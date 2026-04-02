"""Slot attention model with intent-conditioned attention for slot filling.

Extends the standard joint model by using the predicted intent representation
to condition the slot filling attention mechanism. The intuition is that
knowing the intent (e.g., "BookFlight") provides useful context for
determining which slots to expect (e.g., "from_city", "to_city").

References:
  - Goo et al. "Slot-Gated Modeling for Joint Slot Filling and Intent
    Prediction." NAACL-HLT 2018.
  - Liu and Lane. "Attention-Based Recurrent Neural Network Models for
    Joint Intent Detection and Slot Filling." Interspeech 2016.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from src.models.crf import ConditionalRandomField
from src.models.joint_bert import IntentClassifier, JointOutput


class IntentConditionedSlotAttention(nn.Module):
    """Multi-head attention for slot filling, conditioned on intent context.

    The attention mechanism incorporates the intent representation (from the
    [CLS] token) into the key/value computation, allowing the slot filler
    to attend differently based on the detected intent.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
        use_gate: Whether to use a learned gate to modulate intent influence.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_gate = use_gate

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Project intent representation into key/value space
        self.intent_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.intent_value_proj = nn.Linear(hidden_dim, hidden_dim)

        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        sequence_output: torch.Tensor,
        intent_context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply intent-conditioned attention to the token sequence.

        Args:
            sequence_output: Encoder hidden states, shape (B, T, H).
            intent_context: Intent representation from [CLS], shape (B, H).
            attention_mask: Attention mask, shape (B, T).

        Returns:
            Attended representations, shape (B, T, H).
        """
        batch_size, seq_len, hidden_dim = sequence_output.shape

        # Compute queries from token sequence
        queries = self.query_proj(sequence_output)

        # Compute keys and values from token sequence + intent context
        # Expand intent context to match sequence length
        intent_expanded = intent_context.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine token-level and intent-level keys/values
        token_keys = self.key_proj(sequence_output)
        intent_keys = self.intent_key_proj(intent_expanded)
        keys = token_keys + intent_keys

        token_values = self.value_proj(sequence_output)
        intent_values = self.intent_value_proj(intent_expanded)
        values = token_values + intent_values

        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale

        if attention_mask is not None:
            # Expand mask for multi-head: (B, 1, 1, T)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, values)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        attended = self.output_proj(attended)

        # Optional gating: learn how much intent context to incorporate
        if self.use_gate:
            gate_input = torch.cat([sequence_output, attended], dim=-1)
            gate_value = self.gate(gate_input)
            attended = gate_value * attended + (1 - gate_value) * sequence_output

        # Residual connection + layer norm
        output = self.layer_norm(sequence_output + self.dropout(attended))
        return output


class SlotAttentionModel(nn.Module):
    """Joint model with intent-conditioned slot attention.

    Like JointBERT, uses a shared BERT encoder. The key difference is that
    the slot filling head uses a multi-head attention mechanism conditioned
    on the intent representation, allowing the model to dynamically adjust
    its slot extraction behavior based on the predicted intent.

    Args:
        model_name: Pre-trained BERT model name.
        num_intents: Number of intent classes.
        num_slots: Number of slot label types.
        num_attention_heads: Number of heads in the slot attention layer.
        intent_dropout: Dropout for the intent classification head.
        slot_dropout: Dropout for the slot head.
        slot_hidden_dim: Hidden dimension for the slot projection.
        use_crf: Whether to apply a CRF layer after attention.
        use_gate: Whether to use a gating mechanism in attention.
        intent_loss_weight: Weight for intent loss.
        slot_loss_weight: Weight for slot loss.
    """

    def __init__(
        self,
        model_name: str,
        num_intents: int,
        num_slots: int,
        num_attention_heads: int = 8,
        intent_dropout: float = 0.1,
        slot_dropout: float = 0.1,
        slot_hidden_dim: int = 256,
        use_crf: bool = False,
        use_gate: bool = True,
        intent_loss_weight: float = 1.0,
        slot_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.use_crf = use_crf
        self.intent_loss_weight = intent_loss_weight
        self.slot_loss_weight = slot_loss_weight

        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.intent_classifier = IntentClassifier(
            input_dim=hidden_size,
            num_intents=num_intents,
            dropout=intent_dropout,
        )

        self.slot_attention = IntentConditionedSlotAttention(
            hidden_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=slot_dropout,
            use_gate=use_gate,
        )

        self.slot_projection = nn.Sequential(
            nn.Linear(hidden_size, slot_hidden_dim),
            nn.ReLU(),
            nn.Dropout(slot_dropout),
            nn.Linear(slot_hidden_dim, num_slots),
        )

        if use_crf:
            self.crf = ConditionalRandomField(num_tags=num_slots)

        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        intent_label_id: Optional[torch.Tensor] = None,
        slot_label_ids: Optional[torch.Tensor] = None,
    ) -> JointOutput:
        """Forward pass with intent-conditioned slot attention.

        Args:
            input_ids: Token IDs, shape (B, T).
            attention_mask: Attention mask, shape (B, T).
            token_type_ids: Segment IDs, shape (B, T).
            intent_label_id: Ground-truth intent labels, shape (B,).
            slot_label_ids: Ground-truth slot labels, shape (B, T).

        Returns:
            JointOutput with loss and logits.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        cls_output = outputs.pooler_output

        # Intent prediction from [CLS]
        intent_logits = self.intent_classifier(cls_output)

        # Apply intent-conditioned attention to token representations
        attended_output = self.slot_attention(
            sequence_output=sequence_output,
            intent_context=cls_output,
            attention_mask=attention_mask,
        )

        # Slot prediction from attended representations
        slot_logits = self.slot_projection(attended_output)

        total_loss = None
        intent_loss = None
        slot_loss = None

        if intent_label_id is not None and slot_label_ids is not None:
            intent_loss = self.intent_loss_fn(intent_logits, intent_label_id)

            if self.use_crf:
                crf_mask = attention_mask.bool()
                crf_labels = slot_label_ids.clone()
                crf_labels[crf_labels == -100] = 0
                slot_loss = -self.crf(slot_logits, crf_labels, mask=crf_mask)
            else:
                slot_loss = self.slot_loss_fn(
                    slot_logits.view(-1, self.num_slots),
                    slot_label_ids.view(-1),
                )

            total_loss = (
                self.intent_loss_weight * intent_loss
                + self.slot_loss_weight * slot_loss
            )

        return JointOutput(
            loss=total_loss,
            intent_logits=intent_logits,
            slot_logits=slot_logits,
            intent_loss=intent_loss,
            slot_loss=slot_loss,
        )

    def decode_slots(
        self,
        slot_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        """Decode slot predictions using CRF Viterbi or argmax."""
        if self.use_crf:
            return self.crf.decode(slot_logits, mask=attention_mask.bool())
        else:
            return slot_logits.argmax(dim=-1).tolist()
