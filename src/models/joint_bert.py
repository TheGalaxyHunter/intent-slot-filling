"""JointBERT: BERT-based joint model for intent classification and slot filling.

Implements the architecture from Chen et al. (2019), "BERT for Joint Intent
Classification and Slot Filling." The model uses a pre-trained BERT encoder
as a shared backbone with two task-specific heads:

  1. Intent classification head: operates on the [CLS] token representation
  2. Slot filling head: operates on token-level representations with optional CRF

The two tasks share the encoder, enabling the model to learn representations
that are useful for both intent detection and entity extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from src.models.crf import ConditionalRandomField


@dataclass
class JointOutput:
    """Output container for the joint model.

    Attributes:
        loss: Combined intent + slot loss (only when labels are provided).
        intent_logits: Raw logits for intent classification, shape (B, num_intents).
        slot_logits: Raw logits for slot filling, shape (B, seq_len, num_slots).
        intent_loss: Intent classification loss component.
        slot_loss: Slot filling loss component.
    """

    loss: Optional[torch.Tensor] = None
    intent_logits: Optional[torch.Tensor] = None
    slot_logits: Optional[torch.Tensor] = None
    intent_loss: Optional[torch.Tensor] = None
    slot_loss: Optional[torch.Tensor] = None


class IntentClassifier(nn.Module):
    """Classification head for intent prediction from the [CLS] token.

    Args:
        input_dim: Dimensionality of the encoder hidden states.
        num_intents: Number of intent classes.
        dropout: Dropout probability.
        hidden_dim: Optional intermediate hidden layer size. If None,
            projects directly from input_dim to num_intents.
    """

    def __init__(
        self,
        input_dim: int,
        num_intents: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_intents),
            )
        else:
            self.classifier = nn.Linear(input_dim, num_intents)

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        """Classify intent from the [CLS] token representation.

        Args:
            cls_output: [CLS] token hidden state, shape (batch_size, hidden_dim).

        Returns:
            Intent logits, shape (batch_size, num_intents).
        """
        return self.classifier(self.dropout(cls_output))


class SlotClassifier(nn.Module):
    """Token-level classification head for slot filling.

    Args:
        input_dim: Dimensionality of the encoder hidden states.
        num_slots: Number of slot label types (BIO tags).
        dropout: Dropout probability.
        hidden_dim: Optional intermediate hidden layer size.
    """

    def __init__(
        self,
        input_dim: int,
        num_slots: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_slots),
            )
        else:
            self.classifier = nn.Linear(input_dim, num_slots)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """Classify slot labels at each token position.

        Args:
            sequence_output: Encoder hidden states, shape (batch_size, seq_len, hidden_dim).

        Returns:
            Slot logits, shape (batch_size, seq_len, num_slots).
        """
        return self.classifier(self.dropout(sequence_output))


class JointBERT(nn.Module):
    """BERT-based joint model for intent classification and slot filling.

    Uses a shared BERT encoder with separate classification heads for
    intent detection (from [CLS]) and slot filling (from token outputs).
    Optionally applies a CRF layer on top of slot logits for structured prediction.

    Args:
        model_name: Pre-trained BERT model identifier (e.g., "bert-base-uncased").
        num_intents: Number of intent classes.
        num_slots: Number of slot types (BIO tag count).
        intent_dropout: Dropout for the intent head.
        slot_dropout: Dropout for the slot head.
        intent_hidden_dim: Optional hidden layer size for intent head.
        slot_hidden_dim: Optional hidden layer size for slot head.
        use_crf: Whether to apply a CRF layer on slot logits.
        intent_loss_weight: Weight for intent loss in the combined objective.
        slot_loss_weight: Weight for slot loss in the combined objective.
    """

    def __init__(
        self,
        model_name: str,
        num_intents: int,
        num_slots: int,
        intent_dropout: float = 0.1,
        slot_dropout: float = 0.1,
        intent_hidden_dim: Optional[int] = None,
        slot_hidden_dim: Optional[int] = None,
        use_crf: bool = False,
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
            hidden_dim=intent_hidden_dim,
        )

        self.slot_classifier = SlotClassifier(
            input_dim=hidden_size,
            num_slots=num_slots,
            dropout=slot_dropout,
            hidden_dim=slot_hidden_dim,
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
        """Forward pass through the joint model.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len).
            attention_mask: Attention mask, shape (batch_size, seq_len).
            token_type_ids: Segment IDs, shape (batch_size, seq_len).
            intent_label_id: Ground-truth intent IDs, shape (batch_size,).
            slot_label_ids: Ground-truth slot IDs, shape (batch_size, seq_len).
                Positions with value -100 are ignored in the loss.

        Returns:
            JointOutput containing loss, logits, and per-task losses.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # (B, seq_len, H)
        cls_output = outputs.pooler_output             # (B, H)

        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = None
        intent_loss = None
        slot_loss = None

        if intent_label_id is not None and slot_label_ids is not None:
            intent_loss = self.intent_loss_fn(intent_logits, intent_label_id)

            if self.use_crf:
                # CRF computes the negative log-likelihood internally.
                # We need to mask out positions with ignore_index before passing
                # to the CRF, replacing them with 0 (valid tag).
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
        """Decode slot predictions, using CRF Viterbi if available.

        Args:
            slot_logits: Raw slot logits, shape (batch_size, seq_len, num_slots).
            attention_mask: Attention mask, shape (batch_size, seq_len).

        Returns:
            List of predicted slot label ID sequences (one per example).
        """
        if self.use_crf:
            return self.crf.decode(slot_logits, mask=attention_mask.bool())
        else:
            return slot_logits.argmax(dim=-1).tolist()
