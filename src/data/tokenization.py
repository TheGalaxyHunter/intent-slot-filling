"""Subword tokenization with slot label alignment.

When using BERT-style models, words are split into subword tokens. For slot
filling, we need to maintain alignment between original words and subword
tokens so that each word's slot label maps to the correct token position.

Strategy:
  - For special tokens ([CLS], [SEP], [PAD]): assign None as word_id
  - For the first subword of each word: assign the original word index
  - For continuation subwords: assign the word index (used to determine
    that the slot label should be IGNORE_INDEX during training)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


@dataclass
class AlignedTokens:
    """Result of subword tokenization with word-to-token alignment.

    Attributes:
        input_ids: Token IDs for the model.
        attention_mask: Binary mask indicating real tokens (1) vs padding (0).
        token_type_ids: Segment IDs (always 0 for single-sentence input).
        word_ids: Mapping from each token position to the originating word
            index, or None for special/padding tokens.
        tokens: The actual subword token strings (for debugging).
    """

    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    word_ids: list[Optional[int]]
    tokens: list[str]


class SubwordAligner:
    """Handles subword tokenization while maintaining word-level alignment.

    This is critical for slot filling: when a word like "New York" is tokenized
    into ["new", "york"], we need to know that both subwords correspond to
    the same slot label. And when "playing" becomes ["play", "##ing"], only
    the first subword should receive the slot label for loss computation.

    Args:
        model_name: Hugging Face model/tokenizer name (e.g., "bert-base-uncased").
        max_seq_length: Maximum sequence length including special tokens.
    """

    def __init__(self, model_name: str, max_seq_length: int = 50) -> None:
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        self.max_seq_length = max_seq_length

        if not self.tokenizer.is_fast:
            raise ValueError(
                f"Tokenizer for {model_name} is not a fast tokenizer. "
                "Fast tokenizer is required for word_ids() support."
            )

    def align(self, words: list[str]) -> AlignedTokens:
        """Tokenize a list of words and produce aligned outputs.

        Args:
            words: Pre-tokenized words from the original utterance.

        Returns:
            AlignedTokens with input IDs, masks, and word-to-token mapping.
        """
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        word_ids = []
        for i in range(len(encoding["input_ids"])):
            word_idx = encoding.word_ids(0) if hasattr(encoding, "word_ids") else None
            if word_idx is not None:
                break
        # Use the batch encoding's word_ids method
        raw_word_ids = encoding.word_ids(batch_index=0)

        # Pad word_ids to max_seq_length (they should already be, but ensure it)
        word_ids = list(raw_word_ids)
        while len(word_ids) < self.max_seq_length:
            word_ids.append(None)
        word_ids = word_ids[: self.max_seq_length]

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        return AlignedTokens(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding.get("token_type_ids", [0] * len(encoding["input_ids"])),
            word_ids=word_ids,
            tokens=tokens,
        )

    def decode_slots(
        self,
        slot_label_ids: list[int],
        word_ids: list[Optional[int]],
        slot_vocab: "LabelVocab",
    ) -> list[str]:
        """Decode token-level slot predictions back to word-level labels.

        Only the first subword's prediction is used for each word. This inverts
        the alignment performed during tokenization.

        Args:
            slot_label_ids: Predicted slot label IDs for each token position.
            word_ids: Word-to-token mapping from AlignedTokens.
            slot_vocab: The slot label vocabulary for ID-to-label conversion.

        Returns:
            Word-level slot labels (one per original word).
        """
        word_labels: dict[int, str] = {}

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx not in word_labels:
                label_id = slot_label_ids[token_idx]
                word_labels[word_idx] = slot_vocab.id_to_label(label_id)

        max_word_idx = max(word_labels.keys()) if word_labels else -1
        return [word_labels.get(i, "O") for i in range(max_word_idx + 1)]
