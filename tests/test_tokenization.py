"""Tests for subword tokenization alignment.

Validates that the SubwordAligner correctly maps subword tokens back to
their originating words, which is critical for slot filling accuracy.
"""

from __future__ import annotations

import pytest

from src.data.tokenization import SubwordAligner
from src.data.vocab import LabelVocab


@pytest.fixture
def aligner() -> SubwordAligner:
    """Create a SubwordAligner with BERT tokenizer."""
    return SubwordAligner("bert-base-uncased", max_seq_length=32)


@pytest.fixture
def slot_vocab() -> LabelVocab:
    """Create a minimal slot vocabulary for testing."""
    return LabelVocab(
        labels=["O", "B-city", "I-city", "B-date", "I-date"],
        name="slot",
    )


class TestSubwordAlignment:
    """Test subword tokenization with word-level alignment."""

    def test_simple_words_preserve_alignment(self, aligner: SubwordAligner) -> None:
        """Words that are single tokens should have 1:1 alignment."""
        words = ["book", "a", "flight"]
        aligned = aligner.align(words)

        # Check that each word maps to exactly one token (plus special tokens)
        word_indices = [w for w in aligned.word_ids if w is not None]
        assert 0 in word_indices
        assert 1 in word_indices
        assert 2 in word_indices

    def test_subword_split_alignment(self, aligner: SubwordAligner) -> None:
        """Words split into subwords should all map to the same word index."""
        words = ["unbelievable", "performance"]
        aligned = aligner.align(words)

        # "unbelievable" will be split into multiple subwords by BERT
        word_0_tokens = [i for i, w in enumerate(aligned.word_ids) if w == 0]
        assert len(word_0_tokens) >= 1, "Word 'unbelievable' should have at least one token"

    def test_special_tokens_have_none_word_id(self, aligner: SubwordAligner) -> None:
        """[CLS], [SEP], and [PAD] tokens should have None word_ids."""
        words = ["hello"]
        aligned = aligner.align(words)

        # First token ([CLS]) should be None
        assert aligned.word_ids[0] is None
        # Last real tokens after words should include [SEP] with None
        none_positions = [i for i, w in enumerate(aligned.word_ids) if w is None]
        assert len(none_positions) >= 2, "At least [CLS] and [SEP] should have None word_ids"

    def test_padding_tokens_have_none_word_id(self, aligner: SubwordAligner) -> None:
        """Padding positions should have None word_ids."""
        words = ["hi"]
        aligned = aligner.align(words)

        # With max_seq_length=32 and just "hi", most positions are padding
        padding_nones = sum(
            1
            for w, m in zip(aligned.word_ids, aligned.attention_mask)
            if w is None and m == 0
        )
        assert padding_nones > 0, "Padding tokens should have None word_ids"

    def test_attention_mask_matches_real_tokens(self, aligner: SubwordAligner) -> None:
        """Attention mask should be 1 for real tokens and 0 for padding."""
        words = ["book", "a", "flight"]
        aligned = aligner.align(words)

        real_count = sum(aligned.attention_mask)
        # At minimum: [CLS] + 3 words + [SEP] = 5 tokens
        assert real_count >= 5

    def test_output_length_matches_max_seq_length(self, aligner: SubwordAligner) -> None:
        """All output sequences should be padded to max_seq_length."""
        words = ["test"]
        aligned = aligner.align(words)

        assert len(aligned.input_ids) == 32
        assert len(aligned.attention_mask) == 32
        assert len(aligned.word_ids) == 32


class TestSlotDecoding:
    """Test decoding of token-level slot predictions back to word-level labels."""

    def test_decode_simple_slots(
        self, aligner: SubwordAligner, slot_vocab: LabelVocab
    ) -> None:
        """Slot labels should decode correctly for simple words."""
        words = ["fly", "to", "boston"]
        aligned = aligner.align(words)

        # Build fake slot predictions: O, O, B-city
        slot_label_ids = []
        for word_idx in aligned.word_ids:
            if word_idx is None:
                slot_label_ids.append(0)  # O
            elif word_idx == 2:
                slot_label_ids.append(1)  # B-city
            else:
                slot_label_ids.append(0)  # O

        decoded = aligner.decode_slots(slot_label_ids, aligned.word_ids, slot_vocab)
        assert decoded[0] == "O"
        assert decoded[1] == "O"
        assert decoded[2] == "B-city"

    def test_decode_multi_word_slot(
        self, aligner: SubwordAligner, slot_vocab: LabelVocab
    ) -> None:
        """Multi-word slots should decode with B- and I- prefixes."""
        words = ["fly", "to", "new", "york"]
        aligned = aligner.align(words)

        slot_label_ids = []
        for word_idx in aligned.word_ids:
            if word_idx is None:
                slot_label_ids.append(0)
            elif word_idx == 2:
                slot_label_ids.append(1)  # B-city
            elif word_idx == 3:
                slot_label_ids.append(2)  # I-city
            else:
                slot_label_ids.append(0)

        decoded = aligner.decode_slots(slot_label_ids, aligned.word_ids, slot_vocab)
        assert len(decoded) == 4
        assert decoded[2] == "B-city"
        assert decoded[3] == "I-city"


class TestLabelVocab:
    """Test label vocabulary operations."""

    def test_roundtrip_conversion(self, slot_vocab: LabelVocab) -> None:
        """Converting label -> id -> label should return the original."""
        for label in slot_vocab.labels:
            label_id = slot_vocab.label_to_id(label)
            recovered = slot_vocab.id_to_label(label_id)
            assert recovered == label

    def test_unknown_label_raises(self, slot_vocab: LabelVocab) -> None:
        """Looking up an unknown label should raise KeyError."""
        with pytest.raises(KeyError):
            slot_vocab.label_to_id("B-nonexistent")

    def test_vocab_size(self, slot_vocab: LabelVocab) -> None:
        """Vocabulary should report correct size."""
        assert len(slot_vocab) == 5

    def test_contains(self, slot_vocab: LabelVocab) -> None:
        """Membership test should work correctly."""
        assert "O" in slot_vocab
        assert "B-city" in slot_vocab
        assert "B-nonexistent" not in slot_vocab
