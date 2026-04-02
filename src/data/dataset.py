"""PyTorch Dataset for ATIS/SNIPS-format NLU data.

Expects data in the standard format where each example consists of:
  - A tokenized utterance (space-separated words)
  - A sequence of BIO slot labels (one per word)
  - An intent label

Directory structure:
  data/{atis,snips}/
    train/
      seq.in      # utterances, one per line
      seq.out     # slot labels, one per line
      label       # intent labels, one per line
    test/
      seq.in
      seq.out
      label
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.data.tokenization import SubwordAligner
from src.data.vocab import LabelVocab

logger = logging.getLogger(__name__)


@dataclass
class NLUExample:
    """A single NLU training example."""

    words: list[str]
    slot_labels: list[str]
    intent_label: str
    guid: Optional[str] = None


@dataclass
class NLUFeatures:
    """Tokenized and encoded features ready for the model."""

    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    slot_label_ids: list[int]
    intent_label_id: int
    word_ids: list[Optional[int]]


class NLUDataset(Dataset):
    """PyTorch dataset for joint intent classification and slot filling.

    Handles loading raw text data, subword tokenization with slot alignment,
    and conversion to model-ready tensors.

    Args:
        data_dir: Path to the dataset split directory (e.g., data/atis/train).
        tokenizer_name: Hugging Face tokenizer identifier.
        intent_vocab: Label vocabulary for intent classes.
        slot_vocab: Label vocabulary for slot types.
        max_seq_length: Maximum sequence length after tokenization.
        pad_label: Label used for padding and special tokens in slot sequences.
    """

    PAD_LABEL = "PAD"
    IGNORE_INDEX = -100

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer_name: str,
        intent_vocab: LabelVocab,
        slot_vocab: LabelVocab,
        max_seq_length: int = 50,
        pad_label: str = "PAD",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_seq_length = max_seq_length
        self.pad_label = pad_label
        self.intent_vocab = intent_vocab
        self.slot_vocab = slot_vocab

        self.aligner = SubwordAligner(tokenizer_name, max_seq_length=max_seq_length)
        self.examples = self._load_examples()

        logger.info(
            "Loaded %d examples from %s (intents: %d, slots: %d)",
            len(self.examples),
            self.data_dir,
            len(intent_vocab),
            len(slot_vocab),
        )

    def _load_examples(self) -> list[NLUExample]:
        """Read seq.in, seq.out, and label files into NLUExample objects."""
        seq_in_path = self.data_dir / "seq.in"
        seq_out_path = self.data_dir / "seq.out"
        label_path = self.data_dir / "label"

        for path in (seq_in_path, seq_out_path, label_path):
            if not path.exists():
                raise FileNotFoundError(f"Required data file not found: {path}")

        utterances = seq_in_path.read_text(encoding="utf-8").strip().splitlines()
        slot_seqs = seq_out_path.read_text(encoding="utf-8").strip().splitlines()
        intents = label_path.read_text(encoding="utf-8").strip().splitlines()

        if not (len(utterances) == len(slot_seqs) == len(intents)):
            raise ValueError(
                f"Data file lengths do not match: "
                f"seq.in={len(utterances)}, seq.out={len(slot_seqs)}, label={len(intents)}"
            )

        examples = []
        for idx, (utt, slots, intent) in enumerate(zip(utterances, slot_seqs, intents)):
            words = utt.strip().split()
            slot_labels = slots.strip().split()

            if len(words) != len(slot_labels):
                logger.warning(
                    "Skipping example %d: word count (%d) != slot count (%d)",
                    idx,
                    len(words),
                    len(slot_labels),
                )
                continue

            examples.append(
                NLUExample(
                    words=words,
                    slot_labels=slot_labels,
                    intent_label=intent.strip(),
                    guid=f"{self.data_dir.name}-{idx}",
                )
            )

        return examples

    def _convert_to_features(self, example: NLUExample) -> NLUFeatures:
        """Convert an NLUExample to model-ready NLUFeatures.

        This is the critical step where subword tokenization is aligned with
        word-level slot labels. For each word that gets split into multiple
        subwords, only the first subword receives the original slot label;
        subsequent subwords get IGNORE_INDEX so they are excluded from the loss.
        """
        aligned = self.aligner.align(example.words)

        slot_label_ids = []
        for word_idx in aligned.word_ids:
            if word_idx is None:
                slot_label_ids.append(self.IGNORE_INDEX)
            else:
                # Check if this is the first subword for this word
                prev_word_ids = aligned.word_ids[: len(slot_label_ids)]
                if word_idx not in prev_word_ids:
                    label = example.slot_labels[word_idx]
                    slot_label_ids.append(self.slot_vocab.label_to_id(label))
                else:
                    slot_label_ids.append(self.IGNORE_INDEX)

        intent_label_id = self.intent_vocab.label_to_id(example.intent_label)

        return NLUFeatures(
            input_ids=aligned.input_ids,
            attention_mask=aligned.attention_mask,
            token_type_ids=aligned.token_type_ids,
            slot_label_ids=slot_label_ids,
            intent_label_id=intent_label_id,
            word_ids=aligned.word_ids,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        features = self._convert_to_features(self.examples[idx])
        return {
            "input_ids": torch.tensor(features.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(features.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(features.token_type_ids, dtype=torch.long),
            "slot_label_ids": torch.tensor(features.slot_label_ids, dtype=torch.long),
            "intent_label_id": torch.tensor(features.intent_label_id, dtype=torch.long),
        }


def build_vocabs_from_data(data_dir: str | Path) -> tuple[LabelVocab, LabelVocab]:
    """Scan train/test splits to build intent and slot label vocabularies.

    Args:
        data_dir: Root dataset directory (e.g., data/atis/) containing train/ and test/.

    Returns:
        Tuple of (intent_vocab, slot_vocab).
    """
    data_dir = Path(data_dir)
    intent_labels: set[str] = set()
    slot_labels: set[str] = set()

    for split in ("train", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        label_path = split_dir / "label"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").strip().splitlines():
                intent_labels.add(line.strip())

        seq_out_path = split_dir / "seq.out"
        if seq_out_path.exists():
            for line in seq_out_path.read_text(encoding="utf-8").strip().splitlines():
                for label in line.strip().split():
                    slot_labels.add(label)

    intent_vocab = LabelVocab(sorted(intent_labels), name="intent")
    slot_vocab = LabelVocab(sorted(slot_labels), name="slot")

    logger.info("Built intent vocab: %d labels", len(intent_vocab))
    logger.info("Built slot vocab: %d labels", len(slot_vocab))

    return intent_vocab, slot_vocab
