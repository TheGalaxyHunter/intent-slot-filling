"""Intent and slot label vocabularies.

Manages the bidirectional mapping between string labels and integer IDs
used by the model. Handles serialization to/from disk for reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LabelVocab:
    """Bidirectional label-to-ID mapping for intent or slot labels.

    Provides consistent encoding of categorical labels into integer indices
    for model training and decoding of predictions back to human-readable labels.

    Args:
        labels: Ordered list of unique label strings.
        name: Descriptive name for this vocabulary (e.g., "intent" or "slot").
    """

    def __init__(self, labels: list[str], name: str = "labels") -> None:
        self.name = name
        self._labels = list(labels)
        self._label_to_id: dict[str, int] = {label: idx for idx, label in enumerate(labels)}
        self._id_to_label: dict[int, str] = {idx: label for idx, label in enumerate(labels)}

    def label_to_id(self, label: str) -> int:
        """Convert a label string to its integer ID.

        Args:
            label: The label string to look up.

        Returns:
            Integer ID for the label.

        Raises:
            KeyError: If the label is not in the vocabulary.
        """
        if label not in self._label_to_id:
            raise KeyError(
                f"Unknown {self.name} label: '{label}'. "
                f"Vocabulary contains {len(self._labels)} labels."
            )
        return self._label_to_id[label]

    def id_to_label(self, label_id: int) -> str:
        """Convert an integer ID back to its label string.

        Args:
            label_id: The integer ID to look up.

        Returns:
            Label string for the ID.

        Raises:
            KeyError: If the ID is not in the vocabulary.
        """
        if label_id not in self._id_to_label:
            raise KeyError(
                f"Unknown {self.name} label ID: {label_id}. "
                f"Vocabulary contains {len(self._labels)} labels (0-{len(self._labels) - 1})."
            )
        return self._id_to_label[label_id]

    @property
    def labels(self) -> list[str]:
        """Return the ordered list of labels."""
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, label: str) -> bool:
        return label in self._label_to_id

    def __repr__(self) -> str:
        return f"LabelVocab(name='{self.name}', size={len(self)})"

    def save(self, path: str | Path) -> None:
        """Serialize the vocabulary to a JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "labels": self._labels}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %s vocab (%d labels) to %s", self.name, len(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "LabelVocab":
        """Load a vocabulary from a JSON file.

        Args:
            path: Path to the saved vocabulary JSON.

        Returns:
            Reconstructed LabelVocab instance.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls(labels=data["labels"], name=data.get("name", "labels"))
        logger.info("Loaded %s vocab (%d labels) from %s", vocab.name, len(vocab), path)
        return vocab

    @classmethod
    def from_file(cls, path: str | Path, name: Optional[str] = None) -> "LabelVocab":
        """Build a vocabulary from a plain text file with one label per line.

        Args:
            path: Path to a text file containing labels.
            name: Optional name for the vocabulary.

        Returns:
            New LabelVocab instance.
        """
        path = Path(path)
        labels = []
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            label = line.strip()
            if label and label not in seen:
                labels.append(label)
                seen.add(label)

        return cls(labels=labels, name=name or path.stem)
