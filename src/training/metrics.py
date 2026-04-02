"""Evaluation metrics for joint intent classification and slot filling.

Computes three standard metrics for NLU evaluation:
  1. Intent Accuracy: fraction of utterances with correctly predicted intent
  2. Slot F1: entity-level F1 score using the seqeval library (CoNLL-style)
  3. Sentence Accuracy: fraction of utterances where BOTH intent and ALL
     slot labels are correct (the strictest metric)
"""

from __future__ import annotations

from dataclasses import dataclass

from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import classification_report as seqeval_report
from seqeval.scheme import IOB2


@dataclass
class MetricsResult:
    """Container for NLU evaluation metrics.

    Attributes:
        intent_accuracy: Fraction of correctly classified intents.
        slot_f1: Entity-level F1 score for slot filling.
        sentence_accuracy: Fraction of examples with perfect intent + slots.
        slot_report: Detailed per-slot-type classification report.
    """

    intent_accuracy: float
    slot_f1: float
    sentence_accuracy: float
    slot_report: str = ""


def compute_metrics(
    intent_preds: list[int],
    intent_labels: list[int],
    slot_preds: list[list[str]],
    slot_labels: list[list[str]],
) -> MetricsResult:
    """Compute joint NLU evaluation metrics.

    Args:
        intent_preds: Predicted intent IDs, one per utterance.
        intent_labels: Ground-truth intent IDs, one per utterance.
        slot_preds: Predicted slot label sequences (BIO strings), one list per utterance.
        slot_labels: Ground-truth slot label sequences, one list per utterance.

    Returns:
        MetricsResult with all three metrics and a detailed slot report.
    """
    assert len(intent_preds) == len(intent_labels), "Intent prediction count mismatch"
    assert len(slot_preds) == len(slot_labels), "Slot prediction count mismatch"
    assert len(intent_preds) == len(slot_preds), "Intent/slot count mismatch"

    num_examples = len(intent_preds)
    if num_examples == 0:
        return MetricsResult(
            intent_accuracy=0.0,
            slot_f1=0.0,
            sentence_accuracy=0.0,
        )

    # Intent accuracy
    intent_correct = sum(
        1 for pred, label in zip(intent_preds, intent_labels) if pred == label
    )
    intent_accuracy = intent_correct / num_examples

    # Slot F1 (entity-level, using seqeval)
    slot_f1 = seqeval_f1(slot_labels, slot_preds, mode="strict", scheme=IOB2)

    # Detailed classification report
    try:
        slot_report = seqeval_report(slot_labels, slot_preds, mode="strict", scheme=IOB2)
    except Exception:
        slot_report = "Could not generate detailed report."

    # Sentence accuracy: both intent and all slots must be correct
    sentence_correct = 0
    for i_pred, i_label, s_pred, s_label in zip(
        intent_preds, intent_labels, slot_preds, slot_labels
    ):
        if i_pred == i_label and s_pred == s_label:
            sentence_correct += 1
    sentence_accuracy = sentence_correct / num_examples

    return MetricsResult(
        intent_accuracy=intent_accuracy,
        slot_f1=slot_f1,
        sentence_accuracy=sentence_accuracy,
        slot_report=slot_report,
    )
