"""Single-utterance inference pipeline for intent classification and slot filling.

Provides a high-level API for loading a trained model and running predictions
on raw text utterances. Handles tokenization, model inference, and decoding
of results into human-readable intent labels and slot-value pairs.

Usage:
    predictor = IntentSlotPredictor.from_pretrained("runs/joint_bert_atis")
    result = predictor("Book a flight from Boston to New York")
    print(result.intent, result.slots, result.confidence)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from src.data.tokenization import SubwordAligner
from src.data.vocab import LabelVocab
from src.models.joint_bert import JointBERT
from src.models.slot_attention import SlotAttentionModel

logger = logging.getLogger(__name__)


@dataclass
class SlotValue:
    """A single extracted slot with its value and span information.

    Attributes:
        slot_type: The slot type name (e.g., "from_city", "to_city").
        value: The extracted text value (e.g., "Boston").
        start_word: Start word index in the original utterance.
        end_word: End word index (exclusive) in the original utterance.
    """

    slot_type: str
    value: str
    start_word: int
    end_word: int


@dataclass
class PredictionResult:
    """Complete prediction result for a single utterance.

    Attributes:
        intent: Predicted intent label.
        confidence: Confidence score for the predicted intent (softmax probability).
        slots: Dictionary mapping slot types to extracted values.
        slot_details: List of SlotValue objects with span information.
        word_labels: Per-word BIO slot labels.
        words: Original words from the utterance.
    """

    intent: str
    confidence: float
    slots: dict[str, str]
    slot_details: list[SlotValue] = field(default_factory=list)
    word_labels: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)


class IntentSlotPredictor:
    """High-level prediction interface for trained joint models.

    Loads a trained model checkpoint along with its vocabularies and
    provides a simple callable interface for running inference on
    raw text utterances.

    Args:
        model: The trained joint model.
        aligner: Subword tokenizer/aligner.
        intent_vocab: Intent label vocabulary.
        slot_vocab: Slot label vocabulary.
        device: Compute device for inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        aligner: SubwordAligner,
        intent_vocab: LabelVocab,
        slot_vocab: LabelVocab,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.aligner = aligner
        self.intent_vocab = intent_vocab
        self.slot_vocab = slot_vocab
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        checkpoint: str = "best_model.pt",
        device: Optional[str] = None,
    ) -> "IntentSlotPredictor":
        """Load a trained predictor from a saved checkpoint directory.

        Expects the directory to contain:
          - best_model.pt (or specified checkpoint): model weights
          - intent_vocab.json: intent label vocabulary
          - slot_vocab.json: slot label vocabulary
          - model_config.json: model architecture configuration

        Args:
            model_dir: Path to the saved model directory.
            checkpoint: Filename of the model checkpoint.
            device: Device string (e.g., "cuda", "cpu"). Auto-detects if None.

        Returns:
            Initialized IntentSlotPredictor.
        """
        model_dir = Path(model_dir)

        if device is None:
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        # Load vocabularies
        intent_vocab = LabelVocab.load(model_dir / "intent_vocab.json")
        slot_vocab = LabelVocab.load(model_dir / "slot_vocab.json")

        # Load model config
        config_path = model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found at {config_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))

        # Build model from config
        model_name = config["model"]["name"]
        pretrained = config["model"]["pretrained_model_name"]

        if model_name == "joint_bert":
            model = JointBERT(
                model_name=pretrained,
                num_intents=len(intent_vocab),
                num_slots=len(slot_vocab),
                use_crf=config.get("slot_head", {}).get("use_crf", False),
                slot_hidden_dim=config.get("slot_head", {}).get("hidden_dim"),
            )
        elif model_name == "slot_attention":
            model = SlotAttentionModel(
                model_name=pretrained,
                num_intents=len(intent_vocab),
                num_slots=len(slot_vocab),
                use_crf=config.get("slot_head", {}).get("use_crf", False),
                slot_hidden_dim=config.get("slot_head", {}).get("hidden_dim", 256),
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Load checkpoint weights
        checkpoint_path = model_dir / checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device_obj, weights_only=True)
        model.load_state_dict(state_dict)

        # Build aligner
        aligner = SubwordAligner(pretrained, max_seq_length=50)

        logger.info("Loaded model from %s (%s, device=%s)", model_dir, model_name, device_obj)
        return cls(model=model, aligner=aligner, intent_vocab=intent_vocab, slot_vocab=slot_vocab, device=device_obj)

    @torch.no_grad()
    def __call__(self, utterance: str) -> PredictionResult:
        """Run inference on a single utterance.

        Args:
            utterance: Raw text input (e.g., "Book a flight to New York").

        Returns:
            PredictionResult with intent, confidence, and extracted slots.
        """
        words = utterance.strip().split()
        aligned = self.aligner.align(words)

        input_ids = torch.tensor([aligned.input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([aligned.attention_mask], dtype=torch.long, device=self.device)
        token_type_ids = torch.tensor([aligned.token_type_ids], dtype=torch.long, device=self.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Decode intent
        intent_probs = F.softmax(output.intent_logits, dim=-1)
        intent_id = intent_probs.argmax(dim=-1).item()
        confidence = intent_probs[0, intent_id].item()
        intent_label = self.intent_vocab.id_to_label(intent_id)

        # Decode slots
        slot_pred_ids = self.model.decode_slots(output.slot_logits, attention_mask)

        if isinstance(slot_pred_ids[0], list):
            pred_seq = slot_pred_ids[0]
        else:
            pred_seq = slot_pred_ids[0] if isinstance(slot_pred_ids, list) else slot_pred_ids.tolist()[0]

        word_labels = self.aligner.decode_slots(pred_seq, aligned.word_ids, self.slot_vocab)

        # Ensure we have labels for all words (truncation handling)
        while len(word_labels) < len(words):
            word_labels.append("O")
        word_labels = word_labels[: len(words)]

        # Extract slot values from BIO labels
        slot_details = self._extract_slots(words, word_labels)
        slots = {s.slot_type: s.value for s in slot_details}

        return PredictionResult(
            intent=intent_label,
            confidence=confidence,
            slots=slots,
            slot_details=slot_details,
            word_labels=word_labels,
            words=words,
        )

    @staticmethod
    def _extract_slots(words: list[str], labels: list[str]) -> list[SlotValue]:
        """Extract slot-value pairs from BIO-tagged word sequences.

        Handles B-/I- prefixes to group consecutive tokens into slot values.
        For example: ["B-city", "I-city"] over ["New", "York"] yields
        SlotValue(slot_type="city", value="New York").

        Args:
            words: Original words.
            labels: BIO slot labels aligned to words.

        Returns:
            List of extracted SlotValue objects.
        """
        slots: list[SlotValue] = []
        current_type: Optional[str] = None
        current_words: list[str] = []
        current_start: int = 0

        for i, (word, label) in enumerate(zip(words, labels)):
            if label.startswith("B-"):
                # Save previous slot if exists
                if current_type is not None:
                    slots.append(
                        SlotValue(
                            slot_type=current_type,
                            value=" ".join(current_words),
                            start_word=current_start,
                            end_word=i,
                        )
                    )
                current_type = label[2:]
                current_words = [word]
                current_start = i

            elif label.startswith("I-") and current_type == label[2:]:
                current_words.append(word)

            else:
                if current_type is not None:
                    slots.append(
                        SlotValue(
                            slot_type=current_type,
                            value=" ".join(current_words),
                            start_word=current_start,
                            end_word=i,
                        )
                    )
                    current_type = None
                    current_words = []

        # Handle last slot
        if current_type is not None:
            slots.append(
                SlotValue(
                    slot_type=current_type,
                    value=" ".join(current_words),
                    start_word=current_start,
                    end_word=len(words),
                )
            )

        return slots
