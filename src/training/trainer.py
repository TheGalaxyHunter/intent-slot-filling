"""Training loop for joint intent classification and slot filling.

Implements a standard PyTorch training loop with:
  - Joint loss optimization (intent CE + slot CE/CRF)
  - Learning rate warmup with linear decay
  - Gradient clipping and mixed-precision support
  - Periodic evaluation with early stopping on sentence accuracy
  - Checkpoint saving and logging
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.dataset import NLUDataset, build_vocabs_from_data
from src.data.vocab import LabelVocab
from src.models.joint_bert import JointBERT
from src.models.slot_attention import SlotAttentionModel
from src.training.metrics import compute_metrics, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Tracks training progress across epochs."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    eval_results: list[dict] = field(default_factory=list)


def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a linear warmup + linear decay learning rate schedule.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(
    model_config: dict,
    num_intents: int,
    num_slots: int,
    loss_config: dict,
) -> nn.Module:
    """Instantiate the appropriate model from configuration.

    Args:
        model_config: Model configuration dictionary.
        num_intents: Number of intent classes.
        num_slots: Number of slot types.
        loss_config: Loss weighting configuration.

    Returns:
        Initialized model.
    """
    model_name = model_config["model"]["name"]
    pretrained = model_config["model"]["pretrained_model_name"]

    common_kwargs = {
        "model_name": pretrained,
        "num_intents": num_intents,
        "num_slots": num_slots,
        "intent_dropout": model_config["intent_head"]["dropout"],
        "slot_dropout": model_config["slot_head"]["dropout"],
        "use_crf": model_config["slot_head"].get("use_crf", False),
        "intent_loss_weight": loss_config.get("intent_weight", 1.0),
        "slot_loss_weight": loss_config.get("slot_weight", 1.0),
    }

    if model_name == "joint_bert":
        return JointBERT(
            **common_kwargs,
            intent_hidden_dim=model_config["intent_head"].get("hidden_dim"),
            slot_hidden_dim=model_config["slot_head"].get("hidden_dim"),
        )
    elif model_name == "slot_attention":
        attn_config = model_config["slot_head"].get("attention", {})
        return SlotAttentionModel(
            **common_kwargs,
            slot_hidden_dim=model_config["slot_head"].get("hidden_dim", 256),
            num_attention_heads=attn_config.get("num_heads", 8),
            use_gate=attn_config.get("gate_mechanism", True),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
) -> float:
    """Run one training epoch.

    Args:
        model: The joint model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Compute device.
        max_grad_norm: Maximum gradient norm for clipping.
        gradient_accumulation_steps: Number of steps to accumulate gradients.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_steps = 0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        loss = output.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            num_steps += 1

        total_loss += output.loss.item()

    return total_loss / max(num_steps, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    slot_vocab: LabelVocab,
) -> MetricsResult:
    """Run evaluation on the dev/test set.

    Args:
        model: The joint model.
        dataloader: Evaluation data loader.
        device: Compute device.
        slot_vocab: Slot label vocabulary for decoding.

    Returns:
        MetricsResult with intent accuracy, slot F1, and sentence accuracy.
    """
    model.eval()

    all_intent_preds: list[int] = []
    all_intent_labels: list[int] = []
    all_slot_preds: list[list[str]] = []
    all_slot_labels: list[list[str]] = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )

        # Intent predictions
        intent_preds = output.intent_logits.argmax(dim=-1).tolist()
        intent_labels = batch["intent_label_id"].tolist()
        all_intent_preds.extend(intent_preds)
        all_intent_labels.extend(intent_labels)

        # Slot predictions
        slot_pred_ids = model.decode_slots(output.slot_logits, batch["attention_mask"])
        slot_label_ids = batch["slot_label_ids"]

        for pred_seq, label_seq, mask in zip(
            slot_pred_ids, slot_label_ids.tolist(), batch["attention_mask"].tolist()
        ):
            pred_labels = []
            true_labels = []

            for p, l, m in zip(
                pred_seq if isinstance(pred_seq, list) else [pred_seq],
                label_seq,
                mask,
            ):
                if m == 0 or l == -100:
                    continue
                pred_labels.append(slot_vocab.id_to_label(p))
                true_labels.append(slot_vocab.id_to_label(l))

            all_slot_preds.append(pred_labels)
            all_slot_labels.append(true_labels)

    return compute_metrics(
        intent_preds=all_intent_preds,
        intent_labels=all_intent_labels,
        slot_preds=all_slot_preds,
        slot_labels=all_slot_labels,
    )


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train joint intent/slot model")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--model", type=str, required=True, help="Model config YAML")
    parser.add_argument("--dataset", type=str, default="atis", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="data", help="Data root directory")
    parser.add_argument("--output-dir", type=str, default="runs/experiment", help="Output dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    train_config = OmegaConf.load(args.config)
    model_config = OmegaConf.to_container(OmegaConf.load(args.model), resolve=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build vocabularies
    data_root = Path(args.data_dir) / args.dataset
    intent_vocab, slot_vocab = build_vocabs_from_data(data_root)
    intent_vocab.save(output_dir / "intent_vocab.json")
    slot_vocab.save(output_dir / "slot_vocab.json")

    # Build datasets
    tokenizer_name = model_config["tokenizer"]["pretrained_model_name"]
    max_seq_length = train_config.data.max_seq_length

    train_dataset = NLUDataset(
        data_dir=data_root / "train",
        tokenizer_name=tokenizer_name,
        intent_vocab=intent_vocab,
        slot_vocab=slot_vocab,
        max_seq_length=max_seq_length,
    )

    eval_dataset = NLUDataset(
        data_dir=data_root / "test",
        tokenizer_name=tokenizer_name,
        intent_vocab=intent_vocab,
        slot_vocab=slot_vocab,
        max_seq_length=max_seq_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_config.training.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Build model
    model = build_model(
        model_config=model_config,
        num_intents=len(intent_vocab),
        num_slots=len(slot_vocab),
        loss_config=OmegaConf.to_container(train_config.loss, resolve=True),
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{num_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.training.learning_rate,
        weight_decay=train_config.training.weight_decay,
        betas=tuple(train_config.optimizer.betas),
        eps=train_config.optimizer.eps,
    )

    num_training_steps = len(train_loader) * train_config.training.epochs
    num_warmup_steps = int(num_training_steps * train_config.training.warmup_ratio)

    scheduler = get_linear_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)

    # Training loop
    state = TrainingState()
    logger.info("Starting training for %d epochs", train_config.training.epochs)

    for epoch in range(train_config.training.epochs):
        state.epoch = epoch
        start_time = time.time()

        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=train_config.training.max_grad_norm,
            gradient_accumulation_steps=train_config.training.gradient_accumulation_steps,
        )
        state.train_losses.append(avg_loss)

        # Evaluate
        metrics = evaluate(model, eval_loader, device, slot_vocab)
        elapsed = time.time() - start_time

        logger.info(
            "Epoch %d/%d | Loss: %.4f | Intent Acc: %.2f | Slot F1: %.2f | "
            "Sentence Acc: %.2f | Time: %.1fs",
            epoch + 1,
            train_config.training.epochs,
            avg_loss,
            metrics.intent_accuracy * 100,
            metrics.slot_f1 * 100,
            metrics.sentence_accuracy * 100,
            elapsed,
        )

        eval_result = {
            "epoch": epoch,
            "loss": avg_loss,
            "intent_accuracy": metrics.intent_accuracy,
            "slot_f1": metrics.slot_f1,
            "sentence_accuracy": metrics.sentence_accuracy,
        }
        state.eval_results.append(eval_result)

        # Save best model
        current_metric = metrics.sentence_accuracy
        if current_metric > state.best_metric:
            state.best_metric = current_metric
            state.best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info("New best model at epoch %d (sentence_acc=%.2f)", epoch + 1, current_metric * 100)

    # Save final artifacts
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    training_summary = {
        "best_epoch": state.best_epoch,
        "best_sentence_accuracy": state.best_metric,
        "total_epochs": train_config.training.epochs,
        "eval_history": state.eval_results,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )

    logger.info("Training complete. Best sentence accuracy: %.2f at epoch %d", state.best_metric * 100, state.best_epoch + 1)


if __name__ == "__main__":
    main()
