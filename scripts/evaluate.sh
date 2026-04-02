#!/usr/bin/env bash
# Evaluate a trained joint intent/slot model.
#
# Usage:
#   bash scripts/evaluate.sh <model_dir>
#
# Example:
#   bash scripts/evaluate.sh runs/joint_bert_atis

set -euo pipefail

MODEL_DIR="${1:?Usage: evaluate.sh <model_dir>}"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: Model directory not found: ${MODEL_DIR}"
    exit 1
fi

echo "Evaluating model from: ${MODEL_DIR}"
echo ""

python -c "
import json
import torch
from pathlib import Path

model_dir = Path('${MODEL_DIR}')
summary_path = model_dir / 'training_summary.json'

if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print('Training Summary')
    print('=' * 50)
    print(f\"Best epoch: {summary['best_epoch'] + 1}\")
    print(f\"Best sentence accuracy: {summary['best_sentence_accuracy'] * 100:.2f}%\")
    print()
    print('Evaluation History (last 5 epochs):')
    print('-' * 50)
    for result in summary['eval_history'][-5:]:
        print(
            f\"  Epoch {result['epoch'] + 1:3d} | \"
            f\"Intent: {result['intent_accuracy'] * 100:5.2f}% | \"
            f\"Slot F1: {result['slot_f1'] * 100:5.2f}% | \"
            f\"Sent: {result['sentence_accuracy'] * 100:5.2f}%\"
        )
else:
    print('No training summary found. Run training first.')
"
