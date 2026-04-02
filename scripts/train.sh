#!/usr/bin/env bash
# Train a joint intent/slot model.
#
# Usage:
#   bash scripts/train.sh [dataset] [model_config]
#
# Examples:
#   bash scripts/train.sh atis joint_bert
#   bash scripts/train.sh snips slot_attention

set -euo pipefail

DATASET="${1:-atis}"
MODEL="${2:-joint_bert}"
OUTPUT_DIR="runs/${MODEL}_${DATASET}"

echo "Training ${MODEL} on ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python -m src.training.trainer \
    --config configs/train.yaml \
    --model "configs/model/${MODEL}.yaml" \
    --dataset "${DATASET}" \
    --data-dir data \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Training complete. Model saved to ${OUTPUT_DIR}"
