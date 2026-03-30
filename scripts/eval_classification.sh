#!/bin/bash
# Evaluate classification on NIH ChestX-ray14
# Usage: bash scripts/eval_classification.sh [linear_probe|finetune]

set -e

cd "$(dirname "$0")/.."

MODE="${1:-linear_probe}"

if [ "$MODE" = "finetune" ]; then
    CONFIG="configs/eval/nih_finetune.yaml"
    echo "Mode: Fine-tune (encoder unfrozen)"
else
    CONFIG="configs/eval/nih_linear_probe.yaml"
    echo "Mode: Linear probe (encoder frozen)"
fi

echo "============================================"
echo "  Medical I-JEPA Classification Evaluation"
echo "  Dataset: NIH ChestX-ray14 (14 classes)"
echo "  Config: ${CONFIG}"
echo "============================================"

python main_eval_classification.py --fname "${CONFIG}"
