#!/bin/bash
# Evaluate lung segmentation with pretrained ViT-Large encoder
# Usage: bash scripts/eval_segmentation.sh

set -e

cd "$(dirname "$0")/.."

CONFIG="configs/eval/lung_segmentation.yaml"

echo "============================================"
echo "  Medical I-JEPA Segmentation Evaluation"
echo "  Config: ${CONFIG}"
echo "============================================"

python main_eval_segmentation.py --fname "${CONFIG}"
