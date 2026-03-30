#!/bin/bash
# I-JEPA pretraining on NIH ChestX-ray14 with ViT-Large/14
# 1× RTX PRO 6000 (96GB) — 极限显存压榨
# Usage: bash scripts/pretrain.sh

set -e

cd "$(dirname "$0")/.."

DEVICES="cuda:0"
CONFIG="configs/pretrain/nih_vitl14.yaml"

echo "============================================"
echo "  Medical I-JEPA Pretraining"
echo "  Model: ViT-Large/14 (BF16)"
echo "  Dataset: NIH ChestX-ray14 (~109K images)"
echo "  Config: ${CONFIG}"
echo "  Device: ${DEVICES}"
echo "  GPU: RTX PRO 6000 (96GB)"
echo "  Batch: 640 (VRAM ~85%)"
echo "  LR: 0.0025 (linear scaled)"
echo "============================================"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main_pretrain.py \
    --fname "${CONFIG}" \
    --devices ${DEVICES}
