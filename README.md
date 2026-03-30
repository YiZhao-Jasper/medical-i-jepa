# Medical I-JEPA

**Self-Supervised Pre-training for Chest X-ray Disease Classification via Joint-Embedding Predictive Architecture**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository implements **I-JEPA** (Image-based Joint-Embedding Predictive Architecture) for self-supervised pre-training on the **NIH ChestX-ray14** dataset using **ViT-Large/14**, with downstream evaluation via linear probing and full fine-tuning on 14-class multi-label chest disease classification.

---

## Key Results

| Method | Encoder Weights | Best AUROC | Best AUPRC | Best Epoch |
|--------|----------------|-----------|-----------|------------|
| Linear Probe | Random Init | 0.6629 | 0.0957 | 40 / 50 |
| Linear Probe | I-JEPA ep25 | 0.6659 | 0.0937 | 43 / 50 |
| Linear Probe | I-JEPA ep50 | 0.6949 | 0.1022 | 45 / 50 |
| Linear Probe | I-JEPA ep100 | 0.7355 | 0.1230 | 38 / 50 |
| **Fine-tune** | **I-JEPA ep100** | **0.7791** | **0.1724** | **6 / 20** |

> I-JEPA pre-training yields **+11.6% AUROC** improvement over random initialization baseline (0.6629 → 0.7791).

### Per-Class AUROC (Fine-tune Best, Epoch 6)

| Disease | AUROC | | Disease | AUROC |
|---------|-------|-|---------|-------|
| Edema | 0.8836 | | Fibrosis | 0.7674 |
| Cardiomegaly | 0.8798 | | Atelectasis | 0.7584 |
| Effusion | 0.8574 | | Pleural Thickening | 0.7497 |
| Pneumothorax | 0.8123 | | Mass | 0.7418 |
| Consolidation | 0.8083 | | Pneumonia | 0.7087 |
| Hernia | 0.8010 | | Infiltration | 0.6865 |
| Emphysema | 0.8009 | | Nodule | 0.6518 |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Medical I-JEPA                            │
│                                                                  │
│  Input Image ──→ [Context Encoder] ──→ Context Repr.             │
│                    (ViT-L/14, 303M)        │                     │
│                         │                  ↓                     │
│                    EMA update      [Predictor (6-layer, 11.5M)]  │
│                         ↓                  │                     │
│  Input Image ──→ [Target Encoder] ──→ Target Repr. ←── Smooth L1│
│                    (ViT-L/14, EMA)                               │
│                                                                  │
│  Masking: 1 large context block (85-100%) + 4 small pred blocks  │
│           (15-20% each), non-overlapping                         │
└──────────────────────────────────────────────────────────────────┘
```

- **Context Encoder**: ViT-Large/14 (303.2M params) — processes visible patches only
- **Target Encoder**: EMA copy of context encoder — provides prediction targets from full image
- **Predictor**: Lightweight 6-layer ViT (11.5M params) — predicts target representations at masked positions
- **Loss**: Smooth L1 in representation space (not pixel space)
- **Precision**: BFloat16 mixed precision

---

## Project Structure

```
medical-ijepa/
├── configs/
│   ├── pretrain/
│   │   └── nih_vitl14.yaml                # Pre-training config
│   └── eval/
│       ├── nih_linear_probe.yaml          # Linear probe (ep100)
│       ├── nih_lp_ep25.yaml               # Linear probe (ep25)
│       ├── nih_lp_ep50.yaml               # Linear probe (ep50)
│       ├── nih_lp_random.yaml             # Linear probe (random init)
│       ├── nih_finetune.yaml              # Fine-tuning (ep100)
│       └── lung_segmentation.yaml         # Segmentation eval
├── scripts/
│   ├── pretrain.sh                        # Pre-training launcher
│   ├── eval_classification.sh             # Classification eval launcher
│   ├── eval_segmentation.sh               # Segmentation eval launcher
│   ├── extract_dataset.sh                 # Dataset ZIP extraction
│   ├── smoke_test.py                      # Quick sanity check
│   └── ...
├── src/
│   ├── pretrain.py                        # Pre-training loop (I-JEPA)
│   ├── evaluate_classification.py         # Classification eval loop
│   ├── evaluate_segmentation.py           # Segmentation eval loop
│   ├── helper.py                          # Model init, checkpoint loading
│   ├── transforms.py                      # Image transforms
│   ├── datasets/
│   │   ├── nih_chestxray.py               # NIH ChestX-ray14 dataset
│   │   └── segmentation.py               # Segmentation dataset
│   ├── models/
│   │   ├── vision_transformer.py          # ViT encoder & predictor
│   │   └── heads.py                       # Linear / Attentive / Seg heads
│   ├── masks/
│   │   ├── multiblock.py                  # I-JEPA multi-block mask collator
│   │   └── utils.py                       # Mask apply utilities
│   └── utils/
│       ├── distributed.py                 # DDP initialization
│       ├── logging.py                     # CSV logger, GPU timer
│       ├── metrics.py                     # AUROC, AUPRC, Dice
│       ├── schedulers.py                  # LR & weight decay schedulers
│       └── tensors.py                     # Tensor utilities
├── logs/                                  # Training logs & results (see below)
│   ├── pretrain_vitl14/                   # Pre-training logs
│   ├── linear_probe/                      # LP (ep100) results
│   ├── linear_probe_ep25/                 # LP (ep25) results
│   ├── linear_probe_ep50/                 # LP (ep50) results
│   ├── linear_probe_random/               # LP (random) results
│   └── finetune/                          # Fine-tuning results
├── main_pretrain.py                       # Entry: pre-training
├── main_eval_classification.py            # Entry: classification
├── main_eval_segmentation.py              # Entry: segmentation
├── run_all_lp.sh                          # Batch run all linear probes
├── requirements.txt                       # Dependencies
├── EXPERIMENT_REPORT.md                   # Detailed experiment report
├── results_visualization.html             # Interactive result charts
├── LICENSE                                # CC BY-NC 4.0
└── .gitignore
```

---

## Requirements

### Hardware

Pre-training and downstream evaluation were conducted on different hardware:

| Stage | GPU | VRAM | Batch Size |
|-------|-----|------|------------|
| **Pre-training** (Stage 1) | 1× NVIDIA RTX Pro 6000 | 96 GB | 256 |
| **Linear Probe** (Stage 2) | 1× NVIDIA RTX 4090 | 24 GB | 256 |
| **Fine-tuning** (Stage 3) | 1× NVIDIA RTX 4090 | 24 GB | 32 |

**Minimum requirements for reproduction**:

| Component | Pre-training | Downstream Eval |
|-----------|-------------|-----------------|
| GPU VRAM | 48 GB+ (batch 256) or 24 GB (batch 64) | 24 GB |
| RAM | 64 GB | 32 GB |
| Disk | 100 GB (dataset + checkpoints) | 60 GB |

### Software

```
Python >= 3.9
PyTorch >= 2.0 (with BF16 support)
CUDA >= 11.8
```

### Installation

```bash
git clone https://github.com/<your-username>/medical-ijepa.git
cd medical-ijepa
pip install -r requirements.txt
```

---

## Dataset Preparation

### 1. Download NIH ChestX-ray14

Download from one of these sources:

- [NIH Official Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

### 2. Expected Directory Structure

```
<root_path>/
├── Data_Entry_2017.csv            # Label file (required)
├── verified_images.txt            # Verified image whitelist (recommended)
└── images/                        # Extracted PNG images (~109K files)
    ├── 00000001_000.png
    ├── 00000001_001.png
    └── ...
```

### 3. Configuration

Update the `root_path` in YAML config files to point to your data directory:

```yaml
# In configs/pretrain/nih_vitl14.yaml and configs/eval/*.yaml
data:
  root_path: /your/path/to/nih-chest-xrays-data
```

Also update `logging.folder` and `meta.pretrained_path` as needed.

---

## Usage

### Stage 1: Self-Supervised Pre-training

> **Hardware used**: 1× NVIDIA RTX Pro 6000 (96 GB). Training time: ~9 hours for 100 epochs.

```bash
# Single GPU
python main_pretrain.py --fname configs/pretrain/nih_vitl14.yaml --devices cuda:0

# Or use the launcher script
bash scripts/pretrain.sh

# Production mode with crash recovery (recommended)
nohup bash scripts/train_protected.sh &
```

**Pre-training configuration** (`configs/pretrain/nih_vitl14.yaml`):

| Parameter | Value |
|-----------|-------|
| Model | ViT-Large/14 (303.2M + 11.5M predictor) |
| GPU | 1× RTX Pro 6000 (96 GB VRAM) |
| Epochs | 100 |
| Batch size | 256 |
| Learning rate | 0.001 (cosine decay) |
| Warmup | 10 epochs |
| EMA momentum | 0.996 → 1.0 |
| Mask strategy | 1 enc block (85-100%) + 4 pred blocks (15-20%) |

Checkpoints are saved at epochs 25, 50, 75, and 100 (`jepa-ep{N}.pth.tar`), plus `jepa-latest.pth.tar` every epoch.

### Stage 2: Linear Probe Evaluation

> **Hardware used**: 1× NVIDIA RTX 4090 (24 GB). ~1 hour per 50-epoch run.

Freeze the pre-trained encoder and train a linear classification head:

```bash
# Single checkpoint
python main_eval_classification.py --fname configs/eval/nih_linear_probe.yaml

# Run all 4 LP experiments (random, ep25, ep50, ep100)
bash run_all_lp.sh
```

**Linear probe configuration**:

| Parameter | Value |
|-----------|-------|
| GPU | 1× RTX 4090 (24 GB VRAM) |
| Encoder | Frozen |
| Head | LinearClassifier (GAP + LayerNorm + Linear) |
| Epochs | 50 |
| Batch size | 256 |
| Learning rate | 0.001 |

### Stage 3: Fine-tuning Evaluation

> **Hardware used**: 1× NVIDIA RTX 4090 (24 GB). ~5 hours for 20 epochs.

Unfreeze the encoder with differential learning rates:

```bash
python main_eval_classification.py --fname configs/eval/nih_finetune.yaml
```

**Fine-tuning configuration**:

| Parameter | Value |
|-----------|-------|
| GPU | 1× RTX 4090 (24 GB VRAM) |
| Encoder | Unfrozen (LR × 0.1) |
| Head | AttentiveClassifier (cross-attention + MLP) |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 1e-4 (head), 1e-5 (encoder) |
| Weight decay | 0.05 |

### Monitoring

```bash
# Check training CSV
cat logs/finetune/finetune_log.csv

# Or for pre-training
tail -f logs/pretrain_vitl14/train.log
```

---

## Experiment Results

Full results with per-class breakdowns, training curves, and analysis are available in:

- **[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)** — Comprehensive technical report
- **[`results_visualization.html`](results_visualization.html)** — Interactive charts (open in browser)

### Training Logs Included

| Directory | Contents |
|-----------|----------|
| `logs/pretrain_vitl14/` | `params-ijepa.yaml` (config used) |
| `logs/linear_probe/` | `linear_log.csv`, `params.yaml` |
| `logs/linear_probe_ep25/` | `lp_ep25_log.csv`, `params.yaml` |
| `logs/linear_probe_ep50/` | `lp_ep50_log.csv`, `params.yaml` |
| `logs/linear_probe_random/` | `lp_random_log.csv`, `params.yaml` |
| `logs/finetune/` | `finetune_log.csv`, `params.yaml` |

> **Note**: Model checkpoints (`.pth.tar`) are not included in this repository due to their large size (1.2–4.7 GB each). All checkpoints can be reproduced by following the training instructions above.

---

## Key Findings

1. **Self-supervised pre-training is effective on medical images**: +7.3% AUROC (linear probe) and +11.6% AUROC (fine-tune) over random initialization
2. **Pre-training must be sufficient**: ep25 ≈ random (+0.3%), meaningful gains only appear after ep50 (+3.2%) and accelerate by ep100 (+7.3%)
3. **Counter-intuitive loss dynamics**: Pre-training loss *increases* from ep25 (0.009) to ep100 (0.027) due to EMA target encoder evolution, yet downstream performance *improves* — harder prediction targets yield more semantic representations
4. **Fine-tuning outperforms linear probing**: +4.4% AUROC, +40% AUPRC with AttentiveClassifier and end-to-end optimization
5. **Overfitting is the main challenge**: Best fine-tuning epoch is 6/20; performance degrades significantly after, suggesting need for early stopping and data augmentation

---

## Model Details

### ViT-Large/14

| Parameter | Value |
|-----------|-------|
| Patch size | 14 × 14 |
| Embedding dim | 1024 |
| Depth | 24 layers |
| Attention heads | 16 |
| MLP ratio | 4× |
| Position encoding | 2D sincos (fixed) |
| Encoder params | 303.2M |
| Predictor params | 11.5M |
| Input resolution | 224 × 224 |
| Patch grid | 16 × 16 (256 patches) |

### NIH ChestX-ray14

| Property | Value |
|----------|-------|
| Total images | 112,120 |
| Verified images | 108,655 |
| Disease labels | 14 |
| Label source | NLP extraction from radiology reports |
| Image format | Grayscale PNG → 3-channel |
| Original resolution | 1024 × 1024 |
| Data split | Patient-level 80/10/10 (seed=42) |
| Train / Val / Test | 87,078 / 11,013 / ~10K |

### 14 Target Diseases

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

---

## References

- Assran, M. et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." *CVPR*, 2023.
- Wang, X. et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." *CVPR*, 2017.

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Attribution-NonCommercial).
