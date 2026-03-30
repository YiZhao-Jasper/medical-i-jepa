import os
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import src.models.vision_transformer as vit
from src.models.heads import LinearClassifier, AttentiveClassifier
from src.datasets.nih_chestxray import make_nih_classify, NIH_14_LABELS
from src.transforms import make_eval_transforms
from src.helper import load_pretrained_encoder
from src.utils.logging import AverageMeter, CSVLogger
from src.utils.schedulers import WarmupCosineSchedule
from src.utils.metrics import compute_classification_metrics

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):

    # -- META
    model_name = args['meta']['model_name']
    patch_size = args['meta']['patch_size']
    pretrained_path = args['meta']['pretrained_path']
    freeze_encoder = args['meta']['freeze_encoder']
    use_bfloat16 = args['meta'].get('use_bfloat16', False)
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    # -- DATA
    batch_size = args['data']['batch_size']
    crop_size = args['data']['crop_size']
    num_workers = args['data']['num_workers']
    pin_mem = args['data']['pin_mem']
    root_path = args['data']['root_path']
    use_zip = args['data'].get('use_zip', True)

    # -- EVAL
    num_classes = args['eval']['num_classes']
    target_labels = args['eval'].get('target_labels', NIH_14_LABELS)

    # -- OPTIMIZATION
    num_epochs = args['optimization']['epochs']
    lr = args['optimization']['lr']
    wd = args['optimization'].get('weight_decay', 0.0)
    warmup = args['optimization'].get('warmup', 5)
    final_lr = args['optimization'].get('final_lr', 1e-6)
    start_lr = args['optimization'].get('start_lr', 1e-4)
    encoder_lr_scale = args['optimization'].get('encoder_lr_scale', 0.1)

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, 'params.yaml'), 'w') as f:
        yaml.dump(args, f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -- Build encoder
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
    )
    if pretrained_path:
        encoder = load_pretrained_encoder(encoder, pretrained_path)
        logger.info(f'Loaded pretrained encoder from {pretrained_path}')
    else:
        logger.info('No pretrained_path specified — using RANDOM initialization')
    encoder.to(device)

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        logger.info('Encoder FROZEN (linear probe mode)')
        classifier = LinearClassifier(encoder.embed_dim, num_classes).to(device)
    else:
        logger.info('Encoder UNFROZEN (fine-tune mode)')
        classifier = AttentiveClassifier(encoder.embed_dim, num_classes).to(device)

    # -- Data loaders
    train_transform = make_eval_transforms(crop_size=crop_size)
    val_transform = make_eval_transforms(crop_size=crop_size)

    _, train_loader, _ = make_nih_classify(
        root_path=root_path,
        split='train',
        transform=train_transform,
        batch_size=batch_size,
        pin_mem=pin_mem,
        num_workers=num_workers,
        use_zip=use_zip,
    )
    _, val_loader, _ = make_nih_classify(
        root_path=root_path,
        split='val',
        transform=val_transform,
        batch_size=batch_size,
        pin_mem=pin_mem,
        num_workers=num_workers,
        drop_last=False,
        use_zip=use_zip,
    )

    # -- Optimizer
    if freeze_encoder:
        params = classifier.parameters()
    else:
        params = [
            {'params': classifier.parameters(), 'lr': lr},
            {'params': encoder.parameters(), 'lr': lr * encoder_lr_scale},
        ]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup * len(train_loader),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=num_epochs * len(train_loader),
    )
    scaler = None if use_bfloat16 else torch.amp.GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()

    csv_logger = CSVLogger(
        os.path.join(folder, f'{tag}_log.csv'),
        ('%d', 'epoch'),
        ('%.5f', 'train_loss'),
        ('%.4f', 'val_auroc'),
        ('%.4f', 'val_auprc'),
    )

    best_auroc = 0.0

    for epoch in range(num_epochs):
        # -- Train
        classifier.train()
        if not freeze_encoder:
            encoder.train()
        loss_meter = AverageMeter()

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            scheduler.step()

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                if freeze_encoder:
                    with torch.no_grad():
                        features = encoder(imgs)
                else:
                    features = encoder(imgs)
                logits = classifier(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item())

        # -- Validate
        metrics = evaluate(encoder, classifier, val_loader, device, target_labels, freeze_encoder)

        csv_logger.log(
            epoch + 1,
            loss_meter.avg,
            metrics['mean_auroc'],
            metrics['mean_auprc'],
        )

        logger.info(
            f'Epoch {epoch+1}/{num_epochs} | '
            f'loss: {loss_meter.avg:.4f} | '
            f'AUROC: {metrics["mean_auroc"]:.4f} | '
            f'AUPRC: {metrics["mean_auprc"]:.4f}'
        )

        for name in (target_labels or []):
            k_auroc = f'{name}_auroc'
            k_auprc = f'{name}_auprc'
            if k_auroc in metrics:
                logger.info(f'  {name}: AUROC={metrics[k_auroc]:.4f} AUPRC={metrics[k_auprc]:.4f}')

        if metrics['mean_auroc'] > best_auroc:
            best_auroc = metrics['mean_auroc']
            save_dict = {
                'epoch': epoch + 1,
                'classifier': classifier.state_dict(),
                'metrics': metrics,
            }
            if not freeze_encoder:
                save_dict['encoder'] = encoder.state_dict()
            torch.save(save_dict, os.path.join(folder, f'{tag}-best.pth.tar'))
            logger.info(f'  >> New best AUROC: {best_auroc:.4f}')

    logger.info(f'Training complete. Best AUROC: {best_auroc:.4f}')


@torch.no_grad()
def evaluate(encoder, classifier, val_loader, device, label_names=None, freeze_encoder=True):
    classifier.eval()
    if freeze_encoder:
        encoder.eval()

    all_labels = []
    all_preds = []

    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            features = encoder(imgs)
            logits = classifier(features)

        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    return compute_classification_metrics(all_labels, all_preds, label_names)
