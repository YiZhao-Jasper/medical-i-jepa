import os
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import src.models.vision_transformer as vit
from src.models.heads import SegmentationDecoder
from src.datasets.segmentation import LungSegmentationDataset
from src.helper import load_pretrained_encoder
from src.utils.logging import AverageMeter, CSVLogger
from src.utils.schedulers import WarmupCosineSchedule
from src.utils.metrics import compute_segmentation_metrics

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):

    # -- META
    model_name = args['meta']['model_name']
    patch_size = args['meta']['patch_size']
    pretrained_path = args['meta']['pretrained_path']
    freeze_encoder = args['meta'].get('freeze_encoder', True)

    # -- DATA
    batch_size = args['data']['batch_size']
    crop_size = args['data']['crop_size']
    num_workers = args['data']['num_workers']
    pin_mem = args['data']['pin_mem']
    root_path = args['data']['root_path']
    dataset_name = args['data'].get('dataset', 'montgomery')

    # -- EVAL
    num_classes = args['eval'].get('num_classes', 1)
    decoder_channels = args['eval'].get('decoder_channels', [256, 128, 64])

    # -- OPTIMIZATION
    num_epochs = args['optimization']['epochs']
    lr = args['optimization']['lr']
    wd = args['optimization'].get('weight_decay', 0.01)
    warmup = args['optimization'].get('warmup', 5)
    final_lr = args['optimization'].get('final_lr', 1e-6)
    start_lr = args['optimization'].get('start_lr', 1e-4)

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
    encoder = load_pretrained_encoder(encoder, pretrained_path)
    encoder.to(device)

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        logger.info('Encoder FROZEN for segmentation')

    grid_size = crop_size // patch_size

    decoder = SegmentationDecoder(
        embed_dim=encoder.embed_dim,
        num_classes=num_classes,
        channels=decoder_channels,
        img_size=crop_size,
    ).to(device)

    # -- Data loaders
    train_dataset = LungSegmentationDataset(
        root_path=root_path,
        dataset=dataset_name,
        split='train',
        img_size=crop_size,
    )
    val_dataset = LungSegmentationDataset(
        root_path=root_path,
        dataset=dataset_name,
        split='val',
        img_size=crop_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=False,
    )

    # -- Optimizer
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=wd)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup * len(train_loader),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=num_epochs * len(train_loader),
    )
    use_bfloat16 = args['meta'].get('use_bfloat16', False)
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = None if use_bfloat16 else torch.amp.GradScaler('cuda')

    csv_logger = CSVLogger(
        os.path.join(folder, f'{tag}_log.csv'),
        ('%d', 'epoch'),
        ('%.5f', 'train_loss'),
        ('%.4f', 'val_dice'),
        ('%.4f', 'val_iou'),
    )

    best_dice = 0.0

    for epoch in range(num_epochs):
        decoder.train()
        loss_meter = AverageMeter()

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            scheduler.step()

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                if freeze_encoder:
                    with torch.no_grad():
                        features = encoder(imgs)
                else:
                    features = encoder(imgs)
                logits = decoder(features, grid_size)
                logits = logits.squeeze(1)
                loss = dice_bce_loss(logits, masks)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_meter.update(float(loss))

        # -- Validate
        metrics = evaluate_seg(encoder, decoder, val_loader, device, grid_size, freeze_encoder)

        csv_logger.log(
            epoch + 1,
            loss_meter.avg,
            metrics['dice_mean'],
            metrics['iou_mean'],
        )

        logger.info(
            f'Epoch {epoch+1}/{num_epochs} | '
            f'loss: {loss_meter.avg:.4f} | '
            f'Dice: {metrics["dice_mean"]:.4f} ± {metrics["dice_std"]:.4f} | '
            f'IoU: {metrics["iou_mean"]:.4f}'
        )

        if metrics['dice_mean'] > best_dice:
            best_dice = metrics['dice_mean']
            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
                'metrics': metrics,
            }, os.path.join(folder, f'{tag}-best.pth.tar'))
            logger.info(f'  >> New best Dice: {best_dice:.4f}')

    logger.info(f'Training complete. Best Dice: {best_dice:.4f}')


def dice_bce_loss(logits, targets, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return bce + dice.mean()


@torch.no_grad()
def evaluate_seg(encoder, decoder, val_loader, device, grid_size, freeze_encoder=True):
    decoder.eval()
    if freeze_encoder:
        encoder.eval()

    all_preds = []
    all_masks = []

    for imgs, masks in val_loader:
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            features = encoder(imgs)
            logits = decoder(features, grid_size).squeeze(1)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_masks.append(masks.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    return compute_segmentation_metrics(all_preds, all_masks)
