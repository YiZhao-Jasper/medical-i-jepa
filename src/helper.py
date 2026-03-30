import logging
import sys
from collections import OrderedDict

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _adapt_state_dict(state_dict, model):
    """Adapt checkpoint keys between DDP (module.*) and non-DDP formats."""
    model_keys = set(model.state_dict().keys())
    has_module = any(k.startswith('module.') for k in model_keys)
    ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())

    if has_module == ckpt_has_module:
        return state_dict

    adapted = OrderedDict()
    for k, v in state_dict.items():
        if has_module and not ckpt_has_module:
            adapted[f'module.{k}'] = v
        elif not has_module and ckpt_has_module:
            adapted[k.replace('module.', '', 1)] = v
        else:
            adapted[k] = v
    return adapted


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        pretrained_dict = _adapt_state_dict(checkpoint['encoder'], encoder)
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        pretrained_dict = _adapt_state_dict(checkpoint['predictor'], predictor)
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        if target_encoder is not None:
            pretrained_dict = _adapt_state_dict(checkpoint['target_encoder'], target_encoder)
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained target_encoder from epoch {epoch} with msg: {msg}')

        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def load_pretrained_encoder(
    encoder,
    pretrained_path,
    strict=False,
):
    """Load encoder weights from a pretrained I-JEPA checkpoint for downstream tasks."""
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    if 'target_encoder' in checkpoint:
        state_dict = checkpoint['target_encoder']
    elif 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
    else:
        state_dict = checkpoint

    cleaned = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('backbone.', '')
        cleaned[name] = v

    msg = encoder.load_state_dict(cleaned, strict=strict)
    logger.info(f'loaded pretrained encoder from {pretrained_path}: {msg}')
    return encoder


def init_model(
    device,
    patch_size=14,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
    )
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)
    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
        },
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = None if use_bfloat16 else torch.amp.GradScaler('cuda')
    return optimizer, scaler, scheduler, wd_scheduler
