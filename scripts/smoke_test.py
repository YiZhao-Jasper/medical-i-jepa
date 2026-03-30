"""
Smoke test: verify model builds, forward pass works, all imports resolve.
Run without any data to validate the engineering is correct.

Usage: python scripts/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

def test_encoder_builds():
    """Test all ViT variants build correctly."""
    import src.models.vision_transformer as vit
    print('[1/6] Testing encoder builds...')

    for name, expected_dim in [
        ('vit_tiny', 192), ('vit_small', 384), ('vit_base', 768),
        ('vit_large', 1024), ('vit_huge', 1280),
    ]:
        model = vit.__dict__[name](img_size=[224], patch_size=14)
        assert model.embed_dim == expected_dim, f'{name} embed_dim mismatch'
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  {name}: embed_dim={model.embed_dim}, params={n_params:.1f}M')

    print('  PASSED')


def test_pretrain_forward():
    """Test full pretrain forward pass with ViT-B/14."""
    from src.helper import init_model
    from src.masks.multiblock import MaskCollator
    print('[2/6] Testing pretrain forward pass...')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder, predictor = init_model(
        device=device,
        patch_size=14,
        model_name='vit_base',
        crop_size=224,
        pred_depth=6,
        pred_emb_dim=384,
    )
    target_encoder = torch.nn.Module()
    target_encoder.__dict__.update(
        {k: v for k, v in encoder.__dict__.items()})
    import copy
    target_encoder = copy.deepcopy(encoder)

    mask_collator = MaskCollator(
        input_size=224, patch_size=14,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4,
        allow_overlap=False, min_keep=10,
    )

    B = 4
    imgs = torch.randn(B, 3, 224, 224, device=device)
    dummy_batch = [(imgs[i], i) for i in range(B)]
    collated, masks_enc, masks_pred = mask_collator(dummy_batch)

    imgs_batch = collated[0].to(device)
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]

    from src.masks.utils import apply_masks
    from src.utils.tensors import repeat_interleave_batch

    with torch.no_grad():
        h = target_encoder(imgs_batch)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

    z = encoder(imgs_batch, masks_enc)
    z = predictor(z, masks_enc, masks_pred)

    loss = F.smooth_l1_loss(z, h)
    loss.backward()

    print(f'  loss={loss.item():.4f}, z.shape={z.shape}, h.shape={h.shape}')
    print('  PASSED')


def test_classification_heads():
    """Test classification heads."""
    from src.models.heads import LinearClassifier, AttentiveClassifier
    print('[3/6] Testing classification heads...')

    B, N, D = 4, 256, 1024

    tokens = torch.randn(B, N, D)

    lc = LinearClassifier(D, 14)
    out = lc(tokens)
    assert out.shape == (B, 14), f'LinearClassifier output shape wrong: {out.shape}'

    ac = AttentiveClassifier(D, 14)
    out = ac(tokens)
    assert out.shape == (B, 14), f'AttentiveClassifier output shape wrong: {out.shape}'

    print('  PASSED')


def test_segmentation_decoder():
    """Test segmentation decoder."""
    from src.models.heads import SegmentationDecoder
    print('[4/6] Testing segmentation decoder...')

    B, D = 4, 1024
    grid_size = 16
    N = grid_size * grid_size

    tokens = torch.randn(B, N, D)
    decoder = SegmentationDecoder(D, num_classes=1, img_size=224)
    out = decoder(tokens, grid_size)
    assert out.shape == (B, 1, 224, 224), f'Decoder output shape wrong: {out.shape}'

    print(f'  output shape: {out.shape}')
    print('  PASSED')


def test_metrics():
    """Test metric computation."""
    from src.utils.metrics import compute_classification_metrics, compute_segmentation_metrics
    import numpy as np
    print('[5/6] Testing metrics...')

    n_cls = 14
    y_true = np.random.randint(0, 2, (20, n_cls)).astype(float)
    y_score = np.random.rand(20, n_cls)
    labels = [f'Disease_{i}' for i in range(n_cls)]
    result = compute_classification_metrics(y_true, y_score, labels)
    print(f'  mean_auroc={result["mean_auroc"]:.4f}, mean_auprc={result["mean_auprc"]:.4f}')

    pred = np.random.rand(10, 224, 224)
    gt = (np.random.rand(10, 224, 224) > 0.5).astype(float)
    seg_result = compute_segmentation_metrics(pred, gt)
    print(f'  dice={seg_result["dice_mean"]:.4f}, iou={seg_result["iou_mean"]:.4f}')

    print('  PASSED')


def test_transforms():
    """Test transform pipelines."""
    from src.transforms import make_pretrain_transforms, make_eval_transforms
    from PIL import Image
    import numpy as np
    print('[6/6] Testing transforms...')

    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    t_pretrain = make_pretrain_transforms(crop_size=224)
    out = t_pretrain(dummy_img)
    assert out.shape == (3, 224, 224), f'Pretrain transform shape wrong: {out.shape}'

    t_eval = make_eval_transforms(crop_size=224)
    out = t_eval(dummy_img)
    assert out.shape == (3, 224, 224), f'Eval transform shape wrong: {out.shape}'

    print('  PASSED')


if __name__ == '__main__':
    print('=' * 50)
    print('  Medical I-JEPA Smoke Test')
    print('=' * 50)

    test_encoder_builds()
    test_pretrain_forward()
    test_classification_heads()
    test_segmentation_decoder()
    test_metrics()
    test_transforms()

    print('=' * 50)
    print('  ALL TESTS PASSED')
    print('=' * 50)
