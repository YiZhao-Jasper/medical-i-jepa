import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_classification_metrics(y_true, y_score, label_names=None):
    """
    Compute per-class and mean AUROC / AUPRC for multi-label classification.

    Args:
        y_true: (N, C) ground truth binary labels
        y_score: (N, C) predicted probabilities
        label_names: optional list of class names

    Returns:
        dict with per-class and mean metrics
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_classes = y_true.shape[1]

    if label_names is None:
        label_names = [f'class_{i}' for i in range(n_classes)]

    results = {}
    aurocs = []
    auprcs = []

    for i in range(n_classes):
        gt = y_true[:, i]
        pred = y_score[:, i]

        if len(np.unique(gt)) < 2:
            results[f'{label_names[i]}_auroc'] = float('nan')
            results[f'{label_names[i]}_auprc'] = float('nan')
            continue

        auroc = roc_auc_score(gt, pred)
        auprc = average_precision_score(gt, pred)

        results[f'{label_names[i]}_auroc'] = auroc
        results[f'{label_names[i]}_auprc'] = auprc
        aurocs.append(auroc)
        auprcs.append(auprc)

    results['mean_auroc'] = np.mean(aurocs) if aurocs else float('nan')
    results['mean_auprc'] = np.mean(auprcs) if auprcs else float('nan')

    return results


def compute_segmentation_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Compute Dice and IoU for binary segmentation.

    Args:
        pred_mask: (N, H, W) predicted probabilities
        gt_mask: (N, H, W) ground truth binary masks
        threshold: binarization threshold
    """
    pred_mask = np.array(pred_mask)
    gt_mask = np.array(gt_mask)

    pred_bin = (pred_mask > threshold).astype(np.float32)
    gt_bin = gt_mask.astype(np.float32)

    intersection = (pred_bin * gt_bin).sum(axis=(1, 2))
    union = pred_bin.sum(axis=(1, 2)) + gt_bin.sum(axis=(1, 2))

    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    iou = (intersection + 1e-7) / (union - intersection + 1e-7)

    return {
        'dice_mean': float(dice.mean()),
        'dice_std': float(dice.std()),
        'iou_mean': float(iou.mean()),
        'iou_std': float(iou.std()),
    }
