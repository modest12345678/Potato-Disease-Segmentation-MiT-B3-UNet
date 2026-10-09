import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch

def confusion_from_logits(logits, target, num_classes=4, ignore=255):
    pred = logits.argmax(1)
    valid = target != ignore
    pred, tgt = pred[valid], target[valid]
    cm = torch.bincount(num_classes * tgt + pred, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes).cpu()

def per_class_iou_dice(cm):
    cm = cm.float()
    inter = torch.diag(cm)
    gt, pr = cm.sum(1), cm.sum(0)
    union = gt + pr - inter
    iou = inter / union.clamp(min=1)
    dice = 2 * inter / (gt + pr).clamp(min=1)
    return iou, dice