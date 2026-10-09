import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch, torch.nn as nn, torch.nn.functional as F

class SegLoss(nn.Module):
    def __init__(self, class_weights=None, ce_w=1.0, dice_w=1.0, ignore=255, eps=1e-6):
        super().__init__()
        self.ce_w, self.dice_w, self.ignore, self.eps = ce_w, dice_w, ignore, eps
        self.register_buffer("w", class_weights if class_weights is not None
                             else torch.ones(4), persistent=False)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.w, ignore_index=self.ignore)
        valid = (target != self.ignore)
        t = target.clone(); t[~valid] = 0
        v = valid.unsqueeze(1).float()
        p = logits.softmax(1) * v
        oh = F.one_hot(t, logits.shape[1]).permute(0, 3, 1, 2).float() * v
        inter = (p * oh).sum(dim=(0, 2, 3))
        union = p.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        wd = self.w / self.w.sum()
        return self.ce_w * ce + self.dice_w * ((1 - dice) * wd).sum()