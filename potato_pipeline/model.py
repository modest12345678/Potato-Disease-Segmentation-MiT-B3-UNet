import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch.nn as nn
import segmentation_models_pytorch as smp

def fix_first_conv(model, in_ch=6):
    conv = next(m for m in model.encoder.modules() if isinstance(m, nn.Conv2d))
    w = conv.weight.data
    if w.shape[1] == in_ch and in_ch != 3 and w[:, 3:].abs().sum() > 0:
        pre = w[:, :3].clone() * (in_ch / 3.0)
        w.zero_()
        w[:, :3] = pre

def build_model(num_classes=4, in_ch=6, pretrained=True):
    model = smp.Unet(encoder_name="mit_b3",
                     encoder_weights="imagenet" if pretrained else None,
                     in_channels=in_ch, classes=num_classes)
    if pretrained:
        fix_first_conv(model, in_ch)
    return model