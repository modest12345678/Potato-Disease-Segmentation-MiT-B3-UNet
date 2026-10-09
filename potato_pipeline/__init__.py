import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""Potato Disease Segmentation Pipeline (MiT-B3 + U-Net)"""
