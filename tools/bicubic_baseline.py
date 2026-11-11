"""Sanity check of the evaluation protocol: bicubic x4 upsampling of the benchmark LR
images. Literature values (x4, Y channel): Set5 28.42/0.8104, Set14 26.00/0.7027,
BSD100 25.96/0.6675, Urban100 23.14/0.6577."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csea.imresize import imresize
from csea.metrics import psnr_y, ssim_y

s = 4
root = Path(__file__).resolve().parents[1] / "data/processed/bench"
for name in ("Set5", "Set14", "BSD100", "Urban100"):
    ps, ss = [], []
    for hp in sorted((root / name / "HR").glob("*.png")):
        hr = np.array(Image.open(hp).convert("RGB"))
        lr = np.array(Image.open(root / name / f"LRx{s}" / hp.name).convert("RGB"))
        t = torch.from_numpy(lr).permute(2, 0, 1).double().cuda() / 255
        up = (imresize(t, s).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        ps.append(psnr_y(up, hr, s)); ss.append(ssim_y(up, hr, s))
    print(f"{name:9s} bicubic PSNR {np.mean(ps):.2f} SSIM {np.mean(ss):.4f} ({len(ps)} imgs)")
