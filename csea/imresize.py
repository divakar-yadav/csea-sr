"""MATLAB-compatible bicubic imresize (with antialiasing), in PyTorch.

SR benchmarks generate LR images with MATLAB's `imresize(img, 1/s, 'bicubic')`.
PIL/OpenCV bicubic differ slightly, which shifts PSNR by ~0.1 dB, so we
re-implement MATLAB's algorithm exactly (cubic a=-0.5, antialias kernel widening,
symmetric boundary handling).
"""
import math

import torch


def _cubic(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    ax2, ax3 = ax ** 2, ax ** 3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1) +
            (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2))).to(x.dtype)


def _contributions(in_len: int, out_len: int, scale: float, antialias: bool = True,
                   kernel_width: float = 4.0, device="cpu"):
    if scale < 1 and antialias:
        kernel = lambda t: scale * _cubic(scale * t)
        kernel_width = kernel_width / scale
    else:
        kernel = _cubic
    x = torch.arange(1, out_len + 1, dtype=torch.float64, device=device)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)
    p = int(math.ceil(kernel_width)) + 2
    ind = left[:, None] + torch.arange(p, dtype=torch.float64, device=device)[None]
    w = kernel(u[:, None] - ind)
    w = w / w.sum(1, keepdim=True)
    aux = torch.cat([torch.arange(1, in_len + 1, device=device),
                     torch.arange(in_len, 0, -1, device=device)])
    ind = aux[torch.remainder(ind.long() - 1, aux.numel())] - 1  # 0-based, symmetric pad
    keep = w.abs().sum(0) > 0
    return w[:, keep], ind[:, keep]


def _resize_dim(x: torch.Tensor, dim: int, out_len: int, scale: float) -> torch.Tensor:
    w, ind = _contributions(x.shape[dim], out_len, scale, device=x.device)
    x = x.movedim(dim, -1)                       # (..., in_len)
    g = x[..., ind]                              # (..., out_len, P)
    out = (g * w.to(x.dtype)).sum(-1)
    return out.movedim(-1, dim)


def imresize(img: torch.Tensor, scale: float) -> torch.Tensor:
    """img: (..., C, H, W) float tensor (any range). Returns resized float tensor.
    Output size = ceil(H*scale), ceil(W*scale) as in MATLAB."""
    h, w = img.shape[-2:]
    oh, ow = int(math.ceil(h * scale)), int(math.ceil(w * scale))
    x = img.double()
    x = _resize_dim(x, x.dim() - 2, oh, scale)
    x = _resize_dim(x, x.dim() - 1, ow, scale)
    return x.to(img.dtype if img.is_floating_point() else torch.float64)


def bicubic_downsample_uint8(hr_uint8_hwc, scale: int, device="cpu"):
    """HR uint8 HxWx3 numpy array -> LR uint8 numpy array (MATLAB protocol)."""
    import numpy as np
    t = torch.from_numpy(np.ascontiguousarray(hr_uint8_hwc)).to(device).permute(2, 0, 1)
    lr = imresize(t.double() / 255.0, 1.0 / scale)
    lr = (lr.clamp(0, 1) * 255.0).round().byte()   # im2uint8
    return lr.permute(1, 2, 0).cpu().numpy()
