"""PSNR / SSIM on the Y channel, following the standard SR evaluation protocol
(MATLAB rgb2ycbcr, crop `scale` pixels at each border, uint8-quantised output)."""
import numpy as np
import torch
import torch.nn.functional as F


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """img: HxWx3 in [0,255] (any dtype) -> Y in [16,235] float64 (MATLAB rgb2ycbcr)."""
    img = img.astype(np.float64) / 255.0
    return 16.0 + 65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]


def _shave(x, b):
    return x[b:-b, b:-b] if b > 0 else x


def psnr_y(sr: np.ndarray, hr: np.ndarray, border: int) -> float:
    a, b = _shave(rgb_to_y(sr), border), _shave(rgb_to_y(hr), border)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _gauss_window(size=11, sigma=1.5):
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    w = np.outer(g, g)
    return w / w.sum()


def ssim_y(sr: np.ndarray, hr: np.ndarray, border: int) -> float:
    """MATLAB-style SSIM (Wang et al. 2004): 11x11 Gaussian, sigma 1.5, 'valid' filtering."""
    a = torch.from_numpy(_shave(rgb_to_y(sr), border))[None, None]
    b = torch.from_numpy(_shave(rgb_to_y(hr), border))[None, None]
    w = torch.from_numpy(_gauss_window())[None, None]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    s_aa = F.conv2d(a * a, w) - mu_a ** 2
    s_bb = F.conv2d(b * b, w) - mu_b ** 2
    s_ab = F.conv2d(a * b, w) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return float(m.mean())
