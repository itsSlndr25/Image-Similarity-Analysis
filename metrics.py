"""
utils/metrics.py
----------------
6 perceptual/structural image-comparison metrics.

All compute_* functions accept two uint8 RGB np.ndarrays of identical
shape (H, W, 3) and return a single Python float.

To add a new metric:
  1. Write a compute_<n>(a, b) -> float function below.
  2. Add an entry to METRICS at the bottom of this file.
"""

from __future__ import annotations

import warnings

import numpy as np
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr

from .image_io import to_gray, get_hue_channel

# ── Optional heavy dependencies ───────────────────────────────────────────────
# 避免安裝lpips時 crash的措施
try:
    import torch
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    warnings.warn(
        "lpips / torch not found – the LPIPS metric will return NaN.\n"
        "Install with:  pip install lpips torch torchvision",
        stacklevel=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM), averaged over all three channels.

    Range: [-1, 1]; 1 = identical.
    """
    return float(_ssim(a, b, channel_axis=-1, data_range=255)) # data_range → 圖片pixel的範圍


def compute_l2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Root-mean-squared Euclidean (L2) distance, per pixel per channel.

    Range: [0, 255]; 0 = identical.
    """
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt((diff ** 2).mean()))


def compute_hue_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Circular (angular) correlation between the HSV hue channels of two images.

    Standard Pearson correlation cannot handle the circular nature of hue
    (0° and 360° are the same colour), so we use the sine-based formula:

        r = Σ sin(α − ᾱ) sin(β − β̄)
            ─────────────────────────────────────────
            √[ Σ sin²(α − ᾱ) · Σ sin²(β − β̄) ]

    Range: [-1, 1]; 1 = perfectly co-varying hue.
    """
    ha, hb = get_hue_channel(a), get_hue_channel(b)

    # Circular mean via arctan2 to handle the 0°/360° discontinuity
    mean_a = np.arctan2(np.mean(np.sin(ha)), np.mean(np.cos(ha)))
    mean_b = np.arctan2(np.mean(np.sin(hb)), np.mean(np.cos(hb)))

    da = np.sin(ha - mean_a)
    db = np.sin(hb - mean_b)

    denom = np.sqrt((da ** 2).sum() * (db ** 2).sum()) + 1e-12  # avoid division by zero / 避免除以零
    return float((da * db).sum() / denom)


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.

    Returns np.inf when the two images are identical (MSE = 0).
    Range: [0, ∞); higher = more similar.
    """
    return float(_psnr(a, b, data_range=255))


def compute_nlpd(a: np.ndarray, b: np.ndarray, n_levels: int = 6) -> float:
    """
    Normalised Laplacian Pyramid Distance (NLPD).

    Builds a Laplacian pyramid on the luminance channel. Each sub-band is
    divisively normalised by local standard deviation (3x3 window), modelling
    contrast-gain control in early visual cortex.

    Range: [0, ∞); 0 = identical. Lower = more similar.

    Reference:
        Laparra et al., "Perceptual Image Quality Assessment using a
        Normalised Laplacian Pyramid", HVEI 2016.
    """
    from scipy.ndimage import uniform_filter, zoom

    def _gaussian_pyramid(img, levels):
        pyr, cur = [img], img
        for _ in range(levels - 1):
            # Blur then downsample by 2 at each level
            cur = uniform_filter(cur, size=5)[::2, ::2]
            pyr.append(cur)
        return pyr

    def _laplacian_pyramid(img, levels):
        g = _gaussian_pyramid(img, levels)
        lp = []
        for k in range(levels - 1):
            up = zoom(g[k + 1], zoom=2, order=1)
            h, w = g[k].shape
            # Residual between current level and upsampled coarser level
            lp.append(g[k] - up[:h, :w])
        lp.append(g[-1])  # coarsest level kept as-is / 最粗層直接保留
        return lp

    def _normalise(band):
        # Divisive normalisation: divide by local std to mimic cortical processing
        mu    = uniform_filter(band,      size=3)
        mu2   = uniform_filter(band ** 2, size=3)
        sigma = np.sqrt(np.maximum(mu2 - mu ** 2, 0.0)) + 1.0  # +1 avoids /0 / +1 避免除以零
        return band / sigma

    # Convert to luminance before building the pyramid
    gray_a = to_gray(a) * 255.0
    gray_b = to_gray(b) * 255.0

    total_sq, total_n = 0.0, 0
    for la, lb in zip(_laplacian_pyramid(gray_a, n_levels),
                      _laplacian_pyramid(gray_b, n_levels)):
        na, nb = _normalise(la), _normalise(lb)
        total_sq += float(np.sum((na - nb) ** 2))
        total_n  += na.size

    return float(np.sqrt(total_sq / max(total_n, 1)))


# ── LPIPS ─────────────────────────────────────────────────────────────────────
_lpips_model_cache = None


def _get_lpips_model():
    """Lazy-load the LPIPS AlexNet model (cached after first call)."""
    global _lpips_model_cache
    if _lpips_model_cache is None:
        _lpips_model_cache = _lpips_lib.LPIPS(net="alex", verbose=False)
        _lpips_model_cache.eval()
    return _lpips_model_cache


def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) with AlexNet backbone.

    Requires lpips and torch; returns NaN if unavailable.
    Range: [0, ~1]; 0 = identical. Lower = more similar (perceptually).
    """
    if not _LPIPS_AVAILABLE:
        return float("nan")

    def _to_tensor(rgb):
        # Rescale uint8 [0, 255] to float [-1, 1] as expected by LPIPS
        t = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)

    model = _get_lpips_model()
    with torch.no_grad():
        score = model(_to_tensor(a), _to_tensor(b))
    return float(score.item())


# ─────────────────────────────────────────────────────────────────────────────
# Metric catalogue
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: key -> (function, heatmap title, colorbar label)
# This dict is iterated by pipeline.py and visualise.py.

METRICS: dict[str, tuple] = {
    "ssim":      (compute_ssim,     "Structural Similarity Index (SSIM)", "SSIM score"),
    "euclidean": (compute_l2,       "Euclidean (L2) Distance",            "L2 distance"),
    "hue_corr":  (compute_hue_corr, "Hue Circular Correlation",           "Circular correlation"),
    "psnr":      (compute_psnr,     "Peak Signal-to-Noise Ratio (PSNR)",  "PSNR (dB)"),
    "lpips":     (compute_lpips,    "LPIPS Perceptual Similarity",        "LPIPS score"),
    "nlpd":      (compute_nlpd,     "Normalised Laplacian Pyramid Dist.", "NLPD score"),
}
