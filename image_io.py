"""
圖片讀取與色彩空間轉換
-----------------
Image I/O and colour-space helpers.
All functions operate on uint8 RGB numpy arrays of shape (H, W, 3)
"""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_rgb(
    path: Path, # type hint
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Load an image as a uint8 RGB array.

    Args:
        path: Image file to open.
        size: (width, height). If given, the image is resized with
              Lanczos resampling before returning.

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8.
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS) # NEAREST,BILINEAR,BICUBIC,LANCZOS
    return np.asarray(img)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB array to a luminance channel using sRGB / ITU-R BT.709 weights.

    Args:
        rgb: Shape (H, W, 3), dtype uint8.

    Returns:
        Shape (H, W), dtype float64, values in [0, 1].
    """
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    return (rgb.astype(np.float64) @ weights) / 255.0


def get_hue_channel(rgb: np.ndarray) -> np.ndarray:
    """
    Extract the HSV hue channel and convert it to radians.
    Computed from scratch to avoid additional dependencies (cv2 / colorsys).

    Args:
        rgb: Shape (H, W, 3), dtype uint8.

    Returns:
        Flattened hue array, shape (H*W,), dtype float32, values in [0, 2π).
    """
    # [0, 255] 縮放到 [0, 1]
    r, g, b = (rgb[..., c].astype(np.float32) / 255.0 for c in range(3))

    # 找出三個 channel的 Max & Min
    cmax  = np.maximum(np.maximum(r, g), b)
    cmin  = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-9  # avoid division by zero

    hue = np.zeros_like(r)

    # Hue formula differs depending on which channel is dominant
    mask_r = cmax == r
    mask_g = cmax == g
    mask_b = cmax == b

    hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0 # r section：-60° ~ 60°
    hue[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) % 360.0 # g section：60° ~ 180°
    hue[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) % 360.0 # b section：180° ~ 300°

    return np.deg2rad(hue).ravel() # 轉成弧度(0~2π）後return
