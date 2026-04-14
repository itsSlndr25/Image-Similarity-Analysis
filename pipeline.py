"""
圖片載入、圖像兩兩計算與矩陣儲存
-----------
Image loading, pairwise metric computation, and matrix persistence.

"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np

from config import IMG_EXTS, MATRIX_DIR
from utils.image_io import load_image_rgb
from utils.metrics  import METRICS


# ── Image collection ──────────────────────────────────────────────────────────

def collect_images(stimuli_dir: Path) -> list[Path]:
    """
    Return a sorted list of image paths found directly under stimuli_dir.

    Args:
        stimuli_dir: Directory containing the stimulus images.

    Returns:
        Sorted list of image file paths.

    Raises:
        FileNotFoundError: If the directory does not exist or contains no images.
    """
    if not stimuli_dir.is_dir(): # 檢查資料夾存在與否
        raise FileNotFoundError(f"Stimuli directory not found: {stimuli_dir}")

    paths = sorted(
        p for p in stimuli_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not paths: # 檢查資料夾內是否是空的 (len(paths) == 0)
        raise FileNotFoundError(
            f"No images found in {stimuli_dir}.\n"
            "Supported extensions: " + ", ".join(sorted(IMG_EXTS))
        )
    return paths


# ── Pairwise computation ──────────────────────────────────────────────────────

def compute_all_matrices(
    image_paths: list[Path],
    target_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute every metric in METRICS for all pairwise combinations of images.

    Args:
        image_paths: Ordered list of image file paths.
        target_size: (width, height) to resize images before comparison.
                     None = use original size (all images must share resolution).

    Returns:
        Dict mapping metric key -> (N x N) float64 array.
    """
    # 一次讀全部圖 占用memory大 超過某個數量提醒
    n = len(image_paths)
    MEMORY_WARN_THRESHOLD = 500  
    if n > MEMORY_WARN_THRESHOLD:
        warnings.warn(f"Loading {len(image_paths)} images at once may use significant memory.")
    print(f"  Loading {n} image(s) ...")
    images = [load_image_rgb(p, size=target_size) for p in image_paths]

    # Initialise NaN matrices; filled in as pairs are computed
    matrices = {key: np.full((n, n), np.nan, dtype=np.float64) for key in METRICS}

    # ── Off-diagonal pairs (matrix is symmetric) ──────────────────────────────
    pairs = list(itertools.combinations(range(n), 2))
    total = len(pairs)
    report_every = max(1, total // 10) # 每幾個 pair 印一次進度

    for idx, (i, j) in enumerate(pairs, start=1):
        if idx % report_every == 0 or idx == total:
            print(f"    pair {idx}/{total} ...")

        for key, (fn, _, _) in METRICS.items():
            try:
                v = fn(images[i], images[j])
            except Exception as exc:
                warnings.warn(f"[{key}] pair ({i},{j}) failed: {exc}")
                v = float("nan")
            # Fill both (i,j) and (j,i) since the matrix is symmetric
            matrices[key][i, j] = v
            matrices[key][j, i] = v

    # ── Diagonal (self-comparison) ─────────────────────────────────────────────
    for key, (fn, _, _) in METRICS.items():
        for i in range(n):
            try:
                matrices[key][i, i] = fn(images[i], images[i])
            except Exception:
                matrices[key][i, i] = float("nan")

    return matrices


# ── Save & Load ───────────────────────────────────────────────────────────────

def save_matrices(
    matrices:   dict[str, np.ndarray],
    mag:        int,
    matrix_dir: Path = MATRIX_DIR,
) -> None:
    """
    Save each matrix as a .npy file using the naming convention:
    <key>_matrix_1_1x<mag>.npy

    Args:
        matrices:   Dict of metric key -> matrix to save.
        mag:        Magnification level used in the filename.
        matrix_dir: Output directory.
    """
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for key, mat in matrices.items():
        path = matrix_dir / f"{key}_matrix_1_1x{mag}.npy"
        np.save(path, mat)
        print(f"  Saved  {path}")


def load_matrices(
    mag:        int,
    matrix_dir: Path = MATRIX_DIR,
) -> dict[str, np.ndarray]:
    """
    Load all previously saved matrices for a given magnification level.

    Args:
        mag:        Magnification level used in the filename.
        matrix_dir: Directory where the .npy files are stored.

    Returns:
        Dict mapping metric key -> loaded matrix.
        Missing files are skipped with a warning.
    """
    result: dict[str, np.ndarray] = {}
    for key in METRICS:
        path = matrix_dir / f"{key}_matrix_1_1x{mag}.npy"
        if not path.exists():
            warnings.warn(f"Matrix file not found, skipping: {path}")
            continue
        result[key] = np.load(path)
    return result
