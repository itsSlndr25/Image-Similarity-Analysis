"""
比較圖視覺化
------------------
Heatmap rendering for pairwise image-comparison matrices.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts / 無頭後端，適合腳本執行
import matplotlib.pyplot as plt

from ..config import MATRIX_DIR, HMAP_DIR, CMAP, PLOT_DPI
from .metrics import METRICS


def plot_heatmap(
    matrix:      np.ndarray,
    title:       str,
    cbar_label:  str,
    out_path:    Path,
    tick_labels: list[str] | None = None,
) -> None:
    """
    Render a single NxN heatmap and save it to out_path as a PDF.

    Args:
        matrix:      Square similarity/distance matrix.
        title:       Figure title shown above the heatmap.
        cbar_label:  Label for the colour-bar axis.
        out_path:    Destination file path (PDF).
        tick_labels: Optional list of N strings for axis tick labels.
    """
    n = matrix.shape[0]

    fig, ax = plt.subplots(
        1, 1, figsize=(5, 4.5), dpi=PLOT_DPI,
        gridspec_kw={"top": 0.88, "bottom": 0.22, "left": 0.20, "right": 0.96},
    )

    # Replace inf with finite maximum for display (e.g. PSNR of identical images)
    disp = matrix.copy()
    finite = disp[np.isfinite(disp)]
    finite_max = float(np.nanmax(finite)) if finite.size else 0.0
    disp = np.where(np.isinf(disp), finite_max, disp)

    vmin = float(np.nanmin(disp))
    vmax = float(np.nanmax(disp))

    im   = ax.imshow(disp, cmap=CMAP, interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=9, pad=6)
    ax.set_ylabel("Source images", fontsize=8)
    ax.set_xlabel("Target images", fontsize=8)

    ticks = list(range(n))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if tick_labels:
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)
    else:
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_yticklabels(ticks, fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {out_path}")


def visualise_all(
    mag:         int,
    tick_labels: list[str] | None = None,
    matrix_dir:  Path = MATRIX_DIR,
    hmap_dir:    Path = HMAP_DIR,
) -> None:
    """
    Load every saved matrix for a given magnification and render one heatmap per metric.

    Args:
        mag:        Magnification level used to locate the .npy files.
        tick_labels: Optional list of N strings for axis tick labels.
        matrix_dir: Directory where the .npy files are stored.
        hmap_dir:   Directory where the PDF heatmaps will be saved.
    """
    for key, (_, title, cbar_label) in METRICS.items():
        fname = matrix_dir / f"{key}_matrix_1_1x{mag}.npy"
        if not fname.exists():
            warnings.warn(f"Matrix file not found, skipping: {fname}")
            continue

        matrix   = np.load(fname)
        out_path = hmap_dir / f"{fname.stem}.pdf"

        plot_heatmap(
            matrix=matrix,
            title=f"{title}\n(mag x{mag})",
            cbar_label=cbar_label,
            out_path=out_path,
            tick_labels=tick_labels,
        )
