"""
config.py
---------
Central configuration for the image comparison pipeline. #全域設定
All paths, constants, and pipeline parameters are defined here.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
STIMULI_DIR  = Path("./stimuli")             # input images / 輸入圖片資料夾
MATRIX_DIR   = Path("./img_compare_matrix")  # saved .npy matrices / 儲存的矩陣
HMAP_DIR     = Path("./img_compare_hmap")    # output PDF heatmaps / 輸出

# ── Image loading ─────────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── Pipeline settings ─────────────────────────────────────────────────────────
# Magnification levels used for output file naming 
MAG_LIST: list[int] = [2, 3, 4]

# Maximum characters shown per image name in heatmap tick labels
TICK_LABEL_MAX_LEN: int = 10

# ── Visualisation ─────────────────────────────────────────────────────────────
CMAP     = "viridis" # heatmap style
PLOT_DPI = 300
