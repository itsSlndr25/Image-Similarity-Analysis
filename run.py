"""
主程式
------
For the image-comparison pipeline.


Reads images from ./stimuli/, writes matrices to ./img_compare_matrix/,
and saves heatmap PDFs to ./img_compare_hmap/.
"""

from pathlib import Path

from config   import STIMULI_DIR, MATRIX_DIR, HMAP_DIR, MAG_LIST, TICK_LABEL_MAX_LEN
from pipeline import collect_images, compute_all_matrices, save_matrices
from utils    import visualise_all


def main() -> None:
    print("=" * 60)
    print("Image Comparison Metrics Pipeline")
    print("=" * 60)

    # ── 1. Collect source images ──────────────────────────────────────────────
    print(f"\n[1/3]  Scanning {STIMULI_DIR} for images ...")
    try:
        image_paths = collect_images(STIMULI_DIR)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        return

    print(f"  Found {len(image_paths)} image(s):")
    for p in image_paths:
        print(f"    {p.name}")

    # Truncate filenames for use as heatmap tick labels
    tick_labels = [p.stem[:TICK_LABEL_MAX_LEN] for p in image_paths]

    # ── 2. Compute & save matrices ────────────────────────────────────────────
    print(f"\n[2/3]  Computing pairwise metrics ...")
    for mag in MAG_LIST:
        print(f"\n  -- magnification x{mag} --")
        # target_size can be set here if resizing per magnification is needed
        matrices = compute_all_matrices(image_paths, target_size=None)
        save_matrices(matrices, mag, matrix_dir=MATRIX_DIR)

    # ── 3. Visualise ──────────────────────────────────────────────────────────
    print(f"\n[3/3]  Generating heatmaps ...")
    for mag in MAG_LIST:
        print(f"\n  -- magnification x{mag} --")
        visualise_all(
            mag=mag,
            tick_labels=tick_labels,
            matrix_dir=MATRIX_DIR,
            hmap_dir=HMAP_DIR,
        )

    print("\nDone!")
    print(f"  Matrices : {MATRIX_DIR.resolve()}")
    print(f"  Heatmaps : {HMAP_DIR.resolve()}")


if __name__ == "__main__":
    main()
