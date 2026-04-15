# Image Similarity Evaluation Pipeline

## 📋 Overview

This pipeline is designed to:
- Compare multiple images in a dataset
- Compute pairwise similarity metrics
- Generate similarity matrices
- Visualize results as heatmaps
- Save structured outputs for downstream analysis


## 💡 Technical Highlights

- Modular pipeline design
- Multi-metric evaluation (classical + perceptual)
- Matrix-level analysis for dataset comparison
- Clean separation of computation and visualization
- Designed for extensibility and experimental flexibility


## 🏗️ Architecture
```
img_compare/
├── config.py          ← paths & constants
├── pipeline.py        ← image loading, metric computation, matrix generation
├── run.py             ← execute full pipeline
├── image_io.py        ← image loading & color space processing
├── metrics.py         ← functions of the metrics
├── visualization.py   ← heatmap & result plotting
├── 影像相似度分析.pdf  ← summary *chinese version
└── Image Similarity Analysis.pdf ← summary of this project 
```

## 📊 Implemented Metrics

SSIM – Structural Similarity Index
L2 Norm – Pixel-wise Euclidean distance
PSNR – Peak Signal-to-Noise Ratio
Hue Correlation – Color-based similarity
NLPD – Normalized Laplacian Pyramid Distance
LPIPS – Learned Perceptual Image Patch Similarity

* The framework supports both traditional pixel-based metrics and deep feature–based perceptual metrics.

## 📖 How It Works

- Load images from dataset
- Convert color formats if needed (RGB, grayscale, hue channel)
- Compute selected similarity metrics
- Construct similarity matrix
- Save matrix as .npy
- Generate heatmap visualization

## Output

- similarity_matrix.npy - Pairwise similarity matrix
- heatmap.png - Visual representation of similarity distribution
