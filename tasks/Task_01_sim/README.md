# Task 01: sim

Sparse deconvolution for super-resolution fluorescence microscopy using Richardson-Lucy algorithm

## 📄 Paper Information

**Title**: Sparse deconvolution improves the resolution of live-cell super-resolution fluorescence microscopy

**Link**: doi:10.1038/s41587-021-01092-2

**GitHub Repository**: https://github.com/WeisongZhao/sparse-deconv-py

## 📊 Performance Metrics

- **PSNR**: 20.69 dB
- **SSIM**: 0.7439

## 📁 Directory Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── main.py              # Main reconstruction code
│   └── ...                  # Additional utilities
├── notebook/                 # Jupyter notebooks
│   └── visualization.ipynb  # Tutorial and visualization
├── data/                     # Data files (see Hugging Face)
│   ├── input.*              # Input data
│   ├── gt_output.*          # Ground truth output
│   └── recon_output.*       # Reconstruction output
├── test/                     # Test files
│   ├── agents/              # Agent files
│   ├── tests/               # Unit tests
│   ├── docs/                # Documentation
│   └── verification_utils.py # Verification utilities
├── docs/                     # Documentation
│   └── qa.json              # Q&A documentation
└── assets/                   # Visualization results
    └── vis_result.png       # Result visualization

```
