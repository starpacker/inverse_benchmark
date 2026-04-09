# Task 95: eispy2d

2D electrical impedance imaging using Gauss-Newton reconstruction

## 📄 Paper Information

**Title**: EISPY2D: An Open-Source Python Library for the Development and Comparison of Algorithms in Two-Dimensional Electromagnetic Inverse Scattering Problems

**Link**: doi:10.1109/ACCESS.2025.3573679

**GitHub Repository**: https://github.com/andre-batista/eispy2d

## 📊 Performance Metrics

- **PSNR**: 22.15 dB
- **SSIM**: 0.7819

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
