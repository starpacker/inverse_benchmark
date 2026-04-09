# Task 15: lensless_admm

Lensless imaging reconstruction using ADMM (Alternating Direction Method of Multipliers)

## 📄 Paper Information

**Title**: LenslessPiCam: A Hardware and Software Platform for Lensless Computational Imaging with a Raspberry Pi

**Link**: doi:10.21105/joss.04747

**GitHub Repository**: https://github.com/LCAV/LenslessPiCam

## 📊 Performance Metrics

- **PSNR**: 8.74 dB
- **SSIM**: 0.1039

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
