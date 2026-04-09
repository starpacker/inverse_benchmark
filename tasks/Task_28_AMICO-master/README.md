# Task 28: AMICO-master

NODDI diffusion MRI parameter estimation using AMICO framework

## 📄 Paper Information

**Title**: Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data

**Link**: https://doi.org/10.1016/j.neuroimage.2014.10.026

**GitHub Repository**: https://github.com/daducci/AMICO

## 📊 Performance Metrics

- **PSNR**: 28.57 dB ← 🔧 修复前: 2.08 dB
- **SSIM**: 0.9066

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
