# Task 61: DENSS

Small-angle X-ray scattering 3D reconstruction using DENSS

## 📄 Paper Information

**Title**: refellips: Ellipsometry data analysis in Python

**Link**: https://doi.org/10.1016/j.softx.2022.101225

**GitHub Repository**: https://github.com/refnx/refellips

## 📊 Performance Metrics

- **PSNR**: 23.76 dB ← 🔧 修复前: 13.04 dB
- **SSIM**: 0.760

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
