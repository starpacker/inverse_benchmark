# Task 51: tomopy-master

CT reconstruction using TomoPy with FBP and SIRT algorithms

## 📄 Paper Information

**Title**: TomoPy: A framework for the analysis of synchrotron tomographic data

**Link**: https://doi.org/10.1107/S1600577514013939

**GitHub Repository**: https://github.com/tomopy/tomopy

## 📊 Performance Metrics

- **PSNR**: 24.12 dB ← 🔧 修复前: 15.14 dB
- **SSIM**: 0.9132

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
