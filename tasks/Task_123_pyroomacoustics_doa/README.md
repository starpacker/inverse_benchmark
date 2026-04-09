# Task 123: pyroomacoustics_doa

Direction-of-arrival estimation using MUSIC algorithm

## 📄 Paper Information

**Title**: Pyroomacoustics: A Python package for audio room simulations and array processing algorithms

**Link**: https://doi.org/10.1109/icassp.2018.8461310

**GitHub Repository**: https://github.com/LCAV/pyroomacoustics

## 📊 Performance Metrics

- **PSNR**: 25.32 dB ← 🔧 修复前: 15.36 dB

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
