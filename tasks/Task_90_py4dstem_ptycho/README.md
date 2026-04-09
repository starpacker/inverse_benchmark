# Task 90: py4dstem_ptycho

4D-STEM ptychography using py4DSTEM

## 📄 Paper Information

**Title**: py4DSTEM: A Software Package for Four-Dimensional Scanning Transmission Electron Microscopy Data Analysis

**Link**: doi:10.1017/S1431927621000477

**GitHub Repository**: https://github.com/py4dstem/py4DSTEM

## 📊 Performance Metrics

- **PSNR**: 25.21 dB
- **SSIM**: 0.7495

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
