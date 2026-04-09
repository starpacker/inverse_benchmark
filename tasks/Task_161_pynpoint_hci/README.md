# Task 161: pynpoint_hci

High-contrast imaging using PCA-based ADI reduction

## 📄 Paper Information

**Title**: PynPoint: a modular pipeline architecture for processing and analysis of high-contrast imaging data

**Link**: doi:10.1051/0004-6361/201834136

**GitHub Repository**: https://github.com/PynPoint/PynPoint

## 📊 Performance Metrics

- **PSNR**: 28.88 dB

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
