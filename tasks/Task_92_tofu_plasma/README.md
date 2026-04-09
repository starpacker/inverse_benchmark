# Task 92: tofu_plasma

Plasma tomography using Tikhonov regularization

## 📄 Paper Information

**Title**: TOFU: Tomography for Fusion - an IMAS-compatible open-source Python library for tomography diagnostics

**Link**: ⚠️ No formal DOI paper found; Zenodo/GitHub only

**GitHub Repository**: https://github.com/ToFuProject/tofu

## 📊 Performance Metrics

- **PSNR**: 33.11 dB
- **SSIM**: 0.9444

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
