# Task 68: seislib

Seismic surface wave tomography using seislib

## 📄 Paper Information

**Title**: Surface-wave tomography using SeisLib: a Python package for multi-scale seismic imaging

**Link**: https://doi.org/10.1093/gji/ggac236

**GitHub Repository**: https://github.com/fmagrini/seislib

## 📊 Performance Metrics

- **PSNR**: 20.84 dB
- **SSIM**: 0.5402

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
