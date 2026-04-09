# Task 89: pyTFM

Traction force microscopy using FTTC (Fourier Transform Traction Cytometry)

## 📄 Paper Information

**Title**: empymod: Open-source full 3D electromagnetic modeller for 1D VTI media

**Link**: doi:10.5281/zenodo.593094 (Zenodo DOI)

**GitHub Repository**: https://github.com/emsig/empymod

## 📊 Performance Metrics

- **PSNR**: 22.92 dB
- **SSIM**: 0.6963

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
