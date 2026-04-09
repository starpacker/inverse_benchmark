# Task 128: pygimli_ert

Electrical resistivity tomography using pyGIMLi

## 📄 Paper Information

**Title**: pyGIMLi: An open-source library for modelling and inversion in geophysics

**Link**: https://doi.org/10.1016/j.cageo.2017.07.011

**GitHub Repository**: https://github.com/gimli-org/pygimli

## 📊 Performance Metrics

- **PSNR**: 21.64 dB ← 🔧 修复前: 14.43 dB
- **SSIM**: 0.437

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
