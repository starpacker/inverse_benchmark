# Task 130: cobaya_cosmo

Cosmological parameter inference using Cobaya MCMC

## 📄 Paper Information

**Title**: Cobaya: Code for Bayesian Analysis of hierarchical physical models

**Link**: https://doi.org/10.1088/1475-7516/2021/05/057

**GitHub Repository**: https://github.com/CobayaSampler/cobaya

## 📊 Performance Metrics

- **PSNR**: 60.58 dB

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
