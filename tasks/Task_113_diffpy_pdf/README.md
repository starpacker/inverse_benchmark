# Task 113: diffpy_pdf

Pair distribution function refinement for atomic structure

## 📄 Paper Information

**Title**: Complex modeling: a strategy and software program for combining multiple information sources to solve ill posed structure and nanostructure inverse problems

**Link**: doi:10.1107/S2053273315014473 (Juhás et al., Acta Cryst. A, 2015)

**GitHub Repository**: https://github.com/diffpy/diffpy.srfit

## 📊 Performance Metrics

- **PSNR**: 54.09 dB
- **SSIM**: None

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
