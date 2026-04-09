# Task 78: phoeniks

THz time-domain spectroscopy material parameter extraction using phoeniks

## 📄 Paper Information

**Title**: batman: Bad-Ass Transit Model cAlculatioN in Python

**Link**: https://doi.org/10.1086/683602

**GitHub Repository**: https://github.com/lkreidberg/batman

## 📊 Performance Metrics

- **PSNR**: 48.25 dB (refractive index n)
- **SSIM**: N/A (1D spectral data)

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
