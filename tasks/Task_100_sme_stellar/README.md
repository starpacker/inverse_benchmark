# Task 100: sme_stellar

Stellar spectroscopy parameter fitting using SME

## 📄 Paper Information

**Title**: Spectroscopy Made Easy: A New Tool for Fitting Observations with Synthetic Spectra

**Link**: doi:10.1051/0004-6361/201935310 (Wehrhahn et al., A&A, 2023)

**GitHub Repository**: ⚠️ https://github.com/AWehrhahn/SME (Spectroscopy Made Easy Python wrapper)

## 📊 Performance Metrics

- **PSNR**: 63.25 dB
- **SSIM**: N/A (1D spectrum)

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
