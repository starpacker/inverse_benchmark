# Task 145: direct_mri

MRI reconstruction using iterative optimization (POCS/ISTA)

## 📄 Paper Information

**Title**: DIRECT: Deep Image REConstruction Toolkit

**Link**: doi:10.21105/joss.04278

**GitHub Repository**: https://github.com/NKI-AI/direct

## 📊 Performance Metrics

- **PSNR**: 43.00 dB
- **SSIM**: 0.9712

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
