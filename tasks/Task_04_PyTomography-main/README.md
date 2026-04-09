# Task 04: PyTomography-main

SPECT/PET tomographic reconstruction using OSEM (Ordered Subset Expectation Maximization)

## 📄 Paper Information

**Title**: PyTomography: A python library for medical image reconstruction

**Link**: doi:10.1016/j.softx.2024.102020

**GitHub Repository**: https://github.com/qurit/PyTomography

## 📊 Performance Metrics

- **PSNR**: 20.66 dB
- **SSIM**: 0.8078

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
