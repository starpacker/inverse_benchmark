# Task 05: s2ISM-main

Image scanning microscopy (ISM) super-resolution using pixel reassignment and Richardson-Lucy deconvolution

## 📄 Paper Information

**Title**: Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy

**Link**: doi:10.1038/s41566-025-01695-0

**GitHub Repository**: https://github.com/VicidominiLab/s2ISM

## 📊 Performance Metrics

- **PSNR**: 18.88 dB ← 🔧 修复前: 11.25 dB
- **SSIM**: 0.8442

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
