# Task 65: VFM

Virtual fields method for material property identification

## 📄 Paper Information

**Title**: Image Processing and Machine Learning for Hyperspectral Unmixing: An Overview and the HySUPP Python Package

**Link**: https://doi.org/10.1109/TGRS.2024.3393570

**GitHub Repository**: https://github.com/BehnoodRasti/HySUPP

## 📊 Performance Metrics

- **PSNR**: 33.12 dB
- **SSIM**: 0.9689

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
