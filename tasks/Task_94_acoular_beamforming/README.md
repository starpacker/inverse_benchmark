# Task 94: acoular_beamforming

Acoustic beamforming using CLEAN-SC algorithm

## 📄 Paper Information

**Title**: Acoular – Acoustic testing and source mapping software

**Link**: ⚠️ doi:10.1016/j.apacoust.2016.09.015 (Sarradj & Herold, Applied Acoustics, 2017)

**GitHub Repository**: https://github.com/acoular/acoular

## 📊 Performance Metrics

- **PSNR**: 30.12 dB
- **SSIM**: 0.9166

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
