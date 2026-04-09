# Task 98: zdipy_doppler

Zodiacal light Doppler imaging using least-squares inversion

## 📄 Paper Information

**Title**: The evolution of surface magnetic fields in young solar-type stars II: the early main sequence (250-650 Myr)

**Link**: https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.4956F/abstract

**GitHub Repository**: https://github.com/folsomcp/ZDIpy

## 📊 Performance Metrics

- **PSNR**: 20.66 dB ← 🔧 修复前: 8.89 dB
- **SSIM**: 0.8690

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
