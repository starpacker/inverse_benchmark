# Task 175: asteroid_bss

Blind source separation using FastICA

## 📄 Paper Information

**Title**: ⚠️ Asteroid: Audio source separation

**Link**: ⚠️ Interspeech, arXiv:1710.04196

**GitHub Repository**: https://github.com/asteroid-team/asteroid

## 📊 Performance Metrics

- **PSNR**: 31.03 dB ← 🔧 修复前: 16.90 dB
- **SSIM**: N/A (1D 音频信号)

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
