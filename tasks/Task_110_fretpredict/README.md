# Task 110: fretpredict

FRET distance distribution recovery using Tikhonov regularization

## 📄 Paper Information

**Title**: ⚠️ FRETpredict: FRET efficiency prediction for protein structure (no formal paper confirmed)

**Link**: ❌ No formal publication found

**GitHub Repository**: https://github.com/KULL-Centre/FRETpredict

## 📊 Performance Metrics

- **PSNR**: 20.37 dB ← 🔧 修复前: 15.22 dB (补录+修复)
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
