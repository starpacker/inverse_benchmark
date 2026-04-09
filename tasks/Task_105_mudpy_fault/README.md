# Task 105: mudpy_fault

Earthquake slip inversion using Okada Green's functions

## 📄 Paper Information

**Title**: ⚠️ MudPy: Earthquake slip inversion and finite fault modeling (no formal paper found)

**Link**: ❌ No formal publication found

**GitHub Repository**: https://github.com/dmelgarm/MudPy

## 📊 Performance Metrics

- **PSNR**: 20.43 dB ← 🔧 修复前: 11.42 dB
- **SSIM**: 0.885

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
