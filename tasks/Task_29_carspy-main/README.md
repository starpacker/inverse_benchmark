# Task 29: carspy-main

Compressed sensing MRI reconstruction (carspy)

## 📄 Paper Information

**Title**: ⚠️ CARSpy: Synthesizing and fitting coherent anti-Stokes Raman spectra in Python (GitHub repo only, no journal paper)

**Link**: ❌ No formal publication DOI (cite as GitHub repository: Yin, 2021)

**GitHub Repository**: https://github.com/chuckedfromspace/carspy

## 📊 Performance Metrics

- **PSNR**: 40.69 dB ← 🔧 修复前: 11.48 dB
- **SSIM**: 0.9437

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
