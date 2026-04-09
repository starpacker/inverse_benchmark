# Task 107: gprfwi

GPR full-waveform inversion for subsurface permittivity

## 📄 Paper Information

**Title**: ⚠️ No dedicated paper — standard GPR 1D convolutional model with Wiener deconvolution

**Link**: ❌ No formal publication (custom Category A benchmark task)

**GitHub Repository**: https://github.com/nephilim2016/GPR-FWI-Py

## 📊 Performance Metrics

- **PSNR**: 24.35 dB ← 🔧 修复前: 16.14 dB
- **SSIM**: 0.8039

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
