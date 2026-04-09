# Task 30: caustics-main

Strong gravitational lensing modeling using differentiable ray tracing (caustics)

## 📄 Paper Information

**Title**: Caustics: A Python Package for Accelerated Strong Gravitational Lensing Simulations

**Link**: https://doi.org/10.21105/joss.07081 ; https://arxiv.org/abs/2406.15542

**GitHub Repository**: https://github.com/Ciela-Institute/caustics

## 📊 Performance Metrics

- **PSNR**: 31.11 dB ← 🔧 修复前: 14.06 dB
- **SSIM**: 0.9947

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
