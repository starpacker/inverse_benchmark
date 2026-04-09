# Task 173: torch_radon_ct

CT reconstruction using torch-radon with SIRT

## 📄 Paper Information

**Title**: TorchRadon: Fast Differentiable Routines for Computed Tomography

**Link**: arXiv:2009.14788

**GitHub Repository**: https://github.com/matteo-ronchetti/torch-radon

## 📊 Performance Metrics

- **PSNR**: 30.01 dB
- **SSIM**: 0.8513

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
