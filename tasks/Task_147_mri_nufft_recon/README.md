# Task 147: mri_nufft_recon

Non-Cartesian MRI reconstruction using NUFFT

## 📄 Paper Information

**Title**: MRI-NUFFT: Doing non-Cartesian MRI has never been easier

**Link**: doi:10.21105/joss.07743

**GitHub Repository**: https://github.com/mind-inria/mri-nufft

## 📊 Performance Metrics

- **PSNR**: 22.40 dB ← 🔧 修复前: 18.74 dB
- **SSIM**: 0.551

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
