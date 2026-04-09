# Task 118: cryodrgn

Cryo-EM 3D reconstruction using direct Fourier inversion (duplicate)

## 📄 Paper Information

**Title**: CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks

**Link**: doi:10.1038/s41592-020-01049-4 (Zhong et al., Nature Methods, 2021)

**GitHub Repository**: https://github.com/zhonge/cryodrgn

## 📊 Performance Metrics

- **PSNR**: 21.48 dB
- **SSIM**: 0.4760

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
