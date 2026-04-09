# Task 06: semiblindpsfdeconv-master

Semi-blind PSF deconvolution for microscopy using alternating optimization

## 📄 Paper Information

**Title**: Semi-Blind Spatially-Variant Deconvolution in Optical Microscopy with Local Point Spread Function Estimation By Use Of Convolutional Neural Networks

**Link**: doi:10.1109/ICIP.2018.8451736

**GitHub Repository**: https://github.com/idiap/semiblindpsfdeconv

## 📊 Performance Metrics

- **PSNR**: 28.93 dB
- **SSIM**: 0.9332

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
