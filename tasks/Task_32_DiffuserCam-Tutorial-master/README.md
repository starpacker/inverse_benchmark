# Task 32: DiffuserCam-Tutorial-master

DiffuserCam lensless imaging using ADMM with soft thresholding

## 📄 Paper Information

**Title**: ⚠️ DiffuserCam: Lensless Single-exposure 3D Imaging (Antipa et al., Optica 2018; tutorial companion repo)

**Link**: doi:10.1364/OPTICA.5.000001

**GitHub Repository**: https://github.com/Waller-Lab/DiffuserCam-Tutorial

## 📊 Performance Metrics

- **PSNR**: 30.69 dB ← 🔧 修复前: 4.18 dB
- **SSIM**: 0.9265

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
