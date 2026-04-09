# Task 31: CT-and-MR-Perfusion-Tool-main

CT/MR perfusion analysis using gamma variate fitting and deconvolution

## 📄 Paper Information

**Title**: PyPeT: A Python Perfusion Tool for Automated Quantitative Brain CT and MR Perfusion Analysis

**Link**: https://doi.org/10.48550/arXiv.2511.13310

**GitHub Repository**: https://github.com/Marijn311/CT-and-MR-Perfusion-Tool

## 📊 Performance Metrics

- **PSNR**: CBF=-14.50/CBV=-3.99/MTT=13.63/TMAX=7.96 (各灌注图分别评估)
- **SSIM**: CBF=0.00/CBV=0.66/MTT=0.74/TMAX=0.09 (修复后：裁剪异常值)

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
