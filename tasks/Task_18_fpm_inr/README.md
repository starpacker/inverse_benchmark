# Task 18: fpm_inr

Fourier ptychographic microscopy using implicit neural representation (INR)

## 📄 Paper Information

**Title**: Fourier ptychographic microscopy image stack reconstruction using implicit neural representations

**Link**: doi:10.1364/OPTICA.505283

**GitHub Repository**: https://github.com/hwzhou2020/FPM_INR

## 📊 Performance Metrics

- **PSNR**: 39.15 dB

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
