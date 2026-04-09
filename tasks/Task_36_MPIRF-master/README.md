# Task 36: MPIRF-master

Magnetic particle imaging (MPI) reconstruction using system matrix inversion

## 📄 Paper Information

**Title**: A novel software framework for magnetic particle imaging reconstruction

**Link**: doi:10.1002/ima.22707

**GitHub Repository**: https://github.com/XiaoYaoNet/MPIRF

## 📊 Performance Metrics

- **PSNR**: 21.71 dB ← 🔧 修复前: 16.41 dB
- **SSIM**: 0.9802

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
