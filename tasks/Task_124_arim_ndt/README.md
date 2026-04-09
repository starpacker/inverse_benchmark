# Task 124: arim_ndt

Ultrasonic NDT imaging using total focusing method (TFM)

## 📄 Paper Information

**Title**: A Model for Multiview Ultrasonic Array Inspection of Small Two-Dimensional Defects

**Link**: doi:10.1109/TUFFC.2019.2909988 (Budyn et al., IEEE TUFFC, 2019)

**GitHub Repository**: https://github.com/ndtatbristol/arim

## 📊 Performance Metrics

- **PSNR**: 20.61 dB ← 🔧 修复前: 18.43 dB
- **SSIM**: 0.800

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
