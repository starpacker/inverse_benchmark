# Task 58: simpeg_MT

Magnetotelluric 1D inversion using SimPEG

## 📄 Paper Information

**Title**: A Meshless Method to Compute Pressure Fields from Image Velocimetry

**Link**: https://doi.org/10.1088/1361-6501/ac70a9

**GitHub Repository**: https://github.com/mendezVKI/SPICY_VKI

## 📊 Performance Metrics

- **PSNR**: 50.02 dB ← 🔧 修复前: 15.05 dB
- **SSIM**: N/A (1D sounding curve)

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
