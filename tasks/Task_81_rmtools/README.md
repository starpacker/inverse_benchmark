# Task 81: rmtools

Rotation measure synthesis for radio polarimetry using RM-CLEAN

## 📄 Paper Information

**Title**: Retrieving the Aerosol Complex Refractive Index using PyMieScatt: A Mie Computational Package with Visualization Capabilities

**Link**: doi:10.1016/j.jqsrt.2017.10.012

**GitHub Repository**: https://github.com/bsumlin/PyMieScatt

## 📊 Performance Metrics

- **PSNR**: 64.13 dB ← 🔧 修复前: 15.45 dB
- **SSIM**: N/A (1D Faraday 深度谱)

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
