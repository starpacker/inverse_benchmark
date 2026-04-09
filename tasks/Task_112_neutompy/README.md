# Task 112: neutompy

Neutron tomography reconstruction using FBP

## 📄 Paper Information

**Title**: NeuTomPy toolbox, a Python package for tomographic data processing and reconstruction

**Link**: doi:10.1016/j.softx.2019.01.005 (Micieli et al., SoftwareX, 2019)

**GitHub Repository**: https://github.com/dmici/NeuTomPy-toolbox

## 📊 Performance Metrics

- **PSNR**: 31.54 dB
- **SSIM**: 0.6453

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
