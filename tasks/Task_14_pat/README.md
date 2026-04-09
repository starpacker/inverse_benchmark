# Task 14: pat

Photoacoustic tomography with spectral unmixing for blood oxygen saturation estimation

## 📄 Paper Information

**Title**: PATATO: a Python photoacoustic tomography analysis toolkit

**Link**: doi:10.21105/joss.05686

**GitHub Repository**: https://github.com/BohndiekLab/patato

## 📊 Performance Metrics

- **PSNR**: N/A (photoacoustic — SO₂ estimation visual)

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
