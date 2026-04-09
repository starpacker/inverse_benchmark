# Task 34: hcipy-master

Adaptive optics simulation using HCIpy with Fraunhofer propagation

## 📄 Paper Information

**Title**: High Contrast Imaging for Python (HCIPy): an open-source adaptive optics and coronagraph simulator

**Link**: doi:10.1117/12.2314407

**GitHub Repository**: https://github.com/ehpor/hcipy

## 📊 Performance Metrics

- **PSNR**: 197.46 dB

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
