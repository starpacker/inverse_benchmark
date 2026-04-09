# Task 122: impedance_eis

Electrochemical impedance spectroscopy fitting using Randles circuit

## 📄 Paper Information

**Title**: impedance.py: A Python package for electrochemical impedance analysis

**Link**: doi:10.21105/joss.02349 (Murbach et al., JOSS, 2020)

**GitHub Repository**: https://github.com/ECSHackWeek/impedance.py

## 📊 Performance Metrics

- **PSNR**: 57.01 dB
- **SSIM**: N/A (1D spectral fitting)

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
