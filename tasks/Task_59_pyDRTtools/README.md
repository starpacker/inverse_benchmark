# Task 59: pyDRTtools

Electrochemical impedance spectroscopy DRT analysis using pyDRTtools

## 📄 Paper Information

**Title**: Ab initio electron density determination directly from solution scattering data

**Link**: https://doi.org/10.1038/nmeth.4581

**GitHub Repository**: https://github.com/tdgrant1/denss

## 📊 Performance Metrics

- **PSNR**: 26.34 dB

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
