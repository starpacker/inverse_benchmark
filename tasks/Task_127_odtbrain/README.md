# Task 127: odtbrain

Optical diffraction tomography using ODTbrain

## 📄 Paper Information

**Title**: ODTbrain: a Python library for full-view, dense diffraction tomography

**Link**: doi:10.1186/s12859-015-0764-0 (Müller et al., BMC Bioinformatics, 2015)

**GitHub Repository**: https://github.com/RI-imaging/ODTbrain

## 📊 Performance Metrics

- **PSNR**: 20.61 dB
- **SSIM**: 0.8303

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
