# Task 126: prysm_phase

Wavefront sensing using Zernike polynomial fitting

## 📄 Paper Information

**Title**: prysm: A Python optics module

**Link**: doi:10.21105/joss.01352 (Dube, JOSS, 2019)

**GitHub Repository**: https://github.com/brandondube/prysm

## 📊 Performance Metrics

- **PSNR**: 77.07 dB
- **SSIM**: 1.0000

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
