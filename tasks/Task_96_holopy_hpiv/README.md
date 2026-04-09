# Task 96: holopy_hpiv

Holographic particle image velocimetry using digital holography

## 📄 Paper Information

**Title**: HoloPy: Holography and Light Scattering in Python

**Link**: ⚠️ doi:10.5281/zenodo.592838 (Zenodo DOI)

**GitHub Repository**: https://github.com/manoharan-lab/holopy

## 📊 Performance Metrics

- **PSNR**: 21.34 dB
- **SSIM**: N/A (3D positions)

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
