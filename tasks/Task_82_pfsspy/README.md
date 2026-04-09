# Task 82: pfsspy

Solar coronal magnetic field extrapolation using PFSS model

## 📄 Paper Information

**Title**: RM-synthesis, RM-clean and QU-fitting on polarised radio spectra

**Link**: doi:10.3847/1538-4365/ae3dea

**GitHub Repository**: https://github.com/CIRADA-Tools/RM-Tools

## 📊 Performance Metrics

- **PSNR**: 31.59 dB
- **SSIM**: None

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
