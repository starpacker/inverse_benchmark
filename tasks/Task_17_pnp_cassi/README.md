# Task 17: pnp_cassi

Snapshot compressive spectral imaging (CASSI) using plug-and-play ADMM with denoiser

## 📄 Paper Information

**Title**: Deep plug-and-play priors for spectral snapshot compressive imaging

**Link**: doi:10.1364/PRJ.411745

**GitHub Repository**: https://github.com/zsm1211/PnP-CASSI

## 📊 Performance Metrics

- **PSNR**: 21.76 dB

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
