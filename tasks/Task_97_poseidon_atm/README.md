# Task 97: poseidon_atm

Exoplanet atmosphere retrieval using POSEIDON radiative transfer

## 📄 Paper Information

**Title**: POSEIDON: A Multidimensional Atmospheric Retrieval Code for Exoplanet Spectra

**Link**: doi:10.3847/1538-4357/ac47fe (MacDonald & Madhusudhan, ApJ, 2023)

**GitHub Repository**: https://github.com/MartianColonist/POSEIDON

## 📊 Performance Metrics

- **PSNR**: 45.50 dB
- **SSIM**: N/A (1D spectrum)

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
