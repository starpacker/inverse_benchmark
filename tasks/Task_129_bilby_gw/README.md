# Task 129: bilby_gw

Gravitational wave parameter estimation using Bilby

## 📄 Paper Information

**Title**: Bilby: A User-friendly Bayesian Inference Library for Gravitational-wave Astronomy

**Link**: https://doi.org/10.3847/1538-4365/ab06fc

**GitHub Repository**: https://github.com/bilby-dev/bilby

## 📊 Performance Metrics

- **PSNR**: 63.06 dB
- **SSIM**: N/A (waveform match = 0.9994)

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
