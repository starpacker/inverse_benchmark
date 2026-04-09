# Task 20: lenstronomy_simple_ring

Gravitational lensing mass reconstruction for simple Einstein ring using MCMC

## 📄 Paper Information

**Title**: lenstronomy II: A gravitational lensing software ecosystem

**Link**: doi:10.21105/joss.03283

**GitHub Repository**: https://github.com/lenstronomy/lenstronomy

## 📊 Performance Metrics

- **PSNR**: 21.69 dB (model fit)
- **SSIM**: 0.3304

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
