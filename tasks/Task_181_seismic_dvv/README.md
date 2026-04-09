# Task 181: seismic_dvv

Seismic velocity change estimation using stretching method

## 📄 Paper Information

**Title**: ⚠️ SeisMIC: Seismic velocity monitoring

**Link**: ❌ Not documented

**GitHub Repository**: https://github.com/PeterMakus/SeisMIC

## 📊 Performance Metrics

- **PSNR**: 22.12 dB (dv/v time series)
- **SSIM**: N/A (1D time series)

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
