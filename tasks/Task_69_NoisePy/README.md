# Task 69: NoisePy

Ambient noise tomography using NoisePy

## 📄 Paper Information

**Title**: NoisePy: a new high-performance python tool for seismic ambient noise seismology

**Link**: https://doi.org/10.1785/0220190364

**GitHub Repository**: https://github.com/noisepy/NoisePy

## 📊 Performance Metrics

- **PSNR**: 22.12 dB
- **SSIM**: 0.7240

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
