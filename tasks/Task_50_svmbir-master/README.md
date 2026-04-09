# Task 50: svmbir-master

CT reconstruction using SVMBIR (model-based iterative reconstruction)

## 📄 Paper Information

**Title**: ⚠️ Software library with multiple algorithm papers

**Link**: https://doi.org/10.1109/TCI.2016.2599778

**GitHub Repository**: https://github.com/cabouman/svmbir

## 📊 Performance Metrics

- **PSNR**: 23.89 dB
- **SSIM**: 0.7975

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
