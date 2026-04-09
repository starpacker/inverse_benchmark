# Task 64: IHCP

Inverse heat conduction problem using Tikhonov regularization

## 📄 Paper Information

**Title**: Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools

**Link**: https://doi.org/10.1016/j.electacta.2015.09.097

**GitHub Repository**: https://github.com/ciuccislab/pyDRTtools

## 📊 Performance Metrics

- **PSNR**: 21.46 dB ← 🔧 修复前: 14.29 dB
- **SSIM**: 0.712

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
