# Task 111: isdm_scatter

Inverse scattering for molecular structure using shrinkwrap phase retrieval

## 📄 Paper Information

**Title**: ⚠️ No dedicated paper — standard speckle imaging via memory effect + phase retrieval (HIO algorithm)

**Link**: ❌ No formal publication (custom Category A benchmark task)

**GitHub Repository**: ⚠️ https://github.com/yqx7150/ISDM

## 📊 Performance Metrics

- **PSNR**: 98.27 dB ← 🔧 修复前: 12.78 dB
- **SSIM**: 1.000

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
