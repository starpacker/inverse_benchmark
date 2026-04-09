# Task 73: NLOS

Non-line-of-sight imaging using time-resolved backprojection

## 📄 Paper Information

**Title**: Wave-Based Non-Line-of-Sight Imaging using Fast f-k Migration

**Link**: ⚠️ http://www.computationalimaging.org/publications/nlos-fk/

**GitHub Repository**: https://github.com/computational-imaging/nlos-fk

## 📊 Performance Metrics

- **PSNR**: 16.16 dB ← 🔧 修复前: 9.60 dB
- **SSIM**: 0.4058

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
