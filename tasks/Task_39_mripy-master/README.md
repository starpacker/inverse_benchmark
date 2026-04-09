# Task 39: mripy-master

Parallel MRI reconstruction using SENSE and conjugate gradient

## 📄 Paper Information

**Title**: ❌ No associated paper (software toolbox without dedicated publication)

**Link**: ❌ Not documented

**GitHub Repository**: https://github.com/peng-cao/mripy

## 📊 Performance Metrics

- **PSNR**: 37.21 dB (修复后：修正数组维度不匹配)

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
