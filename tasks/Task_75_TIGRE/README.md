# Task 75: TIGRE

CT reconstruction using TIGRE with FBP, SIRT, and CGLS

## 📄 Paper Information

**Title**: TIGRE: A MATLAB-GPU toolbox for CBCT image reconstruction

**Link**: https://doi.org/10.1088/2057-1976/2/5/055010

**GitHub Repository**: https://github.com/CERN/TIGRE

## 📊 Performance Metrics

- **PSNR**: 22.53 dB ← 🔧 修复前: 17.18 dB
- **SSIM**: 0.4644

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
