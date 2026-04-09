# Task 144: fastmri_recon

Accelerated MRI reconstruction using ISTA with TV regularization

## 📄 Paper Information

**Title**: fastMRI: An Open Dataset and Benchmarks for Accelerated MRI

**Link**: arXiv:1811.08839

**GitHub Repository**: https://github.com/facebookresearch/fastMRI

## 📊 Performance Metrics

- **PSNR**: None
- **SSIM**: None

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
