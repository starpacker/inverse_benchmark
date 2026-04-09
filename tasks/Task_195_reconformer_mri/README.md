# Task 195: reconformer_mri

MRI reconstruction using Reconformer transformer

## 📄 Paper Information

**Title**: ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer

**Link**: https://doi.org/10.1109/tmi.2023.3314747

**GitHub Repository**: https://github.com/guopengf/ReconFormer

## 📊 Performance Metrics

- **PSNR**: 27.05 dB
- **SSIM**: 0.9706

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
