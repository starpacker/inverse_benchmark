# Task 02: ptyrad

Ptychographic phase retrieval using automatic differentiation and gradient descent optimization

## 📄 Paper Information

**Title**: PtyRAD: A High-Performance and Flexible Ptychographic Reconstruction Framework with Automatic Differentiation

**Link**: doi:10.1093/mam/ozaf070

**GitHub Repository**: https://github.com/chiahao3/ptyrad

## 📊 Performance Metrics

- **PSNR**: N/A (phase retrieval — no GT phase available)

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
