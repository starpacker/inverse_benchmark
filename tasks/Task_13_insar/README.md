# Task 13: insar

InSAR phase unwrapping using least-squares optimization

## 📄 Paper Information

**Title**: Exploiting Sparsity for Phase Unwrapping

**Link**: ⚠️ IEEE IGARSS 2019 (Chartrand, Calef, Warren)

**GitHub Repository**: https://github.com/scottstanie/spurs

## 📊 Performance Metrics

- **PSNR**: N/A (phase unwrapping — no absolute GT)

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
