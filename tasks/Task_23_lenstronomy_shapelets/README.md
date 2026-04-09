# Task 23: lenstronomy_shapelets

Galaxy light profile reconstruction using Shapelet basis decomposition

## 📄 Paper Information

**Title**: lenstronomy II: A gravitational lensing software ecosystem

**Link**: https://joss.theoj.org/papers/10.21105/joss.03283 ; https://arxiv.org/abs/1803.09746

**GitHub Repository**: https://github.com/lenstronomy/lenstronomy

## 📊 Performance Metrics

- **PSNR**: 27.67 dB

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
