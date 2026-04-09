# Task 74: e2vid

Event-based vision reconstruction using e2vid

## 📄 Paper Information

**Title**: High Speed and High Dynamic Range Video with an Event Camera

**Link**: https://doi.org/10.1109/TPAMI.2019.2963386

**GitHub Repository**: https://github.com/uzh-rpg/rpg_e2vid

## 📊 Performance Metrics

- **PSNR**: 21.48 dB ← 🔧 修复前: 18.26 dB (Round 1 修复前: 12.22 dB)
- **SSIM**: 0.8442

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
