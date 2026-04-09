# Task 114: afm_force

AFM force spectroscopy inversion using Sader method

## 📄 Paper Information

**Title**: ⚠️ No dedicated paper — implements Sader-Jarvis method from: Quantitative force measurements using frequency modulation atomic force microscopy

**Link**: https://doi.org/10.1063/1.1667267

**GitHub Repository**: ⚠️ https://github.com/Probe-Particle/ppafm

## 📊 Performance Metrics

- **PSNR**: 29.12 dB
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
