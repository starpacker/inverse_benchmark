# Task 07: oopao_sh_wfs

Adaptive optics wavefront sensing using Shack-Hartmann sensor and closed-loop control

## 📄 Paper Information

**Title**: OOPAO: Object-Oriented Python Adaptive Optics

**Link**: https://hal.science/hal-04402878v1

**GitHub Repository**: https://github.com/cheritier/OOPAO

## 📊 Performance Metrics

- **PSNR**: N/A (adaptive optics — WFE RMS evaluation)

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
