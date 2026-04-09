# Task 190: dm4ct_bench

CT reconstruction using diffusion model guidance

## 📄 Paper Information

**Title**: ⚠️ DM4CT: Diffusion models for CT

**Link**: ⚠️ ICLR 2026

**GitHub Repository**: https://github.com/DM4CT/DM4CT

## 📊 Performance Metrics

- **PSNR**: 23.49 dB
- **SSIM**: 0.5918

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
