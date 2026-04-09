# Task 196: promptmr_mri

MRI reconstruction using PromptMR with TV regularization

## 📄 Paper Information

**Title**: PromptMR: Prompting for Dynamic and Multi-Contrast MRI Reconstruction

**Link**: arXiv:2309.13839

**GitHub Repository**: https://github.com/hellopipu/PromptMR

## 📊 Performance Metrics

- **PSNR**: 53.55 dB (avg T1=63.47, T2=43.64)
- **SSIM**: 0.9982

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
