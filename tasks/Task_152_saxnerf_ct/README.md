# Task 152: saxnerf_ct

CT reconstruction using SAX-NeRF neural rendering

## 📄 Paper Information

**Title**: Structure-Aware Sparse-View X-ray 3D Reconstruction

**Link**: https://arxiv.org/abs/2311.10959

**GitHub Repository**: https://github.com/caiyuanhao1998/SAX-NeRF

## 📊 Performance Metrics

- **PSNR**: 41.61 dB
- **SSIM**: 0.9985

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
