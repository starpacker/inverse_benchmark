# Task 194: naf_cbct_recon

Cone-beam CT reconstruction using NAF neural fields

## 📄 Paper Information

**Title**: Neural Attenuation Fields for Sparse-View CBCT Reconstruction

**Link**: ⚠️ MICCAI 2022 (Oral)

**GitHub Repository**: https://github.com/Ruyi-Zha/naf_cbct

## 📊 Performance Metrics

- **PSNR**: 34.78 dB
- **SSIM**: 0.9657

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
