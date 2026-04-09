# Task 191: flowdec_deconv

3D fluorescence deconvolution using Richardson-Lucy

## 📄 Paper Information

**Title**: Cytokit: A single-cell analysis toolkit

**Link**: doi:10.1101/460980

**GitHub Repository**: https://github.com/hammerlab/flowdec

## 📊 Performance Metrics

- **PSNR**: 27.65 dB
- **SSIM**: 0.8895

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
