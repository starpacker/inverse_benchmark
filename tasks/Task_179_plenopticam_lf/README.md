# Task 179: plenopticam_lf

Light field depth estimation using plenopticam

## 📄 Paper Information

**Title**: PlenoptiCam v1.0: A Light-Field Imaging Framework

**Link**: doi:10.1109/TIP.2021.3095671

**GitHub Repository**: https://github.com/hahnec/plenopticam

## 📊 Performance Metrics

- **PSNR**: 43.39 dB (sub-aperture), 8.27 dB (depth)
- **SSIM**: 0.983 (sub-aperture)

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
