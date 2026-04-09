# Task 93: myptv

Particle tracking velocimetry using multi-camera triangulation

## 📄 Paper Information

**Title**: ⚠️ MyPTV: A Python package for 3D particle tracking velocimetry (no formal paper confirmed)

**Link**: ❌ No formal publication found

**GitHub Repository**: https://github.com/ronshnapp/MyPTV

## 📊 Performance Metrics

- **PSNR**: 66.86 dB
- **SSIM**: N/A (3D point cloud)

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
