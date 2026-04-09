# Task 106: straintool_geo

Geodetic strain field estimation from GPS velocities

## 📄 Paper Information

**Title**: ⚠️ Strain determination using spatially discrete geodetic data (Shen et al., 2015; benchmark task uses custom reimplementation)

**Link**: ⚠️ doi:10.1785/0120140247 (Shen et al., BSSA, 2015) + Zenodo DOI:10.5281/zenodo.1297565 (StrainTool software)

**GitHub Repository**: ⚠️ Custom benchmark task; reference implementation inspired by https://github.com/DSOlab/StrainTool (Shen et al. method)

## 📊 Performance Metrics

- **PSNR**: 20.01 dB ← 🔧 修复前: 11.50 dB
- **SSIM**: 0.806

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
