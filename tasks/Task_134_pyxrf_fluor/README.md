# Task 134: pyxrf_fluor

X-ray fluorescence spectral fitting using pyXRF

## 📄 Paper Information

**Title**: ⚠️ PyXRF (Proc. SPIE publication)

**Link**: ⚠️ doi:10.1117/12.2272585

**GitHub Repository**: https://github.com/NSLS-II/PyXRF

## 📊 Performance Metrics

- **PSNR**: 40.00 dB (concentration), 55.25 dB (spectrum)

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
