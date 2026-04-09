# Task 03: lfm

Light field microscopy 3D reconstruction using Richardson-Lucy deconvolution

## 📄 Paper Information

**Title**: Artifact-free deconvolution in light field microscopy

**Link**: doi:10.1364/OE.27.031644

**GitHub Repository**: https://github.com/lambdaloop/pyolaf

## 📊 Performance Metrics

- **PSNR**: N/A (volumetric LFM reconstruction — input 2304×2304 raw vs 20×128×128 volume)

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
