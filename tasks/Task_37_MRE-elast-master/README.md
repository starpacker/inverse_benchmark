# Task 37: MRE-elast-master

MR elastography inversion for tissue stiffness using finite element method

## 📄 Paper Information

**Title**: Finite Element Reconstruction of Stiffness Images in MR Elastography Using Statistical Physical Forward Modeling and Proximal Optimization Methods

**Link**: doi:10.1109/ISBI48211.2021.9433873

**GitHub Repository**: https://github.com/narges-mhm/MRE-elast

## 📊 Performance Metrics

- **PSNR**: N/A (MRE — elasticity modulus CNR evaluation)

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
