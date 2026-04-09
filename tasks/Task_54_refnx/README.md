# Task 54: refnx

X-ray reflectometry inversion for thin film structure using refnx

## 📄 Paper Information

**Title**: ⚠️ References TFM and MSM papers but no specific pyTFM paper found

**Link**: ⚠️ https://www.ncbi.nlm.nih.gov/pubmed/11832345

**GitHub Repository**: https://github.com/fabrylab/pyTFM

## 📊 Performance Metrics

- **PSNR**: 85.32 dB
- **SSIM**: N/A (1D reflectivity curve)

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
