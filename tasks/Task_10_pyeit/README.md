# Task 10: pyeit

3D electrical impedance tomography (EIT) using Jacobian-based reconstruction

## 📄 Paper Information

**Title**: pyEIT: A python based framework for Electrical Impedance Tomography

**Link**: doi:10.1016/j.softx.2018.04.002

**GitHub Repository**: https://github.com/liubenyuan/pyEIT

## 📊 Performance Metrics

- **PSNR**: N/A (3D EIT — conductivity distribution visual)

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
