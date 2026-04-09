# Task 09: bayhunt

Bayesian seismic velocity model inversion using MCMC (Markov Chain Monte Carlo)

## 📄 Paper Information

**Title**: BayHunter - McMC transdimensional Bayesian inversion of receiver functions and surface wave dispersion

**Link**: doi:10.5880/GFZ.2.4.2019.001

**GitHub Repository**: https://github.com/jenndrei/BayHunter

## 📊 Performance Metrics

- **PSNR**: N/A (MCMC ensemble — velocity model visual)

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
