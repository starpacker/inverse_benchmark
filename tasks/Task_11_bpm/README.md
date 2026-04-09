# Task 11: bpm

Beam propagation method (BPM) for refractive index reconstruction in optical waveguides

## 📄 Paper Information

**Title**: ⚠️ No dedicated paper — Beam Propagation Method ODT reconstruction with adjoint-state gradient + FISTA acceleration

**Link**: ❌ No formal publication (custom benchmark task)

**GitHub Repository**: ⚠️ Custom benchmark task (no published repo; implements Batched BPM inversion with FISTA-accelerated regularized gradient descent using numpy/scipy)

## 📊 Performance Metrics

- **PSNR**: N/A (BPM — refractive index vs field amplitude)

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
