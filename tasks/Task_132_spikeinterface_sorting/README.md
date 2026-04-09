# Task 132: spikeinterface_sorting

Spike sorting using SpikeInterface

## 📄 Paper Information

**Title**: ⚠️ SpikeInterface (eLife publication)

**Link**: ⚠️ doi:10.7554/eLife.61834

**GitHub Repository**: https://github.com/SpikeInterface/spikeinterface

## 📊 Performance Metrics

- **PSNR**: 35.11 dB (template waveform reconstruction)
- **SSIM**: N/A (1D waveform)

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
