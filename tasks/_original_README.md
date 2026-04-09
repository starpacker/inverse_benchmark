# Inverse Problem Benchmark (Paper2Executable)

A comprehensive benchmark of **200 inverse problem tasks** spanning computational imaging, spectroscopy, medical imaging, astrophysics, geophysics, and more.

## Overview

This repository contains standardized implementations of inverse problems from scientific computing papers. Each task includes:

- **Source code** (`src/`) - Complete implementation
- **Test scripts** (`test_*.py`) - Automated testing
- **Data** (`data/`) - Input data, ground truth outputs, and reconstruction outputs
- **Documentation** (`README.md`) - Task description, setup instructions, and usage
- **Notebook** (`notebook.ipynb`) - Interactive tutorial
- **Requirements** (`requirements.txt`) - Python dependencies
- **Metadata** (`metadata.json`) - Task metadata including metrics

## Task Categories

| Category | Tasks | Examples |
|----------|-------|---------|
| Computational Imaging | 50+ | Ptychography, FPM, Lensless, Holography, Light Field |
| Medical Imaging | 30+ | CT, MRI, PET, Ultrasound, OCT |
| Spectroscopy | 20+ | Raman, NMR, X-ray, EIS |
| Astrophysics | 15+ | Gravitational Lensing, Radio Imaging, Stellar Spectroscopy |
| Geophysics | 15+ | Seismic, GPR, ERT, InSAR |
| Signal Processing | 20+ | Source Separation, DOA, Spike Sorting |
| Microscopy | 15+ | Super-resolution, Deconvolution, Phase Retrieval |
| Other Inverse Problems | 30+ | DIC, Modal Analysis, Rheology, Diffusion |

## Data

Full datasets (input, ground truth, reconstruction outputs) are available on Hugging Face:
🤗 [csusupergear/inverse_benchmark_details](https://huggingface.co/datasets/csusupergear/inverse_benchmark_details)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/starpacker/inverse_benchmark_details.git
cd inverse_benchmark_details

# Navigate to a specific task
cd Task_01_sim

# Install requirements
pip install -r requirements.txt

# Run the test
python test_forward_inverse.py
```

## Task Structure

Each task follows a standardized structure:

```
Task_XX_name/
├── src/                    # Source code
├── data/
│   ├── input/             # Input data
│   ├── gt_output/         # Ground truth output
│   └── recon_output/      # Reconstruction output
├── test_forward_inverse.py # Test script
├── requirements.txt        # Dependencies
├── metadata.json           # Task metadata & metrics
├── notebook.ipynb          # Interactive tutorial
└── README.md               # Documentation
```

## Metrics

Each task is evaluated with relevant metrics (e.g., PSNR, SSIM, MSE, correlation) comparing reconstruction outputs against ground truth.

## License

MIT License

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{inverse_benchmark_2026,
  title={Paper2Executable: A Comprehensive Benchmark for Inverse Problems},
  year={2026},
  url={https://github.com/starpacker/inverse_benchmark_details}
}
```
