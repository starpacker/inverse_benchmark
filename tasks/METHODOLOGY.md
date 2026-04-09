# Methodology: Data Collection and Code Discovery for Inverse Problem Benchmark

This document describes the systematic methodology used to discover, collect, and organize the **200 inverse problem tasks** in this benchmark. Each task involves a scientific inverse problem (e.g., image reconstruction, spectral analysis, signal processing) with corresponding ground-truth code, input data, and reconstruction outputs.

---

## Table of Contents

1. [Overview](#overview)
2. [Task Discovery and Source Identification](#task-discovery-and-source-identification)
3. [Python Code Discovery](#python-code-discovery)
4. [Data File Discovery](#data-file-discovery)
5. [Function Documentation Generation](#function-documentation-generation)
6. [Automated Processing Pipeline](#automated-processing-pipeline)
7. [Directory Structure Standardization](#directory-structure-standardization)
8. [Quality Metrics and Evaluation](#quality-metrics-and-evaluation)
9. [Tools and Scripts Used](#tools-and-scripts-used)

---

## Overview

The benchmark covers **10+ scientific domains**, including:

- **Computational Imaging**: Ptychography, holography, lensless imaging, Fourier ptychography, light field microscopy
- **Medical Imaging**: CT reconstruction, MRI reconstruction, PET/SPECT, ultrasound beamforming
- **Spectroscopy**: Raman, NMR, X-ray fluorescence, impedance spectroscopy
- **Astrophysics**: Gravitational lensing, stellar spectroscopy, CMB analysis, pulsar timing
- **Geophysics**: Seismic inversion, ground-penetrating radar, electrical resistivity tomography
- **Material Science**: X-ray diffraction, SAXS/SANS, AFM force curves, constitutive modeling
- **Neuroscience**: Calcium imaging, spike sorting, fNIRS analysis
- **Optics**: Wavefront sensing, PSF deconvolution, Mueller matrix polarimetry
- **Signal Processing**: Blind source separation, dynamic mode decomposition, modal analysis
- **Neural Operators / Learned Methods**: Physics-informed neural networks, neural operators for PDEs

Each task is derived from an **open-source GitHub repository** with an associated **published paper**.

---

## Task Discovery and Source Identification

### Step 1: Literature Survey

Tasks were identified through a comprehensive survey of open-source scientific computing repositories on GitHub that implement inverse problem solvers. Selection criteria included:

- **Open-source availability**: The code must be publicly available on GitHub
- **Paper association**: Each repository should have an associated published paper (journal article, conference paper, or preprint)
- **Reproducibility**: The code must be executable with available or generatable test data
- **Diversity**: Tasks were selected to cover a broad range of scientific domains and inverse problem types

### Step 2: Metadata Extraction from Report

All 200 tasks were documented in a master report file (`Report.md`) with structured metadata for each task:

```markdown
## Task XX: task_name

- **GitHub Repo**: https://github.com/...
- **Paper Title**: ...
- **Paper Link**: https://doi.org/...
- **Description**: Brief description of the inverse problem
- **Working Folder**: /path/to/sandbox/directory
- **Python Path**: /path/to/conda/env/python
- **GT Code Path**: /path/to/ground_truth_code.py
- **PSNR**: XX.XX dB
- **SSIM**: 0.XXXX
- **专属资源文件夹 (Resource Folder)**: /data/yjh/website_assets/Task_XX_name/
- **核心数据清单 (Core Data Manifest)**:
  - Input: input_file.npy
  - GT Output: gt_output_file.npy
  - Recon Output: recon_output_file.npy
```

The `parse_report.py` script automated the extraction of this metadata using regex-based parsing:

```python
# Key regex patterns used
task_pattern = r'## Task (\d+): (.+)'
field_patterns = {
    'github_repo': r'\*\*GitHub Repo\*\*:\s*(.*)',
    'paper_title': r'\*\*Paper Title\*\*:\s*(.*)',
    'working_folder': r'\*\*Working Folder\*\*:\s*(.*)',
    'gt_code_path': r'\*\*GT Code Path\*\*:\s*(.*)',
    # ... additional fields
}
```

---

## Python Code Discovery

### Ground-Truth Code (`gt_code_path`)

Each task's primary Python code was located via the `gt_code_path` field in `Report.md`. These files represent the **reference implementation** of the inverse problem solver. Common naming conventions include:

| Pattern | Description |
|---------|-------------|
| `sim_code.py` | Simulation and reconstruction code |
| `main.py` | Main entry point |
| `demo.py` | Demonstration script |
| `agent_*.py` | Agent-generated implementation |
| `run_*.py` | Execution scripts |

### Code Location Strategy

Python files were found in **sandbox directories** — isolated environments created for each task:

```
/data/yjh/<repo_name>_sandbox/
├── sim_code.py          # Main reconstruction code
├── agent_code.py        # Agent-generated variant
├── requirements.txt     # Dependencies
└── ...
```

The `batch_process.py` script located these files through the following strategy:

1. **Direct path**: Use the `gt_code_path` from the report metadata
2. **Working folder scan**: Search within the `working_folder` directory for `.py` files
3. **Pattern matching**: Look for known filename patterns (`sim_code.py`, `main.py`, `demo.py`)
4. **Jupyter notebooks**: Also discover `.ipynb` files when present

### Additional Python Files

Beyond the main code, additional Python files were collected:
- **Helper modules**: Utility functions imported by the main script
- **Configuration files**: Task-specific parameters and settings
- **Evaluation scripts**: Code for computing metrics (PSNR, SSIM, domain-specific metrics)

### Dependency Extraction

For each task, Python dependencies were extracted by:
1. Running `pip freeze` in the task's conda environment (using the `python_path`)
2. Parsing `requirements.txt` files in the working directory
3. Analyzing import statements in the source code

---

## Data File Discovery

### Data Types

Each task involves three categories of data files:

| Category | Description | Common Formats |
|----------|-------------|----------------|
| **Input** | Raw measurements / observations | `.npy`, `.npz`, `.mat`, `.h5`, `.fits`, `.tif` |
| **Ground Truth (GT) Output** | Known true solution | `.npy`, `.npz`, `.mat`, `.h5` |
| **Reconstruction Output** | Algorithm's reconstruction result | `.npy`, `.npz`, `.mat`, `.h5` |

### Data Location Strategy

Data files were discovered through a multi-step process:

#### 1. Report Metadata Parsing
The `核心数据清单 (Core Data Manifest)` section in `Report.md` explicitly lists the input, ground truth, and reconstruction output files:

```markdown
- Input: measurements.npy
- GT Output: ground_truth.npy  
- Recon Output: reconstruction.npy
```

#### 2. Resource Folder Scanning
Each task has a dedicated resource folder at:
```
/data/yjh/website_assets/Task_XX_name/
```
This folder contains:
- Pre-processed data files (`.npy`, `.npz`, `.mat`, `.h5`)
- Visualization results (`vis_result.png`)
- Any task-specific supplementary data

#### 3. Sandbox Directory Mining
The working sandbox directories were scanned for data files:
```
/data/yjh/<repo_name>_sandbox/
├── data/                # Data subdirectory
│   ├── input.npy
│   ├── ground_truth.npy
│   └── recon.npy
├── results/             # Output directory
│   └── reconstruction.npy
└── ...
```

#### 4. File Filtering Rules
Large binary files were identified and handled appropriately:
- Files > 100MB flagged for LFS tracking or HuggingFace-only distribution
- Excluded formats from Git: `.npy`, `.npz`, `.h5`, `.pkl`, `.mat`, `.fits`, `.tif`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.wav`, `.mp4`
- These large data files are available on the [HuggingFace dataset](https://huggingface.co/datasets/csusupergear/inverse_benchmark_details)

---

## Function Documentation Generation

### AI-Generated Documentation (`generated_docs/`)

For each task, an AI agent analyzed the source code and generated detailed documentation for key functions. This process produced two types of JSON files:

#### `final_function_list.json`
Lists all key functions identified in the task's codebase:
```json
[
    "function_name_1",
    "function_name_2",
    "..."
]
```

#### `generated_docs_<function_name>.json`
Contains AI-generated documentation for each function:
```json
{
    "function_explanation": "Detailed explanation of what the function does, its algorithmic approach, and its role in the inverse problem pipeline.",
    "function_docstring": "A standardized docstring including parameters, return values, and exceptions.",
    "usage_example": "import numpy as np\n# Complete usage example\nresult = function_name(input_data)\n..."
}
```

### Discovery of Generated Documentation

Generated documentation files were located in sandbox directories:
```
/data/yjh/<repo_name>_sandbox/
├── generated_docs_function1.json
├── generated_docs_function2.json
├── generated_docs_evaluate_results.json
├── final_function_list.json
└── ...
```

The `copy_generated_docs.py` script automated the discovery and collection process:

1. **Load task mapping**: Read `task_mapping.json` to get task metadata
2. **Resolve sandbox path**: Try multiple path patterns to find each task's sandbox:
   - `working_folder` from metadata (if it contains `_sandbox`)
   - `{SANDBOX_BASE}/{task_name}_sandbox`
   - Common path variations (`/data/yjh/`, `/home/yjh/`)
3. **Glob for JSON files**: Search for `generated_docs_*.json` and `final_function_list.json`
4. **Copy to standardized location**: Place files in `processed_tasks/Task_XX_name/generated_docs/`

**Results**: 50 out of 200 tasks had discoverable generated documentation files (the remaining 150 tasks had sandbox directories with different naming conventions or locations not covered by the search patterns).

---

## Automated Processing Pipeline

### Pipeline Architecture

The standardization pipeline (`batch_process.py`) processes each task through **7 sequential steps**:

```
Step 1: Create directory structure
    └── Task_XX_name/
        ├── src/          # Source code
        ├── data/         # Data files
        ├── docs/         # Documentation
        ├── results/      # Output results
        └── metadata.json # Task metadata

Step 2: Copy source code
    └── Copy gt_code_path → src/main.py
    └── Copy additional .py files

Step 3: Copy data files
    └── Copy input/GT/recon files → data/

Step 4: Extract requirements
    └── Run pip freeze → requirements.txt

Step 5: Generate metadata.json
    └── Structured task information

Step 6: Copy visualization
    └── Copy vis_result.png → results/

Step 7: Validate and log
    └── Check completeness → processing_results.json
```

### Parallel Processing

Tasks were processed in parallel using Python's `ThreadPoolExecutor`:
```bash
python batch_process.py --start 1 --end 200 --workers 8
```

### Error Handling

- Tasks with missing files were logged but not skipped entirely
- Partial results were preserved (e.g., if code exists but data is missing)
- A comprehensive `processing_results.json` tracks success/failure for each task

---

## Directory Structure Standardization

### Final Output Structure

Each task is organized into a standardized directory:

```
processed_tasks/
├── README.md                    # Project overview
├── METHODOLOGY.md               # This document
├── task_mapping.json            # Master index of all 200 tasks
├── processing_results.json     # Processing status for each task
│
├── Task_01_sim/
│   ├── src/
│   │   └── main.py             # Main reconstruction code
│   ├── data/
│   │   ├── input.npy           # Input measurements
│   │   ├── gt_output.npy       # Ground truth
│   │   └── recon_output.npy    # Reconstruction result
│   ├── docs/
│   │   └── metadata.json       # Task metadata
│   ├── results/
│   │   └── vis_result.png      # Visualization
│   ├── generated_docs/         # AI-generated function docs
│   │   ├── final_function_list.json
│   │   ├── generated_docs_func1.json
│   │   └── generated_docs_func2.json
│   └── requirements.txt        # Python dependencies
│
├── Task_02_ptyrad/
│   └── ...
├── ...
└── Task_200_vibtest_modal/
    └── ...
```

### Distribution Strategy

- **GitHub** ([starpacker/inverse_benchmark_details](https://github.com/starpacker/inverse_benchmark_details)): Source code, metadata, documentation, generated docs (text-based files)
- **HuggingFace** ([csusupergear/inverse_benchmark_details](https://huggingface.co/datasets/csusupergear/inverse_benchmark_details)): Complete dataset including large binary data files (`.npy`, `.npz`, `.mat`, `.h5`, images)

---

## Quality Metrics and Evaluation

### Standard Metrics

Each task is evaluated using standard image/signal quality metrics when applicable:

| Metric | Description | Range |
|--------|-------------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher is better (dB) |
| **SSIM** | Structural Similarity Index | 0–1 (1 = perfect) |

### Domain-Specific Metrics

Some tasks use additional domain-specific metrics:
- **Optimization Loss**: Convergence of iterative solvers
- **Strehl Ratio**: Optical quality metric
- **DM RMS**: Deformable mirror root-mean-square error
- **Residual Phase Error**: Wavefront sensing accuracy
- **Spectral Angle Mapper (SAM)**: Hyperspectral unmixing accuracy
- **Chi-squared**: Statistical goodness of fit

---

## Tools and Scripts Used

| Script | Purpose |
|--------|---------|
| `automation/parse_report.py` | Parse `Report.md` to extract task metadata into JSON |
| `automation/batch_process.py` | Batch process all 200 tasks: copy code, data, generate metadata |
| `copy_generated_docs.py` | Copy `generated_docs_*.json` and `final_function_list.json` from sandbox directories |
| `upload_to_hf.py` | Upload complete dataset to HuggingFace (via hf-mirror.com) |

### Environment

- **Python**: 3.8+ with conda environments per task
- **Git**: Version control and GitHub hosting
- **HuggingFace Hub**: Dataset hosting for large binary files
- **Processing**: Parallel processing with 8 workers on Linux server

---

## Reproducibility Notes

1. **Conda Environments**: Each task was executed in its own isolated conda environment to avoid dependency conflicts
2. **Sandbox Isolation**: Each task's code ran in a dedicated `_sandbox` directory
3. **Data Provenance**: All input data is either synthetically generated within the code or sourced from the original paper's repository
4. **Deterministic Seeds**: Where applicable, random seeds were fixed for reproducibility

---

## Contact

For questions about this benchmark or the data collection methodology, please refer to the repository's issue tracker or contact the maintainers through the GitHub repository.
