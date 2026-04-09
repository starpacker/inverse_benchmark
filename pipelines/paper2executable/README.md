# Paper2Executable

**Automated Pipeline: From Scientific Papers to Reproducible Executable Code**

An end-to-end system that:
1. **Discovers** computational imaging / inverse problem papers from ArXiv
2. **Extracts** paper content via PDF OCR (PaddleOCR on GPU)
3. **Locates** source code via PapersWithCode API, web search, or OCR-extracted links
4. **Generates** standardized benchmark artifacts using OpenHands AI agents:
   - `input.npy` / `gt_output.npy` — test data
   - `run_gt.py` — consolidated inference script
   - `evaluate.py` — PSNR/SSIM/RMSE evaluation
   - `baseline.npy` — model output for comparison

## What This Pipeline Does

Given an **ArXiv ID** (e.g., `2108.10257`), the pipeline automatically:

```
ArXiv ID → PDF OCR → GitHub Discovery → Clone Repo → OpenHands Agent Tasks → Working Folder
```

**Final Output:** A self-contained `working_folder/` with everything needed to benchmark the paper's algorithm:

```
working_folder/SwinIR/
├── dataset/
│   ├── input.npy          # Input data (e.g., low-res image)
│   ├── gt_output.npy      # Ground truth (e.g., high-res image)
│   ├── baseline.npy       # Model inference output
│   └── aux_data.npy       # Auxiliary parameters
├── run_gt.py              # Consolidated GT inference script
├── evaluate.py            # Evaluation script (PSNR/SSIM/RMSE)
├── result.json            # Evaluation metrics
├── repo/                  # Symlink to cloned repository
└── outputs/
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Paper2Executable Pipeline                                │
├─────────────┬──────────────┬─────────────────┬──────────────┬───────────────────┤
│  Step 1     │  Step 2      │  Step 3         │  Step 4-6    │  Step 7           │
│  ArXiv      │  PDF OCR     │  Code           │  OpenHands   │  Output           │
│  Metadata   │  (GPU)       │  Discovery      │  Agent Tasks │  Assembly         │
├─────────────┼──────────────┼─────────────────┼──────────────┼───────────────────┤
│ • Fetch     │ • PaddleOCR  │ • Parse OCR     │ • Data       │ • Copy to         │
│   title,    │   PPStructV3 │   for GitHub    │   Extraction │   working_folder  │
│   abstract, │ • GPU-accel  │ • PWC API       │ • Code       │ • Reorganize      │
│   authors   │ • Markdown   │ • Web search    │   Consolidate│   dataset/        │
│ • ArXiv ID  │   output     │ • Auto clone    │ • Evaluate   │ • Generate        │
│   validation│              │                 │   Generation │   report.json     │
└─────────────┴──────────────┴─────────────────┴──────────────┴───────────────────┘
```

### OpenHands Agent Tasks (Phase 3)

The pipeline uses **OpenHands CodeActAgent** to perform three key tasks:

| Task | Input | Output | Description |
|------|-------|--------|-------------|
| **Data Extraction** | Repo code | `input.npy`, `gt_output.npy`, `aux_data.npy` | Extract/generate test data from paper's examples |
| **Code Consolidation** | Repo code + data | `run_gt.py`, `baseline.npy` | Flatten repo into single inference script |
| **Evaluate Generation** | `baseline.npy`, `gt_output.npy` | `evaluate.py`, `result.json` | Generate evaluation script with metrics |

## Project Structure

```
paper2executable/
├── config/
│   └── config.yaml           # Main pipeline configuration (OCR, OpenHands, paths)
├── database/
│   ├── models.py             # SQLAlchemy ORM models (Paper, DiscoveryLog, ExecutionLog)
│   ├── manager.py            # Database CRUD operations
│   └── papers.db             # SQLite database (auto-created)
├── phase1_baseline/
│   ├── importer.py           # Import existing benchmark data into index
│   └── dedup.py              # Deduplication by ArXiv ID / title similarity
├── phase2_discovery/
│   ├── pdf_processor.py      # PaddleOCR PDF → Markdown (GPU-accelerated)
│   ├── pwc_api.py            # PapersWithCode API for GitHub lookup
│   ├── web_search.py         # DuckDuckGo + LLM verification
│   └── github_ops.py         # Clone, sparse checkout, structure analysis
├── phase3_openhands/
│   ├── openhands_runner.py   # OpenHands subprocess manager
│   ├── prompt_templates.py   # System prompts for 3 agent tasks
│   ├── run_oh_subprocess.py  # OpenHands execution wrapper
│   └── output_builder.py     # Assemble working_folder + postprocess
├── tools/
│   ├── arxiv_collector.py    # Batch ArXiv paper collection by topic
│   └── run_ocr_tool.py       # Standalone OCR utility
├── utils/
│   ├── arxiv_utils.py        # ArXiv metadata fetching
│   ├── logging_utils.py      # Structured logging
│   └── task_description.py   # Generate task_description.md
├── orchestrator.py           # Main pipeline orchestrator (process_paper_e2e)
├── cli.py                    # Command-line interface
└── requirements.txt          # Python dependencies
```

## Installation

```bash
# 1. Clone and enter directory
cd /home/yjh/paper2executable

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Ensure OpenHands is available
#    (Expected at /home/yjh/OpenHands with 'openhands' conda env)

# 4. Configure GPU for OCR (optional, edit config/config.yaml)
#    ocr.use_gpu: true
#    ocr.gpu_id: 1  # Choose available GPU
```

## CLI Commands

### Core Pipeline

```bash
# Process a single paper (full pipeline)
python cli.py run --arxiv-id 2108.10257

# Process multiple papers
python cli.py run-batch 2108.10257 2209.14687 2206.00942

# Check pipeline status
python cli.py status
```

### Paper Collection

```bash
# Collect papers from ArXiv (default topics: inverse problems, imaging)
python cli.py collect --top-n 20

# Collect with specific topics
python cli.py collect -t "ptychography" -t "phase retrieval" --days-back 90

# Fast mode (skip slow PapersWithCode API check)
python cli.py collect --skip-pwc --no-require-code

# Auto-queue top papers for processing
python cli.py collect --auto-queue --queue-count 5

# Process papers from a collected JSON file
python cli.py process-collected /path/to/collected_20260303.json --top-n 10
```

### Discovery & Search

```bash
# Quick discovery (search + process)
python cli.py discover --topic "computational imaging" --max-results 5

# Initialize database with existing benchmark data
python cli.py init-baseline
```

### Command Reference

| Command | Description |
|---------|-------------|
| `run --arxiv-id ID` | Full E2E pipeline for one paper |
| `run-batch ID1 ID2 ...` | Process multiple papers |
| `collect` | Collect papers from ArXiv by topic |
| `process-collected FILE` | Process papers from a JSON collection |
| `discover --topic TOPIC` | Quick search and process |
| `init-baseline` | Import existing benchmark into DB |
| `status` | Show pipeline statistics |

## Configuration

Edit `config/config.yaml` to customize:

```yaml
general:
  workspace_base: /data/yjh/paper2executable_workspace
  log_level: INFO

ocr:
  use_gpu: true
  gpu_id: 1                    # GPU for PaddleOCR (avoid busy GPU 0)
  timeout: 600                 # OCR timeout in seconds

openhands:
  staging_dir: /data/yjh/paper2executable_workspace/staging
  step_limits:
    data_extraction: 50        # Max iterations per agent task
    code_consolidation: 60
    evaluate_generation: 40

baseline:
  benchmark_dataset_path: /data/yjh/benchmark_dataset
  task_descriptions_path: /data/yjh/task_descriptions
```

## Output Paths

After running `python cli.py run --arxiv-id 2108.10257`:

| Output | Path |
|--------|------|
| **Working Folder** | `/data/yjh/paper2executable_workspace/working/SwinIR/` |
| **Task Description** | `/data/yjh/task_descriptions/SwinIR_description.md` |
| **Staging (temp)** | `/data/yjh/paper2executable_workspace/staging/SwinIR/` |
| **OCR Markdown** | `/data/yjh/paper2executable_workspace/ocr_results/2108.10257.md` |
| **Cloned Repo** | `/data/yjh/paper2executable_workspace/repos/SwinIR/` |
| **Schema Report** | `/data/yjh/paper2executable_workspace/output/reports/SwinIR_report.json` |

## Output Schema

Each processed paper produces:

**Working Folder Structure:**
```
working/SwinIR/
├── dataset/
│   ├── input.npy          # (H,W,C) or (C,H,W) float32
│   ├── gt_output.npy      # Ground truth output
│   ├── baseline.npy       # Model inference result
│   └── aux_data.npy       # [scale, window_size, ...]
├── run_gt.py              # Self-contained inference script
├── evaluate.py            # PSNR/SSIM/RMSE computation
├── result.json            # {"psnr": 32.5, "ssim": 0.89, "rmse": 0.02}
└── repo/                  # Symlink to cloned repository
```

**Schema Report (JSON):**
```json
{
  "task_name": "SwinIR",
  "paper_info": {
    "arxiv_id": "2108.10257",
    "title": "SwinIR: Image Restoration Using Swin Transformer",
    "authors": ["Jingyun Liang", "..."],
    "github_url": "https://github.com/JingyunLiang/SwinIR"
  },
  "outputs": {
    "working_folder": "/data/.../working/SwinIR",
    "gt_code_path": "/data/.../staging/SwinIR/run_gt.py",
    "task_description_path": "/data/.../SwinIR_description.md"
  },
  "status": "ready"
}
```

## Example: Full Pipeline Run

```bash
$ python cli.py run --arxiv-id 2108.10257

[E2E] Processing: 2108.10257
INFO  Step 1: Fetch ArXiv metadata
INFO    Title: SwinIR: Image Restoration Using Swin Transformer
INFO  Step 2: PDF OCR (GPU 1)
INFO    OCR completed in 172.3s, 66908 chars
INFO  Step 3: Code discovery
INFO    Found GitHub: https://github.com/JingyunLiang/SwinIR
INFO  Step 4: Clone repository
INFO  Step 5a: Data extraction (OpenHands)
INFO    ✅ Produced input.npy, gt_output.npy, aux_data.npy
INFO  Step 5b: Code consolidation (OpenHands)
INFO    ✅ Produced run_gt.py, baseline.npy
INFO  Step 5c: Evaluate generation (OpenHands)
INFO    ✅ Produced evaluate.py, result.json
INFO  Step 6: Generate task description
INFO  Step 7: Assemble output artifacts

============================================================
Status: completed
  working_folder: /data/yjh/paper2executable_workspace/working/SwinIR
  gt_code_path: /data/yjh/paper2executable_workspace/staging/SwinIR/run_gt.py
============================================================
```

## Paper Collection Example

```bash
$ python cli.py collect -t "ptychography" --skip-pwc --top-n 5

======================================================================
ArXiv Paper Collector - Computational Imaging / Inverse Problems
======================================================================
Topics: ['ptychography']
Filters: code=False, python=True, days=60, skip_pwc=True
----------------------------------------------------------------------
INFO  Searching ArXiv for: 'ptychography' (max 30)
INFO  Found 15 new papers

================================================================================
COLLECTED PAPERS SUMMARY
================================================================================
Total papers: 8
With code: 5
Python-based: 5

Top 5 by relevance:
--------------------------------------------------------------------------------
 1. [14.0] ✅🐍 2603.01837  Constrained Particle Seeking...
 2. [13.3] ✅🐍 2602.20417  gQIR: Generative Quanta Image Reconstruction
 3. [12.0] ✅🐍 2602.10344  Monte Carlo Maximum Likelihood...
```

## Integration with Agentic Pipeline

The output working folder is compatible with the downstream `agentic_pipeline_dev`:

```
working/SwinIR/
├── dataset/           ← agentic pipeline reads from here
│   ├── input.npy      ← solver loads: np.load('dataset/input.npy')
│   ├── gt_output.npy  ← eval_script loads (hidden from solver)
│   └── baseline.npy   ← eval_script loads (hidden from solver)
├── run_gt.py          ← reference GT code
└── evaluate.py        ← evaluation template
```

## Dependencies

- **Python 3.10+**
- **OpenHands** — AI coding agent (local installation at `/home/yjh/OpenHands`)
- **PaddleOCR** — PDF to Markdown extraction (GPU-accelerated)
- **SQLite** — Paper index database (via SQLAlchemy)
- **arxiv** — ArXiv API client
- **scikit-image** — PSNR/SSIM metrics

```bash
# Core dependencies
pip install arxiv requests sqlalchemy pyyaml click numpy scikit-image

# OCR (separate conda env recommended)
pip install paddlepaddle-gpu paddleocr

# OpenHands (separate conda env)
# See /home/yjh/OpenHands for setup
```

## Default Collection Topics

The paper collector searches for these topics by default:

```python
DEFAULT_TOPICS = [
    "computational imaging",
    "inverse problem",
    "image reconstruction",
    "phase retrieval",
    "ptychography",
    "tomographic reconstruction",
    "deconvolution",
    "super resolution",
    "image restoration",
    "compressed sensing imaging",
    "holography reconstruction",
    "diffraction imaging",
]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OCR fails with CUDA OOM | Set `ocr.gpu_id` to a free GPU in config |
| OpenHands timeout | Increase `step_limits` in config |
| GitHub clone fails | Check network / rate limits |
| PWC API slow | Use `--skip-pwc` flag for fast collection |
| Missing `openhands` env | Run `conda activate openhands` first |

## License

MIT
