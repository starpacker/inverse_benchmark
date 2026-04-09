# New Flow — Automated Tutorial Generation for Scientific Code

An automated pipeline that transforms raw scientific code and research papers into structured tutorials with auto-generated coding questions. The system cleans and refactors code, runs automated unit tests, generates teaching materials, and produces benchmark questions for evaluating code comprehension.

## Overview

The pipeline performs four main stages:

1. **PDF → Markdown (OCR)**: Converts research papers into machine-readable Markdown using PaddleOCR
2. **Code Cleaning & Testing**: Refactors legacy scientific code into a standard 4-function structure, then runs automated unit testing with data capture
3. **Tutorial Generation**: Combines paper content and validated code to produce detailed scientific tutorials with both theoretical and implementation sections
4. **Question Generation**: Automatically creates coding interview questions targeting specific functions for benchmark evaluation

## Pipeline Architecture

```
Paper (PDF)  +  Raw Code
     │              │
     ▼              ▼
  OCR/Parse    Code Clean-up & Refactor
     │         (→ 4 standard functions)
     │              │
     │              ▼
     │         Unit Testing
     │         (Data Capture + Verification)
     │              │
     └──────┬───────┘
            ▼
     Tutorial Writer
     (Theory + Implementation)
            │
            ▼
     Question Generator
     (Coding Interview Questions)
```

### Standard 4-Function Structure

All scientific code is refactored into:
1. `load_and_preprocess_data()` — Data loading and preprocessing
2. `forward_operator()` — Forward model / physical transform
3. `run_inversion()` — Inverse solver / optimization
4. `evaluate_results()` — Result evaluation and metrics

## Project Structure

```
├── run_pipeline.py           # Main pipeline orchestrator
├── run_pipeline.sh           # Shell script runner
├── clean_up_code.py          # Code refactoring via LLM (→ 4-function structure)
├── uni_test.py               # Automated unit testing with data capture
├── tutorial_writer.py        # Tutorial generation with verification loop
├── make_up_question.py       # Coding question generation from tutorials
├── verification_utils.py     # Utilities for output comparison
├── utils.py                  # Helper functions (code loading, tool generation)
├── run_ocr_tool.py           # PDF to Markdown conversion (PaddleOCR)
├── config.yaml               # LLM and prompt configuration
└── input/                    # Input papers (PDF/Markdown)
```

## Usage

```bash
# Full pipeline run
python run_pipeline.py \
  --pdf input/paper.pdf \
  --code /path/to/original_code.py \
  --command "python /path/to/original_code.py" \
  --working_folder /path/to/working/dir \
  --working_folder_file /path/to/working/dir/code.py \
  --saving_folder ./history_task/ \
  --tutorial_name "my_tutorial" \
  --function_folder ./functions/

# Resume from a specific step (0=Start, 1=OCR, 2=CodeClean, 3=Tutorial, 4=Questions)
python run_pipeline.py --step 3 ...
```

## Configuration

Edit `config.yaml` to configure:
- **LLM providers**: Code generation, tutorial writing, tool writing, embedding models
- **Prompts**: Code cleanup instructions, test script generation, tutorial templates
- **Output directories**: Working directories for intermediate files
