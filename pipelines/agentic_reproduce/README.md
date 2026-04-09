# Agentic Reproduce — Paper-Driven Scientific Code Reproduction

An end-to-end system that automatically reproduces scientific inverse problem solvers **directly from research papers**. Given a paper (PDF or Markdown), it generates task descriptions, synthetic test data, evaluation scripts, and a working solver implementation through multi-agent collaboration.

## Overview

Unlike the standard agentic pipeline which requires ground truth code as input, this system is **paper-driven** — it reads a research paper and autonomously:

1. **Extracts task descriptions** from the paper content via LLM
2. **Generates synthetic test data** (`data_gen.py` → `input.npy`, `gt_output.npy`, `baseline.npy`)
3. **Creates evaluation scripts** with automatic baseline metric validation
4. **Produces a working solver** through the Planner → Architect → Coder → Judge pipeline

## Architecture

```
Paper (PDF/Markdown)
        │
        ▼
  ┌─────────────┐
  │ OCR / Parse │ → Task Description
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  Phase 0    │ → data_gen.py + eval_script.py + dataset/
  │ Preparation │
  └─────────────┘
        │
        ▼
  ┌──────────┐  ┌───────────┐  ┌────────┐  ┌───────┐
  │ Planner  │→ │ Architect │→ │ Coder  │→ │ Judge │ ← (iterative loop)
  └──────────┘  └───────────┘  └────────┘  └───────┘
        │
        ▼
  solver.py (validated against eval_script.py)
```

## Key Features

- **Paper-to-Code Pipeline**: No ground truth code required — works from paper content alone
- **Automatic Data Generation**: Creates synthetic datasets for testing
- **Evaluation-Driven**: Auto-generates evaluation scripts; success is determined by PSNR/RMSE thresholds
- **Ticket-Based Repair**: Judge agent assigns fix tickets to Planner/Architect/Coder based on root cause analysis
- **Downstream State Reset**: When re-planning is needed, all downstream artifacts are automatically invalidated
- **Full Experiment Logging**: Each iteration saves plans, skeletons, solver code, execution logs, and judge analysis

## Project Structure

```
├── main_flow.py              # Entry point & InverseProblemWorkflow class
├── agents/
│   ├── planner_agent.py      # Plan generation + Critic review loop
│   ├── architect_agent.py    # Code skeleton generation
│   ├── coder_agent.py        # Function implementation & patching
│   ├── judge_agent.py        # Failure diagnosis & ticket dispatch
│   └── sandbox_manager.py    # DataGen, EvalGen, TaskDesc agents
├── config/
│   ├── config_task.yaml      # Task definitions
│   └── config_llm.yaml       # LLM model configurations
├── paper_archive/             # Cached paper content & task descriptions
└── paper_sandbox/             # Experiment outputs organized by model/task
```

## Usage

```bash
# Run with a specific model on the configured task
python main_flow.py

# The workflow will:
# 1. Read paper from paper_archive/test.md
# 2. Generate task description, data, and eval scripts
# 3. Iteratively generate and refine solver code
# 4. Save all artifacts to paper_sandbox/<model_name>/<task_timestamp>/
```

## Configuration

Edit `config/config_llm.yaml` to configure LLM backends, and modify `main_flow.py` to set the target paper path and model selection.
