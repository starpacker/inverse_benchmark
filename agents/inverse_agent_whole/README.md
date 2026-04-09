# Inverse Agent Whole — End-to-End Multi-Agent Solver Pipeline

An autonomous multi-agent system that solves scientific inverse problems **end-to-end**: given a task, it plans an algorithmic solution, generates executable Python code, runs it in a sandbox, evaluates the output, and iteratively refines the solution through typed feedback loops.

## Overview

The pipeline orchestrates **four phases** with specialized LLM agents:

| Phase | Agent | Role |
|-------|-------|------|
| **Phase 0** | Dataset Constructor | Generates task descriptions, golden plans, and oracle verification scripts from GT code |
| **Phase 1** | Planner + Critic | Produces a structured JSON algorithmic plan, reviewed by a Critic for logical consistency |
| **Phase 2** | Coder + Executor | Translates plan into Python, runs in sandboxed environment, auto-fixes runtime errors |
| **Phase 3** | Evaluator + Feedback | Compares output against ground truth (PSNR), classifies failures, and routes fixes |

### Typed Feedback Loop

When evaluation fails, the system automatically classifies the error:

- **Type A** (Plan Error) → re-plan from Phase 1
- **Type B** (Code Bug) → re-code from Phase 2
- **Type C** (Hyperparameter Tuning) → adjust parameters in Phase 2

## Architecture

```
Phase 0 (Dataset)  →  Phase 1 (Planning)  →  Phase 2 (Coding)  →  Phase 3 (Evaluation)
                           ▲                                            │
                           │     ← Type A (Plan Error) ────────────────┤
                           │                    ← Type B/C (Code Bug) ─┘
```

## Key Features

- **4-Phase Pipeline**: Automated plan → code → execute → evaluate cycle
- **Self-Repair Loop**: Typed error classification routes to correct repair phase (up to 3 outer iterations)
- **ELO Scoring**: Compares generated plans against golden plans via LLM judge
- **Code Scoring**: Structural similarity metrics (AST-based) between generated and GT code
- **Sandbox Isolation**: Each task runs in a fresh sandboxed directory
- **Multi-Model Support**: 7+ LLMs tested (GPT-5.2, Claude Opus 4.5, Gemini 3 Pro, DeepSeek, Qwen, GLM, Kimi)
- **Multi-GPU Parallel Execution**: Distributes tasks across GPUs via multiprocessing
- **Ablation Mode**: Skip planning to isolate coding ability evaluation

## Project Structure

```
├── pipeline.py                 # Main orchestrator (InverseAgentPipeline)
├── pipeline_ablation.py        # Ablation variant (golden plan → code only)
├── run_suite_all_model.py      # Multi-model benchmark runner
├── analyze_pipeline.py         # Execution log analysis
├── agents/
│   ├── phase0_dataset.py       # Dataset & oracle construction
│   ├── phase1_planner.py       # Plan generation + self-critic
│   ├── phase2_coder.py         # Code generation + execution
│   ├── phase3_evaluator.py     # Result analysis + typed feedback
│   ├── elo_scorer.py           # Plan quality scoring (LLM judge)
│   └── code_scorer.py          # Code similarity metrics
├── prompts/
│   ├── phase0.py – phase3.py   # LLM prompt templates per phase
├── config/
│   ├── settings.py             # Global settings (paths, timeouts, defaults)
│   ├── config2.yaml            # Model API configurations
│   ├── config_task.yaml        # Task IO path mappings
│   └── tasks_split_*.yaml      # GPU shard task configs
└── utils/
    ├── llm.py                  # Unified LLM client (OpenAI-compatible)
    ├── sandbox.py              # Sandbox environment management
    └── logger.py               # JSON-lines run logging
```

## Usage

```bash
# Run full pipeline on all tasks with a specific model
python run_suite_all_model.py

# Run ablation (golden plan, coding only)
python pipeline_ablation.py

# Analyze execution results
python analyze_pipeline.py
```

## Configuration

- **`config/config2.yaml`**: LLM model API keys and endpoints
- **`config/config_task.yaml`**: Task definitions with input/output paths
- **`config/settings.py`**: Global settings (timeouts, max retries, default models)
