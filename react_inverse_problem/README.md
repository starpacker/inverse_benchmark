# ReAct Inverse Problem — Multi-Model Benchmark for Scientific Code Generation

A benchmark framework that evaluates multiple LLMs on their ability to generate scientific function implementations for inverse problems. Using a **ReAct (Reasoning + Acting) loop**, models iteratively write code, receive execution feedback, and refine their solutions until tests pass or max rounds are reached.

## Overview

The system benchmarks LLMs on function-level code generation tasks derived from scientific inverse problem codebases. Each task provides:
- A **function signature** and documentation (auto-generated from code)
- A **code skeleton** with imports and helper functions
- An **automated test harness** that validates correctness against reference outputs

Models iterate through a ReAct loop: generate code → execute tests → receive error feedback → refine code.

## Supported Models

| Model | Config Key |
|-------|-----------|
| GPT-5.2 (Thinking) | `gpt-5.2-thinking` |
| Claude Opus 4.5 (Thinking) | `claude-opus-4-5-20251101-thinking` |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` |
| DeepSeek V3.2 | `deepseek-v3.2` |
| Qwen3 Max | `qwen3-max` |
| GLM-4.7 | `glm-4.7` |
| Kimi K2 | `kimi-k2-0905` |

## Architecture

```
Task (function docs + skeleton)
        │
        ▼
  ┌─────────────────┐
  │  LLM generates  │◄──── Error feedback
  │  function code   │        │
  └────────┬────────┘        │
           │                  │
           ▼                  │
  ┌─────────────────┐        │
  │  Code Sandbox   │        │
  │  (Execute test) │────────┘
  └────────┬────────┘  (if failed, up to N rounds)
           │
           ▼ (if passed)
     ✅ SUCCESS
```

## Project Structure

```
├── react_code_log_gen_docs.py        # Main entry point: ReAct loop with doc-based prompts
├── code_dev_env.py                    # Code sandbox execution environment
├── code_dev_env_info.py               # Enhanced sandbox with info extraction
├── filter_question.py                 # Question filtering and skeleton generation
├── compare_results.py                 # Cross-model result comparison
├── schema_validation.py               # Output schema validation
├── benchmark_results/                 # Results organized by model
│   ├── claude-opus-4-5-20251101-thinking/
│   ├── gpt-5.2-thinking/
│   ├── gemini-3-pro-preview/
│   ├── deepseek-v3.2/
│   ├── qwen3-max/
│   ├── glm-4.7/
│   ├── kimi-k2-0905/
│   └── golden_references/            # Ground truth reference outputs
├── config/
│   ├── config.yaml                    # Main config (tasks, paths, settings)
│   ├── config_claude.yaml             # Claude-specific config
│   ├── config_gpt.yaml               # GPT-specific config
│   └── ...                            # Per-model configs
├── scripts/
│   ├── async_llm.py                   # Async LLM client wrapper
│   ├── evaluator.py                   # Result evaluation
│   ├── workflow.py                    # Workflow orchestration
│   └── operators.py                   # LLM operator definitions
└── workspace/                         # Task workspaces with workflow templates
    ├── InverseProb/
    ├── InverseProb2/
    ├── InverseProb3/
    └── InverseProb4/
```

## Usage

```bash
# Run benchmark with a specific model
python react_code_log_gen_docs.py --config_path config/config.yaml --model_name deepseek-v3.2

# Run a specific task only
python react_code_log_gen_docs.py --config_path config/config.yaml --model_name gpt-5.2-thinking --task sim

# Run with custom max rounds
python react_code_log_gen_docs.py --config_path config/config.yaml --model_name gemini-3-pro-preview --max_rounds 15
```

## Benchmark Results

Results are stored in `benchmark_results/<model_name>/` as JSON files, with each file containing per-task metrics. Golden references are available in `benchmark_results/golden_references/` for comparison.

## Configuration

- **Task config** (`config/config.yaml`): Define tasks with working folders, python paths, and question sources
- **Model configs** (`config/config_*.yaml`): Per-model LLM settings
- **Global settings**: Default model, max rounds, sandbox root directory
