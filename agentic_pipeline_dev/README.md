# Agentic Pipeline — Multi-Agent Code Generation for Scientific Inverse Problems

A multi-agent AI system that automatically generates, debugs, and optimizes Python solvers for **scientific inverse problems** — spanning computational imaging, seismic inversion, medical imaging, remote sensing, and more.

## Overview

The system orchestrates **four specialized LLM agents** in a structured pipeline:

| Agent | Role |
|-------|------|
| **Planner** | Generates mathematical/algorithmic solution plans (with Critic review loop) |
| **Architect** | Converts plans into Python code skeletons with class structure and method signatures |
| **Coder** | Implements functions one-by-one using AST-based precise code editing |
| **Judge** | Diagnoses execution failures and dispatches targeted fix tickets |

### Key Features

- **Iterative Self-Repair**: Failed code is analyzed by the Judge and selectively repaired (up to N retries)
- **Persistent Knowledge System**: Three-tier knowledge base (Core / Experience / Instance) accumulates expertise across tasks
- **AST-Based Code Editing**: Function-level precise replacement preserving file structure
- **46+ Scientific Tasks**: Pre-configured benchmarks across optics, seismology, medical imaging, etc.
- **Multi-LLM Support**: Compatible with 11+ LLM backends (Gemini, GPT-5.2, Claude Opus 4.5, Qwen, DeepSeek, Grok, etc.)

## Architecture

```
┌──────────┐    ┌───────────┐    ┌────────┐    ┌───────────────┐
│  Planner │───>│ Architect │───>│ Coder  │───>│ Execute+Judge │
│  Agent   │    │   Agent   │    │ Agent  │    │    Loop       │
└──────────┘    └───────────┘    └────────┘    └───────┬───────┘
     ▲                                                  │
     │              ┌──────────────────┐               │
     └──────────────│  Skill System    │◄──────────────┘
                    │(Core/Exp/Instance)│
                    └──────────────────┘
```

## Project Structure

```
├── main_flow.py              # Entry point, batch orchestration
├── workflow_base.py          # Core workflow engine (Planner→Architect→Coder→Judge loop)
├── reporting.py              # Execution report generation
├── agents/                   # LLM agent implementations
│   ├── planner_agent.py      # Mathematical planning + Critic agent
│   ├── architect_agent.py    # Code skeleton generation
│   ├── coder_agent.py        # Function-by-function implementation
│   ├── judge_agent.py        # Error diagnosis & fix dispatch
│   └── sandbox_manager.py    # Data generation & evaluation agents
├── persistent_skill_system/  # Three-tier knowledge base
│   ├── storage.py            # SQLite storage with vector similarity search
│   ├── manager.py            # Knowledge retrieval & distillation orchestrator
│   ├── teacher.py            # Trajectory → knowledge extraction (LLM-based)
│   └── evolution_manager.py  # Offline knowledge evolution (DBSCAN + LLM induction)
├── config/
│   ├── config_task.yaml      # Task definitions (27 tasks)
│   ├── config_task_2.yaml    # Additional task definitions (26 tasks)
│   └── config_llm.yaml       # LLM model configurations
├── utils/
│   └── code_editor.py        # AST-based code editing tool
└── scripts/
    └── manage_skills.py      # Skill/trajectory inspection CLI
```

## Usage

```bash
# Run pipeline on all tasks
python main_flow.py --config config/config_task.yaml --model gemini_25_pro

# Run specific tasks
python main_flow.py --config config/config_task.yaml --model gpt_52 --tasks sim,deconv

# Check skill database
python scripts/manage_skills.py stats

# Trigger knowledge evolution
python scripts/manage_skills.py evolve
```

## Configuration

- **Task config** (`config/config_task.yaml`): Define tasks with ground truth code path, working directory, conda environment, and max retries
- **LLM config** (`config/config_llm.yaml`): Configure LLM backends using OpenAI-compatible API format

See [TECHNICAL_DOCUMENT_CN.md](TECHNICAL_DOCUMENT_CN.md) for comprehensive technical documentation.
