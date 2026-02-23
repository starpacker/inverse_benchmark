# Inverse Benchmark — LLM-Driven Scientific Inverse Problem Solving

A comprehensive benchmark suite for evaluating and leveraging Large Language Models (LLMs) on **scientific inverse problems** — tasks that recover hidden parameters from observed data, spanning computational imaging, seismic inversion, medical imaging, remote sensing, and more.

## Projects

This repository contains six interconnected projects:

| Project | Description |
|---------|-------------|
| [**agentic_pipeline_dev**](./agentic_pipeline_dev/) | Multi-agent system (Planner→Architect→Coder→Judge) for automatic solver code generation with a persistent knowledge system |
| [**agentic_reproduce**](./agentic_reproduce/) | Paper-driven code reproduction — reads a research paper and autonomously generates working solver implementations |
| [**inverse_agent_whole**](./inverse_agent_whole/) | End-to-end agent pipeline with typed feedback loops (Plan Error / Code Bug / Tuning) and multi-phase autonomous solving |
| [**inverse_planning_eval**](./inverse_planning_eval/) | Planning evaluation framework with TextGrad prompt optimization and multi-judge ELO tournament |
| [**new_flow**](./new_flow/) | Automated pipeline for transforming scientific code into structured tutorials and benchmark questions |
| [**react_inverse_problem**](./react_inverse_problem/) | Multi-model benchmark evaluating 7+ LLMs on function-level scientific code generation via ReAct loops |

## System Overview

```
                    ┌─────────────────────────────┐
                    │     Research Papers (PDF)     │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                          ▼
┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐
│  new_flow        │     │ agentic_reproduce│       │ agentic_pipeline│
│  (Tutorial &     │     │ (Paper → Code   │       │ (GT Code →      │
│   Question Gen)  │     │  Reproduction)  │       │  Auto Solver)   │
└────────┬────────┘     └─────────────────┘       └─────────────────┘
         │
    ┌────┴─────────────────────┐
    ▼                          ▼
┌─────────────────┐  ┌──────────────────────┐
│react_inverse_   │  │inverse_planning_eval │
│problem           │  │(Plan Quality         │
│(Code Generation  │  │ Evaluation + ELO)    │
│ Benchmark)       │  └──────────────────────┘
└─────────────────┘
         │
         ▼
┌──────────────────────┐
│inverse_agent_whole   │
│(End-to-End Agent     │
│ Pipeline Evaluation) │
└──────────────────────┘
```

### Workflow

1. **new_flow** processes scientific code and papers → produces tutorials and coding questions
2. **react_inverse_problem** uses those questions to benchmark multiple LLMs (GPT-5.2, Claude Opus 4.5, Gemini 3 Pro, DeepSeek, Qwen, etc.)
3. **inverse_planning_eval** evaluates LLM plan generation quality via similarity metrics and ELO tournament
4. **inverse_agent_whole** runs end-to-end agent evaluation with typed feedback (plan error → recode → tune)
5. **agentic_pipeline_dev** provides end-to-end solver generation with multi-agent collaboration and knowledge accumulation
6. **agentic_reproduce** extends the pipeline to work directly from research papers without ground truth code

## Covered Domains

- **Optical Imaging**: Microscopy deconvolution, phase retrieval, holographic imaging, ptychography
- **Medical Imaging**: CT reconstruction, MRI, PET, MR elastography
- **Seismology**: Full waveform inversion, seismic tomography, velocity modeling
- **Remote Sensing**: InSAR phase unwrapping, SAR imaging
- **Signal Processing**: Spectral unmixing, compressed sensing, blind deconvolution
- **Physics**: Gravitational lensing, quantum state tomography, electromagnetic inversion

## Supported LLMs

GPT-5.2 · Claude Opus 4.5 · Gemini 3 Pro · DeepSeek V3.2 · Qwen3 Max · GLM-4.7 · Kimi K2 · Grok 3 · and more

## Getting Started

See each project's README for specific setup and usage instructions:
- [agentic_pipeline_dev/README.md](./agentic_pipeline_dev/README.md)
- [agentic_reproduce/README.md](./agentic_reproduce/README.md)
- [inverse_agent_whole/README.md](./inverse_agent_whole/README.md)
- [inverse_planning_eval/README.md](./inverse_planning_eval/README.md)
- [new_flow/README.md](./new_flow/README.md)
- [react_inverse_problem/README.md](./react_inverse_problem/README.md)
