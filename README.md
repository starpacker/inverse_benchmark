# Inverse Benchmark — LLM-Driven Scientific Inverse Problem Research

A consolidated research repository covering **LLM-driven approaches to scientific inverse problems** — tasks that recover hidden parameters from observed data, spanning computational imaging, medical imaging, spectroscopy, astrophysics, geophysics, and beyond.

This umbrella repo brings together the full research line: a 200-task standardized benchmark, multiple paper-to-code generation pipelines, evaluation harnesses, agent variants, and a public showcase website.

---

## 📁 Repository Layout

```
inverse_benchmark/
├── tasks/        ← 200 standardized inverse-problem tasks (the benchmark itself)
├── pipelines/    ← Different paper→code generation pipelines (3 implementations)
├── harnesses/    ← Evaluation harnesses for grading LLM agent outputs
├── agents/       ← Agent implementations / variants used by the harnesses
└── website/      ← Next.js showcase frontend
```

| Folder | What it is | Origin |
|---|---|---|
| [`tasks/`](./tasks/) | 200 standardized inverse problem tasks across 8 categories. Each task ships with `src/`, `test/`, `data/`, `notebook.ipynb`, `requirements.txt`, `metadata.json`. | originally `inverse_benchmark_details` |
| [`pipelines/`](./pipelines/) | Three different "paper → executable code" pipelines: `paper2executable` (4-phase), `agentic_reproduce` (multi-agent), `new_flow` (tutorial-driven). | merged from `Paper2Executable` + previously here |
| [`harnesses/`](./harnesses/) | Evaluation harnesses. The standard is `inverse_101` — 3 modes (plan/function/end2end) × 3 frameworks (claude_code/multi_agent/react). | originally `inverse-101`; replaces deprecated `inverse_planning_eval` |
| [`agents/`](./agents/) | Specialised agent implementations: `agentic_pipeline_dev` (multi-agent solver), `inverse_agent_whole` (typed-feedback E2E), `react_inverse_problem` (ReAct multi-LLM), `openhands_benchmark` (OpenHands wrapper). | previously here |
| [`website/`](./website/) | The Next.js gallery showing task results, comparisons, and reproductions. | originally `agent-imaging-website` |

---

## 🧠 The Big Picture

```
                  ┌─────────────────────────────┐
                  │     Research Papers (PDF)    │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │   pipelines/  (paper→code)   │
                  │  • paper2executable          │
                  │  • agentic_reproduce         │
                  │  • new_flow                  │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │   tasks/   (200 standardized) │
                  │   curated benchmark suite    │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │   agents/  +  harnesses/     │
                  │   solve & evaluate           │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │   website/  (public gallery) │
                  └─────────────────────────────┘
```

**Flow:** papers in → standardized tasks out → agents try to solve them → harnesses score the attempts → results published on the website.

---

## 🗂️ Task Categories (200 tasks)

| Category | Tasks | Examples |
|----------|-------|---------|
| Computational Imaging | 50+ | Ptychography, FPM, Lensless, Holography, Light Field |
| Medical Imaging | 30+ | CT, MRI, PET, Ultrasound, OCT |
| Spectroscopy | 20+ | Raman, NMR, X-ray, EIS |
| Astrophysics | 15+ | Gravitational Lensing, Radio Imaging, Stellar Spectroscopy |
| Geophysics | 15+ | Seismic, GPR, ERT, InSAR |
| Signal Processing | 20+ | Source Separation, DOA, Spike Sorting |
| Microscopy | 15+ | Super-resolution, Deconvolution, Phase Retrieval |
| Other | 30+ | DIC, Modal Analysis, Rheology, Diffusion |

See [`tasks/METHODOLOGY.md`](./tasks/METHODOLOGY.md) for how tasks were curated and standardized.

---

## 🚀 Quick Start

Each subfolder is independently runnable. Pick the entry point that matches what you're trying to do:

| You want to… | Go here |
|---|---|
| Browse the 200-task benchmark | [`tasks/`](./tasks/) |
| Generate executable code from a paper | [`pipelines/paper2executable/`](./pipelines/paper2executable/) or [`pipelines/agentic_reproduce/`](./pipelines/agentic_reproduce/) |
| Evaluate an LLM agent on the benchmark | [`harnesses/inverse_101/`](./harnesses/inverse_101/) |
| Run the multi-agent solver | [`agents/agentic_pipeline_dev/`](./agents/agentic_pipeline_dev/) |
| Spin up the showcase website locally | [`website/`](./website/) |

Each subfolder has its own `README.md` with installation, configuration, and usage instructions.

---

## 🤖 Supported LLMs (by the harnesses)

GPT-5.2 · Claude Opus 4.5/4.6 · Gemini 3 Pro · DeepSeek V3.2 · Qwen3 Max · GLM-4.7 · Kimi K2 · Grok 3 — and more, configured in `harnesses/inverse_101/config_llm.yaml`.

---

## 📜 History & Provenance

This repo is the consolidation of a 5-repo research line developed during 2026-Q1. As of **2026-04-10**, the previously separate repos were merged here:

| Original repo | Now lives at |
|---|---|
| `inverse_benchmark` (this repo, original) | top-level + `agents/` + `pipelines/agentic_reproduce` + `pipelines/new_flow` |
| `inverse_benchmark_details` | `tasks/` |
| `Paper2Executable` | `pipelines/paper2executable/` |
| `inverse-101` | `harnesses/inverse_101/` (replaces deprecated `inverse_planning_eval` which has been removed) |
| `agent-imaging-website` | `website/` |

No git history was preserved during the merge — the latest contents from each source repo were copied directly.

---

## 📄 License

See [LICENSE.md](./LICENSE.md).
