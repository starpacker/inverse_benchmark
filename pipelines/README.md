# pipelines/ — Paper-to-Code Generation Pipelines

This folder contains **three different approaches** to the same problem: given a scientific paper, automatically generate working executable code that reproduces its results.

The three pipelines reflect different design philosophies and were developed in parallel as research artifacts. They are kept side-by-side so future work can compare them fairly.

---

## 📊 Comparison

| Pipeline | Architecture | Input | Output | Strength |
|---|---|---|---|---|
| [`paper2executable/`](./paper2executable/) | **4-phase** (baseline → discovery → openhands → sandbox) | ArXiv ID | `working_folder/` with `input.npy`, `gt_output.npy`, `run_gt.py`, `evaluate.py`, `baseline.npy` | Most structured; integrates PaddleOCR + PapersWithCode + OpenHands |
| [`agentic_reproduce/`](./agentic_reproduce/) | **Multi-agent** (planner → architect → coder → judge) | Paper PDF | Reproduced solver code | Most flexible; uses dedicated agents per role |
| [`new_flow/`](./new_flow/) | **Tutorial-driven** flow | Paper + GT code | Tutorial + benchmark questions | Best for generating standardized benchmark entries |

> **Note:** [`paper2executable/`](./paper2executable/) was originally a standalone repo. The other two were already part of `inverse_benchmark`. They are intentionally kept as separate implementations rather than being merged, because they take fundamentally different architectural approaches to the same problem.

---

## 🎯 Which one should I use?

- **Want a fully automated paper→code pipeline?** → [`paper2executable/`](./paper2executable/)
- **Want fine-grained control via specialized agents?** → [`agentic_reproduce/`](./agentic_reproduce/)
- **Want to turn an existing paper+code into a tutorial/benchmark task?** → [`new_flow/`](./new_flow/)

Each subfolder has its own README with installation and usage instructions.

---

## 🔗 Downstream

The output of any pipeline here can be turned into a standardized task entry under [`../tasks/`](../tasks/), then evaluated by the harnesses in [`../harnesses/`](../harnesses/).
