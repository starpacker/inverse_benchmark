# harnesses/ — LLM Agent Evaluation Harnesses

This folder contains the **evaluation framework** used to grade LLM agents and traditional solvers on the inverse problem benchmark.

The current standard harness is [`inverse_101/`](./inverse_101/). It supersedes the older `inverse_planning_eval` (which used TextGrad + ELO tournament) — that one has been removed from the repo as of 2026-04-10.

---

## 🧪 inverse_101 — current standard

> Originally a standalone repo named `inverse-101`, merged here on 2026-04-10.

A benchmark harness for evaluating LLM agents on computational imaging tasks. It supports **3 evaluation modes × 3 agent frameworks**.

### Evaluation modes

| Mode | What it tests | Scoring |
|---|---|---|
| `plan` | Generate `approach.md` + `design.md` for a task | LLM-as-judge (pairwise + rubric) |
| `function` | Implement a specific function | pytest pass rate |
| `end2end` | Full pipeline from data to reconstruction | Quality metrics (NRMSE, NCC, PSNR, SSIM) |

### Frameworks

| Framework | Description |
|---|---|
| `claude_code` | Wraps the Claude Code CLI as the agent |
| `multi_agent` | Architect + coder + judge multi-agent |
| `react` | ReAct loop with tool use |

### Quick start

```bash
cd inverse_101
pip install -r requirements.txt

# plan mode
python -m evaluation_harness plan --task eht_black_hole_original --model "gemini-3.1-pro-preview"

# function mode
python -m evaluation_harness function --task SSNP_ODT --model "cds/Claude-4.6-opus"
```

See [`inverse_101/README.md`](./inverse_101/README.md) for full configuration and details.

---

## 🗺️ Where the tasks come from

Tasks evaluated by these harnesses live under [`../tasks/`](../tasks/) — each `Task_NN_*/` directory has the data and reference implementation needed to grade an agent's output.

Agents being evaluated are typically in [`../agents/`](../agents/) (or external — the harness is agent-agnostic).
