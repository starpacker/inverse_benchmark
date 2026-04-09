# agents/ — Agent Implementations

This folder collects the **agent implementations and per-paradigm benchmark wrappers** used by (or together with) the evaluation harnesses in [`../harnesses/`](../harnesses/).

---

## 📁 Contents

| Subfolder | What it is |
|---|---|
| [`agentic_pipeline_dev/`](./agentic_pipeline_dev/) | Multi-agent system (Planner → Architect → Coder → Judge) for automatic solver code generation. Has a persistent skill / knowledge system (`persistent_skill_system/`) that accumulates reusable problem-solving experience across tasks. |
| [`inverse_agent_whole/`](./inverse_agent_whole/) | End-to-end agent pipeline with **typed feedback loops**: distinguishes between "plan error / code bug / hyperparameter tuning" failures and dispatches the right repair action. |
| [`react_inverse_problem/`](./react_inverse_problem/) | Multi-model benchmark evaluating 7+ LLMs (GPT-5, Claude, Gemini, DeepSeek, Qwen, etc.) on **function-level** scientific code generation via ReAct loops. |
| [`openhands_benchmark/`](./openhands_benchmark/) | OpenHands wrapper for the inverse problem benchmark — adapts the OpenHands agent runtime to consume tasks from [`../tasks/`](../tasks/). |

---

## 🤝 How agents and harnesses interact

```
   tasks/Task_NN_*/   ← input / data / GT
        │
        ▼
   agents/<impl>/    ← attempts to solve
        │
        ▼
   harnesses/inverse_101/   ← grades the attempt (plan / function / end2end)
        │
        ▼
   results → website/
```

Each agent here can be plugged into [`../harnesses/inverse_101/`](../harnesses/inverse_101/) as a framework, or run standalone with its own entry point.

See each subfolder's README for installation and usage details.
