# tasks/ — 200 Standardized Inverse Problem Tasks

This is the **benchmark itself**: 200 inverse problem tasks curated from scientific computing papers, each standardized into a uniform structure so LLM agents and traditional solvers can be evaluated fairly side-by-side.

> **Note:** This folder was previously a standalone repo named `inverse_benchmark_details`. It was merged into the main `inverse_benchmark` umbrella on 2026-04-10.

---

## 📁 Each task contains

Each `Task_NN_<name>/` directory ships with:

| Item | Description |
|---|---|
| `src/` | Complete reference implementation |
| `test/` | Automated test scripts (`test_*.py`) and per-agent variants (in `test/agents/`) |
| `data/` | Input data, ground-truth outputs, and reconstruction outputs |
| `notebook/` *(or `notebook.ipynb`)* | Interactive tutorial walking through the problem |
| `requirements.txt` | Python dependencies |
| `metadata.json` | Task metadata: domain, metrics, references |
| `README.md` | Task description, setup, and usage |

---

## 🗂️ 200 tasks across 8 categories

| Category | Tasks | Examples |
|---|---|---|
| Computational Imaging | 50+ | Ptychography, FPM, Lensless, Holography, Light Field |
| Medical Imaging | 30+ | CT, MRI, PET, Ultrasound, OCT |
| Spectroscopy | 20+ | Raman, NMR, X-ray, EIS |
| Astrophysics | 15+ | Gravitational Lensing, Radio Imaging, Stellar Spectroscopy |
| Geophysics | 15+ | Seismic, GPR, ERT, InSAR |
| Signal Processing | 20+ | Source Separation, DOA, Spike Sorting |
| Microscopy | 15+ | Super-resolution, Deconvolution, Phase Retrieval |
| Other | 30+ | DIC, Modal Analysis, Rheology, Diffusion |

---

## 📐 Methodology

See [`METHODOLOGY.md`](./METHODOLOGY.md) for the full curation process: how papers were selected, how reference code was extracted, how data was generated, and how the standardized format was designed.

The aggregate processing log is in [`processing_results.json`](./processing_results.json).

---

## 🚀 Using a task

```bash
cd Task_01_sim
pip install -r requirements.txt
python src/main.py
```

For interactive exploration, open the notebook in `notebook/`.

---

## 🔗 How this fits into the larger repo

- **Pipelines** in [`../pipelines/`](../pipelines/) generate task entries from raw papers.
- **Agents** in [`../agents/`](../agents/) and **harnesses** in [`../harnesses/`](../harnesses/) consume tasks here as the evaluation suite.
- **Website** in [`../website/`](../website/) renders results from solving these tasks.
