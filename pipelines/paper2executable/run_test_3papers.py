#!/usr/bin/env python
"""
Test script: Run the E2E pipeline on 3 test papers.

TEST_CASES:
  2108.10257 | SwinIR   | https://github.com/JingyunLiang/SwinIR
  2209.14687 | DPS      | https://github.com/DPS2022/diffusion-posterior-sampling
  2206.00942 | Diffusion Survey (no official code)
"""

import json
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import PipelineOrchestrator
from utils.logging_utils import setup_logging

# Load config
config_path = Path(__file__).parent / "config" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# ensure OCR enabled; requires paddle_env with GPU
config["discovery"]["pdf_ocr"]["enabled"] = True

# Setup logging
setup_logging(level="INFO")

# Init orchestrator
orch = PipelineOrchestrator(config)

# ── Test cases ──
TEST_CASES = [
    {
        "arxiv_id": "2108.10257",
        "note": "SwinIR: Image Restoration Using Swin Transformer",
    },
    {
        "arxiv_id": "2209.14687",
        "note": "DPS: Diffusion Posterior Sampling",
    },
    {
        "arxiv_id": "2206.00942",
        "note": "Diffusion Models Survey (paper-only, no code)",
    },
]


def main():
    all_results = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n{'#'*70}")
        print(f"# TEST {i}/{len(TEST_CASES)}: {tc['arxiv_id']} — {tc['note']}")
        print(f"{'#'*70}\n")

        result = orch.process_paper_e2e(
            arxiv_id=tc["arxiv_id"],
            skip_env_create=True,  # Skip conda env to speed up testing
        )
        all_results.append(result)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  TEST SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        icon = "✅" if r.get("status") == "completed" else "⚠️ " if r.get("status") == "partial" else "❌"
        print(f"  {icon} {r.get('arxiv_id', '?'):15s} | {r.get('task_name', '?'):20s} | {r.get('status', '?')}")
        if r.get("outputs"):
            for k, v in r["outputs"].items():
                exists = "✔" if v and Path(str(v)).exists() else "✘"
                print(f"      {exists} {k}: {v}")
    print(f"{'='*70}")

    # Save all results
    out = Path(config["general"]["workspace_base"]) / "test_3papers_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
