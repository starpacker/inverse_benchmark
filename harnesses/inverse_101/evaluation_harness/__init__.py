"""Evaluation harness for computational imaging benchmarks.

Usage:
    python -m evaluation_harness plan     --task TASK --model MODEL
    python -m evaluation_harness function --task TASK --model MODEL --target MOD.FUNC
    python -m evaluation_harness end2end  --task TASK --model MODEL [--framework react|pipeline|claude-code]
    python -m evaluation_harness prepare  --task TASK [--level L1|L2|L3]
    python -m evaluation_harness collect  --task TASK --workspace-dir PATH
    python -m evaluation_harness summarize --dir PATH
"""
