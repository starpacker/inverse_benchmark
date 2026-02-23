REVISE_PROMPT = """
Review this code for SIM deconvolution with sparsity and continuity priors. Ensure:
1. Priors are integrated DURING iterative deconvolution, not as post-processing
2. Uses mathematical formulation of both priors in the update step
3. Maintains original input/output structure
4. Adds parameter explanations in comments

If correct, output original code. If flawed, output revised code ONLY without explanations.
"""