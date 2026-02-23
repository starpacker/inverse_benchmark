PLAN_PROMPT = """
As an expert in computational microscopy, develop a step-by-step plan to solve the SIM deconvolution problem. Include these critical phases:
1. Data loading and preprocessing: Explain normalization and handling of 5D tensor (time, height, width)
2. PSF modeling: Specify parameters for Gaussian-based point spread function
3. Deconvolution algorithm: Detail how to incorporate both:
   - Sparsity prior (L1 regularization)
   - Continuity prior (total variation regularization)
4. Implementation mechanics: Describe iterative optimization approach
5. Output handling: Specify file format and data preservation

Structure your response as:
1. [Step 1 description]
2. [Step 2 description]
...
"""

CODE_GEN_PROMPT = """
Generate production-ready Python code implementing the provided plan exactly. Include:
1. Required imports (numpy, tifffile, scipy, skimage)
2. Modular functions for each processing stage
3. Parameterized hyperparameters (lambda_sparse, lambda_cont)
4. Memory-efficient time-series processing
5. Validation checks for data integrity

Code requirements:
- MUST preserve input data dimensions (5, 1152, 1152)
- MUST output 32-bit float TIFF
- MUST include progress logging
- MUST handle edge cases (NaN, inf, negative values)

Output ONLY the Python code with no additional text.
"""

VALIDATION_PROMPT = """
Critically analyze the solution against the original plan and problem requirements. Check for:
1. Faithfulness to sparsity/continuity priors implementation
2. Correct handling of 5D time-series data
3. Appropriate numerical stability measures
4. File output specification compliance
5. Computational efficiency considerations

Identify any discrepancies or potential improvements. Structure your response:
1. Compliance assessment: [Match/Mismatch] for each requirement
2. Critical issues: List any fundamental flaws
3. Optimization suggestions: List potential improvements
4. Final verdict: [VALID/INVALID] with justification
"""