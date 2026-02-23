PLAN_PROMPT = """
As an expert in computational microscopy and inverse problems, develop a step-by-step plan to solve the SIM deconvolution problem. The plan must include:
1. Data loading and preprocessing steps
2. PSF modeling considerations
3. Implementation of sparsity prior (e.g., L1 regularization)
4. Implementation of continuity prior (e.g., total variation regularization)
5. Iterative deconvolution procedure
6. Result validation and output saving
Provide specific technical details for each step.
"""

CODE_GEN_PROMPT = """
Generate production-quality Python code implementing the provided plan for SIM deconvolution. Requirements:
1. Use Richardson-Lucy as base with these enhancements:
   - Add sparsity constraint via hard thresholding after each iteration
   - Add continuity constraint via Gaussian smoothing after thresholding
2. Handle 5D TIFF inputs (time, channel, z, y, x)
3. Include memory-efficient batch processing
4. Add progress tracking and parameter validation
5. Save results as 32-bit float TIFF with metadata preservation
Include detailed docstrings and type annotations.
"""

ALT_SOLUTION_PROMPT = """
Generate an alternative implementation approach for the same problem. Consider:
1. Different regularization techniques (e.g., Tikhonov vs TV)
2. Computational efficiency optimizations
3. Alternative data handling strategies
4. Enhanced error handling
5. Multi-threading/GPU acceleration possibilities
Provide complete, runnable code with the same functionality but different implementation strategy.
"""

REVIEW_PROMPT = """
Critically review this solution for SIM deconvolution. Evaluate:
1. Correctness of the mathematical formulation
2. Appropriateness of regularization parameters
3. Handling of edge cases (empty images, NaNs, extreme values)
4. Computational complexity and memory usage
5. Compliance with problem requirements (sparsity/continuity priors)
6. Potential artifacts or numerical instability sources
Provide specific, actionable improvement suggestions.
"""

REVISE_PROMPT = """
Revise the solution based on the provided review. Address all criticism by:
1. Fixing identified errors
2. Implementing suggested improvements
3. Optimizing computational performance
4. Enhancing documentation and comments
5. Adding validation checks
Preserve the core algorithm while incorporating all valid feedback. Output complete final code.
"""