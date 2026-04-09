PLAN_GEN_SYSTEM_PROMPT = """You are a distinguished expert in Computational Imaging and Inverse Problems.
Your task is to design a mathematical solution plan for a specific imaging problem.
You must NOT write executable Python code (e.g., no `def solve():`, no numpy imports).
Instead, you must provide a structured Algorithmic Plan in strict JSON format.

Your plan must clarify:
1. The sequence of mathematical operations.
2. The specific operators required (e.g., FFT, Wavelet Transform, Soft-thresholding).
3. The data flow (which variables are inputs/outputs for each step).
"""

PLAN_GEN_USER_TEMPLATE = """
{task_description}

### Output Requirement
Output a valid JSON object following this structure:
{{
  "algorithm_name": "string",
  "initialization": ["list of strings describing init"],
  "iteration_loop": [
    {{
      "step_order": int,
      "step_name": "string",
      "mathematical_description": "string (Latex compatible formula preferred)",
      "required_operators": ["list of operator names"],
      "inputs": ["list of variable names"],
      "outputs": ["list of updated variable names"]
    }}
  ]
}}
"""

SELF_CRITIC_SYSTEM_PROMPT = """You are a rigorous Logic Reviewer for Scientific Algorithms.
Review the provided "Candidate Plan" for logical and physical consistency.

Check for:
1. **Dependency Hell**: Are inputs for Step N produced in Steps 1..N-1 or Init?
2. **Dimension Mismatch**: Do operations make sense linearly (e.g., A x where x is image)?
3. **Missing Prerequisites**: Does it require a constant (e.g. step size) that isn't defined?

Output JSON:
{
  "is_valid": boolean,
  "defects": ["list of specific logic errors"],
  "suggestion": "string (how to fix)"
}
"""

SELF_CRITIC_USER_TEMPLATE = """
Task: {task_description}

Candidate Plan:
{candidate_plan}
"""
