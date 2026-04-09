PLAN_PROMPT = """
You are an expert computational problem solver. Create a comprehensive implementation plan with these sections:
1. Required Python libraries with import rationale
2. Data processing pipeline steps
3. Core algorithm implementation logic
4. Output validation and formatting requirements
5. Edge case handling strategy
Present as a numbered list without code snippets.
"""

REFINE_PLAN_PROMPT = """
Critique and enhance the provided solution plan. Identify:
1. Missing components or logical gaps
2. Potential efficiency improvements
3. Better library alternatives
4. Additional edge cases to consider
5. Optimization opportunities
Rewrite the improved plan maintaining the original structure. Preserve the numbered list format.
"""

DIRECT_CODE_PROMPT = """
Generate complete executable Python code that:
1. Imports all necessary libraries at the top
2. Implements the Refined Plan exactly
3. Includes detailed inline comments for each logical block
4. Outputs a SINGLE code block with NO additional text
5. Explicitly handles edge cases mentioned in the plan
6. Includes result validation before final output
"""

MODULAR_CODE_PROMPT = """
Generate modular Python code with these requirements:
1. Separate functionality into reusable functions
2. Implement core logic as independent methods
3. Create a main() function coordinating execution flow
4. Include comprehensive docstrings for each function
5. Add input validation guards for all functions
6. Ensure final output is a SINGLE executable code block
"""

SCORE_SOLUTION_PROMPT = """
Evaluate this solution on a 10-point scale considering:
1. Correctness (3 points): Solves all problem requirements
2. Efficiency (3 points): Optimal time/space complexity
3. Robustness (2 points): Handles edge cases properly
4. Readability (2 points): Clear structure and comments
Output ONLY: "Score: X" where X is the total score (0-10).
"""

COMPARE_SOLUTIONS_PROMPT = """
Compare Solution A and Solution B considering:
1. Algorithmic efficiency differences
2. Edge case handling completeness
3. Code maintainability and structure
4. Potential error points
Output ONLY: "Better solution: A" or "Better solution: B".
"""

REPAIR_SOLUTION_PROMPT = """
Identify and fix issues in this solution:
1. Correct syntax errors
2. Fix logical mistakes
3. Add missing edge case handling
4. Optimize inefficient operations
5. Improve output formatting
Output the corrected COMPLETE code in a single block. If no issues found, output the original code unchanged.
"""