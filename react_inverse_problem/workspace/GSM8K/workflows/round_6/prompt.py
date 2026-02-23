SOLVE_PROMPT = """
Solve the math problem step by step. Provide clear reasoning and calculations. 
After reaching the final answer, present it in boxed format: \boxed{final_answer}.
"""

REVIEW_PROMPT = """
Review this math solution for errors. Verify:
1. Correct interpretation of the problem
2. Accurate calculations at each step
3. Units consistency (if applicable)
4. Final answer matches problem requirements

If errors exist, provide corrected reasoning and final answer. 
Always present the final numerical answer in boxed format: \boxed{final_answer}.
"""