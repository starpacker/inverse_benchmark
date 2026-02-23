ANALYSIS_PROMPT = """
Please solve the problem step by step. First, analyze the problem and break it into logical steps. 
Then, derive a precise mathematical expression representing the solution. The expression must:
1. Use Python-compatible operators (*, /, +, -)
2. Include all necessary parentheses
3. End with 'Expression: [your_expression]' exactly
DO NOT calculate the final numerical value.

Problem: 
"""