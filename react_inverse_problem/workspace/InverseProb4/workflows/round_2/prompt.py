PLAN_PROMPT = """
You are an expert in computational problem solving. Break down the problem into clear steps including:
1. Required Python libraries and why
2. Data loading/preprocessing steps
3. Core algorithm implementation
4. Output handling and validation
5. Special considerations for performance/accuracy

Provide this as a numbered list without code snippets.
"""

CODE_PROMPT = """
Generate complete executable Python code that strictly follows these rules:
1. Import all required libraries at the top
2. Implement all steps from the provided Plan
3. Include detailed comments for each logical block
4. Ensure output is a SINGLE code block with NO additional text
5. Handle edge cases and validate input dimensions
6. Include save/export functionality for results
"""