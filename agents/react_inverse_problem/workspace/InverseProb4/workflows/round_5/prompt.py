PLAN_PROMPT = """
You are an expert in computational problem solving. Break down the problem into clear steps including:
1. Required Python libraries and why
2. Data loading/preprocessing steps
3. Core algorithm implementation
4. Output handling and validation
5. Special considerations for performance/accuracy

Provide this as a numbered list without code snippets.
"""

PLAN_REFINE_PROMPT = """
Critically review the provided solution plan and identify potential improvements. Consider:
- Missing edge cases or error handling
- Suboptimal algorithm choices
- Potential performance bottlenecks
- Better library alternatives
- Simplification opportunities

Generate an improved version of the plan addressing these concerns. Maintain the original numbering format.
"""

TEST_CASE_PROMPT = """
Generate diverse test cases for verifying solution correctness. Include:
1. Normal cases with expected outputs
2. Edge cases (min/max values, empty inputs)
3. Error cases (invalid inputs)
4. Performance stress tests

Format as JSON: {"test_cases": [{"input": "...", "expected": "..."}]}
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

VALIDATE_PROMPT = """
Evaluate a solution against test cases and reference solutions. Return JSON with:
- "pass_rate": percentage of passed test cases (0.0-1.0)
- "quality_score": 1-10 rating considering:
  * Correctness of outputs
  * Code efficiency
  * Readability and documentation
  * Error handling
- "issues": list of specific problems found

Test Cases: {test_cases}
Reference Solutions: {reference_solutions}
"""

REFINE_PROMPT = """
Improve this solution based on validation feedback:
1. Fix all identified issues
2. Optimize performance bottlenecks
3. Enhance readability and documentation
4. Maintain original functionality
5. Output ONLY the refined code with no additional text

Validation Feedback:
{validation_feedback}
"""