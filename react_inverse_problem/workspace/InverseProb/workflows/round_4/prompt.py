CODE_GEN_PROMPT = """
You are an expert scientific computing engineer. Given the problem description:
1. Implement a complete, runnable Python solution
2. Include all necessary imports
3. Use efficient numerical computation (NumPy/SciPy)
4. Add detailed comments explaining key steps
5. Output ONLY the raw code without markdown formatting

Structure your solution:
```python
# Required imports
import numpy as np
from scipy import ...

# Main processing function
def solution(input_data):
    # Implementation here
    ...
    
# Execution and output handling
if __name__ == "__main__":
    result = solution(...)
    # Save/output results
```

Ensure the solution:
- Handles real data dimensions and formats
- Includes parameter tuning considerations
- Follows best practices for the domain
"""

REVIEW_PROMPT = """
Critically review the provided code solution. Check for:
1. Mathematical/logical errors
2. Missing edge case handling
3. Performance bottlenecks
4. Incorrect API/library usage
5. Data dimension mismatches
6. Missing output handling

Provide specific feedback in bullet points:
- [ ] Line X: Issue description (severity: high/medium/low)
- [ ] Section Y: Improvement suggestion

Prioritize critical errors affecting correctness. Do NOT rewrite code - only provide diagnostic feedback.
"""

REVISE_PROMPT = """
Incorporate ALL review feedback while maintaining original requirements. Revise the code by:
1. Fixing all identified errors
2. Optimizing performance-critical sections
3. Adding necessary validation checks
4. Preserving original functionality
5. Improving documentation clarity

Output ONLY the final revised code with:
- Updated imports if needed
- Optimized numerical operations
- Added error handling
- Improved comments

Ensure the solution is production-ready and handles all edge cases mentioned in the review.
"""