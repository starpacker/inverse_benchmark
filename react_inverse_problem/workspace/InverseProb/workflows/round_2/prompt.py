CODE_GEN_PROMPT = """
As scientific computing expert, create a complete Python solution with:
1. All required imports (NumPy/SciPy)
2. Efficient numerical implementations
3. Detailed comments for key steps
4. Parameter tuning considerations
5. Real-world data handling
6. Explicit output handling

Structure:
```python
# Imports
import numpy as np
from scipy import ...

# Core solution
def solve(input):
    # Implement optimized solution
    ...
    
# Execution
if __name__ == "__main__":
    result = solve(...)
    # Output results appropriately
```

Output ONLY raw code without markdown formatting.
"""

SCORE_PROMPT = """
Evaluate solution completeness (0.0-1.0 scale):
1. Has all required imports? (0.2)
2. Implements core algorithm correctly? (0.4)
3. Includes parameter handling? (0.2)
4. Has proper output handling? (0.2)

Calculate total score. Respond EXACTLY:
"Score: X.XX"
"""

REVIEW_PROMPT = """
Critique this solution. Identify:
[High Severity]
- Mathematical/logical errors
- Dimension mismatches
- Missing critical imports
- Runtime error risks

[Medium Severity]
- Inefficient computations
- Incomplete edge case handling
- Suboptimal parameter choices

[Low Severity]
- Documentation gaps
- Style violations

Format response:
### Critical Issues (High Severity) ###
- [ ] Line X: Description
### Optimization Opportunities (Medium) ###
- [ ] Section Y: Suggestion
### Minor Improvements (Low) ###
- [ ] Area Z: Observation
"""

REVISE_PROMPT = """
Incorporate ALL feedback while preserving functionality:
1. Fix all critical issues first
2. Optimize performance bottlenecks
3. Add validation checks
4. Improve documentation
5. Maintain original solution structure

Output ONLY revised code with:
- Updated imports if needed
- Optimized numerical operations
- Enhanced error handling
- Clearer comments
"""

VALIDATE_PROMPT = """
Verify solution meets requirements:
1. Contains complete runnable code
2. Handles specified input dimensions
3. Includes proper output handling
4. Has no syntax errors

If ALL pass, respond "PASS". Otherwise, list:
### Missing Requirements ###
- [ ] Requirement not met: Explanation
"""