CODE_GEN_SYSTEM_PROMPT = """You are an expert Scientific Python Programmer.
Translate the provided "Algorithmic Plan" into a complete, executable Python script.
"""

CODE_GEN_USER_TEMPLATE = """
### Task Description
{task_description}

### Algorithmic Plan (The Blueprint)
{plan_json}

### Instructions
Write a Python script to solve this task implementing the EXACT steps from the plan.

**Requirement: Thinking Process**
Before writing the code, you MUST output your thinking process wrapped in `<thinking>` tags.
1. Analyze the Plan step-by-step.
2. Identify potential implementation challenges (e.g., array shapes, library availability).
3. Confirm how you will handle the Input/Output paths.

**Execution Environment:**
The code will be executed in a Python environment with the following libraries available:
[{installed_libs}]

**Strict Constraints:**
1. **Do NOT** attempt to install new packages (no `!pip install`).
2. Only import from the list above or Python's standard libraries.
3. If a library is missing, implement the logic using `numpy` or `scipy`.
4. **IO Requirements**:
   - Output File: `{output_file}`

**Response Format:**
<thinking>
... Step-by-step reasoning ...
</thinking>

```python
... Complete Python Code ...
```
"""

ERROR_REFINER_TEMPLATE = """
The code generated encountered an error during execution.

### Original Plan
{plan_json}

### Previous Code
```python
{previous_code}
```

### Execution Error
{error_logs}

### Instruction
Fix the code to resolve the error. Ensure the logic still follows the Plan.

**Requirement: Thinking Process**
Before fixing the code, analyze the error in `<thinking>` tags:
1. Identify the root cause (Syntax, Shape Mismatch, Import Error, etc.).
2. Propose a specific fix.

**Response Format:**
<thinking>
... Analysis of error and fix ...
</thinking>

```python
... Full Corrected Code ...
```
"""
