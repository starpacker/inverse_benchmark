# End-to-End Computational Imaging Agent Pipeline

This document explains the architecture of the `inverse_agent_whole` pipeline and collects all the prompts used in its various stages.

## Pipeline Overview

The `inverse_agent_whole` is an autonomous agent designed to solve computational imaging tasks (inverse problems) end-to-end. The pipeline is orchestrated by the `InverseAgentPipeline` class and operates in an iterative loop consisting of three main stages: **Planning**, **Coding**, and **Reflection**.

### Core Stages

1.  **Phase 1: Planning**
    *   **Goal**: Generate a mathematically sound algorithmic plan to solve the given task.
    *   **Agent**: `PlanGenerator`
    *   **Process**:
        *   The agent receives a task description and a list of available files in the sandbox.
        *   It produces a structured JSON plan detailing the algorithm (e.g., ADMM, Gradient Descent), initialization steps, and the iteration loop with specific mathematical operators.
        *   **Self-Correction**: A `PlanCritic` agent reviews the generated plan for logical consistency (e.g., dependency errors, dimension mismatches). If defects are found, the plan is refined before moving to the coding stage.

2.  **Phase 2: Coding**
    *   **Goal**: Translate the validated algorithmic plan into executable Python code.
    *   **Agent**: `CodeGenerator`
    *   **Process**:
        *   The agent takes the JSON plan and generates a complete Python script.
        *   It is restricted to using standard libraries and pre-installed scientific packages (numpy, scipy, etc.).
        *   **Execution & Refinement**: The code is executed in a controlled sandbox. If execution fails (runtime errors), an `ErrorRefiner` analyzes the error logs and the original plan to produce a fixed version of the code.

3.  **Phase 3: Reflection & Evaluation**
    *   **Goal**: Verify the solution's correctness and determine the next steps.
    *   **Agent**: `ResultAnalyzer` & `FeedbackGenerator`
    *   **Process**:
        *   The agent's output is verified against a Ground Truth (GT) using an Oracle script.
        *   **Feedback Loop**: If verification fails, the `FeedbackGenerator` analyzes the failure and categorizes it into three types:
            *   **Type A (Plan Error)**: The mathematical approach is fundamentally wrong. The pipeline reverts to **Phase 1 (Planning)**.
            *   **Type B (Implementation Error)**: The plan is correct, but the code implementation is buggy. The pipeline reverts to **Phase 2 (Coding)**.
            *   **Type C (Tuning Error)**: Hyperparameters need adjustment. The pipeline reverts to **Phase 2 (Coding)**.

---

## Collected Prompts

Below are the system prompts and user templates used by the agents in each phase.

### Phase 1: Planning Prompts

**`PLAN_GEN_SYSTEM_PROMPT`**
> You are a distinguished expert in Computational Imaging and Inverse Problems.
> Your task is to design a mathematical solution plan for a specific imaging problem.
> You must NOT write executable Python code (e.g., no `def solve():`, no numpy imports).
> Instead, you must provide a structured Algorithmic Plan in strict JSON format.
>
> Your plan must clarify:
> 1. The sequence of mathematical operations.
> 2. The specific operators required (e.g., FFT, Wavelet Transform, Soft-thresholding).
> 3. The data flow (which variables are inputs/outputs for each step).

**`PLAN_GEN_USER_TEMPLATE`**
```text
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
```

**`SELF_CRITIC_SYSTEM_PROMPT`**
> You are a rigorous Logic Reviewer for Scientific Algorithms.
> Review the provided "Candidate Plan" for logical and physical consistency.
>
> Check for:
> 1. **Dependency Hell**: Are inputs for Step N produced in Steps 1..N-1 or Init?
> 2. **Dimension Mismatch**: Do operations make sense linearly (e.g., A x where x is image)?
> 3. **Missing Prerequisites**: Does it require a constant (e.g. step size) that isn't defined?
>
> Output JSON:
> {
>   "is_valid": boolean,
>   "defects": ["list of specific logic errors"],
>   "suggestion": "string (how to fix)"
> }

**`SELF_CRITIC_USER_TEMPLATE`**
```text
Task: {task_description}

Candidate Plan:
{candidate_plan}
```

### Phase 2: Coding Prompts

**`CODE_GEN_SYSTEM_PROMPT`**
> You are an expert Scientific Python Programmer.
> Translate the provided "Algorithmic Plan" into a complete, executable Python script.

**`CODE_GEN_USER_TEMPLATE`**
```text
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
```

**`ERROR_REFINER_TEMPLATE`**
```text
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
```

### Phase 3: Reflection & Evaluation Prompts

**`FEEDBACK_GENERATOR_SYSTEM_PROMPT`**
> Analyze the Execution Failure and the Oracle Feedback to categorize the error type.
>
> Types:
> - **Type A**: Physical/Mathematical Principle Error (The Plan is wrong).
> - **Type B**: Implementation Error (The Plan is right, Code is buggy).
> - **Type C**: Hyperparameter/Convergence Issue (Needs tuning).
>
> Output JSON:
> {
>   "error_type": "Type A" | "Type B" | "Type C",
>   "diagnosis": "string",
>   "feedback_message": "string (instruction for the agent)"
> }

**`ELO_JUDGE_SYSTEM_PROMPT`**
> You are an objective evaluator for Computational Imaging algorithms.
> Compare a "Candidate Plan" against a "Reference Plan" (Ground Truth) and assess its quality.
>
> Scoring Rubric:
> 1. **Completeness (0.3)**: Are all physical operators present?
> 2. **Correctness (0.4)**: Is the math correct?
> 3. **Logical Dependency (0.3)**: Is the DAG valid?
>
> Output JSON:
> {
>   "step_by_step_analysis": "string",
>   "completeness_score": 1-5,
>   "correctness_score": 1-5,
>   "logical_dependency_score": 1-5,
>   "final_weighted_score": float
> }

**`ELO_JUDGE_USER_TEMPLATE`**
```text
[Task Description]:
{task_description}

[Reference Plan]:
{reference_plan}

[Candidate Plan]:
{candidate_plan}
```
