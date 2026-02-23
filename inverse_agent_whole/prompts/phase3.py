ELO_JUDGE_SYSTEM_PROMPT = """You are an objective evaluator for Computational Imaging algorithms.
Compare a "Candidate Plan" against a "Reference Plan" (Ground Truth) and assess its quality.

Scoring Rubric:
1. **Completeness (0.3)**: Are all physical operators present?
2. **Correctness (0.4)**: Is the math correct?
3. **Logical Dependency (0.3)**: Is the DAG valid?

Output JSON:
{
  "step_by_step_analysis": "string",
  "completeness_score": 1-5,
  "correctness_score": 1-5,
  "logical_dependency_score": 1-5,
  "final_weighted_score": float
}
"""

ELO_JUDGE_USER_TEMPLATE = """
[Task Description]:
{task_description}

[Reference Plan]:
{reference_plan}

[Candidate Plan]:
{candidate_plan}
"""

FEEDBACK_GENERATOR_SYSTEM_PROMPT = """
Analyze the Execution Failure and the Oracle Feedback to categorize the error type.

Types:
- **Type A**: Physical/Mathematical Principle Error (The Plan is wrong).
- **Type B**: Implementation Error (The Plan is right, Code is buggy).
- **Type C**: Hyperparameter/Convergence Issue (Needs tuning).

Output JSON:
{
  "error_type": "Type A" | "Type B" | "Type C",
  "diagnosis": "string",
  "feedback_message": "string (instruction for the agent)"
}
"""
