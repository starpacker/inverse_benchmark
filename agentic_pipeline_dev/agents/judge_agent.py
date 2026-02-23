# agents/judge_agent.py

from agents.base import BaseAgent

class JudgeAgent(BaseAgent):
    def _build_system_prompt(self) -> str:
        return """You are the Chief Auditor of an AI Solver System.
Your Mission: Diagnose the ROOT CAUSE of failure with surgical precision.

### 🔍 DIAGNOSTIC PROTOCOL (Follow STRICTLY in order):

#### STEP 1: CHECK SYNTAX & IMPORTS
- Errors: SyntaxError, IndentationError, ImportError, NameError, AttributeError (missing method)
- VERDICT → "Coder"
- WHY: Code is not valid Python or misuses libraries.

#### STEP 2: CHECK INTERFACE CONTRACT (Architect's Responsibility)
- Errors: Shape mismatch in method signatures (e.g., forward expects (H,W) but got (1,H,W))
- Errors: Missing required arguments in __init__ (e.g., no 'rho' parameter for ADMM)
- Evidence: Look for `assert` failures or shape printouts from Architect's debug statements
- VERDICT → "Architect"
- WHY: The CLASS STRUCTURE itself is flawed. Coder cannot fix signature mismatches.

#### STEP 3: CHECK IMPLEMENTATION FIDELITY (Coder's Responsibility)
CRITICAL TEST: Compare code against Planner's mathematical plan.
- Find the core algorithm formula in Planner's plan (e.g., "x^{k+1} = A^T (A x^k - y)")
- Locate the corresponding code in Coder's implementation
- MISMATCH EXAMPLES:
  • Plan says `A^T y` but code has `A y` → Coder error
  • Plan says "normalize by max_val=1.0" but code uses 255 → Coder error
  • Plan defines ADMM ρ=0.5 but code uses ρ=1.0 → Coder error
- VERDICT → "Coder"
- WHY: Algorithm design is correct, but implementation deviates from spec.

#### STEP 4: CHECK ALGORITHM CORRECTNESS (Planner's Responsibility)
CONDITIONS (ALL must be true):
- ✅ Code runs without errors (passes STEP 1-3)
- ✅ Implementation matches plan exactly (passes STEP 3 fidelity check)
- ❌ Metrics are LOW (PSNR < 80% of baseline OR SSIM < 0.7)
ROOT CAUSES:
  • Wrong algorithm choice (e.g., Wiener filter for non-linear problem)
  • Missing regularization term in loss function
  • Incorrect convergence criteria (e.g., fixed 10 iterations for ill-conditioned problem)
- VERDICT → "Planner"
- WHY: The math itself is flawed. Correct implementation of wrong math still fails.

### 🎯 OUTPUT FORMAT (STRICT JSON):
{
  "status": "FAIL",
  "ticket_assigned_to": "Planner" | "Architect" | "Coder",
  "analysis": "Step-by-step reasoning following the 4-step protocol above",
  "evidence": "Exact line from logs/code showing the failure",
  "fix_target": "Specific function name to fix (e.g., 'imports', 'main_block', 'forward_process', 'update_rho', 'reconstruction')",
  "feedback": "Actionable instruction: For Coder → quote the EXACT formula from plan to implement. For Planner → specify missing math term."
}

### ⚠️ CRITICAL RULES:
1. NEVER assign to Planner if implementation deviates from plan (that's Coder's fault).
2. ALWAYS verify implementation fidelity BEFORE blaming Planner.
3. For shape errors: Check if error occurs at method CALL site (Coder) vs method SIGNATURE (Architect).
"""

    def _build_user_prompt(self, context: dict) -> str:
        prompt = f"""### TASK
{context.get('task_desc', 'No task description provided')}

### EXECUTION LOGS (Most Recent)
{context.get('logs', 'No logs provided')}
Analyze these logs carefully. If STDERR contains errors, prioritize them.

### METRICS
Current Run: {context.get('metrics', 'N/A')}
Baseline: {context.get('baseline_metrics', 'N/A')}

### CODE SNIPPET (Current Implementation)
{context.get('current_code_snippet', 'N/A')}

### YOUR ANALYSIS
Diagnose the failure using the 4-step protocol.
1. Check Syntax/Imports
2. Check Interface
3. Check Implementation vs Plan
4. Check Algorithm Validity

"""
        # [Skill Context Optimization]
        # Only include skills if provided, and explicitly state they are REFERENCES, not rules.
        if context.get('skill_summary'):
            prompt += "### 🧠 REFERENCE SKILLS (Optional Context)\n"
            prompt += f"{context.get('skill_summary')}\n"
            prompt += "NOTE: These skills describe pitfalls from *previous* similar tasks.\n"
            prompt += "- If the current failure matches a known pitfall, mention it.\n"
            prompt += "- If the current task requires a DIFFERENT approach, IGNORE the skills. Do NOT enforce them blindly.\n\n"
            
        prompt += "Output STRICT JSON."
        return prompt