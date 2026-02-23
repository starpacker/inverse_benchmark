from typing import Any, Dict
from agents.base import BaseAgent
import re
import json
import abc

class PlannerAgent(BaseAgent):
    """
    PlannerAgent: 负责将自然语言的任务描述转化为具体的数学模型和算法步骤。
    它是整个 Inverse Problem 求解的大脑。
    """
    def _build_system_prompt(self) -> str:
        return """You are a Principal Scientist in Computational Imaging and Inverse Problems.
                Your Goal: Formulate a rigorous mathematical and algorithmic plan to solve the user's task.

                ### Guidelines:
                1. **Mathematical Modeling**: Explicitly define the Forward Model $y = A(x) + n$. What is $A$? What is the noise $n$?
                2. **Method Selection**: Choose the most robust solver.
                - For linear problems with constraints: Consider ADMM, Primal-Dual, or FISTA.
                - For non-linear/complex priors: Consider Deep Unrolling (Algorithm Unrolling) or End-to-End CNNs (UNet/ResNet).
                - *Constraint*: Prefer stability and standard implementation over experimental papers.
                3. **Dimensionality Awareness**: Mentally check the input and output shapes. If Input is (B, C, H, W), ensure the operations preserve or transform dimensions correctly.

                ### Output Format (Markdown):
                1. **[Problem Formulation]**: The math equation and variable definitions.
                2. **[Proposed Strategy]**: Name of the algorithm/architecture.
                3. **[Step-by-Step Plan]**: 
                - Step 1: Data Preprocessing (Normalization, etc.)
                - Step 2: Forward Operator Implementation (Physics)
                - Step 3: Solver/Network Architecture Details
                - Step 4: Loss Function & Optimizer
                """

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        context needs:
        - 'task_desc': Task description.
        - 'feedback': (Optional) String containing feedback from Critic or Judge.
        """
        prompt = f"### Task Description\n{context['task_desc']}\n"
        
        # Explicitly prompt to use injected skills if present in task_desc
        if "RELEVANT SKILLS" in context['task_desc']:
            prompt += "\n### 🧠 SKILL UTILIZATION\n"
            prompt += "The Task Description above includes 'RELEVANT SKILLS' from past experiences.\n"
            prompt += "1. **Analyze Applicability**: Determine if these skills apply to the CURRENT task. \n"
            prompt += "   - If the task context (e.g., noise type, operator) is different, DO NOT blindly follow the skill.\n"
            prompt += "2. **Explicit Reference**: If applicable, explicitly mention which skill you are using in your plan.\n"
            prompt += "3. **Adaptation**: If the skill suggests a general strategy (e.g., 'Use ADMM'), adapt the specific formulas to the current forward model.\n"
        
        if context.get('feedback'):
            prompt += f"\n\n### ⚠️ CRITICAL FEEDBACK (Previous Plan Failed)\n"
            prompt += f"Review the following feedback and REVISE your plan accordingly:\n"
            prompt += f"\"{context['feedback']}\"\n"
            prompt += "Do not repeat the same mistake. Adjust the math or algorithm logic."
        
        prompt += "\n\nProduce the [Problem Formulation], [Proposed Strategy], and [Step-by-Step Plan] now."
        return prompt

import re
import json
import abc
from typing import Dict, Any

class CriticAgent(BaseAgent):
    """
    CriticAgent: 负责在 Planner 生成计划后立即进行“同行评审”。
    输出严格结构化的 JSON，内置 3 次重试机制确保格式合规。
    """
    def _build_system_prompt(self) -> str:
        return """You are a Senior Technical Reviewer for Inverse Problems in Computational Imaging.
Your sole responsibility: Critically evaluate algorithmic plans BEFORE coding begins.

### STRICT OUTPUT REQUIREMENTS
1. Output ONLY a valid JSON object with EXACTLY these fields:
   {
     "decision": "PASS" | "REJECT",
     "reason": "Concise technical justification (max 100 chars)",
     "suggestion": "Actionable improvement suggestion (if REJECTED, else empty string)"
   }
2. NO MARKDOWN, NO PREFIXES, NO EXPLANATIONS — ONLY RAW JSON.
3. Validate your output with json.loads() before responding.

### Evaluation Checklist (Reject if ANY item fails)
✅ Physics Alignment: Algorithm matches problem type? (e.g., ADMM for inpainting ✅, Wiener filter for missing pixels ❌)
✅ Mathematical Completeness: Forward model H, regularization R, and loss explicitly defined?
✅ Implementation Feasibility: Solvable in <200 lines without external DL frameworks?
✅ Data Flow Clarity: Input → Algorithm → Output pipeline logically sound?

### Examples
PASS:
{"decision": "PASS", "reason": "ADMM plan fully specifies H, R, ρ with clear data flow", "suggestion": ""}

REJECT:
{"decision": "REJECT", "reason": "Missing measurement matrix Φ for compressed sensing", "suggestion": "Define Φ explicitly in Step 2 before ADMM iterations"}
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        return f"""### TASK DESCRIPTION
{context.get('task_desc', 'N/A')}

### PROPOSED PLAN BY PLANNER
{context.get('plan', 'NO PLAN PROVIDED')}

### YOUR MISSION
Review against the STRICT checklist above. Output ONLY valid JSON — nothing else."""

    def generate(self, context: Dict[str, Any]) -> str:
        """
        Enhanced generate with built-in JSON validation and retry logic.
        Returns a VALID JSON string guaranteed to be parseable by json.loads().
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            # 调用父类 generate 获取原始 LLM 响应
            raw_response = super().generate(context)
            
            # 尝试提取 JSON 块（处理 LLM 可能添加的 Markdown 包装）
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            json_candidate = json_match.group(0) if json_match else raw_response.strip()
            
            try:
                # 验证 JSON 格式
                parsed = json.loads(json_candidate)
                
                # 验证必需字段
                if "decision" not in parsed:
                    raise ValueError("Missing required field: 'decision'")
                if parsed["decision"] not in ["PASS", "REJECT"]:
                    raise ValueError(f"Invalid 'decision' value: {parsed['decision']}")
                if "reason" not in parsed:
                    raise ValueError("Missing required field: 'reason'")
                
                # 补全可选字段
                parsed.setdefault("suggestion", "")
                
                # 返回标准化后的 JSON 字符串（确保调用方直接解析）
                normalized_json = json.dumps({
                    "decision": parsed["decision"],
                    "reason": str(parsed["reason"])[:150],  # 截断超长原因
                    "suggestion": str(parsed.get("suggestion", ""))[:200]
                }, ensure_ascii=False)
                
                if attempt > 0:
                    print(f"[Critic] Recovered after {attempt+1} attempts. Valid JSON produced.")
                return normalized_json
                
            except Exception as e:
                last_error = f"Attempt {attempt+1} failed: {str(e)}"
                print(f"[Critic] JSON validation failed: {last_error}")
                print(f"  Raw response snippet: {raw_response[:100]}")
                
                # 注入错误反馈到下一轮提示（仅当还有重试机会时）
                if attempt < max_retries - 1:
                    context["feedback"] = (
                        f"PREVIOUS ATTEMPT FAILED: {last_error}. "
                        f"CRITICAL: Output MUST be valid JSON with fields: decision(PASS/REJECT), reason, suggestion. "
                        f"NO MARKDOWN, NO PREFIXES."
                    )
        
        # 所有重试失败 → 返回安全的 REJECT 决策（避免工作流卡死）
        fallback_response = {
            "decision": "REJECT",
            "reason": f"LLM output parsing failed after {max_retries} attempts",
            "suggestion": "Planner must regenerate plan with explicit mathematical definitions"
        }
        print(f"[Critic] ⚠️ All retries exhausted. Returning fallback REJECT decision.")
        return json.dumps(fallback_response, ensure_ascii=False)