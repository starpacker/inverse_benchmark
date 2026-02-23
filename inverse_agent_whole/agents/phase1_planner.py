from utils.llm import llm_client
from prompts import phase1 as prompts
from config import settings
from utils.logger import RunLogger
import json

class PlanGenerator:
    def __init__(self, model_name: str = settings.DEFAULT_MODEL):
        self.model_name = model_name

    def generate_plan(self, task_description: str, logger: RunLogger = None) -> dict:
        """Generates an initial plan."""
        res = llm_client.call(
            prompts.PLAN_GEN_SYSTEM_PROMPT,
            prompts.PLAN_GEN_USER_TEMPLATE.format(task_description=task_description),
            model=self.model_name,
            json_mode=True
        )
        if logger:
            logger.log_trace("phase1", "plan_gen_thinking", res.get("thinking", ""))
            logger.log_trace("phase1", "plan_gen_response", json.dumps(res.get("content", {}), indent=2))
        return res.get("content", {})

class PlanCritic:
    def __init__(self, model_name: str = settings.DEFAULT_JUDGE_MODEL):
        self.model_name = model_name

    def review_plan(self, task_description: str, candidate_plan: dict, logger: RunLogger = None) -> dict:
        """
        Reviews the plan for logical consistency.
        Returns {is_valid: bool, defects: list, suggestion: str}
        """
        plan_str = json.dumps(candidate_plan, indent=2)
        res = llm_client.call(
            prompts.SELF_CRITIC_SYSTEM_PROMPT,
            prompts.SELF_CRITIC_USER_TEMPLATE.format(
                task_description=task_description,
                candidate_plan=plan_str
            ),
            model=self.model_name,
            json_mode=True
        )
        if logger:
            logger.log_trace("phase1", "critic_thinking", res.get("thinking", ""))
            logger.log_trace("phase1", "critic_response", json.dumps(res.get("content", {}), indent=2))
        return res.get("content", {})
