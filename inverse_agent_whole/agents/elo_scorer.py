from utils.llm import llm_client
from prompts import phase3 as prompts
from config import settings
from utils.logger import RunLogger
import json

class EloJudge:
    def __init__(self, model_name: str = settings.DEFAULT_JUDGE_MODEL):
        self.model_name = model_name

    def evaluate(self, task_description: str, golden_plan: dict, user_plan: dict, logger: RunLogger = None) -> dict:
        """
        Compares User Plan vs Golden Plan.
        Returns detailed scoring.
        """
        res = llm_client.call(
            prompts.ELO_JUDGE_SYSTEM_PROMPT,
            prompts.ELO_JUDGE_USER_TEMPLATE.format(
                task_description=task_description,
                reference_plan=json.dumps(golden_plan, indent=2),
                candidate_plan=json.dumps(user_plan, indent=2)
            ),
            model=self.model_name,
            json_mode=True
        )
        if logger:
            logger.log_trace("elo", "judge_thinking", res.get("thinking", ""))
            logger.log_trace("elo", "judge_response", json.dumps(res.get("content", {}), indent=2))
            
        return res.get("content", {})
