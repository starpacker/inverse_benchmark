from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_6.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.solver = operator.Custom(self.llm)
        self.reviewer = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        # Generate initial solution
        initial_solution = await self.solver(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        # Review and refine solution
        review_input = f"Problem: {problem}\nProposed Solution: {initial_solution['response']}"
        final_solution = await self.reviewer(input=review_input, instruction=prompt_custom.REVIEW_PROMPT)
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
