from typing import Literal
import workspace.InverseProb.workflows.template.operator as operator
import workspace.InverseProb.workflows.round_3.prompt as prompt_custom
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
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str):
        # Generate initial solution
        solution1 = await self.custom_code_generate(problem=problem, entry_point="main", instruction="")
        
        # Review solution and get improvement instructions
        review = await self.custom(input=solution1['response'], instruction=prompt_custom.CODE_REVIEW_PROMPT)
        
        # Generate final solution using review feedback
        solution_final = await self.custom_code_generate(
            problem=problem, 
            entry_point="main", 
            instruction=review['response']
        )
        return solution_final['response'], self.llm.get_usage_summary()["total_cost"]
