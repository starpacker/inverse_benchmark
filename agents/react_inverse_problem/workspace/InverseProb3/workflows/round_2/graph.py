from typing import Literal
import workspace.InverseProb.workflows.template.operator as operator
import workspace.InverseProb.workflows.round_2.prompt as prompt_custom
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
        solution = await self.custom_code_generate(problem=problem, instruction="")
        # Review and revise solution with specialized prompt
        revised = await self.custom(
            input=solution['response'],
            instruction=prompt_custom.REVISE_PROMPT + "\nProblem Context: " + problem
        )
        final_code = revised['response'] if revised['response'].strip() else solution['response']
        return final_code, self.llm.get_usage_summary()["total_cost"]
