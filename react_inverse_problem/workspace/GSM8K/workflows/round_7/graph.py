from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_7.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)  # Added Programmer operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate step-by-step reasoning without final answer
        reasoning = (await self.custom(input=problem, instruction=prompt_custom.REASONING_PROMPT))['response']
        # Generate and execute code based on reasoning
        program_result = await self.programmer(problem=problem, analysis=reasoning)
        return program_result['output'], self.llm.get_usage_summary()["total_cost"]
