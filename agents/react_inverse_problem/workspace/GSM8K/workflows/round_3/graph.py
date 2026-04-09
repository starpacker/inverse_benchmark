from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_3.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        # Generate reasoning with enhanced prompt
        reasoning = await self.custom(input=problem, instruction=prompt_custom.REASONING_PROMPT)
        analysis = reasoning['response']
        
        # Verify through code execution
        code_solution = await self.programmer(problem=problem, analysis=analysis)
        output = code_solution['output']
        
        # Return verified output or fallback to reasoning
        final_answer = output if output and output.strip() else analysis
        return final_answer, self.llm.get_usage_summary()["total_cost"]
