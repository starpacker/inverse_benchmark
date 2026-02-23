from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_9.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)  # Added Programmer operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate two independent solutions with step-by-step reasoning prompt
        sol1 = await self.custom(input=problem, instruction=prompt_custom.STEP_BY_STEP_REASONING_PROMPT)
        sol2 = await self.custom(input=problem, instruction=prompt_custom.STEP_BY_STEP_REASONING_PROMPT)
        
        # Generate code-based solution using first solution's reasoning as analysis
        sol3 = await self.programmer(problem=problem, analysis=sol1['response'])
        
        # Apply self-consistency ensemble
        final_solution = await self.sc_ensemble(
            solutions=[sol1['response'], sol2['response'], sol3['output']],
            problem=problem
        )
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
