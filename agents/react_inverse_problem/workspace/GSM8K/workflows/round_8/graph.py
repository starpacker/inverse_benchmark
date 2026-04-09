from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_8.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate two independent solutions
        sol1 = await self.custom(input=problem, instruction=prompt_custom.MATH_PROMPT)
        sol2 = await self.custom(input=problem, instruction=prompt_custom.MATH_PROMPT)
        
        # Ensemble solutions
        final = await self.sc_ensemble(
            solutions=[sol1['response'], sol2['response']],
            problem=problem
        )
        return final['response'], self.llm.get_usage_summary()["total_cost"]
