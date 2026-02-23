from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_2.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Added ensemble operator

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate three candidate solutions
        solutions = []
        for _ in range(3):
            solution = await self.custom(input=problem, instruction=prompt_custom.BASE_PROMPT)
            solutions.append(solution['response'])
        
        # Apply self-consistency ensemble
        ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
        return ensemble_result['response'], self.llm.get_usage_summary()["total_cost"]
