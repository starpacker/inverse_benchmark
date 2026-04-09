from typing import Literal
import workspace.InverseProb.workflows.template.operator as operator
import workspace.InverseProb.workflows.round_5.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)  # Added ensemble operator
    
    async def __call__(self, problem: str):
        # Generate step-by-step plan
        plan_response = await self.custom(
            input=problem,
            instruction=prompt_custom.PLAN_PROMPT
        )
        plan = plan_response['response']
        
        # Generate 3 candidate solutions using plan + problem
        candidate_solutions = []
        for _ in range(3):  # Loop for ensemble diversity
            solution = await self.custom_code_generate(
                problem=f"Problem: {problem}\nPlan: {plan}",
                instruction=prompt_custom.CODE_GENERATION_PROMPT
            )
            candidate_solutions.append(solution['response'])
        
        # Ensemble selection with error handling
        final_solution = await self.sc_ensemble(
            solutions=candidate_solutions,
            problem=problem
        )
        result = final_solution['response']
        
        # Ensure non-None output
        if not result or "None" in result:
            result = candidate_solutions[0]  # Fallback to first candidate
        
        return result, self.llm.get_usage_summary()["total_cost"]
