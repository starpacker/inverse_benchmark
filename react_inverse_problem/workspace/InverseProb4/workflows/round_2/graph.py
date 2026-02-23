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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # Step 1: Generate detailed solution plan
        plan = await self.custom(
            input=problem,
            instruction=prompt_custom.PLAN_PROMPT
        )
        
        # Step 2: Generate multiple candidate solutions using plan
        solutions = []
        for _ in range(3):  # Generate 3 candidate solutions
            solution = await self.custom_code_generate(
                problem=f"Problem: {problem}\nPlan: {plan['response']}",
                entry_point="main",
                instruction=prompt_custom.CODE_PROMPT
            )
            if solution['response'].strip():  # Ensure non-empty response
                solutions.append(solution['response'])
        
        # Step 3: Select best solution via self-consistency
        if solutions:
            final_solution = await self.sc_ensemble(
                solutions=solutions,
                problem=problem
            )
            return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
        else:
            # Fallback if all solutions empty
            fallback = await self.custom_code_generate(
                problem=problem,
                entry_point="main",
                instruction="Generate complete Python code solution"
            )
            return fallback['response'], self.llm.get_usage_summary()["total_cost"]
