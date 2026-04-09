from typing import Literal
import workspace.InverseProb.workflows.template.operator as operator
import workspace.InverseProb.workflows.round_4.prompt as prompt_custom
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
        # Generate multiple candidate solutions
        solutions = []
        for _ in range(3):  # Generate three independent solutions
            solution = await self.custom_code_generate(
                problem=problem,
                instruction=prompt_custom.CODE_GEN_PROMPT
            )
            if solution['response'].strip():  # Filter empty solutions
                solutions.append(solution['response'])
        
        # Handle case where all generations failed
        if not solutions:
            fallback = await self.custom_code_generate(
                problem=problem,
                instruction="Generate valid Python code solution"
            )
            return fallback['response'], self.llm.get_usage_summary()["total_cost"]
        
        # Ensemble selection of best solution
        candidate = await self.sc_ensemble(
            solutions=solutions,
            problem=problem
        )
        
        # Review and refine the solution
        review = await self.custom(
            input=candidate['response'],
            instruction=prompt_custom.REVIEW_PROMPT
        )
        
        # Revise based on review feedback
        revised = await self.custom_code_generate(
            problem=f"Original problem: {problem}\n\nInitial solution:\n{candidate['response']}\n\nReview feedback:\n{review['response']}",
            instruction=prompt_custom.REVISE_PROMPT
        )
        
        return revised['response'], self.llm.get_usage_summary()["total_cost"]
