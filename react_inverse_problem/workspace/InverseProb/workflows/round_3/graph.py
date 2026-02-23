from typing import Literal
import workspace.InverseProb.workflows.template.operator as operator
import workspace.InverseProb.workflows.round_3.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

import asyncio
import operator
from typing import List

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
        # Step 1: Generate solution plan
        plan = await self.custom(
            input=problem,
            instruction=prompt_custom.PLAN_PROMPT
        )
        plan_text = plan['response']
        
        # Step 2: Generate multiple candidate solutions
        candidates = []
        for _ in range(3):  # Generate 3 candidate solutions
            solution = await self.custom_code_generate(
                problem=f"Problem: {problem}\nPlan: {plan_text}",
                instruction=prompt_custom.CODE_GEN_PROMPT
            )
            candidates.append(solution['response'])
        
        # Step 3: Select best solution using self-consistency
        ensemble = await self.sc_ensemble(
            solutions=candidates,
            problem=problem
        )
        final_solution = ensemble['response']
        
        # Step 4: Validate solution against plan
        validation = await self.custom(
            input=f"Problem: {problem}\nPlan: {plan_text}\nSolution: {final_solution}",
            instruction=prompt_custom.VALIDATION_PROMPT
        )
        
        # Step 5: Return results with cost
        return {
            "plan": plan_text,
            "solution": final_solution,
            "validation": validation['response'],
            "cost": self.llm.get_usage_summary()["total_cost"]
        }
