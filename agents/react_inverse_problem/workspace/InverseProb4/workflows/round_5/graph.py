from typing import Literal
import workspace.InverseProb.workflows.template.operator as operator
import workspace.InverseProb.workflows.round_5.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

import json
import operator
from typing import List, Tuple

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

    async def __call__(self, problem: str) -> Tuple[str, float]:
        # Step 1: Generate initial solution plan
        plan = await self.custom(
            input=problem,
            instruction=prompt_custom.PLAN_PROMPT
        )
        
        # Step 2: Refine plan through critical review
        refined_plan = await self.custom(
            input=f"Original Plan:\n{plan['response']}\nProblem Context:\n{problem}",
            instruction=prompt_custom.PLAN_REFINE_PROMPT
        )
        
        # Step 3: Generate test cases
        test_cases = await self.custom(
            input=problem,
            instruction=prompt_custom.TEST_CASE_PROMPT
        )
        
        # Step 4: Generate candidate solutions
        solutions = []
        for i in range(3):
            solution = await self.custom_code_generate(
                problem=f"Refined Plan:\n{refined_plan['response']}\nProblem:\n{problem}",
                entry_point="main",
                instruction=prompt_custom.CODE_PROMPT
            )
            if solution['response'].strip():
                solutions.append(solution['response'])
        
        # Fallback if no solutions generated
        if not solutions:
            fallback = await self.custom_code_generate(
                problem=problem,
                entry_point="main",
                instruction=prompt_custom.CODE_PROMPT
            )
            return fallback['response'], self.llm.get_usage_summary()["total_cost"]
        
        # Step 5: Cross-validate solutions
        validation_results = []
        for idx, solution in enumerate(solutions):
            other_solutions = [s for i, s in enumerate(solutions) if i != idx]
            validation = await self.custom(
                input=f"Problem:\n{problem}\nTest Cases:\n{test_cases['response']}\nSolution:\n{solution}\nReference Solutions:\n{json.dumps(other_solutions)}",
                instruction=prompt_custom.VALIDATE_PROMPT
            )
            validation_results.append(json.loads(validation['response']))
        
        # Step 6: Weighted solution selection
        scored_solutions = []
        for i, res in enumerate(validation_results):
            score = res['pass_rate'] * 0.7 + res['quality_score'] * 0.3
            scored_solutions.append((solutions[i], score))
        
        # Sort by composite score
        scored_solutions.sort(key=lambda x: x[1], reverse=True)
        top_solution = scored_solutions[0][0]
        
        # Step 7: Iterative refinement
        refined_solution = top_solution
        for _ in range(2):  # Two refinement cycles
            refinement = await self.custom(
                input=f"Problem:\n{problem}\nCurrent Solution:\n{refined_solution}\nValidation Feedback:\n{json.dumps(validation_results)}",
                instruction=prompt_custom.REFINE_PROMPT
            )
            if refinement['response'].strip():
                refined_solution = refinement['response']
        
        return refined_solution, self.llm.get_usage_summary()["total_cost"]
