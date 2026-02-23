from typing import Literal
import workspace.InverseProb.workflows.template.operator as operator
import workspace.InverseProb.workflows.round_3.prompt as prompt_custom
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
        # Step 1: Generate initial solution plan
        plan = await self.custom(
            input=problem,
            instruction=prompt_custom.PLAN_PROMPT
        )
        plan_text = plan['response'] or "No plan generated"
        
        # Step 2: Refine plan through critical review
        refined_plan = await self.custom(
            input=f"Problem: {problem}\nInitial Plan: {plan_text}",
            instruction=prompt_custom.REFINE_PLAN_PROMPT
        )
        refined_plan_text = refined_plan['response'] or plan_text
        
        # Step 3: Generate diverse candidate solutions
        solutions = []
        for i in range(3):  # Generate 3 candidate solutions
            # Alternate between implementation strategies
            if i % 2 == 0:
                instruction = prompt_custom.DIRECT_CODE_PROMPT
            else:
                instruction = prompt_custom.MODULAR_CODE_PROMPT
                
            solution = await self.custom_code_generate(
                problem=f"Problem: {problem}\nRefined Plan: {refined_plan_text}",
                entry_point="main",
                instruction=instruction
            )
            if solution['response'].strip():
                solutions.append(solution['response'])
        
        # Fallback if all solutions empty
        if not solutions:
            fallback = await self.custom_code_generate(
                problem=problem,
                entry_point="main",
                instruction=prompt_custom.DIRECT_CODE_PROMPT
            )
            solutions = [fallback['response']]
        
        # Step 4: Multi-stage solution evaluation
        evaluated_solutions = []
        for idx, solution in enumerate(solutions):
            # Score each solution
            score_result = await self.custom(
                input=f"Problem: {problem}\nSolution:\n{solution}",
                instruction=prompt_custom.SCORE_SOLUTION_PROMPT
            )
            score_text = score_result['response'] or "Score: 0"
            evaluated_solutions.append((solution, score_text))
        
        # Select top 2 solutions by score
        top_solutions = sorted(evaluated_solutions, key=lambda x: int(x[1].split()[-1]), reverse=True)[:2]
        top_codes = [sol[0] for sol in top_solutions]
        
        # Comparative analysis of top solutions
        comparative = await self.custom(
            input=f"Problem: {problem}\nSolution A:\n{top_codes[0]}\n\nSolution B:\n{top_codes[1]}",
            instruction=prompt_custom.COMPARE_SOLUTIONS_PROMPT
        )
        best_idx = 0 if "A" in comparative['response'] else 1
        final_solution = top_codes[best_idx]
        
        # Step 5: Validate and repair solution
        validated = await self.custom(
            input=f"Problem: {problem}\nSolution:\n{final_solution}",
            instruction=prompt_custom.REPAIR_SOLUTION_PROMPT
        )
        final_code = validated['response'] or final_solution
        
        return final_code, self.llm.get_usage_summary()["total_cost"]
