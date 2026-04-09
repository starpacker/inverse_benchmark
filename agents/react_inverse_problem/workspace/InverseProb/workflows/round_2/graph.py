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
        # Generate multiple candidate solutions
        solutions = []
        for _ in range(3):
            solution = await self.custom_code_generate(
                problem=problem,
                instruction=prompt_custom.CODE_GEN_PROMPT
            )
            if solution['response'].strip():
                solutions.append(solution['response'])
        
        # Fallback if all generations fail
        if not solutions:
            fallback = await self.custom_code_generate(
                problem=problem,
                instruction="Generate valid Python code solution"
            )
            return fallback['response'], self.llm.get_usage_summary()["total_cost"]
        
        # Score and filter solutions
        scored_solutions = []
        for sol in solutions:
            score_result = await self.custom(
                input=sol,
                instruction=prompt_custom.SCORE_PROMPT
            )
            try:
                score = float(score_result['response'].split(":")[-1].strip())
                if score >= 0.7:  # Quality threshold
                    scored_solutions.append(sol)
            except:
                continue
        
        # Fallback if no solutions meet threshold
        if not scored_solutions:
            scored_solutions = solutions
        
        # Ensemble selection
        candidate = await self.sc_ensemble(
            solutions=scored_solutions,
            problem=problem
        )
        current_solution = candidate['response']
        accumulated_feedback = ""
        
        # Iterative refinement loop (2 cycles)
        for _ in range(2):
            # Review current solution
            review = await self.custom(
                input=current_solution,
                instruction=prompt_custom.REVIEW_PROMPT
            )
            accumulated_feedback += f"\nReview Cycle {_+1}:\n{review['response']}\n"
            
            # Revise based on accumulated feedback
            revised = await self.custom_code_generate(
                problem=f"Problem: {problem}\nCurrent Solution:\n{current_solution}\nAccumulated Feedback:{accumulated_feedback}",
                instruction=prompt_custom.REVISE_PROMPT
            )
            current_solution = revised['response']
        
        # Final validation
        validation = await self.custom(
            input=current_solution,
            instruction=prompt_custom.VALIDATE_PROMPT
        )
        if "PASS" in validation['response']:
            return current_solution, self.llm.get_usage_summary()["total_cost"]
        else:
            # Return with validation warnings if failed
            return f"{current_solution}\n\n# Validation Warnings:\n{validation['response']}", self.llm.get_usage_summary()["total_cost"]
