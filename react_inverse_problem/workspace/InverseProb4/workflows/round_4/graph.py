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
        self.cost_tracker = CostManage()

    async def __call__(self, problem: str):
        # Track total cost
        total_cost = 0.0
        
        # Step 1: Generate solution plan
        plan = await self.custom(
            input=problem,
            instruction=prompt_custom.PLAN_PROMPT
        )
        self.cost_tracker.add_cost(self.llm.get_usage_summary()["total_cost"])
        
        # Step 2: Generate initial solution using plan
        solution = await self.custom_code_generate(
            problem=f"Problem: {problem}\nPlan: {plan['response']}",
            instruction=prompt_custom.CODE_GEN_PROMPT
        )
        self.cost_tracker.add_cost(self.llm.get_usage_summary()["total_cost"])
        initial_solution = solution['response']
        
        # Step 3: Generate alternative solutions
        alt_solutions = []
        for _ in range(2):  # Generate 2 additional solutions
            alt_solution = await self.custom_code_generate(
                problem=f"Problem: {problem}\nReference Solution: {initial_solution}",
                instruction=prompt_custom.ALT_SOLUTION_PROMPT
            )
            self.cost_tracker.add_cost(self.llm.get_usage_summary()["total_cost"])
            alt_solutions.append(alt_solution['response'])
        
        # Step 4: Self-review of solutions
        reviewed_solutions = []
        all_solutions = [initial_solution] + alt_solutions
        
        for idx, sol in enumerate(all_solutions):
            review = await self.custom(
                input=f"Problem: {problem}\nSolution: {sol}",
                instruction=prompt_custom.REVIEW_PROMPT
            )
            self.cost_tracker.add_cost(self.llm.get_usage_summary()["total_cost"])
            reviewed_solutions.append({
                "solution": sol,
                "review": review['response']
            })
        
        # Step 5: Revise solutions based on reviews
        revised_solutions = []
        for item in reviewed_solutions:
            revised = await self.custom(
                input=f"Problem: {problem}\nSolution: {item['solution']}\nReview: {item['review']}",
                instruction=prompt_custom.REVISE_PROMPT
            )
            self.cost_tracker.add_cost(self.llm.get_usage_summary()["total_cost"])
            revised_solutions.append(revised['response'])
        
        # Step 6: Ensemble selection of best solution
        final_selection = await self.sc_ensemble(
            solutions=revised_solutions,
            problem=problem
        )
        self.cost_tracker.add_cost(self.llm.get_usage_summary()["total_cost"])
        best_solution = final_selection['response']
        
        # Fallback mechanism
        if not best_solution.strip():
            best_solution = max(revised_solutions, key=len, default="")
        if not best_solution.strip():
            best_solution = initial_solution
        
        return best_solution, self.cost_tracker.get_total_cost()
