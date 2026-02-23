
import logging
from litellm import completion

logger = logging.getLogger(__name__)

ELO_JUDGE_PROMPT = """
You are an expert impartial judge evaluating AI-generated test plans for scientific coding tasks.
Your goal is to determine which of two plans is better, or if they are tied.

**Task Context:**
{task_description}

**Plan A (Candidate 1):**
{plan_a}

**Plan B (Candidate 2):**
{plan_b}

**Evaluation Criteria:**
1. **Correctness**: Does the plan correctly implement the logic required by the task?
2. **Completeness**: Does the plan cover all critical steps (initialization, iteration loop, operators)?
3. **Structure**: Is the plan well-structured (JSON format, clear steps)?
4. **Precision**: Does the plan use correct mathematical descriptions and variable names?

**Verdict:**
Compare Plan A against Plan B.
If Plan A is better, output "A is Better".
If Plan B is better, output "B is Better".
If they are functionally equivalent or of equal quality, output "Tie".

Output your decision as one of:
- "A is Better"
- "Tie"
- "B is Better"

**Reasoning:**
Provide a brief explanation for your decision.

Output Format:
[[Verdict]]
Reasoning...
"""

from collections import Counter

class EloEvaluator:
    def __init__(self, judge_config):
        """
        judge_config: {
            "primary": ["model_key1", "model_key2", ...],
            "reserves": ["model_key3", "model_key4", ...],
            "api_configs": { "model_key": {"api_key": "...", "base_url": "...", "provider_prefix": "openai/"}, ... }
        }
        """
        self.judge_config = judge_config
        self.K = 32
        self.initial_rating = 1000

    def _call_judge(self, judge_model_key, prompt):
        """
        Helper to call a specific judge model.
        """
        conf = self.judge_config['api_configs'].get(judge_model_key)
        if not conf:
            logger.error(f"Config not found for judge: {judge_model_key}")
            return None

        try:
            # Set env vars for this specific call (litellm reads env vars, but we might need to be careful with concurrency if threaded)
            # Since this is sequential, os.environ is fine, but passing api_key to completion is safer if supported.
            # Litellm supports passing api_key and base_url directly.
            
            model_name = f"{conf.get('provider_prefix', 'openai/')}{judge_model_key}"
            
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=conf['api_key'],
                base_url=conf['base_url'],
                temperature=0.0 # Deterministic judgment
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Judge {judge_model_key} failed: {e}")
            return None

    def _parse_verdict(self, content):
        if not content:
            return 0.5 # Default to tie on error
            
        if "[[A is Better]]" in content:
            return 1.0
        elif "[[B is Better]]" in content:
            return 0.0
        elif "[[Tie]]" in content:
            return 0.5
        else:
            lower = content.lower()
            if "a is better" in lower:
                return 1.0
            elif "b is better" in lower:
                return 0.0
            elif "tie" in lower:
                return 0.5
            else:
                return 0.5

    def get_judges_for_match(self, model_a, model_b):
        """
        Selects 3 judges.
        Rule: If a judge is the same as model_a or model_b, they must be recused.
        """
        primary = self.judge_config['primary']
        reserves = self.judge_config['reserves']
        
        active_judges = []
        
        # Check primary judges
        for judge in primary:
            # Simple string matching. 
            # Note: Input model names might differ slightly from judge keys, ensure consistency in config.
            if judge == model_a or judge == model_b:
                continue # Recuse
            active_judges.append(judge)
            
        # Fill with reserves if needed
        reserve_idx = 0
        while len(active_judges) < 3 and reserve_idx < len(reserves):
            r_judge = reserves[reserve_idx]
            if r_judge != model_a and r_judge != model_b:
                active_judges.append(r_judge)
            reserve_idx += 1
            
        # If still < 3 (rare edge case where reserves are also participants), we might have to duplicate or accept fewer.
        # For this specific requirement, we assume pool is sufficient.
        
        return active_judges[:3]

    def get_verdict_voting(self, plan_a, plan_b, task_description, model_a_name, model_b_name):
        """
        Queries 3 judges and returns the majority verdict.
        """
        judges = self.get_judges_for_match(model_a_name, model_b_name)
        logger.info(f"    Judges for {model_a_name} vs {model_b_name}: {judges}")
        
        votes = []
        reasonings = {}
        
        prompt = ELO_JUDGE_PROMPT.format(
            task_description=task_description,
            plan_a=plan_a,
            plan_b=plan_b
        )
        
        for judge in judges:
            content = self._call_judge(judge, prompt)
            score = self._parse_verdict(content)
            votes.append(score)
            reasonings[judge] = content if content else "Error"
            
        # Majority Vote
        # 1.0 (A wins), 0.0 (B wins), 0.5 (Tie)
        # We can sum votes and divide? Or count logic.
        # Let's treat it as: sum > 1.5 -> A wins; sum < 1.5 -> B wins; else Tie?
        # Or strict count.
        
        # Count occurrences
        counts = Counter(votes)
        # Most common
        final_verdict = counts.most_common(1)[0][0]
        
        # If there's a 3-way tie (e.g. 1.0, 0.0, 0.5), the mean might be 0.5.
        # Simple mean is also a valid aggregated score for ELO.
        final_score = sum(votes) / len(votes) if votes else 0.5
        
        return final_score, reasonings

    def update_rating(self, rating_a, rating_b, score_a):
        # ... (same as before)
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        new_rating_a = rating_a + self.K * (score_a - expected_a)
        new_rating_b = rating_b + self.K * ((1 - score_a) - (1 - expected_a))
        return new_rating_a, new_rating_b

    def run_tournament(self, plans_dict, task_description):
        # ...
        ratings = {model: self.initial_rating for model in plans_dict.keys()}
        match_history = []
        models = list(plans_dict.keys())
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a = models[i]
                model_b = models[j]
                
                score_a, reasonings = self.get_verdict_voting(
                    plans_dict[model_a], 
                    plans_dict[model_b], 
                    task_description,
                    model_a,
                    model_b
                )
                
                # Update Ratings
                old_ra = ratings[model_a]
                old_rb = ratings[model_b]
                
                new_ra, new_rb = self.update_rating(old_ra, old_rb, score_a)
                
                ratings[model_a] = new_ra
                ratings[model_b] = new_rb
                
                match_history.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "score_a": score_a,
                    "judges": list(reasonings.keys()),
                    "rating_change_a": new_ra - old_ra,
                    "rating_change_b": new_rb - old_rb
                })
                
        return ratings, match_history
