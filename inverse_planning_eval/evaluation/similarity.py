
import os
# Set HF Mirror to avoid timeout in restricted network environments
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SimilarityEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the SimilarityEvaluator with a sentence transformer model.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"SimilarityEvaluator initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            self.model = None

    def compute_embedding_similarity(self, text1, text2):
        """
        Computes cosine similarity between embeddings of two texts.
        """
        if not self.model:
            return 0.0
        
        embeddings = self.model.encode([text1, text2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)

    def compute_structural_similarity(self, plan_json1, plan_json2):
        """
        Computes a heuristic similarity score based on JSON structure.
        Compatible with new complex schema:
        - Checks 'data_preprocessing' steps
        - Checks 'main_inverse_process' -> 'iteration_loop' steps
        - Checks 'objective_function' similarity
        """
        # Parse if strings
        if isinstance(plan_json1, str):
            try:
                plan_json1 = json.loads(plan_json1)
            except:
                return 0.0
        if isinstance(plan_json2, str):
            try:
                plan_json2 = json.loads(plan_json2)
            except:
                return 0.0
                
        # Helper to extract step names from a list of step objects
        def get_step_names(step_list):
            if not isinstance(step_list, list): return set()
            return set([str(s.get('step_name', '')).lower() for s in step_list if isinstance(s, dict)])

        scores = []
        
        # 1. Compare Iteration Loop (Main Inverse Process)
        # Handle both old schema (direct 'iteration_loop') and new schema ('main_inverse_process' -> 'iteration_loop')
        loop1 = plan_json1.get('iteration_loop', [])
        if 'main_inverse_process' in plan_json1:
            loop1 = plan_json1['main_inverse_process'].get('iteration_loop', [])
            
        loop2 = plan_json2.get('iteration_loop', [])
        if 'main_inverse_process' in plan_json2:
            loop2 = plan_json2['main_inverse_process'].get('iteration_loop', [])
            
        names1 = get_step_names(loop1)
        names2 = get_step_names(loop2)
        
        if not names1 and not names2:
            scores.append(1.0)
        elif not names1 or not names2:
            scores.append(0.0)
        else:
            intersection = len(names1.intersection(names2))
            union = len(names1.union(names2))
            scores.append(intersection / union if union > 0 else 0.0)

        # 2. Compare Data Preprocessing (New Schema Feature)
        prep1 = plan_json1.get('data_preprocessing', {}).get('steps', [])
        prep2 = plan_json2.get('data_preprocessing', {}).get('steps', [])
        
        p_names1 = get_step_names(prep1)
        p_names2 = get_step_names(prep2)
        
        if p_names1 or p_names2:
            intersection = len(p_names1.intersection(p_names2))
            union = len(p_names1.union(p_names2))
            scores.append(intersection / union if union > 0 else 0.0)
            
        # 3. Compare Objective Function Keys (New Schema Feature)
        obj1 = plan_json1.get('objective_function', {})
        obj2 = plan_json2.get('objective_function', {})
        if obj1 or obj2:
            # Simple Jaccard on keys presence or basic content match? 
            # Let's just check if both define it.
            if obj1 and obj2:
                scores.append(1.0)
            else:
                scores.append(0.0)
        
        if not scores:
            return 0.0
            
        return sum(scores) / len(scores)

    def evaluate(self, generated_plan, golden_plan):
        """
        Returns a dict of similarity metrics.
        """
        # Convert to string for embedding
        gen_str = json.dumps(generated_plan) if isinstance(generated_plan, (dict, list)) else str(generated_plan)
        gold_str = json.dumps(golden_plan) if isinstance(golden_plan, (dict, list)) else str(golden_plan)
        
        embed_sim = self.compute_embedding_similarity(gen_str, gold_str)
        struct_sim = self.compute_structural_similarity(generated_plan, golden_plan)
        
        return {
            "embedding_similarity": embed_sim,
            "structural_similarity": struct_sim,
            "combined_score": 0.5 * embed_sim + 0.5 * struct_sim
        }
