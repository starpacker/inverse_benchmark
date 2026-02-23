from typing import Any, Dict
from agents.base import BaseAgent
import subprocess

class DataGenAgent(BaseAgent):
    def _build_system_prompt(self) -> str:
        return """You are an expert Data Engineer for scientific computing.
                Your Goal: Create a robust Python script `data_gen.py` to prepare datasets for an Inverse Problem.
                Output Format: Return ONLY the Python code block. No markdown, no explanation.
                Constraints:
                1. The code must be self-contained or import provided GT code correctly.
                2. It MUST generate three files in `dataset/`: 
                - `input.npy` (The measurement/degraded data)
                14. - `gt_output.npy` (The ground truth)
                15. - `baseline.npy` (A simple heuristic result, e.g., identity or simple filter)
                16. 3. **Data Complexity**: 
                    - AVOID trivial data (e.g., all zeros, all ones, or simple constants).
                    - Use `numpy.random` to generate complex, non-trivial signals (e.g., mixtures of sinusoids, Gaussian random fields, or realistic synthetic data relevant to the domain).
                    - Ensure the data has sufficient variance and structure to challenge the solver.
                17. 4. Ensure random seeds are fixed for reproducibility.
                18. 5. **Strict Type Safety**: All saved .npy files MUST be standard numeric arrays (e.g., `.astype(np.float32)`). Do NOT save Python object arrays (lists of lists, ragged arrays) which trigger pickle security errors.
                19. """

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        return f"""
        Task Description: {context['task_desc']}
        
        Reference Ground Truth Code Context:
        {context['gt_code_snippet']}
        
        Please write `data_gen.py`.
        """

class EvalGenAgent(BaseAgent):
    def _build_system_prompt(self) -> str:
        return """You are a QA Engineer for Computer Vision/Signal Processing.
Your Goal: Create an evaluation script `eval_script.py`.
Output Format: Return ONLY the Python code.

35. Requirements:
36. 1. Accept one command-line argument: path to the prediction file (.npy).
37. 2. Load `dataset/gt_output.npy` as the reference. Use `np.load(path, allow_pickle=True)` to avoid security errors with complex data types.
38. 3. Calculate PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity).
39.    - Use `skimage.metrics` or robust numpy implementations.
   - Ensure data range is handled correctly (e.g., if data is [0,1], max_val=1.0).
4. **Metric Guardrails**:
   - Handle minor shape mismatches gracefully (e.g., if pred is (1,H,W) and gt is (H,W), squeeze the singleton dimension).
   - If shapes are fundamentally incompatible (e.g., different spatial dimensions), print JSON with detailed shape info:
     {"psnr": 0.0, "ssim": 0.0, "error": "Shape mismatch: pred shape (X,Y,Z) vs gt shape (A,B,C)"}
     where (X,Y,Z) and (A,B,C) are the actual shapes of prediction and ground truth arrays.
5. Print the result strictly in JSON format to stdout: {"psnr": <float>, "ssim": <float>}
"""
    
    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        return f"""
        Task: {context['task_desc']}
        
        Data Shape Hint: {context.get('data_shape_hint', 'N/A')}
        
        Please generate `eval_script.py` that calculates PSNR and SSIM between `dataset/gt_output.npy` and the provided prediction file.
        """

def get_installed_libraries(python_path: str) -> str:
    """Detects installed libraries in the environment."""
    try:
        result = subprocess.run(
            [python_path, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            import json
            packages = json.loads(result.stdout)
            return ", ".join([p["name"] for p in packages])
    except Exception:
        pass
    return "Standard Python Libraries"
