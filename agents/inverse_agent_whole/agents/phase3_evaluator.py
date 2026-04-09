import os
import sys
import subprocess
import json
import pickle
import numpy as np
from utils.llm import llm_client
from prompts import phase3 as prompts
from config import settings
from utils.logger import RunLogger

try:
    from skimage.metrics import peak_signal_noise_ratio
    from skimage import io
    from PIL import Image
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class ResultAnalyzer:
    def __init__(self):
        self.model_name = settings.DEFAULT_JUDGE_MODEL

    def _load_data(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        ext = os.path.splitext(path)[1].lower()
        
        try:
            if ext == '.npy':
                try:
                    return np.load(path, allow_pickle=True)
                except Exception as e_npy:
                    # Fallback: Check if it's an image renamed to .npy (Phase 0 artifact)
                    try:
                        return np.asarray(Image.open(path))
                    except:
                        raise e_npy
            elif ext == '.npz':
                return np.load(path, allow_pickle=True)
            elif ext == '.mat':
                if not HAS_SCIPY:
                    raise ImportError("scipy is required for .mat files")
                mat = scipy.io.loadmat(path)
                # Heuristic: return the first non-dunder key
                for k in mat:
                    if not k.startswith('__'):
                        return mat[k]
                raise ValueError(f"No valid data found in .mat file: {path}")
            elif ext in ['.pkl', '.pickle']:
                with open(path, "rb") as f:
                    return pickle.load(f)
            elif ext == '.json':
                with open(path, "r") as f:
                    return json.load(f)
            elif ext == '.txt':
                with open(path, "r") as f:
                    return f.read()
            elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                if HAS_SKIMAGE:
                    return io.imread(path)
                else:
                    return np.asarray(Image.open(path))
            else:
                # Try image fallback for unknown extensions
                try:
                    return np.asarray(Image.open(path))
                except:
                    raise ValueError(f"Unsupported file extension: {ext}")
        except Exception as e:
            raise ValueError(f"Failed to load {path}: {str(e)}")

    def _calculate_psnr(self, expected, reference):
        """
        Calculate PSNR.
        - If images (uint8/3-channel): Use skimage.metrics.peak_signal_noise_ratio
        - If arrays: Use custom implementation with 255 normalization as requested.
        """
        try:
            # Ensure numpy arrays
            if not isinstance(expected, np.ndarray):
                expected = np.array(expected)
            if not isinstance(reference, np.ndarray):
                reference = np.array(reference)
            
            # Check for image-like properties
            is_image = False
            if expected.ndim >= 2 and (expected.dtype == np.uint8 or reference.dtype == np.uint8):
                is_image = True
            elif expected.ndim == 3 and expected.shape[2] in [3, 4]: # RGB/RGBA
                is_image = True
            
            if is_image and HAS_SKIMAGE:
                # Skimage PSNR
                # Ensure dimensions match
                if expected.shape != reference.shape:
                     return -1.0 # Mismatch
                
                # Infer data_range if not explicit
                data_range = None
                if expected.dtype == np.uint8 or reference.dtype == np.uint8:
                    data_range = 255
                else:
                    data_range = max(expected.max(), reference.max()) - min(expected.min(), reference.min())
                    if data_range == 0: data_range = 1.0
                
                return peak_signal_noise_ratio(reference, expected, data_range=data_range)
            
            else:
                # Custom Array PSNR (255 Normalization)
                # Flatten for simple MSE
                exp_flat = expected.flatten().astype(np.float64)
                ref_flat = reference.flatten().astype(np.float64)
                
                if exp_flat.shape != ref_flat.shape:
                    return -1.0
                
                def normalize_255(x):
                    x_min, x_max = x.min(), x.max()
                    if x_max == x_min:
                        return np.zeros_like(x)
                    return (x - x_min) / (x_max - x_min) * 255.0

                exp_norm = normalize_255(exp_flat)
                ref_norm = normalize_255(ref_flat)
                
                mse = np.mean((exp_norm - ref_norm) ** 2)
                if mse == 0:
                    return 100.0 # Perfect match
                return 10 * np.log10(255**2 / mse)
                
        except Exception as e:
            # print(f"PSNR Calc Error: {e}")
            return -1.0

    def verify_task(self, expected_output_path: str, gt_output_path: str, gt_reference_path: str) -> dict:
        task_id = os.path.basename(os.path.dirname(expected_output_path))
        result = {
            "task_id": task_id,
            "psnr_expected": 0.0,
            "psnr_gt": 0.0,
            "threshold": 0.0,
            "success": False,
            "error": None
        }

        try:
            # 1. Load Data
            try:
                expected = self._load_data(expected_output_path)
                gt_out = self._load_data(gt_output_path)
                gt_ref = self._load_data(gt_reference_path)
            except Exception as e:
                result["error"] = f"Data Loading Failed: {str(e)}"
                return result

            # 2. Validation Logic
            # Note: shapes might differ slightly if LLM code did something wrong, 
            # but _calculate_psnr handles shape mismatch by returning -1 or error.
            
            # Calculate PSNR (Expected vs Reference)
            psnr_exp = self._calculate_psnr(expected, gt_ref)
            
            # Calculate PSNR (GT Output vs Reference)
            psnr_gt_val = self._calculate_psnr(gt_out, gt_ref)
            
            result["psnr_expected"] = float(psnr_exp)
            result["psnr_gt"] = float(psnr_gt_val)
            result["threshold"] = 0.8 * result["psnr_gt"]
            
            # 3. Success Criteria
            # Handle cases where PSNR calculation failed (-1)
            if psnr_exp == -1 or psnr_gt_val == -1:
                result["success"] = False
                result["error"] = "PSNR Calculation Failed (Shape mismatch or invalid data)"
            else:
                result["success"] = result["psnr_expected"] >= result["threshold"]

        except Exception as e:
            result["success"] = False
            result["error"] = f"Verification Exception: {str(e)}"
            
        return result

    def generate_model_summary(self, results: list) -> str:
        total = len(results)
        passed = sum(1 for r in results if r.get("success", False))
        success_rate = (passed / total * 100) if total > 0 else 0.0
        
        psnr_exps = [r["psnr_expected"] for r in results if "psnr_expected" in r and r["psnr_expected"] != -1]
        psnr_gts = [r["psnr_gt"] for r in results if "psnr_gt" in r and r["psnr_gt"] != -1]
        
        avg_psnr_exp = np.mean(psnr_exps) if psnr_exps else 0.0
        avg_psnr_gt = np.mean(psnr_gts) if psnr_gts else 0.0
        
        failed_tasks = [r.get("task_id", "unknown") for r in results if not r.get("success", False)]
        
        report = f"# Model Evaluation Summary: {self.model_name}\n\n"
        report += "## Overall Statistics\n"
        report += f"- Total Tasks: {total}\n"
        report += f"- Passed Tasks: {passed}\n"
        report += f"- Success Rate: {success_rate:.1f}%\n\n"
        
        report += "## PSNR Distribution\n"
        report += f"- Avg PSNR (Expected): {avg_psnr_exp:.2f} dB\n"
        report += f"- Avg PSNR (GT): {avg_psnr_gt:.2f} dB\n\n"
        
        report += "## Failed Tasks\n"
        for task in failed_tasks:
            report += f"- {task}\n"
        
        return report

    # verify method removed as it depended on oracle_verify.py


class FeedbackGenerator:
    def __init__(self, model_name: str = settings.DEFAULT_JUDGE_MODEL):
        self.model_name = model_name

    def generate_feedback(self, failure_info: dict, plan: dict, code: str, logger: RunLogger = None) -> dict:
        """
        Categorizes error into Type A/B/C and generates feedback.
        """
        prompt = f"""
        Execution/Verification Failed.
        
        Reason: {failure_info.get('reason')}
        Logs: {failure_info.get('logs')}
        
        Plan: {json.dumps(plan, indent=2)}
        
        Code:
        ```python
        {code}
        ```
        """
        
        res = llm_client.call(
            prompts.FEEDBACK_GENERATOR_SYSTEM_PROMPT,
            prompt,
            model=self.model_name,
            json_mode=True
        )
        if logger:
            logger.log_trace("phase3", "feedback_thinking", res.get("thinking", ""))
            logger.log_trace("phase3", "feedback_response", json.dumps(res.get("content", {}), indent=2))
            
        return res.get("content", {})
