import os
import json
import shutil
import glob
import time
import subprocess
import numpy as np
from utils.llm import llm_client
from prompts import phase0 as prompts
from config import settings
from utils.logger import RunLogger
from utils.sandbox import SandboxManager

class DatasetConstructor:
    def __init__(self, model_name: str = settings.DEFAULT_PHASE0_MODEL):
        self.model_name = model_name
        self.sandbox_mgr = SandboxManager()

    def process_task(self, task_name: str, gt_code_path: str = None, output_dir: str = None, working_folder: str = None, 
                     input_path: str = None, gt_output_path: str = None, paper_markdown_path: str = None,
                     model_name: str = None, logger: RunLogger = None, python_path: str = "python") -> dict:
        """
        Main Entry Point for Phase 0.
        Supports two scenarios:
        1. Oracle Pipeline (GT Code available): runs Closed-loop Verification.
        2. Real-world Pipeline (Paper available): runs Structured Extraction.
        """
        if model_name:
            self.model_name = model_name

        if not output_dir:
            output_dir = os.path.join(settings.DATASET_DIR, task_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Result Dictionary
        result = {
            "task_name": task_name,
            "task_description": "",
            "golden_plan": {},
            "oracle_path": None,
            "inferred_io": {}
        }

        # --- Scenario 1: Oracle Pipeline (GT Code) ---
        if gt_code_path and os.path.exists(gt_code_path):
            print(f"[Phase0] Running Oracle Pipeline for {task_name}...")
            if logger: logger.log("phase0", f"Starting Oracle Pipeline with GT Code: {gt_code_path}")

            with open(gt_code_path, 'r') as f:
                code_content = f.read()

            # 1. Closed-loop Verification (The Loop)
            io_info = self._run_oracle_pipeline(
                task_name, gt_code_path, working_folder, output_dir, logger, python_path
            )
            result["inferred_io"] = io_info
            
            # Update paths if generated
            final_input = io_info.get("input_path", input_path)
            final_gt = io_info.get("gt_output_path", gt_output_path)

            # 2. Generate Artifacts based on Code
            result["task_description"] = self._generate_task_description(code_content, logger)
            result["golden_plan"] = self._generate_golden_plan(code_content, logger)
            
            # 3. Generate Oracle Script - REMOVED
            # if final_input and final_gt:
            #    result["oracle_path"] = self.generate_oracle(final_input, final_gt, output_dir, logger)

            # Save artifacts
            with open(os.path.join(output_dir, "task_description.md"), "w") as f:
                f.write(result["task_description"])
            with open(os.path.join(output_dir, "golden_plan.json"), "w") as f:
                json.dump(result["golden_plan"], f, indent=2)
            with open(os.path.join(output_dir, "task_io_info.json"), "w") as f:
                json.dump(result["inferred_io"], f, indent=2)

        # --- Scenario 2: Paper Pipeline (Markdown) ---
        elif paper_markdown_path and os.path.exists(paper_markdown_path):
            print(f"[Phase0] Running Real-world Pipeline (Paper) for {task_name}...")
            if logger: logger.log("phase0", f"Starting Paper Pipeline with: {paper_markdown_path}")
            
            with open(paper_markdown_path, 'r') as f:
                paper_content = f.read()
                
            # 1. Structured Extraction
            result["task_description"] = self._run_paper_pipeline(paper_content, logger)
            
            # Save
            with open(os.path.join(output_dir, "task_description.md"), "w") as f:
                f.write(result["task_description"])
                
        else:
            msg = f"[Phase0] Error: Neither gt_code_path nor paper_markdown_path provided/valid for {task_name}"
            print(msg)
            if logger: logger.log("error", msg)
            
        return result

    # =========================================================================
    # Scenario 1: Oracle Pipeline (Closed-loop Verification)
    # =========================================================================

    def _run_oracle_pipeline(self, task_name, gt_code_path, working_folder, output_dir, logger, python_path):
        """
        Implements the Loop: Analyze -> Synthesize -> Exec -> Verify -> Feedback
        Modified: Accept last attempt's results even on metric failure after max retries
        """
        # 1. Setup Sandbox (保持不变)
        sandbox_path = self.sandbox_mgr.create_sandbox(working_folder, f"{task_name}_phase0_oracle")
        print(f"[Phase0][Oracle] Created Sandbox: {sandbox_path}")
        gt_filename = os.path.basename(gt_code_path)
        sandbox_gt_path = os.path.join(sandbox_path, gt_filename)
        if not os.path.exists(sandbox_gt_path):
            shutil.copy2(gt_code_path, sandbox_gt_path)
            
        with open(sandbox_gt_path, 'r') as f:
            gt_code_content = f.read()

        # 2. Analyze (保持不变)
        print("[Phase0] Analyzing GT Code for Forward Model...")
        analysis = self._analyze_forward_model(gt_code_content, logger)
        
        # 3. Synthesize & Loop (关键修改点)
        max_retries = 2
        feedback = None
        data_gen_code = None
        
        best_io = {}
        last_valid_attempt = None  # 记录最后一次成功生成数据的尝试
        
        for attempt in range(max_retries + 1):
            print(f"[Phase0] Synthesis Loop Iteration {attempt+1}...")
            
            # A. Generate/Refine Data Gen Script (保持不变)
            if attempt == 0:
                data_gen_res = self._synthesize_data_gen(gt_code_content, analysis, gt_filename, logger)
            else:
                data_gen_res = self._refine_data_gen(data_gen_code, feedback, logger)
            
            data_gen_code = data_gen_res.get("data_gen_code")
            x_true_name = data_gen_res.get("x_true_filename", "x_true.npy")
            y_meas_name = data_gen_res.get("y_meas_filename", "input_measurement.npy")
            
            # Write data_gen.py to sandbox (保持不变)
            with open(os.path.join(sandbox_path, "data_gen.py"), "w") as f:
                f.write(data_gen_code)
                
            # B. Exec (Gen): Run data_gen (保持不变)
            print("[Phase0] Executing Data Generator...")
            gen_res = self._run_script(sandbox_path, "data_gen.py", python_path)
            if gen_res["returncode"] != 0:
                feedback = f"Data Generation Failed (Runtime Error):\n{gen_res['stderr']}"
                print(f"[Phase0] Error: {feedback}")
                if logger: logger.log("phase0_warn", f"Data Gen Failed: {feedback}")
                continue
                
            # Check if files exist (保持不变)
            if not os.path.exists(os.path.join(sandbox_path, x_true_name)) or \
            not os.path.exists(os.path.join(sandbox_path, y_meas_name)):
                feedback = f"Data Generator ran but did not produce expected files: {x_true_name}, {y_meas_name}"
                print(f"[Phase0] Error: {feedback}")
                continue

            # C. Exec (Solve): Run GT Code (保持不变)
            print("[Phase0] Executing GT Solver...")
            solve_res = self._run_script(sandbox_path, gt_filename, python_path)
            if solve_res["returncode"] != 0:
                feedback = f"GT Solver Failed (Runtime Error) on synthetic data:\n{solve_res['stderr']}"
                print(f"[Phase0] Error: {feedback}")
                continue

            # D. Evaluate (Verify) - 关键修改：记录每次有效尝试的结果
            output_files = self._find_new_files(sandbox_path, [gt_filename, "data_gen.py", x_true_name, y_meas_name])
            if not output_files:
                feedback = "GT Solver ran but produced no output files."
                print(f"[Phase0] Error: {feedback}")
                continue
                
            # Prefer .npy file if multiple files found (保持不变)
            recon_file = None
            for f in output_files:
                if f.endswith('.npy'):
                    recon_file = f
                    break
            if recon_file is None:
                recon_file = output_files[0]
            
            # Verify Metrics (保持不变)
            verify_res = self._verify_results(
                os.path.join(sandbox_path, x_true_name), 
                os.path.join(sandbox_path, recon_file), 
                logger
            )
            
            # === 关键修改开始 ===
            # 记录本次有效尝试的结果（无论验证是否通过）
            current_attempt = {
                "input_path": os.path.join(sandbox_path, y_meas_name),
                "gt_output_path": os.path.join(sandbox_path, x_true_name),  # x_true is ground truth
                "recon_path": os.path.join(sandbox_path, recon_file),
                "expected_output_filename": "reconstruction.npy",
                "baseline_metrics": verify_res,
                "passed": verify_res["passed"]
            }
            last_valid_attempt = current_attempt  # 更新为最新有效尝试
            
            # 如果验证通过，立即接受结果
            if verify_res["passed"]:
                print(f"[Phase0] Verification Passed! PSNR={verify_res['psnr']:.2f}")
                
                # 复制文件到输出目录
                shutil.copy2(os.path.join(sandbox_path, y_meas_name), os.path.join(output_dir, "input.npy"))
                shutil.copy2(os.path.join(sandbox_path, recon_file), os.path.join(output_dir, "gt_output.npy"))
                shutil.copy2(os.path.join(sandbox_path, x_true_name), os.path.join(output_dir, "gt_reference.npy"))
                
                best_io = {
                    "input_path": os.path.join(output_dir, "input.npy"),
                    "gt_output_path": os.path.join(output_dir, "gt_reference.npy"),  # x_true as ground truth
                    "expected_output_filename": "reconstruction.npy",
                    "baseline_metrics": verify_res
                }
                break  # 提前退出循环
            else:
                feedback = f"Metric Failure: {verify_res['reason']}"
                print(f"[Phase0] Verification Failed: {feedback}")
                # 不退出循环，继续重试
            # === 关键修改结束 ===

        # === 新增逻辑：处理达到最大重试次数的情况 ===
        if not best_io and last_valid_attempt:
            # 达到最大重试次数且从未验证通过，但存在有效数据生成结果
            print(f"[Phase0] Accepting last attempt after {max_retries+1} retries despite metric failure")
            if logger: 
                logger.log("phase0", f"Accepting metric-failed result after {max_retries+1} retries: {last_valid_attempt['baseline_metrics'].get('reason', 'unknown')}")

            # 复制最后一次尝试的文件
            shutil.copy2(last_valid_attempt["input_path"], os.path.join(output_dir, "input.npy"))
            shutil.copy2(last_valid_attempt["recon_path"], os.path.join(output_dir, "gt_output.npy"))
            shutil.copy2(last_valid_attempt["gt_output_path"], os.path.join(output_dir, "gt_reference.npy"))  # x_true
            
            # 构建 best_io（标记为 metric-failed 但接受）
            best_io = {
                "input_path": os.path.join(output_dir, "input.npy"),
                "gt_output_path": os.path.join(output_dir, "gt_reference.npy"),  # x_true as ground truth
                "expected_output_filename": "reconstruction.npy",
                "baseline_metrics": last_valid_attempt["baseline_metrics"],
                "accepted_despite_failure": True  # 新增标记字段
            }
        # === 新增逻辑结束 ===

        if not best_io:
            print("[Phase0] Warning: Closed-loop Verification failed after retries with no valid data generation.")
            if logger: logger.log("warning", "Closed-loop Verification failed with no valid attempts.")
            
        return best_io
    def _analyze_forward_model(self, code, logger):
        res = llm_client.call(
            "You are a Scientific Code Expert.",
            prompts.PROMPT_ANALYZE_FORWARD_MODEL.format(code=code),
            model=self.model_name,
            json_mode=True
        )
        return res.get("content", {})

    def _synthesize_data_gen(self, code, analysis, gt_filename, logger):
        res = llm_client.call(
            "You are a Python Data Scientist.",
            prompts.PROMPT_WRITE_DATA_GEN.format(
                code=code, 
                analysis=json.dumps(analysis), 
                gt_filename=gt_filename
            ),
            model=self.model_name,
            json_mode=True
        )
        return res.get("content", {})

    def _refine_data_gen(self, prev_code, feedback, logger):
        res = llm_client.call(
            "You are a Python Data Scientist.",
            prompts.PROMPT_DATA_GEN_FEEDBACK.format(
                previous_code=prev_code, 
                feedback=feedback
            ),
            model=self.model_name,
            json_mode=True
        )
        return res.get("content", {})

    def _run_script(self, cwd, filename, python_path):
        try:
            res = subprocess.run(
                f"{python_path} {filename}", 
                shell=True, cwd=cwd, capture_output=True, text=True, timeout=300
            )
            return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    def _find_new_files(self, sandbox_path, exclude_names):
        new_files = []
        for root, dirs, files in os.walk(sandbox_path):
            # Exclude __pycache__ and hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, sandbox_path)
                
                # Exclude hidden files, pyc files, and cache
                if any(part.startswith('.') for part in rel_path.split(os.sep)):
                    continue
                if file.endswith('.pyc') or file.endswith('.pyo'):
                    continue
                if '__pycache__' in rel_path:
                    continue
                    
                if os.path.basename(file) not in exclude_names:
                     new_files.append(rel_path)
        return new_files

    def _verify_results(self, x_true_path, x_recon_path, logger):
        try:
            if x_true_path.endswith('.npy'): x_true = np.load(x_true_path, allow_pickle=True)
            else: return {"passed": False, "reason": "Unsupported format for verification"}
            
            if x_recon_path.endswith('.npy'): x_recon = np.load(x_recon_path, allow_pickle=True)
            else: return {"passed": False, "reason": "Unsupported format for verification"}
            
            if x_true.shape != x_recon.shape:
                return {"passed": False, "reason": f"Shape mismatch: {x_true.shape} vs {x_recon.shape}"}
                
            dmin, dmax = x_true.min(), x_true.max()
            drange = dmax - dmin
            if drange == 0: drange = 1.0
            
            from skimage.metrics import peak_signal_noise_ratio as psnr_func
            from skimage.metrics import structural_similarity as ssim_func
            psnr = psnr_func(x_true, x_recon, data_range=drange)
            
            try:
                # Basic SSIM with default heuristics matching oracle_verify
                win_size = min(7, min(x_true.shape))
                if win_size % 2 == 0: win_size -= 1
                if win_size < 3: win_size = None
                
                kwargs = {"data_range": drange}
                if win_size: kwargs["win_size"] = win_size
                
                channel_axis = None
                if x_true.ndim == 3 and x_true.shape[2] <= 4: channel_axis = 2
                elif x_true.ndim == 3 and x_true.shape[0] <= 4: channel_axis = 0
                if channel_axis is not None: kwargs["channel_axis"] = channel_axis
                
                ssim = ssim_func(x_true, x_recon, **kwargs)
            except:
                ssim = -1

            if psnr > 20.0: # Threshold
                return {"passed": True, "psnr": psnr, "ssim": ssim}
            else:
                return {"passed": False, "reason": f"PSNR too low: {psnr:.2f} < 20.0"}
                
        except Exception as e:
            return {"passed": False, "reason": f"Exception: {e}"}

    # =========================================================================
    # Scenario 2: Paper Pipeline
    # =========================================================================

    def _run_paper_pipeline(self, paper_content, logger):
        res = llm_client.call(
            "You are a Scientific Researcher.",
            prompts.PROMPT_EXTRACT_FROM_PAPER.format(paper_content=paper_content),
            model=self.model_name,
            json_mode=True
        )
        extracted = res.get("content", {})
        
        task_desc = f"""# Task Description: {extracted.get('problem_definition')}

## Overview
{extracted.get('problem_definition')}

## Input Specifications
- **Physical Meaning**: {extracted.get('input_spec', {}).get('physical_meaning')}
- **Modality**: {extracted.get('input_spec', {}).get('modality')}
- **Dimensions**: {extracted.get('input_spec', {}).get('dimensions')}

## Output Specifications
- **Physical Meaning**: {extracted.get('output_spec', {}).get('physical_meaning')}
- **Properties**: {extracted.get('output_spec', {}).get('properties')}

## Constraints
{json.dumps(extracted.get('constraints'), indent=2)}

## Evaluation Metrics
{json.dumps(extracted.get('evaluation_metrics'), indent=2)}
"""
        return task_desc

    # =========================================================================
    # Common Artifact Generation
    # =========================================================================

    def _generate_task_description(self, code: str, logger: RunLogger = None) -> str:
        res = llm_client.call(
            "You are a Physics Professor.",
            prompts.PROMPT_GEN_TASK.format(code=code),
            model=self.model_name
        )
        if logger:
            logger.log_trace("phase0", "task_desc_thinking", res.get("thinking", ""))
        return res.get("content", "")

    def _generate_golden_plan(self, code: str, logger: RunLogger = None) -> dict:
        res = llm_client.call(
            "You are a JSON Generator.",
            prompts.PROMPT_GEN_GOLD.format(code=code),
            model=self.model_name,
            json_mode=True
        )
        if logger:
            logger.log_trace("phase0", "golden_plan_thinking", res.get("thinking", ""))
        return res.get("content", {})
    
    # generate_oracle, _get_oracle_script_content REMOVED

