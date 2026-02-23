import os
import json
import argparse
import yaml
import shutil
from config import settings
from utils.logger import RunLogger
from utils.sandbox import SandboxManager
from agents.phase0_dataset import DatasetConstructor
from agents.phase1_planner import PlanGenerator, PlanCritic
from agents.elo_scorer import EloJudge
from agents.phase2_coder import CodeGenerator, CodeExecutor
from agents.phase3_evaluator import ResultAnalyzer, FeedbackGenerator
from agents.code_scorer import CodeScorer

class InverseAgentPipelineAblation:
    def __init__(self, output_dir: str, model_name: str = settings.DEFAULT_MODEL):
        self.output_dir = output_dir
        self.model_name = model_name
        self.logger = RunLogger(output_dir)
        
        # Initialize Agents
        self.dataset_agent = DatasetConstructor()
        # Planner and Critic are NOT used in ablation (using golden plan)
        self.coder = CodeGenerator(model_name)
        self.executor = CodeExecutor()
        self.analyzer = ResultAnalyzer()
        self.feedback_gen = FeedbackGenerator()
        self.sandbox_mgr = SandboxManager()
        self.code_scorer = CodeScorer()
        # Elo Judge is also not strictly needed if we don't generate plans, but maybe for final report?
        # Actually, since we use Golden Plan, Elo score would be comparing Golden Plan against Golden Plan?
        # Or maybe we just skip ELO.

    def run(self, task_name: str, gt_code_path: str, working_folder: str):
        self.logger.log("start", f"Starting ABLATION pipeline for {task_name}")
        
        # --- Config Loading ---
        config_path = os.path.join(settings.BASE_DIR, "config/config_task.yaml")
        input_file = "input.dat"
        expected_output_path = "output.npy"
        input_abs_path = None
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                task_io = config_data.get("task_io_paths", {}).get(task_name, {})
                input_file = os.path.basename(task_io.get("input_path", "input.dat"))
                input_abs_path = task_io.get("input_path")
                expected_output_path = os.path.basename(task_io.get("expected_output_path", "output.npy"))
        
        self.logger.log("config", {"input_file": input_file, "expected_output": expected_output_path})
        print(f"[Ablation] Config loaded: {task_name}, Model: {self.model_name}")

        # --- Phase 0: Dataset Construction ---
        dataset_dir = os.path.join(settings.DATASET_DIR, task_name)
        self.logger.log("phase0", "Generating/Loading Dataset...")
        print(f"[Ablation] Phase 0: Dataset directory is {dataset_dir}")
        
        # Determine GT Path early for phase0
        gt_abs_path = None
        if os.path.exists(config_path):
             with open(config_path, 'r') as f: config_data = yaml.safe_load(f)
             gt_abs_path = config_data.get("task_io_paths", {}).get(task_name, {}).get("expected_output_path")

        # Check cache validity
        cache_valid = False
        golden_plan_path = os.path.join(dataset_dir, "golden_plan.json")
        oracle_path = os.path.join(dataset_dir, "oracle_verify.py")
        
        if os.path.exists(golden_plan_path) and os.path.exists(oracle_path):
            try:
                with open(golden_plan_path, 'r') as f:
                    gp = json.load(f)
                    if gp and isinstance(gp, dict): # Basic check
                        cache_valid = True
            except:
                pass
        
        if cache_valid:
             print("[Ablation] Loading Golden Plan from cache...")
             with open(os.path.join(dataset_dir, "task_description.md")) as f: task_desc = f.read()
             with open(golden_plan_path) as f: golden_plan = json.load(f)
             # oracle_path already defined
             self.logger.log("phase0", "Loaded from cache.")
        else:
            if os.path.exists(golden_plan_path):
                print("[Ablation] Cache exists but is invalid/incomplete (missing oracle). Regenerating...")
            else:
                print("[Ablation] Cache miss. Generating Golden Plan (Phase 0)...")
            
            data_artifacts = self.dataset_agent.process_task(
                task_name, 
                gt_code_path, 
                dataset_dir, 
                input_path=input_abs_path,
                gt_output_path=gt_abs_path,
                model_name=self.model_name,
                logger=self.logger
            )
            task_desc = data_artifacts["task_description"]
            golden_plan = data_artifacts["golden_plan"]
            oracle_path = data_artifacts["oracle_path"]
            self.logger.log("phase0", "Generation complete.")
            print("[Ablation] Phase 0 Generation complete.")

        # Prepare Sandbox
        print(f"[Ablation] Creating sandbox in {working_folder}...")
        sandbox_path = self.sandbox_mgr.create_sandbox(working_folder, task_name)
        self.logger.log("sandbox", f"Created at {sandbox_path}")
        print(f"[Ablation] Sandbox created at {sandbox_path}")
        
        if input_abs_path and os.path.exists(input_abs_path):
            try:
                shutil.copy2(input_abs_path, os.path.join(sandbox_path, input_file))
                self.logger.log("sandbox_io", {"copied_input": os.path.join(sandbox_path, input_file)})
                print(f"[Ablation] Copied input file to sandbox.")
            except Exception as e:
                self.logger.log("sandbox_io_error", str(e))
                print(f"[Ablation] Error copying input: {e}")

        try:
            # --- ABLATION MODIFICATION: SKIP PLANNING ---
            # Directly use Golden Plan
            current_phase = "CODING"
            current_plan = golden_plan
            current_code = None
            feedback_msg = ""
            last_exec_res = None
            last_verify_res = None
            max_iterations = 3
            iteration = 0
            
            self.logger.log("ablation", "Skipping Phase 1 (Planning). Using Golden Plan.")
            self.logger.log_artifact("used_plan", current_plan, "json")
            print("[Ablation] Phase 1 Skipped. Using Golden Plan.")

            while iteration < max_iterations:
                iteration += 1
                self.logger.log("loop_start", f"Iteration {iteration}, Phase: {current_phase}")
                print(f"[Ablation] === Iteration {iteration} | Phase: {current_phase} ===")
                
                # --- Phase 2: Coding ---
                if current_phase == "CODING":
                    self.logger.log("phase2", "Generating Code...")
                    print("[Ablation] Generating/Refining Code...")
                    
                    # If coming from feedback, we might just refine
                    if feedback_msg and current_code:
                        print("[Ablation] Using Feedback to refine code.")
                        current_code = self.coder.refine_code(current_plan, current_code, feedback_msg, self.logger)
                    else:
                        print("[Ablation] Generating initial code.")
                        current_code = self.coder.generate_code(task_desc, current_plan, sandbox_path, input_file, expected_output_path, self.logger)
                    
                    self.logger.log_artifact(f"code_iter{iteration}", current_code, "py")
                    
                    # Execute
                    execution_success = False
                    print("[Ablation] Executing code...")
                    for attempt in range(3):
                        print(f"[Ablation]   Attempt {attempt+1}/3...")
                        exec_res = self.executor.execute(sandbox_path, current_code, expected_output_path, self.logger)
                        # Log execution result for EACH attempt
                        self.logger.log_artifact(f"execution_log_iter{iteration}_try{attempt+1}", exec_res.get("logs", ""), "txt")

                        last_exec_res = exec_res
                        if exec_res["success"]:
                            print(f"[Ablation]   Success!")
                            execution_success = True
                            break
                        else:
                            print(f"[Ablation]   Failed. Retrying with refinement...")
                            self.logger.log("phase2_retry", exec_res)
                            current_code = self.coder.refine_code(current_plan, current_code, exec_res["logs"], self.logger)
                            # Also save refined code
                            self.logger.log_artifact(f"code_iter{iteration}_try{attempt+1}_refined", current_code, "py")
                    
                    if not execution_success:
                        print("[Ablation] All execution attempts failed.")
                        self.logger.log("fail", "Code execution failed after retries.")
                        feedback_msg = "Code execution failed consistently."
                        # --- ABLATION MODIFICATION: NO BACKTRACKING ---
                        # In ablation, if code fails, we do NOT go back to planning.
                        # We just try next iteration (which will just be refining code again if we stay in CODING?)
                        # Or maybe we should stop?
                        # The user said: "如果任务报错也不可能再返回到重新planning的阶段"
                        # So we stay in CODING loop, trying to fix it with feedback.
                        # But wait, if execution failed, we already refined 3 times inside the loop.
                        # So 'feedback_msg' is set, and next iteration of outer loop will trigger refine_code again.
                        continue

                    current_phase = "VERIFYING"

                # --- Phase 3: Validation ---
                if current_phase == "VERIFYING":
                    self.logger.log("phase3", "Verifying...")
                    print("[Ablation] Verifying results...")
                    
                    if not gt_abs_path or not os.path.exists(gt_abs_path):
                        print(f"[Ablation] GT Path invalid: {gt_abs_path}")
                        verify_res = {
                            "verified": False,
                            "reason": f"Ground truth file not found for task={task_name}: {gt_abs_path}",
                            "logs": ""
                        }
                    else:
                        submission_abs_path = os.path.join(sandbox_path, expected_output_path)
                        verify_res = self.analyzer.verify(sandbox_path, oracle_path, submission_abs_path, gt_abs_path, self.logger)
                    last_verify_res = verify_res
                    self.logger.log("phase3_result", verify_res)
                    print(f"[Ablation] Verification Result: {verify_res.get('verified')} ({verify_res.get('reason')})")
                    
                    if verify_res["verified"]:
                        self.logger.log("success", "Task Solved!")
                        print("[Ablation] TASK SOLVED!")
                        break # Exit Loop
                    else:
                        # Generate Feedback
                        print("[Ablation] Verification failed. Generating Feedback...")
                        fb = self.feedback_gen.generate_feedback(verify_res, current_plan, current_code, self.logger)
                        # error_type = fb.get("error_type", "Type B") # Ignore error type in ablation
                        feedback_msg = fb.get("feedback_message", str(fb))
                        self.logger.log("feedback", fb)
                        
                        # --- ABLATION MODIFICATION: FORCE CODING ---
                        self.logger.log("loop", "Routing back to CODING (Ablation: No Planning Backtrack)")
                        current_phase = "CODING"
            
            # --- ELO Scoring Skipped in Ablation ---
            # Because we didn't generate a plan, we used the golden plan.
            
            print("[Ablation] Running Final Code Eval...")
            code_eval = self.code_scorer.evaluate(gt_code_path, current_code or "")
            self.logger.log_artifact("final_code_eval", code_eval, "json")

            final_report = {
                "task_name": task_name,
                "verified": bool(last_verify_res and last_verify_res.get("verified")),
                "iterations": iteration,
                "final_plan": current_plan,
                "last_execution": last_exec_res,
                "last_verification": last_verify_res,
                "code_eval": code_eval,
                "sandbox_path": sandbox_path,
                "dataset_dir": dataset_dir,
                "ablation_mode": True
            }
            self.logger.log_artifact("final_report", final_report, "json")
            print(f"[Ablation] Finished. Report saved to {self.output_dir}")
        
        finally:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--gt_code", required=True)
    parser.add_argument("--working_folder", required=True)
    parser.add_argument("--output_dir", default="results_ablation")
    parser.add_argument("--model", default=settings.DEFAULT_MODEL)
    args = parser.parse_args()
    
    # Create task-specific output subdirectory
    task_output_dir = os.path.join(args.output_dir, args.task_name)
    
    pipeline = InverseAgentPipelineAblation(task_output_dir, model_name=args.model)
    pipeline.run(args.task_name, args.gt_code, args.working_folder)

if __name__ == "__main__":
    main()
