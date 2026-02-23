import os
import sys
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

def collect_sandbox_inputs(
    sandbox_path,
    exclude_extensions=None,
    exclude_dirs=None,
    include_hidden=False
    ):
    """
    智能收集 sandbox 中的输入文件：
    - 默认包含所有文件（避免漏掉未知格式）
    - 仅排除明确属于"工具/环境"的文件
    - 支持用户自定义排除/包含规则
    
    Returns:
        List[str]: 可作为输入数据的相对路径
    """
    # 默认排除：明确是工具/环境的文件
    if exclude_extensions is None:
        exclude_extensions = {
            '.py', '.pyc', '.pyo', '.pyd',  # Python 工具
            '.sh', '.bash', '.zsh', '.ps1',  # 脚本
            '.ipynb',                         # Notebook（通常不是输入数据）
            '.md', '.rst',                    # 文档（可选，根据场景决定）
        }
    
    if exclude_dirs is None:
        exclude_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 
            'node_modules', '.mypy_cache', '.pytest_cache',
            'tools', 'scripts'  # 约定：工具放在特定目录
        }

    input_files = []
    
    for root, dirs, files in os.walk(sandbox_path):
        # 1. 跳过工具/环境目录（原地修改 dirs 避免深入遍历）
        dirs[:] = [d for d in dirs if d not in exclude_dirs 
                   and not d.startswith('.')]
        
        for file in files:
            # 2. 跳过隐藏文件（除非显式要求）
            if not include_hidden and file.startswith('.'):
                continue
                
            # 3. 跳过明确的工具文件
            ext = os.path.splitext(file)[1].lower()
            if ext in exclude_extensions:
                continue
            
            # 4. 特殊豁免：即使在排除目录中，某些文件可能是数据
            # 例如：tools/config.json 可能是输入配置
            # 可通过配置文件精细控制（见下文）
            
            rel_path = os.path.relpath(os.path.join(root, file), sandbox_path)
            input_files.append(rel_path)
    
    return sorted(input_files)

def is_non_empty_json_file(filepath, disallow_empty_dict=False):
    """检查JSON文件是否存在、非空，且（可选）不是空字典{}"""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if disallow_empty_dict and content == {}:
                return False
            return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

def is_non_empty_file(filepath):
    """检查普通文件是否存在且非空"""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

class InverseAgentPipeline:
    def __init__(self, output_dir: str, model_name: str = settings.DEFAULT_MODEL):
        self.output_dir = output_dir
        self.model_name = model_name
        self.logger = RunLogger(output_dir)
        
        # Initialize Agents
        self.dataset_agent = DatasetConstructor()
        self.planner = PlanGenerator(model_name)
        self.critic = PlanCritic()
        self.elo_judge = EloJudge()
        self.coder = CodeGenerator(model_name)
        self.executor = CodeExecutor()
        self.analyzer = ResultAnalyzer()
        self.feedback_gen = FeedbackGenerator()
        self.sandbox_mgr = SandboxManager()
        self.code_scorer = CodeScorer()

    def run(self, task_name: str, gt_code_path: str = None, working_folder: str = None, paper_markdown: str = None):
        success = False  # 初始化成功状态

        self.logger.log("start", f"Starting pipeline for {task_name}")
        
        # --- Config Loading ---
        # Load main config to get python_path
        main_config_path = os.path.join(settings.BASE_DIR, "config/config.yaml")
        task_python_path = sys.executable # Default fallback
        
        if os.path.exists(main_config_path):
            try:
                with open(main_config_path, 'r') as f:
                    main_config = yaml.safe_load(f)
                    tasks_cfg = main_config.get("tasks", [])
                    for t in tasks_cfg:
                        if t.get("name") == task_name:
                            task_python_path = t.get("python_path", sys.executable)
                            print(f"[Pipeline] Found specific Python environment for {task_name}: {task_python_path}")
                            break
            except Exception as e:
                print(f"[Pipeline] Error loading config.yaml: {e}")

        config_path = os.path.join(settings.BASE_DIR, "config/config_task.yaml")
        input_file = None
        expected_output_path = None
        input_abs_path = None
        
        # [MODIFIED] Removed Config Loading for IO paths. 
        # We strictly rely on Phase 0 generation.
        
        self.logger.log("config", {"python_path": task_python_path})
        print(f"[Pipeline] Config loaded: {task_name}, Model: {self.model_name}")

        # --- Phase 0: Dataset Construction ---
        dataset_dir = os.path.join(settings.DATASET_DIR, task_name)
        self.logger.log("phase0", "Generating Dataset...")
        print(f"[Pipeline] Phase 0: Dataset directory is {dataset_dir}")
        
        # Determine GT Path early for phase0 (if in config) - REMOVED
        gt_abs_path = None

        # Check cache validity
        # We require task_io_info.json, golden_plan.json, and task_description.md to be present.
        task_io_info_path = os.path.join(dataset_dir, "task_io_info.json")
        golden_plan_path = os.path.join(dataset_dir, "golden_plan.json")
        task_desc_path = os.path.join(dataset_dir, "task_description.md")

        cache_valid = (
            is_non_empty_json_file(task_io_info_path, disallow_empty_dict=True) and
            is_non_empty_json_file(golden_plan_path) and
            is_non_empty_file(task_desc_path)
        )
        
        inferred_io = {}
        golden_plan = {}
        task_desc = ""

        if cache_valid:
             print("[Pipeline] Phase 0 Cache Hit. Loading artifacts...")
             try:
                 with open(task_io_info_path, 'r') as f: inferred_io = json.load(f)
                 with open(golden_plan_path, 'r') as f: golden_plan = json.load(f)
                 with open(task_desc_path, 'r') as f: task_desc = f.read()
                 self.logger.log("phase0", "Loaded from cache.")
             except Exception as e:
                 print(f"[Pipeline] Error loading cache: {e}. Regenerating...")
                 cache_valid = False
        
        if not cache_valid:
            print("[Pipeline] Cache Miss or Invalid. Running Phase 0 (Oracle Pipeline)...")
            data_artifacts = self.dataset_agent.process_task(
                task_name, 
                gt_code_path=gt_code_path, 
                paper_markdown_path=paper_markdown,
                output_dir=dataset_dir, 
                working_folder=working_folder,
                model_name=self.model_name,
                logger=self.logger,
                python_path=task_python_path
            )
            task_desc = data_artifacts.get("task_description", "")
            golden_plan = data_artifacts.get("golden_plan", {})
            inferred_io = data_artifacts.get("inferred_io", {})
            self.logger.log("phase0", "Generation complete.")
            print("[Pipeline] Phase 0 Generation complete.")

        # Update paths from inferred IO (STRICT MODE)
        if inferred_io:
            gt_abs_path = inferred_io.get("gt_output_path")
            input_abs_path = inferred_io.get("input_path")
            expected_output_path = inferred_io.get("expected_output_filename")
            
            if input_abs_path:
                input_file = os.path.basename(input_abs_path)
            
            print(f"[Pipeline] IO Paths Determined:")
            print(f"  - Input: {input_abs_path}")
            print(f"  - GT Reference: {gt_abs_path}")
            print(f"  - Expected Output Filename: {expected_output_path}")
        else:
            print("[Pipeline] CRITICAL ERROR: No IO info available after Phase 0.")
            raise ValueError("Phase 0 failed to produce task_io_info.json or inferred_io is empty.")

        # Ensure all required paths are present
        if not all([gt_abs_path, input_abs_path, expected_output_path]):
             raise ValueError(f"Missing one or more required paths:\nGT: {gt_abs_path}\nInput: {input_abs_path}\nExpected Output: {expected_output_path}")

        # ========== Fallback REMOVED ==========
        # We strictly require expected_output_path to be set by Phase 0.
        # ======================================


        # Prepare Sandbox
        print(f"[Pipeline] Creating sandbox in {working_folder}...")
        if not os.path.exists(working_folder):
             os.makedirs(working_folder, exist_ok=True)
             
        sandbox_path = self.sandbox_mgr.create_sandbox(working_folder, task_name)
        self.logger.log("sandbox", f"Created at {sandbox_path}")
        print(f"[Pipeline] Sandbox created at {sandbox_path}")
        
        sandbox_files = collect_sandbox_inputs(sandbox_path)
        
        files_str = "\n".join(sandbox_files)
        sandbox_context = f"\n\n[System] Current Sandbox Files:\n{files_str}\n\n[System] Please ensure your code reads the input files from the current directory and writes the output to '{expected_output_path}'."

        try:
            current_phase = "PLANNING"
            current_plan = None
            current_code = None
            feedback_msg = ""
            last_exec_res = None
            last_verify_res = None
            max_iterations = 3
            iteration = 0
            flag_elo = getattr(settings, "ENABLE_ELO", False)
            
            while iteration < max_iterations:
                iteration += 1
                self.logger.log("loop_start", f"Iteration {iteration}, Phase: {current_phase}")
                print(f"[Pipeline] === Iteration {iteration} | Phase: {current_phase} ===")
                
                # --- Phase 1: Planning ---
                if current_phase == "PLANNING":
                    self.logger.log("phase1", "Generating Plan...")
                    print("[Pipeline] Generating Plan...")
                    prompt = task_desc + sandbox_context
                    if feedback_msg:
                        print("[Pipeline] Using Feedback for Planning...")
                        prompt += f"\n\nFEEDBACK FROM PREVIOUS FAILURE:\n{feedback_msg}"
                        
                    current_plan = self.planner.generate_plan(prompt, self.logger)
                    self.logger.log_artifact(f"plan_iter{iteration}", current_plan, "json")
                    
                    # Self-Correction
                    print("[Pipeline] Running Plan Critic...")
                    critic_res = self.critic.review_plan(task_desc, current_plan, self.logger)
                    if not critic_res.get("is_valid", True):
                        print("[Pipeline] Plan Critic found defects. Refining...")
                        self.logger.log("phase1_critic", f"Plan defects: {critic_res.get('defects')}")
                        current_plan = self.planner.generate_plan(
                            f"{task_desc}\n\nCRITIC FEEDBACK: {critic_res.get('suggestion')}", 
                            self.logger
                        )
                        self.logger.log_artifact(f"plan_iter{iteration}_refined", current_plan, "json")
                    else:
                        print("[Pipeline] Plan Critic approved.")
                    
                    current_phase = "CODING"
                
                # --- Phase 2: Coding ---
                if current_phase == "CODING":
                    self.logger.log("phase2", "Generating Code...")
                    print("[Pipeline] Generating/Refining Code...")
                    if not current_plan:
                        self.logger.log("error", "No plan available for coding phase. Reverting to planning.")
                        print("[Pipeline] Error: No plan available. Reverting to PLANNING.")
                        current_phase = "PLANNING"
                        continue

                    if feedback_msg and current_code:
                        print("[Pipeline] Using Feedback to refine code.")
                        refine_context = feedback_msg + sandbox_context
                        current_code = self.coder.refine_code(current_plan, current_code, refine_context, self.logger)
                    else:
                        print("[Pipeline] Generating initial code from plan.")
                        task_with_context = task_desc + sandbox_context
                        current_code = self.coder.generate_code(task_with_context, current_plan, sandbox_path, expected_output_path, self.logger)
                    
                    self.logger.log_artifact(f"code_iter{iteration}", current_code, "py")
                    
                    # Execute
                    execution_success = False
                    print("[Pipeline] Executing code...")
                    for attempt in range(2):
                        print(f"[Pipeline]   Attempt {attempt+1}/2...")
                        exec_res = self.executor.execute(sandbox_path, current_code, expected_output_path, self.logger, python_path=task_python_path)
                        self.logger.log_artifact(f"execution_log_iter{iteration}_try{attempt+1}", exec_res.get("logs", ""), "txt")
                        
                        last_exec_res = exec_res
                        if exec_res["success"]:
                            print(f"[Pipeline]   Success!")
                            execution_success = True
                            break
                        else:
                            print(f"[Pipeline]   Failed. Retrying with refinement...")
                            self.logger.log("phase2_retry", exec_res)
                            current_code = self.coder.refine_code(current_plan, current_code, exec_res["logs"], self.logger)
                            self.logger.log_artifact(f"code_iter{iteration}_try{attempt+1}_refined", current_code, "py")
                    
                    if not execution_success:
                        print("[Pipeline] All execution attempts failed.")
                        self.logger.log("fail", "Code execution failed after retries.")
                        feedback_msg = "Code execution failed consistently."
                        print("[Pipeline] Fallback: Reverting to PLANNING due to persistent execution failure.")
                        current_phase = "PLANNING"
                        continue

                    current_phase = "VERIFYING"

                # --- Phase 3: Validation ---
                if current_phase == "VERIFYING":
                    self.logger.log("phase3", "Verifying...")
                    print("[Pipeline] Verifying results...")
                    
                    # Check if GT is available (It might be missing in Paper Scenario)
                    if not gt_abs_path or not os.path.exists(gt_abs_path):
                        print(f"[Pipeline] GT Path invalid or missing: {gt_abs_path}")
                        # In paper scenario, we might not have GT to verify against?
                        # Or we should have generated it in Phase 0 if possible.
                        # If not, we can't verify.
                        if paper_markdown:
                             print("[Pipeline] Paper Pipeline: Skipping verification (No GT).")
                             verify_res = {"verified": True, "reason": "Skipped (Paper Scenario)"}
                        else:
                             verify_res = {
                                "verified": False,
                                "reason": f"Ground truth file not found for task={task_name}: {gt_abs_path}",
                                "logs": ""
                            }
                    else:
                        submission_abs_path = os.path.join(sandbox_path, expected_output_path)
                        # Construct GT Output Path (Reconstruction by GT Code)
                        gt_recon_path = os.path.join(dataset_dir, "gt_output.npy")
                        
                        # Use new verify_task
                        task_res = self.analyzer.verify_task(submission_abs_path, gt_recon_path, gt_abs_path)
                        
                        verify_res = {
                            "verified": task_res["success"],
                            "reason": task_res.get("error") if not task_res["success"] else f"Passed (PSNR Exp={task_res.get('psnr_expected', 0):.2f} >= {task_res.get('threshold', 0):.2f})",
                            "metrics": {
                                "psnr_expected": task_res.get("psnr_expected", -1),
                                "psnr_gt": task_res.get("psnr_gt", -1),
                                "threshold": task_res.get("threshold", -1),
                                "psnr": task_res.get("psnr_expected", -1),
                                "ssim": task_res.get("ssim", -1)
                            },
                            "logs": str(task_res)
                        }
                    
                    last_verify_res = verify_res
                    self.logger.log("phase3_result", verify_res)
                    print(f"[Pipeline] Verification Result: {verify_res.get('verified')} ({verify_res.get('reason')})")
                    
                    if verify_res["verified"]:
                        self.logger.log("success", "Task Solved!")
                        print("[Pipeline] TASK SOLVED!")
                        success = True  # ✅ 关键：标记成功状态
                        break # Exit Loop
                    else:
                        print("[Pipeline] Verification failed. Generating Feedback...")
                        fb = self.feedback_gen.generate_feedback(verify_res, current_plan, current_code, self.logger)
                        error_type = fb.get("error_type", "Type B")
                        feedback_msg = fb.get("feedback_message", str(fb))
                        self.logger.log("feedback", fb)
                        
                        if "Type A" in error_type:
                            self.logger.log("loop", "Routing back to PLANNING (Type A)")
                            print("[Pipeline] Error Type A: Routing back to PLANNING.")
                            current_phase = "PLANNING"
                        else:
                            self.logger.log("loop", "Routing back to CODING (Type B/C)")
                            print("[Pipeline] Error Type B/C: Routing back to CODING.")
                            current_phase = "CODING"
            
            # --- ELO Scoring (Final Step) ---
            elo_res = None
            if current_plan and flag_elo:
                self.logger.log("elo", "Running ELO Judge on Final Plan...")
                print("[Pipeline] Running ELO Judge...")
                elo_res = self.elo_judge.evaluate(task_desc, golden_plan, current_plan, self.logger)
                self.logger.log("elo_score", elo_res)
                self.logger.log_artifact("final_elo_score", elo_res, "json")

            print("[Pipeline] Running Final Code Eval...")
            if gt_code_path and os.path.exists(gt_code_path):
                 code_eval = self.code_scorer.evaluate(gt_code_path, current_code or "")
                 self.logger.log_artifact("final_code_eval", code_eval, "json")
            else:
                 code_eval = {}

            # --- Final Performance Comparison ---
            performance_comparison = {}
            try:
                # 1. Get Agent Metrics from Verification
                agent_metrics = last_verify_res.get("metrics", {}) if last_verify_res else {}
                
                # 2. Get Baseline Metrics from Phase 0
                baseline_metrics = {}
                if inferred_io and inferred_io.get("baseline_metrics"):
                     # baseline_metrics might be {passed: True, psnr: 30.5} or {passed: True, metrics: {psnr: 30.5}}
                     # Check structure from phase0_dataset.py _verify_results
                     # It returns {"passed": True, "psnr": psnr, "ssim": ssim}
                     bm = inferred_io.get("baseline_metrics", {})
                     baseline_metrics = {
                         "psnr": bm.get("psnr", -1),
                         "ssim": bm.get("ssim", -1)
                     }
                
                # 3. Compute Diff
                if agent_metrics and baseline_metrics:
                    performance_comparison = {
                        "agent_metrics": agent_metrics,
                        "gt_baseline_metrics": baseline_metrics,
                        "improvement": {}
                    }
                    
                    # Compare PSNR
                    if "psnr" in agent_metrics and "psnr" in baseline_metrics:
                        a_p = agent_metrics["psnr"]
                        b_p = baseline_metrics["psnr"]
                        if a_p != -1 and b_p != -1:
                            performance_comparison["improvement"]["psnr_diff"] = a_p - b_p
                            
                    # Compare SSIM
                    if "ssim" in agent_metrics and "ssim" in baseline_metrics:
                        a_s = agent_metrics["ssim"]
                        b_s = baseline_metrics["ssim"]
                        if a_s != -1 and b_s != -1:
                            performance_comparison["improvement"]["ssim_diff"] = a_s - b_s
                            
            except Exception as e:
                print(f"[Pipeline] Warning: Failed to compute performance comparison: {e}")

            final_report = {
                "task_name": task_name,
                "verified": bool(last_verify_res and last_verify_res.get("verified")),
                "iterations": iteration,
                "final_plan": current_plan,
                "last_execution": last_exec_res,
                "last_verification": last_verify_res,
                "elo": elo_res,
                "code_eval": code_eval,
                "performance_comparison": performance_comparison,
                "sandbox_path": sandbox_path,
                "dataset_dir": dataset_dir,
            }
            self.logger.log_artifact("final_report", final_report, "json")
            print(f"[Pipeline] Finished. Report saved to {self.output_dir}")
        
        finally:
            pass
            
        return success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--gt_code", required=False, help="Path to Ground Truth Code")
    parser.add_argument("--paper_markdown", required=False, help="Path to Paper Markdown")
    parser.add_argument("--working_folder", required=True)
    parser.add_argument("--output_dir", default="/data/yjh/results")
    parser.add_argument("--model", default=settings.DEFAULT_MODEL)
    args = parser.parse_args()
    
    if not args.gt_code and not args.paper_markdown:
        print("Error: Must provide either --gt_code or --paper_markdown")
        sys.exit(1)

    # Create task-specific output subdirectory
    task_output_dir = os.path.join(args.output_dir, args.task_name)
    
    pipeline = InverseAgentPipeline(task_output_dir, model_name=args.model)
    final_state = pipeline.run(args.task_name, gt_code_path=args.gt_code, working_folder=args.working_folder, paper_markdown=args.paper_markdown)
    print(final_state)
if __name__ == "__main__":
    main()
