import os
import sys
import shutil
import subprocess
import json
import time
import ast
from typing import List, Dict, Tuple, Any, Optional
import re, hashlib, datetime

from workflow_base import InverseProblemBase

class InverseProblemWorkflow(InverseProblemBase):
    def __init__(self, task_name: str, task_desc: str, gt_code_path: str, python_path: str, working_dir: str, client: Any, model_name: str, root_output_dir: str = "/data/yjh/end_sandbox", skill_manager: Any = None):
        super().__init__(task_name, task_desc, gt_code_path, python_path, working_dir, client, model_name, root_output_dir, skill_manager)

    def run(self):
        self._setup_sandbox()
        self._phase_0_preparation()
        
        # ------------------------------------------------------------------
        # Skill Injection (Moved to per-agent context building)
        # ------------------------------------------------------------------
        if self.skill_manager:
             self._log(">>> [Knowledge System] Hierarchical Memory Activated. Skills will be injected per-agent.")

        # 初始状态
        feedback = None
        ticket = "Planner" # 默认从 Plan 开始
        # action_sequence removed in favor of self.trajectory_steps
        
        while self.retry_count < self.max_retries:
            iter_id = self.retry_count + 1
            self._log(f"\n{'='*20} Iteration {iter_id} (Ticket: {ticket}) {'='*20}")
            

            # =========================================================
            # Stage 1: Planning (with Critic Loop)
            # =========================================================
            if ticket == "Planner":
                self._log(">>> [Agent] Planner...")
                # action_sequence.append(f"Iteration {iter_id}: Planner") removed

                # plan_ctx = {'task_desc': self.task_desc, 'feedback': feedback}
                plan_ctx = self._build_context_with_memory(
                    base_context={'task_desc': self.task_desc, 'feedback': feedback},
                    agent_role="Planner",
                    current_ticket="Planner"
                )
                
                draft_plan = self.planner.generate(plan_ctx)
                
                # --- Critic Loop ---
                # Planner 阶段的 Critic Loop（现有代码只需微调解析方式）
                critic_valid = False
                for critic_retry in range(3):
                    critic_resp_str = self.critic.generate({
                        'task_desc': self.task_desc,
                        'plan': draft_plan
                    })
                    
                    try:
                        critic_resp = json.loads(critic_resp_str)  # 现在 100% 安全
                        if critic_resp["decision"] == "PASS":
                            critic_valid = True
                            break
                        else:
                            # 使用结构化字段构建反馈
                            feedback = f"Critic rejected: {critic_resp['reason']}"
                            if critic_resp.get("suggestion"):
                                feedback += f" | Fix: {critic_resp['suggestion']}"
                            draft_plan = self.planner.generate({'task_desc': self.task_desc, 'feedback': feedback})
                    except Exception as e:
                        # 理论上不会触发（因 CriticAgent 已保证 JSON 合法性）
                        self._log(f"[System] Critic JSON parse failed (unexpected): {e}")
                        break

                if not critic_valid:
                    self._log("[System] Critic rejected plan after max retries. Proceeding with caution...")

                self.current_plan = draft_plan
                
                # ✅ UPDATED RECORD STEP
                self._record_step(iter_id, "Planner", input_data=plan_ctx, output_data={"plan": self.current_plan})
                
                self._save_artifact(f"iter_{iter_id}_plan.md", self.current_plan)
                ticket = "Architect" # Pass baton

            # =========================================================
            # Stage 2: Architecture
            # =========================================================
            if ticket == "Architect":
                self._log(">>> [Agent] Architect...")
                # action_sequence.append(f"Iteration {iter_id}: Architect") removed

                # Custom Retrieval Query for Architect: Task + Plan
                architect_query = f"{self.task_desc}\n\nPLAN CONTEXT:\n{self.current_plan}"
                
                arch_ctx = self._build_context_with_memory(
                    base_context={
                        'task_desc': self.task_desc, # Still pass original for prompt
                        'plan': self.current_plan,
                        'previous_skeleton': self.current_skeleton if self.current_skeleton.strip() else None,
                        'feedback': feedback.get('feedback') if isinstance(feedback, dict) and feedback.get('ticket') == 'Architect' else None
                    },
                    agent_role="Architect",
                    current_ticket="Architect",
                    retrieval_query=architect_query # Pass custom query
                )

                # Architect 生成包含 Class 定义和 pass 方法的骨架
                for attempt in range(3):
                    arch_resp = self.architect.generate(arch_ctx)
                    
                    # Try extracting JSON first (legacy support)
                    extracted_json = self._extract_json(arch_resp)
                    try:
                        arch_json = json.loads(extracted_json)
                        if isinstance(arch_json, dict) and 'skeleton_code' in arch_json:
                             self.current_skeleton = arch_json['skeleton_code']
                        else:
                             # Maybe it's just the code string
                             self.current_skeleton = arch_resp
                    except:
                        # Fallback: assume response is code (with markdown)
                        self.current_skeleton = arch_resp
                    
                    # Ensure skeleton is clean code (Python)
                    self.current_skeleton = self._extract_python(self.current_skeleton)
                    
                    # Validate
                    is_valid, err_msg = self._validate_skeleton(self.current_skeleton)
                    if is_valid:
                        self._log("  [System] Skeleton validated successfully.")
                        break
                    else:
                        self._log(f"  [System] Skeleton validation failed (Attempt {attempt+1}): {err_msg}")
                        # Debug: self._log first few lines of invalid content
                        debug_snippet = self.current_skeleton[:300].replace('\n', '\\n')
                        self._log(f"  [Debug] Invalid Content Start: {debug_snippet}...")
                        arch_ctx['feedback'] = f"Previous skeleton was invalid. Error: {err_msg}\nPlease output ONLY valid Python code containing `class Config` and `class InverseSolver`."
                else:
                    raise RuntimeError("Failed to generate valid skeleton after 3 attempts.")

                # [Hard Logic] Parse function list using AST (No Hallucination)
                self.function_list = self._parse_functions_from_skeleton(self.current_skeleton)
                print(f"[System] Extracted Function List: {self.function_list}")
                
                # Initialize current_code with the skeleton so we start with a valid structure
                self.current_code = self.current_skeleton

                # ✅ UPDATED RECORD STEP
                self._record_step(iter_id, "Architect", input_data=arch_ctx, output_data={"skeleton": self.current_skeleton})
                
                self._save_artifact(f"iter_{iter_id}_skeleton.py", self.current_skeleton)
                ticket = "Coder" # Pass baton

            # =========================================================
            # Stage 3: Coding (The Edit Loop)
            # =========================================================
            if ticket == "Coder":
                self._log(">>> [Agent] Coder...")
                # action_sequence.append(f"Iteration {iter_id}: Coder") removed
                
                # 1. 确定生成任务列表 (Task List)
                # 默认是全量生成 (Fresh Start)
                coding_tasks = [
                    ('imports', None),
                    *[( 'function', func_name ) for func_name in self.function_list],
                    ('main_block', None)
                ]
                
                # 2. 检查是否是“修补模式”
                # 如果有 feedback，且 feedback 指向了特定的函数或模块
                is_patch_mode = False
                if isinstance(feedback, dict) and feedback.get('ticket_assigned_to') == 'Coder':
                    # 假设 Judge 的 feedback 中包含 'fix_target' 字段
                    target = feedback.get('fix_target') 
                    
                    # Normalizing target: Handle "InverseSolver.solve method" -> "solve"
                    if target:
                        # Common patterns: "InverseSolver.solve", "solve method", "function solve"
                        match = re.search(r'\b(solve|__init__|forward|[_a-zA-Z0-9]+)\b', target)
                        if match:
                             # If the extracted name is in our function list (or is main/imports), use it
                             candidate = match.group(1)
                             if candidate in self.function_list or candidate in ['imports', 'main_block']:
                                 target = candidate
                             # Special case: "InverseSolver.solve" -> "solve"
                             elif "." in target:
                                 parts = target.split('.')
                                 if parts[-1] in self.function_list:
                                     target = parts[-1]
                    
                    # 简单的推断逻辑：如果 analysis 里提到了具体的函数名
                    if not target:
                        for func in self.function_list:
                            if func in feedback.get('analysis', ''):
                                target = func
                                break
                    
                    if target:
                        self._log(f"  [System] Smart Patch Mode Activated. Target: {target}")
                        is_patch_mode = True
                        # 只保留需要修改的目标
                        if target == 'imports':
                            coding_tasks = [('imports', None)]
                        elif target == 'main_block' or target == 'main':
                            coding_tasks = [('main_block', None)]
                        elif target in self.function_list:
                            coding_tasks = [('function', target)]
                        else:
                            # If ambiguous, fall back to full rebuild but maybe hint the Coder?
                            # For now, let's try to be smart about "solve method and main execution block"
                            # If multiple targets mentioned, we might need a list.
                            # BUT current architecture supports list of tasks.
                            
                            # Improved: Check for multiple targets in string
                            detected_targets = []
                            if 'main' in str(target).lower() or 'execution' in str(target).lower():
                                detected_targets.append(('main_block', None))
                            
                            for func in self.function_list:
                                if func in str(target):
                                    detected_targets.append(('function', func))
                            
                            if detected_targets:
                                coding_tasks = detected_targets
                                self._log(f"  [System] Multi-target patch detected: {coding_tasks}")
                            else:
                                # 全量修复：按依赖顺序构建任务队列
                                self._log(f"  [System] Target '{target}' ambiguous or unknown. Performing full rebuild...")
                                coding_tasks = [
                                    ('imports', None),                                      # 1. 基础依赖
                                    *[
                                        ('function', func_name) 
                                        for func_name in self.function_list 
                                        if func_name not in {'imports', 'main_block'}  # 避免重复
                                    ],                                                      # 3. 所有业务函数
                                    ('main_block', None)                                    # 4. 入口逻辑
                                ]
                                self._log(f"  [System] Full rebuild tasks: {len(coding_tasks)} items")

                # 3. 执行 Coding Loop
                # Reset to skeleton ONLY if we are doing a full rebuild (Full Fresh Start)
                if not is_patch_mode:
                    # Logic: If this is a RETRY (retry_count > 0) AND we are not patching,
                    # it means we want to discard the previous failed attempt and start clean from skeleton.
                    # BUT if this is the FIRST run (retry_count == 0), self.current_code is ALREADY skeleton (set after Architect).
                    # So we only need to reset if retry_count > 0.
                    # Actually, self.current_code might hold "failed code" from previous iteration.
                    # So yes, reset it.
                    print("  [System] Full rebuild mode: Resetting code to skeleton state.")
                    self.current_code = self.current_skeleton

                for task_type, task_name in coding_tasks:
                    print(f"  [Coder] Processing {task_type}" + (f": {task_name}" if task_name else "") + "...")
                    
                    # --- Custom Retrieval Query for Coder ---
                    # Focus on Function Signature + Task Name (Constraint) + Plan Summary
                    
                    func_signature = None
                    if task_type == 'function' and self.current_skeleton:
                         # Use robust AST-based extraction
                         func_signature = self._extract_function_signature(self.current_skeleton, task_name)
                    
                    # Extract Plan Summary (First 500 chars)
                    plan_summary = self.current_plan[:500] if self.current_plan else "No Plan Available"
                    
                    if func_signature:
                        # Use precise signature + docstring + Plan
                        coder_query = f"Python Implementation for:\n{func_signature}\n\nContext: {self.task_name}\nPlan: {plan_summary}"
                    else:
                        # Fallback
                        coder_query = f"Python Implementation for function: {task_name if task_name else 'Global Scope'}\nContext: {self.task_name}\nPlan: {plan_summary}"
                    
                    if is_patch_mode and feedback.get('analysis'):
                         coder_query += f"\nError Analysis: {feedback.get('analysis')}"
                    
                    ctx = {
                        'target_type': task_type,
                        'skeleton_code': self.current_skeleton,
                        'current_full_code': self.current_code,      # ✅ 上一轮完整实现（自身历史）
                        'plan': self.current_plan,
                        'task_desc': self.task_desc,                 # ✅ Pass task_desc (with skills) to Coder
                        'package_list': self.package_list,
                        'feedback': feedback.get('feedback') if is_patch_mode and isinstance(feedback, dict) else None,
                        'analysis': feedback.get('analysis') if is_patch_mode and isinstance(feedback, dict) else None, # ✅ Inject Analysis
                        'fix_target': target if is_patch_mode else None
                    }
                    
                    if task_type == 'function':
                        ctx['target_function'] = task_name
                    
                    # ====== 关键增强：为Coder注入历史失败 ======
                    ctx = self._build_context_with_memory(
                        base_context=ctx,
                        agent_role="Coder",
                        current_ticket="Coder",
                        retrieval_query=coder_query # Pass custom query
                    )

                    # 调用 Coder Agent
                    # implement_and_merge 内部会自动处理 EDIT 逻辑，保留其他代码不变
                    self.current_code = self.coder.implement_and_merge(ctx)
                    
                    # ✅ UPDATED RECORD STEP
                    self._record_step(iter_id, "Coder", input_data=ctx, output_data={
                        "task_type": task_type,
                        "target": task_name,
                        "code": self.current_code
                    })

                    # 增量保存: 每完成一个任务，就更新 solver.py，允许用户实时查看进度
                    self.current_code = self._extract_python(self.current_code)
                    solver_path = os.path.join(self.sandbox_dir, "solver.py")
                    with open(solver_path, "w") as f:
                        f.write(self.current_code)

                # 4. 保存文件 (Final Save after loop)
                self.current_code = self._extract_python(self.current_code)
                solver_path = os.path.join(self.sandbox_dir, "solver.py")
                with open(solver_path, "w") as f:
                    f.write(self.current_code)

                # --- Syntax Check Loop ---
                self._log("  [System] Checking Syntax...")
                # After generating code and saving to solver.py...

                # === SYNTAX GUARDRAIL: Internal loop until clean ===
                syntax_retry = 0
                MAX_SYNTAX_RETRY = 5
                while syntax_retry < MAX_SYNTAX_RETRY:
                    syn_ok, _, syn_err = self._run_cmd("solver.py", check_syntax_only=True)
                    if syn_ok:
                        self._log(f"  ✅ Syntax check passed (Attempt {syntax_retry+1}).")
                        break
                    
                    self._log(f"  ❌ Syntax Error (Attempt {syntax_retry+1}/{MAX_SYNTAX_RETRY})")
                    
                    # FULL REWRITE - not incremental patch
                    ctx = {
                        'target_type': 'full_rewrite',
                        'skeleton_code': self.current_skeleton,
                        'plan': self.current_plan,
                        'feedback': f"SYNTAX ERROR (Attempt {syntax_retry+1}):\n{syn_err}"
                    }
                    ctx = self._build_context_with_memory(ctx, "Coder", "Coder")
                    self.current_code = self.coder.implement_and_merge(ctx)
                    
                    # ✅ UPDATED RECORD STEP
                    self._record_step(iter_id, "Coder", input_data=ctx, output_data={
                        "task_type": "full_rewrite_syntax",
                        "error": syn_err,
                        "code": self.current_code
                    })

                    # Rewrite file for next check
                    self.current_code = self._extract_python(self.current_code)
                    with open(os.path.join(self.sandbox_dir, "solver.py"), "w") as f:
                        f.write(self.current_code)
                    
                    syntax_retry += 1
                else:
                    # Exhausted retries - escalate to failure history
                    failure_record = {
                        "iteration": self.retry_count + 1,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ticket_assigned_to": "Coder",
                        "fix_target": None,
                        "analysis": f"Persistent syntax errors after {MAX_SYNTAX_RETRY} retries",
                        "evidence": syn_err[-500:],
                        "feedback": "Full rewrite required with strict syntax validation"
                    }
                    self.failure_history.append(failure_record)
                    raise RuntimeError(f"Syntax errors persist after {MAX_SYNTAX_RETRY} internal retries. Aborting iteration.")
                
                
                # 如果语法没问题，进入运行阶段
                self._save_artifact(f"iter_{iter_id}_solver.py", self.current_code)
                ticket = "Execution"

            # =========================================================
            # Stage 4: Execution & Judgment
            # =========================================================
            if ticket == "Execution":
                self._log(">>> [System] Executing...")
                # action_sequence.append(f"Iteration {iter_id}: Execution") removed
                
                # Cleanup stale output to prevent false positives in evaluation
                output_path = "output.npy" # relative to sandbox
                if os.path.exists(os.path.join(self.sandbox_dir, output_path)):
                    try:
                        os.remove(os.path.join(self.sandbox_dir, output_path))
                        self._log("  [System] Removed stale output.npy before execution.")
                    except OSError as e:
                        self._log(f"  [Warning] Failed to remove stale output.npy: {e}")

                success, stdout, stderr = self._run_cmd("solver.py", timeout=600)
                
                logs = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                self._save_artifact(f"iter_{iter_id}_exec_log.txt", logs)
                
                # Metrics Calculation
                metrics = None
                eval_success = False

                if success:
                    self._log("  [System] Execution Success. Running Evaluation...")
                    # 运行 eval_script.py output.npy
                    e_ok, e_out, e_err = self._run_cmd("eval_script.py", args=["output.npy"])
                    self._log(f"  [Eval] Return Code: {'Success' if e_ok else 'Failure'}")
                    self._log(f"  [Eval] STDOUT:\n{e_out}\n  [Eval] STDERR:\n{e_err}")
                    if e_ok:
                        try:
                            metrics = json.loads(e_out)
                            eval_success = True
                            self._log(f"  [Metrics] {metrics}")
                        except:
                            logs += f"\n\n[Eval Error] JSON Parse failed: {e_out}"
                    else:
                        logs += f"\n\n[Eval Error] Script failed: {e_err}"
                else:
                    self._log(f"  [System] Execution Failed. Skipping Evaluation. Error: {stderr}")

                # ✅ UPDATED RECORD STEP
                self._record_step(iter_id, "Execution", input_data="solver.py", output_data={
                    "success": success,
                    "eval_success": eval_success,
                    "metrics": metrics,
                    "stdout": stdout,
                    "stderr": stderr
                })

                if eval_success and metrics:
                    curr_psnr = metrics.get('psnr', 0)
                    
                    # 动态阈值计算：
                    # 1. 基础要求：Baseline 的 80%
                    # 2. 保底要求：至少 20.0 dB (完全可实用的效果)
                    # 取二者中的较大值，确保在 Baseline 极低时仍有高质量要求
                    base_threshold = self.baseline_metrics.get('psnr', 0) * 0.8
                    min_guaranteed_threshold = 20.0 
                    
                    threshold = max(base_threshold, min_guaranteed_threshold)
                    
                    self._log(f"  [Eval] Threshold Logic: max({base_threshold:.2f}, {min_guaranteed_threshold}) = {threshold:.2f}")

                    if curr_psnr >= threshold:
                        self._log(f"🎉 SUCCESS! PSNR {metrics['psnr']} >= {threshold}")
                        self._save_snapshot(self.retry_count + 1, "final_success", {
                            "metrics": metrics,
                            "threshold": threshold,
                            "solver_code_path": f"iter_{self.retry_count+1:02d}_solver.py"
                        })
                        
                        # --- SKILL DISTILLATION (SUCCESS) ---
                        if self.skill_manager:
                            try:
                                # Update credit scores for used knowledge
                                self.skill_manager.update_scores(list(self.used_knowledge_ids), success=True)
                                
                                trajectory = {
                                    "exp_id": self.exp_id,
                                    "task_name": self.task_name,
                                    "task_desc": self.task_desc,
                                    "domain": "General", # Default
                                    "difficulty": "Unknown",
                                    "outcome": "success",
                                    "quality_score": metrics.get('psnr', 0.0) if metrics else 0.0,
                                    "steps": self.trajectory_steps,
                                    "used_knowledge_ids": list(self.used_knowledge_ids),
                                    "retrieval_key": f"Trajectory: {self.task_name} (Success)",
                                    "timestamp": int(time.time()),
                                    "final_plan": self.current_plan,
                                    "final_skeleton": self.current_skeleton,
                                    "final_code": self.current_code,
                                    "final_reward": metrics,
                                    "key_logs": logs[-2000:]
                                }
                                # Store the stats
                                self.distillation_stats = self.skill_manager.distill_and_store(trajectory)
                            except Exception as e:
                                self._log(f"  ⚠️ Skill distillation failed: {e}")
                        
                        # Generate final report
                        self.generate_knowledge_report(success=True)
                        return True  # EXIT WORKFLOW - No Judge needed
                    else:
                        self._log(f"⚠️ Metrics below threshold: PSNR {metrics['psnr']} < {threshold}")
                        
                    
                # --- Judge Agent ---
                self._log("\n>>> [Agent] Judge Analyzing Failure Root Cause...")
                
                # Use clean task desc (no global skills injected)
                clean_task_desc = self.task_desc

                judge_base_ctx = {
                    'task_desc': clean_task_desc,
                    'logs': logs[-1000:], 
                    'metrics': metrics,
                    'baseline_metrics': self.baseline_metrics,
                    'current_code_snippet': self.current_code 
                }
                
                # Custom Retrieval Query for Judge: Task + Error Logs + Low Metrics
                # Focus on "Diagnosis" skills
                judge_query = f"{clean_task_desc}\n\nEXECUTION LOGS (Error Focus):\n{stderr[-500:]}"
                if metrics:
                    judge_query += f"\nMETRICS: {metrics}"
                
                # Inject Knowledge (Core & Experience) using the standard pipeline
                judge_ctx = self._build_context_with_memory(
                    base_context=judge_base_ctx,
                    agent_role="Judge",
                    current_ticket="Judge",
                    retrieval_query=judge_query # Pass custom query
                )
                
                judgment = self.judge.generate(judge_ctx)
                self._save_artifact(f"iter_{iter_id}_judge.json", judgment)
                # 解析 Judge 的 JSON 输出
                try:
                    judgment = self._extract_json(judgment)
                    result = json.loads(judgment)
                    # Enforce evidence field (critical for traceability)
                    if 'evidence' not in result:
                        result['evidence'] = 'MISSING_EVIDENCE_FALLBACK'
                        result['analysis'] = '[SYSTEM OVERRIDE] Judge omitted evidence field. Defaulting to Coder.'
                        result['ticket_assigned_to'] = 'Coder'
                    
                    self._log(f"  [Judge] Ticket: {result['ticket_assigned_to']} | Analysis: {result.get('analysis', 'N/A')}")
                    
                    # ✅ SPECIAL RETRIEVAL KEY GENERATION FOR JUDGE
                    # retrieval_key = f"{error_type}: {execution_summary} in {code_context}"
                    error_type = result.get('ticket_assigned_to', 'Unknown_Error')
                    # Try to find specific error type in analysis (heuristic)
                    if "NaN" in result.get('analysis', ''): error_type = "NaN_Error"
                    elif "Shape" in result.get('analysis', ''): error_type = "Shape_Mismatch"
                    
                    execution_summary = result.get('evidence', 'No evidence')[:50]
                    code_context = result.get('fix_target', 'unknown_context')
                    
                    judge_retrieval_key = f"{error_type}: {execution_summary} in {code_context}"
                    
                    # Compact Output Structure
                    judge_output = {
                        "retrieval_key": judge_retrieval_key,
                        "error_type": error_type,
                        "error_category": result.get('ticket_assigned_to', 'General'),
                        "execution_summary": result.get('evidence', ''),
                        "judgement_summary": result.get('analysis', '')[:200],
                        "outcome": "pending",
                        # Full refs
                        "full_judgement_analysis": result.get('analysis', ''),
                        "ticket": result.get('ticket_assigned_to'),
                        "fix_target": result.get('fix_target'),
                        "feedback": result.get('feedback')
                    }

                    # ✅ UPDATED RECORD STEP
                    self._record_step(iter_id, "Judge", input_data=judge_ctx, output_data=judge_output, retrieval_key=judge_retrieval_key)

                    # Record failure history BEFORE snapshot
                    failure_record = {
                        "iteration": self.retry_count + 1,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ticket_assigned_to": result['ticket_assigned_to'],
                        "fix_target": result.get('fix_target'),
                        "analysis": result.get('analysis', ''),
                        "evidence": result.get('evidence', ''),
                        "feedback": result.get('feedback', ''),
                        "metrics": metrics,
                        "logs_snippet": logs[-500:]
                    }
                    self.failure_history.append(failure_record)
                    self._log(f"  [System] Recorded failure history for {result['ticket_assigned_to']}")
                    

                    # self._reset_sandbox_to_phase0_state()

                    # 失败，分发 Ticket
                    # ticket = result['ticket_assigned_to']

                    new_ticket = result['ticket_assigned_to']
                    self._reset_downstream_state(new_ticket)  # ← 新增调用
                    ticket = new_ticket

                    feedback = result
                    self.retry_count += 1
                        
                except Exception as e:
                    self._log(f"Judge output parse error: {e}")
                    # Fallback
                    ticket = "Coder"
                    feedback = {'analysis': "Judge output invalid, defaulting to check code."}
                    self.retry_count += 1

        self._log("❌ Failed.")
        
        # --- SKILL DISTILLATION (FAILURE) ---
        if self.skill_manager:
            try:
                # Update credit scores for used knowledge (Negative Feedback)
                self.skill_manager.update_scores(list(self.used_knowledge_ids), success=False)
                
                if self.failure_history:
                    last_failure = self.failure_history[-1]
                    trajectory = {
                        "exp_id": self.exp_id,
                        "task_name": self.task_name,
                        "task_desc": self.task_desc,
                        "domain": "General", # Default
                        "difficulty": "Unknown",
                        "outcome": "failure",
                        "quality_score": 0.0,
                        "steps": self.trajectory_steps,
                        "used_knowledge_ids": list(self.used_knowledge_ids),
                        "retrieval_key": f"Trajectory: {self.task_name} (Failure)",
                        "timestamp": int(time.time()),
                        "final_plan": self.current_plan,
                        "final_skeleton": self.current_skeleton,
                        "final_code": self.current_code,
                        "final_reward": last_failure.get('metrics', 0),
                        "key_logs": last_failure.get('analysis', '') + "\n" + last_failure.get('evidence', '')
                    }
                    # Store the stats
                    self.distillation_stats = self.skill_manager.distill_and_store(trajectory)
            except Exception as e:
                self._log(f"  ⚠️ Skill distillation failed: {e}")

        self.failure_history.clear()
        
        # Generate final report
        self.generate_knowledge_report(success=False)
        return False

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    workflow = InverseProblemWorkflow(
        task_desc="Recover a 64x64 image from 50% randomly missing pixels. Use ADMM.",
        gt_code_path="./gt_repo",
        python_path="/usr/bin/python3",
        working_dir="./workspace_run_1"
    )
    workflow.run()
