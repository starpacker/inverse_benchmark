
import os
import sys
import shutil
import subprocess
import json
import time
import ast
import re
import datetime
from typing import List, Dict, Tuple, Any, Optional

from agents.planner_agent import PlannerAgent, CriticAgent
from agents.architect_agent import ArchitectAgent
from agents.coder_agent import CoderAgent
from agents.judge_agent import JudgeAgent
from agents.sandbox_manager import DataGenAgent, EvalGenAgent, get_installed_libraries

class InverseProblemBase:
    def __init__(self, task_name: str, task_desc: str, gt_code_path: str, python_path: str, working_dir: str, client: Any, model_name: str, root_output_dir: str = "/data/yjh/end_sandbox", skill_manager: Any = None):
        self.task_name = task_name
        self.task_desc = task_desc
        self.gt_code_path = gt_code_path
        self.python_path = python_path
        
        # Directory structure
        self.root_dir = os.path.abspath(working_dir)
        self.sandbox_dir = os.path.join(root_output_dir, f"{task_name}_sandbox")
        
        self.package_list = get_installed_libraries(self.python_path)
        
        self.client = client
        self.model_name = model_name
        
        # Skill Manager
        self.skill_manager = skill_manager
        
        # Initialize Agents
        self.planner = PlannerAgent(client, model_name)
        self.critic = CriticAgent(client, model_name)
        self.architect = ArchitectAgent(client, model_name)
        self.coder = CoderAgent(client, model_name)
        self.judge = JudgeAgent(client, model_name)
        self.data_gen_agent = DataGenAgent(client, model_name)
        self.eval_gen_agent = EvalGenAgent(client, model_name)
        
        # Memory State
        self.current_plan = ""
        self.current_skeleton = ""
        self.current_code = ""
        self.function_list = []
        self.failure_history: List[Dict] = []
        self.trajectory_steps: List[Dict] = []
        
        self.max_retries = 8
        self.retry_count = 0

        # Knowledge System: Track used knowledge items
        self.used_knowledge_ids = set()
        
        # Track newly generated knowledge during this run
        self.distillation_stats = {'instances': 0, 'experiences': 0, 'core': 0}

        # Experiment ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Subclasses should append suffix (e.g. _test_) if needed, but base uses standard
        self.exp_id = f"{self.task_name}_{timestamp}"
        
        # Log directory
        self.snapshot_dir = os.path.join(root_output_dir, self.model_name, self.exp_id)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.log_file = os.path.join(self.snapshot_dir, "workflow.log")
        open(self.log_file, 'a').close()
        
        self._log(f"Workflow initialized. Exp ID: {self.exp_id}")
        self._log(f"Sandbox Directory: {self.sandbox_dir}")
        self._log(f"Snapshot Directory: {self.snapshot_dir}")

    def _log(self, message: str):
        timestamp = time.strftime("[%H:%M:%S]")
        formatted_msg = f"{timestamp} {message}"
        print(formatted_msg)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(formatted_msg + "\n")

    def _generate_retrieval_key(self, role: str, input_data: Any, output_data: Any) -> str:
        """
        Auto-generate retrieval key based on input/output content.
        """
        try:
            if role == "Planner":
                # Input usually contains task_desc. Output is plan.
                # Key: Task Summary
                desc = input_data.get('task_desc', '') if isinstance(input_data, dict) else str(input_data)
                return f"Plan for task: {desc[:100]}..."
            
            elif role == "Architect":
                # Input: Plan. Output: Skeleton.
                # Key: Plan Summary
                plan = input_data.get('plan', '') if isinstance(input_data, dict) else str(input_data)
                return f"Architecture for plan: {plan[:100]}..."
            
            elif role == "Coder":
                # Input: Task info. Output: Code.
                # Key: Task Target
                target = input_data.get('target_function') or input_data.get('target_type') or 'unknown'
                return f"Implement {target} in {self.task_name}"
            
            elif role == "Execution":
                return f"Execution of {self.task_name}"
            
            elif role == "Judge":
                # Fallback if not provided explicitly
                return f"Judge analysis for {self.task_name}"
            
            return f"{role} step for {self.task_name}"
        except:
            return f"Step {role} in {self.task_name}"

    def _record_step(self, iteration: int, role: str, input_data: Any, output_data: Any, retrieval_key: str = None):
        """
        Records a detailed step in the trajectory with input, output and retrieval key.
        """
        if retrieval_key is None:
            retrieval_key = self._generate_retrieval_key(role, input_data, output_data)

        step = {
            "step_id": len(self.trajectory_steps) + 1,
            "iteration": iteration,
            "role": role,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data,
            "retrieval_key": retrieval_key
        }
        self.trajectory_steps.append(step)
            
    def _save_artifact(self, filename: str, content: str) -> str:
        path = os.path.join(self.snapshot_dir, f"{self.exp_id}_{filename}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def _save_snapshot(self, iteration: int, stage: str, content: dict):
        path = os.path.join(self.snapshot_dir, f"iter_{iteration:03d}_{stage}.json")
        with open(path, 'w') as f:
            json.dump({
                "exp_id": self.exp_id,
                "iteration": iteration,
                "stage": stage,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **content
            }, f, indent=2)

    # ... (Rest of the methods: _setup_sandbox, _run_cmd, _parse_functions_from_skeleton, _extract_json, _extract_python, _validate_skeleton, _clean_code, _write_file, _format_failure_histories, _build_context_with_memory, _highlight_target_in_code, _load_data_shapes, _phase_0_preparation, _reset_sandbox_to_phase0_state, _reset_downstream_state, generate_knowledge_report) ...
    # I will copy the rest of the methods unchanged.
    
    def _setup_sandbox(self):
        self._log(f"\n[System] Initializing Sandbox at {self.sandbox_dir}...")
        os.makedirs(self.sandbox_dir, exist_ok=True)
        
        dest_gt = os.path.join(self.sandbox_dir, "gt_code")
        if os.path.exists(dest_gt):
            if os.path.isdir(dest_gt): shutil.rmtree(dest_gt)
            else: os.remove(dest_gt)

        if os.path.isdir(self.gt_code_path):
            shutil.copytree(self.gt_code_path, dest_gt)
        elif os.path.isfile(self.gt_code_path):
            os.makedirs(dest_gt, exist_ok=True)
            shutil.copy(self.gt_code_path, dest_gt)
        else:
            raise FileNotFoundError(f"GT code path not found: {self.gt_code_path}")
        
        os.makedirs(os.path.join(self.sandbox_dir, "dataset"), exist_ok=True)

    def _run_cmd(self, script_name: str, args: List[str] = [], timeout=600, check_syntax_only=False):
        if check_syntax_only:
            cmd = [self.python_path, "-m", "py_compile", script_name]
            timeout = 30
        else:
            cmd = [self.python_path, script_name] + args
            
        try:
            result = subprocess.run(
                cmd,
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "TIMEOUT EXPIRED"

    def _parse_functions_from_skeleton(self, skeleton_code: str) -> List[str]:
        func_list = []
        try:
            tree = ast.parse(skeleton_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "InverseSolver":
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            func_list.append(item.name)
        except Exception as e:
            self._log(f"[System] AST Parsing Error: {e}")
        return func_list

    def _extract_json(self, text: str) -> str:
        matches = re.findall(r'```json\s*\n(.*?)\n?\s*```', text, re.DOTALL)
        if matches: return max(matches, key=len).strip()
        matches = re.findall(r'```\s*\n(.*?)\n?\s*```', text, re.DOTALL)
        for m in sorted(matches, key=len, reverse=True):
            if m.strip().startswith('{') and m.strip().endswith('}'): return m.strip()
        try:
            json.loads(text)
            return text
        except: pass
        candidates = re.findall(r'(\{[\s\S]*\})', text)
        if candidates:
            longest = max(candidates, key=len)
            try:
                json.loads(longest)
                return longest
            except: pass
        return text.strip()

    def _extract_python(self, text: str) -> str:
        code = text.strip()
        matches = re.findall(r'```python\s*\n(.*?)\n?\s*```', text, re.DOTALL)
        if matches:
            code = max(matches, key=len).strip()
        else:
            matches = re.findall(r'```\s*\n(.*?)\n?\s*```', text, re.DOTALL)
            if matches:
                code = max(matches, key=len).strip()
            else:
                lines = text.split('\n')
                start_idx = 0
                found_code = False
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ', 'class ', 'def ', '@')):
                        start_idx = i
                        found_code = True
                        break
                if found_code:
                    code = '\n'.join(lines[start_idx:]).strip()
        if len(code) > 2 and (code.startswith('"') or code.startswith("'")):
            try:
                unescaped = ast.literal_eval(code)
                if isinstance(unescaped, str):
                    self._log("  [System] Detected and unescaped string-wrapped code.")
                    return unescaped.strip()
            except: pass
        return code

    def _validate_skeleton(self, code: str) -> Tuple[bool, str]:
        if not code.strip(): return False, "Empty code."
        try:
            tree = ast.parse(code)
            has_solver = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "InverseSolver":
                    has_solver = True
            if not has_solver: return False, "Missing 'class InverseSolver'."
            return True, ""
        except SyntaxError as e: return False, f"Syntax Error: {e}"
        except Exception as e: return False, f"Validation Error: {e}"

    def _clean_code(self, code: str) -> str:
        return self._extract_python(code)

    def _write_file(self, filename: str, content: str):
        content = self._extract_python(content)
        path = os.path.join(self.sandbox_dir, filename)
        with open(path, "w", encoding='utf-8') as f:
            f.write(content)
    
    def _format_failure_histories(self, histories: List[Dict]) -> str:
        if not histories: return ""
        formatted = "\n### ⚠️ PAST FAILURES (Avoid Repeating These Errors) ###\n"
        for i, hist in enumerate(histories[-3:], 1):
            iter_num = hist.get('iteration', '?')
            timestamp = hist.get('timestamp', '').split()[1] if hist.get('timestamp') else ''
            error_type = hist.get('ticket_assigned_to', 'Unknown')
            fix_target = hist.get('fix_target', 'N/A')
            analysis = hist.get('analysis', 'N/A').replace('\n', ' ').strip()
            evidence = hist.get('evidence', '').replace('\n', ' ').strip()[:100]
            formatted += (
                f"\n[Iter {iter_num} | {timestamp}] {error_type} → {fix_target}\n"
                f"  Cause: {analysis[:80]}...\n"
            )
            if evidence: formatted += f"  Evidence: {evidence}...\n"
        return formatted
    
    def _extract_function_signature(self, code: str, func_name: str) -> Optional[str]:
        """
        Extracts the full function signature (including decorators and docstring) for a given function name
        from the provided code (usually skeleton).
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Found the function. Extract source lines.
                    # AST nodes have lineno (1-based).
                    # We want to capture:
                    # 1. Decorators (if any)
                    # 2. def func(...) -> ...:
                    # 3. Docstring (if strictly the first statement)
                    
                    lines = code.split('\n')
                    start_line = node.lineno - 1
                    
                    # Adjust start_line to include decorators
                    if node.decorator_list:
                        start_line = node.decorator_list[0].lineno - 1
                    
                    # Find end of signature (:)
                    # This is tricky with multi-line args.
                    # We can approximate by looking for the docstring or the first statement.
                    
                    end_line = node.lineno # At least the def line
                    
                    # Check for docstring
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # If docstring exists, the signature block effectively ends after the docstring?
                        # Or do we want just the signature?
                        # User requested "signature + docstring".
                        
                        # Find the docstring node in body
                        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                             doc_node = node.body[0]
                             end_line = doc_node.end_lineno if hasattr(doc_node, 'end_lineno') else doc_node.lineno
                    else:
                        # No docstring, try to find the colon of def?
                        # Or just take the first few lines until body starts?
                        if node.body:
                            # The body starts at the first statement
                            # We want everything BEFORE the first statement (excluding docstring if we already handled it)
                            # Actually, let's just grab the text from start_line to the line before first body statement
                            first_stmt = node.body[0]
                            end_line = first_stmt.lineno - 2 # Line before the statement
                            if end_line < start_line: end_line = start_line # Fallback
                        else:
                            end_line = start_line # Empty body
                            
                    # Extract snippet
                    snippet = "\n".join(lines[start_line : end_line + 1])
                    return snippet.strip()
                    
        except Exception as e:
            self._log(f"[System] Signature extraction failed for {func_name}: {e}")
        
        return None

    def _build_context_with_memory(self, base_context: Dict[str, Any], agent_role: str, current_ticket: str, retrieval_query: str = None) -> Dict[str, Any]:
        MAX_HISTORY_LEN = 3
        if len(self.failure_history) > MAX_HISTORY_LEN:
            self.failure_history = sorted(self.failure_history, key=lambda x: x.get('iteration', 0), reverse=True)[:MAX_HISTORY_LEN]

        context = base_context.copy()
        
        # 1. Previous Output Injection
        if agent_role == "Planner" and self.current_plan.strip():
            constraints = []
            if hasattr(self, 'input_shape') and self.input_shape: constraints.append(f"Input shape: {self.input_shape}")
            if hasattr(self, 'output_shape') and self.output_shape: constraints.append(f"Output shape: {self.output_shape}")
            
            if constraints:
                constraint_block = "### 🔑 HARD CONSTRAINTS (NON-NEGOTIABLE)\n" + "\n".join([f"• {c}" for c in constraints]) + "\n\n"
                context["task_desc"] = constraint_block + context.get("task_desc", "")
            
            if self.current_plan.strip():
                context["task_desc"] += (f"\n### PREVIOUS PLAN (Iter {self.retry_count}) ###\n{self.current_plan}")
        
        elif agent_role == "Architect" and self.current_skeleton.strip():
            context["plan"] = context.get("plan", "")
            context["previous_skeleton"] = self.current_skeleton
        
        elif agent_role == "Coder" and self.current_code.strip():
            if base_context.get("fix_target"):
                target = base_context["fix_target"]
                context["current_full_code"] = self._highlight_target_in_code(self.current_code, target)
        
        # 2. Failure History Injection
        if agent_role not in ["Judge", "Critic"]:
            relevant_histories = [h for h in self.failure_history if h.get("ticket_assigned_to") == current_ticket]
            if relevant_histories:
                history_section = self._format_failure_histories(relevant_histories)
                if agent_role == "Planner": context["task_desc"] = context.get("task_desc", "") + history_section
                elif agent_role == "Architect": context["plan"] = history_section + context.get("plan", "")
                elif agent_role == "Coder":
                    existing_fb = context.get("feedback", "")
                    context["feedback"] = history_section + ("\n" + existing_fb if existing_fb else "")
        
        # 3. Knowledge Injection
        if self.skill_manager:
            try:
                # Determine Retrieval Query
                if retrieval_query:
                    query_text = retrieval_query
                else:
                    # Fallback to default (Task Desc)
                    query_text = self.task_desc.split("### 🛡️ CORE KNOWLEDGE")[0].strip()
                
                top_k = 4 if agent_role == "Coder" else 3
                
                knowledge = self.skill_manager.retrieve_knowledge(
                    task_desc=query_text, 
                    agent_role=agent_role, 
                    top_k=top_k
                )
                
                # Track usage
                for k_type in ['core', 'experience', 'instance']:
                    for item in knowledge.get(k_type, []):
                        if 'id' in item:
                            self.used_knowledge_ids.add(item['id'])
                
                knowledge_prompt = self.skill_manager.format_knowledge_for_prompt(knowledge)
                
                if knowledge_prompt:
                    # Use dedicated knowledge_context field
                    context["knowledge_context"] = knowledge_prompt
                    
            except Exception as e:
                self._log(f"  [System] Knowledge injection failed: {e}")

        return context

    def _highlight_target_in_code(self, code: str, target: str) -> str:
        if target in ["imports", "main_block"]:
            marker = {
                "imports": "# >>> TARGET: IMPORTS <<<",
                "main_block": "# >>> TARGET: MAIN BLOCK <<<"
            }.get(target, f"# >>> TARGET: {target} <<<")
            return f"{marker}\n{code}"
        else:
            lines = code.split('\n')
            highlighted = []
            in_target = False
            target_found = False
            for line in lines:
                if f"def {target}(" in line or f"async def {target}(" in line:
                    highlighted.append(f"\n# >>> TARGET FUNCTION: {target} <<<")
                    in_target = True
                    target_found = True
                elif not target_found and f"def " in line and target in line:
                     highlighted.append(f"\n# >>> TARGET FUNCTION (Probable): {target} <<<")
                     in_target = True
                     target_found = True
                elif in_target and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    in_target = False
                highlighted.append(line)
            if not target_found:
                 return f"# >>> TARGET: {target} (Not found in code, please implement) <<<\n{code}"
            return '\n'.join(highlighted)
    
    def _load_data_shapes(self):
        try:
            import numpy as np
            input_path = os.path.join(self.sandbox_dir, "dataset", "input.npy")
            gt_path = os.path.join(self.sandbox_dir, "dataset", "gt_output.npy")
            if os.path.exists(input_path):
                self.input_shape = np.load(input_path).shape
                self._log(f"  ✅ Loaded input shape: {self.input_shape}")
            else:
                self.input_shape = None
                self._log("  ⚠️ input.npy not found - shape unknown")
            if os.path.exists(gt_path):
                self.output_shape = np.load(gt_path).shape
                self._log(f"  ✅ Loaded output shape: {self.output_shape}")
            else:
                self.output_shape = None
                self._log("  ⚠️ gt_output.npy not found - shape unknown")
        except Exception as e:
            self._log(f"  ⚠️ Shape loading failed: {e}")
            self.input_shape = None
            self.output_shape = None

    def _phase_0_preparation(self):
        self._log("\n>>> [Phase 0] Initializing Data & Evaluation Scripts...")
        input_path = os.path.join(self.sandbox_dir, "dataset/input.npy")
        gt_path = os.path.join(self.sandbox_dir, "dataset/gt_output.npy")
        baseline_path = os.path.join(self.sandbox_dir, "dataset/baseline.npy")
        
        # Load GT code snippet for context
        gt_code_content = "N/A"
        try:
            # Assume GT code is in 'solver.py' or similar in the gt_code folder
            gt_folder = os.path.join(self.sandbox_dir, "gt_code")
            # Find the first python file
            py_files = [f for f in os.listdir(gt_folder) if f.endswith('.py')]
            if py_files:
                with open(os.path.join(gt_folder, py_files[0]), 'r') as f:
                    gt_code_content = f.read()[:2000] # Truncate if too long
        except Exception as e:
            self._log(f"  ⚠️ Failed to load GT code content: {e}")

        if os.path.exists(input_path) and os.path.exists(gt_path) and os.path.exists(baseline_path):
             self._log("  [Step 1] Data files found. Skipping Data Generation.")
        else:
            self._log("  [Step 1] Generating Data Generation Script...")
            
            # Inject actual GT code content instead of just path
            gen_ctx = {
                'task_desc': self.task_desc, 
                'gt_code_snippet': f"Here is a snippet of the Ground Truth code for reference on data structure:\n```python\n{gt_code_content}\n```"
            }
            
            for attempt in range(3):
                code = self.data_gen_agent.generate(gen_ctx)
                self._write_file("data_gen.py", code)
                success, out, err = self._run_cmd("data_gen.py", timeout=120)
                input_exists = os.path.exists(os.path.join(self.sandbox_dir, "dataset/input.npy"))
                gt_exists = os.path.exists(os.path.join(self.sandbox_dir, "dataset/gt_output.npy"))
                baseline_exists = os.path.exists(os.path.join(self.sandbox_dir, "dataset/baseline.npy"))
                if success and input_exists and gt_exists and baseline_exists:
                    self._log(f"    ✅ Data generation success (Attempt {attempt+1})")
                    break
                else:
                    self._log(f"    ⚠️ Data generation failed (Attempt {attempt+1}). Error: {err}")
                    gen_ctx['feedback'] = (f"Missing files. Error: {err}")
            else:
                raise RuntimeError("❌ Failed to generate valid data_gen.py after 3 retries.")
        
        self._load_data_shapes()  
        self._log("[Phase 0] Complete.\n")

        self._log("  [Step 2] Preparing Evaluation Script...")
        eval_script_path = os.path.join(self.sandbox_dir, "eval_script.py")
        if not os.path.exists(eval_script_path):
            self._log("  [Step 2] Generating Evaluation Script...")
            
            # Inject actual shape info instead of file path
            shape_info = f"Input Shape: {self.input_shape}, Output Shape: {self.output_shape}" if self.input_shape else "Unknown Shape"
            
            eval_ctx = {
                'task_desc': self.task_desc, 
                'data_shape_hint': f"Data Shapes: {shape_info}. The evaluation script must load 'output.npy' and compare it against 'dataset/baseline.npy' or 'dataset/gt_output.npy'."
            }
            
            for attempt in range(3):
                code = self.eval_gen_agent.generate(eval_ctx)
                self._write_file("eval_script.py", code)
                success, out, err = self._run_cmd("eval_script.py", args=["dataset/baseline.npy"])
                if success:
                    try:
                        metrics = json.loads(out)
                        self.baseline_metrics = metrics
                        self._log(f"    ✅ Baseline Metrics: {metrics}")
                        break 
                    except json.JSONDecodeError:
                        self._log(f"    ⚠️ Eval script output invalid JSON (Attempt {attempt+1}): {out}")
                        eval_ctx['feedback'] = f"Output invalid JSON. Output: {out}\nError: JSONDecodeError"
                else:
                    self._log(f"    ⚠️ Eval script validation failed (Attempt {attempt+1}). Error: {err}")
                    eval_ctx['feedback'] = f"Runtime Error: {err}"
            else:
                raise RuntimeError(f"❌ Eval script validation failed after 3 attempts. Last Error: {err}")
        else:
            self._log("  [Step 2] Found existing eval_script.py. Skipping generation.")
            success, out, err = self._run_cmd("eval_script.py", args=["dataset/baseline.npy"])
            if success:
                try:
                    metrics = json.loads(out)
                    self.baseline_metrics = metrics
                    self._log(f"    ✅ Baseline Metrics: {metrics}")
                except json.JSONDecodeError:
                    raise RuntimeError(f"❌ Eval script output is not valid JSON. Output: {out}")
            else:
                raise RuntimeError(f"❌ Eval script validation failed. Error: {err}")
            
        self._log("[Phase 0] Complete.\n")

    def _reset_sandbox_to_phase0_state(self):
        preserve_paths = {"dirs": ["dataset", "gt_code"], "files": ["eval_script.py", "data_gen.py"]}
        self._log(">>> [System] Resetting sandbox to Phase 0 state...")
        for item in os.listdir(self.sandbox_dir):
            item_path = os.path.join(self.sandbox_dir, item)
            if (os.path.isdir(item_path) and item in preserve_paths["dirs"]) or \
            (os.path.isfile(item_path) and item in preserve_paths["files"]): continue
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path): os.remove(item_path)
                elif os.path.isdir(item_path): shutil.rmtree(item_path)
            except Exception as e: self._log(f"  [Warning] Failed to remove {item}: {str(e)}")
        self._log("  ✅ Sandbox reset complete.")

    def _reset_downstream_state(self, ticket: str):
        if ticket == "Planner":
            self._log("  [State] Resetting downstream state: skeleton + code (Planner ticket)")
            self.current_skeleton = ""
            self.current_code = ""
            self.function_list = []
        elif ticket == "Architect":
            self._log("  [State] Resetting downstream state: code only (Architect ticket)")
            self.current_code = ""

    def generate_knowledge_report(self, success: bool):
        """Generates a final report of knowledge usage and success."""
        report = "\n" + "="*50 + "\n"
        report += "📊 KNOWLEDGE USAGE REPORT\n"
        report += "="*50 + "\n"
        report += f"Task Outcome: {'✅ SUCCESS' if success else '❌ FAILURE'}\n"
        report += f"Knowledge Items Used: {len(self.used_knowledge_ids)}\n"
        
        if self.skill_manager and self.used_knowledge_ids:
            try:
                # We need a method in SkillManager to get details by IDs.
                # Since we don't have it yet, we will rely on a new method or try to fetch from DB manually?
                # Best practice: Add method to SkillManager.
                # Assuming `get_knowledge_details` exists.
                if hasattr(self.skill_manager, 'get_knowledge_details'):
                    details = self.skill_manager.get_knowledge_details(list(self.used_knowledge_ids))
                    
                    # Group by Type
                    by_type = {"core": [], "experience": [], "instance": []}
                    for item in details:
                        k_type = item.get('type', 'unknown')
                        if k_type in by_type: by_type[k_type].append(item)
                    
                    for k_type, items in by_type.items():
                        if items:
                            report += f"\n[{k_type.upper()}]\n"
                            for item in items:
                                score_change = "+0.1" if success else "-0.2"
                                report += f"  - {item['name']} (Score: {item.get('credit_score', 1.0):.2f} -> {score_change})\n"
                else:
                    report += "  (Details unavailable: SkillManager.get_knowledge_details not implemented)\n"
            except Exception as e:
                report += f"  Error generating details: {e}\n"
        else:
            report += "  No knowledge items were injected.\n"
            
        report += "="*50 + "\n"
        self._log(report)
