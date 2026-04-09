import os
import sys
import shutil
import subprocess
import json
import time
import ast
import yaml
from typing import List, Dict, Tuple, Any
import re, hashlib, datetime
from openai import OpenAI
from agents.planner_agent import PlannerAgent, CriticAgent
from agents.architect_agent import ArchitectAgent
from agents.coder_agent import CoderAgent
from agents.judge_agent import JudgeAgent
from agents.sandbox_manager import DataGenAgent, EvalGenAgent, get_installed_libraries, TaskDescAgent

# Add current directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
class InverseProblemWorkflow:
    def __init__(
        self, 
        task_name: str, 
        paper_md_path: str,  # REPLACED gt_code_path
        python_path: str, 
        client: Any, 
        model_name: str,
        root_output_dir: str = "/data/yjh/paper_sandbox"
    ):
        self.task_name = task_name
        self.paper_md_path = paper_md_path
        self.python_path = python_path
        
        # New directory structure per user request
        self.sandbox_dir = os.path.join(root_output_dir, f"{task_name}_sandbox")
        
        self.package_list = get_installed_libraries(self.python_path)
        
        self.client = client
        self.model_name = model_name

        # 初始化 Agents
        self.planner = PlannerAgent(client, model_name)
        self.critic = CriticAgent(client, model_name)
        self.architect = ArchitectAgent(client, model_name)
        self.coder = CoderAgent(client, model_name)
        self.judge = JudgeAgent(client, model_name)
        self.data_gen_agent = DataGenAgent(client, model_name)
        self.eval_gen_agent = EvalGenAgent(client, model_name)
        self.task_desc_generator = TaskDescAgent(client, model_name)

        self.task_desc = None 
        
        # 状态记忆 (Memory)
        self.current_plan = ""
        self.current_skeleton = ""
        self.current_code = ""
        self.function_list = []
        self.failure_history: List[Dict] = []  # 存储结构化失败记录
        
        self.max_retries = 20
        self.retry_count = 0

        # ✅ 在 __init__ 中初始化唯一实验ID
        # safe_task = re.sub(r'[^\w-]', '_', self.task_desc.lower())[:30]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_id = f"{self.task_name}_{timestamp}"
        
        # Log directory: /data/yjh/end_sandbox/<model_name>/<task_name>_<timestamp>
        self.snapshot_dir = os.path.join(root_output_dir, self.model_name, self.exp_id)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.log_file = os.path.join(self.snapshot_dir, "workflow.log")
        open(self.log_file, 'a').close()  # 确保文件存在
        self._log(f"Workflow initialized. Exp ID: {self.exp_id}")
        self._log(f"Sandbox Directory: {self.sandbox_dir}")
        self._log(f"Snapshot Directory: {self.snapshot_dir}")

    def _log(self, message: str):
        """双重日志：打印到屏幕 + 写入文件"""
        timestamp = time.strftime("[%H:%M:%S]")
        formatted_msg = f"{timestamp} {message}"
        print(formatted_msg)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(formatted_msg + "\n")
            
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


    def _setup_sandbox(self):
        """初始化沙箱，仅创建必要目录（无 gt_code）"""
        self._log(f"\n[System] Initializing Sandbox at {self.sandbox_dir}...")
        os.makedirs(self.sandbox_dir, exist_ok=True)
        
        # ✅ Paper-driven 模式：仅创建 dataset 目录
        os.makedirs(os.path.join(self.sandbox_dir, "dataset"), exist_ok=True)
        self._log("  ✅ Paper-driven mode: No GT code required")

    def _run_cmd(self, script_name: str, args: List[str] = [], timeout=600, check_syntax_only=False):
        """通用命令执行器"""
        if check_syntax_only:
            # Python 语法检查命令: python -m py_compile solver.py
            cmd = [self.python_path, "-m", "py_compile", script_name]
            timeout = 30 # 语法检查很快
        else:
            # 实际运行
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
        """
        [Hard-coded Logic]
        不依赖 LLM，而是使用 AST 强制解析 Architect 生成的骨架，
        提取 InverseSolver 类中所有需要实现的方法名。
        """
        func_list = []
        try:
            tree = ast.parse(skeleton_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "InverseSolver":
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            # 排除 __init__ 如果不需要特殊实现，或者保留
                            # 这里通常保留所有方法让 Coder 填充
                            func_list.append(item.name)
        except Exception as e:
            self._log(f"[System] AST Parsing Error: {e}")
        
        return func_list

    def _extract_plain_text(self, text: str) -> str:
        """从 LLM 响应中提取纯文本（移除 markdown/code blocks）"""
        # 移除代码块标记
        text = re.sub(r'```(?:python|json)?\s*', '', text)
        text = re.sub(r'```', '', text)
        # 移除多余空白行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _extract_json(self, text: str) -> str:
        """
        Extracts JSON from text, handling markdown blocks and thinking traces.
        """
        # 1. Try explicit json markdown block
        matches = re.findall(r'```json\s*\n(.*?)\n?\s*```', text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
            
        # 2. Try generic markdown block, but validate if it looks like JSON
        matches = re.findall(r'```\s*\n(.*?)\n?\s*```', text, re.DOTALL)
        for m in sorted(matches, key=len, reverse=True):
            if m.strip().startswith('{') and m.strip().endswith('}'):
                return m.strip()

        # 3. Fallback: Find the last valid JSON object in the raw text
        # This is expensive but robust. We search from the end.
        # Simple heuristic: find last '}' and corresponding '{'
        try:
            # Quick check if the whole text is JSON
            json.loads(text)
            return text
        except:
            pass
            
        # Regex to find potential JSON objects (non-nested simple check or greedy)
        # It's better to rely on the fact that the actual JSON is likely the last thing or substantial.
        # Let's try to find the largest substring starting with { and ending with } that parses.
        
        candidates = re.findall(r'(\{[\s\S]*\})', text)
        if candidates:
            # Check the longest one first
            longest = max(candidates, key=len)
            try:
                json.loads(longest)
                return longest
            except:
                # If longest fails (e.g. multiple json objects merged), try to be smarter?
                # For now, return longest and let caller handle parse error
                pass
                
        return text.strip()

    def _extract_python(self, text: str) -> str:
        """
        Extracts Python code from text.
        """
        code = text.strip()
        
        # 1. Explicit Python block
        matches = re.findall(r'```python\s*\n(.*?)\n?\s*```', text, re.DOTALL)
        if matches:
            code = max(matches, key=len).strip()
        else:
            # 2. Generic block
            matches = re.findall(r'```\s*\n(.*?)\n?\s*```', text, re.DOTALL)
            if matches:
                code = max(matches, key=len).strip()
            else:
                # 3. Fallback: Heuristic search for code start
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

        # 4. Handle Quoted String Representation (Double-Escape Fix)
        # If the code starts with " or ' and looks like a python string literal, unescape it.
        # This handles cases where LLM outputs "import foo\n..." instead of import foo
        if len(code) > 2 and (code.startswith('"') or code.startswith("'")):
            try:
                # Attempt to unescape if it's a valid string literal
                # This handles "line1\nline2" -> line1\nline2
                unescaped = ast.literal_eval(code)
                if isinstance(unescaped, str):
                    self._log("  [System] Detected and unescaped string-wrapped code.")
                    return unescaped.strip()
            except:
                pass
        
        return code

    def _validate_skeleton(self, code: str) -> Tuple[bool, str]:
        """
        Validates that the skeleton is valid Python and contains required classes.
        """
        if not code.strip():
            return False, "Empty code."
        
        try:
            tree = ast.parse(code)
            has_config = False
            has_solver = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name == "InverseSolver":
                        has_solver = True
            
            if not has_solver:
                return False, "Missing 'class InverseSolver'."
                
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax Error: {e}"
        except Exception as e:
            return False, f"Validation Error: {e}"

    def _clean_code(self, code: str) -> str:
        # Deprecated: alias to _extract_python for backward compatibility if needed
        return self._extract_python(code)

    def _write_file(self, filename: str, content: str):
        """辅助写入文件"""
        content = self._extract_python(content)
        path = os.path.join(self.sandbox_dir, filename)
        with open(path, "w", encoding='utf-8') as f:
            f.write(content)
    
    def _format_failure_histories(self, histories: List[Dict]) -> str:
        """
        将失败历史格式化为 LLM 友好的文本片段。
        仅保留关键信息，避免噪声污染上下文。
        """
        if not histories:
            return ""
        
        formatted = "\n### ⚠️ PAST FAILURES (Avoid Repeating These Errors) ###\n"
        
        for i, hist in enumerate(histories[-3:], 1):  # 仅取最近 3 次（防上下文爆炸）
            iter_num = hist.get('iteration', '?')
            timestamp = hist.get('timestamp', '').split()[1] if hist.get('timestamp') else ''
            error_type = hist.get('ticket_assigned_to', 'Unknown')
            fix_target = hist.get('fix_target', 'N/A')
            analysis = hist.get('analysis', 'N/A').replace('\n', ' ').strip()
            evidence = hist.get('evidence', '').replace('\n', ' ').strip()[:100]  # 截断
            
            # 精简格式：单行摘要 + 关键证据
            formatted += (
                f"\n[Iter {iter_num} | {timestamp}] {error_type} → {fix_target}\n"
                f"  Cause: {analysis[:80]}...\n"
            )
            if evidence:
                formatted += f"  Evidence: {evidence}...\n"
        
        return formatted
    
    def _build_context_with_memory(
        self, 
        base_context: Dict[str, Any], 
        agent_role: str,
        current_ticket: str
        ) -> Dict[str, Any]:
        
        # === 限制历史长度：仅保留最近 2 次 ===
        MAX_HISTORY_LEN = 3
        if len(self.failure_history) > MAX_HISTORY_LEN:
            # 保留最近 N 次（按 iteration 排序）
            self.failure_history = sorted(
                self.failure_history, 
                key=lambda x: x.get('iteration', 0),
                reverse=True
            )[:MAX_HISTORY_LEN]

        context = base_context.copy()
        # ====== 1. 注入同类型上一轮输出（核心增强） ======
        if agent_role == "Planner" and self.current_plan.strip():
            constraints = []
        
            # 注入 input/output shape（关键！）
            if hasattr(self, 'input_shape') and self.input_shape:
                constraints.append(f"Input shape: {self.input_shape}")
            if hasattr(self, 'output_shape') and self.output_shape:
                constraints.append(f"Output shape: {self.output_shape}")
            
            
            # 构建约束前缀
            if constraints:
                constraint_block = (
                    "### 🔑 HARD CONSTRAINTS (NON-NEGOTIABLE)\n" +
                    "\n".join([f"• {c}" for c in constraints]) +
                    "\n\n"
                )
                # 注入到 task_desc 开头（最高优先级）
                context["task_desc"] = constraint_block + context.get("task_desc", "")
            
            # 注入历史计划（保持原有逻辑）
            if self.current_plan.strip():
                context["task_desc"] += (
                    f"\n### PREVIOUS PLAN (Iter {self.retry_count}) ###\n"
                    f"{self.current_plan}"
                )
        
        elif agent_role == "Architect" and self.current_skeleton.strip():
            # 标注 skeleton 变更意图（帮助 Architect 聚焦修改点）
            context["plan"] = context.get("plan", "")
            context["previous_skeleton"] = self.current_skeleton  # 显式字段
        
        elif agent_role == "Coder" and self.current_code.strip():
            # Coder 策略：根据 Judge feedback 决定注入粒度
            if base_context.get("fix_target"):
                # 修补模式：注入完整代码 + 高亮目标区域
                target = base_context["fix_target"]
                context["current_full_code"] = self._highlight_target_in_code(self.current_code, target)
                
                # Special handling for ambiguous targets: if full rebuild is triggered, don't use highlight
                # We check this by seeing if coding_tasks in the main flow (not visible here easily)
                # But we can check if target is in our list
                # Actually, simpler: always provide code, but if target is "unknown", highlight won't work well.
                # Let's rely on _highlight_target_in_code to handle fuzzy matching or fallback.
            else:
                # Default case: Use whatever is passed in base_context (which is the accumulator)
                # DO NOT RESET TO SKELETON HERE.
                # The reset logic is handled in the main loop before iterating tasks.
                pass
        
        # ====== 2. 注入失败历史（保持原有逻辑） ======
        if agent_role not in ["Judge", "Critic"]:
            relevant_histories = [
                h for h in self.failure_history 
                if h.get("ticket_assigned_to") == current_ticket
            ]
            if relevant_histories:
                history_section = self._format_failure_histories(relevant_histories)
                if agent_role == "Planner":
                    context["task_desc"] = context.get("task_desc", "") + history_section
                elif agent_role == "Architect":
                    context["plan"] = history_section + context.get("plan", "")
                elif agent_role == "Coder":
                    existing_fb = context.get("feedback", "")
                    context["feedback"] = history_section + ("\n" + existing_fb if existing_fb else "")
        
        return context

    def _highlight_target_in_code(self, code: str, target: str) -> str:
        """在代码中高亮目标函数/区域，帮助 Coder 聚焦"""
        if target in ["imports", "main_block"]:
            marker = {
                "imports": "# >>> TARGET: IMPORTS <<<",
                "main_block": "# >>> TARGET: MAIN BLOCK <<<"
            }.get(target, f"# >>> TARGET: {target} <<<")
            return f"{marker}\n{code}"
        else:
            # 高亮目标函数
            lines = code.split('\n')
            highlighted = []
            in_target = False
            
            # Simple fuzzy check if exact match fails
            target_found = False
            
            for line in lines:
                # Precise match for function definition
                if f"def {target}(" in line or f"async def {target}(" in line:
                    highlighted.append(f"\n# >>> TARGET FUNCTION: {target} <<<")
                    in_target = True
                    target_found = True
                # Heuristic for ambiguous targets (e.g. "solve method")
                elif not target_found and f"def " in line and target in line:
                     highlighted.append(f"\n# >>> TARGET FUNCTION (Probable): {target} <<<")
                     in_target = True
                     target_found = True
                
                elif in_target and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    in_target = False
                highlighted.append(line)
            
            if not target_found:
                 # If we couldn't find the function, prepend a global attention marker
                 return f"# >>> TARGET: {target} (Not found in code, please implement) <<<\n{code}"
                 
            return '\n'.join(highlighted)
    
    def _load_data_shapes(self):
        """Phase 0 后自动加载 input/gt_output shape，作为全局约束"""
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

    def _validate_task_desc(self, desc: str) -> bool:
        """验证 task_description 质量：必须包含具体形状/算法描述"""
        if len(desc) < 80:
            return False
        
    def _phase_0_preparation(self):
        self._log("\n>>> [Phase 0] Paper-Driven Preparation...")
        
        # STEP 1: 生成 task_description（核心新增步骤）
        # 1. Check if cached task description exists in paper_archive
        archive_dir = "/home/yjh/agentic_reproduce/paper_archive"
        # Use a consistent filename as requested, or task-specific if needed. 
        # User requested: "paper_archive/task_description_final.md"
        cached_desc_path = os.path.join(archive_dir, "task_description_final.md")
        
        if os.path.exists(cached_desc_path):
             self._log(f"✅ Found cached task description at {cached_desc_path}")
             with open(cached_desc_path, 'r', encoding='utf-8') as f:
                 self.task_desc = f.read()
             self._save_artifact("task_description_final.md", self.task_desc)
        
        if not self.task_desc:
            with open(self.paper_md_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            
            try:
                self.task_desc = self.task_desc_generator.generate({'paper_content': paper_content})
                
                # Cache it globally
                try:
                    os.makedirs(archive_dir, exist_ok=True)
                    with open(cached_desc_path, 'w', encoding='utf-8') as f:
                        f.write(self.task_desc)
                    self._log(f"✅ Cached task description to {cached_desc_path}")
                except Exception as e:
                    self._log(f"⚠️ Failed to cache task description: {e}")
                
                self._save_artifact("task_description_final.md", self.task_desc)
                self._log(f"✅ Task description generated successfully")
            except Exception as e:
                raise RuntimeError(f"Task description generation failed: {str(e)}") from e
            
        # STEP 2: 生成 data_gen.py（关键：移除 gt_code_snippet）
        self._log("\n>>> [Phase 0] Generating Data Pipeline...")
        input_path = os.path.join(self.sandbox_dir, "dataset/input.npy")
        gt_path = os.path.join(self.sandbox_dir, "dataset/gt_output.npy")
        baseline_path = os.path.join(self.sandbox_dir, "dataset/baseline.npy")
        
        if not (os.path.exists(input_path) and os.path.exists(gt_path) and os.path.exists(baseline_path)):
            # ✅ 仅传入 task_desc，完全移除 gt_code_snippet
            data_ctx = {'task_desc': self.task_desc}
            
            for attempt in range(3):
                code = self.data_gen_agent.generate(data_ctx)
                self._write_file("data_gen.py", code)
                success, out, err = self._run_cmd("data_gen.py", timeout=120)
                
                files_ok = all(os.path.exists(os.path.join(self.sandbox_dir, "dataset", f)) 
                            for f in ["input.npy", "gt_output.npy", "baseline.npy"])
                if success and files_ok:
                    self._log(f"✅ Data generation succeeded (Attempt {attempt+1})")
                    break
                else:
                    self._log(f"⚠️  Data generation failed (Attempt {attempt+1})")
                    data_ctx['feedback'] = f"Error: {err[:300]}..."
            else:
                raise RuntimeError("Data generation failed after 3 attempts")
        
        # STEP 3: 加载 shapes
        self._load_data_shapes()
        
        # STEP 4: 生成 eval_script.py
        self._log("\n>>> [Phase 0] Generating Evaluation Script...")
        eval_script_path = os.path.join(self.sandbox_dir, "eval_script.py")
        if not os.path.exists(eval_script_path):
            eval_ctx = {
                'task_desc': self.task_desc,
                'data_shape_hint': f"Input: {getattr(self, 'input_shape', 'N/A')}, Output: {getattr(self, 'output_shape', 'N/A')}"
            }
            for attempt in range(3):
                code = self.eval_gen_agent.generate(eval_ctx)
                self._write_file("eval_script.py", code)
                success, out, err = self._run_cmd("eval_script.py", args=["dataset/baseline.npy"])
                if success:
                    try:
                        metrics = json.loads(out)
                        self.baseline_metrics = metrics
                        self._log(f"✅ Eval script validated. Baseline metrics: {metrics}")
                        break
                    except:
                        eval_ctx['feedback'] = f"Invalid JSON: {out}"
                else:
                    eval_ctx['feedback'] = f"Runtime error: {err}"
            else:
                raise RuntimeError("Eval script validation failed")
        else:
            # 验证现有 eval_script
            success, out, err = self._run_cmd("eval_script.py", args=["dataset/baseline.npy"])
            if success:
                self.baseline_metrics = json.loads(out)
                self._log(f"✅ Baseline metrics: {self.baseline_metrics}")
            else:
                raise RuntimeError(f"Existing eval script failed: {err}")
        
        self._log("\n>>> [Phase 0] Complete. Ready for planning stage.")

    def _reset_sandbox_to_phase0_state(self):
        """
        清理 sandbox，仅保留 Phase 0 结束后的核心资产：
        - dataset/ (input.npy, gt_output.npy, baseline.npy)
        - eval_script.py
        - data_gen.py (可选，用于可重现性)
        - gt_code/ (原始参考实现)
        """
        preserve_paths = {
            "dirs": ["dataset", "gt_code"],
            "files": ["eval_script.py", "data_gen.py"]
        }
        
        self._log(">>> [System] Resetting sandbox to Phase 0 state...")
        
        for item in os.listdir(self.sandbox_dir):
            item_path = os.path.join(self.sandbox_dir, item)
            
            # 跳过保留项
            if (os.path.isdir(item_path) and item in preserve_paths["dirs"]) or \
            (os.path.isfile(item_path) and item in preserve_paths["files"]):
                continue
            
            # 清理其他所有内容
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                    self._log(f"  [Cleanup] Removed file: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    self._log(f"  [Cleanup] Removed directory: {item}")
            except Exception as e:
                self._log(f"  [Warning] Failed to remove {item}: {str(e)}")
        
        # 验证关键资产存在性
        for d in preserve_paths["dirs"]:
            if not os.path.exists(os.path.join(self.sandbox_dir, d)):
                raise RuntimeError(f"Critical directory missing after cleanup: {d}")
        for f in preserve_paths["files"]:
            if not os.path.exists(os.path.join(self.sandbox_dir, f)):
                self._log(f"  [Warning] Optional file missing: {f}")
        
        self._log("  ✅ Sandbox reset complete. Ready for next iteration.")

    def _reset_downstream_state(self, ticket: str):
        """
        根据当前 ticket 重置下游状态，确保责任链纯净：
        - Planner → 清空 skeleton + code（算法变更，下游全废）
        - Architect → 清空 code（接口变更，实现全废）
        - Coder → 保留所有（仅修补实现）
        """
        if ticket == "Planner":
            self._log("  [State] Resetting downstream state: skeleton + code (Planner ticket)")
            self.current_skeleton = ""
            self.current_code = ""
            self.function_list = []
        elif ticket == "Architect":
            self._log("  [State] Resetting downstream state: code only (Architect ticket)")
            self.current_code = ""
            # 保留 current_skeleton 供 Architect 参考（带修改意图）
        # Coder: 不重置任何状态（增量修补）
        # Note: If Coder ticket, we rely on patch mode to fix specific parts.
        # But if we fall back to full rebuild, that logic is inside the loop.
    

    def run(self):
        self._setup_sandbox()
        self._phase_0_preparation()
        
        # 初始状态
        feedback = None
        ticket = "Planner" # 默认从 Plan 开始
        
        while self.retry_count < self.max_retries:
            iter_id = self.retry_count + 1
            self._log(f"\n{'='*20} Iteration {iter_id} (Ticket: {ticket}) {'='*20}")
            

            # =========================================================
            # Stage 1: Planning (with Critic Loop)
            # =========================================================
            if ticket == "Planner":
                self._log(">>> [Agent] Planner...")

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
                self._save_artifact(f"iter_{iter_id}_plan.md", self.current_plan)
                ticket = "Architect" # Pass baton

            # =========================================================
            # Stage 2: Architecture
            # =========================================================
            if ticket == "Architect":
                self._log(">>> [Agent] Architect...")

                arch_ctx = self._build_context_with_memory(
                    base_context={
                        'task_desc': self.task_desc,
                        'plan': self.current_plan,
                        'previous_skeleton': self.current_skeleton if self.current_skeleton.strip() else None,
                        'feedback': feedback.get('feedback') if isinstance(feedback, dict) and feedback.get('ticket') == 'Architect' else None
                    },
                    agent_role="Architect",
                    current_ticket="Architect"
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

                self._save_artifact(f"iter_{iter_id}_skeleton.py", self.current_skeleton)
                ticket = "Coder" # Pass baton

            # =========================================================
            # Stage 3: Coding (The Edit Loop)
            # =========================================================
            if ticket == "Coder":
                self._log(">>> [Agent] Coder...")
                
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
                    
                    ctx = {
                        'target_type': task_type,
                        'skeleton_code': self.current_skeleton,
                        'current_full_code': self.current_code,      # ✅ 上一轮完整实现（自身历史）
                        'plan': self.current_plan,
                        'package_list': self.package_list,
                        'feedback': feedback.get('feedback') if is_patch_mode and isinstance(feedback, dict) else None,
                        'fix_target': target if is_patch_mode else None
                    }
                    
                    if task_type == 'function':
                        ctx['target_function'] = task_name
                    
                    # ====== 关键增强：为Coder注入历史失败 ======
                    ctx = self._build_context_with_memory(
                        base_context=ctx,
                        agent_role="Coder",
                        current_ticket="Coder"
                    )

                    # 调用 Coder Agent
                    # implement_and_merge 内部会自动处理 EDIT 逻辑，保留其他代码不变
                    self.current_code = self.coder.implement_and_merge(ctx)
                    
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

                if eval_success and metrics:
                    # Logic: Determine success threshold
                    # 1. PSNR Floor: Ensure minimum quality of 25 dB
                    curr_psnr = metrics.get('psnr', 0)
                    baseline_psnr = self.baseline_metrics.get('psnr', 0)
                    
                    # If baseline is extremely low, 0.8 * baseline is meaningless. 
                    # Enforce a hard floor of 25.0 dB for PSNR.
                    psnr_threshold = max(25.0, baseline_psnr * 0.8)
                    
                    is_success = False
                    
                    if 'psnr' in metrics:
                         if curr_psnr >= psnr_threshold:
                             is_success = True
                             self._log(f"🎉 PSNR Success! {curr_psnr:.2f} >= {psnr_threshold:.2f} (Floor: 25.0, Baseline*0.8: {baseline_psnr*0.8:.2f})")
                         else:
                             self._log(f"⚠️ PSNR Fail: {curr_psnr:.2f} < {psnr_threshold:.2f}")
                    
                    # 2. Handle RMSE logic if PSNR not present (Signal tasks)
                    elif 'rmse' in metrics:
                        curr_rmse = metrics.get('rmse', float('inf'))
                        baseline_rmse = self.baseline_metrics.get('rmse', float('inf'))
                        # RMSE: Lower is better. 
                        # Threshold: allow slightly worse than baseline (1.2x) or impose a ceiling if baseline is bad?
                        # User said "RMSE is similar" -> implies we might want a ceiling.
                        # Assuming 0.1 as a generic loose ceiling for normalized data, but respecting baseline.
                        # If baseline is 0.01, threshold 0.012. If baseline is 1.0 (bad), threshold 0.1? No, 1.2.
                        # Let's stick to relative for RMSE as absolute scale is unknown.
                        rmse_threshold = baseline_rmse * 1.2
                        if curr_rmse <= rmse_threshold:
                            is_success = True
                            self._log(f"🎉 RMSE Success! {curr_rmse:.4f} <= {rmse_threshold:.4f}")
                        else:
                            self._log(f"⚠️ RMSE Fail: {curr_rmse:.4f} > {rmse_threshold:.4f}")

                    if is_success:
                        self._log(f"🎉 SUCCESS! Metrics met threshold.")
                        self._save_snapshot(self.retry_count + 1, "final_success", {
                            "metrics": metrics,
                            "threshold": psnr_threshold if 'psnr' in metrics else rmse_threshold,
                            "solver_code_path": f"iter_{self.retry_count+1:02d}_solver.py"
                        })
                        return True  # EXIT WORKFLOW - No Judge needed
                    else:
                        self._log(f"⚠️ Metrics below threshold.")
                        
                    
                # --- Judge Agent ---
                self._log("\n>>> [Agent] Judge Analyzing Failure Root Cause...")
                judge_ctx = {
                    'logs': logs[-1000:],  # Keep recent context
                    'metrics': metrics,
                    'baseline_metrics': self.baseline_metrics,
                    'current_code_snippet': self.current_code # Optional context
                }
                
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
        self.failure_history.clear()
        return False

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_config_path = os.path.join(base_dir, "config", "config_task_2.yaml")
    llm_config_path = os.path.join(base_dir, "config", "config_llm.yaml")

    # 加载配置
    print("Loading configurations...")
    task_config = load_config(task_config_path)
    llm_config = load_config(llm_config_path)

    # 选择模型（固定使用指定模型，也可扩展为参数化）
    # model_key = "claude-opus-4-5-20251101-thinking"
    # model_key = "gpt-5.2-thinking"
    # model_key = "gemini-3-pro-preview"
    model_key = "cds/Claude-4.6-opus"
    
    if model_key not in llm_config['models']:
        raise ValueError(f"Model '{model_key}' not found in {llm_config_path}")
    
    model_conf = llm_config['models'][model_key]
    print(f"Using LLM Model: {model_key}")

    # 初始化客户端（所有任务复用同一客户端）
    client = OpenAI(
        api_key=model_conf['api_key'],
        base_url=model_conf['base_url']
    )


    workflow = InverseProblemWorkflow(
        task_name = "samsung",
        paper_md_path = "/home/yjh/agentic_reproduce/paper_archive/test.md",
        python_path = "/home/yjh/.conda/envs/ragas/bin/python",
        client = client,
        model_name = model_key,
        root_output_dir = "/home/yjh/agentic_reproduce/paper_sandbox"
    )
    workflow.run()