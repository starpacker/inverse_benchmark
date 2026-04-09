import subprocess
import re
import sys
import threading
import os
import json
import ast
import numpy as np
import builtins
import glob
import openai


# --- 更新后的模板 ---
COMBINED_DECORATOR_TEMPLATE = """
import os as _os_
import sys as _sys_
import functools as _functools_
import dill as _dill_
import time as _time_
import inspect as _inspect_
import json as _json_

# Global Registry to track which functions have already been logged in this session
# Stores tuples of (function_name, parent_function_name)
_META_REGISTRY_ = set()

# Try importing ML libraries safely
try:
    import numpy as _np_
except ImportError:
    _np_ = None
try:
    import torch as _torch_
except ImportError:
    _torch_ = None

# --- Helper: Fix Seeds ---
def _fix_seeds_(seed=42):
    import random
    if _np_: _np_.random.seed(seed)
    random.seed(seed)
    if _torch_:
        _torch_.manual_seed(seed)
        if _torch_.cuda.is_available():
            _torch_.cuda.manual_seed_all(seed)

_fix_seeds_(42)

# --- Helper: Object Analysis (For Metadata) ---
def _analyze_obj_(obj):
    if _torch_ and isinstance(obj, _torch_.Tensor):
        return {{
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device)
        }}
    if _np_ and isinstance(obj, _np_.ndarray):
        return {{
            "type": "numpy.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype)
        }}
    if isinstance(obj, (list, tuple)):
        return {{
            "type": type(obj).__name__,
            "length": len(obj),
            "elements": [_analyze_obj_(item) for item in obj]
        }}
    if hasattr(obj, '__dict__'):
        methods = []
        try:
            for m in dir(obj):
                if m.startswith('_'): continue
                try:
                    attr = getattr(obj, m)
                    if callable(attr):
                        methods.append(m)
                except Exception:
                    continue
        except Exception:
            pass
        return {{
            "type": "CustomObject",
            "class_name": obj.__class__.__name__,
            "public_methods": methods,
            "attributes": list(obj.__dict__.keys())
        }}
    try:
        val_str = str(obj)
    except:
        val_str = "<non-stringifiable>"
    return {{"type": type(obj).__name__, "value_sample": val_str}}

# --- Decorator 1: Metadata Recorder (JSON) ---
# Modified to implement "Log Once" logic
def _record_io_decorator_(save_path="./"):
    def decorator(func, parent_function = None):
        @_functools_.wraps(func)
        def wrapper(*args, **kwargs):
            global _META_REGISTRY_
            
            # Create a unique key for this function context
            func_name = func.__name__
            parent_key = str(parent_function)
            registry_key = (func_name, parent_key)
            
            should_record = False
            
            # 1. Memory Check: If not in memory, we proceed to check/record
            if registry_key not in _META_REGISTRY_:
                should_record = True
            
            result = None
            inputs_meta = {{}}
            
            # 2. Analyze Inputs (Only if we plan to record)
            if should_record:
                try:
                    sig = _inspect_.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for name, value in bound_args.arguments.items():
                        inputs_meta[name] = _analyze_obj_(value)
                except Exception as e:
                    inputs_meta = {{"error": f"Analysis failed: {{e}}"}}

            # 3. Execute Original Function
            result = func(*args, **kwargs)
            
            # 4. Analyze Output & Save (Only if we plan to record)
            if should_record:
                try:
                    output_meta = _analyze_obj_(result)
                except Exception:
                    output_meta = "Analysis failed"
                
                try:
                    # Resolve Path
                    final_path = save_path
                    if not final_path.endswith('.json'):
                        if not _os_.path.exists(final_path):
                            _os_.makedirs(final_path, exist_ok=True)
                        if parent_function == None:
                            final_path = _os_.path.join(final_path, f"IO_meta_{{func_name}}.json")
                        else:
                            final_path = _os_.path.join(final_path, f"IO_meta_parent_function_{{parent_function}}_{{func_name}}.json")
                    
                    # Ensure directory exists
                    dir_name = _os_.path.dirname(final_path)
                    if dir_name and not _os_.path.exists(dir_name):
                         _os_.makedirs(dir_name, exist_ok=True)
                    
                    # Load Existing Data
                    existing_data = []
                    file_exists = _os_.path.exists(final_path)
                    if file_exists:
                        try:
                            with open(final_path, 'r') as f:
                                existing_data = _json_.load(f)
                        except: pass
                    
                    # === FILE DEDUPLICATION ===
                    # Check if this function is already documented in the file
                    # This handles the case where code is re-run (persisted across sessions)
                    already_in_file = False
                    for entry in existing_data:
                        if entry.get("function_name") == func_name:
                            # We can add stricter checks here (e.g. input args match) if needed
                            # For now, we assume one schema per function name per file.
                            already_in_file = True
                            break
                    
                    if not already_in_file:
                        func_schema = {{
                            "function_name": func_name,
                            "inputs": inputs_meta,
                            "output": output_meta
                        }}
                        existing_data.append(func_schema)
                        
                        with open(final_path, 'w') as f:
                            _json_.dump(existing_data, f, indent=4)
                        print(f"  [Metadata] Recorded schema for: {{func_name}}")
                    
                    # Mark as recorded in memory so we don't check file again this session
                    _META_REGISTRY_.add(registry_key)

                except Exception as e:
                    print(f"  [Metadata] Warning: {{e}}")

            # 5. Recursive Wrapping
            # CRITICAL: This must happen OUTSIDE the 'should_record' block.
            # Even if we don't log metadata (because it's duplicate), we MUST
            # wrap the result if it's a function, so subsequent calls are caught.
            # 增加检查：如果结果是 PyTorch Module,则不要递归包装
            is_torch_module = False
            if _torch_ and isinstance(result, _torch_.nn.Module):
                is_torch_module = True
            if callable(result) and not isinstance(result, type):
                 # print(f"  [Auto-Wrap] Wrapping returned closure: {{getattr(result, '__name__', 'unknown')}}")
                 return decorator(result, parent_function = func_name)

            return result
        return wrapper
    return decorator

# --- Decorator 2: Payload Capture (Pickle) ---
# (Kept mostly same, assuming you still want data per-call. 
#  If you also want data capture only once, similar logic applies, 
#  but usually data capture is needed for datasets, so we keep it per-call)
def _data_capture_decorator_(func, parent_function = None):
    @_functools_.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        try:
            out_dir = r"{output_dir}"
            if not _os_.path.exists(out_dir):
                _os_.makedirs(out_dir, exist_ok=True)
            
            func_name = func.__name__
            # Use timestamp to allow multiple data captures without overwriting
            # timestamp = str(_time_.time()).replace('.', '_')
            
            if parent_function == None: 
                save_path = _os_.path.join(out_dir, f"standard_data_{{func_name}}.pkl")
            else:
                save_path = _os_.path.join(out_dir, f"standard_data_parent_{{parent_function}}_{{func_name}}.pkl")
            
            def detach_recursive(obj):
                if hasattr(obj, 'detach'): return obj.detach()
                if isinstance(obj, list): return [detach_recursive(x) for x in obj]
                if isinstance(obj, tuple): return tuple(detach_recursive(x) for x in obj)
                if isinstance(obj, dict): return {{k: detach_recursive(v) for k, v in obj.items()}}
                return obj

            payload = {{
                "func_name": func_name,
                "args": detach_recursive(args),
                "kwargs": detach_recursive(kwargs),
                "output": detach_recursive(result)
            }}

            with open(save_path, "wb") as f:
                _dill_.dump(payload, f)
            
        except Exception as e:
            # print(f"  [DataCapture] Warning: {{e}}")
            pass
        
        # 增加检查：如果结果是 PyTorch Module，则不要递归包装
        is_torch_module = False
        if _torch_ and isinstance(result, _torch_.nn.Module):
            is_torch_module = True
        if callable(result) and not isinstance(result, type):
             return _data_capture_decorator_(result, parent_function = func_name)

        return result
    return wrapper
"""

def inject_double_decorators(original_code: str, output_dir: str, metadata_path: str, target_functions: list) -> str:
    """
    Injects two decorators into the target functions.
    Handles 'Log Once' logic for Metadata.
    """
    try:
        tree = ast.parse(original_code)
    except SyntaxError as e:
        raise ValueError(f"Syntax Error in original code: {e}")

    abs_output_dir = os.path.abspath(output_dir).replace("\\", "/")
    abs_metadata_path = os.path.abspath(metadata_path).replace("\\", "/")

    # Format template
    decorator_code = COMBINED_DECORATOR_TEMPLATE.format(
        output_dir=abs_output_dir
    )
    decorator_ast = ast.parse(decorator_code)
    
    # Insert helper code blocks after future imports
    insert_idx = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Str, ast.Constant)):
            insert_idx = i + 1
            continue
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            insert_idx = i + 1
            continue
        break
    
    tree.body[insert_idx:insert_idx] = decorator_ast.body

    class DualDecoratorInjector(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in target_functions:
                # 1. Add Data Capture (Pickle)
                if '_data_capture_decorator_' not in str(node.decorator_list):
                    d1 = ast.Name(id='_data_capture_decorator_', ctx=ast.Load())
                    node.decorator_list.insert(0, d1)

                # 2. Add Record IO (JSON)
                if '_record_io_decorator_' not in str(node.decorator_list):
                    d2 = ast.Call(
                        func=ast.Name(id='_record_io_decorator_', ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(
                                arg='save_path', 
                                value=ast.Constant(value=abs_metadata_path)
                            )
                        ]
                    )
                    node.decorator_list.insert(0, d2)
            return node

    transformer = DualDecoratorInjector()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    if hasattr(ast, 'unparse'):
        return ast.unparse(tree)
    else:
        import astor
        return astor.to_source(tree)
    
def run_command_streaming(command, cwd, log_file_path, timeout=None):
    print(f"  [Exec] Streaming output to console and {log_file_path}...")
    
    f_log = open(log_file_path, 'w', encoding='utf-8')
    
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,     
        stderr=subprocess.STDOUT,
        text=True,                  
        bufsize=1,                  
        encoding='utf-8',
        errors='replace'            
    )
    
    def stream_reader(proc, log_file):
        try:
            for line in proc.stdout:
                # print(line) -> REMOVE THIS (Double printing)
                sys.stdout.write(line) # Writes the line exactly as is
                log_file.write(line)
                # log_file.flush() -> Not strictly necessary every line if bufsize is managed, but safe to keep
        except Exception as e:
            # Print the error instead of passing silently so you know why it fails
            sys.stderr.write(f"\n[Stream Error] {e}\n")

    t = threading.Thread(target=stream_reader, args=(process, f_log))
    t.daemon = True
    t.start()
    
    result_code = -1
    status_msg = ""

    try:
        process.wait(timeout=timeout)
        result_code = process.returncode
        
        # Determine status message based on code
        if result_code == 0:
            status_msg = "Success"
        else:
            status_msg = "Finished with Errors"
        
    except subprocess.TimeoutExpired:
        print(f"\n  [Error] Process timed out after {timeout} seconds. Killing...")
        process.kill()
        result_code = -1
        status_msg = "TIMEOUT_EXPIRED"
        
    except Exception as e:
        print(f"\n  [Error] Exception during execution: {e}")
        process.kill()
        result_code = -1
        status_msg = str(e)
        
    finally:
        # CRITICAL FIX: Remove timeout or make it very long.
        # We must allow the reader thread to finish draining the pipe
        # after the process has exited.
        t.join() 
        f_log.close()

    return result_code, status_msg

def is_generation_complete(finish_reason, content):
    # 1. 明确需要继续的唯一情况
    if finish_reason == "length":
        print("finish reason:",finish_reason,False)
        return False

    # 2. 明确异常中断
    if finish_reason == "content_filter":
        print("finish reason:",finish_reason,False)
        raise RuntimeError("Generation blocked by content filter.")

    if finish_reason is None:
        print("Status: Finish reason is None (Connection interrupted or Timed out).")
        # 如果内容也有，通常说明是被掐断了，最好也返回 False 让它尝试续写（虽然效果不一定好，但比报错强）
        return False

    # 3. DeepSeek / OpenAI 正常结束
    if content and content.strip():
        print("finish reason:",finish_reason,True)
        return True

    # 4. 非 length 但没内容 → 异常情况
    return False

def get_full_response(client, model, messages, max_loops=5,max_tokens=32768, **kwargs):
    """
    通用自动续写函数（全员非流式）。
    无论 DeepSeek 还是 Writer 模型，都统一使用标准 HTTP 请求。
    """
    full_content = ""
    current_messages = list(messages)
    
    for loop_i in range(max_loops):
        print(f"Loop {loop_i+1}: Generating with {model}...")

        # =====================================================
        # 参数预处理
        # =====================================================
        # 复制一份参数，避免修改影响下一次循环
        request_kwargs = kwargs.copy()

        # 针对 DeepSeek 模型，如果用户没传 extra_body，我们给个默认值开启 thinking

        request_kwargs["max_tokens"] = max_tokens

        # =====================================================
        # 统一调用 (非流式)
        # =====================================================
        try:
            response = client.chat.completions.create(
                model=model,
                messages=current_messages,
                **request_kwargs # 透传所有参数 (temperature, max_tokens, etc.)
            )
        except Exception as e:
            print(f"Error during API call: {e}")
            raise e

        # 获取内容
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        # =====================================================
        # 合并结果与截断判断
        # =====================================================
        if content:
            full_content += content
        
        
        if is_generation_complete(finish_reason, content):
            print("Generation complete.")
            return full_content
        
        print(f"\n[Truncated] Output stopped at {len(full_content)} chars. Requesting continuation...")
        
        # 将上一轮回复加入历史，准备下一轮
        current_messages.append({"role": "assistant", "content": content})
        current_messages.append({
            "role": "user",
            "content": """You stopped because of the length limit.
        Continue the code EXACTLY from where it ended.

        STRICT RULES:
        - Do NOT reprint any function or class headers
        - Do NOT repeat previous lines
        - Do NOT add explanations or comments
        - Output ONLY valid code
        """
        })
    
    print("Warning: Max continuation loops reached.")
    return full_content

def _extract_code_from_markdown(text: str) -> str:
    """
    Robustly extract code from LLM response.
    Priority:
    1. ```python ... ``` blocks (Case insensitive)
    2. ``` ... ``` generic blocks
    3. Raw text fallback (if explicit Python keywords are detected and no blocks found)
    """
    if not text or not text.strip():
        return ""

    # --- 策略 1: 标准 Python Markdown 块 ---
    # re.IGNORECASE: 兼容 ```Python, ```PYTHON, ```py
    # re.DOTALL: 让 . 匹配换行符，抓取跨行代码
    python_pattern = r"```(?:python|py)\s*([\s\S]*?)```"
    python_matches = re.findall(python_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if python_matches:
        # 拼接多个块（防止 LLM 把 import 和 def 分在两个块里）
        return "\n\n".join(python_matches).strip()
    
    # --- 策略 2: 通用 Markdown 块 ---
    generic_pattern = r"```\s*([\s\S]*?)```"
    generic_matches = re.findall(generic_pattern, text, re.DOTALL)
    
    if generic_matches:
        return "\n\n".join(generic_matches).strip()

    # --- 策略 3: 兜底机制 (Raw Text Fallback) ---
    # 如果没找到 Markdown 框，但文本里包含大量 Python 关键字，
    # 我们假设 LLM 忘了打标签，直接返回原始内容。
    
    # 定义特征关键词
    indicators = [
        "import ", "from ", "def ", "class ", 
        "if __name__", "return ", "print("
    ]
    
    # 统计出现了几个特征词
    hit_count = sum(1 for k in indicators if k in text)
    
    # 阈值判断：如果包含至少 2 个特征词，且文本长度适中，认为它是裸代码
    if hit_count >= 2:
        print("  [Warn] No markdown tags found. Detected raw code, returning full text.")
        
        # 可选：简单的清洗，尝试去掉开头的 "Here is the code:" 等废话
        lines = text.split('\n')
        
        # 寻找代码起始行
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 跳过空行
            if not stripped:
                continue
            # 如果这一行是以代码关键字开头，或者是装饰器，或者是注释，我们就认为代码开始了
            if stripped.startswith(('import ', 'from ', 'def ', 'class ', '@', 'if __name__', '#')):
                start_idx = i
                break
        
        if start_idx > 0:
            print(f"  [Auto-Clean] Stripped {start_idx} lines of conversational text.")
            return "\n".join(lines[start_idx:]).strip()
             
        return text.strip()

    # 确实没找到代码
    return ""

def load_code_from_file(filepath):
    """Reads the raw source code from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_processed_functions(working_folder):
    """
    从 working_folder 中读取 final_function_list.json。
    Args:
        working_folder (str): 包含 json 文件的文件夹路径。
    Returns:
        list: 包含函数名的列表。如果文件不存在或解析失败，返回空列表。
    """
    json_path = os.path.join(working_folder, "final_function_list.json")
    
    if not os.path.exists(json_path):
        print(f"[Warning] File not found: {json_path}")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            function_list = json.load(f)
        
        if isinstance(function_list, list):
            print(f"[Success] Loaded {len(function_list)} functions from {json_path}")
            return function_list
        else:
            print(f"[Error] JSON content is not a list.")
            return []
            
    except json.JSONDecodeError:
        print(f"[Error] Failed to decode JSON from {json_path}")
        return []
    except Exception as e:
        print(f"[Error] Unexpected error reading file: {e}")
        return []

def generate_sabotaged_code_deterministic(original_code, func_name, sabotage_type):
    """
    Deterministic mutator using Python's AST.
    Safely replaces function body while keeping imports and signature intact.
    """
    try:
        tree = ast.parse(original_code)
    except SyntaxError:
        return original_code

    class BodyReplacer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name != func_name:
                return node
            if node.name == func_name:
                new_body = []
                # Keep docstring if exists
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    new_body.append(node.body[0])

                if sabotage_type == 'return_zero':
                    return_count = 1  # 默认为 1 (return 0)
                    # 遍历原函数体，寻找第一个有效的 return 语句
                    # 使用 ast.walk 可以找到嵌套在 if/for/try 内部的 return
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value is not None:
                            # 如果是 return a, b, c 这种形式，AST 会解析为 Tuple
                            if isinstance(child.value, ast.Tuple):
                                return_count = len(child.value.elts)
                            else:
                                # 否则 (return x, return [1,2], return func()) 都视为返回 1 个对象
                                return_count = 1
                            break  # 找到一个就停止，假设函数返回形状一致
                    # --- 构建新的 return 语句 ---
                    if return_count > 1:
                        # 构造 (0, 0, ..., 0)
                        zeros = [ast.Constant(value=0) for _ in range(return_count)]
                        ret_val = ast.Tuple(elts=zeros, ctx=ast.Load())
                    else:
                        # 构造 0
                        ret_val = ast.Constant(value=0)
                    
                    new_body.append(ast.Return(value=ret_val))
                    # new_body.append(ast.Return(value=ast.Constant(value=0)))
                
                elif sabotage_type == 'assert_error':
                    new_body.append(ast.Assert(
                        test=ast.Compare(
                            left=ast.Constant(value=0),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value=1)]
                        ),
                        msg=ast.Constant(value="Intentional Crash")
                    ))
                node.body = new_body
                return node
            return node

    transformer = BodyReplacer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except AttributeError:
        import astor 
        return astor.to_source(new_tree)

def validate_test_reliability(working_folder, func_name, test_filename, standard_code, **kwargs):
    """
    Performs Meta-Testing (Mutation Testing) on the generated test_{func}.py.
    
    NOTE: This function assumes the test ALREADY PASSED the standard_code check in the main loop.
    It checks if the test correctly FAILS when the code is broken.
    
    Returns (bool, str): (Passed/Failed, Reason)
    """
    print(f"    [Meta-Test] Verifying reliability of {test_filename}...")
    
    agent_filename = f"agent_{func_name}.py"
    agent_path = os.path.join(working_folder, agent_filename)
    
    # Use target_python_path if provided, otherwise default to python
    python_exec = "python"
    if 'target_python_path' in kwargs and kwargs['target_python_path']:
        python_exec = kwargs['target_python_path']
        
    test_cmd = f"{python_exec} {test_filename}"
    
    # We skip Scenario 1 (Standard Code) because the main loop already verified it passes.
    
    # --- Scenario 2: Return 0 (Expect FAIL) ---
    print("      -> Scenario 2: Agent returns 0 (Should FAIL)")
    code_return_zero = generate_sabotaged_code_deterministic(standard_code, func_name, 'return_zero')
    
    # Ensure directory exists (defensive programming)
    os.makedirs(os.path.dirname(agent_path), exist_ok=True)
    
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.write(code_return_zero)
        
    res2 = subprocess.run(test_cmd, shell=True, cwd=working_folder, capture_output=True, text=True)
    
    # If returncode is 0, it means the test passed even though the code returned 0 -> BAD TEST
    if res2.returncode == 0:
        # Restore standard code before returning
        with open(agent_path, 'w', encoding='utf-8') as f:
            f.write(standard_code)
        return False, "Failed Scenario 2. Test passed even when agent returned 0 (False Positive)."

    # --- Scenario 3: Assert 0==1 (Expect FAIL) ---
    print("      -> Scenario 3: Agent raises assert error (Should FAIL)")
    code_assert_error = generate_sabotaged_code_deterministic(standard_code, func_name, 'assert_error')
    
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.write(code_assert_error)
    
    res3 = subprocess.run(test_cmd, shell=True, cwd=working_folder, capture_output=True, text=True)
    
    # If returncode is 0, it means the test swallowed the crash -> BAD TEST
    if res3.returncode == 0:
        # Restore standard code before returning
        with open(agent_path, 'w', encoding='utf-8') as f:
            f.write(standard_code)
        return False, "Failed Scenario 3. Test passed even when agent crashed (False Positive)."

    # --- Restore Standard Code ---
    # Always leave the agent in a working state
    with open(agent_path, 'w', encoding='utf-8') as f:
        f.write(standard_code)

    print("      -> Meta-Test Passed: Test script correctly discriminates working vs broken code.")
    return True, "Meta-Test Passed"

def validate_semantic_similarity(reference_code, generated_code, client, embedding_model):
    """
    Calculates the Cosine Similarity between the embeddings of two code snippets.
    Returns a score between 0 and 100.
    """
    # 1. Get Embeddings
    # 'text-embedding-3-small' is cheap and very good at code semantics
    response = client.embeddings.create(
        input=[reference_code, generated_code],
        model=embedding_model
    )
    
    vec_ref = response.data[0].embedding
    vec_gen = response.data[1].embedding
    
    # 2. Compute Cosine Similarity
    # Similarity = (A . B) / (||A|| * ||B||)
    dot_product = np.dot(vec_ref, vec_gen)
    norm_ref = np.linalg.norm(vec_ref)
    norm_gen = np.linalg.norm(vec_gen)
    
    similarity = dot_product / (norm_ref * norm_gen)
    
    return similarity * 100

def validate_section_clarity(explanation_text, ground_truth_code, coder_client, embedding_client,coder_model="gpt-4o", embedding_model = None):
    """
    Returns a score (0-100) indicating how well the explanation describes the code.
    """
    # Prompt the Coder LLM to reconstruct code based ONLY on the explanation
    coder_prompt = f"""
    You are an expert Python coder. 
    Read the following technical explanation of one or several functions:
    
    \"\"\"
    {explanation_text}
    \"\"\"
    
    Based STRICTLY on this explanation, write the corresponding Python functions.
    - Use specific variable names if mentioned in the text.
    - If the text is vague, make your best guess (this will lower the score, which is intended).
    - Output ONLY the python code block.
    - If the technical explanation mentioned several functions, you should implement all of them.
    """
    # print("%"*15)
    # print("coder_prompt", coder_prompt)
    response = coder_client.chat.completions.create(
        model=coder_model,
        messages=[{"role": "user", "content": coder_prompt}],
        temperature=0.0
    )
    
    reconstructed_code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    print("*"*15)
    print("ground_truth_code", ground_truth_code)
    print("#"*15)
    print("reconstructed_code", reconstructed_code)
    # Calculate Similarity Ratio
    # Normalize whitespace to ensure fair comparison
    similarity_score = validate_semantic_similarity(reconstructed_code, ground_truth_code, embedding_client, embedding_model)
    
    return similarity_score, reconstructed_code

def validate_section_clarity_by_eval_code(target_python_path, explanation_text, coder_client, coder_model, eval_working_folder, title, prompts_config, function_name = None):
    """
    Returns a score (0-100) indicating how well the explanation describes the code.
    """
    # Prompt the Coder LLM to reconstruct code based ONLY on the explanation
    coder_prompt_template = prompts_config['validator']['reconstruct_code']
    coder_prompt = coder_prompt_template.format(explanation_text=explanation_text)

    # response = coder_client.chat.completions.create(
    #     model=coder_model,
    #     messages=[{"role": "user", "content": coder_prompt}],
    #     temperature=0.0,
    #     timeout = 30000
    # )
    
    # reconstructed_code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()

    raw_content = get_full_response(
        client=coder_client,
        model=coder_model,
        messages=[{"role": "user", "content": coder_prompt}],
        temperature=0.0,
        timeout=30000  # timeout 会作为 kwargs 透传进去
    )

    # 2. 清洗 Markdown 标记
    reconstructed_code = raw_content.replace("```python", "").replace("```", "").strip()
    if title == "Data Preprocessing":
        function_name = "load_and_preprocess_data"
    elif title == "Forward Operator Implementation":
        function_name = "forward_operator"
    elif title == "Core Loop of Inverse Algorithm (Focus!)":
        function_name = "run_inversion"
    elif title == "Definition and Implementation of Evaluation Metrics":
        function_name = "evaluate_results"
    else:
        function_name = function_name
    gt_code = f"agent_{function_name}.py"
    gt_code_path = os.path.join(eval_working_folder, gt_code)
    gt_save_path =  os.path.join(eval_working_folder, f"gt_{function_name}.py")
    if os.path.exists(gt_save_path):
        pass
    else:
        with open(gt_code_path, 'r', encoding='utf-8', errors='ignore') as f:
            gt_code = f.read()
        with open(gt_save_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(gt_code)
    with open(gt_code_path, 'w', encoding='utf-8') as f:
        f.write(reconstructed_code)
    test_filename = f"test_{function_name}.py"

    test_cmd = f"{target_python_path} {test_filename}"
    execution_log_path = os.path.join(eval_working_folder, "log","execution_log.txt")
    return_code, _ = run_command_streaming(
                test_cmd,
                cwd=eval_working_folder,
                log_file_path=execution_log_path,
                timeout=None 
            )
    
    if os.path.exists(execution_log_path):
        with open(execution_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            test_output_content = f.read()

    if return_code == 0:
        status = "success"
        hint_for_tutorial = None
        
    else:
        print("  -> FAILED (Runtime Error or Assertion Error)")
        # print(f"STDERR snippet: {test_res.stderr[:300]}...")
        # 提取错误日志的最后一部分给 LLM
        # 避免把几百兆的训练日志全喂给 LLM
        max_chars = 2000
        error_tail = test_output_content[-max_chars:] if len(test_output_content) > max_chars else test_output_content
        
        # print(f"Error Tail: {error_tail[-200:]}")
        
        # 将错误信息存入变量，供下一次循环构建 Prompt 使用
        test_res_stderr = f"Execution failed. Log tail:\n\n{error_tail}"
        status = "failed"
        hint_for_tutorial = test_res_stderr
    # print("hint_for_tutorial", hint_for_tutorial)
    print("hint", hint_for_tutorial)
    # assert 1==0
   
    return status, hint_for_tutorial, reconstructed_code

def analyze_code_structure(file_path):
    try:
        with open(file_path, "r") as source:
            tree = ast.parse(source.read())
    except FileNotFoundError:
        # print(f"Error: Could not find file '{file_path}'")
        return

    # print(f"--- Analysis of {file_path} ---\n")

    # 1. Find Standalone (Global) Functions
    # print("GLOBAL FUNCTIONS:")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    # if functions:
    #     for func in functions:
    #         print(f"{func.name}")
    # else:
    #     print("  None")

    # 2. Find Classes and their Methods
    # print("\nCLASSES & METHODS:")
    # classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    
    # if classes:
    #     for cls in classes:
    #         print(f"  class {cls.name}:")
    #         methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
    #         for method in methods:
    #             print(f"    - def {method.name}(...)")
    #         print("") # Empty line between classes
    # else:
    #     print("  None")
    function_names = []
    for func in functions:
            # print(f"{func.name}")
            function_names.append(f"{func.name}")
    # print(function_names)
    return function_names

class DependencyExtractor:
    def __init__(self, source_code: str):
        self.tree = ast.parse(source_code)
        self.source_code = source_code
        self.source_lines = source_code.splitlines()

        # Indexed definitions
        self.definitions = {}        # name -> FunctionDef | ClassDef
        self.constants = {}          # name -> Assign
        self.import_map = {}         # name -> container node
        self.wildcard_imports = []   # list of container nodes

        # Global setup (sys.path, config calls, etc.)
        self.global_setup_nodes = []

        # Dependency traversal
        self.collected_nodes = set()
        self.visited = set()

        self._index_file()

    # ------------------------------------------------------------------
    # Index phase
    # ------------------------------------------------------------------

    def _index_file(self):
        """Index top-level definitions, imports, constants, and global setup."""
        for node in self.tree.body:

            # 1. 遇到 __main__ 守卫：直接跳过
            # 我们假设 main 块包含的是执行逻辑而非定义，提取依赖时不应包含它。
            if self._is_main_guard(node):
                continue

            # 2. 检测全局环境设置 (sys.path, warnings, matplotlib settings 等)
            # 或者顶层的函数调用 (例如 torch.set_default_dtype)
            # 这些通常是代码运行的前提，建议保留
            if self._is_global_setup_or_side_effect(node):
                self.global_setup_nodes.append(node)
                self._collect_setup_dependencies(node)
                continue

            # 3. 常规索引 (函数定义、类定义、常量赋值、导入)
            self._scan_node_for_defs_and_imports(node, container=node)

    def _is_global_setup_or_side_effect(self, node):
        """
        判断节点是否为全局设置或副作用。
        包括：sys.path 修改，或者顶层的函数调用（如 torch.set_default_dtype）。
        """
        # 显式的 sys.path 修改
        if self._is_sys_path_modification(node):
            return True
        
        # 顶层表达式 (通常是函数调用，如 torch.set_default_dtype(...))
        # 注意：文档字符串 (Docstring) 也是 Expr，AST 解析时通常作为独立的 Expr 节点存在，
        # 如果是纯字符串常量的 Expr，我们通常忽略，避免输出无用的字符串。
        if isinstance(node, ast.Expr):
            # 排除纯字符串 (Docstrings)
            if isinstance(node.value, (ast.Str, ast.Constant)):
                return False
            return True
            
        return False

    def _collect_setup_dependencies(self, node):
        """
        对于全局设置节点，扫描其使用的依赖 (如 sys, os, torch)。
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                name = child.id
                if name not in dir(builtins):
                    self.get_dependencies(name)

    def _is_main_guard(self, node):
        """Detects: if __name__ == '__main__':"""
        if not isinstance(node, ast.If):
            return False

        test = node.test
        if not isinstance(test, ast.Compare):
            return False

        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            return False

        left = test.left
        right = test.comparators[0]

        def is_name(n):
            return isinstance(n, ast.Name) and n.id == "__name__"

        def is_main_str(n):
            return (
                isinstance(n, ast.Constant) and n.value == "__main__"
            ) or (
                isinstance(n, ast.Str) and n.s == "__main__"
            )

        return (is_name(left) and is_main_str(right)) or \
               (is_name(right) and is_main_str(left))

    def _is_sys_path_modification(self, node):
        """Heuristic detection of sys.path modification."""
        for child in ast.walk(node):
            # sys.path.append / insert / extend / remove
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Attribute):
                    if func.attr in {"append", "insert", "extend", "remove"}:
                        if self._is_sys_path(func.value):
                            return True
            # sys.path += ...
            if isinstance(child, ast.AugAssign):
                if self._is_sys_path(child.target):
                    return True
            # sys.path = ...
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if self._is_sys_path(target):
                        return True
        return False

    def _is_sys_path(self, node):
        """Check if node represents sys.path."""
        return (
            isinstance(node, ast.Attribute) and
            node.attr == "path" and
            isinstance(node.value, ast.Name) and
            node.value.id == "sys"
        )

    def _scan_node_for_defs_and_imports(self, node, container):
        """
        Recursively scan node to find: imports, definitions, constants
        """
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._register_import(node, container)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.definitions[node.name] = node

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.constants[target.id] = node
        
        # 递归扫描 If/Try/With 等结构内部的定义
        # 注意：这里我们不再处理 __main__，因为在 _index_file 已经被过滤了
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for child in ast.iter_child_nodes(node):
                self._scan_node_for_defs_and_imports(child, container)

    def _register_import(self, node, container):
        """Register imports, mapping imported names to their container."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.import_map[name] = container
                if not alias.asname:
                    self.import_map[alias.name.split(".")[0]] = container

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    self.wildcard_imports.append(container)
                else:
                    name = alias.asname if alias.asname else alias.name
                    self.import_map[name] = container

    # ------------------------------------------------------------------
    # Dependency traversal
    # ------------------------------------------------------------------

    def get_dependencies(self, target_name: str):
        """Recursively resolve dependencies for a given symbol name."""
        if target_name in self.visited:
            return
        self.visited.add(target_name)

        # 1. Function / class definition
        if target_name in self.definitions:
            node = self.definitions[target_name]
            self.collected_nodes.add(node)
            self._scan_node_for_usage(node)
            return

        # 2. Global constant
        if target_name in self.constants:
            node = self.constants[target_name]
            self.collected_nodes.add(node)
            self._scan_node_for_usage(node)
            return

        # 3. Import
        if target_name in self.import_map:
            node = self.import_map[target_name]
            self.collected_nodes.add(node)
            self._scan_node_for_usage(node)
            return

        # 4. Fallback: wildcard imports
        for node in self.wildcard_imports:
            self.collected_nodes.add(node)

    def _scan_node_for_usage(self, node):
        """Scan node body to discover referenced names."""
        for child in ast.walk(node):

            # Name usage
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                name = child.id
                if name not in dir(builtins):
                    self.get_dependencies(name)

            # Decorators
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                for dec in child.decorator_list:
                    if isinstance(dec, ast.Name):
                        self.get_dependencies(dec.id)
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                        self.get_dependencies(dec.func.id)

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def generate_code(self) -> str:
        """Reconstruct extracted source code."""
        output_nodes = set()

        # Always include global setup
        for node in self.global_setup_nodes:
            output_nodes.add(node)

        # Include collected dependencies
        for node in self.collected_nodes:
            output_nodes.add(node)

        # Sort by source order to maintain readability and execution flow
        sorted_nodes = sorted(output_nodes, key=lambda n: n.lineno)

        output = []
        for node in sorted_nodes:
            segment = ast.get_source_segment(self.source_code, node)
            if segment:
                output.append(segment)
                output.append("")

        return "\n".join(output)

def get_io_metadata(base_dir, func_name): 
    parent_path = os.path.join(base_dir, f"IO_meta_{func_name}.json") 
    if not os.path.exists(parent_path): 
        print(f"Warning: Parent metadata not found at {parent_path}") 
        return {}, None
    with open(parent_path, 'r') as f:
        parent_io = json.load(f)

    # 2. Search for Child Metadata (Factory Pattern)
    # Pattern: IO_meta_parent_function_{parent_name}_{child_name}.json
    # We use glob because we might not know the exact {child_name}
    search_pattern = os.path.join(base_dir, f"IO_meta_parent_function_{func_name}_*.json")
    # IO_meta_parent_function_{{parent_function}}_{{func_name}}.json
    found_files = glob.glob(search_pattern)

    child_io = None
    if found_files:
        child_path = found_files[0] # Take the first match
        print(f"   -> Detected Factory Pattern! Loading child info: {os.path.basename(child_path)}")
        with open(child_path, 'r') as f:
            child_io = json.load(f)

    return parent_io, child_io

def load_code_and_imports(folder, filename_suffix):
        """
        尝试加载 gt_ 或 agent_ 文件。
        返回: (source_code, extracted_imports)
        """
        gt_path = os.path.join(folder, f"gt_{filename_suffix}")
        agent_path = os.path.join(folder, f"agent_{filename_suffix}")
        
        target_path = None
        if os.path.exists(gt_path):
            target_path = gt_path
        elif os.path.exists(agent_path):
            target_path = agent_path
            
        if target_path:
            code = load_code_from_file(target_path)
            imports = extract_imports_from_code(code)
            return code, imports
        else:
            return None, ""

def extract_imports_from_code(code_source):
    """
    使用 AST 解析代码，提取所有的 import 语句（保留原始格式）。
    """
    if not code_source:
        return ""
    
    try:
        tree = ast.parse(code_source)
    except SyntaxError:
        # 如果代码本身有语法错误，无法解析 AST，则返回空或做简单处理
        return ""

    import_statements = []
    lines = code_source.splitlines()

    for node in ast.walk(tree):
        # 筛选出 Import 和 ImportFrom 节点
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # 获取该节点在源码中的起始行和结束行
            # lineno 从 1 开始，所以列表切片要 -1
            start_line = node.lineno - 1
            end_line = node.end_lineno
            
            # 提取完整的代码块（包括多行 import）
            stmt = "\n".join(lines[start_line:end_line])
            import_statements.append(stmt)
            
    return "\n".join(import_statements)

class DataSummarizer:
    """
    Parses rich metadata (types, shapes, value_samples) into LLM-friendly descriptions.
    """
    @staticmethod
    def summarize(info):
        # 1. Handle "value_sample" (e.g., Integers, Strings, simple Types)
        if isinstance(info, dict) and "value_sample" in info:
            return f"{info.get('type')} (Example: {info['value_sample']})"

        # 2. Handle "CustomObject" (e.g., nufft_ob)
        if isinstance(info, dict) and info.get("type") == "CustomObject":
            class_name = info.get("class_name", "UnknownClass")
            methods = info.get("public_methods", [])
            attributes = info.get("attributes", [])
            
            # Heuristic: If it has 'load_state_dict', it's likely a PyTorch Module
            if "load_state_dict" in methods:
                return f"PyTorch Module instance ({class_name})"
            
            # For other Custom Objects, list the API surface (Methods & Attributes)
            methods_str = ", ".join(methods) if methods else "None"
            attrs_str = ", ".join(attributes) if attributes else "None"
            
            return (f"Custom Object ({class_name}) | "
                    f"Methods: [{methods_str}] | "
                    f"Attributes: [{attrs_str}]")
        
        if isinstance(info, dict) and "type" not in info and "value_sample" not in info:
            # 这可能是一个普通的 kwargs 字典
            summarized_dict = {k: DataSummarizer.summarize(v) for k, v in info.items()}
            return f"Dict with keys: {list(summarized_dict.keys())}, values: {summarized_dict}"

        # 3. Handle Lists with "elements" (e.g., cphase_ind_list)
        if isinstance(info, dict) and info.get("type") == "list" and "elements" in info:
            elements = info["elements"]
            count = len(elements)
            if count == 0:
                return "Empty List"
            
            # Check if all elements are Tensors
            first_elem = elements[0]
            if first_elem.get("type") == "torch.Tensor":
                # Summarize as a group if uniform
                shape = first_elem.get("shape", "?")
                dtype = first_elem.get("dtype", "?")
                return f"List of {count} Tensors (Shape: {shape}, Dtype: {dtype})"
            
            return f"List of {count} items (Type: {first_elem.get('type')})"

        # 4. Handle Direct Tensor Definitions (e.g., ktraj_vis)
        if isinstance(info, dict) and info.get("type") == "torch.Tensor":
            return f"Tensor(shape={info.get('shape')}, dtype={info.get('dtype')}, device={info.get('device')})"

        # Fallback
        return str(info)

class LLMDocGenerator:
    def __init__(self, api_key, base_url,model="gpt-4"):
        self.client = openai.OpenAI(api_key=api_key, base_url = base_url)
        self.model = model

    def generate_docs(self, source_code, parent_io_info, test_code, child_io_info=None):
        # --- KEY CHANGE HERE ---
        # We updated the system prompt to enforce Google Style and Shape Generalization
        system_prompt = (
            "You are an expert Python documentation generator for scientific code. "
            "Your task is to analyze a function, its concrete input/output trace data, and its test case.\n\n"
            
            "RULES FOR DOCSTRING GENERATION:\n"
            "1. **Analyze Inputs**: Use the provided 'INPUT DATA ANALYSIS' to write the 'Args' section. "
            "   Translate technical summaries (e.g., 'List of 3 Tensors') into natural language (e.g., 'A list containing three tensors with shape of, representing...').\n"
            "   If any input argument is a 'CustomObject', you must explicitly list the public methods and attributes used in the SOURCE CODE. Cross check if you list out all the the public methods and attributes used in the SOURCE CODE"
            "   This is required to prevent the LLM from hallucinating non-existent attributes in using the source code."
            "2. Generalization: The provided I/O Info contains specific numbers (e.g., shape [32, 32, 32]). "
            "   Do NOT hardcode these numbers in the docstring. Instead, generalize them symbolically based the provided code"
            "   (e.g., use '(N, H, W)', '(..., H, W)', or '(Batch, ...)') based on the code logic.\n"
            "   Only use the specific constraints (like dtype or device) if they are strictly required by the code.\n\n"
            "3. FACTORY MODE: If a 'RETURNED FUNCTION RUNTIME EXAMPLE' is provided, it means the main function "
            "   is a Factory. You must document the 'Returns' section as a 'Callable' or 'function' "
            "   and explicitly describe the input/output signature of that returned function also in Google Style(sections: 'Args:', 'Returns:')."
            "   IMPORTANT: When documenting the returned function, clearly distinguish its arguments from the factory's arguments. "
            "   In the 'Returns' section of the factory, describe the returned object as 'Callable[[InnerArgs], InnerReturn]' and explain what the inner function does."
            
            "Return a valid JSON object with these keys:\n"
            "1. 'function_explanation': A concise, one-sentence summary.\n"
            "2. 'function_docstring': A Google-style docstring using generalized shapes (e.g., use 'B' instead of '32').\n"
            "3. 'usage_example': A code snippet using the function."
        )
        raw_inputs = parent_io_info.get("inputs", {})
        
        input_descriptions = []
        for arg_name, arg_info in raw_inputs.items():
            summary = DataSummarizer.summarize(arg_info)
            input_descriptions.append(f"- {arg_name}: {summary}")
        
        inputs_block = "\n".join(input_descriptions)
        print("inputs_block", inputs_block)
        io_section = f"""
            ### INPUT DATA ANALYSIS (Use this to describe Args)
            {inputs_block}
            ```json
            {json.dumps(parent_io_info, indent=2)}
            """ 
        if child_io_info: 
            io_section += f"""

            RETURNED FUNCTION I/O (The Inner Function)

            This is the I/O trace of the function returned by the factory.

            JSON
            {json.dumps(child_io_info, indent=2)}
            """

        user_prompt = f"""
            Please generate the documentation.

            ### SOURCE CODE
            ```python
            {source_code}

            I/O RUNTIME EXAMPLE (Use this for type inference, but generalize the shapes)
            {io_section}

            Python
            {test_code}
            """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0, 
                response_format={"type": "json_object"}
            )
            # print("response", response)
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            return {"error": str(e)}

def generate_tools(config, standard_data_dir, working_folder, final_eval_function_lists):
    for FUNC_NAME in final_eval_function_lists:
        
        # FUNC_NAME = "evaluate_results" # Change this to your factory function name if testing that
        output_filename =  os.path.join(working_folder,f"generated_docs_{FUNC_NAME}.json")
        
        if os.path.exists(output_filename):
            continue
        # 1. Load Code
        code_path = os.path.join(working_folder, f"gt_{FUNC_NAME}.py")
        if not os.path.exists(code_path):
            code_path = os.path.join(working_folder, f"agent_{FUNC_NAME}.py")
        test_path = os.path.join(working_folder, f"test_{FUNC_NAME}.py")

        code_snippet = load_code_from_file(code_path)
        test_snippet = load_code_from_file(test_path)

        # 2. Load Metadata (Detects Factory automatically)
        print(f"--- analyzing {FUNC_NAME} ---")
        parent_io, child_io = get_io_metadata(standard_data_dir, FUNC_NAME)
        if isinstance(parent_io, list):
            parent_io = parent_io[0] # Take the first object
        else:
            parent_io = parent_io

        if isinstance(child_io, list):
            child_io = child_io[0] # Take the first object
        else:
            child_io = child_io

        # 3. Run Generator
        generator = LLMDocGenerator(
            api_key=config['llm']['tool_writer']['api_key'], 
            base_url=config['llm']['tool_writer']['base_url'], 
            model=config['llm']['tool_writer']['model']
        )

        docs = generator.generate_docs(code_snippet, parent_io, test_snippet, child_io)

        # 4. Output
        if "error" in docs:
            print(f"❌ FAILED: {docs['error']}")
        else:
            print("\n[GENERATED DOCSTRING]:\n")
            print(docs.get('function_explanation'))
            print(docs.get('function_docstring'))
            print(docs.get('usage_example'))
            

        # --- SAVE TO FILE ---
        print(f"Saving documentation to {FUNC_NAME}...")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                # indent=4 makes it readable for humans
                # ensure_ascii=False allows special characters (like symbols) to appear correctly
                json.dump(docs, f, indent=4, ensure_ascii=False)
            print("✅ Save successful.")
        except Exception as e:
            print(f"❌ Failed to save: {e}")