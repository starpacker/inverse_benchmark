import os
import re
import json
import subprocess
import sys
from utils.llm import llm_client
from utils.sandbox import SandboxManager
from prompts import phase2 as prompts
from config import settings
from utils.logger import RunLogger

class CodeGenerator:
    def __init__(self, model_name: str = settings.DEFAULT_MODEL):
        self.model_name = model_name

    def generate_code(self, task_description: str, plan: dict, sandbox_path: str, input_file: str, output_file: str, logger: RunLogger = None) -> str:
        # Detect environment
        installed_libs = SandboxManager.get_installed_libraries(sys.executable)
        
        prompt = prompts.CODE_GEN_USER_TEMPLATE.format(
            task_description=task_description,
            plan_json=json.dumps(plan, indent=2),
            installed_libs=installed_libs,
            input_file=input_file,
            output_file=output_file
        )
        
        res = llm_client.call(prompts.CODE_GEN_SYSTEM_PROMPT, prompt, model=self.model_name)
        code = self._extract_code(res.get("content", ""))
        
        if logger:
            logger.log_trace("phase2", "code_gen_thinking", res.get("thinking", ""))
            logger.log_trace("phase2", "code_gen_response", res.get("content", ""))
            logger.log_artifact("agent_solution", code, "py")
            
        return code

    def refine_code(self, plan: dict, previous_code: str, error_logs: str, logger: RunLogger = None) -> str:
        prompt = prompts.ERROR_REFINER_TEMPLATE.format(
            plan_json=json.dumps(plan, indent=2),
            previous_code=previous_code,
            error_logs=error_logs
        )
        res = llm_client.call("You are an expert Debugger.", prompt, model=self.model_name)
        code = self._extract_code(res.get("content", ""))
        
        if logger:
            logger.log_trace("phase2", "refine_thinking", res.get("thinking", ""))
            logger.log_trace("phase2", "refine_response", res.get("content", ""))
            logger.log_artifact("refined_solution", code, "py")
            
        return code

    @staticmethod
    def _extract_code(response: str) -> str:
        match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip()

class CodeExecutor:
    def execute(self, sandbox_path: str, code: str, output_file: str, logger: RunLogger = None, python_path: str = sys.executable) -> dict:
        """
        Runs the code in the sandbox.
        Returns: {success: bool, logs: str, return_code: int}
        """
        script_path = os.path.join(sandbox_path, "agent_solution.py")
        with open(script_path, "w") as f:
            f.write(code)
            
        try:
            # Use the specified python_path or fallback to sys.executable
            cmd = [python_path, "agent_solution.py"]
            
            result = subprocess.run(
                cmd,
                cwd=sandbox_path,
                capture_output=True,
                text=True,
                timeout=settings.EXECUTION_TIMEOUT
            )
            
            logs = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            if logger:
                logger.log_artifact("execution_log", logs, "txt")
            
            if result.returncode != 0:
                return {"success": False, "logs": logs, "return_code": result.returncode, "error_type": "Runtime Error"}
            
            # Check if output exists
            if not os.path.exists(os.path.join(sandbox_path, output_file)):
                 return {"success": False, "logs": logs + "\n\nError: Output file not created.", "return_code": 0, "error_type": "Missing Output"}
                 
            return {"success": True, "logs": logs, "return_code": 0}
            
        except subprocess.TimeoutExpired:
            return {"success": False, "logs": "Timeout Expired", "return_code": -1, "error_type": "Timeout"}
        except Exception as e:
            return {"success": False, "logs": str(e), "return_code": -1, "error_type": "System Error"}
