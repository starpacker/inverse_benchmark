"""
Phase 4 - Sandbox Executor

Manages the execution of the consolidated code (run_gt.py) and evaluation (evaluate.py)
within a safe, reproducible environment (Conda).
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SandboxExecutor:
    """Runs code in a sandboxed Conda environment."""

    def __init__(self, config: dict):
        self.config = config
        self.conda_base = config["openhands"]["sandbox"]["conda_base"]
        self.env_prefix = config["sandbox"]["conda_env_prefix"]
        self.timeout = config["sandbox"]["execution_timeout"]

    def setup_environment(self, task_name: str, requirements_path: Optional[str] = None) -> str:
        """Create or update a dedicated Conda environment for the task."""
        env_name = f"{self.env_prefix}{task_name}"
        logger.info(f"Setting up Conda env: {env_name}")

        # Basic create command
        cmd = [
            "conda", "create", "-y", "-n", env_name,
            "python=3.10", "numpy", "scikit-image", "scipy", "pytorch"
        ]
        
        try:
            # Check if env exists
            check = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True, text=True
            )
            if env_name in check.stdout:
                logger.info(f"Environment {env_name} exists.")
            else:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            # Install extra requirements if provided
            if requirements_path and Path(requirements_path).exists():
                subprocess.run(
                    ["conda", "run", "-n", env_name, "pip", "install", "-r", requirements_path],
                    check=True
                )
            
            return env_name

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup environment: {e}")
            raise

    def execute_script(
        self,
        script_path: str,
        env_name: str,
        cwd: str,
        timeout: Optional[int] = None
    ) -> dict:
        """Run a Python script in the specified environment."""
        timeout = timeout or self.timeout
        start_time = time.time()
        
        cmd = [
            "conda", "run", "-n", env_name,
            "--no-capture-output",  # Stream output directly
            "python", script_path
        ]

        logger.info(f"Executing {script_path} in {env_name} (cwd={cwd})")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            return {
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Execution timed out after {timeout}s")
            return {
                "success": False,
                "returncode": -1,
                "error": "TimeoutExpired",
                "duration": timeout
            }
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {
                "success": False,
                "returncode": -1,
                "error": str(e),
                "duration": time.time() - start_time
            }
