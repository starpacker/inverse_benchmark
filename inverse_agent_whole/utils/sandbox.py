import os
import shutil
import uuid
import datetime
import subprocess
from config import settings

class SandboxManager:
    def __init__(self, base_path: str = settings.SANDBOX_DIR, keep_files: bool = False):
        self.base_path = base_path
        self.keep_files = keep_files
        os.makedirs(self.base_path, exist_ok=True)
        self.current_sandbox = None

    def create_sandbox(self, source_path: str, task_name: str) -> str:
        """Creates a new sandbox by copying the source directory."""
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sandbox_name = f"{task_name}_{timestamp}_{unique_id}"
        self.current_sandbox = os.path.join(self.base_path, sandbox_name)
        
        if os.path.exists(self.current_sandbox):
            shutil.rmtree(self.current_sandbox)
            
        # Ensure source exists, if not, create empty
        if not os.path.exists(source_path):
             os.makedirs(source_path, exist_ok=True)

        shutil.copytree(source_path, self.current_sandbox)
        return self.current_sandbox

    def cleanup(self):
        if not self.keep_files and self.current_sandbox and os.path.exists(self.current_sandbox):
            try:
                shutil.rmtree(self.current_sandbox)
            except Exception as e:
                print(f"[Sandbox] Warning: Cleanup failed for {self.current_sandbox}: {e}")

    @staticmethod
    def get_installed_libraries(python_path: str) -> str:
        """Detects installed libraries in the environment."""
        try:
            result = subprocess.run(
                [python_path, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0:
                import json
                packages = json.loads(result.stdout)
                return ", ".join([p["name"] for p in packages])
        except Exception:
            pass
        return "Standard Python Libraries"
