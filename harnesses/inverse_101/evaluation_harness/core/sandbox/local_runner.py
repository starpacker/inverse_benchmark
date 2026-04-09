"""Local filesystem sandbox for executing agent-generated code (no Docker).

Replaces DockerRunner for environments without Docker access.
Creates a temporary workspace, copies visible files, and runs commands
directly on the host.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 8000


class LocalRunner:
    """Manages a temporary directory workspace for sandboxed execution.

    Drop-in replacement for DockerRunner — same interface.
    """

    def __init__(self, image: str, task_dir: Path, timeout: int = 600) -> None:
        self.image = image  # unused, kept for API compat
        self.task_dir = task_dir.resolve()
        self.timeout = timeout
        self.workspace: Path | None = None
        self.container: str | None = None  # for API compat with scorer

    def start(self, visible_paths: list[str] | None = None) -> None:
        """Create a temp workspace and seed it with visible files."""
        self.workspace = Path(tempfile.mkdtemp(prefix="imaging101-local-"))
        self.container = str(self.workspace)  # for API compat
        log.info("Local workspace: %s", self.workspace)

        if visible_paths:
            for p in visible_paths:
                src = self.task_dir / p
                dst = self.workspace / p
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                elif src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                else:
                    log.warning("Visible path not found: %s", src)
        else:
            shutil.copytree(self.task_dir, self.workspace, dirs_exist_ok=True)

        # Install per-task requirements in isolated venv
        req = self.task_dir / "requirements.txt"
        if req.exists():
            log.info("Installing task requirements in isolated environment…")
            venv_path = self.workspace / ".venv"
            self.exec(f"python -m venv {venv_path}")
            # Windows uses Scripts/, Unix uses bin/
            import sys
            if sys.platform == "win32":
                venv_bin = venv_path / "Scripts"
            else:
                venv_bin = venv_path / "bin"
            self.exec(f"{venv_bin}/pip install -q -r {req}")
            # Store venv path so exec() can prepend it to PATH
            self._venv_bin = str(venv_bin)
        else:
            self._venv_bin = None

    def exec(self, command: str) -> tuple[str, int]:
        """Run *command* inside the workspace directory."""
        assert self.workspace, "Workspace not started"
        # Replace /workspace references with actual workspace path
        command = command.replace("/workspace_src/", str(self.task_dir) + "/")
        command = command.replace("/workspace_src", str(self.task_dir))
        command = command.replace("/workspace/", str(self.workspace) + "/")
        command = command.replace("/workspace", str(self.workspace))
        
        # Minimal whitelist environment
        import sys
        path_sep = ";" if sys.platform == "win32" else ":"
        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": str(self.workspace),
            # Use real HOME to avoid matplotlib/config init errors
            "HOME": os.environ.get("HOME", os.environ.get("USERPROFILE", str(self.workspace))),
            "USERPROFILE": os.environ.get("USERPROFILE", str(self.workspace)),
            # Windows needs these for temp files
            "TEMP": os.environ.get("TEMP", ""),
            "TMP": os.environ.get("TMP", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        }

        # Prepend venv bin to PATH so `python` resolves to venv Python
        venv_bin = getattr(self, "_venv_bin", None)
        if venv_bin:
            safe_env["PATH"] = venv_bin + path_sep + safe_env["PATH"]
        
        try:
            if sys.platform == "win32":
                # On Windows, use cmd.exe instead of bash
                result = subprocess.run(
                    ["cmd.exe", "/c", command],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=str(self.workspace),
                    env=safe_env,
                )
            else:
                result = subprocess.run(
                    ["bash", "-c", command],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=str(self.workspace),
                    env=safe_env,
                )
            # Preserve stderr separately to avoid truncating critical errors
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if len(stderr) > MAX_OUTPUT_CHARS // 2:
                stderr = stderr[:MAX_OUTPUT_CHARS // 2] + "\n… [stderr truncated]"
            if len(stdout) > MAX_OUTPUT_CHARS // 2:
                stdout = stdout[:MAX_OUTPUT_CHARS // 2] + "\n… [stdout truncated]"
            
            output = (stdout + "\n" + stderr).strip() if stderr else stdout
            return output, result.returncode
        except subprocess.TimeoutExpired:
            return "[Command timed out]", 1
        except Exception as e:
            return f"[Error: {e}]", 1

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, stripping /workspace prefix if present.
        
        Validates that the resolved path stays within workspace to prevent
        path traversal attacks.
        """
        # The LLM agent thinks it's in /workspace, strip that prefix
        if path.startswith("/workspace/"):
            path = path[len("/workspace/"):]
        elif path.startswith("/workspace"):
            path = path[len("/workspace"):]
        if path.startswith("/"):
            path = path[1:]
        
        # Resolve to absolute path and validate it's within workspace
        full_path = (self.workspace / path).resolve()
        
        # Security check: prevent path traversal
        try:
            full_path.relative_to(self.workspace.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {path} escapes workspace")
        
        return full_path

    def write_file(self, path: str, content: str) -> None:
        """Write content to a path inside the workspace."""
        assert self.workspace, "Workspace not started"
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def read_file(self, path: str) -> str:
        """Read a file from the workspace.

        Mirrors :meth:`DockerRunner.read_file` behaviour.  Known binary
        extensions are rejected early with a helpful hint; unknown binary
        files fall back gracefully instead of crashing the evaluation.
        """
        assert self.workspace, "Workspace not started"
        try:
            full_path = self._resolve_path(path)
        except ValueError as e:
            return f"[Error reading {path}]: {e}"
        
        if not full_path.exists():
            return f"[Error reading {path}]: File not found"
        # Layer 1: known binary extensions → helpful hint
        binary_exts = {'.npz', '.npy', '.pkl', '.pickle', '.h5', '.hdf5', '.pt', '.pth', '.bin'}
        if full_path.suffix.lower() in binary_exts:
            return (f"[Error reading {path}]: Binary file ({full_path.suffix}). "
                    f"Use RUN with `python -c \"import numpy as np; "
                    f"print(np.load('{path}').files)\"` to inspect.")
        # Layer 2: fallback for unknown binary files
        try:
            return full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"[Error reading {path}]: Binary file, cannot read as text."

    def stop(self) -> None:
        """Archive model-generated code, then clean up the temporary workspace."""
        if self.workspace and self.workspace.exists():
            # --- Archive model-generated code before deletion ---
            try:
                # Archive to project-local directory
                archive_base = Path(__file__).resolve().parent.parent / "results" / "code_archive"
                archive_base.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Extract task name from task_dir for a meaningful folder name
                task_name = self.task_dir.name if self.task_dir else "unknown_task"
                archive_dir = archive_base / f"{timestamp}_{task_name}"
                archive_dir.mkdir(parents=True, exist_ok=True)

                # Copy src/ directory (model-generated code)
                ws_src = self.workspace / "src"
                if ws_src.is_dir():
                    shutil.copytree(ws_src, archive_dir / "src", dirs_exist_ok=True)
                    log.info("Archived model code: %s/src -> %s/src", self.workspace, archive_dir)

                # Also copy plan/ if it exists (model-generated plans)
                ws_plan = self.workspace / "plan"
                if ws_plan.is_dir():
                    shutil.copytree(ws_plan, archive_dir / "plan", dirs_exist_ok=True)
                    log.info("Archived model plan: %s/plan -> %s/plan", self.workspace, archive_dir)

                # Also copy main.py
                ws_main = self.workspace / "main.py"
                if ws_main.is_file():
                    shutil.copy2(ws_main, archive_dir / "main.py")
                    log.info("Archived main.py")

                # Also copy output/ if it exists
                ws_output = self.workspace / "output"
                if ws_output.is_dir():
                    shutil.copytree(ws_output, archive_dir / "output", dirs_exist_ok=True)
                    log.info("Archived model output: %s/output -> %s/output", self.workspace, archive_dir)

                # Save a metadata file with run info
                meta_path = archive_dir / "archive_meta.txt"
                meta_path.write_text(
                    f"workspace: {self.workspace}\n"
                    f"task_dir: {self.task_dir}\n"
                    f"archived_at: {datetime.now().isoformat()}\n",
                    encoding="utf-8",
                )
            except Exception as e:
                log.warning("Failed to archive workspace: %s", e)

            # --- Original cleanup ---
            log.info("Cleaning up workspace %s", self.workspace)
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None
            self.container = None
