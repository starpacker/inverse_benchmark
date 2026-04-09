"""
Phase 3 - OpenHands Task Builder

Prepares the OpenHands workspace with necessary files and configurations.
"""

import shutil
from pathlib import Path
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class TaskBuilder:
    """Prepares workspace for OpenHands execution."""

    def __init__(self, config: dict):
        self.config = config
        self.staging_dir = Path(config["openhands"]["staging_dir"])

    def prepare_workspace(
        self,
        task_name: str,
        repo_path: Optional[str] = None,
        context_files: Optional[dict] = None
    ) -> Path:
        """Set up the task directory."""
        workspace = self.staging_dir / task_name
        
        # Clean existing workspace if needed (careful!)
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)

        # Copy repo if provided
        if repo_path:
            repo_src = Path(repo_path)
            if repo_src.exists():
                logger.info(f"Copying repo from {repo_src} to {workspace}")
                shutil.copytree(repo_src, workspace / "repo")
            else:
                logger.warning(f"Repo path {repo_src} does not exist.")

        # Create dummy context files (e.g., paper summary)
        if context_files:
            for fname, content in context_files.items():
                (workspace / fname).write_text(content, encoding="utf-8")

        return workspace
