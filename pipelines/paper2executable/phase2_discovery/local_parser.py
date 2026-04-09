"""
Phase 2 - Code Discovery (Level 1): Local Parser

Scans the paper's full text (if available) for GitHub links.
"""

import re
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class LocalParser:
    """Finds GitHub links in local text/markdown."""

    def __init__(self, config: dict):
        self.config = config
        self.regex_pattern = re.compile(
            r'https?://github\.com/([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)',
            re.IGNORECASE
        )

    def find_repo_url(self, text: str) -> Optional[str]:
        """Scan text for the first GitHub repository URL."""
        if not text:
            return None

        # Find all matches
        matches = self.regex_pattern.findall(text)
        if not matches:
            return None

        # Filter out common false positives (frameworks, not the paper's code)
        # e.g., pytorch/pytorch, tensorflow/tensorflow, keras-team/keras
        ignored_repos = {
            "pytorch/pytorch", "tensorflow/tensorflow",
            "keras-team/keras", "huggingface/transformers",
            "microsoft/vscode", "facebookresearch/detectron2",
            "open-mmlab/mmdetection"
        }

        candidates = []
        for owner, repo in matches:
            full_repo = f"{owner}/{repo}"
            if full_repo.lower() not in ignored_repos:
                candidates.append(f"https://github.com/{full_repo}")

        if candidates:
            # Heuristic: The first non-framework link is often the winner
            # But could be improved by checking proximity to keywords like "code available at"
            logger.info(f"LocalParser found candidates: {candidates}")
            return candidates[0]

        return None
