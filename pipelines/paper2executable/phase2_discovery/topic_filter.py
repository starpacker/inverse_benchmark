"""
Phase 2 - Code Discovery (Level 4): Topic Filter

Uses an LLM to determine if a paper is relevant to the target domain
(Computational Imaging / Inverse Problems).
"""

import json
from typing import List, Optional

from utils.llm_client import LLMClient
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class TopicFilter:
    """Filters papers by topic relevance."""

    def __init__(self, llm_client: LLMClient, config: dict):
        self.llm = llm_client
        self.config = config
        self.target_topics = config.get("discovery", {}).get("topic_filter", {}).get("target_topics", [])
        self.strict_mode = config.get("discovery", {}).get("topic_filter", {}).get("strict_mode", True)

    def is_relevant(self, title: str, abstract: str) -> tuple[bool, List[str]]:
        """Determine if the paper matches the target domain."""
        if not self.target_topics:
            logger.warning("No target topics configured; passing everything.")
            return True, ["unknown"]

        prompt = (
            f"Paper Title: {title}\n\n"
            f"Abstract: {abstract}\n\n"
            f"Target Topics: {', '.join(self.target_topics)}\n\n"
            "Task: Identify if this paper strictly belongs to the field of "
            "computational imaging or inverse problems (e.g., reconstruction, "
            "denoising, super-resolution, tomography, etc.).\n"
            "If it is a pure computer vision paper (e.g., classification, "
            "detection, segmentation on natural images) without an inverse "
            "problem formulation, mark it as irrelevant.\n\n"
            "Output JSON:\n"
            "{\n"
            '  "is_relevant": true/false,\n'
            '  "primary_topic": "string",\n'
            '  "reasoning": "string"\n'
            "}"
        )

        try:
            response = self.llm.completion(
                prompt,
                system_prompt="You are an expert in computational imaging.",
                json_mode=True
            )
            data = json.loads(response)
            relevant = data.get("is_relevant", False)
            topic = data.get("primary_topic", "unknown")
            reason = data.get("reasoning", "")

            if relevant:
                logger.info(f"Topic MATCH: {title[:50]}... ({topic})")
            else:
                logger.info(f"Topic REJECT: {title[:50]}... ({reason})")

            return relevant, [topic]

        except Exception as e:
            logger.error(f"Topic filter failed: {e}")
            # In strict mode, fail open or closed? Here closed.
            return False, []
