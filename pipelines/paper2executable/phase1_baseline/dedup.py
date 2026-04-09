"""
Phase 1 - Deduplication Engine

Provides uniqueness checks against the global index:
1. ArXiv ID exact match (primary key)
2. Title cosine similarity with TF-IDF (catches title variants)
"""

import re
from typing import Optional

from rapidfuzz import fuzz

from database.manager import DatabaseManager
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class DeduplicationEngine:
    """Checks whether a paper already exists in the global index."""

    def __init__(self, db: DatabaseManager, config: dict):
        self.db = db
        self.config = config
        dedup_cfg = config.get("baseline", {}).get("dedup", {})
        self.title_threshold = dedup_cfg.get("title_similarity_threshold", 0.85)
        self.use_arxiv_first = dedup_cfg.get("use_arxiv_id_first", True)

    def is_duplicate(
        self,
        arxiv_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if a paper is a duplicate.

        Returns:
            (is_dup: bool, reason: str | None)
            reason is a human-readable explanation if duplicate.
        """
        # --- Level 1: ArXiv ID exact match ---
        if arxiv_id and self.use_arxiv_first:
            existing = self.db.get_paper_by_arxiv_id(arxiv_id)
            if existing:
                reason = (
                    f"ArXiv ID exact match: {arxiv_id} → "
                    f"existing paper_id={existing.id} '{existing.title[:60]}'"
                )
                logger.info(f"DEDUP HIT: {reason}")
                return True, reason

        # --- Level 2: Title similarity ---
        if title:
            normalized_title = self._normalize_title(title)
            all_papers = self.db.get_all_papers()

            best_score = 0.0
            best_match = None

            for paper in all_papers:
                existing_normalized = self._normalize_title(paper.title)
                # Use token_sort_ratio for word-order-invariant matching
                score = fuzz.token_sort_ratio(normalized_title, existing_normalized) / 100.0

                if score > best_score:
                    best_score = score
                    best_match = paper

            if best_score >= self.title_threshold and best_match:
                reason = (
                    f"Title similarity {best_score:.2%}: "
                    f"'{title[:60]}' ≈ '{best_match.title[:60]}' "
                    f"(paper_id={best_match.id})"
                )
                logger.info(f"DEDUP HIT: {reason}")
                return True, reason

            if best_match:
                logger.debug(
                    f"Best title match for '{title[:40]}' is "
                    f"'{best_match.title[:40]}' at {best_score:.2%} "
                    f"(below threshold {self.title_threshold:.2%})"
                )

        return False, None

    def _normalize_title(self, title: str) -> str:
        """Normalize a title for comparison.

        - Lowercase
        - Remove punctuation
        - Collapse whitespace
        - Remove common prefixes/suffixes
        """
        t = title.lower().strip()
        # Remove version strings like v1, v2, etc.
        t = re.sub(r'\bv\d+\b', '', t)
        # Remove arXiv references
        t = re.sub(r'arxiv:\s*\d+\.\d+', '', t)
        # Remove punctuation except hyphens
        t = re.sub(r'[^\w\s\-]', ' ', t)
        # Collapse whitespace
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def check_batch(
        self,
        papers: list[dict],
    ) -> dict:
        """Check a batch of papers for duplicates.

        Args:
            papers: list of dicts with optional 'arxiv_id' and 'title' keys.

        Returns:
            {"new": [list of non-dup paper dicts], "duplicates": [list of (paper, reason)]}
        """
        new_papers = []
        duplicates = []

        for paper in papers:
            is_dup, reason = self.is_duplicate(
                arxiv_id=paper.get("arxiv_id"),
                title=paper.get("title"),
            )
            if is_dup:
                duplicates.append((paper, reason))
            else:
                new_papers.append(paper)

        logger.info(
            f"Batch dedup: {len(new_papers)} new, "
            f"{len(duplicates)} duplicates out of {len(papers)} total"
        )
        return {"new": new_papers, "duplicates": duplicates}
