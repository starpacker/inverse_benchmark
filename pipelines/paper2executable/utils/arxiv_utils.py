"""
ArXiv Utilities

Fetch paper metadata from the ArXiv API.
"""

import re
import time
from typing import Optional, Dict
from dataclasses import dataclass, field

import arxiv

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ArxivPaperInfo:
    """Structured metadata about an ArXiv paper."""
    arxiv_id: str
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    pdf_url: str = ""
    published: str = ""
    categories: list[str] = field(default_factory=list)


def fetch_arxiv_metadata(arxiv_id: str) -> Optional[ArxivPaperInfo]:
    """
    Fetch metadata for a single ArXiv paper by its ID.

    Args:
        arxiv_id: ArXiv ID like "2108.10257" or "2108.10257v1"

    Returns:
        ArxivPaperInfo or None on failure.
    """
    # Normalize ID (strip version suffix for search, keep for record)
    clean_id = arxiv_id.split("v")[0].strip()

    try:
        search = arxiv.Search(id_list=[clean_id])
        results = list(search.results())
        if not results:
            logger.warning(f"No ArXiv result for ID: {clean_id}")
            return None

        paper = results[0]
        info = ArxivPaperInfo(
            arxiv_id=clean_id,
            title=paper.title.strip(),
            abstract=paper.summary.strip(),
            authors=[a.name for a in paper.authors],
            pdf_url=paper.pdf_url,
            published=paper.published.isoformat() if paper.published else "",
            categories=paper.categories,
        )
        logger.info(f"Fetched ArXiv metadata: {info.title[:80]}...")
        return info

    except Exception as e:
        logger.error(f"Failed to fetch ArXiv metadata for {arxiv_id}: {e}")
        return None


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """Check if a string looks like a valid ArXiv ID."""
    # Patterns: 2108.10257, 2108.10257v1, hep-ph/9905221
    pattern = r"^(\d{4}\.\d{4,5})(v\d+)?$|^[a-z\-]+/\d{7}$"
    return bool(re.match(pattern, arxiv_id.strip()))
