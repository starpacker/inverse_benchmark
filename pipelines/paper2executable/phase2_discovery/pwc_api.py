"""
Phase 2 - Code Discovery (Level 2): PapersWithCode API

Queries the PapersWithCode public API for official repository links.
"""

import time
from typing import Optional

import requests

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PapersWithCodeClient:
    """Interface to PapersWithCode API for code discovery."""

    def __init__(self, config: dict):
        self.config = config
        self.base_url = "https://paperswithcode.com/api/v1"
        self.timeout = config.get("discovery", {}).get("pwc_api", {}).get("timeout", 30)

    def find_repo_by_arxiv_id(self, arxiv_id: str) -> Optional[str]:
        """Find official code repository using ArXiv ID."""
        if not arxiv_id:
            return None

        # PWC endpoint: /papers/?arxiv_id={arxiv_id}
        url = f"{self.base_url}/papers/"
        params = {"arxiv_id": arxiv_id}

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data["count"] > 0:
                    paper_id = data["results"][0]["id"]
                    return self._get_repo_for_paper_id(paper_id)
            else:
                logger.warning(
                    f"PWC API returned {response.status_code} for {arxiv_id}"
                )
        except Exception as e:
            logger.error(f"PWC API error for {arxiv_id}: {e}")

        return None

    def find_repo_by_title(self, title: str) -> Optional[str]:
        """Find code by title query."""
        if not title:
            return None

        # PWC endpoint: /papers/?q={title}
        url = f"{self.base_url}/papers/"
        params = {"q": title}

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data["count"] > 0:
                    # Check first result for reasonable title match
                    top_match = data["results"][0]
                    # Could add fuzzy match check here
                    return self._get_repo_for_paper_id(top_match["id"])
        except Exception as e:
            logger.error(f"PWC API error for title query '{title}': {e}")

        return None

    def _get_repo_for_paper_id(self, paper_id: str) -> Optional[str]:
        """Fetch repositories linked to a PWC paper ID."""
        url = f"{self.base_url}/papers/{paper_id}/repositories/"
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data["count"] > 0:
                    # Prefer official implementation if flagged
                    repos = data["results"]
                    official = [r for r in repos if r.get("is_official")]
                    if official:
                        return official[0]["url"]
                    # Otherwise take the most starred/popular one
                    # PWC usually sorts by stars descending by default
                    return repos[0]["url"]
        except Exception as e:
            logger.error(f"PWC repository fetch error for {paper_id}: {e}")

        return None
