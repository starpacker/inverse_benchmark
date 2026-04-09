"""
Phase 2 - Code Discovery (Level 3): Web Search + LLM Verification

Actively searches for code repositories using DuckDuckGo/SerpAPI,
then uses an LLM to read the README/snippets and verify relevance.
"""

import json
from typing import Optional, List

from duckduckgo_search import DDGS

from utils.llm_client import LLMClient
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class WebSearchDiscovery:
    """Finds code via search engine + LLM filtering."""

    def __init__(self, llm_client: LLMClient, config: dict):
        self.llm = llm_client
        self.config = config
        self.search_engine = DDGS()
        self.max_results = config.get("discovery", {}).get("web_search", {}).get("max_results", 5)
        self.verify_top_n = config.get("discovery", {}).get("web_search", {}).get("verify_top_n", 3)

    def find_repo(self, title: str, arxiv_id: Optional[str] = None) -> Optional[str]:
        """Perform search and verify candidates."""
        query = f'"{title}" github code "inverse problem"'
        logger.info(f"Searching web for code: {query}")

        try:
            # Execute search
            results = self.search_engine.text(query, max_results=self.max_results)
            if not results:
                logger.warning(f"No search results for '{title}'")
                return None

            # Filter for GitHub links only
            candidates = []
            for res in results:
                url = res.get("href", "")
                if "github.com" in url and "/blob/" not in url and "/issues/" not in url:
                    # Clean up URL (remove trailing slash, .git, etc.)
                    url = url.split("?")[0].rstrip("/")
                    if url.endswith(".git"):
                        url = url[:-4]
                    if url not in candidates:
                        candidates.append({
                            "url": url,
                            "title": res.get("title", ""),
                            "snippet": res.get("body", "")
                        })

            if not candidates:
                logger.info("No GitHub links found in search results.")
                return None

            # Use LLM to verify top N candidates
            best_match = self._verify_candidates(title, candidates[:self.verify_top_n])
            return best_match

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return None

    def _verify_candidates(self, paper_title: str, candidates: List[dict]) -> Optional[str]:
        """Ask LLM to identify the correct repository from search snippets."""
        prompt = (
            f"I am looking for the official code repository for the paper: '{paper_title}'.\n"
            "Below are search results from GitHub. Please identify which one is the "
            "most likely official implementation or a high-quality reproduction.\n\n"
        )

        for i, cand in enumerate(candidates):
            prompt += (
                f"Candidate {i+1}:\n"
                f"URL: {cand['url']}\n"
                f"Title: {cand['title']}\n"
                f"Snippet: {cand['snippet']}\n\n"
            )

        prompt += (
            "Analyze the snippets carefully. Look for phrases like 'official implementation', "
            "'code for paper', or matching authors/titles.\n"
            "Respond in JSON format with:\n"
            "{\n"
            '  "best_candidate_index": 1,  // 1-based index, or 0 if none match\n'
            '  "confidence": "high|medium|low",\n'
            '  "reasoning": "..."\n'
            "}"
        )

        try:
            response = self.llm.completion(
                prompt,
                system_prompt="You are a research assistant verifying code repositories.",
                json_mode=True
            )
            data = json.loads(response)

            idx = data.get("best_candidate_index", 0)
            if idx > 0 and idx <= len(candidates):
                chosen = candidates[idx - 1]
                logger.info(
                    f"LLM selected {chosen['url']} (Confidence: {data.get('confidence')})"
                )
                return chosen["url"]
            else:
                logger.info("LLM could not verify any candidate as the correct repo.")
                return None

        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return None
