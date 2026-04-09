#!/usr/bin/env python3
"""
ArXiv Paper Collector

Continuously collect relevant papers from ArXiv based on topics/keywords,
filter for code availability (Python-based), and queue for the pipeline.

Usage:
    python -m tools.arxiv_collector --topics "computational imaging" "inverse problem" --max-results 50
    python -m tools.arxiv_collector --config collector_config.yaml --daemon
"""

import argparse
import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

import arxiv
import requests

from database.manager import DatabaseManager
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# ArXiv categories relevant to computational imaging / inverse problems
RELEVANT_CATEGORIES = {
    "cs.CV",        # Computer Vision
    "cs.LG",        # Machine Learning
    "eess.IV",      # Image and Video Processing
    "eess.SP",      # Signal Processing
    "physics.optics",
    "physics.comp-ph",
    "stat.ML",
}

# Default search topics for inverse problems / computational imaging
DEFAULT_TOPICS = [
    "computational imaging",
    "inverse problem",
    "image reconstruction",
    "phase retrieval",
    "ptychography",
    "tomographic reconstruction",
    "deconvolution",
    "super resolution",
    "image restoration",
    "compressed sensing imaging",
    "holography reconstruction",
    "diffraction imaging",
]

# Keywords indicating Python implementation
PYTHON_INDICATORS = [
    "pytorch", "tensorflow", "keras", "jax", "numpy", "scipy",
    "python", "pip install", ".py", "torch", "tf.",
    "github.com", "colab", "jupyter",
]

# PapersWithCode API
PWC_API_BASE = "https://paperswithcode.com/api/v1"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CollectedPaper:
    """A paper collected from ArXiv with relevance metadata."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published: str
    pdf_url: str
    # Relevance info
    matched_topics: List[str] = field(default_factory=list)
    has_code: bool = False
    github_url: str = ""
    is_python: bool = False
    relevance_score: float = 0.0
    # Collection metadata
    collected_at: str = ""
    collection_source: str = "arxiv_search"


# ═══════════════════════════════════════════════════════════════════════════════
# ArXiv Search
# ═══════════════════════════════════════════════════════════════════════════════

class ArxivCollector:
    """Collects papers from ArXiv based on topic queries."""

    def __init__(
        self,
        topics: List[str] = None,
        categories: Set[str] = None,
        db: DatabaseManager = None,
        output_dir: str = "/data/yjh/paper2executable_workspace/collected_papers",
        days_back: int = 30,
        require_code: bool = True,
        require_python: bool = True,
    ):
        self.topics = topics or DEFAULT_TOPICS
        self.categories = categories or RELEVANT_CATEGORIES
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.days_back = days_back
        self.require_code = require_code
        self.require_python = require_python

        # Track collected papers to avoid duplicates within session
        self._seen_ids: Set[str] = set()
        self._collected: List[CollectedPaper] = []

    def search_topic(self, topic: str, max_results: int = 100) -> List[CollectedPaper]:
        """Search ArXiv for papers matching a topic."""
        logger.info(f"Searching ArXiv for: '{topic}' (max {max_results})")

        # Build query: search in title and abstract
        query = f'all:"{topic}"'

        # Add category filter
        if self.categories:
            cat_query = " OR ".join([f"cat:{c}" for c in self.categories])
            query = f"({query}) AND ({cat_query})"

        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            cutoff_date = datetime.now() - timedelta(days=self.days_back)

            for result in search.results():
                # Skip if too old
                if result.published and result.published.replace(tzinfo=None) < cutoff_date:
                    continue

                arxiv_id = result.entry_id.split("/")[-1].split("v")[0]

                # Skip duplicates
                if arxiv_id in self._seen_ids:
                    continue

                # Skip if already in database
                if self.db and self.db.paper_exists(arxiv_id=arxiv_id):
                    logger.debug(f"Skipping already-indexed paper: {arxiv_id}")
                    continue

                # Check category relevance
                paper_cats = set(result.categories)
                if not paper_cats.intersection(self.categories):
                    continue

                paper = CollectedPaper(
                    arxiv_id=arxiv_id,
                    title=result.title.strip().replace("\n", " "),
                    abstract=result.summary.strip().replace("\n", " "),
                    authors=[a.name for a in result.authors[:10]],
                    categories=result.categories,
                    published=result.published.isoformat() if result.published else "",
                    pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                    matched_topics=[topic],
                    collected_at=datetime.now().isoformat(),
                )

                self._seen_ids.add(arxiv_id)
                papers.append(paper)

                # Rate limiting
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"ArXiv search failed for '{topic}': {e}")

        logger.info(f"Found {len(papers)} new papers for topic '{topic}'")
        return papers

    def search_all_topics(self, max_per_topic: int = 50) -> List[CollectedPaper]:
        """Search all configured topics and merge results."""
        all_papers: Dict[str, CollectedPaper] = {}

        for topic in self.topics:
            papers = self.search_topic(topic, max_results=max_per_topic)
            for p in papers:
                if p.arxiv_id in all_papers:
                    # Merge matched topics
                    all_papers[p.arxiv_id].matched_topics.extend(p.matched_topics)
                else:
                    all_papers[p.arxiv_id] = p

            # Rate limiting between topics
            time.sleep(1.0)

        return list(all_papers.values())

    def check_code_availability(self, papers: List[CollectedPaper]) -> List[CollectedPaper]:
        """Check PapersWithCode for code links and filter."""
        logger.info(f"Checking code availability for {len(papers)} papers...")

        for paper in papers:
            github_url, is_python = self._check_pwc(paper.arxiv_id, paper.title)
            if github_url:
                paper.has_code = True
                paper.github_url = github_url
                paper.is_python = is_python
            else:
                # Check abstract for Python indicators
                paper.is_python = self._check_python_indicators(paper.abstract)

            # Rate limiting
            time.sleep(0.2)

        # Filter based on requirements
        if self.require_code:
            papers = [p for p in papers if p.has_code]
            logger.info(f"After code filter: {len(papers)} papers")

        if self.require_python:
            papers = [p for p in papers if p.is_python]
            logger.info(f"After Python filter: {len(papers)} papers")

        return papers

    def _check_pwc(self, arxiv_id: str, title: str) -> tuple[str, bool]:
        """Check PapersWithCode for a paper's code repository."""
        try:
            # Try by ArXiv ID first
            url = f"{PWC_API_BASE}/papers/?arxiv_id={arxiv_id}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    paper_data = data["results"][0]
                    repos = paper_data.get("repositories", [])
                    if repos:
                        repo = repos[0]
                        github_url = repo.get("url", "")
                        # Check if Python
                        is_python = repo.get("framework", "").lower() in ("pytorch", "tensorflow", "jax", "other")
                        # Also check description
                        if not is_python:
                            desc = str(repo.get("description", "")).lower()
                            is_python = any(ind in desc for ind in PYTHON_INDICATORS)
                        return github_url, is_python
        except Exception as e:
            logger.debug(f"PWC lookup failed for {arxiv_id}: {e}")

        return "", False

    def _check_python_indicators(self, text: str) -> bool:
        """Check if text contains Python-related keywords."""
        text_lower = text.lower()
        return any(ind in text_lower for ind in PYTHON_INDICATORS)

    def compute_relevance_scores(self, papers: List[CollectedPaper]) -> List[CollectedPaper]:
        """Compute relevance scores and sort papers."""
        for paper in papers:
            score = 0.0

            # More matched topics = higher relevance
            score += len(paper.matched_topics) * 2.0

            # Has code = bonus
            if paper.has_code:
                score += 5.0

            # Is Python = bonus
            if paper.is_python:
                score += 3.0

            # Recent = bonus (decay over 30 days)
            if paper.published:
                try:
                    pub_date = datetime.fromisoformat(paper.published.replace("Z", "+00:00"))
                    days_ago = (datetime.now(pub_date.tzinfo) - pub_date).days
                    score += max(0, (30 - days_ago) / 10)
                except:
                    pass

            # Keywords in title = bonus
            title_lower = paper.title.lower()
            keywords = ["inverse", "reconstruction", "imaging", "restoration", "recovery", "neural", "deep"]
            score += sum(1.0 for kw in keywords if kw in title_lower)

            paper.relevance_score = round(score, 2)

        # Sort by relevance
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers

    def collect(self, max_per_topic: int = 50, skip_pwc_check: bool = False) -> List[CollectedPaper]:
        """
        Main collection workflow:
        1. Search all topics
        2. Check code availability (optional, can be slow)
        3. Compute relevance scores
        4. Return sorted list
        
        Args:
            max_per_topic: Max papers to fetch per topic
            skip_pwc_check: Skip PapersWithCode API check (faster but less accurate code detection)
        """
        logger.info(f"Starting collection with {len(self.topics)} topics...")

        # Step 1: Search
        papers = self.search_all_topics(max_per_topic=max_per_topic)
        logger.info(f"Total unique papers found: {len(papers)}")

        if not papers:
            return []

        # Step 2: Code check (optional - PWC API is slow)
        if not skip_pwc_check:
            papers = self.check_code_availability(papers)
        else:
            # Quick heuristic: check abstract for Python indicators
            for paper in papers:
                paper.is_python = self._check_python_indicators(paper.abstract)
                paper.has_code = paper.is_python  # Assume Python indicator means code exists
            
            # Apply filters
            if self.require_code:
                papers = [p for p in papers if p.has_code]
            if self.require_python:
                papers = [p for p in papers if p.is_python]
            logger.info(f"After quick filters: {len(papers)} papers")

        # Step 3: Relevance scoring
        papers = self.compute_relevance_scores(papers)

        self._collected = papers
        return papers

    def save_results(self, papers: List[CollectedPaper] = None, filename: str = None):
        """Save collected papers to JSON file."""
        papers = papers or self._collected
        if not papers:
            logger.warning("No papers to save")
            return

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collected_{timestamp}.json"

        output_path = self.output_dir / filename
        data = {
            "collected_at": datetime.now().isoformat(),
            "topics": self.topics,
            "categories": list(self.categories),
            "total_papers": len(papers),
            "papers": [asdict(p) for p in papers],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(papers)} papers to {output_path}")
        return output_path

    def print_summary(self, papers: List[CollectedPaper] = None, top_n: int = 20):
        """Print a summary of collected papers."""
        papers = papers or self._collected
        if not papers:
            print("No papers collected.")
            return

        print(f"\n{'='*80}")
        print(f"COLLECTED PAPERS SUMMARY")
        print(f"{'='*80}")
        print(f"Total papers: {len(papers)}")
        print(f"With code: {sum(1 for p in papers if p.has_code)}")
        print(f"Python-based: {sum(1 for p in papers if p.is_python)}")
        print(f"\nTop {min(top_n, len(papers))} by relevance:")
        print(f"{'-'*80}")

        for i, p in enumerate(papers[:top_n], 1):
            code_badge = "✅" if p.has_code else "❌"
            py_badge = "🐍" if p.is_python else "  "
            print(f"{i:2d}. [{p.relevance_score:5.1f}] {code_badge}{py_badge} {p.arxiv_id}")
            print(f"    {p.title[:70]}...")
            if p.github_url:
                print(f"    → {p.github_url}")
            print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Collect relevant papers from ArXiv")
    parser.add_argument(
        "--topics", nargs="+", default=None,
        help="Topics to search for (default: built-in inverse problems topics)"
    )
    parser.add_argument(
        "--max-per-topic", type=int, default=50,
        help="Maximum results per topic (default: 50)"
    )
    parser.add_argument(
        "--days-back", type=int, default=30,
        help="Only papers from last N days (default: 30)"
    )
    parser.add_argument(
        "--require-code", action="store_true", default=True,
        help="Only include papers with code (default: True)"
    )
    parser.add_argument(
        "--no-require-code", action="store_false", dest="require_code",
        help="Include papers without code"
    )
    parser.add_argument(
        "--require-python", action="store_true", default=True,
        help="Only include Python-based papers (default: True)"
    )
    parser.add_argument(
        "--no-require-python", action="store_false", dest="require_python",
        help="Include non-Python papers"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/data/yjh/paper2executable_workspace/collected_papers",
        help="Output directory for results"
    )
    parser.add_argument(
        "--db-path", type=str, default="database/papers.db",
        help="Path to paper database for dedup"
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip database dedup check"
    )
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Show top N papers in summary (default: 30)"
    )

    args = parser.parse_args()

    # Initialize database (for dedup)
    db = None
    if not args.no_db:
        try:
            db = DatabaseManager(args.db_path)
        except Exception as e:
            logger.warning(f"Could not connect to database: {e}")

    # Create collector
    collector = ArxivCollector(
        topics=args.topics,
        db=db,
        output_dir=args.output_dir,
        days_back=args.days_back,
        require_code=args.require_code,
        require_python=args.require_python,
    )

    # Run collection
    papers = collector.collect(max_per_topic=args.max_per_topic)

    # Save and print
    if papers:
        collector.save_results(papers)
        collector.print_summary(papers, top_n=args.top_n)

        # Print command to process top paper
        if papers:
            top = papers[0]
            print(f"\n{'='*80}")
            print(f"To process the top paper:")
            print(f"  cd /home/yjh/paper2executable")
            print(f"  python cli.py run --arxiv-id {top.arxiv_id}")
            print(f"{'='*80}")
    else:
        print("No papers matched the criteria.")


if __name__ == "__main__":
    main()
