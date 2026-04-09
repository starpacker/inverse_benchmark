"""
Paper2Executable Orchestrator

The central nervous system of the pipeline.
Coordinates the flow from paper ingestion -> discovery -> execution -> validation.

Final pipeline outputs per paper:
  1. working-folder   — directory with repo code + data
  2. gt-code-path     — path to the main/consolidated script
  3. target-python-env — conda python executable path
  4. task-description  — markdown file in task_descriptions_path
  5. schema report     — JSON summary
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, List

from database.manager import DatabaseManager
from phase1_baseline.importer import BaselineImporter
from phase1_baseline.dedup import DeduplicationEngine
from phase2_discovery.local_parser import LocalParser
from phase2_discovery.pwc_api import PapersWithCodeClient
from phase2_discovery.web_search import WebSearchDiscovery
from phase2_discovery.topic_filter import TopicFilter
from phase2_discovery.pdf_processor import PDFProcessor
from phase3_openhands.openhands_runner import OpenHandsRunner
from phase3_openhands.task_desc_generator import TaskDescriptionGenerator
from phase3_openhands.output_builder import OutputBuilder
from phase4_sandbox.sandbox_executor import SandboxExecutor
from phase4_sandbox.evaluator import Evaluator
from utils.llm_client import LLMClient
from utils.arxiv_utils import fetch_arxiv_metadata, is_valid_arxiv_id
from utils.github_utils import (
    clone_repo,
    load_from_zip,
    find_main_scripts,
    find_requirements,
    get_repo_name_from_url,
    CloneFailedNeedManualUpload,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Manages the lifecycle of a paper through the P2E pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.db = DatabaseManager(config["general"]["database_path"])
        self.llm = LLMClient(config)

        # Phase 1
        self.importer = BaselineImporter(self.db, config)
        self.dedup = DeduplicationEngine(self.db, config)

        # Phase 2
        self.pdf_ocr = PDFProcessor(config)
        self.local_parser = LocalParser(config)
        self.pwc = PapersWithCodeClient(config)
        self.web_search = WebSearchDiscovery(self.llm, config)
        self.topic_filter = TopicFilter(self.llm, config)

        # Phase 3
        self.openhands = OpenHandsRunner(config)
        self.task_desc_gen = TaskDescriptionGenerator(self.llm, config)
        self.output_builder = OutputBuilder(config)

        # Phase 4
        self.executor = SandboxExecutor(config)
        self.evaluator = Evaluator(self.db, config)

        # Workspace
        self.workspace_base = Path(
            config.get("general", {}).get(
                "workspace_base", "/data/yjh/paper2executable_workspace"
            )
        )

    async def run_pipeline(self, arxiv_id: Optional[str] = None, title: Optional[str] = None):
        """Execute the full pipeline for a single paper."""
        
        # 1. Check existence (Dedup)
        is_dup, reason = self.dedup.is_duplicate(arxiv_id, title)
        if is_dup:
            logger.info(f"Skipping duplicate paper: {reason}")
            return

        # 2. Create initial record
        paper = self.db.add_paper(
            arxiv_id=arxiv_id,
            title=title or f"ArXiv:{arxiv_id}",
            pipeline_stage="discovered"
        )
        logger.info(f"Processing Paper ID {paper.id}: {paper.title}")

        # 3. Discovery (Find Code)
        # Try finding PDF content first to boost discovery
        pdf_content = self.pdf_ocr.process_paper(arxiv_id) if arxiv_id else None
        
        repo_url = await self._discover_code(paper, pdf_content)
        if not repo_url:
            self._fail_paper(paper, "Code not found")
            return

        # 4. Filter by Topic
        abstract = paper.abstract or (pdf_content[:2000] if pdf_content else "")
        is_relevant, topics = self.topic_filter.is_relevant(paper.title, abstract)
        self.db.update_paper(paper.id, topics=topics)
        if not is_relevant:
            self._fail_paper(paper, "Topic irrelevant")
            return

        # 5. OpenHands Execution (Consolidation)
        success = await self._run_openhands_flow(paper, repo_url)
        if not success:
            self._fail_paper(paper, "OpenHands flow failed")
            return

        # 6. Sandbox Validation
        metrics = self._validate_sandbox(paper)
        if not metrics:
            self._fail_paper(paper, "Validation failed")
            return

        logger.info(f"Pipeline SUCCESS for Paper {paper.id}!")

    async def _discover_code(self, paper, pdf_content: Optional[str] = None) -> Optional[str]:
        """Try all 3 levels of code discovery."""
        
        # Level 1: Local Parse (now with PDF OCR content!)
        if pdf_content:
            url = self.local_parser.find_repo_url(pdf_content)
            if url:
                logger.info(f"Code found via PDF OCR: {url}")
                self.db.update_paper(paper.id, github_url=url, github_discovered_by="pdf_ocr_local_parse")
                return url
        
        # Level 2: PWC API (Fastest)
        url = self.pwc.find_repo_by_arxiv_id(paper.arxiv_id)
        if url:
            logger.info(f"Code found via PWC: {url}")
            self.db.update_paper(paper.id, github_url=url, github_discovered_by="pwc_api")
            return url

        # Level 3: Web Search (Slower)
        url = self.web_search.find_repo(paper.title, paper.arxiv_id)
        if url:
            logger.info(f"Code found via Web Search: {url}")
            self.db.update_paper(paper.id, github_url=url, github_discovered_by="web_search")
            return url

        logger.warning(f"No code found for Paper {paper.id}")
        return None

    async def _run_openhands_flow(self, paper, repo_url: str) -> bool:
        """Run the 3-step OpenHands process."""
        
        task_name = f"paper_{paper.id}"
        
        # Step 1: Data Extraction
        res1 = await self.openhands.run_task(task_name, repo_url, "data_extraction")
        if res1["status"] != "finished":
            return False

        # Step 2: Code Consolidation
        res2 = await self.openhands.run_task(task_name, None, "code_consolidation")
        if res2["status"] != "finished":
            return False

        # Step 3: Evaluation Script
        res3 = await self.openhands.run_task(task_name, None, "evaluate_generation")
        if res3["status"] != "finished":
            return False

        self.db.update_paper(paper.id, pipeline_stage="consolidated")
        return True

    def _validate_sandbox(self, paper) -> Optional[dict]:
        """Run the code in the sandbox and evaluate."""
        task_name = f"paper_{paper.id}"
        workspace = Path(self.config["openhands"]["staging_dir"]) / task_name
        
        # Setup Env
        env_name = self.executor.setup_environment(task_name)

        # Run GT
        run_res = self.executor.execute_script("run_gt.py", env_name, cwd=str(workspace))
        if not run_res["success"]:
            logger.error(f"run_gt.py failed: {run_res['stderr']}")
            return None

        # Run Eval
        eval_res = self.executor.execute_script("evaluate.py", env_name, cwd=str(workspace))
        if not eval_res["success"]:
            logger.error(f"evaluate.py failed: {eval_res['stderr']}")
            return None

        # Process Metrics
        return self.evaluator.process_results(paper.id, str(workspace))

    def _fail_paper(self, paper, reason: str):
        """Mark paper as failed."""
        logger.error(f"Paper {paper.id} FAILED: {reason}")
        self.db.update_paper(paper.id, pipeline_stage="failed", failure_reason=reason)

    def import_baseline(self):
        """Run Phase 1 import."""
        return self.importer.import_all()

    # ================================================================
    # NEW: End-to-End pipeline producing the 5 standard outputs
    # ================================================================

    def process_paper_e2e(
        self,
        arxiv_id: str,
        skip_env_create: bool = False,
    ) -> Dict:
        """
        End-to-end pipeline for a single paper, producing:
          1. working-folder
          2. gt-code-path
          3. target-python-env
          4. task-description  (in /data/yjh/task_descriptions/)
          5. schema report

        This method is synchronous and calls async helpers internally.

        Args:
            arxiv_id: ArXiv paper ID (e.g. "2108.10257")
            skip_env_create: If True, skip conda env creation (use base python)

        Returns:
            dict with all output paths and status info
        """
        result = {
            "arxiv_id": arxiv_id,
            "status": "started",
            "steps": {},
        }

        # ── Step 1: Fetch ArXiv metadata ──
        logger.info(f"{'='*60}")
        logger.info(f"[E2E] Processing ArXiv: {arxiv_id}")
        logger.info(f"{'='*60}")

        paper_info = fetch_arxiv_metadata(arxiv_id)
        if not paper_info:
            logger.error(f"Cannot fetch ArXiv metadata for {arxiv_id}")
            result["status"] = "failed"
            result["error"] = "arxiv_metadata_fetch_failed"
            return result

        # Derive a clean task name from the arxiv_id
        task_name = self._make_task_name(arxiv_id, paper_info.title)
        logger.info(f"[E2E] Title : {paper_info.title}")
        logger.info(f"[E2E] Task  : {task_name}")
        result["task_name"] = task_name
        result["title"] = paper_info.title
        result["steps"]["arxiv_metadata"] = "ok"

        # ── Step 2: PDF → Markdown (OCR) ──
        logger.info(f"[E2E] Step 2: PDF → Markdown OCR")
        paper_markdown = None
        try:
            paper_markdown = self.pdf_ocr.process_paper(arxiv_id)
            if paper_markdown:
                logger.info(f"[E2E]   OCR success ({len(paper_markdown):,} chars)")
                result["steps"]["pdf_ocr"] = "ok"
            else:
                # OCR enabled but returned nothing
                logger.warning(f"[E2E]   OCR returned nothing")
                result["steps"]["pdf_ocr"] = "failed"
        except Exception as e:
            logger.warning(f"[E2E]   OCR error: {e}")
            result["steps"]["pdf_ocr"] = f"error: {e}"

        # ── Step 3: Code Discovery ──
        logger.info(f"[E2E] Step 3: Code Discovery")
        # manual override no longer allowed; always run discovery
        repo_url = None
        has_code = False

        repo_url = self._discover_code_sync(
            arxiv_id, paper_info.title, paper_markdown
        )

        if repo_url:
            logger.info(f"[E2E]   Repo URL: {repo_url}")
            result["steps"]["code_discovery"] = repo_url
            has_code = True
        else:
            logger.warning(f"[E2E]   No code repository found")
            result["steps"]["code_discovery"] = "not_found"

        # ── Step 4: Clone repo (if found) ──
        repo_dir = None
        gt_code_path = None
        code_content = None

        if has_code and repo_url:
            logger.info(f"[E2E] Step 4: Clone repository")
            repo_name = get_repo_name_from_url(repo_url)
            clone_target = str(self.workspace_base / "repos" / task_name)
            try:
                repo_dir = clone_repo(repo_url, clone_target)
            except CloneFailedNeedManualUpload as e:
                # ── Git clone failed (firewall etc.) — WAIT for human upload ──
                zip_dest = f"{clone_target}.zip"
                logger.warning(f"[E2E]   ⚠️  GIT CLONE FAILED — need manual zip upload")
                logger.warning(f"[E2E]   Repo URL : {repo_url}")
                logger.warning(f"[E2E]   Try downloading: {e.zip_download_url}")
                logger.warning(f"[E2E]   Then place zip at: {zip_dest}")
                print(f"\n{'!'*60}")
                print(f"  ACTION REQUIRED: git clone failed (likely firewall)")
                print(f"  Please download the repo as zip and place it at:")
                print(f"    {zip_dest}")
                print(f"  Download URL (try main or master):")
                print(f"    {e.zip_download_url}")
                print(f"  Press ENTER after uploading, or type 'skip' to continue without code.")
                print(f"{'!'*60}\n")

                user_input = input(">>> ").strip().lower()

                if user_input == "skip":
                    logger.info(f"[E2E]   User chose to skip — paper-only mode")
                    result["steps"]["clone"] = "skipped_by_user"
                    repo_dir = None
                else:
                    # User claims they uploaded — try loading from zip
                    repo_dir = load_from_zip(zip_dest, clone_target) if Path(zip_dest).exists() else None
                    if not repo_dir:
                        # Also try looking for any zip nearby
                        from utils.github_utils import _find_manual_zip
                        found = _find_manual_zip(Path(clone_target))
                        if found:
                            repo_dir = load_from_zip(str(found), clone_target)
                    if repo_dir:
                        logger.info(f"[E2E]   Loaded from zip → {repo_dir}")
                        result["steps"]["clone"] = str(repo_dir)
                    else:
                        logger.error(f"[E2E]   Zip not found at {zip_dest}, cannot proceed with code")
                        result["steps"]["clone"] = "zip_not_found"
                        repo_dir = None

            if repo_dir:
                logger.info(f"[E2E]   Repo ready → {repo_dir}")
                result["steps"]["clone"] = str(repo_dir)

                # ── Step 4.5: Run OpenHands consolidation ──
                logger.info(f"[E2E] Step 4.5: OpenHands consolidation")
                staging_dir = Path(self.config["openhands"]["staging_dir"]) / task_name

                # --- Data extraction ---
                try:
                    res1 = asyncio.run(self.openhands.run_task(task_name, repo_dir, "data_extraction"))
                    result["steps"]["openhands_data"] = res1.get("status")
                except Exception as e:
                    logger.error(f"OpenHands data_extraction error: {e}")
                    result["steps"]["openhands_data"] = f"error: {e}"
                    res1 = {"status": "failed"}

                # Check actual success by verifying expected output files,
                # because OpenHands may report ERROR (max-iter) yet still
                # have produced the required .npy files.
                data_files_ok = all(
                    (staging_dir / f).exists()
                    for f in ("input.npy", "gt_output.npy")
                )
                if data_files_ok:
                    logger.info(f"[E2E]   ✅ Data extraction produced output files")
                    result["steps"]["openhands_data"] = "ok"
                elif res1.get("status") not in ("success", "finished"):
                    logger.warning(f"[E2E]   ⚠️ Data extraction ended with {res1.get('status')} and no output files")

                # --- Code consolidation (only if data step produced files) ---
                data_step_ok = data_files_ok or res1.get("status") in ("success", "finished", "max_iterations")
                if data_step_ok:
                    try:
                        res2 = asyncio.run(self.openhands.run_task(task_name, None, "code_consolidation"))
                        result["steps"]["openhands_consolidation"] = res2.get("status")
                    except Exception as e:
                        logger.error(f"OpenHands code_consolidation error: {e}")
                        result["steps"]["openhands_consolidation"] = f"error: {e}"
                        res2 = {"status": "failed"}
                else:
                    logger.info(f"[E2E]   Skipping code_consolidation (data extraction incomplete)")
                    res2 = {"status": "skipped"}

                # Check consolidation success by file presence too
                run_gt_exists = (staging_dir / "run_gt.py").exists()
                if run_gt_exists:
                    logger.info(f"[E2E]   ✅ Code consolidation produced run_gt.py")
                    result["steps"]["openhands_consolidation"] = "ok"

                # --- Evaluate generation ---
                consol_ok = res2.get("status") in ("success", "finished", "max_iterations") or run_gt_exists
                if consol_ok:
                    try:
                        res3 = asyncio.run(self.openhands.run_task(task_name, None, "evaluate_generation"))
                        result["steps"]["openhands_evaluate"] = res3.get("status")
                    except Exception as e:
                        logger.error(f"OpenHands evaluate_generation error: {e}")
                        result["steps"]["openhands_evaluate"] = f"error: {e}"
                
                # Find main scripts
                main_scripts = find_main_scripts(str(repo_dir))
                if main_scripts:
                    gt_code_path = str(Path(repo_dir) / main_scripts[0])
                    logger.info(f"[E2E]   Main script candidates: {main_scripts[:5]}")
                    logger.info(f"[E2E]   Selected gt_code: {gt_code_path}")

                    # Read code content for task description generation
                    try:
                        code_content = Path(gt_code_path).read_text(
                            encoding="utf-8", errors="replace"
                        )
                        if len(code_content) > 100000:
                            code_content = code_content[:100000]
                    except Exception as exc:
                        logger.warning(f"[E2E]   Could not read code: {exc}")
                        code_content = None
        else:
            logger.info(f"[E2E] Step 4: No repo to clone (paper-only mode)")
            result["steps"]["clone"] = "skipped"

        # ── Step 5: Generate Task Description ──
        logger.info(f"[E2E] Step 5: Generate task description")
        task_desc_path = None
        try:
            if code_content:
                # Mode: has code → use code-analysis approach
                task_desc_path = self.task_desc_gen.generate(
                    task_name=task_name,
                    code_content=code_content,
                )
            elif paper_markdown:
                # Mode: paper-only → use 4-stage extraction
                task_desc_path = self.task_desc_gen.generate(
                    task_name=task_name,
                    paper_markdown=paper_markdown,
                )
            else:
                # Last resort: use abstract as a mini-paper
                mini_paper = f"# {paper_info.title}\n\n{paper_info.abstract}"
                task_desc_path = self.task_desc_gen.generate(
                    task_name=task_name,
                    paper_markdown=mini_paper,
                )

            if task_desc_path:
                logger.info(f"[E2E]   Task desc → {task_desc_path}")
                result["steps"]["task_description"] = task_desc_path
            else:
                logger.warning(f"[E2E]   Task description generation failed")
                result["steps"]["task_description"] = "failed"
        except Exception as e:
            logger.error(f"[E2E]   Task desc error: {e}")
            result["steps"]["task_description"] = f"error: {e}"

        # ── Step 6: Setup Python environment ──
        # Environment convention:
        #   ragas  = main pipeline env (read-only, no packages installed)
        #   paddle_env = OCR only
        #   openhands env = OpenHands only
        # Sandbox will create a per-task conda env if additional packages are needed.
        logger.info(f"[E2E] Step 6: Python environment")
        RAGAS_PYTHON = "/home/yjh/.conda/envs/ragas/bin/python"
        python_path = RAGAS_PYTHON
        conda_env_name = "ragas"

        if skip_env_create:
            logger.info(f"[E2E]   Using ragas env: {python_path}")
            result["steps"]["python_env"] = python_path
        else:
            # Always default to ragas for inspection; sandbox handles missing deps
            if Path(RAGAS_PYTHON).exists():
                logger.info(f"[E2E]   Using ragas env: {python_path}")
                result["steps"]["python_env"] = python_path
            else:
                python_path = self.output_builder._get_fallback_python()
                logger.warning(f"[E2E]   ragas env not found, fallback: {python_path}")
                result["steps"]["python_env"] = f"fallback: {python_path}"

        # ── Step 7: Build outputs & schema report ──
        logger.info(f"[E2E] Step 7: Assemble output artifacts")
        try:
            outputs = self.output_builder.build_outputs(
                task_name=task_name,
                repo_dir=str(repo_dir) if repo_dir else None,
                gt_code_path=gt_code_path,
                task_desc_path=task_desc_path,
                conda_env_name=conda_env_name,
                paper_info={
                    "arxiv_id": paper_info.arxiv_id,
                    "title": paper_info.title,
                    "abstract": paper_info.abstract[:500] if paper_info.abstract else "",
                    "authors": paper_info.authors[:5],
                    "pdf_url": paper_info.pdf_url,
                    "categories": paper_info.categories,
                    "github_url": repo_url or "",
                    "has_code": has_code,
                },
            )
            result["outputs"] = outputs
            result["steps"]["output_assembly"] = "ok"
        except Exception as e:
            logger.error(f"[E2E]   Output assembly error: {e}", exc_info=True)
            result["steps"]["output_assembly"] = f"error: {e}"

        # ── Summary ──
        all_ok = all(
            v in ("ok",) or (isinstance(v, str) and not v.startswith("error") and v != "not_found")
            for v in result["steps"].values()
        )
        result["status"] = "completed" if all_ok else "partial"

        logger.info(f"{'='*60}")
        logger.info(f"[E2E] Pipeline {'COMPLETED' if all_ok else 'PARTIAL'} for {arxiv_id}")
        if result.get("outputs"):
            for k, v in result["outputs"].items():
                logger.info(f"  {k}: {v}")
        logger.info(f"{'='*60}")

        return result

    def process_batch(
        self,
        papers: List[Dict[str, str]],
        skip_env_create: bool = False,
    ) -> List[Dict]:
        """
        Process a batch of papers.

        Args:
            papers: list of dicts with keys "arxiv_id" and optional "repo_url"
            skip_env_create: skip conda env creation

        Returns:
            List of result dicts from process_paper_e2e
        """
        results = []
        for i, entry in enumerate(papers, 1):
            arxiv_id = entry.get("arxiv_id", "").strip()
            repo_url = entry.get("repo_url", "").strip() or None
            if not arxiv_id:
                logger.warning(f"Skipping entry {i}: no arxiv_id")
                continue

            logger.info(f"\n{'#'*60}")
            logger.info(f"# Paper {i}/{len(papers)}: {arxiv_id}")
            logger.info(f"{'#'*60}")

            try:
                if repo_url:
                    logger.warning("Ignoring manual repo_url override; discovery will run automatically.")
                res = self.process_paper_e2e(
                    arxiv_id=arxiv_id,
                    skip_env_create=skip_env_create,
                )
                results.append(res)
            except Exception as e:
                logger.error(f"Unhandled error processing {arxiv_id}: {e}", exc_info=True)
                results.append({
                    "arxiv_id": arxiv_id,
                    "status": "error",
                    "error": str(e),
                })

        # Print summary table
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH SUMMARY: {len(results)}/{len(papers)} processed")
        logger.info(f"{'='*60}")
        for r in results:
            status_icon = "✅" if r.get("status") == "completed" else "⚠️" if r.get("status") == "partial" else "❌"
            logger.info(f"  {status_icon} {r.get('arxiv_id', '?'):20s} | {r.get('status', '?'):10s} | {r.get('task_name', '?')}")
        return results

    # ── Helpers ──

    def _discover_code_sync(
        self,
        arxiv_id: str,
        title: str,
        paper_markdown: Optional[str] = None,
    ) -> Optional[str]:
        """Synchronous version of code discovery (no DB record needed)."""
        # First check if we already have a repo url recorded in database
        existing = self.db.get_paper_by_arxiv_id(arxiv_id)
        if existing and existing.github_url:
            logger.info(f"  Using cached repo url from database: {existing.github_url}")
            return existing.github_url

        # Level 1: Parse paper markdown for GitHub links
        if paper_markdown:
            url = self.local_parser.find_repo_url(paper_markdown)
            if url:
                logger.info(f"  Code found via PDF parse: {url}")
                return url

        # Level 2: PapersWithCode API
        url = self.pwc.find_repo_by_arxiv_id(arxiv_id)
        if url:
            logger.info(f"  Code found via PWC: {url}")
            return url

        # Level 3: Web search + LLM verify
        try:
            url = self.web_search.find_repo(title, arxiv_id)
            if url:
                logger.info(f"  Code found via web search: {url}")
                return url
        except Exception as e:
            logger.warning(f"  Web search failed: {e}")

        return None

    @staticmethod
    def _make_task_name(arxiv_id: str, title: str) -> str:
        """Derive a clean task name from arxiv ID and title."""
        import re
        # Take the first meaningful word(s) from the title
        # e.g. "SwinIR: Image Restoration Using Swin Transformer" → "SwinIR"
        clean = title.split(":")[0].strip()
        # Remove special chars, keep alphanumeric and spaces
        clean = re.sub(r'[^a-zA-Z0-9\s\-]', '', clean)
        words = clean.split()
        if words:
            # Use first word (or first 2 if first is very short)
            if len(words[0]) <= 3 and len(words) > 1:
                name = f"{words[0]}_{words[1]}"
            else:
                name = words[0]
        else:
            name = arxiv_id.replace(".", "_")
        # Sanitize
        name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        return name
