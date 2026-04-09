"""
Phase 2 - PDF Processor (OCR Wrapper)

Wraps the existing PaddleOCR-based PDF-to-Markdown tool.
Downloads PDFs from ArXiv if needed, then converts them to clean Markdown
for better code link extraction.
"""

import subprocess
import shutil
import time
import os
from pathlib import Path
from typing import Optional

import requests
import arxiv

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PDFProcessor:
    """Handles PDF downloading and OCR conversion."""

    def __init__(self, config: dict, project_root: Optional[str] = None):
        self.config = config
        self.ocr_config = config["discovery"].get("pdf_ocr", {})
        self.enabled = self.ocr_config.get("enabled", False)

        self.python_env = self.ocr_config.get("python_env")
        # Resolve script_path relative to project root
        raw_script = self.ocr_config.get("script_path", "tools/run_ocr_tool.py")
        if project_root:
            self.script_path = str(Path(project_root) / raw_script)
        else:
            self.script_path = str(Path(__file__).resolve().parent.parent / raw_script)
        self.output_base = Path(self.ocr_config.get("output_dir", "ocr_results"))
        self.output_base.mkdir(parents=True, exist_ok=True)

    def process_paper(self, arxiv_id: str, pdf_url: Optional[str] = None) -> Optional[str]:
        """
        Download PDF (if needed) and convert to Markdown.
        Returns the content of the markdown file.
        """
        if not self.enabled:
            logger.info("PDF OCR is disabled in config.")
            return None

        # 1. Download PDF
        pdf_path = self._get_pdf(arxiv_id, pdf_url)
        if not pdf_path:
            return None

        # 2. Run OCR
        md_path = self._run_ocr(pdf_path, arxiv_id)
        if not md_path:
            return None

        # 3. Read Content
        try:
            return md_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read generated markdown {md_path}: {e}")
            return None

    def _get_pdf(self, arxiv_id: str, pdf_url: Optional[str]) -> Optional[Path]:
        """Download PDF from ArXiv or use provided URL."""
        target_path = self.output_base / f"{arxiv_id}.pdf"
        
        # Return if already exists
        if target_path.exists() and target_path.stat().st_size > 1024:
            logger.debug(f"PDF already exists: {target_path}")
            return target_path

        # Try downloading
        url = pdf_url
        if not url:
            # Construct ArXiv PDF URL
            # Handle versioned IDs (2403.12345v1 -> 2403.12345)
            clean_id = arxiv_id.split('v')[0]
            url = f"https://arxiv.org/pdf/{clean_id}.pdf"

        logger.info(f"Downloading PDF for {arxiv_id} from {url}")
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                target_path.write_bytes(response.content)
                return target_path
            else:
                logger.error(f"Failed to download PDF: {response.status_code}")
                # Fallback: Try using arxiv library
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(search.results(), None)
                if paper:
                    paper.download_pdf(dirpath=str(self.output_base), filename=f"{arxiv_id}.pdf")
                    return target_path
        except Exception as e:
            logger.error(f"PDF download error: {e}")
        
        return None

    def _run_ocr(self, pdf_path: Path, arxiv_id: str) -> Optional[Path]:
        """Execute the external OCR script."""
        
        # Expected output path by the script (it uses the pdf filename stem)
        # If input is /path/to/2403.12345.pdf, output dir is /out
        # It typically produces /out/2403.12345.md
        expected_md = self.output_base / f"{pdf_path.stem}.md"
        
        if expected_md.exists():
            logger.info(f"Markdown already exists: {expected_md}")
            return expected_md

        cmd = [
            self.python_env,
            self.script_path,
            "--pdf", str(pdf_path),
            "--output_dir", str(self.output_base)
        ]
        # GPU configuration
        use_gpu = self.ocr_config.get("use_gpu", False)
        gpu_id = self.ocr_config.get("gpu_id", 1)  # default to GPU 1 to avoid busy GPU 0
        if use_gpu:
            cmd.extend(["--use_gpu", "--gpu_id", str(gpu_id)])
        
        logger.info(f"Running OCR for {arxiv_id}...")
        start_time = time.time()
        
        try:
            # Run the command — PaddleOCR model loading alone takes ~2 min,
            # plus processing time scales with page count.
            ocr_timeout = int(self.ocr_config.get("timeout", 600))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=ocr_timeout,
            )
            
            if result.returncode != 0:
                stderr = result.stderr or ""
                logger.error(f"OCR failed: {stderr}")
                # if the failure looks like a GPU out-of-memory situation,
                # retry on CPU once
                if "Out of memory" in stderr or "ResourceExhaustedError" in stderr:
                    logger.warning("Retrying OCR on CPU due to GPU OOM...")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = ""
                    cpu_cmd = [self.python_env, self.script_path, "--pdf", str(pdf_path), "--output_dir", str(self.output_base)]
                    # do not pass --use_gpu
                    try:
                        result2 = subprocess.run(
                            cpu_cmd,
                            capture_output=True,
                            text=True,
                            timeout=ocr_timeout,
                            env=env,
                        )
                        if result2.returncode == 0:
                            for line in result2.stdout.splitlines():
                                if line.startswith("RESULT_PATH:"):
                                    parsed_path = Path(line.split("RESULT_PATH:", 1)[1].strip())
                                    if parsed_path.exists():
                                        logger.info(f"OCR (CPU) complete → {parsed_path}")
                                        return parsed_path
                            if expected_md.exists():
                                logger.info("OCR (CPU) complete")
                                return expected_md
                    except Exception:
                        pass
                return None

            # Parse RESULT_PATH from stdout (the script prints it with a prefix)
            for line in result.stdout.splitlines():
                if line.startswith("RESULT_PATH:"):
                    parsed_path = Path(line.split("RESULT_PATH:", 1)[1].strip())
                    if parsed_path.exists():
                        logger.info(f"OCR complete in {time.time() - start_time:.1f}s → {parsed_path}")
                        return parsed_path

            # Fallback: check the expected filename
            if expected_md.exists():
                logger.info(f"OCR complete in {time.time() - start_time:.1f}s")
                return expected_md
            else:
                logger.error(f"OCR finished but output markdown was not found.")
                return None

        except subprocess.TimeoutExpired:
            logger.error("OCR timed out.")
            return None
        except Exception as e:
            logger.error(f"Error running OCR script: {e}")
            return None
