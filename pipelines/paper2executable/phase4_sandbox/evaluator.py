"""
Phase 4 - Evaluator

Computes standard metrics (PSNR, SSIM, RMSE, MSE) to assess reconstruction quality.
Reads results from `evaluate.py` output (result.json) and logs them to the database.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from database.manager import DatabaseManager
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Parses evaluation results and updates paper metrics."""

    def __init__(self, db: DatabaseManager, config: dict):
        self.db = db
        self.config = config
        self.thresholds = config["sandbox"]["thresholds"]

    def process_results(
        self,
        paper_id: int,
        results_dir: str
    ) -> Dict[str, float]:
        """Read result.json and update database with metrics."""
        
        results_path = Path(results_dir) / "result.json"
        
        if not results_path.exists():
            logger.error(f"result.json not found in {results_dir}")
            return {}

        try:
            with open(results_path, "r") as f:
                metrics = json.load(f)

            # Update database
            paper = self.db.update_paper(
                paper_id,
                psnr=metrics.get("psnr"),
                ssim=metrics.get("ssim"),
                rmse=metrics.get("rmse"),
                mse=metrics.get("mse"),
                evaluate_status="success",
                pipeline_stage="validated"
            )

            if paper:
                logger.info(
                    f"Metrics updated for Paper {paper_id}: "
                    f"PSNR={metrics.get('psnr'):.2f}, SSIM={metrics.get('ssim'):.2f}"
                )

            # Check thresholds
            self._check_success(metrics, paper_id)

            return metrics

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {results_path}")
            return {}
        except Exception as e:
            logger.error(f"Error processing metrics: {e}")
            return {}

    def _check_success(self, metrics: dict, paper_id: int):
        """Log success/failure based on thresholds."""
        psnr = metrics.get("psnr", 0)
        ssim = metrics.get("ssim", 0)

        min_psnr = self.thresholds.get("psnr_min", 20.0)
        min_ssim = self.thresholds.get("ssim_min", 0.5)

        if psnr >= min_psnr and ssim >= min_ssim:
            logger.info(f"Paper {paper_id} PASSED validation thresholds.")
            self.db.update_paper(paper_id, run_gt_status="success")
        else:
            reason = f"Low metrics: PSNR={psnr:.2f} < {min_psnr}, SSIM={ssim:.2f} < {min_ssim}"
            logger.warning(f"Paper {paper_id} FAILED validation: {reason}")
            self.db.update_paper(
                paper_id,
                run_gt_status="failed",
                failure_reason=reason
            )
