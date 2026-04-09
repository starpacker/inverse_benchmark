"""
Phase 1 - Baseline Importer

Imports existing benchmark dataset papers into the global index database.
Scans /data/yjh/benchmark_dataset/ and /data/yjh/task_descriptions/
to bootstrap the dedup index with known papers.
"""

import json
import re
from pathlib import Path
from typing import Optional

import yaml

from database.manager import DatabaseManager
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaselineImporter:
    """Imports existing benchmark data into the paper index."""

    def __init__(self, db: DatabaseManager, config: dict):
        self.db = db
        self.config = config
        self.benchmark_path = Path(config["baseline"]["benchmark_dataset_path"])
        self.descriptions_path = Path(config["baseline"]["task_descriptions_path"])

    def import_all(self) -> dict:
        """Import all tasks from the existing benchmark dataset.

        Returns summary of imported/skipped counts.
        """
        stats = {"imported": 0, "skipped": 0, "errors": 0}

        if not self.benchmark_path.exists():
            logger.error(f"Benchmark dataset path not found: {self.benchmark_path}")
            return stats

        # Enumerate task directories
        task_dirs = sorted([
            d for d in self.benchmark_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        logger.info(f"Found {len(task_dirs)} task directories in benchmark dataset")

        for task_dir in task_dirs:
            task_name = task_dir.name
            try:
                result = self._import_task(task_name, task_dir)
                if result == "imported":
                    stats["imported"] += 1
                elif result == "skipped":
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Error importing task {task_name}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Baseline import complete: {stats['imported']} imported, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )
        return stats

    def _import_task(self, task_name: str, task_dir: Path) -> str:
        """Import a single task. Returns 'imported' or 'skipped'."""

        # Check if already imported (use task name as pseudo-title)
        if self.db.paper_exists(title=task_name):
            logger.debug(f"Skipping already-imported task: {task_name}")
            return "skipped"

        # Read task description for abstract/algorithm info
        abstract = self._read_task_description(task_name)

        # Read task_io_info.json for metadata
        io_info = self._read_io_info(task_dir)

        # Read golden_plan.json for algorithm details
        golden_plan = self._read_golden_plan(task_dir)

        # Determine data status
        data_status = self._check_data_status(task_dir)

        # Attempt to extract topics from description
        topics = self._extract_topics_from_name(task_name)

        # Build the paper record
        paper_data = {
            "title": task_name,
            "abstract": abstract,
            "topics": topics,
            "data_status": data_status,
            "source": "baseline_import",
            "pipeline_stage": "validated",  # Already validated in existing benchmark
            "workspace_path": str(task_dir),
            "output_path": str(task_dir),
        }

        # Add metrics if available from io_info
        if io_info and "baseline_metrics" in io_info:
            metrics = io_info["baseline_metrics"]
            if isinstance(metrics, dict):
                paper_data["psnr"] = metrics.get("psnr")
                paper_data["ssim"] = metrics.get("ssim")

        paper = self.db.add_paper(**paper_data)
        logger.info(f"Imported task: {task_name} (paper_id={paper.id})")
        return "imported"

    def _read_task_description(self, task_name: str) -> Optional[str]:
        """Read the markdown task description for a task."""
        desc_file = self.descriptions_path / f"{task_name}_description.md"
        if desc_file.exists():
            text = desc_file.read_text(encoding="utf-8", errors="replace")
            # Truncate to reasonable size for abstract field
            if len(text) > 5000:
                return text[:5000] + "\n... [truncated]"
            return text
        return None

    def _read_io_info(self, task_dir: Path) -> Optional[dict]:
        """Read task_io_info.json if available."""
        io_file = task_dir / "task_io_info.json"
        if io_file.exists():
            try:
                with open(io_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {io_file}")
        return None

    def _read_golden_plan(self, task_dir: Path) -> Optional[dict]:
        """Read golden_plan.json if available."""
        plan_file = task_dir / "golden_plan.json"
        if plan_file.exists():
            try:
                with open(plan_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {plan_file}")
        return None

    def _check_data_status(self, task_dir: Path) -> str:
        """Check what data files exist in the task directory."""
        has_input = (task_dir / "input.npy").exists()
        has_gt = (task_dir / "gt_output.npy").exists() or (task_dir / "gt_reference.npy").exists()

        if has_input and has_gt:
            return "real_data"
        elif has_input:
            return "partial"
        else:
            return "mock_only"

    def _extract_topics_from_name(self, task_name: str) -> list[str]:
        """Heuristic topic extraction from task name."""
        topic_keywords = {
            "tomography": ["tomography", "tomo", "ct", "sart", "astra"],
            "ptychography": ["ptycho", "ptylab", "ptyrad"],
            "deconvolution": ["deconv", "deblur", "psf"],
            "holography": ["holo", "dhm", "holoscope"],
            "light_field": ["lfm", "flfm", "light_field", "lensless"],
            "phase_retrieval": ["phase", "fpm"],
            "mri_reconstruction": ["mri", "mrf"],
            "electrical_impedance": ["eit", "pyeit"],
            "optical_coherence": ["oct", "cbort"],
            "compressed_sensing": ["cs", "cassi", "compressive"],
            "beam_propagation": ["bpm", "beam"],
            "seismic_inversion": ["seismic", "insar", "bayhunt"],
            "acoustic_imaging": ["acoustic", "pat", "photoacoustic"],
            "gravitational_lensing": ["lenstronomy", "quasar"],
            "fluorescence_microscopy": ["fluorescence", "ism", "smlm", "storm"],
            "super_resolution": ["super_res", "sr"],
            "adaptive_optics": ["oopao", "wavefront", "zernike"],
            "spectroscopy": ["spectral", "raman", "carspy", "phasor"],
            "inverse_problems": [],  # Added to all
        }

        name_lower = task_name.lower()
        found_topics = ["inverse_problems"]  # Always include base topic

        for topic, keywords in topic_keywords.items():
            for kw in keywords:
                if kw in name_lower:
                    found_topics.append(topic)
                    break

        return list(set(found_topics))
