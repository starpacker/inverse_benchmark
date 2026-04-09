"""
Phase 3 - Output Builder

Assembles the final pipeline outputs for a processed paper:
  1. working-folder  — cloned repo + data files
  2. gt-code-path    — path to the consolidated run_gt.py
  3. target-python-env — conda environment python path
  4. task-description — saved to /data/yjh/task_descriptions/
  5. schema report    — JSON summary of the paper record

Also integrates with the downstream agentic pipeline config format
(gt_code_index.yaml style).
"""

import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutputBuilder:
    """Builds the final output artifacts for a processed paper."""

    def __init__(self, config: dict):
        self.config = config
        self.workspace_base = Path(
            config.get("general", {}).get(
                "workspace_base", "/data/yjh/paper2executable_workspace"
            )
        )
        self.task_desc_dir = Path(
            config.get("baseline", {}).get(
                "task_descriptions_path", "/data/yjh/task_descriptions"
            )
        )
        self.output_dir = Path(
            config.get("openhands", {}).get(
                "output_dir", "/data/yjh/paper2executable_workspace/output"
            )
        )
        self.conda_env_prefix = config.get("sandbox", {}).get(
            "conda_env_prefix", "p2e_"
        )

    def build_outputs(
        self,
        task_name: str,
        repo_dir: Optional[str] = None,
        gt_code_path: Optional[str] = None,
        task_desc_path: Optional[str] = None,
        conda_env_name: Optional[str] = None,
        paper_info: Optional[dict] = None,
    ) -> Dict[str, str]:
        """
        Assemble and return all output paths.

        Returns a dict with keys:
            working_folder, gt_code_path, python_path,
            task_description_path, schema_report_path
        """
        # 1. Working folder
        working_folder = self._prepare_working_folder(task_name, repo_dir)

        # if OpenHands staging produced artifacts, copy them in
        staging = Path(self.config.get("openhands", {}).get("staging_dir", "")) / task_name
        if staging.exists():
            # .npy data files → dataset/  subfolder
            dataset_dir = working_folder / "dataset"
            dataset_dir.mkdir(exist_ok=True)

            # Collect ALL .npy from staging
            npy_files = list(staging.glob("*.npy"))
            for npy in npy_files:
                target = dataset_dir / npy.name
                if not target.exists():
                    try:
                        shutil.copy2(str(npy), str(target))
                        logger.info(f"Copied {npy.name} from staging → dataset/")
                    except Exception:
                        pass

            # Script files → working root
            for script_name in ("run_gt.py", "evaluate.py"):
                src = staging / script_name
                if src.exists():
                    target = working_folder / script_name
                    if not target.exists():
                        try:
                            shutil.copy2(str(src), str(target))
                            logger.info(f"Copied {script_name} from staging to working")
                        except Exception:
                            pass

            # result.json → working root
            result_src = staging / "result.json"
            if result_src.exists():
                result_target = working_folder / "result.json"
                if not result_target.exists():
                    try:
                        shutil.copy2(str(result_src), str(result_target))
                        logger.info("Copied result.json from staging to working")
                    except Exception:
                        pass

        # Rewrite data paths in scripts to use dataset/ prefix
        self._rewrite_data_paths(working_folder)

        # 2. GT code path — either exists already or we create a placeholder
        # prefer staging run_gt if present
        staging_gt = staging / "run_gt.py"
        if staging_gt.exists():
            final_gt_path = staging_gt
        elif gt_code_path and Path(gt_code_path).exists():
            final_gt_path = gt_code_path
        else:
            # Look inside working folder for run_gt.py or the main script
            final_gt_path = self._find_or_create_gt_script(working_folder)

        # 3. Python environment
        python_path = self._resolve_python_env(task_name, conda_env_name)

        # 4. Task description (should already be generated)
        if not task_desc_path:
            task_desc_path = str(
                self.task_desc_dir / f"{task_name}_description.md"
            )

        # 5. Schema report
        report_path = self._write_schema_report(
            task_name=task_name,
            working_folder=str(working_folder),
            gt_code_path=str(final_gt_path) if final_gt_path else "",
            python_path=python_path,
            task_desc_path=task_desc_path,
            paper_info=paper_info,
        )

        outputs = {
            "working_folder": str(working_folder),
            "gt_code_path": str(final_gt_path) if final_gt_path else "",
            "python_path": python_path,
            "task_description_path": task_desc_path,
            "schema_report_path": report_path,
        }

        logger.info(f"[{task_name}] Output artifacts assembled:")
        for k, v in outputs.items():
            logger.info(f"  {k}: {v}")

        return outputs

    # ──────────────────────────────────────────────────────────

    def _rewrite_data_paths(self, working_folder: Path):
        """
        Rewrite run_gt.py and evaluate.py so that all .npy references
        point to dataset/ instead of the working root.

        This ensures consistency with the downstream agentic pipeline
        which expects:
            dataset/input.npy, dataset/gt_output.npy,
            dataset/baseline.npy, dataset/aux_data.npy, ...
        """
        npy_names = set()
        dataset_dir = working_folder / "dataset"
        if dataset_dir.exists():
            npy_names = {f.name for f in dataset_dir.glob("*.npy")}

        if not npy_names:
            return

        for script_name in ("run_gt.py", "evaluate.py"):
            script_path = working_folder / script_name
            if not script_path.exists():
                continue

            content = script_path.read_text()
            original = content

            for npy_name in npy_names:
                # Rewrite bare string references like "input.npy" → "dataset/input.npy"
                # But skip if already prefixed with dataset/
                # Match patterns: "input.npy", 'input.npy', SCRIPT_DIR, "input.npy"
                # 1) Replace os.path.join(SCRIPT_DIR, "input.npy") → os.path.join(SCRIPT_DIR, "dataset/input.npy")
                content = content.replace(
                    f'os.path.join(SCRIPT_DIR, "{npy_name}")',
                    f'os.path.join(SCRIPT_DIR, "dataset/{npy_name}")',
                )
                content = content.replace(
                    f"os.path.join(SCRIPT_DIR, '{npy_name}')",
                    f"os.path.join(SCRIPT_DIR, 'dataset/{npy_name}')",
                )
                # 2) Replace standalone string literals like "input.npy" when used
                #    as arguments to load_and_prepare, np.load, np.save, open, etc.
                #    Use a negative lookbehind to avoid double-prefixing.
                for quote in ('"', "'"):
                    bare = f'{quote}{npy_name}{quote}'
                    prefixed = f'{quote}dataset/{npy_name}{quote}'
                    content = content.replace(bare, prefixed)
                    # Clean up any double dataset/dataset/ that might occur
                    content = content.replace(
                        f'{quote}dataset/dataset/{npy_name}{quote}',
                        f'{quote}dataset/{npy_name}{quote}',
                    )

            if content != original:
                script_path.write_text(content)
                logger.info(f"Rewrote data paths in {script_name} → dataset/ prefix")

    @staticmethod
    def postprocess_working_folder(working_folder_path: str):
        """
        Standalone post-processing: reorganize an existing working folder
        so that all .npy files live under dataset/.

        Can be called from CLI or as a utility:
            OutputBuilder.postprocess_working_folder("/path/to/working/SwinIR")
        """
        wf = Path(working_folder_path)
        dataset_dir = wf / "dataset"
        dataset_dir.mkdir(exist_ok=True)

        # Move all root-level .npy into dataset/
        moved = []
        for npy in wf.glob("*.npy"):
            target = dataset_dir / npy.name
            if not target.exists():
                shutil.move(str(npy), str(target))
                moved.append(npy.name)
            elif npy.resolve() != target.resolve():
                # target already exists, remove the duplicate at root
                npy.unlink()
                moved.append(f"{npy.name} (dup removed)")

        if moved:
            logger.info(f"Moved to dataset/: {moved}")

        # Collect all npy names now in dataset/
        npy_names = {f.name for f in dataset_dir.glob("*.npy")}

        # Rewrite scripts
        for script_name in ("run_gt.py", "evaluate.py"):
            script_path = wf / script_name
            if not script_path.exists():
                continue

            content = script_path.read_text()
            original = content

            for npy_name in npy_names:
                content = content.replace(
                    f'os.path.join(SCRIPT_DIR, "{npy_name}")',
                    f'os.path.join(SCRIPT_DIR, "dataset/{npy_name}")',
                )
                content = content.replace(
                    f"os.path.join(SCRIPT_DIR, '{npy_name}')",
                    f"os.path.join(SCRIPT_DIR, 'dataset/{npy_name}')",
                )
                for quote in ('"', "'"):
                    bare = f'{quote}{npy_name}{quote}'
                    prefixed = f'{quote}dataset/{npy_name}{quote}'
                    content = content.replace(bare, prefixed)
                    # Clean up any double dataset/dataset/ that might occur
                    content = content.replace(
                        f'{quote}dataset/dataset/{npy_name}{quote}',
                        f'{quote}dataset/{npy_name}{quote}',
                    )

            if content != original:
                script_path.write_text(content)
                logger.info(f"Rewrote data paths in {script_name} → dataset/ prefix")

        return npy_names

    def _prepare_working_folder(
        self, task_name: str, repo_dir: Optional[str]
    ) -> Path:
        """
        Prepare the working folder.
        If a repo was cloned, use it. Otherwise create a fresh directory.
        """
        working = self.workspace_base / "working" / task_name
        working.mkdir(parents=True, exist_ok=True)

        if repo_dir and Path(repo_dir).exists():
            # If repo was cloned elsewhere, symlink or copy relevant pieces
            repo_path = Path(repo_dir)
            target_repo = working / "repo"
            if not target_repo.exists():
                if repo_path != target_repo:
                    # Create a symlink for convenience
                    target_repo.symlink_to(repo_path)
                    logger.info(
                        f"Linked repo {repo_path} → {target_repo}"
                    )

        # Create standard sub-dirs
        (working / "dataset").mkdir(exist_ok=True)
        (working / "outputs").mkdir(exist_ok=True)

        return working

    def _find_or_create_gt_script(self, working_folder: Path) -> Optional[Path]:
        """Find or create a run_gt.py in the working folder."""
        # Check common locations
        for candidate in [
            working_folder / "run_gt.py",
            working_folder / "repo" / "run_gt.py",
            working_folder / "repo" / "main.py",
            working_folder / "repo" / "demo.py",
            working_folder / "repo" / "test.py",
        ]:
            if candidate.exists():
                return candidate

        # Search recursively for .py files in repo/
        repo_dir = working_folder / "repo"
        if repo_dir.exists():
            py_files = sorted(repo_dir.rglob("*.py"))
            # Prefer files with telltale names
            priority_names = [
                "run_gt.py", "main.py", "demo.py", "test.py",
                "inference.py", "evaluate.py", "predict.py",
            ]
            for pname in priority_names:
                for f in py_files:
                    if f.name == pname:
                        return f
            # Fall back to first .py
            if py_files:
                return py_files[0]

        return None

    def _resolve_python_env(
        self, task_name: str, conda_env_name: Optional[str]
    ) -> str:
        """
        Resolve the python executable path for the task.

        Environment convention:
          ragas      = main pipeline env (default)
          paddle_env = OCR only
          openhands  = OpenHands only

        We do NOT create new conda envs automatically — just look up existing ones.
        """
        # Priority 1: explicit env name
        if conda_env_name:
            python_path = self._get_conda_python(conda_env_name)
            if python_path:
                return python_path

        # Priority 2: ragas (the standard main env)
        ragas_path = "/home/yjh/.conda/envs/ragas/bin/python"
        if Path(ragas_path).exists():
            return ragas_path

        # Priority 3: any p2e_{task} env that was previously created
        env_name = f"{self.conda_env_prefix}{task_name}"
        python_path = self._get_conda_python(env_name)
        if python_path:
            return python_path

        # Fallback: system python
        return self._get_fallback_python()

    def _get_conda_python(self, env_name: str) -> Optional[str]:
        """Get the python path for a conda env if it exists (fast path check only)."""
        for base in ["/home/yjh/.conda/envs", "/usr/local/anaconda3/envs"]:
            candidate = Path(base) / env_name / "bin" / "python"
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _get_fallback_python() -> str:
        """Return a reasonable fallback python path."""
        try:
            result = subprocess.run(
                ["which", "python"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "/usr/bin/python3"

    def _write_schema_report(
        self,
        task_name: str,
        working_folder: str,
        gt_code_path: str,
        python_path: str,
        task_desc_path: str,
        paper_info: Optional[dict] = None,
    ) -> str:
        """Write the schema report JSON."""
        report_dir = self.output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{task_name}_report.json"

        report = {
            "task_name": task_name,
            "generated_at": datetime.utcnow().isoformat(),
            "pipeline": "paper2executable",
            "outputs": {
                "working_folder": working_folder,
                "gt_code_path": gt_code_path,
                "python_path": python_path,
                "task_description_path": task_desc_path,
            },
            "paper_info": paper_info or {},
            "status": "ready" if gt_code_path and Path(gt_code_path).exists() else "partial",
            "validation": {
                "working_folder_exists": Path(working_folder).exists(),
                "gt_code_exists": bool(gt_code_path) and Path(gt_code_path).exists(),
                "task_desc_exists": Path(task_desc_path).exists() if task_desc_path else False,
                "python_path_exists": Path(python_path).exists() if python_path else False,
            },
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Schema report written → {report_path}")
        return str(report_path)
