"""
GitHub Utilities

Handles repository cloning and basic repo operations.
When git clone fails (e.g. firewall), provides human-feedback mechanism:
  user downloads zip manually → uploads to target_dir → pipeline continues.
"""

import subprocess
import shutil
import zipfile
import glob
from pathlib import Path
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class CloneFailedNeedManualUpload(Exception):
    """Raised when git clone fails and manual zip upload is required."""

    def __init__(self, repo_url: str, target_dir: str):
        self.repo_url = repo_url
        self.target_dir = target_dir
        self.zip_download_url = repo_url.rstrip("/") + "/archive/refs/heads/main.zip"
        super().__init__(
            f"Git clone failed for {repo_url}.\n"
            f"Please manually download the repo zip and place it at:\n"
            f"  {target_dir}.zip\n"
            f"Download URL (try main or master branch):\n"
            f"  {self.zip_download_url}\n"
            f"Then re-run the pipeline."
        )


def clone_repo(
    repo_url: str,
    target_dir: str,
    depth: int = 1,
    timeout: int = 120,
) -> Optional[Path]:
    """
    Clone a GitHub repository to a target directory.

    Fallback chain:
      1. git clone --depth=1
      2. git clone (full)
      3. Check if user placed a .zip at target_dir.zip (manual upload)
      4. Raise CloneFailedNeedManualUpload so caller can report to human

    Returns:
        Path to the cloned/unzipped directory, or None on failure.
    Raises:
        CloneFailedNeedManualUpload: when clone fails and no zip found.
    """
    target = Path(target_dir)

    # Already done?
    if target.exists() and _dir_has_content(target):
        logger.info(f"Target directory already exists and is non-empty: {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)

    # Normalize URL
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    clone_url = url + ".git"

    # ── Attempt 1: shallow clone ──
    logger.info(f"Cloning {clone_url} → {target}")
    ok = _try_git_clone(["git", "clone", "--depth", str(depth), clone_url, str(target)], timeout)
    if ok:
        return target

    # ── Attempt 2: full clone ──
    logger.warning("Shallow clone failed, trying full clone...")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
    ok = _try_git_clone(["git", "clone", clone_url, str(target)], timeout)
    if ok:
        return target

    # ── Attempt 3: wget zip download (works when git is blocked by firewall) ──
    logger.warning("Git clone failed, trying wget zip download...")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
    for branch in ["main", "master"]:
        zip_url = f"{url}/archive/refs/heads/{branch}.zip"
        zip_dest = target.parent / f"{target.name}.zip"
        downloaded = _try_wget_zip(zip_url, str(zip_dest), timeout)
        if downloaded:
            result = _unzip_repo(zip_dest, target)
            if result:
                logger.info(f"Wget zip download succeeded (branch={branch})")
                return result

    # ── Attempt 4: check for manually uploaded zip ──
    zip_path = _find_manual_zip(target)
    if zip_path:
        logger.info(f"Found manual zip: {zip_path}")
        return _unzip_repo(zip_path, target)

    # ── All failed → raise for human action ──
    raise CloneFailedNeedManualUpload(repo_url, str(target))


def load_from_zip(zip_path: str, target_dir: str) -> Optional[Path]:
    """Unzip a manually downloaded repo zip into target_dir."""
    target = Path(target_dir)
    zp = Path(zip_path)
    if not zp.exists():
        logger.error(f"Zip file not found: {zp}")
        return None
    return _unzip_repo(zp, target)


# ── Internal helpers ──

def _try_git_clone(cmd: list, timeout: int) -> bool:
    """Run a git clone command. Returns True on success."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            logger.info(f"Clone successful: {cmd[-1]}")
            return True
        logger.error(f"git clone failed (rc={result.returncode}): {result.stderr.strip()[:200]}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Clone timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"Clone error: {e}")
        return False


def _try_wget_zip(url: str, dest: str, timeout: int) -> bool:
    """Download a zip file using wget. Returns True on success."""
    try:
        logger.info(f"wget: {url} → {dest}")
        result = subprocess.run(
            ["wget", "-q", "--timeout=30", "-O", dest, url],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and Path(dest).exists() and Path(dest).stat().st_size > 1024:
            logger.info(f"wget succeeded: {Path(dest).stat().st_size:,} bytes")
            return True
        logger.warning(f"wget failed (rc={result.returncode}): {result.stderr.strip()[:200]}")
        # Clean up empty/broken file
        if Path(dest).exists():
            Path(dest).unlink()
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"wget timed out after {timeout}s for {url}")
        if Path(dest).exists():
            Path(dest).unlink()
        return False
    except Exception as e:
        logger.warning(f"wget error: {e}")
        return False


def _find_manual_zip(target: Path) -> Optional[Path]:
    """Check common locations for a manually uploaded zip."""
    candidates = [
        target.parent / f"{target.name}.zip",          # repos/SwinIR.zip
        target / "repo.zip",                             # repos/SwinIR/repo.zip
        target.parent / f"{target.name}-main.zip",      # repos/SwinIR-main.zip
        target.parent / f"{target.name}-master.zip",     # repos/SwinIR-master.zip
    ]
    for c in candidates:
        if c.exists():
            return c
    # Also check glob
    for g in target.parent.glob(f"{target.name}*.zip"):
        return g
    return None


def _unzip_repo(zip_path: Path, target: Path) -> Optional[Path]:
    """Unzip a repo zip, handling the common single-root-folder pattern."""
    try:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)

        # GitHub zips typically have a single root folder like "SwinIR-main/"
        # Move contents up one level if so
        children = list(target.iterdir())
        if len(children) == 1 and children[0].is_dir():
            inner = children[0]
            # Move all inner contents to target
            for item in inner.iterdir():
                shutil.move(str(item), str(target / item.name))
            inner.rmdir()
            logger.info(f"Unzipped and flattened: {zip_path} → {target}")
        else:
            logger.info(f"Unzipped: {zip_path} → {target}")

        return target
    except Exception as e:
        logger.error(f"Unzip failed: {e}")
        return None


def _dir_has_content(d: Path) -> bool:
    """Check if directory has at least one real file (not just .git)."""
    for item in d.iterdir():
        if item.name != ".git":
            return True
    return False


def find_main_scripts(repo_dir: str) -> list[str]:
    """
    Find candidate main/entry scripts in a cloned repo.
    Returns a list of relative paths to Python files that look like entry points.
    """
    repo = Path(repo_dir)
    candidates = []

    # Priority patterns for entry-point scripts
    priority_names = [
        "main.py", "run.py", "test.py", "demo.py", "eval.py",
        "evaluate.py", "inference.py", "predict.py", "reconstruct.py",
        "train.py", "example.py", "run_gt.py",
    ]

    # Search top-level first, then one level deep
    for depth_pattern in ["*.py", "*/*.py", "examples/*.py", "demo/*.py", "scripts/*.py"]:
        for f in repo.glob(depth_pattern):
            if f.name in priority_names:
                candidates.insert(0, str(f.relative_to(repo)))
            elif f.name.startswith("test_") or f.name.startswith("demo_"):
                candidates.append(str(f.relative_to(repo)))

    return candidates


def find_requirements(repo_dir: str) -> Optional[str]:
    """
    Find a requirements file in the repo.
    Returns the path to the first found requirements file.
    """
    repo = Path(repo_dir)
    for name in [
        "requirements.txt",
        "requirements_test.txt",
        "requirements-dev.txt",
        "setup.py",
        "pyproject.toml",
        "environment.yml",
        "environment.yaml",
    ]:
        candidate = repo / name
        if candidate.exists():
            return str(candidate)
    return None


def get_repo_name_from_url(url: str) -> str:
    """Extract the repo name from a GitHub URL."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.split("/")[-1]
