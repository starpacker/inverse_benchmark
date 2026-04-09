"""Function mode runner: implement target function, score with pytest."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from evaluation_harness.core.config import RunConfig
from evaluation_harness.core.llm_client import LLMClient
from evaluation_harness.core.scorer import EvalResult, Scorer
from evaluation_harness.frameworks.react.agent import Agent, AgentResult
from .prompts import function_prompt

log = logging.getLogger(__name__)

VISIBLE_PATHS = ["README.md", "plan", "evaluation", "data"]


def _make_runner(config: RunConfig):
    if shutil.which("docker"):
        from evaluation_harness.core.sandbox.docker_runner import DockerRunner
        return DockerRunner(
            image=config.docker_image,
            task_dir=config.task.task_dir,
            timeout=config.timeout_seconds,
        )
    else:
        log.warning("Docker not available - using local runner")
        from evaluation_harness.core.sandbox.local_runner import LocalRunner
        return LocalRunner(
            image=config.docker_image,
            task_dir=config.task.task_dir,
            timeout=config.timeout_seconds,
        )


def run_function_mode(config: RunConfig) -> EvalResult:
    """Execute function-mode evaluation."""
    target = config.task.target_function
    if not target:
        raise ValueError("function mode requires --target")

    task_dir = config.task.task_dir
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    client = LLMClient(config.llm)
    runner = _make_runner(config)
    runner.start(visible_paths=VISIBLE_PATHS)
    t0 = time.time()

    try:
        readme = _read_host_file(task_dir, "README.md")
        approach = _read_host_file(task_dir, "plan/approach.md")
        design = _read_host_file(task_dir, "plan/design.md")

        module = target.split(".")[0]
        test_path = f"evaluation/tests/test_{module}.py"
        test_content = _read_host_file(task_dir, test_path)

        # Seed dependency modules
        _seed_dependency_modules(task_dir, module, runner)

        prompt = function_prompt(readme, approach, design, target, test_content)
        agent = Agent(client, runner, config.max_iterations,
                      mode="function", log_file=config.log_file)
        agent_result = agent.run(prompt)

        wall_time = time.time() - t0
        scorer = Scorer(runner, config, llm_client=client)
        result = scorer.score(agent_result, client.total_usage, wall_time,
                              llm_calls=client.call_count)
        scorer.save(result, config.output_dir)
        return result
    finally:
        runner.stop()


def _seed_dependency_modules(task_dir: Path, target_module: str, runner) -> None:
    """Copy reference src/ modules (except the target) into the sandbox."""
    src_dir = task_dir / "src"
    if not src_dir.is_dir():
        log.warning("No src/ directory found in task - skipping dependency seeding")
        return

    runner.write_file("src/__init__.py", "")
    for py_file in sorted(src_dir.glob("*.py")):
        mod_name = py_file.stem
        if mod_name == "__init__":
            continue
        if mod_name == target_module:
            log.info("Skipping target module src/%s.py (agent must implement)", mod_name)
            continue
        content = py_file.read_text(encoding="utf-8")
        runner.write_file(f"src/{py_file.name}", content)
        log.info("Seeded dependency: src/%s.py", mod_name)


def _read_host_file(task_dir: Path, rel_path: str) -> str:
    p = task_dir / rel_path
    if not p.exists():
        return f"[File not found: {rel_path}]"
    return p.read_text(encoding="utf-8")
