"""Plan mode runner: generate approach.md + design.md, score with LLM-as-judge."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from evaluation_harness.core.config import RunConfig
from evaluation_harness.core.llm_client import LLMClient
from evaluation_harness.core.scorer import EvalResult, Scorer
from evaluation_harness.frameworks.react.agent import Agent, AgentResult
from .prompts import plan_approach_prompt, plan_design_prompt

log = logging.getLogger(__name__)

VISIBLE_PATHS = ["README.md", "data"]


def _make_runner(config: RunConfig):
    """Create a Docker or local sandbox runner."""
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


def run_plan_mode(config: RunConfig) -> EvalResult:
    """Execute plan-mode evaluation."""
    task_dir = config.task.task_dir
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    client = LLMClient(config.llm)
    runner = _make_runner(config)
    runner.start(visible_paths=VISIBLE_PATHS)
    t0 = time.time()

    try:
        readme = _read_host_file(task_dir, "README.md")
        meta_data = _read_host_file(task_dir, "data/meta_data")

        agent = Agent(client, runner, config.max_iterations,
                      mode="plan", log_file=config.log_file)

        # Phase 1: approach
        log.info("Plan mode - generating approach.md")
        prompt1 = plan_approach_prompt(readme, meta_data)
        result1 = agent.run(prompt1)

        # Phase 2: design
        approach = runner.read_file("plan/approach.md")
        log.info("Plan mode - generating design.md")
        prompt2 = plan_design_prompt(readme, approach)
        result2 = agent.run(prompt2)

        # Merge results
        result2.files_written = result1.files_written + result2.files_written
        result2.iterations = result1.iterations + result2.iterations

        wall_time = time.time() - t0
        scorer = Scorer(runner, config, llm_client=client)
        result = scorer.score(result2, client.total_usage, wall_time,
                              llm_calls=client.call_count)
        scorer.save(result, config.output_dir)
        return result
    finally:
        runner.stop()


def _read_host_file(task_dir: Path, rel_path: str) -> str:
    p = task_dir / rel_path
    if not p.exists():
        return f"[File not found: {rel_path}]"
    return p.read_text(encoding="utf-8")
