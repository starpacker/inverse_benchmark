"""End-to-end mode runner: full pipeline, multiple framework options.

Frameworks:
  - react:      ReAct single-agent loop
  - pipeline:   Multi-agent pipeline (Plan->Architect->Code->Judge)
  - claude-code: Prepare sandbox for Claude Code (manual interaction)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from evaluation_harness.core.config import RunConfig
from evaluation_harness.core.llm_client import LLMClient
from evaluation_harness.core.scorer import EvalResult, Scorer
from evaluation_harness.frameworks.react.agent import Agent, AgentResult
from evaluation_harness.frameworks.multi_agent.pipeline import MultiAgentPipeline
from .prompts import (
    end_to_end_plan_prompt,
    end_to_end_impl_prompt,
    end_to_end_L2_plan_prompt,
    end_to_end_L2_impl_prompt,
    end_to_end_L3_impl_prompt,
)

log = logging.getLogger(__name__)


def _get_visible_paths(level: str) -> list[str]:
    """Return visible paths for end-to-end mode based on level."""
    base = ["README.md", "data", "requirements.txt"]
    if level == "L2":
        return base + ["plan/approach.md"]
    elif level == "L3":
        return base + ["plan"]
    return base  # L1


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


def run_end2end_mode(config: RunConfig) -> EvalResult:
    """Execute end-to-end evaluation, dispatching to the chosen framework."""
    task_dir = config.task.task_dir
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    framework = config.framework
    if framework == "claude-code":
        return _run_claude_code(config)

    # react or pipeline - both use sandbox + LLM
    client = LLMClient(config.llm)
    runner = _make_runner(config)
    visible = _get_visible_paths(config.task.level)
    runner.start(visible_paths=visible)
    t0 = time.time()

    try:
        if framework == "pipeline":
            agent_result = _run_multi_agent(config, client, runner)
        else:
            agent_result = _run_react(config, client, runner)

        wall_time = time.time() - t0
        scorer = Scorer(runner, config, llm_client=client)
        result = scorer.score(agent_result, client.total_usage, wall_time,
                              llm_calls=client.call_count)
        scorer.save(result, config.output_dir)
        return result
    finally:
        runner.stop()


# ---------------------------------------------------------------------------
# ReAct framework
# ---------------------------------------------------------------------------

def _run_react(config: RunConfig, client: LLMClient, runner) -> AgentResult:
    """End-to-end using the ReAct single-agent framework."""
    readme = _read_host_file(config.task.task_dir, "README.md")
    meta_data = _read_host_file(config.task.task_dir, "data/meta_data")
    level = config.task.level

    if level == "L3":
        approach = _read_host_file(config.task.task_dir, "plan/approach.md")
        design = _read_host_file(config.task.task_dir, "plan/design.md")
        runner.write_file("plan/approach.md", approach)
        runner.write_file("plan/design.md", design)
        impl_agent = Agent(client, runner, config.max_iterations,
                           mode="end_to_end", log_file=config.log_file)
        log.info("End-to-end [L3] - implementation only")
        impl_prompt = end_to_end_L3_impl_prompt(readme, meta_data, approach, design)
        return impl_agent.run(impl_prompt)

    elif level == "L2":
        approach = _read_host_file(config.task.task_dir, "plan/approach.md")
        plan_agent = Agent(client, runner, config.max_iterations,
                           mode="plan", log_file=config.log_file)
        log.info("End-to-end [L2] - Phase 1: design only")
        plan_prompt = end_to_end_L2_plan_prompt(readme, meta_data, approach)
        plan_result = plan_agent.run(plan_prompt)

        impl_agent = Agent(client, runner, config.max_iterations,
                           mode="end_to_end", log_file=config.log_file)
        approach_sandbox = runner.read_file("plan/approach.md")
        design_sandbox = runner.read_file("plan/design.md")
        log.info("End-to-end [L2] - Phase 2: implementation")
        impl_prompt = end_to_end_L2_impl_prompt(approach_sandbox, design_sandbox)
        impl_result = impl_agent.run(impl_prompt)
        impl_result.files_written = plan_result.files_written + impl_result.files_written
        impl_result.iterations = plan_result.iterations + impl_result.iterations
        return impl_result

    else:  # L1
        plan_agent = Agent(client, runner, config.max_iterations,
                           mode="plan", log_file=config.log_file)
        log.info("End-to-end [L1] - Phase 1: planning")
        plan_prompt = end_to_end_plan_prompt(readme, meta_data)
        plan_result = plan_agent.run(plan_prompt)

        impl_agent = Agent(client, runner, config.max_iterations,
                           mode="end_to_end", log_file=config.log_file)
        approach = runner.read_file("plan/approach.md")
        design = runner.read_file("plan/design.md")
        log.info("End-to-end [L1] - Phase 2: implementation")
        impl_prompt = end_to_end_impl_prompt(approach, design)
        impl_result = impl_agent.run(impl_prompt)
        impl_result.files_written = plan_result.files_written + impl_result.files_written
        impl_result.iterations = plan_result.iterations + impl_result.iterations
        return impl_result


# ---------------------------------------------------------------------------
# Multi-agent pipeline framework
# ---------------------------------------------------------------------------

def _run_multi_agent(config: RunConfig, client: LLMClient, runner) -> AgentResult:
    """End-to-end using multi-agent pipeline."""
    readme = _read_host_file(config.task.task_dir, "README.md")
    meta_data = _read_host_file(config.task.task_dir, "data/meta_data")
    requirements = _read_host_file(config.task.task_dir, "requirements.txt")
    level = config.task.level

    given_approach = None
    given_design = None
    if level in ("L2", "L3"):
        given_approach = _read_host_file(config.task.task_dir, "plan/approach.md")
        if given_approach.startswith("[File not found"):
            log.warning("L2/L3 requested but approach.md not found - falling back to L1")
            given_approach = None
            level = "L1"
    if level == "L3":
        given_design = _read_host_file(config.task.task_dir, "plan/design.md")
        if given_design.startswith("[File not found"):
            log.warning("L3 requested but design.md not found - falling back to L2")
            given_design = None
            level = "L2"

    log.info("End-to-end [pipeline, %s] - running multi-agent pipeline", level)
    pipeline = MultiAgentPipeline(
        client=client,
        runner=runner,
        max_iterations=config.max_iterations,
        log_file=config.log_file,
    )
    return pipeline.run(
        task_desc=readme,
        data_spec=meta_data,
        requirements=requirements,
        level=level,
        given_approach=given_approach,
        given_design=given_design,
    )


# ---------------------------------------------------------------------------
# Claude Code (sandbox-only) framework
# ---------------------------------------------------------------------------

def _run_claude_code(config: RunConfig) -> EvalResult:
    """End-to-end using Claude Code - just prepare sandbox."""
    from evaluation_harness.frameworks.claude_code.sandbox import (
        prepare_copilot_evaluation,
        collect_copilot_results,
    )
    from evaluation_harness.frameworks.claude_code.scorer import score_copilot_results

    task_dir = config.task.task_dir
    level = config.task.level

    print("=" * 60)
    print("CLAUDE CODE EVALUATION (sandbox mode)")
    print("=" * 60)

    prep_result = prepare_copilot_evaluation(
        task_name=config.task.task_name,
        task_dir=task_dir,
        level=level,
    )

    print(f"\nSandbox prepared at: {prep_result['workspace_path']}")
    print(f"Prompt saved to:    {prep_result['prompt_file']}")
    print(f"Level:              {level}")
    print(f"\nInstructions:")
    print(f"  1. cd {prep_result['workspace_path']}")
    print(f"  2. Paste the prompt from .prompt.md into Claude Code")
    print(f"  3. Let it produce output/reconstruction.npy")
    print(f"  4. Press ENTER here when done")
    print("=" * 60)

    try:
        input("\nPress ENTER when the agent has finished... ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)

    workspace_path = Path(prep_result["workspace_path"])
    collect_copilot_results(workspace_path, task_dir, level)

    eval_result = score_copilot_results(
        workspace_path=workspace_path,
        task_dir=task_dir,
        task_name=config.task.task_name,
        level=level,
        agent_name="claude-code",
        framework="claude-code",
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{config.task.task_name}_end_to_end_{level}_claude-code_{ts}.json"
    result_path = output_dir / name
    with open(result_path, "w") as f:
        json.dump(dataclasses.asdict(eval_result), f, indent=2, default=str)
    print(f"\nResults saved to: {result_path}")

    return eval_result


def _read_host_file(task_dir: Path, rel_path: str) -> str:
    p = task_dir / rel_path
    if not p.exists():
        return f"[File not found: {rel_path}]"
    return p.read_text(encoding="utf-8")
