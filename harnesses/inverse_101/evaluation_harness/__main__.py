"""CLI entry point: python -m evaluation_harness <mode> ..."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .core.config import LLMConfig, RunConfig, TaskConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="evaluation_harness",
        description="Run computational-imaging benchmark evaluations.",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- shared argument helpers ----
    def _add_common(p):
        p.add_argument("--task", required=True, help="Task name, e.g. eht_black_hole_original")
        p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    def _add_llm(p):
        p.add_argument("--model", required=True,
                        help="Model name (looked up in config_llm.yaml, or use --base-url/--api-key)")
        p.add_argument("--base-url", default=None, help="Override: OpenAI-compatible base URL")
        p.add_argument("--api-key", default=None, help="Override: API key")

    def _add_run_opts(p):
        p.add_argument("--max-iterations", type=int, default=20)
        p.add_argument("--docker-image", default="imaging101-sandbox")
        p.add_argument("--timeout", type=int, default=600)
        p.add_argument("--output", default="results", help="Output directory")
        p.add_argument("--log-file", default=None, help="Interaction log path")

    # ---- plan ----
    plan_p = sub.add_parser("plan", help="Plan mode: generate approach.md + design.md, LLM-as-judge scoring")
    _add_common(plan_p)
    _add_llm(plan_p)
    _add_run_opts(plan_p)

    # ---- function ----
    func_p = sub.add_parser("function", help="Function mode: implement a target function, pytest scoring")
    _add_common(func_p)
    _add_llm(func_p)
    _add_run_opts(func_p)
    func_p.add_argument("--target", required=True,
                        help="module.function_name, e.g. preprocessing.load_observation")

    # ---- end2end ----
    e2e_p = sub.add_parser("end2end", help="End-to-end mode: full pipeline with quality metric scoring")
    _add_common(e2e_p)
    _add_llm(e2e_p)
    _add_run_opts(e2e_p)
    e2e_p.add_argument("--framework", default="react",
                       choices=["react", "pipeline", "claude-code"],
                       help="Agent framework (default: react)")
    e2e_p.add_argument("--level", default="L1", choices=["L1", "L2", "L3"],
                       help="Difficulty: L1=from scratch, L2=+approach, L3=+approach+design")

    # ---- prepare (claude-code sandbox) ----
    prep_p = sub.add_parser("prepare", help="Prepare sandbox for third-party agent evaluation")
    _add_common(prep_p)
    prep_p.add_argument("--level", default="L1", choices=["L1", "L2", "L3"])
    prep_p.add_argument("--workspace-dir", default=None, help="Custom workspace path")

    # ---- collect (claude-code results) ----
    coll_p = sub.add_parser("collect", help="Collect and score third-party agent results")
    _add_common(coll_p)
    coll_p.add_argument("--workspace-dir", required=True, help="Workspace path to collect from")
    coll_p.add_argument("--level", default="L1", choices=["L1", "L2", "L3"])
    coll_p.add_argument("--agent-name", default="unknown", help="Name of the agent")
    coll_p.add_argument("--output", default="results")

    # ---- summarize (function mode) ----
    sum_p = sub.add_parser("summarize", help="Generate summary.json for function-mode run directory")
    sum_p.add_argument("--dir", required=True, help="Run directory path")
    sum_p.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "plan": _handle_plan,
        "function": _handle_function,
        "end2end": _handle_end2end,
        "prepare": _handle_prepare,
        "collect": _handle_collect,
        "summarize": _handle_summarize,
    }
    dispatch[args.command](args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_task_dir(task_name: str) -> Path:
    """Resolve task directory from name."""
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "tasks" / task_name
    if not task_dir.exists():
        print(f"Error: task directory not found: {task_dir}", file=sys.stderr)
        sys.exit(1)
    return task_dir


def _build_llm_config(args) -> LLMConfig:
    """Build LLMConfig from CLI args: try config_llm.yaml first, then overrides."""
    if args.base_url and args.api_key:
        return LLMConfig(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    try:
        cfg = LLMConfig.from_yaml(args.model)
    except (FileNotFoundError, KeyError) as e:
        # Fall back to manual args / env
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = args.base_url or "https://api.openai.com/v1"
        if not api_key:
            print(f"Warning: {e}", file=sys.stderr)
            print("Provide --api-key or set OPENAI_API_KEY.", file=sys.stderr)
            sys.exit(1)
        cfg = LLMConfig(model=args.model, base_url=base_url, api_key=api_key)
    # Allow CLI overrides on top of YAML
    if args.base_url:
        cfg.base_url = args.base_url
    if args.api_key:
        cfg.api_key = args.api_key
    return cfg


def _make_log_file(args, mode: str) -> Path:
    """Build an interaction log file path."""
    if getattr(args, "log_file", None):
        return Path(args.log_file)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    extras = ""
    if mode == "function":
        extras = f"_{args.target}"
    elif mode == "end_to_end":
        extras = f"_{args.level}"
    log_dir = Path("logs") / "interactions"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{args.task}_{mode}{extras}_{ts}.md"


def _build_config(args, mode: str, **overrides) -> RunConfig:
    """Build RunConfig from parsed args."""
    llm = _build_llm_config(args)
    task_dir = _resolve_task_dir(args.task)
    return RunConfig(
        llm=llm,
        task=TaskConfig(
            task_name=args.task,
            task_dir=task_dir,
            mode=mode,
            target_function=getattr(args, "target", None),
            level=getattr(args, "level", "L1"),
        ),
        max_iterations=getattr(args, "max_iterations", 20),
        docker_image=getattr(args, "docker_image", "imaging101-sandbox"),
        timeout_seconds=getattr(args, "timeout", 600),
        output_dir=Path(getattr(args, "output", "results")),
        log_file=_make_log_file(args, mode),
        framework=getattr(args, "framework", "react"),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_plan(args) -> None:
    """Plan mode: approach.md + design.md generation, LLM-as-judge scoring."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _build_config(args, mode="plan")
    from .modes.plan.runner import run_plan_mode
    result = run_plan_mode(config)
    _print_summary(result, args)


def _handle_function(args) -> None:
    """Function mode: implement target function, pytest scoring."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _build_config(args, mode="function")
    from .modes.function.runner import run_function_mode
    result = run_function_mode(config)
    _print_summary(result, args)


def _handle_end2end(args) -> None:
    """End-to-end mode: dispatch to react/pipeline/claude-code framework."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _build_config(args, mode="end_to_end")
    from .modes.end2end.runner import run_end2end_mode
    result = run_end2end_mode(config)
    _print_summary(result, args)


def _handle_prepare(args) -> None:
    """Prepare a sandbox workspace for third-party agents."""
    from .frameworks.claude_code.sandbox import prepare_copilot_evaluation

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    task_dir = _resolve_task_dir(args.task)
    workspace_dir = Path(args.workspace_dir) if args.workspace_dir else None

    prep_result = prepare_copilot_evaluation(
        task_name=args.task,
        task_dir=task_dir,
        level=args.level,
        workspace_dir=workspace_dir,
    )

    print("=" * 60)
    print("SANDBOX PREPARED")
    print("=" * 60)
    print(f"  Task:      {args.task}")
    print(f"  Level:     {args.level}")
    print(f"  Workspace: {prep_result['workspace_path']}")
    print(f"  Prompt:    {prep_result['prompt_file']}")
    print(f"\nNext steps:")
    print(f"  1. cd {prep_result['workspace_path']}")
    print(f"  2. Paste .prompt.md into your agent")
    print(f"  3. Let the agent produce output/reconstruction.npy")
    print(f"  4. python -m evaluation_harness collect \\")
    print(f"       --task {args.task} --workspace-dir {prep_result['workspace_path']} \\")
    print(f"       --level {args.level} --agent-name <name>")
    print("=" * 60)


def _handle_collect(args) -> None:
    """Collect and score results from a third-party agent workspace."""
    from .frameworks.claude_code.sandbox import collect_copilot_results
    from .frameworks.claude_code.scorer import score_copilot_results

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    task_dir = _resolve_task_dir(args.task)
    workspace_path = Path(args.workspace_dir)

    if not workspace_path.exists():
        print(f"Error: workspace not found: {workspace_path}", file=sys.stderr)
        sys.exit(1)

    collect_copilot_results(workspace_path, task_dir, args.level)

    eval_result = score_copilot_results(
        workspace_path=workspace_path,
        task_dir=task_dir,
        task_name=args.task,
        level=args.level,
        agent_name=args.agent_name,
        framework="claude-code",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_agent = args.agent_name.replace("/", "_").replace(" ", "_")
    name = f"{args.task}_end_to_end_{args.level}_claude-code_{safe_agent}_{ts}.json"
    result_path = output_dir / name
    with open(result_path, "w") as f:
        json.dump(dataclasses.asdict(eval_result), f, indent=2, default=str)

    print(f"\nResults saved to: {result_path}")
    _print_summary(eval_result, None)


def _handle_summarize(args) -> None:
    """Generate summary.json for a function-mode run directory."""
    run_dir = Path(args.dir)
    if not run_dir.is_dir():
        print(f"Error: directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    modules = {}
    total_tests = total_passed = total_failed = total_tokens = 0
    total_wall_time = 0.0
    task_name = model = framework = ""

    for mod_dir in sorted(run_dir.iterdir()):
        result_file = mod_dir / "result.json"
        if not mod_dir.is_dir() or not result_file.exists():
            continue
        with open(result_file) as f:
            r = json.load(f)
        mod_name = mod_dir.name
        if not task_name:
            task_name = r.get("task_name", "")
            model = r.get("model", "")
            framework = r.get("framework", "")

        modules[mod_name] = {
            "tests_total": r["tests_total"],
            "tests_passed": r["tests_passed"],
            "tests_failed": r["tests_failed"],
            "test_pass_rate": r["test_pass_rate"],
            "iterations": r["iterations"],
            "wall_time_seconds": r["wall_time_seconds"],
            "total_tokens": r["total_tokens"],
            "stopped_reason": r["stopped_reason"],
        }
        total_tests += r["tests_total"]
        total_passed += r["tests_passed"]
        total_failed += r["tests_failed"]
        total_tokens += r["total_tokens"]
        total_wall_time += r["wall_time_seconds"]

    if not modules:
        print("Error: no result.json files found", file=sys.stderr)
        sys.exit(1)

    summary = {
        "task_name": task_name,
        "mode": "function",
        "model": model,
        "framework": framework,
        "aggregate": {
            "tests_total": total_tests,
            "tests_passed": total_passed,
            "tests_failed": total_failed,
            "test_pass_rate": round(total_passed / total_tests, 4) if total_tests else 0,
            "total_tokens": total_tokens,
            "total_wall_time_seconds": round(total_wall_time, 1),
        },
        "per_module": modules,
    }

    out = run_dir / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    agg = summary["aggregate"]
    print(f"\n{'=' * 60}")
    print(f"FUNCTION-MODE SUMMARY: {task_name}")
    print(f"{'=' * 60}")
    print(f"  Model:    {model}")
    print(f"  Overall:  {agg['tests_passed']}/{agg['tests_total']} passed ({agg['test_pass_rate']:.0%})")
    print(f"  Tokens:   {agg['total_tokens']:,}")
    print(f"  Time:     {agg['total_wall_time_seconds']:.0f}s")
    print(f"\nSaved to: {out}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_summary(result, args) -> None:
    """Print evaluation result summary."""
    print("\n" + "=" * 60)
    print(f"Task:      {result.task_name}")
    print(f"Mode:      {result.mode}")
    print(f"Model:     {result.model}")
    if hasattr(result, "framework") and result.framework:
        print(f"Framework: {result.framework}")
    if args and hasattr(args, "level"):
        print(f"Level:     {args.level}")
    print(f"Status:    {result.stopped_reason}")
    if result.tests_total > 0:
        print(f"Tests:     {result.tests_passed}/{result.tests_total} ({result.test_pass_rate:.0%})")
    if result.quality_metrics:
        qm = result.quality_metrics
        if "error" in qm:
            print(f"Quality:   ERROR - {qm['error']}")
        else:
            print(f"Quality:   NRMSE={qm.get('nrmse', 'N/A')}, NCC={qm.get('ncc', 'N/A')}, "
                  f"PSNR={qm.get('psnr', 'N/A')}, SSIM={qm.get('ssim', 'N/A')}")
    print(f"Tokens:    {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})")
    if hasattr(result, "llm_calls"):
        print(f"LLM calls: {result.llm_calls}")
    print(f"Time:      {result.wall_time_seconds:.1f}s ({result.iterations} iterations)")
    print("=" * 60)


if __name__ == "__main__":
    main()
