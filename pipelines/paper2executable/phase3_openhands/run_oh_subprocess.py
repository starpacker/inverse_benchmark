#!/usr/bin/env python3
"""
Phase 3 - OpenHands Subprocess Invoker

Runs OpenHands as a **subprocess** in the `openhands` conda environment,
because importing openhands in the `ragas` env hangs.

This script is designed to be called from the orchestrator:
    conda run -n openhands python run_oh_subprocess.py \
        --workspace /path/to/workspace \
        --prompt-file /path/to/prompt.txt \
        --max-iterations 40 \
        --config-toml /home/yjh/OpenHands/config.toml

It can also be executed standalone for testing.
"""

import asyncio
import json
import os
import sys
import time
import uuid
import argparse
from pathlib import Path
from datetime import datetime

# OpenHands must be importable in THIS env (openhands conda env)
sys.path.insert(0, "/home/yjh/OpenHands")

from openhands.core.main import run_controller
from openhands.core.config import load_openhands_config
from openhands.core.setup import generate_sid
from openhands.core.schema import AgentState
from openhands.events.action import MessageAction


def auto_continue(state) -> MessageAction:
    """Auto-reply when agent asks for human input."""
    return MessageAction(content="Please continue working on the task.")


async def run_task(
    workspace: str,
    prompt: str,
    max_iterations: int,
    config_toml: str,
    session_name: str = "",
) -> dict:
    """Run a single OpenHands task and return result dict."""
    start = time.time()

    config = load_openhands_config(config_file=config_toml)
    config.workspace_base = workspace
    config.max_iterations = max_iterations

    if not session_name:
        session_name = f"p2e_{uuid.uuid4().hex[:8]}"
    sid = generate_sid(config, session_name)
    initial_action = MessageAction(content=prompt)

    print(f"[OH] Starting session {session_name} (max_iter={max_iterations})")
    print(f"[OH] Workspace: {workspace}")

    state = await run_controller(
        config=config,
        initial_user_action=initial_action,
        sid=sid,
        fake_user_response_fn=auto_continue,
        headless_mode=True,
    )

    elapsed = time.time() - start
    result = {
        "session_name": session_name,
        "elapsed_seconds": round(elapsed, 1),
        "workspace": workspace,
    }

    if state:
        agent_state = state.agent_state
        result["agent_state"] = str(agent_state)
        result["iteration_count"] = getattr(state, "iteration", 0)
        result["last_error"] = getattr(state, "last_error", None)

        if agent_state in (AgentState.FINISHED, AgentState.STOPPED):
            result["status"] = "success"
        elif agent_state == AgentState.PAUSED:
            result["status"] = "max_iterations"
        elif agent_state == AgentState.ERROR:
            result["status"] = "error"
        else:
            result["status"] = str(agent_state)
    else:
        result["status"] = "no_state"
        result["agent_state"] = "NONE"

    # List files created in workspace (top-level only)
    ws_files = []
    for fn in sorted(os.listdir(workspace)):
        fp = os.path.join(workspace, fn)
        if os.path.isfile(fp):
            ws_files.append({"name": fn, "size": os.path.getsize(fp)})
        elif os.path.isdir(fp) and fn != ".eval_gt":
            subfiles = os.listdir(fp) if os.path.isdir(fp) else []
            ws_files.append({"name": fn + "/", "children": len(subfiles)})
    result["workspace_files"] = ws_files

    return result


def main():
    parser = argparse.ArgumentParser(description="Run OpenHands task via subprocess")
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt text file")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--config-toml", default="/home/yjh/OpenHands/config.toml")
    parser.add_argument("--session-name", default="")
    parser.add_argument("--result-file", default="", help="Write JSON result to this file")
    args = parser.parse_args()

    # Read prompt
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    result = asyncio.run(run_task(
        workspace=args.workspace,
        prompt=prompt,
        max_iterations=args.max_iterations,
        config_toml=args.config_toml,
        session_name=args.session_name,
    ))

    # Write result
    result_file = args.result_file or os.path.join(args.workspace, "oh_result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OH] Result written to {result_file}")
    print(f"[OH] Status: {result['status']}")

    # Also print to stdout for parsing
    print("---OH_RESULT_JSON---")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
