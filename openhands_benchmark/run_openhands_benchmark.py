#!/usr/bin/env python3
"""
OpenHands Benchmark Runner v2
==============================
Properly isolated sandbox per task:
  - Copies dataset/ + eval_script.py from end_sandbox (Phase-0 artifacts)
  - Does NOT copy gt_code/ (no answer leaking)
  - Creates a fresh workspace for each task
  - Cleans up runtime processes between tasks

Usage:
    python run_openhands_benchmark_v2.py                                    # all 26 tasks
    python run_openhands_benchmark_v2.py --task-filter "AMICO-master"       # single task
    python run_openhands_benchmark_v2.py --max-iterations 30                # more budget
"""

import os
import sys
import yaml
import json
import time
import shutil
import signal
import asyncio
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ── OpenHands imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openhands.core.main import run_controller
from openhands.core.config import load_openhands_config
from openhands.core.setup import generate_sid
from openhands.core.schema import AgentState
from openhands.events.action import MessageAction

# ── Constants (configure these for your environment, or set via environment variables) ──
TASK_DESCRIPTION_DIR = os.environ.get("BENCH_TASK_DESC_DIR", "/data/task_descriptions")
SOURCE_SANDBOX_DIR = os.environ.get("BENCH_SANDBOX_DIR", "/data/end_sandbox")           # Phase-0 prepared sandboxes
OPENHANDS_WORKSPACE_ROOT = os.environ.get("BENCH_WORKSPACE_DIR", "/data/openhands_bench")  # Fresh workspaces created here
RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR", "/data/openhands_results")
PIPELINE_CONFIG_DIR = os.environ.get("BENCH_CONFIG_DIR", "./config")
OPENHANDS_DIR = os.path.dirname(os.path.abspath(__file__))

# GPU_ID is set at runtime via --gpu-id; used for CUDA_VISIBLE_DEVICES and results isolation
GPU_ID: int | None = None


# ═══════════════════════════════════════════════════════════════════
# Sandbox Preparation
# ═══════════════════════════════════════════════════════════════════

def prepare_sandbox(task_name: str, python_path: str) -> str:
    """
    Create a clean, isolated workspace for a single task.

    Copies from end_sandbox/{task}_sandbox/:
      - dataset/          (input.npy, baseline.npy, ... but NOT gt_output.npy)
      - eval_script.py    (evaluation script, patched to read hidden gt)

    Anti-cheat measures:
      - gt_output.npy is moved to a hidden directory .eval_gt/ (not in dataset/)
      - eval_script.py is patched to read gt from .eval_gt/gt_output.npy
      - The agent cannot see or access gt_output.npy
      - No gt_code/ is copied

    Returns the absolute path to the fresh workspace.
    """
    source_dir = os.path.join(SOURCE_SANDBOX_DIR, f"{task_name}_sandbox")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source sandbox not found: {source_dir}")

    workspace = os.path.join(OPENHANDS_WORKSPACE_ROOT, f"{task_name}_workspace")

    # Always start fresh
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)

    # 1) Copy dataset/ directory, EXCLUDING gt_output.npy
    src_dataset = os.path.join(source_dir, "dataset")
    dst_dataset = os.path.join(workspace, "dataset")
    if os.path.isdir(src_dataset):
        os.makedirs(dst_dataset, exist_ok=True)
        copied_count = 0
        for fn in os.listdir(src_dataset):
            if fn == "gt_output.npy":
                continue  # Skip gt — will be hidden separately
            shutil.copy2(os.path.join(src_dataset, fn), os.path.join(dst_dataset, fn))
            copied_count += 1
        print(f"    ✓ Copied dataset/ ({copied_count} files, gt_output.npy excluded)")
    else:
        raise FileNotFoundError(f"dataset/ not found in {source_dir}")

    # 2) Hide gt_output.npy in .eval_gt/ (invisible to casual ls)
    gt_src = os.path.join(src_dataset, "gt_output.npy")
    hidden_gt_dir = os.path.join(workspace, ".eval_gt")
    os.makedirs(hidden_gt_dir, exist_ok=True)
    if os.path.isfile(gt_src):
        shutil.copy2(gt_src, os.path.join(hidden_gt_dir, "gt_output.npy"))
        print(f"    ✓ Hidden gt_output.npy in .eval_gt/")
    else:
        raise FileNotFoundError(f"gt_output.npy not found in {src_dataset}")

    # 3) Copy eval_script.py and PATCH it to read gt from hidden path
    src_eval = os.path.join(source_dir, "eval_script.py")
    dst_eval = os.path.join(workspace, "eval_script.py")
    if os.path.isfile(src_eval):
        with open(src_eval, "r", encoding="utf-8") as f:
            eval_content = f.read()
        # Patch: replace gt_path to point to hidden location
        eval_content = eval_content.replace(
            '"dataset/gt_output.npy"',
            '"{}"'.format(os.path.join(hidden_gt_dir, "gt_output.npy"))
        )
        # Also handle single-quote variant
        eval_content = eval_content.replace(
            "'dataset/gt_output.npy'",
            "'{}'".format(os.path.join(hidden_gt_dir, "gt_output.npy"))
        )
        with open(dst_eval, "w", encoding="utf-8") as f:
            f.write(eval_content)
        print(f"    ✓ Copied eval_script.py (patched gt path)")
    else:
        raise FileNotFoundError(f"eval_script.py not found in {source_dir}")

    # 4) Verify: no gt_code leak, no gt_output in dataset/
    assert not os.path.exists(os.path.join(workspace, "gt_code")), "gt_code should not exist!"
    assert not os.path.exists(os.path.join(dst_dataset, "gt_output.npy")), "gt_output.npy should not be in dataset/!"

    # 5) Write a README for the agent
    readme = f"""# Workspace: {task_name}
## Contents
- dataset/         — Contains input.npy (measurements), baseline.npy, and other data files
- eval_script.py   — Evaluation script. Run: `{python_path} eval_script.py output.npy` to evaluate your solution

## Goal
Write a Python solver script (solver.py) that produces `output.npy` in this directory.
The eval_script.py will compare your output.npy against the ground truth.
"""
    with open(os.path.join(workspace, "README.md"), "w") as f:
        f.write(readme)

    print(f"    ✓ Workspace ready: {workspace}")
    print(f"    Contents: {sorted(os.listdir(workspace))}")
    return workspace


# ═══════════════════════════════════════════════════════════════════
# Prompt Construction
# ═══════════════════════════════════════════════════════════════════

def load_task_description(task_name: str) -> str:
    desc_path = os.path.join(TASK_DESCRIPTION_DIR, f"{task_name}_description.md")
    if os.path.isfile(desc_path):
        with open(desc_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"Solve the inverse problem for task: {task_name}"


def get_dataset_info(workspace: str) -> str:
    """Summarize what's in the dataset/ directory."""
    dataset_dir = os.path.join(workspace, "dataset")
    if not os.path.isdir(dataset_dir):
        return "No dataset directory found."
    files = sorted(os.listdir(dataset_dir))
    lines = []
    for fn in files:
        fp = os.path.join(dataset_dir, fn)
        size = os.path.getsize(fp)
        lines.append(f"  - {fn} ({size:,} bytes)")
    return "\n".join(lines)


def get_eval_script_content(workspace: str) -> str:
    """Read the eval_script.py so the agent knows the evaluation criteria."""
    eval_path = os.path.join(workspace, "eval_script.py")
    if os.path.isfile(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            return f.read()
    return "# eval_script.py not found"


def build_task_prompt(task_name: str, task_description: str,
                      workspace: str, python_path: str) -> str:
    """
    Build a self-contained prompt for OpenHands.
    No gt_code path is provided — the agent must work from the task description,
    dataset files, and eval_script.py alone (same info level as a human).
    """
    dataset_info = get_dataset_info(workspace)
    eval_script = get_eval_script_content(workspace)

    prompt = f"""# Inverse Problem Solving Task: {task_name}

## Task Description
{task_description}

## Your Working Directory
`{workspace}`

## Available Files
### dataset/ directory:
{dataset_info}

### eval_script.py (Evaluation Script):
```python
{eval_script}
```

## Python Interpreter
Use: `{python_path}`

## Goal
You must write an **inverse solver Python script** that:
1. Loads the measurement data from `dataset/input.npy`
2. Recovers the original signal/parameters (the inverse problem)
3. Saves the result as `output.npy` in the working directory (`{workspace}/output.npy`)

## ⚠️ MANDATORY REQUIREMENTS (MUST follow ALL of these)

1. **You MUST create a Python file called `solver.py`** in the working directory.
   - Use the `str_replace_editor` tool with command `create` to write `solver.py`.
   - Do NOT rely solely on `python -c "..."` inline commands for the final solution.
   - `solver.py` must be a complete, self-contained, runnable script.

2. **You MUST run `solver.py`** to generate `output.npy`:
   ```bash
   cd {workspace} && {python_path} solver.py
   ```

3. **You MUST NOT copy, rename, or symlink any ground truth file as your output.**
   - Do NOT use `gt_output.npy` in any way to produce `output.npy`.
   - `output.npy` must be the result of YOUR inverse solver computation.
   - The ground truth is NOT available to you — it is hidden for evaluation only.

4. **After producing `output.npy`, run the evaluation:**
   ```bash
   cd {workspace} && {python_path} eval_script.py output.npy
   ```

## Evaluation
The eval script prints a JSON with metrics (e.g., PSNR/SSIM or relative error).
Higher PSNR (>20 dB) and higher SSIM (>0.8) indicate better reconstruction.

## Suggested Approach
1. First, inspect the dataset: load and print shapes/dtypes of all .npy files in `dataset/`
2. Read the task description to understand the forward model (physics/math)
3. Study `eval_script.py` to understand the expected output format and shape
4. Write `solver.py` implementing your inverse solver
5. Run `solver.py` to produce `output.npy`
6. Run `eval_script.py` to check your score
7. If the score is poor, edit `solver.py` and iterate

## Important Rules
- Work ONLY within `{workspace}`
- Use the Python interpreter `{python_path}`
- Your final deliverable is BOTH `solver.py` AND `output.npy`
- `solver.py` must be runnable: `{python_path} solver.py` should produce `output.npy`

Please begin!
"""
    return prompt


# ═══════════════════════════════════════════════════════════════════
# Runtime Cleanup
# ═══════════════════════════════════════════════════════════════════

def kill_action_servers(wait: int = 2):
    """Kill any leftover action_execution_server processes to free ports."""
    try:
        subprocess.run(
            ["pkill", "-f", "action_execution_server"],
            capture_output=True, text=True, timeout=10
        )
    except Exception:
        pass
    # Also kill any file_viewer_server leftovers
    try:
        subprocess.run(
            ["pkill", "-f", "file_viewer_server"],
            capture_output=True, text=True, timeout=10
        )
    except Exception:
        pass
    time.sleep(wait)


def is_infrastructure_error(result: dict) -> bool:
    """Check if a task failure was caused by infrastructure (not agent intelligence)."""
    err = result.get("error", "") or ""
    infra_patterns = [
        "RetryError", "Server process died", "Connection refused",
        "ConnectError", "RuntimeError", "EXCEPTION",
    ]
    return any(p in err for p in infra_patterns) and result.get("total_events", 0) < 5


# ═══════════════════════════════════════════════════════════════════
# Sandbox Artifact Saving (trajectory, thinking log, solver code)
# ═══════════════════════════════════════════════════════════════════

def save_sandbox_artifacts(workspace: str, config, task_name: str, python_path: str):
    """
    After the agent finishes, save all artifacts into the workspace:
      1. trajectory.json      — full trajectory (copy from trajectories dir)
      2. thinking_log.md      — human-readable thinking/reasoning log
      3. agent_solver.py      — extracted Python code that produces output.npy
    """
    traj_dir = getattr(config, 'save_trajectory_path', None)
    if not traj_dir or not os.path.isdir(traj_dir):
        print("    ⚠ No trajectory dir, skipping artifact save")
        return

    # Find the most recent trajectory for this task
    traj_files = sorted(
        [f for f in Path(traj_dir).glob('*.json')
         if task_name.replace('-', '') in f.stem.replace('-', '') or task_name[:12] in f.stem],
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not traj_files:
        # Fallback: just pick the most recent file
        traj_files = sorted(Path(traj_dir).glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)

    if not traj_files:
        print("    ⚠ No trajectory files found")
        return

    traj_path = traj_files[0]
    try:
        with open(traj_path) as f:
            traj = json.load(f)
    except Exception as e:
        print(f"    ⚠ Failed to read trajectory: {e}")
        return

    # ── 1. Copy trajectory.json into workspace ──
    dst_traj = os.path.join(workspace, "trajectory.json")
    shutil.copy2(traj_path, dst_traj)
    print(f"    ✓ Saved trajectory.json ({len(traj)} events)")

    # ── 2. Generate thinking_log.md ──
    thinking_log = _generate_thinking_log(traj, task_name)
    log_path = os.path.join(workspace, "thinking_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(thinking_log)
    print(f"    ✓ Saved thinking_log.md")

    # ── 3. Extract agent_solver.py ──
    solver_code = _extract_solver_code(traj, python_path)
    if solver_code:
        solver_path = os.path.join(workspace, "agent_solver.py")
        with open(solver_path, "w", encoding="utf-8") as f:
            f.write(solver_code)
        print(f"    ✓ Saved agent_solver.py ({len(solver_code)} chars)")
    else:
        print("    ⚠ Could not extract solver code (agent may not have produced output.npy)")


def _generate_thinking_log(traj: list, task_name: str) -> str:
    """Generate a human-readable Markdown log of the agent's reasoning process."""
    lines = [
        f"# OpenHands Agent Thinking Log: {task_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total events: {len(traj)}",
        f"",
        "---",
        "",
    ]

    step = 0
    for ev in traj:
        action = ev.get('action', '')
        observation = ev.get('observation', '')
        ts = ev.get('timestamp', '')[:19]  # trim microseconds

        if action == 'think':
            step += 1
            thought = ev.get('args', {}).get('thought', '') or ev.get('message', '')
            # Clean up the thought text
            if thought.startswith('I am thinking...: '):
                thought = thought[len('I am thinking...: '):]
            lines.append(f"## Step {step} — 💭 Think ({ts})")
            lines.append("")
            lines.append(thought.strip())
            lines.append("")

        elif action == 'run':
            step += 1
            cmd = ev.get('args', {}).get('command', '')
            lines.append(f"## Step {step} — 🔧 Execute ({ts})")
            lines.append("")
            lines.append("```bash")
            lines.append(cmd)
            lines.append("```")
            lines.append("")

        elif action == 'str_replace_editor':
            step += 1
            args = ev.get('args', {})
            sub_cmd = args.get('command', '')
            path = args.get('path', '')
            lines.append(f"## Step {step} — ✏️ Edit File ({ts})")
            lines.append(f"")
            lines.append(f"**Command**: `{sub_cmd}` on `{path}`")
            if sub_cmd == 'create':
                content = args.get('file_text', '')
                lines.append("")
                lines.append("```python")
                lines.append(content)
                lines.append("```")
            elif sub_cmd == 'str_replace':
                old = args.get('old_str', '')
                new = args.get('new_str', '')
                lines.append(f"\n**Replace**:\n```\n{old}\n```\n**With**:\n```\n{new}\n```")
            lines.append("")

        elif observation == 'run':
            content = ev.get('content', '')
            # Truncate very long outputs
            if len(content) > 2000:
                content = content[:1000] + f"\n\n... (truncated, {len(content)} chars total) ...\n\n" + content[-500:]
            lines.append("**Output:**")
            lines.append("```")
            lines.append(content)
            lines.append("```")
            lines.append("")

        elif observation == 'error':
            content = ev.get('content', '')
            lines.append(f"**❌ Error:** {content}")
            lines.append("")

        elif action == 'finish':
            step += 1
            content = ev.get('args', {}).get('content', '') or ev.get('message', '')
            lines.append(f"## Step {step} — ✅ Finish ({ts})")
            lines.append("")
            lines.append(content.strip())
            lines.append("")

        # Also capture the LLM's reasoning text from tool_call_metadata
        tcm = ev.get('tool_call_metadata', {})
        if tcm:
            mr = tcm.get('model_response', {})
            choices = mr.get('choices', [])
            if choices:
                msg = choices[0].get('message', {})
                reasoning = msg.get('content', '')
                if reasoning and reasoning.strip() and action in ('run', 'str_replace_editor', 'finish'):
                    lines.append(f"> **Agent reasoning:** {reasoning.strip()}")
                    lines.append("")

    return "\n".join(lines)


def _extract_solver_code(traj: list, python_path: str) -> str | None:
    """
    Extract the final Python code that produces output.npy.

    Strategy:
    1. Look for str_replace_editor 'create' commands that wrote a .py file
       which was then used to produce output.npy
    2. Look for the last `python -c "..."` command that called np.save('output.npy', ...)
    3. Look for any file the agent created in the workspace that produces output.npy
    """
    # Strategy 1: Find str_replace_editor file creations
    created_files = {}  # path -> content
    for ev in traj:
        if ev.get('action') == 'str_replace_editor':
            args = ev.get('args', {})
            if args.get('command') == 'create':
                fpath = args.get('path', '')
                content = args.get('file_text', '')
                if fpath and content:
                    created_files[fpath] = content

    # Strategy 2: Find the last inline python -c that saves output.npy
    last_inline_code = None
    for ev in traj:
        if ev.get('action') == 'run':
            cmd = ev.get('args', {}).get('command', '')
            # Match: python -c "...np.save('output.npy'..."
            if 'python' in cmd and '-c' in cmd and 'output.npy' in cmd and 'save' in cmd.lower():
                # Extract the code between -c "..." or -c '...'
                code = _extract_inline_python(cmd)
                if code:
                    last_inline_code = code

    # Strategy 3: Find if a created file was run to produce output.npy
    solver_from_file = None
    for ev in traj:
        if ev.get('action') == 'run':
            cmd = ev.get('args', {}).get('command', '')
            if 'output.npy' not in cmd:  # The run itself may not mention output.npy
                for fpath, content in created_files.items():
                    if os.path.basename(fpath) in cmd and 'output.npy' in content:
                        solver_from_file = content
                        break
            # Also check if agent explicitly ran a solver file
            for fpath, content in created_files.items():
                if fpath in cmd and ('output.npy' in content or 'np.save' in content):
                    solver_from_file = content

    # Prefer file-based solver, fallback to inline
    code = solver_from_file or None

    if not code and last_inline_code:
        # Wrap inline code into a proper script
        code = f'''#!/usr/bin/env python3
"""
Agent-generated solver (extracted from inline python -c command).
Run: {python_path} agent_solver.py
"""
{last_inline_code}
'''

    if not code:
        # Last resort: look for any created .py file with np.save
        for fpath, content in created_files.items():
            if 'np.save' in content or 'numpy' in content:
                code = content
                break

    return code


def _extract_inline_python(cmd: str) -> str | None:
    """Extract Python code from a `python -c "..."` command."""
    import re
    # Try to find -c "..." or -c '...'
    # The command often looks like: cd /path && python -c "code here"
    # Handle escaped quotes within the code

    # Pattern 1: python -c "..."  (the code may contain escaped quotes)
    match = re.search(r'python[^\s]*\s+-c\s+"(.*)"', cmd, re.DOTALL)
    if match:
        code = match.group(1)
        # Unescape common escapes
        code = code.replace('\\"', '"').replace('\\n', '\n')
        return code

    # Pattern 2: python -c '...'
    match = re.search(r"python[^\s]*\s+-c\s+'(.*)'", cmd, re.DOTALL)
    if match:
        code = match.group(1)
        code = code.replace("\\'", "'").replace('\\n', '\n')
        return code

    # Pattern 3: The command IS the python invocation with newlines
    # e.g., python -c "\nimport numpy...\n"
    idx = cmd.find('-c ')
    if idx >= 0:
        rest = cmd[idx+3:].strip()
        if rest.startswith('"'):
            # Find matching end quote (handle multiline)
            end = rest.rfind('"')
            if end > 0:
                code = rest[1:end]
                code = code.replace('\\"', '"')
                return code
        elif rest.startswith("'"):
            end = rest.rfind("'")
            if end > 0:
                code = rest[1:end]
                code = code.replace("\\'", "'")
                return code

    return None


# ═══════════════════════════════════════════════════════════════════
# Task Execution
# ═══════════════════════════════════════════════════════════════════

def auto_continue_response(state) -> MessageAction:
    return MessageAction(content="Please continue working on the task.")


async def run_single_task(
    task_info: dict,
    config,
    max_iterations: int,
) -> Dict[str, Any]:
    """Run a single task with OpenHands in a clean sandbox."""
    task_name = task_info["name"]
    python_path = task_info.get("python_path", "python")
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"▶ Task: {task_name}")
    print(f"{'='*60}")

    try:
        # Step 1: Prepare clean sandbox
        print("  [Step 1] Preparing sandbox...")
        workspace = prepare_sandbox(task_name, python_path)

        # Step 2: Load task description & build prompt
        print("  [Step 2] Building prompt...")
        task_description = load_task_description(task_name)
        task_prompt = build_task_prompt(task_name, task_description, workspace, python_path)

        # Step 3: Configure OpenHands for this task
        config.max_iterations = max_iterations
        config.workspace_base = workspace

        # Use unique session name to avoid accumulating history from previous runs
        import uuid
        unique_suffix = uuid.uuid4().hex[:8]
        session_name = f"oh_{task_name}_{unique_suffix}"
        sid = generate_sid(config, session_name)
        initial_action = MessageAction(content=task_prompt)

        # Step 4: Run agent
        print(f"  [Step 3] Running CodeActAgent (max {max_iterations} iterations)...")
        state = await run_controller(
            config=config,
            initial_user_action=initial_action,
            sid=sid,
            fake_user_response_fn=auto_continue_response,
            headless_mode=True,
        )

        elapsed = time.time() - start_time

        # Step 5: Analyze result
        success = False
        error_msg = None
        agent_state_str = "UNKNOWN"
        iteration_count = 0
        metrics_str = ""
        output_exists = os.path.isfile(os.path.join(workspace, "output.npy"))

        if state:
            agent_state = state.agent_state
            agent_state_str = str(agent_state)

            if agent_state == AgentState.FINISHED:
                success = output_exists
                if not output_exists:
                    error_msg = "Agent finished but output.npy not found"
            elif agent_state == AgentState.STOPPED:
                success = output_exists
                if not output_exists:
                    error_msg = "Agent stopped but output.npy not found"
            elif agent_state == AgentState.ERROR:
                error_msg = getattr(state, 'last_error', '') or "Agent ERROR state"
                # partial_success: agent errored but still produced output
                if output_exists:
                    success = True
            elif agent_state == AgentState.PAUSED:
                error_msg = "Max iterations reached"
                success = output_exists  # might still have produced output

            if hasattr(state, 'metrics') and state.metrics:
                metrics_str = str(state.metrics)

            for attr in ('iteration', 'local_iteration'):
                val = getattr(state, attr, None)
                if val is not None:
                    iteration_count = val
                    break

        # Step 6: Validate output integrity & check for solver.py
        solver_exists = False
        solver_file = None
        cheat_detected = False

        # Check for solver.py or any .py file the agent created
        for candidate in ['solver.py', 'solve.py', 'inverse_solver.py', 'agent_solver.py']:
            p = os.path.join(workspace, candidate)
            if os.path.isfile(p):
                solver_exists = True
                solver_file = candidate
                break

        # If no standard name, check for any .py the agent created (excluding eval_script.py, README.md)
        if not solver_exists:
            for fn in os.listdir(workspace):
                if fn.endswith('.py') and fn not in ('eval_script.py',):
                    fp = os.path.join(workspace, fn)
                    with open(fp, 'r', errors='replace') as f:
                        content = f.read()
                    if 'output.npy' in content or 'np.save' in content:
                        solver_exists = True
                        solver_file = fn
                        break

        # Anti-cheat: verify output.npy is not a byte-for-byte copy of gt_output.npy
        if output_exists:
            gt_hidden = os.path.join(workspace, ".eval_gt", "gt_output.npy")
            if os.path.isfile(gt_hidden):
                import hashlib
                def file_md5(path):
                    h = hashlib.md5()
                    with open(path, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            h.update(chunk)
                    return h.hexdigest()
                out_hash = file_md5(os.path.join(workspace, "output.npy"))
                gt_hash = file_md5(gt_hidden)
                if out_hash == gt_hash:
                    cheat_detected = True
                    success = False
                    error_msg = "CHEAT DETECTED: output.npy is identical to gt_output.npy"
                    print(f"  🚨 CHEAT: output.npy is a copy of gt_output.npy!")

        if not solver_exists and output_exists:
            print(f"  ⚠ No solver .py file found (agent used inline code only)")

        # Step 7: Run eval_script if output.npy exists and not cheating
        eval_result = None
        if output_exists and not cheat_detected:
            try:
                proc = subprocess.run(
                    [python_path, "eval_script.py", "output.npy"],
                    cwd=workspace, capture_output=True, text=True, timeout=120
                )
                if proc.returncode == 0:
                    eval_result = proc.stdout.strip()
                    try:
                        eval_result = json.loads(eval_result)
                    except json.JSONDecodeError:
                        pass
                    print(f"  📊 Eval result: {eval_result}")
                else:
                    print(f"  ⚠ Eval failed: {proc.stderr[:200]}")
            except Exception as e:
                print(f"  ⚠ Eval exception: {e}")

        # Step 8: Save sandbox artifacts (trajectory, thinking_log, solver code)
        print("  [Step 4] Saving artifacts to sandbox...")
        save_sandbox_artifacts(workspace, config, task_name, python_path)

        # Extract trajectory metrics if available
        traj_metrics = extract_trajectory_metrics(config)

        # Determine partial_success: agent didn't FINISH cleanly but produced valid output
        partial_success = (success and agent_state_str != "AgentState.FINISHED")

        result = {
            "task_name": task_name,
            "success": success,
            "partial_success": partial_success,
            "output_exists": output_exists,
            "solver_exists": solver_exists,
            "solver_file": solver_file,
            "cheat_detected": cheat_detected,
            "elapsed_sec": round(elapsed, 2),
            "error": error_msg,
            "iterations": iteration_count,
            "agent_state": agent_state_str,
            "eval_result": eval_result,
            "metrics": metrics_str,
            "workspace": workspace,
            "gpu_id": GPU_ID,
            **traj_metrics,
        }

        icon = "✓" if (success and not partial_success) else ("◐" if partial_success else ("⚠" if output_exists else "✗"))
        ps_tag = " [partial_success]" if partial_success else ""
        print(f"\n  [{icon}] '{task_name}' — {elapsed:.1f}s, output={'YES' if output_exists else 'NO'}, state={agent_state_str}{ps_tag}")
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        is_max_iter = "maximum iteration" in str(e).lower()

        if is_max_iter:
            # Check if output was produced before hitting max iterations
            ws = os.path.join(OPENHANDS_WORKSPACE_ROOT, f"{task_name}_workspace")
            output_exists = os.path.isfile(os.path.join(ws, "output.npy"))
            print(f"\n  ⚠ '{task_name}' max iterations in {elapsed:.1f}s, output={'YES' if output_exists else 'NO'}")
            
            # Run eval if output exists
            eval_result = None
            if output_exists:
                try:
                    python_path_task = task_info.get("python_path", "python")
                    proc = subprocess.run(
                        [python_path_task, "eval_script.py", "output.npy"],
                        cwd=ws, capture_output=True, text=True, timeout=120
                    )
                    if proc.returncode == 0:
                        eval_result = proc.stdout.strip()
                        try:
                            eval_result = json.loads(eval_result)
                        except json.JSONDecodeError:
                            pass
                        print(f"  📊 Eval result: {eval_result}")
                except Exception:
                    pass

            # Save artifacts even on max iteration
            try:
                save_sandbox_artifacts(ws, config, task_name, python_path)
            except Exception:
                pass

            return {
                "task_name": task_name,
                "success": output_exists,
                "output_exists": output_exists,
                "solver_exists": os.path.isfile(os.path.join(ws, "solver.py")),
                "solver_file": "solver.py" if os.path.isfile(os.path.join(ws, "solver.py")) else None,
                "cheat_detected": False,
                "elapsed_sec": round(elapsed, 2),
                "error": "Max iterations reached",
                "iterations": max_iterations,
                "agent_state": "MAX_ITERATIONS",
                "eval_result": eval_result,
                "metrics": "",
                "workspace": ws,
            }
        else:
            print(f"\n  ✗ EXCEPTION '{task_name}': {error_msg}")
            traceback.print_exc()
            ws = os.path.join(OPENHANDS_WORKSPACE_ROOT, f"{task_name}_workspace")
            try:
                save_sandbox_artifacts(ws, config, task_name, python_path)
            except Exception:
                pass
            return {
                "task_name": task_name,
                "success": False,
                "output_exists": False,
                "solver_exists": False,
                "solver_file": None,
                "cheat_detected": False,
                "elapsed_sec": round(elapsed, 2),
                "error": error_msg,
                "iterations": 0,
                "agent_state": "EXCEPTION",
                "eval_result": None,
                "metrics": "",
                "workspace": ws,
            }
    finally:
        # Always clean up runtime servers after each task
        kill_action_servers(wait=3)

        # Also clean up any leaked global state in OpenHands
        try:
            from openhands.runtime.impl.local.local_runtime import _RUNNING_SERVERS
            for sid_key in list(_RUNNING_SERVERS.keys()):
                try:
                    info = _RUNNING_SERVERS[sid_key]
                    if info.process and info.process.poll() is None:
                        info.process.terminate()
                        info.process.wait(timeout=5)
                except Exception:
                    pass
            _RUNNING_SERVERS.clear()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════

def extract_trajectory_metrics(config) -> dict:
    """Extract multi-dimensional metrics from the latest trajectory."""
    traj_dir = getattr(config, 'save_trajectory_path', None)
    if not traj_dir or not os.path.isdir(traj_dir):
        return {}
    # Find the most recent trajectory file
    files = sorted(Path(traj_dir).glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {}
    try:
        with open(files[0]) as f:
            traj = json.load(f)
        action_types = ['run', 'think', 'finish', 'str_replace_editor', 'execute_ipython_cell']
        total_actions = sum(1 for ev in traj if ev.get('action') in action_types)
        error_count = sum(1 for ev in traj if ev.get('observation') == 'error')
        effective_actions = total_actions  # errors are separate events, not actions
        action_breakdown = {}
        for a in action_types:
            cnt = sum(1 for ev in traj if ev.get('action') == a)
            if cnt > 0:
                action_breakdown[a] = cnt
        return {
            'total_events': len(traj),
            'effective_actions': effective_actions,
            'error_events': error_count,
            'action_breakdown': action_breakdown,
            'trajectory_file': str(files[0]),
        }
    except Exception:
        return {}


def save_results(results: List[Dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    gpu_suffix = f"_gpu{GPU_ID}" if GPU_ID is not None else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"openhands_results{gpu_suffix}_{ts}.json")
    report = {
        "timestamp": ts,
        "framework": "OpenHands",
        "agent": "CodeActAgent",
        "model": "cds/Claude-4.6-opus",
        "total_tasks": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "partial_success": sum(1 for r in results if r.get("partial_success")),
        "full_success": sum(1 for r in results if r["success"] and not r.get("partial_success")),
        "output_produced": sum(1 for r in results if r.get("output_exists")),
        "failed": sum(1 for r in results if not r["success"]),
        "total_time_sec": round(sum(r["elapsed_sec"] for r in results), 2),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Results saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main_async(args):
    task_config = load_yaml(args.task_config)
    all_tasks = task_config.get("tasks", [])
    if not all_tasks:
        raise ValueError(f"No tasks in {args.task_config}")

    # Filter
    task_filter = args.task_filter or os.environ.get("TASK_NAMES", "").strip()
    if task_filter:
        names = [n.strip() for n in task_filter.split(",") if n.strip()]
        tasks = [t for t in all_tasks if t["name"] in names]
        if not tasks:
            raise ValueError(f"No matching tasks: {names}")
        print(f"Filtered: {len(tasks)}/{len(all_tasks)} tasks")
    else:
        tasks = all_tasks
        print(f"Running all {len(tasks)} tasks")

    # Skip already-done tasks if requested
    if args.skip_done:
        done_tasks = set()
        import glob as glob_mod
        for f in glob_mod.glob(os.path.join(RESULTS_DIR, "openhands_results*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                for r in data.get("results", []):
                    # Skip only truly successful tasks (not infra failures)
                    if r.get("success") and not is_infrastructure_error(r):
                        done_tasks.add(r["task_name"])
            except Exception:
                pass
        if done_tasks:
            before = len(tasks)
            tasks = [t for t in tasks if t["name"] not in done_tasks]
            print(f"Skipping {before - len(tasks)} already-successful tasks: {sorted(done_tasks)}")
            print(f"Remaining: {len(tasks)} tasks")

    # Load config
    config_path = os.path.abspath(args.openhands_config)
    config = load_openhands_config(set_logging_levels=True, config_file=config_path)
    llm = config.get_llm_config()
    print(f"  Model   : {llm.model}")
    print(f"  Base URL: {llm.base_url}")
    print(f"  Runtime : {config.runtime}")
    print(f"  Max iter: {args.max_iterations}")

    # Clean up before start
    kill_action_servers(wait=3)

    results: List[Dict] = []
    t0 = time.time()
    max_retries = args.max_retries

    for idx, task in enumerate(tasks, 1):
        print(f"\n[Task {idx}/{len(tasks)}]")

        # Try up to max_retries times for infrastructure failures
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"  🔄 Retry attempt {attempt}/{max_retries} for '{task['name']}'...")
                kill_action_servers(wait=5)  # Extra cleanup on retry

            result = await run_single_task(task, config, args.max_iterations)

            if result["success"] or not is_infrastructure_error(result):
                break  # Either succeeded, or it's an agent-level failure (no point retrying)
            else:
                print(f"  ⚠ Infrastructure failure (attempt {attempt}/{max_retries}): {result.get('error', '')[:80]}")

        results.append(result)
        if idx < len(tasks):
            time.sleep(5)  # Extra breathing room between tasks

    total = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    succ = [r for r in results if r["success"]]
    full_succ = [r for r in results if r["success"] and not r.get("partial_success")]
    partial = [r for r in results if r.get("partial_success")]
    out = [r for r in results if r.get("output_exists")]
    fail = [r for r in results if not r["success"]]
    print(f"  Tasks           : {len(results)}")
    print(f"  Success (total) : {len(succ)} ({len(full_succ)} full + {len(partial)} partial)")
    print(f"  Output.npy      : {len(out)} produced")
    print(f"  Failed          : {len(fail)} ✗")
    print(f"  Time            : {total:.0f}s ({total/60:.1f} min)")

    if partial:
        print("\n  Partial success tasks (output produced but agent didn't FINISH):")
        for r in partial:
            eval_r = r.get('eval_result', {})
            psnr = eval_r.get('psnr', '?') if isinstance(eval_r, dict) else '?'
            print(f"    ◐ {r['task_name']}: state={r.get('agent_state','')} psnr={psnr} err={r.get('error','')[:60]}")

    if fail:
        print("\n  Failed tasks:")
        for r in fail:
            print(f"    ✗ {r['task_name']}: {r['error'] or 'no output'}")

    save_results(results, RESULTS_DIR)
    print(f"{'='*60}")
    return len(fail)


def main():
    parser = argparse.ArgumentParser(description="OpenHands Benchmark v2")
    parser.add_argument("--task-config",
        default=os.path.join(PIPELINE_CONFIG_DIR, "config_task_2.yaml"))
    parser.add_argument("--openhands-config",
        default=os.path.join(OPENHANDS_DIR, "config.toml"))
    parser.add_argument("--task-filter", help="Comma-separated task names")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--gpu-id", type=int, default=None,
        help="GPU ID (0-7). Sets CUDA_VISIBLE_DEVICES and isolates results.")
    parser.add_argument("--skip-done", action="store_true",
        help="Skip tasks that already have successful results in RESULTS_DIR")
    parser.add_argument("--max-retries", type=int, default=2,
        help="Max retries for infrastructure failures (RetryError, ConnectError)")
    args = parser.parse_args()

    # Set GPU if specified
    global GPU_ID
    if args.gpu_id is not None:
        GPU_ID = args.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f"  GPU ID    : {args.gpu_id} (CUDA_VISIBLE_DEVICES={args.gpu_id})")

    print("=" * 60)
    print("OpenHands Benchmark Runner v2")
    print("=" * 60)
    print(f"  Task Config : {args.task_config}")
    print(f"  OH Config   : {args.openhands_config}")
    print(f"  Max Iter    : {args.max_iterations}")
    print(f"  Results Dir : {RESULTS_DIR}")
    print(f"  Workspace   : {OPENHANDS_WORKSPACE_ROOT}")
    print()

    n_fail = asyncio.run(main_async(args))
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
