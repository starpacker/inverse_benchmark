"""
Phase 3 - OpenHands Runner

Manages the interaction with the OpenHands agent to:
1. Initialize the workspace
2. Run data extraction (Step 1)
3. Run code consolidation (Step 2)
4. Run evaluation script generation (Step 3)
"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# Import OpenHands core components (adjust import based on actual installation)
try:
    from openhands.core.main import run_controller
    from openhands.core.config import load_openhands_config
    from openhands.core.setup import generate_sid
    from openhands.core.schema import AgentState
    from openhands.events.action import MessageAction
except ImportError:
    # Fallback for when OpenHands is not installed in this env
    run_controller = None
    load_openhands_config = None
    generate_sid = None
    AgentState = None
    # Dummy MessageAction so type annotation doesn't fail
    class MessageAction:
        def __init__(self, content=""): self.content = content

from utils.logging_utils import get_logger
from phase3_openhands.prompt_templates import (
    STEP1_DATA_EXTRACTION_PROMPT,
    STEP2_CODE_CONSOLIDATION_PROMPT,
    STEP3_EVALUATION_SCRIPT_PROMPT,
    STEP4_DEBUG_PROMPT,
)

logger = get_logger(__name__)


class OpenHandsRunner:
    """Orchestrates OpenHands agent tasks."""

    def __init__(self, config: dict):
        self.config = config
        self.oh_config_path = config["openhands"]["config_path"]
        self.workspace_base = Path(config["openhands"]["staging_dir"])

        if not run_controller:
            logger.warning("OpenHands modules not found. Mock mode active?")

    async def run_task(
        self,
        task_name: str,
        repo_path: str,
        step: str,
        context: Optional[dict] = None,
    ) -> dict:
        """Run a specific step of the pipeline using OpenHands."""
        
        # 1. Setup workspace
        workspace_dir = self.workspace_base / task_name
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # If repo_path is provided, copy it to workspace if not already there
        if repo_path and Path(repo_path).exists():
            target_repo = workspace_dir / "repo"
            if not target_repo.exists():
                shutil.copytree(repo_path, target_repo)

        # 2. Select prompt based on step
        prompt = self._get_prompt_for_step(step, context)
        if not prompt:
            return {"status": "error", "message": f"Unknown step: {step}"}

        # 3. Decide execution mode: in-process vs subprocess
        use_subproc = self.config.get("openhands", {}).get("use_subprocess", True)
        if use_subproc or run_controller is None:
            # write prompt to file
            prompt_file = workspace_dir / f"{task_name}_{step}_prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            oh_script = Path(__file__).parent / "run_oh_subprocess.py"
            cmd = [
                "conda", "run", "-n", self.config.get("openhands", {}).get("conda_env", "openhands"),
                "python", str(oh_script),
                "--workspace", str(workspace_dir),
                "--prompt-file", str(prompt_file),
                "--max-iterations", str(self.config["openhands"]["step_limits"].get(step, 30)),
                "--config-toml", self.oh_config_path,
            ]
            logger.info(f"Invoking OpenHands subprocess: {' '.join(cmd[:5])} ...")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"OH subprocess failed: {result.stderr[:200]}")
                    return {"status": "error", "error": "subprocess_failed"}
                # prefer result file
                json_path = workspace_dir / "oh_result.json"
                if json_path.exists():
                    try:
                        import json
                        state = json.loads(json_path.read_text())
                    except Exception:
                        logger.warning("Failed to parse oh_result.json, falling back to stdout")
                        state = None
                else:
                    # parse stdout after marker
                    for line in result.stdout.splitlines():
                        if line.strip().startswith("{" ):
                            try:
                                state = json.loads(line)
                                break
                            except Exception:
                                continue
                    else:
                        state = None
                return self._process_state(state)
            except Exception as e:
                logger.error(f"OpenHands subprocess error: {e}")
                return {"status": "failed", "error": str(e)}

        # else run in-process
        try:
            oh_config = load_openhands_config(
                config_file=self.oh_config_path
            )
            oh_config.workspace_base = str(workspace_dir)
            oh_config.max_iterations = self.config["openhands"]["step_limits"].get(step, 30)
            
            # 4. Initialize session
            sid = generate_sid(oh_config, session_name=f"{task_name}_{step}")
            initial_action = MessageAction(content=prompt)

            # 5. Run controller
            logger.info(f"Starting OpenHands task: {task_name} step={step}")
            state = await run_controller(
                config=oh_config,
                initial_user_action=initial_action,
                sid=sid,
                headless_mode=True,
                fake_user_response_fn=self._auto_reply
            )

            # 6. Process result
            return self._process_state(state)

        except Exception as e:
            logger.error(f"OpenHands execution failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _get_prompt_for_step(self, step: str, context: dict) -> str:
        """Hydrate the prompt template with context variables."""
        if context is None:
            context = {}
        if step == "data_extraction":
            return STEP1_DATA_EXTRACTION_PROMPT
        elif step == "code_consolidation":
            return STEP2_CODE_CONSOLIDATION_PROMPT
        elif step == "evaluate_generation":
            return STEP3_EVALUATION_SCRIPT_PROMPT
        elif step == "debug":
            # Only the debug prompt has template variables ({failed_script}, {error_log})
            return STEP4_DEBUG_PROMPT.format(**context)
        return ""

    def _auto_reply(self, state) -> MessageAction:
        """Simple auto-reply to keep the agent moving if it asks confirmation."""
        return MessageAction(content="Please continue. If you are stuck, try a different approach.")

    def _process_state(self, state) -> dict:
        """Convert OpenHands state to result dict."""
        if not state:
            return {"status": "unknown"}

        # subprocess invocation writes a JSON file that is parsed back into
        # a dict here; in that case `state` is a plain dict rather than an
        # object with attributes.  Handle both cleanly.
        if isinstance(state, dict):
            # Normalize the status field.  The subprocess wrapper writes
            # a high-level "status" (success/error/max_iterations) AND a
            # raw "agent_state" string like "AgentState.FINISHED".
            # Prefer the high-level status, falling back to agent_state.
            raw_status = state.get("status", state.get("agent_state", "unknown"))
            # Normalize common variants so the orchestrator can compare
            # against a small set of known values.
            status_map = {
                "success": "finished",
                "AgentState.FINISHED": "finished",
                "AgentState.STOPPED": "finished",
                "max_iterations": "max_iterations",
                "AgentState.PAUSED": "max_iterations",
                "AgentState.ERROR": "error",
                "error": "error",
            }
            normalized = status_map.get(raw_status, raw_status)
            return {
                "status": normalized,
                "last_error": state.get("last_error"),
                "metrics": state.get("metrics", {}),
                "iterations": state.get("iteration", state.get("iterations", 0)),
            }

        # otherwise assume it's an AgentState-like object
        return {
            "status": state.agent_state.value if hasattr(state.agent_state, 'value') else str(state.agent_state),
            "last_error": state.last_error if hasattr(state, 'last_error') else None,
            "metrics": state.metrics if hasattr(state, 'metrics') else {},
            "iterations": state.iteration if hasattr(state, 'iteration') else 0
        }
