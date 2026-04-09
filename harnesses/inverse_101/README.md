# inverse-101: Evaluation Harness

Benchmark harness for evaluating LLM agents on computational imaging tasks.

## Three Evaluation Modes

| Mode | What it tests | Scoring |
|------|--------------|---------|
| `plan` | Generate approach.md + design.md | LLM-as-judge (pairwise + rubric) |
| `function` | Implement a specific function | pytest pass rate |
| `end2end` | Full pipeline from data to reconstruction | Quality metrics (NRMSE, NCC, PSNR, SSIM) |

## Quick Start

```bash
pip install -r requirements.txt
```

### Plan mode
```bash
python -m evaluation_harness plan \
    --task eht_black_hole_original \
    --model "gemini-3.1-pro-preview"
```

### Function mode
```bash
python -m evaluation_harness function \
    --task SSNP_ODT \
    --model "cds/Claude-4.6-opus" \
    --target preprocessing.load_observation
```

### End-to-end mode

**ReAct (single-agent):**
```bash
python -m evaluation_harness end2end \
    --task eht_black_hole_original \
    --model "cds/Claude-4.6-opus" \
    --framework react --level L1
```

**Multi-agent pipeline (Plan -> Architect -> Code -> Judge):**
```bash
python -m evaluation_harness end2end \
    --task eht_black_hole_original \
    --model "gemini-3.1-pro-preview" \
    --framework pipeline --level L1
```

**Claude Code (sandbox-only, for third-party agents):**
```bash
python -m evaluation_harness end2end \
    --task eht_black_hole_original \
    --model dummy \
    --framework claude-code --level L1
```

Or use the two-step workflow:
```bash
# Step 1: Prepare sandbox
python -m evaluation_harness prepare --task eht_black_hole_original --level L1

# Step 2: (run your agent in the sandbox)

# Step 3: Collect and score
python -m evaluation_harness collect \
    --task eht_black_hole_original \
    --workspace-dir /path/to/workspace \
    --level L1 --agent-name claude-code
```

## Switching Models

Models are configured in `config_llm.yaml`. Just use `--model <name>`:

```bash
# Test with Claude
python -m evaluation_harness end2end --task TASK --model "cds/Claude-4.6-opus" --framework react

# Test with Gemini
python -m evaluation_harness end2end --task TASK --model "gemini-3.1-pro-preview" --framework react
```

## Difficulty Levels (End-to-end)

| Level | Given to agent | Agent must |
|-------|---------------|-----------|
| L1 | Task description only | Plan + implement from scratch |
| L2 | Task description + approach.md | Design + implement |
| L3 | Task description + approach.md + design.md | Implement only |

## Project Structure

```
evaluation_harness/
    __main__.py              # CLI entry point
    core/
        config.py            # LLMConfig, TaskConfig, RunConfig
        llm_client.py        # OpenAI-compatible HTTP client
        scorer.py            # EvalResult + mode-specific scoring
        plan_scorer.py       # LLM-as-judge (pairwise + rubric)
        visualizer.py        # Matplotlib evaluation figures
        sandbox/
            docker_runner.py # Docker container sandbox
            local_runner.py  # Local filesystem sandbox
    frameworks/
        react/
            agent.py         # ReAct single-agent loop
            prompts.py       # All prompt templates
        multi_agent/
            pipeline.py      # Plan->Architect->Code->Judge pipeline
            agents/          # Individual agent implementations
        claude_code/
            sandbox.py       # Third-party agent sandbox setup
            prompts.py       # Level-specific prompt generation
            scorer.py        # Copilot result scoring
    modes/
        plan/runner.py       # Plan mode orchestration
        function/runner.py   # Function mode orchestration
        end2end/runner.py    # End-to-end mode orchestration
```
