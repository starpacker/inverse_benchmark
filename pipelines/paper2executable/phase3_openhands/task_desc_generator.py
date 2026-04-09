"""
Phase 3 - Task Description Generator

Two modes:
  1. Paper-only (no code found): 4-stage LLM extraction from paper markdown
     (adapted from sandbox_manager.py TaskDescAgent)
  2. With code: Code-analysis approach
     (adapted from generate_task_descriptions.py)

Output: a structured task description markdown file saved to task_descriptions_path.
"""

import os
from pathlib import Path
from typing import Optional

from utils.llm_client import LLMClient
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class TaskDescriptionGenerator:
    """Generate task descriptions for the downstream agentic pipeline."""

    def __init__(self, llm_client: LLMClient, config: dict):
        self.llm = llm_client
        self.config = config
        self.output_dir = Path(
            config.get("baseline", {}).get(
                "task_descriptions_path", "/data/yjh/task_descriptions"
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def generate(
        self,
        task_name: str,
        paper_markdown: Optional[str] = None,
        code_content: Optional[str] = None,
        force: bool = False,
    ) -> Optional[str]:
        """
        Generate a task description and save it.

        Args:
            task_name: Identifier used as filename stem (e.g. "SwinIR")
            paper_markdown: Full paper text in markdown (from PDF OCR)
            code_content: Source code of the GT script (if available)
            force: Overwrite existing description

        Returns:
            Path to the saved description file, or None on failure.
        """
        output_path = self.output_dir / f"{task_name}_description.md"
        if output_path.exists() and not force:
            logger.info(f"Task description already exists: {output_path}")
            return str(output_path)

        if code_content:
            logger.info(f"[{task_name}] Generating task desc from CODE")
            desc = self._generate_from_code(code_content)
        elif paper_markdown:
            logger.info(f"[{task_name}] Generating task desc from PAPER (4-stage)")
            desc = self._generate_from_paper(paper_markdown)
        else:
            logger.error(f"[{task_name}] No paper markdown or code provided!")
            return None

        if not desc:
            logger.error(f"[{task_name}] Task description generation failed")
            return None

        output_path.write_text(desc, encoding="utf-8")
        logger.info(f"[{task_name}] Task description saved → {output_path}")
        return str(output_path)

    # ──────────────────────────────────────────────────────────
    # Mode 1: From Code (ref: generate_task_descriptions.py)
    # ──────────────────────────────────────────────────────────

    def _generate_from_code(self, code_content: str) -> Optional[str]:
        """Analyze code and produce a structured task description."""
        system_prompt = _CODE_ANALYSIS_SYSTEM_PROMPT
        user_prompt = f"Here is the Ground Truth Code:\n\n{code_content}"

        try:
            result = self.llm.completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=8192,
            )
            return result.strip() if result else None
        except Exception as e:
            logger.error(f"Code-based desc generation failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # Mode 2: From Paper (ref: TaskDescAgent 4-stage)
    # ──────────────────────────────────────────────────────────

    def _generate_from_paper(self, paper_content: str) -> Optional[str]:
        """4-stage extraction: Problem → Forward Model → Method → Experiment."""
        stages = {
            "problem_setting": (
                _STAGE1_SYSTEM, _stage1_user(paper_content)
            ),
            "forward_model": (
                _STAGE2_SYSTEM, _stage2_user(paper_content)
            ),
            "algorithm": (
                _STAGE3_SYSTEM, _stage3_user(paper_content)
            ),
            "experiment": (
                _STAGE4_SYSTEM, _stage4_user(paper_content)
            ),
        }

        results = {}
        for stage_name, (sys_prompt, usr_prompt) in stages.items():
            logger.info(f"  Stage: {stage_name}")
            try:
                resp = self.llm.completion(
                    prompt=usr_prompt,
                    system_prompt=sys_prompt,
                    temperature=0.2,
                    max_tokens=4096,
                )
                results[stage_name] = resp.strip() if resp else ""
            except Exception as e:
                logger.error(f"  Stage {stage_name} failed: {e}")
                results[stage_name] = ""

        return _compose_paper_description(results)


# ══════════════════════════════════════════════════════════════
# Prompt constants
# ══════════════════════════════════════════════════════════════

_CODE_ANALYSIS_SYSTEM_PROMPT = """You are a code analysis expert. Read the input code, extract its core logic, and generate a structured Task Description that will instruct another LLM to reproduce the algorithm.

**Purpose:** The description you generate serves as an executable specification—another system will use it to implement the algorithm and produce structured JSON output.

**Output Structure Requirements:**
- Use snake_case section headers (e.g., `## data_preprocessing`, `## objective_function`)
- Each section header maps to a JSON key; subsections become nested objects
- Use imperative language: specify what MUST be included, not just what exists
- Use consistent table formats with explicit column headers

**Sections** (include only applicable ones; starred sections are REQUIRED):

1. `overview`: Brief summary of the code's purpose and pipeline stages

2. `data_preprocessing`: One-time transformations before main computation (mark as initialization steps)

3. `variables`*: Categorize ALL variables in table format: | name | shape | dtype | role | description |
   - Create **distinct subsections** by role: `primal` (optimized), `dual` (multipliers), `constants`, `observations`

4. `objective_function`*: For optimization/inverse problems:
   - `full_expression`: Complete mathematical formulation
   - `data_fidelity_term`: Measurement consistency term with formula
   - `regularization_terms`: Each regularizer with name, formula, weight
   - `constraints`: Mathematical constraints (type: equality/inequality/bound + expression)

5. `initialization`: Table format: | variable | value | shape | source/justification |

6. `main_process`*:
   - `algorithm_framework`: Name the optimization method
   - `stopping_criterion`: Termination condition type and expression
   - `iteration_loop`: Numbered per-iteration update sequence
   - `validation_checks`: Post-hoc filters or sanity checks

7. `evaluation`*: Table of metrics: | metric | formula | reference_required | output_type |

8. `output_schema`*: Explicitly list all JSON keys that MUST appear in any implementation output, with expected types and structure. Map pipeline stages to their corresponding output sections.

**Format Guidelines:**
- Consolidate mathematical expressions before decomposing
- Clearly distinguish one-time setup from repeated iteration steps
- Cover all logical steps, edge cases, and constraints with precision
- Frame each section as requirements for reproduction, not just documentation"""


# ── Stage 1: Problem Setting ──

_STAGE1_SYSTEM = """You are a Computational Imaging Scientist specializing in inverse problems.
MISSION: Extract the precise PROBLEM SETTING from the ENTIRE paper.

## CRITICAL PRINCIPLES
1. **PRESERVE FULL COMPLEXITY**:
   - Extract multi-constraint recovery settings verbatim (e.g., "simultaneous denoising + super-resolution + motion correction").
   - Preserve hybrid modality specifications exactly.

2. **INPUT/OUTPUT SPECIFICATION**:
   - Extract EXACT shapes, dtypes, and channel counts.
   - **CRITICAL**: Identify the **DATA RANGE** (e.g., [0, 1] float32 vs [0, 255] uint8).
   - **CRITICAL**: Identify the **DOMAIN** (Pixel domain vs Frequency domain/k-space).

3. **PROBLEM CONTEXT**:
   - State application domain (e.g., "cardiac MRI", "natural image deblurring")."""


def _stage1_user(paper: str) -> str:
    return f"""# FULL PAPER CONTENT
{paper[:60000]}

# TASK
Extract the PROBLEM SETTING:
1. What is the goal? (e.g., MRI Reconstruction)
2. What are input/output shapes and types?
3. **CRITICAL**: What is the valid data range? [0,1] or [0,255]? Complex or Real?
4. Domain: Pixel or Frequency?

Output a structured description:"""


# ── Stage 2: Forward Model ──

_STAGE2_SYSTEM = """You are a Physics Simulation Engineer for computational imaging.
MISSION: Extract the EXACT MATHEMATICAL FORWARD MODEL (y = A(x) + n) used to generate synthetic data.

## CRITICAL PRINCIPLES
1. **DEGRADATION EQUATION**: Extract the exact formula.
2. **NOISE DETAILS**: Exact type (Gaussian, Poisson, Rician), exact parameters and units.
3. **SAMPLING/BLUR KERNELS**:
   - MRI: Mask type, Sampling Rate, Center Fraction.
   - Blur: Kernel size, Type, std dev.
   - Inpainting: Mask distribution.

## STRICT RULES
- IF PAPER USES MULTIPLE SETTINGS, EXTRACT THE **MAIN** SETTING used in the primary results table.
- Focus on NUMBERS and EQUATIONS."""


def _stage2_user(paper: str) -> str:
    return f"""# FULL PAPER CONTENT
{paper[:60000]}

# TASK
Extract the FORWARD MODEL (y = A(x) + n) specifics:
1. **Equation**: How is y generated from x?
2. **Noise**: Exact distribution and standard deviation (value and scale).
3. **Operator A**: Mask/kernel/sampling details.

Output a structured description:"""


# ── Stage 3: Algorithm ──

_STAGE3_SYSTEM = """You are an Algorithm Implementation Specialist for inverse problems.
MISSION: Extract the FULL algorithmic complexity from the ENTIRE paper.

## CRITICAL PRINCIPLES
1. **ARCHITECTURAL BLUEPRINT**: Extract backbone, modules, iteration counts.
2. **MATHEMATICAL OPERATIONS**: Loss functions, normalization layers.
3. **HYPERPARAMETERS**: LR, batch size, optimizer, iteration counts, loss weights."""


def _stage3_user(paper: str) -> str:
    return f"""# FULL PAPER CONTENT
{paper[:60000]}

# TASK
Extract the ALGORITHM details:
1. Network Architecture / Iterative Algorithm steps.
2. **Loss Function**: Exact formula and weights.
3. Optimizer, LR, Scheduler.

Output a structured description:"""


# ── Stage 4: Experiment ──

_STAGE4_SYSTEM = """You are an Experimental Reproducibility Engineer.
MISSION: Extract the COMPLETE experimental protocol.

## CRITICAL PRINCIPLES
1. **DATASET FIDELITY**: Exact Dataset Name, Train/Val/Test Splits.
2. **EVALUATION METRICS**: Primary Metric, evaluation context (Y-channel? range?).
3. **BASELINES**: List key comparison methods."""


def _stage4_user(paper: str) -> str:
    return f"""# FULL PAPER CONTENT
{paper[:60000]}

# TASK
Extract EXPERIMENTAL PROTOCOL:
1. Dataset used (Train/Test counts).
2. **Preprocessing**: Crops, Normalization range.
3. **Metrics**: PSNR/SSIM calculation details.

Output a structured description:"""


# ── Compose final ──

def _compose_paper_description(stages: dict) -> str:
    return f"""## PROBLEM SETTING
{stages.get('problem_setting', 'N/A')}

## FORWARD MODEL & DEGRADATION (CRITICAL)
{stages.get('forward_model', 'N/A')}

## ALGORITHMIC BLUEPRINT
{stages.get('algorithm', 'N/A')}

## EXPERIMENTAL PROTOCOL
{stages.get('experiment', 'N/A')}

## REPRODUCTION CRITERION
Implement the complete pipeline preserving all architectural nuances and experimental details exactly as described. Reproduction is successful when reported metrics are matched within ±0.3dB PSNR (or equivalent tolerance) under identical data splits, preprocessing, and evaluation protocol."""
