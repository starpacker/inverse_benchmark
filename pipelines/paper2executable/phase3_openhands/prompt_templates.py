"""
Phase 3 - OpenHands Prompt Templates

Core prompts that drive the agent behavior at each pipeline stage.
"""

# --- Step 1: Data Extraction ---
STEP1_DATA_EXTRACTION_PROMPT = """
# Task: Extract or Generate Test Data
You are an expert data engineer for computational imaging.
Your goal is to prepare the input/output data needed to run the paper's code.

> **ENVIRONMENT RULE**: The base pipeline environment (`ragas`) is readonly. You **may not** install or modify packages there.  
> If the code requires additional Python packages, you must note them for the sandbox step and a **temporary conda environment** will be created there. Do not attempt to `pip install` into ragas.

## Instructions
1. **Search for Data**: Scan the repository for `test_data/`, `demo/`, or `examples/`.
   - If a download script (sh/py) exists, run it.
   - If a URL is in the README, download it using `wget` or `curl`.

2. **Generate Mock Data (Fallback)**:
   - If no real data is available or download fails, you MUST generate mock data.
   - Analyze the code (especially `dataloader` or `forward_model`) to find input dimensions.
   - Create a Python script `generate_mock_data.py` that saves:
     - `input.npy`: The measurement/input (e.g., blurred image, sinogram).
     - `gt_output.npy`: The ground truth (e.g., clean image).
     - `aux_data.npy`: Any physics constants (mask, kernel, sensitivity map).
   - Use standard Gaussian noise or simple shapes (e.g., `skimage.data.shepp_logan_phantom`) if exact distribution is unknown.

3. **Standardize Output**:
   - Save all .npy files to the current directory.
   - Ensure data types are `float32`.

## Constraints
- Do NOT ask for user permission.
- If data is too large (>1GB), generate mock data instead.

## Completion
Once you have successfully saved `input.npy`, `gt_output.npy`, and `aux_data.npy` to the current directory, verify they exist with `ls -la *.npy` and then **stop** — your task is done.
"""

# --- Step 2: Code Consolidation ---
STEP2_CODE_CONSOLIDATION_PROMPT = """
# Task: Create Single-File Run Script (run_gt.py)
You are an expert research software engineer.
Your goal is to consolidate the paper's inference code into a single standalone script.

> **ENVIRONMENT RULE**: Remember that you are executing inside the `ragas` base environment which cannot be modified. If the original code requires additional Python packages, **do not install them here**; instead, list them clearly in comments at the top of `run_gt.py` so the sandbox phase can create a temporary conda env with those dependencies.

## Instructions
1. **Identify Inference Entry Point**: Find the main script (e.g., `test.py`, `eval.py`, `demo.ipynb`).
2. **Flatten Logic**:
   - Copy relevant model definitions (Classes), forward operators, and utility functions into `run_gt.py`.
   - Remove dependencies on complex local file imports (copy code in if needed).
   - Remove argument parsers (`argparse`); hardcode necessary parameters for the `input.npy` you created.
3. **Weight Loading**:
   - If the model needs weights (`.pth`, `.ckpt`), download them if a link exists.
   - **CRITICAL**: If no weights are available, initialize the model with RANDOM weights (`model.apply(weights_init)`). The code must run, even if output is garbage.
4. **I/O Contract**:
   - Input: Read `input.npy` (and `aux_data.npy` if needed).
   - Processing: Run the reconstruction/inference.
   - Output: Save the result as `baseline.npy` (float32).

## Constraints
- Do NOT use `argparse`.
- Do NOT rely on external config files unless you create them.
- Handle `FileNotFoundError` gracefully.
- Write code using `cat << 'EOF' > run_gt.py` or Python file I/O to avoid context limits.

## Completion
Once `run_gt.py` is written and you have verified it runs to completion (producing `baseline.npy`), **stop** — your task is done.
"""

# --- Step 3: Evaluation Script ---
STEP3_EVALUATION_SCRIPT_PROMPT = """
# Task: Create Evaluation Script (evaluate.py)
You are a metrics specialist.
Your goal is to create a standard script to measure reconstruction quality.

> **ENVIRONMENT NOTE**: This script will run later in a temporary conda environment created by the sandbox step. Any package imports should be common ones already available or noted above.

## Instructions
1. Create `evaluate.py`.
2. **Load Data**:
   - `baseline.npy` (Model output)
   - `gt_output.npy` (Ground truth)
3. **Normalize**:
   - Ensure both arrays are comparable (e.g., range [0, 1] or [0, 255]).
   - Handle shape mismatches (crop center or resize if differ by <5 pixels).
4. **Compute Metrics**:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - RMSE (Root Mean Square Error)
5. **Output**:
   - Print metrics to stdout.
   - Save detailed JSON to `result.json`: `{"psnr": 25.4, "ssim": 0.88, ...}`.

## Constraints
- Use `skimage.metrics` or `torchmetrics`.
- Handle `NaN` or `Inf` values (replace with 0.0 and log warning).

## Completion
Once `evaluate.py` is written and you have verified it runs to completion (producing `result.json`), **stop** — your task is done.
"""

# --- Step 4: Auto-Debug ---
STEP4_DEBUG_PROMPT = """
# Task: Fix Execution Failure
The previous attempt to run `{failed_script}` failed.

## Error Log
```
{error_log}
```

## Instructions
1. Analyze the traceback above.
2. Fix the code in `{failed_script}`.
   - If `ModuleNotFoundError`: Install via pip or copy missing code.
   - If `RuntimeError` (Shapes): Fix tensor dimensions (check `input.npy` shape).
   - If `OutOfMemory`: Reduce batch size or simplify model.
3. Verify the fix by running the script again.

## Constraints
- Do not change the I/O contract (input.npy -> baseline.npy).
- You have 5 iterations to fix this.
"""
