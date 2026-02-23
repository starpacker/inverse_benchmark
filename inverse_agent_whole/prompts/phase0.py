PROMPT_GEN_TASK = """ 
You are a deterministic algorithm extraction engine. Convert code into a structured **Algorithmic Specification Document** that will be parsed by downstream systems for JSON extraction.

**CRITICAL BEHAVIORAL CONSTRAINTS**:
- Your response MUST begin with "## 0. ALGORITHM IDENTIFICATION"—any preamble is invalid
- Produce ONLY the specification document—never ask questions, suggest improvements, diagnose issues, or add conversational elements
- Use IMPERATIVE language: "The output MUST include X" not "The algorithm uses X"
- Assume the code is correct; document what it DOES, not what it SHOULD do
- If ambiguous, state your interpretation and proceed
- NEVER write "Refer to [function/code]"—expand all formulas inline

**OUTPUT STRUCTURE** (all sections mandatory; each maps to a JSON field):

## 0. ALGORITHM IDENTIFICATION [→ algorithm_identification]
- algorithm_name, algorithm_family, one-sentence summary

## 1. OBJECTIVE FUNCTION [→ objective_function]
- **full_expression**: Complete optimization problem in standard form
- **data_fidelity_term**: Formula with explicit definition (verify L1/L2/other from code operations)
- **regularization_term**: Formula for EACH component separately (or "None")
- **constraints**: Include box constraints from clamp/projection operations; use indicator notation

## 2. VARIABLES [→ variables]
Present in GROUPED SUBSECTIONS by category, each with table (Symbol | Shape | Dtype | Initialization | Description):
### 2.1 Primal Variables
### 2.2 Dual Variables  
### 2.3 Auxiliary Variables (include ALL tracking/temporary variables like `x_old`)
### 2.4 Constants (list EVERY hardcoded value with exact values)
### 2.5 Observations
For nested procedures, create sub-tables marked [INNER].

## 3. PREPROCESSING PIPELINE [→ data_preprocessing]
For each step: step_order, step_name, formula (explicit LaTeX), inputs (with shapes), outputs (with shapes)
Include parameter derivation as Step 0 if physical constants are computed.

## 4. ALGORITHM STEPS [→ initialization + iteration_loop + finalization]
### 4.1 INITIALIZATION [→ initialization]
### 4.2 ITERATION LOOP [→ iteration_loop]
Use BEGIN_LOOP/END_LOOP markers. Each step performs ONE atomic operation:
(1) forward model, (2) residual/loss, (3) gradient, (4) descent update, (5) proximal operator, (6) momentum update
For each: step_name, formula, inputs, outputs, computational_note
For nested loops, create sub-blocks with own BEGIN/END markers.
### 4.3 FINALIZATION [→ finalization]
Document output formats, post-processing, and metric computation.

## 5. STOPPING CRITERION [→ stopping_criterion]
- type: (iteration_limit | convergence_threshold | combined)
- expression: Mathematical condition with exact threshold values
- parameters: All thresholds and limits from code
- check_frequency: When evaluated

## 6. EVALUATION METRICS [→ evaluation]
For each metric: name, formula (complete definition), description

## 7. REQUIRED OUTPUT FIELDS
List all JSON fields that MUST appear in downstream output derived from this specification.

**REQUIREMENTS**:
- "Nothing implicit": document every value, default, and function definition
- For proximal operators: provide BOTH definition AND closed-form solution
- Mark operators as [FORWARD] or [ADJOINT] where applicable
- Use consistent notation (define symbols before use)
- Every operation MUST have its explicit formula
Here is the Ground Truth Code:
{code}
"""

PROMPT_GEN_GOLD = """
You are an Expert Algorithm Architect.
Your task is to reverse-engineer a "Golden Plan" (Algorithmic Logic) from the provided code.
The output must be a strict JSON object describing the algorithm with a complete schema.

Output JSON Schema (strict JSON, no markdown):
{{
  "algorithm_name": "string",

  "data_preprocessing": {{
    "description": "string (high-level purpose of preprocessing)",
    "steps": [
      {{
        "step_order": 1,
        "step_name": "string",
        "operation": "string (mathematical or algorithmic description)",
        "mathematical_formula": "string (LaTeX compatible, optional)",
        "input_data": ["list of inputs"],
        "output_data": ["list of outputs"],
        "assumptions": ["list of assumptions or conditions"]
      }}
    ]
  }},

  "objective_function": {{
    "full_expression": "string (complete mathematical objective)",
    "data_fidelity_term": "string (mathematical formula)",
    "regularization_term": "string (mathematical formula)",
    "constraints": ["list of explicit constraints, if any"]
  }},

  "variables": {{
    "primal": ["list of primal variables"],
    "dual": ["list of dual / auxiliary variables"],
    "constants": ["list of constants / hyperparameters"],
    "observations": ["list of observed data variables"]
  }},

  "initialization": [
    {{
      "variable": "string",
      "value": "string description",
      "shape": "string description",
      "source": "string (e.g., zeros, random, from data, analytical)"
    }}
  ],

  "main_inverse_process": {{
    "algorithm_framework": "string (e.g., GD, ADMM, PDHG, ISTA)",
    "iteration_loop": [
      {{
        "step_order": 1,
        "step_name": "string",
        "step_type": "string (e.g., primal update / dual update / proximal step)",
        "mathematical_formula": "string (LaTeX compatible)",
        "operator_requirements": ["list of required operators"],
        "input_variables": ["list of variables"],
        "output_variables": ["list of variables"],
        "computational_notes": "string (e.g., FFT-based, linear solve, proximal closed-form)"
      }}
    ],
    "stopping_criterion": {{
      "type": "string (e.g., max_iter / tolerance / relative_change)",
      "expression": "string (mathematical or logical condition)"
    }}
  }},

  "evaluation": {{
    "description": "string (purpose of evaluation)",
    "metrics": [
      {{
        "metric_name": "string (e.g., MSE, PSNR, SSIM, residual norm)",
        "definition": "string (mathematical definition)",
        "reference_data": "string (e.g., ground truth, observations)",
        "output": "string (scalar / vector / curve)"
      }}
    ],
    "post_processing": [
      {{
        "operation": "string (e.g., normalization, thresholding, visualization)",
        "input_variables": ["list of variables"],
        "output_variables": ["list of variables"]
      }}
    ]
  }}
}}

Here is the Ground Truth Code:
{code}
"""

PROMPT_ORACLE_ADAPTER = """
You are a Python Data Expert.
Analyze the provided Ground Truth Code to determine the file format of the "Expected Output".

Standard Formats (Supported natively): [.npy, .npz, .json, .txt, .csv, .log, .md, .png, .jpg, .jpeg, .tif, .tiff, .bmp]

Your Goal:
1. Identify the output file extension.
2. If it is NOT a Standard Format, provide a Python function `custom_load(path)` that:
   - Can return ANY Python object (dict, list, numpy array, etc.)
   - Must preserve the FULL STRUCTURE of the data
   - Include ALL necessary imports inside the function
   - Handle file closing properly

Output JSON Schema:
{{
  "is_standard": boolean,
  "file_extension": "string",
  "custom_loader_code": "string (code for custom_load, can return any type)",
  "reasoning": "string"
}}

Ground Truth Code:
{code}
"""

PROMPT_INSTRUMENT_GT = """
You are a Python Expert. Your task is to analyze and if necessary, instrument the provided Ground Truth (GT) code to ensure it saves the final reconstruction result to a specific file.

**Goal**:
We need to run this code to get the "Ground Truth Reconstruction" (the result of the inverse problem, e.g., the recovered image or volume). We also need to know what input file it used.

**Instructions**:
1. Analyze the code to find:
   - The main entry point.
   - The variable holding the final reconstruction result (NOT the evaluation metrics, but the actual image/data).
   - The input file path being used.

2. Check if the code ALREADY saves this final result to a file.
   - If YES, identify the filename.
   - If NO, or if it saves to a complex path, you must REWRITE the code to save the final result to a file named `gt_output_ref.npy` (or .tif/.png if appropriate) in the CURRENT working directory.

3. Check if the input file path is hardcoded or relative.
   - If it's relative, ensure it works when run from the script's directory.

4. **Output**:
   - Return a JSON object containing the modified (instrumented) code and metadata.

**Output JSON Schema**:
{{
  "instrumented_code": "string (The full python code, possibly modified to save output)",
  "detected_input_path": "string (The input file path found in the code, or null if not found)",
  "output_filename": "string (The name of the file that the code WILL save, e.g. 'gt_output_ref.npy')",
  "modifications_made": "string (Brief description of changes made)"
}}

**Ground Truth Code**:
{code}
"""

PROMPT_ANALYZE_FORWARD_MODEL = """
You are a Scientific Code Expert.
Your task is to analyze the provided Ground Truth Code to identify the "Forward Model" (the mathematical operator that generates measurements from the ground truth object).

**Goal**:
We want to create a synthetic dataset. We need to know:
1. Which function/operator implements `y = A(x)`?
2. What are the expected shapes/types of `x` (ground truth) and `y` (measurements)?
3. Are there any specific constraints (e.g., x must be positive)?

**Output JSON Schema**:
{{
  "forward_model_function": "string (Name of the function/class method)",
  "forward_model_call_signature": "string (Example code snippet of how to call it)",
  "x_true_shape_desc": "string (Description of the shape of the ground truth object, e.g., '256x256 image' or '1D signal of length 1024')",
  "y_meas_shape_desc": "string (Description of the shape of the measurements)",
  "constraints": "string (Any constraints on x or y)"
}}

**Ground Truth Code**:
{code}
"""

PROMPT_WRITE_DATA_GEN = """
You are a Python Data Scientist.
Your task is to write a standalone Python script `data_gen.py` that generates a synthetic dataset for the Inverse Problem defined in the provided Ground Truth Code.

**Context**:
- We have the Ground Truth Code which solves `y = A(x)`.
- We analyzed it and found the Forward Model info: {analysis}

**Requirements for `data_gen.py`**:
1. **Import/Reuse**: It must import or copy the necessary "Forward Model" logic from the provided GT code. (Assume the GT code file is in the same directory, named `{gt_filename}`).
2. **Generate x_true**: Create a realistic "Ground Truth Object" `x_true` (e.g., a Shepp-Logan phantom, a natural image, or a smooth signal). It MUST match the shape/type expected by the GT code.
3. **Generate y_meas**: Apply the forward model: `y_meas = ForwardModel(x_true)`.
4. **Add Noise**: Add appropriate Gaussian noise to `y_meas` (e.g., SNR=30dB).
5. **Save Files**:
   - Save `x_true` to `x_true.npy` (or .png/.tif if image).
   - Save `y_meas` to `input_measurement.npy` (or whatever format/name the GT code expects as input!). *Crucial: Ensure the filename matches what the GT code reads, or use a standard name `input.npy` if the GT code is flexible.*
6. **Robustness**: Ensure data types (float32/64) are correct.

**Output JSON Schema**:
{{
  "data_gen_code": "string (The full python script)",
  "x_true_filename": "string",
  "y_meas_filename": "string"
}}

**Ground Truth Code**:
{code}
"""

PROMPT_EXTRACT_FROM_PAPER = """
You are a Scientific Researcher.
Your task is to extract structured information from the provided Research Paper (Markdown) to define an Inverse Problem Task.

**Goal**: Convert unstructured text into a standard Task Description.

**Output JSON Schema**:
{{
  "problem_definition": "string (e.g., 'Sparse-view CT Reconstruction')",
  "input_spec": {{
    "physical_meaning": "string",
    "modality": "string (image/time-series/etc)",
    "dimensions": "string"
  }},
  "output_spec": {{
    "physical_meaning": "string",
    "properties": "string (e.g., 'High resolution, deniosed')"
  }},
  "constraints": ["list of strings"],
  "evaluation_metrics": ["list of strings (e.g., PSNR, SSIM)"]
}}

**Paper Content**:
{paper_content}
"""

PROMPT_DATA_GEN_FEEDBACK = """
You are a Python Data Scientist.
The previous `data_gen.py` failed to generate valid data for the GT Code.

**Feedback/Error**:
{feedback}

**Task**:
Rewrite `data_gen.py` to fix the issue.
- If "Shape Mismatch", check dimensions.
- If "Metric Failure", maybe noise is too high? Or x_true is too simple/complex?
- If "Runtime Error", fix the code.

**Previous Data Gen Code**:
{previous_code}

**Output JSON Schema**:
{{
  "data_gen_code": "string (The fixed python script)",
  "reasoning": "string"
}}
"""
