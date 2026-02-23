## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "AMICO-NODDI",
  "algorithm_family": "Dictionary-based sparse reconstruction with LASSO",
  "summary": "Accelerated Microstructure Imaging via Convex Optimization for NODDI parameter estimation using a precomputed biophysical dictionary and L1-regularized least squares."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{x_{\\text{iso}} \\geq 0} \\|\\mathbf{y} - x_{\\text{iso}} \\mathbf{a}_{\\text{iso}}\\|_2^2 \\quad \\text{then} \\quad \\min_{\\mathbf{x}_{\\text{wm}} \\geq 0} \\frac{1}{2n}\\|\\mathbf{y}_{\\text{wm}} - \\mathbf{A}_{\\text{wm}} \\mathbf{x}_{\\text{wm}}\\|_2^2 + \\lambda \\|\\mathbf{x}_{\\text{wm}}\\|_1",
  "data_fidelity_term": "\\frac{1}{2n}\\|\\mathbf{y}_{\\text{wm}} - \\mathbf{A}_{\\text{wm}} \\mathbf{x}_{\\text{wm}}\\|_2^2 where n is the number of measurements",
  "regularization_term": "\\lambda \\|\\mathbf{x}_{\\text{wm}}\\|_1 where \\lambda = \\frac{\\lambda_1}{n} = \\frac{0.5}{n}",
  "constraints": "x_{\\text{iso}} \\geq 0, \\quad \\mathbf{x}_{\\text{wm}} \\geq 0 \\quad (\\text{non-negativity via NNLS and positive=True in LASSO})"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x_{\text{iso}}$ | scalar | float64 | NNLS solution | Isotropic (CSF) compartment coefficient |
| $\mathbf{x}_{\text{wm}}$ | $(N_{\text{atoms}},)$ where $N_{\text{atoms}}=144$ | float64 | LASSO solution | White matter dictionary coefficients |
| $\mathbf{x}_{\text{wm,scaled}}$ | $(144,)$ | float64 | LASSO output | Scaled WM coefficients before norm correction |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| N/A | — | — | — | LASSO uses coordinate descent internally (no explicit dual) |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}_{\text{wm}}$ | $(n_S,)$ | float64 | $\mathbf{y} - x_{\text{iso}} \mathbf{a}_{\text{iso}}$ | Residual signal after isotropic subtraction |
| $\text{idx}_{\text{dir}}$ | scalar | int | KDTree query | Index of closest direction in 500-point sphere |
| $\text{sum}_{x_{\text{wm}}}$ | scalar | float64 | $\sum_j x_{\text{wm},j}$ | Sum of WM coefficients for normalization |
| $\text{NDI}$ | scalar | float64 | weighted average | Neurite Density Index for current voxel |
| $\text{ODI}$ | scalar | float64 | computed from $\kappa$ | Orientation Dispersion Index for current voxel |
| $\text{FWF}$ | scalar | float64 | $x_{\text{iso}}/(\text{sum}_{x_{\text{wm}}} + x_{\text{iso}})$ | Free Water Fraction |

### 2.4 Constants
| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $\lambda_1$ | scalar | float64 | $5 \times 10^{-1}$ | Base regularization parameter |
| $d_{\text{par}}$ | scalar | float64 | $1.7 \times 10^{-3}$ mm²/s | Parallel diffusivity |
| $d_{\text{iso}}$ | scalar | float64 | $3.0 \times 10^{-3}$ mm²/s | Isotropic diffusivity |
| $\gamma$ | scalar | float64 | $2.675987 \times 10^{8}$ rad/(s·T) | Gyromagnetic ratio |
| $l_{\max}$ | scalar | int | 12 | Maximum spherical harmonic order |
| $n_{\text{dirs}}$ | scalar | int | 500 | Number of directions on high-resolution sphere |
| $N_{\text{OD}}$ | scalar | int | 12 | Number of orientation dispersion samples |
| $N_{\text{VF}}$ | scalar | int | 12 | Number of volume fraction samples |
| IC_VFs | $(12,)$ | float64 | $\text{linspace}(0.1, 0.99, 12)$ | Intracellular volume fraction grid |
| IC_ODs | $(12,)$ | float64 | $[0.03, 0.06] \cup \text{linspace}(0.09, 0.99, 10)$ | Orientation dispersion grid |
| GRAD_500 | $(500, 3)$ | float64 | Predefined | High-resolution gradient directions |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(n_S,)$ | float64 | Measured diffusion signal for single voxel |
| $\mathbf{Y}$ | $(n_x, n_y, n_z, n_S)$ | float64 | Full 4D diffusion-weighted image |
| $\mathbf{M}$ | $(n_x, n_y, n_z)$ | bool | Binary brain mask |
| $\mathbf{D}$ | $(n_x, n_y, n_z, 3)$ | float64 | Principal diffusion directions from DTI |
| scheme | object | — | Acquisition scheme with b-values and gradients |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": 1,
      "step_name": "Load DWI Data",
      "formula": "\\mathbf{Y} = \\text{nib.load}(\\text{dwi\\_file}).\\text{get\\_fdata}()",
      "inputs": {"dwi_file": "string path"},
      "outputs": {"Y": "(n_x, n_y, n_z, n_S)", "affine": "(4, 4)"},
      "assumptions": "NIfTI format, 4D array"
    },
    {
      "step_order": 2,
      "step_name": "Load Acquisition Scheme",
      "formula": "\\text{scheme} = \\text{Scheme}(\\text{scheme\\_file}, b_0^{\\text{thr}}=10)",
      "inputs": {"scheme_file": "string path"},
      "outputs": {"scheme.b": "(n_S,)", "scheme.raw": "(n_S, 4 \\text{ or } 7)", "scheme.shells": "list of shell dicts"},
      "assumptions": "Columns: [gx, gy, gz, b] or STEJSKALTANNER format"
    },
    {
      "step_order": 3,
      "step_name": "Normalize Gradient Directions",
      "formula": "\\mathbf{g}_i \\leftarrow \\frac{\\mathbf{g}_i}{\\|\\mathbf{g}_i\\|_2 + 10^{-16}} \\quad \\forall i",
      "inputs": {"scheme.raw[:,:3]": "(n_S, 3)"},
      "outputs": {"scheme.raw[:,:3]": "(n_S, 3) unit vectors"},
      "assumptions": "Non-zero gradient vectors"
    },
    {
      "step_order": 4,
      "step_name": "Identify b=0 and DWI Indices",
      "formula": "\\text{b0\\_idx} = \\{i : b_i \\leq 10\\}, \\quad \\text{dwi\\_idx} = \\{i : b_i > 10\\}",
      "inputs": {"scheme.b": "(n_S,)"},
      "outputs": {"scheme.b0_idx": "(n_{b0},)", "scheme.dwi_idx": "(n_{dwi},)"},
      "assumptions": "At least one b=0 volume"
    },
    {
      "step_order": 5,
      "step_name": "Group into Shells",
      "formula": "\\text{shells}[s] = \\{i : \\lfloor b_i / 100 \\rfloor \\cdot 100 = b_s\\} \\text{ for unique } b_s",
      "inputs": {"scheme.b": "(n_S,)", "scheme.dwi_idx": "(n_{dwi},)"},
      "outputs": {"scheme.shells": "list of {b, idx, grad, ...}"},
      "assumptions": "Shell separation >= 100 s/mm²"
    },
    {
      "step_order": 6,
      "step_name": "DTI Fit for Principal Directions",
      "formula": "\\mathbf{D}_{xyz} = \\text{evecs}(\\text{DTI}(\\mathbf{Y}_{xyz}))[..., 0]",
      "inputs": {"Y": "(n_x, n_y, n_z, n_S)", "scheme": "object", "M": "(n_x, n_y, n_z)"},
      "outputs": {"D": "(n_x, n_y, n_z, 3)"},
      "assumptions": "Single fiber population approximation"
    },
    {
      "step_order": 7,
      "step_name": "Create High-Resolution Scheme",
      "formula": "\\text{scheme\\_hr} = \\text{Scheme with } 500 \\times n_{\\text{shells}} \\text{ measurements using GRAD\\_500}",
      "inputs": {"scheme.shells": "list", "GRAD_500": "(500, 3)"},
      "outputs": {"scheme_hr": "Scheme object with (500·n_shells, 7)"},
      "assumptions": "Each shell replicated on 500-point sphere"
    },
    {
      "step_order": 8,
      "step_name": "Precompute Rotation Matrices",
      "formula": "\\mathbf{Y}_{\\text{inv}} = (\\mathbf{Y}_{\\text{SH}}^T \\mathbf{Y}_{\\text{SH}})^{-1} \\mathbf{Y}_{\\text{SH}}^T, \\quad \\mathbf{Y}_{\\text{SH}} = \\text{real\\_sh\\_descoteaux}(l_{\\max}, \\theta, \\phi)",
      "inputs": {"GRAD_500": "(500, 3)", "l_max": "12"},
      "outputs": {"AUX['fit']": "(n_{SH}, 500)", "AUX['Ylm_rot']": "(500,) of (n_{SH},)", "AUX['const']": "(n_{SH},)", "AUX['idx_m0']": "(n_{SH},)"},
      "assumptions": "n_SH = (l_max + 1)(l_max + 2)/2 = 91"
    },
    {
      "step_order": 9,
      "step_name": "Generate Resampling Structures",
      "formula": "\\mathbf{Y}_{\\text{out}}[i, :] = \\text{real\\_sh\\_descoteaux}(l_{\\max}, \\theta_i, \\phi_i) \\text{ for each DWI direction}",
      "inputs": {"scheme.shells": "list", "l_max": "12"},
      "outputs": {"idx_out_res": "(n_{dwi},)", "Ylm_out": "(n_{dwi}, n_{SH} \\cdot n_{shells})"},
      "assumptions": "Maps SH coefficients to actual acquisition directions"
    }
  ]
}
```

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION
```json
{
  "steps": [
    {
      "step_order": 1,
      "variable_name": "IC_KAPPAs",
      "initial_value": "\\kappa_i = \\frac{1}{\\tan(\\text{IC\\_ODs}_i \\cdot \\pi / 2)} \\quad \\text{for } i = 0, \\ldots, 11",
      "shape": "(12,)",
      "dtype": "float64",
      "source": "Derived from IC_ODs"
    },
    {
      "step_order": 2,
      "variable_name": "kernels_lut",
      "initial_value": "\\text{empty list}",
      "shape": "list of (500, n_{SH} \\cdot n_{shells})",
      "dtype": "float32",
      "source": "To be populated"
    },
    {
      "step_order": 3,
      "variable_name": "KERNELS['wm']",
      "initial_value": "\\mathbf{0}",
      "shape": "(144, 500, n_S)",
      "dtype": "float32",
      "source": "np.zeros"
    },
    {
      "step_order": 4,
      "variable_name": "KERNELS['iso']",
      "initial_value": "\\mathbf{0}",
      "shape": "(n_S,)",
      "dtype": "float32",
      "source": "np.zeros"
    },
    {
      "step_order": 5,
      "variable_name": "KERNELS['icvf']",
      "initial_value": "\\mathbf{0}",
      "shape": "(144,)",
      "dtype": "float64",
      "source": "np.zeros"
    },
    {
      "step_order": 6,
      "variable_name": "KERNELS['kappa']",
      "initial_value": "\\mathbf{0}",
      "shape": "(144,)",
      "dtype": "float64",
      "source": "np.zeros"
    },
    {
      "step_order": 7,
      "variable_name": "KERNELS['norms']",
      "initial_value": "\\mathbf{0}",
      "shape": "(n_{dwi}, 144)",
      "dtype": "float64",
      "source": "np.zeros"
    },
    {
      "step_order": 8,
      "variable_name": "results",
      "initial_value": "\\mathbf{0}",
      "shape": "(n_{voxels}, 3)",
      "dtype": "float64",
      "source": "np.zeros"
    },
    {
      "step_order": 9,
      "variable_name": "KDTree",
      "initial_value": "\\text{KDTree}(\\text{GRAD\\_500})",
      "shape": "tree structure",
      "dtype": "object",
      "source": "scipy.spatial.KDTree"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "steps": [
    {
      "step_order": 10,
      "step_name": "BEGIN_LOOP: Dictionary Generation (kappa)",
      "formula": "\\text{for } k = 0 \\text{ to } N_{OD} - 1:",
      "inputs": {"IC_KAPPAs": "(12,)"},
      "outputs": {"k": "loop index", "kappa": "scalar"},
      "computational_note": "Outer loop over orientation dispersion values"
    },
    {
      "step_order": 11,
      "step_name": "[FORWARD] Compute Intracellular Signal",
      "formula": "\\mathbf{S}_{IC}^{(k)} = E_{IC}(d_{par}, \\kappa_k) \\text{ where } E_{IC} = \\frac{1}{2} \\sum_{l=0}^{6} \\text{lgi}_l \\cdot c_l(\\kappa) \\cdot \\sqrt{\\frac{l + 0.25}{\\pi}} P_l^0(\\cos\\theta) \\cdot e^{-q^2 \\tau \\cdot 0}",
      "inputs": {"d_par": "1.7e-3 mm²/s", "kappa_k": "scalar", "scheme_hr": "object"},
      "outputs": {"S_IC": "(500 · n_{shells},)"},
      "computational_note": "Watson-distributed cylinder model with Neuman perpendicular = 0"
    },
    {
      "step_order": 12,
      "step_name": "BEGIN_LOOP: Dictionary Generation (volume fraction)",
      "formula": "\\text{for } j = 0 \\text{ to } N_{VF} - 1:",
      "inputs": {"IC_VFs": "(12,)"},
      "outputs": {"j": "loop index", "v_{ic}": "scalar"},
      "computational_note": "Inner loop over intracellular volume fractions"
    },
    {
      "step_order": 13,
      "step_name": "[FORWARD] Compute Extracellular Signal",
      "formula": "\\mathbf{S}_{EC}^{(k,j)} = \\exp\\left(-b \\left[(d_\\parallel - d_\\perp) \\cos^2\\theta + d_\\perp\\right]\\right)",
      "inputs": {"d_par": "1.7e-3", "kappa_k": "scalar", "v_ic_j": "scalar", "scheme_hr": "object"},
      "outputs": {"S_EC": "(500 · n_{shells},)"},
      "computational_note": "d_perp = d_par(1 - v_ic), tortuosity-based anisotropic Gaussian"
    },
    {
      "step_order": 14,
      "step_name": "Combine IC and EC Signals",
      "formula": "\\mathbf{S}^{(k,j)} = v_{ic}^{(j)} \\cdot \\mathbf{S}_{IC}^{(k)} + (1 - v_{ic}^{(j)}) \\cdot \\mathbf{S}_{EC}^{(k,j)}",
      "inputs": {"S_IC": "(500 · n_{shells},)", "S_EC": "(500 · n_{shells},)", "v_ic_j": "scalar"},
      "outputs": {"S_combined": "(500 · n_{shells},)"},
      "computational_note": "Linear mixture of compartments"
    },
    {
      "step_order": 15,
      "step_name": "Rotate Kernel to SH Basis",
      "formula": "\\mathbf{K}_{lm}^{(s)} = \\mathbf{Y}_{inv} \\mathbf{S}^{(k,j)}[\\text{idx\\_in}[s]] \\text{ for each shell } s, \\quad \\mathbf{KR}_{lm}[i, :] = c_l \\cdot K_{l,0}^{(s)} \\cdot Y_{lm}(\\theta_i, \\phi_i)",
      "inputs": {"S_combined": "(500 · n_{shells},)", "AUX": "dict"},
      "outputs": {"KRlm": "(500, n_{SH} · n_{shells})"},
      "computational_note": "Spherical convolution theorem: rotation by zonal SH multiplication"
    },
    {
      "step_order": 16,
      "step_name": "Append to Kernel LUT",
      "formula": "\\text{kernels\\_lut.append}(\\mathbf{KR}_{lm})",
      "inputs": {"KRlm": "(500, n_{SH} · n_{shells})"},
      "outputs": {"kernels_lut": "list extended"},
      "computational_note": "Index = k · N_VF + j"
    },
    {
      "step_order": 17,
      "step_name": "END_LOOP: volume fraction",
      "formula": "\\text{end for } j",
      "inputs": {},
      "outputs": {},
      "computational_note": "Completes inner loop"
    },
    {
      "step_order": 18,
      "step_name": "END_LOOP: kappa",
      "formula": "\\text{end for } k",
      "inputs": {},
      "outputs": {},
      "computational_note": "Completes outer loop, kernels_lut has 144 entries"
    },
    {
      "step_order": 19,
      "step_name": "[FORWARD] Compute Isotropic Signal",
      "formula": "\\mathbf{S}_{iso} = \\exp(-b \\cdot d_{iso}) \\text{ where } d_{iso} = 3.0 \\times 10^{-3} \\text{ mm}^2/\\text{s}",
      "inputs": {"d_iso": "3.0e-3", "scheme_hr": "object"},
      "outputs": {"S_iso": "(500 · n_{shells},)"},
      "computational_note": "Isotropic diffusion (CSF)"
    },
    {
      "step_order": 20,
      "step_name": "Rotate Isotropic Kernel",
      "formula": "\\mathbf{KR}_{lm}^{iso}[\\text{idx\\_out}[s]] = \\mathbf{Y}_{inv} \\mathbf{S}_{iso}[\\text{idx\\_in}[s]]",
      "inputs": {"S_iso": "(500 · n_{shells},)", "AUX": "dict"},
      "outputs": {"KRlm_iso": "(n_{SH} · n_{shells},)"},
      "computational_note": "No directional dependence for isotropic"
    },
    {
      "step_order": 21,
      "step_name": "Append Isotropic to LUT",
      "formula": "\\text{kernels\\_lut.append}(\\mathbf{KR}_{lm}^{iso})",
      "inputs": {"KRlm_iso": "(n_{SH} · n_{shells},)"},
      "outputs": {"kernels_lut": "list with 145 entries"},
      "computational_note": "Last entry is isotropic"
    },
    {
      "step_order": 22,
      "step_name": "BEGIN_LOOP: Resample WM Kernels",
      "formula": "\\text{for } \\text{idx} = 0 \\text{ to } 143:",
      "inputs": {"kernels_lut": "list"},
      "outputs": {"idx": "loop index"},
      "computational_note": "Resample each atom to acquisition scheme"
    },
    {
      "step_order": 23,
      "step_name": "Resample Kernel to Acquisition Directions",
      "formula": "\\mathbf{K}_{\\text{wm}}[\\text{idx}, i, \\text{idx\\_out\\_res}] = \\mathbf{Y}_{out} \\cdot \\mathbf{KR}_{lm}[i, :] \\quad \\forall i \\in [0, 499]",
      "inputs": {"KRlm": "(500, n_{SH} · n_{shells})", "Ylm_out": "(n_{dwi}, n_{SH} · n_{shells})", "idx_out_res": "(n_{dwi},)"},
      "outputs": {"KERNELS['wm'][idx]": "(500, n_S)"},
      "computational_note": "K[idx, i, j] = 1 for b0 indices"
    },
    {
      "step_order": 24,
      "step_name": "Store Atom Parameters",
      "formula": "\\text{KERNELS}['\\kappa'][\\text{idx}] = \\kappa_k, \\quad \\text{KERNELS}['\\text{icvf}'][\\text{idx}] = v_{ic}^{(j)}",
      "inputs": {"IC_KAPPAs": "(12,)", "IC_VFs": "(12,)", "idx": "scalar"},
      "outputs": {"KERNELS['kappa'][idx]": "scalar", "KERNELS['icvf'][idx]": "scalar"},
      "computational_note": "k = idx // 12, j = idx % 12"
    },
    {
      "step_order": 25,
      "step_name": "Compute Atom Norm",
      "formula": "n_{\\text{idx}} = \\|\\mathbf{K}_{\\text{wm}}[\\text{idx}, 0, \\text{dwi\\_idx}]\\|_2, \\quad \\text{KERNELS}['\\text{norms}'][:, \\text{idx}] = 1/n_{\\text{idx}}",
      "inputs": {"KERNELS['wm'][idx, 0, :]": "(n_S,)", "scheme.dwi_idx": "(n_{dwi},)"},
      "outputs": {"KERNELS['norms'][:, idx]": "(n_{dwi},)"},
      "computational_note": "For dictionary normalization in LASSO"
    },
    {
      "step_order": 26,
      "step_name": "END_LOOP: Resample WM Kernels",
      "formula": "\\text{end for idx}",
      "inputs": {},
      "outputs": {},
      "computational_note": "KERNELS['wm'] fully populated"
    },
    {
      "step_order": 27,
      "step_name": "Resample Isotropic Kernel",
      "formula": "\\text{KERNELS}['\\text{iso}'][\\text{idx\\_out\\_res}] = \\mathbf{Y}_{out} \\cdot \\mathbf{KR}_{lm}^{iso}",
      "inputs": {"KRlm_iso": "(n_{SH} · n_{shells},)", "Ylm_out": "(n_{dwi}, n_{SH} · n_{shells})"},
      "outputs": {"KERNELS['iso']": "(n_S,)"},
      "computational_note": "Single isotropic atom"
    },
    {
      "step_order": 28,
      "step_name": "Extract Masked Voxels",
      "formula": "\\mathbf{y}_{\\text{data}} = \\mathbf{Y}[\\text{mask\\_indices}], \\quad \\mathbf{d}_{\\text{dirs}} = \\mathbf{D}[\\text{mask\\_indices}]",
      "inputs": {"Y": "(n_x, n_y, n_z, n_S)", "D": "(n_x, n_y, n_z, 3)", "M": "(n_x, n_y, n_z)"},
      "outputs": {"y_data": "(n_{voxels}, n_S)", "d_dirs": "(n_{voxels}, 3)"},
      "computational_note": "Linearize masked voxels"
    },
    {
      "step_order": 29,
      "step_name": "BEGIN_LOOP: Voxel-wise Inversion",
      "formula": "\\text{for } i = 0 \\text{ to } n_{voxels} - 1:",
      "inputs": {"n_voxels": "scalar"},
      "outputs": {"i": "loop index"},
      "computational_note": "Main reconstruction loop"
    },
    {
      "step_order": 30,
      "step_name": "Extract Voxel Signal and Direction",
      "formula": "\\mathbf{y} = \\mathbf{y}_{\\text{data}}[i, :], \\quad \\mathbf{d} = \\mathbf{d}_{\\text{dirs}}[i, :]",
      "inputs": {"y_data": "(n_{voxels}, n_S)", "d_dirs": "(n_{voxels}, 3)", "i": "scalar"},
      "outputs": {"y": "(n_S,)", "d": "(3,)"},
      "computational_note": "Single voxel extraction"
    },
    {
      "step_order": 31,
      "step_name": "Normalize Direction",
      "formula": "\\mathbf{d} \\leftarrow \\frac{\\mathbf{d}}{\\|\\mathbf{d}\\|_2 + 10^{-16}}",
      "inputs": {"d": "(3,)"},
      "outputs": {"d": "(3,) unit vector"},
      "computational_note": "Ensure unit norm"
    },
    {
      "step_order": 32,
      "step_name": "Find Closest Dictionary Direction",
      "formula": "\\text{idx}_{\\text{dir}} = \\arg\\min_j \\|\\mathbf{d} - \\text{GRAD\\_500}[j]\\|_2, \\quad \\text{also check } -\\mathbf{d}",
      "inputs": {"d": "(3,)", "GRAD_500": "(500, 3)", "KDTree": "object"},
      "outputs": {"idx_dir": "scalar in [0, 499]"},
      "computational_note