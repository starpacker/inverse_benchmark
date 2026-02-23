## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "FISTA-BPM Optical Diffraction Tomography",
  "algorithm_family": "Proximal Gradient Methods / FISTA with Adjoint-State Gradient",
  "summary": "Reconstructs 3D refractive index distribution from holographic measurements using FISTA-accelerated gradient descent with a Beam Propagation Method (BPM) forward model and composite regularization (TV + box constraints + sparsity)."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\Delta n} \\frac{1}{N_{\\text{angles}}} \\sum_{j=1}^{N_{\\text{angles}}} \\|\\mathcal{A}_j(\\Delta n) - u_{\\text{out}}^{(j)}\\|_1 + \\tau \\|\\nabla \\Delta n\\|_{2,1} + \\lambda_s \\|\\Delta n\\|_1 + \\mathcal{I}_{[n_{\\min}, n_{\\max}]}(\\Delta n_{\\text{ROI}})",
  "data_fidelity_term": "\\frac{1}{N_{\\text{angles}}} \\sum_{j=1}^{N_{\\text{angles}}} \\|\\mathcal{A}_j(\\Delta n) - u_{\\text{out}}^{(j)}\\|_1 where $\\|\\cdot\\|_1$ is element-wise L1 norm (mean of absolute values)",
  "regularization_term": {
    "total_variation": "$\\tau \\|\\nabla \\Delta n\\|_{2,1} = \\tau \\sum_{i,j,k} \\sqrt{(\\partial_z \\Delta n)^2 + (\\partial_y \\Delta n)^2 + (\\partial_x \\Delta n)^2}$ with $\\tau = 1.75 \\times 10^{-6}$",
    "sparsity": "$\\lambda_s \\|\\Delta n\\|_1$ with $\\lambda_s = \\text{sparse\\_param} \\times \\text{step\\_size}$ (None in config, so disabled)",
    "box_constraint": "$\\mathcal{I}_{[n_{\\min}, n_{\\max}]}(\\Delta n_{\\text{ROI}}) = \\begin{cases} 0 & \\text{if } n_{\\min} \\leq \\Delta n_{\\text{ROI}} \\leq n_{\\max} \\\\ +\\infty & \\text{otherwise} \\end{cases}$ with $n_{\\min}=0, n_{\\max}=100$"
  },
  "constraints": "Box constraint on ROI: $0 \\leq \\Delta n[s_0:e_0, s_1:e_1, s_2:e_2] \\leq 100$"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Delta n$ | $(N_z, N_y, N_x)$ | float32 | Zero tensor | Refractive index perturbation (primary unknown) |
| $s$ | $(N_z, N_y, N_x)$ | float32 | Clone of $\Delta n$ | FISTA extrapolated point for outer loop |
| $x^{(k)}$ | $(N_z, N_y, N_x)$ | float32 | Clone of $\Delta n$ | Current iterate in outer FISTA |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g$ | $(N_z, N_y, N_x, 3)$ | float32 | $\nabla z$ (gradient of input) | Dual variable for TV regularization inner FISTA |
| $d$ | $(N_z, N_y, N_x, 3)$ | float32 | Clone of $g$ | FISTA extrapolated dual variable in TV inner loop |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $q_1^{\text{outer}}$ | scalar | float64 | 1.0 | FISTA momentum parameter (outer loop) |
| $q_2^{\text{outer}}$ | scalar | float64 | Computed | Updated momentum (outer loop) |
| $q_1^{\text{inner}}$ | scalar | float64 | 1.0 | FISTA momentum parameter (TV inner loop) |
| $q_2^{\text{inner}}$ | scalar | float64 | Computed | Updated momentum (TV inner loop) |
| $u$ | $(N_{\text{batch}}, N_y, N_x)$ | complex64 | $u_{\text{inlet}}^{(\text{batch})}$ | Propagating field in forward model |
| $r$ | $(N_{\text{batch}}, N_y, N_x)$ | complex64 | $u - u_{\text{outlet}}$ | Residual field for adjoint computation |
| $S$ | $(N_{\text{batch}}, N_z, N_y, N_x)$ | complex64 | Zero tensor | Stored intermediate fields for adjoint |
| $\text{grad}$ | $(N_z, N_y, N_x)$ | float32 | Zero tensor | Accumulated gradient |
| $z$ | $(N_z, N_y, N_x)$ | float32 | Computed | Gradient descent intermediate |

### 2.4 Constants
| Symbol | Value/Formula | Description |
|--------|---------------|-------------|
| $\lambda$ | 0.532 μm | Wavelength |
| $n_{\text{med}}$ | 1.33 | Medium refractive index |
| $k_0$ | $2\pi/\lambda = 11.814$ rad/μm | Vacuum wavenumber |
| $k_m$ | $k_0 \cdot n_{\text{med}} = 15.712$ rad/μm | Medium wavenumber |
| $\Delta_{\text{pix}}$ | $4.5/100 = 0.045$ μm | Effective pixel size |
| $\Delta_{\text{spec}}$ | $2\pi/(\Delta_{\text{pix}} \cdot N_{\text{crop}})$ | Spectral pixel size |
| $\Delta z$ | $\Delta_{\text{pix}} \cdot N_{\text{crop}} / N_z$ | Axial resolution |
| $\alpha$ | 0.0005 | Step size |
| $\tau$ | $1.75 \times 10^{-6}$ | TV regularization weight |
| $N_{\text{TV}}$ | 50 | TV inner iterations |
| $\gamma$ | $1/(12\tau)$ | TV dual step size |
| $N_{\text{iter}}$ | 1 (config) | Outer iterations |
| $N_{\text{batch}}$ | 2 | Batch size for gradient computation |
| ROI | $(0, 256, 60, -60, 60, -60)$ | Region of interest bounds |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $u_{\text{in}}$ | $(N_{\text{angles}}, N_y, N_x)$ | complex64 | Raw input fields (loaded from file) |
| $u_{\text{out}}$ | $(N_{\text{angles}}, N_y, N_x)$ | complex64 | Raw output fields (loaded from file) |
| $k_{\text{scan}}$ | $(N_{\text{angles}}, 2)$ | float64 | Illumination k-vector samples |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": 1,
      "step_name": "Compute physical parameters",
      "formula": "$k_0 = \\frac{2\\pi}{\\lambda}, \\quad k_m = k_0 \\cdot n_{\\text{med}}, \\quad \\Delta_{\\text{pix}} = \\frac{\\Delta_{\\text{cam}}}{M}, \\quad \\Delta_{\\text{spec}} = \\frac{2\\pi}{\\Delta_{\\text{pix}} \\cdot N_{\\text{crop}}}$",
      "inputs": {"wavelength": "scalar", "camera_pixel_size": "scalar", "magnification": "scalar", "n_medium": "scalar", "crop_size": "(2,)"},
      "outputs": {"k0": "scalar", "km": "scalar", "pixel_size": "scalar", "spec_pixel_size": "scalar"},
      "assumptions": "All physical units consistent (micrometers)"
    },
    {
      "step_order": 2,
      "step_name": "Compute resolution",
      "formula": "$\\Delta = \\frac{\\Delta_{\\text{pix}} \\cdot N_{\\text{crop}}}{N_{\\text{domain}}}$ element-wise for each dimension",
      "inputs": {"pixel_size": "scalar", "crop_size": "(2,)", "domain_size": "(3,)"},
      "outputs": {"resolution": "(3,)"},
      "assumptions": "domain_size defines reconstruction grid"
    },
    {
      "step_order": 3,
      "step_name": "Load measurement data",
      "formula": "$u_{\\text{in}} \\leftarrow \\texttt{load}(\\texttt{u\\_in.npy}), \\quad u_{\\text{out}} \\leftarrow \\texttt{load}(\\texttt{u\\_out.npy}), \\quad k_{\\text{scan}} \\leftarrow \\texttt{load}(\\texttt{k\\_samp.npy})$",
      "inputs": {"file_paths": "strings"},
      "outputs": {"u_in": "(N_angles, N_y, N_x) complex", "u_out": "(N_angles, N_y, N_x) complex", "k_scan": "(N_angles, 2) float"},
      "assumptions": "Files exist in data_folder/"
    },
    {
      "step_order": 4,
      "step_name": "Compute oblique illumination correction factor",
      "formula": "$\\cos\\theta_j = \\cos\\left(\\arcsin\\left(\\sqrt{\\left(\\frac{k_{\\text{scan},j,0} \\cdot \\Delta_{\\text{spec}}}{k_m}\\right)^2 + \\left(\\frac{k_{\\text{scan},j,1} \\cdot \\Delta_{\\text{spec}}}{k_m}\\right)^2}\\right)\\right)$",
      "inputs": {"k_scan": "(N_angles, 2)", "spec_pixel_size": "scalar", "km": "scalar"},
      "outputs": {"bpm_cosFactor": "(N_angles, 1) float32"},
      "assumptions": "Illumination angles within NA"
    },
    {
      "step_order": 5,
      "step_name": "Construct angular spectrum propagation kernel",
      "formula": "$K_x[i] = \\left(i - \\frac{N_y}{2}\\right) \\cdot \\Delta_{\\text{spec}}, \\quad K_y[j] = \\left(j - \\frac{N_x}{2}\\right) \\cdot \\Delta_{\\text{spec}}$ for $i,j \\in [0, N-1]$; $K_z = \\sqrt{k_m^2 - K_x^2 - K_y^2}$ (set to 0 where argument negative); $H = e^{i K_z \\Delta z}$; $H_{\\text{kernel}} = \\texttt{fftshift}(H)$",
      "inputs": {"domain_size": "(3,)", "spec_pixel_size": "scalar", "resolution": "(3,)", "km": "scalar"},
      "outputs": {"p_kernel": "(N_y, N_x) complex64"},
      "assumptions": "Evanescent waves zeroed out"
    },
    {
      "step_order": 6,
      "step_name": "Propagate input field to sample inlet plane [FORWARD]",
      "formula": "$u_{\\text{inlet}} = \\mathcal{F}^{-1}\\left\\{\\mathcal{F}\\{u_{\\text{in}}\\} \\cdot H^{*(N_z^{\\text{region}} - N_z^{\\text{bg}})}\\right\\}$",
      "inputs": {"u_in": "(N_angles, N_y, N_x)", "p_kernel": "(N_y, N_x)", "region_z": "scalar", "bg_z": "scalar"},
      "outputs": {"u_inlet": "(N_angles, N_y, N_x) complex64"},
      "assumptions": "Conjugate for backward propagation"
    },
    {
      "step_order": 7,
      "step_name": "Propagate output field to sample outlet plane [FORWARD]",
      "formula": "$u_{\\text{outlet}} = \\mathcal{F}^{-1}\\left\\{\\mathcal{F}\\{u_{\\text{out}}\\} \\cdot H^{N_z^{\\text{bg}}}\\right\\}$",
      "inputs": {"u_out": "(N_angles, N_y, N_x)", "p_kernel": "(N_y, N_x)", "bg_z": "scalar"},
      "outputs": {"u_outlet": "(N_angles, N_y, N_x) complex64"},
      "assumptions": "Forward propagation to outlet"
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
      "variable_name": "init_delta_ri (Δn)",
      "initial_value": "$\\Delta n^{(0)} = \\mathbf{0}$",
      "shape": "(N_z, N_y, N_x) = (256, 502, 502)",
      "dtype": "float32",
      "source": "Zero initialization"
    },
    {
      "step_order": 2,
      "variable_name": "s (extrapolated point)",
      "initial_value": "$s^{(0)} = \\Delta n^{(0)}$",
      "shape": "(N_z, N_y, N_x)",
      "dtype": "float32",
      "source": "Clone of init_delta_ri"
    },
    {
      "step_order": 3,
      "variable_name": "x_1 (previous iterate)",
      "initial_value": "$x^{(0)} = \\Delta n^{(0)}$",
      "shape": "(N_z, N_y, N_x)",
      "dtype": "float32",
      "source": "Clone of init_delta_ri"
    },
    {
      "step_order": 4,
      "variable_name": "q_1 (FISTA momentum)",
      "initial_value": "$q^{(0)} = 1$",
      "shape": "scalar",
      "dtype": "float64",
      "source": "Standard FISTA initialization"
    },
    {
      "step_order": 5,
      "variable_name": "ol_factor (phase modulation factor)",
      "initial_value": "$\\phi_j = \\frac{k_0 \\cdot \\Delta z}{\\cos\\theta_j}$ for each angle $j$",
      "shape": "(N_angles, 1)",
      "dtype": "float32",
      "source": "Computed from k0, resolution[0], bpm_cosFactor"
    },
    {
      "step_order": 6,
      "variable_name": "p_kernel_expanded",
      "initial_value": "$H_{\\text{exp}} = H[\\texttt{newaxis}, :, :]$",
      "shape": "(1, N_y, N_x)",
      "dtype": "complex64",
      "source": "Unsqueeze p_kernel for broadcasting"
    },
    {
      "step_order": 7,
      "variable_name": "bp_kernel (backpropagation kernel)",
      "initial_value": "$H^{\\dagger} = H^*[\\texttt{newaxis}, :, :]$",
      "shape": "(1, N_y, N_x)",
      "dtype": "complex64",
      "source": "Conjugate of p_kernel_expanded"
    },
    {
      "step_order": 8,
      "variable_name": "loss_history",
      "initial_value": "Empty list $[]$",
      "shape": "list",
      "dtype": "float64",
      "source": "For tracking convergence"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "steps": [
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "iteration",
      "range": "[0, N_iter)",
      "description": "Outer FISTA loop"
    },
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "i (batch index)",
      "range": "[0, num_batches)",
      "description": "Batch loop for gradient computation"
    },
    {
      "step_order": 9,
      "step_name": "Extract batch indices",
      "formula": "$i_{\\text{start}} = i \\cdot N_{\\text{batch}}, \\quad i_{\\text{end}} = \\min((i+1) \\cdot N_{\\text{batch}}, N_{\\text{angles}})$",
      "inputs": {"i": "scalar", "batch_size": "scalar", "N_angles": "scalar"},
      "outputs": {"start_idx": "scalar", "end_idx": "scalar"},
      "computational_note": "Integer indexing"
    },
    {
      "step_order": 10,
      "step_name": "Slice batch data",
      "formula": "$u_{\\text{in}}^{(b)} = u_{\\text{inlet}}[i_{\\text{start}}:i_{\\text{end}}], \\quad u_{\\text{out}}^{(b)} = u_{\\text{outlet}}[i_{\\text{start}}:i_{\\text{end}}], \\quad \\phi^{(b)} = \\phi[i_{\\text{start}}:i_{\\text{end}}]$",
      "inputs": {"u_inlet": "(N_angles, N_y, N_x)", "u_outlet": "(N_angles, N_y, N_x)", "ol_factor": "(N_angles, 1)"},
      "outputs": {"sub_u_in": "(B, N_y, N_x)", "sub_u_out": "(B, N_y, N_x)", "sub_ol_factor": "(B, 1)"},
      "computational_note": "B = actual_batch_size"
    },
    {
      "step_order": 11,
      "step_name": "Initialize field storage",
      "formula": "$S = \\mathbf{0} \\in \\mathbb{C}^{B \\times N_z \\times N_y \\times N_x}$",
      "inputs": {"actual_batch_size": "scalar", "domain_size": "(3,)"},
      "outputs": {"s_fields": "(B, N_z, N_y, N_x) complex64"},
      "computational_note": "Stores intermediate fields for adjoint"
    },
    {
      "step_order": 12,
      "step_name": "Initialize propagating field",
      "formula": "$u = u_{\\text{in}}^{(b)}$",
      "inputs": {"sub_u_in": "(B, N_y, N_x)"},
      "outputs": {"u": "(B, N_y, N_x) complex64"},
      "computational_note": "Clone operation"
    },
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "m (z-slice)",
      "range": "[0, N_z)",
      "description": "Forward BPM propagation"
    },
    {
      "step_order": 13,
      "step_name": "Angular spectrum propagation [FORWARD]",
      "formula": "$u \\leftarrow \\mathcal{F}^{-1}\\left\\{\\mathcal{F}\\{u\\} \\odot H_{\\text{exp}}\\right\\}$",
      "inputs": {"u": "(B, N_y, N_x)", "p_kernel_expanded": "(1, N_y, N_x)"},
      "outputs": {"u": "(B, N_y, N_x) complex64"},
      "computational_note": "2D FFT along last two dims, element-wise multiply, 2D IFFT"
    },
    {
      "step_order": 14,
      "step_name": "Store intermediate field",
      "formula": "$S[:, m, :, :] = u$",
      "inputs": {"u": "(B, N_y, N_x)", "m": "scalar"},
      "outputs": {"s_fields": "(B, N_z, N_y, N_x) updated"},
      "computational_note": "Clone to avoid aliasing"
    },
    {
      "step_order": 15,
      "step_name": "Phase modulation [FORWARD]",
      "formula": "$u \\leftarrow u \\odot \\exp\\left(i \\cdot \\phi^{(b)} \\cdot s[m, :, :]\\right)$",
      "inputs": {"u": "(B, N_y, N_x)", "sub_ol_factor": "(B, 1)", "s": "(N_z, N_y, N_x)", "m": "scalar"},
      "outputs": {"u": "(B, N_y, N_x) complex64"},
      "computational_note": "s[m] unsqueezed to (1, N_y, N_x) for broadcasting"
    },
    {
      "marker": "END_LOOP",
      "loop_variable": "m"
    },
    {
      "step_order": 16,
      "step_name": "Compute residual",
      "formula": "$r = u - u_{\\text{out}}^{(b)}$",
      "inputs": {"u": "(B, N_y, N_x)", "sub_u_out": "(B, N_y, N_x)"},
      "outputs": {"r": "(B, N_y, N_x) complex64"},
      "computational_note": "Complex subtraction"
    },
    {
      "step_order": 17,
      "step_name": "Compute batch loss (L1)",
      "formula": "$\\mathcal{L}_{\\text{batch}} = \\frac{1}{B \\cdot N_y \\cdot N_x} \\sum_{b,i,j} |r_{b,i,j}|$",
      "inputs": {"r": "(B, N_y, N_x)"},
      "outputs": {"batch_loss": "scalar float"},
      "computational_note": "Mean of absolute values"
    },
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "m (z-slice, reversed)",
      "range": "[N_z-1, ..., 0]",
      "description": "Adjoint BPM backpropagation"
    },
    {
      "step_order": 18,
      "step_name": "Adjoint phase modulation [ADJOINT]",
      "formula": "$r \\leftarrow r \\odot \\exp\\left(-i \\cdot \\phi^{(b)} \\cdot s[m, :, :]\\right)$",
      "inputs": {"r": "(B, N_y, N_x)", "sub_ol_factor": "(B, 1)", "s": "(N_z, N_y, N_x)", "m": "scalar"},
      "outputs": {"r": "(B, N_y, N_x) complex64"},
      "computational_note": "Conjugate of forward phase modulation"
    },
    {
      "step_order": 19,
      "step_name": "Compute slice gradient",
      "formula": "$G_m^{(b)} = -i \\cdot \\phi^{(b)} \\cdot S[:, m, :, :]^* \\odot r$",
      "inputs": {"sub_ol_factor": "(B, 1)", "s_fields": "(B, N_z, N_y, N_x)", "r": "(B, N_y, N_x)", "m": "scalar"},
      "outputs": {"batch_grad": "(B, N_y, N_x) complex64"},
      "computational_note": "Wirtinger derivative contribution"
    },
    {
      "step_order": 20,
      "step_name": "Accumulate gradient",
      "formula": "$\\nabla_{\\Delta n}[m, :, :] \\mathrel{+}= \\sum_{b=1}^{B} \\text{Re}(G_m^{(b)})$",
      "inputs": {"grad": "(N_z, N_y, N_x)", "batch_grad": "(B, N_y, N_x)", "m": "scalar"},
      "outputs": {"grad": "(N_z, N_y, N_x) updated"},
      "computational_note": "Sum over batch dimension, take real part"
    },
    {
      "step_order": 21,
      "step_name": "Adjoint angular spectrum propagation [ADJOINT]",
      "formula": "$r \\leftarrow \\mathcal{F}^{-1}\\left\\{\\mathcal{F}\\{r\\} \\odot H^{\\dagger}\\right\\}$",
      "inputs": {"r": "(B, N_y, N_x)", "bp_kernel": "(1, N_y, N_x)"},
      "outputs": {"r": "(B, N_y, N_x) complex64"},
      "computational_note": "Conjugate kernel for adjoint propagation"
    },
    {
      "marker": "END_LOOP",
      "loop_variable": "m (reversed)"
    },
    {
      "marker": "END_LOOP",
      "loop_variable": "i (batch)"
    },
    {
      "step_order": 22,
      "step_name": "Normalize gradient",
      "formula": "$\\nabla_{\\Delta n} \\leftarrow \\frac{\\nabla_{\\Delta n}}{N_{\\text{angles}}}$",
      "inputs": {"grad": "(N_z, N_y, N_x)", "N_angles": "scalar"},
      "outputs": {"grad": "(N_z, N_y, N_x)"},
      "computational_note": "Average over all illumination angles"
    },
    {
      "step_order": 23,
      "step_name": "Gradient descent step",
      "formula": "$z = s - \\alpha \\cdot \\nabla_{\\Delta n}$",
      "inputs": {"s": "(N_z, N_y, N_x)", "grad": "(N_z, N_y, N_x)", "step_size": "scalar = 0.0005"},
      "outputs": {"z": "(N_z, N_y, N_x)"},
      "computational_note": "No gradient tracking (torch.no_grad)"
    },
    {
      "step_order": 24,
      "step_name": "Apply composite proximal operator",
      "formula": "$x^{(k+1)} = \\text{prox}_{\\text{TV}+\\text{box}+\\text{sparse}}(z)$",
      "inputs": {"z": "(N_z, N_y, N_x)"},
      "outputs": {"x_2": "(N_z, N_y, N_x)"},
      "computational_note": "Calls fista_regu → see nested TV-FISTA below"
    },
    {
      "marker": "BEGIN_NESTED_BLOCK",
      "block_name": "TV-FISTA Proximal Operator (fista_regu)",
      "description": "Inner FISTA for TV denoising"
    },
    {
      "step_order": "24a",
      "step_name": "Initialize TV dual variable",
      "formula": "$g^{(0)} = \\nabla z$ where $(\\nabla z)_{i,j,k,0} = z_{i+1,j,k} - z_{i,j,k}$, $(\\nabla z)_{i,j,k,1} = z_{i,j+1,k} - z_{i,j,k}$, $(\\nabla z)_{i,j,k,2} = z_{i,j,k+1} - z_{i,j,k}$ with zero-padding at boundaries",
      "inputs": {"z": "(N_z, N_y, N_x)"},
      "outputs": {"g_1":