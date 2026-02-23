## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Zernike Modal Decomposition with FFT-based PSF Forward Model",
  "algorithm_family": "Linear Least Squares / Modal Projection",
  "summary": "Decomposes atmospheric turbulence optical path difference (OPD) maps into Zernike polynomial coefficients using pseudoinverse projection, with an FFT-based forward model for PSF computation."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\mathbf{c}} \\|\\boldsymbol{\\phi}_{\\text{masked}} - \\mathbf{Z} \\mathbf{c}\\|_2^2",
  "data_fidelity_term": "\\|\\boldsymbol{\\phi}_{\\text{masked}} - \\mathbf{Z} \\mathbf{c}\\|_2^2 = \\sum_{p \\in \\mathcal{P}} \\left(\\phi_p - \\sum_{j=1}^{J} Z_{p,j} c_j\\right)^2",
  "regularization_term": "None (unregularized least squares)",
  "constraints": "None (unconstrained)"
}
```

**Notes**: 
- $\boldsymbol{\phi}_{\text{masked}} \in \mathbb{R}^{N_p}$ contains OPD values at pupil pixels
- $\mathbf{Z} \in \mathbb{R}^{N_p \times J}$ is the Zernike basis matrix restricted to pupil
- $\mathcal{P}$ denotes the set of pixel indices where pupil equals 1
- The closed-form solution is $\mathbf{c}^* = \mathbf{Z}^+ \boldsymbol{\phi}_{\text{masked}}$

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{c}$ | $(J,)$ | float64 | Computed from projection | Zernike coefficient vector |

### 2.2 Dual Variables
None (direct method, not iterative optimization)

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\phi}_{\text{masked}}$ | $(N_p,)$ | float64 | Extracted from OPD | OPD values inside pupil |
| $\text{OPD}_{\text{rec}}$ | $(N, N)$ | float64 | Zero array | Reconstructed OPD map |
| $\mathbf{E}$ | $(N, N)$ | complex128 | Computed | Electric field in pupil plane |
| $\mathbf{E}_{\text{pad}}$ | $(N_{\text{pad}}, N_{\text{pad}})$ | complex128 | Zero-padded | Padded electric field |
| $\mathbf{F}$ | $(N_{\text{pad}}, N_{\text{pad}})$ | complex128 | Computed | Complex amplitude in focal plane |
| $\text{PSF}$ | $(N_{\text{pad}}, N_{\text{pad}})$ | float64 | Computed | Point spread function |
| $\text{RMSE}_t$ | scalar | float64 | Computed | Root mean square error at iteration $t$ |
| $R_{n,m}(\rho)$ | $(N, N)$ | float64 | Computed | Zernike radial polynomial |
| $Z_{n,m}(x,y)$ | $(N, N)$ | float64 | Computed | Single Zernike mode |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $N$ | 120 | Telescope resolution (pixels) |
| $D$ | 8.0 m | Telescope diameter |
| $\Delta t$ | 0.001 s | Sampling time |
| $\epsilon_{\text{obs}}$ | 0.0 | Central obstruction ratio |
| $J$ | 100 | Number of Zernike modes |
| $T$ | 10 | Number of time iterations |
| $r_0$ | 0.15 m | Fried parameter |
| $L_0$ | 25 m | Outer scale |
| $v_{\text{wind}}$ | 10 m/s | Wind speed |
| $\theta_{\text{wind}}$ | 0° | Wind direction |
| $h$ | 0 m | Turbulence altitude |
| $\lambda$ | ~0.9 μm | Source wavelength (I-band) |
| $P_{\text{zero}}$ | 4 | Zero-padding factor for FFT |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\text{OPD}_t$ | $(N, N)$ | float64 | Atmospheric OPD at time $t$ [meters] |
| $\mathbf{A}$ | $(N, N)$ | float64 | Telescope pupil binary mask |
| $\mathbf{Z}$ | $(N_p, J)$ | float64 | Zernike basis matrix (pupil pixels) |
| $\mathbf{Z}_{\text{full}}$ | $(N^2, J)$ | float64 | Zernike basis (full resolution) |
| $\mathbf{Z}^+$ | $(J, N_p)$ | float64 | Pseudoinverse of Zernike basis |

## 3. PREPROCESSING PIPELINE
```json
{
  "preprocessing_steps": [
    {
      "step_order": 1,
      "step_name": "Coordinate Grid Generation",
      "formula": "x_{i,j} = \\left(j - \\frac{N}{2}\\right) \\cdot \\Delta_p, \\quad y_{i,j} = \\left(i - \\frac{N}{2}\\right) \\cdot \\Delta_p, \\quad \\Delta_p = \\frac{D}{N}",
      "inputs": ["N: (scalar)", "D: (scalar)"],
      "outputs": ["x: (N, N)", "y: (N, N)"],
      "assumptions": "Grid centered at telescope axis"
    },
    {
      "step_order": 2,
      "step_name": "Normalized Polar Coordinates",
      "formula": "\\rho_{i,j} = \\frac{2\\sqrt{x_{i,j}^2 + y_{i,j}^2}}{D}, \\quad \\theta_{i,j} = \\arctan2(y_{i,j}, x_{i,j})",
      "inputs": ["x: (N, N)", "y: (N, N)", "D: (scalar)"],
      "outputs": ["ρ: (N, N)", "θ: (N, N)"],
      "assumptions": "ρ ∈ [0, 1] inside pupil"
    },
    {
      "step_order": 3,
      "step_name": "Zernike Radial Polynomial Computation",
      "formula": "R_n^{|m|}(\\rho) = \\sum_{k=0}^{(n-|m|)/2} \\frac{(-1)^k (n-k)!}{k! \\left(\\frac{n+|m|}{2}-k\\right)! \\left(\\frac{n-|m|}{2}-k\\right)!} \\rho^{n-2k}",
      "inputs": ["n: (scalar)", "m: (scalar)", "ρ: (N, N)"],
      "outputs": ["R_n^m: (N, N)"],
      "assumptions": "n - |m| must be even and non-negative"
    },
    {
      "step_order": 4,
      "step_name": "Zernike Mode Generation",
      "formula": "Z_n^m(\\rho, \\theta) = \\begin{cases} \\sqrt{n+1} \\, R_n^0(\\rho) & m = 0 \\\\ \\sqrt{2(n+1)} \\, R_n^{|m|}(\\rho) \\cos(m\\theta) & m > 0 \\\\ \\sqrt{2(n+1)} \\, R_n^{|m|}(\\rho) \\sin(|m|\\theta) & m < 0 \\end{cases}",
      "inputs": ["n: (scalar)", "m: (scalar)", "ρ: (N, N)", "θ: (N, N)"],
      "outputs": ["Z_n^m: (N, N)"],
      "assumptions": "Mode is zero outside pupil (ρ > 1)"
    },
    {
      "step_order": 5,
      "step_name": "Zernike Basis Matrix Assembly",
      "formula": "\\mathbf{Z}_{p,j} = Z_{n_j}^{m_j}(\\rho_p, \\theta_p) \\quad \\forall p \\in \\mathcal{P}, \\; j = 1, \\ldots, J",
      "inputs": ["Zernike modes: J × (N, N)", "pupil mask: (N, N)"],
      "outputs": ["Z: (N_p, J)"],
      "assumptions": "Noll indexing maps j → (n, m)"
    },
    {
      "step_order": 6,
      "step_name": "Pseudoinverse Computation",
      "formula": "\\mathbf{Z}^+ = (\\mathbf{Z}^\\top \\mathbf{Z})^{-1} \\mathbf{Z}^\\top",
      "inputs": ["Z: (N_p, J)"],
      "outputs": ["Z⁺: (J, N_p)"],
      "assumptions": "Z has full column rank"
    },
    {
      "step_order": 7,
      "step_name": "Atmosphere Initialization",
      "formula": "\\text{Initialize phase screens with von Kármán spectrum: } \\Phi(\\kappa) = 0.023 \\, r_0^{-5/3} (\\kappa^2 + \\kappa_0^2)^{-11/6}, \\quad \\kappa_0 = \\frac{2\\pi}{L_0}",
      "inputs": ["r0: (scalar)", "L0: (scalar)", "telescope parameters"],
      "outputs": ["atm: Atmosphere object with OPD generator"],
      "assumptions": "Kolmogorov/von Kármán turbulence statistics"
    },
    {
      "step_order": 8,
      "step_name": "Demonstration Phase Map Creation",
      "formula": "\\phi_{\\text{demo}} = 0.5 \\cdot Z_2^0 + 0.5 \\cdot Z_2^2 \\quad \\text{(Defocus + Astigmatism)}",
      "inputs": ["Z_2^0: (N, N)", "Z_2^2: (N, N)"],
      "outputs": ["phase_map: (N, N)"],
      "assumptions": "Coefficients in radians"
    },
    {
      "step_order": 9,
      "step_name": "Phase to OPD Conversion",
      "formula": "\\text{OPD} = \\frac{\\phi \\cdot \\lambda}{2\\pi}",
      "inputs": ["phase_map: (N, N)", "λ: (scalar)"],
      "outputs": ["opd_map: (N, N)"],
      "assumptions": "OPD in meters"
    }
  ]
}
```

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION
```json
{
  "initialization": [
    {
      "step_order": 1,
      "variable_name": "rmse_history",
      "initial_value": "[] \\text{ (empty list)}",
      "shape": "grows to (T,)",
      "dtype": "list → float64 array",
      "source": "Accumulator for RMSE values"
    },
    {
      "step_order": 2,
      "variable_name": "all_coeffs",
      "initial_value": "[] \\text{ (empty list)}",
      "shape": "grows to list of (J,) arrays",
      "dtype": "list",
      "source": "Storage for coefficient history"
    },
    {
      "step_order": 3,
      "variable_name": "all_opd_original",
      "initial_value": "[] \\text{ (empty list)}",
      "shape": "grows to list of (N, N) arrays",
      "dtype": "list",
      "source": "Storage for original OPD maps"
    },
    {
      "step_order": 4,
      "variable_name": "all_opd_reconstructed",
      "initial_value": "[] \\text{ (empty list)}",
      "shape": "grows to list of (N, N) arrays",
      "dtype": "list",
      "source": "Storage for reconstructed OPD maps"
    },
    {
      "step_order": 5,
      "variable_name": "t",
      "initial_value": "0",
      "shape": "scalar",
      "dtype": "int",
      "source": "Iteration counter"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "iteration_loop": [
    {
      "step_order": 6,
      "marker": "BEGIN_LOOP",
      "condition": "t < T",
      "description": "Loop over T atmospheric time frames"
    },
    {
      "step_order": 7,
      "step_name": "Atmosphere Update",
      "formula": "\\text{OPD}_t \\leftarrow \\text{atm.update()}",
      "inputs": ["atm: Atmosphere object", "v_wind: (scalar)", "Δt: (scalar)"],
      "outputs": ["OPD_t: (N, N)"],
      "computational_note": "Advances turbulence phase screen by wind translation"
    },
    {
      "step_order": 8,
      "step_name": "Copy Current OPD",
      "formula": "\\text{current\\_opd} = \\text{OPD}_t.\\text{copy}()",
      "inputs": ["OPD_t: (N, N)"],
      "outputs": ["current_opd: (N, N)"],
      "computational_note": "Deep copy to preserve original"
    },
    {
      "step_order": 9,
      "step_name": "Extract Pupil-Masked OPD",
      "formula": "\\boldsymbol{\\phi}_{\\text{masked}} = \\text{current\\_opd}[\\mathbf{A} = 1]",
      "inputs": ["current_opd: (N, N)", "A (pupil): (N, N)"],
      "outputs": ["φ_masked: (N_p,)"],
      "computational_note": "N_p = number of pixels where pupil equals 1"
    },
    {
      "step_order": 10,
      "step_name": "Zernike Coefficient Projection [INVERSE]",
      "formula": "\\mathbf{c}_t = \\mathbf{Z}^+ \\boldsymbol{\\phi}_{\\text{masked}}",
      "inputs": ["Z⁺: (J, N_p)", "φ_masked: (N_p,)"],
      "outputs": ["c_t: (J,)"],
      "computational_note": "Matrix-vector multiplication; solves least-squares problem"
    },
    {
      "step_order": 11,
      "step_name": "OPD Reconstruction [FORWARD]",
      "formula": "\\text{opd\\_rec\\_flat} = \\mathbf{Z}_{\\text{full}} \\mathbf{c}_t",
      "inputs": ["Z_full: (N², J)", "c_t: (J,)"],
      "outputs": ["opd_rec_flat: (N²,)"],
      "computational_note": "Synthesis using full-resolution Zernike basis"
    },
    {
      "step_order": 12,
      "step_name": "Reshape Reconstructed OPD",
      "formula": "\\text{OPD}_{\\text{rec}} = \\text{reshape}(\\text{opd\\_rec\\_flat}, (N, N))",
      "inputs": ["opd_rec_flat: (N²,)"],
      "outputs": ["OPD_rec: (N, N)"],
      "computational_note": "np.squeeze applied to remove singleton dimension"
    },
    {
      "step_order": 13,
      "step_name": "Compute Residual Map",
      "formula": "\\mathbf{R}_t = (\\text{current\\_opd} - \\text{OPD}_{\\text{rec}}) \\odot \\mathbf{A}",
      "inputs": ["current_opd: (N, N)", "OPD_rec: (N, N)", "A: (N, N)"],
      "outputs": ["R_t: (N, N)"],
      "computational_note": "Element-wise multiplication with pupil mask"
    },
    {
      "step_order": 14,
      "step_name": "Compute RMSE",
      "formula": "\\text{RMSE}_t = \\sqrt{\\frac{1}{N_p} \\sum_{p \\in \\mathcal{P}} R_{t,p}^2} = \\text{std}(\\mathbf{R}_t[\\mathbf{A}=1])",
      "inputs": ["R_t: (N, N)", "A: (N, N)"],
      "outputs": ["RMSE_t: scalar"],
      "computational_note": "Standard deviation of residual over pupil pixels"
    },
    {
      "step_order": 15,
      "step_name": "Append RMSE to History",
      "formula": "\\text{rmse\\_history.append}(\\text{RMSE}_t)",
      "inputs": ["RMSE_t: scalar"],
      "outputs": ["rmse_history: updated list"],
      "computational_note": "Accumulate for later analysis"
    },
    {
      "step_order": 16,
      "step_name": "Store Coefficient Vector",
      "formula": "\\text{all\\_coeffs.append}(\\mathbf{c}_t.\\text{copy}())",
      "inputs": ["c_t: (J,)"],
      "outputs": ["all_coeffs: updated list"],
      "computational_note": "Deep copy to preserve values"
    },
    {
      "step_order": 17,
      "step_name": "Store Original OPD",
      "formula": "\\text{all\\_opd\\_original.append}(\\text{current\\_opd.copy}())",
      "inputs": ["current_opd: (N, N)"],
      "outputs": ["all_opd_original: updated list"],
      "computational_note": "Deep copy for later comparison"
    },
    {
      "step_order": 18,
      "step_name": "Store Reconstructed OPD",
      "formula": "\\text{all\\_opd\\_reconstructed.append}(\\text{OPD}_{\\text{rec}}.\\text{copy}())",
      "inputs": ["OPD_rec: (N, N)"],
      "outputs": ["all_opd_reconstructed: updated list"],
      "computational_note": "Deep copy for later comparison"
    },
    {
      "step_order": 19,
      "step_name": "Increment Counter",
      "formula": "t \\leftarrow t + 1",
      "inputs": ["t: scalar"],
      "outputs": ["t: scalar"],
      "computational_note": "Advance to next iteration"
    },
    {
      "step_order": 20,
      "marker": "END_LOOP",
      "description": "End of temporal iteration loop"
    }
  ]
}
```

### 4.3 FINALIZATION
```json
{
  "finalization": [
    {
      "step_order": 21,
      "step_name": "Convert RMSE History to Array",
      "formula": "\\text{rmse\\_array} = \\text{np.array}(\\text{rmse\\_history})",
      "inputs": ["rmse_history: list of T scalars"],
      "outputs": ["rmse_array: (T,)"],
      "computational_note": "Convert list to numpy array"
    },
    {
      "step_order": 22,
      "step_name": "Compute Final RMSE",
      "formula": "\\text{final\\_rmse} = \\text{rmse\\_array}[T-1]",
      "inputs": ["rmse_array: (T,)"],
      "outputs": ["final_rmse: scalar"],
      "computational_note": "Last iteration's RMSE"
    },
    {
      "step_order": 23,
      "step_name": "Compute Mean RMSE",
      "formula": "\\text{mean\\_rmse} = \\frac{1}{T} \\sum_{t=1}^{T} \\text{RMSE}_t",
      "inputs": ["rmse_array: (T,)"],
      "outputs": ["mean_rmse: scalar"],
      "computational_note": "Average over all iterations"
    },
    {
      "step_order": 24,
      "step_name": "Forward Model: Create Electric Field",
      "formula": "\\mathbf{E} = \\mathbf{A} \\odot \\exp(i \\cdot \\phi_{\\text{demo}})",
      "inputs": ["A (pupil): (N, N)", "phase_map: (N, N)"],
      "outputs": ["E: (N, N) complex"],
      "computational_note": "Complex pupil function"
    },
    {
      "step_order": 25,
      "step_name": "Forward Model: Zero Padding",
      "formula": "\\mathbf{E}_{\\text{pad}} = \\text{pad}(\\mathbf{E}, \\text{width}=\\frac{N(P_{\\text{zero}}-1)}{2})",
      "inputs": ["E: (N, N)", "P_zero: 4"],
      "outputs": ["E_pad: (N·P_zero, N·P_zero) = (480, 480)"],
      "computational_note": "Symmetric zero-padding around electric field"
    },
    {
      "step_order": 26,
      "step_name": "Forward Model: FFT to Focal Plane [FORWARD]",
      "formula": "\\mathbf{F} = \\mathcal{F}_{\\text{shift}}\\{\\mathcal{F}_2\\{\\mathcal{F}_{\\text{ishift}}\\{\\mathbf{E}_{\\text{pad}}\\}\\}\\}",
      "inputs": ["E_pad: (480, 480)"],
      "outputs": ["F: (480, 480) complex"],
      "computational_note": "ifftshift → fft2 → fftshift sequence for centered spectrum"
    },
    {
      "step_order": 27,
      "step_name": "Forward Model: Compute PSF Intensity",
      "formula": "\\text{PSF} = |\\mathbf{F}|^2 = \\mathbf{F} \\odot \\mathbf{F}^*",
      "inputs": ["F: (480, 480) complex"],
      "outputs": ["PSF: (480, 480) float"],
      "computational_note": "Squared modulus of complex focal field"
    },
    {
      "step_order": 28,
      "step_name": "Forward Model: Normalize PSF",
      "formula": "\\text{PSF}_{\\text{norm}} = \\frac{\\text{PSF}}{\\max(\\text{PSF})}",
      "inputs": ["PSF: (480, 480)"],
      "outputs": ["PSF_norm: (480, 480)"],
      "computational_note": "Peak normalization to unity"
    }
  ]
}
```

## 5. STOPPING CRITERION
```json
{
  "type": "iteration_limit",
  "expression": "t \\geq T",
  "parameters": {
    "T": 10,
    "description": "Fixed number of atmospheric time frames"
  },
  "check_frequency": "After each iteration (step 19)"
}
```

## 6. EVALUATION METRICS
```json
{
  "evaluation_metrics": [
    {
      "name": "RMSE (per iteration)",
      "formula": "\\text{RMSE}_t = \\sqrt{\\frac{1}{N_p} \\sum_{p \\in \\mathcal{P}} \\left(\\text{OPD}_{t,p} - \\text{OPD}_{\\text{rec},t,p}\\right)^2}",
      "reference_data": "current_opd (original atmospheric OPD)",
      "predicted_data": "OPD_rec (Zernike-reconstructed OPD)",
      "description": "Root mean square error of OPD reconstruction within pupil"
    },
    {
      "name": "Mean RMSE",
      "formula": "\\overline{\\text{RMSE}} = \\frac{1}{T} \\sum_{t=1}^{T} \\text{RMSE}_t",
      "reference_data": "All RMSE values",
      "predicted_data": "N/A",
      "description": "Average reconstruction error across all time frames"
    },
    {
      "name": "Final RMSE",
      "formula": "\\text{RMSE}_{T}",
      "reference_data": "Last original OPD",
      "predicted_data": "Last reconstructed OPD",
      "description": "Reconstruction error at final iteration"
    },
    {
      "name": "RMSE Standard Deviation",
      "formula": "\\sigma_{\\text{RMSE}} = \\sqrt{\\frac{1}{T-1} \\sum_{t=1}^{T} (\\text{RMSE}_t - \\overline{\\text{RMSE}})^2}",
      "reference_data": "All RMSE values",
      "predicted_data": "N/A",
      "description": "Variability of reconstruction quality over time"
    },
    {
      "name": "Min/Max RMSE",
      "formula": "\\text{RMSE}_{\\min} = \\min_t \\text{RMSE}_t, \\quad \\text{RMSE}_{\\max} = \\max_t \\text{RMSE}_t",
      "reference_data": "All RMSE values",
      "predicted_data": "N/A",
      "description": "Range of reconstruction quality"
    }
  ]
}
```