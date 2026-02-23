## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Closed-Loop Adaptive Optics with Integral Control",
  "algorithm_family": "Linear Feedback Control / Wavefront Reconstruction",
  "summary": "Iterative closed-loop control algorithm that minimizes residual wavefront error by measuring Shack-Hartmann slopes and applying integral control to deformable mirror commands using a modal reconstructor derived from SVD-based pseudoinverse of the interaction matrix."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\mathbf{u}} \\| \\mathbf{s}(\\mathbf{u}) - \\mathbf{s}_{\\text{ref}} \\|_2^2 \\quad \\text{subject to} \\quad \\mathbf{s}(\\mathbf{u}) = \\mathcal{W}(\\phi_{\\text{atm}} - \\mathbf{B}\\mathbf{u})",
  "data_fidelity_term": "\\| \\mathbf{s}_{\\text{meas}} - \\mathbf{s}_{\\text{ref}} \\|_2^2 where $\\mathbf{s}_{\\text{meas}} \\in \\mathbb{R}^{n_{\\text{meas}}}$ are measured WFS slopes and $\\mathbf{s}_{\\text{ref}}$ are reference slopes from flat wavefront",
  "regularization_term": "None (implicit regularization via SVD truncation with threshold $\\tau = 10^{-3}$)",
  "constraints": "None (unconstrained DM commands)"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{u}^{(k)}$ | $(n_{\text{act}},)$ | float64 | $\mathbf{0}$ | DM actuator commands at iteration $k$, where $n_{\text{act}} = (n_{\text{sub}}+1)^2$ |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| — | — | — | — | No explicit dual variables (not a primal-dual algorithm) |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{s}_{\text{meas}}^{(k)}$ | $(n_{\text{meas}},)$ | float64 | Computed | Raw measured slopes at iteration $k$ |
| $\Delta\mathbf{s}^{(k)}$ | $(n_{\text{meas}},)$ | float64 | Computed | Residual slopes: $\mathbf{s}_{\text{meas}}^{(k)} - \mathbf{s}_{\text{ref}}$ |
| $\Delta\mathbf{u}^{(k)}$ | $(n_{\text{act}},)$ | float64 | Computed | Command update: $\mathbf{R} \Delta\mathbf{s}^{(k)}$ |
| $\mathbf{E}_{\text{cube}}^{(k)}$ | $(n_{\text{sub}}^2, n_{\text{pix}}, n_{\text{pix}})$ | complex128 | Computed | Electric field at lenslet array |
| $\tilde{\mathbf{E}}_{\text{cube}}^{(k)}$ | $(n_{\text{sub}}^2, n_{\text{pix}}, n_{\text{pix}})$ | complex128 | Computed | FFT of electric field (spot formation) |
| $\mathbf{I}_{\text{spots}}^{(k)}$ | $(n_{\text{sub}}^2, n_{\text{pix}}, n_{\text{pix}})$ | float64 | Computed | Spot intensities: $|\tilde{\mathbf{E}}_{\text{cube}}|^2$ |
| $\text{SR}^{(k)}$ | scalar | float64 | Computed | Strehl ratio at iteration $k$ (percentage) |
| $\mathbf{PSF}^{(k)}$ | $(2n_{\text{res}}, 2n_{\text{res}})$ | float64 | Computed | Science camera PSF at iteration $k$ |

### 2.4 Constants
| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $n_{\text{sub}}$ | scalar | int | 20 | Number of subapertures per dimension |
| $n_{\text{modes}}$ | scalar | int | 20 | Number of KL modes for control |
| $n_{\text{res}}$ | scalar | int | $6 \times n_{\text{sub}} = 120$ | Pupil resolution (pixels) |
| $n_{\text{pix}}$ | scalar | int | $n_{\text{res}} / n_{\text{sub}} = 6$ | Pixels per subaperture |
| $n_{\text{act}}$ | scalar | int | $(n_{\text{sub}}+1)^2 = 441$ | Number of DM actuators |
| $n_{\text{meas}}$ | scalar | int | $2 \times n_{\text{valid}}$ | Number of slope measurements |
| $n_{\text{valid}}$ | scalar | int | Computed | Number of valid subapertures |
| $g$ | scalar | float64 | 0.4 | Integral control gain |
| $K_{\text{iter}}$ | scalar | int | 20 | Maximum number of iterations |
| $\sigma_{\text{stroke}}$ | scalar | float64 | $10^{-8}$ | Calibration stroke amplitude (meters) |
| $\tau$ | scalar | float64 | $10^{-3}$ | SVD truncation threshold |
| $D$ | scalar | float64 | 8.0 | Telescope diameter (meters) |
| $r_0$ | scalar | float64 | 0.15 | Fried parameter (meters) |
| $L_0$ | scalar | float64 | 25 | Outer scale (meters) |
| $\mathbf{R}$ | $(n_{\text{act}}, n_{\text{meas}})$ | float64 | Calibrated | Final reconstructor matrix |
| $\mathbf{M}_{2C}$ | $(n_{\text{act}}, n_{\text{modes}})$ | float64 | Calibrated | Mode-to-command basis (KL modes) |
| $\mathbf{s}_{\text{ref}}$ | $(n_{\text{meas}},)$ | float64 | Calibrated | Reference slopes (flat wavefront) |
| $\mathbf{PSF}_{\text{ref}}$ | $(2n_{\text{res}}, 2n_{\text{res}})$ | float64 | Calibrated | Diffraction-limited reference PSF |
| $\mathbf{X}$ | $(n_{\text{pix}}, n_{\text{pix}})$ | float64 | Precomputed | X-coordinate grid for centroiding |
| $\mathbf{Y}$ | $(n_{\text{pix}}, n_{\text{pix}})$ | float64 | Precomputed | Y-coordinate grid for centroiding |
| $\mathbf{v}_{\text{valid}}$ | $(n_{\text{sub}}^2,)$ | bool | From WFS | Valid subaperture mask |

### 2.5 Observations
| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $\phi_{\text{atm}}^{(k)}$ | $(n_{\text{res}}, n_{\text{res}})$ | float64 | Atmosphere simulation | Atmospheric phase screen at iteration $k$ |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": "P1",
      "step_name": "Initialize Telescope",
      "formula": "n_{\\text{res}} = 6 \\times n_{\\text{sub}}, \\quad \\text{tel} = \\text{Telescope}(n_{\\text{res}}, D=8.0, T_s=0.001)",
      "inputs": {"n_sub": "scalar int"},
      "outputs": {"tel": "Telescope object", "n_res": "120"},
      "assumptions": "No central obstruction"
    },
    {
      "step_order": "P2",
      "step_name": "Initialize Source",
      "formula": "\\text{ngs} = \\text{Source}(\\lambda_I, m=8, [0,0]), \\quad \\text{ngs} * \\text{tel}",
      "inputs": {"tel": "Telescope object"},
      "outputs": {"ngs": "Source object coupled to telescope"},
      "assumptions": "Natural guide star at infinity, I-band"
    },
    {
      "step_order": "P3",
      "step_name": "Initialize Atmosphere",
      "formula": "\\text{atm} = \\text{Atmosphere}(r_0=0.15, L_0=25, v=10, \\theta=0, h=0)",
      "inputs": {"tel": "Telescope object"},
      "outputs": {"atm": "Atmosphere object with single turbulent layer"},
      "assumptions": "Single layer at ground, Kolmogorov turbulence"
    },
    {
      "step_order": "P4",
      "step_name": "Initialize Deformable Mirror",
      "formula": "\\text{dm} = \\text{DeformableMirror}(n_{\\text{sub}}, \\alpha=0.35), \\quad n_{\\text{act}} = (n_{\\text{sub}}+1)^2",
      "inputs": {"tel": "Telescope object", "n_sub": "20"},
      "outputs": {"dm": "DM object with 441 actuators"},
      "assumptions": "Mechanical coupling factor 0.35"
    },
    {
      "step_order": "P5",
      "step_name": "Initialize Wavefront Sensor",
      "formula": "\\text{wfs} = \\text{ShackHartmann}(n_{\\text{sub}}=20, \\text{lightRatio}=0.5)",
      "inputs": {"tel": "Telescope object", "n_sub": "20"},
      "outputs": {"wfs": "WFS object", "n_valid": "number of valid subapertures", "n_meas": "2 × n_valid"},
      "assumptions": "50% light ratio threshold for valid subapertures"
    },
    {
      "step_order": "P6",
      "step_name": "Initialize Science Camera",
      "formula": "\\text{sci\\_cam} = \\text{Detector}(2 \\times n_{\\text{res}})",
      "inputs": {"n_res": "120"},
      "outputs": {"sci_cam": "Detector with 240×240 pixels"},
      "assumptions": "2× Nyquist sampling"
    },
    {
      "step_order": "P7",
      "step_name": "Compute Reference PSF",
      "formula": "\\text{tel.resetOPD()}, \\quad \\text{ngs} * \\text{tel} * \\text{sci\\_cam}, \\quad \\mathbf{PSF}_{\\text{ref}} = \\text{sci\\_cam.frame}",
      "inputs": {"tel": "Telescope", "ngs": "Source", "sci_cam": "Detector"},
      "outputs": {"PSF_ref": "(240, 240) float64"},
      "assumptions": "Flat wavefront represents diffraction limit"
    },
    {
      "step_order": "P8",
      "step_name": "Compute Reference Slopes",
      "formula": "\\mathbf{s}_{\\text{ref}} = \\mathcal{W}(\\phi=0)",
      "inputs": {"wfs": "WFS object", "tel": "Telescope", "ngs": "Source"},
      "outputs": {"s_ref": "(n_meas,) float64"},
      "assumptions": "Reference slopes for null wavefront"
    },
    {
      "step_order": "P9",
      "step_name": "Compute KL Modal Basis",
      "formula": "\\mathbf{M}_{2C} = \\text{compute\\_KL\\_basis}(\\text{tel}, \\text{atm}, \\text{dm})[:, :n_{\\text{modes}}]",
      "inputs": {"tel": "Telescope", "atm": "Atmosphere", "dm": "DM", "n_modes": "20"},
      "outputs": {"M2C": "(n_act, n_modes) = (441, 20) float64"},
      "assumptions": "KL modes optimized for given atmosphere statistics"
    },
    {
      "step_order": "P10",
      "step_name": "Build Interaction Matrix via Push-Pull",
      "formula": "\\text{For } i = 1, \\ldots, n_{\\text{modes}}: \\quad \\mathbf{IM}[:, i] = \\frac{\\mathbf{s}_{+i} - \\mathbf{s}_{-i}}{2\\sigma_{\\text{stroke}}}",
      "inputs": {"M2C": "(441, 20)", "stroke": "1e-8", "wfs": "WFS", "dm": "DM"},
      "outputs": {"IM": "(n_meas, n_modes) float64"},
      "assumptions": "Linear response assumed for small stroke"
    },
    {
      "step_order": "P10.1",
      "step_name": "Push Measurement (per mode i)",
      "formula": "\\mathbf{u}_{\\text{dm}} = \\mathbf{M}_{2C}[:, i] \\cdot \\sigma_{\\text{stroke}}, \\quad \\mathbf{s}_{+i} = \\mathcal{W}(\\phi_{\\text{dm}})",
      "inputs": {"M2C[:, i]": "(n_act,)", "stroke": "1e-8"},
      "outputs": {"s_push": "(n_meas,)"},
      "assumptions": "DM applies positive stroke"
    },
    {
      "step_order": "P10.2",
      "step_name": "Pull Measurement (per mode i)",
      "formula": "\\mathbf{u}_{\\text{dm}} = -\\mathbf{M}_{2C}[:, i] \\cdot \\sigma_{\\text{stroke}}, \\quad \\mathbf{s}_{-i} = \\mathcal{W}(\\phi_{\\text{dm}})",
      "inputs": {"M2C[:, i]": "(n_act,)", "stroke": "1e-8"},
      "outputs": {"s_pull": "(n_meas,)"},
      "assumptions": "DM applies negative stroke"
    },
    {
      "step_order": "P11",
      "step_name": "SVD of Interaction Matrix",
      "formula": "\\mathbf{IM} = \\mathbf{U} \\boldsymbol{\\Sigma} \\mathbf{V}^T, \\quad \\mathbf{U} \\in \\mathbb{R}^{n_{\\text{meas}} \\times n_{\\text{modes}}}, \\quad \\boldsymbol{\\Sigma} = \\text{diag}(\\sigma_1, \\ldots, \\sigma_{n_{\\text{modes}}})",
      "inputs": {"IM": "(n_meas, n_modes)"},
      "outputs": {"U": "(n_meas, n_modes)", "s": "(n_modes,)", "Vt": "(n_modes, n_modes)"},
      "assumptions": "Reduced SVD (full_matrices=False)"
    },
    {
      "step_order": "P12",
      "step_name": "Compute Truncated Pseudoinverse",
      "formula": "\\sigma_i^{+} = \\begin{cases} 1/\\sigma_i & \\text{if } \\sigma_i > \\tau \\\\ 0 & \\text{otherwise} \\end{cases}, \\quad \\mathbf{R}_{\\text{modal}} = \\mathbf{V} \\text{diag}(\\boldsymbol{\\sigma}^{+}) \\mathbf{U}^T",
      "inputs": {"U": "(n_meas, n_modes)", "s": "(n_modes,)", "Vt": "(n_modes, n_modes)", "tau": "1e-3"},
      "outputs": {"R_modal": "(n_modes, n_meas) float64"},
      "assumptions": "Truncation regularizes ill-conditioned modes"
    },
    {
      "step_order": "P13",
      "step_name": "Convert Modal to Zonal Reconstructor",
      "formula": "\\mathbf{R} = \\mathbf{M}_{2C} \\mathbf{R}_{\\text{modal}}",
      "inputs": {"M2C": "(n_act, n_modes)", "R_modal": "(n_modes, n_meas)"},
      "outputs": {"R": "(n_act, n_meas) float64"},
      "assumptions": "Final reconstructor maps slopes directly to actuator commands"
    },
    {
      "step_order": "P14",
      "step_name": "Build Centroid Coordinate Grids",
      "formula": "x_j = j - \\lfloor n_{\\text{pix}}/2 \\rfloor \\text{ for } j=0,\\ldots,n_{\\text{pix}}-1, \\quad \\mathbf{X}, \\mathbf{Y} = \\text{meshgrid}(\\mathbf{x}, \\mathbf{x})",
      "inputs": {"n_pix": "6"},
      "outputs": {"X": "(6, 6) float64", "Y": "(6, 6) float64"},
      "assumptions": "Zero-centered coordinate system for CoG"
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
      "variable_name": "$\\mathbf{u}^{(0)}$",
      "initial_value": "$\\mathbf{0} \\in \\mathbb{R}^{n_{\\text{act}}}$",
      "shape": "(441,)",
      "dtype": "float64",
      "source": "Flat DM initialization (dm.coefs[:] = 0)"
    },
    {
      "step_order": 2,
      "variable_name": "$\\text{strehl\\_history}$",
      "initial_value": "Empty list []",
      "shape": "dynamic",
      "dtype": "list of float64",
      "source": "Tracking array for convergence"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "loop_structure": "for k = 0, 1, ..., K_iter - 1",
  "steps": [
    {
      "marker": "BEGIN_LOOP k = 0 to K_iter - 1"
    },
    {
      "step_order": 3,
      "step_name": "Update Atmosphere Phase Screen",
      "formula": "$\\phi_{\\text{atm}}^{(k)} = \\text{atm.update}()$",
      "inputs": {"atm": "Atmosphere object"},
      "outputs": {"phi_atm^{(k)}": "(n_res, n_res) = (120, 120) float64"},
      "computational_note": "Translates frozen-flow turbulence layer by wind velocity × timestep"
    },
    {
      "step_order": 4,
      "step_name": "Forward Propagation to WFS [FORWARD]",
      "formula": "$\\phi_{\\text{total}}^{(k)} = \\phi_{\\text{atm}}^{(k)} - \\mathbf{B} \\mathbf{u}^{(k)}$, where $\\mathbf{B}$ is the DM influence matrix",
      "inputs": {"phi_atm^{(k)}": "(120, 120)", "u^{(k)}": "(441,)", "dm": "DM object"},
      "outputs": {"phi_total^{(k)}": "(120, 120) float64 on telescope pupil"},
      "computational_note": "atm * ngs * tel * dm propagates through optical train"
    },
    {
      "step_order": 5,
      "step_name": "Extract Lenslet Electric Fields",
      "formula": "$\\mathbf{E}_{\\text{cube}}^{(k)} = \\text{wfs.get\\_lenslet\\_em\\_field}(\\phi_{\\text{total}}^{(k)})$",
      "inputs": {"phi_total^{(k)}": "(120, 120)"},
      "outputs": {"E_cube^{(k)}": "(n_sub^2, n_pix, n_pix) = (400, 6, 6) complex128"},
      "computational_note": "Segments pupil into subaperture electric fields"
    },
    {
      "step_order": 6,
      "step_name": "FFT for Spot Formation [FORWARD]",
      "formula": "$\\tilde{\\mathbf{E}}_{\\text{cube}}^{(k)}[i, :, :] = \\text{FFT}_2(\\mathbf{E}_{\\text{cube}}^{(k)}[i, :, :])$ for $i = 0, \\ldots, n_{\\text{sub}}^2 - 1$",
      "inputs": {"E_cube^{(k)}": "(400, 6, 6) complex128"},
      "outputs": {"E_tilde_cube^{(k)}": "(400, 6, 6) complex128"},
      "computational_note": "np.fft.fft2 with axes=[1,2] performs 2D FFT on each subaperture"
    },
    {
      "step_order": 7,
      "step_name": "Compute Spot Intensities",
      "formula": "$\\mathbf{I}_{\\text{spots}}^{(k)} = |\\tilde{\\mathbf{E}}_{\\text{cube}}^{(k)}|^2$",
      "inputs": {"E_tilde_cube^{(k)}": "(400, 6, 6) complex128"},
      "outputs": {"I_spots^{(k)}": "(400, 6, 6) float64"},
      "computational_note": "Element-wise squared magnitude"
    },
    {
      "step_order": 8,
      "step_name": "Center of Gravity Centroiding",
      "formula": "For each valid subaperture $i$: $c_x^{(i)} = \\frac{\\sum_{p,q} I^{(k)}_{i,p,q} X_{p,q}}{\\sum_{p,q} I^{(k)}_{i,p,q}}, \\quad c_y^{(i)} = \\frac{\\sum_{p,q} I^{(k)}_{i,p,q} Y_{p,q}}{\\sum_{p,q} I^{(k)}_{i,p,q}}$",
      "inputs": {"I_spots^{(k)}": "(400, 6, 6)", "X": "(6, 6)", "Y": "(6, 6)", "v_valid": "(400,) bool"},
      "outputs": {"slopes_raw": "(n_valid, 2) float64"},
      "computational_note": "Loop over valid subapertures; skip if flux ≤ 0"
    },
    {
      "step_order": 9,
      "step_name": "Flatten Slopes Vector",
      "formula": "$\\mathbf{s}_{\\text{meas}}^{(k)} = [c_x^{(0)}, \\ldots, c_x^{(n_{\\text{valid}}-1)}, c_y^{(0)}, \\ldots, c_y^{(n_{\\text{valid}}-1)}]^T$",
      "inputs": {"slopes_raw": "(n_valid, 2)"},
      "outputs": {"s_meas^{(k)}": "(n_meas,) = (2×n_valid,) float64"},
      "computational_note": "np.concatenate((slopes[:,0], slopes[:,1]))"
    },
    {
      "step_order": 10,
      "step_name": "Compute Residual Slopes",
      "formula": "$\\Delta\\mathbf{s}^{(k)} = \\mathbf{s}_{\\text{meas}}^{(k)} - \\mathbf{s}_{\\text{ref}}$",
      "inputs": {"s_meas^{(k)}": "(n_meas,)", "s_ref": "(n_meas,)"},
      "outputs": {"Delta_s^{(k)}": "(n_meas,) float64"},
      "computational_note": "Element-wise subtraction"
    },
    {
      "step_order": 11,
      "step_name": "Compute Command Update",
      "formula": "$\\Delta\\mathbf{u}^{(k)} = \\mathbf{R} \\Delta\\mathbf{s}^{(k)}$",
      "inputs": {"R": "(n_act, n_meas)", "Delta_s^{(k)}": "(n_meas,)"},
      "outputs": {"Delta_u^{(k)}": "(n_act,) = (441,) float64"},
      "computational_note": "np.matmul(reconstructor, slopes_meas)"
    },
    {
      "step_order": 12,
      "step_name": "Apply Integral Control Law",
      "formula": "$\\mathbf{u}^{(k+1)} = \\mathbf{u}^{(k)} - g \\cdot \\Delta\\mathbf{u}^{(k)}$",
      "inputs": {"u^{(k)}": "(441,)", "Delta_u^{(k)}": "(441,)", "g": "0.4"},
      "outputs": {"u^{(k+1)}": "(441,) float64"},
      "computational_note": "dm.coefs = dm.coefs - gain * delta_command"
    },
    {
      "step_order": 13,
      "step_name": "Science Path Propagation [FORWARD]",
      "formula": "$\\mathbf{PSF}^{(k)} = |\\mathcal{F}\\{P \\cdot e^{j\\phi_{\\text{total}}^{(k)}}\\}|^2$",
      "inputs": {"atm": "Atmosphere", "ngs": "Source", "tel": "Telescope", "dm": "DM with u^{(k+1)}", "sci_cam": "Detector"},
      "outputs": {"PSF^{(k)}": "(240, 240) float64"},
      "computational_note": "atm * ngs * tel * dm * sci_cam; sci_cam.frame contains PSF"
    },
    {
      "step_order": 14,
      "step_name": "Compute OTF of Current PSF",
      "formula": "$\\mathbf{OTF}^{(k)} = |\\text{fftshift}(\\text{FFT}_2(\\mathbf{PSF}^{(k)}))|$",
      "inputs": {"PSF^{(k)}": "(240, 240)"},
      "outputs": {"OTF^{(k)}": "(240, 240) float64"},
      "computational_note": "np.abs(np.fft.fftshift(np.fft.fft2(psf)))"
    },
    {
      "step_order": 15,
      "step_name": "Compute OTF of Reference PSF",
      "formula": "$\\mathbf{OTF}_{\\text{ref}} = |\\text{fftshift}(\\text{FFT}_2(\\mathbf{PSF}_{\\text{ref}}))|$",
      "inputs": {"PSF_ref": "(240, 240)"},
      "outputs": {"OTF_ref": "(240, 240) float64"},
      "computational_note": "Computed each iteration (could be cached)"
    },
    {
      "step_order": 16,
      "step_name": "Compute Strehl Ratio",
      "formula": "$\\text{SR}^{(k)} = 100 \\times \\frac{\\sum_{p,q} \\mathbf{OTF}^{(k)}_{p,q}}{\\sum_{p,q} (\\mathbf{OTF}_{\\text{ref}})_{p,q}}$",
      "inputs": {"OTF^{(k)}": "(240, 240)", "OTF_ref": "(240, 240)"},
      "outputs": {"SR^{(k)}": "scalar float64 (percentage)"},
      "computational_note": "OTF-based Strehl approximation"
    },
    {
      "step_order": 17,
      "step_name": "Record Strehl to History",