## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "JAC (Jacobian-based Linear Reconstruction)",
  "algorithm_family": "Linear Inverse Problems / Tikhonov Regularization",
  "summary": "Single-step linearized reconstruction for 3D Electrical Impedance Tomography using Jacobian-based inversion with Tikhonov-type regularization to recover conductivity changes from boundary voltage differences."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\Delta\\sigma} \\frac{1}{2}\\|J\\Delta\\sigma - \\Delta v\\|_2^2 + \\frac{\\lambda}{2}\\Delta\\sigma^T R \\Delta\\sigma",
  "data_fidelity_term": "\\frac{1}{2}\\|J\\Delta\\sigma - \\Delta v\\|_2^2 where $J \\in \\mathbb{R}^{M \\times N_e}$ is the Jacobian, $\\Delta\\sigma \\in \\mathbb{R}^{N_e}$ is conductivity change, $\\Delta v = v_1 - v_0 \\in \\mathbb{R}^M$ is voltage difference",
  "regularization_term": "\\frac{\\lambda}{2}\\Delta\\sigma^T R \\Delta\\sigma where $R$ depends on method: Kotre: $R = \\text{diag}((J^T J)_{ii}^p)$; LM: $R = \\text{diag}((J^T J)_{ii})$; DGN: $R = I$",
  "constraints": "None (unconstrained quadratic problem)"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Delta\sigma$ | $(N_e,)$ | float64 | Computed via closed-form solution | Reconstructed element-wise conductivity change |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| — | — | — | — | None (closed-form solution, no dual formulation) |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $K_g$ | $(N_n, N_n)$ | sparse CSR | Assembled from $K_e$, $\sigma$, tri | Global stiffness matrix |
| $\mathbf{b}$ | $(N_{ex}, N_n)$ | float64 | Current injection pattern | Excitation right-hand-side matrix |
| $\mathbf{f}$ | $(N_{ex}, N_n)$ | float64 | Solved from $K_g \mathbf{f}^T = \mathbf{b}^T$ | Nodal potential solutions for each excitation |
| $R_{mat}$ | $(N_{el}, N_n)$ | float64 | $K_g^{-1}[\text{el\_pos}, :]$ | Electrode rows of inverse stiffness |
| $R_{el}$ | $(N_{ex}, N_{el}, N_n)$ | float64 | Broadcast of $R_{mat}$ | Per-excitation electrode response |
| $r_i$ | $(M, N_{el})$ | float64 | Differenced from $R_{el}$ | Measurement-pattern differenced responses |
| $J$ | $(M, N_e)$ | float64 | Computed from $r_i$, $K_e$, $\mathbf{f}$ | Jacobian matrix |
| $J^T J$ | $(N_e, N_e)$ | float64 | $J^T J$ | Gram matrix of Jacobian |
| $R$ | $(N_e, N_e)$ | float64 | Depends on method | Regularization matrix |
| $H$ | $(N_e, M)$ | float64 | $(J^T J + \lambda R)^{-1} J^T$ | Reconstruction matrix |
| $\Delta v$ | $(M,)$ | float64 | $v_1 - v_0$ | Voltage difference vector |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{el}$ | 16 | Number of electrodes |
| $h_0$ | 0.2 | Mesh element size parameter |
| bbox | $[[-1,-1,-1],[1,1,1]]$ | Mesh bounding box |
| $d_{exc}$ | 7 | Distance between excitation electrodes |
| $s_{meas}$ | 1 | Step for measurement pattern |
| $\sigma_{bg}$ | 1.0 | Background conductivity |
| $\sigma_{anom}$ | 100.0 | Anomaly conductivity |
| $c_{anom}$ | $[0.4, 0.4, 0.0]$ | Anomaly center |
| $r_{anom}$ | 0.3 | Anomaly radius |
| $p$ | 0.50 | Kotre regularization exponent |
| $\lambda$ | $10^{-3}$ | Regularization parameter |
| method | "kotre" | Regularization method |
| normalize | False | Whether to use log-normalized difference |
| $N_n$ | mesh-dependent | Number of mesh nodes |
| $N_e$ | mesh-dependent | Number of mesh elements |
| $N_{ex}$ | 16 | Number of excitation patterns |
| $M$ | protocol-dependent | Number of measurements |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $v_0$ | $(M,)$ | float64 | Baseline boundary voltage measurements (homogeneous) |
| $v_1$ | $(M,)$ | float64 | Perturbed boundary voltage measurements (with anomaly) |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": 1,
      "step_name": "Mesh Generation",
      "formula": "\\text{mesh\\_obj} = \\text{create}(N_{el}, h_0, \\text{bbox}, \\text{ball})",
      "inputs": ["$N_{el}$: int", "$h_0$: float", "bbox: $2 \\times 3$", "fd: ball function"],
      "outputs": ["pts: $(N_n, 3)$ node coordinates", "tri: $(N_e, 4)$ tetrahedron connectivity", "el_pos: $(N_{el},)$ electrode node indices", "ref_node: int reference node index"],
      "assumptions": "3D ball domain with tetrahedral elements"
    },
    {
      "step_order": 2,
      "step_name": "Build Excitation Pattern",
      "formula": "\\text{ex\\_mat}[i] = [i, (i + d_{exc}) \\mod N_{el}] \\quad \\forall i \\in [0, N_{el})",
      "inputs": ["$N_{el}$: int", "$d_{exc}$: int"],
      "outputs": ["ex_mat: $(N_{ex}, 2)$ excitation electrode pairs"],
      "assumptions": "Adjacent excitation mode with configurable distance"
    },
    {
      "step_order": 3,
      "step_name": "Build Measurement Pattern",
      "formula": "\\text{For each excitation } k: m = [0,1,...,N_{el}-1], n = (m + s_{meas}) \\mod N_{el}; \\text{keep if } m,n \\notin \\{a_k, b_k\\}",
      "inputs": ["ex_mat: $(N_{ex}, 2)$", "$N_{el}$: int", "$s_{meas}$: int"],
      "outputs": ["meas_mat: $(M, 3)$ where columns are [n, m, excitation_index]", "keep_ba: $(N_{ex} \\times N_{el},)$ boolean mask"],
      "assumptions": "Excludes measurements at excitation electrodes"
    },
    {
      "step_order": 4,
      "step_name": "Compute Local Stiffness Matrices",
      "formula": "K_e[e] = \\frac{\\mathbf{a}\\mathbf{a}^T}{36 V_e} \\text{ where } V_e = \\frac{1}{6}\\det([s_0, s_1, s_2]), s_i = \\text{edge vectors}, \\mathbf{a}_j = \\pm(s_i \\times s_k)",
      "inputs": ["pts: $(N_n, 3)$", "tri: $(N_e, 4)$"],
      "outputs": ["$K_e$: $(N_e, 4, 4)$ local stiffness matrices"],
      "assumptions": "Tetrahedron finite elements"
    },
    {
      "step_order": 5,
      "step_name": "Compute Baseline Measurements",
      "formula": "K_g^{(0)} = \\text{assemble}(K_e, \\text{tri}, \\sigma_{bg}), \\mathbf{f}^{(0)} = (K_g^{(0)})^{-1}\\mathbf{b}, v_0 = \\text{diff}(\\mathbf{f}^{(0)}[:, \\text{el\\_pos}])",
      "inputs": ["$K_e$: $(N_e, 4, 4)$", "tri: $(N_e, 4)$", "$\\sigma_{bg}$: float"],
      "outputs": ["$v_0$: $(M,)$ baseline measurements"],
      "assumptions": "Homogeneous background conductivity"
    },
    {
      "step_order": 6,
      "step_name": "Compute Perturbed Measurements",
      "formula": "\\sigma_{anom}[e] = \\begin{cases} \\sigma_{anom} & \\text{if element } e \\text{ inside anomaly} \\\\ \\sigma_{bg} & \\text{otherwise} \\end{cases}, v_1 = \\text{forward}(\\sigma_{anom})",
      "inputs": ["$c_{anom}$: $(3,)$", "$r_{anom}$: float", "$\\sigma_{anom}$: float", "$\\sigma_{bg}$: float"],
      "outputs": ["$v_1$: $(M,)$ perturbed measurements"],
      "assumptions": "Spherical anomaly region"
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
      "variable_name": "$\\sigma$",
      "initial_value": "$\\sigma = \\sigma_{baseline}$ (copy of baseline permittivity array)",
      "shape": "$(N_e,)$",
      "dtype": "float64",
      "source": "From preprocessing (perm_baseline)"
    },
    {
      "step_order": 2,
      "variable_name": "$J$",
      "initial_value": "$J = \\mathbf{0}$",
      "shape": "$(M, N_e)$",
      "dtype": "float64",
      "source": "Zero initialization"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "steps": [
    {
      "step_order": 3,
      "step_name": "Assemble Global Stiffness Matrix",
      "formula": "$K_g = \\sum_{e=1}^{N_e} \\sigma_e \\cdot P_e^T K_e[e] P_e$ where $P_e$ is the assembly operator mapping local to global indices; Dirichlet condition at ref_node enforced by setting row/col to identity",
      "inputs": ["$K_e$: $(N_e, 4, 4)$", "tri: $(N_e, 4)$", "$\\sigma$: $(N_e,)$", "$N_n$: int", "ref_node: int"],
      "outputs": ["$K_g$: $(N_n, N_n)$ sparse CSR"],
      "computational_note": "Sparse assembly using COO format converted to CSR"
    },
    {
      "step_order": 4,
      "step_name": "Build Excitation RHS Matrix",
      "formula": "$\\mathbf{b}[k, \\text{el\\_pos}[\\text{ex\\_mat}[k,0]]] = 1, \\quad \\mathbf{b}[k, \\text{el\\_pos}[\\text{ex\\_mat}[k,1]]] = -1$",
      "inputs": ["ex_mat: $(N_{ex}, 2)$", "el_pos: $(N_{el},)$", "$N_n$: int"],
      "outputs": ["$\\mathbf{b}$: $(N_{ex}, N_n)$"],
      "computational_note": "Unit current injection at electrode pairs"
    },
    {
      "step_order": 5,
      "step_name": "BEGIN_LOOP: Solve Forward Problems",
      "formula": "for $i = 0$ to $N_{ex} - 1$",
      "inputs": [],
      "outputs": [],
      "computational_note": "Loop over excitation patterns"
    },
    {
      "step_order": 6,
      "step_name": "Solve Linear System [FORWARD]",
      "formula": "$\\mathbf{f}[i, :] = K_g^{-1} \\mathbf{b}[i, :]$",
      "inputs": ["$K_g$: $(N_n, N_n)$ sparse", "$\\mathbf{b}[i,:]$: $(N_n,)$"],
      "outputs": ["$\\mathbf{f}[i,:]$: $(N_n,)$"],
      "computational_note": "Sparse direct solve using scipy.sparse.linalg.spsolve"
    },
    {
      "step_order": 7,
      "step_name": "END_LOOP: Solve Forward Problems",
      "formula": "end for",
      "inputs": [],
      "outputs": ["$\\mathbf{f}$: $(N_{ex}, N_n)$ complete potential matrix"],
      "computational_note": ""
    },
    {
      "step_order": 8,
      "step_name": "Compute Dense Inverse at Electrodes",
      "formula": "$R_{mat} = (K_g^{-1})[\\text{el\\_pos}, :] \\in \\mathbb{R}^{N_{el} \\times N_n}$",
      "inputs": ["$K_g$: $(N_n, N_n)$ sparse"],
      "outputs": ["$R_{mat}$: $(N_{el}, N_n)$"],
      "computational_note": "Requires full dense inversion of $K_g$, then row extraction"
    },
    {
      "step_order": 9,
      "step_name": "Broadcast Electrode Response",
      "formula": "$R_{el}[k, :, :] = R_{mat} \\quad \\forall k \\in [0, N_{ex})$",
      "inputs": ["$R_{mat}$: $(N_{el}, N_n)$"],
      "outputs": ["$R_{el}$: $(N_{ex}, N_{el}, N_n)$"],
      "computational_note": "np.full broadcast"
    },
    {
      "step_order": 10,
      "step_name": "Compute Measurement-Differenced Response",
      "formula": "$r_i[m, :] = R_{el}[\\text{idx}[m], \\text{meas\\_mat}[m,0], :] - R_{el}[\\text{idx}[m], \\text{meas\\_mat}[m,1], :]$ where $\\text{idx} = \\text{meas\\_mat}[:,2]$",
      "inputs": ["$R_{el}$: $(N_{ex}, N_{el}, N_n)$", "meas_mat: $(M, 3)$"],
      "outputs": ["$r_i$: $(M, N_n)$"],
      "computational_note": "Vectorized row differencing per measurement pattern"
    },
    {
      "step_order": 11,
      "step_name": "Extract Potentials for Jacobian",
      "formula": "$\\mathbf{f}_n = \\mathbf{f}[\\text{meas\\_mat}[:,2], :]$",
      "inputs": ["$\\mathbf{f}$: $(N_{ex}, N_n)$", "meas_mat: $(M, 3)$"],
      "outputs": ["$\\mathbf{f}_n$: $(M, N_n)$"],
      "computational_note": "Index by excitation pattern for each measurement"
    },
    {
      "step_order": 12,
      "step_name": "BEGIN_LOOP: Compute Jacobian Columns",
      "formula": "for $e = 0$ to $N_e - 1$",
      "inputs": [],
      "outputs": [],
      "computational_note": "Loop over elements"
    },
    {
      "step_order": 13,
      "step_name": "Compute Jacobian Column for Element e",
      "formula": "$J[:, e] = \\sum_{\\text{axis}=1} \\left( (r_i[:, \\text{tri}[e]] \\cdot K_e[e]) \\odot \\mathbf{f}_n[:, \\text{tri}[e]] \\right)$ where $\\text{tri}[e]$ gives the 4 node indices of element $e$",
      "inputs": ["$r_i$: $(M, N_n)$", "$K_e[e]$: $(4, 4)$", "$\\mathbf{f}_n$: $(M, N_n)$", "tri: $(N_e, 4)$"],
      "outputs": ["$J[:, e]$: $(M,)$"],
      "computational_note": "For each measurement: $J_{m,e} = r_i[m, \\text{ijk}]^T K_e[e] \\mathbf{f}_n[m, \\text{ijk}]$"
    },
    {
      "step_order": 14,
      "step_name": "END_LOOP: Compute Jacobian Columns",
      "formula": "end for",
      "inputs": [],
      "outputs": ["$J$: $(M, N_e)$ complete Jacobian"],
      "computational_note": ""
    },
    {
      "step_order": 15,
      "step_name": "Compute Gram Matrix",
      "formula": "$G = J^T J$",
      "inputs": ["$J$: $(M, N_e)$"],
      "outputs": ["$G$: $(N_e, N_e)$"],
      "computational_note": "Dense matrix multiplication"
    },
    {
      "step_order": 16,
      "step_name": "Build Regularization Matrix (Kotre)",
      "formula": "$R = \\text{diag}(G_{ii}^p) = \\text{diag}((J^T J)_{ii}^{0.5})$",
      "inputs": ["$G$: $(N_e, N_e)$", "$p = 0.5$"],
      "outputs": ["$R$: $(N_e, N_e)$"],
      "computational_note": "For method='kotre'; alternative: method='lm' uses $p=1$, method='dgn' uses $R=I$"
    },
    {
      "step_order": 17,
      "step_name": "Compute Regularized Inverse",
      "formula": "$H = (J^T J + \\lambda R)^{-1} J^T = (G + \\lambda R)^{-1} J^T$",
      "inputs": ["$G$: $(N_e, N_e)$", "$R$: $(N_e, N_e)$", "$J$: $(M, N_e)$", "$\\lambda = 10^{-3}$"],
      "outputs": ["$H$: $(N_e, M)$"],
      "computational_note": "Dense matrix inversion via scipy.linalg.inv"
    },
    {
      "step_order": 18,
      "step_name": "Compute Voltage Difference",
      "formula": "$\\Delta v = v_1 - v_0$",
      "inputs": ["$v_1$: $(M,)$", "$v_0$: $(M,)$"],
      "outputs": ["$\\Delta v$: $(M,)$"],
      "computational_note": "When normalize=True: $\\Delta v = \\log(|v_1|/|v_0|) \\cdot \\text{sign}(\\Re(v_0))$"
    },
    {
      "step_order": 19,
      "step_name": "Apply Reconstruction Matrix",
      "formula": "$\\Delta\\sigma = -H \\cdot \\Delta v$",
      "inputs": ["$H$: $(N_e, M)$", "$\\Delta v$: $(M,)$"],
      "outputs": ["$\\Delta\\sigma$: $(N_e,)$"],
      "computational_note": "Negative sign accounts for Jacobian sign convention"
    }
  ]
}
```

### 4.3 FINALIZATION
```json
{
  "steps": [
    {
      "step_order": 20,
      "step_name": "Interpolate to Nodes",
      "formula": "$\\Delta\\sigma_{node}[n] = \\frac{\\sum_{e: n \\in \\text{tri}[e]} \\Re(\\Delta\\sigma[e])}{\\max(\\text{count}[n], 1)}$ where count[n] = number of elements containing node n",
      "inputs": ["$\\Delta\\sigma$: $(N_e,)$", "pts: $(N_n, 3)$", "tri: $(N_e, 4)$"],
      "outputs": ["$\\Delta\\sigma_{node}$: $(N_n,)$"],
      "computational_note": "Averaging element values to nodes for visualization"
    },
    {
      "step_order": 21,
      "step_name": "Return Reconstruction",
      "formula": "output $\\leftarrow \\Delta\\sigma$",
      "inputs": ["$\\Delta\\sigma$: $(N_e,)$"],
      "outputs": ["$\\Delta\\sigma$: $(N_e,)$ element-wise conductivity change"],
      "computational_note": "Final reconstructed conductivity perturbation"
    }
  ]
}
```

## 5. STOPPING CRITERION
```json
{
  "type": "single_step",
  "expression": "N/A (non-iterative algorithm)",
  "parameters": {
    "description": "This is a single-step linear reconstruction method. The closed-form solution is computed directly without iteration."
  },
  "check_frequency": "N/A"
}
```

## 6. EVALUATION METRICS
```json
{
  "metrics": [
    {
      "name": "Minimum Conductivity Change",
      "formula": "$\\text{ds\\_min} = \\min_e \\Re(\\Delta\\sigma[e])$",
      "reference_data": "N/A",
      "predicted_data": "$\\Delta\\sigma$",
      "description": "Minimum reconstructed conductivity change across all elements"
    },
    {
      "name": "Maximum Conductivity Change",
      "formula": "$\\text{ds\\_max} = \\max_e \\Re(\\Delta\\sigma[e])$",
      "reference_data": "N/A",
      "predicted_data": "$\\Delta\\sigma$",
      "description": "Maximum reconstructed conductivity change across all elements"
    },
    {
      "name": "Mean Conductivity Change",
      "formula": "$\\text{ds\\_mean} = \\frac{1}{N_e}\\sum_{e=1}^{N_e} \\Re(\\Delta\\sigma[e])$",
      "reference_data": "N/A",
      "predicted_data": "$\\Delta\\sigma$",
      "description": "Average reconstructed conductivity change"
    },
    {
      "name": "Standard Deviation of Conductivity Change",
      "formula": "$\\text{ds\\_std} = \\sqrt{\\frac{1}{N_e}\\sum_{e=1}^{N_e}(\\Re(\\Delta\\sigma[e]) - \\text{ds\\_mean})^2}$",
      "reference_data": "N/A",
      "predicted_data": "$\\Delta\\sigma$",
      "description": "Spread of reconstructed conductivity values"
    },
    {
      "name": "Data Residual Norm",
      "formula": "$\\|\\Delta v\\|_2 = \\sqrt{\\sum_{m=1}^{M}(v_1[m] - v_0[m])^2}$",
      "reference_data": "$v_0$",
      "predicted_data": "$v_1$",
      "description": "L2 norm of voltage difference"
    },
    {
      "name": "Relative Voltage Change",
      "formula": "$\\frac{\\|v_1 - v_0\\|_2}{\\|v_0\\|_2}$",
      "reference_data": "$v_0$",
      "predicted_data": "$v_1$",
      "description": "Normalized measure of measurement perturbation"
    },
    {
      "name": "Element with Maximum Absolute Change",
      "formula": "$e^* = \\arg\\max_e |\\Re(\\Delta\\sigma[e])|$",
      "reference_data": "N/A",
      "predicted_data": "$\\Delta\\sigma$",
      "description": "Index of element with largest absolute conductivity change"
    },
    {
      "name": "Forward Operator Consistency",
      "formula": "$\\|\\mathcal{F}(\\sigma_{baseline}) - v_0\\|_2$",
      "reference_data": "$v_0$",
      "predicted_data": "$\\mathcal{F}(\\sigma_{baseline})$",
      "description": "Verification that forward operator reproduces baseline measurements"
    }
  ]
}
```