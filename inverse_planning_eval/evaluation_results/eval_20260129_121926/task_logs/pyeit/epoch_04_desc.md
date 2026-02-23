## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: JAC-EIT (Jacobian-based Electrical Impedance Tomography Reconstruction)
- **algorithm_family**: Linear Inverse Problem with Tikhonov Regularization
- **summary**: One-step linearized reconstruction of conductivity changes from boundary voltage differences using a regularized pseudo-inverse of the Jacobian matrix computed via finite element discretization.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\Delta\sigma} \frac{1}{2}\|J\Delta\sigma - \Delta v\|_2^2 + \frac{\lambda}{2}\Delta\sigma^T R \Delta\sigma$$

where:
- $\Delta\sigma \in \mathbb{R}^{N_e}$ is the conductivity change vector (element-wise)
- $\Delta v = v_1 - v_0 \in \mathbb{R}^{N_m}$ is the voltage difference
- $J \in \mathbb{R}^{N_m \times N_e}$ is the Jacobian (sensitivity matrix)
- $R \in \mathbb{R}^{N_e \times N_e}$ is the regularization matrix
- $\lambda > 0$ is the regularization parameter

### data_fidelity_term
$$\mathcal{D}(\Delta\sigma) = \frac{1}{2}\|J\Delta\sigma - \Delta v\|_2^2 = \frac{1}{2}\sum_{m=1}^{N_m}\left(\sum_{e=1}^{N_e} J_{me}\Delta\sigma_e - \Delta v_m\right)^2$$

### regularization_term
$$\mathcal{R}(\Delta\sigma) = \frac{\lambda}{2}\Delta\sigma^T R \Delta\sigma$$

where $R$ depends on method parameter:
- **Kotre** ($\texttt{method}=\text{"kotre"}$): $R = \text{diag}\left((J^TJ)_{ii}^p\right)$ with $p=0.50$
- **Levenberg-Marquardt** ($\texttt{method}=\text{"lm"}$): $R = \text{diag}\left((J^TJ)_{ii}\right)$
- **Damped Gauss-Newton** ($\texttt{method}=\text{"dgn"}$): $R = I_{N_e}$

### constraints
None (unconstrained linear solve)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Delta\sigma$ (`ds`) | $(N_e,)$ | `float64` | Computed directly | Reconstructed conductivity change per element |

### 2.2 Dual Variables

None (closed-form solution, no dual variables)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $K_g$ (`kg`) | $(N_n, N_n)$ sparse | `float64` | Assembled from $K^{(e)}$ | Global stiffness matrix |
| $\phi$ (`f`) | $(N_{ex}, N_n)$ | `float64` | Solved from $K_g\phi=b$ | Potential field for each excitation |
| $J$ (`jac`) | $(N_m, N_e)$ | `float64` | `np.zeros((n_meas, n_elems))` | Jacobian/sensitivity matrix |
| $H$ | $(N_e, N_m)$ | `float64` | Computed | Reconstruction matrix (regularized pseudo-inverse) |
| $R_{\text{mat}}$ (`r_mat`) | $(N_{el}, N_n)$ | `float64` | From $K_g^{-1}$ | Rows of inverse stiffness at electrode positions |
| $R_i$ (`ri`) | $(N_m, 4)$ | `float64` | Differenced from `r_el` | Measurement sensitivity rows |
| $J^TJ$ (`j_w_j`) | $(N_e, N_e)$ | `float64` | Computed | Gram matrix of Jacobian |
| $R$ (`r_mat_reg`) | $(N_e, N_e)$ | `float64` | Method-dependent | Regularization matrix |
| $\Delta v$ (`dv`) | $(N_m,)$ | `float64` | $v_1 - v_0$ or normalized | Measurement difference |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{el}$ (`n_el`) | 16 | Number of electrodes |
| $h_0$ | 0.2 | Mesh element size parameter |
| `bbox` | $[[-1,-1,-1],[1,1,1]]$ | Bounding box for mesh |
| `dist_exc` | 7 | Excitation electrode separation |
| `step_meas` | 1 | Measurement electrode step |
| `anomaly_center` | $[0.4, 0.4, 0.0]$ | Anomaly location |
| `anomaly_r` | 0.3 | Anomaly radius |
| $\sigma_{\text{anom}}$ | 100.0 | Anomaly permittivity |
| $\sigma_{\text{bg}}$ | 1.0 | Background permittivity |
| $p$ | 0.50 | Kotre regularization exponent |
| $\lambda$ (`lamb`) | $10^{-3}$ | Regularization parameter |
| `method` | "kotre" | Regularization method selection |
| `normalize` | False | Voltage normalization flag |
| `ref_node` | From mesh | Reference node for Dirichlet BC |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $v_0$ | $(N_m,)$ | `float64` | Baseline boundary voltage measurements (homogeneous) |
| $v_1$ | $(N_m,)$ | `float64` | Perturbed boundary voltage measurements (with anomaly) |

---

## 3. PREPROCESSING PIPELINE

**Step 0: Mesh Generation**
- **step_order**: 0
- **step_name**: Generate 3D tetrahedral mesh
- **formula**: Call external `mesh.create(n_el, h0, bbox, fd=ball)` producing:
  - $\mathbf{p} \in \mathbb{R}^{N_n \times 3}$ (node coordinates)
  - $\mathbf{T} \in \mathbb{Z}^{N_e \times 4}$ (element connectivity)
  - $\mathbf{e} \in \mathbb{Z}^{N_{el}}$ (electrode node indices)
- **inputs**: $N_{el}=16$, $h_0=0.2$, `bbox`, `fd=ball`
- **outputs**: `mesh_obj` containing `pts` $(N_n,3)$, `tri` $(N_e,4)$, `el_pos` $(N_{el},)$

**Step 1: Build Excitation Pattern**
- **step_order**: 1
- **step_name**: Adjacent excitation pattern
- **formula**: 
$$E_{i,:} = [i, \; (i+d) \mod N_{el}], \quad i=0,\ldots,N_{el}-1$$
where $d=7$ (dist_exc)
- **inputs**: $N_{el}=16$, $d=7$
- **outputs**: `ex_mat` $(N_{el}, 2)$

**Step 2: Build Measurement Pattern**
- **step_order**: 2
- **step_name**: Differential measurement pattern
- **formula**: For each excitation $(a,b)$, form electrode pairs $(m,n)$ where $m \in \{0,\ldots,N_{el}-1\}$, $n=(m+1)\mod N_{el}$, excluding pairs where $m,n \in \{a,b\}$
- **inputs**: `ex_mat`, $N_{el}$, `step=1`
- **outputs**: `meas_mat` $(N_m, 3)$ storing $(n, m, \text{exc\_idx})$, `keep_ba` mask

**Step 3: Compute Local Stiffness Matrices**
- **step_order**: 3
- **step_name**: Element stiffness computation
- **formula**: For each tetrahedron $e$ with vertices $\mathbf{x}_0,\mathbf{x}_1,\mathbf{x}_2,\mathbf{x}_3$:
$$\mathbf{s}_k = \mathbf{x}_{(k+2)\mod 4} - \mathbf{x}_{(k+1)\mod 4}$$
$$V_e = \frac{1}{6}\det([\mathbf{s}_0, \mathbf{s}_1, \mathbf{s}_2])$$
$$\mathbf{a}_k = \text{sign}_k \cdot (\mathbf{s}_{i_k} \times \mathbf{s}_{j_k})$$
where $(i,j,\text{sign}) \in \{(0,1,+1),(1,2,-1),(2,3,+1),(3,0,-1)\}$
$$K^{(e)}_{ij} = \frac{\mathbf{a}_i \cdot \mathbf{a}_j}{36 V_e}$$
- **inputs**: `pts` $(N_n, 3)$, `tri` $(N_e, 4)$
- **outputs**: `se` $(N_e, 4, 4)$

**Step 4: Baseline Forward Solution**
- **step_order**: 4
- **step_name**: Compute baseline measurements
- **formula**: 
  1. Assemble: $K_g = \sum_e \sigma_e^{(0)} K^{(e)}$ with Dirichlet BC at ref node
  2. Solve: $K_g \phi_i = b_i$ for each excitation $i$
  3. Measure: $v_0^{(i,m)} = \phi_i[e_n^{(m)}] - \phi_i[e_m^{(m)}]$
- **inputs**: `se`, `tri`, $\sigma^{(0)}=\mathbf{1}$, `ex_mat`, `meas_mat`, `el_pos`
- **outputs**: `v0` $(N_m,)$, `f_baseline` $(N_{ex}, N_n)$, `kg_baseline`

**Step 5: Perturbed Forward Solution**
- **step_order**: 5
- **step_name**: Compute perturbed measurements with anomaly
- **formula**: Same as Step 4 but with $\sigma^{(1)}$ where elements inside anomaly sphere have $\sigma=100$
- **inputs**: `se`, `tri`, $\sigma^{(1)}$, `ex_mat`, `meas_mat`, `el_pos`
- **outputs**: `v1` $(N_m,)$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Extract mesh data | $N_n \leftarrow$ `mesh_obj.n_nodes`, $N_e \leftarrow$ `mesh_obj.n_elems` | `mesh_obj` | Dimensions |
| I.2 | Set baseline permittivity | $\sigma = \sigma_{\text{baseline}} \in \mathbb{R}^{N_e}$ | `perm_baseline` | $\sigma$ |
| I.3 | Assemble global stiffness | $K_g = \texttt{assemble}(K^{(e)}, \mathbf{T}, \sigma, N_n, \text{ref})$ | `se`, `tri`, $\sigma$ | $K_g \in \mathbb{R}^{N_n \times N_n}$ sparse |
| I.4 | Build excitation RHS | $b_{i,j} = \begin{cases}+1 & j = e_{E_{i,0}} \\ -1 & j = e_{E_{i,1}} \\ 0 & \text{otherwise}\end{cases}$ | `ex_mat`, `el_pos` | $b \in \mathbb{R}^{N_{ex} \times N_n}$ |
| I.5 | Solve forward problems | $\phi_i = K_g^{-1} b_i, \quad i=1,\ldots,N_{ex}$ | $K_g$, $b$ | $\phi \in \mathbb{R}^{N_{ex} \times N_n}$ |
| I.6 | Compute inverse at electrodes | $R_{\text{mat}} = (K_g^{-1})[\mathbf{e},:]$ | $K_g$, `el_pos` | $R_{\text{mat}} \in \mathbb{R}^{N_{el} \times N_n}$ |
| I.7 | Initialize Jacobian | $J \leftarrow \mathbf{0}^{N_m \times N_e}$ | Dimensions | $J$ |

### 4.2 ITERATION LOOP

**Note**: This algorithm does NOT iterate. The reconstruction is a single-step linear solve. The Jacobian computation involves a loop over elements but is not an optimization iteration.

**BEGIN_LOOP** (Element loop for Jacobian assembly, $e = 0, \ldots, N_e-1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| J.1 | Compute measurement sensitivity | $R_i = R_{\text{mat}}[\mathbf{n},:] - R_{\text{mat}}[\mathbf{m},:]$ | `r_mat`, `meas_mat` | $R_i \in \mathbb{R}^{N_m \times N_n}$ | Difference per measurement |
| J.2 | Extract excitation potentials | $\phi_{\text{meas}} = \phi[\text{exc\_idx},:]$ | $\phi$, `meas_mat[:,2]` | $\phi_{\text{meas}} \in \mathbb{R}^{N_m \times N_n}$ | Index by measurement's excitation |
| J.3 | Compute Jacobian column | $J_{:,e} = \sum_{k \in \text{ijk}} \left( R_i[:,\text{ijk}] \cdot K^{(e)} \right) \odot \phi_{\text{meas}}[:,\text{ijk}]$ summed over local nodes | $R_i$, $K^{(e)}$, $\phi_{\text{meas}}$, `tri[e]` | $J_{:,e} \in \mathbb{R}^{N_m}$ | Element sensitivity |

**END_LOOP**

**Regularized Reconstruction (Non-iterative):**

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| R.1 | Compute Gram matrix | $G = J^T J$ | $J$ | $G \in \mathbb{R}^{N_e \times N_e}$ | |
| R.2 | Build regularization matrix | **Kotre**: $R = \text{diag}(G_{ii}^p)$, $p=0.50$ | $G$, $p$, `method` | $R \in \mathbb{R}^{N_e \times N_e}$ | Diagonal scaling |
| R.3 | Form reconstruction matrix | $H = (G + \lambda R)^{-1} J^T$ | $G$, $R$, $\lambda=10^{-3}$, $J$ | $H \in \mathbb{R}^{N_e \times N_m}$ | Pseudo-inverse |
| R.4 | Compute data difference | $\Delta v = v_1 - v_0$ (if `normalize=False`) | $v_0$, $v_1$ | $\Delta v \in \mathbb{R}^{N_m}$ | Alt: $\Delta v = \log|v_1/v_0| \cdot \text{sign}(v_0)$ |
| R.5 | Reconstruct | $\Delta\sigma = -H \cdot \Delta v$ | $H$, $\Delta v$ | $\Delta\sigma \in \mathbb{R}^{N_e}$ | Sign convention |

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Extract real part | $\Delta\sigma_{\text{real}} = \text{Re}(\Delta\sigma)$ | $\Delta\sigma$ | `ds_real` |
| F.2 | Interpolate to nodes | $(\Delta\sigma_{\text{node}})_j = \frac{\sum_{e: j \in e} \Delta\sigma_e}{\text{count}_j}$ | $\Delta\sigma$, `pts`, `tri` | `node_ds` $(N_n,)$ |
| F.3 | Generate visualization | 3D scatter plot colored by `node_ds` | `pts`, `node_ds` | PNG file |

---

## 5. STOPPING CRITERION

- **type**: N/A (non-iterative, single-step closed-form solution)
- **expression**: The algorithm computes $\Delta\sigma = -H \cdot \Delta v$ directly without iteration
- **parameters**: None
- **check_frequency**: N/A

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| `ds_min` | $\min_e \text{Re}(\Delta\sigma_e)$ | Minimum conductivity change |
| `ds_max` | $\max_e \text{Re}(\Delta\sigma_e)$ | Maximum conductivity change |
| `ds_mean` | $\frac{1}{N_e}\sum_e \text{Re}(\Delta\sigma_e)$ | Mean conductivity change |
| `ds_std` | $\sqrt{\frac{1}{N_e}\sum_e (\text{Re}(\Delta\sigma_e) - \bar{\Delta\sigma})^2}$ | Standard deviation |
| `data_residual` | $\|\Delta v\|_2 = \|v_1 - v_0\|_2$ | Norm of voltage difference |
| `relative_change` | $\frac{\|v_1 - v_0\|_2}{\|v_0\|_2}$ | Relative voltage perturbation |
| `max_change_element` | $\arg\max_e |\text{Re}(\Delta\sigma_e)|$ | Element index with largest change |
| `max_change_value` | $\text{Re}(\Delta\sigma_{\text{argmax}})$ | Value at max-change element |
| Forward consistency | $\|F(\sigma_{\text{baseline}}) - v_0\|_2$ | Forward model validation (should be $\approx 0$) |

---

## 7. REQUIRED OUTPUT FIELDS

```
algorithm_identification: {algorithm_name, algorithm_family, summary}
objective_function: {full_expression, data_fidelity_term, regularization_term, constraints}
variables: {primal, dual, auxiliary, constants, observations}
data_preprocessing: [{step_order, step_name, formula, inputs, outputs}, ...]
initialization: [{step, name, formula, inputs, outputs}, ...]
iteration_loop: null  # Non-iterative algorithm
finalization: [{step, name, formula, inputs, outputs}, ...]
stopping_criterion: {type: "non_iterative", expression: null}
evaluation: [{name, formula, description}, ...]
forward_model: {assembly_formula, solve_formula, measurement_formula}
jacobian_computation: {element_sensitivity_formula, assembly_method}
regularization: {method, matrix_formula, parameter_lambda, parameter_p}
reconstruction_formula: "ds = -(J^T J + lambda * R)^{-1} J^T @ dv"
```