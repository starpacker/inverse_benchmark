## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: JAC (Jacobian-based) EIT Inversion
- **algorithm_family**: Linearized inverse problems / Tikhonov-regularized least squares
- **summary**: Reconstructs 3D conductivity changes from boundary voltage measurements using a one-step Jacobian-based inversion with Gauss-Newton regularization.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\Delta\sigma} = \arg\min_{\Delta\sigma} \left\| \mathbf{J} \Delta\sigma - \Delta\mathbf{v} \right\|_2^2 + \lambda \left\| \mathbf{R}^{1/2} \Delta\sigma \right\|_2^2$$

Closed-form solution:
$$\hat{\Delta\sigma} = -(\mathbf{J}^\top \mathbf{J} + \lambda \mathbf{R})^{-1} \mathbf{J}^\top \Delta\mathbf{v}$$

### data_fidelity_term
$$\mathcal{D}(\Delta\sigma) = \left\| \mathbf{J} \Delta\sigma - \Delta\mathbf{v} \right\|_2^2$$

where $\Delta\mathbf{v} = \mathbf{v}_1 - \mathbf{v}_0$ (unnormalized) or $\Delta\mathbf{v} = \log\left|\frac{\mathbf{v}_1}{\mathbf{v}_0}\right| \cdot \text{sign}(\text{Re}(\mathbf{v}_0))$ (normalized)

### regularization_term
$$\mathcal{R}(\Delta\sigma) = \lambda \cdot \Delta\sigma^\top \mathbf{R} \, \Delta\sigma$$

where $\mathbf{R}$ depends on method:
- **Kotre**: $\mathbf{R} = \text{diag}\left((\text{diag}(\mathbf{J}^\top\mathbf{J}))^p\right)$
- **Levenberg-Marquardt (lm)**: $\mathbf{R} = \text{diag}(\text{diag}(\mathbf{J}^\top\mathbf{J}))$
- **Damped Gauss-Newton (dgn)**: $\mathbf{R} = \mathbf{I}_{n_\text{elems}}$

### constraints
None

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\Delta\sigma$ (`ds`) | primal | $(n_\text{elems},)$ | float64 | Computed via closed-form solution | Reconstructed element-wise conductivity change |

### Dual Variables
None (closed-form solution, no iterative dual updates)

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{J}$ (`jac`) | auxiliary | $(n_\text{meas}, n_\text{elems})$ | float64 | Zeros, then computed | Jacobian/sensitivity matrix |
| $\mathbf{K}_g$ (`kg`) | auxiliary | $(n_\text{nodes}, n_\text{nodes})$ | sparse CSR | Assembled from local stiffness | Global stiffness matrix |
| $\mathbf{f}$ (`f`) | auxiliary | $(n_\text{exc}, n_\text{nodes})$ | float64 | Empty array | Nodal potential solutions for each excitation |
| $\mathbf{R}_\text{mat}$ (`r_mat`) | auxiliary | $(n_\text{el}, n_\text{nodes})$ | float64 | Submatrix of $\mathbf{K}_g^{-1}$ | Reciprocity matrix at electrode positions |
| $\mathbf{r}_i$ (`ri`) | auxiliary | $(n_\text{meas}, n_\text{el})$ | float64 | Computed from `r_mat` | Differential reciprocity vectors |
| $\mathbf{H}$ (`H`) | auxiliary | $(n_\text{elems}, n_\text{meas})$ | float64 | Computed | Reconstruction matrix $(\mathbf{J}^\top\mathbf{J} + \lambda\mathbf{R})^{-1}\mathbf{J}^\top$ |
| $\Delta\mathbf{v}$ (`dv`) | auxiliary | $(n_\text{meas},)$ | float64 | $\mathbf{v}_1 - \mathbf{v}_0$ | Voltage difference vector |

### Constants
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $n_\text{el}$ | constants | scalar | int | 16 | Number of electrodes |
| $h_0$ | constants | scalar | float | 0.2 | Mesh element size parameter |
| bbox | constants | $(2, 3)$ | list | $[[-1,-1,-1],[1,1,1]]$ | Bounding box for mesh generation |
| $d_\text{exc}$ (`dist_exc`) | constants | scalar | int | 7 | Distance between excitation electrodes |
| $s_\text{meas}$ (`step_meas`) | constants | scalar | int | 1 | Step for measurement pattern |
| $\mathbf{c}_\text{anom}$ | constants | $(3,)$ | list | $[0.4, 0.4, 0.0]$ | Anomaly center coordinates |
| $r_\text{anom}$ | constants | scalar | float | 0.3 | Anomaly radius |
| $\sigma_\text{anom}$ | constants | scalar | float | 100.0 | Anomaly permittivity |
| $\sigma_\text{bg}$ | constants | scalar | float | 1.0 | Background permittivity |
| $p$ | constants | scalar | float | 0.50 | Kotre regularization exponent |
| $\lambda$ (`lamb`) | constants | scalar | float | $10^{-3}$ | Regularization parameter |
| method | constants | string | str | "kotre" | Regularization method |
| normalize | constants | boolean | bool | False | Whether to use normalized difference |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{v}_0$ (`v0`) | observations | $(n_\text{meas},)$ | float64 | Forward solve with $\sigma_\text{bg}$ | Baseline boundary voltages |
| $\mathbf{v}_1$ (`v1`) | observations | $(n_\text{meas},)$ | float64 | Forward solve with anomaly | Perturbed boundary voltages |

### Mesh and Protocol Variables
| Symbol | Category | Shape | Dtype | Description |
|--------|----------|-------|-------|-------------|
| $\mathbf{P}$ (`pts`) | constants | $(n_\text{nodes}, 3)$ | float64 | Node coordinates |
| $\mathbf{T}$ (`tri`) | constants | $(n_\text{elems}, 4)$ | int | Tetrahedron connectivity |
| $\mathbf{S}_e$ (`se`) | constants | $(n_\text{elems}, 4, 4)$ | float64 | Local stiffness matrices |
| $\mathbf{E}_\text{exc}$ (`ex_mat`) | constants | $(n_\text{el}, 2)$ | int | Excitation electrode pairs |
| $\mathbf{M}_\text{meas}$ (`meas_mat`) | constants | $(n_\text{meas}, 3)$ | int | Measurement pattern [n, m, exc_id] |
| $\text{el\_pos}$ | constants | $(n_\text{el},)$ | int | Node indices of electrodes |
| $\text{ref}$ | constants | scalar | int | mesh_obj.ref_node | Reference node for Dirichlet BC |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Mesh Generation | $\text{mesh\_obj} = \text{mesh.create}(n_\text{el}, h_0, \text{bbox}, \text{ball})$ | $n_\text{el}=16$, $h_0=0.2$, bbox, ball function | $\mathbf{P} \in \mathbb{R}^{n_\text{nodes} \times 3}$, $\mathbf{T} \in \mathbb{Z}^{n_\text{elems} \times 4}$ |
| 2 | Excitation Pattern | $\mathbf{E}_\text{exc}[i] = [i, (i + d_\text{exc}) \mod n_\text{el}]$ for $i = 0, \ldots, n_\text{el}-1$ | $n_\text{el}=16$, $d_\text{exc}=7$ | $\mathbf{E}_\text{exc} \in \mathbb{Z}^{16 \times 2}$ |
| 3 | Measurement Pattern | For each excitation $(a,b)$: $m = 0,\ldots,n_\text{el}-1$; $n = (m+1) \mod n_\text{el}$; keep if $m,n \notin \{a,b\}$ | $\mathbf{E}_\text{exc}$, $n_\text{el}$, $s_\text{meas}=1$ | $\mathbf{M}_\text{meas} \in \mathbb{Z}^{n_\text{meas} \times 3}$ |
| 4 | Local Stiffness | $\mathbf{S}_e^{(k)} = \frac{\mathbf{A}^{(k)} (\mathbf{A}^{(k)})^\top}{36 V_k}$ where $V_k = \frac{1}{6}\det([\mathbf{s}_0, \mathbf{s}_1, \mathbf{s}_2])$, $\mathbf{A}^{(k)}_{i} = \pm(\mathbf{s}_{i_1} \times \mathbf{s}_{i_2})$ | $\mathbf{P}$, $\mathbf{T}$ | $\mathbf{S}_e \in \mathbb{R}^{n_\text{elems} \times 4 \times 4}$ |
| 5 | Baseline Stiffness Assembly | $\mathbf{K}_g^{(0)} = \sum_{k=1}^{n_\text{elems}} \sigma_\text{bg} \cdot \mathbf{L}_k^\top \mathbf{S}_e^{(k)} \mathbf{L}_k$ with Dirichlet BC at ref node | $\mathbf{S}_e$, $\mathbf{T}$, $\sigma_\text{bg}=1.0$ | $\mathbf{K}_g^{(0)} \in \mathbb{R}^{n_\text{nodes} \times n_\text{nodes}}$ (sparse) |
| 6 | Excitation RHS | $\mathbf{b}_i[\text{el\_pos}[\mathbf{E}_\text{exc}[i,0]]] = 1$, $\mathbf{b}_i[\text{el\_pos}[\mathbf{E}_\text{exc}[i,1]]] = -1$ | $\mathbf{E}_\text{exc}$, el\_pos | $\mathbf{B} \in \mathbb{R}^{n_\text{exc} \times n_\text{nodes}}$ |
| 7 | Baseline Forward Solve | $\mathbf{f}_i^{(0)} = (\mathbf{K}_g^{(0)})^{-1} \mathbf{b}_i$ for each excitation $i$ | $\mathbf{K}_g^{(0)}$, $\mathbf{B}$ | $\mathbf{F}^{(0)} \in \mathbb{R}^{n_\text{exc} \times n_\text{nodes}}$ |
| 8 | Baseline Voltages | $v_0^{(j)} = \mathbf{f}^{(0)}_{\text{idx}[j]}[\text{el\_pos}[n_j]] - \mathbf{f}^{(0)}_{\text{idx}[j]}[\text{el\_pos}[m_j]]$ | $\mathbf{F}^{(0)}$, $\mathbf{M}_\text{meas}$, el\_pos | $\mathbf{v}_0 \in \mathbb{R}^{n_\text{meas}}$ |
| 9 | Anomaly Permittivity | $\sigma^{(1)}_k = \begin{cases} \sigma_\text{anom} & \text{if centroid}_k \in \text{Ball}(\mathbf{c}_\text{anom}, r_\text{anom}) \\ \sigma_\text{bg} & \text{otherwise} \end{cases}$ | $\mathbf{c}_\text{anom}$, $r_\text{anom}$, $\sigma_\text{anom}$, $\sigma_\text{bg}$ | $\boldsymbol{\sigma}^{(1)} \in \mathbb{R}^{n_\text{elems}}$ |
| 10 | Anomaly Stiffness Assembly | $\mathbf{K}_g^{(1)} = \sum_{k=1}^{n_\text{elems}} \sigma^{(1)}_k \cdot \mathbf{L}_k^\top \mathbf{S}_e^{(k)} \mathbf{L}_k$ | $\mathbf{S}_e$, $\boldsymbol{\sigma}^{(1)}$ | $\mathbf{K}_g^{(1)} \in \mathbb{R}^{n_\text{nodes} \times n_\text{nodes}}$ |
| 11 | Anomaly Forward Solve | $\mathbf{f}_i^{(1)} = (\mathbf{K}_g^{(1)})^{-1} \mathbf{b}_i$ | $\mathbf{K}_g^{(1)}$, $\mathbf{B}$ | $\mathbf{F}^{(1)} \in \mathbb{R}^{n_\text{exc} \times n_\text{nodes}}$ |
| 12 | Perturbed Voltages | $v_1^{(j)} = \mathbf{f}^{(1)}_{\text{idx}[j]}[\text{el\_pos}[n_j]] - \mathbf{f}^{(1)}_{\text{idx}[j]}[\text{el\_pos}[m_j]]$ | $\mathbf{F}^{(1)}$, $\mathbf{M}_\text{meas}$ | $\mathbf{v}_1 \in \mathbb{R}^{n_\text{meas}}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Assemble Global Stiffness | $\mathbf{K}_g = \sum_{k=1}^{n_\text{elems}} \sigma_{\text{bg},k} \cdot \mathbf{L}_k^\top \mathbf{S}_e^{(k)} \mathbf{L}_k$ with $\mathbf{K}_g[\text{ref}, :] = \mathbf{K}_g[:, \text{ref}] = 0$, $\mathbf{K}_g[\text{ref}, \text{ref}] = 1$ | $\mathbf{S}_e$, $\mathbf{T}$, $\boldsymbol{\sigma}_\text{baseline}$ | $\mathbf{K}_g \in \mathbb{R}^{n_\text{nodes} \times n_\text{nodes}}$ |
| I.2 | Construct RHS Matrix | $\mathbf{B}[i, \text{el\_pos}[\mathbf{E}_\text{exc}[i,:]]] = [1, -1]$ | $\mathbf{E}_\text{exc}$, el\_pos | $\mathbf{B} \in \mathbb{R}^{n_\text{exc} \times n_\text{nodes}}$ |
| I.3 | Solve Forward Problems | $\mathbf{f}_i = \mathbf{K}_g^{-1} \mathbf{b}_i$ via sparse LU for $i = 1, \ldots, n_\text{exc}$ | $\mathbf{K}_g$, $\mathbf{B}$ | $\mathbf{F} \in \mathbb{R}^{n_\text{exc} \times n_\text{nodes}}$ |
| I.4 | Compute Reciprocity Matrix | $\mathbf{R}_\text{mat} = (\mathbf{K}_g^{-1})[\text{el\_pos}, :]$ | $\mathbf{K}_g$, el\_pos | $\mathbf{R}_\text{mat} \in \mathbb{R}^{n_\text{el} \times n_\text{nodes}}$ |
| I.5 | Differential Reciprocity | $\mathbf{r}_i^{(j)} = \mathbf{R}_\text{mat}[n_j, :] - \mathbf{R}_\text{mat}[m_j, :]$ for measurement $j$ | $\mathbf{R}_\text{mat}$, $\mathbf{M}_\text{meas}$ | $\mathbf{r}_i \in \mathbb{R}^{n_\text{meas} \times n_\text{nodes}}$ |
| I.6 | Initialize Jacobian | $\mathbf{J} = \mathbf{0}_{n_\text{meas} \times n_\text{elems}}$ | $n_\text{meas}$, $n_\text{elems}$ | $\mathbf{J} \in \mathbb{R}^{n_\text{meas} \times n_\text{elems}}$ |

### 4.2 ITERATION LOOP (Jacobian Assembly)

**BEGIN_LOOP** (over elements $e = 0, \ldots, n_\text{elems}-1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | Extract Element Nodes | $\text{ijk} = \mathbf{T}[e, :]$ | $\mathbf{T}$, $e$ | ijk $\in \mathbb{Z}^4$ | Indices of 4 nodes |
| L.2 | Extract Forward Potentials | $\mathbf{F}_n = \mathbf{F}[\text{idx}, :]$ where $\text{idx} = \mathbf{M}_\text{meas}[:, 2]$ | $\mathbf{F}$, $\mathbf{M}_\text{meas}$ | $\mathbf{F}_n \in \mathbb{R}^{n_\text{meas} \times n_\text{nodes}}$ | Potentials for relevant excitations |
| L.3 | Compute Jacobian Column | $\mathbf{J}[:, e] = \sum_{l=1}^{4} \left( \mathbf{r}_i[:, \text{ijk}] \cdot \mathbf{S}_e^{(e)} \right) \odot \mathbf{F}_n[:, \text{ijk}]$ summed over local nodes | $\mathbf{r}_i$, $\mathbf{S}_e^{(e)}$, $\mathbf{F}_n$, ijk | $\mathbf{J}[:, e] \in \mathbb{R}^{n_\text{meas}}$ | Sensitivity of all measurements to element $e$ |

**END_LOOP**

### 4.3 FINALIZATION (Inversion)

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Compute Gram Matrix | $\mathbf{G} = \mathbf{J}^\top \mathbf{J}$ | $\mathbf{J}$ | $\mathbf{G} \in \mathbb{R}^{n_\text{elems} \times n_\text{elems}}$ |
| F.2 | Construct Regularization | **Kotre**: $\mathbf{R} = \text{diag}((\text{diag}(\mathbf{G}))^p)$ with $p = 0.50$; **LM**: $\mathbf{R} = \text{diag}(\text{diag}(\mathbf{G}))$; **DGN**: $\mathbf{R} = \mathbf{I}$ | $\mathbf{G}$, $p$, method | $\mathbf{R} \in \mathbb{R}^{n_\text{elems} \times n_\text{elems}}$ |
| F.3 | Compute Reconstruction Matrix | $\mathbf{H} = (\mathbf{G} + \lambda \mathbf{R})^{-1} \mathbf{J}^\top$ with $\lambda = 10^{-3}$ | $\mathbf{G}$, $\mathbf{R}$, $\mathbf{J}$, $\lambda$ | $\mathbf{H} \in \mathbb{R}^{n_\text{elems} \times n_\text{meas}}$ |
| F.4 | Compute Voltage Difference | **Unnormalized**: $\Delta\mathbf{v} = \mathbf{v}_1 - \mathbf{v}_0$; **Normalized**: $\Delta\mathbf{v} = \log\left|\frac{\mathbf{v}_1}{\mathbf{v}_0}\right| \cdot \text{sign}(\text{Re}(\mathbf{v}_0))$ | $\mathbf{v}_0$, $\mathbf{v}_1$, normalize | $\Delta\mathbf{v} \in \mathbb{R}^{n_\text{meas}}$ |
| F.5 | Solve Inverse Problem | $\Delta\boldsymbol{\sigma} = -\mathbf{H} \cdot \Delta\mathbf{v}$ | $\mathbf{H}$, $\Delta\mathbf{v}$ | $\Delta\boldsymbol{\sigma} \in \mathbb{R}^{n_\text{elems}}$ |

---

## 5. STOPPING CRITERION

- **type**: single_pass (non-iterative closed-form solution)
- **expression**: Algorithm terminates after one-step Jacobian computation and matrix inversion; no iterative convergence criterion
- **parameters**: 
  - Element loop: $e = 0, \ldots, n_\text{elems} - 1$ (exhaustive)
  - Excitation loop: $i = 0, \ldots, n_\text{exc} - 1$ where $n_\text{exc} = n_\text{el} = 16$
- **check_frequency**: N/A (single-pass algorithm)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| ds_min | $\min_e \text{Re}(\Delta\sigma_e)$ | Minimum reconstructed conductivity change |
| ds_max | $\max_e \text{Re}(\Delta\sigma_e)$ | Maximum reconstructed conductivity change |
| ds_mean | $\frac{1}{n_\text{elems}} \sum_{e=1}^{n_\text{elems}} \text{Re}(\Delta\sigma_e)$ | Mean reconstructed conductivity change |
| ds_std | $\sqrt{\frac{1}{n_\text{elems}} \sum_{e=1}^{n_\text{elems}} (\text{Re}(\Delta\sigma_e) - \overline{\Delta\sigma})^2}$ | Standard deviation of conductivity change |
| data_residual | $\|\mathbf{v}_1 - \mathbf{v}_0\|_2$ | L2 norm of voltage difference |
| relative_change | $\frac{\|\mathbf{v}_1 - \mathbf{v}_0\|_2}{\|\mathbf{v}_0\|_2}$ | Relative voltage change magnitude |
| max_change_element | $\arg\max_e |\text{Re}(\Delta\sigma_e)|$ | Element index with largest absolute change |
| max_change_value | $\text{Re}(\Delta\sigma_{\text{max\_idx}})$ | Value at element with largest change |
| forward_consistency | $\|\mathbf{v}_\text{pred} - \mathbf{v}_0\|_2$ where $\mathbf{v}_\text{pred} = \mathcal{F}(\boldsymbol{\sigma}_\text{baseline})$ | Forward operator self-consistency check |