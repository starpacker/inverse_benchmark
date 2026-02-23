## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Jacobian-Based EIT Reconstruction (JAC)
- **algorithm_family**: Linearized Gauss-Newton with Tikhonov Regularization
- **summary**: Single-step linear reconstruction of 3D conductivity changes from boundary voltage measurements using Jacobian sensitivity matrix and regularized pseudoinverse.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\Delta\sigma} \frac{1}{2}\|\mathbf{J}\Delta\sigma - \Delta\mathbf{v}\|_2^2 + \frac{\lambda}{2}\|\mathbf{R}\Delta\sigma\|_2^2$$

### data_fidelity_term
$$\frac{1}{2}\|\mathbf{J}\Delta\sigma - \Delta\mathbf{v}\|_2^2 = \frac{1}{2}\sum_{i=1}^{M}\left([\mathbf{J}\Delta\sigma]_i - \Delta v_i\right)^2$$

where $\Delta\mathbf{v} = \mathbf{v}_1 - \mathbf{v}_0$ (difference measurements), $M$ = number of measurements.

### regularization_term
$$\frac{\lambda}{2}\|\mathbf{R}\Delta\sigma\|_2^2$$

where the regularization matrix $\mathbf{R}$ depends on method:
- **Kotre**: $\mathbf{R} = \text{diag}((\mathbf{J}^T\mathbf{J})_{ii}^{p/2})$ with $p \in [0,1]$
- **Levenberg-Marquardt (lm)**: $\mathbf{R} = \text{diag}(\sqrt{(\mathbf{J}^T\mathbf{J})_{ii}})$
- **Damped Gauss-Newton (dgn)**: $\mathbf{R} = \mathbf{I}$

### constraints
None (unconstrained linear inverse problem)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Delta\sigma$ (`ds`) | $(N_e,)$ | float64 | Computed directly | Reconstructed conductivity change per element |

### 2.2 Dual Variables

None (closed-form solution, no dual formulation)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{K}_g$ (`kg`) | $(N_n, N_n)$ | sparse CSR | Assembled from $\mathbf{K}_e$ | Global stiffness matrix |
| $\mathbf{b}$ (`b`) | $(N_{ex}, N_n)$ | float64 | Zeros with ±1 at electrodes | Excitation current patterns |
| $\mathbf{f}$ (`f`) | $(N_{ex}, N_n)$ | float64 | Undefined | Nodal potential solutions |
| $\mathbf{R}_{el}$ (`r_mat`) | $(N_{el}, N_n)$ | float64 | From $\mathbf{K}_g^{-1}$ | Electrode rows of inverse stiffness |
| $\mathbf{R}_i$ (`ri`) | $(M, 4)$ | float64 | Computed | Differential reconstruction vectors |
| $\mathbf{J}$ (`jac`) | $(M, N_e)$ | float64 | Zeros | Jacobian matrix |
| $\mathbf{J}^T\mathbf{J}$ (`j_w_j`) | $(N_e, N_e)$ | float64 | Computed | Jacobian Gramian |
| $\mathbf{R}_{reg}$ (`r_mat_reg`) | $(N_e, N_e)$ | float64 | Method-dependent | Regularization matrix |
| $\mathbf{H}$ (`H`) | $(N_e, M)$ | float64 | Computed | Reconstruction matrix (pseudoinverse) |
| $\Delta\mathbf{v}$ (`dv`) | $(M,)$ | float64 | $\mathbf{v}_1 - \mathbf{v}_0$ | Measurement difference |
| $\mathbf{f}_n$ (`f_n`) | $(M, N_n)$ | float64 | Indexed from $\mathbf{f}$ | Potentials for each measurement |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{el}$ (`n_el`) | 16 | Number of electrodes |
| $h_0$ (`h0`) | 0.2 | Mesh element size parameter |
| $d_{exc}$ (`dist_exc`) | 7 | Excitation electrode distance |
| $s_{meas}$ (`step_meas`) | 1 | Measurement step |
| $p$ (`p`) | 0.50 | Kotre regularization exponent |
| $\lambda$ (`lamb`) | 0.001 | Regularization parameter |
| `method` | "kotre" | Regularization method |
| `normalize` | False | Whether to use log-ratio normalization |
| $\sigma_{bg}$ (`background_perm`) | 1.0 | Background conductivity |
| $\sigma_{anom}$ (`anomaly_perm`) | 100.0 | Anomaly conductivity |
| $r_{anom}$ (`anomaly_r`) | 0.3 | Anomaly radius |
| $\mathbf{c}_{anom}$ (`anomaly_center`) | [0.4, 0.4, 0.0] | Anomaly center |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{v}_0$ (`v0`) | $(M,)$ | float64 | Baseline boundary voltage measurements |
| $\mathbf{v}_1$ (`v1`) | $(M,)$ | float64 | Perturbed boundary voltage measurements |
| $\mathbf{P}$ (`pts`) | $(N_n, 3)$ | float64 | Mesh node coordinates |
| $\mathbf{T}$ (`tri`) | $(N_e, 4)$ | int | Element connectivity (tetrahedra) |
| $\boldsymbol{\sigma}_0$ (`perm_baseline`) | $(N_e,)$ | float64 | Baseline conductivity distribution |
| $\mathbf{K}_e$ (`se`) | $(N_e, 4, 4)$ | float64 | Pre-computed local stiffness matrices |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Mesh Generation | $(\mathbf{P}, \mathbf{T}) = \text{mesh.create}(N_{el}, h_0, \text{bbox}, \text{ball})$ | $N_{el}: \text{int}$, $h_0: \text{float}$, bbox: list | $\mathbf{P}: (N_n, 3)$, $\mathbf{T}: (N_e, 4)$ |
| 2 | Excitation Pattern | $\mathbf{E}_{ex}[i,:] = [i, (i + d_{exc}) \mod N_{el}]$ for $i \in [0, N_{el})$ | $N_{el}$, $d_{exc}$ | $\mathbf{E}_{ex}: (N_{el}, 2)$ |
| 3 | Measurement Pattern | For each excitation $(a,b)$: $m = [0,\ldots,N_{el}-1]$, $n = (m + s_{meas}) \mod N_{el}$; keep where $m \neq a \land m \neq b \land n \neq a \land n \neq b$ | $\mathbf{E}_{ex}$, $N_{el}$, $s_{meas}$ | $\mathbf{M}_{pat}: (M, 3)$ where columns are $(n, m, \text{exc\_idx})$ |
| 4 | Local Stiffness Matrices | For tetrahedron with vertices $\mathbf{xy} \in \mathbb{R}^{4 \times 3}$: $\mathbf{s} = \mathbf{xy}_{[2,3,0,1]} - \mathbf{xy}_{[0,1,2,3]}$, $V_t = \frac{1}{6}\det(\mathbf{s}_{[0,1,2]})$, $\mathbf{a}_k = \text{sign}_k \cdot (\mathbf{s}_i \times \mathbf{s}_j)$ for pairs $[(0,1),(1,2),(2,3),(3,0)]$ with signs $[1,-1,1,-1]$, $\mathbf{K}_e^{(e)} = \frac{\mathbf{a}\mathbf{a}^T}{36 V_t}$ | $\mathbf{P}$, $\mathbf{T}$ | $\mathbf{K}_e: (N_e, 4, 4)$ |
| 5 | Baseline Stiffness Assembly | $\mathbf{K}_g = \sum_{e=1}^{N_e} \sigma_0^{(e)} \mathbf{A}_e^T \mathbf{K}_e^{(e)} \mathbf{A}_e$ with Dirichlet BC at ref node | $\mathbf{K}_e$, $\mathbf{T}$, $\boldsymbol{\sigma}_0$ | $\mathbf{K}_g: (N_n, N_n)$ sparse |
| 6 | Baseline Forward Solve | $\mathbf{b}_{i, \text{el\_pos}[\mathbf{E}_{ex}[i,0]]} = 1$, $\mathbf{b}_{i, \text{el\_pos}[\mathbf{E}_{ex}[i,1]]} = -1$; $\mathbf{f}_i = \mathbf{K}_g^{-1}\mathbf{b}_i$ | $\mathbf{K}_g$, $\mathbf{E}_{ex}$, el\_pos | $\mathbf{f}: (N_{ex}, N_n)$ |
| 7 | Baseline Measurements | $v_0^{(k)} = \mathbf{f}_{\text{idx}[k], \text{el\_pos}[n_k]} - \mathbf{f}_{\text{idx}[k], \text{el\_pos}[m_k]}$ where $(n_k, m_k, \text{idx}_k) = \mathbf{M}_{pat}[k,:]$ | $\mathbf{f}$, $\mathbf{M}_{pat}$, el\_pos | $\mathbf{v}_0: (M,)$ |
| 8 | Anomaly Conductivity | $\sigma_1^{(e)} = \sigma_{anom}$ if centroid of element $e$ within ball$(\mathbf{c}_{anom}, r_{anom})$, else $\sigma_{bg}$ | mesh, anomaly params | $\boldsymbol{\sigma}_1: (N_e,)$ |
| 9 | Perturbed Forward Solve | Same as steps 5-7 with $\boldsymbol{\sigma}_1$ | $\mathbf{K}_e$, $\boldsymbol{\sigma}_1$ | $\mathbf{v}_1: (M,)$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Assemble baseline stiffness | $\mathbf{K}_g = \text{assemble}(\mathbf{K}_e, \mathbf{T}, \boldsymbol{\sigma}_0, N_n, \text{ref})$ | $\mathbf{K}_e: (N_e,4,4)$, $\mathbf{T}: (N_e,4)$, $\boldsymbol{\sigma}_0: (N_e,)$ | $\mathbf{K}_g: (N_n, N_n)$ sparse | CSR format with Dirichlet at ref node |
| 2 | Build excitation vectors | $\mathbf{b}_{i,j} = \begin{cases} 1 & j = \text{el\_pos}[\mathbf{E}_{ex}[i,0]] \\ -1 & j = \text{el\_pos}[\mathbf{E}_{ex}[i,1]] \\ 0 & \text{otherwise} \end{cases}$ | $\mathbf{E}_{ex}: (N_{ex},2)$, el\_pos | $\mathbf{b}: (N_{ex}, N_n)$ | Current injection pattern |
| 3 | Solve forward problems | $\mathbf{f}_{i,:} = \text{spsolve}(\mathbf{K}_g, \mathbf{b}_{i,:})$ for $i = 0, \ldots, N_{ex}-1$ | $\mathbf{K}_g$, $\mathbf{b}$ | $\mathbf{f}: (N_{ex}, N_n)$ | [FORWARD] sparse LU solve per excitation |
| 4 | Compute electrode inverse | $\mathbf{R}_{el} = (\mathbf{K}_g^{-1})[\text{el\_pos}, :]$ | $\mathbf{K}_g$, el\_pos: $(N_{el},)$ | $\mathbf{R}_{el}: (N_{el}, N_n)$ | Full inverse, extract electrode rows |
| 5 | Replicate for excitations | $\mathbf{R}_{el}^{(i)} = \mathbf{R}_{el}$ for $i = 0, \ldots, N_{ex}-1$ | $\mathbf{R}_{el}$ | $\mathbf{R}_{rep}: (N_{ex}, N_{el}, N_n)$ | Broadcast for vectorized computation |
| 6 | Compute differential reconstruction | $\mathbf{R}_i[k,:] = \mathbf{R}_{rep}[\text{idx}_k, n_k, :] - \mathbf{R}_{rep}[\text{idx}_k, m_k, :]$ | $\mathbf{R}_{rep}$, $\mathbf{M}_{pat}$ | $\mathbf{R}_i: (M, N_n)$ | Measurement differencing |
| 7 | Initialize Jacobian | $\mathbf{J} = \mathbf{0}_{M \times N_e}$ | $M$, $N_e$ | $\mathbf{J}: (M, N_e)$ | Zero matrix |
| 8 | Index potentials by measurement | $\mathbf{f}_n = \mathbf{f}[\mathbf{M}_{pat}[:,2], :]$ | $\mathbf{f}$, $\mathbf{M}_{pat}$ | $\mathbf{f}_n: (M, N_n)$ | Select excitation for each measurement |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (element loop: $e = 0, \ldots, N_e - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 9 | Extract element nodes | $\text{ijk} = \mathbf{T}[e, :]$ | $\mathbf{T}$, $e$ | ijk: $(4,)$ | Node indices for element $e$ |
| 10 | Compute Jacobian column | $\mathbf{J}[:,e] = \sum_{q=0}^{3} \left[ (\mathbf{R}_i[:, \text{ijk}] \cdot \mathbf{K}_e^{(e)}) \odot \mathbf{f}_n[:, \text{ijk}] \right]_{:,q}$ | $\mathbf{R}_i: (M, N_n)$, $\mathbf{K}_e^{(e)}: (4,4)$, $\mathbf{f}_n: (M, N_n)$, ijk | $\mathbf{J}[:,e]: (M,)$ | $J_{k,e} = \sum_{i,j \in \text{ijk}} R_i[k,i] K_e^{(e)}[i,j] f_n[k,j]$ |

**END_LOOP**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 11 | Compute Jacobian Gramian | $\mathbf{G} = \mathbf{J}^T \mathbf{J}$ | $\mathbf{J}: (M, N_e)$ | $\mathbf{G}: (N_e, N_e)$ | Symmetric positive semi-definite |
| 12 | Build regularization matrix | $\mathbf{R}_{reg} = \begin{cases} \text{diag}(\mathbf{G}_{ii}^p) & \text{method} = \text{"kotre"} \\ \text{diag}(\mathbf{G}_{ii}) & \text{method} = \text{"lm"} \\ \mathbf{I}_{N_e} & \text{otherwise} \end{cases}$ | $\mathbf{G}$, $p=0.5$, method | $\mathbf{R}_{reg}: (N_e, N_e)$ | Diagonal scaling matrix |
| 13 | Compute reconstruction matrix | $\mathbf{H} = (\mathbf{G} + \lambda \mathbf{R}_{reg})^{-1} \mathbf{J}^T$ | $\mathbf{G}$, $\mathbf{R}_{reg}$, $\mathbf{J}$, $\lambda=0.001$ | $\mathbf{H}: (N_e, M)$ | Tikhonov-regularized pseudoinverse |
| 14 | Compute measurement difference | $\Delta\mathbf{v} = \mathbf{v}_1 - \mathbf{v}_0$ | $\mathbf{v}_1: (M,)$, $\mathbf{v}_0: (M,)$ | $\Delta\mathbf{v}: (M,)$ | If normalize=True: $\Delta\mathbf{v} = \log(|\mathbf{v}_1|/|\mathbf{v}_0|) \cdot \text{sign}(\Re(\mathbf{v}_0))$ |
| 15 | Reconstruct conductivity change | $\Delta\boldsymbol{\sigma} = -\mathbf{H} \Delta\mathbf{v}$ | $\mathbf{H}: (N_e, M)$, $\Delta\mathbf{v}: (M,)$ | $\Delta\boldsymbol{\sigma}: (N_e,)$ | Negative sign from physics convention |

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 16 | Interpolate to nodes | For each node $n$: $\Delta\sigma_n^{(\text{node})} = \frac{\sum_{e: n \in e} \Delta\sigma_e}{|\{e: n \in e\}|}$ | $\Delta\boldsymbol{\sigma}: (N_e,)$, $\mathbf{T}$ | $\Delta\boldsymbol{\sigma}^{(\text{node})}: (N_n,)$ | Average element values to nodes for visualization |

---

## 5. STOPPING CRITERION

- **type**: single_step (non-iterative)
- **expression**: Algorithm terminates after one linear solve: $\Delta\boldsymbol{\sigma} = -\mathbf{H}\Delta\mathbf{v}$
- **parameters**: None (no convergence criteria)
- **check_frequency**: N/A (single-step closed-form solution)

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Minimum conductivity change | $\Delta\sigma_{\min} = \min_e \Re(\Delta\sigma_e)$ | N/A | $\Delta\boldsymbol{\sigma}$ | Smallest reconstructed change |
| Maximum conductivity change | $\Delta\sigma_{\max} = \max_e \Re(\Delta\sigma_e)$ | N/A | $\Delta\boldsymbol{\sigma}$ | Largest reconstructed change |
| Mean conductivity change | $\bar{\Delta\sigma} = \frac{1}{N_e}\sum_{e=1}^{N_e} \Re(\Delta\sigma_e)$ | N/A | $\Delta\boldsymbol{\sigma}$ | Average reconstructed change |
| Standard deviation | $\text{std}(\Delta\sigma) = \sqrt{\frac{1}{N_e}\sum_{e=1}^{N_e}(\Re(\Delta\sigma_e) - \bar{\Delta\sigma})^2}$ | N/A | $\Delta\boldsymbol{\sigma}$ | Spread of reconstructed values |
| Data residual norm | $\|\Delta\mathbf{v}\|_2 = \sqrt{\sum_{k=1}^{M}(v_1^{(k)} - v_0^{(k)})^2}$ | $\mathbf{v}_0$ | $\mathbf{v}_1$ | Magnitude of voltage change |
| Relative voltage change | $\frac{\|\mathbf{v}_1 - \mathbf{v}_0\|_2}{\|\mathbf{v}_0\|_2}$ | $\mathbf{v}_0$ | $\mathbf{v}_1$ | Normalized signal change |
| Maximum change element | $e^* = \arg\max_e |\Re(\Delta\sigma_e)|$ | N/A | $\Delta\boldsymbol{\sigma}$ | Element index with largest absolute change |
| Maximum change value | $\Delta\sigma_{e^*}$ | N/A | $\Delta\boldsymbol{\sigma}$ | Value at maximum change element |