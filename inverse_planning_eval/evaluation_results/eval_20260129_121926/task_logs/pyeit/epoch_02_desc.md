# Algorithmic Specification Document: 3D EIT Jacobian-Based Inversion

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\Delta\boldsymbol{\sigma}} \frac{1}{2} \|\mathbf{J}\Delta\boldsymbol{\sigma} - \Delta\mathbf{v}\|_2^2 + \frac{\lambda}{2} \|\mathbf{R}^{1/2}\Delta\boldsymbol{\sigma}\|_2^2$$

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}}(\Delta\boldsymbol{\sigma}) = \frac{1}{2}\sum_{m=1}^{M}\left(\sum_{e=1}^{E} J_{me}\Delta\sigma_e - \Delta v_m\right)^2 = \frac{1}{2}\|\mathbf{J}\Delta\boldsymbol{\sigma} - \Delta\mathbf{v}\|_2^2$$

where:
- $M$ = number of measurements
- $E$ = number of elements
- $\Delta v_m = v_m^{(1)} - v_m^{(0)}$ (voltage difference between perturbed and baseline)

**regularization_term**:
$$\mathcal{R}(\Delta\boldsymbol{\sigma}) = \frac{\lambda}{2}\Delta\boldsymbol{\sigma}^\top \mathbf{R} \Delta\boldsymbol{\sigma}$$

where $\mathbf{R} \in \mathbb{R}^{E \times E}$ depends on method:

| Method | Regularization Matrix $\mathbf{R}$ |
|--------|-----------------------------------|
| `kotre` | $R_{ii} = \left[(\mathbf{J}^\top\mathbf{J})_{ii}\right]^p$, $R_{ij}=0$ for $i\neq j$ |
| `lm` | $R_{ii} = (\mathbf{J}^\top\mathbf{J})_{ii}$, $R_{ij}=0$ for $i\neq j$ |
| `dgn` | $\mathbf{R} = \mathbf{I}_E$ |

**constraints**:
- None (unconstrained optimization with closed-form solution)

---

## 2. VARIABLES

### Primal Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\Delta\boldsymbol{\sigma}$ | `primal` | $(E,)$ | `float64` | Computed directly | Reconstructed conductivity change per element |

### Dual Variables

None (closed-form solution, no iterative dual updates)

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\lambda$ | `constants` | scalar | `float64` | User-specified (`lamb=0.001`) | Regularization weight |
| $p$ | `constants` | scalar | `float64` | User-specified (`p=0.20`) | Kotre exponent parameter |
| $n_{el}$ | `constants` | scalar | `int` | User-specified (`n_el=16`) | Number of electrodes |
| $\boldsymbol{\sigma}_0$ | `constants` | $(E,)$ | `float64` | `perm_baseline` | Baseline conductivity distribution |
| $\mathbf{P}$ | `constants` | $(N, 3)$ | `float64` | From mesh | Node coordinates |
| $\mathbf{T}$ | `constants` | $(E, 4)$ | `int` | From mesh | Element connectivity (tetrahedra) |
| $\mathbf{S}^{(e)}$ | `constants` | $(E, 4, 4)$ | `float64` | Precomputed | Local stiffness matrices |
| $\mathbf{E}_{ex}$ | `constants` | $(n_{ex}, 2)$ | `int` | From protocol | Excitation electrode pairs |
| $\mathbf{E}_{meas}$ | `constants` | $(M, 3)$ | `int` | From protocol | Measurement pattern $[n, m, \text{exc\_idx}]$ |
| $\mathbf{e}_{pos}$ | `constants` | $(n_{el},)$ | `int` | From mesh | Electrode node indices |
| $n_{ref}$ | `constants` | scalar | `int` | From mesh | Reference node index (ground) |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{v}^{(0)}$ | `observations` | $(M,)$ | `float64` | Computed from baseline | Baseline boundary voltages |
| $\mathbf{v}^{(1)}$ | `observations` | $(M,)$ | `float64` | Computed with anomaly | Perturbed boundary voltages |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Mesh Generation

| Field | Value |
|-------|-------|
| `step_order` | 1 |
| `step_name` | Generate 3D tetrahedral mesh |
| `formula` | $\{\mathbf{P}, \mathbf{T}, \mathbf{e}_{pos}, n_{ref}\} = \texttt{mesh.create}(n_{el}, h_0, \text{bbox}, \text{ball})$ |
| `inputs` | $n_{el}$: scalar, $h_0$: scalar, bbox: $(2,3)$ |
| `outputs` | $\mathbf{P}$: $(N, 3)$, $\mathbf{T}$: $(E, 4)$, $\mathbf{e}_{pos}$: $(n_{el},)$, $n_{ref}$: scalar |

### Step 3.2: Protocol Generation - Excitation Pattern

| Field | Value |
|-------|-------|
| `step_order` | 2 |
| `step_name` | Build excitation pattern (adjacent mode) |
| `formula` | $(\mathbf{E}_{ex})_i = [i, (i + d_{exc}) \mod n_{el}]$ for $i = 0, \ldots, n_{el}-1$ |
| `inputs` | $n_{el}$: scalar, $d_{exc}$: scalar |
| `outputs` | $\mathbf{E}_{ex}$: $(n_{el}, 2)$ |

### Step 3.3: Protocol Generation - Measurement Pattern

| Field | Value |
|-------|-------|
| `step_order` | 3 |
| `step_name` | Build measurement pattern with electrode exclusion |
| `formula` | For each excitation $k$ with electrodes $(a_k, b_k)$: $m_j = j$, $n_j = (j + s_{meas}) \mod n_{el}$. Keep measurement $j$ iff $(m_j \neq a_k) \land (m_j \neq b_k) \land (n_j \neq a_k) \land (n_j \neq b_k)$ |
| `inputs` | $\mathbf{E}_{ex}$: $(n_{el}, 2)$, $n_{el}$: scalar, $s_{meas}$: scalar |
| `outputs` | $\mathbf{E}_{meas}$: $(M, 3)$ where each row is $[n, m, \text{exc\_idx}]$ |

### Step 3.4: Local Stiffness Matrix Computation

| Field | Value |
|-------|-------|
| `step_order` | 4 |
| `step_name` | Compute local stiffness matrices for all tetrahedra |
| `formula` | For element $e$ with vertices $\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3$: Define edge vectors $\mathbf{s}_i = \mathbf{x}_{(i+2)\mod 4} - \mathbf{x}_{(i+1)\mod 4}$. Volume: $V_e = \frac{1}{6}\det([\mathbf{s}_0, \mathbf{s}_1, \mathbf{s}_2])$. Face normals: $\mathbf{a}_0 = \mathbf{s}_0 \times \mathbf{s}_1$, $\mathbf{a}_1 = -\mathbf{s}_1 \times \mathbf{s}_2$, $\mathbf{a}_2 = \mathbf{s}_2 \times \mathbf{s}_3$, $\mathbf{a}_3 = -\mathbf{s}_3 \times \mathbf{s}_0$. Local stiffness: $S^{(e)}_{ij} = \frac{\mathbf{a}_i \cdot \mathbf{a}_j}{36 V_e}$ |
| `inputs` | $\mathbf{P}$: $(N, 3)$, $\mathbf{T}$: $(E, 4)$ |
| `outputs` | $\mathbf{S}^{(e)}$: $(E, 4, 4)$ |

### Step 3.5: Baseline Forward Solve

| Field | Value |
|-------|-------|
| `step_order` | 5 |
| `step_name` | Assemble global stiffness and solve baseline forward problem |
| `formula` | Global stiffness assembly: $K_{ij}(\boldsymbol{\sigma}) = \sum_{e: i,j \in \mathbf{T}_e} \sigma_e S^{(e)}_{local(i), local(j)}$ with Dirichlet BC at $n_{ref}$: $K_{n_{ref}, n_{ref}} = 1$, $K_{n_{ref}, j} = K_{i, n_{ref}} = 0$ for $i,j \neq n_{ref}$. Source vector for excitation $k$: $b^{(k)}_i = \begin{cases} 1 & i = \mathbf{e}_{pos}[(\mathbf{E}_{ex})_{k,0}] \\ -1 & i = \mathbf{e}_{pos}[(\mathbf{E}_{ex})_{k,1}] \\ 0 & \text{otherwise} \end{cases}$. Solve: $\mathbf{K}(\boldsymbol{\sigma}_0)\mathbf{u}^{(k)} = \mathbf{b}^{(k)}$ |
| `inputs` | $\mathbf{S}^{(e)}$: $(E,4,4)$, $\mathbf{T}$: $(E,4)$, $\boldsymbol{\sigma}_0$: $(E,)$, $\mathbf{E}_{ex}$: $(n_{el},2)$ |
| `outputs` | $\mathbf{K}_0$: $(N, N)$ sparse, $\mathbf{U}_0$: $(n_{el}, N)$ potential fields |

### Step 3.6: Baseline Voltage Extraction

| Field | Value |
|-------|-------|
| `step_order` | 6 |
| `step_name` | Extract differential boundary voltages |
| `formula` | For measurement $m$ with pattern $[n_m, m_m, k_m]$: $v^{(0)}_m = u^{(k_m)}_{\mathbf{e}_{pos}[n_m]} - u^{(k_m)}_{\mathbf{e}_{pos}[m_m]}$ |
| `inputs` | $\mathbf{U}_0$: $(n_{el}, N)$, $\mathbf{e}_{pos}$: $(n_{el},)$, $\mathbf{E}_{meas}$: $(M, 3)$ |
| `outputs` | $\mathbf{v}^{(0)}$: $(M,)$ |

### Step 3.7: Anomaly Forward Solve

| Field | Value |
|-------|-------|
| `step_order` | 7 |
| `step_name` | Compute perturbed measurements with anomaly |
| `formula` | $\sigma^{(1)}_e = \begin{cases} \sigma_{anomaly} & \|\mathbf{c}_e - \mathbf{c}_{anom}\|_2 \leq r_{anom} \\ \sigma_{bg} & \text{otherwise} \end{cases}$ where $\mathbf{c}_e$ is element centroid. Solve: $\mathbf{K}(\boldsymbol{\sigma}^{(1)})\mathbf{u}^{(1,k)} = \mathbf{b}^{(k)}$. Extract: $v^{(1)}_m = u^{(1,k_m)}_{\mathbf{e}_{pos}[n_m]} - u^{(1,k_m)}_{\mathbf{e}_{pos}[m_m]}$ |
| `inputs` | $\mathbf{S}^{(e)}$, anomaly parameters, $\mathbf{b}^{(k)}$ |
| `outputs` | $\mathbf{v}^{(1)}$: $(M,)$ |

---

## 4. ALGORITHM STEPS

### Initialization

**BEGIN INITIALIZATION**

#### Step I.1: Compute Global Stiffness at Baseline

| Field | Value |
|-------|-------|
| `step_name` | Assemble baseline global stiffness matrix |
| `formula` | $\mathbf{K}_0 = \texttt{assemble}(\mathbf{S}^{(e)}, \mathbf{T}, \boldsymbol{\sigma}_0, N, n_{ref})$ as defined in Step 3.5 |
| `inputs` | $\mathbf{S}^{(e)}$: $(E,4,4)$, $\mathbf{T}$: $(E,4)$, $\boldsymbol{\sigma}_0$: $(E,)$ |
| `outputs` | $\mathbf{K}_0$: $(N, N)$ sparse CSR |
| `computational_note` | Sparse matrix assembly with COO to CSR conversion |

#### Step I.2: Solve Forward Problem at Baseline

| Field | Value |
|-------|-------|
| `step_name` | Solve for potential fields under all excitations |
| `formula` | $\mathbf{u}^{(k)} = \mathbf{K}_0^{-1}\mathbf{b}^{(k)}$ for $k = 0, \ldots, n_{el}-1$ |
| `inputs` | $\mathbf{K}_0$: $(N, N)$, $\mathbf{B}$: $(n_{el}, N)$ |
| `outputs` | $\mathbf{F}$: $(n_{el}, N)$ where row $k$ is $\mathbf{u}^{(k)}$ |
| `computational_note` | Uses `scipy.sparse.linalg.spsolve` for each excitation |

#### Step I.3: Compute Inverse of Stiffness at Electrodes

| Field | Value |
|-------|-------|
| `step_name` | Compute electrode-restricted inverse stiffness |
| `formula` | $\mathbf{R}_{el} = (\mathbf{K}_0^{-1})[\mathbf{e}_{pos}, :]$ |
| `inputs` | $\mathbf{K}_0$: $(N, N)$, $\mathbf{e}_{pos}$: $(n_{el},)$ |
| `outputs` | $\mathbf{R}_{el}$: $(n_{el}, N)$ |
| `computational_note` | Requires dense inverse computation: $\mathbf{K}_0^{-1} = \texttt{la.inv}(\mathbf{K}_0.\texttt{toarray}())$ |

#### Step I.4: Compute Differential Response Matrix

| Field | Value |
|-------|-------|
| `step_name` | Apply measurement differencing to inverse matrix |
| `formula` | For measurement $m$ with pattern $[n_m, m_m, k_m]$: $\mathbf{r}_m = (\mathbf{R}_{el})_{n_m, :} - (\mathbf{R}_{el})_{m_m, :}$ |
| `inputs` | $\mathbf{R}_{el}$: $(n_{el}, N)$ replicated per excitation, $\mathbf{E}_{meas}$: $(M, 3)$ |
| `outputs` | $\mathbf{R}_i$: $(M, N)$ differential response vectors |
| `computational_note` | Vectorized subtraction using measurement indices |

#### Step I.5: Compute Jacobian Matrix

| Field | Value |
|-------|-------|
| `step_name` | Assemble Jacobian using sensitivity theorem |
| `formula` | $J_{me} = \sum_{i,j \in \mathbf{T}_e} (\mathbf{r}_m)_{\mathbf{T}_{e,i}} \cdot S^{(e)}_{ij} \cdot (\mathbf{f}_{k_m})_{\mathbf{T}_{e,j}}$ Equivalently: $J_{me} = (\mathbf{r}_m[\mathbf{T}_e])^\top \mathbf{S}^{(e)} (\mathbf{f}_{k_m}[\mathbf{T}_e])$ |
| `inputs` | $\mathbf{R}_i$: $(M, N)$, $\mathbf{S}^{(e)}$: $(E, 4, 4)$, $\mathbf{F}$: $(n_{el}, N)$, $\mathbf{E}_{meas}$: $(M, 3)$, $\mathbf{T}$: $(E, 4)$ |
| `outputs` | $\mathbf{J}$: $(M, E)$ |
| `computational_note` | This implements the EIT sensitivity theorem: $\frac{\partial v}{\partial \sigma_e} = -\int_{\Omega_e} \nabla u^+ \cdot \nabla u^- \, d\Omega$ |

#### Step I.6: Compute Regularization Matrix

| Field | Value |
|-------|-------|
| `step_name` | Build regularization matrix based on method |
| `formula` | $\mathbf{G} = \mathbf{J}^\top \mathbf{J}$. Then: $R_{ii} = \begin{cases} G_{ii}^p & \text{method} = \texttt{kotre} \\ G_{ii} & \text{method} = \texttt{lm} \\ 1 & \text{method} = \texttt{dgn} \end{cases}$, $R_{ij} = 0$ for $i \neq j$ |
| `inputs` | $\mathbf{J}$: $(M, E)$, $p$: scalar, method: string |
| `outputs` | $\mathbf{R}$: $(E, E)$ diagonal |
| `computational_note` | Diagonal extraction: `np.diag(np.diag(J.T @ J))` |

#### Step I.7: Compute Reconstruction Matrix

| Field | Value |
|-------|-------|
| `step_name` | Compute pseudo-inverse reconstruction operator |
| `formula` | $\mathbf{H} = (\mathbf{J}^\top\mathbf{J} + \lambda\mathbf{R})^{-1}\mathbf{J}^\top$ |
| `inputs` | $\mathbf{J}$: $(M, E)$, $\mathbf{R}$: $(E, E)$, $\lambda$: scalar |
| `outputs` | $\mathbf{H}$: $(E, M)$ |
| `computational_note` | Dense matrix inversion using `scipy.linalg.inv` |

**END INITIALIZATION**

---

### Iteration Loop

**BEGIN ITERATION LOOP**

This algorithm uses a single-step (one-shot) linearized reconstruction. There is no iterative loop.

#### Step S.1: Compute Data Difference

| Field | Value |
|-------|-------|
| `step_name` | Calculate voltage difference vector |
| `formula` | If `normalize=False`: $\Delta\mathbf{v} = \mathbf{v}^{(1)} - \mathbf{v}^{(0)}$. If `normalize=True`: $\Delta v_m = \ln\left(\frac{|v^{(1)}_m|}{|v^{(0)}_m|}\right) \cdot \text{sign}(\Re(v^{(0)}_m))$ |
| `inputs` | $\mathbf{v}^{(0)}$: $(M,)$, $\mathbf{v}^{(1)}$: $(M,)$, normalize: bool |
| `outputs` | $\Delta\mathbf{v}$: $(M,)$ |
| `computational_note` | Normalized difference provides scale-invariance for varying contact impedances |

#### Step S.2: Apply Reconstruction Operator

| Field | Value |
|-------|-------|
| `step_name` | Reconstruct conductivity change |
| `formula` | $\Delta\boldsymbol{\sigma} = -\mathbf{H}\Delta\mathbf{v}$ |
| `inputs` | $\mathbf{H}$: $(E, M)$, $\Delta\mathbf{v}$: $(M,)$ |
| `outputs` | $\Delta\boldsymbol{\sigma}$: $(E,)$ |
| `computational_note` | Negative sign because $\mathbf{J}$ represents $\partial v / \partial \sigma$ and increased conductivity decreases voltage |

**END ITERATION LOOP**

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| `type` | `single_step` (non-iterative) |
| `expression` | N/A - Algorithm terminates after single matrix-vector multiplication |
| `parameters` | None |

**Interpretation**: This is a linearized one-step reconstruction method. The solution is obtained directly via the pre-computed reconstruction matrix $\mathbf{H}$. No iterative convergence is required.

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Minimum Conductivity Change | $\Delta\sigma_{\min} = \min_e \Re(\Delta\sigma_e)$ | Minimum reconstructed change across elements |
| Maximum Conductivity Change | $\Delta\sigma_{\max} = \max_e \Re(\Delta\sigma_e)$ | Maximum reconstructed change across elements |
| Mean Conductivity Change | $\bar{\Delta\sigma} = \frac{1}{E}\sum_{e=1}^{E}\Re(\Delta\sigma_e)$ | Average reconstructed change |
| Standard Deviation | $\text{std}(\Delta\boldsymbol{\sigma}) = \sqrt{\frac{1}{E}\sum_{e=1}^{E}(\Re(\Delta\sigma_e) - \bar{\Delta\sigma})^2}$ | Spread of reconstructed values |
| Data Residual Norm | $\|\Delta\mathbf{v}\|_2 = \sqrt{\sum_{m=1}^{M}(v^{(1)}_m - v^{(0)}_m)^2}$ | Magnitude of measurement change |
| Relative Voltage Change | $\frac{\|\Delta\mathbf{v}\|_2}{\|\mathbf{v}^{(0)}\|_2}$ | Normalized measurement perturbation |
| Maximum Change Element | $e^* = \arg\max_e |\Re(\Delta\sigma_e)|$ | Element index with largest absolute change |
| Maximum Change Value | $\Delta\sigma_{e^*}$ | Conductivity change at element $e^*$ |
| Node-Interpolated Values | $(\Delta\sigma_{node})_n = \frac{\sum_{e: n \in \mathbf{T}_e} \Delta\sigma_e}{\max(|\{e: n \in \mathbf{T}_e\}|, 1)}$ | Element-to-node averaging for visualization |