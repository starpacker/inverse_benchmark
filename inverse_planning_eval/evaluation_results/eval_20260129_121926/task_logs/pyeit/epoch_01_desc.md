# Algorithmic Specification Document: 3D EIT Jacobian-Based Reconstruction

## 1. Mathematical Formulation

### 1.1 Forward Problem (Complete-Electrode Model)

The forward problem solves the elliptic PDE governing electrical conduction:

$$\nabla \cdot (\sigma(\mathbf{r}) \nabla u(\mathbf{r})) = 0, \quad \mathbf{r} \in \Omega \subset \mathbb{R}^3$$

where $\sigma(\mathbf{r})$ is the conductivity distribution and $u(\mathbf{r})$ is the electric potential.

**FEM Discretization:** The weak form yields a linear system:

$$\mathbf{K}(\boldsymbol{\sigma}) \boldsymbol{\phi} = \mathbf{b}$$

where:
- Global stiffness matrix: $\mathbf{K} = \sum_{e=1}^{N_e} \sigma_e \mathbf{K}_e^{(loc)}$
- Local stiffness matrix for tetrahedron $e$: 
$$K_{ij}^{(loc)} = \frac{\mathbf{a}_i \cdot \mathbf{a}_j}{36 V_e}$$
where $\mathbf{a}_i = (-1)^{i+1}(\mathbf{s}_{i_1} \times \mathbf{s}_{i_2})$ are gradient coefficients derived from edge vectors, and $V_e = \frac{1}{6}|\det[\mathbf{s}_0, \mathbf{s}_1, \mathbf{s}_2]|$

**Boundary Voltage Extraction:**
$$v_k = \phi_{m_k} - \phi_{n_k}$$
for measurement pair $(m_k, n_k)$ under excitation pattern $k$.

---

### 1.2 Inverse Problem Formulation

**Objective Function:**

$$\min_{\Delta\boldsymbol{\sigma}} \mathcal{L}(\Delta\boldsymbol{\sigma}) = \underbrace{\|\mathbf{v}_1 - \mathbf{v}_0 - \mathbf{J}\Delta\boldsymbol{\sigma}\|_2^2}_{\text{Data Fidelity}} + \underbrace{\lambda \cdot \Delta\boldsymbol{\sigma}^T \mathbf{R} \Delta\boldsymbol{\sigma}}_{\text{Regularization}}$$

**Linearized Approximation:**
$$\Delta\mathbf{v} = \mathbf{v}_1 - \mathbf{v}_0 \approx \mathbf{J} \Delta\boldsymbol{\sigma}$$

**Closed-Form Solution (Tikhonov):**
$$\Delta\hat{\boldsymbol{\sigma}} = -(\mathbf{J}^T\mathbf{J} + \lambda\mathbf{R})^{-1}\mathbf{J}^T \Delta\mathbf{v}$$

**Regularization Matrix Options:**

| Method | $\mathbf{R}$ Formula |
|--------|---------------------|
| `kotre` | $R_{ii} = ([\mathbf{J}^T\mathbf{J}]_{ii})^p$ |
| `lm` (Levenberg-Marquardt) | $R_{ii} = [\mathbf{J}^T\mathbf{J}]_{ii}$ |
| `dgn` (Damped Gauss-Newton) | $\mathbf{R} = \mathbf{I}$ |

---

### 1.3 Jacobian Computation

The Jacobian $\mathbf{J} \in \mathbb{R}^{N_m \times N_e}$ relates conductivity perturbations to voltage changes:

$$J_{k,e} = \frac{\partial v_k}{\partial \sigma_e} = \sum_{i,j \in \text{element } e} r_{k,i} \cdot K_{ij}^{(loc)} \cdot \phi_{k,j}$$

where:
- $\boldsymbol{\phi}_k$ = potential field for excitation pattern $k$
- $\mathbf{r}_k = (\mathbf{K}^{-1})_{m_k,:} - (\mathbf{K}^{-1})_{n_k,:}$ = sensitivity row for measurement $(m_k, n_k)$

**Explicit Formula:**
$$J_{k,e} = \mathbf{r}_{k,\mathcal{I}_e}^T \mathbf{K}_e^{(loc)} \boldsymbol{\phi}_{k,\mathcal{I}_e}$$
where $\mathcal{I}_e$ denotes node indices of element $e$.

---

## 2. Variables Taxonomy

### 2.1 Optimization/Primal Variables

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `ds` | $\Delta\boldsymbol{\sigma}$ | $(N_e,)$ | `float64` | Reconstructed element-wise conductivity change |

### 2.2 Auxiliary/Intermediate Variables

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `f` / `f_baseline` | $\boldsymbol{\Phi}$ | $(N_{exc}, N_{nodes})$ | `float64` | Nodal potentials for each excitation |
| `kg` | $\mathbf{K}$ | $(N_{nodes}, N_{nodes})$ | `csr_matrix` | Global stiffness matrix (sparse) |
| `jac` | $\mathbf{J}$ | $(N_m, N_e)$ | `float64` | Jacobian/sensitivity matrix |
| `H` | $\mathbf{H}$ | $(N_e, N_m)$ | `float64` | Inverse operator $(\mathbf{J}^T\mathbf{J}+\lambda\mathbf{R})^{-1}\mathbf{J}^T$ |
| `r_mat` | $\mathbf{K}^{-1}_{el\_pos,:}$ | $(N_{el}, N_{nodes})$ | `float64` | Rows of inverse stiffness at electrodes |
| `ri` | $\mathbf{r}_k$ | $(N_m, N_{nodes})$ | `float64` | Differenced sensitivity rows |
| `se` / `ke_array` | $\{\mathbf{K}_e^{(loc)}\}$ | $(N_e, 4, 4)$ | `float64` | Local stiffness matrices (precomputed) |

### 2.3 Hyperparameters/Constants

| Variable | Symbol | Default | Dtype | Description |
|----------|--------|---------|-------|-------------|
| `lamb` | $\lambda$ | `0.001` | `float` | Tikhonov regularization weight |
| `p` | $p$ | `0.20` | `float` | Kotre method exponent |
| `method` | — | `"kotre"` | `str` | Regularization method: `{"kotre", "lm", "dgn"}` |
| `normalize` | — | `False` | `bool` | Use log-ratio normalization |
| `n_el` | $N_{el}$ | `16` | `int` | Number of electrodes |
| `h0` | $h_0$ | `0.2` | `float` | Mesh refinement parameter |
| `dist_exc` | — | `7` | `int` | Electrode skip for excitation |
| `step_meas` | — | `1` | `int` | Electrode skip for measurement |

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `v0` | $\mathbf{v}_0$ | $(N_m,)$ | `float64` | Baseline boundary voltages |
| `v1` | $\mathbf{v}_1$ | $(N_m,)$ | `float64` | Perturbed boundary voltages |
| `pts` | $\mathbf{P}$ | $(N_{nodes}, 3)$ | `float64` | Node coordinates |
| `tri` | $\mathbf{T}$ | $(N_e, 4)$ | `int` | Tetrahedron connectivity |
| `perm_baseline` | $\boldsymbol{\sigma}_0$ | $(N_e,)$ | `float64` | Baseline conductivity (typically $\mathbf{1}$) |
| `ex_mat` | — | $(N_{exc}, 2)$ | `int` | Excitation electrode pairs |
| `meas_mat` | — | $(N_m, 3)$ | `int` | Measurement pattern $[n, m, exc\_id]$ |

---

## 3. Preprocessing Pipeline

| Step | Operation | Input | Output | Formula/Description |
|------|-----------|-------|--------|---------------------|
| **P1** | Mesh Generation | $N_{el}, h_0, \text{bbox}$ | `mesh_obj` | Create 3D tetrahedral mesh in unit ball domain |
| **P2** | Excitation Pattern | $N_{el}, \text{dist\_exc}$ | `ex_mat` $\in \mathbb{Z}^{N_{exc} \times 2}$ | $\text{ex\_mat}[i] = [i, (i + \text{dist}) \mod N_{el}]$ |
| **P3** | Measurement Pattern | `ex_mat`, $N_{el}$, `step` | `meas_mat`, `keep_ba` | $m = [0..N_{el}-1], n = (m+\text{step}) \mod N_{el}$; exclude electrodes in excitation pair |
| **P4** | Local Stiffness | `pts`, `tri` | `se` $\in \mathbb{R}^{N_e \times 4 \times 4}$ | $K_{ij}^{(loc)} = \frac{\mathbf{a}_i \cdot \mathbf{a}_j}{36 V_e}$ for each tetrahedron |
| **P5** | Global Assembly (baseline) | `se`, `tri`, $\boldsymbol{\sigma}_0$ | `kg_baseline` | $\mathbf{K} = \sum_e \sigma_e \mathbf{K}_e^{(loc)}$ with Dirichlet BC at `ref_node` |
| **P6** | RHS Construction | `ex_mat`, `el_pos` | `b` $\in \mathbb{R}^{N_{exc} \times N_{nodes}}$ | $b[k, \text{el\_pos}[\text{ex\_mat}[k]]] = [+1, -1]$ |
| **P7** | Forward Solve (baseline) | `kg_baseline`, `b` | `f_baseline` | $\boldsymbol{\phi}_k = \mathbf{K}^{-1}\mathbf{b}_k$ for each $k$ |
| **P8** | Voltage Extraction | `f_baseline`, `meas_mat` | `v0` | $v_0[k] = \phi_{exc[k]}[n_k] - \phi_{exc[k]}[m_k]$ |
| **P9** | Anomaly Perturbation | `mesh_obj`, anomaly params | `perm_anomaly` | Set $\sigma_e = \sigma_{anomaly}$ for elements in ball region |
| **P10** | Forward Solve (perturbed) | Repeat P5-P8 with `perm_anomaly` | `v1` | Perturbed measurements |

---

## 4. Algorithm Steps (JAC Inversion)

### Initialization Phase

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| **I1** | Load Preprocessed Data | — | All preprocessing outputs | `mesh_obj`, `protocol_obj`, `se`, `v0`, `v1`, `perm_baseline` |
| **I2** | Initialize conductivity | $\boldsymbol{\sigma} \leftarrow \boldsymbol{\sigma}_0$ | `perm_baseline` | `perm` $\in \mathbb{R}^{N_e}$ |

### Jacobian Computation Phase

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| **J1** | Assemble Stiffness | $\mathbf{K} = \sum_{e=1}^{N_e} \sigma_e \mathbf{K}_e^{(loc)}$ | `se`, `tri`, `perm` | `kg` (sparse) |
| **J2** | Construct RHS | $\mathbf{b}[k, \text{el\_pos}[\text{ex}[k]]] = [+1, -1]$ | `ex_mat`, `el_pos` | `b` $\in \mathbb{R}^{N_{exc} \times N_{nodes}}$ |
| **J3** | Forward Solve | $\boldsymbol{\phi}_k = \text{spsolve}(\mathbf{K}, \mathbf{b}_k)$ | `kg`, `b` | `f` $\in \mathbb{R}^{N_{exc} \times N_{nodes}}$ |
| **J4** | Compute Inverse Kernel | $\mathbf{R}_{mat} = (\mathbf{K}^{-1})_{\text{el\_pos},:}$ | `kg`, `el_pos` | `r_mat` $\in \mathbb{R}^{N_{el} \times N_{nodes}}$ |
| **J5** | Expand to Excitations | $\mathbf{R}_{el}[k] = \mathbf{R}_{mat}$ for all $k$ | `r_mat`, $N_{exc}$ | `r_el` $\in \mathbb{R}^{N_{exc} \times N_{el} \times N_{nodes}}$ |
| **J6** | Difference for Measurements | $\mathbf{r}_i = \mathbf{R}_{el}[\text{exc}_i, n_i, :] - \mathbf{R}_{el}[\text{exc}_i, m_i, :]$ | `r_el`, `meas_mat` | `ri` $\in \mathbb{R}^{N_m \times N_{nodes}}$ |
| **J7** | Jacobian Assembly | $J_{k,e} = \sum_{i,j \in e} r_{k,i} \cdot K_{ij}^{(loc)} \cdot \phi_{exc[k],j}$ | `ri`, `se`, `f`, `tri` | `jac` $\in \mathbb{R}^{N_m \times N_e}$ |

### Regularized Inversion Phase

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| **R1** | Compute JᵀJ | $\mathbf{G} = \mathbf{J}^T \mathbf{J}$ | `jac` | `j_w_j` $\in \mathbb{R}^{N_e \times N_e}$ |
| **R2** | Build Regularization | $\mathbf{R} = \text{diag}(\text{diag}(\mathbf{G})^p)$ (Kotre) | `j_w_j`, $p$ | `r_mat_reg` $\in \mathbb{R}^{N_e \times N_e}$ |
| **R3** | Compute Inverse Operator | $\mathbf{H} = (\mathbf{G} + \lambda\mathbf{R})^{-1}\mathbf{J}^T$ | `j_w_j`, `r_mat_reg`, `jac`, $\lambda$ | `H` $\in \mathbb{R}^{N_e \times N_m}$ |
| **R4** | Compute Data Difference | $\Delta\mathbf{v} = \mathbf{v}_1 - \mathbf{v}_0$ (standard) | `v1`, `v0` | `dv` $\in \mathbb{R}^{N_m}$ |
| **R4'** | (Alternative) Normalized | $\Delta\mathbf{v} = \log(|\mathbf{v}_1|/|\mathbf{v}_0|) \cdot \text{sign}(\mathbf{v}_0)$ | `v1`, `v0` | `dv` $\in \mathbb{R}^{N_m}$ |
| **R5** | Solve Inverse | $\Delta\boldsymbol{\sigma} = -\mathbf{H} \cdot \Delta\mathbf{v}$ | `H`, `dv` | `ds` $\in \mathbb{R}^{N_e}$ |

---

## 5. Stopping Criterion

This implementation uses a **single-step linearized reconstruction** (one Gauss-Newton iteration). No iterative refinement is performed.

| Criterion | Condition | Value |
|-----------|-----------|-------|
| Iteration Limit | $k = 1$ | Fixed (non-iterative) |
| Convergence Check | Not applicable | — |

**Note:** For iterative variants, typical criteria would be:
- $\|\Delta\boldsymbol{\sigma}^{(k)} - \Delta\boldsymbol{\sigma}^{(k-1)}\|_2 < \epsilon_{tol}$
- $\|\mathbf{v}_1 - \mathcal{F}(\boldsymbol{\sigma}_0 + \Delta\boldsymbol{\sigma}^{(k)})\|_2 < \epsilon_{data}$
- $k > k_{max}$

---

## 6. Evaluation Metrics

| Metric | Symbol | Formula | Description |
|--------|--------|---------|-------------|
| **Data Residual Norm** | $\|\Delta\mathbf{v}\|_2$ | $\sqrt{\sum_{i=1}^{N_m}(v_1^{(i)} - v_0^{(i)})^2}$ | Magnitude of voltage change |
| **Relative Voltage Change** | $\eta$ | $\frac{\|\mathbf{v}_1 - \mathbf{v}_0\|_2}{\|\mathbf{v}_0\|_2}$ | Normalized perturbation strength |
| **Min Conductivity Change** | $\Delta\sigma_{min}$ | $\min_e \text{Re}(\Delta\sigma_e)$ | Minimum reconstructed change |
| **Max Conductivity Change** | $\Delta\sigma_{max}$ | $\max_e \text{Re}(\Delta\sigma_e)$ | Maximum reconstructed change |
| **Mean Conductivity Change** | $\bar{\Delta\sigma}$ | $\frac{1}{N_e}\sum_{e=1}^{N_e}\text{Re}(\Delta\sigma_e)$ | Average change |
| **Std Conductivity Change** | $s_{\Delta\sigma}$ | $\sqrt{\frac{1}{N_e}\sum_{e=1}^{N_e}(\text{Re}(\Delta\sigma_e) - \bar{\Delta\sigma})^2}$ | Spread of reconstruction |
| **Max Change Element** | $e^*$ | $\arg\max_e |\text{Re}(\Delta\sigma_e)|$ | Element with largest absolute change |
| **Forward Consistency** | $\epsilon_{fwd}$ | $\|\mathcal{F}(\boldsymbol{\sigma}_0) - \mathbf{v}_0\|_2$ | Verification of forward model (should be $\approx 0$) |

---

## 7. Algorithm Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PHASE                          │
├─────────────────────────────────────────────────────────────────┤
│  Mesh Generation → Protocol Setup → Local Stiffness Ke          │
│  → Baseline Forward Solve → v₀                                  │
│  → Anomaly Forward Solve → v₁                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    JACOBIAN COMPUTATION                         │
├─────────────────────────────────────────────────────────────────┤
│  K(σ₀) Assembly → Forward Solve Φ → K⁻¹ Computation             │
│  → Sensitivity ri → Jₖₑ = riᵀ Kₑ Φₑ                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 REGULARIZED INVERSION                           │
├─────────────────────────────────────────────────────────────────┤
│  JᵀJ → R (Kotre/LM/DGN) → H = (JᵀJ + λR)⁻¹Jᵀ                   │
│  → Δv = v₁ - v₀ → Δσ = -H·Δv                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATION                                 │
├─────────────────────────────────────────────────────────────────┤
│  Statistics → Visualization → Metrics Export                    │
└─────────────────────────────────────────────────────────────────┘
```