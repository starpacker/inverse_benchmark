## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: ADMM Phase Unwrapping
- **algorithm_family**: Alternating Direction Method of Multipliers (ADMM) / Proximal Splitting
- **summary**: Recovers the unwrapped phase from a wrapped interferogram by minimizing a generalized p-norm penalty on the gradient residual using variable splitting and DCT-based Poisson solvers.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{F} \| \nabla F - \boldsymbol{\phi} \|_p$$

Reformulated via ADMM variable splitting:
$$\min_{F, \mathbf{w}} \| \mathbf{w} \|_p \quad \text{subject to} \quad \nabla F - \boldsymbol{\phi} - \mathbf{w} = \mathbf{0}$$

Augmented Lagrangian:
$$\mathcal{L}(F, \mathbf{w}, \boldsymbol{\Lambda}) = \|\mathbf{w}\|_p + \langle \boldsymbol{\Lambda}, \nabla F - \boldsymbol{\phi} - \mathbf{w} \rangle + \frac{\rho}{2}\|\nabla F - \boldsymbol{\phi} - \mathbf{w}\|_2^2$$

where $\mathbf{w} = (w_x, w_y)$, $\boldsymbol{\Lambda} = (\Lambda_x, \Lambda_y)$, $\boldsymbol{\phi} = (\phi_x, \phi_y)$, and $\nabla F = (D_x F, D_y F)$.

### data_fidelity_term
$$\| \nabla F - \boldsymbol{\phi} \|_p = \left( \sum_{i,j} \left[ (F_x - \phi_x)_{i,j}^2 + (F_y - \phi_y)_{i,j}^2 \right]^{p/2} \right)$$

where $F_x = D_x F$ and $F_y = D_y F$ are discrete gradients. The p-norm is applied to the 2D gradient vector at each pixel.

### regularization_term
None (the objective is purely data fidelity to wrapped gradients).

### constraints
None explicit. The ADMM penalty parameter $\rho$ is implicitly set to 1.0 (absorbed into scaled dual variables).

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $F$ | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Unwrapped phase estimate |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Lambda_x$ | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Scaled dual variable for x-gradient constraint |
| $\Lambda_y$ | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Scaled dual variable for y-gradient constraint |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $w_x$ | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Auxiliary split variable for x-gradient |
| $w_y$ | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Auxiliary split variable for y-gradient |
| $F_{\text{old}}$ | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Previous iterate for convergence check |
| $F_x$ | $(M, N)$ | float32 | Computed | Current x-gradient of $F$ |
| $F_y$ | $(M, N)$ | float32 | Computed | Current y-gradient of $F$ |
| $r_x$ | $(MN,)$ | float32 | Computed | RHS component for x in Poisson solve |
| $r_y$ | $(MN,)$ | float32 | Computed | RHS component for y in Poisson solve |
| $\text{RHS}$ | $(MN,)$ | float32 | Computed | Full right-hand side for Poisson equation |
| $\hat{\rho}$ | $(M, N)$ | float32 | Computed | DCT of RHS |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `max_iters` | 50 | Maximum ADMM iterations |
| `tol` | 0.08 | Convergence tolerance on max absolute change |
| $\lambda$ | 2.0 | Shrinkage threshold parameter |
| $p$ | 0.01 | Norm exponent (0 = L0, 1 = L1) |
| $c$ | 1.6 | Dual variable update scaling factor |
| $\epsilon$ | 0 | Mollification parameter for p-shrinkage |
| `boundary_conditions` | "neumann" | Boundary condition type |
| `dtype` | "float32" | Numerical precision |
| `debug` | True | Print iteration information |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $I$ | $(M, N)$ | complex64 | Raw complex interferogram |
| $f_{\text{wrapped}}$ | $(M, N)$ | float32 | Wrapped phase $= \arg(I)$ |
| $\text{mag}$ | $(M, N)$ | float32 | Interferogram magnitude $= |I|$ |
| $\phi_x$ | $(M, N)$ | float32 | Wrapped gradient in x-direction |
| $\phi_y$ | $(M, N)$ | float32 | Wrapped gradient in y-direction |
| $D_x$ | $(MN, MN)$ | sparse float32 | Forward difference operator in x |
| $D_y$ | $(MN, MN)$ | sparse float32 | Forward difference operator in y |
| $K$ | $(M, N)$ | float32 | Inverse Laplacian eigenvalues for DCT solver |
| $M$ | scalar | int | Number of rows |
| $N$ | scalar | int | Number of columns |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Load Interferogram
- **step_order**: 0
- **step_name**: Load Complex Interferogram
- **formula**: $I = \text{fromfile}(\text{filename}, \text{complex64}).\text{reshape}(-1, N)$
- **inputs**: filename (str), columns $N$ from RSC file
- **outputs**: $I \in \mathbb{C}^{M \times N}$

### Step 1: Extract Wrapped Phase and Magnitude
- **step_order**: 1
- **step_name**: Phase and Magnitude Extraction
- **formula**: 
  - $\text{mag}_{i,j} = |I_{i,j}|$
  - $f_{\text{wrapped}, i,j} = \arg(I_{i,j}) = \text{atan2}(\text{Im}(I_{i,j}), \text{Re}(I_{i,j}))$
- **inputs**: $I \in \mathbb{C}^{M \times N}$
- **outputs**: $\text{mag} \in \mathbb{R}^{M \times N}$, $f_{\text{wrapped}} \in [-\pi, \pi]^{M \times N}$

### Step 2: Construct Differentiation Matrices
- **step_order**: 2
- **step_name**: Build Sparse Gradient Operators (Neumann BC)
- **formula**: 
  - Base 1D forward difference: $D^{(1)} \in \mathbb{R}^{n \times n}$ with $(D^{(1)})_{i,i} = -1$, $(D^{(1)})_{i,i+1} = +1$
  - Neumann modification: $(D^{(1)})_{n-1,n-1} = 0$ (zero flux at boundary)
  - $D_x = I_M \otimes D^{(1)}_N \in \mathbb{R}^{MN \times MN}$
  - $D_y = D^{(1)}_M \otimes I_N \in \mathbb{R}^{MN \times MN}$
- **inputs**: $M$, $N$, boundary_conditions="neumann"
- **outputs**: $D_x$, $D_y$ (sparse CSR matrices)

### Step 3: Estimate Wrapped Gradients
- **step_order**: 3
- **step_name**: Wrapped Gradient Estimation with Phase Unwrap Correction
- **formula**:
  - Raw gradient: $\tilde{\phi}_x = D_x \cdot \text{vec}(f_{\text{wrapped}})$, reshaped to $(M, N)$
  - Raw gradient: $\tilde{\phi}_y = D_y \cdot \text{vec}(f_{\text{wrapped}})$, reshaped to $(M, N)$
  - Wrap correction for x:
    $$\phi_{x,i,j} = \begin{cases} \tilde{\phi}_{x,i,j} - 2\pi \cdot \text{sign}(\tilde{\phi}_{x,i,j}) & \text{if } |\tilde{\phi}_{x,i,j}| > \pi \\ \tilde{\phi}_{x,i,j} & \text{otherwise} \end{cases}$$
  - Wrap correction for y:
    $$\phi_{y,i,j} = \begin{cases} \tilde{\phi}_{y,i,j} - 2\pi \cdot \text{sign}(\tilde{\phi}_{y,i,j}) & \text{if } |\tilde{\phi}_{y,i,j}| > \pi \\ \tilde{\phi}_{y,i,j} & \text{otherwise} \end{cases}$$
- **inputs**: $f_{\text{wrapped}} \in \mathbb{R}^{M \times N}$, $D_x$, $D_y$
- **outputs**: $\phi_x, \phi_y \in \mathbb{R}^{M \times N}$

### Step 4: Compute Inverse Laplacian Kernel
- **step_order**: 4
- **step_name**: DCT-Domain Laplacian Inverse (Eigenvalues)
- **formula**:
  - $\xi_y^{(m)} = 2 - 2\cos\left(\frac{\pi m}{M}\right)$ for $m = 0, \ldots, M-1$
  - $\xi_x^{(n)} = 2 - 2\cos\left(\frac{\pi n}{N}\right)$ for $n = 0, \ldots, N-1$
  - Laplacian eigenvalues: $\Lambda_{m,n} = \xi_y^{(m)} + \xi_x^{(n)}$
  - Inverse kernel: $K_{m,n} = \begin{cases} 0 & \text{if } \Lambda_{m,n} = 0 \\ 1/\Lambda_{m,n} & \text{otherwise} \end{cases}$
- **inputs**: $M$, $N$
- **outputs**: $K \in \mathbb{R}^{M \times N}$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Operation | Formula | Output |
|------|-----------|---------|--------|
| I1 | Initialize primal | $F^{(0)} = \mathbf{0}_{M \times N}$ | $F^{(0)}$ |
| I2 | Initialize auxiliary x | $w_x^{(0)} = \mathbf{0}_{M \times N}$ | $w_x^{(0)}$ |
| I3 | Initialize auxiliary y | $w_y^{(0)} = \mathbf{0}_{M \times N}$ | $w_y^{(0)}$ |
| I4 | Initialize dual x | $\Lambda_x^{(0)} = \mathbf{0}_{M \times N}$ | $\Lambda_x^{(0)}$ |
| I5 | Initialize dual y | $\Lambda_y^{(0)} = \mathbf{0}_{M \times N}$ | $\Lambda_y^{(0)}$ |
| I6 | Initialize tracking | $F_{\text{old}}^{(0)} = \mathbf{0}_{M \times N}$ | $F_{\text{old}}^{(0)}$ |
| I7 | Set iteration counter | $k = 0$ | $k$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, \ldots, \text{max\_iters}-1$)

#### Step 1: Compute RHS for Poisson Equation
- **step_name**: Assemble Poisson RHS
- **formula**:
  $$r_x = \text{vec}(w_x^{(k)}) + \text{vec}(\phi_x) - \text{vec}(\Lambda_x^{(k)})$$
  $$r_y = \text{vec}(w_y^{(k)}) + \text{vec}(\phi_y) - \text{vec}(\Lambda_y^{(k)})$$
  $$\text{RHS} = D_x^T r_x + D_y^T r_y$$
- **inputs**: $w_x^{(k)}, w_y^{(k)}, \phi_x, \phi_y, \Lambda_x^{(k)}, \Lambda_y^{(k)}, D_x, D_y$
- **outputs**: $\text{RHS} \in \mathbb{R}^{MN}$
- **computational_note**: [ADJOINT] operators $D_x^T$, $D_y^T$ applied

#### Step 2: Solve Poisson Equation via DCT
- **step_name**: F-Subproblem (Poisson Solve)
- **formula**:
  $$\hat{\rho} = \text{DCT-II}(\text{reshape}(\text{RHS}, (M, N)))$$
  $$F^{(k+1)} = \text{IDCT-II}(\hat{\rho} \odot K)$$
  where DCT-II uses orthonormal normalization.
- **inputs**: $\text{RHS} \in \mathbb{R}^{MN}$, $K \in \mathbb{R}^{M \times N}$
- **outputs**: $F^{(k+1)} \in \mathbb{R}^{M \times N}$
- **computational_note**: Solves $-\nabla^2 F = \nabla \cdot (\mathbf{w} + \boldsymbol{\phi} - \boldsymbol{\Lambda})$ with Neumann BC

#### Step 3: Compute Gradients of Updated F
- **step_name**: Forward Gradient Computation
- **formula**:
  $$F_x^{(k+1)} = \text{reshape}(D_x \cdot \text{vec}(F^{(k+1)}), (M, N))$$
  $$F_y^{(k+1)} = \text{reshape}(D_y \cdot \text{vec}(F^{(k+1)}), (M, N))$$
- **inputs**: $F^{(k+1)}$, $D_x$, $D_y$
- **outputs**: $F_x^{(k+1)}, F_y^{(k+1)} \in \mathbb{R}^{M \times N}$
- **computational_note**: [FORWARD] operators applied

#### Step 4: Compute Shrinkage Input
- **step_name**: Form Shrinkage Arguments
- **formula**:
  $$u_x = F_x^{(k+1)} - \phi_x + \Lambda_x^{(k)}$$
  $$u_y = F_y^{(k+1)} - \phi_y + \Lambda_y^{(k)}$$
- **inputs**: $F_x^{(k+1)}, F_y^{(k+1)}, \phi_x, \phi_y, \Lambda_x^{(k)}, \Lambda_y^{(k)}$
- **outputs**: $u_x, u_y \in \mathbb{R}^{M \times N}$

#### Step 5: w-Subproblem (p-Shrinkage)
- **step_name**: Proximal Operator for p-Norm
- **definition**: The p-shrinkage operator computes:
  $$\text{prox}_{\lambda \|\cdot\|_p}(\mathbf{u}) = \arg\min_{\mathbf{z}} \|\mathbf{z}\|_p + \frac{1}{2\lambda^{2-p}}\|\mathbf{z} - \mathbf{u}\|_2^2$$
- **closed_form_solution**:
  - Stack inputs: $X = \begin{bmatrix} u_x \\ u_y \end{bmatrix} \in \mathbb{R}^{2 \times M \times N}$
  - Compute magnitude per pixel: $\text{mag}_{i,j} = \sqrt{u_{x,i,j}^2 + u_{y,i,j}^2}$
  - Handle zeros: $\text{nonzero}_{i,j} = \begin{cases} 1 & \text{if } \text{mag}_{i,j} = 0 \\ \text{mag}_{i,j} & \text{otherwise} \end{cases}$
  - Shrinkage factor:
    $$s_{i,j} = \frac{\max\left(\text{mag}_{i,j} - \lambda^{2-p}(\text{nonzero}_{i,j}^2 + \epsilon)^{p/2 - 0.5}, 0\right)}{\text{nonzero}_{i,j}}$$
  - Apply: $(w_x^{(k+1)}, w_y^{(k+1)}) = s \odot (u_x, u_y)$
- **inputs**: $u_x, u_y$, $\lambda=2.0$, $p=0.01$, $\epsilon=0$
- **outputs**: $w_x^{(k+1)}, w_y^{(k+1)} \in \mathbb{R}^{M \times N}$

#### Step 6: Dual Variable Update
- **step_name**: Lagrange Multiplier Update
- **formula**:
  $$\Lambda_x^{(k+1)} = \Lambda_x^{(k)} + c \cdot (F_x^{(k+1)} - \phi_x - w_x^{(k+1)})$$
  $$\Lambda_y^{(k+1)} = \Lambda_y^{(k)} + c \cdot (F_y^{(k+1)} - \phi_y - w_y^{(k+1)})$$
- **inputs**: $\Lambda_x^{(k)}, \Lambda_y^{(k)}, F_x^{(k+1)}, F_y^{(k+1)}, \phi_x, \phi_y, w_x^{(k+1)}, w_y^{(k+1)}$, $c=1.6$
- **outputs**: $\Lambda_x^{(k+1)}, \Lambda_y^{(k+1)}$
- **computational_note**: Scaled form; effective ADMM penalty $\rho = 1/c$

#### Step 7: Convergence Check
- **step_name**: Compute Change Metric
- **formula**:
  $$\Delta^{(k+1)} = \max_{i,j} |F^{(k+1)}_{i,j} - F^{(k)}_{i,j}|$$
- **inputs**: $F^{(k+1)}$, $F_{\text{old}}^{(k)}$
- **outputs**: $\Delta^{(k+1)}$ (scalar)

#### Step 8: Update Tracking Variable
- **step_name**: Store Previous Iterate
- **formula**: $F_{\text{old}}^{(k+1)} = F^{(k+1)}$ (deep copy)
- **inputs**: $F^{(k+1)}$
- **outputs**: $F_{\text{old}}^{(k+1)}$

**END_LOOP**

### 4.3 FINALIZATION

| Step | Operation | Description |
|------|-----------|-------------|
| F1 | Return solution | Output $F^{(k_{\text{final}})}$ as unwrapped phase |
| F2 | Save output | `np.save(io_dir/output.npy, F)` |
| F3 | Optional format conversion | Save as GeoTIFF (.tif) or binary (.unw) with magnitude |

---

## 5. STOPPING CRITERION

- **type**: combined
- **expression**: 
  $$\text{STOP if } \left( \Delta^{(k+1)} < \text{tol} \right) \lor \left( \text{isnan}(\Delta^{(k+1)}) \right) \lor \left( k \geq \text{max\_iters} - 1 \right)$$
- **parameters**:
  - `tol` = 0.08
  - `max_iters` = 50
- **check_frequency**: Every iteration (after Step 7)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Phase Range | $[\min_{i,j} F_{i,j}, \max_{i,j} F_{i,j}]$ | Bounds of unwrapped phase |
| Mean Phase | $\bar{F} = \frac{1}{MN}\sum_{i,j} F_{i,j}$ | Average unwrapped phase value |
| Std Phase | $\sigma_F = \sqrt{\frac{1}{MN}\sum_{i,j}(F_{i,j} - \bar{F})^2}$ | Standard deviation of unwrapped phase |
| Iteration Change | $\Delta^{(k)} = \max_{i,j}|F^{(k)}_{i,j} - F^{(k-1)}_{i,j}|$ | Maximum absolute change between iterates |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include:

- `algorithm_identification`: {algorithm_name, algorithm_family, summary}
- `objective_function`: {full_expression, data_fidelity_term, regularization_term, constraints}
- `variables`: {primal_variables, dual_variables, auxiliary_variables, constants, observations}
- `data_preprocessing`: [{step_order, step_name, formula, inputs, outputs}, ...]
- `initialization`: [{step, operation, formula, output}, ...]
- `iteration_loop`: [{step_name, formula, inputs, outputs, computational_note}, ...]
- `finalization`: [{step, operation, description}, ...]
- `stopping_criterion`: {type, expression, parameters, check_frequency}
- `evaluation`: [{name, formula, description}, ...]
- `hyperparameters`: {max_iters: 50, tol: 0.08, lambda: 2.0, p: 0.01, c: 1.6, epsilon: 0, boundary_conditions: "neumann"}
- `output_shape`: (M, N) matching input interferogram dimensions
- `output_dtype`: float32