## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: ADMM Phase Unwrapping with p-Shrinkage
- **algorithm_family**: Alternating Direction Method of Multipliers (ADMM) / Proximal Splitting Methods
- **summary**: Recovers unwrapped phase from wrapped interferometric SAR observations by minimizing the p-quasi-norm of the residual between the gradient of the unwrapped phase and the wrapped gradient estimates.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{F} \| \nabla F - \phi \|_p^p$$

where $\nabla F = (D_x F, D_y F)$ and $\phi = (\phi_x, \phi_y)$ is the wrapped gradient.

Equivalently, via ADMM reformulation:
$$\min_{F, w} \| w \|_p^p \quad \text{subject to} \quad \nabla F - \phi = w$$

### data_fidelity_term
$$\| \nabla F - \phi \|_p^p = \sum_{i,j} \left( (D_x F - \phi_x)_{i,j}^2 + (D_y F - \phi_y)_{i,j}^2 \right)^{p/2}$$

where $D_x, D_y$ are discrete gradient operators and $\phi_x, \phi_y$ are wrapped gradient observations.

### regularization_term
None (the sparsity is enforced through the p-quasi-norm on the residual, which is the data fidelity term itself)

### constraints
None (unconstrained optimization after ADMM reformulation absorbs constraints into augmented Lagrangian)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $F$ | primal | $(M, N)$ | float32 | `np.zeros((rows, columns))` | Unwrapped phase estimate |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $w_x$ | auxiliary | $(M, N)$ | float32 | `np.zeros_like(phi_x)` | ADMM splitting variable for x-gradient residual |
| $w_y$ | auxiliary | $(M, N)$ | float32 | `np.zeros_like(phi_y)` | ADMM splitting variable for y-gradient residual |
| $F_{\text{old}}$ | auxiliary | $(M, N)$ | float32 | `np.zeros_like(f_wrapped)` | Previous iterate for convergence check |

### Dual Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\Lambda_x$ | dual | $(M, N)$ | float32 | `np.zeros_like(phi_x)` | Lagrange multiplier for x-constraint |
| $\Lambda_y$ | dual | $(M, N)$ | float32 | `np.zeros_like(phi_y)` | Lagrange multiplier for y-constraint |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $f_{\text{wrapped}}$ | observations | $(M, N)$ | float32 | `np.angle(igram)` | Wrapped phase from interferogram |
| $\phi_x$ | observations | $(M, N)$ | float32 | Computed via wrapped gradient estimation | Wrapped gradient in x-direction |
| $\phi_y$ | observations | $(M, N)$ | float32 | Computed via wrapped gradient estimation | Wrapped gradient in y-direction |
| $\text{mag}$ | observations | $(M, N)$ | complex64→float | `np.abs(igram)` | Interferogram magnitude |

### Constants/Operators
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $D_x$ | constants | $(MN, MN)$ sparse | float32 | Constructed via `make_differentiation_matrices` | Forward difference operator in x-direction |
| $D_y$ | constants | $(MN, MN)$ sparse | float32 | Constructed via `make_differentiation_matrices` | Forward difference operator in y-direction |
| $K$ | constants | $(M, N)$ | float32 | Computed via `make_laplace_kernel` | Inverse Laplacian eigenvalues for DCT domain |
| $\lambda$ | constants | scalar | float | `2.0` (main block) | Regularization/shrinkage parameter |
| $p$ | constants | scalar | float | `0.01` (main block) | p-quasi-norm exponent |
| $c$ | constants | scalar | float | `1.6` (main block) | Dual variable update step size |
| $\epsilon$ | constants | scalar | float | `0` | Mollification parameter for p-shrinkage |
| $\text{max\_iters}$ | constants | scalar | int | `50` (main block) | Maximum iteration count |
| $\text{tol}$ | constants | scalar | float | `0.08` (main block) | Convergence tolerance threshold |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Load Interferogram | $\text{igram} = \text{fromfile}(\text{filename}, \text{complex64}).\text{reshape}(-1, N)$ | filename, columns $N$ from .rsc file | igram: $(M, N)$ complex64 |
| 2 | Extract Magnitude | $\text{mag} = |\text{igram}|$ | igram: $(M, N)$ | mag: $(M, N)$ |
| 3 | Extract Wrapped Phase | $f_{\text{wrapped}} = \arg(\text{igram})$ | igram: $(M, N)$ | $f_{\text{wrapped}}$: $(M, N)$ float32 |
| 4 | Construct $D_x$ | $D_x = I_M \otimes D^{(N)}$ where $D^{(N)}_{i,i}=-1, D^{(N)}_{i,i+1}=1$, $D^{(N)}_{N-1,N-1}=0$ (Neumann BC) | $M$, $N$, boundary="neumann" | $D_x$: $(MN, MN)$ sparse |
| 5 | Construct $D_y$ | $D_y = D^{(M)} \otimes I_N$ where $D^{(M)}_{i,i}=-1, D^{(M)}_{i,i+1}=1$, $D^{(M)}_{M-1,M-1}=0$ (Neumann BC) | $M$, $N$, boundary="neumann" | $D_y$: $(MN, MN)$ sparse |
| 6 | Compute Raw Gradients | $\tilde{\phi}_x = (D_x \cdot \text{vec}(f_{\text{wrapped}})).\text{reshape}(M,N)$, $\tilde{\phi}_y = (D_y \cdot \text{vec}(f_{\text{wrapped}})).\text{reshape}(M,N)$ | $f_{\text{wrapped}}$, $D_x$, $D_y$ | $\tilde{\phi}_x, \tilde{\phi}_y$: $(M,N)$ |
| 7 | Apply Wrapping Correction (x) | $\phi_x = \begin{cases} \tilde{\phi}_x - 2\pi \cdot \text{sign}(\tilde{\phi}_x) & \text{if } |\tilde{\phi}_x| > \pi \\ \tilde{\phi}_x & \text{otherwise} \end{cases}$ | $\tilde{\phi}_x$ | $\phi_x$: $(M,N)$ |
| 8 | Apply Wrapping Correction (y) | $\phi_y = \begin{cases} \tilde{\phi}_y - 2\pi \cdot \text{sign}(\tilde{\phi}_y) & \text{if } |\tilde{\phi}_y| > \pi \\ \tilde{\phi}_y & \text{otherwise} \end{cases}$ | $\tilde{\phi}_y$ | $\phi_y$: $(M,N)$ |
| 9 | Compute Laplacian Eigenvalues | $\xi_y^{(i)} = 2 - 2\cos\left(\frac{\pi i}{M}\right), \quad \xi_x^{(j)} = 2 - 2\cos\left(\frac{\pi j}{N}\right)$ | $M$, $N$ | $\xi_y$: $(M,1)$, $\xi_x$: $(1,N)$ |
| 10 | Compute Inverse Laplacian Kernel | $K_{i,j} = \begin{cases} 0 & \text{if } \xi_y^{(i)} + \xi_x^{(j)} = 0 \\ \frac{1}{\xi_y^{(i)} + \xi_x^{(j)}} & \text{otherwise} \end{cases}$ | $\xi_y$, $\xi_x$ | $K$: $(M,N)$ float32 |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Initialize Primal | $F^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $F^{(0)}$ |
| 2 | Initialize Auxiliary | $w_x^{(0)} = \mathbf{0}_{M \times N}, \quad w_y^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $w_x^{(0)}, w_y^{(0)}$ |
| 3 | Initialize Dual | $\Lambda_x^{(0)} = \mathbf{0}_{M \times N}, \quad \Lambda_y^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $\Lambda_x^{(0)}, \Lambda_y^{(0)}$ |
| 4 | Initialize History | $F_{\text{old}}^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $F_{\text{old}}^{(0)}$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, \text{max\_iters}-1$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 1 | Compute RHS (x-component) | $r_x = \text{vec}(w_x^{(k)} + \phi_x - \Lambda_x^{(k)})$ | $w_x^{(k)}, \phi_x, \Lambda_x^{(k)}$ | $r_x \in \mathbb{R}^{MN}$ | Flatten to vector |
| 2 | Compute RHS (y-component) | $r_y = \text{vec}(w_y^{(k)} + \phi_y - \Lambda_y^{(k)})$ | $w_y^{(k)}, \phi_y, \Lambda_y^{(k)}$ | $r_y \in \mathbb{R}^{MN}$ | Flatten to vector |
| 3 | Assemble Poisson RHS | $\rho = D_x^T r_x + D_y^T r_y$ | $D_x, D_y, r_x, r_y$ | $\rho \in \mathbb{R}^{MN}$ | Sparse matrix-vector products |
| 4 | Forward DCT | $\hat{\rho} = \text{DCT-II}(\text{reshape}(\rho, M, N))$ | $\rho$ reshaped to $(M,N)$ | $\hat{\rho} \in \mathbb{R}^{M \times N}$ | Orthonormal DCT-II, parallel workers |
| 5 | Apply Inverse Laplacian | $\hat{F} = K \odot \hat{\rho}$ | $K$, $\hat{\rho}$ | $\hat{F} \in \mathbb{R}^{M \times N}$ | Element-wise multiplication |
| 6 | Inverse DCT | $F^{(k+1)} = \text{IDCT-II}(\hat{F})$ | $\hat{F}$ | $F^{(k+1)} \in \mathbb{R}^{M \times N}$ | Orthonormal IDCT-II |
| 7 | Compute x-gradient | $F_x^{(k+1)} = (D_x \cdot \text{vec}(F^{(k+1)})).\text{reshape}(M,N)$ | $D_x$, $F^{(k+1)}$ | $F_x^{(k+1)} \in \mathbb{R}^{M \times N}$ | Forward operator |
| 8 | Compute y-gradient | $F_y^{(k+1)} = (D_y \cdot \text{vec}(F^{(k+1)})).\text{reshape}(M,N)$ | $D_y$, $F^{(k+1)}$ | $F_y^{(k+1)} \in \mathbb{R}^{M \times N}$ | Forward operator |
| 9 | Shrinkage input (x) | $u_x = F_x^{(k+1)} - \phi_x + \Lambda_x^{(k)}$ | $F_x^{(k+1)}, \phi_x, \Lambda_x^{(k)}$ | $u_x \in \mathbb{R}^{M \times N}$ | - |
| 10 | Shrinkage input (y) | $u_y = F_y^{(k+1)} - \phi_y + \Lambda_y^{(k)}$ | $F_y^{(k+1)}, \phi_y, \Lambda_y^{(k)}$ | $u_y \in \mathbb{R}^{M \times N}$ | - |
| 11 | Stack for vectorial shrinkage | $U = \begin{bmatrix} u_x \\ u_y \end{bmatrix}$ | $u_x, u_y$ | $U \in \mathbb{R}^{2 \times M \times N}$ | Stack along axis 0 |
| 12 | Compute magnitude | $m_{i,j} = \sqrt{u_{x,i,j}^2 + u_{y,i,j}^2}$ | $U$ | $m \in \mathbb{R}^{M \times N}$ | Per-pixel 2D magnitude |
| 13 | Safe magnitude | $\tilde{m}_{i,j} = \begin{cases} 1 & \text{if } m_{i,j} = 0 \\ m_{i,j} & \text{otherwise} \end{cases}$ | $m$ | $\tilde{m} \in \mathbb{R}^{M \times N}$ | Avoid division by zero |
| 14 | p-Shrinkage scaling | $s_{i,j} = \frac{\max\left(m_{i,j} - \lambda^{2-p}(\tilde{m}_{i,j}^2 + \epsilon)^{p/2-0.5}, 0\right)}{\tilde{m}_{i,j}}$ | $m, \tilde{m}, \lambda, p, \epsilon$ | $s \in \mathbb{R}^{M \times N}$ | Generalized shrinkage |
| 15 | Apply shrinkage | $w_x^{(k+1)} = s \odot u_x, \quad w_y^{(k+1)} = s \odot u_y$ | $s, u_x, u_y$ | $w_x^{(k+1)}, w_y^{(k+1)}$ | Element-wise scaling |
| 16 | Update dual (x) | $\Lambda_x^{(k+1)} = \Lambda_x^{(k)} + c(F_x^{(k+1)} - \phi_x - w_x^{(k+1)})$ | $\Lambda_x^{(k)}, F_x^{(k+1)}, \phi_x, w_x^{(k+1)}, c$ | $\Lambda_x^{(k+1)}$ | Dual ascent |
| 17 | Update dual (y) | $\Lambda_y^{(k+1)} = \Lambda_y^{(k)} + c(F_y^{(k+1)} - \phi_y - w_y^{(k+1)})$ | $\Lambda_y^{(k)}, F_y^{(k+1)}, \phi_y, w_y^{(k+1)}, c$ | $\Lambda_y^{(k+1)}$ | Dual ascent |
| 18 | Compute change | $\Delta^{(k)} = \max_{i,j} |F_{i,j}^{(k+1)} - F_{\text{old},i,j}^{(k)}|$ | $F^{(k+1)}, F_{\text{old}}^{(k)}$ | $\Delta^{(k)} \in \mathbb{R}$ | Convergence metric |
| 19 | Check convergence | If $\Delta^{(k)} < \text{tol}$ or $\text{isnan}(\Delta^{(k)})$: **BREAK** | $\Delta^{(k)}, \text{tol}$ | - | Early termination |
| 20 | Update history | $F_{\text{old}}^{(k+1)} = F^{(k+1)}.\text{copy}()$ | $F^{(k+1)}$ | $F_{\text{old}}^{(k+1)}$ | Store for next iteration |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Return Solution | Output $F^{(k_{\text{final}})}$ | Final iterate | Unwrapped phase $F$ |

---

## 5. STOPPING CRITERION

- **type**: combined
- **expression**: 
$$\text{STOP if } \left( \max_{i,j}|F_{i,j}^{(k+1)} - F_{i,j}^{(k)}| < \text{tol} \right) \text{ OR } \left( k \geq \text{max\_iters} - 1 \right) \text{ OR } \left( \Delta^{(k)} = \text{NaN} \right)$$
- **parameters**:
  - `tol = 0.08` (convergence threshold)
  - `max_iters = 50` (iteration limit)
- **check_frequency**: Every iteration (after dual variable update)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Phase Range | $[\min_{i,j} F_{i,j}, \max_{i,j} F_{i,j}]$ | Min and max values of unwrapped phase |
| Mean Phase | $\bar{F} = \frac{1}{MN}\sum_{i,j} F_{i,j}$ | Spatial average of unwrapped phase |
| Standard Deviation | $\sigma_F = \sqrt{\frac{1}{MN}\sum_{i,j}(F_{i,j} - \bar{F})^2}$ | Spatial variability of unwrapped phase |
| Iteration Change | $\Delta^{(k)} = \|F^{(k+1)} - F^{(k)}\|_\infty$ | Maximum absolute change between consecutive iterates |