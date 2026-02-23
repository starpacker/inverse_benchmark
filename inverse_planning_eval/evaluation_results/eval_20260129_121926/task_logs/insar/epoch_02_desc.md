# Algorithmic Specification Document: InSAR Phase Unwrapping via ADMM

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{F} \| \nabla F - \boldsymbol{\phi} \|_p$$

Reformulated as ADMM-compatible constrained problem:
$$\min_{F, \mathbf{w}} \| \mathbf{w} \|_p \quad \text{subject to} \quad \mathbf{w} = \nabla F - \boldsymbol{\phi}$$

Augmented Lagrangian formulation:
$$\mathcal{L}(F, \mathbf{w}, \boldsymbol{\Lambda}) = \|\mathbf{w}\|_p + \langle \boldsymbol{\Lambda}, \nabla F - \boldsymbol{\phi} - \mathbf{w} \rangle + \frac{1}{2}\|\nabla F - \boldsymbol{\phi} - \mathbf{w}\|_2^2$$

**data_fidelity_term**:
$$\mathcal{D}(F) = \|\nabla F - \boldsymbol{\phi}\|_2^2 = \|D_x F - \phi_x\|_2^2 + \|D_y F - \phi_y\|_2^2$$

where $\boldsymbol{\phi} = (\phi_x, \phi_y)$ represents the wrapped gradient observations.

**regularization_term**:
$$\mathcal{R}(\mathbf{w}) = \|\mathbf{w}\|_p = \sum_{i,j} \left( w_x^2[i,j] + w_y^2[i,j] \right)^{p/2}$$

For $p=0$: counts non-zero gradient magnitudes (sparsity-promoting).
For $p=1$: total variation norm.

**constraints**:
- $w_x = F_x - \phi_x$ (gradient residual in x-direction)
- $w_y = F_y - \phi_y$ (gradient residual in y-direction)
- Neumann boundary conditions: $\frac{\partial F}{\partial n}\big|_{\partial\Omega} = 0$

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $F$ | primal | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Unwrapped phase estimate |
| $w_x$ | primal | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Auxiliary variable for x-gradient residual |
| $w_y$ | primal | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Auxiliary variable for y-gradient residual |

### Dual Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\Lambda_x$ | dual | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Lagrange multiplier for x-constraint |
| $\Lambda_y$ | dual | $(M, N)$ | float32 | $\mathbf{0}_{M \times N}$ | Lagrange multiplier for y-constraint |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $D_x$ | constants | $(MN, MN)$ | sparse float32 | Constructed via Kronecker product | Forward difference operator in x-direction |
| $D_y$ | constants | $(MN, MN)$ | sparse float32 | Constructed via Kronecker product | Forward difference operator in y-direction |
| $K$ | constants | $(M, N)$ | float32 | Computed from DCT eigenvalues | Inverse Laplacian kernel in Fourier domain |
| $\lambda$ | constants | scalar | float | User-specified (default: 1.0) | Shrinkage regularization weight |
| $p$ | constants | scalar | float | User-specified (default: 0) | Norm order for sparsity |
| $c$ | constants | scalar | float | User-specified (default: 1.3) | Lagrange multiplier update factor |
| $M$ | constants | scalar | int | From data | Number of rows |
| $N$ | constants | scalar | int | From data | Number of columns |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $I$ | observations | $(M, N)$ | complex64 | Loaded from file | Complex interferogram |
| $f_{\text{wrapped}}$ | observations | $(M, N)$ | float32 | $\arg(I)$ | Wrapped phase |
| $\phi_x$ | observations | $(M, N)$ | float32 | From wrapped gradient | Wrapped gradient in x-direction |
| $\phi_y$ | observations | $(M, N)$ | float32 | From wrapped gradient | Wrapped gradient in y-direction |
| $\text{mag}$ | observations | $(M, N)$ | float32 | $|I|$ | Interferogram magnitude |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Load Interferogram
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load Complex Interferogram |
| formula | $I[i,j] = \text{fromfile}(\texttt{filename}, \text{complex64}).\text{reshape}(M, N)$ |
| inputs | filename (string), N (int, from .rsc file) |
| outputs | $I \in \mathbb{C}^{M \times N}$ |

### Step 3.2: Extract Phase and Magnitude
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Phase and Magnitude Extraction |
| formula | $f_{\text{wrapped}}[i,j] = \arg(I[i,j]) = \text{atan2}(\text{Im}(I[i,j]), \text{Re}(I[i,j]))$ |
| | $\text{mag}[i,j] = |I[i,j]| = \sqrt{\text{Re}(I[i,j])^2 + \text{Im}(I[i,j])^2}$ |
| inputs | $I \in \mathbb{C}^{M \times N}$ |
| outputs | $f_{\text{wrapped}} \in \mathbb{R}^{M \times N}$, $\text{mag} \in \mathbb{R}^{M \times N}$ |

### Step 3.3: Construct Differentiation Matrices
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Build Finite Difference Operators |
| formula | **1D x-derivative (Neumann BC):** |
| | $D^{(1D)}_x = \begin{pmatrix} -1 & 1 & 0 & \cdots & 0 \\ 0 & -1 & 1 & \cdots & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & \cdots & 0 & -1 & 1 \\ 0 & \cdots & 0 & 0 & 0 \end{pmatrix} \in \mathbb{R}^{N \times N}$ |
| | **2D x-derivative:** $D_x = I_M \otimes D^{(1D)}_x \in \mathbb{R}^{MN \times MN}$ |
| | **1D y-derivative (Neumann BC):** |
| | $D^{(1D)}_y = \begin{pmatrix} -1 & 1 & 0 & \cdots & 0 \\ 0 & -1 & 1 & \cdots & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & \cdots & 0 & -1 & 1 \\ 0 & \cdots & 0 & 0 & 0 \end{pmatrix} \in \mathbb{R}^{M \times M}$ |
| | **2D y-derivative:** $D_y = D^{(1D)}_y \otimes I_N \in \mathbb{R}^{MN \times MN}$ |
| inputs | $M$ (int), $N$ (int), boundary_conditions="neumann" |
| outputs | $D_x \in \mathbb{R}^{MN \times MN}$ (sparse CSR), $D_y \in \mathbb{R}^{MN \times MN}$ (sparse CSR) |

### Step 3.4: Estimate Wrapped Gradients
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Wrapped Gradient Estimation with Phase Correction |
| formula | **Raw gradients:** |
| | $\tilde{\phi}_x = \text{reshape}(D_x \cdot \text{vec}(f_{\text{wrapped}}), (M, N))$ |
| | $\tilde{\phi}_y = \text{reshape}(D_y \cdot \text{vec}(f_{\text{wrapped}}), (M, N))$ |
| | **Wrapping correction:** |
| | $\phi_x[i,j] = \begin{cases} \tilde{\phi}_x[i,j] - 2\pi \cdot \text{sign}(\tilde{\phi}_x[i,j]) & \text{if } |\tilde{\phi}_x[i,j]| > \pi \\ \tilde{\phi}_x[i,j] & \text{otherwise} \end{cases}$ |
| | $\phi_y[i,j] = \begin{cases} \tilde{\phi}_y[i,j] - 2\pi \cdot \text{sign}(\tilde{\phi}_y[i,j]) & \text{if } |\tilde{\phi}_y[i,j]| > \pi \\ \tilde{\phi}_y[i,j] & \text{otherwise} \end{cases}$ |
| inputs | $f_{\text{wrapped}} \in \mathbb{R}^{M \times N}$, $D_x$, $D_y$ |
| outputs | $\phi_x \in \mathbb{R}^{M \times N}$, $\phi_y \in \mathbb{R}^{M \times N}$ |

### Step 3.5: Compute Laplacian Kernel
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Inverse Laplacian Kernel Construction |
| formula | **DCT-II eigenvalues of discrete Laplacian:** |
| | $\xi_y[m] = 2 - 2\cos\left(\frac{\pi m}{M}\right), \quad m = 0, \ldots, M-1$ |
| | $\xi_x[n] = 2 - 2\cos\left(\frac{\pi n}{N}\right), \quad n = 0, \ldots, N-1$ |
| | **Combined eigenvalues:** $\Xi[m,n] = \xi_y[m] + \xi_x[n]$ |
| | **Inverse kernel:** $K[m,n] = \begin{cases} 0 & \text{if } \Xi[m,n] = 0 \\ \frac{1}{\Xi[m,n]} & \text{otherwise} \end{cases}$ |
| inputs | $M$ (int), $N$ (int) |
| outputs | $K \in \mathbb{R}^{M \times N}$ |

---

## 4. ALGORITHM STEPS

### Initialization

| Field | Value |
|-------|-------|
| step_name | Variable Initialization |
| formula | $F^{(0)} = \mathbf{0}_{M \times N}$ |
| | $F_{\text{old}}^{(0)} = \mathbf{0}_{M \times N}$ |
| | $w_x^{(0)} = \mathbf{0}_{M \times N}$ |
| | $w_y^{(0)} = \mathbf{0}_{M \times N}$ |
| | $\Lambda_x^{(0)} = \mathbf{0}_{M \times N}$ |
| | $\Lambda_y^{(0)} = \mathbf{0}_{M \times N}$ |
| | $k = 0$ |
| inputs | $M$, $N$, dtype |
| outputs | All initialized variables |
| computational_note | All arrays allocated with zeros; iteration counter set to 0 |

---

### Iteration Loop

**BEGIN ITERATION LOOP** ($k = 0, 1, \ldots, k_{\max}-1$)

---

#### Step 4.1: Compute Right-Hand Side for F-Update
| Field | Value |
|-------|-------|
| step_name | Assemble Linear System RHS |
| formula | $r_x = \text{vec}(w_x^{(k)} + \phi_x - \Lambda_x^{(k)})$ |
| | $r_y = \text{vec}(w_y^{(k)} + \phi_y - \Lambda_y^{(k)})$ |
| | $\text{RHS} = D_x^\top r_x + D_y^\top r_y$ |
| inputs | $w_x^{(k)}, w_y^{(k)}, \phi_x, \phi_y, \Lambda_x^{(k)}, \Lambda_y^{(k)}, D_x, D_y$ |
| outputs | $\text{RHS} \in \mathbb{R}^{MN}$ |
| computational_note | Sparse matrix-vector products; RHS corresponds to divergence of combined field |

#### Step 4.2: Solve for F via DCT
| Field | Value |
|-------|-------|
| step_name | Poisson Solve in Fourier Domain |
| formula | $\hat{\rho} = \text{DCT-II}(\text{reshape}(\text{RHS}, (M, N)))$ |
| | where $\text{DCT-II}: \hat{\rho}[m,n] = \sum_{i=0}^{M-1}\sum_{j=0}^{N-1} \rho[i,j] \cos\left(\frac{\pi m(2i+1)}{2M}\right)\cos\left(\frac{\pi n(2j+1)}{2N}\right)$ |
| | with orthonormal normalization |
| | $F^{(k+1)} = \text{IDCT-II}(\hat{\rho} \odot K)$ |
| | where $\text{IDCT-II}$ is the inverse DCT-II with orthonormal normalization |
| inputs | $\text{RHS} \in \mathbb{R}^{MN}$, $K \in \mathbb{R}^{M \times N}$ |
| outputs | $F^{(k+1)} \in \mathbb{R}^{M \times N}$ |
| computational_note | $\odot$ denotes element-wise multiplication; DCT diagonalizes Laplacian under Neumann BC |

#### Step 4.3: Forward Operator (Gradient Computation)
| Field | Value |
|-------|-------|
| step_name | Compute Gradients of Updated Phase |
| formula | $F_x^{(k+1)} = \text{reshape}(D_x \cdot \text{vec}(F^{(k+1)}), (M, N))$ |
| | $F_y^{(k+1)} = \text{reshape}(D_y \cdot \text{vec}(F^{(k+1)}), (M, N))$ |
| inputs | $F^{(k+1)} \in \mathbb{R}^{M \times N}$, $D_x$, $D_y$ |
| outputs | $F_x^{(k+1)} \in \mathbb{R}^{M \times N}$, $F_y^{(k+1)} \in \mathbb{R}^{M \times N}$ |
| computational_note | Sparse matrix-vector multiplication followed by reshape |

#### Step 4.4: Compute Shrinkage Input
| Field | Value |
|-------|-------|
| step_name | Prepare Input for Proximal Operator |
| formula | $u_x^{(k+1)} = F_x^{(k+1)} - \phi_x + \Lambda_x^{(k)}$ |
| | $u_y^{(k+1)} = F_y^{(k+1)} - \phi_y + \Lambda_y^{(k)}$ |
| inputs | $F_x^{(k+1)}, F_y^{(k+1)}, \phi_x, \phi_y, \Lambda_x^{(k)}, \Lambda_y^{(k)}$ |
| outputs | $u_x^{(k+1)} \in \mathbb{R}^{M \times N}$, $u_y^{(k+1)} \in \mathbb{R}^{M \times N}$ |
| computational_note | Element-wise operations |

#### Step 4.5: p-Shrinkage (Proximal Operator)
| Field | Value |
|-------|-------|
| step_name | Auxiliary Variable Update via Generalized Shrinkage |
| formula | **Stack inputs:** $\mathbf{U}^{(k+1)} = \begin{pmatrix} u_x^{(k+1)} \\ u_y^{(k+1)} \end{pmatrix} \in \mathbb{R}^{2 \times M \times N}$ |
| | **Compute magnitude:** $\mu[i,j] = \sqrt{(u_x^{(k+1)}[i,j])^2 + (u_y^{(k+1)}[i,j])^2}$ |
| | **Safe magnitude:** $\tilde{\mu}[i,j] = \begin{cases} 1 & \text{if } \mu[i,j] = 0 \\ \mu[i,j] & \text{otherwise} \end{cases}$ |
| | **Shrinkage factor:** $s[i,j] = \frac{\max\left(\mu[i,j] - \lambda^{2-p} \cdot (\tilde{\mu}[i,j]^2 + \epsilon)^{p/2 - 1/2}, 0\right)}{\tilde{\mu}[i,j]}$ |
| | **Apply shrinkage:** $w_x^{(k+1)}[i,j] = s[i,j] \cdot u_x^{(k+1)}[i,j]$ |
| | $w_y^{(k+1)}[i,j] = s[i,j] \cdot u_y^{(k+1)}[i,j]$ |
| inputs | $u_x^{(k+1)}, u_y^{(k+1)} \in \mathbb{R}^{M \times N}$, $\lambda$, $p$, $\epsilon=0$ |
| outputs | $w_x^{(k+1)}, w_y^{(k+1)} \in \mathbb{R}^{M \times N}$ |
| computational_note | Isotropic shrinkage operates on gradient magnitude; for $p=0$ approximates hard thresholding; for $p=1$ reduces to soft thresholding |

#### Step 4.6: Dual Variable Update
| Field | Value |
|-------|-------|
| step_name | Lagrange Multiplier Update |
| formula | $\Lambda_x^{(k+1)} = \Lambda_x^{(k)} + c \cdot (F_x^{(k+1)} - \phi_x - w_x^{(k+1)})$ |
| | $\Lambda_y^{(k+1)} = \Lambda_y^{(k)} + c \cdot (F_y^{(k+1)} - \phi_y - w_y^{(k+1)})$ |
| inputs | $\Lambda_x^{(k)}, \Lambda_y^{(k)}, F_x^{(k+1)}, F_y^{(k+1)}, \phi_x, \phi_y, w_x^{(k+1)}, w_y^{(k+1)}, c$ |
| outputs | $\Lambda_x^{(k+1)}, \Lambda_y^{(k+1)} \in \mathbb{R}^{M \times N}$ |
| computational_note | Scaled form of ADMM dual update; $c$ controls step size |

#### Step 4.7: Convergence Check
| Field | Value |
|-------|-------|
| step_name | Evaluate Stopping Criterion |
| formula | $\Delta^{(k+1)} = \max_{i,j} |F^{(k+1)}[i,j] - F_{\text{old}}^{(k)}[i,j]|$ |
| | **Converged:** $\Delta^{(k+1)} < \tau$ OR $\text{isnan}(\Delta^{(k+1)})$ |
| | **Continue:** $F_{\text{old}}^{(k+1)} \leftarrow F^{(k+1)}$, $k \leftarrow k + 1$ |
| inputs | $F^{(k+1)}, F_{\text{old}}^{(k)}, \tau$ |
| outputs | convergence_flag (boolean), $\Delta^{(k+1)}$ (scalar) |
| computational_note | Maximum absolute change provides worst-case convergence measure |

---

**END ITERATION LOOP**

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | combined |
| expression | $\left( \Delta^{(k)} < \tau \right) \lor \left( \text{isnan}(\Delta^{(k)}) \right) \lor \left( k \geq k_{\max} \right)$ |
| | where $\Delta^{(k)} = \|F^{(k)} - F^{(k-1)}\|_\infty = \max_{i,j}|F^{(k)}[i,j] - F^{(k-1)}[i,j]|$ |
| parameters | $\tau = 1.6$ (default tolerance, code uses `tol`) |
| | $k_{\max} = 500$ (default maximum iterations, code uses `max_iters`) |
| | Runtime parameters: $\tau = 0.08$, $k_{\max} = 50$ |

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Phase Minimum | $F_{\min} = \min_{i,j} F[i,j]$ | Minimum value of unwrapped phase |
| Phase Maximum | $F_{\max} = \max_{i,j} F[i,j]$ | Maximum value of unwrapped phase |
| Phase Mean | $\bar{F} = \frac{1}{MN}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1} F[i,j]$ | Mean unwrapped phase value |
| Phase Standard Deviation | $\sigma_F = \sqrt{\frac{1}{MN}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}(F[i,j] - \bar{F})^2}$ | Standard deviation of unwrapped phase |
| Phase Range | $[F_{\min}, F_{\max}]$ | Dynamic range of reconstruction |