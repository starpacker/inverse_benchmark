## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: ADMM Phase Unwrapping
- **algorithm_family**: Alternating Direction Method of Multipliers (ADMM) / Proximal Splitting
- **summary**: Solves InSAR phase unwrapping by minimizing the p-norm of the residual between the gradient of the unwrapped phase and the wrapped gradient observations, using ADMM with DCT-based Poisson solver and generalized shrinkage.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{F} \left\| \nabla F - \boldsymbol{\phi} \right\|_p$$

Reformulated with auxiliary variable $\mathbf{w} = (w_x, w_y)$:
$$\min_{F, \mathbf{w}} \left\| \mathbf{w} \right\|_p \quad \text{subject to} \quad \nabla F - \boldsymbol{\phi} = \mathbf{w}$$

Augmented Lagrangian:
$$\mathcal{L}_\rho(F, \mathbf{w}, \boldsymbol{\Lambda}) = \left\| \mathbf{w} \right\|_p + \langle \Lambda_x, F_x - \phi_x - w_x \rangle + \langle \Lambda_y, F_y - \phi_y - w_y \rangle + \frac{\rho}{2}\left( \|F_x - \phi_x - w_x\|_2^2 + \|F_y - \phi_y - w_y\|_2^2 \right)$$

where $F_x = D_x F$, $F_y = D_y F$, and $\rho = 1$ (implicit in formulation).

### data_fidelity_term
$$\left\| \nabla F - \boldsymbol{\phi} \right\|_p = \left( \sum_{i,j} \left[ (F_x - \phi_x)_{i,j}^2 + (F_y - \phi_y)_{i,j}^2 \right]^{p/2} \right)^{1/p}$$

where $p = 0.01$ (near-L0 sparsity).

### regularization_term
None (the objective is purely data fidelity on gradient residuals).

### constraints
$$F \in \mathbb{R}^{M \times N} \quad \text{(unconstrained)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $F$ | $(M, N)$ | float32 | $F^{(0)} = \mathbf{0}_{M \times N}$ | Unwrapped phase estimate |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Lambda_x$ | $(M, N)$ | float32 | $\Lambda_x^{(0)} = \mathbf{0}_{M \times N}$ | Lagrange multiplier for x-gradient constraint |
| $\Lambda_y$ | $(M, N)$ | float32 | $\Lambda_y^{(0)} = \mathbf{0}_{M \times N}$ | Lagrange multiplier for y-gradient constraint |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $w_x$ | $(M, N)$ | float32 | $w_x^{(0)} = \mathbf{0}_{M \times N}$ | Auxiliary variable for x-gradient residual |
| $w_y$ | $(M, N)$ | float32 | $w_y^{(0)} = \mathbf{0}_{M \times N}$ | Auxiliary variable for y-gradient residual |
| $F_{\text{old}}$ | $(M, N)$ | float32 | $F_{\text{old}}^{(0)} = \mathbf{0}_{M \times N}$ | Previous iteration estimate for convergence check |
| $F_x$ | $(M, N)$ | float32 | Computed each iteration | Gradient of $F$ in x-direction |
| $F_y$ | $(M, N)$ | float32 | Computed each iteration | Gradient of $F$ in y-direction |
| $r_x$ | $(MN,)$ | float32 | Computed each iteration | Flattened RHS component for x |
| $r_y$ | $(MN,)$ | float32 | Computed each iteration | Flattened RHS component for y |
| $\text{RHS}$ | $(MN,)$ | float32 | Computed each iteration | Right-hand side of Poisson equation |
| $\hat{\rho}$ | $(M, N)$ | float32 | Computed each iteration | DCT of RHS |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $\lambda$ | scalar | float | 2.0 | Shrinkage regularization parameter |
| $p$ | scalar | float | 0.01 | p-norm exponent (near-L0) |
| $c$ | scalar | float | 1.6 | Dual variable update scaling factor |
| $\tau$ | scalar | float | 0.08 | Convergence tolerance |
| $k_{\max}$ | scalar | int | 50 | Maximum iterations |
| $\epsilon$ | scalar | float | 0 | Mollification parameter for p-shrink |
| $\rho$ | scalar | float | 1.0 | ADMM penalty parameter (implicit) |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $I$ | $(M, N)$ | complex64 | Raw complex interferogram |
| $f_{\text{wrapped}}$ | $(M, N)$ | float32 | Wrapped phase: $\angle(I)$ |
| $\phi_x$ | $(M, N)$ | float32 | Wrapped gradient in x-direction (adjusted) |
| $\phi_y$ | $(M, N)$ | float32 | Wrapped gradient in y-direction (adjusted) |
| $D_x$ | $(MN, MN)$ | sparse float32 | Forward difference matrix in x-direction |
| $D_y$ | $(MN, MN)$ | sparse float32 | Forward difference matrix in y-direction |
| $K$ | $(M, N)$ | float32 | Inverse Laplacian eigenvalues for DCT solver |
| $M$ | scalar | int | Number of rows |
| $N$ | scalar | int | Number of columns |

---

## 3. PREPROCESSING PIPELINE

### Step P1: Load Interferogram
- **step_order**: P1
- **step_name**: Load Complex Interferogram
- **formula**: $I = \text{fromfile}(\texttt{filename}, \text{dtype=complex64}).\text{reshape}(M, N)$
- **inputs**: filename (string), N (from RSC file)
- **outputs**: $I \in \mathbb{C}^{M \times N}$

### Step P2: Extract Wrapped Phase
- **step_order**: P2
- **step_name**: Compute Wrapped Phase
- **formula**: $f_{\text{wrapped}}[i,j] = \arg(I[i,j]) = \arctan2(\text{Im}(I[i,j]), \text{Re}(I[i,j]))$
- **inputs**: $I \in \mathbb{C}^{M \times N}$
- **outputs**: $f_{\text{wrapped}} \in \mathbb{R}^{M \times N}$, range $(-\pi, \pi]$

### Step P3: Extract Magnitude
- **step_order**: P3
- **step_name**: Compute Magnitude
- **formula**: $\text{mag}[i,j] = |I[i,j]| = \sqrt{\text{Re}(I[i,j])^2 + \text{Im}(I[i,j])^2}$
- **inputs**: $I \in \mathbb{C}^{M \times N}$
- **outputs**: $\text{mag} \in \mathbb{R}^{M \times N}$

### Step P4: Construct X-Differentiation Matrix
- **step_order**: P4
- **step_name**: Build $D_x$ with Neumann BC
- **formula**: 
$$D_{\text{1D},x} = \begin{bmatrix} -1 & 1 & 0 & \cdots & 0 \\ 0 & -1 & 1 & \cdots & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & \cdots & 0 & -1 & 1 \\ 0 & \cdots & 0 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{N \times N}$$
$$D_x = I_M \otimes D_{\text{1D},x} \in \mathbb{R}^{MN \times MN}$$
(Neumann BC: last row is zero)
- **inputs**: $M$, $N$
- **outputs**: $D_x$ (sparse CSR)

### Step P5: Construct Y-Differentiation Matrix
- **step_order**: P5
- **step_name**: Build $D_y$ with Neumann BC
- **formula**: 
$$D_{\text{1D},y} = \begin{bmatrix} -1 & 1 & 0 & \cdots & 0 \\ 0 & -1 & 1 & \cdots & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & \cdots & 0 & -1 & 1 \\ 0 & \cdots & 0 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{M \times M}$$
$$D_y = D_{\text{1D},y} \otimes I_N \in \mathbb{R}^{MN \times MN}$$
- **inputs**: $M$, $N$
- **outputs**: $D_y$ (sparse CSR)

### Step P6: Compute Raw X-Gradient
- **step_order**: P6
- **step_name**: Compute Raw Wrapped Gradient X
- **formula**: $\tilde{\phi}_x = \text{reshape}(D_x \cdot \text{vec}(f_{\text{wrapped}}), (M, N))$
- **inputs**: $D_x \in \mathbb{R}^{MN \times MN}$, $f_{\text{wrapped}} \in \mathbb{R}^{M \times N}$
- **outputs**: $\tilde{\phi}_x \in \mathbb{R}^{M \times N}$

### Step P7: Compute Raw Y-Gradient
- **step_order**: P7
- **step_name**: Compute Raw Wrapped Gradient Y
- **formula**: $\tilde{\phi}_y = \text{reshape}(D_y \cdot \text{vec}(f_{\text{wrapped}}), (M, N))$
- **inputs**: $D_y \in \mathbb{R}^{MN \times MN}$, $f_{\text{wrapped}} \in \mathbb{R}^{M \times N}$
- **outputs**: $\tilde{\phi}_y \in \mathbb{R}^{M \times N}$

### Step P8: Apply Wrapping Adjustment to X-Gradient
- **step_order**: P8
- **step_name**: Phase Wrap Correction X
- **formula**: 
$$\phi_x[i,j] = \begin{cases} \tilde{\phi}_x[i,j] - 2\pi \cdot \text{sign}(\tilde{\phi}_x[i,j]) & \text{if } |\tilde{\phi}_x[i,j]| > \pi \\ \tilde{\phi}_x[i,j] & \text{otherwise} \end{cases}$$
- **inputs**: $\tilde{\phi}_x \in \mathbb{R}^{M \times N}$
- **outputs**: $\phi_x \in \mathbb{R}^{M \times N}$, range $[-\pi, \pi]$

### Step P9: Apply Wrapping Adjustment to Y-Gradient
- **step_order**: P9
- **step_name**: Phase Wrap Correction Y
- **formula**: 
$$\phi_y[i,j] = \begin{cases} \tilde{\phi}_y[i,j] - 2\pi \cdot \text{sign}(\tilde{\phi}_y[i,j]) & \text{if } |\tilde{\phi}_y[i,j]| > \pi \\ \tilde{\phi}_y[i,j] & \text{otherwise} \end{cases}$$
- **inputs**: $\tilde{\phi}_y \in \mathbb{R}^{M \times N}$
- **outputs**: $\phi_y \in \mathbb{R}^{M \times N}$, range $[-\pi, \pi]$

### Step P10: Compute Laplacian Eigenvalues
- **step_order**: P10
- **step_name**: Build DCT Laplacian Kernel
- **formula**: 
$$\xi_y[i] = 2 - 2\cos\left(\frac{\pi i}{M}\right), \quad i = 0, \ldots, M-1$$
$$\xi_x[j] = 2 - 2\cos\left(\frac{\pi j}{N}\right), \quad j = 0, \ldots, N-1$$
$$\text{eigvals}[i,j] = \xi_y[i] + \xi_x[j]$$
$$K[i,j] = \begin{cases} 0 & \text{if } \text{eigvals}[i,j] = 0 \\ \frac{1}{\text{eigvals}[i,j]} & \text{otherwise} \end{cases}$$
- **inputs**: $M$, $N$
- **outputs**: $K \in \mathbb{R}^{M \times N}$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize Dual X | $\Lambda_x^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $\Lambda_x^{(0)}$ | Zero array, dtype=float32 |
| 2 | Initialize Dual Y | $\Lambda_y^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $\Lambda_y^{(0)}$ | Zero array, dtype=float32 |
| 3 | Initialize Auxiliary X | $w_x^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $w_x^{(0)}$ | Zero array, dtype=float32 |
| 4 | Initialize Auxiliary Y | $w_y^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $w_y^{(0)}$ | Zero array, dtype=float32 |
| 5 | Initialize Primal | $F^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $F^{(0)}$ | Zero array, dtype=float32 |
| 6 | Initialize Previous | $F_{\text{old}}^{(0)} = \mathbf{0}_{M \times N}$ | $M$, $N$ | $F_{\text{old}}^{(0)}$ | For convergence tracking |
| 7 | Initialize Iteration Counter | $k = 0$ | — | $k$ | Integer counter |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, k_{\max} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 8 | Compute RHS X-Component | $r_x = \text{vec}(w_x^{(k)}) + \text{vec}(\phi_x) - \text{vec}(\Lambda_x^{(k)})$ | $w_x^{(k)}$, $\phi_x$, $\Lambda_x^{(k)}$ | $r_x \in \mathbb{R}^{MN}$ | Flatten and combine |
| 9 | Compute RHS Y-Component | $r_y = \text{vec}(w_y^{(k)}) + \text{vec}(\phi_y) - \text{vec}(\Lambda_y^{(k)})$ | $w_y^{(k)}$, $\phi_y$, $\Lambda_y^{(k)}$ | $r_y \in \mathbb{R}^{MN}$ | Flatten and combine |
| 10 | Compute Divergence (RHS) | $\text{RHS} = D_x^T r_x + D_y^T r_y$ | $D_x$, $D_y$, $r_x$, $r_y$ | $\text{RHS} \in \mathbb{R}^{MN}$ | [ADJOINT] Discrete divergence |
| 11 | Reshape RHS | $\text{RHS}_{2D} = \text{reshape}(\text{RHS}, (M, N))$ | $\text{RHS}$ | $\text{RHS}_{2D} \in \mathbb{R}^{M \times N}$ | — |
| 12 | Forward DCT | $\hat{\rho} = \text{DCT-II}(\text{RHS}_{2D})$ | $\text{RHS}_{2D}$ | $\hat{\rho} \in \mathbb{R}^{M \times N}$ | Orthonormal DCT-II |
| 13 | Apply Inverse Laplacian | $\hat{F} = \hat{\rho} \odot K$ | $\hat{\rho}$, $K$ | $\hat{F} \in \mathbb{R}^{M \times N}$ | Elementwise multiply |
| 14 | Inverse DCT | $F^{(k+1)} = \text{IDCT-II}(\hat{F})$ | $\hat{F}$ | $F^{(k+1)} \in \mathbb{R}^{M \times N}$ | Orthonormal IDCT-II, solves $\nabla^2 F = \text{RHS}$ |
| 15 | Compute X-Gradient | $F_x^{(k+1)} = \text{reshape}(D_x \cdot \text{vec}(F^{(k+1)}), (M, N))$ | $D_x$, $F^{(k+1)}$ | $F_x^{(k+1)} \in \mathbb{R}^{M \times N}$ | [FORWARD] |
| 16 | Compute Y-Gradient | $F_y^{(k+1)} = \text{reshape}(D_y \cdot \text{vec}(F^{(k+1)}), (M, N))$ | $D_y$, $F^{(k+1)}$ | $F_y^{(k+1)} \in \mathbb{R}^{M \times N}$ | [FORWARD] |
| 17 | Compute Shrinkage Input X | $u_x = F_x^{(k+1)} - \phi_x + \Lambda_x^{(k)}$ | $F_x^{(k+1)}$, $\phi_x$, $\Lambda_x^{(k)}$ | $u_x \in \mathbb{R}^{M \times N}$ | — |
| 18 | Compute Shrinkage Input Y | $u_y = F_y^{(k+1)} - \phi_y + \Lambda_y^{(k)}$ | $F_y^{(k+1)}$, $\phi_y$, $\Lambda_y^{(k)}$ | $u_y \in \mathbb{R}^{M \times N}$ | — |
| 19 | Stack Shrinkage Inputs | $\mathbf{U} = \text{stack}(u_x, u_y) \in \mathbb{R}^{2 \times M \times N}$ | $u_x$, $u_y$ | $\mathbf{U}$ | Stack along axis 0 |
| 20 | Compute Magnitude | $\text{mag}[i,j] = \sqrt{u_x[i,j]^2 + u_y[i,j]^2}$ | $\mathbf{U}$ | $\text{mag} \in \mathbb{R}^{M \times N}$ | Pointwise L2 norm |
| 21 | Compute Safe Denominator | $\text{nonzero}[i,j] = \begin{cases} 1 & \text{if } \text{mag}[i,j] = 0 \\ \text{mag}[i,j] & \text{otherwise} \end{cases}$ | $\text{mag}$ | $\text{nonzero} \in \mathbb{R}^{M \times N}$ | Avoid division by zero |
| 22 | Compute Shrinkage Factor | $s[i,j] = \frac{\max\left(\text{mag}[i,j] - \lambda^{2-p} \cdot (\text{nonzero}[i,j]^2 + \epsilon)^{p/2 - 0.5}, 0\right)}{\text{nonzero}[i,j]}$ | $\text{mag}$, $\text{nonzero}$, $\lambda=2.0$, $p=0.01$, $\epsilon=0$ | $s \in \mathbb{R}^{M \times N}$ | Generalized p-shrinkage coefficient |
| 23 | Apply Shrinkage | $\mathbf{W} = s \odot \mathbf{U}$ (broadcast over axis 0) | $s$, $\mathbf{U}$ | $\mathbf{W} \in \mathbb{R}^{2 \times M \times N}$ | Elementwise scaling |
| 24 | Extract Auxiliary X | $w_x^{(k+1)} = \mathbf{W}[0]$ | $\mathbf{W}$ | $w_x^{(k+1)} \in \mathbb{R}^{M \times N}$ | — |
| 25 | Extract Auxiliary Y | $w_y^{(k+1)} = \mathbf{W}[1]$ | $\mathbf{W}$ | $w_y^{(k+1)} \in \mathbb{R}^{M \times N}$ | — |
| 26 | Update Dual X | $\Lambda_x^{(k+1)} = \Lambda_x^{(k)} + c \cdot (F_x^{(k+1)} - \phi_x - w_x^{(k+1)})$ | $\Lambda_x^{(k)}$, $F_x^{(k+1)}$, $\phi_x$, $w_x^{(k+1)}$, $c=1.6$ | $\Lambda_x^{(k+1)}$ | Scaled dual ascent |
| 27 | Update Dual Y | $\Lambda_y^{(k+1)} = \Lambda_y^{(k)} + c \cdot (F_y^{(k+1)} - \phi_y - w_y^{(k+1)})$ | $\Lambda_y^{(k)}$, $F_y^{(k+1)}$, $\phi_y$, $w_y^{(k+1)}$, $c=1.6$ | $\Lambda_y^{(k+1)}$ | Scaled dual ascent |
| 28 | Compute Change | $\Delta = \max_{i,j} |F^{(k+1)}[i,j] - F_{\text{old}}^{(k)}[i,j]|$ | $F^{(k+1)}$, $F_{\text{old}}^{(k)}$ | $\Delta \in \mathbb{R}$ | L∞ norm of difference |
| 29 | Check Convergence | If $\Delta < \tau$ OR $\text{isnan}(\Delta)$: EXIT LOOP | $\Delta$, $\tau=0.08$ | convergence flag | — |
| 30 | Update Previous Estimate | $F_{\text{old}}^{(k+1)} = \text{copy}(F^{(k+1)})$ | $F^{(k+1)}$ | $F_{\text{old}}^{(k+1)}$ | Deep copy required |
| 31 | Increment Counter | $k \leftarrow k + 1$ | $k$ | $k$ | — |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 32 | Return Unwrapped Phase | $F^* = F^{(k+1)}$ | $F^{(k+1)}$ | $F^* \in \mathbb{R}^{M \times N}$ | Final estimate |
| 33 | Save Result | $\text{np.save}(\texttt{io/output.npy}, F^*)$ | $F^*$ | file | NumPy binary format |

---

## 5. STOPPING CRITERION

- **type**: combined
- **expression**: 
$$\text{STOP if } \left( \max_{i,j} |F^{(k+1)}_{i,j} - F^{(k)}_{i,j}| < \tau \right) \text{ OR } \left( k \geq k_{\max} \right) \text{ OR } \text{isnan}(\Delta)$$
- **parameters**:
  - $\tau = 0.08$ (convergence tolerance)
  - $k_{\max} = 50$ (maximum iterations)
- **check_frequency**: Every iteration, evaluated at step 29

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Minimum Phase | $F_{\min} = \min_{i,j} F^*[i,j]$ | — | $F^*$ | Minimum value of unwrapped phase |
| Maximum Phase | $F_{\max} = \max_{i,j} F^*[i,j]$ | — | $F^*$ | Maximum value of unwrapped phase |
| Mean Phase | $\bar{F} = \frac{1}{MN}\sum_{i,j} F^*[i,j]$ | — | $F^*$ | Mean of unwrapped phase |
| Standard Deviation | $\sigma_F = \sqrt{\frac{1}{MN}\sum_{i,j}(F^*[i,j] - \bar{F})^2}$ | — | $F^*$ | Standard deviation of unwrapped phase |
| Iteration Change | $\Delta^{(k)} = \max_{i,j}|F^{(k+1)}[i,j] - F^{(k)}[i,j]|$ | $F^{(k)}$ | $F^{(k+1)}$ | Per-iteration convergence metric (L∞ norm) |