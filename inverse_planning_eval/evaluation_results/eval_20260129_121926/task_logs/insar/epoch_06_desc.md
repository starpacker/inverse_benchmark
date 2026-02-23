## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "ADMM Phase Unwrapping",
  "algorithm_family": "Alternating Direction Method of Multipliers (ADMM)",
  "summary": "Solves InSAR phase unwrapping by minimizing a p-norm penalty on the difference between the gradient of the unwrapped phase and the wrapped gradient estimates, using variable splitting and DCT-based linear solves."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{F} \left\| \nabla F - \boldsymbol{\phi} \right\|_p$$

where $\nabla F = (D_x F, D_y F)$ and $\boldsymbol{\phi} = (\phi_x, \phi_y)$.

**ADMM reformulation** (with variable splitting):
$$\min_{F, \mathbf{w}} \left\| \mathbf{w} \right\|_p \quad \text{subject to} \quad \nabla F - \boldsymbol{\phi} = \mathbf{w}$$

**Augmented Lagrangian**:
$$\mathcal{L}(F, \mathbf{w}, \boldsymbol{\Lambda}) = \left\| \mathbf{w} \right\|_p + \langle \boldsymbol{\Lambda}, \nabla F - \boldsymbol{\phi} - \mathbf{w} \rangle + \frac{1}{2}\left\| \nabla F - \boldsymbol{\phi} - \mathbf{w} \right\|_2^2$$

**data_fidelity_term**:
$$\frac{1}{2}\left\| D_x F - \phi_x - w_x \right\|_2^2 + \frac{1}{2}\left\| D_y F - \phi_y - w_y \right\|_2^2$$

**regularization_term**:
$$\left\| \mathbf{w} \right\|_p = \sum_{i,j} \left( w_x[i,j]^2 + w_y[i,j]^2 \right)^{p/2}$$

where $p \in [0, 1]$ (p=0 for L0 pseudo-norm, p=1 for L1 norm).

**constraints**: None (unconstrained optimization on $F$).

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $F$ | $(N_r, N_c)$ | float32 | $\mathbf{0}_{N_r \times N_c}$ | Unwrapped phase estimate |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Lambda_x$ | $(N_r, N_c)$ | float32 | $\mathbf{0}_{N_r \times N_c}$ | Lagrange multiplier for x-gradient constraint |
| $\Lambda_y$ | $(N_r, N_c)$ | float32 | $\mathbf{0}_{N_r \times N_c}$ | Lagrange multiplier for y-gradient constraint |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $w_x$ | $(N_r, N_c)$ | float32 | $\mathbf{0}_{N_r \times N_c}$ | ADMM splitting variable for x-gradient residual |
| $w_y$ | $(N_r, N_c)$ | float32 | $\mathbf{0}_{N_r \times N_c}$ | ADMM splitting variable for y-gradient residual |
| $F_{\text{old}}$ | $(N_r, N_c)$ | float32 | $\mathbf{0}_{N_r \times N_c}$ | Previous iteration phase for convergence check |
| $F_x$ | $(N_r, N_c)$ | float32 | computed | Gradient of $F$ in x-direction |
| $F_y$ | $(N_r, N_c)$ | float32 | computed | Gradient of $F$ in y-direction |
| $r_x$ | $(N_r \cdot N_c,)$ | float32 | computed | RHS component for x |
| $r_y$ | $(N_r \cdot N_c,)$ | float32 | computed | RHS component for y |
| $\text{RHS}$ | $(N_r \cdot N_c,)$ | float32 | computed | Right-hand side of normal equations |
| $\hat{\rho}$ | $(N_r, N_c)$ | float32 | computed | DCT of RHS |
| $\text{input}_x$ | $(N_r, N_c)$ | float32 | computed | Input to shrinkage for x |
| $\text{input}_y$ | $(N_r, N_c)$ | float32 | computed | Input to shrinkage for y |
| $\text{change}$ | scalar | float32 | computed | Maximum absolute change in $F$ |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $N_r$ | from data | Number of rows in interferogram |
| $N_c$ | from data | Number of columns in interferogram |
| $\lambda$ | 2.0 | Regularization parameter for shrinkage |
| $p$ | 0.01 | Norm parameter (near L0) |
| $c$ | 1.6 | Dual variable update scaling factor |
| $\epsilon$ | 0 | Mollification parameter for shrinkage |
| $\tau$ | 0.08 | Convergence tolerance |
| $K_{\max}$ | 50 | Maximum number of iterations |
| boundary | "neumann" | Boundary condition type |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $I$ | $(N_r, N_c)$ | complex64 | Raw complex interferogram |
| $f_{\text{wrapped}}$ | $(N_r, N_c)$ | float32 | Wrapped phase: $\angle(I)$ |
| $\text{mag}$ | $(N_r, N_c)$ | float32 | Interferogram magnitude: $|I|$ |
| $\phi_x$ | $(N_r, N_c)$ | float32 | Wrapped gradient estimate in x |
| $\phi_y$ | $(N_r, N_c)$ | float32 | Wrapped gradient estimate in y |
| $D_x$ | $(N_r N_c, N_r N_c)$ | sparse float32 | Differentiation matrix for x-direction |
| $D_y$ | $(N_r N_c, N_r N_c)$ | sparse float32 | Differentiation matrix for y-direction |
| $K$ | $(N_r, N_c)$ | float32 | Inverse Laplacian eigenvalues (DCT domain) |

## 3. PREPROCESSING PIPELINE

### Step P1: Load Interferogram
- **step_order**: P1
- **step_name**: Load Complex Interferogram
- **formula**: $I = \text{fromfile}(\texttt{filename}, \texttt{complex64}).\text{reshape}(-1, N_c)$
- **inputs**: filename (string), $N_c$ from RSC metadata
- **outputs**: $I \in \mathbb{C}^{N_r \times N_c}$
- **assumptions**: Binary file in row-major complex64 format

### Step P2: Extract Magnitude
- **step_order**: P2
- **step_name**: Compute Magnitude
- **formula**: $\text{mag}[i,j] = |I[i,j]| = \sqrt{\text{Re}(I[i,j])^2 + \text{Im}(I[i,j])^2}$
- **inputs**: $I \in \mathbb{C}^{N_r \times N_c}$
- **outputs**: $\text{mag} \in \mathbb{R}^{N_r \times N_c}$
- **assumptions**: None

### Step P3: Extract Wrapped Phase
- **step_order**: P3
- **step_name**: Compute Wrapped Phase
- **formula**: $f_{\text{wrapped}}[i,j] = \text{atan2}(\text{Im}(I[i,j]), \text{Re}(I[i,j])) \in (-\pi, \pi]$
- **inputs**: $I \in \mathbb{C}^{N_r \times N_c}$
- **outputs**: $f_{\text{wrapped}} \in \mathbb{R}^{N_r \times N_c}$
- **assumptions**: None

### Step P4: Construct x-Differentiation Matrix
- **step_order**: P4
- **step_name**: Build $D_x$ with Neumann BC
- **formula**: 
  - Base 1D operator $D \in \mathbb{R}^{N_c \times N_c}$: $D[i,i] = -1$, $D[i,i+1] = 1$ for $i = 0, \ldots, N_c-2$
  - Neumann: $D[N_c-1, N_c-1] = 0$
  - $D_x = I_{N_r} \otimes D$ (Kronecker product)
- **inputs**: $N_r$, $N_c$
- **outputs**: $D_x \in \mathbb{R}^{N_r N_c \times N_r N_c}$ (sparse CSR)
- **assumptions**: Forward difference scheme

### Step P5: Construct y-Differentiation Matrix
- **step_order**: P5
- **step_name**: Build $D_y$ with Neumann BC
- **formula**: 
  - Base 1D operator $D \in \mathbb{R}^{N_r \times N_r}$: $D[i,i] = -1$, $D[i,i+1] = 1$ for $i = 0, \ldots, N_r-2$
  - Neumann: $D[N_r-1, N_r-1] = 0$
  - $D_y = D \otimes I_{N_c}$ (Kronecker product)
- **inputs**: $N_r$, $N_c$
- **outputs**: $D_y \in \mathbb{R}^{N_r N_c \times N_r N_c}$ (sparse CSR)
- **assumptions**: Forward difference scheme

### Step P6: Compute Raw Gradients
- **step_order**: P6
- **step_name**: Apply Differentiation to Wrapped Phase
- **formula**: 
  - $\tilde{\phi}_x = (D_x \cdot \text{vec}(f_{\text{wrapped}})).\text{reshape}(N_r, N_c)$
  - $\tilde{\phi}_y = (D_y \cdot \text{vec}(f_{\text{wrapped}})).\text{reshape}(N_r, N_c)$
- **inputs**: $f_{\text{wrapped}} \in \mathbb{R}^{N_r \times N_c}$, $D_x$, $D_y$
- **outputs**: $\tilde{\phi}_x, \tilde{\phi}_y \in \mathbb{R}^{N_r \times N_c}$
- **assumptions**: None

### Step P7: Wrap Gradient Adjustment (x-component)
- **step_order**: P7
- **step_name**: Apply Wrapping Correction to x-gradient
- **formula**: 
$$\phi_x[i,j] = \begin{cases} \tilde{\phi}_x[i,j] - 2\pi \cdot \text{sign}(\tilde{\phi}_x[i,j]) & \text{if } |\tilde{\phi}_x[i,j]| > \pi \\ \tilde{\phi}_x[i,j] & \text{otherwise} \end{cases}$$
- **inputs**: $\tilde{\phi}_x \in \mathbb{R}^{N_r \times N_c}$
- **outputs**: $\phi_x \in \mathbb{R}^{N_r \times N_c}$
- **assumptions**: Gradients should be in $(-\pi, \pi]$

### Step P8: Wrap Gradient Adjustment (y-component)
- **step_order**: P8
- **step_name**: Apply Wrapping Correction to y-gradient
- **formula**: 
$$\phi_y[i,j] = \begin{cases} \tilde{\phi}_y[i,j] - 2\pi \cdot \text{sign}(\tilde{\phi}_y[i,j]) & \text{if } |\tilde{\phi}_y[i,j]| > \pi \\ \tilde{\phi}_y[i,j] & \text{otherwise} \end{cases}$$
- **inputs**: $\tilde{\phi}_y \in \mathbb{R}^{N_r \times N_c}$
- **outputs**: $\phi_y \in \mathbb{R}^{N_r \times N_c}$
- **assumptions**: Gradients should be in $(-\pi, \pi]$

### Step P9: Compute Laplacian Eigenvalue Grid
- **step_order**: P9
- **step_name**: Generate DCT Eigenvalue Arrays
- **formula**: 
  - $\xi_y[k] = 2 - 2\cos\left(\frac{\pi k}{N_r}\right)$ for $k = 0, \ldots, N_r - 1$
  - $\xi_x[\ell] = 2 - 2\cos\left(\frac{\pi \ell}{N_c}\right)$ for $\ell = 0, \ldots, N_c - 1$
  - $\text{eigvals}[k, \ell] = \xi_y[k] + \xi_x[\ell]$
- **inputs**: $N_r$, $N_c$
- **outputs**: $\text{eigvals} \in \mathbb{R}^{N_r \times N_c}$
- **assumptions**: Neumann BC implies DCT-II diagonalization

### Step P10: Compute Inverse Laplacian Kernel
- **step_order**: P10
- **step_name**: Invert Eigenvalues with Zero Handling
- **formula**: 
$$K[k, \ell] = \begin{cases} 0 & \text{if } \text{eigvals}[k, \ell] = 0 \\ \frac{1}{\text{eigvals}[k, \ell]} & \text{otherwise} \end{cases}$$
- **inputs**: $\text{eigvals} \in \mathbb{R}^{N_r \times N_c}$
- **outputs**: $K \in \mathbb{R}^{N_r \times N_c}$
- **assumptions**: $K[0,0] = 0$ handles the null space of Laplacian

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $\Lambda_x$ | $\mathbf{0}$ | $(N_r, N_c)$ | float32 | np.zeros_like(phi_x) |
| 2 | $\Lambda_y$ | $\mathbf{0}$ | $(N_r, N_c)$ | float32 | np.zeros_like(phi_y) |
| 3 | $w_x$ | $\mathbf{0}$ | $(N_r, N_c)$ | float32 | np.zeros_like(phi_x) |
| 4 | $w_y$ | $\mathbf{0}$ | $(N_r, N_c)$ | float32 | np.zeros_like(phi_y) |
| 5 | $F_{\text{old}}$ | $\mathbf{0}$ | $(N_r, N_c)$ | float32 | np.zeros_like(f_wrapped) |
| 6 | $F$ | $\mathbf{0}$ | $(N_r, N_c)$ | float32 | np.zeros_like(f_wrapped) |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, K_{\max} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 7 | Compute x-residual vector | $r_x = \text{vec}(w_x) + \text{vec}(\phi_x) - \text{vec}(\Lambda_x)$ | $w_x \in \mathbb{R}^{N_r \times N_c}$, $\phi_x \in \mathbb{R}^{N_r \times N_c}$, $\Lambda_x \in \mathbb{R}^{N_r \times N_c}$ | $r_x \in \mathbb{R}^{N_r N_c}$ | Element-wise vector addition |
| 8 | Compute y-residual vector | $r_y = \text{vec}(w_y) + \text{vec}(\phi_y) - \text{vec}(\Lambda_y)$ | $w_y \in \mathbb{R}^{N_r \times N_c}$, $\phi_y \in \mathbb{R}^{N_r \times N_c}$, $\Lambda_y \in \mathbb{R}^{N_r \times N_c}$ | $r_y \in \mathbb{R}^{N_r N_c}$ | Element-wise vector addition |
| 9 | Apply adjoint x-operator [ADJOINT] | $\text{RHS}_x = D_x^T r_x$ | $D_x \in \mathbb{R}^{N_r N_c \times N_r N_c}$, $r_x \in \mathbb{R}^{N_r N_c}$ | $\text{RHS}_x \in \mathbb{R}^{N_r N_c}$ | Sparse matrix-vector multiply |
| 10 | Apply adjoint y-operator [ADJOINT] | $\text{RHS}_y = D_y^T r_y$ | $D_y \in \mathbb{R}^{N_r N_c \times N_r N_c}$, $r_y \in \mathbb{R}^{N_r N_c}$ | $\text{RHS}_y \in \mathbb{R}^{N_r N_c}$ | Sparse matrix-vector multiply |
| 11 | Sum RHS components | $\text{RHS} = \text{RHS}_x + \text{RHS}_y$ | $\text{RHS}_x \in \mathbb{R}^{N_r N_c}$, $\text{RHS}_y \in \mathbb{R}^{N_r N_c}$ | $\text{RHS} \in \mathbb{R}^{N_r N_c}$ | Element-wise addition |
| 12 | Reshape RHS to 2D | $\text{RHS}_{2D} = \text{RHS}.\text{reshape}(N_r, N_c)$ | $\text{RHS} \in \mathbb{R}^{N_r N_c}$ | $\text{RHS}_{2D} \in \mathbb{R}^{N_r \times N_c}$ | Memory view reshape |
| 13 | Forward DCT-II | $\hat{\rho} = \text{DCT-II}(\text{RHS}_{2D}, \text{norm}=\texttt{'ortho'})$ | $\text{RHS}_{2D} \in \mathbb{R}^{N_r \times N_c}$ | $\hat{\rho} \in \mathbb{R}^{N_r \times N_c}$ | 2D orthonormal DCT type 2 |
| 14 | Apply inverse Laplacian in DCT domain | $\hat{F} = \hat{\rho} \odot K$ | $\hat{\rho} \in \mathbb{R}^{N_r \times N_c}$, $K \in \mathbb{R}^{N_r \times N_c}$ | $\hat{F} \in \mathbb{R}^{N_r \times N_c}$ | Element-wise multiplication |
| 15 | Inverse DCT-II | $F = \text{IDCT-II}(\hat{F}, \text{norm}=\texttt{'ortho'})$ | $\hat{F} \in \mathbb{R}^{N_r \times N_c}$ | $F \in \mathbb{R}^{N_r \times N_c}$ | 2D orthonormal inverse DCT type 2 |
| 16 | Compute x-gradient [FORWARD] | $F_x = (D_x \cdot \text{vec}(F)).\text{reshape}(N_r, N_c)$ | $D_x \in \mathbb{R}^{N_r N_c \times N_r N_c}$, $F \in \mathbb{R}^{N_r \times N_c}$ | $F_x \in \mathbb{R}^{N_r \times N_c}$ | Sparse matvec then reshape |
| 17 | Compute y-gradient [FORWARD] | $F_y = (D_y \cdot \text{vec}(F)).\text{reshape}(N_r, N_c)$ | $D_y \in \mathbb{R}^{N_r N_c \times N_r N_c}$, $F \in \mathbb{R}^{N_r \times N_c}$ | $F_y \in \mathbb{R}^{N_r \times N_c}$ | Sparse matvec then reshape |
| 18 | Compute shrinkage input x | $\text{input}_x = F_x - \phi_x + \Lambda_x$ | $F_x \in \mathbb{R}^{N_r \times N_c}$, $\phi_x \in \mathbb{R}^{N_r \times N_c}$, $\Lambda_x \in \mathbb{R}^{N_r \times N_c}$ | $\text{input}_x \in \mathbb{R}^{N_r \times N_c}$ | Element-wise |
| 19 | Compute shrinkage input y | $\text{input}_y = F_y - \phi_y + \Lambda_y$ | $F_y \in \mathbb{R}^{N_r \times N_c}$, $\phi_y \in \mathbb{R}^{N_r \times N_c}$, $\Lambda_y \in \mathbb{R}^{N_r \times N_c}$ | $\text{input}_y \in \mathbb{R}^{N_r \times N_c}$ | Element-wise |
| 20 | Stack inputs | $X = \text{stack}([\text{input}_x, \text{input}_y], \text{axis}=0)$ | $\text{input}_x, \text{input}_y \in \mathbb{R}^{N_r \times N_c}$ | $X \in \mathbb{R}^{2 \times N_r \times N_c}$ | Array concatenation |
| 21 | Compute vector magnitude | $\text{mag}[i,j] = \sqrt{X[0,i,j]^2 + X[1,i,j]^2}$ | $X \in \mathbb{R}^{2 \times N_r \times N_c}$ | $\text{mag} \in \mathbb{R}^{N_r \times N_c}$ | Point-wise L2 norm across channel axis |
| 22 | Handle zero magnitudes | $\text{nonzero}[i,j] = \begin{cases} 1 & \text{if } \text{mag}[i,j] = 0 \\ \text{mag}[i,j] & \text{otherwise} \end{cases}$ | $\text{mag} \in \mathbb{R}^{N_r \times N_c}$ | $\text{nonzero} \in \mathbb{R}^{N_r \times N_c}$ | Avoid division by zero |
| 23 | Compute shrinkage scale | $\text{scale}[i,j] = \frac{\max\left(\text{mag}[i,j] - \lambda^{2-p} \cdot (\text{nonzero}[i,j]^2 + \epsilon)^{p/2 - 0.5}, 0\right)}{\text{nonzero}[i,j]}$ | $\text{mag}, \text{nonzero} \in \mathbb{R}^{N_r \times N_c}$, $\lambda=2.0$, $p=0.01$, $\epsilon=0$ | $\text{scale} \in \mathbb{R}^{N_r \times N_c}$ | Generalized p-shrinkage threshold |
| 24 | Apply scale to x-input | $w_x = \text{scale} \odot \text{input}_x$ | $\text{scale}, \text{input}_x \in \mathbb{R}^{N_r \times N_c}$ | $w_x \in \mathbb{R}^{N_r \times N_c}$ | Element-wise multiplication |
| 25 | Apply scale to y-input | $w_y = \text{scale} \odot \text{input}_y$ | $\text{scale}, \text{input}_y \in \mathbb{R}^{N_r \times N_c}$ | $w_y \in \mathbb{R}^{N_r \times N_c}$ | Element-wise multiplication |
| 26 | Compute x-constraint residual | $\Delta_x = F_x - \phi_x - w_x$ | $F_x, \phi_x, w_x \in \mathbb{R}^{N_r \times N_c}$ | $\Delta_x \in \mathbb{R}^{N_r \times N_c}$ | Element-wise |
| 27 | Compute y-constraint residual | $\Delta_y = F_y - \phi_y - w_y$ | $F_y, \phi_y, w_y \in \mathbb{R}^{N_r \times N_c}$ | $\Delta_y \in \mathbb{R}^{N_r \times N_c}$ | Element-wise |
| 28 | Update x dual variable | $\Lambda_x \leftarrow \Lambda_x + c \cdot \Delta_x$ | $\Lambda_x, \Delta_x \in \mathbb{R}^{N_r \times N_c}$, $c = 1.6$ | $\Lambda_x \in \mathbb{R}^{N_r \times N_c}$ | Scaled dual ascent |
| 29 | Update y dual variable | $\Lambda_y \leftarrow \Lambda_