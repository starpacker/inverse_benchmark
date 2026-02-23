# Algorithmic Specification Document: InSAR Phase Unwrapping via ADMM

## 1. Mathematical Formulation

### 1.1 Problem Statement

**Objective**: Recover the unwrapped (true) phase $F \in \mathbb{R}^{M \times N}$ from a wrapped phase observation $f_{\text{wrapped}} \in [-\pi, \pi)^{M \times N}$.

### 1.2 Optimization Problem

The algorithm solves the following sparse gradient recovery problem:

$$
\min_{F} \left\| \nabla F - \boldsymbol{\phi} \right\|_p
$$

where:
- $\nabla F = (D_x F, D_y F)$ denotes the discrete gradient of $F$
- $\boldsymbol{\phi} = (\phi_x, \phi_y)$ is the **wrapped gradient** (observed data)
- $\|\cdot\|_p$ is the vector $\ell_p$-norm applied column-wise (with $p \in [0,1]$)

### 1.3 ADMM Reformulation

Introducing auxiliary splitting variables $\mathbf{w} = (w_x, w_y)$:

$$
\min_{F, w_x, w_y} \left\| \mathbf{w} \right\|_p \quad \text{subject to} \quad \nabla F - \boldsymbol{\phi} = \mathbf{w}
$$

**Augmented Lagrangian**:

$$
\mathcal{L}_\rho(F, \mathbf{w}, \boldsymbol{\Lambda}) = \|\mathbf{w}\|_p + \langle \boldsymbol{\Lambda}, \nabla F - \boldsymbol{\phi} - \mathbf{w} \rangle + \frac{\rho}{2}\|\nabla F - \boldsymbol{\phi} - \mathbf{w}\|_2^2
$$

where $\boldsymbol{\Lambda} = (\Lambda_x, \Lambda_y)$ are dual variables (Lagrange multipliers).

### 1.4 Component Definitions

**Discrete Gradient Operators** (Neumann boundary conditions):

$$
D_x = I_M \otimes D^{(N)}, \quad D_y = D^{(M)} \otimes I_N
$$

where the 1D forward difference matrix $D^{(K)} \in \mathbb{R}^{K \times K}$:

$$
D^{(K)}_{ij} = \begin{cases}
-1 & \text{if } i = j \text{ and } i < K \\
+1 & \text{if } j = i+1 \text{ and } i < K \\
0 & \text{otherwise}
\end{cases}
$$

**Wrapped Gradient Adjustment** (Equations 2-3 in Itoh's method):

$$
\phi_x[i,j] \leftarrow \phi_x[i,j] - 2\pi \cdot \text{sign}(\phi_x[i,j]) \quad \text{if } |\phi_x[i,j]| > \pi
$$

$$
\phi_y[i,j] \leftarrow \phi_y[i,j] - 2\pi \cdot \text{sign}(\phi_y[i,j]) \quad \text{if } |\phi_y[i,j]| > \pi
$$

**p-Shrinkage Operator** (proximal operator for $\ell_p$ norm):

$$
\mathcal{S}_p(\mathbf{X}; \lambda, \epsilon) = \frac{\max\left( \|\mathbf{X}\|_2 - \lambda^{2-p}\left(\|\mathbf{X}\|_2^2 + \epsilon\right)^{\frac{p}{2} - \frac{1}{2}}, 0 \right)}{\|\mathbf{X}\|_2} \cdot \mathbf{X}
$$

where $\|\mathbf{X}\|_2 = \sqrt{\sum_k X_k^2}$ is computed along the stacked gradient dimension.

**Laplacian Eigenvalues** (for DCT-II with Neumann BC):

$$
\xi_y[i] = 2 - 2\cos\left(\frac{\pi i}{M}\right), \quad i = 0, \ldots, M-1
$$

$$
\xi_x[j] = 2 - 2\cos\left(\frac{\pi j}{N}\right), \quad j = 0, \ldots, N-1
$$

$$
K[i,j] = \begin{cases}
\frac{1}{\xi_y[i] + \xi_x[j]} & \text{if } \xi_y[i] + \xi_x[j] \neq 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 2. Variables Taxonomy

### 2.1 Observed Data / Inputs

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| Complex interferogram | $\text{igram}$ | $(M, N)$ | `complex64` | Raw input data |
| Wrapped phase | $f_{\text{wrapped}}$ | $(M, N)$ | `float32` | $f_{\text{wrapped}} = \angle(\text{igram})$ |
| Magnitude | $\text{mag}$ | $(M, N)$ | `float32` | $\text{mag} = |\text{igram}|$ |
| Wrapped gradient (x) | $\phi_x$ | $(M, N)$ | `float32` | Adjusted wrapped gradient in x-direction |
| Wrapped gradient (y) | $\phi_y$ | $(M, N)$ | `float32` | Adjusted wrapped gradient in y-direction |

### 2.2 Primal / Optimization Variables

| Variable | Symbol | Shape | Dtype | Initialization |
|----------|--------|-------|-------|----------------|
| Unwrapped phase | $F$ | $(M, N)$ | `float32` | $F^{(0)} = \mathbf{0}_{M \times N}$ |

### 2.3 Auxiliary / Dual Variables

| Variable | Symbol | Shape | Dtype | Initialization |
|----------|--------|-------|-------|----------------|
| Auxiliary (x) | $w_x$ | $(M, N)$ | `float32` | $w_x^{(0)} = \mathbf{0}_{M \times N}$ |
| Auxiliary (y) | $w_y$ | $(M, N)$ | `float32` | $w_y^{(0)} = \mathbf{0}_{M \times N}$ |
| Lagrange multiplier (x) | $\Lambda_x$ | $(M, N)$ | `float32` | $\Lambda_x^{(0)} = \mathbf{0}_{M \times N}$ |
| Lagrange multiplier (y) | $\Lambda_y$ | $(M, N)$ | `float32` | $\Lambda_y^{(0)} = \mathbf{0}_{M \times N}$ |
| Previous iterate | $F_{\text{old}}$ | $(M, N)$ | `float32` | $F_{\text{old}}^{(0)} = \mathbf{0}_{M \times N}$ |

### 2.4 Precomputed Operators / Constants

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| Differentiation matrix (x) | $D_x$ | $(MN, MN)$ | sparse `float32` | $D_x = I_M \otimes D^{(N)}$ |
| Differentiation matrix (y) | $D_y$ | $(MN, MN)$ | sparse `float32` | $D_y = D^{(M)} \otimes I_N$ |
| Laplacian kernel | $K$ | $(M, N)$ | `float32` | Inverse eigenvalues of Laplacian |

### 2.5 Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Maximum iterations | $N_{\text{iter}}$ | 500 | Upper bound on ADMM iterations |
| Convergence tolerance | $\tau$ | 1.6 | Threshold for $\|F^{(k)} - F^{(k-1)}\|_\infty$ |
| Regularization parameter | $\lambda$ | 1.0 | Controls shrinkage strength |
| Norm parameter | $p$ | 0 | Sparsity-inducing norm ($p \in [0,1]$) |
| Dual update factor | $c$ | 1.3 | Step size for Lagrange multiplier update |
| Mollification parameter | $\epsilon$ | 0 | Smoothing for p-shrinkage |
| Boundary conditions | — | "neumann" | Type of boundary handling |

---

## 3. Preprocessing Pipeline

| Step | Operation | Input | Output | Formula |
|------|-----------|-------|--------|---------|
| 1 | Load interferogram | filename | $\text{igram} \in \mathbb{C}^{M \times N}$ | Binary file read + reshape |
| 2 | Extract magnitude | $\text{igram}$ | $\text{mag} \in \mathbb{R}^{M \times N}$ | $\text{mag}[i,j] = |\text{igram}[i,j]|$ |
| 3 | Extract wrapped phase | $\text{igram}$ | $f_{\text{wrapped}} \in \mathbb{R}^{M \times N}$ | $f_{\text{wrapped}}[i,j] = \angle(\text{igram}[i,j])$ |
| 4 | Build differentiation matrices | $(M, N, \text{BC})$ | $(D_x, D_y)$ | See Section 1.4 |
| 5 | Compute raw gradients | $(f_{\text{wrapped}}, D_x, D_y)$ | $(\tilde{\phi}_x, \tilde{\phi}_y)$ | $\tilde{\phi}_x = D_x \cdot \text{vec}(f_{\text{wrapped}})$, $\tilde{\phi}_y = D_y \cdot \text{vec}(f_{\text{wrapped}})$ |
| 6 | Apply wrapping correction | $(\tilde{\phi}_x, \tilde{\phi}_y)$ | $(\phi_x, \phi_y)$ | See Section 1.4 (wrapped adjustment) |
| 7 | Compute Laplacian kernel | $(M, N)$ | $K \in \mathbb{R}^{M \times N}$ | See Section 1.4 |

---

## 4. Algorithm Steps (ADMM Iteration)

### Initialization
Set $k = 0$, initialize all variables to zero matrices as specified in Section 2.

### Per-Iteration Updates

For $k = 0, 1, 2, \ldots, N_{\text{iter}} - 1$:

---

**Step 1: F-Update (Primal Variable — Poisson Solve via DCT)**

| Aspect | Specification |
|--------|---------------|
| **Name** | Phase Update |
| **Input** | $w_x^{(k)}, w_y^{(k)}, \phi_x, \phi_y, \Lambda_x^{(k)}, \Lambda_y^{(k)}, D_x, D_y, K$ |
| **Output** | $F^{(k+1)}$ |

**Formula**:

$$
r_x = \text{vec}(w_x^{(k)} + \phi_x - \Lambda_x^{(k)})
$$

$$
r_y = \text{vec}(w_y^{(k)} + \phi_y - \Lambda_y^{(k)})
$$

$$
\text{RHS} = D_x^\top r_x + D_y^\top r_y
$$

$$
\hat{\rho} = \text{DCT-II}\left(\text{reshape}(\text{RHS}, M, N)\right)
$$

$$
F^{(k+1)} = \text{IDCT-II}\left(\hat{\rho} \odot K\right)
$$

where DCT-II uses orthonormal normalization.

---

**Step 2: Forward Operator (Gradient Computation)**

| Aspect | Specification |
|--------|---------------|
| **Name** | Gradient Computation |
| **Input** | $F^{(k+1)}, D_x, D_y$ |
| **Output** | $F_x^{(k+1)}, F_y^{(k+1)}$ |

**Formula**:

$$
F_x^{(k+1)} = \text{reshape}\left(D_x \cdot \text{vec}(F^{(k+1)}), M, N\right)
$$

$$
F_y^{(k+1)} = \text{reshape}\left(D_y \cdot \text{vec}(F^{(k+1)}), M, N\right)
$$

---

**Step 3: w-Update (Auxiliary Variable — Shrinkage)**

| Aspect | Specification |
|--------|---------------|
| **Name** | Shrinkage Step |
| **Input** | $F_x^{(k+1)}, F_y^{(k+1)}, \phi_x, \phi_y, \Lambda_x^{(k)}, \Lambda_y^{(k)}, \lambda, p, \epsilon$ |
| **Output** | $w_x^{(k+1)}, w_y^{(k+1)}$ |

**Formula**:

$$
\text{input}_x = F_x^{(k+1)} - \phi_x + \Lambda_x^{(k)}
$$

$$
\text{input}_y = F_y^{(k+1)} - \phi_y + \Lambda_y^{(k)}
$$

$$
\mathbf{X} = \begin{pmatrix} \text{input}_x \\ \text{input}_y \end{pmatrix} \in \mathbb{R}^{2 \times M \times N}
$$

$$
\|\mathbf{X}\|[i,j] = \sqrt{\text{input}_x[i,j]^2 + \text{input}_y[i,j]^2}
$$

$$
\tilde{m}[i,j] = \begin{cases} \|\mathbf{X}\|[i,j] & \text{if } \|\mathbf{X}\|[i,j] \neq 0 \\ 1 & \text{otherwise} \end{cases}
$$

$$
s[i,j] = \frac{\max\left(\|\mathbf{X}\|[i,j] - \lambda^{2-p}\left(\tilde{m}[i,j]^2 + \epsilon\right)^{\frac{p-1}{2}}, 0\right)}{\tilde{m}[i,j]}
$$

$$
w_x^{(k+1)} = s \odot \text{input}_x, \quad w_y^{(k+1)} = s \odot \text{input}_y
$$

---

**Step 4: Λ-Update (Dual Variable)**

| Aspect | Specification |
|--------|---------------|
| **Name** | Lagrange Multiplier Update |
| **Input** | $\Lambda_x^{(k)}, \Lambda_y^{(k)}, F_x^{(k+1)}, F_y^{(k+1)}, \phi_x, \phi_y, w_x^{(k+1)}, w_y^{(k+1)}, c$ |
| **Output** | $\Lambda_x^{(k+1)}, \Lambda_y^{(k+1)}$ |

**Formula**:

$$
\Lambda_x^{(k+1)} = \Lambda_x^{(k)} + c \left(F_x^{(k+1)} - \phi_x - w_x^{(k+1)}\right)
$$

$$
\Lambda_y^{(k+1)} = \Lambda_y^{(k)} + c \left(F_y^{(k+1)} - \phi_y - w_y^{(k+1)}\right)
$$

---

**Step 5: Convergence Check**

| Aspect | Specification |
|--------|---------------|
| **Name** | Convergence Evaluation |
| **Input** | $F^{(k+1)}, F^{(k)}$ |
| **Output** | Boolean: continue/stop |

**Formula**:

$$
\Delta^{(k+1)} = \max_{i,j} \left| F^{(k+1)}[i,j] - F^{(k)}[i,j] \right|
$$

Update: $F_{\text{old}} \leftarrow F^{(k+1)}$

---

## 5. Stopping Criterion

The algorithm terminates when **any** of the following conditions is met:

| Condition | Formula | Description |
|-----------|---------|-------------|
| Convergence | $\Delta^{(k+1)} < \tau$ | Maximum absolute change below tolerance |
| Numerical failure | $\Delta^{(k+1)} = \text{NaN}$ | Numerical instability detected |
| Iteration limit | $k \geq N_{\text{iter}} - 1$ | Maximum iterations reached |

**Output**: $F^{(k+1)}$ (final unwrapped phase estimate)

---

## 6. Evaluation Metrics

| Metric | Symbol | Formula |
|--------|--------|---------|
| Minimum | $F_{\min}$ | $\displaystyle\min_{i,j} F[i,j]$ |
| Maximum | $F_{\max}$ | $\displaystyle\max_{i,j} F[i,j]$ |
| Mean | $\bar{F}$ | $\displaystyle\frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N} F[i,j]$ |
| Standard Deviation | $\sigma_F$ | $\displaystyle\sqrt{\frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N}\left(F[i,j] - \bar{F}\right)^2}$ |
| Phase Range | — | $[F_{\min}, F_{\max}]$ |

---

## 7. Algorithm Summary (Pseudocode)

```
ALGORITHM: ADMM Phase Unwrapping

INPUT: igram ∈ ℂ^(M×N), λ, p, c, τ, N_iter
OUTPUT: F ∈ ℝ^(M×N)

─────────────────────────────────────────────────
PREPROCESSING:
─────────────────────────────────────────────────
1.  f_wrapped ← angle(igram)
2.  (D_x, D_y) ← BuildDifferentiationMatrices(M, N, "neumann")
3.  (φ_x, φ_y) ← EstimateWrappedGradient(f_wrapped, D_x, D_y)
4.  K ← BuildLaplacianKernel(M, N)

─────────────────────────────────────────────────
INITIALIZATION:
─────────────────────────────────────────────────
5.  F, F_old, w_x, w_y, Λ_x, Λ_y ← 0_(M×N)

─────────────────────────────────────────────────
ADMM LOOP: for k = 0, 1, ..., N_iter - 1
─────────────────────────────────────────────────
6.  r_x ← vec(w_x + φ_x - Λ_x)
7.  r_y ← vec(w_y + φ_y - Λ_y)
8.  RHS ← D_x^T r_x + D_y^T r_y
9.  ρ̂ ← DCT-II(reshape(RHS, M, N))
10. F ← IDCT-II(ρ̂ ⊙ K)
11. (F_x, F_y) ← ForwardOperator(F, D_x, D_y)
12. (w_x, w_y) ← pShrink([F_x - φ_x + Λ_x; F_y - φ_y + Λ_y], λ, p)
13. Λ_x ← Λ_x + c·(F_x - φ_x - w_x)
14. Λ_y ← Λ_y + c·(F_y - φ_y - w_y)
15. Δ ← max|F - F_old|
16. IF Δ < τ OR isnan(Δ): BREAK
17. F_old ← F

─────────────────────────────────────────────────
RETURN F
```