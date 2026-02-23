# Algorithmic Specification Document: ADMM-Based Lensless Image Reconstruction

## 1. Mathematical Formulation

### 1.1 Problem Statement

Reconstruct a latent image $\mathbf{x}$ from a lensless camera measurement $\mathbf{y}$ given a known point spread function (PSF) $\mathbf{h}$.

### 1.2 Forward Imaging Model

$$\mathbf{y} = \mathbf{h} \circledast \mathbf{x} + \boldsymbol{\epsilon}$$

where $\circledast$ denotes 2D convolution and $\boldsymbol{\epsilon}$ represents measurement noise.

### 1.3 Optimization Problem (Primal Form)

$$\min_{\mathbf{x}} \quad \frac{1}{2}\|\mathbf{H}\mathbf{x} - \mathbf{y}\|_2^2 + \tau \|\nabla \mathbf{x}\|_1 \quad \text{subject to} \quad \mathbf{x} \geq 0$$

**Term Decomposition:**
| Term | Type | Formula |
|------|------|---------|
| Data Fidelity | $\frac{1}{2}\|\mathbf{H}\mathbf{x} - \mathbf{y}\|_2^2$ | Squared $\ell_2$ norm of residual |
| TV Regularization | $\tau\|\nabla \mathbf{x}\|_1$ | Anisotropic Total Variation |
| Constraint | $\mathbf{x} \geq 0$ | Element-wise non-negativity |

where $\mathbf{H}$ is the linear convolution operator with kernel $\mathbf{h}$.

### 1.4 ADMM Reformulation (Variable Splitting)

Introduce auxiliary variables to decouple the problem:

$$\min_{\mathbf{x}, \mathbf{X}, \mathbf{U}, \mathbf{W}} \quad \frac{1}{2}\|\mathbf{X} - \mathbf{y}\|_2^2 + \tau\|\mathbf{U}\|_1 + \mathcal{I}_{\geq 0}(\mathbf{W})$$

$$\text{subject to:} \quad \mathbf{H}\mathbf{x} = \mathbf{X}, \quad \nabla\mathbf{x} = \mathbf{U}, \quad \mathbf{x} = \mathbf{W}$$

where $\mathcal{I}_{\geq 0}(\cdot)$ is the indicator function for the non-negative orthant.

### 1.5 Augmented Lagrangian

$$\mathcal{L}(\mathbf{x}, \mathbf{X}, \mathbf{U}, \mathbf{W}, \boldsymbol{\xi}, \boldsymbol{\eta}, \boldsymbol{\rho}) = \frac{1}{2}\|\mathbf{X} - \mathbf{y}\|_2^2 + \tau\|\mathbf{U}\|_1 + \mathcal{I}_{\geq 0}(\mathbf{W})$$

$$+ \langle \boldsymbol{\xi}, \mathbf{H}\mathbf{x} - \mathbf{X} \rangle + \frac{\mu_1}{2}\|\mathbf{H}\mathbf{x} - \mathbf{X}\|_2^2$$

$$+ \langle \boldsymbol{\eta}, \nabla\mathbf{x} - \mathbf{U} \rangle + \frac{\mu_2}{2}\|\nabla\mathbf{x} - \mathbf{U}\|_2^2$$

$$+ \langle \boldsymbol{\rho}, \mathbf{x} - \mathbf{W} \rangle + \frac{\mu_3}{2}\|\mathbf{x} - \mathbf{W}\|_2^2$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | Dtype | Initialization | Description |
|----------|--------|-------|-------|----------------|-------------|
| Latent Image | $\mathbf{x}$ | `[1, D, H_pad, W_pad, C]` | `float32` | $\mathbf{0}$ | Primary reconstruction variable |

### 2.2 Auxiliary Variables (ADMM Splitting)

| Variable | Symbol | Shape | Dtype | Initialization | Description |
|----------|--------|-------|-------|----------------|-------------|
| Data Fidelity Aux | $\mathbf{X}$ | `[1, D, H_pad, W_pad, C]` | `float32` | $\mathbf{0}$ | Proxy for $\mathbf{H}\mathbf{x}$ |
| TV Aux | $\mathbf{U}$ | `[1, D, H_pad, W_pad, C, 2]` | `float32` | $\mathbf{0}$ | Proxy for $\nabla\mathbf{x}$ |
| Non-negativity Aux | $\mathbf{W}$ | `[1, D, H_pad, W_pad, C]` | `float32` | $\mathbf{0}$ | Proxy for $\mathbf{x}$ |

### 2.3 Dual Variables (Lagrange Multipliers)

| Variable | Symbol | Shape | Dtype | Initialization | Description |
|----------|--------|-------|-------|----------------|-------------|
| Data Fidelity Dual | $\boldsymbol{\xi}$ | `[1, D, H_pad, W_pad, C]` | `float32` | $\mathbf{0}$ | Multiplier for $\mathbf{H}\mathbf{x} = \mathbf{X}$ |
| TV Dual | $\boldsymbol{\eta}$ | `[1, D, H_pad, W_pad, C, 2]` | `float32` | $\mathbf{0}$ | Multiplier for $\nabla\mathbf{x} = \mathbf{U}$ |
| Non-negativity Dual | $\boldsymbol{\rho}$ | `[1, D, H_pad, W_pad, C]` | `float32` | $\mathbf{0}$ | Multiplier for $\mathbf{x} = \mathbf{W}$ |

### 2.4 Hyperparameters/Constants

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Iterations | $N_{\text{iter}}$ | 50 | Maximum ADMM iterations |
| Data Penalty | $\mu_1$ | $10^{-6}$ | Penalty weight for data fidelity constraint |
| TV Penalty | $\mu_2$ | $10^{-5}$ | Penalty weight for TV constraint |
| Non-neg Penalty | $\mu_3$ | $4 \times 10^{-5}$ | Penalty weight for non-negativity constraint |
| TV Weight | $\tau$ | $10^{-4}$ | TV regularization strength |
| Downsample Factor | $d$ | 4 | Spatial downsampling factor |
| FFT Normalization | - | `"ortho"` | Orthonormal FFT scaling |

### 2.5 Observed Data/Inputs

| Data | Symbol | Shape | Dtype | Description |
|------|--------|-------|-------|-------------|
| Measurement | $\mathbf{y}$ | `[D, H, W, C]` | `float32` | Raw lensless camera capture |
| PSF | $\mathbf{h}$ | `[D, H, W, C]` | `float32` | Point spread function |

---

## 3. Preprocessing Pipeline

### Step 1: Data Loading
- **Input:** File paths `psf_path`, `data_path`, downsample factor $d$
- **Operation:** Load images via `load_data()` with normalization
- **Output:** $\mathbf{h} \in \mathbb{R}^{D \times H \times W \times C}$, $\mathbf{y} \in \mathbb{R}^{D \times H \times W \times C}$
- **Parameters:** `dtype="float32"`, `normalize=True`, `bayer=False`

### Step 2: Padded Shape Computation
- **Input:** PSF shape $(D, H, W, C)$
- **Operation:**
  $$H_{\text{pad}} = \texttt{next\_fast\_len}(2H - 1), \quad W_{\text{pad}} = \texttt{next\_fast\_len}(2W - 1)$$
- **Output:** Padded shape $(D, H_{\text{pad}}, W_{\text{pad}}, C)$

### Step 3: Padding Index Computation
- **Input:** Original shape $(H, W)$, padded shape $(H_{\text{pad}}, W_{\text{pad}})$
- **Operation:**
  $$s_h = \lfloor(H_{\text{pad}} - H)/2\rfloor, \quad s_w = \lfloor(W_{\text{pad}} - W)/2\rfloor$$
  $$e_h = s_h + H, \quad e_w = s_w + W$$
- **Output:** Start indices $(s_h, s_w)$, end indices $(e_h, e_w)$

### Step 4: PSF Frequency Response Precomputation
- **Input:** Padded PSF $\mathbf{h}_{\text{pad}}$
- **Operation:**
  $$\hat{\mathbf{H}} = \mathcal{F}_{\text{rfft2}}\{\mathbf{h}_{\text{pad}}\}, \quad \hat{\mathbf{H}}^* = \text{conj}(\hat{\mathbf{H}})$$
- **Output:** $\hat{\mathbf{H}}, \hat{\mathbf{H}}^*$ (frequency domain PSF and its conjugate)

### Step 5: Finite Difference Gram Matrix Precomputation
- **Input:** Padded shape, dtype
- **Operation:** Construct spatial kernel $\mathbf{g}$ where:
  - For 2D: $\mathbf{g}[0,0]=4$, $\mathbf{g}[0,1]=\mathbf{g}[0,-1]=\mathbf{g}[1,0]=\mathbf{g}[-1,0]=-1$
  - For 3D: $\mathbf{g}[0,0,0]=6$, neighbors $=-1$
  $$\hat{\mathbf{\Psi}}^T\hat{\mathbf{\Psi}} = \mathcal{F}_{\text{rfft2}}\{\mathbf{g}\}$$
- **Output:** $\hat{\mathbf{\Psi}}^T\hat{\mathbf{\Psi}}$ (frequency domain TV Gram matrix)

### Step 6: Division Matrix Precomputation
- **Input:** $\hat{\mathbf{H}}$, $\hat{\mathbf{H}}^*$, $\hat{\mathbf{\Psi}}^T\hat{\mathbf{\Psi}}$, $\mu_1, \mu_2, \mu_3$
- **Operation:**
  $$\mathbf{R}_{\text{div}} = \frac{1}{\mu_1|\hat{\mathbf{H}}^*\hat{\mathbf{H}}| + \mu_2|\hat{\mathbf{\Psi}}^T\hat{\mathbf{\Psi}}| + \mu_3}$$
  $$\mathbf{X}_{\text{div}} = \frac{1}{\mathbf{M} + \mu_1}$$
  where $\mathbf{M}$ is a mask of ones in the valid region
- **Output:** $\mathbf{R}_{\text{div}}$, $\mathbf{X}_{\text{div}}$

### Step 7: Measurement Padding
- **Input:** Measurement $\mathbf{y}$, padding indices
- **Operation:** Zero-pad $\mathbf{y}$ to padded shape with centered placement
- **Output:** $\mathbf{y}_{\text{pad}} \in \mathbb{R}^{D \times H_{\text{pad}} \times W_{\text{pad}} \times C}$

---

## 4. Algorithm Steps (ADMM Iterations)

### Initialization
Set iteration counter $k = 0$ and initialize all variables to zero:
$$\mathbf{x}^{(0)} = \mathbf{0}, \quad \mathbf{X}^{(0)} = \mathbf{0}, \quad \mathbf{U}^{(0)} = \mathbf{0}, \quad \mathbf{W}^{(0)} = \mathbf{0}$$
$$\boldsymbol{\xi}^{(0)} = \mathbf{0}, \quad \boldsymbol{\eta}^{(0)} = \mathbf{0}, \quad \boldsymbol{\rho}^{(0)} = \mathbf{0}$$

---

### Per-Iteration Updates

#### Step 1: TV Auxiliary Update (Soft Thresholding)
- **Name:** U-update (Anisotropic TV Proximal)
- **Input:** $\mathbf{x}^{(k)}$, $\boldsymbol{\eta}^{(k)}$, $\mu_2$, $\tau$
- **Formula:**
  $$\mathbf{v} = \nabla\mathbf{x}^{(k)} + \frac{\boldsymbol{\eta}^{(k)}}{\mu_2}$$
  $$\mathbf{U}^{(k+1)} = \text{sign}(\mathbf{v}) \odot \max\left(|\mathbf{v}| - \frac{\tau}{\mu_2}, 0\right)$$
- **Output:** $\mathbf{U}^{(k+1)}$

**Finite Difference Operator Definition:**
$$(\nabla\mathbf{x})_{i,j,\cdot} = \begin{pmatrix} x_{i+1,j} - x_{i,j} \\ x_{i,j+1} - x_{i,j} \end{pmatrix}$$

---

#### Step 2: Data Fidelity Auxiliary Update
- **Name:** X-update (Quadratic Proximal)
- **Input:** $\mathbf{x}^{(k)}$, $\boldsymbol{\xi}^{(k)}$, $\mathbf{y}_{\text{pad}}$, $\mu_1$
- **Formula:**
  $$\mathbf{X}^{(k+1)} = \mathbf{X}_{\text{div}} \odot \left(\boldsymbol{\xi}^{(k)} + \mu_1 (\mathbf{H}\mathbf{x}^{(k)}) + \mathbf{y}_{\text{pad}}\right)$$
  
  where convolution is computed as:
  $$\mathbf{H}\mathbf{x} = \mathcal{F}^{-1}_{\text{irfft2}}\left\{\text{ifftshift}\left(\mathcal{F}_{\text{rfft2}}\{\mathbf{x}\} \odot \hat{\mathbf{H}}\right)\right\}$$
- **Output:** $\mathbf{X}^{(k+1)}$

---

#### Step 3: Non-negativity Auxiliary Update (Projection)
- **Name:** W-update (ReLU Projection)
- **Input:** $\mathbf{x}^{(k)}$, $\boldsymbol{\rho}^{(k)}$, $\mu_3$
- **Formula:**
  $$\mathbf{W}^{(k+1)} = \max\left(\mathbf{x}^{(k)} + \frac{\boldsymbol{\rho}^{(k)}}{\mu_3}, 0\right)$$
- **Output:** $\mathbf{W}^{(k+1)}$

---

#### Step 4: Primal Variable Update (Frequency Domain Solve)
- **Name:** x-update (Linear System Solve)
- **Input:** $\mathbf{X}^{(k+1)}$, $\mathbf{U}^{(k+1)}$, $\mathbf{W}^{(k+1)}$, $\boldsymbol{\xi}^{(k)}$, $\boldsymbol{\eta}^{(k)}$, $\boldsymbol{\rho}^{(k)}$
- **Formula:**

  Compute right-hand side:
  $$\mathbf{r}^{(k)} = (\mu_3\mathbf{W}^{(k+1)} - \boldsymbol{\rho}^{(k)}) + \nabla^T(\mu_2\mathbf{U}^{(k+1)} - \boldsymbol{\eta}^{(k)}) + \mathbf{H}^T(\mu_1\mathbf{X}^{(k+1)} - \boldsymbol{\xi}^{(k)})$$

  Solve in frequency domain:
  $$\mathbf{x}^{(k+1)} = \mathcal{F}^{-1}_{\text{irfft2}}\left\{\mathbf{R}_{\text{div}} \odot \mathcal{F}_{\text{rfft2}}\{\mathbf{r}^{(k)}\}\right\}$$

- **Output:** $\mathbf{x}^{(k+1)}$

**Adjoint Operators:**
$$\mathbf{H}^T\mathbf{z} = \mathcal{F}^{-1}_{\text{irfft2}}\left\{\text{ifftshift}\left(\mathcal{F}_{\text{rfft2}}\{\mathbf{z}\} \odot \hat{\mathbf{H}}^*\right)\right\}$$

$$(\nabla^T\mathbf{u})_{i,j} = (u^{(1)}_{i-1,j} - u^{(1)}_{i,j}) + (u^{(2)}_{i,j-1} - u^{(2)}_{i,j})$$

---

#### Step 5: Dual Variable Updates (Gradient Ascent)
- **Name:** Lagrange Multiplier Updates
- **Input:** $\mathbf{x}^{(k+1)}$, $\mathbf{X}^{(k+1)}$, $\mathbf{U}^{(k+1)}$, $\mathbf{W}^{(k+1)}$, $\boldsymbol{\xi}^{(k)}$, $\boldsymbol{\eta}^{(k)}$, $\boldsymbol{\rho}^{(k)}$
- **Formulas:**
  $$\boldsymbol{\xi}^{(k+1)} = \boldsymbol{\xi}^{(k)} + \mu_1\left(\mathbf{H}\mathbf{x}^{(k+1)} - \mathbf{X}^{(k+1)}\right)$$
  $$\boldsymbol{\eta}^{(k+1)} = \boldsymbol{\eta}^{(k)} + \mu_2\left(\nabla\mathbf{x}^{(k+1)} - \mathbf{U}^{(k+1)}\right)$$
  $$\boldsymbol{\rho}^{(k+1)} = \boldsymbol{\rho}^{(k)} + \mu_3\left(\mathbf{x}^{(k+1)} - \mathbf{W}^{(k+1)}\right)$$
- **Output:** $\boldsymbol{\xi}^{(k+1)}$, $\boldsymbol{\eta}^{(k+1)}$, $\boldsymbol{\rho}^{(k+1)}$

---

#### Step 6: Post-Processing (After Final Iteration)
- **Name:** Crop and Reshape
- **Input:** $\mathbf{x}^{(N_{\text{iter}})}$
- **Formula:**
  $$\mathbf{x}_{\text{out}} = \mathbf{x}^{(N_{\text{iter}})}[\cdot, s_h:e_h, s_w:e_w, \cdot]$$
  Remove batch dimension if shape is `[1, D, H, W, C]`
- **Output:** Final reconstruction $\mathbf{x}_{\text{out}} \in \mathbb{R}^{D \times H \times W \times C}$

---

## 5. Stopping Criterion

| Criterion Type | Condition | Implementation |
|----------------|-----------|----------------|
| **Maximum Iterations** | $k \geq N_{\text{iter}}$ | Primary termination (default $N_{\text{iter}}=50$) |

**Note:** The implementation uses a fixed iteration count without explicit convergence checking based on primal/dual residuals.

---

## 6. Evaluation Metrics

### 6.1 Reconstruction Statistics

**Mean:**
$$\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$$

**Standard Deviation:**
$$\sigma_x = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2}$$

**Range:**
$$x_{\min} = \min_i(x_i), \quad x_{\max} = \max_i(x_i)$$

### 6.2 Forward Model Consistency (Implicit)

The forward operator can be applied to verify reconstruction quality:
$$\hat{\mathbf{y}} = \mathbf{H} \circledast \mathbf{x}_{\text{out}}$$

**Residual Norm** (not explicitly computed but implied):
$$\|\hat{\mathbf{y}} - \mathbf{y}\|_2$$

---

## 7. Summary of Complete ADMM Algorithm

```
Algorithm: ADMM for TV-Regularized Lensless Image Reconstruction
─────────────────────────────────────────────────────────────────
Input: Measurement y, PSF h, parameters (μ₁, μ₂, μ₃, τ, N_iter)
Output: Reconstructed image x

1. PRECOMPUTE:
   Ĥ ← rfft2(pad(h))
   Ĥ* ← conj(Ĥ)
   Ψ̂ᵀΨ̂ ← rfft2(finite_diff_gram_kernel)
   R_div ← 1 / (μ₁|Ĥ*Ĥ| + μ₂|Ψ̂ᵀΨ̂| + μ₃)
   X_div ← 1 / (M + μ₁)
   y_pad ← pad(y)

2. INITIALIZE: x, X, U, W, ξ, η, ρ ← 0

3. FOR k = 0 TO N_iter - 1:
   
   (a) U ← shrink(∇x + η/μ₂, τ/μ₂)
   
   (b) X ← X_div ⊙ (ξ + μ₁·conv(x,h) + y_pad)
   
   (c) W ← max(x + ρ/μ₃, 0)
   
   (d) r ← (μ₃W - ρ) + ∇ᵀ(μ₂U - η) + convᵀ(μ₁X - ξ, h)
       x ← irfft2(R_div ⊙ rfft2(r))
   
   (e) ξ ← ξ + μ₁(conv(x,h) - X)
       η ← η + μ₂(∇x - U)
       ρ ← ρ + μ₃(x - W)

4. RETURN crop(x)
```