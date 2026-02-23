# Algorithmic Specification Document: Sparse Hessian Deconvolution for Image Reconstruction

---

## 1. Mathematical Formulation

### 1.1 Overall Problem Statement

The algorithm solves a two-stage inverse problem for image reconstruction:

**Stage 1 (Sparse Hessian Reconstruction):** Given observed data $f$, recover latent image $g$ by solving:

$$\min_{g} \underbrace{\frac{\lambda_f}{2\mu}\|g - f\|_2^2}_{\text{Data Fidelity}} + \underbrace{\lambda_s \|g\|_1}_{\text{Sparsity}} + \underbrace{\mathcal{R}_H(g)}_{\text{Hessian Regularization}}$$

where the Hessian regularization term is:

$$\mathcal{R}_H(g) = \|D_{xx}g\|_1 + \|D_{yy}g\|_1 + \alpha^2\|D_{zz}g\|_1 + 2\|D_{xy}g\|_1 + 2\alpha\|D_{xz}g\|_1 + 2\alpha\|D_{yz}g\|_1$$

with $\alpha = \sqrt{t_{\text{continuity}}}$ controlling temporal/z-axis regularization strength.

**Stage 2 (Iterative Deconvolution):** Given sparse reconstruction $g^*$ and PSF kernel $h$, solve:

$$\min_{u} \mathcal{D}(Hu, g^*)$$

where $H$ is the convolution operator with kernel $h$, and $\mathcal{D}$ is a divergence measure (KL for Richardson-Lucy, $\ell_2$ for Landweber).

### 1.2 Detailed Formulations

#### 1.2.1 Finite Difference Operators

Forward difference along dimension $d$ with step $\Delta$:
$$(\nabla_d^+ g)_{i,j,k} = \frac{g_{i+\delta_d^0, j+\delta_d^1, k+\delta_d^2} - g_{i,j,k}}{\Delta}$$

Backward difference along dimension $d$:
$$(\nabla_d^- g)_{i,j,k} = \frac{g_{i,j,k} - g_{i-\delta_d^0, j-\delta_d^1, k-\delta_d^2}}{\Delta}$$

where $\delta_d^m = 1$ if $m=d$, else $0$.

#### 1.2.2 Second-Order Derivative Operators

| Operator | Definition | Fourier Filter Kernel |
|----------|------------|----------------------|
| $D_{xx}$ | $\nabla_x^- \nabla_x^+ g$ | $[1, -2, 1]$ |
| $D_{yy}$ | $\nabla_y^- \nabla_y^+ g$ | $[1, -2, 1]^T$ |
| $D_{zz}$ | $\nabla_z^- \nabla_z^+ g$ | $[1, -2, 1]$ (along z) |
| $D_{xy}$ | $\nabla_x^+ \nabla_y^+ g$ | $\begin{bmatrix}1 & -1\\-1 & 1\end{bmatrix}$ |
| $D_{xz}$ | $\nabla_x^+ \nabla_z^+ g$ | $\begin{bmatrix}1 & -1\\-1 & 1\end{bmatrix}$ (x-z plane) |
| $D_{yz}$ | $\nabla_y^+ \nabla_z^+ g$ | $\begin{bmatrix}1 & -1\\-1 & 1\end{bmatrix}$ (y-z plane) |

#### 1.2.3 PSF/OTF Model

Gaussian PSF with full-width half-maximum $\sigma$:
$$h(x,y) = \frac{1}{Z}\exp\left(-\frac{x^2}{2s_x^2} - \frac{y^2}{2s_y^2}\right)$$

where $s = \frac{\sigma}{2\sqrt{2\ln 2}}$ and $Z$ is the normalization constant.

Optical Transfer Function (OTF):
$$\hat{h} = \mathcal{F}\{\text{circshift}(\text{pad}(h, N), -\lfloor N_h/2 \rfloor)\}$$

#### 1.2.4 Forward Imaging Model

$$y_{\text{pred}} = \mathcal{F}^{-1}\{\hat{h} \cdot \mathcal{F}\{x\}\}$$

---

## 2. Variables Taxonomy

### 2.1 Observed Data / Inputs

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `img` | $f_{\text{raw}}$ | $(T, H, W)$ or $(H, W)$ | float32 | Raw input image stack |
| `sigma` | $\sigma$ | scalar or $(2,)$ | float32 | PSF FWHM parameter |
| `input_path` | - | string | - | Path to input file |

### 2.2 Hyperparameters / Constants

| Variable | Symbol | Default | Description |
|----------|--------|---------|-------------|
| `fidelity` | $\lambda_f$ | 150 | Data fidelity weight |
| `sparsity` | $\lambda_s$ | 10 | $\ell_1$ sparsity weight |
| `tcontinuity` | $t_c$ | 0.5 | Z-axis continuity ($\alpha = \sqrt{t_c}$) |
| `mu` | $\mu$ | 1 | ADMM penalty parameter |
| `sparse_iter` | $K_{\text{sparse}}$ | 1000 | Max sparse Hessian iterations |
| `deconv_iter` | $K_{\text{deconv}}$ | 7 | Max deconvolution iterations |
| `deconv_type` | - | 1 | 0: None, 1: RL, 2: Landweber |
| `tol` | $\epsilon$ | $10^{-4}$ | Convergence tolerance |
| `dlevel` | $L_w$ | 7 | Wavelet decomposition level |
| `wavename` | - | 'db6' | Wavelet basis |

### 2.3 Primal / Optimization Variables

| Variable | Symbol | Shape | Initialization | Description |
|----------|--------|-------|----------------|-------------|
| `g` | $g^{(k)}$ | $(T, H, W)$ | $\frac{f}{\lambda_f/\mu}$ | Sparse reconstruction (primal) |
| `yk` | $y^{(k)}$ | $(H', W')$ | $f_{\text{padded}}$ | Deconvolution iterate |
| `xk` | $x^{(k)}$ | $(H', W')$ | $\mathbf{0}$ | Deconvolution auxiliary |

### 2.4 Auxiliary / Dual Variables (Bregman Splitting)

| Variable | Symbol | Shape | Initialization | Description |
|----------|--------|-------|----------------|-------------|
| `bxx` | $b_{xx}^{(k)}$ | $(T, H, W)$ | $\mathbf{0}$ | Bregman variable for $D_{xx}$ |
| `byy` | $b_{yy}^{(k)}$ | $(T, H, W)$ | $\mathbf{0}$ | Bregman variable for $D_{yy}$ |
| `bzz` | $b_{zz}^{(k)}$ | $(T, H, W)$ | $\mathbf{0}$ | Bregman variable for $D_{zz}$ |
| `bxy` | $b_{xy}^{(k)}$ | $(T, H, W)$ | $\mathbf{0}$ | Bregman variable for $D_{xy}$ |
| `bxz` | $b_{xz}^{(k)}$ | $(T, H, W)$ | $\mathbf{0}$ | Bregman variable for $D_{xz}$ |
| `byz` | $b_{yz}^{(k)}$ | $(T, H, W)$ | $\mathbf{0}$ | Bregman variable for $D_{yz}$ |
| `bl1` | $b_s^{(k)}$ | $(T, H, W)$ | $\mathbf{0}$ | Bregman variable for sparsity |

### 2.5 Intermediate/Derived Variables

| Variable | Symbol | Shape | Description |
|----------|--------|-------|-------------|
| `normlize` | $\mathcal{N}$ | $(T, H, W)$ | Fourier-domain normalizer |
| `otf` | $\hat{h}$ | $(H, W)$ | Optical transfer function |
| `kernel` | $h$ | $(N_h, N_h)$ | Gaussian PSF kernel |
| `scaler` | $s$ | scalar | Original max value for rescaling |
| `backgrounds` | $B$ | $(T, H, W)$ | Estimated background |

---

## 3. Preprocessing Pipeline

### Step 1: Image Loading
- **Input:** `input_path` (string)
- **Output:** $f_{\text{raw}} \in \mathbb{R}^{T \times H \times W}$ (float32)
- **Operation:** `io.imread(input_path).astype('float32')`

### Step 2: Intensity Normalization
- **Input:** $f_{\text{raw}}$
- **Output:** $f_{\text{norm}}$, $s$
$$s = \max(f_{\text{raw}}), \quad f_{\text{norm}} = \frac{f_{\text{raw}}}{s}$$

### Step 3: Background Estimation (Wavelet-based)
- **Input:** $f_{\text{norm}}$
- **Output:** $B \in \mathbb{R}^{T \times H \times W}$

For each frame $t$ and iteration $i \in \{1, \ldots, 3\}$:
1. Compute wavelet decomposition: $\{c_A^{(L)}, (c_H^{(\ell)}, c_V^{(\ell)}, c_D^{(\ell)})\}_{\ell=1}^{L}$ = `wavedec2(res, 'db6', level=7)`
2. Zero out detail coefficients: $\tilde{c}_H^{(\ell)} = \tilde{c}_V^{(\ell)} = \tilde{c}_D^{(\ell)} = \mathbf{0}$
3. Reconstruct low-frequency: $B_{\text{iter}} = $ `waverec2({cA, (0,0,0), ...})`
4. Threshold update:
$$\epsilon = \frac{\sqrt{|f_{\text{norm}}|}}{2}, \quad \text{res}[f_{\text{norm}} > B_{\text{iter}} + \epsilon] = B_{\text{iter}} + \epsilon$$

### Step 4: Background Subtraction
- **Input:** $f_{\text{norm}}$, $B$
- **Output:** $f_{\text{sub}}$
$$f_{\text{sub}} = \max\left(\frac{f_{\text{norm}} - B}{\max(f_{\text{norm}} - B)}, 0\right)$$

### Step 5: Optional Upsampling
- **Input:** $f_{\text{sub}}$
- **Output:** $f \in \mathbb{R}^{T \times H' \times W'}$

| Mode | Operation | Output Size |
|------|-----------|-------------|
| 0 | None | $(T, H, W)$ |
| 1 | Fourier interpolation | $(T, 2H, 2W)$ |
| 2 | Spatial zero-insertion | $(T, 2H, 2W)$ |

### Step 6: Dimension Padding (for 2D inputs)
- **Input:** $f$ with $\text{ndim} = 2$
- **Output:** $f \in \mathbb{R}^{3 \times H \times W}$ (replicate along z-axis)

---

## 4. Algorithm Steps

### 4.1 Stage 1: Sparse Hessian Reconstruction (Split Bregman / ADMM)

#### Initialization

**Step 0.1:** Compute Fourier-domain Hessian operator magnitudes
$$|\hat{D}_{ij}|^2 = \mathcal{F}\{\delta_{ij}\} \cdot \overline{\mathcal{F}\{\delta_{ij}\}}$$

where $\delta_{ij}$ are the finite difference kernels.

**Step 0.2:** Compute normalizer
$$\mathcal{N} = \frac{\lambda_f}{\mu} + \lambda_s^2 + |\hat{D}_{xx}|^2 + |\hat{D}_{yy}|^2 + \alpha^2|\hat{D}_{zz}|^2 + 2|\hat{D}_{xy}|^2 + 2\alpha|\hat{D}_{xz}|^2 + 2\alpha|\hat{D}_{yz}|^2$$

**Step 0.3:** Initialize variables
$$g^{(0)} = f, \quad b_{ij}^{(0)} = \mathbf{0}, \quad b_s^{(0)} = \mathbf{0}, \quad \hat{g}_{\text{update}}^{(0)} = \frac{\lambda_f}{\mu}\mathcal{F}\{f\}$$

#### Per-Iteration Updates (for $k = 0, 1, \ldots, K_{\text{sparse}}-1$)

**Step 1: Primal Update (Fourier Domain Solve)**
- **Input:** $\hat{g}_{\text{update}}^{(k)}$, $\mathcal{N}$
- **Output:** $g^{(k)}$

$$g^{(k)} = \mathcal{F}^{-1}\left\{\frac{\hat{g}_{\text{update}}^{(k)}}{\mathcal{N}}\right\}$$

(For $k=0$: $g^{(0)} = \mathcal{F}^{-1}\left\{\frac{\hat{g}_{\text{update}}^{(0)}}{\lambda_f/\mu}\right\}$)

**Step 2: Reset Accumulator**
$$\hat{g}_{\text{update}}^{(k+1)} \leftarrow \frac{\lambda_f}{\mu}\mathcal{F}\{f\}$$

**Step 3: Hessian Subproblem Updates** (for each operator $D_{ij} \in \{D_{xx}, D_{yy}, D_{zz}, D_{xy}, D_{xz}, D_{yz}\}$)

- **Input:** $g^{(k)}$, $b_{ij}^{(k)}$
- **Output:** $b_{ij}^{(k+1)}$, contribution $L_{ij}$

Let $w_{ij}$ be the weight ($1$ for $xx,yy$; $\alpha^2$ for $zz$; $2$ for $xy$; $2\alpha$ for $xz, yz$):

$$\begin{aligned}
v_{ij} &= D_{ij} g^{(k)} \\
d_{ij} &= \text{shrink}(v_{ij} + b_{ij}^{(k)}, \mu) = \text{sign}(v_{ij} + b_{ij}^{(k)}) \cdot \max(|v_{ij} + b_{ij}^{(k)}| - \mu^{-1}, 0) \\
b_{ij}^{(k+1)} &= b_{ij}^{(k)} + v_{ij} - d_{ij} \\
L_{ij} &= w_{ij} \cdot D_{ij}^T (d_{ij} - b_{ij}^{(k+1)})
\end{aligned}$$

**Step 4: Sparsity Subproblem Update**
- **Input:** $g^{(k)}$, $b_s^{(k)}$
- **Output:** $b_s^{(k+1)}$, contribution $L_s$

$$\begin{aligned}
d_s &= \text{shrink}(g^{(k)} + b_s^{(k)}, \mu) \\
b_s^{(k+1)} &= b_s^{(k)} + g^{(k)} - d_s \\
L_s &= \lambda_s (d_s - b_s^{(k+1)})
\end{aligned}$$

**Step 5: Accumulate Updates**
$$\hat{g}_{\text{update}}^{(k+1)} \leftarrow \hat{g}_{\text{update}}^{(k+1)} + \mathcal{F}\{L_{xx} + L_{yy} + L_{zz} + L_{xy} + L_{xz} + L_{yz} + L_s\}$$

**Step 6: Non-negativity Projection**
$$g^{(k)} \leftarrow \max(g^{(k)}, 0)$$

---

### 4.2 Stage 2: Iterative Deconvolution

#### 4.2.1 Richardson-Lucy Deconvolution

**Initialization:**
$$y^{(0)} = f_{\text{padded}}, \quad x^{(0)} = \mathbf{0}, \quad v^{(0)} = \mathbf{0}$$

**Per-Iteration Updates (for $k = 0, 1, \ldots, K_{\text{deconv}}-1$):**

**Step 1: RL Core Update**
- **Input:** $y^{(k)}$, $\hat{h}$, $f$
- **Output:** $x^{(k+1)}$

$$r^{(k)} = \mathcal{F}\left\{\frac{f}{\max(\mathcal{F}^{-1}\{\hat{h} \cdot \mathcal{F}\{y^{(k)}\}\}, 10^{-6})}\right\}$$

$$x^{(k+1)} = \max\left(y^{(k)} \cdot \frac{\mathcal{F}^{-1}\{\bar{\hat{h}} \cdot r^{(k)}\}}{\mathcal{F}^{-1}\{\hat{h} \cdot \mathcal{F}\{\mathbf{1}\}\}}, 10^{-6}\right)$$

**Step 2: Momentum Update**
- **Input:** $x^{(k+1)}$, $x^{(k)}$, $v^{(k)}$
- **Output:** $y^{(k+1)}$, $v^{(k+1)}$

$$v^{(k+1)} = \max(x^{(k+1)} - y^{(k)}, 10^{-6})$$

$$\alpha^{(k+1)} = \text{clip}\left(\frac{\langle v^{(k)}, v^{(k+1)} \rangle}{\|v^{(k)}\|^2 + 10^{-10}}, 10^{-6}, 1\right)$$

$$y^{(k+1)} = \max(x^{(k+1)} + \alpha^{(k+1)}(x^{(k+1)} - x^{(k)}), 10^{-6})$$

#### 4.2.2 Landweber Deconvolution

**Initialization:**
$$x^{(0)} = f, \quad \gamma_1 = 1, \quad t = 1$$

**Per-Iteration Updates (for $k = 1, \ldots, K_{\text{deconv}}-1$):**

**Step 1: Momentum Coefficient**
$$\gamma_2 = \frac{1}{2}\sqrt{4\gamma_1^2 + \gamma_1^4} - \gamma_1^2, \quad \beta = -\gamma_2(1 - \gamma_1^{-1})$$

**Step 2: Extrapolation**
$$\tilde{y}^{(k)} = x^{(k)} + \beta(x^{(k)} - x^{(k-1)})$$

**Step 3: Gradient Step**
$$y^{(k+1)} = \max\left(\tilde{y}^{(k)} + t \cdot \mathcal{F}^{-1}\{\bar{\hat{h}} \cdot (\mathcal{F}\{f\} - \hat{h} \cdot \mathcal{F}\{\tilde{y}^{(k)}\})\}, 10^{-6}\right)$$

**Step 4: Update**
$$x^{(k+1)} = y^{(k+1)}, \quad \gamma_1 \leftarrow \gamma_2$$

---

## 5. Stopping Criterion

### 5.1 Sparse Hessian Stage

| Criterion | Formula | Threshold |
|-----------|---------|-----------|
| Maximum iterations | $k \geq K_{\text{sparse}}$ | $K_{\text{sparse}} = 1000$ |
| Relative residual change | $\frac{|r^{(k)} - r^{(k-20)}|}{r^{(k-20)} + 10^{-12}} < \epsilon$ | $\epsilon = 10^{-4}$ |

where $r^{(k)} = \|f - g^{(k)}\|_2$ (checked every 20 iterations).

### 5.2 Deconvolution Stage

| Criterion | Formula | Threshold |
|-----------|---------|-----------|
| Maximum iterations | $k \geq K_{\text{deconv}}$ | $K_{\text{deconv}} = 7$ |

---

## 6. Evaluation Metrics

### 6.1 Peak Signal-to-Noise Ratio (PSNR)

$$\text{PSNR}(u, v) = 10 \cdot \log_{10}\left(\frac{R^2}{\text{MSE}(u, v)}\right) \text{ dB}$$

where $R = \max(v) - \min(v)$ is the dynamic range of the reference image $v$.

### 6.2 Structural Similarity Index (SSIM)

$$\text{SSIM}(u, v) = \frac{(2\mu_u\mu_v + c_1)(2\sigma_{uv} + c_2)}{(\mu_u^2 + \mu_v^2 + c_1)(\sigma_u^2 + \sigma_v^2 + c_2)}$$

where:
- $\mu_u, \mu_v$: local means
- $\sigma_u^2, \sigma_v^2$: local variances
- $\sigma_{uv}$: local covariance
- $c_1 = (k_1 R)^2$, $c_2 = (k_2 R)^2$ with $k_1 = 0.01$, $k_2 = 0.03$

### 6.3 Mean Squared Error (MSE)

$$\text{MSE}(u, v) = \frac{1}{N}\sum_{i=1}^{N}(u_i - v_i)^2$$

---

## 7. Algorithm Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Raw Image f_raw                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING                                                       │
│  1. Normalize: f = f_raw / max(f_raw)                               │
│  2. Background estimation (wavelet): B = wavelet_lowpass(f)          │
│  3. Subtract & clip: f ← max((f - B)/max(f - B), 0)                 │
│  4. Optional upsampling (Fourier/Spatial)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: SPARSE HESSIAN (Split Bregman)                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  FOR k = 0 to K_sparse:                                        │ │
│  │    g ← F⁻¹{ĝ_update / N}           [Primal update]            │ │
│  │    FOR each D_ij ∈ {Dxx, Dyy, Dzz, Dxy, Dxz, Dyz}:            │ │
│  │      d_ij ← shrink(D_ij·g + b_ij, μ)                          │ │
│  │      b_ij ← b_ij + D_ij·g - d_ij   [Bregman update]           │ │
│  │    d_s ← shrink(g + b_s, μ)                                    │ │
│  │    b_s ← b_s + g - d_s                                         │ │
│  │    Accumulate ĝ_update                                         │ │
│  │    g ← max(g, 0)                   [Non-negativity]            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  OUTPUT: g* (sparse reconstruction)                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: ITERATIVE DECONVOLUTION                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Richardson-Lucy (rule=1):                                     │ │
│  │    x ← y · F⁻¹{ĥ* · F{f / max(F⁻¹{ĥ·F{y}}, ε)}} / F⁻¹{ĥ·F{1}}│ │
│  │    α ← <v_old, v_new> / ||v_old||²                             │ │
│  │    y ← x + α(x - x_old)                                        │ │
│  │  ──────────────────────────────────────────────────────────────│ │
│  │  Landweber (rule=2):                                           │ │
│  │    ỹ ← x + β(x - x_old)                                        │ │
│  │    y ← ỹ + t·F⁻¹{ĥ*·(F{f} - ĥ·F{ỹ})}                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  OUTPUT: u* (final reconstruction)                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POST-PROCESSING & EVALUATION                                        │
│  1. Rescale: output = scaler × u*                                   │
│  2. Compute PSNR, SSIM, MSE against reference                        │
└─────────────────────────────────────────────────────────────────────┘
```