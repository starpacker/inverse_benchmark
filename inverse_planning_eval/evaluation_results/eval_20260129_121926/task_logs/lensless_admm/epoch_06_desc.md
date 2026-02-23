## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "ADMM-TV Lensless Image Reconstruction",
  "algorithm_family": "Alternating Direction Method of Multipliers (ADMM)",
  "summary": "Reconstructs an image from lensless camera measurements by solving a convex optimization problem with data fidelity, total variation regularization, and non-negativity constraints using variable splitting in the frequency domain."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{x \geq 0} \frac{1}{2}\|Hx - y\|_2^2 + \tau \|\nabla x\|_1$$

Subject to: $x \geq 0$ (element-wise non-negativity)

**data_fidelity_term**:
$$\mathcal{D}(x) = \frac{1}{2}\|Hx - y\|_2^2$$
where $H$ is the convolution operator with the PSF, $y$ is the measured data (padded).

**regularization_term**:
$$\mathcal{R}(x) = \tau \|\nabla x\|_1 = \tau \sum_{i,j,c} \sqrt{|(\nabla_1 x)_{i,j,c}|^2 + |(\nabla_2 x)_{i,j,c}|^2}$$
where $\nabla_1$ is the finite difference along axis -3 (height) and $\nabla_2$ is along axis -2 (width).

**Note**: The code implements anisotropic TV (separate L1 norms), so:
$$\mathcal{R}(x) = \tau \|\nabla x\|_1 = \tau \left( \|\nabla_1 x\|_1 + \|\nabla_2 x\|_1 \right)$$

**constraints**:
$$\mathcal{I}_{\geq 0}(x) = \begin{cases} 0 & \text{if } x_{i,j,c} \geq 0 \; \forall i,j,c \\ +\infty & \text{otherwise} \end{cases}$$

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x$ (`image_est`) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Estimated image in padded domain |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\xi$ (`xi`) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Lagrange multiplier for data fidelity constraint $Hx = X$ |
| $\eta$ (`eta`) | $[1, D, H_p, W_p, C, 2]$ | float32 | $\mathbf{0}$ | Lagrange multiplier for TV constraint $\nabla x = U$ |
| $\rho$ (`rho`) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Lagrange multiplier for non-negativity constraint $x = W$ |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $X$ | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Auxiliary variable for data fidelity splitting |
| $U$ | $[1, D, H_p, W_p, C, 2]$ | float32 | $\mathbf{0}$ | Auxiliary variable for TV splitting |
| $W$ | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Auxiliary variable for non-negativity splitting |
| $\hat{H}$ (`H`) | $[D, H_p, W_p//2+1, C]$ | complex64 | Precomputed | FFT of padded PSF |
| $\hat{H}^*$ (`Hadj`) | $[D, H_p, W_p//2+1, C]$ | complex64 | Precomputed | Complex conjugate of $\hat{H}$ |
| $\hat{\Psi}^T\hat{\Psi}$ (`PsiTPsi`) | $[D, H_p, W_p//2+1, C]$ | complex64 | Precomputed | FFT of finite difference Gram matrix |
| $R_{\text{div}}$ (`R_divmat`) | $[D, H_p, W_p//2+1, C]$ | complex64 | Precomputed | Precomputed denominator for $x$-update |
| $X_{\text{div}}$ (`X_divmat`) | $[1, D, H_p, W_p, C]$ | float32 | Precomputed | Precomputed denominator for $X$-update |
| $y_{\text{pad}}$ (`data_padded`) | $[1, D, H_p, W_p, C]$ | float32 | Padded measurement | Zero-padded measurement data |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $\mu_1$ | $1 \times 10^{-6}$ | ADMM penalty for data fidelity constraint |
| $\mu_2$ | $1 \times 10^{-5}$ | ADMM penalty for TV constraint |
| $\mu_3$ | $4 \times 10^{-5}$ | ADMM penalty for non-negativity constraint |
| $\tau$ | $1 \times 10^{-4}$ | TV regularization weight |
| $N_{\text{iter}}$ | 50 | Maximum number of iterations |
| `norm` | `"ortho"` | FFT normalization mode |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y$ (`measurement`) | $[D, H, W, C]$ | float32 | Raw lensless camera measurement |
| $h$ (`psf`) | $[D, H, W, C]$ | float32 | Point spread function |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Load and normalize data | $y, h = \texttt{load\_data}(\cdot)$ with `normalize=True` | File paths | $y: [D,H,W,C]$, $h: [D,H,W,C]$ | Images are RGB, same spatial dimensions |
| P2 | Compute padded dimensions | $H_p = \texttt{next\_fast\_len}(2H - 1)$, $W_p = \texttt{next\_fast\_len}(2W - 1)$ | $h$ shape $[D,H,W,C]$ | $[D, H_p, W_p, C]$ | Padding for linear (non-circular) convolution |
| P3 | Compute padding indices | $s_0 = \lfloor(H_p - H)/2\rfloor$, $s_1 = \lfloor(W_p - W)/2\rfloor$, $e_0 = s_0 + H$, $e_1 = s_1 + W$ | Padded shape, original shape | Start/end indices | Centered padding |
| P4 | Pad PSF | $h_{\text{pad}}[d, s_0:e_0, s_1:e_1, c] = h[d,:,:,c]$, zeros elsewhere | $h: [D,H,W,C]$ | $h_{\text{pad}}: [D, H_p, W_p, C]$ | Zero-padding |
| P5 | Compute PSF frequency response | $\hat{H} = \mathcal{F}_{\text{2D}}^{\text{rfft}}\{h_{\text{pad}}\}$ with `norm="ortho"` along axes $(-3,-2)$ | $h_{\text{pad}}: [D,H_p,W_p,C]$ | $\hat{H}: [D, H_p, W_p//2+1, C]$ | Orthonormal FFT |
| P6 | Compute adjoint frequency response | $\hat{H}^* = \text{conj}(\hat{H})$ | $\hat{H}$ | $\hat{H}^*: [D, H_p, W_p//2+1, C]$ | Complex conjugate |
| P7 | Construct finite difference Gram kernel | For $D=1$: $g[0,0,0]=4$, $g[0,0,1]=g[0,0,-1]=g[0,1,0]=g[0,-1,0]=-1$; For $D>1$: $g[0,0,0]=6$, add $g[1,0,0]=g[-1,0,0]=-1$ | Padded shape | $g: [D, H_p, W_p, C]$ | Periodic boundary conditions |
| P8 | Compute TV Gram in frequency | $\hat{\Psi}^T\hat{\Psi} = \mathcal{F}_{\text{2D}}^{\text{rfft}}\{g\}$ along axes $(-3,-2)$ | $g: [D,H_p,W_p,C]$ | $\hat{\Psi}^T\hat{\Psi}: [D, H_p, W_p//2+1, C]$ | Orthonormal FFT |
| P9 | Precompute $x$-update denominator | $R_{\text{div}} = \frac{1}{\mu_1 |\hat{H}^* \odot \hat{H}| + \mu_2 |\hat{\Psi}^T\hat{\Psi}| + \mu_3}$ | $\hat{H}, \hat{\Psi}^T\hat{\Psi}, \mu_1, \mu_2, \mu_3$ | $R_{\text{div}}: [D, H_p, W_p//2+1, C]$ (complex64) | Element-wise operations |
| P10 | Precompute $X$-update denominator | $X_{\text{div}} = \frac{1}{\texttt{pad}(\mathbf{1}_{[D,H,W,C]}) + \mu_1}$ | $\mu_1$, padded shape | $X_{\text{div}}: [1, D, H_p, W_p, C]$ | Mask is 1 in valid region, 0 in padding |
| P11 | Pad measurement | $y_{\text{pad}} = \texttt{pad}(y)$ (zeros outside $[s_0:e_0, s_1:e_1]$) | $y: [D,H,W,C]$ | $y_{\text{pad}}: [1, D, H_p, W_p, C]$ | Centered zero-padding |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $x$ (`image_est`) | $\mathbf{0}$ | $[1, D, H_p, W_p, C]$ | float32 | Zero initialization |
| 2 | $X$ | $\mathbf{0}$ | $[1, D, H_p, W_p, C]$ | float32 | Zero initialization |
| 3 | $U$ | $\mathbf{0}$ | $[1, D, H_p, W_p, C, 2]$ | float32 | Zero initialization |
| 4 | $W$ | $\mathbf{0}$ | $[1, D, H_p, W_p, C]$ | float32 | Zero initialization |
| 5 | $\xi$ | $\mathbf{0}$ | $[1, D, H_p, W_p, C]$ | float32 | Zero initialization |
| 6 | $\eta$ | $\mathbf{0}$ | $[1, D, H_p, W_p, C, 2]$ | float32 | Zero initialization |
| 7 | $\rho$ | $\mathbf{0}$ | $[1, D, H_p, W_p, C]$ | float32 | Zero initialization |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{iter}}-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 8 | Compute finite differences of $x$ | $(\nabla x)^{(k)} = \texttt{stack}\left( \texttt{roll}(x^{(k)}, 1, \text{axis}=-3) - x^{(k)}, \; \texttt{roll}(x^{(k)}, 1, \text{axis}=-2) - x^{(k)} \right)$ | $x^{(k)}: [1,D,H_p,W_p,C]$ | $(\nabla x)^{(k)}: [1,D,H_p,W_p,C,2]$ | Stack along new last axis; periodic boundary via roll |
| 9 | Compute soft-threshold argument | $v_U = (\nabla x)^{(k)} + \frac{\eta^{(k)}}{\mu_2}$ | $(\nabla x)^{(k)}, \eta^{(k)}: [1,D,H_p,W_p,C,2]$; $\mu_2$ | $v_U: [1,D,H_p,W_p,C,2]$ | Element-wise |
| 10 | U-update: soft thresholding [PROX] | $U^{(k+1)} = \text{sign}(v_U) \odot \max\left(0, |v_U| - \frac{\tau}{\mu_2}\right)$ | $v_U: [1,D,H_p,W_p,C,2]$; $\tau, \mu_2$ | $U^{(k+1)}: [1,D,H_p,W_p,C,2]$ | Proximal operator: $\text{prox}_{\tau/\mu_2 \|\cdot\|_1}(v) = \text{soft}(v, \tau/\mu_2)$ |
| 11 | Forward convolution [FORWARD] | Compute $\hat{x}^{(k)} = \mathcal{F}_{\text{2D}}^{\text{rfft}}\{x^{(k)}\}$ along axes $(-3,-2)$ | $x^{(k)}: [1,D,H_p,W_p,C]$ | $\hat{x}^{(k)}: [1,D,H_p,W_p//2+1,C]$ | No normalization (default) |
| 12 | Multiply in frequency domain | $\hat{z} = \hat{x}^{(k)} \odot \hat{H}$ | $\hat{x}^{(k)}, \hat{H}$ | $\hat{z}: [1,D,H_p,W_p//2+1,C]$ | Element-wise complex multiplication |
| 13 | Inverse FFT | $z = \mathcal{F}_{\text{2D}}^{\text{irfft}}\{\hat{z}\}$ with output shape $(H_p, W_p)$ along axes $(-3,-2)$ | $\hat{z}: [1,D,H_p,W_p//2+1,C]$ | $z: [1,D,H_p,W_p,C]$ | Real-valued output |
| 14 | Apply ifftshift | $(Hx)^{(k)} = \texttt{ifftshift}(z, \text{axes}=(-3,-2))$ | $z: [1,D,H_p,W_p,C]$ | $(Hx)^{(k)}: [1,D,H_p,W_p,C]$ | Centers the convolution result |
| 15 | X-update: data fidelity | $X^{(k+1)} = X_{\text{div}} \odot \left( \xi^{(k)} + \mu_1 (Hx)^{(k)} + y_{\text{pad}} \right)$ | $X_{\text{div}}, \xi^{(k)}, (Hx)^{(k)}, y_{\text{pad}}: [1,D,H_p,W_p,C]$; $\mu_1$ | $X^{(k+1)}: [1,D,H_p,W_p,C]$ | Closed-form solution to $\min_X \|X-y_{\text{pad}}\|^2 + \frac{\mu_1}{2}\|(Hx)-X\|^2 + \langle\xi, (Hx)-X\rangle$ |
| 16 | W-update: non-negativity [PROX] | $W^{(k+1)} = \max\left( \frac{\rho^{(k)}}{\mu_3} + x^{(k)}, \; 0 \right)$ | $\rho^{(k)}, x^{(k)}: [1,D,H_p,W_p,C]$; $\mu_3$ | $W^{(k+1)}: [1,D,H_p,W_p,C]$ | Proximal operator: $\text{prox}_{\mathcal{I}_{\geq 0}}(v) = \max(v, 0)$ |
| 17 | Compute $r_k$ term 1 | $r_1 = \mu_3 W^{(k+1)} - \rho^{(k)}$ | $W^{(k+1)}, \rho^{(k)}: [1,D,H_p,W_p,C]$; $\mu_3$ | $r_1: [1,D,H_p,W_p,C]$ | Part of RHS for $x$-update |
| 18 | Compute scaled U minus eta | $v_\eta = \mu_2 U^{(k+1)} - \eta^{(k)}$ | $U^{(k+1)}, \eta^{(k)}: [1,D,H_p,W_p,C,2]$; $\mu_2$ | $v_\eta: [1,D,H_p,W_p,C,2]$ | Intermediate for adjoint |
| 19 | Finite difference adjoint | $r_2 = \left(\texttt{roll}(v_\eta[...,0], -1, \text{axis}=-3) - v_\eta[...,0]\right) + \left(\texttt{roll}(v_\eta[...,1], -1, \text{axis}=-2) - v_\eta[...,1]\right)$ | $v_\eta: [1,D,H_p,W_p,C,2]$ | $r_2: [1,D,H_p,W_p,C]$ | Adjoint of finite_diff: $\nabla^T$ |
| 20 | Compute scaled X minus xi | $v_\xi = \mu_1 X^{(k+1)} - \xi^{(k)}$ | $X^{(k+1)}, \xi^{(k)}: [1,D,H_p,W_p,C]$; $\mu_1$ | $v_\xi: [1,D,H_p,W_p,C]$ | Intermediate for adjoint convolution |
| 21 | Adjoint convolution: FFT [ADJOINT] | $\hat{v}_\xi = \mathcal{F}_{\text{2D}}^{\text{rfft}}\{v_\xi\}$ along axes $(-3,-2)$ | $v_\xi: [1,D,H_p,W_p,C]$ | $\hat{v}_\xi: [1,D,H_p,W_p//2+1,C]$ | Forward FFT of input |
| 22 | Adjoint convolution: multiply | $\hat{w} = \hat{v}_\xi \odot \hat{H}^*$ | $\hat{v}_\xi, \hat{H}^*$ | $\hat{w}: [1,D,H_p,W_p//2+1,C]$ | Convolution with conjugate kernel |
| 23 | Adjoint convolution: IFFT | $w = \mathcal{F}_{\text{2D}}^{\text{irfft}}\{\hat{w}\}$ with output shape $(H_p, W_p)$ | $\hat{w}: [1,D,H_p,W_p//2+1,C]$ | $w: [1,D,H_p,W_p,C]$ | Real output |
| 24 | Adjoint convolution: ifftshift | $r_3 = \texttt{ifftshift}(w, \text{axes}=(-3,-2))$ | $w: [1,D,H_p,W_p,C]$ | $r_3: [1,D,H_p,W_p,C]$ | $r_3 = H^T(\mu_1 X^{(k+1)} - \xi^{(k)})$ |
| 25 | Assemble RHS | $r_k = r_1 + r_2 + r_3$ | $r_1, r_2, r_3: [1,D,H_p,W_p,C]$ | $r_k: [1,D,H_p,W_p,C]$ | Full RHS for frequency-domain solve |
| 26 | x-update: FFT of RHS | $\hat{r}_k = \mathcal{F}_{\text{2D}}^{\text{rfft}}\{r_k\}$ along axes $(-3,-2)$ | $r_k: [1,D,H_p,W_p,C]$ | $\hat{r}_k: [1,D,H_p,W_p//2+1,C]$ | Transform to frequency domain |
| 27 | x-update: frequency division | $\hat{x}^{(k+1)} = R_{\text{div}} \odot \hat{r}_k$ | $R_{\text{div}}, \hat{r}_k: [1,D,H_p,W_p//2+1,C]$ | $\hat{x}^{(k+1)}: [1,D,H_p,W_p//2+1,C]$ | Solves $(μ_1 H^TH + μ_2∇^T∇ + μ_3 I)x = r_k$ |
| 28 | x-update: IFFT | $x^{(k+1)} = \mathcal{F}_{\text{2D}}^{\text{irfft}}\{\hat{x}^{(k+1)}\}$ with output shape $(H_p, W_p)$ | $\hat{x}^{(k+1)}$ | $x^{(k+1)}: [1,D,H_p,W_p,C]$ | Updated primal variable |
| 29 | Recompute forward convolution for dual update | $(Hx)^{(k+1)} = H x^{(k+1)}$ (steps 11-14 repeated with $x^{(k+1)}$) | $x^{(k+1)}: [1,D,H_p,W_p,C]$, $\hat{H}$ | $(Hx)^{(k+1)}: [1,D,H_p,W_p,C]$ | Same as steps 11-14 |
| 30 | Recompute finite differences for dual update | $(\nabla x)^{(k+1)} = \nabla x^{(k+1)}$ (same as step 8 with $x^{(k+1)}$) | $x^{(k+1)}: [1,D,H_p,W_p,C]$ | $(\nabla x)^{(k+1)}: [1,D,H_p,W_p,C,2]$ | Same as step 8 |
| 31 | Dual update: $\xi$ | $\xi^{(k+1)} = \xi^{(k)} + \mu_1 \left( (Hx)^{(k+1)} - X^{(k+1)} \right)$ | $\xi^{(k)}, (Hx)^{(k+1)}, X^{(k+1)}$; $\mu_1$ | $\xi^{(k+1)}: [1,D,H_p,W_p,C]$ | Gradient ascent on dual |
| 32 | Dual update: $\eta$ | $\eta^{(k+1)} = \eta^{(k)} + \mu_2 \left( (\nabla x)^{(k+1)} - U^{(k+1)} \right)$ | $\eta^{(k)}, (\nabla x)^{(k+1)}, U^{(k+1)}$; $\mu_2$ | $\eta^{(k+1)}: [1,D,H_p,W_p,C,2]$ | Gradient ascent on dual |
| 33 | Dual update: $\rho$ | $\rho^{(k+1)} = \rho^{(k)} + \mu_3 \left( x^{(k+1)} - W^{(k+1)} \right)$ | $\rho^{(k)}, x^{(k+1)}, W^{(k+1)}$; $\mu_3$ | $\rho^{(k+1)}: [1,D,H_p,W_p,C]$ | Gradient ascent on dual |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 34 | Crop to original size | $x_{\text{out}} = x^{(N_{\text{iter}})}[..., s_0:e_0, s_1:e_1, :]$ | $x^{(N_{\text{iter}})}: [1,D,H_p,W_p,C]$, indices $s_0,e_0,s_1,e_1$ | $x_{\text{out}}: [1,D,H,W,C]$ | Extract valid region |
| 35 | Remove batch dimension | $\hat{x} = x_{\text{out}}[0]$ if batch size is 1 | $x_{\text{out}}: [1,D,H,W,C]$ | $\hat{x}: [D,H,W,C]$ | Squeeze leading dimension |

## 5. STOPPING CRITERION

```json
{
  "type": "iteration_limit",
  "expression": "k < N_{iter}",
  "parameters": {
    "N_iter": 50,
    "default_in_main": 5
  },
  "check_frequency": "Every iteration (at loop condition)"
}
```

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Reconstruction Range | $(\min(\hat{x}), \max(\hat{x}))$ | N/A | $\hat{x}$ | Min and max pixel values of reconstruction |
| Reconstruction Mean | $\bar{x} = \frac{1}{N}\sum_{i,j,c} \hat{x}_{i,j,c}$ where $N = D \cdot H \cdot W \cdot C$ | N/A | $\hat{x}$ | Mean pixel intensity |
| Reconstruction Std | $\sigma_x = \sqrt{\frac{1}{N}\sum_{i,j,c}(\hat{x}_{i,j,c} - \bar{x})^2}$ | N/A | $\hat{x}$ | Standard deviation of pixel values |
| Forward Consistency | $y_{\text{sim}} = H\hat{x}$ (output shape reported) | $y$ (measurement) | $H\hat{x}$ | Simulated measurement from reconstruction |