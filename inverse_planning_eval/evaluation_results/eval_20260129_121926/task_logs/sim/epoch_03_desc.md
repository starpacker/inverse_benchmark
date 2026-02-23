## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Sparse Hessian Deconvolution with Richardson-Lucy Refinement
- **algorithm_family**: Split Bregman / ADMM for convex optimization with iterative deconvolution
- **summary**: Two-stage image reconstruction combining L1-regularized Hessian-based sparse deconvolution via split Bregman iterations followed by accelerated Richardson-Lucy deblurring.

---

## 1. OBJECTIVE FUNCTION

### **full_expression**

**Stage 1 (Sparse Hessian):**
$$\min_{g \geq 0} \; \frac{\lambda_f}{2}\|g - f\|_2^2 + \lambda_s \|g\|_1 + \|\partial_{xx}g\|_1 + \|\partial_{yy}g\|_1 + \gamma^2\|\partial_{zz}g\|_1 + 2\|\partial_{xy}g\|_1 + 2\gamma\|\partial_{xz}g\|_1 + 2\gamma\|\partial_{yz}g\|_1$$

**Stage 2 (Richardson-Lucy):**
$$\max_{x \geq 0} \; \sum_{i,j} \left[ y_{ij} \log((H * x)_{ij}) - (H * x)_{ij} \right]$$

where $H$ is the PSF convolution operator.

### **data_fidelity_term**

**Stage 1:** $\frac{\lambda_f}{2}\|g - f\|_2^2$ where $\lambda_f = \frac{\text{fidelity}}{\mu} = 150$

**Stage 2:** Poisson log-likelihood: $\sum_{i,j} y_{ij} \log((H * x)_{ij}) - (H * x)_{ij}$

### **regularization_term**

**Stage 1:**
$$R(g) = \lambda_s \|g\|_1 + \sum_{(p,q) \in \mathcal{H}} w_{pq} \|\partial_{pq}g\|_1$$

where $\mathcal{H} = \{(x,x), (y,y), (z,z), (x,y), (x,z), (y,z)\}$ with weights:
- $w_{xx} = w_{yy} = 1$
- $w_{zz} = \gamma^2$
- $w_{xy} = 2$
- $w_{xz} = w_{yz} = 2\gamma$

**Stage 2:** None (implicit non-negativity)

### **constraints**

- $g \geq 0$ (enforced by projection after Stage 1)
- $x \geq 10^{-6}$ (enforced by clipping in Stage 2)

---

## 2. VARIABLES

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $y_{\text{raw}}$ | observations | $(T, H, W)$ or $(H, W)$ | float32 | Loaded from file | Raw input image |
| $f$ | observations | $(Z, H, W)$ | float32 | Preprocessed $y_{\text{raw}}$ | Background-subtracted, normalized input |

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $g$ | primal | $(Z, H, W)$ | float32 | $g^{(0)} = f$ | Sparse Hessian reconstruction |
| $x_k$ | primal | $(H+2B, W+2B)$ | float32 | $x_0 = y_{\text{padded}}$ | R-L iterate |
| $y_k$ | primal | $(H+2B, W+2B)$ | float32 | $y_0 = y_{\text{padded}}$ | R-L accelerated iterate |

### Dual/Auxiliary Variables (Split Bregman)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $b_{xx}$ | dual | $(Z, H, W)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xx}$ |
| $b_{yy}$ | dual | $(Z, H, W)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{yy}$ |
| $b_{zz}$ | dual | $(Z, H, W)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{zz}$ |
| $b_{xy}$ | dual | $(Z, H, W)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xy}$ |
| $b_{xz}$ | dual | $(Z, H, W)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xz}$ |
| $b_{yz}$ | dual | $(Z, H, W)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{yz}$ |
| $b_{\ell_1}$ | dual | $(Z, H, W)$ | float32 | $\mathbf{0}$ | Bregman variable for sparsity |
| $d_{pq}$ | auxiliary | $(Z, H, W)$ | float32 | Computed per iteration | Shrinkage output for derivative $(p,q)$ |
| $v_k$ | auxiliary | $(H+2B, W+2B)$ | float32 | $\mathbf{0}$ | R-L momentum variable |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\lambda_f$ | constants | scalar | float32 | $\frac{\text{fidelity}}{\mu} = \frac{150}{1} = 150$ | Fidelity weight ratio |
| $\lambda_s$ | constants | scalar | float32 | $\text{sparsity} = 10$ | Sparsity regularization weight |
| $\gamma$ | constants | scalar | float32 | $\sqrt{\text{tcontinuity}} = \sqrt{0.5} \approx 0.7071$ | Z-axis continuity factor |
| $\mu$ | constants | scalar | float32 | $1$ | ADMM penalty parameter |
| $\sigma$ | constants | scalar | float32 | $\frac{280}{65} \approx 4.3077$ | PSF Gaussian sigma |
| $K_{\max}^{(1)}$ | constants | scalar | int | $1000$ | Max sparse iterations |
| $K_{\max}^{(2)}$ | constants | scalar | int | $7$ | Max deconvolution iterations |
| $\tau$ | constants | scalar | float32 | $10^{-4}$ | Convergence tolerance |
| $\epsilon_{\text{clip}}$ | constants | scalar | float32 | $10^{-6}$ | Minimum clipping value |
| $\epsilon_{\text{div}}$ | constants | scalar | float32 | $10^{-12}$ | Division stability constant |
| $B$ | constants | scalar | int | $\lfloor \min(H,W)/6 \rfloor$ | Padding width for deconvolution |

### Background Estimation Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\text{dlevel}$ | constants | scalar | int | $7$ | Wavelet decomposition level |
| $\text{wavename}$ | constants | string | - | `'db6'` | Wavelet basis name |
| $\text{bg\_iter}$ | constants | scalar | int | $3$ | Background estimation iterations |
| $\text{th}$ | constants | scalar | float32 | $1$ | Threshold flag for iterative refinement |
| $\text{scale\_bg}$ | constants | scalar | float32 | $2.5$ | Background scaling divisor |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Load Image | $y_{\text{raw}} = \text{imread}(\text{path})$ | path: string | $y_{\text{raw}}$: $(T,H,W)$ or $(H,W)$, uint8/uint16 |
| 2 | Type Conversion | $y = \text{float32}(y_{\text{raw}})$ | $y_{\text{raw}}$ | $y$: float32 |
| 3 | Compute Scaler | $s = \max(y)$ | $y$ | $s$: scalar |
| 4 | Initial Normalization | $y \leftarrow y / s$ | $y, s$ | $y \in [0,1]$ |
| 5 | Background Estimation | $\text{bg} = \text{BackgroundEstimate}(y/2.5)$ | $y$ | $\text{bg}$: $(T,H,W)$ |
| 5a | └─ Wavelet Decompose | $\{c_A^{(L)}, (c_H^{(\ell)}, c_V^{(\ell)}, c_D^{(\ell)})\}_{\ell=1}^L = \text{wavedec2}(I, \text{'db6'}, L=7)$ | $I$: $(H,W)$ | Wavelet coefficients |
| 5b | └─ Zero Detail Coeffs | $\tilde{c}_H^{(\ell)} = \tilde{c}_V^{(\ell)} = \tilde{c}_D^{(\ell)} = \mathbf{0}, \; \forall \ell$ | Detail coeffs | Zeroed detail coeffs |
| 5c | └─ Wavelet Reconstruct | $B_0 = \text{waverec2}(\{c_A^{(L)}, \mathbf{0}\})$ | Modified coeffs | $B_0$: $(H,W)$ |
| 5d | └─ Iterative Refinement | For $i=1,\ldots,3$: $\epsilon = \frac{1}{2}\sqrt{|I|}$; where $I > B_{i-1}+\epsilon$: $I \leftarrow B_{i-1}+\epsilon$; recompute $B_i$ | $I, B_{i-1}$ | $B_i$: $(H,W)$ |
| 6 | Background Subtract | $y \leftarrow y - \text{bg}$ | $y, \text{bg}$ | $y$: background-free |
| 7 | Renormalize | $y \leftarrow y / \max(y)$ | $y$ | $y \in [0,1]$ |
| 8 | Clip Negatives | $y \leftarrow \max(y, 0)$ | $y$ | $y \geq 0$ |
| 9 | Dimension Handling | If 2D: $f = \text{stack}([y,y,y])$; If 3D with $T<3$: pad to 3 slices | $y$ | $f$: $(Z,H,W)$ with $Z \geq 3$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Compute Continuity | $\gamma = \sqrt{t_{\text{cont}}} = \sqrt{0.5}$ | $t_{\text{cont}}=0.5$ | $\gamma \approx 0.7071$ |
| 2 | Build $\widehat{\Delta_{xx}}$ | $\widehat{\Delta_{xx}} = |\mathcal{F}\{[1,-2,1]\}|^2$ | imgsize $(Z,H,W)$ | $\widehat{\Delta_{xx}}$: $(Z,H,W)$ complex |
| 3 | Build $\widehat{\Delta_{yy}}$ | $\widehat{\Delta_{yy}} = |\mathcal{F}\{[1;-2;1]\}|^2$ | imgsize | $\widehat{\Delta_{yy}}$: $(Z,H,W)$ |
| 4 | Build $\widehat{\Delta_{zz}}$ | $\widehat{\Delta_{zz}} = |\mathcal{F}\{\delta_{zz}\}|^2$ where $\delta_{zz}[0,0,0]=1, \delta_{zz}[1,0,0]=-2, \delta_{zz}[2,0,0]=1$ | imgsize | $\widehat{\Delta_{zz}}$: $(Z,H,W)$ |
| 5 | Build $\widehat{\Delta_{xy}}$ | $\widehat{\Delta_{xy}} = |\mathcal{F}\{[[1,-1];[-1,1]]\}|^2$ | imgsize | $\widehat{\Delta_{xy}}$: $(Z,H,W)$ |
| 6 | Build $\widehat{\Delta_{xz}}$ | $\widehat{\Delta_{xz}} = |\mathcal{F}\{\delta_{xz}\}|^2$ | imgsize | $\widehat{\Delta_{xz}}$: $(Z,H,W)$ |
| 7 | Build $\widehat{\Delta_{yz}}$ | $\widehat{\Delta_{yz}} = |\mathcal{F}\{\delta_{yz}\}|^2$ | imgsize | $\widehat{\Delta_{yz}}$: $(Z,H,W)$ |
| 8 | Build Normalizer | $\widehat{N} = \lambda_f + \lambda_s^2 + \widehat{\Delta_{xx}} + \widehat{\Delta_{yy}} + \gamma^2\widehat{\Delta_{zz}} + 2\widehat{\Delta_{xy}} + 2\gamma\widehat{\Delta_{xz}} + 2\gamma\widehat{\Delta_{yz}}$ | All $\widehat{\Delta}$, $\lambda_f=150$, $\lambda_s=10$, $\gamma$ | $\widehat{N}$: $(Z,H,W)$ |
| 9 | Initialize Bregman | $b_{xx} = b_{yy} = b_{zz} = b_{xy} = b_{xz} = b_{yz} = b_{\ell_1} = \mathbf{0}$ | imgsize | All $b$: $(Z,H,W)$ zeros |
| 10 | Initialize Update | $u^{(0)} = \lambda_f \cdot f$ | $f$, $\lambda_f$ | $u^{(0)}$: $(Z,H,W)$ |
| 11 | Initialize Residual | $r_{\text{prev}} = +\infty$ | - | $r_{\text{prev}}$: scalar |

### 4.2 ITERATION LOOP

#### 4.2.1 STAGE 1: SPARSE HESSIAN (Split Bregman)

**BEGIN_LOOP** (index $k = 0, \ldots, K_{\max}^{(1)}-1$ where $K_{\max}^{(1)} = 1000$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| 1 | Fourier Transform Update | $\widehat{u}^{(k)} = \mathcal{F}\{u^{(k)}\}$ | $u^{(k)}$ | $\widehat{u}^{(k)}$ | 3D FFT |
| 2 | Primal Update | If $k=0$: $g^{(k)} = \mathcal{F}^{-1}\{\widehat{u}^{(k)}/\lambda_f\}$; else: $g^{(k)} = \mathcal{F}^{-1}\{\widehat{u}^{(k)}/\widehat{N}\}$ | $\widehat{u}^{(k)}$, $\widehat{N}$ or $\lambda_f$ | $g^{(k)}$ | Real part only |
| 3 | Reset Accumulator | $u^{(k+1)} = \lambda_f \cdot f$ | $f$, $\lambda_f$ | $u^{(k+1)}$ | - |
| 4 | XX-derivative Update | $g_{xx} = D_x^- D_x^+ g^{(k)}$ | $g^{(k)}$ | $g_{xx}$ | Backward-forward diff |
| 5 | XX-shrinkage | $d_{xx} = \text{shrink}(g_{xx} + b_{xx}, \mu) = \text{sign}(g_{xx}+b_{xx})\max(|g_{xx}+b_{xx}|-1/\mu, 0)$ | $g_{xx}$, $b_{xx}$, $\mu=1$ | $d_{xx}$ | Soft threshold |
| 6 | XX-Bregman Update | $b_{xx} \leftarrow b_{xx} + g_{xx} - d_{xx}$ | $b_{xx}$, $g_{xx}$, $d_{xx}$ | $b_{xx}$ | - |
| 7 | XX-accumulate | $L_{xx} = D_x^- D_x^+ (d_{xx} - b_{xx})$; $u^{(k+1)} \leftarrow u^{(k+1)} + L_{xx}$ | $d_{xx}$, $b_{xx}$ | $u^{(k+1)}$ | Weight $w_{xx}=1$ |
| 8-11 | YY-updates | Same as steps 4-7 with $D_y^-, D_y^+$ | - | $b_{yy}$, $u^{(k+1)}$ | Weight $w_{yy}=1$ |
| 12-15 | ZZ-updates | Same as steps 4-7 with $D_z^-, D_z^+$; $L_{zz} = \gamma^2 D_z^- D_z^+(d_{zz}-b_{zz})$ | - | $b_{zz}$, $u^{(k+1)}$ | Weight $w_{zz}=\gamma^2$ |
| 16-19 | XY-updates | $g_{xy} = D_x^+ D_y^+ g$; shrink; $L_{xy} = 2 D_y^- D_x^-(d_{xy}-b_{xy})$ | - | $b_{xy}$, $u^{(k+1)}$ | Weight $w_{xy}=2$ |
| 20-23 | XZ-updates | $g_{xz} = D_x^+ D_z^+ g$; shrink; $L_{xz} = 2\gamma D_z^- D_x^-(d_{xz}-b_{xz})$ | - | $b_{xz}$, $u^{(k+1)}$ | Weight $w_{xz}=2\gamma$ |
| 24-27 | YZ-updates | $g_{yz} = D_y^+ D_z^+ g$; shrink; $L_{yz} = 2\gamma D_z^- D_y^-(d_{yz}-b_{yz})$ | - | $b_{yz}$, $u^{(k+1)}$ | Weight $w_{yz}=2\gamma$ |
| 28-31 | Sparsity updates | $d_{\ell_1} = \text{shrink}(g^{(k)} + b_{\ell_1}, \mu)$; $b_{\ell_1} \leftarrow b_{\ell_1} + g^{(k)} - d_{\ell_1}$; $u^{(k+1)} \leftarrow u^{(k+1)} + \lambda_s(d_{\ell_1}-b_{\ell_1})$ | - | $b_{\ell_1}$, $u^{(k+1)}$ | Weight $\lambda_s=10$ |
| 32 | Convergence Check | If $k \mod 20 = 0$: $r^{(k)} = \|f - g^{(k)}\|_2$; if $\frac{|r^{(k)}-r_{\text{prev}}|}{r_{\text{prev}}+10^{-12}} < \tau$: **BREAK** | $f$, $g^{(k)}$, $r_{\text{prev}}$, $\tau=10^{-4}$ | convergence flag | Every 20 iterations |
| 33 | Update Previous Residual | $r_{\text{prev}} \leftarrow r^{(k)}$ | $r^{(k)}$ | $r_{\text{prev}}$ | - |

**END_LOOP**

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 34 | Non-negativity Projection | $g \leftarrow \max(g, 0)$ | $g$ | $g \geq 0$ |
| 35 | Extract 2D (if applicable) | If input was 2D: $g_{\text{sparse}} = g[1,:,:]$; else $g_{\text{sparse}} = g$ | $g$, flag | $g_{\text{sparse}}$ |
| 36 | Normalize | $g_{\text{sparse}} \leftarrow g_{\text{sparse}} / \max(g_{\text{sparse}})$ | $g_{\text{sparse}}$ | $g_{\text{sparse}} \in [0,1]$ |

#### 4.2.2 STAGE 2: RICHARDSON-LUCY DECONVOLUTION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 37 | Build Gaussian PSF | $h[i,j] = \exp\left(-\frac{i^2}{2\tilde{\sigma}^2} - \frac{j^2}{2\tilde{\sigma}^2}\right)$; $h \leftarrow h/\sum h$; where $\tilde{\sigma} = \sigma/(2\sqrt{2\ln 2})$, $\sigma = 280/65$ | $\sigma$ | $h$: $(N_h, N_h)$ |
| 38 | Compute PSF Support | $N_h = 2\lceil \sigma/\sqrt{8\ln 2} \cdot \sqrt{-2\ln(0.0002)} \rceil + 3$ | $\sigma$ | $N_h$: odd integer |
| 39 | Compute Padding | $B = \lfloor \min(H, W) / 6 \rfloor$ | $H, W$ | $B$: integer |
| 40 | Pad Input | $y_{\text{pad}} = \text{pad}(g_{\text{sparse}}, B, \text{mode='edge'})$ | $g_{\text{sparse}}$, $B$ | $y_{\text{pad}}$: $(H+2B, W+2B)$ |
| 41 | Compute OTF | $\widehat{H} = \text{psf2otf}(h, \text{shape}(y_{\text{pad}}))$ | $h$, shape | $\widehat{H}$: $(H+2B, W+2B)$ complex |
| 42 | Initialize R-L | $y_0 = y_{\text{pad}}$; $x_0 = \mathbf{0}$; $v_0 = \mathbf{0}$ | $y_{\text{pad}}$ | $y_0, x_0, v_0$ |

**BEGIN_LOOP** (index $k = 0, \ldots, K_{\max}^{(2)}-1$ where $K_{\max}^{(2)} = 7$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| 43 | Store Previous | $x_{k-1} \leftarrow x_k$ | $x_k$ | $x_{k-1}$ | - |
| 44 | Compute Ratio | $r_k = \frac{y_{\text{pad}}}{\max(\mathcal{F}^{-1}\{\widehat{H} \cdot \mathcal{F}\{y_k\}\}, 10^{-6})}$ | $y_{\text{pad}}$, $y_k$, $\widehat{H}$ | $r_k$ | Clipped denominator |
| 45 | R-L Update | $x_{k+1} = \frac{y_k \cdot \text{Re}(\mathcal{F}^{-1}\{\widehat{H}^* \cdot \mathcal{F}\{r_k\}\})}{\text{Re}(\mathcal{F}^{-1}\{\widehat{H} \cdot \mathcal{F}\{\mathbf{1}\}\})}$ | $y_k$, $r_k$, $\widehat{H}$ | $x_{k+1}$ | - |
| 46 | Clip Primal | $x_{k+1} \leftarrow \max(x_{k+1}, 10^{-6})$ | $x_{k+1}$ | $x_{k+1} \geq 10^{-6}$ | - |
| 47 | Store Previous Momentum | $v_{k-1} \leftarrow v_k$ | $v_k$ | $v_{k-1}$ | - |
| 48 | Update Momentum | $v_{k+1} = \max(x_{k+1} - y_k, 10^{-6})$ | $x_{k+1}$, $y_k$ | $v_{k+1}$ | - |
| 49 | Compute Acceleration | If $k=0$: $\alpha_k = 0$; else: $\alpha_k = \text{clip}\left(\frac{\langle v_{k-1}, v_k \rangle}{\|v_{k-1}\|_2^2 + 10^{-10}}, 10^{-6}, 1\right)$ | $v_{k-1}$, $v_k$ | $\alpha_k \in [10^{-6}, 1]$ | Adaptive step |
| 50 | Accelerated Update | $y_{k+1} = \max(x_{k+1} + \alpha_k(x_{k+1} - x_k), 10^{-6})$ | $x_{k+1}$, $x_k$, $\alpha_k$ | $y_{k+1}$ | - |
| 51 | NaN Handling | $y_{k+1}[\text{isnan}(y_{k+1})] \leftarrow 10^{-6}$ | $y_{k+1}$ | $y_{k+1}$ | - |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 52 | Final Clipping | $y_{\text{final}} \leftarrow \max(y_K, 0)$ | $y_K$ | $y_{\text{final}} \geq 0$ |
| 53 | Remove Padding | $x_{\text{out}} = y_{\text{final}}[B:H+B, B:W+B]$ | $y_{\text{final}}$, $B$ | $x_{\text{out}}$: $(H, W)$ |
| 54 | Rescale Output | $x_{\text{result}} = s \cdot x_{\text{out}}$ | $x_{\text{out}}$, scaler $s$ | $x_{\text{result}}$ |

---

## 5. STOPPING CRITERION

| Property | Value |
|----------|-------|
| **type** | combined |
| **expression** | **Stage 1:** $(k \geq K_{\max}^{(1)}) \lor \left(k \mod 20 = 0 \land \frac{|r^{(k)} - r^{(k-20)}|}{r^{(k-20)} + 10^{-12}} < 10^{-4}\right)$; **Stage 2:** $k \geq K_{\max}^{(2)}$ |
| **parameters** | $K_{\max}^{(1)} = 1000$, $K_{\max}^{(2)} = 7$, $\tau = 10^{-4}$, $\epsilon_{\text{div}} = 10^{-12}$ |
| **check_frequency** | Stage 1: every 20 iterations; Stage 2: never (fixed iterations) |

---

## 6. EVALUATION METRICS

| Name | Formula