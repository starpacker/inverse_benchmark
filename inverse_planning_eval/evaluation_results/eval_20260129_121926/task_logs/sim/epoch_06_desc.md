## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Sparse Hessian Deconvolution with Split Bregman",
  "algorithm_family": "ADMM / Split Bregman with Iterative Deconvolution",
  "summary": "A two-stage image reconstruction algorithm combining Split Bregman optimization for sparse Hessian regularization followed by accelerated Richardson-Lucy or Landweber iterative deconvolution."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{g \geq 0} \frac{\lambda_f}{2\mu} \|g - f\|_2^2 + \lambda_s \|g\|_1 + \|\partial_{xx} g\|_1 + \|\partial_{yy} g\|_1 + \gamma^2 \|\partial_{zz} g\|_1 + 2\|\partial_{xy} g\|_1 + 2\gamma\|\partial_{xz} g\|_1 + 2\gamma\|\partial_{yz} g\|_1$$

followed by deconvolution stage solving:
$$\min_{x \geq 0} D_{KL}(y \| Hx) \quad \text{(Richardson-Lucy)}$$
or
$$\min_{x \geq 0} \|y - Hx\|_2^2 \quad \text{(Landweber)}$$

**data_fidelity_term**:
- Stage 1 (Sparse Hessian): $\frac{\lambda_f}{2\mu} \|g - f\|_2^2$ where $\lambda_f = \texttt{fidelity}$, $\mu = \texttt{mu}$
- Stage 2 (Richardson-Lucy): $D_{KL}(y \| Hx) = \sum_{i,j} \left[ y_{i,j} \log\frac{y_{i,j}}{(Hx)_{i,j}} - y_{i,j} + (Hx)_{i,j} \right]$
- Stage 2 (Landweber): $\frac{1}{2}\|y - Hx\|_2^2$

**regularization_term**:
- L1 sparsity on image: $\lambda_s \|g\|_1$ where $\lambda_s = \texttt{sparsity}$
- Hessian xx: $\|\partial_{xx} g\|_1$
- Hessian yy: $\|\partial_{yy} g\|_1$
- Hessian zz: $\gamma^2 \|\partial_{zz} g\|_1$ where $\gamma = \sqrt{\texttt{tcontinuity}}$
- Hessian xy: $2\|\partial_{xy} g\|_1$
- Hessian xz: $2\gamma\|\partial_{xz} g\|_1$
- Hessian yz: $2\gamma\|\partial_{yz} g\|_1$

**constraints**:
$$g \geq 0, \quad x \geq \epsilon \quad \text{where } \epsilon = 10^{-6}$$

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g$ | $(N_z, N_x, N_y)$ | float32 | $\mathcal{F}^{-1}\left(\frac{\mathcal{F}(g\_update)}{\lambda_f/\mu}\right)$ | Sparse Hessian reconstructed image |
| $x_k$ | $(N_x+2B, N_y+2B)$ | float32 | $y_{\text{padded}}$ | Deconvolution iterate (current estimate) |
| $y_k$ | $(N_x+2B, N_y+2B)$ | float32 | $y_{\text{padded}}$ | Deconvolution accelerated iterate |

### 2.2 Dual Variables (Bregman Variables)

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $b_{xx}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xx}$ |
| $b_{yy}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{yy}$ |
| $b_{zz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{zz}$ |
| $b_{xy}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xy}$ |
| $b_{xz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xz}$ |
| $b_{yz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{yz}$ |
| $b_{l1}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for L1 sparsity |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g\_update$ | $(N_z, N_x, N_y)$ | float32/complex128 | $\frac{\lambda_f}{\mu} f$ | Accumulated update in spatial/Fourier domain |
| $d_{xx}$ | $(N_z, N_x, N_y)$ | float32 | Computed | Shrinkage result for $\partial_{xx}$ |
| $d_{yy}$ | $(N_z, N_x, N_y)$ | float32 | Computed | Shrinkage result for $\partial_{yy}$ |
| $d_{zz}$ | $(N_z, N_x, N_y)$ | float32 | Computed | Shrinkage result for $\partial_{zz}$ |
| $d_{xy}$ | $(N_z, N_x, N_y)$ | float32 | Computed | Shrinkage result for $\partial_{xy}$ |
| $d_{xz}$ | $(N_z, N_x, N_y)$ | float32 | Computed | Shrinkage result for $\partial_{xz}$ |
| $d_{yz}$ | $(N_z, N_x, N_y)$ | float32 | Computed | Shrinkage result for $\partial_{yz}$ |
| $d_{sparse}$ | $(N_z, N_x, N_y)$ | float32 | Computed | Shrinkage result for L1 sparsity |
| $v_k$ | $(N_x+2B, N_y+2B)$ | float32 | $\mathbf{0}$ | Velocity variable for RL acceleration |
| $x_{k,\text{update}}$ | $(N_x+2B, N_y+2B)$ | float32 | Computed | Previous deconv iterate |
| $v_{k,\text{update}}$ | $(N_x+2B, N_y+2B)$ | float32 | Computed | Previous velocity |
| $\text{residual}$ | scalar | float64 | $\infty$ | Convergence tracking variable |
| $\text{residual\_prev}$ | scalar | float64 | $\infty$ | Previous residual value |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $\lambda_f$ | scalar | int | 150 | Fidelity parameter |
| $\lambda_s$ | scalar | int | 10 | Sparsity parameter |
| $\mu$ | scalar | float | 1.0 | ADMM penalty parameter |
| $\gamma$ | scalar | float | $\sqrt{0.5} \approx 0.707$ | Z-axis continuity weight |
| $K_{\text{sparse}}$ | scalar | int | 1000 | Maximum sparse iterations |
| $K_{\text{deconv}}$ | scalar | int | 7 | Maximum deconvolution iterations |
| $\sigma_{\text{PSF}}$ | scalar | float | $280/65 \approx 4.308$ | PSF sigma parameter |
| $\text{tol}$ | scalar | float | $10^{-4}$ | Convergence tolerance |
| $\epsilon$ | scalar | float | $10^{-6}$ | Numerical floor |
| $\delta_{xx}$ | $(1,1,3)$ | float32 | $[[[1, -2, 1]]]$ | Second derivative kernel (x) |
| $\delta_{yy}$ | $(1,3,1)$ | float32 | $[[[1], [-2], [1]]]$ | Second derivative kernel (y) |
| $\delta_{zz}$ | $(3,1,1)$ | float32 | $[[[1]], [[-2]], [[1]]]$ | Second derivative kernel (z) |
| $\delta_{xy}$ | $(1,2,2)$ | float32 | $[[[1, -1], [-1, 1]]]$ | Mixed derivative kernel (xy) |
| $\delta_{xz}$ | $(2,1,2)$ | float32 | $[[[1, -1]], [[-1, 1]]]$ | Mixed derivative kernel (xz) |
| $\delta_{yz}$ | $(2,2,1)$ | float32 | $[[[1], [-1]], [[-1], [1]]]$ | Mixed derivative kernel (yz) |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y_{\text{raw}}$ | $(N_z, N_x, N_y)$ or $(N_x, N_y)$ | float32 | Raw input image from file |
| $f$ | $(N_z, N_x, N_y)$ | float32 | Preprocessed normalized observation |
| $y_{\text{padded}}$ | $(N_x+2B, N_y+2B)$ | float32 | Padded data for deconvolution |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Load Image | $y_{\text{raw}} = \texttt{io.imread}(\text{path})$ | File path | $y_{\text{raw}}$: $(N_z, N_x, N_y)$ or $(N_x, N_y)$ | Valid image file |
| P2 | Convert to Float | $y_{\text{float}} = \texttt{float32}(y_{\text{raw}})$ | $y_{\text{raw}}$ | $y_{\text{float}}$: same shape, float32 | — |
| P3 | Compute Scaler | $s = \max(y_{\text{float}})$ | $y_{\text{float}}$ | $s$: scalar | $\max > 0$ |
| P4 | Initial Normalization | $y_{\text{norm}} = y_{\text{float}} / s$ | $y_{\text{float}}$, $s$ | $y_{\text{norm}}$: same shape | — |
| P5 | Background Estimation | For each slice $t$: (a) $\text{coeffs} = \texttt{wavedec2}(y_{\text{norm}}[t]/2.5, \text{'db6'}, 7)$; (b) Zero all detail coefficients; (c) $B_{\text{init}} = \texttt{waverec2}(\text{coeffs\_zeroed})$; (d) $\epsilon_t = \sqrt{|y_{\text{norm}}[t]/2.5|}/2$; (e) Where $y > B + \epsilon$: $\text{res} \leftarrow B + \epsilon$; (f) Repeat wavelet decomp on res; (g) $\text{Background}[t] = \texttt{waverec2}(\text{coeffs\_final})$ | $y_{\text{norm}}$: $(N_z, N_x, N_y)$ | $\text{Background}$: $(N_z, N_x, N_y)$ | 7-level wavelet decomposition possible |
| P6 | Background Subtraction | $y_{\text{sub}} = y_{\text{norm}} - \text{Background}$ | $y_{\text{norm}}$, Background | $y_{\text{sub}}$: same shape | — |
| P7 | Renormalization | $y_{\text{renorm}} = y_{\text{sub}} / \max(y_{\text{sub}})$ | $y_{\text{sub}}$ | $y_{\text{renorm}}$: same shape | $\max > 0$ |
| P8 | Clamp Negatives | $y_{\text{clamp}}[i,j,k] = \max(y_{\text{renorm}}[i,j,k], 0)$ | $y_{\text{renorm}}$ | $y_{\text{clamp}}$: same shape | — |
| P9 | Optional Upsampling | If $\texttt{up\_sample}=1$: Fourier upsample by 2; If $\texttt{up\_sample}=2$: Spatial upsample by 2; Else: no change | $y_{\text{clamp}}$ | $y_{\text{up}}$: $(N_z, 2N_x, 2N_y)$ or same | — |
| P10 | Final Normalization | $f = y_{\text{up}} / \max(y_{\text{up}})$ | $y_{\text{up}}$ | $f$: preprocessed observation | — |
| P11 | Dimension Handling | If $\text{ndim}=2$: $\gamma \leftarrow 0$, $f \leftarrow \text{stack}([f,f,f])$ along axis 0; If $\text{ndim}=3$ and $N_z < 3$: $\gamma \leftarrow 0$, pad to 3 slices | $f$ | $f$: $(N_z, N_x, N_y)$ with $N_z \geq 3$ | — |
| P12 | Compute Hessian OTFs | $\hat{\Delta}_{xx} = |\mathcal{F}(\delta_{xx}, \text{imgsize})|^2$, $\hat{\Delta}_{yy} = |\mathcal{F}(\delta_{yy}, \text{imgsize})|^2$, $\hat{\Delta}_{zz} = |\mathcal{F}(\delta_{zz}, \text{imgsize})|^2$, $\hat{\Delta}_{xy} = |\mathcal{F}(\delta_{xy}, \text{imgsize})|^2$, $\hat{\Delta}_{xz} = |\mathcal{F}(\delta_{xz}, \text{imgsize})|^2$, $\hat{\Delta}_{yz} = |\mathcal{F}(\delta_{yz}, \text{imgsize})|^2$ | Kernel arrays, imgsize | Each $\hat{\Delta}$: $(N_z, N_x, N_y)$, complex128 | — |
| P13 | Compute Combined Operator | $\hat{\Phi} = \hat{\Delta}_{xx} + \hat{\Delta}_{yy} + \gamma^2 \hat{\Delta}_{zz} + 2\hat{\Delta}_{xy} + 2\gamma\hat{\Delta}_{xz} + 2\gamma\hat{\Delta}_{yz}$ | All $\hat{\Delta}$ | $\hat{\Phi}$: $(N_z, N_x, N_y)$ | — |
| P14 | Compute Normalization | $\mathcal{N} = \frac{\lambda_f}{\mu} + \lambda_s^2 + \hat{\Phi}$ | $\lambda_f$, $\mu$, $\lambda_s$, $\hat{\Phi}$ | $\mathcal{N}$: $(N_z, N_x, N_y)$ | — |
| P15 | Compute PSF Kernel | $\sigma_{\text{std}} = \frac{\sigma_{\text{PSF}}}{2\sqrt{2\ln 2}}$; $N_{\text{psf}} = 2\lceil\frac{\sigma_{\text{PSF}}}{\sqrt{8\ln 2}}\sqrt{-2\ln(0.0002)}\rceil + 1$; For $i,j \in [-\lfloor N/2 \rfloor, \lceil N/2 \rceil)$: $K[i,j] = \exp\left(-\frac{i^2 + j^2}{2\sigma_{\text{std}}^2}\right)$; $K \leftarrow K / \sum K$ | $\sigma_{\text{PSF}}$ | $K$: $(N_{\text{psf}}, N_{\text{psf}})$, float32 | — |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $b_{xx}$ | $\mathbf{0}$ | $(N_z, N_x, N_y)$ | float32 | Zero array |
| 2 | $b_{yy}$ | $\mathbf{0}$ | $(N_z, N_x, N_y)$ | float32 | Zero array |
| 3 | $b_{zz}$ | $\mathbf{0}$ | $(N_z, N_x, N_y)$ | float32 | Zero array |
| 4 | $b_{xy}$ | $\mathbf{0}$ | $(N_z, N_x, N_y)$ | float32 | Zero array |
| 5 | $b_{xz}$ | $\mathbf{0}$ | $(N_z, N_x, N_y)$ | float32 | Zero array |
| 6 | $b_{yz}$ | $\mathbf{0}$ | $(N_z, N_x, N_y)$ | float32 | Zero array |
| 7 | $b_{l1}$ | $\mathbf{0}$ | $(N_z, N_x, N_y)$ | float32 | Zero array |
| 8 | $g\_update$ | $\frac{\lambda_f}{\mu} \cdot f$ | $(N_z, N_x, N_y)$ | float32 | Scaled observation |
| 9 | $\text{residual\_prev}$ | $+\infty$ | scalar | float64 | Infinity |
| 10 | $k$ | $0$ | scalar | int | Iteration counter |

### 4.2 ITERATION LOOP

**--- BEGIN_LOOP: Sparse Hessian Iterations (for $k = 0, 1, \ldots, K_{\text{sparse}}-1$) ---**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 11 | FFT of g_update | $\hat{g}\_update = \mathcal{F}(g\_update)$ | $g\_update$: $(N_z, N_x, N_y)$ | $\hat{g}\_update$: $(N_z, N_x, N_y)$, complex128 | [FORWARD] 3D FFT |
| 12 | Update g (first iter) | If $k=0$: $g = \Re\left\{\mathcal{F}^{-1}\left(\frac{\hat{g}\_update}{\lambda_f/\mu}\right)\right\}$ | $\hat{g}\_update$, $\lambda_f$, $\mu$ | $g$: $(N_z, N_x, N_y)$, float32 | Division by scalar |
| 13 | Update g (subsequent) | If $k>0$: $g = \Re\left\{\mathcal{F}^{-1}\left(\frac{\hat{g}\_update}{\mathcal{N}}\right)\right\}$ | $\hat{g}\_update$, $\mathcal{N}$ | $g$: $(N_z, N_x, N_y)$, float32 | Element-wise division then IFFT |
| 14 | Reset g_update | $g\_update = \frac{\lambda_f}{\mu} \cdot f$ | $f$, $\lambda_f$, $\mu$ | $g\_update$: $(N_z, N_x, N_y)$ | Scalar multiplication |
| 15 | Compute $g_{xx}$ | $g_{xx} = D_x^- D_x^+ g$ where $(D_x^+ g)[z,i,j] = g[z,i+1,j] - g[z,i,j]$ and $(D_x^- g)[z,i,j] = g[z,i,j] - g[z,i-1,j]$ with zero boundary | $g$ | $g_{xx}$: $(N_z, N_x, N_y)$ | Second derivative along axis 1 |
| 16 | Shrinkage $d_{xx}$ | $d_{xx} = \text{sign}(g_{xx} + b_{xx}) \cdot \max(|g_{xx} + b_{xx}| - 1/\mu, 0)$ | $g_{xx}$, $b_{xx}$, $\mu$ | $d_{xx}$: $(N_z, N_x, N_y)$ | Soft thresholding |
| 17 | Update $b_{xx}$ | $b_{xx} \leftarrow b_{xx} + g_{xx} - d_{xx}$ | $b_{xx}$, $g_{xx}$, $d_{xx}$ | $b_{xx}$: $(N_z, N_x, N_y)$ | Bregman update |
| 18 | Compute $L_{xx}$ | $L_{xx} = 1 \cdot D_x^- D_x^+ (d_{xx} - b_{xx})$ | $d_{xx}$, $b_{xx}$ | $L_{xx}$: $(N_z, N_x, N_y)$ | para=1 for xx |
| 19 | Accumulate $L_{xx}$ | $g\_update \leftarrow g\_update + L_{xx}$ | $g\_update$, $L_{xx}$ | $g\_update$: $(N_z, N_x, N_y)$ | Element-wise add |
| 20 | Compute $g_{yy}$ | $g_{yy} = D_y^- D_y^+ g$ where $(D_y^+ g)[z,i,j] = g[z,i,j+1] - g[z,i,j]$ | $g$ | $g_{yy}$: $(N_z, N_x, N_y)$ | Second derivative along axis 2 |
| 21 | Shrinkage $d_{yy}$ | $d_{yy} = \text{sign}(g_{yy} + b_{yy}) \cdot \max(|g_{yy} + b_{yy}| - 1/\mu, 0)$ | $g_{yy}$, $b_{yy}$, $\mu$ | $d_{yy}$: $(N_z, N_x, N_y)$ | Soft thresholding |
| 22 | Update $b_{yy}$ | $b_{yy} \leftarrow b_{yy} + g_{yy} - d_{yy}$ | $b_{yy}$, $g_{yy}$, $d_{yy}$ | $b_{yy}$: $(N_z, N_x, N_y)$ | Bregman update |
| 23 | Compute $L_{yy}$ | $L_{yy} = 1 \cdot D_y^- D_y^+ (d_{yy} - b_{yy})$ | $d_{yy}$, $b_{yy}$ | $L_{yy}$: $(N_z, N_x, N_y)$ | para=1 for yy |
| 24 | Accumulate $L_{yy}$ | $g\_update \leftarrow g\_update + L_{yy}$ | $g\_update$, $L_{yy}$ | $g\_update$: $(N_z, N_x, N_y)$ | Element-wise add |
| 25 | Compute $g_{zz}$ | $g_{zz} = D_z^- D_z^+ g$ where $(D_z^+ g)[z,i,j] = g[z+1,i,j] - g[z,i,j]$ | $g$ | $g_{zz}$: $(N_z, N_x, N_y)$ | Second derivative along axis 0 |
| 26 | Shrinkage $d_{zz}$ | $d_{zz} = \text{sign}(g_{zz} + b_{zz}) \cdot \max(|g_{zz} + b_{zz}| - 1/\mu, 0)$ | $g_{zz}$, $b_{zz}$, $\mu$ | $d_{zz}$: $(N_z, N_x, N_y)$ | Soft thresholding |
| 27 | Update $b_{zz}$ | $b_{zz} \leftarrow b_{zz} + g_{zz} - d_{zz}$ | $b_{zz}$, $g_{zz}$, $d_{zz}$ | $b_{zz}$: $(N_z, N_x, N_y)$ | Bregman update |
| 28 | Compute $L_{zz}$ | $L_{zz} = \gamma^2 \cdot D_z^- D_z^+ (d_{zz} - b_{zz})$ | $d_{zz}$, $b_{zz}$, $\gamma$ | $L_{zz}$: $(N_z, N_x, N_y)$ | para=$\gamma^2$ for zz |
| 29 | Accumulate $L_{zz}$ | $g\_update \leftarrow g\_update + L_{zz}$ | $g\_update$, $L_{zz}$ | $g\_update$: $(N_z, N_x, N_y)$ | Element-wise add |
| 30 | Compute $g_{xy}$ | $g_{xy} = D_y^+ D_x^+ g$ | $g$ | $g_{xy}$: $(N_z, N_x, N_y)$ | Mixed derivative axes 1,2 |
| 31 | Shrinkage $d_{xy}$ | $d_{xy} = \text{sign}(g_{xy} + b_{xy}) \cdot \max(|g_{xy} + b_{xy}| - 1/\mu, 0)$ | $g_{xy}$, $b_{xy}$, $\mu$ | $d_{xy}$: $(N_z, N_x, N_y)$ | Soft thresholding |
| 32 | Update $b_{xy}$ | $b_{xy} \leftarrow b_{xy} + g_{xy} - d_{xy}$ | $b_{xy}$, $g_{xy}$, $d_{xy}$ | $b_{xy}$: $(N_z, N_x, N_y)$ | Bregman update |
| 33 | Compute $L_{xy}$ | $L_{xy} = 2 \cdot D_x^- D_y^- (d_{xy} - b_{xy})$ | $d_{xy}$, $b_{xy}$ | $L_{xy}$: $(N_z, N_x, N_y)$ | para=2 for xy |
| 34 | Accumulate $L_{xy}$ | $g\_update \leftarrow g\_update + L_{xy}$ | $g\_update$, $L_{xy}$ | $g\_update$: $(N_z, N_x, N_y)$ | Element-wise add |
| 35 | Compute $g_{xz}$ | $g_{xz} = D_z^+ D_x^+ g$ | $g$ | $g_{xz}$: $(N_z, N_x, N_y)$ | Mixed derivative axes 1,0 |
| 36 | Shrinkage $d_{xz}$ | $d_{xz} = \text{sign}(g_{xz} + b_{xz}) \cdot \max(|g_{xz} + b_{xz}| - 1/\mu, 0)$ | $g_{xz}$