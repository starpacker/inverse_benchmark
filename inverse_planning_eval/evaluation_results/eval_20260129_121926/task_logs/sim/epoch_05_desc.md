## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Sparse Hessian Deconvolution with Accelerated Richardson-Lucy/Landweber Refinement
- **algorithm_family**: Split Bregman / ADMM for Sparse Reconstruction + Iterative Deconvolution
- **summary**: A two-stage image reconstruction algorithm that first solves a Hessian-regularized sparse optimization problem via split Bregman iterations in the Fourier domain, then refines the result using accelerated Richardson-Lucy or Landweber deconvolution.

---

## 1. OBJECTIVE FUNCTION

### full_expression

**Stage 1 (Sparse Hessian Reconstruction):**
$$\min_{g \geq 0} \frac{\lambda_f}{\mu}\|f - g\|_2^2 + \lambda_s^2 \|g\|_1 + \|\partial_{xx}g\|_1 + \|\partial_{yy}g\|_1 + \gamma^2\|\partial_{zz}g\|_1 + 2\|\partial_{xy}g\|_1 + 2\gamma\|\partial_{xz}g\|_1 + 2\gamma\|\partial_{yz}g\|_1$$

**Stage 2 (Richardson-Lucy Deconvolution):**
$$\max_{x \geq 0} \sum_{i,j} \left[ y_{ij} \log((H * x)_{ij}) - (H * x)_{ij} \right]$$

where $H$ is the PSF convolution operator.

### data_fidelity_term

**Stage 1:**
$$\mathcal{D}_1(g) = \frac{\lambda_f}{\mu}\|f - g\|_2^2 = \frac{\lambda_f}{\mu}\sum_{i,j,k}(f_{ijk} - g_{ijk})^2$$

**Stage 2 (Richardson-Lucy):**
$$\mathcal{D}_2(x) = \sum_{i,j}\left[-y_{ij}\log\left(\max((H*x)_{ij}, 10^{-6})\right) + (H*x)_{ij}\right]$$

### regularization_term

**Sparsity (L1) regularization:**
$$R_{\text{sparse}}(g) = \lambda_s^2 \|g\|_1 = \lambda_s^2 \sum_{i,j,k}|g_{ijk}|$$

**Hessian regularization components:**
$$R_{xx}(g) = \|\partial_{xx}g\|_1, \quad R_{yy}(g) = \|\partial_{yy}g\|_1, \quad R_{zz}(g) = \gamma^2\|\partial_{zz}g\|_1$$
$$R_{xy}(g) = 2\|\partial_{xy}g\|_1, \quad R_{xz}(g) = 2\gamma\|\partial_{xz}g\|_1, \quad R_{yz}(g) = 2\gamma\|\partial_{yz}g\|_1$$

### constraints

$$g \geq 0 \quad \text{(enforced via projection: } g \leftarrow \max(g, 0)\text{)}$$
$$x \geq 10^{-6} \quad \text{(enforced via projection in deconvolution)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g$ | $(N_z, N_x, N_y)$ | float32 | $g^{(0)} = f$ | Reconstructed image (sparse Hessian stage) |
| $x_k$ | $(N_x+2B, N_y+2B)$ | float32 | $x_k^{(0)} = y_{\text{padded}}$ | Current estimate (deconvolution stage) |
| $y_k$ | $(N_x+2B, N_y+2B)$ | float32 | $y_k^{(0)} = y_{\text{padded}}$ | Accelerated iterate (deconvolution stage) |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $b_{xx}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xx}$ |
| $b_{yy}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{yy}$ |
| $b_{zz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{zz}$ |
| $b_{xy}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xy}$ |
| $b_{xz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{xz}$ |
| $b_{yz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\partial_{yz}$ |
| $b_{\ell_1}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for sparsity |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g_{\text{update}}$ | $(N_z, N_x, N_y)$ | float32/complex128 | $\frac{\lambda_f}{\mu}f$ | Accumulated update in Fourier domain |
| $d_{xx}, d_{yy}, \ldots$ | $(N_z, N_x, N_y)$ | float32 | Computed per iteration | Soft-thresholded Hessian components |
| $v_k$ | $(N_x+2B, N_y+2B)$ | float32 | $\mathbf{0}$ | Velocity for RL acceleration |
| $\alpha$ | scalar | float32 | $0$ | Momentum coefficient (RL) |
| $\gamma_1, \gamma_2$ | scalar | float32 | $1$ | Acceleration parameters (Landweber) |
| $\text{residual}_{\text{prev}}$ | scalar | float64 | $+\infty$ | Previous residual for convergence |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $\lambda_f$ | scalar | int | $150$ | Fidelity parameter |
| $\lambda_s$ | scalar | int | $10$ | Sparsity parameter |
| $\gamma$ | scalar | float32 | $\sqrt{0.5} \approx 0.707$ | Z-continuity weight ($\sqrt{\text{tcontinuity}}$) |
| $\mu$ | scalar | float32 | $1.0$ | ADMM penalty parameter |
| $K_{\text{sparse}}$ | scalar | int | $1000$ | Max sparse iterations |
| $K_{\text{deconv}}$ | scalar | int | $7$ | Max deconvolution iterations |
| $\tau$ | scalar | float64 | $10^{-4}$ | Convergence tolerance |
| $\epsilon$ | scalar | float32 | $10^{-6}$ | Numerical floor |
| $\sigma_{\text{PSF}}$ | scalar | float32 | $280/65 \approx 4.31$ | PSF sigma |
| $\Delta_{xx}$ | $(1,1,3)$ | float32 | $[[[1,-2,1]]]$ | Second derivative kernel (x) |
| $\Delta_{yy}$ | $(1,3,1)$ | float32 | $[[[1],[-2],[1]]]$ | Second derivative kernel (y) |
| $\Delta_{zz}$ | $(3,1,1)$ | float32 | $[[[1]],[[-2]],[[1]]]$ | Second derivative kernel (z) |
| $\Delta_{xy}$ | $(1,2,2)$ | float32 | $[[[1,-1],[-1,1]]]$ | Mixed derivative kernel (xy) |
| $\Delta_{xz}$ | $(2,1,2)$ | float32 | $[[[1,-1]],[[-1,1]]]$ | Mixed derivative kernel (xz) |
| $\Delta_{yz}$ | $(2,2,1)$ | float32 | $[[[1],[-1]],[[-1],[1]]]$ | Mixed derivative kernel (yz) |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $I_{\text{raw}}$ | $(N_z, N_x, N_y)$ or $(N_x, N_y)$ | uint8/uint16 | Raw input image from file |
| $f$ | $(N_z, N_x, N_y)$ | float32 | Preprocessed normalized observation |
| $s$ | scalar | float32 | Original maximum intensity (scaler) |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| P1 | Load and normalize | $I_{\text{norm}} = \frac{I_{\text{raw}}}{\max(I_{\text{raw}})}$, $s = \max(I_{\text{raw}})$ | $I_{\text{raw}}$: $(N_x, N_y)$ | $I_{\text{norm}}$: $(N_x, N_y)$, $s$: scalar |
| P2 | Background estimation | For each slice: $B = \text{waverec2}(\text{LF}(\text{wavedec2}(I_{\text{norm}}/2.5, \text{db6}, 7)))$ iterated 3 times with $\text{res}[I_{\text{norm}} > B + \frac{\sqrt{|res|}}{2}] \leftarrow B + \frac{\sqrt{|res|}}{2}$ | $I_{\text{norm}}$: $(N_x, N_y)$ | $B$: $(N_x, N_y)$ |
| P3 | Background subtraction | $I_{\text{sub}} = I_{\text{norm}} - B$ | $I_{\text{norm}}$, $B$: $(N_x, N_y)$ | $I_{\text{sub}}$: $(N_x, N_y)$ |
| P4 | Final normalization and clipping | $f = \max\left(\frac{I_{\text{sub}}}{\max(I_{\text{sub}})}, 0\right)$ | $I_{\text{sub}}$: $(N_x, N_y)$ | $f$: $(N_x, N_y)$ |
| P5 | Dimension expansion (if 2D) | If $\text{ndim}(f)=2$: $f_{\text{3D}}[i,:,:] = f$ for $i \in \{0,1,2\}$, set $\gamma=0$ | $f$: $(N_x, N_y)$ | $f_{\text{3D}}$: $(3, N_x, N_y)$ |
| P6 | Compute Hessian power spectra | $\hat{\Phi}_{xx} = |\mathcal{F}(\Delta_{xx})|^2$, similarly for $yy, zz, xy, xz, yz$ | $\Delta_{xx}$, imgsize | $\hat{\Phi}_{xx}, \hat{\Phi}_{yy}, \ldots$: $(N_z, N_x, N_y)$ |
| P7 | Build normalizer | $\mathcal{N} = \frac{\lambda_f}{\mu} + \lambda_s^2 + \hat{\Phi}_{xx} + \hat{\Phi}_{yy} + \gamma^2\hat{\Phi}_{zz} + 2\hat{\Phi}_{xy} + 2\gamma\hat{\Phi}_{xz} + 2\gamma\hat{\Phi}_{yz}$ | All $\hat{\Phi}$, $\lambda_f$, $\lambda_s$, $\gamma$, $\mu$ | $\mathcal{N}$: $(N_z, N_x, N_y)$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize dual variables | $b_{xx}^{(0)} = b_{yy}^{(0)} = b_{zz}^{(0)} = b_{xy}^{(0)} = b_{xz}^{(0)} = b_{yz}^{(0)} = b_{\ell_1}^{(0)} = \mathbf{0}_{N_z \times N_x \times N_y}$ | imgsize | All $b^{(0)}$ | Zero arrays allocation |
| 2 | Initialize update accumulator | $g_{\text{update}}^{(0)} = \frac{\lambda_f}{\mu} \cdot f$ | $f$, $\lambda_f$, $\mu$ | $g_{\text{update}}^{(0)}$: $(N_z, N_x, N_y)$ | Element-wise multiply |
| 3 | Initialize convergence tracking | $r_{\text{prev}} = +\infty$ | None | $r_{\text{prev}}$: scalar | |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (Sparse Hessian): $k = 0, 1, \ldots, K_{\text{sparse}}-1$

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 4 | FFT of update | $\hat{g}_{\text{update}}^{(k)} = \mathcal{F}(g_{\text{update}}^{(k)})$ | $g_{\text{update}}^{(k)}$ | $\hat{g}_{\text{update}}^{(k)}$: complex | [FORWARD] 3D FFT |
| 5 | Primal update | If $k=0$: $g^{(k)} = \mathcal{F}^{-1}\left(\frac{\hat{g}_{\text{update}}^{(k)}}{\lambda_f/\mu}\right)$; else: $g^{(k)} = \mathcal{F}^{-1}\left(\frac{\hat{g}_{\text{update}}^{(k)}}{\mathcal{N}}\right)$ | $\hat{g}_{\text{update}}^{(k)}$, $\mathcal{N}$ | $g^{(k)}$: $(N_z, N_x, N_y)$ real | [ADJOINT] Element-wise division then IFFT |
| 6 | Reset update | $g_{\text{update}}^{(k+1)} = \frac{\lambda_f}{\mu} \cdot f$ | $f$, $\lambda_f$, $\mu$ | $g_{\text{update}}^{(k+1)}$ | Base fidelity term |
| 7 | Compute $\partial_{xx}g$ | $g_{xx} = \nabla^-_x(\nabla^+_x(g^{(k)}))$ where $\nabla^+_x(g)[i,j,k] = g[i,j,k+1] - g[i,j,k]$, $\nabla^-_x(g)[i,j,k] = g[i,j,k-1] - g[i,j,k]$ | $g^{(k)}$ | $g_{xx}$: $(N_z, N_x, N_y)$ | Backward-forward difference |
| 8 | Soft threshold xx | $d_{xx}^{(k)} = \text{sign}(g_{xx} + b_{xx}^{(k)}) \cdot \max(|g_{xx} + b_{xx}^{(k)}| - \frac{1}{\mu}, 0)$ | $g_{xx}$, $b_{xx}^{(k)}$, $\mu$ | $d_{xx}^{(k)}$ | Proximal of $\|\cdot\|_1$ |
| 9 | Update dual xx | $b_{xx}^{(k+1)} = b_{xx}^{(k)} + g_{xx} - d_{xx}^{(k)}$ | $b_{xx}^{(k)}$, $g_{xx}$, $d_{xx}^{(k)}$ | $b_{xx}^{(k+1)}$ | Bregman update |
| 10 | Compute Lxx contribution | $L_{xx} = 1 \cdot \nabla^-_x(\nabla^+_x(d_{xx}^{(k)} - b_{xx}^{(k+1)}))$ | $d_{xx}^{(k)}$, $b_{xx}^{(k+1)}$ | $L_{xx}$ | Adjoint of Hessian |
| 11 | Accumulate Lxx | $g_{\text{update}}^{(k+1)} \leftarrow g_{\text{update}}^{(k+1)} + L_{xx}$ | $g_{\text{update}}^{(k+1)}$, $L_{xx}$ | $g_{\text{update}}^{(k+1)}$ | |
| 12 | Compute $\partial_{yy}g$ | $g_{yy} = \nabla^-_y(\nabla^+_y(g^{(k)}))$ | $g^{(k)}$ | $g_{yy}$ | |
| 13 | Soft threshold yy | $d_{yy}^{(k)} = \text{sign}(g_{yy} + b_{yy}^{(k)}) \cdot \max(|g_{yy} + b_{yy}^{(k)}| - \frac{1}{\mu}, 0)$ | $g_{yy}$, $b_{yy}^{(k)}$ | $d_{yy}^{(k)}$ | |
| 14 | Update dual yy | $b_{yy}^{(k+1)} = b_{yy}^{(k)} + g_{yy} - d_{yy}^{(k)}$ | | $b_{yy}^{(k+1)}$ | |
| 15 | Compute and accumulate Lyy | $L_{yy} = \nabla^-_y(\nabla^+_y(d_{yy}^{(k)} - b_{yy}^{(k+1)}))$; $g_{\text{update}}^{(k+1)} \leftarrow g_{\text{update}}^{(k+1)} + L_{yy}$ | | $g_{\text{update}}^{(k+1)}$ | |
| 16 | Compute $\partial_{zz}g$ | $g_{zz} = \nabla^-_z(\nabla^+_z(g^{(k)}))$ | $g^{(k)}$ | $g_{zz}$ | |
| 17 | Soft threshold zz | $d_{zz}^{(k)} = \text{sign}(g_{zz} + b_{zz}^{(k)}) \cdot \max(|g_{zz} + b_{zz}^{(k)}| - \frac{1}{\mu}, 0)$ | | $d_{zz}^{(k)}$ | |
| 18 | Update dual zz and accumulate | $b_{zz}^{(k+1)} = b_{zz}^{(k)} + g_{zz} - d_{zz}^{(k)}$; $L_{zz} = \gamma^2 \nabla^-_z(\nabla^+_z(d_{zz}^{(k)} - b_{zz}^{(k+1)}))$; $g_{\text{update}}^{(k+1)} \leftarrow g_{\text{update}}^{(k+1)} + L_{zz}$ | | | |
| 19 | Compute $\partial_{xy}g$ | $g_{xy} = \nabla^+_x(\nabla^+_y(g^{(k)}))$ | $g^{(k)}$ | $g_{xy}$ | Forward-forward |
| 20 | Soft threshold, update dual, accumulate xy | $d_{xy}^{(k)} = \mathcal{S}_{1/\mu}(g_{xy} + b_{xy}^{(k)})$; $b_{xy}^{(k+1)} = b_{xy}^{(k)} + g_{xy} - d_{xy}^{(k)}$; $L_{xy} = 2 \nabla^-_y(\nabla^-_x(d_{xy}^{(k)} - b_{xy}^{(k+1)}))$; accumulate | | | $\mathcal{S}_\tau(x) = \text{sign}(x)\max(|x|-\tau,0)$ |
| 21 | Compute $\partial_{xz}g$ | $g_{xz} = \nabla^+_x(\nabla^+_z(g^{(k)}))$ | $g^{(k)}$ | $g_{xz}$ | |
| 22 | Process xz component | $d_{xz}^{(k)} = \mathcal{S}_{1/\mu}(g_{xz} + b_{xz}^{(k)})$; $b_{xz}^{(k+1)} = b_{xz}^{(k)} + g_{xz} - d_{xz}^{(k)}$; $L_{xz} = 2\gamma \nabla^-_z(\nabla^-_x(d_{xz}^{(k)} - b_{xz}^{(k+1)}))$; accumulate | | | |
| 23 | Compute $\partial_{yz}g$ | $g_{yz} = \nabla^+_y(\nabla^+_z(g^{(k)}))$ | $g^{(k)}$ | $g_{yz}$ | |
| 24 | Process yz component | $d_{yz}^{(k)} = \mathcal{S}_{1/\mu}(g_{yz} + b_{yz}^{(k)})$; $b_{yz}^{(k+1)} = b_{yz}^{(k)} + g_{yz} - d_{yz}^{(k)}$; $L_{yz} = 2\gamma \nabla^-_z(\nabla^-_y(d_{yz}^{(k)} - b_{yz}^{(k+1)}))$; accumulate | | | |
| 25 | Process sparsity | $d_{\ell_1}^{(k)} = \mathcal{S}_{1/\mu}(g^{(k)} + b_{\ell_1}^{(k)})$; $b_{\ell_1}^{(k+1)} = b_{\ell_1}^{(k)} + g^{(k)} - d_{\ell_1}^{(k)}$; $L_{\text{sparse}} = \lambda_s^2 (d_{\ell_1}^{(k)} - b_{\ell_1}^{(k+1)})$; $g_{\text{update}}^{(k+1)} \leftarrow g_{\text{update}}^{(k+1)} + L_{\text{sparse}}$ | | | Sparsity term |
| 26 | Convergence check (every 20 iter) | If $k \mod 20 = 0$: $r^{(k)} = \|f - g^{(k)}\|_2$; if $\frac{|r^{(k)} - r_{\text{prev}}|}{r_{\text{prev}} + 10^{-12}} < \tau$: BREAK; else $r_{\text{prev}} \leftarrow r^{(k)}$ | $f$, $g^{(k)}$, $r_{\text{prev}}$, $\tau$ | $r_{\text{prev}}$ | Early stopping |

**END_LOOP** (Sparse Hessian)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 27 | Non-negativity projection | $g \leftarrow \max(g, 0)$ | $g$ | $g$ | Element-wise |
| 28 | Extract 2D result | If original 2D: $g_{\text{sparse}} = g[1,:,:]$; else $g_{\text{sparse}} = g$ | $g$ | $g_{\text{sparse}}$ | Slice middle frame |
| 29 | Normalize sparse result | $g_{\text{sparse}} \leftarrow \frac{g_{\text{sparse}}}{\max(g_{\text{sparse}})}$ | $g_{\text{sparse}}$ | $g_{\text{sparse}}$ | |
| 30 | Generate PSF kernel | $H_{ij} = \frac{1}{Z}\exp\left(-\frac{i^2}{2\sigma_x^2} - \frac{j^2}{2\sigma_y^2}\right)$ where $\sigma_x = \sigma_y = \frac{\sigma_{\text{PSF}}}{2\sqrt{2\ln 2}}$, $Z = \sum_{i,j}H_{ij}$, support size $N = 2\lceil\frac{\sigma_{\text{PSF}}}{\sqrt{8\ln 2}}\sqrt{-2\ln(0.0002)}\rceil + 3$ | $\sigma_{\text{PSF}}$ | $H$: $(N, N)$ | Gaussian PSF |
| 31 | Pad data for deconvolution | $B = \lfloor\frac{\min(N_x, N_y)}{6}\rfloor$; $y_{\text{pad}} = \text{pad}_{\text{edge}}(g_{\text{sparse}}, B)$ | $g_{\text{sparse}}$, $B$ | $y_{\text{pad}}$: $(N_x+2B, N_y+2B)$ | Replicate boundary |
| 32 | Compute OTF | $\text{OTF} = \mathcal{F}(\text{circshift}(\text{zeropad}(H, (N_x+2B, N_y+2B))))$ | $H$, padded size | OTF: complex | psf2otf |
| 33 | Initialize deconv variables | $y_k^{(0)} = y_{\text{pad}}$; $x_k^{(0)} = \mathbf{0}$; $v_k^{(0)} = \mathbf{0}$; $\alpha^{(0)} = 0$ | | | |

**BEGIN_LOOP** (Richardson-Lucy): $n = 0, 1, \ldots, K_{\text{deconv}}-1$

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 34 | Store previous estimate | $x_{k,\text{prev}}^{(n)} = x_k^{(n)}$ | $x_k^{(n)}$ | $x_{k,\text{prev}}^{(n)}$ | |
| 35 | Compute RL ratio in Fourier | $\hat{R}^{(n)} = \mathcal{F}\left(\frac{y_{\text{pad}}}{\max(\mathcal{F}^{-1}(\text{OTF} \cdot \mathcal{F}(y_k^{(n)})), 10^{-6})}\right)$ | $y_{\text{pad}}$, $y_k^{(n)}$, OTF | $\hat{R}^{(n)}$: complex | |
| 36 | RL multiplicative update | $x_k^{(n+1)} = \frac{y_k^{(n)} \cdot \text{Re}(\mathcal{F}^{-1}(\text{OTF}^* \cdot \hat{R}^{(n)}))}{\text{Re}(\mathcal{F}^{-1}(\mathcal{F}(\mathbf{1}) \cdot \text{OTF}))}$ | $y_k^{(n)}$, $\hat{R}^{(n)}$, OTF | $x_k^{(n+1)}$ | Denominator is PSF sum |
| 37 | Apply floor | $x_k^{(n+1)} \leftarrow \max(x_k^{(n+1)}, 10^{-6})$ | $x_k^