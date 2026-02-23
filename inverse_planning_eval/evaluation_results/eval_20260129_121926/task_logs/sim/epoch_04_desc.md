## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Sparse Hessian Deconvolution with Iterative Refinement (SHD-IR)
- **algorithm_family**: Split Bregman / ADMM with Richardson-Lucy or Landweber post-processing
- **summary**: A two-stage image reconstruction algorithm that first applies sparse Hessian regularization via split Bregman iterations in Fourier domain, then refines with accelerated iterative deconvolution.

---

## 1. OBJECTIVE FUNCTION

### **full_expression**

**Stage 1 (Sparse Hessian):**
$$\min_{g \geq 0} \; \frac{\lambda_f}{\mu} \|f - g\|_2^2 + \lambda_s \|g\|_1 + \|\nabla_{xx} g\|_1 + \|\nabla_{yy} g\|_1 + \gamma^2 \|\nabla_{zz} g\|_1 + 2\|\nabla_{xy} g\|_1 + 2\gamma \|\nabla_{xz} g\|_1 + 2\gamma \|\nabla_{yz} g\|_1 + \iota_{\geq 0}(g)$$

**Stage 2 (Richardson-Lucy Deconvolution):**
$$\max_{x \geq 0} \; \sum_{i} \left[ y_i \log(Hx)_i - (Hx)_i \right]$$

**Stage 2 (Landweber Deconvolution, alternative):**
$$\min_{x \geq 0} \; \frac{1}{2}\|y - Hx\|_2^2$$

### **data_fidelity_term**

**Stage 1:**
$$\mathcal{D}_1(g; f) = \frac{\lambda_f}{\mu} \|f - g\|_2^2 = \frac{\lambda_f}{\mu} \sum_{i,j,k} (f_{i,j,k} - g_{i,j,k})^2$$

**Stage 2 (RL - Poisson negative log-likelihood):**
$$\mathcal{D}_2^{RL}(x; y) = -\sum_{i} \left[ y_i \log(Hx)_i - (Hx)_i \right]$$

**Stage 2 (Landweber - L2):**
$$\mathcal{D}_2^{LW}(x; y) = \frac{1}{2}\|y - Hx\|_2^2$$

### **regularization_term**

**Hessian Frobenius norm (split into components):**
$$\mathcal{R}_{xx}(g) = \|\nabla_{xx} g\|_1 = \sum_{i,j,k} |\nabla_x^- \nabla_x^+ g_{i,j,k}|$$

$$\mathcal{R}_{yy}(g) = \|\nabla_{yy} g\|_1 = \sum_{i,j,k} |\nabla_y^- \nabla_y^+ g_{i,j,k}|$$

$$\mathcal{R}_{zz}(g) = \gamma^2 \|\nabla_{zz} g\|_1 = \gamma^2 \sum_{i,j,k} |\nabla_z^- \nabla_z^+ g_{i,j,k}|$$

$$\mathcal{R}_{xy}(g) = 2\|\nabla_{xy} g\|_1 = 2\sum_{i,j,k} |\nabla_x^+ \nabla_y^+ g_{i,j,k}|$$

$$\mathcal{R}_{xz}(g) = 2\gamma \|\nabla_{xz} g\|_1 = 2\gamma \sum_{i,j,k} |\nabla_x^+ \nabla_z^+ g_{i,j,k}|$$

$$\mathcal{R}_{yz}(g) = 2\gamma \|\nabla_{yz} g\|_1 = 2\gamma \sum_{i,j,k} |\nabla_y^+ \nabla_z^+ g_{i,j,k}|$$

**Sparsity regularization:**
$$\mathcal{R}_{sparse}(g) = \lambda_s \|g\|_1 = \lambda_s \sum_{i,j,k} |g_{i,j,k}|$$

### **constraints**

$$\iota_{\geq 0}(g) = \begin{cases} 0 & \text{if } g_{i,j,k} \geq 0 \; \forall i,j,k \\ +\infty & \text{otherwise} \end{cases}$$

$$\iota_{\geq \epsilon}(x) = \begin{cases} 0 & \text{if } x_{i,j} \geq \epsilon \; \forall i,j \\ +\infty & \text{otherwise} \end{cases}$$

where $\epsilon = 10^{-6}$ (enforced via `np.maximum`)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g$ | $(N_z, N_x, N_y)$ | float32 | $\mathcal{F}^{-1}\left(\frac{\hat{g}_{update}}{\lambda_f/\mu}\right)$ at iter 0 | Reconstructed image (sparse Hessian stage) |
| $x_k$ | $(N_x + 2B, N_y + 2B)$ | float32 | $y$ (padded data) | Primal estimate (deconvolution stage) |
| $y_k$ | $(N_x + 2B, N_y + 2B)$ | float32 | $y$ (padded data) | Extrapolated primal (deconvolution stage) |

### 2.2 Dual Variables (Bregman variables)

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $b_{xx}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\nabla_{xx}$ |
| $b_{yy}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\nabla_{yy}$ |
| $b_{zz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\nabla_{zz}$ |
| $b_{xy}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\nabla_{xy}$ |
| $b_{xz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\nabla_{xz}$ |
| $b_{yz}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for $\nabla_{yz}$ |
| $b_{l1}$ | $(N_z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Bregman variable for sparsity |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g_{update}$ | $(N_z, N_x, N_y)$ | float32 | $(\lambda_f/\mu) \cdot f$ | Accumulated update for $g$ |
| $d_{xx}, d_{yy}, \ldots$ | $(N_z, N_x, N_y)$ | float32 | Computed each iteration | Shrinkage outputs (auxiliary splitting variables) |
| $v_k$ | $(N_x + 2B, N_y + 2B)$ | float32 | $\mathbf{0}$ | Velocity term for RL acceleration |
| $v_{k-1}$ | $(N_x + 2B, N_y + 2B)$ | float32 | $\mathbf{0}$ | Previous velocity |
| $x_{k-1}$ | $(N_x + 2B, N_y + 2B)$ | float32 | $y$ | Previous primal (for momentum) |
| $y_{k-1}$ | $(N_x + 2B, N_y + 2B)$ | float32 | N/A | Previous extrapolated (Landweber) |
| $\gamma_1$ | scalar | float32 | $1$ | FISTA parameter (Landweber) |
| $\text{residual}_{prev}$ | scalar | float64 | $+\infty$ | Previous residual for convergence check |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda_f$ | 150 | Fidelity parameter |
| $\lambda_s$ | 10 | Sparsity parameter |
| $\gamma$ | $\sqrt{0.5} \approx 0.707$ | Z-continuity weight ($\sqrt{\text{tcontinuity}}$) |
| $\mu$ | 1 | Split Bregman penalty parameter |
| $K_{sparse}$ | 1000 | Maximum sparse iterations |
| $K_{deconv}$ | 7 | Maximum deconvolution iterations |
| $\sigma_{PSF}$ | $280/65 \approx 4.308$ | Gaussian PSF sigma |
| $\epsilon$ | $10^{-6}$ | Numerical floor for positivity |
| $\text{tol}$ | $10^{-4}$ | Relative convergence tolerance |
| $t$ | 1 | Landweber step size |
| $B$ | $\lfloor \min(N_x, N_y)/6 \rfloor$ | Padding size for deconvolution |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y_{raw}$ | $(N_z, N_x, N_y)$ or $(N_x, N_y)$ | float32 | Raw input image |
| $f$ | $(N_z, N_x, N_y)$ | float32 | Preprocessed observation (background-subtracted, normalized) |
| $\text{scaler}$ | scalar | float32 | Original maximum intensity for rescaling |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Load Image | $y_{raw} \leftarrow \text{imread}(\text{path})$ | path: str | $y_{raw}$: $(N_z, N_x, N_y)$ |
| 1 | Extract Scaler | $\text{scaler} \leftarrow \max(y_{raw})$ | $y_{raw}$: $(N_z, N_x, N_y)$ | scaler: scalar |
| 2 | Normalize | $y_{norm} \leftarrow y_{raw} / \text{scaler}$ | $y_{raw}$, scaler | $y_{norm}$: $(N_z, N_x, N_y)$ |
| 3 | Background Estimation | $B_{bg} \leftarrow \text{WaveletLowPass}(y_{norm}/2.5)$ | $y_{norm}$: $(N_z, N_x, N_y)$ | $B_{bg}$: $(N_z, N_x, N_y)$ |
| 4 | Background Subtraction | $y_{sub} \leftarrow y_{norm} - B_{bg}$ | $y_{norm}$, $B_{bg}$ | $y_{sub}$: $(N_z, N_x, N_y)$ |
| 5 | Re-normalize | $y_{sub} \leftarrow y_{sub} / \max(y_{sub})$ | $y_{sub}$ | $y_{sub}$: $(N_z, N_x, N_y)$ |
| 6 | Clip Negatives | $f_{i,j,k} \leftarrow \max(y_{sub,i,j,k}, 0)$ | $y_{sub}$ | $f$: $(N_z, N_x, N_y)$ |
| 7 | Optional Upsample | $f \leftarrow \text{Upsample}(f, \text{method})$ | $f$, method $\in \{0,1,2\}$ | $f$: potentially upsampled |
| 8 | Final Normalize | $f \leftarrow f / \max(f)$ | $f$ | $f$: $(N_z, N_x, N_y)$ normalized $\in [0,1]$ |

**Background Estimation Subroutine (Step 3 expanded):**

For each z-slice, for `iter` $\in \{1,2,3\}$:
1. Compute wavelet decomposition: $\text{coeffs} \leftarrow \text{wavedec2}(I, \text{'db6'}, \text{level}=7)$
2. Zero high-frequency coefficients: $\text{coeffs}[1:] \leftarrow \mathbf{0}$
3. Reconstruct: $B \leftarrow \text{waverec2}(\text{coeffs})$
4. Update mask: $\epsilon \leftarrow \sqrt{|I|}/2$; where $I > B + \epsilon$: $I \leftarrow B + \epsilon$
5. Repeat wavelet decomposition on updated $I$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_name | formula | inputs | outputs |
|-----------|---------|--------|---------|
| Expand 2D to 3D | If $f$ is 2D: $f^{(3D)}_{i,:,:} \leftarrow f$ for $i \in \{0,1,2\}$; set $\gamma \leftarrow 0$ | $f$: $(N_x, N_y)$ | $f$: $(3, N_x, N_y)$, $\gamma$: scalar |
| Compute Hessian Operators | $\hat{\Delta}_{xx} \leftarrow |\mathcal{F}([1,-2,1])|^2$, etc. | imgsize: $(N_z, N_x, N_y)$ | $\hat{\Delta}_{xx}, \hat{\Delta}_{yy}, \hat{\Delta}_{zz}, \hat{\Delta}_{xy}, \hat{\Delta}_{xz}, \hat{\Delta}_{yz}$: $(N_z, N_x, N_y)$ |
| Build Normalizer | $\hat{N} \leftarrow \frac{\lambda_f}{\mu} + \lambda_s^2 + \hat{\Delta}_{xx} + \hat{\Delta}_{yy} + \gamma^2 \hat{\Delta}_{zz} + 2\hat{\Delta}_{xy} + 2\gamma\hat{\Delta}_{xz} + 2\gamma\hat{\Delta}_{yz}$ | All $\hat{\Delta}$ operators | $\hat{N}$: $(N_z, N_x, N_y)$ |
| Initialize Bregman Variables | $b_{xx}, b_{yy}, b_{zz}, b_{xy}, b_{xz}, b_{yz}, b_{l1} \leftarrow \mathbf{0}$ | imgsize | All $b$: $(N_z, N_x, N_y)$ zeros |
| Initialize Update Accumulator | $g_{update} \leftarrow (\lambda_f/\mu) \cdot f$ | $f$, $\lambda_f$, $\mu$ | $g_{update}$: $(N_z, N_x, N_y)$ |
| Initialize Convergence Tracking | $\text{residual}_{prev} \leftarrow +\infty$ | None | $\text{residual}_{prev}$: scalar |

### 4.2 ITERATION LOOP

#### BEGIN_LOOP [Sparse Hessian: $k = 0, 1, \ldots, K_{sparse}-1$]

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| FFT of Update | $\hat{g}_{update} \leftarrow \mathcal{F}(g_{update})$ | $g_{update}$: $(N_z, N_x, N_y)$ | $\hat{g}_{update}$: $(N_z, N_x, N_y)$ complex | [FORWARD] FFT |
| Solve Primal (iter 0) | $g \leftarrow \mathcal{F}^{-1}\left(\frac{\hat{g}_{update}}{\lambda_f/\mu}\right)$ | $\hat{g}_{update}$, $\lambda_f$, $\mu$ | $g$: $(N_z, N_x, N_y)$ real | Only at $k=0$ |
| Solve Primal (iter > 0) | $g \leftarrow \mathcal{F}^{-1}\left(\frac{\hat{g}_{update}}{\hat{N}}\right)$ | $\hat{g}_{update}$, $\hat{N}$ | $g$: $(N_z, N_x, N_y)$ real | [ADJOINT] inverse FFT |
| Reset Update | $g_{update} \leftarrow (\lambda_f/\mu) \cdot f$ | $f$, $\lambda_f$, $\mu$ | $g_{update}$: $(N_z, N_x, N_y)$ | Data fidelity contribution |
| XX Hessian Term | See inner block below | $g$, $b_{xx}$ | $L_{xx}$, $b_{xx}$ | |
| Accumulate XX | $g_{update} \leftarrow g_{update} + L_{xx}$ | $g_{update}$, $L_{xx}$ | $g_{update}$ | |
| YY Hessian Term | Analogous to XX with $\nabla_y$ | $g$, $b_{yy}$ | $L_{yy}$, $b_{yy}$ | |
| Accumulate YY | $g_{update} \leftarrow g_{update} + L_{yy}$ | | | |
| ZZ Hessian Term | Analogous with $\nabla_z$, weight $\gamma^2$ | $g$, $b_{zz}$ | $L_{zz}$, $b_{zz}$ | |
| Accumulate ZZ | $g_{update} \leftarrow g_{update} + L_{zz}$ | | | |
| XY Hessian Term | Mixed derivative, weight $2$ | $g$, $b_{xy}$ | $L_{xy}$, $b_{xy}$ | |
| Accumulate XY | $g_{update} \leftarrow g_{update} + L_{xy}$ | | | |
| XZ Hessian Term | Mixed derivative, weight $2\gamma$ | $g$, $b_{xz}$ | $L_{xz}$, $b_{xz}$ | |
| Accumulate XZ | $g_{update} \leftarrow g_{update} + L_{xz}$ | | | |
| YZ Hessian Term | Mixed derivative, weight $2\gamma$ | $g$, $b_{yz}$ | $L_{yz}$, $b_{yz}$ | |
| Accumulate YZ | $g_{update} \leftarrow g_{update} + L_{yz}$ | | | |
| Sparsity Term | $d_s \leftarrow \mathcal{S}_{1/\mu}(g + b_{l1})$; $b_{l1} \leftarrow b_{l1} + g - d_s$; $L_s \leftarrow \lambda_s(d_s - b_{l1})$ | $g$, $b_{l1}$, $\lambda_s$, $\mu$ | $L_s$, $b_{l1}$ | Soft thresholding |
| Accumulate Sparsity | $g_{update} \leftarrow g_{update} + L_s$ | | | |
| Convergence Check | Every 20 iterations: $r \leftarrow \|f - g\|_2$; if $|r - r_{prev}|/(r_{prev} + 10^{-12}) < \text{tol}$: BREAK | $f$, $g$, $r_{prev}$, tol | BREAK or continue | |
| Update Previous Residual | $r_{prev} \leftarrow r$ | $r$ | $r_{prev}$ | |

#### END_LOOP [Sparse Hessian]

**[INNER] Hessian Sub-iteration (e.g., iter_xx):**

| step_name | formula | inputs | outputs |
|-----------|---------|--------|---------|
| Compute 2nd Derivative | $g_{xx} \leftarrow \nabla_x^- (\nabla_x^+ g)$ where $\nabla_x^+ g_{i,j,k} = g_{i,j,k+1} - g_{i,j,k}$, $\nabla_x^- g_{i,j,k} = g_{i,j,k} - g_{i,j,k-1}$ | $g$ | $g_{xx}$ |
| Soft Threshold | $d_{xx} \leftarrow \mathcal{S}_{1/\mu}(g_{xx} + b_{xx})$ where $\mathcal{S}_\tau(x) = \text{sign}(x) \cdot \max(|x| - \tau, 0)$ | $g_{xx}$, $b_{xx}$, $\mu$ | $d_{xx}$ |
| Bregman Update | $b_{xx} \leftarrow b_{xx} + g_{xx} - d_{xx}$ | $b_{xx}$, $g_{xx}$, $d_{xx}$ | $b_{xx}$ |
| Compute Contribution | $L_{xx} \leftarrow w \cdot \nabla_x^- (\nabla_x^+ (d_{xx} - b_{xx}))$ | $d_{xx}$, $b_{xx}$, weight $w$ | $L_{xx}$ |

| step_name | formula | inputs | outputs |
|-----------|---------|--------|---------|
| Positivity Projection | $g_{i,j,k} \leftarrow \max(g_{i,j,k}, 0) \; \forall i,j,k$ | $g$ | $g$ |
| Extract 2D if needed | If input was 2D: $g_{out} \leftarrow g_{1,:,:}$ | $g$: $(3, N_x, N_y)$ | $g_{out}$: $(N_x, N_y)$ |
| Normalize Sparse Output | $g_{sparse} \leftarrow g_{out} / \max(g_{out})$ | $g_{out}$ | $g_{sparse}$ |

---

#### BEGIN_LOOP [Deconvolution (Richardson-Lucy): $k = 0, 1, \ldots, K_{deconv}-1$]

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Compute OTF | $\hat{H} \leftarrow \text{psf2otf}(\text{kernel}, \text{shape})$ | kernel, data shape | $\hat{H}$: complex | Precomputed once |
| Forward Model | $\hat{y}_k \leftarrow \hat{H} \cdot \mathcal{F}(y_k)$ | $y_k$, $\hat{H}$ | $\hat{y}_k$ | [FORWARD] |
| Inverse to Spatial | $(Hy_k)_{pred} \leftarrow \mathcal{F}^{-1}(\hat{y}_k)$ | $\hat{y}_k$ | $(Hy_k)_{pred}$ | |
| Compute Ratio | $r \leftarrow y / \max((Hy_k)_{pred}, 10^{-6})$ | $y$, $(Hy_k)_{pred}$ | $r$ | Division stabilized |
| Backproject Ratio | $\hat{r} \leftarrow \mathcal{F}(r)$ | $r$ | $\hat{r}$ | |
| Adjoint Operation | $H^T r \leftarrow \mathcal{F}^{-1}(\hat{H}^* \cdot \hat{r})$ | $\hat{H}$, $\hat{r}$ | $H^T r$ | [ADJOINT] |
| Normalization Factor | $N_{RL} \leftarrow \mathcal{F}^{-1}(\mathcal{F}(\mathbf{1}) \cdot \hat{H})$ | $\hat{H}$, shape | $N_{RL}$ | PSF sum normalization |
| RL Update | $x_k \leftarrow y_k \cdot (H^T r) / N_{RL}$ | $y_k$, $H^T r$, $N_{RL}$ | $x_k$ | Multiplicative update |
| Positivity Clamp | $x_k \leftarrow \max(x_k, 10^{-6})$ | $x_k$ | $x_k$ | |
| Compute Velocity | $v_k \leftarrow \max(x_k - y_k, 10^{-6})$ | $x_k$, $y_k$ | $v_k$ | |
| Acceleration Parameter | $\alpha \leftarrow \frac{\langle v_{k-1}, v_k \rangle}{\langle v_{k-1}, v_{k-1} \rangle + 10^{-10}}$; $\alpha \leftarrow \text{clip}(\alpha, 10^{-6}, 1)$ | $v_k$, $v_{k-1}$ | $\alpha$ | For $k > 0$; $\alpha=0$ for $k=0$ |
| Extrapolate | $y_{k+1} \leftarrow x_k + \alpha (x_k - x_{k-1})$ | $x_k$, $x_{k-1}$, $\alpha$ | $y_{k+1}$ | For $k > 0$; $y_1 = x_0$ |
| Positivity Clamp | $y_{k+1} \leftarrow \max(y_{k+1}, 10^{-6})$ | $y_{k+1}$ | $y_{k+1}$ | |
| NaN Check | If $\text{isnan}(y_{k+1})$: $y_{k+1} \leftarrow 10^{-6}$ | $y_{k+1}$ | $y_{k+1}$ | |
| Store Previous | $x_{k-1} \leftarrow x_k$; $v_{k-1} \leftarrow v_k$ | $x_k$, $v_k$ | $x_{k-1}$, $v_{k-1}$ | |

#### END_LOOP [Richardson-Lucy]

---

#### BEGIN_LOOP [Deconvolution (Landweber, alternative): $k = 0, 1, \ldots, K_{deconv}-1$]

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Forward Model | $\hat{y}_{pred} \leftarrow \hat{H} \cdot \mathcal{F}(y_k)$ | $y_k$, $\hat{H}$ | $(Hy_k)_{pred}$ | [FORWARD] |
| Compute Residual | $e \leftarrow y - (Hy_k)_{pred}$ | $y$, $(Hy_k)_{pred}$ | $e$ | In Fourier: $\hat{e} = \mathcal{F}(y) - \hat{H}\mathcal{F}(y_k)$ |
| Adjoint of Residual | $H^T e \leftarrow \mathcal{F}^{-1}(\hat{H}^* \cdot \hat{e})$ | $\hat{H}$, $\hat{e}$ | $H^T e$ | [ADJOINT] |
| FISTA Parameter Update | $\gamma_2 \leftarrow \frac{1}{2}\sqrt{4\gamma_1^2 + \gamma_1^4} - \gamma_1^2$ | $\gamma_1$ | $\gamma_2$ | For $k > 0$ |
| Momentum Parameter | $\beta \leftarrow -\gamma_2(1 - 1/\gamma_1)$ | $\gamma_1$, $\gamma_2$ | $\beta$ | For $k > 0$ |
| Extrapolate | $y_k^{ext} \leftarrow x_k + \beta(x_k - x_{k-1})$ | $x_k