# Algorithmic Specification Document: Sparse Hessian Deconvolution

## 1. OBJECTIVE FUNCTION

### full_expression

The algorithm solves a two-stage optimization problem:

**Stage 1 (Sparse Hessian Reconstruction):**
$$\min_{g \geq 0} \frac{\lambda_f}{2\mu} \|f - g\|_2^2 + \lambda_s \|g\|_1 + \|\nabla_{xx} g\|_1 + \|\nabla_{yy} g\|_1 + \gamma^2 \|\nabla_{zz} g\|_1 + 2\|\nabla_{xy} g\|_1 + 2\gamma \|\nabla_{xz} g\|_1 + 2\gamma \|\nabla_{yz} g\|_1$$

**Stage 2 (Iterative Deconvolution):**
$$\min_{u \geq 0} D_{KL}(g_{sparse} \| H * u) \quad \text{(Richardson-Lucy)}$$
or
$$\min_{u \geq 0} \|g_{sparse} - H * u\|_2^2 \quad \text{(Landweber)}$$

### data_fidelity_term

**Stage 1:**
$$\mathcal{D}_1(g) = \frac{\lambda_f}{2\mu} \|f - g\|_2^2 = \frac{\lambda_f}{2\mu} \sum_{i,j,k} (f_{i,j,k} - g_{i,j,k})^2$$

**Stage 2 (Richardson-Lucy):**
$$\mathcal{D}_2(u) = D_{KL}(g_{sparse} \| H * u) = \sum_{i,j} \left[ (H * u)_{i,j} - g_{sparse,i,j} \log((H * u)_{i,j}) \right]$$

**Stage 2 (Landweber):**
$$\mathcal{D}_2(u) = \frac{1}{2}\|g_{sparse} - H * u\|_2^2$$

### regularization_term

**Stage 1 (Sparse Hessian):**
$$\mathcal{R}(g) = \lambda_s \|g\|_1 + \|\nabla_{xx} g\|_1 + \|\nabla_{yy} g\|_1 + \gamma^2 \|\nabla_{zz} g\|_1 + 2\|\nabla_{xy} g\|_1 + 2\gamma \|\nabla_{xz} g\|_1 + 2\gamma \|\nabla_{yz} g\|_1$$

where:
- $\nabla_{xx} g = \nabla_x^- \nabla_x^+ g$ (second derivative in x)
- $\nabla_{xy} g = \nabla_x^+ \nabla_y^+ g$ (mixed derivative)
- $\gamma = \sqrt{\tau_c}$ (z-continuity weight)

### constraints

| Constraint | Expression |
|------------|------------|
| Non-negativity (Stage 1) | $g_{i,j,k} \geq 0 \quad \forall i,j,k$ |
| Non-negativity (Stage 2) | $u_{i,j} \geq \epsilon \quad \forall i,j$ where $\epsilon = 10^{-6}$ |

---

## 2. VARIABLES

### Observations and Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $f$ | observations | $(Z, N_x, N_y)$ | float32 | Preprocessed input image | Observed/background-subtracted image |
| $\lambda_f$ | constants | scalar | int | 150 | Fidelity parameter |
| $\lambda_s$ | constants | scalar | int | 10 | Sparsity parameter |
| $\tau_c$ | constants | scalar | float | 0.5 | Temporal/z-continuity parameter |
| $\gamma$ | constants | scalar | float | $\sqrt{\tau_c}$ | Z-continuity weight |
| $\mu$ | constants | scalar | float | 1.0 | ADMM penalty parameter |
| $\sigma$ | constants | scalar or (2,) | float | $280/65$ | Gaussian PSF standard deviation |
| $H$ | constants | $(K_x, K_y)$ | float32 | Gaussian kernel | Point spread function |
| $\mathcal{F}\{H\}$ | constants | $(N_x, N_y)$ | complex64 | psf2otf(H, shape) | Optical transfer function |

### Primal Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $g$ | primal | $(Z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Sparse Hessian reconstruction |
| $g_{update}$ | primal | $(Z, N_x, N_y)$ | float32 | $\frac{\lambda_f}{\mu} f$ | Accumulated update in frequency domain |
| $y_k$ | primal | $(N_x+2B, N_y+2B)$ | float32 | Padded $g_{sparse}$ | Accelerated iterate (deconvolution) |
| $x_k$ | primal | $(N_x+2B, N_y+2B)$ | float32 | $\mathbf{0}$ | Primary iterate (deconvolution) |
| $v_k$ | primal | $(N_x+2B, N_y+2B)$ | float32 | $\mathbf{0}$ | Momentum term (RL deconvolution) |

### Dual Variables (Scaled Lagrange Multipliers)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $b_{xx}$ | dual | $(Z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Dual variable for $\nabla_{xx}$ |
| $b_{yy}$ | dual | $(Z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Dual variable for $\nabla_{yy}$ |
| $b_{zz}$ | dual | $(Z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Dual variable for $\nabla_{zz}$ |
| $b_{xy}$ | dual | $(Z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Dual variable for $\nabla_{xy}$ |
| $b_{xz}$ | dual | $(Z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Dual variable for $\nabla_{xz}$ |
| $b_{yz}$ | dual | $(Z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Dual variable for $\nabla_{yz}$ |
| $b_{l1}$ | dual | $(Z, N_x, N_y)$ | float32 | $\mathbf{0}$ | Dual variable for sparsity |

### Auxiliary Variables (Split Variables)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $d_{xx}$ | auxiliary | $(Z, N_x, N_y)$ | float32 | Computed | Split variable for $\nabla_{xx} g$ |
| $d_{yy}$ | auxiliary | $(Z, N_x, N_y)$ | float32 | Computed | Split variable for $\nabla_{yy} g$ |
| $d_{zz}$ | auxiliary | $(Z, N_x, N_y)$ | float32 | Computed | Split variable for $\nabla_{zz} g$ |
| $d_{xy}$ | auxiliary | $(Z, N_x, N_y)$ | float32 | Computed | Split variable for $\nabla_{xy} g$ |
| $d_{xz}$ | auxiliary | $(Z, N_x, N_y)$ | float32 | Computed | Split variable for $\nabla_{xz} g$ |
| $d_{yz}$ | auxiliary | $(Z, N_x, N_y)$ | float32 | Computed | Split variable for $\nabla_{yz} g$ |
| $d_{sparse}$ | auxiliary | $(Z, N_x, N_y)$ | float32 | Computed | Split variable for sparsity |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Image Loading and Type Conversion

| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load and convert image |
| formula | $I_{raw} = \text{imread}(\text{path}), \quad I = \text{float32}(I_{raw})$ |
| inputs | path: string |
| outputs | $I \in \mathbb{R}^{Z \times N_x \times N_y}$ (float32) |

### Step 3.2: Intensity Scaling

| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Normalize by maximum |
| formula | $s = \max(I), \quad I_{norm} = \frac{I}{s}$ |
| inputs | $I \in \mathbb{R}^{Z \times N_x \times N_y}$ |
| outputs | $I_{norm} \in [0,1]^{Z \times N_x \times N_y}$, $s \in \mathbb{R}^+$ (scaler) |

### Step 3.3: Background Estimation (Wavelet-Based)

| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Iterative wavelet background estimation |
| formula | For each z-slice and iteration $i = 1, \ldots, N_{iter}$: |
| | $C = \text{wavedec2}(R^{(i)}, \psi, L)$ where $\psi$ = 'db6', $L$ = 7 |
| | $\tilde{C} = \text{LowFreqResolve}(C, L)$ — zero all detail coefficients |
| | $B^{(i)} = \text{waverec2}(\tilde{C}, \psi)$ |
| | $\epsilon = \frac{1}{2}\sqrt{|R^{(i)}|}$ |
| | $R^{(i+1)}_{p,q} = \begin{cases} B^{(i)}_{p,q} + \epsilon_{p,q} & \text{if } R^{(i)}_{p,q} > B^{(i)}_{p,q} + \epsilon_{p,q} \\ R^{(i)}_{p,q} & \text{otherwise} \end{cases}$ |
| | Initialize: $R^{(1)} = I_{norm} / 2.5$ |
| inputs | $I_{norm} \in \mathbb{R}^{Z \times N_x \times N_y}$, wavelet='db6', dlevel=7, iter=3 |
| outputs | $B \in \mathbb{R}^{Z \times N_x \times N_y}$ (background estimate) |

### Step 3.4: Background Subtraction

| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Subtract background |
| formula | $I_{sub} = I_{norm} - B$ |
| inputs | $I_{norm}, B \in \mathbb{R}^{Z \times N_x \times N_y}$ |
| outputs | $I_{sub} \in \mathbb{R}^{Z \times N_x \times N_y}$ |

### Step 3.5: Re-normalization and Clipping

| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Normalize and enforce non-negativity |
| formula | $I_{clip} = \frac{I_{sub}}{\max(I_{sub})}, \quad f_{i,j,k} = \max(I_{clip,i,j,k}, 0)$ |
| inputs | $I_{sub} \in \mathbb{R}^{Z \times N_x \times N_y}$ |
| outputs | $f \in [0,1]^{Z \times N_x \times N_y}$ |

### Step 3.6: Optional Upsampling

| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Spatial or Fourier upsampling |
| formula | **Spatial** ($n=2$): $f^{up}_{2i, 2j} = f_{i,j}$, zeros elsewhere |
| | **Fourier**: Zero-padding in frequency domain with scaling $\frac{n^2}{N_x N_y}$ |
| inputs | $f \in \mathbb{R}^{Z \times N_x \times N_y}$, upsample_type $\in \{0,1,2\}$ |
| outputs | $f \in \mathbb{R}^{Z \times nN_x \times nN_y}$ (if upsampling enabled) |

### Step 3.7: Dimensional Padding for 2D Input

| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Ensure 3D structure |
| formula | If $\dim(f) = 2$: $f_{3D} \in \mathbb{R}^{3 \times N_x \times N_y}$ with $f_{3D}[z,:,:] = f \; \forall z \in \{0,1,2\}$, set $\gamma = 0$ |
| inputs | $f \in \mathbb{R}^{N_x \times N_y}$ or $\mathbb{R}^{Z \times N_x \times N_y}$ |
| outputs | $f \in \mathbb{R}^{Z' \times N_x \times N_y}$ where $Z' \geq 3$ |

### Step 3.8: Precompute Hessian Operators in Frequency Domain

| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | Compute FFT of finite difference operators |
| formula | $\hat{\Delta}_{xx} = |\mathcal{F}\{[1, -2, 1]\}|^2$ |
| | $\hat{\Delta}_{yy} = |\mathcal{F}\{[1; -2; 1]\}|^2$ |
| | $\hat{\Delta}_{zz} = |\mathcal{F}\{\delta_{zz}\}|^2$ where $\delta_{zz}[0,0,0]=1, \delta_{zz}[1,0,0]=-2, \delta_{zz}[2,0,0]=1$ |
| | $\hat{\Delta}_{xy} = |\mathcal{F}\{[[1,-1],[-1,1]]\}|^2$ |
| | $\hat{\Delta}_{xz} = |\mathcal{F}\{\delta_{xz}\}|^2$, $\hat{\Delta}_{yz} = |\mathcal{F}\{\delta_{yz}\}|^2$ |
| | $\hat{\Lambda} = \hat{\Delta}_{xx} + \hat{\Delta}_{yy} + \gamma^2 \hat{\Delta}_{zz} + 2\hat{\Delta}_{xy} + 2\gamma\hat{\Delta}_{xz} + 2\gamma\hat{\Delta}_{yz}$ |
| inputs | imgsize = $(Z, N_x, N_y)$ |
| outputs | $\hat{\Lambda} \in \mathbb{C}^{Z \times N_x \times N_y}$ |

### Step 3.9: Compute Normalization Denominator

| Field | Value |
|-------|-------|
| step_order | 9 |
| step_name | Precompute denominator for g-update |
| formula | $\mathcal{N} = \frac{\lambda_f}{\mu} + \lambda_s^2 + \hat{\Lambda}$ |
| inputs | $\lambda_f, \mu, \lambda_s, \hat{\Lambda}$ |
| outputs | $\mathcal{N} \in \mathbb{C}^{Z \times N_x \times N_y}$ |

---

## 4. ALGORITHM STEPS

### Stage 1: Sparse Hessian Reconstruction (ADMM/Split Bregman)

#### Initialization

| Step | Field | Value |
|------|-------|-------|
| I.1 | step_name | Initialize dual variables |
| | formula | $b_{xx}^{(0)} = b_{yy}^{(0)} = b_{zz}^{(0)} = b_{xy}^{(0)} = b_{xz}^{(0)} = b_{yz}^{(0)} = b_{l1}^{(0)} = \mathbf{0} \in \mathbb{R}^{Z \times N_x \times N_y}$ |
| | inputs | imgsize = $(Z, N_x, N_y)$ |
| | outputs | All dual variables initialized to zero |
| I.2 | step_name | Initialize update accumulator |
| | formula | $g_{update}^{(0)} = \frac{\lambda_f}{\mu} f$ |
| | inputs | $f, \lambda_f, \mu$ |
| | outputs | $g_{update}^{(0)} \in \mathbb{R}^{Z \times N_x \times N_y}$ |
| I.3 | step_name | Initialize convergence tracking |
| | formula | $r^{(-1)} = +\infty, \quad \tau = 10^{-4}$ |
| | inputs | None |
| | outputs | $r^{(-1)}$ (previous residual), $\tau$ (tolerance) |

---

#### Iteration Loop

**BEGIN ITERATION LOOP** ($k = 0, 1, \ldots, K_{sparse}-1$ where $K_{sparse}$ = sparse_iter)

---

| Step | Field | Value |
|------|-------|-------|
| L.1 | step_name | FFT of update accumulator |
| | formula | $\hat{g}_{update}^{(k)} = \mathcal{F}\{g_{update}^{(k)}\}$ |
| | inputs | $g_{update}^{(k)} \in \mathbb{R}^{Z \times N_x \times N_y}$ |
| | outputs | $\hat{g}_{update}^{(k)} \in \mathbb{C}^{Z \times N_x \times N_y}$ |
| | computational_note | 3D FFT over all dimensions |

---

| Step | Field | Value |
|------|-------|-------|
| L.2 | step_name | Primal update (g-subproblem) |
| | formula | $g^{(k+1)} = \begin{cases} \mathcal{F}^{-1}\left\{ \frac{\hat{g}_{update}^{(k)}}{\lambda_f/\mu} \right\} & k = 0 \\ \mathcal{F}^{-1}\left\{ \frac{\hat{g}_{update}^{(k)}}{\mathcal{N}} \right\} & k > 0 \end{cases}$ |
| | inputs | $\hat{g}_{update}^{(k)}, \mathcal{N}, \lambda_f, \mu, k$ |
| | outputs | $g^{(k+1)} \in \mathbb{R}^{Z \times N_x \times N_y}$ (real part taken) |
| | computational_note | Element-wise division in frequency domain, then inverse FFT |

---

| Step | Field | Value |
|------|-------|-------|
| L.3 | step_name | Reset update accumulator with fidelity term |
| | formula | $g_{update}^{(k+1)} = \frac{\lambda_f}{\mu} f$ |
| | inputs | $f, \lambda_f, \mu$ |
| | outputs | $g_{update}^{(k+1)} \in \mathbb{R}^{Z \times N_x \times N_y}$ |

---

| Step | Field | Value |
|------|-------|-------|
| L.4a | step_name | Compute second derivative $\nabla_{xx}$ |
| | formula | $(\nabla_{xx} g)_{z,i,j} = g_{z,i+1,j} - 2g_{z,i,j} + g_{z,i-1,j}$ |
| | | Implemented as: $\nabla_{xx} g = \nabla_x^- (\nabla_x^+ g)$ |
| | | $(\nabla_x^+ g)_{z,i,j} = g_{z,i+1,j} - g_{z,i,j}$ (forward difference, negated) |
| | | $(\nabla_x^- g)_{z,i,j} = g_{z,i,j} - g_{z,i-1,j}$ (backward difference) |
| | inputs | $g^{(k+1)}$ |
| | outputs | $g_{xx} \in \mathbb{R}^{Z \times N_x \times N_y}$ |

| Step | Field | Value |
|------|-------|-------|
| L.4b | step_name | Soft thresholding for $d_{xx}$ (d-subproblem) |
| | formula | $d_{xx}^{(k+1)} = \mathcal{S}_{\mu^{-1}}(g_{xx} + b_{xx}^{(k)})$ |
| | | where $\mathcal{S}_\lambda(x) = \text{sign}(x) \cdot \max(|x| - \lambda, 0)$ |
| | inputs | $g_{xx}, b_{xx}^{(k)}, \mu$ |
| | outputs | $d_{xx}^{(k+1)} \in \mathbb{R}^{Z \times N_x \times N_y}$ |

| Step | Field | Value |
|------|-------|-------|
| L.4c | step_name | Dual update for $b_{xx}$ |
| | formula | $b_{xx}^{(k+1)} = b_{xx}^{(k)} + g_{xx} - d_{xx}^{(k+1)}$ |
| | inputs | $b_{xx}^{(k)}, g_{xx}, d_{xx}^{(k+1)}$ |
| | outputs | $b_{xx}^{(k+1)} \in \mathbb{R}^{Z \times N_x \times N_y}$ |

| Step | Field | Value |
|------|-------|-------|
| L.4d | step_name | Compute Hessian xx contribution to update |
| | formula | $L_{xx} = 1 \cdot \nabla_x^- \nabla_x^+ (d_{xx}^{(k+1)} - b_{xx}^{(k+1)})$ |
| | | $g_{update}^{(k+1)} \leftarrow g_{update}^{(k+1)} + L_{xx}$ |
| | inputs | $d_{xx}^{(k+1)}, b_{xx}^{(k+1)}, g_{update}^{(k+1)}$ |
| | outputs | Updated $g_{update}^{(k+1)}$ |

---

| Step | Field | Value |
|------|-------|-------|
| L.5 | step_name | Process $\nabla_{yy}$ term |
| | formula | $g_{yy} = \nabla_y^- \nabla_y^+ g^{(k+1)}$ |
| | | $d_{yy}^{(k+1)} = \mathcal{S}_{\mu^{-1}}(g_{yy} + b_{yy}^{(k)})$ |
| | | $b_{yy}^{(k+1)} = b_{yy}^{(k)} + g_{yy} - d_{yy}^{(k+1)}$ |
| | | $L_{yy} = 1 \cdot \nabla_y^- \nabla_y^+ (d_{yy}^{(k+1)} - b_{yy}^{(k+1)})$ |
| | | $g_{update}^{(k+1)} \leftarrow g_{update}^{(k+1)} + L_{yy}$ |
| | inputs | $g^{(k+1)}, b_{yy}^{(k)}, g_{update}^{(k+1)}, \mu$ |
| | outputs | $b_{yy}^{(k+1)}$, updated $g_{update}^{(k+1)}$ |

---

| Step | Field | Value |
|------|-------|-------|
| L.6 | step_name | Process $\nabla_{zz}$ term |
| | formula | $g_{zz} = \nabla_z^- \nabla_z^+ g^{(k+1)}$ |
| | | $d_{zz}^{(k+1)} = \mathcal{S}_{\mu^{-1}}(g_{zz} + b_{zz}^{(k)})$ |
| | | $b_{zz}^{(k+1)} = b_{zz}^{(k)} + g_{zz} - d_{zz}^{(k+1)}$ |
| | | $L_{zz} = \gamma^2 \cdot \nabla_z^- \nabla_z^+ (d_{zz}^{(k+1)} - b_{zz}^{(k+1)})$ |
| | | $g_{update}^{(k+1)} \leftarrow g_{update}^{(k+1)} + L_{zz}$ |
| | inputs | $g^{(k+1)}, b_{zz}^{(k)}, g_{update}^{(k+1)}, \mu, \gamma$ |
| | outputs | $b_{zz}^{(k+1)}$, updated $g_{update}^{(k+1)}$ |

---

| Step | Field | Value |
|------|-------|-------|
| L.7 | step_name | Process $\nabla_{xy}$ term |
| | formula | $g_{xy} = \nabla_x^+ \nabla_y^+ g^{(k+1)}$ |
| | | $d_{xy}^{(k+1)} = \mathcal{S}_{\mu^{-1}}(g_{xy} + b_{xy}^{(k)})$ |
| | | $b_{xy}^{(k+1)} = b_{xy}^{(k)} + g_{xy} - d_{xy}^{(k+1)}$ |
| | | $L_{xy} = 2 \cdot \nabla_y^- \nabla_x^- (d_{xy}^{(k+1)} - b_{xy}^{(k+1)})$ |
| | | $g_{update}^{(k+1)} \leftarrow g_{update}^{(k+1)} + L_{xy}$ |
| | inputs | $g^{(k+1)}, b_{xy}^{(k)}, g_{update}^{(k+1)}, \mu$ |
| | outputs | $b_{xy}^{(k+1)}$, updated $g_{update}^{(k+1)}$ |

---

| Step | Field | Value |
|------|-------|-------|
| L.8 | step_name | Process $\nabla_{xz}$ term |
| | formula | $g_{xz} = \nabla_x^+ \nabla_z^+ g^{(k+1)}$ |
| | | $d_{xz}^{(k+1)} = \mathcal{S}_{\mu^{-1}}(g_{xz} + b_{xz}^{(k)})$ |
| | | $b_{xz}^{(k+1)} = b_{xz}^{(k)} + g_{xz} - d_{xz}^{(k+1)}$ |
| | | $L_{xz} = 2\gamma \cdot \nabla_z^- \nabla_x^- (d_{xz}^{(k+1)} - b_{xz}^{(k+1)})$ |
| | | $g_{update}^{(k+1)} \leftarrow g_{update}^{(k+1)} + L_{xz}$ |
| | inputs | $g^{(k+1)}, b_{xz}^{(k)}, g_{update}^{(k+1)}, \mu, \gamma$ |
| | outputs | $b_{xz}^{(k+1)}$, updated $g_{update}^{(k+1)}$ |

---

| Step | Field | Value |
|------|-------|-------|
| L.9 | step_name | Process $\nabla_{yz}$ term |
| | formula | $g_{yz} = \nabla_y^+ \nabla_z^+ g^{(k+1)}$ |
| | | $d_{yz}^{(k+1)} = \mathcal{S}_{\mu^{-1