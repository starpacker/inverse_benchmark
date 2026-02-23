## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: ADMM-TV Lensless Image Reconstruction
- **algorithm_family**: Alternating Direction Method of Multipliers (ADMM) with Total Variation regularization
- **summary**: Reconstructs an image from lensless camera measurements by solving a convex optimization problem with data fidelity, anisotropic total variation regularization, and non-negativity constraints via variable splitting in the Fourier domain.

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{\mathbf{x}} \quad \frac{1}{2}\|\mathbf{y} - \mathbf{H}\mathbf{x}\|_2^2 + \tau \|\boldsymbol{\Psi}\mathbf{x}\|_1 + \mathcal{I}_{\geq 0}(\mathbf{x})$$

Subject to ADMM splitting constraints:
$$\mathbf{H}\mathbf{x} = \mathbf{X}, \quad \boldsymbol{\Psi}\mathbf{x} = \mathbf{U}, \quad \mathbf{x} = \mathbf{W}$$

### data_fidelity_term

$$\frac{1}{2}\|\mathbf{X} - \mathbf{y}\|_2^2$$

where $\mathbf{X}$ is the auxiliary variable constrained to equal $\mathbf{H}\mathbf{x}$, and $\mathbf{y}$ is the measured data (padded to shape $[D, H_p, W_p, C]$).

**Verification**: The X-update step uses L2 norm (quadratic penalty with soft constraint via ADMM).

### regularization_term

**TV Regularization (Anisotropic)**:
$$\tau \|\mathbf{U}\|_1 = \tau \sum_{d,i,j,c,k} |U_{d,i,j,c,k}|$$

where $\mathbf{U} \in \mathbb{R}^{D \times H_p \times W_p \times C \times 2}$ contains horizontal and vertical finite differences, and $\tau = 0.0001$ (default).

### constraints

**Non-negativity constraint**:
$$\mathcal{I}_{\geq 0}(\mathbf{x}) = \begin{cases} 0 & \text{if } x_{d,i,j,c} \geq 0 \; \forall d,i,j,c \\ +\infty & \text{otherwise} \end{cases}$$

Enforced via auxiliary variable $\mathbf{W}$ with projection $\mathbf{W} = \max(\mathbf{x} + \boldsymbol{\rho}/\mu_3, 0)$.

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}$ (`image_est`) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Reconstructed image in padded domain |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\xi}$ (`xi`) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Dual variable for constraint $\mathbf{H}\mathbf{x} = \mathbf{X}$ |
| $\boldsymbol{\eta}$ (`eta`) | $[1, D, H_p, W_p, C, 2]$ | float32 | $\mathbf{0}$ | Dual variable for constraint $\boldsymbol{\Psi}\mathbf{x} = \mathbf{U}$ |
| $\boldsymbol{\rho}$ (`rho`) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Dual variable for constraint $\mathbf{x} = \mathbf{W}$ |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}$ (`X`) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Auxiliary for data fidelity (split from $\mathbf{H}\mathbf{x}$) |
| $\mathbf{U}$ (`U`) | $[1, D, H_p, W_p, C, 2]$ | float32 | $\mathbf{0}$ | Auxiliary for TV term (split from $\boldsymbol{\Psi}\mathbf{x}$) |
| $\mathbf{W}$ (`W`) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Auxiliary for non-negativity (split from $\mathbf{x}$) |
| $\mathbf{r}_k$ (`rk`) | $[1, D, H_p, W_p, C]$ | float32 | Computed | Right-hand side for $\mathbf{x}$-subproblem |
| `Psi_out` | $[1, D, H_p, W_p, C, 2]$ | float32 | Computed | Finite difference output $\boldsymbol{\Psi}\mathbf{x}$ |
| `forward_out` | $[1, D, H_p, W_p, C]$ | float32 | Computed | Forward model output $\mathbf{H}\mathbf{x}$ |

### 2.4 Constants

| Symbol | Value | Dtype | Description |
|--------|-------|-------|-------------|
| $n_{\text{iter}}$ | 50 (default) | int | Maximum number of ADMM iterations |
| $\mu_1$ | $1 \times 10^{-6}$ | float32 | ADMM penalty for data fidelity constraint |
| $\mu_2$ | $1 \times 10^{-5}$ | float32 | ADMM penalty for TV constraint |
| $\mu_3$ | $4 \times 10^{-5}$ | float32 | ADMM penalty for non-negativity constraint |
| $\tau$ | $1 \times 10^{-4}$ | float32 | TV regularization weight |
| `downsample` | 4 | int | Spatial downsampling factor for data loading |
| `norm` | `"ortho"` | str | FFT normalization mode |
| `pad` | `True` (load) / `False` (ADMM) | bool | Padding flag for convolution |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ (`measurement`) | $[D, H, W, C]$ | float32 | Raw lensless measurement (normalized to $[0,1]$) |
| $\mathbf{y}_p$ (`data_padded`) | $[D, H_p, W_p, C]$ | float32 | Zero-padded measurement |
| $\mathbf{h}$ (`psf`) | $[D, H, W, C]$ | float32 | Point spread function (normalized to $[0,1]$) |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 0 | Data Loading | $\mathbf{y}, \mathbf{h} \leftarrow \text{load\_data}(\cdot, \text{normalize}=\text{True})$ | `psf_path`, `data_path`, `downsample=4` | $\mathbf{y}: [D, H, W, C]$, $\mathbf{h}: [D, H, W, C]$ |
| 1 | Padded Shape Computation | $H_p = \text{next\_fast\_len}(2H - 1), \; W_p = \text{next\_fast\_len}(2W - 1)$ | PSF shape $[D, H, W, C]$ | `_padded_shape`: $[D, H_p, W_p, C]$ |
| 2 | Padding Index Computation | $s_h = \lfloor(H_p - H)/2\rfloor, \; s_w = \lfloor(W_p - W)/2\rfloor$ | $H_p, W_p, H, W$ | `_start_idx`: $[s_h, s_w]$, `_end_idx`: $[s_h + H, s_w + W]$ |
| 3 | PSF Padding | $\mathbf{h}_p[s_h:e_h, s_w:e_w, :] = \mathbf{h}$, zeros elsewhere | $\mathbf{h}: [D, H, W, C]$ | $\mathbf{h}_p: [D, H_p, W_p, C]$ |
| 4 | PSF DFT | $\mathbf{H} = \mathcal{F}_{2D}\{\mathbf{h}_p\}$ with `norm="ortho"` | $\mathbf{h}_p$ | $\mathbf{H}: [D, H_p, W_p//2+1, C]$ (complex64) |
| 5 | Adjoint PSF DFT | $\mathbf{H}^* = \overline{\mathbf{H}}$ (complex conjugate) | $\mathbf{H}$ | $\mathbf{H}^*: [D, H_p, W_p//2+1, C]$ (complex64) |
| 6 | Measurement Padding | $\mathbf{y}_p[s_h:e_h, s_w:e_w, :] = \mathbf{y}$, zeros elsewhere | $\mathbf{y}: [D, H, W, C]$ | $\mathbf{y}_p: [D, H_p, W_p, C]$ |
| 7 | TV Gram Matrix | $(\boldsymbol{\Psi}^T\boldsymbol{\Psi})_{\text{freq}} = \mathcal{F}_{2D}\{\mathbf{G}\}$ where $\mathbf{G}$ is the Laplacian kernel | Shape $[D, H_p, W_p, C]$ | `PsiTPsi`: $[D, H_p, W_p//2+1, C]$ (complex) |
| 8 | X Division Matrix | $\mathbf{M}_X = \frac{1}{\mathbf{1}_{\Omega} + \mu_1}$ where $\mathbf{1}_{\Omega}$ is padded indicator | $\mu_1$, `_padded_shape` | `X_divmat`: $[D, H_p, W_p, C]$ |
| 9 | R Division Matrix | $\mathbf{M}_R = \frac{1}{\mu_1 |\mathbf{H}^* \mathbf{H}| + \mu_2 |(\boldsymbol{\Psi}^T\boldsymbol{\Psi})_{\text{freq}}| + \mu_3}$ | $\mathbf{H}, \mu_1, \mu_2, \mu_3$, `PsiTPsi` | `R_divmat`: $[D, H_p, W_p//2+1, C]$ (complex64) |

**TV Gram Kernel Definition** (for 2D case, $D=1$):
$$\mathbf{G}[0,0,0] = 4, \quad \mathbf{G}[0,0,\pm 1] = \mathbf{G}[0,\pm 1,0] = -1$$

**TV Gram Kernel Definition** (for 3D case, $D>1$):
$$\mathbf{G}[0,0,0] = 6, \quad \mathbf{G}[0,0,\pm 1] = \mathbf{G}[0,\pm 1,0] = \mathbf{G}[\pm 1,0,0] = -1$$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Outputs |
|------|------|---------|---------|
| I.1 | Initialize primal | $\mathbf{x}^{(0)} = \mathbf{0}$ | `image_est`: $[1, D, H_p, W_p, C]$ |
| I.2 | Initialize X auxiliary | $\mathbf{X}^{(0)} = \mathbf{0}$ | `X`: $[1, D, H_p, W_p, C]$ |
| I.3 | Initialize U auxiliary | $\mathbf{U}^{(0)} = \mathbf{0}$ | `U`: $[1, D, H_p, W_p, C, 2]$ |
| I.4 | Initialize W auxiliary | $\mathbf{W}^{(0)} = \mathbf{0}$ | `W`: $[1, D, H_p, W_p, C]$ |
| I.5 | Initialize dual xi | $\boldsymbol{\xi}^{(0)} = \mathbf{0}$ | `xi`: $[1, D, H_p, W_p, C]$ |
| I.6 | Initialize dual eta | $\boldsymbol{\eta}^{(0)} = \mathbf{0}$ | `eta`: $[1, D, H_p, W_p, C, 2]$ |
| I.7 | Initialize dual rho | $\boldsymbol{\rho}^{(0)} = \mathbf{0}$ | `rho`: $[1, D, H_p, W_p, C]$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, n_{\text{iter}} - 1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | Compute finite differences | $\boldsymbol{\Psi}\mathbf{x}^{(k)} = \begin{bmatrix} \mathcal{R}_{+1}^{(h)}(\mathbf{x}^{(k)}) - \mathbf{x}^{(k)} \\ \mathcal{R}_{+1}^{(w)}(\mathbf{x}^{(k)}) - \mathbf{x}^{(k)} \end{bmatrix}$ | $\mathbf{x}^{(k)}$ | `Psi_out` | $\mathcal{R}_{+1}^{(h)}$ = roll by +1 along axis $-3$ |
| L.2 | U-update (TV soft threshold) | $\mathbf{U}^{(k+1)} = \text{sign}\left(\boldsymbol{\Psi}\mathbf{x}^{(k)} + \frac{\boldsymbol{\eta}^{(k)}}{\mu_2}\right) \odot \max\left(\left|\boldsymbol{\Psi}\mathbf{x}^{(k)} + \frac{\boldsymbol{\eta}^{(k)}}{\mu_2}\right| - \frac{\tau}{\mu_2}, 0\right)$ | `Psi_out`, $\boldsymbol{\eta}^{(k)}$, $\mu_2$, $\tau$ | $\mathbf{U}^{(k+1)}$ | Proximal for $\frac{\tau}{\mu_2}\|\cdot\|_1$ |
| L.3 | Forward model [FORWARD] | $\mathbf{H}\mathbf{x}^{(k)} = \mathcal{F}^{-1}_{2D}\left\{\mathcal{F}_{2D}\{\mathbf{x}^{(k)}\} \odot \mathbf{H}\right\}$, then `ifftshift` | $\mathbf{x}^{(k)}$, $\mathbf{H}$ | `forward_out` | Circular convolution in padded domain |
| L.4 | X-update (data fidelity) | $\mathbf{X}^{(k+1)} = \mathbf{M}_X \odot \left(\boldsymbol{\xi}^{(k)} + \mu_1 \cdot \mathbf{H}\mathbf{x}^{(k)} + \mathbf{y}_p\right)$ | `forward_out`, $\boldsymbol{\xi}^{(k)}$, $\mathbf{y}_p$, $\mu_1$, $\mathbf{M}_X$ | $\mathbf{X}^{(k+1)}$ | Closed-form quadratic minimization |
| L.5 | W-update (non-negativity projection) | $\mathbf{W}^{(k+1)} = \max\left(\mathbf{x}^{(k)} + \frac{\boldsymbol{\rho}^{(k)}}{\mu_3}, 0\right)$ | $\mathbf{x}^{(k)}$, $\boldsymbol{\rho}^{(k)}$, $\mu_3$ | $\mathbf{W}^{(k+1)}$ | Proximal for $\mathcal{I}_{\geq 0}$ |
| L.6 | Compute RHS for x-update | $\mathbf{r}_k = (\mu_3 \mathbf{W}^{(k+1)} - \boldsymbol{\rho}^{(k)}) + \boldsymbol{\Psi}^T(\mu_2 \mathbf{U}^{(k+1)} - \boldsymbol{\eta}^{(k)}) + \mathbf{H}^T(\mu_1 \mathbf{X}^{(k+1)} - \boldsymbol{\xi}^{(k)})$ | All updated auxiliaries and duals | `rk` | See L.6a, L.6b below |
| L.6a | Adjoint finite differences [ADJOINT] | $\boldsymbol{\Psi}^T \mathbf{v} = \left(\mathcal{R}_{-1}^{(h)}(\mathbf{v}_0) - \mathbf{v}_0\right) + \left(\mathcal{R}_{-1}^{(w)}(\mathbf{v}_1) - \mathbf{v}_1\right)$ | $\mathbf{v} \in \mathbb{R}^{...\times 2}$ | Scalar field | $\mathcal{R}_{-1}$ = roll by $-1$ |
| L.6b | Adjoint convolution [ADJOINT] | $\mathbf{H}^T \mathbf{v} = \mathcal{F}^{-1}_{2D}\left\{\mathcal{F}_{2D}\{\mathbf{v}\} \odot \mathbf{H}^*\right\}$, then `ifftshift` | $\mathbf{v}$, $\mathbf{H}^*$ | Deconvolved field | Correlation with PSF |
| L.7 | x-update (frequency domain) | $\mathbf{x}^{(k+1)} = \mathcal{F}^{-1}_{2D}\left\{\mathbf{M}_R \odot \mathcal{F}_{2D}\{\mathbf{r}_k\}\right\}$ | `rk`, $\mathbf{M}_R$ | $\mathbf{x}^{(k+1)}$ | Solves normal equations in Fourier domain |
| L.8 | Recompute forward model | $\mathbf{H}\mathbf{x}^{(k+1)}$ (same as L.3) | $\mathbf{x}^{(k+1)}$ | `forward_out` | For dual update |
| L.9 | Recompute finite differences | $\boldsymbol{\Psi}\mathbf{x}^{(k+1)}$ (same as L.1) | $\mathbf{x}^{(k+1)}$ | `Psi_out` | For dual update |
| L.10 | Dual update xi | $\boldsymbol{\xi}^{(k+1)} = \boldsymbol{\xi}^{(k)} + \mu_1 \left(\mathbf{H}\mathbf{x}^{(k+1)} - \mathbf{X}^{(k+1)}\right)$ | `forward_out`, $\mathbf{X}^{(k+1)}$, $\boldsymbol{\xi}^{(k)}$, $\mu_1$ | $\boldsymbol{\xi}^{(k+1)}$ | Gradient ascent on dual |
| L.11 | Dual update eta | $\boldsymbol{\eta}^{(k+1)} = \boldsymbol{\eta}^{(k)} + \mu_2 \left(\boldsymbol{\Psi}\mathbf{x}^{(k+1)} - \mathbf{U}^{(k+1)}\right)$ | `Psi_out`, $\mathbf{U}^{(k+1)}$, $\boldsymbol{\eta}^{(k)}$, $\mu_2$ | $\boldsymbol{\eta}^{(k+1)}$ | Gradient ascent on dual |
| L.12 | Dual update rho | $\boldsymbol{\rho}^{(k+1)} = \boldsymbol{\rho}^{(k)} + \mu_3 \left(\mathbf{x}^{(k+1)} - \mathbf{W}^{(k+1)}\right)$ | $\mathbf{x}^{(k+1)}$, $\mathbf{W}^{(k+1)}$, $\boldsymbol{\rho}^{(k)}$, $\mu_3$ | $\boldsymbol{\rho}^{(k+1)}$ | Gradient ascent on dual |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Crop to original size | $\hat{\mathbf{x}} = \mathbf{x}^{(n_{\text{iter}})}[..., s_h:e_h, s_w:e_w, :]$ | $\mathbf{x}^{(n_{\text{iter}})}$, `_start_idx`, `_end_idx` | `result`: $[1, D, H, W, C]$ |
| F.2 | Remove batch dimension | $\hat{\mathbf{x}} = \hat{\mathbf{x}}[0]$ if batch size is 1 | `result` | `result`: $[D, H, W, C]$ |
| F.3 | Save PNG | `plot_image(result)` $\rightarrow$ `output_path` | `result` | PNG file |
| F.4 | Save NumPy array | `np.save(result)` | `result` | `.npy` file |

---

## 5. STOPPING CRITERION

- **type**: `iteration_limit`
- **expression**: $k < n_{\text{iter}}$
- **parameters**: 
  - `n_iter`: 50 (default), can be set to any positive integer (example uses 5)
- **check_frequency**: Every iteration (loop termination condition)

---

## 6. EVALUATION METRICS

| Metric | Formula | Description |
|--------|---------|-------------|
| Reconstruction Mean | $\bar{x} = \frac{1}{N}\sum_{d,i,j,c} \hat{x}_{d,i,j,c}$ where $N = D \cdot H \cdot W \cdot C$ | Average pixel intensity of reconstruction |
| Reconstruction Std | $\sigma_x = \sqrt{\frac{1}{N}\sum_{d,i,j,c}(\hat{x}_{d,i,j,c} - \bar{x})^2}$ | Standard deviation of pixel intensities |
| Reconstruction Min | $\min_{d,i,j,c} \hat{x}_{d,i,j,c}$ | Minimum pixel value |
| Reconstruction Max | $\max_{d,i,j,c} \hat{x}_{d,i,j,c}$ | Maximum pixel value |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include:

- `algorithm_identification`: Contains `algorithm_name`, `algorithm_family`, `summary`
- `objective_function`: Contains `full_expression`, `data_fidelity_term`, `regularization_term`, `constraints`
- `variables`: Contains `primal_variables`, `dual_variables`, `auxiliary_variables`, `constants`, `observations`
- `data_preprocessing`: Array of steps with `step_order`, `step_name`, `formula`, `inputs`, `outputs`
- `initialization`: Array of initialization steps
- `iteration_loop`: Contains `loop_type`, `termination_condition`, `steps` (array with `step_name`, `formula`, `inputs`, `outputs`, `computational_note`)
- `finalization`: Array of post-processing steps
- `stopping_criterion`: Contains `type`, `expression`, `parameters`, `check_frequency`
-