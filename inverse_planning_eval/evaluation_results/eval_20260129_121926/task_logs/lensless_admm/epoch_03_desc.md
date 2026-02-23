## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: ADMM-TV Lensless Image Reconstruction
- **algorithm_family**: Alternating Direction Method of Multipliers (ADMM) with Total Variation regularization
- **summary**: Reconstructs an image from lensless camera measurements by solving a convex optimization problem with data fidelity, total variation regularization, and non-negativity constraints via variable splitting and alternating minimization in the Fourier domain.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{x}} \frac{1}{2}\|\mathbf{M} \odot (\mathbf{H}\mathbf{x} - \mathbf{y})\|_2^2 + \tau \|\nabla \mathbf{x}\|_1 \quad \text{subject to} \quad \mathbf{x} \geq 0$$

where the ADMM reformulation introduces auxiliary variables:
$$\min_{\mathbf{x}, \mathbf{X}, \mathbf{U}, \mathbf{W}} \frac{1}{2}\|\mathbf{X} - \mathbf{y}\|_2^2 + \tau \|\mathbf{U}\|_1 + \mathcal{I}_{\geq 0}(\mathbf{W})$$
$$\text{subject to} \quad \mathbf{H}\mathbf{x} = \mathbf{X}, \quad \nabla\mathbf{x} = \mathbf{U}, \quad \mathbf{x} = \mathbf{W}$$

### data_fidelity_term
$$\mathcal{D}(\mathbf{x}) = \frac{1}{2}\|\mathbf{M} \odot (\mathbf{H}\mathbf{x} - \mathbf{y})\|_2^2$$

where $\mathbf{M}$ is a binary mask equal to 1 in the valid PSF region and 0 in the padding region, $\mathbf{H}$ is the convolution operator with the PSF, $\mathbf{y}$ is the zero-padded measurement.

### regularization_term
$$\mathcal{R}(\mathbf{x}) = \tau \|\nabla \mathbf{x}\|_1 = \tau \sum_{d,i,j,c} \sqrt{|(\nabla_h \mathbf{x})_{d,i,j,c}|^2 + |(\nabla_v \mathbf{x})_{d,i,j,c}|^2}$$

implemented as anisotropic TV:
$$\mathcal{R}(\mathbf{x}) = \tau \left( \|\nabla_h \mathbf{x}\|_1 + \|\nabla_v \mathbf{x}\|_1 \right)$$

### constraints
1. $\mathbf{x}_{d,i,j,c} \geq 0$ for all indices (non-negativity)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{x}$ | primal | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{x}^{(0)} = \mathbf{0}$ | Reconstructed image in padded space |

### Auxiliary Variables (ADMM Splitting)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{X}$ | auxiliary | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{X}^{(0)} = \mathbf{0}$ | Auxiliary for forward model constraint |
| $\mathbf{U}$ | auxiliary | $[1, D, H_p, W_p, C, 2]$ | float32 | $\mathbf{U}^{(0)} = \mathbf{0}$ | Auxiliary for TV constraint (2 gradient directions) |
| $\mathbf{W}$ | auxiliary | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{W}^{(0)} = \mathbf{0}$ | Auxiliary for non-negativity constraint |

### Dual Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\boldsymbol{\xi}$ | dual | $[1, D, H_p, W_p, C]$ | float32 | $\boldsymbol{\xi}^{(0)} = \mathbf{0}$ | Dual variable for $\mathbf{Hx} = \mathbf{X}$ |
| $\boldsymbol{\eta}$ | dual | $[1, D, H_p, W_p, C, 2]$ | float32 | $\boldsymbol{\eta}^{(0)} = \mathbf{0}$ | Dual variable for $\nabla\mathbf{x} = \mathbf{U}$ |
| $\boldsymbol{\rho}$ | dual | $[1, D, H_p, W_p, C]$ | float32 | $\boldsymbol{\rho}^{(0)} = \mathbf{0}$ | Dual variable for $\mathbf{x} = \mathbf{W}$ |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}$ | observations | $[D, H, W, C]$ | float32 | Loaded from file, normalized | Raw lensless measurement |
| $\mathbf{y}_{\text{pad}}$ | observations | $[D, H_p, W_p, C]$ | float32 | Zero-padded $\mathbf{y}$ | Measurement in padded domain |
| $\mathbf{h}$ | observations | $[D, H, W, C]$ | float32 | Loaded from file, normalized | Point spread function |

### Precomputed Operators (Constants)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\hat{\mathbf{H}}$ | constants | $[D, H_p, W_p//2+1, C]$ | complex64 | $\mathcal{F}\{\text{pad}(\mathbf{h})\}$ | FFT of zero-padded PSF |
| $\hat{\mathbf{H}}^*$ | constants | $[D, H_p, W_p//2+1, C]$ | complex64 | $\overline{\hat{\mathbf{H}}}$ | Complex conjugate of $\hat{\mathbf{H}}$ |
| $\hat{\boldsymbol{\Psi}}^T\hat{\boldsymbol{\Psi}}$ | constants | $[D, H_p, W_p//2+1, C]$ | complex64 | FFT of TV Gram kernel | Fourier transform of $\nabla^T\nabla$ |
| $\mathbf{R}$ | constants | $[D, H_p, W_p//2+1, C]$ | complex64 | $\frac{1}{\mu_1|\hat{\mathbf{H}}|^2 + \mu_2|\hat{\boldsymbol{\Psi}}^T\hat{\boldsymbol{\Psi}}| + \mu_3}$ | Precomputed division matrix for $\mathbf{x}$-update |
| $\mathbf{D}_X$ | constants | $[D, H_p, W_p, C]$ | float32 | $\frac{1}{\mathbf{M} + \mu_1}$ | Division matrix for $\mathbf{X}$-update |

### Algorithm Constants
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\mu_1$ | constants | scalar | float | $1 \times 10^{-6}$ | ADMM penalty for forward model constraint |
| $\mu_2$ | constants | scalar | float | $1 \times 10^{-5}$ | ADMM penalty for TV constraint |
| $\mu_3$ | constants | scalar | float | $4 \times 10^{-5}$ | ADMM penalty for non-negativity constraint |
| $\tau$ | constants | scalar | float | $1 \times 10^{-4}$ | TV regularization weight |
| $N_{\text{iter}}$ | constants | scalar | int | $5$ (default in main) | Maximum number of ADMM iterations |
| downsample | constants | scalar | int | $4$ | Spatial downsampling factor for data loading |
| norm | constants | string | - | "ortho" | FFT normalization mode |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Load Data | $\mathbf{y}, \mathbf{h} = \texttt{load\_data}(\cdot)$ with normalize=True | psf_path, data_path | $\mathbf{y}: [D,H,W,C]$, $\mathbf{h}: [D,H,W,C]$ |
| 2 | Compute Padded Shape | $H_p = \texttt{next\_fast\_len}(2H-1)$, $W_p = \texttt{next\_fast\_len}(2W-1)$ | $\mathbf{h}.\text{shape}$ | $(H_p, W_p)$ |
| 3 | Compute Padding Indices | $s_h = \lfloor(H_p - H)/2\rfloor$, $s_w = \lfloor(W_p - W)/2\rfloor$, $e_h = s_h + H$, $e_w = s_w + W$ | $(H, W, H_p, W_p)$ | $(s_h, s_w, e_h, e_w)$ |
| 4 | Pad PSF | $\mathbf{h}_{\text{pad}}[s_h:e_h, s_w:e_w, :] = \mathbf{h}$, elsewhere $0$ | $\mathbf{h}: [D,H,W,C]$ | $\mathbf{h}_{\text{pad}}: [D,H_p,W_p,C]$ |
| 5 | Compute PSF FFT | $\hat{\mathbf{H}} = \mathcal{F}_{\text{2D}}^{\text{real}}\{\mathbf{h}_{\text{pad}}\}$ with norm="ortho" | $\mathbf{h}_{\text{pad}}$ | $\hat{\mathbf{H}}: [D,H_p,W_p//2+1,C]$ |
| 6 | Compute Adjoint | $\hat{\mathbf{H}}^* = \overline{\hat{\mathbf{H}}}$ | $\hat{\mathbf{H}}$ | $\hat{\mathbf{H}}^*$ |
| 7 | Pad Measurement | $\mathbf{y}_{\text{pad}}[s_h:e_h, s_w:e_w, :] = \mathbf{y}$, elsewhere $0$ | $\mathbf{y}: [D,H,W,C]$ | $\mathbf{y}_{\text{pad}}: [D,H_p,W_p,C]$ |
| 8 | Build TV Gram Kernel | For $D=1$: $\mathbf{G}[0,0,0]=4$, $\mathbf{G}[0,0,1]=\mathbf{G}[0,0,-1]=\mathbf{G}[0,1,0]=\mathbf{G}[0,-1,0]=-1$ | $(D,H_p,W_p,C)$ | $\mathbf{G}: [D,H_p,W_p,C]$ |
| 9 | Compute TV Gram FFT | $\hat{\boldsymbol{\Psi}}^T\hat{\boldsymbol{\Psi}} = \mathcal{F}_{\text{2D}}^{\text{real}}\{\mathbf{G}\}$ | $\mathbf{G}$ | $\hat{\boldsymbol{\Psi}}^T\hat{\boldsymbol{\Psi}}$ |
| 10 | Precompute $\mathbf{R}$ | $\mathbf{R} = \left(\mu_1 |\hat{\mathbf{H}}^* \odot \hat{\mathbf{H}}| + \mu_2 |\hat{\boldsymbol{\Psi}}^T\hat{\boldsymbol{\Psi}}| + \mu_3\right)^{-1}$ | $\hat{\mathbf{H}}, \hat{\mathbf{H}}^*, \hat{\boldsymbol{\Psi}}^T\hat{\boldsymbol{\Psi}}, \mu_1, \mu_2, \mu_3$ | $\mathbf{R}: [D,H_p,W_p//2+1,C]$ |
| 11 | Precompute $\mathbf{D}_X$ | $\mathbf{D}_X = (\mathbf{M} + \mu_1)^{-1}$ where $\mathbf{M} = \text{pad}(\mathbf{1}_{[D,H,W,C]})$ | $\mu_1$, shape | $\mathbf{D}_X: [D,H_p,W_p,C]$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Initialize Primal | $\mathbf{x}^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ | shape | $\mathbf{x}^{(0)}$ |
| I.2 | Initialize X | $\mathbf{X}^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ | shape | $\mathbf{X}^{(0)}$ |
| I.3 | Initialize U | $\mathbf{U}^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C,2]}$ | shape | $\mathbf{U}^{(0)}$ |
| I.4 | Initialize W | $\mathbf{W}^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ | shape | $\mathbf{W}^{(0)}$ |
| I.5 | Initialize $\boldsymbol{\xi}$ | $\boldsymbol{\xi}^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ | shape | $\boldsymbol{\xi}^{(0)}$ |
| I.6 | Initialize $\boldsymbol{\eta}$ | $\boldsymbol{\eta}^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C,2]}$ | shape | $\boldsymbol{\eta}^{(0)}$ |
| I.7 | Initialize $\boldsymbol{\rho}$ | $\boldsymbol{\rho}^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ | shape | $\boldsymbol{\rho}^{(0)}$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{iter}}-1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | Compute Gradient | $\boldsymbol{\Psi}_{\text{out}} = \nabla \mathbf{x}^{(k)}$ | $\mathbf{x}^{(k)}$ | $\boldsymbol{\Psi}_{\text{out}}$ | See finite difference definition below |
| L.2a | U-update: Compute Argument | $\mathbf{v} = \boldsymbol{\Psi}_{\text{out}} + \boldsymbol{\eta}^{(k)}/\mu_2$ | $\boldsymbol{\Psi}_{\text{out}}, \boldsymbol{\eta}^{(k)}, \mu_2$ | $\mathbf{v}$ | - |
| L.2b | U-update: Soft Threshold | $\mathbf{U}^{(k+1)} = \text{sign}(\mathbf{v}) \odot \max(|\mathbf{v}| - \tau/\mu_2, 0)$ | $\mathbf{v}, \tau, \mu_2$ | $\mathbf{U}^{(k+1)}$ | Element-wise soft thresholding, threshold $= \tau/\mu_2 = 10$ |
| L.3a | X-update: Forward Model | $\mathbf{f} = \mathbf{H} * \mathbf{x}^{(k)}$ | $\mathbf{x}^{(k)}, \hat{\mathbf{H}}$ | $\mathbf{f}$ | Convolution via FFT |
| L.3b | X-update: Closed Form | $\mathbf{X}^{(k+1)} = \mathbf{D}_X \odot (\boldsymbol{\xi}^{(k)} + \mu_1 \mathbf{f} + \mathbf{y}_{\text{pad}})$ | $\mathbf{D}_X, \boldsymbol{\xi}^{(k)}, \mu_1, \mathbf{f}, \mathbf{y}_{\text{pad}}$ | $\mathbf{X}^{(k+1)}$ | Element-wise |
| L.4 | W-update: Projection | $\mathbf{W}^{(k+1)} = \max(\boldsymbol{\rho}^{(k)}/\mu_3 + \mathbf{x}^{(k)}, 0)$ | $\boldsymbol{\rho}^{(k)}, \mu_3, \mathbf{x}^{(k)}$ | $\mathbf{W}^{(k+1)}$ | Proximal of indicator $\mathcal{I}_{\geq 0}$ |
| L.5a | x-update: RHS Term 1 | $\mathbf{r}_1 = \mu_3 \mathbf{W}^{(k+1)} - \boldsymbol{\rho}^{(k)}$ | $\mu_3, \mathbf{W}^{(k+1)}, \boldsymbol{\rho}^{(k)}$ | $\mathbf{r}_1$ | - |
| L.5b | x-update: RHS Term 2 | $\mathbf{r}_2 = \nabla^T(\mu_2 \mathbf{U}^{(k+1)} - \boldsymbol{\eta}^{(k)})$ | $\mu_2, \mathbf{U}^{(k+1)}, \boldsymbol{\eta}^{(k)}$ | $\mathbf{r}_2$ | Adjoint finite difference |
| L.5c | x-update: RHS Term 3 | $\mathbf{r}_3 = \mathbf{H}^T * (\mu_1 \mathbf{X}^{(k+1)} - \boldsymbol{\xi}^{(k)})$ | $\mu_1, \mathbf{X}^{(k+1)}, \boldsymbol{\xi}^{(k)}, \hat{\mathbf{H}}^*$ | $\mathbf{r}_3$ | Adjoint convolution via FFT |
| L.5d | x-update: Combine RHS | $\mathbf{r}_k = \mathbf{r}_1 + \mathbf{r}_2 + \mathbf{r}_3$ | $\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3$ | $\mathbf{r}_k$ | - |
| L.5e | x-update: FFT | $\hat{\mathbf{r}}_k = \mathcal{F}_{\text{2D}}^{\text{real}}\{\mathbf{r}_k\}$ | $\mathbf{r}_k$ | $\hat{\mathbf{r}}_k$ | - |
| L.5f | x-update: Frequency Solve | $\hat{\mathbf{x}}^{(k+1)} = \mathbf{R} \odot \hat{\mathbf{r}}_k$ | $\mathbf{R}, \hat{\mathbf{r}}_k$ | $\hat{\mathbf{x}}^{(k+1)}$ | Element-wise multiplication |
| L.5g | x-update: IFFT | $\mathbf{x}^{(k+1)} = \mathcal{F}_{\text{2D}}^{-1,\text{real}}\{\hat{\mathbf{x}}^{(k+1)}\}$ | $\hat{\mathbf{x}}^{(k+1)}$ | $\mathbf{x}^{(k+1)}$ | - |
| L.6a | Dual: Recompute Forward | $\mathbf{f}' = \mathbf{H} * \mathbf{x}^{(k+1)}$ | $\mathbf{x}^{(k+1)}, \hat{\mathbf{H}}$ | $\mathbf{f}'$ | - |
| L.6b | Dual: Recompute Gradient | $\boldsymbol{\Psi}'_{\text{out}} = \nabla \mathbf{x}^{(k+1)}$ | $\mathbf{x}^{(k+1)}$ | $\boldsymbol{\Psi}'_{\text{out}}$ | - |
| L.7a | Dual: $\boldsymbol{\xi}$-update | $\boldsymbol{\xi}^{(k+1)} = \boldsymbol{\xi}^{(k)} + \mu_1(\mathbf{f}' - \mathbf{X}^{(k+1)})$ | $\boldsymbol{\xi}^{(k)}, \mu_1, \mathbf{f}', \mathbf{X}^{(k+1)}$ | $\boldsymbol{\xi}^{(k+1)}$ | - |
| L.7b | Dual: $\boldsymbol{\eta}$-update | $\boldsymbol{\eta}^{(k+1)} = \boldsymbol{\eta}^{(k)} + \mu_2(\boldsymbol{\Psi}'_{\text{out}} - \mathbf{U}^{(k+1)})$ | $\boldsymbol{\eta}^{(k)}, \mu_2, \boldsymbol{\Psi}'_{\text{out}}, \mathbf{U}^{(k+1)}$ | $\boldsymbol{\eta}^{(k+1)}$ | - |
| L.7c | Dual: $\boldsymbol{\rho}$-update | $\boldsymbol{\rho}^{(k+1)} = \boldsymbol{\rho}^{(k)} + \mu_3(\mathbf{x}^{(k+1)} - \mathbf{W}^{(k+1)})$ | $\boldsymbol{\rho}^{(k)}, \mu_3, \mathbf{x}^{(k+1)}, \mathbf{W}^{(k+1)}$ | $\boldsymbol{\rho}^{(k+1)}$ | - |

**END_LOOP**

#### Operator Definitions

**Finite Difference (Forward):**
$$(\nabla \mathbf{x})_{d,i,j,c,0} = \mathbf{x}_{d,(i+1)\mod H_p,j,c} - \mathbf{x}_{d,i,j,c}$$
$$(\nabla \mathbf{x})_{d,i,j,c,1} = \mathbf{x}_{d,i,(j+1)\mod W_p,c} - \mathbf{x}_{d,i,j,c}$$

**Finite Difference (Adjoint):**
$$(\nabla^T \mathbf{u})_{d,i,j,c} = (\mathbf{u}_{d,(i-1)\mod H_p,j,c,0} - \mathbf{u}_{d,i,j,c,0}) + (\mathbf{u}_{d,i,(j-1)\mod W_p,c,1} - \mathbf{u}_{d,i,j,c,1})$$

**Convolution (Forward):**
$$\mathbf{H} * \mathbf{x} = \text{ifftshift}\left(\mathcal{F}^{-1}\left\{\mathcal{F}\{\mathbf{x}\} \odot \hat{\mathbf{H}}\right\}\right)$$

**Convolution (Adjoint):**
$$\mathbf{H}^T * \mathbf{x} = \text{ifftshift}\left(\mathcal{F}^{-1}\left\{\mathcal{F}\{\mathbf{x}\} \odot \hat{\mathbf{H}}^*\right\}\right)$$

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Crop Result | $\mathbf{x}_{\text{out}} = \mathbf{x}^{(N_{\text{iter}})}[s_h:e_h, s_w:e_w, :]$ | $\mathbf{x}^{(N_{\text{iter}})}, (s_h,s_w,e_h,e_w)$ | $\mathbf{x}_{\text{out}}: [D,H,W,C]$ |
| F.2 | Remove Batch Dim | $\mathbf{x}_{\text{final}} = \mathbf{x}_{\text{out}}[0]$ if batch size is 1 | $\mathbf{x}_{\text{out}}$ | $\mathbf{x}_{\text{final}}: [H,W,C]$ |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k \geq N_{\text{iter}}$
- **parameters**: 
  - $N_{\text{iter}} = 5$ (as set in main execution; function default is 50)
- **check_frequency**: After each complete iteration (after dual variable updates)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Minimum Value | $\min_{d,i,j,c}(\mathbf{x}_{\text{final}})_{d,i,j,c}$ | Minimum pixel intensity in reconstruction |
| Maximum Value | $\max_{d,i,j,c}(\mathbf{x}_{\text{final}})_{d,i,j,c}$ | Maximum pixel intensity in reconstruction |
| Mean | $\bar{x} = \frac{1}{N}\sum_{d,i,j,c}(\mathbf{x}_{\text{final}})_{d,i,j,c}$ where $N = D \cdot H \cdot W \cdot C$ | Mean pixel intensity |
| Standard Deviation | $\sigma_x = \sqrt{\frac{1}{N}\sum_{d,i,j,c}\left((\mathbf{x}_{\text{final}})_{d,i,j,c} - \bar{x}\right)^2}$ | Standard deviation of pixel intensities |