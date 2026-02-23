# Algorithmic Specification Document: Gravitational Lens Source Reconstruction via Shapelet Linear Inversion

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\hat{\mathbf{a}} = \arg\min_{\mathbf{a}} \left\| \mathbf{W} \left( \mathbf{d} - \mathbf{M} \mathbf{a} \right) \right\|_2^2$$

**data_fidelity_term**:
$$\mathcal{L}(\mathbf{a}) = \sum_{i=1}^{N_{\text{pix}}} w_i^2 \left( d_i - [\mathbf{M}\mathbf{a}]_i \right)^2$$

where:
- $d_i$ is the observed flux at pixel $i$
- $[\mathbf{M}\mathbf{a}]_i$ is the predicted flux at pixel $i$
- $w_i = \sigma_i^{-1}$ is the inverse noise standard deviation at pixel $i$

**regularization_term**: None (unconstrained linear least squares)

**constraints**: None (unconstrained optimization)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{a}$ | primal | $(N_{\text{coeff}},)$ | float64 | Solved via least squares | Shapelet basis coefficients |

where $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $n_{\max}$ | constants | scalar | int | User-specified (8) | Maximum shapelet order |
| $\beta$ | constants | scalar | float64 | User-specified (0.2) | Shapelet scale parameter [arcsec] |
| $(c_x^s, c_y^s)$ | constants | $(2,)$ | float64 | User-specified (0,0) | Shapelet center coordinates [arcsec] |
| $\theta_E$ | constants | scalar | float64 | User-specified (1.0) | Einstein radius [arcsec] |
| $(e_1, e_2)$ | constants | $(2,)$ | float64 | User-specified | SIE ellipticity components |
| $(\gamma_1, \gamma_2)$ | constants | $(2,)$ | float64 | User-specified | External shear components |
| $\sigma_{\text{bkg}}$ | constants | scalar | float64 | User-specified (0.05) | Background noise RMS |
| $t_{\text{exp}}$ | constants | scalar | float64 | User-specified (100) | Exposure time [s] |
| $\Delta_{\text{pix}}$ | constants | scalar | float64 | User-specified (0.05) | Pixel scale [arcsec/pixel] |
| $\text{FWHM}$ | constants | scalar | float64 | User-specified (0.1) | PSF full width at half maximum [arcsec] |
| $N_{\text{pix}}$ | constants | scalar | int | User-specified (100) | Image dimension per side |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{d}$ | observations | $(N_{\text{pix}}^2,)$ | float64 | Simulated with noise | Observed image data (flattened) |
| $\mathbf{D}$ | observations | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Simulated with noise | Observed image data (2D) |

### Derived/Intermediate Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{M}$ | derived | $(N_{\text{pix}}^2, N_{\text{coeff}})$ | float64 | Computed | Design matrix (response matrix) |
| $\mathbf{W}$ | derived | $(N_{\text{pix}}^2, N_{\text{pix}}^2)$ | float64 | Computed from noise | Diagonal weight matrix |
| $\boldsymbol{\sigma}$ | derived | $(N_{\text{pix}}^2,)$ | float64 | Computed | Per-pixel noise standard deviation |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Coordinate System Initialization
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Coordinate Transform Setup |
| formula | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ |
| | $c_x = c_y = \frac{N_{\text{pix}} - 1}{2}$ |
| | $\text{RA}_0 = -(T_{00} \cdot c_x + T_{01} \cdot c_y)$ |
| | $\text{Dec}_0 = -(T_{10} \cdot c_x + T_{11} \cdot c_y)$ |
| inputs | $N_{\text{pix}}$ (scalar), $\Delta_{\text{pix}}$ (scalar) |
| outputs | $\mathbf{T}$ (2×2), $\text{RA}_0$ (scalar), $\text{Dec}_0$ (scalar) |

### Step 3.2: PSF Configuration
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Gaussian PSF Construction |
| formula | $\text{PSF}(\mathbf{r}) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{|\mathbf{r}|^2}{2\sigma_{\text{PSF}}^2}\right)$ |
| | $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$ |
| inputs | FWHM (scalar), $\Delta_{\text{pix}}$ (scalar), truncation=3 (scalar) |
| outputs | PSF kernel (discrete, truncated at $3\sigma_{\text{PSF}}$) |

### Step 3.3: True Source Image Simulation
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Clean Lensed Image Generation |
| formula | $I_{\text{clean}}(\boldsymbol{\theta}) = \left[ \text{PSF} * S\left(\boldsymbol{\beta}(\boldsymbol{\theta})\right) \right]$ |
| | where $\boldsymbol{\beta}(\boldsymbol{\theta}) = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$ (lens equation) |
| inputs | kwargs_lens (lens parameters), kwargs_source_true (source parameters) |
| outputs | $\mathbf{I}_{\text{clean}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

### Step 3.4: Poisson Noise Generation
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Poisson Noise Simulation |
| formula | $C_{ij} = I_{\text{clean},ij} \cdot t_{\text{exp}}$ |
| | $\tilde{C}_{ij} = \max(C_{ij}, 0)$ |
| | $C_{ij}^{\text{Poisson}} \sim \text{Poisson}(\tilde{C}_{ij})$ |
| | $N_{ij}^{\text{Poisson}} = \frac{C_{ij}^{\text{Poisson}}}{t_{\text{exp}}} - I_{\text{clean},ij}$ |
| inputs | $\mathbf{I}_{\text{clean}}$ $(N_{\text{pix}} \times N_{\text{pix}})$, $t_{\text{exp}}$ (scalar) |
| outputs | $\mathbf{N}^{\text{Poisson}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

### Step 3.5: Gaussian Background Noise Generation
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Background Noise Simulation |
| formula | $N_{ij}^{\text{bkg}} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$ |
| inputs | $\sigma_{\text{bkg}}$ (scalar), shape $(N_{\text{pix}}, N_{\text{pix}})$ |
| outputs | $\mathbf{N}^{\text{bkg}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

### Step 3.6: Noisy Image Assembly
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Observed Data Construction |
| formula | $\mathbf{D} = \mathbf{I}_{\text{clean}} + \mathbf{N}^{\text{bkg}} + \mathbf{N}^{\text{Poisson}}$ |
| inputs | $\mathbf{I}_{\text{clean}}$, $\mathbf{N}^{\text{bkg}}$, $\mathbf{N}^{\text{Poisson}}$ (all $N_{\text{pix}} \times N_{\text{pix}}$) |
| outputs | $\mathbf{D}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

---

## 4. ALGORITHM STEPS

### Initialization

| Field | Value |
|-------|-------|
| step_name | Define Shapelet Basis Parameters |
| formula | $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$ |
| inputs | $n_{\max}$ (scalar), $\beta$ (scalar), $(c_x^s, c_y^s)$ (2,) |
| outputs | Shapelet configuration dictionary |

| Field | Value |
|-------|-------|
| step_name | Construct Design Matrix $\mathbf{M}$ |
| formula | For each shapelet basis function $\phi_k$ (indexed by $k \in \{1, \ldots, N_{\text{coeff}}\}$): |
| | $\mathbf{M}_{:,k} = \text{vec}\left( \text{PSF} * \phi_k\left(\boldsymbol{\beta}(\boldsymbol{\theta})\right) \right)$ |
| | where $\phi_k$ is the $k$-th Cartesian shapelet basis: |
| | $\phi_{n_1,n_2}(\mathbf{x}; \beta) = \frac{1}{\beta} \chi_{n_1}\left(\frac{x_1}{\beta}\right) \chi_{n_2}\left(\frac{x_2}{\beta}\right)$ |
| | $\chi_n(x) = \frac{1}{\sqrt{2^n n! \sqrt{\pi}}} H_n(x) e^{-x^2/2}$ |
| | $H_n(x)$ = Hermite polynomial of order $n$ |
| | Column $k$ corresponds to $(n_1, n_2)$ with $n_1 + n_2 \leq n_{\max}$ |
| inputs | Shapelet params, lens model, PSF, coordinate grid |
| outputs | $\mathbf{M}$ $(N_{\text{pix}}^2 \times N_{\text{coeff}})$ |
| computational_note | Lenstronomy's `ImageLinearFit` constructs this internally via ray-tracing |

| Field | Value |
|-------|-------|
| step_name | Construct Weight Matrix $\mathbf{W}$ |
| formula | $\sigma_i = \sigma_{\text{bkg}}$ (constant noise model as implemented) |
| | $W_{ii} = \sigma_i^{-1}$, $W_{ij} = 0$ for $i \neq j$ |
| inputs | $\sigma_{\text{bkg}}$ (scalar) |
| outputs | $\mathbf{W}$ (diagonal, $N_{\text{pix}}^2 \times N_{\text{pix}}^2$) |
| computational_note | Stored implicitly; applied element-wise |

---

### Iteration Loop

**BEGIN ITERATION LOOP**

This algorithm solves a linear system directly (non-iterative); the "iteration" is a single-step closed-form solution.

| Field | Value |
|-------|-------|
| step_name | Form Weighted Normal Equations |
| formula | $\mathbf{A} = \mathbf{M}^T \mathbf{W}^T \mathbf{W} \mathbf{M} = \mathbf{M}^T \mathbf{W}^2 \mathbf{M}$ |
| | $\mathbf{b} = \mathbf{M}^T \mathbf{W}^2 \mathbf{d}$ |
| inputs | $\mathbf{M}$ $(N_{\text{pix}}^2 \times N_{\text{coeff}})$, $\mathbf{W}$ (diagonal), $\mathbf{d}$ $(N_{\text{pix}}^2,)$ |
| outputs | $\mathbf{A}$ $(N_{\text{coeff}} \times N_{\text{coeff}})$, $\mathbf{b}$ $(N_{\text{coeff}},)$ |

| Field | Value |
|-------|-------|
| step_name | Solve Linear System |
| formula | $\hat{\mathbf{a}} = \mathbf{A}^{-1} \mathbf{b}$ |
| | Equivalently: solve $\mathbf{A} \hat{\mathbf{a}} = \mathbf{b}$ |
| inputs | $\mathbf{A}$ $(N_{\text{coeff}} \times N_{\text{coeff}})$, $\mathbf{b}$ $(N_{\text{coeff}},)$ |
| outputs | $\hat{\mathbf{a}}$ $(N_{\text{coeff}},)$ |
| computational_note | Solved via `numpy.linalg.solve` or equivalent; `inv_bool=False` indicates direct solve |

| Field | Value |
|-------|-------|
| step_name | Compute Model Image |
| formula | $\hat{\mathbf{d}}_{\text{model}} = \mathbf{M} \hat{\mathbf{a}}$ |
| | Reshape: $\hat{\mathbf{I}}_{\text{model}} = \text{reshape}(\hat{\mathbf{d}}_{\text{model}}, (N_{\text{pix}}, N_{\text{pix}}))$ |
| inputs | $\mathbf{M}$ $(N_{\text{pix}}^2 \times N_{\text{coeff}})$, $\hat{\mathbf{a}}$ $(N_{\text{coeff}},)$ |
| outputs | $\hat{\mathbf{I}}_{\text{model}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

| Field | Value |
|-------|-------|
| step_name | Compute Covariance Matrix (Optional) |
| formula | $\mathbf{C}_{\text{param}} = \mathbf{A}^{-1} = \left( \mathbf{M}^T \mathbf{W}^2 \mathbf{M} \right)^{-1}$ |
| inputs | $\mathbf{A}$ $(N_{\text{coeff}} \times N_{\text{coeff}})$ |
| outputs | $\mathbf{C}_{\text{param}}$ $(N_{\text{coeff}} \times N_{\text{coeff}})$ |
| computational_note | Returned when solution is computed |

| Field | Value |
|-------|-------|
| step_name | Convergence Check |
| formula | N/A (direct solution; no iteration required) |
| inputs | — |
| outputs | Algorithm terminates after single solve |

**END ITERATION LOOP**

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | direct_solution |
| expression | Not applicable; closed-form solution obtained in single step |
| parameters | None |

---

## 6. EVALUATION METRICS

### Metric 6.1: Reduced Chi-Square
| Field | Value |
|-------|-------|
| name | Reduced Chi-Square |
| formula | $\chi^2_{\text{red}} = \frac{1}{N_{\text{dof}}} \sum_{i=1}^{N_{\text{pix}}^2} \frac{(d_i - \hat{d}_{\text{model},i})^2}{\sigma_i^2}$ |
| | $N_{\text{dof}} = N_{\text{pix}}^2 - N_{\text{coeff}}$ |
| description | Goodness-of-fit normalized by degrees of freedom; $\chi^2_{\text{red}} \approx 1$ indicates good fit |

### Metric 6.2: Residual RMS
| Field | Value |
|-------|-------|
| name | Residual Root Mean Square |
| formula | $\text{RMS}_{\text{res}} = \sqrt{\frac{1}{N_{\text{pix}}^2} \sum_{i,j} \left( D_{ij} - \hat{I}_{\text{model},ij} \right)^2}$ |
| description | Standard deviation of residuals; compare to $\sigma_{\text{bkg}}$ for quality assessment |

### Metric 6.3: Peak Signal-to-Noise Ratio
| Field | Value |
|-------|-------|
| name | PSNR |
| formula | $\text{PSNR} = 20 \log_{10}\left( \frac{\max_{i,j} |\hat{I}_{\text{model},ij}|}{\text{RMS}_{\text{res}}} \right)$ |
| description | Ratio of peak model signal to residual noise in decibels |

### Metric 6.4: Coefficient Statistics
| Field | Value |
|-------|-------|
| name | Shapelet Coefficient Statistics |
| formula | $\bar{a} = \frac{1}{N_{\text{coeff}}} \sum_{k=1}^{N_{\text{coeff}}} \hat{a}_k$ |
| | $\sigma_a = \sqrt{\frac{1}{N_{\text{coeff}}} \sum_{k=1}^{N_{\text{coeff}}} (\hat{a}_k - \bar{a})^2}$ |
| | $a_{\max} = \max_k |\hat{a}_k|$ |
| description | Summary statistics of recovered shapelet amplitudes |

### Metric 6.5: Number of Coefficients
| Field | Value |
|-------|-------|
| name | Basis Dimension |
| formula | $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$ |
| description | Total number of shapelet basis functions used in reconstruction |

### Metric 6.6: Forward Operator Consistency
| Field | Value |
|-------|-------|
| name | Forward Model Consistency Check |
| formula | $\Delta_{\text{forward}} = \max_{i,j} \left| I_{\text{forward},ij} - \hat{I}_{\text{model},ij} \right|$ |
| | where $I_{\text{forward}} = \mathcal{F}(\hat{\mathbf{a}})$ via independent forward evaluation |
| description | Verification that forward operator reproduces inversion result |

---

## APPENDIX: Forward Operator Definition

The forward operator $\mathcal{F}: \mathbb{R}^{N_{\text{coeff}}} \to \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ is defined as:

$$\mathcal{F}(\mathbf{a}) = \text{PSF} * \left[ \sum_{k=1}^{N_{\text{coeff}}} a_k \, \phi_k\left( \boldsymbol{\beta}(\boldsymbol{\theta}) \right) \right]$$

where:
- $\boldsymbol{\theta}$ denotes image-plane coordinates
- $\boldsymbol{\beta}(\boldsymbol{\theta}) = \boldsymbol{\theta} - \boldsymbol{\alpha}_{\text{SIE}}(\boldsymbol{\theta}) - \boldsymbol{\alpha}_{\text{shear}}(\boldsymbol{\theta})$ is the source-plane position via lens equation
- $\boldsymbol{\alpha}_{\text{SIE}}$ is the deflection from singular isothermal ellipsoid
- $\boldsymbol{\alpha}_{\text{shear}}$ is the deflection from external shear
- $*$ denotes 2D convolution