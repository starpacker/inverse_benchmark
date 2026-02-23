## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Weighted Least Squares Shapelet Inversion for Gravitational Lens Source Reconstruction
- **algorithm_family**: Linear Inverse Problems / Weighted Least Squares
- **summary**: Reconstructs the source light distribution of a gravitationally lensed system by solving a weighted least squares problem for shapelet basis coefficients, where the forward model includes ray-tracing through a lens mass model and PSF convolution.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\mathbf{a}} = \arg\min_{\mathbf{a}} \left\| \mathbf{W}(\mathbf{y} - \mathbf{H}\mathbf{a}) \right\|_2^2$$

Equivalently:
$$\hat{\mathbf{a}} = \arg\min_{\mathbf{a}} \sum_{i=1}^{N_{\text{pix}}} \frac{(y_i - [\mathbf{H}\mathbf{a}]_i)^2}{\sigma_i^2}$$

### data_fidelity_term
$$\chi^2 = (\mathbf{y} - \mathbf{H}\mathbf{a})^\top \mathbf{C}^{-1} (\mathbf{y} - \mathbf{H}\mathbf{a})$$

where:
- $\mathbf{C} = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_{N_{\text{pix}}}^2)$ is the noise covariance matrix
- $\mathbf{W} = \mathbf{C}^{-1/2} = \text{diag}(1/\sigma_1, 1/\sigma_2, \ldots, 1/\sigma_{N_{\text{pix}}})$

### regularization_term
None (unregularized weighted least squares)

### constraints
None (unconstrained optimization)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{a}$ | $(N_{\text{coeff}},)$ | float64 | Solved analytically | Shapelet basis coefficients |

where $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$

### 2.2 Dual Variables

None (direct analytical solution, no iterative dual formulation)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{H}$ | $(N_{\text{pix}}, N_{\text{coeff}})$ | float64 | Computed | Lensing response matrix (design matrix) |
| $\mathbf{y}_{\text{pred}}$ | $(N_y, N_x)$ | float64 | Computed | Model-predicted image |
| $\mathbf{r}$ | $(N_y, N_x)$ | float64 | Computed | Residual image |
| $\boldsymbol{\sigma}$ | $(N_y, N_x)$ | float64 | Computed | Per-pixel noise standard deviation (error map) |
| $\mathbf{M}$ | $(N_{\text{coeff}}, N_{\text{coeff}})$ | float64 | Computed | Normal matrix $\mathbf{H}^\top \mathbf{C}^{-1} \mathbf{H}$ |
| $\mathbf{b}$ | $(N_{\text{coeff}},)$ | float64 | Computed | Right-hand side vector $\mathbf{H}^\top \mathbf{C}^{-1} \mathbf{y}$ |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $N_x$ | scalar | int | 100 | Number of pixels in x-direction |
| $N_y$ | scalar | int | 100 | Number of pixels in y-direction |
| $N_{\text{pix}}$ | scalar | int | $N_x \times N_y = 10000$ | Total number of image pixels |
| $\Delta_{\text{pix}}$ | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\sigma_{\text{bkg}}$ | scalar | float64 | 0.05 | Background noise RMS |
| $t_{\text{exp}}$ | scalar | float64 | 100 | Exposure time (seconds) |
| $\text{FWHM}$ | scalar | float64 | 0.1 | PSF full-width at half-maximum (arcsec) |
| $n_{\max}$ | scalar | int | 8 | Maximum shapelet order |
| $\beta$ | scalar | float64 | 0.2 | Shapelet scale parameter (arcsec) |
| $x_c^{\text{src}}$ | scalar | float64 | 0.0 | Shapelet center x-coordinate |
| $y_c^{\text{src}}$ | scalar | float64 | 0.0 | Shapelet center y-coordinate |
| $\theta_E$ | scalar | float64 | 1.0 | Einstein radius of SIE lens |
| $e_1^{\text{lens}}$ | scalar | float64 | 0.1 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | scalar | float64 | -0.1 | Lens ellipticity component 2 |
| $\gamma_1$ | scalar | float64 | 0.05 | External shear component 1 |
| $\gamma_2$ | scalar | float64 | 0.01 | External shear component 2 |
| seed | scalar | int | 42 | Random seed for reproducibility |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(N_y, N_x)$ | float64 | Observed noisy image data |
| $\mathbf{y}_{\text{clean}}$ | $(N_y, N_x)$ | float64 | Clean simulated image (ground truth) |

---

## 3. PREPROCESSING PIPELINE

### Step P1: Coordinate System Setup
- **step_order**: P1
- **step_name**: Define pixel-to-angular coordinate transformation
- **formula**: 
$$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$
$$c_x = c_y = \frac{N_x - 1}{2}$$
$$\text{RA}_0 = -(\mathbf{T}_{00} \cdot c_x + \mathbf{T}_{01} \cdot c_y) = \frac{(N_x-1)\Delta_{\text{pix}}}{2}$$
$$\text{Dec}_0 = -(\mathbf{T}_{10} \cdot c_x + \mathbf{T}_{11} \cdot c_y) = -\frac{(N_x-1)\Delta_{\text{pix}}}{2}$$
- **inputs**: $N_x = 100$, $\Delta_{\text{pix}} = 0.05$
- **outputs**: $\mathbf{T} \in \mathbb{R}^{2 \times 2}$, $\text{RA}_0$, $\text{Dec}_0$

### Step P2: PSF Configuration
- **step_order**: P2
- **step_name**: Initialize Gaussian PSF kernel
- **formula**: 
$$\text{PSF}(\mathbf{x}) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{|\mathbf{x}|^2}{2\sigma_{\text{PSF}}^2}\right)$$
$$\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} = \frac{0.1}{2.355} \approx 0.0425$$
- **inputs**: $\text{FWHM} = 0.1$, $\Delta_{\text{pix}} = 0.05$, truncation = 3
- **outputs**: PSF kernel (discretized, normalized)

### Step P3: Simulate Clean Lensed Image
- **step_order**: P3
- **step_name**: Generate lensed source image via ray-tracing
- **formula**: For each image pixel $(x_i, y_i)$:
$$(\beta_x, \beta_y) = (x_i, y_i) - \boldsymbol{\alpha}(x_i, y_i)$$
$$I_{\text{source}}(\beta_x, \beta_y) = \sum_{k} I_k^{\text{Sersic}}(\beta_x, \beta_y)$$
$$\mathbf{y}_{\text{unlensed}}(x_i, y_i) = I_{\text{source}}(\beta_x, \beta_y)$$
$$\mathbf{y}_{\text{clean}} = \text{PSF} * \mathbf{y}_{\text{unlensed}}$$

where deflection angle $\boldsymbol{\alpha}$ includes SIE and external shear contributions.
- **inputs**: Lens parameters, source parameters, coordinate grid
- **outputs**: $\mathbf{y}_{\text{clean}} \in \mathbb{R}^{100 \times 100}$

### Step P4: Add Poisson Noise
- **step_order**: P4
- **step_name**: Simulate photon counting noise
- **formula**: 
$$\mathbf{y}_{\text{counts}} = \max(\mathbf{y}_{\text{clean}} \cdot t_{\text{exp}}, 0)$$
$$\mathbf{y}_{\text{Poisson}} \sim \text{Poisson}(\mathbf{y}_{\text{counts}})$$
$$\mathbf{n}_{\text{Poisson}} = \frac{\mathbf{y}_{\text{Poisson}}}{t_{\text{exp}}} - \mathbf{y}_{\text{clean}}$$
- **inputs**: $\mathbf{y}_{\text{clean}}$, $t_{\text{exp}} = 100$
- **outputs**: $\mathbf{n}_{\text{Poisson}} \in \mathbb{R}^{100 \times 100}$

### Step P5: Add Background Noise
- **step_order**: P5
- **step_name**: Add Gaussian background noise
- **formula**: 
$$\mathbf{n}_{\text{bkg}} = \sigma_{\text{bkg}} \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
- **inputs**: $\sigma_{\text{bkg}} = 0.05$, shape $(100, 100)$
- **outputs**: $\mathbf{n}_{\text{bkg}} \in \mathbb{R}^{100 \times 100}$

### Step P6: Generate Observed Data
- **step_order**: P6
- **step_name**: Combine signal and noise
- **formula**: 
$$\mathbf{y} = \mathbf{y}_{\text{clean}} + \mathbf{n}_{\text{bkg}} + \mathbf{n}_{\text{Poisson}}$$
- **inputs**: $\mathbf{y}_{\text{clean}}$, $\mathbf{n}_{\text{bkg}}$, $\mathbf{n}_{\text{Poisson}}$
- **outputs**: $\mathbf{y} \in \mathbb{R}^{100 \times 100}$ (observed noisy image)

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

**Step 1**: Compute number of shapelet coefficients
- **step_order**: 1
- **step_name**: Calculate shapelet basis dimension
- **formula**: 
$$N_{\text{coeff}} = \frac{(n_{\max} + 1)(n_{\max} + 2)}{2} = \frac{(8+1)(8+2)}{2} = 45$$
- **inputs**: $n_{\max} = 8$
- **outputs**: $N_{\text{coeff}} = 45$
- **computational_note**: Number of 2D shapelet basis functions up to order $n_{\max}$

**Step 2**: Initialize shapelet basis parameters
- **step_order**: 2
- **step_name**: Set shapelet configuration
- **formula**: 
$$\text{kwargs}_{\text{source}} = \{n_{\max}: 8, \beta: 0.2, x_c: 0.0, y_c: 0.0\}$$
- **inputs**: $n_{\max}$, $\beta$, $x_c^{\text{src}}$, $y_c^{\text{src}}$
- **outputs**: kwargs_source_reconstruct (dictionary)
- **computational_note**: These parameters define the shapelet basis but not the amplitudes

### 4.2 ITERATION LOOP

**[Note: This algorithm has an analytical closed-form solution, not an iterative loop. The "loop" constructs the design matrix column-by-column.]**

**BEGIN_LOOP** (over shapelet basis functions $j = 1, \ldots, N_{\text{coeff}}$)

**Step 3**: Construct design matrix column $j$
- **step_order**: 3
- **step_name**: Compute response to $j$-th shapelet basis [FORWARD]
- **formula**: For basis function $B_j(\boldsymbol{\beta}; n_{\max}, \beta, x_c, y_c)$:
$$[\mathbf{H}]_{:,j} = \text{vec}\left(\text{PSF} * \mathcal{L}[B_j]\right)$$
where $\mathcal{L}[\cdot]$ denotes the lensing operator (ray-tracing):
$$\mathcal{L}[B_j](x_i, y_i) = B_j\left((x_i, y_i) - \boldsymbol{\alpha}(x_i, y_i)\right)$$
- **inputs**: Shapelet basis index $j$, lens model, PSF, coordinate grid
- **outputs**: $\mathbf{H}_{:,j} \in \mathbb{R}^{N_{\text{pix}}}$
- **computational_note**: The shapelet basis $B_j$ is a 2D polar shapelet of order $(n_1, n_2)$ with $n_1 + n_2 \leq n_{\max}$

**END_LOOP**

**Step 4**: Compute noise covariance (error map)
- **step_order**: 4
- **step_name**: Estimate per-pixel noise variance
- **formula**: 
$$\sigma_i^2 = \sigma_{\text{bkg}}^2 + \frac{\max(y_i, 0)}{t_{\text{exp}}}$$
$$\mathbf{C} = \text{diag}(\sigma_1^2, \ldots, \sigma_{N_{\text{pix}}}^2)$$
- **inputs**: $\mathbf{y}$, $\sigma_{\text{bkg}} = 0.05$, $t_{\text{exp}} = 100$
- **outputs**: $\boldsymbol{\sigma} \in \mathbb{R}^{N_{\text{pix}}}$, $\mathbf{C} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ (diagonal)
- **computational_note**: Combines background noise and Poisson noise contributions

**Step 5**: Form normal equations
- **step_order**: 5
- **step_name**: Compute weighted normal matrix and RHS vector
- **formula**: 
$$\mathbf{M} = \mathbf{H}^\top \mathbf{C}^{-1} \mathbf{H} = \sum_{i=1}^{N_{\text{pix}}} \frac{1}{\sigma_i^2} \mathbf{H}_{i,:}^\top \mathbf{H}_{i,:}$$
$$\mathbf{b} = \mathbf{H}^\top \mathbf{C}^{-1} \mathbf{y} = \sum_{i=1}^{N_{\text{pix}}} \frac{y_i}{\sigma_i^2} \mathbf{H}_{i,:}^\top$$
- **inputs**: $\mathbf{H} \in \mathbb{R}^{10000 \times 45}$, $\boldsymbol{\sigma}$, $\mathbf{y}$
- **outputs**: $\mathbf{M} \in \mathbb{R}^{45 \times 45}$, $\mathbf{b} \in \mathbb{R}^{45}$
- **computational_note**: $\mathbf{M}$ is symmetric positive semi-definite

**Step 6**: Solve linear system
- **step_order**: 6
- **step_name**: Compute optimal shapelet coefficients
- **formula**: 
$$\hat{\mathbf{a}} = \mathbf{M}^{-1} \mathbf{b}$$
Explicit solution via matrix inversion or linear solver:
$$\hat{\mathbf{a}} = (\mathbf{H}^\top \mathbf{C}^{-1} \mathbf{H})^{-1} \mathbf{H}^\top \mathbf{C}^{-1} \text{vec}(\mathbf{y})$$
- **inputs**: $\mathbf{M}$, $\mathbf{b}$
- **outputs**: $\hat{\mathbf{a}} \in \mathbb{R}^{45}$ (shapelet_coeffs)
- **computational_note**: Solved using numpy's linear algebra routines; inv_bool=False indicates pseudo-inverse is not used

### 4.3 FINALIZATION

**Step 7**: Compute model image
- **step_order**: 7
- **step_name**: Generate reconstructed lensed image [FORWARD]
- **formula**: 
$$\mathbf{y}_{\text{pred}} = \text{reshape}(\mathbf{H} \hat{\mathbf{a}}, (N_y, N_x))$$
Equivalently:
$$\mathbf{y}_{\text{pred}}(x, y) = \text{PSF} * \left[\sum_{j=1}^{N_{\text{coeff}}} \hat{a}_j \cdot \mathcal{L}[B_j](x, y)\right]$$
- **inputs**: $\mathbf{H}$, $\hat{\mathbf{a}}$
- **outputs**: $\mathbf{y}_{\text{pred}} \in \mathbb{R}^{100 \times 100}$ (model_image)
- **computational_note**: This is the wls_model returned by image_linear_solve

**Step 8**: Compute residuals
- **step_order**: 8
- **step_name**: Calculate data-model difference
- **formula**: 
$$\mathbf{r} = \mathbf{y} - \mathbf{y}_{\text{pred}}$$
- **inputs**: $\mathbf{y}$, $\mathbf{y}_{\text{pred}}$
- **outputs**: $\mathbf{r} \in \mathbb{R}^{100 \times 100}$ (residuals)

---

## 5. STOPPING CRITERION

- **type**: single_iteration (analytical solution)
- **expression**: Not applicable—the weighted least squares problem has a closed-form solution computed in one step.
- **parameters**: None
- **check_frequency**: N/A (no iteration)

---

## 6. EVALUATION METRICS

### Reduced Chi-Square
- **name**: Reduced Chi-Square ($\chi^2_\nu$)
- **formula**: 
$$\chi^2_\nu = \frac{1}{N_{\text{dof}}} \sum_{i=1}^{N_{\text{pix}}} \frac{(y_i - [\mathbf{y}_{\text{pred}}]_i)^2}{\sigma_i^2}$$
$$N_{\text{dof}} = N_{\text{pix}} - N_{\text{coeff}} = 10000 - 45 = 9955$$
- **reference_data**: $\mathbf{y}$ (observed image)
- **predicted_data**: $\mathbf{y}_{\text{pred}}$ (model image)
- **description**: Goodness-of-fit statistic; values near 1.0 indicate good fit

### Residual RMS
- **name**: Root Mean Square of Residuals
- **formula**: 
$$\text{RMS}_r = \sqrt{\frac{1}{N_{\text{pix}}} \sum_{i=1}^{N_{\text{pix}}} r_i^2} = \sqrt{\frac{1}{N_{\text{pix}}} \|\mathbf{y} - \mathbf{y}_{\text{pred}}\|_2^2}$$
- **reference_data**: $\mathbf{y}$
- **predicted_data**: $\mathbf{y}_{\text{pred}}$
- **description**: Overall residual magnitude; should be comparable to $\sigma_{\text{bkg}}$

### Peak Signal-to-Noise Ratio (PSNR)
- **name**: PSNR
- **formula**: 
$$\text{PSNR} = 20 \log_{10}\left(\frac{\max_i |[\mathbf{y}_{\text{pred}}]_i|}{\text{RMS}_r}\right) \text{ dB}$$
- **reference_data**: N/A
- **predicted_data**: $\mathbf{y}_{\text{pred}}$
- **description**: Signal quality metric; higher values indicate better reconstruction

### Forward Operator Consistency
- **name**: Forward Model Verification
- **formula**: 
$$\Delta_{\text{forward}} = \max_i \left| [\mathbf{y}_{\text{pred}}]_i - [\mathbf{y}_{\text{forward}}(\hat{\mathbf{a}})]_i \right|$$
- **reference_data**: $\mathbf{y}_{\text{pred}}$ (from inversion)
- **predicted_data**: $\mathbf{y}_{\text{forward}}(\hat{\mathbf{a}})$ (from forward_operator function)
- **description**: Verifies consistency between inversion model and explicit forward operator; should be $\approx 0$