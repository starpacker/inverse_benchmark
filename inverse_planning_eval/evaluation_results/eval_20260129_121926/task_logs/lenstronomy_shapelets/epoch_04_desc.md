## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Weighted Linear Least Squares Shapelet Inversion for Gravitational Lens Source Reconstruction
- **algorithm_family**: Linear Inverse Problems / Weighted Least Squares
- **summary**: Reconstructs a gravitationally lensed source by solving a linear system for shapelet basis coefficients using weighted least squares, where the forward model comprises shapelet basis evaluation, gravitational lensing deflection, and PSF convolution.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\mathbf{c}} = \arg\min_{\mathbf{c}} \frac{1}{2} \| \mathbf{W}(\mathbf{y} - \mathbf{A}\mathbf{c}) \|_2^2$$

where $\mathbf{W} = \text{diag}(\sigma_i^{-1})$ is the diagonal weight matrix with $\sigma_i$ being the per-pixel noise standard deviation.

### data_fidelity_term
$$\mathcal{D}(\mathbf{c}) = \frac{1}{2} \sum_{i=1}^{N_{\text{pix}}} \frac{(y_i - [\mathbf{A}\mathbf{c}]_i)^2}{\sigma_i^2}$$

where:
- $y_i$ is the observed flux at pixel $i$
- $[\mathbf{A}\mathbf{c}]_i = \sum_{j=1}^{N_c} A_{ij} c_j$ is the predicted flux at pixel $i$
- $\sigma_i^2 = \sigma_{\text{bkg}}^2 + \max(y_i, 0) / t_{\text{exp}}$ is the total noise variance (background + Poisson)

### regularization_term
None (unconstrained weighted least squares)

### constraints
None (the linear solve is unconstrained)

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{c}$ | $(N_c,)$ where $N_c = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$ | float64 | Solved via WLS | Shapelet coefficient vector |

### 2.2 Dual Variables
None (direct solve, not iterative dual method)

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{A}$ | $(N_{\text{pix}}, N_c)$ | float64 | Computed from basis | Response matrix; column $j$ is lensed+convolved image of $j$-th shapelet basis |
| $\mathbf{y}$ | $(N_{\text{pix}},)$ | float64 | From observation | Flattened observed noisy image |
| $\boldsymbol{\sigma}$ | $(N_{\text{pix}},)$ | float64 | Computed from noise model | Per-pixel noise standard deviation |
| $\hat{\mathbf{y}}$ | $(N_{\text{pix}},)$ | float64 | $\mathbf{A}\hat{\mathbf{c}}$ | Predicted/reconstructed image (flattened) |
| $\mathbf{r}$ | $(N_{\text{pix}},)$ | float64 | $\mathbf{y} - \hat{\mathbf{y}}$ | Residual vector |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{\text{pix,side}}$ | 100 | Number of pixels per image side |
| $N_{\text{pix}}$ | 10000 | Total number of pixels ($100 \times 100$) |
| $\Delta_{\text{pix}}$ | 0.05 arcsec | Pixel scale |
| $\sigma_{\text{bkg}}$ | 0.05 | Background RMS noise (flux units) |
| $t_{\text{exp}}$ | 100 s | Exposure time |
| $\text{FWHM}$ | 0.1 arcsec | PSF full width at half maximum |
| $n_{\max}$ | 8 | Maximum shapelet order |
| $\beta$ | 0.2 arcsec | Shapelet scale parameter |
| $x_c^{\text{src}}$ | 0.0 arcsec | Source center x-coordinate |
| $y_c^{\text{src}}$ | 0.0 arcsec | Source center y-coordinate |
| $\theta_E$ | 1.0 arcsec | Einstein radius (SIE lens) |
| $e_1^{\text{lens}}$ | 0.1 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | -0.1 | Lens ellipticity component 2 |
| $\gamma_1$ | 0.05 | External shear component 1 |
| $\gamma_2$ | 0.01 | External shear component 2 |
| PSF truncation | 3 | Truncation radius in units of $\sigma_{\text{PSF}}$ |
| random_seed | 42 | Random number generator seed |
| inv_bool | False | Flag for covariance matrix inversion (disabled) |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{Y}_{\text{obs}}$ | $(100, 100)$ | float64 | Observed noisy lensed image |
| $\mathbf{Y}_{\text{clean}}$ | $(100, 100)$ | float64 | Clean simulated image (ground truth) |

## 3. PREPROCESSING PIPELINE

**Step 0: Coordinate System Setup**
- step_order: 0
- step_name: Pixel-to-Angular Coordinate Transformation
- formula: 
$$\mathbf{M} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$
$$c_x = c_y = \frac{N_{\text{pix,side}} - 1}{2} = 49.5$$
$$\text{RA}_0 = -(\mathbf{M}_{00} c_x + \mathbf{M}_{01} c_y) = 2.475 \text{ arcsec}$$
$$\text{Dec}_0 = -(\mathbf{M}_{10} c_x + \mathbf{M}_{11} c_y) = -2.475 \text{ arcsec}$$
- inputs: $N_{\text{pix,side}}$ (int), $\Delta_{\text{pix}}$ (float)
- outputs: $\mathbf{M}$ (2×2), $\text{RA}_0$ (float), $\text{Dec}_0$ (float)

**Step 1: PSF Configuration**
- step_order: 1
- step_name: Gaussian PSF Initialization
- formula: 
$$\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} = \frac{0.1}{2.355} \approx 0.0425 \text{ arcsec}$$
$$\text{PSF}(\mathbf{x}) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{|\mathbf{x}|^2}{2\sigma_{\text{PSF}}^2}\right)$$
- inputs: FWHM (float), $\Delta_{\text{pix}}$ (float), truncation (int)
- outputs: PSF kernel (discretized)

**Step 2: Clean Image Simulation**
- step_order: 2
- step_name: Forward Model for True Source
- formula: 
$$\mathbf{Y}_{\text{clean}} = \mathbf{P} * \mathcal{L}[S_{\text{true}}](\mathbf{x})$$
where $\mathcal{L}$ denotes the lensing operator mapping source to image plane and $\mathbf{P}$ is PSF convolution.
- inputs: True source parameters (Sérsic profiles), lens parameters
- outputs: $\mathbf{Y}_{\text{clean}}$ (100×100)

**Step 3: Noise Generation**
- step_order: 3
- step_name: Poisson + Gaussian Noise Addition
- formula:
$$\mathbf{Y}_{\text{counts}} = \max(\mathbf{Y}_{\text{clean}} \cdot t_{\text{exp}}, 0)$$
$$\mathbf{Y}_{\text{Poisson}} \sim \frac{\text{Poisson}(\mathbf{Y}_{\text{counts}})}{t_{\text{exp}}}$$
$$\mathbf{N}_{\text{Poisson}} = \mathbf{Y}_{\text{Poisson}} - \mathbf{Y}_{\text{clean}}$$
$$\mathbf{N}_{\text{bkg}} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$$
$$\mathbf{Y}_{\text{obs}} = \mathbf{Y}_{\text{clean}} + \mathbf{N}_{\text{bkg}} + \mathbf{N}_{\text{Poisson}}$$
- inputs: $\mathbf{Y}_{\text{clean}}$, $t_{\text{exp}}$, $\sigma_{\text{bkg}}$, random_seed
- outputs: $\mathbf{Y}_{\text{obs}}$ (100×100)

**Step 4: Shapelet Basis Enumeration**
- step_order: 4
- step_name: Compute Number of Shapelet Coefficients
- formula:
$$N_c = \frac{(n_{\max} + 1)(n_{\max} + 2)}{2} = \frac{(8+1)(8+2)}{2} = 45$$
- inputs: $n_{\max}$
- outputs: $N_c$ (int)

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

**Step I.1: Initialize ImageLinearFit Object**
- step_name: Construct Linear Fitting Framework
- formula: Instantiate ImageLinearFit with data, PSF, lens model, and shapelet source model
- inputs: $\mathbf{Y}_{\text{obs}}$, PSF, LensModel, kwargs_numerics
- outputs: ImageLinearFit instance

**Step I.2: Set Source Model Parameters**
- step_name: Configure Shapelet Parameters
- formula:
$$\text{kwargs\_source} = \{n_{\max}: 8, \beta: 0.2, x_c: 0.0, y_c: 0.0\}$$
- inputs: $n_{\max}$, $\beta$, $x_c^{\text{src}}$, $y_c^{\text{src}}$
- outputs: kwargs_source_reconstruct (list of dict)

### 4.2 ITERATION LOOP

The algorithm is a **direct solve** (non-iterative), implemented via `image_linear_solve`. The internal procedure:

**BEGIN_LOOP** (Internal to lenstronomy - single pass)

**Step L.1: Construct Response Matrix** [FORWARD]
- step_name: Build Design Matrix
- formula: For each shapelet basis function $\phi_j$ ($j = 1, \ldots, N_c$):
$$A_{:,j} = \text{vec}\left(\mathbf{P} * \mathcal{L}[\phi_j](\mathbf{x})\right)$$
where the 2D shapelet basis function is:
$$\phi_{n_1, n_2}(x, y) = \frac{1}{\beta} \chi_{n_1}\left(\frac{x - x_c}{\beta}\right) \chi_{n_2}\left(\frac{y - y_c}{\beta}\right)$$
$$\chi_n(\xi) = \frac{1}{\sqrt{2^n n! \sqrt{\pi}}} H_n(\xi) e^{-\xi^2/2}$$
$H_n$ is the $n$-th Hermite polynomial.
- inputs: Shapelet parameters, lens model, PSF
- outputs: $\mathbf{A} \in \mathbb{R}^{N_{\text{pix}} \times N_c}$

**Step L.2: Construct Weight Matrix**
- step_name: Compute Noise Weights
- formula:
$$\sigma_i = \sqrt{\sigma_{\text{bkg}}^2 + \frac{\max(y_i, 0)}{t_{\text{exp}}}}$$
$$W_{ii} = \sigma_i^{-1}$$
- inputs: $\mathbf{y}$, $\sigma_{\text{bkg}}$, $t_{\text{exp}}$
- outputs: $\mathbf{W} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ (diagonal)

**Step L.3: Solve Weighted Normal Equations**
- step_name: Weighted Least Squares Solution
- formula:
$$(\mathbf{A}^T \mathbf{W}^2 \mathbf{A}) \hat{\mathbf{c}} = \mathbf{A}^T \mathbf{W}^2 \mathbf{y}$$
Solved as:
$$\hat{\mathbf{c}} = (\mathbf{A}^T \mathbf{W}^2 \mathbf{A})^{-1} \mathbf{A}^T \mathbf{W}^2 \mathbf{y}$$
- inputs: $\mathbf{A}$, $\mathbf{W}$, $\mathbf{y}$
- outputs: $\hat{\mathbf{c}} \in \mathbb{R}^{N_c}$
- computational_note: Uses numpy.linalg.lstsq or equivalent; `inv_bool=False` means covariance matrix NOT explicitly inverted for output

**Step L.4: Compute Model Image**
- step_name: Reconstruct Predicted Image
- formula:
$$\hat{\mathbf{y}} = \mathbf{A}\hat{\mathbf{c}}$$
$$\hat{\mathbf{Y}}_{\text{model}} = \text{reshape}(\hat{\mathbf{y}}, (N_{\text{pix,side}}, N_{\text{pix,side}}))$$
- inputs: $\mathbf{A}$, $\hat{\mathbf{c}}$
- outputs: $\hat{\mathbf{Y}}_{\text{model}} \in \mathbb{R}^{100 \times 100}$

**END_LOOP**

### 4.3 FINALIZATION

**Step F.1: Compute Error Map**
- step_name: Per-Pixel Uncertainty Estimation
- formula:
$$\text{error\_map}_{i} = \sigma_i$$
- inputs: Noise model parameters
- outputs: error_map $(100 \times 100)$

**Step F.2: Extract Coefficients**
- step_name: Store Shapelet Coefficients
- formula:
$$\texttt{shapelet\_coeffs} = \hat{\mathbf{c}}$$
- inputs: param (from linear solve)
- outputs: numpy array of shape $(N_c,)$

**Step F.3: Compute Reduced Chi-Square**
- step_name: Goodness-of-Fit Metric
- formula:
$$\chi^2_{\text{red}} = \frac{1}{N_{\text{pix}} - N_c} \sum_{i=1}^{N_{\text{pix}}} \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}$$
- inputs: $\mathbf{y}$, $\hat{\mathbf{y}}$, $\boldsymbol{\sigma}$, $N_{\text{pix}}$, $N_c$
- outputs: $\chi^2_{\text{red}}$ (float)

## 5. STOPPING CRITERION

- **type**: single_pass (direct solve, no iteration)
- **expression**: N/A (matrix equation solved in one step)
- **parameters**: None
- **check_frequency**: N/A

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Reduced Chi-Square | $\chi^2_{\text{red}} = \frac{1}{N_{\text{pix}} - N_c} \sum_{i=1}^{N_{\text{pix}}} \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}$ | Goodness of fit normalized by degrees of freedom |
| Residual RMS | $\text{RMS}_r = \sqrt{\frac{1}{N_{\text{pix}}} \sum_{i=1}^{N_{\text{pix}}} (y_i - \hat{y}_i)^2}$ | Root mean square of residuals |
| PSNR | $\text{PSNR} = 20 \log_{10}\left(\frac{\max_i |\hat{y}_i|}{\text{RMS}_r}\right)$ | Peak signal-to-noise ratio in dB |
| Coefficient Mean | $\bar{c} = \frac{1}{N_c} \sum_{j=1}^{N_c} c_j$ | Mean of shapelet coefficients |
| Coefficient Std | $\sigma_c = \sqrt{\frac{1}{N_c} \sum_{j=1}^{N_c} (c_j - \bar{c})^2}$ | Standard deviation of coefficients |
| Coefficient Max Abs | $c_{\max} = \max_j |c_j|$ | Maximum absolute coefficient value |
| Forward Consistency | $\Delta_{\text{fwd}} = \max_i |y_i^{\text{pred}} - \hat{y}_i|$ | Maximum difference between forward operator output and model image |

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include:
- `algorithm_identification`: {algorithm_name, algorithm_family, summary}
- `objective_function`: {full_expression, data_fidelity_term, regularization_term, constraints}
- `variables`: {primal, dual, auxiliary, constants, observations}
- `data_preprocessing`: [{step_order, step_name, formula, inputs, outputs}]
- `initialization`: [{step_name, formula, inputs, outputs}]
- `iteration_loop`: [{step_name, formula, inputs, outputs, computational_note}] (or indication of direct solve)
- `finalization`: [{step_name, formula, inputs, outputs}]
- `stopping_criterion`: {type, expression, parameters, check_frequency}
- `evaluation`: [{metric_name, formula, description}]
- `outputs`: {model_image, shapelet_coeffs, chi2_reduced, residual_rms, psnr, elapsed_time, num_coeffs}