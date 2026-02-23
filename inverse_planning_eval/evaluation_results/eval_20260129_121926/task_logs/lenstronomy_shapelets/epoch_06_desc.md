## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Weighted Least Squares Shapelet Source Reconstruction",
  "algorithm_family": "Linear Inverse Problems / Gravitational Lens Modeling",
  "summary": "Reconstructs a gravitational lens source by solving a weighted least squares problem for shapelet basis coefficients, where the forward model includes gravitational lensing and PSF convolution."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{a} \in \mathbb{R}^{N_{\text{coeff}}}} \frac{1}{2} \left\| \mathbf{W} \left( \mathbf{y} - \boldsymbol{\Phi} \mathbf{a} \right) \right\|_2^2$$

**data_fidelity_term**:
$$\mathcal{D}(\mathbf{a}) = \frac{1}{2} \sum_{i=1}^{N_{\text{pix}}} w_i^2 \left( y_i - [\boldsymbol{\Phi} \mathbf{a}]_i \right)^2$$

where:
- $y_i$ is the $i$-th pixel of the observed noisy image
- $[\boldsymbol{\Phi} \mathbf{a}]_i$ is the $i$-th pixel of the predicted image
- $w_i = 1/\sigma_i$ is the inverse noise standard deviation at pixel $i$
- $\sigma_i^2 = \sigma_{\text{bkg}}^2 + \max(y_i, 0) / t_{\text{exp}}$ (background + Poisson variance)

**regularization_term**: None (unconstrained weighted least squares)

**constraints**: None

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{a}$ | $(N_{\text{coeff}},)$ | float64 | Solved via WLS | Shapelet basis coefficients |

### 2.2 Dual Variables
None (direct linear solve, not iterative)

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\Phi}$ | $(N_{\text{pix}}, N_{\text{coeff}})$ | float64 | Computed | Response matrix (lensed+convolved basis functions) |
| $\mathbf{W}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Computed | Diagonal weight matrix with $W_{ii} = 1/\sigma_i$ |
| $\mathbf{y}_{\text{pred}}$ | $(N_y, N_x)$ | float64 | Computed | Predicted model image |
| $\mathbf{r}$ | $(N_y, N_x)$ | float64 | Computed | Residual image |
| $\mathbf{M}$ | $(N_{\text{coeff}}, N_{\text{coeff}})$ | float64 | Computed | Normal equation matrix $\boldsymbol{\Phi}^T \mathbf{W}^2 \boldsymbol{\Phi}$ |
| $\mathbf{b}$ | $(N_{\text{coeff}},)$ | float64 | Computed | Normal equation RHS $\boldsymbol{\Phi}^T \mathbf{W}^2 \mathbf{y}$ |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $N_x$ | 100 | Image width in pixels |
| $N_y$ | 100 | Image height in pixels |
| $N_{\text{pix}}$ | $N_x \times N_y = 10000$ | Total number of pixels |
| $\Delta_{\text{pix}}$ | 0.05 arcsec | Pixel scale |
| $\sigma_{\text{bkg}}$ | 0.05 | Background noise RMS |
| $t_{\text{exp}}$ | 100 s | Exposure time |
| $\text{FWHM}$ | 0.1 arcsec | PSF full width at half maximum |
| $n_{\max}$ | 8 | Maximum shapelet order |
| $\beta$ | 0.2 arcsec | Shapelet scale parameter |
| $N_{\text{coeff}}$ | $(n_{\max}+1)(n_{\max}+2)/2 = 45$ | Number of shapelet coefficients |
| $(c_x^s, c_y^s)$ | $(0.0, 0.0)$ | Shapelet center coordinates |
| $\theta_E$ | 1.0 arcsec | Einstein radius (SIE lens) |
| $(e_1, e_2)_{\text{SIE}}$ | $(0.1, -0.1)$ | SIE ellipticity components |
| $(\gamma_1, \gamma_2)$ | $(0.05, 0.01)$ | External shear components |
| $\sigma_{\text{PSF}}$ | $\text{FWHM}/(2\sqrt{2\ln 2})$ | PSF Gaussian width |
| random_seed | 42 | Random seed for reproducibility |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(N_y, N_x)$ | float64 | Observed noisy lensed image |
| $\mathbf{y}_{\text{clean}}$ | $(N_y, N_x)$ | float64 | True clean lensed image (for evaluation) |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Set random seed | `np.random.seed(42)` | seed=42 | RNG state | Reproducibility |
| P2 | Compute transformation matrix | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}$ | $\mathbf{T} \in \mathbb{R}^{2 \times 2}$ | Linear mapping pixel→angular |
| P3 | Compute image center pixel | $c_x = c_y = (N_x - 1)/2 = 49.5$ | $N_x, N_y$ | $(c_x, c_y)$ | Image centered at origin |
| P4 | Compute reference RA/Dec | $\text{RA}_0 = -(\mathbf{T}_{00} c_x + \mathbf{T}_{01} c_y) = 2.475$, $\text{Dec}_0 = -(\mathbf{T}_{10} c_x + \mathbf{T}_{11} c_y) = -2.475$ | $\mathbf{T}, c_x, c_y$ | $(\text{RA}_0, \text{Dec}_0)$ | Origin at image center |
| P5 | Initialize ImageData class | Create `data_class` with coordinate system | $\mathbf{T}, \text{RA}_0, \text{Dec}_0, \sigma_{\text{bkg}}, t_{\text{exp}}$ | `data_class` | — |
| P6 | Initialize PSF class | Gaussian PSF with $\sigma_{\text{PSF}} = \text{FWHM}/(2\sqrt{2\ln 2})$, truncation at $3\sigma$ | FWHM, $\Delta_{\text{pix}}$ | `psf_class` | PSF is space-invariant |
| P7 | Initialize Lens Model | SIE + external shear: $\boldsymbol{\alpha}(\boldsymbol{\theta}) = \boldsymbol{\alpha}_{\text{SIE}}(\boldsymbol{\theta}) + \boldsymbol{\alpha}_{\text{shear}}(\boldsymbol{\theta})$ | $\theta_E, e_1, e_2, \gamma_1, \gamma_2$ | `lens_model_class` | Fixed lens parameters |
| P8 | Initialize true source model | Sérsic ellipse + Sérsic profile | kwargs_source_true | `source_model_class_true` | For simulation only |
| P9 | Simulate clean lensed image | $\mathbf{y}_{\text{clean}} = \mathbf{P} * \mathcal{L}[S_{\text{true}}](\boldsymbol{\theta})$ where $\mathcal{L}$ is lens mapping, $\mathbf{P}$ is PSF | lens model, source model, PSF | $\mathbf{y}_{\text{clean}} \in \mathbb{R}^{N_y \times N_x}$ | — |
| P10 | Generate Poisson noise | $\mathbf{n}_{\text{Poisson}} = \frac{\text{Poisson}(\max(\mathbf{y}_{\text{clean}} \cdot t_{\text{exp}}, 0))}{t_{\text{exp}}} - \mathbf{y}_{\text{clean}}$ | $\mathbf{y}_{\text{clean}}, t_{\text{exp}}$ | $\mathbf{n}_{\text{Poisson}} \in \mathbb{R}^{N_y \times N_x}$ | Poisson statistics |
| P11 | Generate background noise | $\mathbf{n}_{\text{bkg}} = \sigma_{\text{bkg}} \cdot \mathcal{N}(0, 1)^{N_y \times N_x}$ | $\sigma_{\text{bkg}}$ | $\mathbf{n}_{\text{bkg}} \in \mathbb{R}^{N_y \times N_x}$ | i.i.d. Gaussian |
| P12 | Create noisy observation | $\mathbf{y} = \mathbf{y}_{\text{clean}} + \mathbf{n}_{\text{bkg}} + \mathbf{n}_{\text{Poisson}}$ | $\mathbf{y}_{\text{clean}}, \mathbf{n}_{\text{bkg}}, \mathbf{n}_{\text{Poisson}}$ | $\mathbf{y} \in \mathbb{R}^{N_y \times N_x}$ | Additive noise model |
| P13 | Update data class | Store $\mathbf{y}$ in `data_class` | $\mathbf{y}$ | Updated `data_class` | — |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | `source_model_list` | `['SHAPELETS']` | list | — | Constant |
| 2 | `source_model_class` | `LightModel(['SHAPELETS'])` | object | — | Constructor |
| 3 | `imageLinearFit` | `ImageLinearFit(data_class, psf_class, lens_model_class, source_model_class, kwargs_numerics)` | object | — | Constructor |
| 4 | `kwargs_source` | `[{'n_max': 8, 'beta': 0.2, 'center_x': 0.0, 'center_y': 0.0}]` | list | — | Parameters |

### 4.2 ITERATION LOOP

**Note**: This algorithm uses a direct linear solve (non-iterative), but the internal computation follows these logical steps:

---
**BEGIN_LOOP** (Single pass - direct solve)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 5 | Generate shapelet basis index pairs | For $n = 0, \ldots, n_{\max}$: $(n_1, n_2)$ where $n_1 + n_2 = n$, $n_1 \geq 0$, $n_2 \geq 0$. Total: $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2} = 45$ | $n_{\max}=8$ | Index set $\{(n_1^{(k)}, n_2^{(k)})\}_{k=1}^{N_{\text{coeff}}}$ | Polar shapelet indexing |
| 6 | Compute pixel coordinate grids | $x_{ij} = \text{RA}_0 + \mathbf{T}_{00} \cdot i + \mathbf{T}_{01} \cdot j$, $y_{ij} = \text{Dec}_0 + \mathbf{T}_{10} \cdot i + \mathbf{T}_{11} \cdot j$ for $i,j \in [0, N-1]$ | $\mathbf{T}, \text{RA}_0, \text{Dec}_0, N_x, N_y$ | $\mathbf{X}, \mathbf{Y} \in \mathbb{R}^{N_y \times N_x}$ | Image plane coordinates |
| 7 | [FORWARD] Compute source plane coordinates via lens equation | $\boldsymbol{\beta}_{ij} = \boldsymbol{\theta}_{ij} - \boldsymbol{\alpha}(\boldsymbol{\theta}_{ij})$ where $\boldsymbol{\alpha} = \boldsymbol{\alpha}_{\text{SIE}} + \boldsymbol{\alpha}_{\text{shear}}$ | $(\mathbf{X}, \mathbf{Y})$, lens parameters | $(\mathbf{X}_s, \mathbf{Y}_s) \in \mathbb{R}^{N_y \times N_x}$ each | Ray-tracing through lens |
| 8 | Compute $k$-th shapelet basis function in source plane | $B_k(\mathbf{X}_s, \mathbf{Y}_s) = \phi_{n_1^{(k)}}\left(\frac{\mathbf{X}_s - c_x^s}{\beta}\right) \phi_{n_2^{(k)}}\left(\frac{\mathbf{Y}_s - c_y^s}{\beta}\right)$ where $\phi_n(u) = \frac{1}{\sqrt{2^n n! \sqrt{\pi}}} H_n(u) e^{-u^2/2}$, $H_n$ is Hermite polynomial | $(n_1^{(k)}, n_2^{(k)}), \mathbf{X}_s, \mathbf{Y}_s, \beta, c_x^s, c_y^s$ | $\mathbf{B}_k \in \mathbb{R}^{N_y \times N_x}$ | Repeated for $k=1,\ldots,N_{\text{coeff}}$ |
| 9 | [FORWARD] Convolve basis function with PSF | $\tilde{\mathbf{B}}_k = \mathbf{B}_k * \mathbf{P}$ where $\mathbf{P}(x,y) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{x^2+y^2}{2\sigma_{\text{PSF}}^2}\right)$ | $\mathbf{B}_k, \mathbf{P}$ | $\tilde{\mathbf{B}}_k \in \mathbb{R}^{N_y \times N_x}$ | FFT-based convolution |
| 10 | Assemble response matrix | $\boldsymbol{\Phi}_{:,k} = \text{vec}(\tilde{\mathbf{B}}_k)$ for $k=1,\ldots,N_{\text{coeff}}$ | $\{\tilde{\mathbf{B}}_k\}_{k=1}^{N_{\text{coeff}}}$ | $\boldsymbol{\Phi} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{coeff}}}$ | Column-wise stacking |
| 11 | Compute noise variance map | $\sigma_i^2 = \sigma_{\text{bkg}}^2 + \frac{\max(y_i, 0)}{t_{\text{exp}}}$ for each pixel $i$ | $\mathbf{y}, \sigma_{\text{bkg}}, t_{\text{exp}}$ | $\boldsymbol{\sigma}^2 \in \mathbb{R}^{N_{\text{pix}}}$ | Poisson + Gaussian variance |
| 12 | Construct weight vector | $w_i = 1/\sigma_i$ | $\boldsymbol{\sigma}^2$ | $\mathbf{w} \in \mathbb{R}^{N_{\text{pix}}}$ | Inverse variance weighting |
| 13 | Form weighted response matrix | $\tilde{\boldsymbol{\Phi}}_{ij} = w_i \cdot \boldsymbol{\Phi}_{ij}$ | $\boldsymbol{\Phi}, \mathbf{w}$ | $\tilde{\boldsymbol{\Phi}} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{coeff}}}$ | Row scaling |
| 14 | Form weighted data vector | $\tilde{\mathbf{y}}_i = w_i \cdot y_i$ | $\mathbf{y}, \mathbf{w}$ | $\tilde{\mathbf{y}} \in \mathbb{R}^{N_{\text{pix}}}$ | Vectorized observation |
| 15 | Compute normal equation matrix | $\mathbf{M} = \tilde{\boldsymbol{\Phi}}^T \tilde{\boldsymbol{\Phi}} = \boldsymbol{\Phi}^T \mathbf{W}^2 \boldsymbol{\Phi}$ | $\tilde{\boldsymbol{\Phi}}$ | $\mathbf{M} \in \mathbb{R}^{N_{\text{coeff}} \times N_{\text{coeff}}}$ | Symmetric positive semi-definite |
| 16 | Compute normal equation RHS | $\mathbf{b} = \tilde{\boldsymbol{\Phi}}^T \tilde{\mathbf{y}} = \boldsymbol{\Phi}^T \mathbf{W}^2 \mathbf{y}$ | $\tilde{\boldsymbol{\Phi}}, \tilde{\mathbf{y}}$ | $\mathbf{b} \in \mathbb{R}^{N_{\text{coeff}}}$ | Matrix-vector product |
| 17 | Solve normal equations | $\mathbf{a} = \mathbf{M}^{-1} \mathbf{b}$ via `np.linalg.lstsq` | $\mathbf{M}, \mathbf{b}$ | $\mathbf{a} \in \mathbb{R}^{N_{\text{coeff}}}$ | Direct solve (inv_bool=False) |

**END_LOOP**

---

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 18 | Compute model image | $\mathbf{y}_{\text{pred}} = \text{reshape}(\boldsymbol{\Phi} \mathbf{a}, (N_y, N_x))$ | $\boldsymbol{\Phi}, \mathbf{a}$ | $\mathbf{y}_{\text{pred}} \in \mathbb{R}^{N_y \times N_x}$ | Matrix-vector product + reshape |
| 19 | Compute error map | $\boldsymbol{\sigma}_{\text{err}} = \text{reshape}(\boldsymbol{\sigma}, (N_y, N_x))$ | $\boldsymbol{\sigma}$ | $\boldsymbol{\sigma}_{\text{err}} \in \mathbb{R}^{N_y \times N_x}$ | Pixel-wise noise estimate |
| 20 | Compute residuals | $\mathbf{r} = \mathbf{y} - \mathbf{y}_{\text{pred}}$ | $\mathbf{y}, \mathbf{y}_{\text{pred}}$ | $\mathbf{r} \in \mathbb{R}^{N_y \times N_x}$ | Element-wise subtraction |
| 21 | Compute chi-squared | $\chi^2 = \sum_{i=1}^{N_{\text{pix}}} \left(\frac{r_i}{\sigma_i}\right)^2$ | $\mathbf{r}, \boldsymbol{\sigma}_{\text{err}}$ | $\chi^2 \in \mathbb{R}$ | Sum of squared standardized residuals |
| 22 | Compute reduced chi-squared | $\chi^2_{\text{red}} = \frac{\chi^2}{N_{\text{pix}} - N_{\text{coeff}}} = \frac{\chi^2}{10000 - 45}$ | $\chi^2, N_{\text{pix}}, N_{\text{coeff}}$ | $\chi^2_{\text{red}} \in \mathbb{R}$ | Degrees of freedom correction |
| 23 | Extract shapelet coefficients | $\texttt{shapelet\_coeffs} = \mathbf{a}$ | $\mathbf{a}$ | `shapelet_coeffs` $\in \mathbb{R}^{45}$ | Final output |

## 5. STOPPING CRITERION

```json
{
  "type": "direct_solve",
  "expression": "Single-pass direct linear solve via normal equations; no iterative stopping criterion",
  "parameters": {
    "method": "numpy.linalg.lstsq (via lenstronomy)",
    "inv_bool": false
  },
  "check_frequency": "N/A (non-iterative)"
}
```

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Reduced Chi-Squared | $\chi^2_{\text{red}} = \frac{1}{N_{\text{pix}} - N_{\text{coeff}}} \sum_{i=1}^{N_{\text{pix}}} \left(\frac{y_i - [\mathbf{y}_{\text{pred}}]_i}{\sigma_i}\right)^2$ | $\mathbf{y}$ (noisy observation) | $\mathbf{y}_{\text{pred}}$ (model) | Goodness of fit; ideal value ≈ 1 |
| Residual RMS | $\text{RMS}_r = \sqrt{\frac{1}{N_{\text{pix}}} \sum_{i=1}^{N_{\text{pix}}} r_i^2}$ | $\mathbf{y}$ | $\mathbf{y}_{\text{pred}}$ | Root mean square of residuals |
| Peak Signal-to-Noise Ratio | $\text{PSNR} = 20 \log_{10}\left(\frac{\max_i |[\mathbf{y}_{\text{pred}}]_i|}{\text{RMS}_r}\right)$ dB | — | $\mathbf{y}_{\text{pred}}$ | Image quality metric |
| Coefficient Statistics | $\bar{a} = \frac{1}{N_{\text{coeff}}} \sum_k a_k$, $\sigma_a = \sqrt{\frac{1}{N_{\text{coeff}}} \sum_k (a_k - \bar{a})^2}$, $\|a\|_\infty = \max_k |a_k|$ | — | $\mathbf{a}$ | Shapelet coefficient distribution |
| Number of Coefficients | $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$ | Expected: 45 | Actual count | Verification of basis size |
| Forward Operator Consistency | $\epsilon = \max_{i,j} |\mathbf{y}_{\text{pred}}^{\text{(inv)}}[i,j] - \mathbf{y}_{\text{pred}}^{\text{(fwd)}}[i,j]|$ | Inversion model image | Forward operator output | Should be $\approx 0$ (numerical precision) |