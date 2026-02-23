## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Shapelet-Based Linear Inversion for Gravitational Lens Source Reconstruction
- **algorithm_family**: Linear Least Squares / Weighted Least Squares Inversion
- **summary**: Reconstructs a background source from a gravitationally lensed observation by solving a linear inverse problem where the source is parameterized as a linear combination of shapelet basis functions.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\mathbf{a}} = \arg\min_{\mathbf{a}} \| \mathbf{W}^{1/2}(\mathbf{y} - \boldsymbol{\Phi}\mathbf{a}) \|_2^2$$

Equivalently in weighted least squares form:
$$\hat{\mathbf{a}} = \arg\min_{\mathbf{a}} (\mathbf{y} - \boldsymbol{\Phi}\mathbf{a})^\top \mathbf{W} (\mathbf{y} - \boldsymbol{\Phi}\mathbf{a})$$

### data_fidelity_term
$$\mathcal{D}(\mathbf{a}) = \sum_{i=1}^{N_{\text{pix}}} w_i \left( y_i - [\boldsymbol{\Phi}\mathbf{a}]_i \right)^2$$

where:
- $y_i$: observed pixel value at pixel $i$
- $[\boldsymbol{\Phi}\mathbf{a}]_i$: predicted pixel value from forward model
- $w_i = 1/\sigma_i^2$: inverse variance weight for pixel $i$
- $\sigma_i^2 = \sigma_{\text{bkg}}^2 + \max(y_i, 0)/t_{\text{exp}}$: total variance (background + Poisson)

### regularization_term
None (unregularized linear least squares)

### constraints
None

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{a}$ | `primal` | $(N_{\text{coeff}},)$ | float64 | Solved directly via WLS | Shapelet coefficient vector |

where $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}$ | `observations` | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Simulated noisy lensed image | Observed image data |
| $\mathbf{y}_{\text{clean}}$ | `observations` | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Forward simulation | Noise-free simulated image |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\boldsymbol{\Phi}$ | `auxiliary` | $(N_{\text{pix}}^2, N_{\text{coeff}})$ | float64 | Constructed from basis evaluation | Design matrix (forward operator) |
| $\mathbf{W}$ | `auxiliary` | $(N_{\text{pix}}^2, N_{\text{pix}}^2)$ | float64 | Diagonal from noise model | Weight matrix (diagonal) |
| $\hat{\mathbf{y}}$ | `auxiliary` | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | $\boldsymbol{\Phi}\hat{\mathbf{a}}$ | Reconstructed model image |
| $\mathbf{C}_{\mathbf{a}}$ | `auxiliary` | $(N_{\text{coeff}}, N_{\text{coeff}})$ | float64 | $(\boldsymbol{\Phi}^\top\mathbf{W}\boldsymbol{\Phi})^{-1}$ | Parameter covariance matrix |

### Constants — Imaging Configuration
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $N_{\text{pix}}$ | `constants` | scalar | int | 100 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | `constants` | scalar | float64 | 0.05 arcsec | Pixel scale |
| $\sigma_{\text{bkg}}$ | `constants` | scalar | float64 | 0.05 | Background RMS noise (counts/s) |
| $t_{\text{exp}}$ | `constants` | scalar | float64 | 100 s | Exposure time |
| $\text{FWHM}$ | `constants` | scalar | float64 | 0.1 arcsec | PSF full width half maximum |
| $\text{PSF}_{\text{trunc}}$ | `constants` | scalar | int | 3 | PSF truncation in units of $\sigma$ |
| $f_{\text{supersample}}$ | `constants` | scalar | int | 1 | Supersampling factor |
| $r_{\text{seed}}$ | `constants` | scalar | int | 42 | Random seed for noise generation |

### Constants — Lens Model Parameters
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\theta_E$ | `constants` | scalar | float64 | 1.0 arcsec | Einstein radius (SIE) |
| $e_1^{\text{lens}}$ | `constants` | scalar | float64 | 0.1 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | `constants` | scalar | float64 | -0.1 | Lens ellipticity component 2 |
| $x_c^{\text{lens}}$ | `constants` | scalar | float64 | 0.0 arcsec | Lens center x |
| $y_c^{\text{lens}}$ | `constants` | scalar | float64 | 0.0 arcsec | Lens center y |
| $\gamma_1$ | `constants` | scalar | float64 | 0.05 | External shear component 1 |
| $\gamma_2$ | `constants` | scalar | float64 | 0.01 | External shear component 2 |

### Constants — True Source Parameters (Simulation Only)
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $A_1$ | `constants` | scalar | float64 | 200 | Amplitude of Sersic ellipse |
| $R_{s,1}$ | `constants` | scalar | float64 | 0.3 arcsec | Sersic radius (component 1) |
| $n_{s,1}$ | `constants` | scalar | float64 | 1 | Sersic index (component 1) |
| $e_1^{s,1}$ | `constants` | scalar | float64 | 0.1 | Source ellipticity e1 (comp 1) |
| $e_2^{s,1}$ | `constants` | scalar | float64 | 0.1 | Source ellipticity e2 (comp 1) |
| $x_c^{s,1}$ | `constants` | scalar | float64 | 0.1 arcsec | Source center x (component 1) |
| $y_c^{s,1}$ | `constants` | scalar | float64 | 0.1 arcsec | Source center y (component 1) |
| $A_2$ | `constants` | scalar | float64 | 100 | Amplitude of Sersic (comp 2) |
| $R_{s,2}$ | `constants` | scalar | float64 | 0.1 arcsec | Sersic radius (component 2) |
| $n_{s,2}$ | `constants` | scalar | float64 | 2 | Sersic index (component 2) |
| $x_c^{s,2}$ | `constants` | scalar | float64 | -0.2 arcsec | Source center x (component 2) |
| $y_c^{s,2}$ | `constants` | scalar | float64 | 0.0 arcsec | Source center y (component 2) |

### Constants — Shapelet Reconstruction Parameters
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $n_{\max}$ | `constants` | scalar | int | 8 | Maximum shapelet order |
| $\beta$ | `constants` | scalar | float64 | 0.2 arcsec | Shapelet scale parameter |
| $x_c^{\text{shp}}$ | `constants` | scalar | float64 | 0.0 arcsec | Shapelet center x |
| $y_c^{\text{shp}}$ | `constants` | scalar | float64 | 0.0 arcsec | Shapelet center y |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula / Operation | Inputs | Outputs |
|------|------|---------------------|--------|---------|
| 1 | Set random seed | `np.random.seed(42)` | $r_{\text{seed}}=42$ | RNG state |
| 2 | Build transformation matrix | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}$ | $\mathbf{T} \in \mathbb{R}^{2\times2}$ |
| 3 | Compute image center | $c_x = c_y = \frac{N_{\text{pix}}-1}{2} = 49.5$ | $N_{\text{pix}}$ | $c_x, c_y$ (scalar) |
| 4 | Compute RA/Dec at origin | $\alpha_0 = -(T_{00}c_x + T_{01}c_y)$, $\delta_0 = -(T_{10}c_x + T_{11}c_y)$ | $\mathbf{T}, c_x, c_y$ | $\alpha_0=2.475$, $\delta_0=-2.475$ arcsec |
| 5 | Initialize ImageData class | Create data container with zero image | $\sigma_{\text{bkg}}, t_{\text{exp}}, \alpha_0, \delta_0, \mathbf{T}$ | `data_class` object |
| 6 | Configure Gaussian PSF | $\text{PSF}(r) = \frac{1}{2\pi\sigma_{\text{PSF}}^2}\exp\left(-\frac{r^2}{2\sigma_{\text{PSF}}^2}\right)$, where $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$ | $\text{FWHM}=0.1$, $\Delta_{\text{pix}}$, $\text{trunc}=3$ | `psf_class` object |
| 7 | Initialize SIE lens model | $\boldsymbol{\alpha}_{\text{SIE}}(\boldsymbol{\theta}) = \theta_E \frac{\sqrt{q}}{f(q)} \begin{pmatrix} \arctan\left(\frac{f(q)\theta_1}{\psi}\right) \\ \text{arctanh}\left(\frac{f(q)\theta_2}{\psi + q^2}\right) \end{pmatrix}$ | $\theta_E, e_1, e_2, x_c, y_c$ | Deflection field |
| 8 | Initialize external shear | $\boldsymbol{\alpha}_{\text{shear}}(\boldsymbol{\theta}) = \begin{pmatrix} \gamma_1\theta_1 + \gamma_2\theta_2 \\ \gamma_2\theta_1 - \gamma_1\theta_2 \end{pmatrix}$ | $\gamma_1, \gamma_2$ | Shear deflection |
| 9 | Initialize true source model | Sersic profile: $I(r) = A \exp\left(-b_n\left[\left(\frac{r}{R_s}\right)^{1/n_s} - 1\right]\right)$ | Source parameters | `source_model_class_true` |
| 10 | Simulate clean lensed image | $\mathbf{y}_{\text{clean}} = \mathbf{P} \ast \mathcal{L}[S_{\text{true}}]$ | Lens model, source model, PSF | $\mathbf{y}_{\text{clean}} \in \mathbb{R}^{100\times100}$ |
| 11 | Generate Poisson noise | $\mathbf{n}_{\text{Poisson}} = \frac{\text{Poisson}(\max(\mathbf{y}_{\text{clean}} \cdot t_{\text{exp}}, 0))}{t_{\text{exp}}} - \mathbf{y}_{\text{clean}}$ | $\mathbf{y}_{\text{clean}}, t_{\text{exp}}$ | $\mathbf{n}_{\text{Poisson}} \in \mathbb{R}^{100\times100}$ |
| 12 | Generate background noise | $\mathbf{n}_{\text{bkg}} = \sigma_{\text{bkg}} \cdot \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ | $\sigma_{\text{bkg}}=0.05$ | $\mathbf{n}_{\text{bkg}} \in \mathbb{R}^{100\times100}$ |
| 13 | Construct noisy observation | $\mathbf{y} = \mathbf{y}_{\text{clean}} + \mathbf{n}_{\text{bkg}} + \mathbf{n}_{\text{Poisson}}$ | $\mathbf{y}_{\text{clean}}, \mathbf{n}_{\text{bkg}}, \mathbf{n}_{\text{Poisson}}$ | $\mathbf{y} \in \mathbb{R}^{100\times100}$ |
| 14 | Update data class | Store $\mathbf{y}$ in data container | $\mathbf{y}$ | Updated `data_class` |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Define shapelet basis | $B_{n_1,n_2}(x,y;\beta) = \phi_{n_1}\left(\frac{x-x_c^{\text{shp}}}{\beta}\right)\phi_{n_2}\left(\frac{y-y_c^{\text{shp}}}{\beta}\right)$ where $\phi_n(\xi) = \frac{1}{\sqrt{2^n n! \sqrt{\pi}}} H_n(\xi) e^{-\xi^2/2}$ | $n_{\max}=8, \beta=0.2, x_c^{\text{shp}}=0, y_c^{\text{shp}}=0$ | Basis function set |
| I.2 | Compute number of coefficients | $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2} = \frac{9 \times 10}{2} = 45$ | $n_{\max}=8$ | $N_{\text{coeff}}=45$ |
| I.3 | Initialize reconstruction source model | Create `LightModel` with `SHAPELETS` profile | $n_{\max}, \beta, x_c^{\text{shp}}, y_c^{\text{shp}}$ | `source_model_class_reconstruct` |
| I.4 | Initialize ImageLinearFit | Create linear fitting object combining data, PSF, lens, source models | `data_class`, `psf_class`, `lens_model_class`, `source_model_class_reconstruct` | `imageLinearFit` object |

### 4.2 ITERATION LOOP

**Note**: This algorithm performs a **direct linear solve** (non-iterative). The solution is computed in closed form.

--- BEGIN_SOLVE ---

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| S.1 | Construct design matrix | For $j = 1, \ldots, N_{\text{coeff}}$: $\boldsymbol{\Phi}_{:,j} = \text{vec}\left(\mathbf{P} \ast \mathcal{L}[B_j]\right)$ where $\mathcal{L}$ is ray-tracing through lens | Shapelet basis $\{B_j\}$, lens model, PSF | $\boldsymbol{\Phi} \in \mathbb{R}^{N_{\text{pix}}^2 \times N_{\text{coeff}}}$ | Each column is forward model of one basis function |
| S.2 | Construct weight matrix | $W_{ii} = \frac{1}{\sigma_i^2}$ where $\sigma_i^2 = \sigma_{\text{bkg}}^2 + \frac{\max(y_i, 0)}{t_{\text{exp}}}$ | $\mathbf{y}, \sigma_{\text{bkg}}, t_{\text{exp}}$ | $\mathbf{W} \in \mathbb{R}^{N_{\text{pix}}^2 \times N_{\text{pix}}^2}$ (diagonal) | Inverse variance weighting |
| S.3 | Form normal equations | $\mathbf{M} = \boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi}$, $\mathbf{b} = \boldsymbol{\Phi}^\top \mathbf{W} \text{vec}(\mathbf{y})$ | $\boldsymbol{\Phi}, \mathbf{W}, \mathbf{y}$ | $\mathbf{M} \in \mathbb{R}^{N_{\text{coeff}} \times N_{\text{coeff}}}$, $\mathbf{b} \in \mathbb{R}^{N_{\text{coeff}}}$ | Gram matrix and projection |
| S.4 | Solve linear system | $\hat{\mathbf{a}} = \mathbf{M}^{-1}\mathbf{b}$ | $\mathbf{M}, \mathbf{b}$ | $\hat{\mathbf{a}} \in \mathbb{R}^{45}$ | Direct solve (not iterative) |
| S.5 | Compute model image | $\hat{\mathbf{y}} = \text{reshape}(\boldsymbol{\Phi}\hat{\mathbf{a}}, (N_{\text{pix}}, N_{\text{pix}}))$ | $\boldsymbol{\Phi}, \hat{\mathbf{a}}$ | $\hat{\mathbf{y}} \in \mathbb{R}^{100\times100}$ | Forward model evaluation |
| S.6 | Compute error map | $\boldsymbol{\sigma} = \sqrt{\sigma_{\text{bkg}}^2 + \frac{\max(\hat{\mathbf{y}}, 0)}{t_{\text{exp}}}}$ (per-pixel) | $\hat{\mathbf{y}}, \sigma_{\text{bkg}}, t_{\text{exp}}$ | $\boldsymbol{\sigma} \in \mathbb{R}^{100\times100}$ | Model-based noise estimate |

--- END_SOLVE ---

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Compute reduced chi-squared | $\chi^2_{\text{red}} = \frac{1}{N_{\text{dof}}} \sum_{i=1}^{N_{\text{pix}}^2} \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}$ where $N_{\text{dof}} = N_{\text{pix}}^2 - N_{\text{coeff}}$ | $\mathbf{y}, \hat{\mathbf{y}}, \boldsymbol{\sigma}$ | $\chi^2_{\text{red}}$ (scalar) |
| F.2 | Extract shapelet coefficients | $\hat{\mathbf{a}} = [\hat{a}_1, \hat{a}_2, \ldots, \hat{a}_{45}]^\top$ | Solution from S.4 | Coefficient vector |
| F.3 | Verify forward operator | $\Delta_{\max} = \max_i \left| [\boldsymbol{\Phi}\hat{\mathbf{a}}]_i - \hat{y}_i \right|$ | $\hat{\mathbf{a}}, \hat{\mathbf{y}}$ | Consistency check (should be $\approx 0$) |

---

## 5. STOPPING CRITERION

- **type**: direct_solve (not applicable — single-step solution)
- **expression**: Solution obtained via closed-form weighted least squares: $\hat{\mathbf{a}} = (\boldsymbol{\Phi}^\top\mathbf{W}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\mathbf{W}\mathbf{y}$
- **parameters**: 
  - `inv_bool = False`: Parameter covariance is NOT returned as inverse (returns $(\boldsymbol{\Phi}^\top\mathbf{W}\boldsymbol{\Phi})^{-1}$ directly)
- **check_frequency**: N/A (non-iterative)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Reduced Chi-Squared | $\chi^2_{\text{red}} = \frac{1}{N_{\text{pix}}^2 - N_{\text{coeff}}} \sum_{i=1}^{N_{\text{pix}}^2} \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}$ | Goodness of fit normalized by degrees of freedom; $\chi^2_{\text{red}} \approx 1$ indicates good fit |
| Residual RMS | $\text{RMS}_{\text{res}} = \sqrt{\frac{1}{N_{\text{pix}}^2}\sum_{i=1}^{N_{\text{pix}}^2}(y_i - \hat{y}_i)^2}$ | Root mean square of residuals |
| Peak Signal-to-Noise Ratio | $\text{PSNR} = 20\log_{10}\left(\frac{\max_i|\hat{y}_i|}{\text{RMS}_{\text{res}}}\right)$ dB | Ratio of peak signal to residual noise in decibels |
| Coefficient Mean | $\bar{a} = \frac{1}{N_{\text{coeff}}}\sum_{j=1}^{N_{\text{coeff}}} \hat{a}_j$ | Mean of reconstructed shapelet coefficients |
| Coefficient Std | $\sigma_a = \sqrt{\frac{1}{N_{\text{coeff}}}\sum_{j=1}^{N_{\text{coeff}}}(\hat{a}_j - \bar{a})^2}$ | Standard deviation of shapelet coefficients |
| Coefficient Max Abs | $\|a\|_\infty = \max_j |\hat{a}_j|$ | Maximum absolute coefficient value |
| Forward Operator Consistency | $\Delta_{\text{fwd}} = \max_i |[\mathcal{F}(\hat{\mathbf{a}})]_i - \hat{y}_i|$ | Verification that forward operator $\mathcal{F}$ reproduces model image |