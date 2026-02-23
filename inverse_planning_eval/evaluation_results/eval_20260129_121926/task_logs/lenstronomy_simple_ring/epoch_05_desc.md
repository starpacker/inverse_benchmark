## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Gravitational Lens Parameter Estimation via PSO-MCMC
- **algorithm_family**: Bayesian Inference / Metaheuristic Optimization (Particle Swarm Optimization + Markov Chain Monte Carlo)
- **summary**: Estimates gravitational lens mass distribution and light profile parameters by minimizing chi-squared residuals between observed and forward-modeled lensed images using sequential PSO global search followed by MCMC posterior sampling.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \left[ \frac{1}{2} \sum_{i,j} \frac{(d_{ij} - \mathcal{F}(\boldsymbol{\theta})_{ij})^2}{\sigma_{\text{bkg}}^2} \right] \quad \text{subject to} \quad \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$$

where $\boldsymbol{\theta} = (\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\theta}_{\text{source}}, \boldsymbol{\theta}_{\text{lens\_light}})$

### data_fidelity_term
$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2\sigma_{\text{bkg}}^2} \| \mathbf{d} - \mathcal{F}(\boldsymbol{\theta}) \|_2^2 = \frac{1}{2\sigma_{\text{bkg}}^2} \sum_{i=0}^{N_{\text{pix}}-1} \sum_{j=0}^{N_{\text{pix}}-1} (d_{ij} - \mathcal{F}(\boldsymbol{\theta})_{ij})^2$$

This is an **L2 (Gaussian) likelihood** corresponding to chi-squared minimization.

### regularization_term
None explicitly defined in code. Implicit regularization via parameter bounds only.

### constraints
Box constraints on all parameters:
$$\mathcal{I}_{\mathcal{C}}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}} \\ +\infty & \text{otherwise} \end{cases}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 0.7 | Einstein radius of SIE lens [arcsec] |
| $e_1^{\text{lens}}$ | scalar | float64 | 0.0 | First ellipticity component of SIE lens |
| $e_2^{\text{lens}}$ | scalar | float64 | 0.0 | Second ellipticity component of SIE lens |
| $x_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center x-coordinate [arcsec] |
| $y_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center y-coordinate [arcsec] |
| $\gamma_1$ | scalar | float64 | 0.0 | External shear component 1 |
| $\gamma_2$ | scalar | float64 | 0.0 | External shear component 2 |
| $A_{\text{src}}$ | scalar | float64 | 16.0 | Source amplitude |
| $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.2 | Source Sersic half-light radius [arcsec] |
| $n_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | scalar | float64 | 0.0 | Source ellipticity component 1 |
| $e_2^{\text{src}}$ | scalar | float64 | 0.0 | Source ellipticity component 2 |
| $x_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center x [arcsec] |
| $y_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center y [arcsec] |
| $A_{\text{lens\_light}}$ | scalar | float64 | 16.0 | Lens light amplitude |
| $R_{\text{sersic}}^{\text{lens\_light}}$ | scalar | float64 | 0.5 | Lens light Sersic radius [arcsec] |
| $n_{\text{sersic}}^{\text{lens\_light}}$ | scalar | float64 | 2.0 | Lens light Sersic index |
| $e_1^{\text{lens\_light}}$ | scalar | float64 | 0.0 | Lens light ellipticity component 1 |
| $e_2^{\text{lens\_light}}$ | scalar | float64 | 0.0 | Lens light ellipticity component 2 |
| $x_c^{\text{lens\_light}}$ | scalar | float64 | 0.0 | Lens light center x [arcsec] |
| $y_c^{\text{lens\_light}}$ | scalar | float64 | 0.0 | Lens light center y [arcsec] |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| — | — | — | — | No explicit dual variables (constrained optimization handled via bounds) |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}^{(k)}$ | $(N_p, N_{\text{params}})$ | float64 | Random within bounds | PSO particle positions at iteration $k$ |
| $\mathbf{V}^{(k)}$ | $(N_p, N_{\text{params}})$ | float64 | Random | PSO particle velocities |
| $\mathbf{P}_{\text{best}}$ | $(N_p, N_{\text{params}})$ | float64 | $\mathbf{X}^{(0)}$ | Personal best positions per particle |
| $\mathbf{G}_{\text{best}}$ | $(N_{\text{params}},)$ | float64 | Best of $\mathbf{X}^{(0)}$ | Global best position |
| $f_{\text{pbest}}$ | $(N_p,)$ | float64 | $\infty$ | Personal best fitness values |
| $f_{\text{gbest}}$ | scalar | float64 | $\infty$ | Global best fitness value |
| $\mathbf{W}$ | $(N_w, N_{\text{params}})$ | float64 | Gaussian around PSO result | MCMC walker positions |
| $\boldsymbol{\beta}_{\text{src}}$ | $(i,j) \to (\beta_x, \beta_y)$ | float64 | Computed | Source plane coordinates after ray-tracing |
| $\mathbf{I}_{\text{model}}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Computed | Forward model image |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{\text{pix}}$ | 60 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | 0.05 arcsec | Pixel scale |
| $\sigma_{\text{bkg}}$ | 0.005 | Background RMS noise level |
| $t_{\text{exp}}$ | 500.0 s | Exposure time |
| $\text{FWHM}$ | 0.05 arcsec | PSF full width at half maximum |
| $N_p$ | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | 10 | PSO iterations |
| $N_w$ | 50 | Number of MCMC walkers |
| $N_{\text{burn}}$ | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | 10 | MCMC sampling steps |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | 1.0 | PSO sigma scale factor |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | 0.1 | MCMC proposal sigma scale |
| $\text{truncation}$ | 3 | PSF truncation radius in units of $\sigma$ |
| $b_n$ | $\approx 1.9992n - 0.3271$ | Sersic $b_n$ approximation |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{d}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Observed (simulated) lensed image with noise |
| $\mathbf{P}$ | $(N_{\text{psf}}, N_{\text{psf}})$ | float64 | Point spread function kernel |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Define pixel-to-angle transformation | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}$ (scalar) | $\mathbf{T}$ (2×2) |
| 2 | Compute image center in pixel coordinates | $c_x = c_y = \frac{N_{\text{pix}} - 1}{2} = 29.5$ | $N_{\text{pix}}$ (scalar) | $c_x, c_y$ (scalars) |
| 3 | Compute RA/Dec at pixel origin | $\alpha_0 = -(T_{00} c_x + T_{01} c_y) = 1.475$ arcsec; $\delta_0 = -(T_{10} c_x + T_{11} c_y) = -1.475$ arcsec | $\mathbf{T}$ (2×2), $c_x, c_y$ (scalars) | $\alpha_0, \delta_0$ (scalars) |
| 4 | Generate coordinate grid | $\alpha_{ij} = \alpha_0 + T_{00} i + T_{01} j$; $\delta_{ij} = \delta_0 + T_{10} i + T_{11} j$ for $i,j \in [0, N_{\text{pix}}-1]$ | $\mathbf{T}$, $\alpha_0$, $\delta_0$, $N_{\text{pix}}$ | $(\boldsymbol{\alpha}, \boldsymbol{\delta})$ each $(N_{\text{pix}}, N_{\text{pix}})$ |
| 5 | Generate Gaussian PSF kernel | $P_{kl} = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{k^2 + l^2}{2\sigma_{\text{PSF}}^2}\right)$ where $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}\Delta_{\text{pix}}}$ | FWHM, $\Delta_{\text{pix}}$, truncation | $\mathbf{P}$ (kernel array) |
| 6 | Compute noise-free model image | $\mathbf{I}_{\text{true}} = \mathcal{F}(\boldsymbol{\theta}_{\text{true}})$ | True parameters, coordinate grid, PSF | $\mathbf{I}_{\text{true}}$ $(N_{\text{pix}}, N_{\text{pix}})$ |
| 7 | Convert to photon counts | $\mathbf{C} = \mathbf{I}_{\text{true}} \cdot t_{\text{exp}}$; $C_{ij} \leftarrow \max(C_{ij}, 0)$ | $\mathbf{I}_{\text{true}}$, $t_{\text{exp}}$ | $\mathbf{C}$ $(N_{\text{pix}}, N_{\text{pix}})$ |
| 8 | Apply Poisson noise | $\tilde{C}_{ij} \sim \text{Poisson}(C_{ij})$; $\mathbf{I}_{\text{poisson}} = \tilde{\mathbf{C}} / t_{\text{exp}}$ | $\mathbf{C}$, $t_{\text{exp}}$ | $\mathbf{I}_{\text{poisson}}$ $(N_{\text{pix}}, N_{\text{pix}})$ |
| 9 | Add Gaussian background noise | $\mathbf{n} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2 \mathbf{I})$; $\mathbf{d} = \mathbf{I}_{\text{poisson}} + \mathbf{n}$ | $\mathbf{I}_{\text{poisson}}$, $\sigma_{\text{bkg}}$ | $\mathbf{d}$ $(N_{\text{pix}}, N_{\text{pix}})$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize lens parameters | $\boldsymbol{\theta}_{\text{lens}}^{(0)} = (\theta_E^{(0)}, e_1^{(0)}, e_2^{(0)}, x_c^{(0)}, y_c^{(0)}, \gamma_1^{(0)}, \gamma_2^{(0)}) = (0.7, 0, 0, 0, 0, 0, 0)$ | Constants from Section 2.4 | $\boldsymbol{\theta}_{\text{lens}}^{(0)}$ (7 scalars) | Initial guess for lens model |
| 2 | Initialize source parameters | $\boldsymbol{\theta}_{\text{src}}^{(0)} = (A^{(0)}, R^{(0)}, n^{(0)}, e_1^{(0)}, e_2^{(0)}, x_c^{(0)}, y_c^{(0)}) = (16, 0.2, 1, 0, 0, 0, 0)$ | Constants | $\boldsymbol{\theta}_{\text{src}}^{(0)}$ (7 scalars) | Initial guess for source |
| 3 | Initialize lens light parameters | $\boldsymbol{\theta}_{\text{ll}}^{(0)} = (A^{(0)}, R^{(0)}, n^{(0)}, e_1^{(0)}, e_2^{(0)}, x_c^{(0)}, y_c^{(0)}) = (16, 0.5, 2, 0, 0, 0, 0)$ | Constants | $\boldsymbol{\theta}_{\text{ll}}^{(0)}$ (7 scalars) | Initial guess for lens light |
| 4 | Stack full parameter vector | $\boldsymbol{\theta}^{(0)} = [\boldsymbol{\theta}_{\text{lens}}^{(0)}, \boldsymbol{\theta}_{\text{src}}^{(0)}, \boldsymbol{\theta}_{\text{ll}}^{(0)}]$ | Steps 1-3 outputs | $\boldsymbol{\theta}^{(0)}$ $(N_{\text{params}},)$ | $N_{\text{params}} = 21$ total free parameters |
| 5 | Initialize PSO particle positions | $X_{pi}^{(0)} \sim \mathcal{U}(\theta_i^{\text{lower}}, \theta_i^{\text{upper}})$ for $p \in [1, N_p]$, with one particle at $\boldsymbol{\theta}^{(0)}$ | $\boldsymbol{\theta}^{(0)}$, bounds | $\mathbf{X}^{(0)}$ $(N_p, N_{\text{params}})$ | Uniform random initialization within bounds |
| 6 | Initialize PSO velocities | $V_{pi}^{(0)} \sim \mathcal{U}(-|\theta_i^{\text{upper}} - \theta_i^{\text{lower}}|, |\theta_i^{\text{upper}} - \theta_i^{\text{lower}}|) \cdot \sigma_{\text{scale}}$ | Bounds, $\sigma_{\text{scale}}^{\text{PSO}}$ | $\mathbf{V}^{(0)}$ $(N_p, N_{\text{params}})$ | Scaled random velocities |
| 7 | Initialize personal bests | $\mathbf{P}_{\text{best}} = \mathbf{X}^{(0)}$; $f_{\text{pbest},p} = +\infty$ for all $p$ | $\mathbf{X}^{(0)}$ | $\mathbf{P}_{\text{best}}$, $f_{\text{pbest}}$ | Personal best tracking |
| 8 | Initialize global best | $\mathbf{G}_{\text{best}} = \boldsymbol{\theta}^{(0)}$; $f_{\text{gbest}} = +\infty$ | $\boldsymbol{\theta}^{(0)}$ | $\mathbf{G}_{\text{best}}$, $f_{\text{gbest}}$ | Global best tracking |

### 4.2 ITERATION LOOP

**--- BEGIN_LOOP: PSO Optimization (k = 0 to $N_{\text{iter}}^{\text{PSO}}-1$) ---**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 9 | **BEGIN_LOOP: Particle Evaluation (p = 0 to $N_p - 1$)** | — | — | — | Nested loop over particles |
| 10 | Extract particle parameters | $\boldsymbol{\theta}_p = \mathbf{X}^{(k)}[p, :]$ | $\mathbf{X}^{(k)}$ | $\boldsymbol{\theta}_p$ $(N_{\text{params}},)$ | Current particle position |
| 11 | [FORWARD] Ray-trace through SIE lens | $\beta_x = \alpha - \alpha_{\text{SIE}}(\alpha, \delta; \theta_E, e_1, e_2, x_c, y_c)$; $\beta_y = \delta - \delta_{\text{SIE}}(\alpha, \delta; \theta_E, e_1, e_2, x_c, y_c)$ where deflection angles: $\boldsymbol{\alpha}_{\text{SIE}} = \frac{\theta_E}{\sqrt{1-e^2}} \left( \arctan\left(\frac{\sqrt{1-e^2}\tilde{x}}{\psi}\right), \text{arctanh}\left(\frac{\sqrt{1-e^2}\tilde{y}}{\psi + 1 - e^2}\right) \right)$, $\psi = \sqrt{(1-e^2)\tilde{x}^2 + \tilde{y}^2}$, rotated coords: $(\tilde{x}, \tilde{y})$ from $(e_1, e_2)$ | $(\boldsymbol{\alpha}, \boldsymbol{\delta})$, $\boldsymbol{\theta}_{\text{lens},p}$ | $(\boldsymbol{\beta}_x, \boldsymbol{\beta}_y)$ each $(N_{\text{pix}}, N_{\text{pix}})$ | Maps image plane to source plane |
| 12 | [FORWARD] Apply external shear | $\beta_x \leftarrow \beta_x - \gamma_1 \alpha - \gamma_2 \delta$; $\beta_y \leftarrow \beta_y - \gamma_2 \alpha + \gamma_1 \delta$ | $(\boldsymbol{\beta}_x, \boldsymbol{\beta}_y)$, $\gamma_1$, $\gamma_2$, $(\boldsymbol{\alpha}, \boldsymbol{\delta})$ | $(\boldsymbol{\beta}_x, \boldsymbol{\beta}_y)$ updated | Additional deflection from shear |
| 13 | Evaluate source Sersic profile in source plane | $I_{\text{src},ij} = A_{\text{src}} \exp\left( -b_n \left[ \left(\frac{r_{ij}}{R_{\text{sersic}}}\right)^{1/n} - 1 \right] \right)$ where $r_{ij} = \sqrt{q \tilde{\beta}_{x,ij}^2 + \tilde{\beta}_{y,ij}^2/q}$, $q = \frac{1-e}{1+e}$, $e = \sqrt{e_1^2 + e_2^2}$, $(\tilde{\beta}_x, \tilde{\beta}_y)$ rotated by $\phi = \frac{1}{2}\arctan(e_2/e_1)$ relative to $(x_c^{\text{src}}, y_c^{\text{src}})$ | $(\boldsymbol{\beta}_x, \boldsymbol{\beta}_y)$, $\boldsymbol{\theta}_{\text{src},p}$ | $\mathbf{I}_{\text{src}}$ $(N_{\text{pix}}, N_{\text{pix}})$ | Sersic profile evaluated at ray-traced positions |
| 14 | Evaluate lens light Sersic profile in image plane | $I_{\text{ll},ij} = A_{\text{ll}} \exp\left( -b_n \left[ \left(\frac{r_{ij}}{R_{\text{sersic}}^{\text{ll}}}\right)^{1/n^{\text{ll}}} - 1 \right] \right)$ with same elliptical radius formula using $(\alpha_{ij}, \delta_{ij})$ relative to $(x_c^{\text{ll}}, y_c^{\text{ll}})$ | $(\boldsymbol{\alpha}, \boldsymbol{\delta})$, $\boldsymbol{\theta}_{\text{ll},p}$ | $\mathbf{I}_{\text{ll}}$ $(N_{\text{pix}}, N_{\text{pix}})$ | Lens galaxy light contribution |
| 15 | [FORWARD] Convolve lensed source with PSF | $\mathbf{I}_{\text{src,conv}} = \mathbf{I}_{\text{src}} \ast \mathbf{P}$ | $\mathbf{I}_{\text{src}}$, $\mathbf{P}$ | $\mathbf{I}_{\text{src,conv}}$ $(N_{\text{pix}}, N_{\text{pix}})$ | 2D convolution via FFT |
| 16 | [FORWARD] Convolve lens light with PSF | $\mathbf{I}_{\text{ll,conv}} = \mathbf{I}_{\text{ll}} \ast \mathbf{P}$ | $\mathbf{I}_{\text{ll}}$, $\mathbf{P}$ | $\mathbf{I}_{\text{ll,conv}}$ $(N_{\text{pix}}, N_{\text{pix}})$ | 2D convolution via FFT |
| 17 | Combine model components | $\mathbf{I}_{\text{model},p} = \mathbf{I}_{\text{src,conv}} + \mathbf{I}_{\text{ll,conv}}$ | $\mathbf{I}_{\text{src,conv}}$, $\mathbf{I}_{\text{ll,conv}}$ | $\mathbf{I}_{\text{model},p}$ $(N_{\text{pix}}, N_{\text{pix}})$ | Total model image |
| 18 | Compute chi-squared fitness | $f_p = \chi^2_p = \sum_{i,j} \frac{(d_{ij} - I_{\text{model},p,ij})^2}{\sigma_{\text{bkg}}^2}$ | $\mathbf{d}$, $\mathbf{I}_{\text{model},p}$, $\sigma_{\text{bkg}}$ | $f_p$ (scalar) | Negative log-likelihood (up to constant) |
| 19 | Update personal best | If $f_p < f_{\text{pbest},p}$: $\mathbf{P}_{\text{best}}[p,:] \leftarrow \boldsymbol{\theta}_p$; $f_{\text{pbest},p} \leftarrow f_p$ | $f_p$, $\boldsymbol{\theta}_p$, $\mathbf{P}_{\text{best}}$, $f_{\text{pbest}}$ | $\mathbf{P}_{\text{best}}$, $f_{\text{pbest}}$ (updated) | Track best per particle |
| 20 | Update global best | If $f_p < f_{\text{gbest}}$: $\mathbf{G}_{\text{best}} \leftarrow \boldsymbol{\theta}_p$; $f_{\text{gbest}} \leftarrow f_p$ | $f_p$, $\boldsymbol{\theta}_p$, $\mathbf{G}_{\text{best}}$, $f_{\text{gbest}}$ | $\mathbf{G}_{\text{best}}$, $f_{\text{gbest}}$ (updated) | Track global best |
| 21 | **END_LOOP: Particle Evaluation** | — | — | — | — |
| 22 | Update particle velocities | $V_{pi}^{(k+1)} = \omega V_{pi}^{(k)} + c_1 r_1 (P_{\text{best},pi} - X_{pi}^{(k)}) + c_2 r_2 (G_{\text{best},i} - X_{pi}^{(k)})$ where $\omega \approx 0.7$ (inertia), $c_1, c_2 \approx 1.5$ (cognitive/social), $r_1, r_2 \sim \mathcal{U}(0,1)$ | $\mathbf{V}^{(k)}$, $\mathbf{X}^{(k)}$, $\mathbf{P}_{\text{best}}$, $\mathbf{G}_{\text{best}}$ | $\mathbf{V}^{(k+1)}$ $(N_p, N_{\text{params}})$ | Standard PSO velocity update |
| 23 | Update particle positions | $X_{pi}^{(k+1)} = X_{pi}^{(k)} + V_{pi}^{(k+1)}$ | $\mathbf{X}^{(k)}$, $\mathbf{V}^{(k+1)}$ | $\mathbf{X}^{(k+1)}$ $(N_p, N_{\text{params}})$ | Position update |
| 24 | Apply box constraints (projection) | $X_{pi}^{(k+1)} \leftarrow \text{clip}(X_{pi}^{(k+1)}, \theta_i^{\text{lower}}, \theta_i^{\text{upper}})$ | $\mathbf{X}^{(k+1)}$, bounds | $\mathbf{X}^{(k+1)}$ (clipped) | Enforce feasibility |

**--- END_LOOP: PSO Optimization ---**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 25 | Extract PSO result | $\boldsymbol{\theta}^{\text{PSO}} = \mathbf{G}_{\text{best}}$ | $\mathbf{G}_{\text{best}}$ | $\boldsymbol{\theta}^{\text{PSO}}$ $(N_{\text{params}},)$ | Best parameters from PSO |
| 26 | Initialize MCMC walkers | $W_{w,i}^{(0)} = \theta_i^{\text{PSO}} + \sigma_i \cdot \sigma_{\text{scale}}^{\text{MCMC}} \cdot \mathcal{N}(0,1)$ for $w \in [1, N_w]$ | $\boldsymbol{\theta}^{\text{PSO}}$, $\boldsymbol{\sigma}$, $\sigma_{\text{scale}}^{\text{MCMC}}$ | $\mathbf{W}^{(0)}$ $(N_w, N_{\text{params}})$ | Gaussian ball around PSO solution |

**--- BEGIN_LOOP: MCMC Burn-in (t = 0 to $