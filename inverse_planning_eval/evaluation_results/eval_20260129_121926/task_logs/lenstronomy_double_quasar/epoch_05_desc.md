## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Gravitational Lens Modeling via PSO-MCMC Hybrid Optimization
- **algorithm_family**: Hybrid Global Optimization (Particle Swarm Optimization + Markov Chain Monte Carlo)
- **summary**: Reconstructs gravitational lens system parameters by minimizing the chi-squared residual between observed and forward-modeled images using sequential PSO global search followed by MCMC posterior sampling.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta} \in \Omega}{\arg\min} \; \chi^2(\boldsymbol{\theta}) = \underset{\boldsymbol{\theta} \in \Omega}{\arg\min} \sum_{i,j} \frac{\left( d_{ij} - m_{ij}(\boldsymbol{\theta}) \right)^2}{\sigma_{ij}^2}$$

where $\boldsymbol{\theta} = \{\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\theta}_{\text{source}}, \boldsymbol{\theta}_{\text{lens\_light}}, \boldsymbol{\theta}_{\text{ps}}\}$

### data_fidelity_term
$$\chi^2(\boldsymbol{\theta}) = \sum_{i=0}^{N_{\text{pix}}-1} \sum_{j=0}^{N_{\text{pix}}-1} \frac{\left( d_{ij} - m_{ij}(\boldsymbol{\theta}) \right)^2}{\sigma_{\text{bg}}^2}$$

where:
- $d_{ij}$ is the observed noisy image at pixel $(i,j)$
- $m_{ij}(\boldsymbol{\theta})$ is the forward model prediction
- $\sigma_{\text{bg}} = 0.5$ is the background RMS noise

### regularization_term
None (implicit regularization through bounded parameter space)

### constraints
Box constraints on all parameters:
$$\boldsymbol{\theta} \in \Omega = \prod_{k} [\theta_k^{\text{lower}}, \theta_k^{\text{upper}}]$$

With indicator function formulation:
$$\mathcal{I}_\Omega(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta} \in \Omega \\ +\infty & \text{otherwise} \end{cases}$$

Additional constraints:
- Joint source-point source constraint: Source center linked to point source origin
- Lens equation solver constraint: Image positions must satisfy lens equation within tolerance $\delta_{\text{src}} = 0.001$ arcsec

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 1.1 | Einstein radius (arcsec) |
| $\gamma_{\text{EPL}}$ | scalar | float64 | 2.0 | Power-law slope of mass profile |
| $e_1^{\text{lens}}$ | scalar | float64 | 0.0 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | scalar | float64 | 0.0 | Lens ellipticity component 2 |
| $x_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center x-coordinate (arcsec) |
| $y_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center y-coordinate (arcsec) |
| $\gamma_1$ | scalar | float64 | 0.0 | External shear component 1 |
| $\gamma_2$ | scalar | float64 | 0.0 | External shear component 2 |
| $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.03 | Source Sersic half-light radius (arcsec) |
| $n_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | scalar | float64 | 0.0 | Source ellipticity component 1 |
| $e_2^{\text{src}}$ | scalar | float64 | 0.0 | Source ellipticity component 2 |
| $x_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center x (arcsec) |
| $y_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center y (arcsec) |
| $R_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 0.1 | Lens light Sersic radius (arcsec) |
| $n_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 1.0 | Lens light Sersic index |
| $e_1^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light ellipticity component 1 |
| $e_2^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light ellipticity component 2 |
| $x_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center x (arcsec) |
| $y_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center y (arcsec) |
| $\mathbf{x}_{\text{img}}$ | $(N_{\text{img}},)$ | float64 | $\mathbf{x}_{\text{img}}^{\text{true}} + 0.01$ | Point source image x-positions |
| $\mathbf{y}_{\text{img}}$ | $(N_{\text{img}},)$ | float64 | $\mathbf{y}_{\text{img}}^{\text{true}} - 0.01$ | Point source image y-positions |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| — | — | — | — | No explicit dual variables (MCMC samples posterior directly) |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{P}^{(k)}$ | $(N_p, N_{\text{params}})$ | float64 | Random in bounds | PSO particle positions at iteration $k$ |
| $\mathbf{V}^{(k)}$ | $(N_p, N_{\text{params}})$ | float64 | Scaled random | PSO particle velocities |
| $\mathbf{p}_{\text{best}}^{(k)}$ | $(N_p, N_{\text{params}})$ | float64 | Initial positions | Personal best positions per particle |
| $\mathbf{g}_{\text{best}}^{(k)}$ | $(N_{\text{params}},)$ | float64 | Best initial | Global best position |
| $\mathbf{W}^{(m)}$ | $(N_w, N_{\text{params}})$ | float64 | From PSO best | MCMC walker positions at step $m$ |
| $\boldsymbol{\alpha}$ | $(N_{\text{basis}},)$ | float64 | Linear solve | Linear amplitude parameters (marginalized) |
| $\mathbf{m}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Computed | Current model image |
| $\boldsymbol{\beta}_{\text{src}}$ | $(2,)$ | float64 | Lens equation solve | Source plane position from image positions |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $\sigma_{\text{bg}}$ | scalar | float64 | 0.5 | Background RMS noise (counts/s) |
| $t_{\text{exp}}$ | scalar | float64 | 100.0 | Exposure time (seconds) |
| $N_{\text{pix}}$ | scalar | int | 100 | Image dimension in pixels |
| $\Delta_{\text{pix}}$ | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\text{FWHM}$ | scalar | float64 | 0.2 | PSF full-width half-max (arcsec) |
| $N_p$ | scalar | int | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | scalar | int | 10 | PSO iterations |
| $N_{\text{burn}}$ | scalar | int | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | scalar | int | 10 | MCMC sampling steps |
| $r_w$ | scalar | int | 4 | MCMC walker ratio |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | scalar | float64 | 1.0 | PSO sigma scale factor |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | scalar | float64 | 0.1 | MCMC proposal scale factor |
| $\delta_{\text{img}}$ | scalar | float64 | 0.004 | Image position uncertainty (arcsec) |
| $\delta_{\text{src}}$ | scalar | float64 | 0.001 | Source position tolerance (arcsec) |
| $\mathbf{T}$ | $(2,2)$ | float64 | $\begin{pmatrix} -0.05 & 0 \\ 0 & 0.05 \end{pmatrix}$ | Pixel to angle transformation |
| $b_n$ | scalar | float64 | $\approx 1.9992 n - 0.3271$ | Sersic $b_n$ approximation |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{d}$ | $(100, 100)$ | float64 | Observed noisy image (counts/s) |
| $\mathbf{x}_{\text{img}}^{\text{init}}$ | $(N_{\text{img}},)$ | float64 | Initial image x-positions from lens equation solver |
| $\mathbf{y}_{\text{img}}^{\text{init}}$ | $(N_{\text{img}},)$ | float64 | Initial image y-positions from lens equation solver |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Coordinate Grid Generation | $\alpha_{ij} = \alpha_0 + T_{00} \cdot i + T_{01} \cdot j$, $\delta_{ij} = \delta_0 + T_{10} \cdot i + T_{11} \cdot j$ where $\alpha_0 = -T_{00} c_x - T_{01} c_y$, $\delta_0 = -T_{10} c_x - T_{11} c_y$, $c_x = c_y = (N_{\text{pix}}-1)/2 = 49.5$ | $N_{\text{pix}}=100$, $\mathbf{T} \in \mathbb{R}^{2 \times 2}$ | $\boldsymbol{\alpha}, \boldsymbol{\delta} \in \mathbb{R}^{100 \times 100}$ (RA/Dec grids) |
| 2 | PSF Kernel Construction | $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} = \frac{0.2}{2.355}$, $K_{kl} = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{(k-k_c)^2 + (l-l_c)^2}{2(\sigma_{\text{PSF}}/\Delta_{\text{pix}})^2}\right)$ with truncation at $3\sigma$ | FWHM=0.2, $\Delta_{\text{pix}}=0.05$, truncation=3 | $\mathbf{K} \in \mathbb{R}^{n_k \times n_k}$ (normalized PSF kernel) |
| 3 | True Lens Model Setup | EPL parameters: $\{\theta_E=1, \gamma=1.96, x_c=0, y_c=0, e_1=0.07, e_2=-0.03\}$; Shear: $\{\gamma_1=0.01, \gamma_2=0.01\}$ | True parameter values | $\boldsymbol{\theta}_{\text{lens}}^{\text{true}}$ |
| 4 | Lens Equation Solving | Find $\{(\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}})\}$ such that $\boldsymbol{\beta} = \mathbf{x}_{\text{img}} - \boldsymbol{\alpha}_{\text{defl}}(\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}})$ where $\boldsymbol{\beta} = (0.1, 0.3)$ is source position | $\boldsymbol{\beta}_{\text{src}}=(0.1, 0.3)$, $\boldsymbol{\theta}_{\text{lens}}^{\text{true}}$, search window=$5$ arcsec | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}} \in \mathbb{R}^{N_{\text{img}}}$ |
| 5 | Magnification Computation | $\mu_i = \frac{1}{\det(\mathbf{A}_i)}$ where $\mathbf{A}_i = \mathbf{I} - \nabla\boldsymbol{\alpha}_{\text{defl}}|_{(\mathbf{x}_{\text{img}}^i, \mathbf{y}_{\text{img}}^i)}$ | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}, \boldsymbol{\theta}_{\text{lens}}^{\text{true}}$ | $\boldsymbol{\mu} \in \mathbb{R}^{N_{\text{img}}}$ |
| 6 | Point Source Amplitude | $A_i^{\text{ps}} = 100 \cdot |\mu_i|$ | $\boldsymbol{\mu}$ | $\mathbf{A}^{\text{ps}} \in \mathbb{R}^{N_{\text{img}}}$ |
| 7 | Clean Image Simulation | $\mathbf{s}_{\text{clean}} = \mathcal{F}(\boldsymbol{\theta}^{\text{true}})$ (see forward model in Section 4.2) | All true parameters, $\mathbf{K}$, coordinate grids | $\mathbf{s}_{\text{clean}} \in \mathbb{R}^{100 \times 100}$ |
| 8 | Poisson Noise Generation | $\tilde{n}_{ij}^{\text{Poisson}} = \text{Poisson}(\max(s_{ij}^{\text{clean}} \cdot t_{\text{exp}}, 0))$, $n_{ij}^{\text{Poisson}} = \tilde{n}_{ij}^{\text{Poisson}} / t_{\text{exp}} - s_{ij}^{\text{clean}}$ | $\mathbf{s}_{\text{clean}}$, $t_{\text{exp}}=100$ | $\mathbf{n}^{\text{Poisson}} \in \mathbb{R}^{100 \times 100}$ |
| 9 | Gaussian Noise Generation | $n_{ij}^{\text{bg}} = \sigma_{\text{bg}} \cdot z_{ij}$ where $z_{ij} \sim \mathcal{N}(0,1)$ | $\sigma_{\text{bg}}=0.5$ | $\mathbf{n}^{\text{bg}} \in \mathbb{R}^{100 \times 100}$ |
| 10 | Noisy Image Construction | $d_{ij} = s_{ij}^{\text{clean}} + n_{ij}^{\text{Poisson}} + n_{ij}^{\text{bg}}$ | $\mathbf{s}_{\text{clean}}, \mathbf{n}^{\text{Poisson}}, \mathbf{n}^{\text{bg}}$ | $\mathbf{d} \in \mathbb{R}^{100 \times 100}$ |
| 11 | Parameter Bounds Setup | $\boldsymbol{\theta}^{\text{lower}}, \boldsymbol{\theta}^{\text{upper}}$ for each parameter group (see constraints) | Bound specifications | Bound arrays |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize PSO particles | $P_{pi}^{(0)} = \theta_i^{\text{lower}} + U_{pi} \cdot (\theta_i^{\text{upper}} - \theta_i^{\text{lower}})$ where $U_{pi} \sim \text{Uniform}(0,1)$ | $\boldsymbol{\theta}^{\text{lower}}, \boldsymbol{\theta}^{\text{upper}}$, $N_p=50$ | $\mathbf{P}^{(0)} \in \mathbb{R}^{50 \times N_{\text{params}}}$ | Random initialization within bounds |
| 2 | Initialize PSO velocities | $V_{pi}^{(0)} = \sigma_{\text{scale}} \cdot \sigma_i \cdot (2 U_{pi} - 1)$ where $\sigma_i$ is parameter-specific scale | $\boldsymbol{\sigma}$ (from kwargs_*_sigma), $\sigma_{\text{scale}}=1.0$ | $\mathbf{V}^{(0)} \in \mathbb{R}^{50 \times N_{\text{params}}}$ | Scaled random velocities |
| 3 | Evaluate initial fitness | $f_p^{(0)} = \chi^2(\mathbf{P}_p^{(0)})$ for each particle $p \in \{1, \ldots, N_p\}$ | $\mathbf{P}^{(0)}, \mathbf{d}$, forward model | $\mathbf{f}^{(0)} \in \mathbb{R}^{50}$ | Requires $N_p$ forward model evaluations |
| 4 | Initialize personal bests | $\mathbf{p}_{\text{best},p}^{(0)} = \mathbf{P}_p^{(0)}$, $f_{\text{best},p}^{(0)} = f_p^{(0)}$ | $\mathbf{P}^{(0)}, \mathbf{f}^{(0)}$ | $\mathbf{p}_{\text{best}}^{(0)}, \mathbf{f}_{\text{best}}^{(0)}$ | Copy initial values |
| 5 | Initialize global best | $\mathbf{g}_{\text{best}}^{(0)} = \mathbf{p}_{\text{best},p^*}^{(0)}$ where $p^* = \arg\min_p f_{\text{best},p}^{(0)}$ | $\mathbf{p}_{\text{best}}^{(0)}, \mathbf{f}_{\text{best}}^{(0)}$ | $\mathbf{g}_{\text{best}}^{(0)} \in \mathbb{R}^{N_{\text{params}}}$, $f_{\text{global}}^{(0)}$ | Find minimum |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (PSO: $k = 1, \ldots, N_{\text{iter}}^{\text{PSO}}=10$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 6 | Update velocities | $V_{pi}^{(k)} = \omega V_{pi}^{(k-1)} + c_1 r_1 (p_{\text{best},pi}^{(k-1)} - P_{pi}^{(k-1)}) + c_2 r_2 (g_{\text{best},i}^{(k-1)} - P_{pi}^{(k-1)})$ where $\omega=0.72984$, $c_1=c_2=1.49618$, $r_1,r_2 \sim U(0,1)$ | $\mathbf{V}^{(k-1)}, \mathbf{P}^{(k-1)}, \mathbf{p}_{\text{best}}^{(k-1)}, \mathbf{g}_{\text{best}}^{(k-1)}$ | $\mathbf{V}^{(k)}$ | Standard PSO velocity update |
| 7 | Update positions | $\tilde{P}_{pi}^{(k)} = P_{pi}^{(k-1)} + V_{pi}^{(k)}$ | $\mathbf{P}^{(k-1)}, \mathbf{V}^{(k)}$ | $\tilde{\mathbf{P}}^{(k)}$ | Unconstrained update |
| 8 | Apply box constraints | $P_{pi}^{(k)} = \min(\max(\tilde{P}_{pi}^{(k)}, \theta_i^{\text{lower}}), \theta_i^{\text{upper}})$ | $\tilde{\mathbf{P}}^{(k)}, \boldsymbol{\theta}^{\text{lower}}, \boldsymbol{\theta}^{\text{upper}}$ | $\mathbf{P}^{(k)}$ | Projection onto feasible set |
| 9 | Compute deflection angles (EPL) [FORWARD] | $\alpha_x^{\text{EPL}} = \frac{\theta_E}{1-\frac{\gamma-1}{2}} \cdot \frac{x'}{(q^2 x'^2 + y'^2)^{(\gamma-1)/2}}$, $\alpha_y^{\text{EPL}} = \frac{\theta_E}{1-\frac{\gamma-1}{2}} \cdot \frac{y'}{(q^2 x'^2 + y'^2)^{(\gamma-1)/2}}$ where $(x', y')$ are elliptically transformed coordinates | $\boldsymbol{\alpha}, \boldsymbol{\delta}$, $\theta_E, \gamma, e_1, e_2, x_c, y_c$ | $\boldsymbol{\alpha}_x^{\text{EPL}}, \boldsymbol{\alpha}_y^{\text{EPL}} \in \mathbb{R}^{100 \times 100}$ | EPL deflection field |
| 10 | Compute deflection angles (Shear) [FORWARD] | $\alpha_x^{\text{shear}} = \gamma_1 \cdot (\alpha - \alpha_0) + \gamma_2 \cdot (\delta - \delta_0)$, $\alpha_y^{\text{shear}} = \gamma_2 \cdot (\alpha - \alpha_0) - \gamma_1 \cdot (\delta - \delta_0)$ | $\boldsymbol{\alpha}, \boldsymbol{\delta}, \gamma_1, \gamma_2$ | $\boldsymbol{\alpha}_x^{\text{shear}}, \boldsymbol{\alpha}_y^{\text{shear}}$ | Linear shear deflection |
| 11 | Total deflection | $\boldsymbol{\alpha}_x^{\text{tot}} = \boldsymbol{\alpha}_x^{\text{EPL}} + \boldsymbol{\alpha}_x^{\text{shear}}$, $\boldsymbol{\alpha}_y^{\text{tot}} = \boldsymbol{\alpha}_y^{\text{EPL}} + \boldsymbol{\alpha}_y^{\text{shear}}$ | $\boldsymbol{\alpha}_x^{\text{EPL}}, \boldsymbol{\alpha}_y^{\text{EPL}}, \boldsymbol{\alpha}_x^{\text{shear}}, \boldsymbol{\alpha}_y^{\text{shear}}$ | $\boldsymbol{\alpha}_x^{\text{tot}}, \boldsymbol{\alpha}_y^{\text{tot}}$ | Sum deflections |
| 12 | Ray-trace to source plane | $\beta_x = \alpha - \alpha_x^{\text{tot}}$, $\beta_y = \delta - \alpha_y^{\text{tot}}$ | $\boldsymbol{\alpha}, \boldsymbol{\delta}, \boldsymbol{\alpha}_x^{\text{tot}}, \boldsymbol{\alpha}_y^{\text{tot}}$ | $\boldsymbol{\beta}_x, \boldsymbol{\beta}_y \in \mathbb{R}^{100 \times 100}$ | Source plane coordinates |
| 13 | Evaluate source Sersic profile | $I_{ij}^{\text{src}} = A_{\text{src}} \exp\left(-b_{n_s} \left[\left(\frac{r'_{ij}}{R_s}\right)^{1/n_s} - 1\right]\right)$ where $r'_{ij} = \sqrt{q_s^2 (\beta_x - x_c^s)^2 + (\beta_y - y_c^s)^2}$, $b_{n_s} \approx 1.9992 n_s - 0.3271$ | $\boldsymbol{\beta}_x, \boldsymbol{\beta}_y$, source params | $\mathbf{I}^{\text{src}} \in \mathbb{R}^{100 \times 100}$ | Lensed source light |
| 14 | Evaluate lens light Sersic profile | $I_{ij}^{\text{ll}} = A_{\text{ll}} \exp\left(-b_{n_l} \left[\left(\frac{r_{ij}}{R_l}\right)^{1/n_l} - 1\right]\right)$ where $r_{ij} = \sqrt{(\alpha_{ij} - x_c^{ll})^2 + (\delta_{ij} - y_c^{ll})^2}$ | $\boldsymbol{\alpha}, \boldsymbol{\delta}$, lens light params | $\mathbf{I}^{\text{ll}} \in \mathbb{R}^{100 \times 100}$ | Lens galaxy light |
| 15 | Render point sources | $I_{ij}^{\text{ps}} = \sum_{m=1}^{N_{\text{img}}} A_m^{\text{ps}} \cdot \delta(\alpha_{ij} - x_m^{\text{img}}) \cdot \delta(\delta_{ij} - y_m^{\text{img}})$ (subpixel interpolation applied) | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}, \mathbf{A}^{\text{ps}}$ | $\mathbf{I}^{\text{ps}} \in \mathbb{R}^{100 \times 100}$ | Point source contribution |
| 16 | Combine light components | $\mathbf{I}^{\text{total}} = \mathbf{I}^{\text{src}} + \mathbf{I}^{\text{ll}} + \mathbf{I}^{\text{ps}}$ | $\mathbf{I}^{\text{src}}, \mathbf{I}^{\text{ll}}, \mathbf{I}^{\text{ps}}$ | $\mathbf{I}^{\text{total}}$ | Pre-convolution image |
| 17 | PSF convolution [FORWARD] | $m_{ij} = (\mathbf{I}^{\text{total}} * \mathbf{K})_{ij} = \sum_{k,l} I_{i-k,j-l}^{\text{total}} K_{kl}$ | $\mathbf{I}^{\text{total}}, \mathbf{K}$ | $\mathbf{m} \in \mathbb{R}^{100 \times 100}$ | FFT-based convolution |
| 18 | Solve linear amplitudes | $\boldsymbol{\alpha}^* = \arg\min_{\boldsymbol{\alpha}} \|\mathbf{d} - \sum_c \alpha_c \mathbf{B}_c\|_2^2$ with solution $\boldsymbol{\alpha}^* = (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T \mathbf{d}$ | $\mathbf{d}$, basis images $\{\mathbf{B}_c\}$ | $\boldsymbol{\alpha}^*$ | Linear solve for amp parameters |
| 19 | Evaluate chi-squared | $f_p