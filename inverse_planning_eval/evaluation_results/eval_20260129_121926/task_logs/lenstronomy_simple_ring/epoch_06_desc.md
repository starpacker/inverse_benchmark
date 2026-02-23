## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Gravitational Lens Modeling via PSO-MCMC",
  "algorithm_family": "Hybrid Metaheuristic-Bayesian Inference",
  "summary": "Parametric gravitational lens model fitting using Particle Swarm Optimization for initial exploration followed by Markov Chain Monte Carlo for posterior sampling, minimizing chi-squared residuals between observed and model lensed images."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\theta}_{\text{source}}, \boldsymbol{\theta}_{\text{lens\_light}}} \frac{1}{2} \sum_{i,j} \left( \frac{I^{\text{obs}}_{i,j} - I^{\text{model}}_{i,j}(\boldsymbol{\theta})}{\sigma_{\text{bkg}}} \right)^2$$
subject to box constraints on all parameters.

**data_fidelity_term**:
$$\mathcal{D}(\boldsymbol{\theta}) = \frac{1}{2\sigma_{\text{bkg}}^2} \| \mathbf{I}^{\text{obs}} - \mathbf{I}^{\text{model}}(\boldsymbol{\theta}) \|_F^2$$

where $\|\cdot\|_F$ denotes the Frobenius norm (L2 norm of flattened image), and the model image is:
$$I^{\text{model}}_{i,j} = \text{PSF} \ast \left[ S(\boldsymbol{\beta}(\boldsymbol{\theta}_{\text{lens}}); \boldsymbol{\theta}_{\text{source}}) + L(\mathbf{x}; \boldsymbol{\theta}_{\text{lens\_light}}) \right]$$

**regularization_term**: None (purely likelihood-based fitting)

**constraints**:
$$\mathcal{C} = \mathbb{I}_{[\theta_E^{\min}, \theta_E^{\max}]}(\theta_E) \cdot \mathbb{I}_{[e_1^{\min}, e_1^{\max}]}(e_1) \cdot \ldots$$

Box constraints for each parameter (see Section 2.4 for bounds).

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 0.7 | Einstein radius of SIE lens (arcsec) |
| $e_1^{\text{lens}}$ | scalar | float64 | 0.0 | First ellipticity component of SIE |
| $e_2^{\text{lens}}$ | scalar | float64 | 0.0 | Second ellipticity component of SIE |
| $x_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center x-coordinate (arcsec) |
| $y_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center y-coordinate (arcsec) |
| $\gamma_1$ | scalar | float64 | 0.0 | First external shear component |
| $\gamma_2$ | scalar | float64 | 0.0 | Second external shear component |
| $A^{\text{src}}$ | scalar | float64 | 16.0 | Source amplitude |
| $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.2 | Source Sersic half-light radius |
| $n_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | scalar | float64 | 0.0 | Source first ellipticity component |
| $e_2^{\text{src}}$ | scalar | float64 | 0.0 | Source second ellipticity component |
| $x_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center x-coordinate |
| $y_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center y-coordinate |
| $A^{\text{ll}}$ | scalar | float64 | 16.0 | Lens light amplitude |
| $R_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 0.5 | Lens light Sersic half-light radius |
| $n_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 2.0 | Lens light Sersic index |
| $e_1^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light first ellipticity |
| $e_2^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light second ellipticity |
| $x_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center x-coordinate |
| $y_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center y-coordinate |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| N/A | — | — | — | No explicit dual variables (non-convex optimization) |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}^{(k)}$ | $(N_{\text{particles}}, N_{\text{params}})$ | float64 | Gaussian perturbation from init | PSO particle positions at iteration $k$ |
| $\mathbf{V}^{(k)}$ | $(N_{\text{particles}}, N_{\text{params}})$ | float64 | Zero or small random | PSO particle velocities |
| $\mathbf{P}_{\text{best}}$ | $(N_{\text{particles}}, N_{\text{params}})$ | float64 | $\mathbf{X}^{(0)}$ | Personal best positions |
| $\mathbf{g}_{\text{best}}$ | $(N_{\text{params}},)$ | float64 | Best of $\mathbf{X}^{(0)}$ | Global best position |
| $f_{\text{pbest}}$ | $(N_{\text{particles}},)$ | float64 | $\chi^2(\mathbf{X}^{(0)})$ | Personal best fitness values |
| $f_{\text{gbest}}$ | scalar | float64 | $\min(f_{\text{pbest}})$ | Global best fitness |
| $\mathbf{W}^{(k)}$ | $(N_{\text{walkers}}, N_{\text{params}})$ | float64 | PSO result + perturbation | MCMC walker positions |
| $\ln \mathcal{L}^{(k)}$ | $(N_{\text{walkers}},)$ | float64 | Computed | Log-likelihood for each walker |
| $\boldsymbol{\beta}$ | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | float64 | Computed | Source plane coordinates via lens equation |
| $\mathbf{I}^{\text{model}}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Computed | Forward model image |
| $\mathbf{R}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Computed | Residual image |

### 2.4 Constants

| Symbol | Value | Dtype | Description |
|--------|-------|-------|-------------|
| $N_{\text{pix}}$ | 60 | int | Image dimension |
| $\Delta_{\text{pix}}$ | 0.05 | float64 | Pixel scale (arcsec/pixel) |
| $\sigma_{\text{bkg}}$ | 0.005 | float64 | Background RMS noise (flux units) |
| $t_{\text{exp}}$ | 500.0 | float64 | Exposure time (seconds) |
| $\text{FWHM}_{\text{PSF}}$ | 0.05 | float64 | PSF full-width half-maximum (arcsec) |
| $N_{\text{particles}}$ | 50 | int | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | 10 | int | PSO iterations |
| $N_{\text{walkers}}$ | 50 | int | Number of MCMC walkers |
| $N_{\text{burn}}$ | 10 | int | MCMC burn-in steps |
| $N_{\text{run}}$ | 10 | int | MCMC production steps |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | 1.0 | float64 | PSO initialization scale |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | 0.1 | float64 | MCMC proposal scale |
| $\theta_E^{\min}, \theta_E^{\max}$ | 0.01, 10.0 | float64 | Einstein radius bounds |
| $e_{1,2}^{\min}, e_{1,2}^{\max}$ | -0.5, 0.5 | float64 | Ellipticity bounds |
| $\gamma_{1,2}^{\min}, \gamma_{1,2}^{\max}$ | -0.2, 0.2 | float64 | Shear bounds |
| $x_c^{\min}, x_c^{\max}$ | -10.0, 10.0 | float64 | Center position bounds |
| $R_{\text{sersic}}^{\min}, R_{\text{sersic}}^{\max}$ | 0.001, 10.0 | float64 | Sersic radius bounds |
| $n_{\text{sersic}}^{\min}, n_{\text{sersic}}^{\max}$ | 0.5, 5.0 | float64 | Sersic index bounds |
| $A^{\min}, A^{\max}$ | 0.0, 100.0 | float64 | Amplitude bounds |
| $\text{truncation}_{\text{PSF}}$ | 3 | int | PSF truncation radius in FWHM units |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{I}^{\text{obs}}$ | $(60, 60)$ | float64 | Observed (simulated) image with noise |
| $\mathbf{T}$ | $(2, 2)$ | float64 | Pixel-to-angle transformation matrix |
| $(\alpha_0, \delta_0)$ | $(2,)$ | float64 | RA/Dec at pixel (0,0) |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Set random seed | $\text{np.random.seed}(42)$ | seed=42 | RNG state | Reproducibility required |
| P2 | Compute transformation matrix | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}=0.05$ | $\mathbf{T} \in \mathbb{R}^{2 \times 2}$ | Standard coordinate convention |
| P3 | Compute image center in pixels | $c_x = c_y = \frac{N_{\text{pix}} - 1}{2} = 29.5$ | $N_{\text{pix}}=60$ | $c_x, c_y$ scalars | Image centered at origin |
| P4 | Compute RA at (0,0) | $\alpha_0 = -(\mathbf{T}_{00} \cdot c_x + \mathbf{T}_{01} \cdot c_y) = -(-0.05 \cdot 29.5 + 0) = 1.475$ | $\mathbf{T}, c_x, c_y$ | $\alpha_0 = 1.475$ arcsec | — |
| P5 | Compute Dec at (0,0) | $\delta_0 = -(\mathbf{T}_{10} \cdot c_x + \mathbf{T}_{11} \cdot c_y) = -(0 + 0.05 \cdot 29.5) = -1.475$ | $\mathbf{T}, c_x, c_y$ | $\delta_0 = -1.475$ arcsec | — |
| P6 | Create Gaussian PSF kernel | $\text{PSF}(r) = \exp\left(-\frac{r^2}{2\sigma_{\text{PSF}}^2}\right)$, where $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} \approx 0.0212$ | FWHM=0.05, truncation=3 | PSF kernel array | PSF normalized |
| P7 | Generate noise-free model image | $\mathbf{I}^{\text{true}} = \text{ImageModel.image}(\boldsymbol{\theta}^{\text{true}}_{\text{lens}}, \boldsymbol{\theta}^{\text{true}}_{\text{source}}, \boldsymbol{\theta}^{\text{true}}_{\text{ll}})$ | True parameters, PSF | $\mathbf{I}^{\text{true}} \in \mathbb{R}^{60 \times 60}$ | Forward model exact |
| P8 | Convert to photon counts | $\mathbf{C}^{\text{true}} = \mathbf{I}^{\text{true}} \cdot t_{\text{exp}}$ | $\mathbf{I}^{\text{true}}$, $t_{\text{exp}}=500$ | $\mathbf{C}^{\text{true}} \in \mathbb{R}^{60 \times 60}$ | — |
| P9 | Clip negative counts | $\mathbf{C}^{\text{clipped}}_{i,j} = \max(\mathbf{C}^{\text{true}}_{i,j}, 0)$ | $\mathbf{C}^{\text{true}}$ | $\mathbf{C}^{\text{clipped}} \geq 0$ | Physical constraint |
| P10 | Apply Poisson noise | $\mathbf{C}^{\text{poisson}}_{i,j} \sim \text{Poisson}(\mathbf{C}^{\text{clipped}}_{i,j})$ | $\mathbf{C}^{\text{clipped}}$ | $\mathbf{C}^{\text{poisson}} \in \mathbb{Z}_{\geq 0}^{60 \times 60}$ | Shot noise model |
| P11 | Convert back to flux | $\mathbf{I}^{\text{poisson}} = \mathbf{C}^{\text{poisson}} / t_{\text{exp}}$ | $\mathbf{C}^{\text{poisson}}$, $t_{\text{exp}}$ | $\mathbf{I}^{\text{poisson}} \in \mathbb{R}^{60 \times 60}$ | — |
| P12 | Generate background noise | $\mathbf{N}^{\text{bkg}}_{i,j} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$ | $\sigma_{\text{bkg}}=0.005$ | $\mathbf{N}^{\text{bkg}} \in \mathbb{R}^{60 \times 60}$ | i.i.d. Gaussian |
| P13 | Create observed image | $\mathbf{I}^{\text{obs}} = \mathbf{I}^{\text{poisson}} + \mathbf{N}^{\text{bkg}}$ | $\mathbf{I}^{\text{poisson}}$, $\mathbf{N}^{\text{bkg}}$ | $\mathbf{I}^{\text{obs}} \in \mathbb{R}^{60 \times 60}$ | Additive noise |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $\boldsymbol{\theta}^{(0)}_{\text{lens}}$ | $[\theta_E=0.7, e_1=0, e_2=0, x_c=0, y_c=0, \gamma_1=0, \gamma_2=0]$ | $(7,)$ | float64 | `kwargs_lens_init` |
| 2 | $\boldsymbol{\theta}^{(0)}_{\text{source}}$ | $[A=16, R_s=0.2, n_s=1, e_1=0, e_2=0, x_c=0, y_c=0]$ | $(7,)$ | float64 | `kwargs_source_init` |
| 3 | $\boldsymbol{\theta}^{(0)}_{\text{ll}}$ | $[A=16, R_s=0.5, n_s=2, e_1=0, e_2=0, x_c=0, y_c=0]$ | $(7,)$ | float64 | `kwargs_lens_light_init` |
| 4 | $\boldsymbol{\sigma}$ | $[\sigma_{\theta_E}=0.2, \sigma_{e_1}=0.05, \ldots]$ | $(21,)$ | float64 | `kwargs_*_sigma` |
| 5 | $\boldsymbol{\theta}^{\min}$ | Concatenation of all lower bounds | $(21,)$ | float64 | `kwargs_lower_*` |
| 6 | $\boldsymbol{\theta}^{\max}$ | Concatenation of all upper bounds | $(21,)$ | float64 | `kwargs_upper_*` |
| 7 | $N_{\text{params}}$ | $7 + 7 + 7 = 21$ (unfixed parameters) | scalar | int | Parameter count |
| 8 | $\mathbf{X}^{(0)}$ | $X^{(0)}_{p,d} = \theta^{(0)}_d + \sigma_{\text{scale}}^{\text{PSO}} \cdot \sigma_d \cdot z_{p,d}$, where $z_{p,d} \sim \mathcal{N}(0,1)$ | $(50, 21)$ | float64 | PSO initialization |
| 9 | Clip to bounds | $X^{(0)}_{p,d} \leftarrow \text{clip}(X^{(0)}_{p,d}, \theta^{\min}_d, \theta^{\max}_d)$ | $(50, 21)$ | float64 | Enforce constraints |
| 10 | $\mathbf{V}^{(0)}$ | $\mathbf{0}$ or small random values | $(50, 21)$ | float64 | PSO velocities |
| 11 | Evaluate initial fitness | $f^{(0)}_p = \chi^2(\mathbf{X}^{(0)}_p) = \sum_{i,j} \left(\frac{I^{\text{obs}}_{i,j} - I^{\text{model}}_{i,j}(\mathbf{X}^{(0)}_p)}{\sigma_{\text{bkg}}}\right)^2$ | $(50,)$ | float64 | Fitness computation |
| 12 | $\mathbf{P}_{\text{best}}$ | $\mathbf{X}^{(0)}$ | $(50, 21)$ | float64 | Personal best init |
| 13 | $f_{\text{pbest}}$ | $f^{(0)}$ | $(50,)$ | float64 | Personal best fitness |
| 14 | $p^* = \arg\min_p f^{(0)}_p$ | Index of best particle | scalar | int | Find global best |
| 15 | $\mathbf{g}_{\text{best}}$ | $\mathbf{X}^{(0)}_{p^*}$ | $(21,)$ | float64 | Global best position |
| 16 | $f_{\text{gbest}}$ | $f^{(0)}_{p^*}$ | scalar | float64 | Global best fitness |

### 4.2 ITERATION LOOP

#### PSO Phase

**BEGIN_LOOP** (PSO: $k = 1, \ldots, N_{\text{iter}}^{\text{PSO}} = 10$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 17 | Compute inertia weight | $w^{(k)} = w_{\max} - (w_{\max} - w_{\min}) \cdot \frac{k}{N_{\text{iter}}^{\text{PSO}}}$ | $k$, $w_{\max} \approx 0.9$, $w_{\min} \approx 0.4$ | $w^{(k)} \in \mathbb{R}$ | Linear decay schedule |
| 18 | Sample cognitive random | $r_1^{(k)} \sim U(0,1)^{N_{\text{particles}} \times N_{\text{params}}}$ | — | $r_1^{(k)} \in \mathbb{R}^{50 \times 21}$ | Per-particle, per-dimension |
| 19 | Sample social random | $r_2^{(k)} \sim U(0,1)^{N_{\text{particles}} \times N_{\text{params}}}$ | — | $r_2^{(k)} \in \mathbb{R}^{50 \times 21}$ | Per-particle, per-dimension |
| 20 | Update velocities | $V^{(k)}_{p,d} = w^{(k)} V^{(k-1)}_{p,d} + c_1 r_{1,p,d}^{(k)} (P_{\text{best},p,d} - X^{(k-1)}_{p,d}) + c_2 r_{2,p,d}^{(k)} (g_{\text{best},d} - X^{(k-1)}_{p,d})$ | $\mathbf{V}^{(k-1)}, \mathbf{X}^{(k-1)}, \mathbf{P}_{\text{best}}, \mathbf{g}_{\text{best}}, r_1, r_2$ | $\mathbf{V}^{(k)} \in \mathbb{R}^{50 \times 21}$ | $c_1 = c_2 \approx 1.5$ |
| 21 | Update positions | $X^{(k)}_{p,d} = X^{(k-1)}_{p,d} + V^{(k)}_{p,d}$ | $\mathbf{X}^{(k-1)}, \mathbf{V}^{(k)}$ | $\mathbf{X}^{(k)} \in \mathbb{R}^{50 \times 21}$ | Position update |
| 22 | Clip positions to bounds | $X^{(k)}_{p,d} \leftarrow \text{clip}(X^{(k)}_{p,d}, \theta^{\min}_d, \theta^{\max}_d)$ | $\mathbf{X}^{(k)}, \boldsymbol{\theta}^{\min}, \boldsymbol{\theta}^{\max}$ | $\mathbf{X}^{(k)} \in \mathbb{R}^{50 \times 21}$ | Box constraint projection |

**BEGIN_INNER_LOOP** (Particle evaluation: $p = 1, \ldots, 50$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 23 | Extract lens parameters | $\boldsymbol{\theta}_{\text{lens},p} = [X^{(k)}_{p,0}, \ldots, X^{(k)}_{p,6}]$ | $\mathbf{X}^{(k)}_p$ | $\boldsymbol{\theta}_{\text{lens},p} \in \mathbb{R}^7$ | SIE + SHEAR params |
| 24 | Extract source parameters | $\boldsymbol{\theta}_{\text{source},p} = [X^{(k)}_{p,7}, \ldots, X^{(k)}_{p,13}]$ | $\mathbf{X}^{(k)}_p$ | $\boldsymbol{\theta}_{\text{source},p} \in \mathbb{R}^7$ | Sersic params |
| 25 | Extract lens light parameters | $\boldsymbol{\theta}_{\text{ll},p} = [X^{(k)}_{p,14}, \ldots, X^{(k)}_{p,20}]$ | $\mathbf{X}^{(k)}_p$ | $\boldsymbol{\theta}_{\text{ll},p} \in \mathbb{R}^7$ | Sersic params |
| 26 | [FORWARD] Compute pixel coordinates | For each pixel $(i,j)$: $\alpha_{i,j} = \alpha_0 + \mathbf{T}_{00} \cdot i + \mathbf{T}_{01} \cdot j$, $\delta_{i,j} = \delta_0 + \mathbf{T}_{10} \cdot i + \mathbf{T}_{11} \cdot j$ | $\mathbf{T}, \alpha_0, \delta_0$ | $(\boldsymbol{\alpha}, \boldsymbol{\delta}) \in \mathbb{R}^{60 \times 60} \times \mathbb{R}^{60 \times 60}$ | Image plane coords |
| 27 | [FORWARD] Compute SIE deflection angles | $\hat{\alpha}_{\text{SIE},x} = \theta_E \frac{q}{\sqrt{1-q^2}} \arctan\left(\frac{\sqrt{1-q^2} \tilde{x}}{\psi + q^2 s}\right)$, $\hat{\alpha}_{\text{SIE},y} = \theta_E \frac{q}{\sqrt{1-q^2}} \text{arctanh}\left(\frac{\sqrt{1-q^2} \tilde{y}}{\psi + s}\right)$ where $q = \frac{1-|e|}{1+|e|}$, $|e| = \sqrt{e_1^2 + e_2^2}$, $\psi = \sqrt{q^2(\tilde{x}^2 + s^2) + \tilde{y}^2}$, $s$ is core radius (0), $(\tilde{x}, \tilde{y})$ rotated coordinates | $\boldsymbol{\alpha}, \boldsymbol{\delta}, \boldsymbol{\theta}_{\text{lens},p}$ | $(\hat{\boldsymbol{\alpha}}_{\text{SIE}}, \hat{\boldsymbol{\delta}}_{\text{SIE}}) \in \mathbb{R}^{60 \times 60} \times \mathbb{R}^{60 \times 60}$ | SIE lens model |
| 28 | [FORWARD] Compute shear deflection | $\hat{\alpha}_{\text{shear}} = \gamma_1 (\alpha - \alpha_{\text{ref}}) + \gamma_2 (\delta - \delta_{\text{ref}})$, $\hat{\delta}_{\text{shear}} = \gamma_2 (\alpha - \alpha_{\text{ref}}) - \gamma_1 (\delta - \delta_{\text{ref}})$ | $\boldsymbol{\alpha}, \boldsymbol{\delta}, \gamma_1, \gamma_2$ | $(\hat{\boldsymbol{\alpha}}_{\text{shear}}, \hat{\boldsymbol{\delta}}_{\text{shear}})$ | External shear at ref=(0,0) |
| 29 | [FORWARD] Apply lens equation | $\beta_x = \alpha - \hat{\alpha}_{\text{SIE}} - \hat{\alpha}_{\text{shear}}$, $\beta_y = \delta - \hat{\delta}_{\text{SIE}} - \hat{\delta}_{\text{shear}}$ | $\boldsymbol{\alpha}, \boldsymbol{\delta}, \hat{\boldsymbol{\alpha}}_{\text{total}}, \hat{\boldsymbol{\delta}}_{\text{total}}$ | $(\boldsymbol{\beta}_x, \boldsymbol{\beta}_y) \in \mathbb{R}^{60 \times 60} \times \mathbb{R}^{60 \times 60}$ | Source plane mapping |
| 30 | [FORWARD] Compute lensed source | $I^{\text{src}}_{i,j} = A^{\text{src}} \exp\left(-b_n \left[\left(\frac{r_{i,j}}{R_{\text{sersic}}^{\text{src}}}\right)^{1/n_{\text{sersic}}^{\text{src}}} - 1\right]\right)$ where $r_{i,j} = \sqrt{q_s \tilde{\beta}_{x,i,j}^2 + \tilde{\beta}_{y,i,j}^2/q_s}$, $b_n \approx 1.9992 n - 0.3271$, $(\tilde{\beta}_x, \tilde{\beta}_y)$ centered and rotated by source ellipticity | $\boldsymbol{\beta}, \boldsymbol{\theta}_