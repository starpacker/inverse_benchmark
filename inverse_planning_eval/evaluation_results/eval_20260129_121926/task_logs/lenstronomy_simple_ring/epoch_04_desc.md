## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Gravitational Lens Model Fitting via PSO-MCMC Hybrid Optimization
- **algorithm_family**: Bayesian Inference / Metaheuristic-MCMC Hybrid Optimization
- **summary**: A two-stage optimization algorithm that fits parametric gravitational lens models to observed images by first exploring parameter space with Particle Swarm Optimization (PSO), then refining posterior distributions with Markov Chain Monte Carlo (MCMC) sampling, minimizing a chi-squared likelihood under box constraints.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta} \in \Omega} \left[ -\log \mathcal{L}(\boldsymbol{\theta}) \right] = \arg\min_{\boldsymbol{\theta} \in \Omega} \left[ \frac{1}{2} \sum_{i,j} \frac{(y_{ij} - \mathcal{F}(\boldsymbol{\theta})_{ij})^2}{\sigma_{\text{bg}}^2} \right]$$

where $\boldsymbol{\theta} = \{\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\theta}_{\text{source}}, \boldsymbol{\theta}_{\text{lens\_light}}\}$ and $\Omega$ is the feasible parameter box.

### data_fidelity_term
$$\mathcal{D}(\boldsymbol{\theta}) = \frac{1}{2\sigma_{\text{bg}}^2} \|\mathbf{y} - \mathcal{F}(\boldsymbol{\theta})\|_2^2 = \frac{1}{2\sigma_{\text{bg}}^2} \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} (y_{ij} - \mathcal{F}(\boldsymbol{\theta})_{ij})^2$$

where:
- $\mathbf{y} \in \mathbb{R}^{N_x \times N_y}$: observed image
- $\mathcal{F}(\boldsymbol{\theta})$: forward model prediction
- $\sigma_{\text{bg}} = 0.005$: background RMS noise level (homoscedastic approximation)

### regularization_term
None (implicit regularization via box constraints and prior distributions in MCMC)

### constraints
$$\mathbb{I}_{\Omega}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}} \\ +\infty & \text{otherwise} \end{cases}$$

**Lens (SIE) bounds**:
- $\theta_E \in [0.01, 10.0]$
- $e_1, e_2 \in [-0.5, 0.5]$
- $\text{center}_x, \text{center}_y \in [-10, 10]$

**Lens (SHEAR) bounds**:
- $\gamma_1, \gamma_2 \in [-0.2, 0.2]$

**Source (SERSIC_ELLIPSE) bounds**:
- $R_{\text{sersic}} \in [0.001, 10]$
- $n_{\text{sersic}} \in [0.5, 5.0]$
- $e_1, e_2 \in [-0.5, 0.5]$
- $\text{center}_x, \text{center}_y \in [-10, 10]$
- $\text{amp} \in [0, 100]$

**Lens Light (SERSIC_ELLIPSE) bounds**: Same as Source bounds

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 0.7 | Einstein radius of SIE lens (arcsec) |
| $e_1^{\text{lens}}$ | scalar | float64 | 0.0 | First ellipticity component of SIE lens |
| $e_2^{\text{lens}}$ | scalar | float64 | 0.0 | Second ellipticity component of SIE lens |
| $x_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center x-coordinate (arcsec) |
| $y_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center y-coordinate (arcsec) |
| $\gamma_1$ | scalar | float64 | 0.0 | First external shear component |
| $\gamma_2$ | scalar | float64 | 0.0 | Second external shear component |
| $A_{\text{src}}$ | scalar | float64 | 16.0 | Source amplitude |
| $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.2 | Source Sersic half-light radius (arcsec) |
| $n_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | scalar | float64 | 0.0 | Source first ellipticity component |
| $e_2^{\text{src}}$ | scalar | float64 | 0.0 | Source second ellipticity component |
| $x_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center x-coordinate (arcsec) |
| $y_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center y-coordinate (arcsec) |
| $A_{\text{ll}}$ | scalar | float64 | 16.0 | Lens light amplitude |
| $R_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 0.5 | Lens light Sersic half-light radius |
| $n_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 2.0 | Lens light Sersic index |
| $e_1^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light first ellipticity component |
| $e_2^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light second ellipticity component |
| $x_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center x-coordinate |
| $y_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center y-coordinate |

### 2.2 Dual Variables

None (direct optimization, no Lagrangian dual formulation)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}_{\text{particles}}$ | $(N_p, d)$ | float64 | Gaussian around init | PSO particle positions ($N_p=50$, $d=21$) |
| $\mathbf{V}_{\text{particles}}$ | $(N_p, d)$ | float64 | Scaled Gaussian | PSO particle velocities |
| $\mathbf{p}_{\text{best}}$ | $(N_p, d)$ | float64 | Copy of $\mathbf{X}$ | Personal best positions per particle |
| $\mathbf{g}_{\text{best}}$ | $(d,)$ | float64 | Best of $\mathbf{p}$ | Global best position |
| $f_{\text{pbest}}$ | $(N_p,)$ | float64 | $\infty$ | Personal best fitness values |
| $f_{\text{gbest}}$ | scalar | float64 | $\infty$ | Global best fitness value |
| $\mathbf{W}_{\text{walkers}}$ | $(N_w, d)$ | float64 | Ball around PSO result | MCMC walker positions ($N_w=50$) |
| $\mathbf{y}_{\text{pred}}$ | $(N_x, N_y)$ | float64 | Computed | Current model prediction |
| $\mathbf{r}$ | $(N_x, N_y)$ | float64 | Computed | Residual image |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_x$ | 60 | Number of pixels in x-direction |
| $N_y$ | 60 | Number of pixels in y-direction |
| $\Delta_{\text{pix}}$ | 0.05 arcsec | Pixel scale |
| $\sigma_{\text{bg}}$ | 0.005 | Background RMS noise |
| $t_{\text{exp}}$ | 500.0 s | Exposure time |
| $\text{FWHM}$ | 0.05 arcsec | PSF full-width at half-maximum |
| $N_p$ | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | 10 | PSO iterations |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | 1.0 | PSO initialization scale factor |
| $N_w$ | 50 | Number of MCMC walkers |
| $N_{\text{burn}}$ | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | 10 | MCMC production steps |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | 0.1 | MCMC initialization ball radius scale |
| truncation | 3 | PSF truncation radius in units of $\sigma$ |
| supersampling_factor | 1 | No supersampling |
| ra_0 (shear) | 0 | Fixed shear reference RA |
| dec_0 (shear) | 0 | Fixed shear reference Dec |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(60, 60)$ | float64 | Observed (simulated) lensed image with noise |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Coordinate System Setup | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}=0.05$ | $\mathbf{T} \in \mathbb{R}^{2 \times 2}$ |
| 1 | Image Center Computation | $c_x = c_y = \frac{N_x - 1}{2} = 29.5$ | $N_x = 60$ | $c_x, c_y \in \mathbb{R}$ |
| 2 | Reference Coordinate Computation | $\text{RA}_0 = -(\mathbf{T}_{00} c_x + \mathbf{T}_{01} c_y) = 1.475$, $\text{Dec}_0 = -(\mathbf{T}_{10} c_x + \mathbf{T}_{11} c_y) = -1.475$ | $\mathbf{T}, c_x, c_y$ | $\text{RA}_0, \text{Dec}_0 \in \mathbb{R}$ |
| 3 | PSF Generation | $\text{PSF}(r) = \exp\left(-\frac{r^2}{2\sigma_{\text{PSF}}^2}\right)$ where $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$ | FWHM=0.05, type='GAUSSIAN' | PSF kernel (truncated at $3\sigma$) |
| 4 | True Model Image Generation | $\mathbf{I}_{\text{model}} = \mathcal{F}(\boldsymbol{\theta}_{\text{true}})$ | True parameters | $\mathbf{I}_{\text{model}} \in \mathbb{R}^{60 \times 60}$ |
| 5 | Poisson Noise Application | $\mathbf{I}_{\text{counts}} = \max(\mathbf{I}_{\text{model}} \cdot t_{\text{exp}}, 0)$, $\mathbf{I}_{\text{poisson}} = \frac{\text{Poisson}(\mathbf{I}_{\text{counts}})}{t_{\text{exp}}}$ | $\mathbf{I}_{\text{model}}, t_{\text{exp}}=500$ | $\mathbf{I}_{\text{poisson}} \in \mathbb{R}^{60 \times 60}$ |
| 6 | Gaussian Noise Addition | $\mathbf{n}_{\text{bg}} = \sigma_{\text{bg}} \cdot \mathcal{N}(0, \mathbf{I})$ | $\sigma_{\text{bg}}=0.005$ | $\mathbf{n}_{\text{bg}} \in \mathbb{R}^{60 \times 60}$ |
| 7 | Final Observation | $\mathbf{y} = \mathbf{I}_{\text{poisson}} + \mathbf{n}_{\text{bg}}$ | Steps 5,6 outputs | $\mathbf{y} \in \mathbb{R}^{60 \times 60}$ |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Initialize parameter vector | $\boldsymbol{\theta}^{(0)} = \boldsymbol{\theta}_{\text{init}}$ (21 parameters total) | Initial guesses from tables | $\boldsymbol{\theta}^{(0)} \in \mathbb{R}^{21}$ | Concatenation of all model parameters |
| Initialize PSO particles | $\mathbf{X}_i^{(0)} = \boldsymbol{\theta}^{(0)} + \sigma_{\text{scale}} \cdot \boldsymbol{\sigma}_{\text{param}} \cdot \mathcal{N}(0,1)$ for $i=1,\ldots,N_p$ | $\boldsymbol{\theta}^{(0)}, \boldsymbol{\sigma}_{\text{param}}, \sigma_{\text{scale}}=1.0$ | $\mathbf{X} \in \mathbb{R}^{50 \times 21}$ | $\boldsymbol{\sigma}_{\text{param}}$ from sigma tables |
| Clip to bounds | $\mathbf{X}_i^{(0)} = \text{clip}(\mathbf{X}_i^{(0)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$ | $\mathbf{X}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}$ | $\mathbf{X} \in \Omega$ | Enforce box constraints |
| Initialize personal bests | $\mathbf{p}_i = \mathbf{X}_i^{(0)}$, $f_{\text{pbest},i} = \chi^2(\mathbf{X}_i^{(0)})$ | $\mathbf{X}^{(0)}$ | $\mathbf{p}, f_{\text{pbest}}$ | Evaluate initial fitness |
| Initialize global best | $\mathbf{g} = \arg\min_i f_{\text{pbest},i}$, $f_{\text{gbest}} = \min_i f_{\text{pbest},i}$ | $\mathbf{p}, f_{\text{pbest}}$ | $\mathbf{g}, f_{\text{gbest}}$ | Track best solution |

### 4.2 ITERATION LOOP

**BEGIN_LOOP (PSO Stage)**: $k = 1, \ldots, N_{\text{iter}}^{\text{PSO}}$ where $N_{\text{iter}}^{\text{PSO}} = 10$

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Update velocities | $\mathbf{V}_i^{(k)} = \omega \mathbf{V}_i^{(k-1)} + c_1 r_1 (\mathbf{p}_i - \mathbf{X}_i^{(k-1)}) + c_2 r_2 (\mathbf{g} - \mathbf{X}_i^{(k-1)})$ | $\mathbf{V}^{(k-1)}, \mathbf{X}^{(k-1)}, \mathbf{p}, \mathbf{g}$ | $\mathbf{V}^{(k)}$ | $\omega, c_1, c_2$ are PSO coefficients; $r_1, r_2 \sim U(0,1)$ |
| Update positions | $\mathbf{X}_i^{(k)} = \mathbf{X}_i^{(k-1)} + \mathbf{V}_i^{(k)}$ | $\mathbf{X}^{(k-1)}, \mathbf{V}^{(k)}$ | $\mathbf{X}^{(k)}$ | Position update |
| Enforce bounds | $\mathbf{X}_i^{(k)} = \text{clip}(\mathbf{X}_i^{(k)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$ | $\mathbf{X}^{(k)}$ | $\mathbf{X}^{(k)} \in \Omega$ | Box constraint projection |
| Forward model [FORWARD] | $\mathbf{y}_{\text{pred},i} = \mathcal{F}(\mathbf{X}_i^{(k)}) = \text{PSF} * [I_{\text{src}}(\boldsymbol{\beta}(\boldsymbol{\alpha})) + I_{\text{ll}}]$ | $\mathbf{X}_i^{(k)}$ | $\mathbf{y}_{\text{pred},i} \in \mathbb{R}^{60 \times 60}$ | See Forward Model Detail below |
| Compute fitness | $f_i^{(k)} = \chi^2(\mathbf{X}_i^{(k)}) = \sum_{p,q} \frac{(y_{pq} - \mathbf{y}_{\text{pred},i,pq})^2}{\sigma_{\text{bg}}^2}$ | $\mathbf{y}, \mathbf{y}_{\text{pred},i}, \sigma_{\text{bg}}$ | $f_i^{(k)}$ | Chi-squared computation |
| Update personal best | If $f_i^{(k)} < f_{\text{pbest},i}$: $\mathbf{p}_i = \mathbf{X}_i^{(k)}$, $f_{\text{pbest},i} = f_i^{(k)}$ | $f_i^{(k)}, f_{\text{pbest},i}$ | $\mathbf{p}_i, f_{\text{pbest},i}$ | Personal best update |
| Update global best | If $\min_i f_{\text{pbest},i} < f_{\text{gbest}}$: $\mathbf{g} = \arg\min_i f_{\text{pbest},i}$, $f_{\text{gbest}} = \min_i f_{\text{pbest},i}$ | $f_{\text{pbest}}, f_{\text{gbest}}$ | $\mathbf{g}, f_{\text{gbest}}$ | Global best update |

**END_LOOP (PSO Stage)**

**BEGIN_LOOP (MCMC Stage - Burn-in)**: $k = 1, \ldots, N_{\text{burn}}$ where $N_{\text{burn}} = 10$

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Initialize walkers | $\mathbf{W}_i^{(0)} = \mathbf{g} + \sigma_{\text{scale}}^{\text{MCMC}} \cdot \boldsymbol{\sigma}_{\text{param}} \cdot \mathcal{N}(0,1)$ for $i=1,\ldots,N_w$ | $\mathbf{g}, \sigma_{\text{scale}}^{\text{MCMC}}=0.1$ | $\mathbf{W}^{(0)} \in \mathbb{R}^{50 \times 21}$ | Ball initialization around PSO result |
| Propose stretch move | $\mathbf{W}_i^* = \mathbf{W}_j + Z(\mathbf{W}_i^{(k-1)} - \mathbf{W}_j)$ where $Z \sim g(z) \propto \frac{1}{\sqrt{z}}$ for $z \in [1/a, a]$, $a=2$ | $\mathbf{W}^{(k-1)}$, random complementary walker $j$ | $\mathbf{W}_i^*$ | Affine-invariant ensemble sampler |
| Compute log-likelihood | $\log\mathcal{L}(\mathbf{W}_i^*) = -\frac{1}{2}\chi^2(\mathbf{W}_i^*)$ | $\mathbf{W}_i^*$ | $\log\mathcal{L}^*$ | Requires forward model evaluation |
| Accept/Reject | $\mathbf{W}_i^{(k)} = \begin{cases} \mathbf{W}_i^* & \text{with prob } \min(1, Z^{d-1} e^{\log\mathcal{L}^* - \log\mathcal{L}^{(k-1)}}) \\ \mathbf{W}_i^{(k-1)} & \text{otherwise} \end{cases}$ | $\mathbf{W}_i^*, \mathbf{W}_i^{(k-1)}, Z$ | $\mathbf{W}_i^{(k)}$ | Metropolis-Hastings acceptance |

**END_LOOP (MCMC Burn-in)**

**BEGIN_LOOP (MCMC Stage - Production)**: $k = 1, \ldots, N_{\text{run}}$ where $N_{\text{run}} = 10$

(Same steps as burn-in, but samples are stored)

**END_LOOP (MCMC Production)**

---

**Forward Model Detail** [FORWARD]:

The forward operator $\mathcal{F}(\boldsymbol{\theta})$ computes:

$$\mathcal{F}(\boldsymbol{\theta}) = \text{PSF} * \left[ I_{\text{src}}(\boldsymbol{\beta}(\mathbf{x}; \boldsymbol{\theta}_{\text{lens}}); \boldsymbol{\theta}_{\text{src}}) + I_{\text{ll}}(\mathbf{x}; \boldsymbol{\theta}_{\text{ll}}) \right]$$

**Lens equation (SIE + SHEAR)**:
$$\boldsymbol{\beta} = \mathbf{x} - \boldsymbol{\alpha}_{\text{SIE}}(\mathbf{x}) - \boldsymbol{\alpha}_{\text{SHEAR}}(\mathbf{x})$$

**SIE deflection**:
$$\boldsymbol{\alpha}_{\text{SIE}} = \theta_E \frac{\sqrt{q}}{1-q^2} \begin{pmatrix} \arctan\left(\frac{(1-q^2)^{1/2} x'}{\psi + q^2 s}\right) \\ \text{arctanh}\left(\frac{(1-q^2)^{1/2} y'}{\psi + s}\right) \end{pmatrix}$$
where $q = \sqrt{\frac{1-e}{1+e}}$, $e = \sqrt{e_1^2 + e_2^2}$, $\psi = \sqrt{q^2 x'^2 + y'^2}$, $(x', y')$ are rotated coordinates centered on lens

**SHEAR deflection**:
$$\boldsymbol{\alpha}_{\text{SHEAR}} = \begin{pmatrix} \gamma_1 x + \gamma_2 y \\ \gamma_2 x - \gamma_1 y \end{pmatrix}$$

**Sersic profile** (for source and lens light):
$$I_{\text{Sersic}}(R) = A \exp\left[ -b_n \left( \left(\frac{R}{R_{\text{sersic}}}\right)^{1/n} - 1 \right) \right]$$
where $b_n \approx 2n - 1/3 + 4/(405n)$, and $R$ is the elliptical radius:
$$R = \sqrt{q_s x''^2 + y''^2/q_s}$$

**PSF convolution**:
$$(\text{PSF} * I)_{ij} = \sum_{m,n} \text{PSF}_{mn} \cdot I_{i-m, j-n}$$

### 4.3 FINALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Extract best fit | $\boldsymbol{\theta}^* = \arg\max_{\text{chain}} \mathcal{L}(\boldsymbol{\theta})$ | MCMC chain | $\boldsymbol{\theta}^*$ | Maximum likelihood from chain |
| Compute model image | $\mathbf{y}_{\text{model}} = \mathcal{F}(\boldsymbol{\theta}^*)$ | $\boldsymbol{\theta}^*$ | $\mathbf{y}_{\text{model}} \in \mathbb{R}^{60 \times 60}$ | Final forward evaluation |
| Compute residuals | $\mathbf{r} = \mathbf{y} - \mathbf{y}_{\text{model}}$ | $\mathbf{y}, \mathbf{y}_{\text{model}}$ | $\mathbf{r} \in \mathbb{R}^{60 \times 60}$ | Residual image |
| Compute chi-squared | $\chi^2 = \sum_{i,j} \left(\frac{r_{ij}}{\sigma_{\text{bg}}}\right)^2$ | $\mathbf{r}, \sigma_{\text{bg}}$ | $\chi^2 \in \mathbb{R}$ | Goodness of fit |
| Compute reduced chi-squared | $\chi^2_{\text{red}} = \frac{\chi^2}{N_{\text{pix}} - 1} = \frac{\chi^2}{3599}$ | $\chi^2, N_{\text{pix}}=3600$ | $\chi^2_{\text{red}} \in \mathbb{R}$ | Normalized fit quality |

## 5. STOPPING CRITERION

- **type**: iteration_limit (for both PSO and MCMC stages)
- **expression**: 
  - PSO: $k = N_{\text{iter}}^{\text{PSO}} = 10$
  - MCMC burn-in: $k = N_{\text{burn}} = 10$
  - MCMC production: $k = N_{\text{run}} = 10$
- **parameters**: 
  - `n_iterations` (PSO) = 10
  - `n_particles` (PSO) = 50
  - `n_burn` (MCMC) = 10
  - `n_run` (MCMC) = 10
  - `n_walkers` (MCMC) = 50
- **check_frequency**: After each iteration/step

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| Chi-squared | $\chi^2 = \sum_{i,j} \frac{(y_{ij} - \hat{y}_{ij})^2}{\sigma_{\text{bg}}^2}$ | Sum of squared normalized residuals |
| Reduced Chi-squared | $\chi^2_{\text{red}} = \frac{\chi^2}{N_{\text{pix}} - 1}$ where $N_{\text{pix}} = 3600$ | Chi-squared per degree of freedom (approximated) |
| Residual RMS | $\text{RMS}_r = \sqrt{\frac{1}{N_{\text{pix}}} \sum_{i,j} r_{ij}^2} = \text{std}(\mathbf{r})$ | Standard deviation of residual image |
| Parameter Error | $\Delta\theta_k = \theta_k^{\text{fitted}} - \theta_k^{\text{true}}$ | Difference between fitted and true parameter values |

## 7. REQUIRED OUTPUT FIELDS

The following JSON fields MUST appear in downstream output:

```
algorithm_identification:
  - algorithm_name
  - algorithm_family
  - summary

objective_function:
  - full_expression
  - data_fidelity_term
  - regularization_term
  - constraints

variables:
  - primal_variables (list of {symbol, shape, dtype, initialization, description})
  - dual_variables
  - auxiliary_variables
  - constants (list of {symbol, value, description})
  - observations

data_preprocessing:
  - steps (list of {step_order, step_name, formula, inputs, outputs})

initialization:
  - steps (list of {step_name, formula, inputs, outputs, computational_note})

iteration_loop:
  - pso_stage:
    - max_iterations: 10
    - steps (list per iteration)
  - mcmc_burn_stage:
    - max_iterations: 10
    - steps (list per iteration)
  - mcmc_production_stage:
    - max_iterations: 10
    - steps (list per iteration)

finalization:
  - steps (list of {step_name, formula, inputs, outputs})

stopping_criterion:
  - type
  - expression
  -