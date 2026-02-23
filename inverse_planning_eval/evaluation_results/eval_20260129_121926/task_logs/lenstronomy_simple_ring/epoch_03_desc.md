## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Gravitational Lens Modeling via PSO-MCMC Hybrid Optimization
- **algorithm_family**: Bayesian Parameter Estimation with Metaheuristic Initialization
- **summary**: Fits a parametric gravitational lens model (SIE lens + external shear + Sérsic light profiles) to simulated imaging data using Particle Swarm Optimization for global search followed by Markov Chain Monte Carlo for posterior sampling.

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta} \in \Omega} \left[ -\log \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) \right]$$

where $\boldsymbol{\theta} = (\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\theta}_{\text{source}}, \boldsymbol{\theta}_{\text{lens\_light}})$ and $\Omega$ is the constrained parameter space.

### data_fidelity_term

$$-\log \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) = \frac{1}{2} \sum_{i,j} \left( \frac{d_{ij} - m_{ij}(\boldsymbol{\theta})}{\sigma_{\text{bkg}}} \right)^2$$

where:
- $d_{ij}$: observed image pixel value at position $(i,j)$
- $m_{ij}(\boldsymbol{\theta})$: model-predicted image pixel value
- $\sigma_{\text{bkg}} = 0.005$: background RMS noise

### regularization_term

None (pure maximum likelihood estimation)

### constraints

**Box constraints on lens parameters (SIE)**:
- $0.01 \leq \theta_E \leq 10.0$
- $-0.5 \leq e_1 \leq 0.5$
- $-0.5 \leq e_2 \leq 0.5$
- $-10 \leq x_{\text{lens}} \leq 10$
- $-10 \leq y_{\text{lens}} \leq 10$

**Box constraints on lens parameters (SHEAR)**:
- $-0.2 \leq \gamma_1 \leq 0.2$
- $-0.2 \leq \gamma_2 \leq 0.2$

**Box constraints on source parameters (SERSIC_ELLIPSE)**:
- $0.001 \leq R_{\text{sersic}} \leq 10$
- $0.5 \leq n_{\text{sersic}} \leq 5.0$
- $-0.5 \leq e_1 \leq 0.5$
- $-0.5 \leq e_2 \leq 0.5$
- $-10 \leq x_{\text{source}} \leq 10$
- $-10 \leq y_{\text{source}} \leq 10$
- $0 \leq A_{\text{source}} \leq 100$

**Box constraints on lens light parameters (SERSIC_ELLIPSE)**:
- $0.001 \leq R_{\text{sersic}} \leq 10$
- $0.5 \leq n_{\text{sersic}} \leq 5.0$
- $-0.5 \leq e_1 \leq 0.5$
- $-0.5 \leq e_2 \leq 0.5$
- $-10 \leq x_{\text{lens\_light}} \leq 10$
- $-10 \leq y_{\text{lens\_light}} \leq 10$
- $0 \leq A_{\text{lens\_light}} \leq 100$

**Fixed parameters**:
- SHEAR reference point: $(\text{ra}_0, \text{dec}_0) = (0, 0)$

---

## 2. VARIABLES

### Primal Variables (Fitted Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta_E$ | primal | scalar | float64 | 0.7 | Einstein radius of SIE lens (arcsec) |
| $e_1^{\text{lens}}$ | primal | scalar | float64 | 0.0 | First ellipticity component of SIE |
| $e_2^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Second ellipticity component of SIE |
| $x_{\text{lens}}$ | primal | scalar | float64 | 0.0 | SIE center x-coordinate (arcsec) |
| $y_{\text{lens}}$ | primal | scalar | float64 | 0.0 | SIE center y-coordinate (arcsec) |
| $\gamma_1$ | primal | scalar | float64 | 0.0 | First external shear component |
| $\gamma_2$ | primal | scalar | float64 | 0.0 | Second external shear component |
| $A_{\text{source}}$ | primal | scalar | float64 | 16.0 | Source amplitude |
| $R_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 0.2 | Source Sérsic half-light radius (arcsec) |
| $n_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 1.0 | Source Sérsic index |
| $e_1^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source first ellipticity component |
| $e_2^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source second ellipticity component |
| $x_{\text{src}}$ | primal | scalar | float64 | 0.0 | Source center x-coordinate (arcsec) |
| $y_{\text{src}}$ | primal | scalar | float64 | 0.0 | Source center y-coordinate (arcsec) |
| $A_{\text{lens\_light}}$ | primal | scalar | float64 | 16.0 | Lens light amplitude |
| $R_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 0.5 | Lens light Sérsic half-light radius |
| $n_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 2.0 | Lens light Sérsic index |
| $e_1^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light first ellipticity component |
| $e_2^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light second ellipticity component |
| $x_{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light center x-coordinate |
| $y_{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light center y-coordinate |

### Auxiliary Variables (PSO-specific)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{p}_i^{(k)}$ | auxiliary | $(N_{\text{dim}},)$ | float64 | Sampled from prior within bounds | Position of particle $i$ at iteration $k$ |
| $\mathbf{v}_i^{(k)}$ | auxiliary | $(N_{\text{dim}},)$ | float64 | Zero or small random | Velocity of particle $i$ at iteration $k$ |
| $\mathbf{p}_i^{\text{best}}$ | auxiliary | $(N_{\text{dim}},)$ | float64 | $\mathbf{p}_i^{(0)}$ | Personal best position of particle $i$ |
| $\mathbf{g}^{\text{best}}$ | auxiliary | $(N_{\text{dim}},)$ | float64 | Best of initial positions | Global best position across all particles |

### Auxiliary Variables (MCMC-specific)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{w}_j^{(k)}$ | auxiliary | $(N_{\text{dim}},)$ | float64 | PSO result $\pm \sigma_{\text{scale}} \cdot \boldsymbol{\sigma}$ | Position of walker $j$ at step $k$ |
| $\mathbf{S}^{(k)}$ | auxiliary | $(N_{\text{walkers}}, N_{\text{dim}})$ | float64 | Initialized around PSO best | Ensemble state at MCMC step $k$ |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{d}$ | observations | $(60, 60)$ | float64 | Simulated with noise | Observed image data (flux/time) |
| $\mathbf{m}_{\text{true}}$ | observations | $(60, 60)$ | float64 | Forward model with true params | Noise-free ground truth model |

### Constants

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $N_{\text{pix}}$ | constants | scalar | int | 60 | Image size in pixels |
| $\Delta_{\text{pix}}$ | constants | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\sigma_{\text{bkg}}$ | constants | scalar | float64 | 0.005 | Background RMS noise |
| $t_{\text{exp}}$ | constants | scalar | float64 | 500.0 | Exposure time (seconds) |
| $\text{FWHM}$ | constants | scalar | float64 | 0.05 | PSF full width half maximum (arcsec) |
| $N_{\text{particles}}$ | constants | scalar | int | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | constants | scalar | int | 10 | Number of PSO iterations |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | constants | scalar | float64 | 1.0 | PSO sigma scale factor |
| $N_{\text{walkers}}$ | constants | scalar | int | 50 | Number of MCMC walkers |
| $N_{\text{burn}}$ | constants | scalar | int | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | constants | scalar | int | 10 | MCMC sampling steps |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | constants | scalar | float64 | 0.1 | MCMC initialization scatter |
| $\text{truncation}_{\text{PSF}}$ | constants | scalar | int | 3 | PSF truncation radius (in FWHM units) |
| $\text{supersampling}$ | constants | scalar | int | 1 | Supersampling factor for numerics |
| random_seed | constants | scalar | int | 42 | Random seed for reproducibility |

### True Parameters (Ground Truth)

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\theta_E^{\text{true}}$ | constants | scalar | float64 | 0.66 | True Einstein radius |
| $e_1^{\text{lens,true}}$ | constants | scalar | float64 | 0.07 | True SIE $e_1$ |
| $e_2^{\text{lens,true}}$ | constants | scalar | float64 | -0.03 | True SIE $e_2$ |
| $x_{\text{lens}}^{\text{true}}$ | constants | scalar | float64 | 0.05 | True SIE center x |
| $y_{\text{lens}}^{\text{true}}$ | constants | scalar | float64 | 0.0 | True SIE center y |
| $\gamma_1^{\text{true}}$ | constants | scalar | float64 | 0.0 | True shear $\gamma_1$ |
| $\gamma_2^{\text{true}}$ | constants | scalar | float64 | -0.05 | True shear $\gamma_2$ |
| $A_{\text{src}}^{\text{true}}$ | constants | scalar | float64 | 16.0 | True source amplitude |
| $R_{\text{src}}^{\text{true}}$ | constants | scalar | float64 | 0.1 | True source half-light radius |
| $n_{\text{src}}^{\text{true}}$ | constants | scalar | float64 | 1.0 | True source Sérsic index |
| $e_1^{\text{src,true}}$ | constants | scalar | float64 | -0.1 | True source $e_1$ |
| $e_2^{\text{src,true}}$ | constants | scalar | float64 | 0.1 | True source $e_2$ |
| $x_{\text{src}}^{\text{true}}$ | constants | scalar | float64 | 0.1 | True source center x |
| $y_{\text{src}}^{\text{true}}$ | constants | scalar | float64 | 0.0 | True source center y |
| $A_{\text{ll}}^{\text{true}}$ | constants | scalar | float64 | 16.0 | True lens light amplitude |
| $R_{\text{ll}}^{\text{true}}$ | constants | scalar | float64 | 0.6 | True lens light half-light radius |
| $n_{\text{ll}}^{\text{true}}$ | constants | scalar | float64 | 2.0 | True lens light Sérsic index |
| $e_1^{\text{ll,true}}$ | constants | scalar | float64 | -0.1 | True lens light $e_1$ |
| $e_2^{\text{ll,true}}$ | constants | scalar | float64 | 0.1 | True lens light $e_2$ |
| $x_{\text{ll}}^{\text{true}}$ | constants | scalar | float64 | 0.05 | True lens light center x |
| $y_{\text{ll}}^{\text{true}}$ | constants | scalar | float64 | 0.0 | True lens light center y |

### Parameter Uncertainties (PSO/MCMC step sizes)

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\sigma_{\theta_E}$ | constants | scalar | float64 | 0.2 | Step size for Einstein radius |
| $\sigma_{e}^{\text{lens}}$ | constants | scalar | float64 | 0.05 | Step size for lens ellipticity |
| $\sigma_{xy}^{\text{lens}}$ | constants | scalar | float64 | 0.05 | Step size for lens center |
| $\sigma_{\gamma}$ | constants | scalar | float64 | 0.1 | Step size for shear |
| $\sigma_{n}^{\text{src}}$ | constants | scalar | float64 | 0.5 | Step size for source Sérsic index |
| $\sigma_{R}^{\text{src}}$ | constants | scalar | float64 | 0.1 | Step size for source radius |
| $\sigma_{e}^{\text{src}}$ | constants | scalar | float64 | 0.05 | Step size for source ellipticity |
| $\sigma_{xy}^{\text{src}}$ | constants | scalar | float64 | 0.2 | Step size for source center |
| $\sigma_{A}^{\text{src}}$ | constants | scalar | float64 | 10.0 | Step size for source amplitude |
| $\sigma_{n}^{\text{ll}}$ | constants | scalar | float64 | 1.0 | Step size for lens light Sérsic index |
| $\sigma_{R}^{\text{ll}}$ | constants | scalar | float64 | 0.3 | Step size for lens light radius |
| $\sigma_{e}^{\text{ll}}$ | constants | scalar | float64 | 0.05 | Step size for lens light ellipticity |
| $\sigma_{xy}^{\text{ll}}$ | constants | scalar | float64 | 0.1 | Step size for lens light center |
| $\sigma_{A}^{\text{ll}}$ | constants | scalar | float64 | 10.0 | Step size for lens light amplitude |

---

## 3. PREPROCESSING PIPELINE

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 1 | Set Random Seed | `np.random.seed(42)` | seed=42 | RNG state initialized |
| 2 | Define Coordinate Transform | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix} = \begin{pmatrix} -0.05 & 0 \\ 0 & 0.05 \end{pmatrix}$ | $\Delta_{\text{pix}}=0.05$ | $\mathbf{T} \in \mathbb{R}^{2 \times 2}$ |
| 3 | Compute Image Center | $c_x = c_y = \frac{N_{\text{pix}} - 1}{2} = \frac{60-1}{2} = 29.5$ | $N_{\text{pix}}=60$ | $c_x, c_y \in \mathbb{R}$ |
| 4 | Compute Reference Coordinates | $\text{RA}_0 = -(T_{00} c_x + T_{01} c_y) = -(-0.05 \times 29.5 + 0) = 1.475$ <br> $\text{Dec}_0 = -(T_{10} c_x + T_{11} c_y) = -(0 + 0.05 \times 29.5) = -1.475$ | $\mathbf{T}, c_x, c_y$ | $\text{RA}_0, \text{Dec}_0 \in \mathbb{R}$ |
| 5 | Initialize Empty Image | $\mathbf{I}_0 = \mathbf{0}_{60 \times 60}$ | $N_{\text{pix}}=60$ | $\mathbf{I}_0 \in \mathbb{R}^{60 \times 60}$ |
| 6 | Create Data Class | `ImageData(kwargs_data)` | $\sigma_{\text{bkg}}, t_{\text{exp}}, \text{RA}_0, \text{Dec}_0, \mathbf{T}, \mathbf{I}_0$ | `data_class` object |
| 7 | Create Gaussian PSF | $\text{PSF}(x,y) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$ where $\sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} = \frac{0.05}{2.355} \approx 0.0212$ | FWHM=0.05, $\Delta_{\text{pix}}$=0.05, truncation=3 | `psf_class` object |
| 8 | Create Lens Model | `LensModel(['SIE', 'SHEAR'])` | lens_model_list | `lens_model_class` |
| 9 | Create Source Light Model | `LightModel(['SERSIC_ELLIPSE'])` | source_model_list | `source_model_class` |
| 10 | Create Lens Light Model | `LightModel(['SERSIC_ELLIPSE'])` | lens_light_model_list | `lens_light_model_class` |
| 11 | Instantiate Image Model | `ImageModel(data_class, psf_class, ...)` | All model classes, kwargs_numerics | `imageModel` object |
| 12 | Compute Noise-Free Model | $\mathbf{m}_{\text{true}} = \mathcal{F}(\boldsymbol{\theta}^{\text{true}})$ (see forward model below) | True parameters | $\mathbf{m}_{\text{true}} \in \mathbb{R}^{60 \times 60}$ |
| 13 | Convert to Counts | $\mathbf{C} = \mathbf{m}_{\text{true}} \cdot t_{\text{exp}}$; $C_{ij} \leftarrow \max(C_{ij}, 0)$ | $\mathbf{m}_{\text{true}}, t_{\text{exp}}=500$ | $\mathbf{C} \in \mathbb{R}_{\geq 0}^{60 \times 60}$ |
| 14 | Add Poisson Noise | $\tilde{C}_{ij} \sim \text{Poisson}(C_{ij})$; $\mathbf{I}_{\text{Poisson}} = \tilde{\mathbf{C}} / t_{\text{exp}}$ | $\mathbf{C}$ | $\mathbf{I}_{\text{Poisson}} \in \mathbb{R}^{60 \times 60}$ |
| 15 | Add Gaussian Noise | $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{bkg}}^2 \mathbf{I})$ | $\sigma_{\text{bkg}}=0.005$ | $\mathbf{n} \in \mathbb{R}^{60 \times 60}$ |
| 16 | Generate Observed Image | $\mathbf{d} = \mathbf{I}_{\text{Poisson}} + \mathbf{n}$ | $\mathbf{I}_{\text{Poisson}}, \mathbf{n}$ | $\mathbf{d} \in \mathbb{R}^{60 \times 60}$ |
| 17 | Update Data Class | `data_class.update_data(d)` | $\mathbf{d}$ | Updated `data_class` |

---

## 4. ALGORITHM STEPS

### Forward Model Definition

The forward operator $\mathcal{F}: \boldsymbol{\theta} \mapsto \mathbf{m}$ computes:

$$\mathbf{m}(\boldsymbol{\theta}) = \text{PSF} * \left[ I_{\text{source}}(\boldsymbol{\beta}(\boldsymbol{\alpha})) + I_{\text{lens\_light}}(\boldsymbol{\alpha}) \right]$$

where:

**Lens Equation (SIE + SHEAR)**:
$$\boldsymbol{\beta} = \boldsymbol{\alpha} - \nabla\psi_{\text{SIE}}(\boldsymbol{\alpha}) - \nabla\psi_{\text{SHEAR}}(\boldsymbol{\alpha})$$

**SIE Deflection** (Singular Isothermal Ellipsoid):
$$\psi_{\text{SIE}}(\boldsymbol{\alpha}) = \theta_E \sqrt{q} \left[ \alpha_1 \arctan\left(\frac{\alpha_1 \sqrt{1-q^2}}{w + q\alpha_2}\right) + \alpha_2 \text{arctanh}\left(\frac{\alpha_2\sqrt{1-q^2}}{w + \alpha_1}\right) \right]$$
where $q = \sqrt{\frac{1-e}{1+e}}$, $e = \sqrt{e_1^2 + e_2^2}$, and $w = \sqrt{q^2\alpha_1^2 + \alpha_2^2}$

**SHEAR Deflection**:
$$\nabla\psi_{\text{SHEAR}} = \begin{pmatrix} \gamma_1 \alpha_1 + \gamma_2 \alpha_2 \\ \gamma_2 \alpha_1 - \gamma_1 \alpha_2 \end{pmatrix}$$

**Sérsic Profile**:
$$I_{\text{Sersic}}(r) = A \exp\left[ -b_n \left( \left(\frac{r}{R_{\text{sersic}}}\right)^{1/n} - 1 \right) \right]$$
where $b_n \approx 1.9992n - 0.3271$ for $0.5 < n < 10$

**Elliptical Radius**:
$$r_{\text{ell}} = \sqrt{q_s x'^2 + y'^2/q_s}$$
where $(x', y')$ are rotated coordinates and $q_s$ is derived from $(e_1, e_2)$

---

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| 1 | Setup Parameter Bounds | Define $\boldsymbol{\theta}_{\min}, \boldsymbol{\theta}_{\max}$ from constraint tables | Bound specifications | Bound arrays | See Section 1 constraints |
| 2 | Setup Initial Guess | $\boldsymbol{\theta}^{(0)} = \boldsymbol{\theta}_{\text{init}}$ (see Variable table) | Init specifications | $\boldsymbol{\theta}^{(0)} \in \mathbb{R}^{N_{\text{dim}}}$ | $N_{\text{dim}} = 21$ (total free parameters) |
| 3 | Setup Step Sizes | $\boldsymbol{\sigma} = (\sigma_{\theta_E}, \sigma_{e}^{\text{lens}}, \ldots)$ | Sigma specifications | $\boldsymbol{\sigma} \in \mathbb{R}^{N_{\text{dim}}}$ | |
| 4 | Create FittingSequence | `FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)` | All setup dicts | `fitting_seq` object | `linear_solver=True`, `source_marg=False` |

---

### 4.2 ITERATION LOOP

#### 4.2.1 Phase 1: Particle Swarm Optimization (PSO)

**BEGIN_LOOP** (PSO: $k = 1, \ldots, N_{\text{iter}}^{\text{PSO}} = 10$)

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| 1 | Evaluate Particle Fitness | $f_i^{(k)} = -\log\mathcal{L}(\mathbf{d} | \mathbf{p}_i^{(k)}) = \frac{1}{2}\sum_{i,j}\left(\frac{d_{ij} - m_{ij}(\mathbf{p}_i^{(k)})}{\sigma_{\text{bkg}}}\right)^2$ | $\mathbf{p}_i^{(k)}$, $\mathbf{d}$ | $f_i^{(k)} \in \mathbb{R}$ | For all $i = 1, \ldots, 50$ |
| 2 | Update Personal Best | $\mathbf{p}_i^{\text{best}} \leftarrow \mathbf{p}_i^{(k)}$ if $f_i^{(k)} < f(\mathbf{p}_i^{\text{best}})$ | $f_i^{(k)}, \mathbf{p}_i^{(k)}, \mathbf{p}_i^{\text{best}}$ | Updated $\mathbf{p}_i^{\text{best}}$ | |
| 3 | Update Global Best | $\mathbf{g}^{\text{best}} \leftarrow \arg\min_i f(\mathbf{p}_i^{\text{best}})$ | All $\mathbf{p}_i^{\text{best}}$ | Updated $\mathbf{g}^{\text{best}}$ | |
| 4 | Update Velocities | $\mathbf{v}_i^{(k+1)} = \omega \mathbf{v}_i^{(k)} + c_1 r_1 (\mathbf{p}_i^{\text{best}} - \mathbf{p}_i^{(k)}) + c_2 r_2 (\mathbf{g}^{\text{best}} - \mathbf{p}_i^{(k)})$ | $\mathbf{v}_i^{(k)}, \mathbf{p}_i^{(k)}, \mathbf{p}_i^{\text{best}}, \mathbf{g}^{\text{best}}$ | $\mathbf{v}_i^{(k+1)}$ | $\omega, c_1, c_2, r_1, r_2$ from lenstronomy defaults |
| 5 | Update Positions | $\mathbf{p}_i^{(k+1)} = \mathbf{p}_i^{(k)} + \mathbf{v}_i^{(k+1)}$ | $\mathbf{p}_i^{(k)}, \mathbf{v}_i^{(k+1)}$ | $\mathbf{p}_i^{(k+1)}$ | |
| 6 | Enforce Bounds | $p_{i,j}^{(k+1)} \leftarrow \text{clip}(p_{i,j}^{(k+1)}, \theta_{\min,