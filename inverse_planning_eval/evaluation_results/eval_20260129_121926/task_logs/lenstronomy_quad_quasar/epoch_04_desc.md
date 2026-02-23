## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: Gravitational Lens Model Fitting via PSO-MCMC
- **algorithm_family**: Hybrid Metaheuristic-Bayesian Optimization (Particle Swarm Optimization followed by Markov Chain Monte Carlo)
- **summary**: Fits a parametric gravitational lensing model to observed imaging data by first exploring the parameter space with PSO, then sampling the posterior distribution with MCMC to recover lens mass distribution, source light, lens light, and point source parameters.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \left[ \frac{1}{2} \chi^2(\boldsymbol{\theta}) - \ln \pi(\boldsymbol{\theta}) \right]$$

subject to:
$$\boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$$

where the posterior is:
$$\mathcal{L}(\boldsymbol{\theta} | \mathbf{d}) \propto \exp\left(-\frac{1}{2}\chi^2(\boldsymbol{\theta})\right) \cdot \pi(\boldsymbol{\theta})$$

### data_fidelity_term
**Chi-squared (L2-norm) data fidelity**:
$$\chi^2(\boldsymbol{\theta}) = \sum_{i,j} \frac{\left(d_{ij} - m_{ij}(\boldsymbol{\theta})\right)^2}{\sigma_{ij}^2}$$

where:
- $d_{ij}$: observed image pixel value at position $(i,j)$
- $m_{ij}(\boldsymbol{\theta})$: model-predicted pixel value
- $\sigma_{ij}^2 = \sigma_{\text{bkg}}^2 + \frac{m_{ij}(\boldsymbol{\theta})}{t_{\text{exp}}}$ (background variance + Poisson variance)

### regularization_term
**Gaussian Prior on Lens Ellipticity**:
$$-\ln \pi(\boldsymbol{\theta}) = \frac{(e_1^{\text{lens}} - \mu_{e_1})^2}{2\sigma_{e_1}^2} + \frac{(e_2^{\text{lens}} - \mu_{e_2})^2}{2\sigma_{e_2}^2}$$

with $\mu_{e_1} = 0$, $\sigma_{e_1} = 0.2$, $\mu_{e_2} = 0$, $\sigma_{e_2} = 0.2$

### constraints
**Box constraints** (indicator function $\mathbb{I}_{\mathcal{C}}$):
$$\mathbb{I}_{\mathcal{C}}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}} \\ +\infty & \text{otherwise} \end{cases}$$

**Source position constraint**:
$$\|\mathbf{r}_{\text{source}}^{\text{(light)}} - \mathbf{r}_{\text{source}}^{\text{(PS)}}\| \leq \tau_{\text{src}}$$
where $\tau_{\text{src}} = 0.001$ arcsec

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 1.2 | Einstein radius (arcsec) |
| $\gamma_{\text{lens}}$ | scalar | float64 | 2.0 | Power-law slope of mass profile |
| $e_1^{\text{lens}}$ | scalar | float64 | 0.0 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | scalar | float64 | 0.0 | Lens ellipticity component 2 |
| $x_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center x-coordinate (arcsec) |
| $y_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center y-coordinate (arcsec) |
| $\gamma_1^{\text{shear}}$ | scalar | float64 | 0.0 | External shear component 1 |
| $\gamma_2^{\text{shear}}$ | scalar | float64 | 0.0 | External shear component 2 |
| $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.03 | Source Sersic half-light radius |
| $n_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | scalar | float64 | 0.0 | Source ellipticity component 1 |
| $e_2^{\text{src}}$ | scalar | float64 | 0.0 | Source ellipticity component 2 |
| $x_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center x-coordinate |
| $y_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center y-coordinate |
| $R_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 0.1 | Lens light Sersic radius |
| $n_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 1.0 | Lens light Sersic index |
| $e_1^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light ellipticity 1 |
| $e_2^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light ellipticity 2 |
| $x_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center x |
| $y_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center y |
| $\mathbf{x}_{\text{img}}$ | (4,) | float64 | $x_{\text{true}} + 0.01$ | Point source image RA positions |
| $\mathbf{y}_{\text{img}}$ | (4,) | float64 | $y_{\text{true}} - 0.01$ | Point source image Dec positions |

### 2.2 Dual Variables
None (optimization is unconstrained in primal form with box projection)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{p}_k^{(i)}$ | $(N_{\text{dim}},)$ | float64 | Random in bounds | PSO particle $i$ position at iteration $k$ |
| $\mathbf{v}_k^{(i)}$ | $(N_{\text{dim}},)$ | float64 | Zero | PSO particle $i$ velocity at iteration $k$ |
| $\mathbf{p}_{\text{best}}^{(i)}$ | $(N_{\text{dim}},)$ | float64 | $\mathbf{p}_0^{(i)}$ | Personal best position for particle $i$ |
| $\mathbf{g}_{\text{best}}$ | $(N_{\text{dim}},)$ | float64 | Best of initial | Global best position across all particles |
| $\mathbf{w}_k^{(j)}$ | $(N_{\text{dim}},)$ | float64 | From PSO result | MCMC walker $j$ position at step $k$ |
| $\chi^2_{\text{current}}$ | scalar | float64 | Computed | Current chi-squared value |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\sigma_{\text{bkg}}$ | 0.5 | Background noise RMS (counts/s) |
| $t_{\text{exp}}$ | 100 | Exposure time (seconds) |
| $N_{\text{pix}}$ | 100 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | 0.05 | Pixel scale (arcsec/pixel) |
| $\text{FWHM}$ | 0.1 | PSF full width half maximum (arcsec) |
| $N_{\text{particles}}$ | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | 10 | PSO iterations |
| $N_{\text{walkers}}$ | 50 | Number of MCMC walkers |
| $N_{\text{burn}}$ | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | 10 | MCMC sampling steps |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | 1.0 | PSO sigma scaling factor |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | 0.1 | MCMC proposal sigma scaling |
| $\sigma_{\text{pos}}$ | 0.004 | Image position uncertainty (arcsec) |
| $\tau_{\text{src}}$ | 0.001 | Source position tolerance (arcsec) |
| $\text{trunc}$ | 5 | PSF truncation (in FWHM units) |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{d}$ | $(100, 100)$ | float64 | Observed image data (counts/s) |
| $\mathbf{T}$ | $(2, 2)$ | float64 | Pixel-to-angle transformation matrix |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Coordinate System Setup | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}$ (scalar) | $\mathbf{T}$ (2×2) |
| 1 | Reference Point Calculation | $\text{RA}_0 = -(\mathbf{T}_{00} \cdot c_x + \mathbf{T}_{01} \cdot c_y)$, $\text{Dec}_0 = -(\mathbf{T}_{10} \cdot c_x + \mathbf{T}_{11} \cdot c_y)$ where $c_x = c_y = (N_{\text{pix}}-1)/2$ | $\mathbf{T}$, $N_{\text{pix}}$ | $\text{RA}_0$, $\text{Dec}_0$ (scalars) |
| 2 | Shear Conversion | $\gamma_1 = \gamma \cos(2\phi)$, $\gamma_2 = \gamma \sin(2\phi)$ where $\phi=0.1$, $\gamma=0.02$ | $\phi$, $\gamma$ (scalars) | $\gamma_1$, $\gamma_2$ (scalars) |
| 3 | True Image Generation | $\mathbf{m}_{\text{true}} = \mathcal{F}(\boldsymbol{\theta}_{\text{true}})$ | True parameters | $\mathbf{m}_{\text{true}}$ (100×100) |
| 4 | Poisson Noise Generation | $\mathbf{n}_{\text{Poisson}} = \frac{\text{Poisson}(\max(\mathbf{m}_{\text{true}} \cdot t_{\text{exp}}, 0))}{t_{\text{exp}}} - \mathbf{m}_{\text{true}}$ | $\mathbf{m}_{\text{true}}$, $t_{\text{exp}}$ | $\mathbf{n}_{\text{Poisson}}$ (100×100) |
| 5 | Background Noise Generation | $\mathbf{n}_{\text{bkg}} = \sigma_{\text{bkg}} \cdot \mathcal{N}(0,1)^{N_{\text{pix}} \times N_{\text{pix}}}$ | $\sigma_{\text{bkg}}$ | $\mathbf{n}_{\text{bkg}}$ (100×100) |
| 6 | Observed Data Assembly | $\mathbf{d} = \mathbf{m}_{\text{true}} + \mathbf{n}_{\text{bkg}} + \mathbf{n}_{\text{Poisson}}$ | $\mathbf{m}_{\text{true}}$, $\mathbf{n}_{\text{bkg}}$, $\mathbf{n}_{\text{Poisson}}$ | $\mathbf{d}$ (100×100) |
| 7 | Point Source Position Solving | $(\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}) = \text{LensEqSolver}(\mathbf{r}_{\text{src}}, \boldsymbol{\theta}_{\text{lens}})$ | Source position, lens parameters | Image positions (4,), (4,) |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Parameter Vector Assembly | $\boldsymbol{\theta}_{\text{init}} = [\theta_E^{(0)}, \gamma^{(0)}, e_1^{(0)}, e_2^{(0)}, x_c^{(0)}, y_c^{(0)}, \gamma_1^{(0)}, \gamma_2^{(0)}, \ldots]$ | Initial kwargs | $\boldsymbol{\theta}_{\text{init}} \in \mathbb{R}^{N_{\text{dim}}}$ | Flattened parameter vector |
| PSO Particle Initialization | $\mathbf{p}_0^{(i)} = \boldsymbol{\theta}_{\text{init}} + \boldsymbol{\sigma} \cdot \mathcal{N}(0,1)^{N_{\text{dim}}} \cdot \sigma_{\text{scale}}^{\text{PSO}}$ for $i=1,\ldots,N_{\text{particles}}$ | $\boldsymbol{\theta}_{\text{init}}$, $\boldsymbol{\sigma}$ | $\{\mathbf{p}_0^{(i)}\}_{i=1}^{50}$ | Gaussian scatter around init |
| Velocity Initialization | $\mathbf{v}_0^{(i)} = \mathbf{0}$ for all $i$ | — | $\{\mathbf{v}_0^{(i)}\}_{i=1}^{50}$ | Zero initial velocity |
| Bound Enforcement | $\mathbf{p}_0^{(i)} = \text{clip}(\mathbf{p}_0^{(i)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$ | Particles, bounds | Clipped particles | Projection onto feasible set |
| Initial Fitness Evaluation | $f^{(i)} = \chi^2(\mathbf{p}_0^{(i)}) - 2\ln\pi(\mathbf{p}_0^{(i)})$ for all $i$ | Particles | Fitness values | Negative log-posterior |
| Personal/Global Best Init | $\mathbf{p}_{\text{best}}^{(i)} = \mathbf{p}_0^{(i)}$; $\mathbf{g}_{\text{best}} = \arg\min_i f^{(i)}$ | Fitness values | Best positions | Track optima |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (PSO: $k = 1, \ldots, N_{\text{iter}}^{\text{PSO}}$)

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Velocity Update | $\mathbf{v}_{k}^{(i)} = \omega \mathbf{v}_{k-1}^{(i)} + c_1 r_1 (\mathbf{p}_{\text{best}}^{(i)} - \mathbf{p}_{k-1}^{(i)}) + c_2 r_2 (\mathbf{g}_{\text{best}} - \mathbf{p}_{k-1}^{(i)})$ | Previous velocity, positions, bests | Updated velocity | $\omega, c_1, c_2$ from lenstronomy defaults; $r_1, r_2 \sim U(0,1)$ |
| Position Update | $\mathbf{p}_{k}^{(i)} = \mathbf{p}_{k-1}^{(i)} + \mathbf{v}_{k}^{(i)}$ | Position, velocity | New position | Standard PSO step |
| Bound Projection | $\mathbf{p}_{k}^{(i)} = \text{clip}(\mathbf{p}_{k}^{(i)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$ | Position, bounds | Projected position | Box constraint enforcement |
| Forward Model [FORWARD] | $\mathbf{m}^{(i)} = \mathcal{F}(\mathbf{p}_{k}^{(i)}) = \mathbf{I}_{\text{lens}} + \mathbf{I}_{\text{source}} + \mathbf{I}_{\text{PS}}$ | Parameters | Model image | See forward model details below |
| Chi-squared Computation | $\chi^{2(i)} = \sum_{j,l} \frac{(d_{jl} - m_{jl}^{(i)})^2}{\sigma_{\text{bkg}}^2 + m_{jl}^{(i)}/t_{\text{exp}}}$ | Data, model | $\chi^2$ value | L2 residual |
| Prior Evaluation | $\ln\pi^{(i)} = -\frac{(e_1^{(i)})^2}{2(0.2)^2} - \frac{(e_2^{(i)})^2}{2(0.2)^2}$ | Ellipticities | Log-prior | Gaussian prior on $e_1, e_2$ |
| Fitness Computation | $f^{(i)} = \chi^{2(i)} - 2\ln\pi^{(i)}$ | $\chi^2$, prior | Fitness | Negative log-posterior (×2) |
| Personal Best Update | $\mathbf{p}_{\text{best}}^{(i)} = \mathbf{p}_{k}^{(i)}$ if $f^{(i)} < f_{\text{best}}^{(i)}$ | Fitness, previous best | Updated personal best | Greedy update |
| Global Best Update | $\mathbf{g}_{\text{best}} = \arg\min_i \mathbf{p}_{\text{best}}^{(i)}$ | All personal bests | Updated global best | Best across swarm |

**END_LOOP** (PSO)

**BEGIN_LOOP** (MCMC: $k = 1, \ldots, N_{\text{burn}} + N_{\text{run}}$)

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Walker Initialization (k=0) | $\mathbf{w}_0^{(j)} = \mathbf{g}_{\text{best}} + \sigma_{\text{scale}}^{\text{MCMC}} \cdot \boldsymbol{\sigma} \cdot \mathcal{N}(0,1)$ for $j=1,\ldots,N_{\text{walkers}}$ | PSO best, sigma | Walker positions | Initialize around PSO optimum |
| Stretch Move Proposal | $\mathbf{w}_{k,\text{prop}}^{(j)} = \mathbf{w}_{k-1}^{(j')} + z(\mathbf{w}_{k-1}^{(j)} - \mathbf{w}_{k-1}^{(j')})$ where $z \sim g(z) \propto 1/\sqrt{z}$ for $z \in [1/a, a]$, $a=2$ | Complementary walker $j'$ | Proposed position | Affine-invariant ensemble sampler |
| Bound Check | Valid $= (\boldsymbol{\theta}_{\text{lower}} \leq \mathbf{w}_{k,\text{prop}}^{(j)} \leq \boldsymbol{\theta}_{\text{upper}})$ | Proposal, bounds | Boolean | Reject if outside bounds |
| Log-Likelihood Evaluation | $\ln\mathcal{L}^{(j)} = -\frac{1}{2}\chi^2(\mathbf{w}_{k,\text{prop}}^{(j)}) + \ln\pi(\mathbf{w}_{k,\text{prop}}^{(j)})$ | Proposal | Log-likelihood | Forward model + prior |
| Acceptance Probability | $\alpha = \min\left(1, z^{N_{\text{dim}}-1} \exp(\ln\mathcal{L}_{\text{prop}} - \ln\mathcal{L}_{\text{current}})\right)$ | Log-likelihoods, $z$ | Acceptance prob | Stretch move acceptance |
| Metropolis Accept/Reject | $\mathbf{w}_k^{(j)} = \begin{cases} \mathbf{w}_{k,\text{prop}}^{(j)} & \text{if } u < \alpha, u \sim U(0,1) \\ \mathbf{w}_{k-1}^{(j)} & \text{otherwise} \end{cases}$ | Proposal, current, $\alpha$ | Accepted position | Standard MH step |

**END_LOOP** (MCMC)

#### Forward Model Detail [FORWARD]

The model image $\mathbf{m}(\boldsymbol{\theta})$ is computed as:

$$\mathbf{m} = \underbrace{\text{PSF} \ast \mathbf{I}_{\text{source}}^{\text{lensed}}}_{\text{Lensed source}} + \underbrace{\text{PSF} \ast \mathbf{I}_{\text{lens light}}}_{\text{Lens galaxy}} + \underbrace{\sum_{n=1}^{4} A_n \cdot \text{PSF}(\mathbf{x}_n^{\text{img}})}_{\text{Point sources}}$$

**Lens Deflection (EPL + Shear)**:
$$\boldsymbol{\alpha}(\boldsymbol{\xi}) = \boldsymbol{\alpha}_{\text{EPL}}(\boldsymbol{\xi}; \theta_E, \gamma, e_1, e_2, x_c, y_c) + \boldsymbol{\alpha}_{\text{shear}}(\boldsymbol{\xi}; \gamma_1, \gamma_2)$$

**Source Position Mapping**:
$$\boldsymbol{\beta} = \boldsymbol{\xi} - \boldsymbol{\alpha}(\boldsymbol{\xi})$$

**Sersic Profile**:
$$I_{\text{Sersic}}(r) = A \exp\left(-b_n \left[\left(\frac{r}{R_{\text{sersic}}}\right)^{1/n} - 1\right]\right)$$
where $b_n \approx 2n - 1/3 + 4/(405n)$

**PSF Convolution**:
$$(\text{PSF} \ast I)_{ij} = \sum_{k,l} \text{PSF}_{kl} \cdot I_{i-k, j-l}$$

with Gaussian PSF:
$$\text{PSF}(r) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{r^2}{2\sigma_{\text{PSF}}^2}\right), \quad \sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$$

### 4.3 FINALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Best Fit Extraction | $\boldsymbol{\theta}_{\text{best}} = \arg\max_k \ln\mathcal{L}(\mathbf{w}_k)$ | MCMC chain | Best parameters | Maximum likelihood from chain |
| Parameter Unpacking | $\{\text{kwargs\_lens}, \text{kwargs\_source}, \text{kwargs\_lens\_light}, \text{kwargs\_ps}\} = \text{unpack}(\boldsymbol{\theta}_{\text{best}})$ | Best vector | Parameter dicts | Restore structured format |
| Reconstructed Image | $\mathbf{m}_{\text{recon}} = \mathcal{F}(\boldsymbol{\theta}_{\text{best}})$ | Best parameters | Image (100×100) | Final forward model |
| Chain Storage | Store $\{(\mathbf{w}_k^{(j)}, \ln\mathcal{L}_k^{(j)})\}$ for $k > N_{\text{burn}}$ | MCMC samples | Chain list | Discard burn-in |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: 
  - PSO: $k_{\text{PSO}} = N_{\text{iter}}^{\text{PSO}} = 10$
  - MCMC: $k_{\text{MCMC}} = N_{\text{burn}} + N_{\text{run}} = 10 + 10 = 20$
- **parameters**:
  - `n_iterations` (PSO): 10
  - `n_burn` (MCMC): 10
  - `n_run` (MCMC): 10
  - `n_particles`: 50
  - `n_walkers`: 50
- **check_frequency**: After each complete iteration of all particles/walkers

---

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| Residual RMS | $\text{RMS} = \sqrt{\frac{1}{N_{\text{pix}}^2} \sum_{i,j} (d_{ij} - m_{ij}^{\text{recon}})^2}$ | Root mean square of image residuals |
| Einstein Radius Error | $\Delta\theta_E = |\theta_E^{\text{fit}} - \theta_E^{\text{true}}|$ | Absolute error in Einstein radius recovery |
| Power-Law Slope Error | $\Delta\gamma = |\gamma^{\text{fit}} - \gamma^{\text{true}}|$ | Absolute error in mass profile slope |
| Ellipticity Error (e1) | $\Delta e_1 = |e_1^{\text{fit}} - e_1^{\text{true}}|$ | Absolute error in ellipticity component 1 |
| Ellipticity Error (e2) | $\Delta e_2 = |e_2^{\text{fit}} - e_2^{\text{true}}|$ | Absolute error in ellipticity component 2 |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include:

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
  - primal_variables[]
  - auxiliary_variables[]
  - constants[]
  - observations[]

data_preprocessing:
  - steps[] (each with: step_order, step_name, formula, inputs, outputs)

initialization:
  - steps[] (each with: step_name, formula, inputs, outputs)

iteration_loop:
  - pso_loop:
    - n_iterations
    - steps[]
  - mcmc_loop:
    - n_burn
    - n_run
    - n_walkers
    - steps[]

finalization:
  - steps[]

stopping_criterion:
  - type
  - expression
  - parameters

evaluation:
  - metrics[] (each with: name, formula, description)

fitted_parameters:
  - kwargs_lens
  - kwargs_source
  - kwargs_lens_light
  - kwargs_ps

outputs:
  - image_reconstructed
  - residuals
  - chain_list
  - residual_rms
  - parameter_errors
```