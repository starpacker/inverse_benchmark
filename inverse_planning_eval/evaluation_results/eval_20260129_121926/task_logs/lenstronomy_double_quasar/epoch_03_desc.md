## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Gravitational Lens Modeling via PSO-MCMC Hybrid Optimization
- **algorithm_family**: Bayesian Inference / Metaheuristic Optimization (Particle Swarm Optimization + Markov Chain Monte Carlo)
- **summary**: Reconstructs gravitational lens system parameters by minimizing chi-squared residuals between observed and forward-modeled images using sequential PSO global search followed by MCMC posterior sampling.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta} \in \Omega} \mathcal{L}(\boldsymbol{\theta} | \mathbf{d})$$

where $\boldsymbol{\theta} = \{\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\theta}_{\text{source}}, \boldsymbol{\theta}_{\text{lens\_light}}, \boldsymbol{\theta}_{\text{ps}}\}$ and $\Omega$ is the bounded parameter space.

### data_fidelity_term
$$\mathcal{L}(\boldsymbol{\theta} | \mathbf{d}) = -\frac{1}{2} \chi^2(\boldsymbol{\theta}) = -\frac{1}{2} \sum_{i,j} \left( \frac{d_{ij} - m_{ij}(\boldsymbol{\theta})}{\sigma_{\text{bkg}}} \right)^2$$

where:
- $d_{ij}$: observed image pixel value at position $(i,j)$
- $m_{ij}(\boldsymbol{\theta}) = \mathcal{F}(\boldsymbol{\theta})$: forward model prediction
- $\sigma_{\text{bkg}} = 0.5$: background RMS noise

### regularization_term
None (uniform priors within bounds; hard constraints only)

### constraints
1. **Parameter bounds**: $\boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$ (see Section 2 for explicit values)
2. **Joint source-point source constraint**: Source center linked to point source position via index mapping $[[0, 0]]$
3. **Image position uncertainty**: $\sigma_{\text{pos}} = 0.004$ arcsec
4. **Source position tolerance**: $\tau_{\text{source}} = 0.001$ arcsec
5. **Shear origin fixed**: $(\text{ra}_0, \text{dec}_0) = (0, 0)$ for SHEAR component

---

## 2. VARIABLES

### Primal Variables (Optimized Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta_E$ | primal | scalar | float64 | 1.1 | Einstein radius (arcsec) |
| $\gamma_{\text{EPL}}$ | primal | scalar | float64 | 2.0 | EPL power-law slope |
| $e_1^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens ellipticity component 2 |
| $x_c^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens center x (arcsec) |
| $y_c^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens center y (arcsec) |
| $\gamma_1$ | primal | scalar | float64 | 0.0 | External shear component 1 |
| $\gamma_2$ | primal | scalar | float64 | 0.0 | External shear component 2 |
| $R_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 0.03 | Source Sersic radius (arcsec) |
| $n_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source ellipticity component 1 |
| $e_2^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source ellipticity component 2 |
| $x_c^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source center x (arcsec) |
| $y_c^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source center y (arcsec) |
| $R_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 0.1 | Lens light Sersic radius |
| $n_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 1.0 | Lens light Sersic index |
| $e_1^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light ellipticity 1 |
| $e_2^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light ellipticity 2 |
| $x_c^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light center x |
| $y_c^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light center y |
| $\mathbf{x}_{\text{img}}$ | primal | $(N_{\text{img}},)$ | float64 | $\mathbf{x}_{\text{solved}} + 0.01$ | Point source RA positions |
| $\mathbf{y}_{\text{img}}$ | primal | $(N_{\text{img}},)$ | float64 | $\mathbf{y}_{\text{solved}} - 0.01$ | Point source Dec positions |

### Parameter Bounds (Constants)

| Parameter | Lower | Upper | $\sigma$ (PSO/MCMC) |
|-----------|-------|-------|---------------------|
| $\theta_E$ | 0 | 10 | 0.1 |
| $\gamma_{\text{EPL}}$ | 1.5 | 2.5 | 0.1 |
| $e_1^{\text{lens}}, e_2^{\text{lens}}$ | -0.5 | 0.5 | 0.2 |
| $x_c^{\text{lens}}, y_c^{\text{lens}}$ | -10 | 10 | 0.1 |
| $\gamma_1, \gamma_2$ | -0.2 | 0.2 | 0.1 |
| $R_{\text{sersic}}^{\text{src}}$ | 0.001 | 10 | 0.2 |
| $n_{\text{sersic}}^{\text{src}}$ | 0.5 | 5.0 | 0.5 |
| $e_1^{\text{src}}, e_2^{\text{src}}$ | -0.5 | 0.5 | 0.2 |
| $x_c^{\text{src}}, y_c^{\text{src}}$ | -10 | 10 | 0.1 |
| $R_{\text{sersic}}^{\text{ll}}$ | 0.001 | 10 | 0.1 |
| $n_{\text{sersic}}^{\text{ll}}$ | 0.5 | 5.0 | 0.5 |
| $e_1^{\text{ll}}, e_2^{\text{ll}}$ | -0.5 | 0.5 | 0.0 (fixed at init) |
| $x_c^{\text{ll}}, y_c^{\text{ll}}$ | -10 | 10 | 0.1 |
| $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}$ | -10 | 10 | 0.02 |

### Observations

| Symbol | Category | Shape | Dtype | Description |
|--------|----------|-------|-------|-------------|
| $\mathbf{d}$ | observations | $(100, 100)$ | float64 | Noisy observed image |

### Constants

| Symbol | Category | Value | Description |
|--------|----------|-------|-------------|
| $\sigma_{\text{bkg}}$ | constants | 0.5 | Background RMS noise (flux units) |
| $t_{\text{exp}}$ | constants | 100 | Exposure time (seconds) |
| $N_{\text{pix}}$ | constants | 100 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | constants | 0.05 | Pixel scale (arcsec/pixel) |
| FWHM | constants | 0.2 | PSF full-width half-maximum (arcsec) |
| $\tau_{\text{PSF}}$ | constants | 3 | PSF truncation radius (in FWHM units) |
| $N_{\text{particles}}$ | constants | 50 | PSO swarm size |
| $N_{\text{iter}}^{\text{PSO}}$ | constants | 10 | PSO iterations |
| $N_{\text{burn}}$ | constants | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | constants | 10 | MCMC sampling steps |
| $r_{\text{walker}}$ | constants | 4 | MCMC walker ratio |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | constants | 1.0 | PSO sigma scale factor |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | constants | 0.1 | MCMC sigma scale factor |
| $\sigma_{\text{pos}}$ | constants | 0.004 | Image position uncertainty (arcsec) |
| $\tau_{\text{source}}$ | constants | 0.001 | Source position tolerance (arcsec) |

### True Parameters (for simulation)

| Symbol | Value | Description |
|--------|-------|-------------|
| $\theta_E^{\text{true}}$ | 1.0 | True Einstein radius |
| $\gamma_{\text{EPL}}^{\text{true}}$ | 1.96 | True EPL slope |
| $e_1^{\text{lens,true}}$ | 0.07 | True lens ellipticity 1 |
| $e_2^{\text{lens,true}}$ | -0.03 | True lens ellipticity 2 |
| $\gamma_1^{\text{true}}$ | 0.01 | True shear component 1 |
| $\gamma_2^{\text{true}}$ | 0.01 | True shear component 2 |
| $\text{amp}^{\text{ll}}$ | 400.0 | Lens light amplitude |
| $R_{\text{sersic}}^{\text{ll,true}}$ | 1.0 | True lens light Sersic radius |
| $n_{\text{sersic}}^{\text{ll,true}}$ | 2 | True lens light Sersic index |
| $\text{amp}^{\text{src}}$ | 160 | Source amplitude |
| $R_{\text{sersic}}^{\text{src,true}}$ | 0.5 | True source Sersic radius |
| $n_{\text{sersic}}^{\text{src,true}}$ | 7 | True source Sersic index |
| $x_{\text{src}}^{\text{true}}$ | 0.1 | True source x position |
| $y_{\text{src}}^{\text{true}}$ | 0.3 | True source y position |
| $e_2^{\text{src,true}}$ | 0.1 | True source ellipticity 2 |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula / Operation | Inputs | Outputs |
|------|------|---------------------|--------|---------|
| 1 | Set random seed | `np.random.seed(42)` | seed=42 | Deterministic RNG state |
| 2 | Compute transformation matrix | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}=0.05$ | $\mathbf{T} \in \mathbb{R}^{2\times2}$ |
| 3 | Compute reference coordinates | $c_x = c_y = \frac{N_{\text{pix}}-1}{2} = 49.5$ <br> $\text{ra}_0 = -(\mathbf{T}_{00} c_x + \mathbf{T}_{01} c_y) = 2.475$ <br> $\text{dec}_0 = -(\mathbf{T}_{10} c_x + \mathbf{T}_{11} c_y) = -2.475$ | $\mathbf{T}$, $N_{\text{pix}}=100$ | $\text{ra}_0, \text{dec}_0 \in \mathbb{R}$ |
| 4 | Initialize ImageData | Create ImageData object with $\mathbf{d}_0 = \mathbf{0}_{100\times100}$ | kwargs_data | data_class |
| 5 | Configure Gaussian PSF | $\text{PSF}(\mathbf{r}) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{|\mathbf{r}|^2}{2\sigma^2}\right)$, $\sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$ | FWHM=0.2, truncation=3 | psf_class |
| 6 | Construct EPL lens model | $\boldsymbol{\alpha}^{\text{EPL}}(\mathbf{x}) = \theta_E \left(\frac{\theta_E}{|\mathbf{x}'|}\right)^{\gamma-1} \frac{\mathbf{x}'}{|\mathbf{x}'|}$ where $\mathbf{x}' = $ elliptically transformed coordinates | kwargs_lens_true | lens_model_class |
| 7 | Construct shear model | $\boldsymbol{\alpha}^{\text{shear}} = \begin{pmatrix} \gamma_1 x + \gamma_2 y \\ \gamma_2 x - \gamma_1 y \end{pmatrix}$ | kwargs_shear | Added to lens_model_class |
| 8 | Construct lens light model | $I^{\text{ll}}(\mathbf{r}) = \text{amp} \cdot \exp\left[-b_n\left(\left(\frac{r}{R_{\text{sersic}}}\right)^{1/n} - 1\right)\right]$ | kwargs_sersic | lens_light_model_class |
| 9 | Construct source light model | $I^{\text{src}}(\mathbf{r}) = \text{amp} \cdot \exp\left[-b_n\left(\left(\frac{r'}{R_{\text{sersic}}}\right)^{1/n} - 1\right)\right]$ with elliptical $r'$ | kwargs_sersic_ellipse | source_model_class |
| 10 | Solve lens equation | Find $\{\mathbf{x}_i\}$ such that $\mathbf{x}_i - \boldsymbol{\alpha}(\mathbf{x}_i) = \mathbf{y}_{\text{src}}$ | $\mathbf{y}_{\text{src}}=(0.1, 0.3)$, kwargs_lens_true, numImages=4 | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}} \in \mathbb{R}^{N_{\text{img}}}$ |
| 11 | Compute magnifications | $\mu_i = \frac{1}{\det(\mathbf{I} - \nabla\boldsymbol{\alpha}(\mathbf{x}_i))}$ | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}$, kwargs_lens_true | $\boldsymbol{\mu} \in \mathbb{R}^{N_{\text{img}}}$ |
| 12 | Set point source amplitudes | $A_i = 100 \cdot |\mu_i|$ | $\boldsymbol{\mu}$ | kwargs_ps_true |
| 13 | Simulate clean image | $\mathbf{m} = \text{PSF} * \left[ I^{\text{ll}}(\mathbf{r}) + \sum_i \mathcal{L}^{-1}[I^{\text{src}}](\mathbf{r}_i) + \sum_i A_i \delta(\mathbf{r}-\mathbf{r}_i) \right]$ | All model components | $\mathbf{m} \in \mathbb{R}^{100\times100}$ |
| 14 | Add Poisson noise | $\mathbf{n}^P_{ij} = \frac{\text{Poisson}(\max(0, m_{ij} \cdot t_{\text{exp}}))}{t_{\text{exp}}} - m_{ij}$ | $\mathbf{m}$, $t_{\text{exp}}=100$ | $\mathbf{n}^P \in \mathbb{R}^{100\times100}$ |
| 15 | Add Gaussian noise | $\mathbf{n}^G_{ij} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$ | $\sigma_{\text{bkg}}=0.5$ | $\mathbf{n}^G \in \mathbb{R}^{100\times100}$ |
| 16 | Generate noisy observation | $\mathbf{d} = \mathbf{m} + \mathbf{n}^P + \mathbf{n}^G$ | $\mathbf{m}$, $\mathbf{n}^P$, $\mathbf{n}^G$ | $\mathbf{d} \in \mathbb{R}^{100\times100}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula / Operation | Inputs | Outputs |
|------|------|---------------------|--------|---------|
| 1 | Construct FittingSequence | Assemble kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params | All configuration dicts | fitting_seq object |
| 2 | Define fitting pipeline | fitting_kwargs_list = [['PSO', {...}], ['MCMC', {...}]] | Optimization parameters | Ordered fitting stages |

### 4.2 ITERATION LOOP

#### Stage 1: Particle Swarm Optimization (PSO)

**BEGIN_LOOP** (PSO: $k = 1, \ldots, N_{\text{iter}}^{\text{PSO}}$)

| Step | Name | Formula | Inputs | Outputs | Notes |
|------|------|---------|--------|---------|-------|
| 1.1 | Initialize swarm | $\mathbf{x}_i^{(0)} \sim \mathcal{U}(\boldsymbol{\theta}_{\text{init}} - \sigma_{\text{scale}} \cdot \boldsymbol{\sigma}, \boldsymbol{\theta}_{\text{init}} + \sigma_{\text{scale}} \cdot \boldsymbol{\sigma})$ for $i=1,\ldots,N_{\text{particles}}$ | kwargs_init, kwargs_sigma, $\sigma_{\text{scale}}=1.0$ | $\{\mathbf{x}_i^{(0)}\}_{i=1}^{50}$ | First iteration only |
| 1.2 | Initialize velocities | $\mathbf{v}_i^{(0)} \sim \mathcal{U}(-|\boldsymbol{\theta}_{\text{upper}} - \boldsymbol{\theta}_{\text{lower}}|, |\boldsymbol{\theta}_{\text{upper}} - \boldsymbol{\theta}_{\text{lower}}|)$ | Parameter bounds | $\{\mathbf{v}_i^{(0)}\}_{i=1}^{50}$ | First iteration only |
| 1.3 | Evaluate fitness | $f_i^{(k)} = \mathcal{L}(\mathbf{x}_i^{(k)} | \mathbf{d})$ via forward_operator | $\mathbf{x}_i^{(k)}$, $\mathbf{d}$ | $\{f_i^{(k)}\}_{i=1}^{50}$ | Parallel evaluation |
| 1.4 | Update personal best | $\mathbf{p}_i^{(k)} = \begin{cases} \mathbf{x}_i^{(k)} & \text{if } f_i^{(k)} > f(\mathbf{p}_i^{(k-1)}) \\ \mathbf{p}_i^{(k-1)} & \text{otherwise} \end{cases}$ | $\mathbf{x}_i^{(k)}$, $\mathbf{p}_i^{(k-1)}$, $f_i^{(k)}$ | $\{\mathbf{p}_i^{(k)}\}_{i=1}^{50}$ | |
| 1.5 | Update global best | $\mathbf{g}^{(k)} = \arg\max_{\mathbf{p}_i^{(k)}} f(\mathbf{p}_i^{(k)})$ | $\{\mathbf{p}_i^{(k)}\}_{i=1}^{50}$ | $\mathbf{g}^{(k)}$ | |
| 1.6 | Update velocities | $\mathbf{v}_i^{(k+1)} = \omega \mathbf{v}_i^{(k)} + c_1 r_1 (\mathbf{p}_i^{(k)} - \mathbf{x}_i^{(k)}) + c_2 r_2 (\mathbf{g}^{(k)} - \mathbf{x}_i^{(k)})$ | $\mathbf{v}_i^{(k)}$, $\mathbf{x}_i^{(k)}$, $\mathbf{p}_i^{(k)}$, $\mathbf{g}^{(k)}$, $r_1,r_2 \sim \mathcal{U}(0,1)$ | $\{\mathbf{v}_i^{(k+1)}\}$ | $\omega, c_1, c_2$: lenstronomy defaults |
| 1.7 | Update positions | $\mathbf{x}_i^{(k+1)} = \text{clip}(\mathbf{x}_i^{(k)} + \mathbf{v}_i^{(k+1)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$ | $\mathbf{x}_i^{(k)}$, $\mathbf{v}_i^{(k+1)}$, bounds | $\{\mathbf{x}_i^{(k+1)}\}$ | Enforce bounds |

**END_LOOP** (PSO)

| Step | Name | Formula | Outputs |
|------|------|---------|---------|
| 1.8 | Extract PSO best | $\boldsymbol{\theta}^{\text{PSO}} = \mathbf{g}^{(N_{\text{iter}}^{\text{PSO}})}$ | Best parameters from PSO |

#### Stage 2: Markov Chain Monte Carlo (MCMC)

**BEGIN_LOOP** (MCMC Burn-in: $t = 1, \ldots, N_{\text{burn}}$)

| Step | Name | Formula | Inputs | Outputs | Notes |
|------|------|---------|--------|---------|-------|
| 2.1 | Initialize walkers | $\mathbf{w}_j^{(0)} = \boldsymbol{\theta}^{\text{PSO}} + \sigma_{\text{scale}}^{\text{MCMC}} \cdot \boldsymbol{\sigma} \cdot \boldsymbol{\epsilon}_j$, $\boldsymbol{\epsilon}_j \sim \mathcal{N}(0, \mathbf{I})$ for $j=1,\ldots,N_w$ | $\boldsymbol{\theta}^{\text{PSO}}$, $\sigma_{\text{scale}}=0.1$, $N_w = r_{\text{walker}} \cdot \dim(\boldsymbol{\theta})$ | $\{\mathbf{w}_j^{(0)}\}_{j=1}^{N_w}$ | $r_{\text{walker}}=4$ |
| 2.2 | Propose stretch move | $\mathbf{w}_j^* = \mathbf{w}_k^{(t)} + z(\mathbf{w}_j^{(t)} - \mathbf{w}_k^{(t)})$, $z \sim g(z) \propto \frac{1}{\sqrt{z}}$ for $z \in [1/a, a]$, $a=2$ | $\mathbf{w}_j^{(t)}$, randomly chosen $\mathbf{w}_k^{(t)}$ | $\mathbf{w}_j^*$ | Affine-invariant ensemble sampler |
| 2.3 | Compute acceptance | $\alpha = \min\left(1, z^{d-1} \frac{\mathcal{L}(\mathbf{w}_j^* | \mathbf{d})}{\mathcal{L}(\mathbf{w}_j^{(t)} | \mathbf{d})}\right)$ | $\mathbf{w}_j^*$, $\mathbf{w}_j^{(t)}$, $d=\dim(\boldsymbol{\theta})$ | $\alpha \in [0,1]$ | |
| 2.4 | Accept/Reject | $\mathbf{w}_j^{(t+1)} = \begin{cases} \mathbf{w}_j^* & \text{if } u < \alpha, \; u \sim \mathcal{U}(0,1) \\ \mathbf{w}_j^{(t)} & \text{otherwise} \end{cases}$ | $\mathbf{w}_j^*$, $\alpha$, $u$ | $\mathbf{w}_j^{(t+1)}$ | |

**END_LOOP** (MCMC Burn-in)

**BEGIN_LOOP** (MCMC Sampling: $t = 1, \ldots, N_{\text{run}}$)

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 2.5 | Stretch move proposal | Same as Step 2.2 | Walker ensemble | Proposed positions |
| 2.6 | Acceptance probability | Same as Step 2.3 | Proposals, current positions | Acceptance ratios |
| 2.7 | Accept/Reject & store | Same as Step 2.4; append accepted samples to chain | Proposals, acceptances | $\{\mathbf{w}_j^{(t)}\}$ stored |

**END_LOOP** (MCMC Sampling)

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Extract best fit | $\boldsymbol{\theta}^* = \arg\max_{\mathbf{w} \in \text{chain}} \mathcal{L}(\mathbf{w} | \mathbf{d})$ | Full MCMC chain | kwargs_result |
| 2 | Compute model image | $\mathbf{m}^* = \mathcal{F}(\boldsymbol{\theta}^*)$ | $\boldsymbol{\theta}^*$ | model_image |
| 3 | Compute residuals | $\mathbf{r} = \mathbf{d} - \mathbf{m}^*$ | $\mathbf{d}$, $\mathbf{m}^*$ | residual_image |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit (sequential stages)
- **expression**: 
  - PSO: Terminate when $k = N_{\text{iter}}^{\text{PSO}} = 10$
  - MCMC: Terminate when $t_{\text{burn}} = N_{\text{burn}} = 10$ AND $t_{\text{run}} = N_{\text{run}} = 10$
- **parameters**:
  - $N_{\text{iter}}^{\text{PSO}} = 10$
  - $N_{\text{burn}} = 10$
  - $N_{\text{run}} = 10$
  - $N_{\text{particles}} = 50$
  - $N_{\text{walkers}} = 4 \times \dim(\boldsymbol{\theta})$
- **check_frequency**: After each complete iteration of respective stage

---

## 6. EVALUATION METRICS

| Metric | Formula | Description |
|--------|---------|-------------|
| Chi-squared | $\chi^2 = \sum_{i,j} \left(\frac{d_{ij} - m_{ij}^*}{\sigma_{\text{bkg}}}\right)^2$ | Sum of squared normalized residuals |
| Reduced chi-squared | $\chi^2_{\text{red}} = \frac{\chi^2}{N_{\text{pix}}^2 - 1} = \frac{\chi^2}{9999}$ | Chi-squared per degree of freedom |
| Residual RMS | $\text{RMS} = \sqrt{\frac{1}{N_{\text{pix}}^2}\sum_{i,j}(d_{ij} - m_{ij}^*)^2}$ | Root mean square of residuals |
| Max absolute residual | $\|r\|_{\