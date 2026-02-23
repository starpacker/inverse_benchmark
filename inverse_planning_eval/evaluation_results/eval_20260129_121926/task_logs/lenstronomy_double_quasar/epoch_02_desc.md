# Algorithmic Specification Document: Gravitational Lens Modeling Inversion

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \left[ -\ln \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) \right] \quad \text{subject to} \quad \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$$

where $\boldsymbol{\theta} = \{\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\theta}_{\text{source}}, \boldsymbol{\theta}_{\text{lens\_light}}, \boldsymbol{\theta}_{\text{ps}}\}$

**data_fidelity_term**:
$$-\ln \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) = \frac{1}{2} \sum_{i,j} \left( \frac{d_{ij} - m_{ij}(\boldsymbol{\theta})}{\sigma_{\text{bkg}}} \right)^2 + \frac{1}{2} \sum_{k=1}^{N_{\text{img}}} \frac{(\mathbf{x}_k^{\text{obs}} - \mathbf{x}_k^{\text{model}})^2}{\sigma_{\text{pos}}^2}$$

where:
- $d_{ij}$: observed image pixel value at position $(i,j)$
- $m_{ij}(\boldsymbol{\theta})$: forward model prediction
- $\sigma_{\text{bkg}} = 0.5$: background RMS noise
- $\sigma_{\text{pos}} = 0.004$: image position uncertainty
- $\mathbf{x}_k$: image positions (RA, Dec)

**regularization_term**: None (implicit regularization through parameter bounds and priors)

**constraints**:
| Constraint | Expression |
|------------|------------|
| Lens Einstein radius | $0 \leq \theta_E \leq 10$ |
| Power-law slope | $1.5 \leq \gamma \leq 2.5$ |
| Ellipticity components | $-0.5 \leq e_1, e_2 \leq 0.5$ |
| Shear components | $-0.2 \leq \gamma_1, \gamma_2 \leq 0.2$ |
| Centroid positions | $-10 \leq x_c, y_c \leq 10$ |
| Sersic radius | $0.001 \leq R_{\text{sersic}} \leq 10$ |
| Sersic index | $0.5 \leq n_{\text{sersic}} \leq 5$ |
| Source position tolerance | $|\mathbf{x}_{\text{source}}^{\text{ps}} - \mathbf{x}_{\text{source}}^{\text{light}}| < 0.001$ |

---

## 2. VARIABLES

### Primal Variables (Parameters to be optimized)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta_E$ | primal | scalar | float64 | 1.1 | Einstein radius of EPL lens |
| $\gamma$ | primal | scalar | float64 | 2.0 | Power-law slope of mass profile |
| $e_1^{\text{lens}}, e_2^{\text{lens}}$ | primal | (2,) | float64 | (0, 0) | Lens ellipticity components |
| $x_c^{\text{lens}}, y_c^{\text{lens}}$ | primal | (2,) | float64 | (0, 0) | Lens centroid position |
| $\gamma_1, \gamma_2$ | primal | (2,) | float64 | (0, 0) | External shear components |
| $R_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 0.03 | Source Sersic effective radius |
| $n_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}, e_2^{\text{src}}$ | primal | (2,) | float64 | (0, 0) | Source ellipticity components |
| $x_c^{\text{src}}, y_c^{\text{src}}$ | primal | (2,) | float64 | (0, 0) | Source centroid (linked to point source) |
| $R_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 0.1 | Lens light Sersic radius |
| $n_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 1.0 | Lens light Sersic index |
| $e_1^{\text{ll}}, e_2^{\text{ll}}$ | primal | (2,) | float64 | (0, 0) | Lens light ellipticity |
| $x_c^{\text{ll}}, y_c^{\text{ll}}$ | primal | (2,) | float64 | (0, 0) | Lens light centroid |
| $\boldsymbol{\alpha}_{\text{img}}$ | primal | $(N_{\text{img}},)$ | float64 | $\mathbf{x}_{\text{img}} + 0.01$ | Point source RA positions |
| $\boldsymbol{\delta}_{\text{img}}$ | primal | $(N_{\text{img}},)$ | float64 | $\mathbf{y}_{\text{img}} - 0.01$ | Point source Dec positions |
| $\mathbf{A}_{\text{ps}}$ | primal | $(N_{\text{img}},)$ | float64 | linear solve | Point source amplitudes |
| $A_{\text{src}}, A_{\text{ll}}$ | primal | (2,) | float64 | linear solve | Light model amplitudes |

### Constants

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $N_{\text{pix}}$ | constants | scalar | int | 100 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | constants | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\sigma_{\text{bkg}}$ | constants | scalar | float64 | 0.5 | Background RMS noise |
| $t_{\text{exp}}$ | constants | scalar | float64 | 100 | Exposure time (seconds) |
| $\text{FWHM}$ | constants | scalar | float64 | 0.2 | PSF full-width half-maximum |
| $N_{\text{particles}}$ | constants | scalar | int | 50 | PSO particle count |
| $N_{\text{iter}}^{\text{PSO}}$ | constants | scalar | int | 10 | PSO iterations |
| $N_{\text{burn}}$ | constants | scalar | int | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | constants | scalar | int | 10 | MCMC sampling steps |
| $r_{\text{walker}}$ | constants | scalar | int | 4 | MCMC walker ratio |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{d}$ | observations | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | simulated + noise | Observed noisy image |
| $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}$ | observations | $(N_{\text{img}},)$ | float64 | lens equation solution | Observed image positions |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Coordinate System Setup
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | coordinate_transform_initialization |
| formula | $$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$ $$\alpha_0 = -\left(T_{00} \cdot \frac{N_{\text{pix}}-1}{2} + T_{01} \cdot \frac{N_{\text{pix}}-1}{2}\right)$$ $$\delta_0 = -\left(T_{10} \cdot \frac{N_{\text{pix}}-1}{2} + T_{11} \cdot \frac{N_{\text{pix}}-1}{2}\right)$$ |
| inputs | $N_{\text{pix}}$: scalar, $\Delta_{\text{pix}}$: scalar |
| outputs | $\mathbf{T}$: $(2,2)$, $\alpha_0$: scalar, $\delta_0$: scalar |

### Step 3.2: PSF Kernel Construction
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | gaussian_psf_construction |
| formula | $$\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$$ $$\text{PSF}(x,y) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{x^2 + y^2}{2\sigma_{\text{PSF}}^2}\right)$$ Truncated at $3\sigma_{\text{PSF}}$ |
| inputs | $\text{FWHM}$: scalar, $\Delta_{\text{pix}}$: scalar |
| outputs | $\mathbf{K}_{\text{PSF}}$: $(N_k, N_k)$ where $N_k = 2\lfloor 3\sigma_{\text{PSF}}/\Delta_{\text{pix}} \rfloor + 1$ |

### Step 3.3: Lens Equation Solution
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | image_position_finding |
| formula | Solve for $\mathbf{x}_{\text{img}}$ satisfying: $$\boldsymbol{\beta} = \mathbf{x}_{\text{img}} - \boldsymbol{\alpha}(\mathbf{x}_{\text{img}})$$ where deflection angle: $$\boldsymbol{\alpha}(\mathbf{x}) = \boldsymbol{\alpha}_{\text{EPL}}(\mathbf{x}; \theta_E, \gamma, e_1, e_2) + \boldsymbol{\alpha}_{\text{shear}}(\mathbf{x}; \gamma_1, \gamma_2)$$ |
| inputs | $\boldsymbol{\beta} = (0.1, 0.3)$: source position, $\boldsymbol{\theta}_{\text{lens}}^{\text{true}}$: lens params |
| outputs | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}$: $(N_{\text{img}},)$ each |

### Step 3.4: Magnification Computation
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | magnification_calculation |
| formula | $$\mu_k = \frac{1}{\det(\mathbf{J}_k)}$$ where Jacobian: $$\mathbf{J}_k = \mathbf{I} - \frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_k} = \begin{pmatrix} 1 - \kappa - \gamma_{\text{eff},1} & -\gamma_{\text{eff},2} \\ -\gamma_{\text{eff},2} & 1 - \kappa + \gamma_{\text{eff},1} \end{pmatrix}$$ Point source amplitudes: $A_k^{\text{ps}} = |\mu_k| \times 100$ |
| inputs | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}$: $(N_{\text{img}},)$, $\boldsymbol{\theta}_{\text{lens}}$ |
| outputs | $\boldsymbol{\mu}$: $(N_{\text{img}},)$, $\mathbf{A}^{\text{ps}}$: $(N_{\text{img}},)$ |

### Step 3.5: Clean Image Simulation
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | forward_model_simulation |
| formula | $$m_{ij}^{\text{clean}} = \left[\mathbf{K}_{\text{PSF}} * \left( I_{\text{lens\_light}} + I_{\text{source\_lensed}} + I_{\text{ps}} \right)\right]_{ij}$$ Component definitions in Step 4.4 |
| inputs | All true model parameters, $\mathbf{K}_{\text{PSF}}$: $(N_k, N_k)$ |
| outputs | $\mathbf{m}^{\text{clean}}$: $(N_{\text{pix}}, N_{\text{pix}})$ |

### Step 3.6: Poisson Noise Generation
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | poisson_noise_addition |
| formula | $$C_{ij} = \max(0, m_{ij}^{\text{clean}} \cdot t_{\text{exp}})$$ $$\tilde{C}_{ij} \sim \text{Poisson}(\lfloor C_{ij} \rfloor)$$ $$\eta_{ij}^{\text{Poisson}} = \frac{\tilde{C}_{ij}}{t_{\text{exp}}} - m_{ij}^{\text{clean}}$$ |
| inputs | $\mathbf{m}^{\text{clean}}$: $(N_{\text{pix}}, N_{\text{pix}})$, $t_{\text{exp}}$: scalar |
| outputs | $\boldsymbol{\eta}^{\text{Poisson}}$: $(N_{\text{pix}}, N_{\text{pix}})$ |

### Step 3.7: Gaussian Background Noise Generation
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | gaussian_noise_addition |
| formula | $$\eta_{ij}^{\text{bkg}} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$$ |
| inputs | $\sigma_{\text{bkg}}$: scalar, image shape: $(N_{\text{pix}}, N_{\text{pix}})$ |
| outputs | $\boldsymbol{\eta}^{\text{bkg}}$: $(N_{\text{pix}}, N_{\text{pix}})$ |

### Step 3.8: Noisy Observation Assembly
| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | observation_combination |
| formula | $$d_{ij} = m_{ij}^{\text{clean}} + \eta_{ij}^{\text{bkg}} + \eta_{ij}^{\text{Poisson}}$$ |
| inputs | $\mathbf{m}^{\text{clean}}$, $\boldsymbol{\eta}^{\text{bkg}}$, $\boldsymbol{\eta}^{\text{Poisson}}$: all $(N_{\text{pix}}, N_{\text{pix}})$ |
| outputs | $\mathbf{d}$: $(N_{\text{pix}}, N_{\text{pix}})$ |

---

## 4. ALGORITHM STEPS

### Initialization

| Step | Field | Value |
|------|-------|-------|
| I.1 | step_name | parameter_initialization |
| | formula | $$\boldsymbol{\theta}^{(0)} = \boldsymbol{\theta}_{\text{init}}$$ $$\boldsymbol{\sigma}_{\theta} = \text{[defined per parameter]}$$ |
| | inputs | Initial parameter dictionaries |
| | outputs | $\boldsymbol{\theta}^{(0)}$: parameter vector, $\boldsymbol{\sigma}_{\theta}$: step sizes |
| | computational_note | Parameters grouped: lens (8), source (6), lens_light (6), point_source ($2N_{\text{img}}$) |
| I.2 | step_name | pso_swarm_initialization |
| | formula | For particle $p = 1, \ldots, N_{\text{particles}}$: $$\boldsymbol{\theta}_p^{(0)} \sim \mathcal{U}(\boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$$ $$\mathbf{v}_p^{(0)} \sim \mathcal{U}(-|\boldsymbol{\theta}_{\text{upper}} - \boldsymbol{\theta}_{\text{lower}}|, |\boldsymbol{\theta}_{\text{upper}} - \boldsymbol{\theta}_{\text{lower}}|)$$ $$\mathbf{p}_p^{\text{best}} = \boldsymbol{\theta}_p^{(0)}, \quad f_p^{\text{best}} = f(\boldsymbol{\theta}_p^{(0)})$$ $$\mathbf{g}^{\text{best}} = \arg\min_p f_p^{\text{best}}$$ |
| | inputs | $N_{\text{particles}}$, bounds $\boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}$ |
| | outputs | $\{\boldsymbol{\theta}_p^{(0)}\}_{p=1}^{N_{\text{particles}}}$, $\{\mathbf{v}_p^{(0)}\}$, $\mathbf{g}^{\text{best}}$ |
| | computational_note | $f(\boldsymbol{\theta}) = -\ln \mathcal{L}(\mathbf{d}|\boldsymbol{\theta})$ is the objective |

---

### Iteration Loop: PSO Phase

**BEGIN PSO_ITERATION_LOOP** (for $t = 1, \ldots, N_{\text{iter}}^{\text{PSO}}$)

| Step | Field | Value |
|------|-------|-------|
| P.1 | step_name | velocity_update |
| | formula | For each particle $p$: $$\mathbf{v}_p^{(t)} = \omega \mathbf{v}_p^{(t-1)} + c_1 r_1 \odot (\mathbf{p}_p^{\text{best}} - \boldsymbol{\theta}_p^{(t-1)}) + c_2 r_2 \odot (\mathbf{g}^{\text{best}} - \boldsymbol{\theta}_p^{(t-1)})$$ where $\omega = 0.7298$ (inertia), $c_1 = c_2 = 1.4962$ (cognitive/social coefficients), $r_1, r_2 \sim \mathcal{U}(0,1)^{d}$ |
| | inputs | $\mathbf{v}_p^{(t-1)}$, $\boldsymbol{\theta}_p^{(t-1)}$, $\mathbf{p}_p^{\text{best}}$, $\mathbf{g}^{\text{best}}$ |
| | outputs | $\mathbf{v}_p^{(t)}$: $(d,)$ |
| | computational_note | $\odot$ denotes element-wise multiplication; $\sigma_{\text{scale}}=1$ scales initial velocity range |
| P.2 | step_name | position_update |
| | formula | $$\tilde{\boldsymbol{\theta}}_p^{(t)} = \boldsymbol{\theta}_p^{(t-1)} + \mathbf{v}_p^{(t)}$$ $$\boldsymbol{\theta}_p^{(t)} = \text{clip}(\tilde{\boldsymbol{\theta}}_p^{(t)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$$ |
| | inputs | $\boldsymbol{\theta}_p^{(t-1)}$, $\mathbf{v}_p^{(t)}$, bounds |
| | outputs | $\boldsymbol{\theta}_p^{(t)}$: $(d,)$ |
| | computational_note | Boundary handling via clipping |
| P.3 | step_name | fitness_evaluation |
| | formula | $$f_p^{(t)} = -\ln \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}_p^{(t)})$$ (computed via forward model, see Step P.4) |
| | inputs | $\boldsymbol{\theta}_p^{(t)}$, $\mathbf{d}$ |
| | outputs | $f_p^{(t)}$: scalar |
| | computational_note | Includes bounds checking; returns $\infty$ if constraints violated |
| P.4 | step_name | forward_model_evaluation |
| | formula | **Lens light (Sersic):** $$I_{\text{ll}}(\mathbf{x}) = A_{\text{ll}} \exp\left( -b_n \left[ \left(\frac{r}{R_{\text{sersic}}^{\text{ll}}}\right)^{1/n_{\text{sersic}}^{\text{ll}}} - 1 \right] \right)$$ where $b_n \approx 1.9992n - 0.3271$ **Lensed source (Sersic ellipse):** $$I_{\text{src}}(\mathbf{x}) = A_{\text{src}} \exp\left( -b_n \left[ \left(\frac{r_{\text{ell}}}{R_{\text{sersic}}^{\text{src}}}\right)^{1/n_{\text{sersic}}^{\text{src}}} - 1 \right] \right)$$ evaluated at source plane: $\boldsymbol{\beta} = \mathbf{x} - \boldsymbol{\alpha}(\mathbf{x})$ **Point sources:** $$I_{\text{ps}}(\mathbf{x}) = \sum_{k=1}^{N_{\text{img}}} A_k^{\text{ps}} \cdot \delta(\mathbf{x} - \mathbf{x}_k^{\text{img}})$$ **Total model:** $$m_{ij} = \left[ \mathbf{K}_{\text{PSF}} * (I_{\text{ll}} + I_{\text{src}} + I_{\text{ps}}) \right]_{ij}$$ |
| | inputs | $\boldsymbol{\theta}_p^{(t)}$, $\mathbf{K}_{\text{PSF}}$, pixel grid |
| | outputs | $\mathbf{m}$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| | computational_note | Amplitudes solved via linear least squares (multi-linear mode) |
| P.5 | step_name | personal_best_update |
| | formula | $$\text{If } f_p^{(t)} < f_p^{\text{best}}: \quad \mathbf{p}_p^{\text{best}} \leftarrow \boldsymbol{\theta}_p^{(t)}, \quad f_p^{\text{best}} \leftarrow f_p^{(t)}$$ |
| | inputs | $f_p^{(t)}$, $f_p^{\text{best}}$, $\boldsymbol{\theta}_p^{(t)}$ |
| | outputs | Updated $\mathbf{p}_p^{\text{best}}$, $f_p^{\text{best}}$ |
| | computational_note | Per-particle memory |
| P.6 | step_name | global_best_update |
| | formula | $$p^* = \arg\min_p f_p^{\text{best}}$$ $$\text{If } f_{p^*}^{\text{best}} < f(\mathbf{g}^{\text{best}}): \quad \mathbf{g}^{\text{best}} \leftarrow \mathbf{p}_{p^*}^{\text{best}}$$ |
| | inputs | All $\{f_p^{\text{best}}, \mathbf{p}_p^{\text{best}}\}_{p=1}^{N_{\text{particles}}}$ |
| | outputs | Updated $\mathbf{g}^{\text{best}}$ |
| | computational_note | Swarm-wide synchronization |

**END PSO_ITERATION_LOOP**

---

### Iteration Loop: MCMC Phase

| Step | Field | Value |
|------|-------|-------|
| M.0 | step_name | mcmc_initialization |
| | formula | $$\boldsymbol{\theta}_w^{(0)} = \mathbf{g}^{\text{best}} + \sigma_{\text{scale}} \cdot \boldsymbol{\sigma}_{\theta} \odot \boldsymbol{\xi}_w, \quad \boldsymbol{\xi}_w \sim \mathcal{N}(0, \mathbf{I})$$ for $w = 1, \ldots, N_{\text{walkers}}$ where $N_{\text{walkers}} = r_{\text{walker}} \cdot d$ |
| | inputs | $\mathbf{g}^{\text{best}}$, $\boldsymbol{\sigma}_{\theta}$, $\sigma_{\text{scale}} = 0.1$, $r_{\text{walker}} = 4$ |
| | outputs | $\{\boldsymbol{\theta}_w^{(0)}\}_{w=1}^{N_{\text{walkers}}}$ |
| | computational_note | Walkers initialized in ball around PSO best |

**BEGIN MCMC_BURN_IN_LOOP** (for $t = 1, \ldots, N_{\text{burn}}$)

| Step | Field | Value |
|------|-------|-------|
| M.1 | step_name | stretch_move_proposal |
| | formula | For walker $w$, select complementary walker $w' \neq w$ uniformly: $$z \sim g(z) \propto \frac{1}{\sqrt{z}} \quad \text{for } z \in [1/a, a], \; a=2$$ $$\boldsymbol{\theta}_w^{\text{prop}} = \boldsymbol{\theta}_{w'}^{(t-1)} + z \cdot (\boldsymbol{\theta}_w^{(t-1)} - \boldsymbol{\theta}_{w'}^{(t-1)})$$ |
| | inputs | $\boldsymbol{\theta}_w^{(t-1)}$, complementary set |
| | outputs | $\boldsymbol{\theta}_w^{\text{prop}}$, $z$ |
| | computational_note | Affine-invariant ensemble sampler (emcee algorithm) |
| M.2 | step_name | acceptance_probability |
| | formula | $$q = z^{d-1} \cdot \frac{\mathcal{L}(\mathbf{d}|\boldsymbol{\theta}_w^{\text{prop}})}{\mathcal{L}(\mathbf{d}|\boldsymbol{\theta}_w^{(t-1)})} = z^{d-1} \cdot \exp\left( f(\boldsymbol{\theta}_w^{(t-1)}) - f(\boldsymbol{\theta}_w^{\text{prop}}) \right)$$ $$\alpha = \min(1, q)$$ |
| | inputs | $\boldsymbol{\theta}_w^{\text{prop}}$, $\boldsymbol{\theta}_w^{(t-1)}$, $z$, $d$ |
| | outputs | $\alpha$: scalar |
| | computational_note | $d$ is parameter dimension |
| M.3 | step_name | accept_reject_step |
| | formula | $$u \sim \mathcal{U}(0,1)$$ $$\boldsymbol{\theta}_w^{(t)} = \begin{cases} \boldsymbol{\theta}_w^{\text{prop}} & \text{if } u < \alpha \\ \boldsymbol{\theta}_w^{(t-1)} & \text{otherwise} \end{cases}$$ |
| | inputs | $\alpha$, $\boldsymbol{\theta}_w^{\text{prop}}$, $\boldsymbol{\theta}_w^{(t-1)}$ |
| | outputs | $\boldsymbol{\theta}_w^{(t)}$ |
| | computational_note | Metropolis-Hastings acceptance |

**END MCMC_BURN_IN_LOOP**

**BEGIN MCMC_SAMPLING_LOOP** (for $t = N_{\text{burn}}+1, \ldots, N_{\text{burn}}+N_{\text{run}}$)

| Step |