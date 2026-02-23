## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Gravitational Lens Modeling via PSO-MCMC Hybrid Optimization
- **algorithm_family**: Bayesian Inference / Metaheuristic Optimization (Particle Swarm Optimization + Markov Chain Monte Carlo)
- **summary**: Estimates gravitational lens mass distribution and light profile parameters from simulated lensed images using sequential PSO for global search followed by MCMC for posterior sampling.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \left[ \log \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) + \log \pi(\boldsymbol{\theta}) \right]$$

subject to:
$$\boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$$

### data_fidelity_term
$$\log \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) = -\frac{1}{2} \sum_{i,j} \frac{\left( d_{i,j} - \mathcal{F}(\boldsymbol{\theta})_{i,j} \right)^2}{\sigma_{i,j}^2}$$

where:
- $d_{i,j}$: observed image pixel value at position $(i,j)$
- $\mathcal{F}(\boldsymbol{\theta})_{i,j}$: forward model prediction
- $\sigma_{i,j}^2 = \sigma_{\text{bkg}}^2 + \max(0, \mathcal{F}(\boldsymbol{\theta})_{i,j}) / t_{\text{exp}}$ (combined background and Poisson variance)

### regularization_term
**Gaussian Prior on Lens Ellipticity:**
$$\log \pi(\boldsymbol{\theta}) = -\frac{1}{2}\left( \frac{e_1^2}{\sigma_{e_1}^2} + \frac{e_2^2}{\sigma_{e_2}^2} \right)$$

with $\sigma_{e_1} = \sigma_{e_2} = 0.2$, centered at $\mu_{e_1} = \mu_{e_2} = 0$

### constraints
| Constraint | Expression |
|------------|------------|
| Lens Einstein radius | $0 \leq \theta_E \leq 10$ |
| Lens power-law slope | $1.5 \leq \gamma \leq 2.5$ |
| Lens ellipticity | $-0.5 \leq e_1, e_2 \leq 0.5$ |
| Lens center | $-10 \leq c_x, c_y \leq 10$ |
| External shear | $-0.5 \leq \gamma_1, \gamma_2 \leq 0.5$ |
| Source Sersic radius | $0.001 \leq R_{\text{sersic}} \leq 10$ |
| Source Sersic index | $0.5 \leq n_{\text{sersic}} \leq 5$ |
| Point source positions | $-10 \leq \text{ra}, \text{dec} \leq 10$ |
| Source position tolerance | $\|\boldsymbol{\beta}_{\text{ps}} - \boldsymbol{\beta}_{\text{source}}\| < 0.001$ arcsec |

---

## 2. VARIABLES

### Primal Variables (Fitted Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta_E$ | primal | scalar | float64 | 1.2 | Einstein radius (arcsec) |
| $\gamma$ | primal | scalar | float64 | 2.0 | EPL power-law slope |
| $e_1^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens ellipticity component 2 |
| $c_x^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens center x (arcsec) |
| $c_y^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens center y (arcsec) |
| $\gamma_1$ | primal | scalar | float64 | 0.0 | External shear component 1 |
| $\gamma_2$ | primal | scalar | float64 | 0.0 | External shear component 2 |
| $R_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 0.03 | Source Sersic radius (arcsec) |
| $n_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source ellipticity component 1 |
| $e_2^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source ellipticity component 2 |
| $c_x^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source center x (arcsec) |
| $c_y^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source center y (arcsec) |
| $R_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 0.1 | Lens light Sersic radius |
| $n_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 1.0 | Lens light Sersic index |
| $e_1^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light ellipticity 1 |
| $e_2^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light ellipticity 2 |
| $c_x^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light center x |
| $c_y^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light center y |
| $\boldsymbol{\alpha}^{\text{ps}}$ | primal | (4,) | float64 | $x_{\text{image}}^{\text{true}} + 0.01$ | Point source RA positions |
| $\boldsymbol{\delta}^{\text{ps}}$ | primal | (4,) | float64 | $y_{\text{image}}^{\text{true}} - 0.01$ | Point source Dec positions |

### Auxiliary Variables (Linear Amplitude Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $A^{\text{src}}$ | auxiliary | scalar | float64 | Linear solve | Source amplitude |
| $A^{\text{ll}}$ | auxiliary | scalar | float64 | Linear solve | Lens light amplitude |
| $\mathbf{A}^{\text{ps}}$ | auxiliary | (4,) | float64 | Linear solve | Point source amplitudes |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{d}$ | observations | (100, 100) | float64 | Simulated data | Observed image (counts/s) |

### Constants

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\sigma_{\text{bkg}}$ | constants | scalar | float64 | 0.5 | Background RMS noise |
| $t_{\text{exp}}$ | constants | scalar | float64 | 100.0 | Exposure time (s) |
| $N_{\text{pix}}$ | constants | scalar | int | 100 | Image size (pixels) |
| $\Delta_{\text{pix}}$ | constants | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\text{FWHM}$ | constants | scalar | float64 | 0.1 | PSF FWHM (arcsec) |
| $N_{\text{particles}}$ | constants | scalar | int | 50 | PSO particle count |
| $N_{\text{iter}}^{\text{PSO}}$ | constants | scalar | int | 10 | PSO iterations |
| $N_{\text{walkers}}$ | constants | scalar | int | 50 | MCMC walker count |
| $N_{\text{burn}}$ | constants | scalar | int | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | constants | scalar | int | 10 | MCMC production steps |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | constants | scalar | float64 | 1.0 | PSO sigma scale |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | constants | scalar | float64 | 0.1 | MCMC sigma scale |
| $\sigma_{\text{pos}}$ | constants | scalar | float64 | 0.004 | Image position uncertainty (arcsec) |
| $\tau_{\text{src}}$ | constants | scalar | float64 | 0.001 | Source position tolerance (arcsec) |
| $\sigma_{\text{src}}$ | constants | scalar | float64 | 0.001 | Source position sigma (arcsec) |
| $\text{trunc}$ | constants | scalar | int | 5 | PSF truncation (in FWHM units) |

### True Parameters (Ground Truth for Simulation)

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\theta_E^{\text{true}}$ | constants | scalar | float64 | 1.0 | True Einstein radius |
| $\gamma^{\text{true}}$ | constants | scalar | float64 | 1.96 | True power-law slope |
| $e_1^{\text{lens,true}}$ | constants | scalar | float64 | 0.1 | True lens ellipticity 1 |
| $e_2^{\text{lens,true}}$ | constants | scalar | float64 | 0.2 | True lens ellipticity 2 |
| $\phi^{\text{shear}}$ | constants | scalar | float64 | 0.1 | Shear angle (rad) |
| $\gamma^{\text{shear}}$ | constants | scalar | float64 | 0.02 | Shear magnitude |
| $A^{\text{ll,true}}$ | constants | scalar | float64 | 400.0 | True lens light amplitude |
| $R_{\text{sersic}}^{\text{ll,true}}$ | constants | scalar | float64 | 1.0 | True lens light Sersic radius |
| $n_{\text{sersic}}^{\text{ll,true}}$ | constants | scalar | float64 | 2.0 | True lens light Sersic index |
| $A^{\text{src,true}}$ | constants | scalar | float64 | 4000.0 | True source amplitude |
| $R_{\text{sersic}}^{\text{src,true}}$ | constants | scalar | float64 | 0.1 | True source Sersic radius |
| $n_{\text{sersic}}^{\text{src,true}}$ | constants | scalar | float64 | 3.0 | True source Sersic index |
| $\beta_x^{\text{true}}$ | constants | scalar | float64 | 0.0 | True source position x |
| $\beta_y^{\text{true}}$ | constants | scalar | float64 | 0.1 | True source position y |
| $e_1^{\text{src,true}}$ | constants | scalar | float64 | -0.1 | True source ellipticity 1 |
| $e_2^{\text{src,true}}$ | constants | scalar | float64 | 0.01 | True source ellipticity 2 |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Coordinate Transform Setup | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}$ | $\mathbf{T} \in \mathbb{R}^{2 \times 2}$ |
| 2 | Reference Coordinate Computation | $\begin{aligned} c_x &= (N_{\text{pix}} - 1)/2 \\ c_y &= (N_{\text{pix}} - 1)/2 \\ \text{RA}_0 &= -(T_{00} c_x + T_{01} c_y) \\ \text{Dec}_0 &= -(T_{10} c_x + T_{11} c_y) \end{aligned}$ | $\mathbf{T}, N_{\text{pix}}$ | $\text{RA}_0, \text{Dec}_0 \in \mathbb{R}$ |
| 3 | PSF Construction | $\text{PSF}(r) = \exp\left(-\frac{r^2}{2\sigma_{\text{PSF}}^2}\right), \quad \sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$ | FWHM=0.1, $\Delta_{\text{pix}}$=0.05, trunc=5 | PSF kernel |
| 4 | Shear Cartesian Conversion | $\begin{aligned} \gamma_1^{\text{true}} &= \gamma^{\text{shear}} \cos(2\phi^{\text{shear}}) \\ \gamma_2^{\text{true}} &= \gamma^{\text{shear}} \sin(2\phi^{\text{shear}}) \end{aligned}$ | $\phi=0.1$, $\gamma=0.02$ | $\gamma_1^{\text{true}}, \gamma_2^{\text{true}}$ |
| 5 | Lens Equation Solving | Find $\{\mathbf{x}_i\}_{i=1}^4$ such that $\boldsymbol{\beta}^{\text{true}} = \mathbf{x}_i - \boldsymbol{\alpha}(\mathbf{x}_i; \boldsymbol{\theta}_{\text{lens}}^{\text{true}})$ | $\boldsymbol{\beta}^{\text{true}}$, lens params | $\mathbf{x}_{\text{image}} \in \mathbb{R}^{4 \times 2}$ |
| 6 | Magnification Computation | $\mu_i = \left| \det\left( \mathbf{I} - \frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{x}} \bigg|_{\mathbf{x}_i} \right) \right|^{-1}$ | $\mathbf{x}_{\text{image}}$, lens params | $\boldsymbol{\mu} \in \mathbb{R}^4$ |
| 7 | Point Source Amplitude | $A_i^{\text{ps}} = 100 \cdot (\mu_i + \epsilon_i), \quad \epsilon_i \sim \mathcal{N}(0, 0.5)$ | $\boldsymbol{\mu}$ | $\mathbf{A}^{\text{ps}} \in \mathbb{R}^4$ |
| 8 | Clean Image Generation | $\mathbf{I}_{\text{clean}} = \mathcal{F}(\boldsymbol{\theta}^{\text{true}})$ | All true params | $\mathbf{I}_{\text{clean}} \in \mathbb{R}^{100 \times 100}$ |
| 9 | Poisson Noise Generation | $\begin{aligned} \mathbf{C} &= \max(0, \mathbf{I}_{\text{clean}} \cdot t_{\text{exp}}) \\ \mathbf{C}_{\text{noisy}} &\sim \text{Poisson}(\mathbf{C}) \\ \mathbf{n}_{\text{Poisson}} &= \mathbf{C}_{\text{noisy}}/t_{\text{exp}} - \mathbf{I}_{\text{clean}} \end{aligned}$ | $\mathbf{I}_{\text{clean}}$, $t_{\text{exp}}=100$ | $\mathbf{n}_{\text{Poisson}} \in \mathbb{R}^{100 \times 100}$ |
| 10 | Gaussian Noise Generation | $\mathbf{n}_{\text{bkg}} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2 \mathbf{I})$ | $\sigma_{\text{bkg}}=0.5$ | $\mathbf{n}_{\text{bkg}} \in \mathbb{R}^{100 \times 100}$ |
| 11 | Noisy Image Formation | $\mathbf{d} = \mathbf{I}_{\text{clean}} + \mathbf{n}_{\text{bkg}} + \mathbf{n}_{\text{Poisson}}$ | $\mathbf{I}_{\text{clean}}, \mathbf{n}_{\text{bkg}}, \mathbf{n}_{\text{Poisson}}$ | $\mathbf{d} \in \mathbb{R}^{100 \times 100}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Initialize Parameter Vector | $\boldsymbol{\theta}^{(0)} = [\theta_E^{(0)}, \gamma^{(0)}, e_1^{(0)}, e_2^{(0)}, c_x^{(0)}, c_y^{(0)}, \gamma_1^{(0)}, \gamma_2^{(0)}, \ldots]$ | Initial values from Section 2 | $\boldsymbol{\theta}^{(0)} \in \mathbb{R}^{D}$ |
| 2 | Initialize PSO Particles | $\boldsymbol{\theta}_p^{(0)} = \boldsymbol{\theta}^{(0)} + \sigma_{\text{scale}}^{\text{PSO}} \cdot \boldsymbol{\sigma} \cdot \mathbf{z}_p, \quad \mathbf{z}_p \sim \mathcal{N}(0, \mathbf{I})$ for $p = 1, \ldots, 50$ | $\boldsymbol{\theta}^{(0)}, \boldsymbol{\sigma}$ | $\{\boldsymbol{\theta}_p^{(0)}\}_{p=1}^{50}$ |
| 3 | Initialize Particle Velocities | $\mathbf{v}_p^{(0)} \sim \mathcal{U}(-|\boldsymbol{\sigma}|, |\boldsymbol{\sigma}|)$ for $p = 1, \ldots, 50$ | $\boldsymbol{\sigma}$ | $\{\mathbf{v}_p^{(0)}\}_{p=1}^{50}$ |
| 4 | Evaluate Initial Fitness | $f_p^{(0)} = \log \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}_p^{(0)}) + \log \pi(\boldsymbol{\theta}_p^{(0)})$ for all $p$ | $\{\boldsymbol{\theta}_p^{(0)}\}, \mathbf{d}$ | $\{f_p^{(0)}\}_{p=1}^{50}$ |
| 5 | Initialize Personal/Global Best | $\mathbf{p}_p^{(0)} = \boldsymbol{\theta}_p^{(0)}, \quad \mathbf{g}^{(0)} = \arg\max_p f_p^{(0)}$ | $\{\boldsymbol{\theta}_p^{(0)}\}, \{f_p^{(0)}\}$ | $\{\mathbf{p}_p^{(0)}\}, \mathbf{g}^{(0)}$ |

### 4.2 ITERATION LOOP

#### Phase 1: Particle Swarm Optimization

**BEGIN_LOOP** (PSO: $k = 1, \ldots, N_{\text{iter}}^{\text{PSO}} = 10$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| 1 | Update Velocity | $\mathbf{v}_p^{(k)} = w \mathbf{v}_p^{(k-1)} + c_1 r_1 (\mathbf{p}_p - \boldsymbol{\theta}_p^{(k-1)}) + c_2 r_2 (\mathbf{g}^{(k-1)} - \boldsymbol{\theta}_p^{(k-1)})$ | $\mathbf{v}_p^{(k-1)}, \mathbf{p}_p, \mathbf{g}^{(k-1)}, \boldsymbol{\theta}_p^{(k-1)}$ | $\mathbf{v}_p^{(k)}$ | $w, c_1, c_2, r_1, r_2$ from lenstronomy defaults |
| 2 | Update Position | $\tilde{\boldsymbol{\theta}}_p^{(k)} = \boldsymbol{\theta}_p^{(k-1)} + \mathbf{v}_p^{(k)}$ | $\boldsymbol{\theta}_p^{(k-1)}, \mathbf{v}_p^{(k)}$ | $\tilde{\boldsymbol{\theta}}_p^{(k)}$ | |
| 3 | Apply Bounds | $\boldsymbol{\theta}_p^{(k)} = \text{clip}(\tilde{\boldsymbol{\theta}}_p^{(k)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$ | $\tilde{\boldsymbol{\theta}}_p^{(k)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}$ | $\boldsymbol{\theta}_p^{(k)}$ | |
| 4 | Forward Model | $\mathcal{F}(\boldsymbol{\theta}_p^{(k)}) = \text{PSF} * \left( I_{\text{lens}}(\boldsymbol{\theta}_p^{(k)}) + I_{\text{source}}(\boldsymbol{\theta}_p^{(k)}) + I_{\text{ps}}(\boldsymbol{\theta}_p^{(k)}) \right)$ | $\boldsymbol{\theta}_p^{(k)}$ | $\hat{\mathbf{d}}_p^{(k)} \in \mathbb{R}^{100 \times 100}$ | Linear amplitudes solved analytically |
| 5 | Evaluate Fitness | $f_p^{(k)} = -\frac{1}{2}\sum_{i,j} \frac{(d_{i,j} - \hat{d}_{p,i,j}^{(k)})^2}{\sigma_{i,j}^2} - \frac{(e_1^{(k)})^2 + (e_2^{(k)})^2}{2 \cdot 0.2^2}$ | $\mathbf{d}, \hat{\mathbf{d}}_p^{(k)}, \boldsymbol{\theta}_p^{(k)}$ | $f_p^{(k)}$ | |
| 6 | Update Personal Best | $\mathbf{p}_p = \begin{cases} \boldsymbol{\theta}_p^{(k)} & \text{if } f_p^{(k)} > f_p^{\text{best}} \\ \mathbf{p}_p & \text{otherwise} \end{cases}$ | $\boldsymbol{\theta}_p^{(k)}, f_p^{(k)}, \mathbf{p}_p, f_p^{\text{best}}$ | $\mathbf{p}_p, f_p^{\text{best}}$ | |
| 7 | Update Global Best | $\mathbf{g}^{(k)} = \arg\max_p f_p^{\text{best}}$ | $\{\mathbf{p}_p\}, \{f_p^{\text{best}}\}$ | $\mathbf{g}^{(k)}$ | |

**END_LOOP** (PSO)

#### Phase 2: MCMC Sampling

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| T1 | Initialize Walkers | $\boldsymbol{\theta}_w^{(0)} = \mathbf{g}^{\text{PSO}} + \sigma_{\text{scale}}^{\text{MCMC}} \cdot \boldsymbol{\sigma} \cdot \mathbf{z}_w$ for $w = 1, \ldots, 50$ | $\mathbf{g}^{\text{PSO}}$, $\sigma_{\text{scale}}^{\text{MCMC}}=0.1$ | $\{\boldsymbol{\theta}_w^{(0)}\}_{w=1}^{50}$ | Ensemble sampler |

**BEGIN_LOOP** (MCMC Burn-in: $t = 1, \ldots, N_{\text{burn}} = 10$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| B1 | Propose Move (Stretch) | $\boldsymbol{\theta}_w^* = \boldsymbol{\theta}_j^{(t-1)} + z(\boldsymbol{\theta}_w^{(t-1)} - \boldsymbol{\theta}_j^{(t-1)})$, $z \sim g(z) \propto 1/\sqrt{z}$ for $z \in [1/a, a]$ | $\boldsymbol{\theta}_w^{(t-1)}, \boldsymbol{\theta}_j^{(t-1)}$ (random complementary walker) | $\boldsymbol{\theta}_w^*$ | $a=2$ (stretch parameter) |
| B2 | Compute Log-Posterior | $\log P(\boldsymbol{\theta}_w^*) = \log \mathcal{L}(\mathbf{d}|\boldsymbol{\theta}_w^*) + \log \pi(\boldsymbol{\theta}_w^*)$ | $\boldsymbol{\theta}_w^*, \mathbf{d}$ | $\log P_w^*$ | |
| B3 | Acceptance Probability | $\alpha = \min\left(1, z^{D-1} \exp(\log P_w^* - \log P_w^{(t-1)})\right)$ | $z, D, \log P_w^*, \log P_w^{(t-1)}$ | $\alpha$ | |
| B4 | Accept/Reject | $\boldsymbol{\theta}_w^{(t)} = \begin{cases} \boldsymbol{\theta}_w^* & \text{if } u < \alpha, u \sim \mathcal{U}(0,1) \\ \boldsymbol{\theta}_w^{(t-1)} & \text{otherwise} \end{cases}$ | $\boldsymbol{\theta}_w^*, \boldsymbol{\theta}_w^{(t-1)}, \alpha$ | $\boldsymbol{\theta}_w^{(t)}$ | |

**END_LOOP** (Burn-in)

**BEGIN_LOOP** (MCMC Production: $t = 1, \ldots, N_{\text{run}} = 10$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| P1 | Propose Move | Same as B1 | | $\boldsymbol{\theta}_w^*$ | |
| P2 | Compute Log-Posterior | Same as B2 | | $\log P_w^*$ | |
| P3 | Acceptance Probability | Same as B3 | | $\alpha$ | |
| P4 | Accept/Reject | Same as B4 | | $\boldsymbol{\theta}_w^{(t)}$ | |
| P5 | Store Sample | $\text{chain}[t, w] = \boldsymbol{\theta}_