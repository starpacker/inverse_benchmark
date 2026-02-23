# Algorithmic Specification Document: Gravitational Lens Modeling and Parameter Inference

## 1. Mathematical Formulation

### 1.1 Complete Objective Function

The algorithm solves a **Bayesian inverse problem** for gravitational lens parameter inference. The objective is to find parameters $\boldsymbol{\Theta}$ that maximize the posterior probability:

$$\boldsymbol{\Theta}^* = \arg\max_{\boldsymbol{\Theta}} \, p(\boldsymbol{\Theta} | \mathbf{D}) = \arg\max_{\boldsymbol{\Theta}} \, p(\mathbf{D} | \boldsymbol{\Theta}) \cdot p(\boldsymbol{\Theta})$$

where $\mathbf{D}$ is the observed image data and $\boldsymbol{\Theta}$ comprises all model parameters.

### 1.2 Data Fidelity Term (Log-Likelihood)

Assuming Gaussian noise statistics, the negative log-likelihood is:

$$-\ln \mathcal{L}(\boldsymbol{\Theta}) = \frac{1}{2} \chi^2(\boldsymbol{\Theta}) = \frac{1}{2} \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} \frac{\left( D_{ij} - M_{ij}(\boldsymbol{\Theta}) \right)^2}{\sigma_{ij}^2}$$

where:
- $D_{ij}$: observed pixel value at position $(i,j)$
- $M_{ij}(\boldsymbol{\Theta})$: model-predicted pixel value
- $\sigma_{ij}^2$: noise variance at pixel $(i,j)$

### 1.3 Forward Model $M(\boldsymbol{\Theta})$

The forward model computes the predicted image through gravitational lensing physics:

$$M(\boldsymbol{\theta}) = \left[ I_{\text{source}}\left(\boldsymbol{\beta}(\boldsymbol{\theta})\right) + I_{\text{lens}}(\boldsymbol{\theta}) \right] \ast P$$

#### 1.3.1 Gravitational Lens Equation

$$\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$$

where:
- $\boldsymbol{\theta} = (\theta_1, \theta_2)$: image plane angular coordinates
- $\boldsymbol{\beta} = (\beta_1, \beta_2)$: source plane angular coordinates  
- $\boldsymbol{\alpha}(\boldsymbol{\theta})$: deflection angle field

#### 1.3.2 SIE (Singular Isothermal Ellipsoid) Deflection

The deflection angles for SIE with ellipticity $(e_1, e_2)$ and Einstein radius $\theta_E$:

$$\alpha_1 = \theta_E \frac{q}{\sqrt{1-q^2}} \arctan\left(\frac{\sqrt{1-q^2} \, \tilde{\theta}_1}{\psi + q^2 s}\right)$$

$$\alpha_2 = \theta_E \frac{q}{\sqrt{1-q^2}} \text{arctanh}\left(\frac{\sqrt{1-q^2} \, \tilde{\theta}_2}{\psi + s}\right)$$

where:
- $q = \frac{1-|e|}{1+|e|}$ (axis ratio), with $|e| = \sqrt{e_1^2 + e_2^2}$
- $\psi = \sqrt{q^2(\tilde{\theta}_1^2 + s^2) + \tilde{\theta}_2^2}$
- $(\tilde{\theta}_1, \tilde{\theta}_2)$ are rotated coordinates aligned with ellipse axes
- $s$ is a core radius (effectively 0 for SIE)

#### 1.3.3 External Shear Contribution

$$\boldsymbol{\alpha}_{\text{shear}} = \begin{pmatrix} \gamma_1 & \gamma_2 \\ \gamma_2 & -\gamma_1 \end{pmatrix} \begin{pmatrix} \theta_1 - \theta_{1,0} \\ \theta_2 - \theta_{2,0} \end{pmatrix}$$

**Total deflection:**
$$\boldsymbol{\alpha}_{\text{total}}(\boldsymbol{\theta}) = \boldsymbol{\alpha}_{\text{SIE}}(\boldsymbol{\theta}) + \boldsymbol{\alpha}_{\text{shear}}(\boldsymbol{\theta})$$

#### 1.3.4 Sérsic Light Profile

For source and lens light distributions:

$$I_{\text{Sérsic}}(\boldsymbol{r}) = A \exp\left[ -b_n \left( \left(\frac{r_{\text{eff}}}{R_{\text{sersic}}}\right)^{1/n} - 1 \right) \right]$$

where:
- $r_{\text{eff}} = \sqrt{q_L \tilde{r}_1^2 + \tilde{r}_2^2 / q_L}$ (elliptical radius)
- $A$: amplitude
- $R_{\text{sersic}}$: effective (half-light) radius
- $n$: Sérsic index
- $b_n \approx 2n - 1/3 + 4/(405n)$ (approximation ensuring half-light definition)

#### 1.3.5 PSF Convolution

$$M_{ij} = \sum_{k,l} \left[ I_{\text{source}}(\boldsymbol{\beta}_{kl}) + I_{\text{lens}}(\boldsymbol{\theta}_{kl}) \right] \cdot P_{i-k, j-l}$$

For Gaussian PSF:
$$P(r) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left( -\frac{r^2}{2\sigma_{\text{PSF}}^2} \right), \quad \sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$$

### 1.4 Noise Model (Data Generation)

The simulated observation follows:

$$D_{ij} = \frac{\text{Poisson}\left( M_{ij}^{\text{true}} \cdot t_{\text{exp}} \right)}{t_{\text{exp}}} + \mathcal{N}(0, \sigma_{\text{bg}}^2)$$

where:
- $t_{\text{exp}}$: exposure time
- $\sigma_{\text{bg}}$: background RMS noise

### 1.5 Constraint Terms

**Box constraints** on all parameters:
$$\boldsymbol{\Theta}^{\text{lower}} \leq \boldsymbol{\Theta} \leq \boldsymbol{\Theta}^{\text{upper}}$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | dtype | Init Value | Lower Bound | Upper Bound |
|----------|--------|-------|-------|------------|-------------|-------------|
| **SIE Einstein radius** | $\theta_E$ | scalar | float64 | 0.7 | 0.01 | 10.0 |
| **SIE ellipticity 1** | $e_1^{\text{lens}}$ | scalar | float64 | 0.0 | -0.5 | 0.5 |
| **SIE ellipticity 2** | $e_2^{\text{lens}}$ | scalar | float64 | 0.0 | -0.5 | 0.5 |
| **SIE center x** | $x_c^{\text{lens}}$ | scalar | float64 | 0.0 | -10.0 | 10.0 |
| **SIE center y** | $y_c^{\text{lens}}$ | scalar | float64 | 0.0 | -10.0 | 10.0 |
| **Shear γ₁** | $\gamma_1$ | scalar | float64 | 0.0 | -0.2 | 0.2 |
| **Shear γ₂** | $\gamma_2$ | scalar | float64 | 0.0 | -0.2 | 0.2 |
| **Source amplitude** | $A_s$ | scalar | float64 | 16.0 | 0.0 | 100.0 |
| **Source Sérsic radius** | $R_s^{\text{src}}$ | scalar | float64 | 0.2 | 0.001 | 10.0 |
| **Source Sérsic index** | $n_s^{\text{src}}$ | scalar | float64 | 1.0 | 0.5 | 5.0 |
| **Source ellipticity 1** | $e_1^{\text{src}}$ | scalar | float64 | 0.0 | -0.5 | 0.5 |
| **Source ellipticity 2** | $e_2^{\text{src}}$ | scalar | float64 | 0.0 | -0.5 | 0.5 |
| **Source center x** | $x_c^{\text{src}}$ | scalar | float64 | 0.0 | -10.0 | 10.0 |
| **Source center y** | $y_c^{\text{src}}$ | scalar | float64 | 0.0 | -10.0 | 10.0 |
| **Lens light amplitude** | $A_{\ell}$ | scalar | float64 | 16.0 | 0.0 | 100.0 |
| **Lens light Sérsic radius** | $R_s^{\ell}$ | scalar | float64 | 0.5 | 0.001 | 10.0 |
| **Lens light Sérsic index** | $n_s^{\ell}$ | scalar | float64 | 2.0 | 0.5 | 5.0 |
| **Lens light ellipticity 1** | $e_1^{\ell}$ | scalar | float64 | 0.0 | -0.5 | 0.5 |
| **Lens light ellipticity 2** | $e_2^{\ell}$ | scalar | float64 | 0.0 | -0.5 | 0.5 |
| **Lens light center x** | $x_c^{\ell}$ | scalar | float64 | 0.0 | -10.0 | 10.0 |
| **Lens light center y** | $y_c^{\ell}$ | scalar | float64 | 0.0 | -10.0 | 10.0 |

**Total optimization dimensions:** 21 parameters

### 2.2 Auxiliary/Internal Variables

| Variable | Symbol | Shape | Description |
|----------|--------|-------|-------------|
| Particle positions (PSO) | $\mathbf{X}^{(k)}$ | $(N_p, d)$ | Positions of $N_p=50$ particles in $d$-dim space |
| Particle velocities (PSO) | $\mathbf{V}^{(k)}$ | $(N_p, d)$ | Velocity vectors |
| Personal best positions | $\mathbf{P}_{\text{best}}$ | $(N_p, d)$ | Best position per particle |
| Global best position | $\mathbf{G}_{\text{best}}$ | $(d,)$ | Best position found globally |
| Walker positions (MCMC) | $\mathbf{W}^{(k)}$ | $(N_w, d)$ | Positions of $N_w=50$ walkers |
| Deflection field | $\boldsymbol{\alpha}$ | $(N_x, N_y, 2)$ | Deflection angles per pixel |
| Source plane coordinates | $\boldsymbol{\beta}$ | $(N_x, N_y, 2)$ | Ray-traced source positions |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Image size | $N_x = N_y$ | 60 | Number of pixels per dimension |
| Pixel scale | $\Delta\theta$ | 0.05 arcsec/pix | Angular size per pixel |
| Background RMS | $\sigma_{\text{bg}}$ | 0.005 | Gaussian noise standard deviation |
| Exposure time | $t_{\text{exp}}$ | 500.0 s | Integration time |
| PSF FWHM | FWHM | 0.05 arcsec | Point spread function width |
| PSF truncation | - | 3 | Truncation radius in FWHM units |
| Supersampling factor | - | 1 | No supersampling |
| PSO particles | $N_p$ | 50 | Number of swarm particles |
| PSO iterations | $K_{\text{PSO}}$ | 10 | Maximum PSO iterations |
| PSO sigma scale | $\sigma_{\text{scale}}$ | 1.0 | Initial distribution scale |
| MCMC walkers | $N_w$ | 50 | Number of ensemble walkers |
| MCMC burn-in | $K_{\text{burn}}$ | 10 | Burn-in iterations |
| MCMC production | $K_{\text{run}}$ | 10 | Production iterations |
| MCMC sigma scale | - | 0.1 | Proposal distribution scale |
| Random seed | - | 42 | Reproducibility seed |

### 2.4 Observed Data/Inputs (Ground Truth for Simulation)

| Variable | Symbol | Value/Shape | Description |
|----------|--------|-------------|-------------|
| Observed image | $\mathbf{D}$ | $(60, 60)$, float64 | Simulated noisy observation |
| True $\theta_E$ | $\theta_E^{\text{true}}$ | 0.66 | Ground truth Einstein radius |
| True SIE center | $(x_c, y_c)^{\text{true}}$ | (0.05, 0.0) | Ground truth lens center |
| True SIE ellipticity | $(e_1, e_2)^{\text{true}}$ | (0.07, -0.03) | Ground truth ellipticity |
| True shear | $(\gamma_1, \gamma_2)^{\text{true}}$ | (0.0, -0.05) | Ground truth external shear |
| True source params | - | See config | Source Sérsic parameters |
| True lens light params | - | See config | Lens light Sérsic parameters |

---

## 3. Preprocessing Pipeline

| Step | Operation | Input | Output | Formula/Description |
|------|-----------|-------|--------|---------------------|
| **1** | Set random seed | seed=42 | RNG state | `np.random.seed(42)` |
| **2** | Define coordinate transform | $\Delta\theta$, $N_x$ | $\mathbf{T}$ | $\mathbf{T} = \begin{pmatrix} -\Delta\theta & 0 \\ 0 & \Delta\theta \end{pmatrix}$ |
| **3** | Compute reference point | $\mathbf{T}$, $N_x$ | $(\text{RA}_0, \text{Dec}_0)$ | $c = \frac{N_x - 1}{2}$; $\text{RA}_0 = -T_{00}c - T_{01}c$; $\text{Dec}_0 = -T_{10}c - T_{11}c$ |
| **4** | Initialize image array | $N_x$ | $\mathbf{D}_0$ | $\mathbf{D}_0 = \mathbf{0}_{N_x \times N_x}$ |
| **5** | Construct PSF kernel | FWHM, $\Delta\theta$ | $\mathbf{P}$ | Gaussian kernel with $\sigma = \text{FWHM}/(2\sqrt{2\ln 2})$, truncated at 3×FWHM |
| **6** | Build lens model | `['SIE', 'SHEAR']` | `LensModel` | Composite deflector class |
| **7** | Build source model | `['SERSIC_ELLIPSE']` | `LightModel` | Source brightness class |
| **8** | Build lens light model | `['SERSIC_ELLIPSE']` | `LightModel` | Lens galaxy brightness class |
| **9** | Generate noise-free image | $\boldsymbol{\Theta}^{\text{true}}$, models | $\mathbf{M}^{\text{true}}$ | $M_{ij} = \left[\sum_s I_s(\boldsymbol{\beta}_{ij}) + \sum_\ell I_\ell(\boldsymbol{\theta}_{ij})\right] \ast P$ |
| **10** | Convert to counts | $\mathbf{M}^{\text{true}}$, $t_{\text{exp}}$ | $\mathbf{C}$ | $C_{ij} = \max(M_{ij}^{\text{true}} \cdot t_{\text{exp}}, 0)$ |
| **11** | Add Poisson noise | $\mathbf{C}$ | $\mathbf{C}'$ | $C'_{ij} \sim \text{Poisson}(C_{ij})$ |
| **12** | Convert back to flux | $\mathbf{C}'$, $t_{\text{exp}}$ | $\mathbf{M}'$ | $M'_{ij} = C'_{ij} / t_{\text{exp}}$ |
| **13** | Generate background noise | $\sigma_{\text{bg}}$, shape | $\mathbf{N}$ | $N_{ij} \sim \mathcal{N}(0, \sigma_{\text{bg}}^2)$ |
| **14** | Create final observation | $\mathbf{M}'$, $\mathbf{N}$ | $\mathbf{D}$ | $D_{ij} = M'_{ij} + N_{ij}$ |

**Output of preprocessing:** Dictionary containing $\mathbf{D}$, all model objects, true parameters, and configuration.

---

## 4. Algorithm Steps

### 4.1 Initialization Phase

| Step | Name | Formula/Procedure | Input | Output |
|------|------|-------------------|-------|--------|
| **I.1** | Load preprocessed data | Extract from dict | `preprocessed_data` | $\mathbf{D}$, models, configs |
| **I.2** | Define parameter bounds | Tables §2.1 | - | $\boldsymbol{\Theta}^{\text{lower}}$, $\boldsymbol{\Theta}^{\text{upper}}$ |
| **I.3** | Set initial parameters | Tables §2.1 | - | $\boldsymbol{\Theta}^{(0)}$ |
| **I.4** | Set parameter uncertainties | $\boldsymbol{\sigma}$ from tables | - | Proposal widths for each parameter |
| **I.5** | Construct `FittingSequence` | All configs | - | Fitting object with likelihood evaluator |

### 4.2 Phase 1: Particle Swarm Optimization (PSO)

**Objective:** Global optimization to find region near optimum.

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **P.1** | Initialize particles | $\mathbf{x}_i^{(0)} \sim \mathcal{N}(\boldsymbol{\Theta}^{(0)}, \sigma_{\text{scale}}^2 \boldsymbol{\sigma}^2)$, clipped to bounds | $\boldsymbol{\Theta}^{(0)}$, bounds | $\{\mathbf{x}_i^{(0)}\}_{i=1}^{N_p}$ |
| **P.2** | Initialize velocities | $\mathbf{v}_i^{(0)} = \mathbf{0}$ | - | $\{\mathbf{v}_i^{(0)}\}_{i=1}^{N_p}$ |
| **P.3** | Evaluate fitness | $f_i = -\ln\mathcal{L}(\mathbf{x}_i) = \frac{1}{2}\chi^2(\mathbf{x}_i)$ | $\{\mathbf{x}_i\}$, $\mathbf{D}$ | $\{f_i\}_{i=1}^{N_p}$ |
| **P.4** | Initialize personal bests | $\mathbf{p}_i = \mathbf{x}_i^{(0)}$, $f_i^{\text{pbest}} = f_i$ | - | $\{\mathbf{p}_i, f_i^{\text{pbest}}\}$ |
| **P.5** | Initialize global best | $\mathbf{g} = \arg\min_i f_i$, $f^{\text{gbest}} = \min_i f_i$ | - | $\mathbf{g}$, $f^{\text{gbest}}$ |

**Per-iteration update (for $k = 1, \ldots, K_{\text{PSO}}$):**

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **P.6** | Update velocity | $\mathbf{v}_i^{(k)} = \omega \mathbf{v}_i^{(k-1)} + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^{(k-1)}) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^{(k-1)})$ | Previous state, $r_1, r_2 \sim U(0,1)$ | $\mathbf{v}_i^{(k)}$ |
| **P.7** | Update position | $\mathbf{x}_i^{(k)} = \text{clip}(\mathbf{x}_i^{(k-1)} + \mathbf{v}_i^{(k)}, \boldsymbol{\Theta}^{\text{lower}}, \boldsymbol{\Theta}^{\text{upper}})$ | $\mathbf{x}_i^{(k-1)}$, $\mathbf{v}_i^{(k)}$ | $\mathbf{x}_i^{(k)}$ |
| **P.8** | Evaluate fitness | $f_i^{(k)} = \frac{1}{2}\chi^2(\mathbf{x}_i^{(k)})$ | $\mathbf{x}_i^{(k)}$ | $f_i^{(k)}$ |
| **P.9** | Update personal best | If $f_i^{(k)} < f_i^{\text{pbest}}$: $\mathbf{p}_i \leftarrow \mathbf{x}_i^{(k)}$, $f_i^{\text{pbest}} \leftarrow f_i^{(k)}$ | - | Updated $\mathbf{p}_i$ |
| **P.10** | Update global best | If $\min_i f_i^{(k)} < f^{\text{gbest}}$: $\mathbf{g} \leftarrow \arg\min$, $f^{\text{gbest}} \leftarrow \min$ | - | Updated $\mathbf{g}$ |

**PSO Output:** $\boldsymbol{\Theta}^{\text{PSO}} = \mathbf{g}$ (best found position)

### 4.3 Phase 2: Markov Chain Monte Carlo (MCMC)

**Objective:** Sample posterior distribution for uncertainty quantification.

Uses **Affine-Invariant Ensemble Sampler** (emcee-style):

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **M.1** | Initialize walkers | $\mathbf{w}_j^{(0)} \sim \mathcal{N}(\boldsymbol{\Theta}^{\text{PSO}}, (0.1 \cdot \boldsymbol{\sigma})^2)$, $j=1,\ldots,N_w$ | $\boldsymbol{\Theta}^{\text{PSO}}$ | $\{\mathbf{w}_j^{(0)}\}_{j=1}^{N_w}$ |
| **M.2** | Evaluate log-posterior | $\ln p_j^{(0)} = \ln\mathcal{L}(\mathbf{w}_j^{(0)}) + \ln\pi(\mathbf{w}_j^{(0)})$ | Walkers, bounds | $\{\ln p_j^{(0)}\}$ |

**Per-iteration update (for $k = 1, \ldots, K_{\text{burn}} + K_{\text{run}}$):**

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **M.3** | Select complementary walker | $\mathbf{w}_{\ell}$ chosen uniformly from other walkers | Walker ensemble | $\mathbf{w}_{\ell}$ |
| **M.4** | Draw stretch factor | $z \sim g(z)$ where $g(z) \propto 1/\sqrt{z}$ for $z \in [1/a, a]$, $a=2$ | - | $z$ |
| **M.5** | Propose new position | $\mathbf{w}_j' = \mathbf{w}_{\ell} + z(\mathbf{w}_j^{(k-1)} - \mathbf{w}_{\ell})$ | $\mathbf{w}_j^{(k-1)}$, $\mathbf{w}_{\ell}$, $z$ | $\mathbf{w}_j'$ |
| **M.6** | Compute acceptance probability | $\alpha = \min\left(1, z^{d-1} \frac{p(\mathbf{w}_j')}{p(\mathbf{w}_j^{(k-1)})}\right)$ | Log-posteriors | $\alpha$ |
| **M.7** | Accept/reject | $u \sim U(0,1)$; if $u < \alpha$: $\mathbf{w}_j^{(k)} = \mathbf{w}_j'$, else $\mathbf{w}_j^{(k)} = \mathbf{w}_j^{(k-1)}$ | $\alpha$, $u$ | $\mathbf{w}_j^{(k)}$ |
| **M.8** | Store sample | If $k > K_{\text{burn}}$: append $\mathbf{w}_j^{(k)}$ to chain | - | Chain samples |

**MCMC Output:** 
- Chain: $\{\mathbf{w}_j^{(k)}\}$ for $k > K_{\text{