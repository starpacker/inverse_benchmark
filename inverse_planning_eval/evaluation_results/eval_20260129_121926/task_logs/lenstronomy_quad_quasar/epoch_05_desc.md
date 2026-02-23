## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Gravitational Lens Parameter Estimation via PSO-MCMC
- **algorithm_family**: Hybrid Metaheuristic-Bayesian Optimization (Particle Swarm Optimization + Markov Chain Monte Carlo)
- **summary**: Estimates gravitational lens model parameters by first performing global optimization using Particle Swarm Optimization (PSO) to find an approximate maximum likelihood solution, followed by Markov Chain Monte Carlo (MCMC) sampling to characterize the posterior distribution.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} \left[ \log \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) + \log \pi(\boldsymbol{\theta}) \right]$$

subject to:
$$\boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$$
$$\|\mathbf{x}_{\text{source}}^{\text{(model)}} - \mathbf{x}_{\text{source}}^{\text{(ps)}}\| < \tau_{\text{source}}$$

### data_fidelity_term
$$\log \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) = -\frac{1}{2} \sum_{i,j} \frac{\left( d_{ij} - f_{ij}(\boldsymbol{\theta}) \right)^2}{\sigma_{ij}^2}$$

where:
- $d_{ij}$ is the observed image pixel value at position $(i,j)$
- $f_{ij}(\boldsymbol{\theta})$ is the forward model prediction
- $\sigma_{ij}^2 = \sigma_{\text{bkg}}^2 + \frac{\max(f_{ij}, 0)}{t_{\text{exp}}}$ is the pixel variance (background + Poisson)

### regularization_term
**Gaussian Prior on Lens Ellipticity:**
$$\log \pi_{\text{lens}}(\boldsymbol{\theta}) = -\frac{(e_1^{\text{(lens)}} - \mu_{e_1})^2}{2\sigma_{e_1}^2} - \frac{(e_2^{\text{(lens)}} - \mu_{e_2})^2}{2\sigma_{e_2}^2}$$

where $\mu_{e_1} = 0$, $\sigma_{e_1} = 0.2$, $\mu_{e_2} = 0$, $\sigma_{e_2} = 0.2$

### constraints
**Box Constraints (Indicator Functions):**
$$\mathcal{I}_{\text{box}}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}} \\ -\infty & \text{otherwise} \end{cases}$$

**Source Position Constraint:**
$$\mathcal{I}_{\text{source}}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \|\mathbf{x}_{\text{source}}^{\text{(model)}} - \mathbf{x}_{\text{source}}^{\text{(ps)}}\| < 0.001 \\ -\infty & \text{otherwise} \end{cases}$$

**Image Position Likelihood Contribution:**
$$\log \mathcal{L}_{\text{pos}} = -\frac{1}{2} \sum_{k=1}^{4} \frac{\|\mathbf{x}_k^{\text{(obs)}} - \mathbf{x}_k^{\text{(model)}}\|^2}{\sigma_{\text{pos}}^2}$$

where $\sigma_{\text{pos}} = 0.004$ arcsec

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 1.2 | Einstein radius of EPL lens (arcsec) |
| $\gamma_{\text{EPL}}$ | scalar | float64 | 2.0 | Power-law slope of EPL mass profile |
| $e_1^{\text{(lens)}}$ | scalar | float64 | 0.0 | First ellipticity component of lens |
| $e_2^{\text{(lens)}}$ | scalar | float64 | 0.0 | Second ellipticity component of lens |
| $x_c^{\text{(lens)}}$ | scalar | float64 | 0.0 | Lens center x-coordinate (arcsec) |
| $y_c^{\text{(lens)}}$ | scalar | float64 | 0.0 | Lens center y-coordinate (arcsec) |
| $\gamma_1$ | scalar | float64 | 0.0 | First external shear component |
| $\gamma_2$ | scalar | float64 | 0.0 | Second external shear component |
| $R_{\text{sersic}}^{\text{(src)}}$ | scalar | float64 | 0.03 | Source Sérsic effective radius (arcsec) |
| $n_{\text{sersic}}^{\text{(src)}}$ | scalar | float64 | 1.0 | Source Sérsic index |
| $e_1^{\text{(src)}}$ | scalar | float64 | 0.0 | Source first ellipticity component |
| $e_2^{\text{(src)}}$ | scalar | float64 | 0.0 | Source second ellipticity component |
| $x_c^{\text{(src)}}$ | scalar | float64 | 0.0 | Source center x-coordinate (arcsec) |
| $y_c^{\text{(src)}}$ | scalar | float64 | 0.0 | Source center y-coordinate (arcsec) |
| $R_{\text{sersic}}^{\text{(ll)}}$ | scalar | float64 | 0.1 | Lens light Sérsic effective radius |
| $n_{\text{sersic}}^{\text{(ll)}}$ | scalar | float64 | 1.0 | Lens light Sérsic index |
| $e_1^{\text{(ll)}}$ | scalar | float64 | 0.0 | Lens light first ellipticity |
| $e_2^{\text{(ll)}}$ | scalar | float64 | 0.0 | Lens light second ellipticity |
| $x_c^{\text{(ll)}}$ | scalar | float64 | 0.0 | Lens light center x-coordinate |
| $y_c^{\text{(ll)}}$ | scalar | float64 | 0.0 | Lens light center y-coordinate |
| $\mathbf{x}_{\text{img}}$ | (4,) | float64 | $\mathbf{x}_{\text{true}} + 0.01$ | Point source image RA positions |
| $\mathbf{y}_{\text{img}}$ | (4,) | float64 | $\mathbf{y}_{\text{true}} - 0.01$ | Point source image Dec positions |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| N/A | — | — | — | No explicit dual variables (embedded in MCMC sampler) |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{P}^{(t)}$ | (50, $N_{\text{params}}$) | float64 | Sampled from $\mathcal{N}(\boldsymbol{\theta}_0, \boldsymbol{\Sigma})$ | PSO particle positions |
| $\mathbf{V}^{(t)}$ | (50, $N_{\text{params}}$) | float64 | Zeros | PSO particle velocities |
| $\mathbf{P}_{\text{best}}^{\text{(local)}}$ | (50, $N_{\text{params}}$) | float64 | $\mathbf{P}^{(0)}$ | Personal best positions |
| $\mathbf{p}_{\text{best}}^{\text{(global)}}$ | ($N_{\text{params}}$,) | float64 | Best of $\mathbf{P}^{(0)}$ | Global best position |
| $\mathbf{W}^{(t)}$ | (50, $N_{\text{params}}$) | float64 | Initialized around best PSO result | MCMC walker positions |
| $\chi^{(t)}$ | (50,) | float64 | Computed | Log-posterior values for walkers |
| $A^{\text{(src)}}$ | scalar | float64 | Solved via linear optimization | Source amplitude (linear parameter) |
| $A^{\text{(ll)}}$ | scalar | float64 | Solved via linear optimization | Lens light amplitude (linear parameter) |
| $\mathbf{A}_{\text{ps}}$ | (4,) | float64 | Solved via linear optimization | Point source amplitudes |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\sigma_{\text{bkg}}$ | 0.5 | Background RMS noise (counts/s/pixel) |
| $t_{\text{exp}}$ | 100.0 | Exposure time (seconds) |
| $N_{\text{pix}}$ | 100 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | 0.05 | Pixel scale (arcsec/pixel) |
| $\text{FWHM}$ | 0.1 | PSF full width at half maximum (arcsec) |
| $N_{\text{particles}}$ | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | 10 | PSO iterations |
| $N_{\text{walkers}}$ | 50 | Number of MCMC walkers |
| $N_{\text{burn}}$ | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | 10 | MCMC production steps |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | 1.0 | PSO initialization scale |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | 0.1 | MCMC initialization scale |
| $\sigma_{\text{pos}}$ | 0.004 | Image position uncertainty (arcsec) |
| $\tau_{\text{source}}$ | 0.001 | Source position tolerance (arcsec) |
| $\sigma_{\text{source}}$ | 0.001 | Source position sigma (arcsec) |
| $\mu_{e_1}, \mu_{e_2}$ | 0.0 | Prior means for lens ellipticity |
| $\sigma_{e_1}, \sigma_{e_2}$ | 0.2 | Prior std for lens ellipticity |
| $\text{truncation}$ | 5 | PSF truncation radius in FWHM units |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{d}$ | (100, 100) | float64 | Observed (simulated) lensed image with noise |
| $\mathbf{T}$ | (2, 2) | float64 | Pixel-to-angle transformation matrix |
| $(\alpha_0, \delta_0)$ | (2,) | float64 | RA/Dec at pixel (0,0) |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Compute transformation matrix | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}$ (scalar) | $\mathbf{T}$ (2×2) |
| 2 | Compute reference coordinates | $c_x = c_y = \frac{N_{\text{pix}} - 1}{2}$, $\alpha_0 = -(T_{00} c_x + T_{01} c_y)$, $\delta_0 = -(T_{10} c_x + T_{11} c_y)$ | $N_{\text{pix}}$ (scalar), $\mathbf{T}$ (2×2) | $\alpha_0, \delta_0$ (scalars) |
| 3 | Convert shear to Cartesian | $\gamma_1^{\text{true}} = \gamma \cos(2\phi)$, $\gamma_2^{\text{true}} = \gamma \sin(2\phi)$ where $\phi=0.1$, $\gamma=0.02$ | $\phi, \gamma$ (scalars) | $\gamma_1^{\text{true}}, \gamma_2^{\text{true}}$ (scalars) |
| 4 | Solve lens equation | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}} = \text{LensEquationSolver}(\alpha_s, \delta_s, \boldsymbol{\theta}_{\text{lens}})$ where $(\alpha_s, \delta_s) = (0, 0.1)$ | Source position, lens params | $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}$ (4,) each |
| 5 | Compute magnifications | $\mu_k = \left| \det \left( \mathbf{I} - \frac{\partial \boldsymbol{\alpha}}{\partial \boldsymbol{\theta}} \right)^{-1} \right|$ for each image $k$ | Image positions, lens params | $\boldsymbol{\mu}$ (4,) |
| 6 | Perturb magnifications | $\tilde{\mu}_k \sim \mathcal{N}(\mu_k, 0.5)$ | $\boldsymbol{\mu}$ (4,) | $\tilde{\boldsymbol{\mu}}$ (4,) |
| 7 | Compute point source amplitudes | $A_k^{\text{ps}} = 100 \cdot \tilde{\mu}_k$ | $\tilde{\boldsymbol{\mu}}$ (4,) | $\mathbf{A}^{\text{ps}}$ (4,) |
| 8 | Generate noise-free image | $\mathbf{f} = \mathcal{M}(\boldsymbol{\theta}_{\text{true}})$ | All true model params | $\mathbf{f}$ (100×100) |
| 9 | Generate Poisson noise | $\mathbf{n}_{\text{Poisson}} = \frac{\text{Poisson}(\max(\mathbf{f} \cdot t_{\text{exp}}, 0))}{t_{\text{exp}}} - \mathbf{f}$ | $\mathbf{f}$ (100×100), $t_{\text{exp}}$ | $\mathbf{n}_{\text{Poisson}}$ (100×100) |
| 10 | Generate Gaussian noise | $\mathbf{n}_{\text{bkg}} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2 \mathbf{I})$ | $\sigma_{\text{bkg}}$ (scalar) | $\mathbf{n}_{\text{bkg}}$ (100×100) |
| 11 | Combine to form observation | $\mathbf{d} = \mathbf{f} + \mathbf{n}_{\text{Poisson}} + \mathbf{n}_{\text{bkg}}$ | $\mathbf{f}, \mathbf{n}_{\text{Poisson}}, \mathbf{n}_{\text{bkg}}$ (all 100×100) | $\mathbf{d}$ (100×100) |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize parameter vector | $\boldsymbol{\theta}_0 = [\theta_E^{(0)}, \gamma^{(0)}, e_1^{(0)}, e_2^{(0)}, x_c^{(0)}, y_c^{(0)}, \gamma_1^{(0)}, \gamma_2^{(0)}, \ldots]^T$ with values from init dicts | Init parameter dicts | $\boldsymbol{\theta}_0$ ($N_{\text{params}}$,) | Concatenate all model parameters |
| 2 | Initialize sigma vector | $\boldsymbol{\sigma} = [\sigma_{\theta_E}, \sigma_{\gamma}, \sigma_{e_1}, \ldots]^T$ with values $[0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, \ldots]$ | Sigma parameter dicts | $\boldsymbol{\sigma}$ ($N_{\text{params}}$,) | Step sizes for initialization |
| 3 | Initialize lower bounds | $\boldsymbol{\theta}_{\text{lower}} = [0, 1.5, -0.5, -0.5, -10, -10, -0.5, -0.5, \ldots]^T$ | Lower bound dicts | $\boldsymbol{\theta}_{\text{lower}}$ ($N_{\text{params}}$,) | Box constraint lower limits |
| 4 | Initialize upper bounds | $\boldsymbol{\theta}_{\text{upper}} = [10, 2.5, 0.5, 0.5, 10, 10, 0.5, 0.5, \ldots]^T$ | Upper bound dicts | $\boldsymbol{\theta}_{\text{upper}}$ ($N_{\text{params}}$,) | Box constraint upper limits |
| 5 | Initialize PSO particles | $P_i^{(0)} = \boldsymbol{\theta}_0 + \sigma_{\text{scale}}^{\text{PSO}} \cdot \boldsymbol{\sigma} \odot \mathbf{z}_i$ where $\mathbf{z}_i \sim \mathcal{N}(0, \mathbf{I})$ for $i = 1, \ldots, 50$ | $\boldsymbol{\theta}_0, \boldsymbol{\sigma}, \sigma_{\text{scale}}^{\text{PSO}}$ | $\mathbf{P}^{(0)}$ (50, $N_{\text{params}}$) | Sample particles around initial guess |
| 6 | Clip particles to bounds | $P_{ij}^{(0)} \leftarrow \max(\theta_{\text{lower},j}, \min(\theta_{\text{upper},j}, P_{ij}^{(0)}))$ | $\mathbf{P}^{(0)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}$ | $\mathbf{P}^{(0)}$ (50, $N_{\text{params}}$) | Enforce box constraints |
| 7 | Initialize velocities | $V_{ij}^{(0)} = 0$ for all $i, j$ | — | $\mathbf{V}^{(0)}$ (50, $N_{\text{params}}$) | Zero initial velocities |
| 8 | Evaluate initial fitness | $\chi_i^{(0)} = \log \mathcal{L}(\mathbf{d} | \mathbf{P}_i^{(0)}) + \log \pi(\mathbf{P}_i^{(0)})$ for $i = 1, \ldots, 50$ | $\mathbf{P}^{(0)}, \mathbf{d}$ | $\boldsymbol{\chi}^{(0)}$ (50,) | Evaluate log-posterior for each particle |
| 9 | Initialize personal bests | $\mathbf{P}_{\text{best},i}^{\text{(local)}} = \mathbf{P}_i^{(0)}$, $\chi_{\text{best},i}^{\text{(local)}} = \chi_i^{(0)}$ | $\mathbf{P}^{(0)}, \boldsymbol{\chi}^{(0)}$ | $\mathbf{P}_{\text{best}}^{\text{(local)}}$ (50, $N_{\text{params}}$) | Store initial as best |
| 10 | Initialize global best | $i^* = \arg\max_i \chi_i^{(0)}$, $\mathbf{p}_{\text{best}}^{\text{(global)}} = \mathbf{P}_{i^*}^{(0)}$, $\chi_{\text{best}}^{\text{(global)}} = \chi_{i^*}^{(0)}$ | $\mathbf{P}^{(0)}, \boldsymbol{\chi}^{(0)}$ | $\mathbf{p}_{\text{best}}^{\text{(global)}}$ ($N_{\text{params}}$,), $\chi_{\text{best}}^{\text{(global)}}$ (scalar) | Find best particle |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (PSO: $t = 1, \ldots, N_{\text{iter}}^{\text{PSO}} = 10$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 11 | Generate random coefficients | $r_1, r_2 \sim \mathcal{U}(0, 1)^{N_{\text{params}}}$ | — | $\mathbf{r}_1, \mathbf{r}_2$ ($N_{\text{params}}$,) | Per-dimension random vectors |
| 12 | Update velocities | $V_{ij}^{(t)} = \omega V_{ij}^{(t-1)} + c_1 r_{1j} (P_{\text{best},ij}^{\text{(local)}} - P_{ij}^{(t-1)}) + c_2 r_{2j} (p_{\text{best},j}^{\text{(global)}} - P_{ij}^{(t-1)})$ | $\mathbf{V}^{(t-1)}, \mathbf{P}^{(t-1)}, \mathbf{P}_{\text{best}}^{\text{(local)}}, \mathbf{p}_{\text{best}}^{\text{(global)}}, \mathbf{r}_1, \mathbf{r}_2$ | $\mathbf{V}^{(t)}$ (50, $N_{\text{params}}$) | $\omega \approx 0.7$, $c_1, c_2 \approx 1.5$ (lenstronomy defaults) |
| 13 | Update positions | $P_{ij}^{(t)} = P_{ij}^{(t-1)} + V_{ij}^{(t)}$ | $\mathbf{P}^{(t-1)}, \mathbf{V}^{(t)}$ | $\mathbf{P}^{(t)}$ (50, $N_{\text{params}}$) | Position update |
| 14 | Clip to bounds | $P_{ij}^{(t)} \leftarrow \max(\theta_{\text{lower},j}, \min(\theta_{\text{upper},j}, P_{ij}^{(t)}))$ | $\mathbf{P}^{(t)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}$ | $\mathbf{P}^{(t)}$ (50, $N_{\text{params}}$) | Enforce box constraints |
| 15 | Evaluate fitness | $\chi_i^{(t)} = \log \mathcal{L}(\mathbf{d} | \mathbf{P}_i^{(t)}) + \log \pi(\mathbf{P}_i^{(t)})$ | $\mathbf{P}^{(t)}, \mathbf{d}$ | $\boldsymbol{\chi}^{(t)}$ (50,) | [FORWARD] model evaluation |

**BEGIN_SUBLOOP** (Update bests: $i = 1, \ldots, 50$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 16 | Update personal best | If $\chi_i^{(t)} > \chi_{\text{best},i}^{\text{(local)}}$: $\mathbf{P}_{\text{best},i}^{\text{(local)}} \leftarrow \mathbf{P}_i^{(t)}$, $\chi_{\text{best},i}^{\text{(local)}} \leftarrow \chi_i^{(t)}$ | $\mathbf{P}_i^{(t)}, \chi_i^{(t)}, \mathbf{P}_{\text{best},i}^{\text{(local)}}, \chi_{\text{best},i}^{\text{(local)}}$ | Updated $\mathbf{P}_{\text{best},i}^{\text{(local)}}, \chi_{\text{best},i}^{\text{(local)}}$ | Conditional update |
| 17 | Update global best | If $\chi_i^{(t)} > \chi_{\text{best}}^{\text{(global)}}$: $\mathbf{p}_{\text{best}}^{\text{(global)}} \leftarrow \mathbf{P}_i^{(t)}$, $\chi_{\text{best}}^{\text{(global)}} \leftarrow \chi_i^{(t)}$ | $\mathbf{P}_i^{(t)}, \chi_i^{(t)}, \mathbf{p}_{\text{best}}^{\text{(global)}}, \chi_{\text{best}}^{\text{(global)}}$ | Updated $\mathbf{p}_{\text{best}}^{\text{(global)}}, \chi_{\text{best}}^{\text{(global)}}$ | Conditional update |

**END_SUBLOOP**

**END_LOOP** (PSO)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 18 | Store PSO result | $\boldsymbol{\theta}_{\text{PSO}} = \mathbf{p}_{\text{best}}^{\text{(global)}}$ | $\mathbf{p}_{\text{best}}^{\text{(global)}}$ | $\boldsymbol{\theta}_{\text{PSO}}$ ($N_{\text{params}}$,) | Best solution from PSO |
| 19 | Initialize MCMC walkers | $W_i^{(0)} = \boldsymbol{\theta}_{\text{PSO}} + \sigma_{\text{scale}}^{\text{MCMC}} \cdot \boldsymbol{\sigma} \odot \mathbf{z}_i$ where $\mathbf{z}_i \sim \mathcal{N}(0, \mathbf{I})$ for $i = 1, \ldots, 50$ | $\boldsymbol{\theta}_{\text{PSO}}, \boldsymbol{\sigma}, \sigma_{\text{scale}}^{\text{MCMC}}$ | $\mathbf{W}^{(0)}$ (50, $N_{\text{params}}$) | Initialize around PSO solution |
| 20 | Clip walkers to bounds | $W_{ij}^{(0)} \leftarrow \max(\theta_{\text{lower},j}, \min(\theta_{\text{upper},j}, W_{ij}^{(0)}))$ | $\mathbf{W}^{(0)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}$ | $\mathbf{W}^{(0)}$ (50, $N_{\text{params}}$) | Enforce constraints |
| 21 | Evaluate initial walker fitness | $\chi_i^{(0)} = \log \mathcal{L}(\mathbf{d} | \mathbf{W}_i^{(0)}) + \log \pi(\mathbf{W}_i^{(0)})$ | $\mathbf{W}^{(0)}, \mathbf{d}$ | $\boldsymbol{\chi}^{(0)}$ (50,) | Log-posterior for each walker |

**BEGIN_LOOP** (MCMC Burn-in: $t = 1, \ldots, N_{\text{burn}} = 10$)

**BEGIN_SUBLOOP** (Walker updates: $i = 1, \ldots, 50$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 22 | Select complementary walker | $j \sim \mathcal{U}\{1, \ldots, 50\} \setminus \{i\}$ | Walker index $i$ | $j$ (scalar) | Random selection excluding $i$ |
| 23 | Generate stretch factor | $z \sim g(z) \propto \f