## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Quasar-Host Galaxy Decomposition via Particle Swarm Optimization
- **algorithm_family**: Metaheuristic Global Optimization (Particle Swarm Optimization)
- **summary**: Decomposes astronomical images of active galactic nuclei into point source (quasar) and extended host galaxy (disk + bulge Sérsic profiles) components by minimizing χ² residuals using PSO.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\boldsymbol{\theta}} \chi^2(\boldsymbol{\theta}) = \sum_{i,j} \frac{\left(y_{ij}^{\text{obs}} - y_{ij}^{\text{model}}(\boldsymbol{\theta})\right)^2}{\sigma_{ij}^2}$$

subject to:
$$\boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$$

### data_fidelity_term
$$\chi^2(\boldsymbol{\theta}) = \sum_{i=1}^{N_{\text{pix}}} \sum_{j=1}^{N_{\text{pix}}} \frac{\left(y_{ij}^{\text{obs}} - y_{ij}^{\text{model}}(\boldsymbol{\theta})\right)^2}{\sigma_{ij}^2}$$

where:
- $y_{ij}^{\text{obs}}$: observed image pixel value at position $(i,j)$
- $y_{ij}^{\text{model}}(\boldsymbol{\theta})$: forward model prediction
- $\sigma_{ij}^2 = \sigma_{\text{bkg}}^2 + \frac{y_{ij}^{\text{model}}}{t_{\text{exp}}}$: noise variance (background + Poisson)

### regularization_term
None

### constraints
**Box constraints on parameters:**

| Parameter | Lower Bound | Upper Bound |
|-----------|-------------|-------------|
| $e_1^{\text{disk}}$ | -0.5 | 0.5 |
| $e_2^{\text{disk}}$ | -0.5 | 0.5 |
| $R_{\text{sersic}}^{\text{disk}}$ | 0.001 | 10 |
| $R_{\text{sersic}}^{\text{bulge}}$ | 0.001 | 10 |
| $x_c, y_c$ | -10 | 10 |
| $\alpha_{\text{ps}}, \delta_{\text{ps}}$ | -10 | 10 |

**Equality constraints (joint centers):**
$$x_c^{\text{disk}} = x_c^{\text{bulge}} = \alpha_{\text{ps}}$$
$$y_c^{\text{disk}} = y_c^{\text{bulge}} = \delta_{\text{ps}}$$

**Fixed parameters:**
$$n_{\text{sersic}}^{\text{disk}} = 1, \quad n_{\text{sersic}}^{\text{bulge}} = 4$$

---

## 2. VARIABLES

### Primal Variables (Optimization Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $R_{\text{sersic}}^{\text{disk}}$ | primal | scalar | float64 | 1.0 | Disk effective radius (arcsec) |
| $e_1^{\text{disk}}$ | primal | scalar | float64 | 0.0 | Disk ellipticity component 1 |
| $e_2^{\text{disk}}$ | primal | scalar | float64 | 0.0 | Disk ellipticity component 2 |
| $R_{\text{sersic}}^{\text{bulge}}$ | primal | scalar | float64 | 0.5 | Bulge effective radius (arcsec) |
| $x_c$ | primal | scalar | float64 | 0.0 | Shared center x-coordinate (arcsec) |
| $y_c$ | primal | scalar | float64 | 0.0 | Shared center y-coordinate (arcsec) |
| $\alpha_{\text{ps}}$ | primal | scalar | float64 | 0.0 | Point source RA position (arcsec) |
| $\delta_{\text{ps}}$ | primal | scalar | float64 | 0.0 | Point source Dec position (arcsec) |
| $A_{\text{disk}}$ | primal | scalar | float64 | Linear solve | Disk amplitude (solved linearly) |
| $A_{\text{bulge}}$ | primal | scalar | float64 | Linear solve | Bulge amplitude (solved linearly) |
| $A_{\text{ps}}$ | primal | scalar | float64 | Linear solve | Point source amplitude (solved linearly) |

### PSO Auxiliary Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{x}_p^{(k)}$ | auxiliary | $(N_{\text{part}}, N_{\text{dim}})$ | float64 | Uniform in bounds, scaled by $\sigma$ | Particle positions at iteration $k$ |
| $\mathbf{v}_p^{(k)}$ | auxiliary | $(N_{\text{part}}, N_{\text{dim}})$ | float64 | Zero or small random | Particle velocities at iteration $k$ |
| $\mathbf{p}_{\text{best},p}$ | auxiliary | $(N_{\text{part}}, N_{\text{dim}})$ | float64 | $\mathbf{x}_p^{(0)}$ | Personal best position for particle $p$ |
| $\mathbf{g}_{\text{best}}$ | auxiliary | $(N_{\text{dim}},)$ | float64 | Best of initial positions | Global best position |
| $f_{\text{best},p}$ | auxiliary | $(N_{\text{part}},)$ | float64 | $\chi^2(\mathbf{x}_p^{(0)})$ | Personal best fitness |
| $f_{\text{gbest}}$ | auxiliary | scalar | float64 | $\min_p f_{\text{best},p}^{(0)}$ | Global best fitness |

### Constants

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $N_{\text{part}}$ | constant | scalar | int | 50 | Number of PSO particles |
| $N_{\text{iter}}$ | constant | scalar | int | 50 | Number of PSO iterations |
| $\sigma_{\text{scale}}$ | constant | scalar | float64 | 1.0 | PSO sigma scale factor |
| $N_{\text{pix}}$ | constant | scalar | int | 80 | Image dimension (pixels) |
| $\Delta_{\text{pix}}$ | constant | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\sigma_{\text{bkg}}$ | constant | scalar | float64 | 0.1 | Background RMS noise |
| $t_{\text{exp}}$ | constant | scalar | float64 | 100.0 | Exposure time (seconds) |
| $\text{FWHM}$ | constant | scalar | float64 | 0.1 | PSF FWHM (arcsec) |
| $\tau_{\text{PSF}}$ | constant | scalar | float64 | 3 | PSF truncation (units of FWHM) |
| $n_{\text{disk}}$ | constant | scalar | float64 | 1 | Fixed disk Sérsic index |
| $n_{\text{bulge}}$ | constant | scalar | float64 | 4 | Fixed bulge Sérsic index |
| $f_{\text{ss}}$ | constant | scalar | int | 1 | Supersampling factor |
| seed | constant | scalar | int | 42 | Random seed |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{Y}^{\text{obs}}$ | observation | $(80, 80)$ | float64 | Simulated with noise | Observed image data |

### Ground Truth (for simulation)

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $x_c^{\text{true}}$ | constant | scalar | float64 | 0.02 | True center x |
| $y_c^{\text{true}}$ | constant | scalar | float64 | 0.01 | True center y |
| $A_{\text{ps}}^{\text{true}}$ | constant | scalar | float64 | 10000 | True point source amplitude |
| $A_{\text{disk}}^{\text{true}}$ | constant | scalar | float64 | 400 | True disk amplitude |
| $A_{\text{bulge}}^{\text{true}}$ | constant | scalar | float64 | 400 | True bulge amplitude |
| $R_{\text{disk}}^{\text{true}}$ | constant | scalar | float64 | 0.7 | True disk effective radius |
| $R_{\text{bulge}}^{\text{true}}$ | constant | scalar | float64 | 0.3 | True bulge effective radius |
| $\phi^{\text{true}}$ | constant | scalar | float64 | 0.3 | True disk position angle |
| $q^{\text{true}}$ | constant | scalar | float64 | 0.6 | True disk axis ratio |

---

## 3. PREPROCESSING PIPELINE

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 1 | Ellipticity Conversion | $e_1 = \frac{1-q}{1+q}\cos(2\phi), \quad e_2 = \frac{1-q}{1+q}\sin(2\phi)$ | $\phi=0.3$, $q=0.6$ | $e_1, e_2$ (scalars) |
| 2 | Coordinate Transform Matrix | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ | $\Delta_{\text{pix}}=0.05$ | $\mathbf{T} \in \mathbb{R}^{2\times2}$ |
| 3 | Reference Coordinates | $\alpha_0 = -(\mathbf{T}_{00} c_x + \mathbf{T}_{01} c_y), \quad \delta_0 = -(\mathbf{T}_{10} c_x + \mathbf{T}_{11} c_y)$ where $c_x = c_y = \frac{N_{\text{pix}}-1}{2} = 39.5$ | $\mathbf{T}$, $N_{\text{pix}}=80$ | $\alpha_0=1.975$, $\delta_0=-1.975$ |
| 4 | PSF Construction | $P(r) = \frac{1}{2\pi\sigma_{\text{PSF}}^2}\exp\left(-\frac{r^2}{2\sigma_{\text{PSF}}^2}\right)$ where $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$ | FWHM=0.1, $\tau=3$, $\Delta_{\text{pix}}=0.05$ | PSF kernel (truncated at $3\times$FWHM) |
| 5 | Clean Image Simulation | $\mathbf{Y}^{\text{clean}} = P \ast (I_{\text{disk}} + I_{\text{bulge}}) + P \ast (A_{\text{ps}} \cdot \delta(\mathbf{r} - \mathbf{r}_{\text{ps}}))$ | True parameters, PSF | $\mathbf{Y}^{\text{clean}} \in \mathbb{R}^{80\times80}$ |
| 6 | Poisson Noise Generation | $\mathbf{C} = \max(\mathbf{Y}^{\text{clean}} \cdot t_{\text{exp}}, 0), \quad \mathbf{N}_{\text{Poisson}} = \frac{\text{Poisson}(\mathbf{C})}{t_{\text{exp}}} - \mathbf{Y}^{\text{clean}}$ | $\mathbf{Y}^{\text{clean}}$, $t_{\text{exp}}=100$ | $\mathbf{N}_{\text{Poisson}} \in \mathbb{R}^{80\times80}$ |
| 7 | Background Noise Generation | $\mathbf{N}_{\text{bkg}} = \sigma_{\text{bkg}} \cdot \mathbf{Z}, \quad \mathbf{Z}_{ij} \sim \mathcal{N}(0,1)$ | $\sigma_{\text{bkg}}=0.1$ | $\mathbf{N}_{\text{bkg}} \in \mathbb{R}^{80\times80}$ |
| 8 | Observed Image Assembly | $\mathbf{Y}^{\text{obs}} = \mathbf{Y}^{\text{clean}} + \mathbf{N}_{\text{bkg}} + \mathbf{N}_{\text{Poisson}}$ | Steps 5-7 outputs | $\mathbf{Y}^{\text{obs}} \in \mathbb{R}^{80\times80}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 1 | Parameter Bounds Definition | $\boldsymbol{\theta}_{\text{lower}} = [-0.5, -0.5, 0.001, 0.001, -10, -10]^T$ <br> $\boldsymbol{\theta}_{\text{upper}} = [0.5, 0.5, 10, 10, 10, 10]^T$ | Hardcoded values | Bound vectors |
| 2 | Initial Parameter Guess | $\boldsymbol{\theta}^{(0)} = [0, 0, 1.0, 0.5, 0, 0]^T$ for $[e_1, e_2, R_{\text{disk}}, R_{\text{bulge}}, x_c, y_c]$ | Hardcoded values | $\boldsymbol{\theta}^{(0)}$ |
| 3 | Parameter Sigma Definition | $\boldsymbol{\sigma}_{\theta} = [0.1, 0.1, 0.5, 0.3, 0.1, 0.1]^T$ | Hardcoded values | Search scale |
| 4 | Constraint Configuration | Joint centers: $\{(\text{disk}, \text{bulge}): [x_c, y_c]\}$, $\{(\text{disk}, \text{ps}), (\text{bulge}, \text{ps})\}$ | Model structure | Constraint mapping |
| 5 | PSO Swarm Initialization | $\mathbf{x}_p^{(0)} \sim \mathcal{U}(\boldsymbol{\theta}^{(0)} - \sigma_{\text{scale}}\boldsymbol{\sigma}_{\theta}, \boldsymbol{\theta}^{(0)} + \sigma_{\text{scale}}\boldsymbol{\sigma}_{\theta})$ for $p=1,\ldots,N_{\text{part}}$ | $\boldsymbol{\theta}^{(0)}$, $\boldsymbol{\sigma}_{\theta}$, $\sigma_{\text{scale}}=1$ | $\{\mathbf{x}_p^{(0)}\}_{p=1}^{50}$ |
| 6 | Initial Fitness Evaluation | $f_p^{(0)} = \chi^2(\mathbf{x}_p^{(0)})$ for all $p$ | Particle positions | Fitness values |
| 7 | Personal/Global Best Init | $\mathbf{p}_{\text{best},p} = \mathbf{x}_p^{(0)}$, $f_{\text{best},p} = f_p^{(0)}$, $\mathbf{g}_{\text{best}} = \arg\min_p f_p^{(0)}$ | Step 6 outputs | Best trackers |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 1, \ldots, N_{\text{iter}}$ where $N_{\text{iter}} = 50$)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| 1 | Velocity Update | $\mathbf{v}_p^{(k)} = \omega \mathbf{v}_p^{(k-1)} + c_1 r_1 (\mathbf{p}_{\text{best},p} - \mathbf{x}_p^{(k-1)}) + c_2 r_2 (\mathbf{g}_{\text{best}} - \mathbf{x}_p^{(k-1)})$ | Previous velocity, positions, bests; $r_1, r_2 \sim \mathcal{U}(0,1)$ | $\mathbf{v}_p^{(k)}$ | PSO standard update with inertia $\omega$, cognitive $c_1$, social $c_2$ |
| 2 | Position Update | $\mathbf{x}_p^{(k)} = \mathbf{x}_p^{(k-1)} + \mathbf{v}_p^{(k)}$ | Previous position, new velocity | $\mathbf{x}_p^{(k)}$ | Unconstrained move |
| 3 | Boundary Enforcement | $x_{p,d}^{(k)} = \text{clip}(x_{p,d}^{(k)}, \theta_{\text{lower},d}, \theta_{\text{upper},d})$ | Position, bounds | Clipped $\mathbf{x}_p^{(k)}$ | Reflective or absorbing boundary |
| 4 | Constraint Application | Apply joint center constraints: set $x_c^{\text{bulge}} = x_c^{\text{disk}}$, $y_c^{\text{bulge}} = y_c^{\text{disk}}$, $\alpha_{\text{ps}} = x_c^{\text{disk}}$, $\delta_{\text{ps}} = y_c^{\text{disk}}$ | Parameter vector | Constrained $\boldsymbol{\theta}_p$ | Reduces effective dimensionality |
| 5 | Forward Model Evaluation | | | | **BEGIN_SUBLOOP** (Linear solve for amplitudes) |
| 5.1 | Sérsic Disk Profile | $I_{\text{disk}}(\mathbf{r}) = A_{\text{disk}} \exp\left(-b_1 \left[\left(\frac{r'}{R_{\text{disk}}}\right)^{1/n_{\text{disk}}} - 1\right]\right)$ where $r' = \sqrt{q_{\text{disk}}((x-x_c)\cos\theta + (y-y_c)\sin\theta)^2 + ((y-y_c)\cos\theta - (x-x_c)\sin\theta)^2/q_{\text{disk}}}$, $b_1 \approx 1.678$ | $R_{\text{disk}}, e_1, e_2, x_c, y_c$ | Disk surface brightness map | Ellipticity: $q = \frac{1-|e|}{1+|e|}$, $\theta = \frac{1}{2}\arctan(e_2/e_1)$ |
| 5.2 | Sérsic Bulge Profile | $I_{\text{bulge}}(\mathbf{r}) = A_{\text{bulge}} \exp\left(-b_4 \left[\left(\frac{r}{R_{\text{bulge}}}\right)^{1/4} - 1\right]\right)$ where $b_4 \approx 7.669$ | $R_{\text{bulge}}, x_c, y_c$ | Bulge surface brightness map | Circular profile (no ellipticity) |
| 5.3 | Point Source Model | $I_{\text{ps}}(\mathbf{r}) = A_{\text{ps}} \cdot \delta(\mathbf{r} - \mathbf{r}_{\text{ps}})$ | $\alpha_{\text{ps}}, \delta_{\text{ps}}$ | Delta function at PS position | Subpixel interpolation applied |
| 5.4 | PSF Convolution | $\tilde{I}_{\text{disk}} = P \ast I_{\text{disk}}$, $\tilde{I}_{\text{bulge}} = P \ast I_{\text{bulge}}$, $\tilde{I}_{\text{ps}} = P \ast I_{\text{ps}}$ | Component maps, PSF $P$ | Convolved components | FFT-based convolution |
| 5.5 | Design Matrix Assembly | $\mathbf{M} = [\text{vec}(\tilde{I}_{\text{disk}}), \text{vec}(\tilde{I}_{\text{bulge}}), \text{vec}(\tilde{I}_{\text{ps}})]$ with unit amplitudes | Convolved normalized components | $\mathbf{M} \in \mathbb{R}^{6400 \times 3}$ | Columns are basis images |
| 5.6 | Linear Amplitude Solve | $\mathbf{a}^* = \arg\min_{\mathbf{a} \geq 0} \|\mathbf{y}^{\text{obs}} - \mathbf{M}\mathbf{a}\|_{\mathbf{C}^{-1}}^2$ where $\mathbf{a} = [A_{\text{disk}}, A_{\text{bulge}}, A_{\text{ps}}]^T$ | $\mathbf{M}$, $\mathbf{y}^{\text{obs}}$, noise covariance | Optimal amplitudes $\mathbf{a}^*$ | Non-negative least squares |
| | | | | | **END_SUBLOOP** |
| 6 | Model Image Construction | $\mathbf{Y}^{\text{model}} = A_{\text{disk}}^* \tilde{I}_{\text{disk}} + A_{\text{bulge}}^* \tilde{I}_{\text{bulge}} + A_{\text{ps}}^* \tilde{I}_{\text{ps}}$ | Optimal amplitudes, convolved components | $\mathbf{Y}^{\text{model}} \in \mathbb{R}^{80\times80}$ | Full forward model |
| 7 | Chi-squared Computation | $\chi^2_p = \sum_{i,j} \frac{(Y_{ij}^{\text{obs}} - Y_{ij}^{\text{model}})^2}{\sigma_{\text{bkg}}^2 + Y_{ij}^{\text{model}}/t_{\text{exp}}}$ | Model, observation, noise params | $f_p^{(k)} = \chi^2_p$ | Per-pixel noise model |
| 8 | Personal Best Update | If $f_p^{(k)} < f_{\text{best},p}$: $\mathbf{p}_{\text{best},p} \leftarrow \mathbf{x}_p^{(k)}$, $f_{\text{best},p} \leftarrow f_p^{(k)}$ | Current fitness, personal best | Updated personal best | Per-particle update |
| 9 | Global Best Update | If $\min_p f_p^{(k)} < f_{\text{gbest}}$: $\mathbf{g}_{\text{best}} \leftarrow \arg\min_p \mathbf{x}_p^{(k)}$, $f_{\text{gbest}} \leftarrow \min_p f_p^{(k)}$ | All particle fitnesses | Updated global best | Swarm-wide comparison |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 1 | Extract Best Parameters | $\boldsymbol{\theta}^* = \mathbf{g}_{\text{best}}$ | Global best from PSO | Optimal nonlinear parameters |
| 2 | Final Linear Solve | Recompute $\mathbf{a}^*$ at $\boldsymbol{\theta}^*$ using `ImageLinearFit.image_linear_solve()` | Best structural params | Final amplitudes |
| 3 | Reconstruct Final Image | $\mathbf{Y}^{\text{recon}} = \mathcal{F}(\boldsymbol{\theta}^*, \mathbf{a}^*)$ | All optimal parameters | Reconstructed image |
| 4 | Compute Residual | $\mathbf{R} = \mathbf{Y}^{\text{obs}} - \mathbf{Y}^{\text{recon}}$ | Observation, reconstruction | Residual map |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k = N_{\text{iter}} = 50$
- **parameters**: 
  - $N_{\text{iter}} = 50$ (maximum iterations)
  - $N_{\text{part}} = 50$ (swarm size)
- **check_frequency**: After each complete iteration over all particles

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Residual RMS | $\text{RMS} = \sqrt{\frac{1}{N_{\text{pix}}^2} \sum_{i,j} (Y_{ij}^{\text{obs}} - Y_{ij}^{\text{recon}})^2}$ | Root mean square of pixel residuals |
| Reconstructed Sum | $S_{\text{recon}} = \sum_{i,j} Y_{ij}^{\text{recon}}$ | Total flux in reconstructed image |
| True Sum | $S_{\text{true}} = \sum_{i,j} Y_{ij}^{\text{obs}}$ | Total flux in observed image |
| Flux Conservation Ratio | $\rho = \frac{S_{\text{recon}}}{S_{\text{true}}}$ | Ratio of reconstructed to observed total flux |
| Chi-squared | $\chi^2 = \sum_{i,j} \frac{(Y_{ij}^{\text{obs}} - Y_{ij}^{\text{recon}})^2}{\sigma_{ij}^2}$ | Goodness of fit statistic |
| Reduced Chi-squared | $\chi^2_{\nu} = \frac{\chi^2}{N_{\text{pix}}^2 - N_{\text{params}}}$ | Chi-squared per degree of freedom |
| Fitting Time | $T_{\text{fit}} = t_{\text{end}} - t_{\text{start}}$ | Wall-clock time for optimization (seconds) |