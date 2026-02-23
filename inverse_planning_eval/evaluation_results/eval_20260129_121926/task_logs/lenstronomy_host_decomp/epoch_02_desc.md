# Algorithmic Specification Document: Astronomical Image Decomposition via Particle Swarm Optimization

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\boldsymbol{\theta}_{\text{NL}}} \left[ \min_{\mathbf{a}} \left\| \mathbf{y}^{\text{obs}} - \mathbf{M}(\boldsymbol{\theta}_{\text{NL}}) \mathbf{a} \right\|_{\mathbf{C}^{-1}}^2 \right]$$

where the weighted norm is defined as:
$$\left\| \mathbf{r} \right\|_{\mathbf{C}^{-1}}^2 = \mathbf{r}^\top \mathbf{C}^{-1} \mathbf{r} = \sum_{i=1}^{N_{\text{pix}}} \frac{r_i^2}{\sigma_i^2}$$

### data_fidelity_term
$$\chi^2(\boldsymbol{\theta}) = \sum_{i=1}^{N_{\text{pix}}} \frac{\left( y_i^{\text{obs}} - y_i^{\text{model}}(\boldsymbol{\theta}) \right)^2}{\sigma_i^2}$$

where the noise variance per pixel:
$$\sigma_i^2 = \sigma_{\text{bkg}}^2 + \frac{\max(y_i^{\text{model}}, 0)}{t_{\text{exp}}}$$

### regularization_term
None (unconstrained maximum likelihood estimation)

### constraints
| Constraint Type | Expression | Description |
|-----------------|------------|-------------|
| Joint center (light components) | $c_x^{\text{disk}} = c_x^{\text{bulge}}$, $c_y^{\text{disk}} = c_y^{\text{bulge}}$ | Disk and bulge share center |
| Joint center (PS with light) | $x_{\text{PS}} = c_x^{\text{disk}}$, $y_{\text{PS}} = c_y^{\text{disk}}$ | Point source at galaxy center |
| Fixed Sérsic indices | $n_{\text{disk}} = 1$, $n_{\text{bulge}} = 4$ | Exponential disk, de Vaucouleurs bulge |
| Ellipticity bounds | $-0.5 \leq e_1, e_2 \leq 0.5$ | Physical ellipticity range |
| Size bounds | $0.001 \leq R_{\text{sersic}} \leq 10$ | Half-light radius in arcsec |
| Position bounds | $-10 \leq c_x, c_y \leq 10$ | Center position in arcsec |

---

## 2. VARIABLES

### Primal Variables (Nonlinear Parameters)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $R_{\text{sersic}}^{\text{disk}}$ | primal | scalar | float64 | 1.0 | Half-light radius of disk [arcsec] |
| $e_1^{\text{disk}}$ | primal | scalar | float64 | 0.0 | Ellipticity component 1 of disk |
| $e_2^{\text{disk}}$ | primal | scalar | float64 | 0.0 | Ellipticity component 2 of disk |
| $R_{\text{sersic}}^{\text{bulge}}$ | primal | scalar | float64 | 0.5 | Half-light radius of bulge [arcsec] |
| $c_x$ | primal | scalar | float64 | 0.0 | Shared center x-coordinate [arcsec] |
| $c_y$ | primal | scalar | float64 | 0.0 | Shared center y-coordinate [arcsec] |

### Primal Variables (Linear Amplitudes)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $a_{\text{disk}}$ | primal | scalar | float64 | solved | Surface brightness amplitude of disk |
| $a_{\text{bulge}}$ | primal | scalar | float64 | solved | Surface brightness amplitude of bulge |
| $a_{\text{PS}}$ | primal | scalar | float64 | solved | Point source flux amplitude |

### PSO State Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{X}^{(k)}$ | primal | $(N_p, D)$ | float64 | uniform in bounds | Particle positions at iteration $k$ |
| $\mathbf{V}^{(k)}$ | primal | $(N_p, D)$ | float64 | scaled random | Particle velocities at iteration $k$ |
| $\mathbf{P}_{\text{best}}$ | primal | $(N_p, D)$ | float64 | $\mathbf{X}^{(0)}$ | Personal best positions |
| $\mathbf{g}_{\text{best}}$ | primal | $(D,)$ | float64 | best of $\mathbf{X}^{(0)}$ | Global best position |
| $f_{\text{pbest}}$ | primal | $(N_p,)$ | float64 | $f(\mathbf{X}^{(0)})$ | Personal best fitness values |
| $f_{\text{gbest}}$ | primal | scalar | float64 | $\min(f_{\text{pbest}})$ | Global best fitness value |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $N_{\text{pix}}$ | constants | scalar | int | 80 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | constants | scalar | float64 | 0.05 | Pixel scale [arcsec/pixel] |
| $\sigma_{\text{bkg}}$ | constants | scalar | float64 | 0.1 | Background RMS noise [counts/s] |
| $t_{\text{exp}}$ | constants | scalar | float64 | 100.0 | Exposure time [seconds] |
| $\text{FWHM}$ | constants | scalar | float64 | 0.1 | PSF full-width half-maximum [arcsec] |
| $n_{\text{disk}}$ | constants | scalar | float64 | 1 | Sérsic index for disk (fixed) |
| $n_{\text{bulge}}$ | constants | scalar | float64 | 4 | Sérsic index for bulge (fixed) |
| $N_p$ | constants | scalar | int | 50 | Number of PSO particles |
| $N_{\text{iter}}$ | constants | scalar | int | 50 | Number of PSO iterations |
| $D$ | constants | scalar | int | 6 | Dimension of search space |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{Y}^{\text{obs}}$ | observations | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | simulated | Observed image data [counts/s] |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Coordinate System Setup
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Pixel-to-Angular Coordinate Transformation |
| formula | $$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$ $$\text{RA}_0 = -\left( T_{00} \cdot \frac{N_{\text{pix}}-1}{2} + T_{01} \cdot \frac{N_{\text{pix}}-1}{2} \right)$$ $$\text{Dec}_0 = -\left( T_{10} \cdot \frac{N_{\text{pix}}-1}{2} + T_{11} \cdot \frac{N_{\text{pix}}-1}{2} \right)$$ |
| inputs | $N_{\text{pix}}$ (scalar), $\Delta_{\text{pix}}$ (scalar) |
| outputs | $\mathbf{T}$ (2×2), $\text{RA}_0$ (scalar), $\text{Dec}_0$ (scalar) |

### Step 3.2: Angular Coordinate Grid Generation
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Compute Angular Coordinates per Pixel |
| formula | For pixel $(i, j)$: $$\alpha_{ij} = \text{RA}_0 + T_{00} \cdot i + T_{01} \cdot j$$ $$\delta_{ij} = \text{Dec}_0 + T_{10} \cdot i + T_{11} \cdot j$$ |
| inputs | $\mathbf{T}$ (2×2), $\text{RA}_0$ (scalar), $\text{Dec}_0$ (scalar), pixel indices $(i,j)$ |
| outputs | $\boldsymbol{\alpha}$ $(N_{\text{pix}} \times N_{\text{pix}})$, $\boldsymbol{\delta}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

### Step 3.3: PSF Kernel Construction
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Gaussian PSF Kernel Generation |
| formula | $$\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2 \ln 2}}$$ $$K_{mn} = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left( -\frac{m^2 + n^2}{2\sigma_{\text{PSF}}^2} \right)$$ Kernel size: $N_K = 2 \cdot \lfloor 3\sigma_{\text{PSF}} / \Delta_{\text{pix}} \rfloor + 1$ Normalized: $$\tilde{K}_{mn} = \frac{K_{mn}}{\sum_{m,n} K_{mn}}$$ |
| inputs | FWHM (scalar), $\Delta_{\text{pix}}$ (scalar), truncation=3 |
| outputs | $\tilde{\mathbf{K}}$ $(N_K \times N_K)$ |

### Step 3.4: True Image Simulation
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Forward Model Evaluation (Ground Truth) |
| formula | See Algorithm Step 4.2.1 for full forward model |
| inputs | $\boldsymbol{\theta}^{\text{true}}$ (true parameters), $\tilde{\mathbf{K}}$ (PSF kernel) |
| outputs | $\mathbf{Y}^{\text{clean}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

### Step 3.5: Poisson Noise Generation
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Shot Noise Simulation |
| formula | $$C_{ij} = \max(Y_{ij}^{\text{clean}} \cdot t_{\text{exp}}, 0)$$ $$\tilde{C}_{ij} \sim \text{Poisson}(C_{ij})$$ $$N_{ij}^{\text{Poisson}} = \frac{\tilde{C}_{ij}}{t_{\text{exp}}} - Y_{ij}^{\text{clean}}$$ |
| inputs | $\mathbf{Y}^{\text{clean}}$ $(N_{\text{pix}} \times N_{\text{pix}})$, $t_{\text{exp}}$ (scalar) |
| outputs | $\mathbf{N}^{\text{Poisson}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

### Step 3.6: Background Noise Generation
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Gaussian Background Noise |
| formula | $$N_{ij}^{\text{bkg}} = \sigma_{\text{bkg}} \cdot z_{ij}, \quad z_{ij} \sim \mathcal{N}(0, 1)$$ |
| inputs | $\sigma_{\text{bkg}}$ (scalar), shape $(N_{\text{pix}}, N_{\text{pix}})$ |
| outputs | $\mathbf{N}^{\text{bkg}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

### Step 3.7: Noisy Image Assembly
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Combine Signal and Noise |
| formula | $$Y_{ij}^{\text{obs}} = Y_{ij}^{\text{clean}} + N_{ij}^{\text{bkg}} + N_{ij}^{\text{Poisson}}$$ |
| inputs | $\mathbf{Y}^{\text{clean}}$, $\mathbf{N}^{\text{bkg}}$, $\mathbf{N}^{\text{Poisson}}$ (all $N_{\text{pix}} \times N_{\text{pix}}$) |
| outputs | $\mathbf{Y}^{\text{obs}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |

---

## 4. ALGORITHM STEPS

### Initialization

#### Step 4.1.1: Parameter Space Definition
| Field | Value |
|-------|-------|
| step_name | Define Search Bounds |
| formula | $$\boldsymbol{\theta}^{\text{lower}} = (R_{\text{disk}}^{\min}, e_1^{\min}, e_2^{\min}, R_{\text{bulge}}^{\min}, c_x^{\min}, c_y^{\min})$$ $$= (0.001, -0.5, -0.5, 0.001, -10, -10)$$ $$\boldsymbol{\theta}^{\text{upper}} = (R_{\text{disk}}^{\max}, e_1^{\max}, e_2^{\max}, R_{\text{bulge}}^{\max}, c_x^{\max}, c_y^{\max})$$ $$= (10, 0.5, 0.5, 10, 10, 10)$$ |
| inputs | problem specification |
| outputs | $\boldsymbol{\theta}^{\text{lower}}$ $(D,)$, $\boldsymbol{\theta}^{\text{upper}}$ $(D,)$ |
| computational_note | $D=6$ free nonlinear parameters after applying constraints |

#### Step 4.1.2: Particle Position Initialization
| Field | Value |
|-------|-------|
| step_name | Initialize Particle Swarm Positions |
| formula | $$X_{pd}^{(0)} = \theta_d^{\text{lower}} + U_{pd} \cdot (\theta_d^{\text{upper}} - \theta_d^{\text{lower}})$$ where $U_{pd} \sim \text{Uniform}(0, 1)$ for $p = 1, \ldots, N_p$ and $d = 1, \ldots, D$ |
| inputs | $\boldsymbol{\theta}^{\text{lower}}$ $(D,)$, $\boldsymbol{\theta}^{\text{upper}}$ $(D,)$, $N_p$ (scalar) |
| outputs | $\mathbf{X}^{(0)}$ $(N_p \times D)$ |
| computational_note | Initial positions centered around initial guess with sigma_scale=1 |

#### Step 4.1.3: Particle Velocity Initialization
| Field | Value |
|-------|-------|
| step_name | Initialize Particle Velocities |
| formula | $$V_{pd}^{(0)} = \sigma_d \cdot z_{pd}, \quad z_{pd} \sim \mathcal{N}(0, 1)$$ where $\sigma_d$ is the parameter-specific step size from kwargs_sigma |
| inputs | $\boldsymbol{\sigma}$ $(D,)$ (sigma values), $N_p$ (scalar) |
| outputs | $\mathbf{V}^{(0)}$ $(N_p \times D)$ |
| computational_note | sigma_scale=1.0 multiplier applied |

#### Step 4.1.4: Initial Fitness Evaluation
| Field | Value |
|-------|-------|
| step_name | Evaluate Initial Population Fitness |
| formula | For each particle $p$: $$f_p^{(0)} = \chi^2(\mathbf{X}_p^{(0)})$$ (see Step 4.2.4 for $\chi^2$ computation) |
| inputs | $\mathbf{X}^{(0)}$ $(N_p \times D)$, $\mathbf{Y}^{\text{obs}}$ |
| outputs | $\mathbf{f}^{(0)}$ $(N_p,)$ |
| computational_note | Includes linear amplitude solve per evaluation |

#### Step 4.1.5: Personal and Global Best Initialization
| Field | Value |
|-------|-------|
| step_name | Initialize Best Tracking Variables |
| formula | $$\mathbf{P}_{\text{best}} = \mathbf{X}^{(0)}$$ $$\mathbf{f}_{\text{pbest}} = \mathbf{f}^{(0)}$$ $$p^* = \arg\min_p f_p^{(0)}$$ $$\mathbf{g}_{\text{best}} = \mathbf{X}_{p^*}^{(0)}$$ $$f_{\text{gbest}} = f_{p^*}^{(0)}$$ |
| inputs | $\mathbf{X}^{(0)}$ $(N_p \times D)$, $\mathbf{f}^{(0)}$ $(N_p,)$ |
| outputs | $\mathbf{P}_{\text{best}}$ $(N_p \times D)$, $\mathbf{f}_{\text{pbest}}$ $(N_p,)$, $\mathbf{g}_{\text{best}}$ $(D,)$, $f_{\text{gbest}}$ (scalar) |
| computational_note | Minimization problem: lower fitness is better |

---

### Iteration Loop

**BEGIN ITERATION LOOP** ($k = 0, 1, \ldots, N_{\text{iter}} - 1$)

#### Step 4.2.1: Forward Model - Sérsic Profile Evaluation
| Field | Value |
|-------|-------|
| step_name | Compute Sérsic Surface Brightness Templates |
| formula | **Elliptical radius transformation:** $$q = \frac{1 - \sqrt{e_1^2 + e_2^2}}{1 + \sqrt{e_1^2 + e_2^2}}$$ $$\phi = \frac{1}{2}\arctan\left(\frac{e_2}{e_1}\right)$$ $$x' = (\alpha - c_x)\cos\phi + (\delta - c_y)\sin\phi$$ $$y' = -(\alpha - c_x)\sin\phi + (\delta - c_y)\cos\phi$$ $$r_{\text{ell}} = \sqrt{x'^2 + \frac{y'^2}{q^2}}$$ **Sérsic profile (unnormalized template):** $$I_{\text{Sérsic}}(r; R_s, n) = \exp\left\{ -b_n \left[ \left(\frac{r}{R_s}\right)^{1/n} - 1 \right] \right\}$$ where $b_n \approx 1.9992n - 0.3271$ (approximation for $0.5 < n < 10$) **Component templates:** $$\mathbf{S}^{\text{disk}}_{ij} = I_{\text{Sérsic}}(r_{\text{ell},ij}^{\text{disk}}; R_s^{\text{disk}}, n=1)$$ $$\mathbf{S}^{\text{bulge}}_{ij} = I_{\text{Sérsic}}(r_{\text{ell},ij}^{\text{bulge}}; R_s^{\text{bulge}}, n=4)$$ |
| inputs | $\boldsymbol{\alpha}$, $\boldsymbol{\delta}$ (coordinate grids), $(R_s^{\text{disk}}, e_1, e_2, R_s^{\text{bulge}}, c_x, c_y)$ |
| outputs | $\mathbf{S}^{\text{disk}}$ $(N_{\text{pix}} \times N_{\text{pix}})$, $\mathbf{S}^{\text{bulge}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |
| computational_note | Bulge uses spherical Sérsic (no ellipticity parameters in this configuration) |

#### Step 4.2.2: Forward Model - Point Source Template
| Field | Value |
|-------|-------|
| step_name | Compute Point Source Template |
| formula | $$\mathbf{S}^{\text{PS}}_{ij} = \delta(i - i_{\text{PS}}, j - j_{\text{PS}})$$ where $(i_{\text{PS}}, j_{\text{PS}})$ is the sub-pixel position computed via: $$i_{\text{PS}} = \frac{\text{RA}_{\text{PS}} - \text{RA}_0}{T_{00}}$$ $$j_{\text{PS}} = \frac{\text{Dec}_{\text{PS}} - \text{Dec}_0}{T_{11}}$$ Implemented via bilinear interpolation for sub-pixel positioning. |
| inputs | $(c_x, c_y)$ as point source position, coordinate transformation |
| outputs | $\mathbf{S}^{\text{PS}}$ $(N_{\text{pix}} \times N_{\text{pix}})$ |
| computational_note | Joint constraint: PS position equals galaxy center |

#### Step 4.2.3: Forward Model - PSF Convolution and Design Matrix Assembly
| Field | Value |
|-------|-------|
| step_name | Convolve Templates and Build Design Matrix |
| formula | **PSF convolution (2D):** $$\tilde{\mathbf{S}}^{(c)} = \tilde{\mathbf{K}} * \mathbf{S}^{(c)}$$ defined as: $$\tilde{S}^{(c)}_{ij} = \sum_{m,n} \tilde{K}_{mn} \cdot S^{(c)}_{i-m, j-n}$$ for $c \in \{\text{disk}, \text{bulge}, \text{PS}\}$ **Vectorization and design matrix:** $$\mathbf{m}^{(c)} = \text{vec}(\tilde{\mathbf{S}}^{(c)}) \in \mathbb{R}^{N_{\text{pix}}^2}$$ $$\mathbf{M} = \begin{pmatrix} \mathbf{m}^{\text{disk}} & \mathbf{m}^{\text{bulge}} & \mathbf{m}^{\text{PS}} \end{pmatrix} \in \mathbb{R}^{N_{\text{pix}}^2 \times 3}$$ |
| inputs | $\mathbf{S}^{\text{disk}}$, $\mathbf{S}^{\text{bulge}}$, $\mathbf{S}^{\text{PS}}$ (templates), $\tilde{\mathbf{K}}$ (PSF) |
| outputs | $\mathbf{M}$ $(N_{\text{pix}}^2 \times 3)$ |
| computational_note | Convolution performed via FFT for efficiency |

#### Step 4.2.4: Linear Amplitude Solve
| Field | Value |
|-------|-------|
| step_name | Solve for Optimal Linear Amplitudes |
| formula | **Weighted least squares:** $$\mathbf{a}^* = \arg\min_{\mathbf{a}} \left\| \mathbf{y} - \mathbf{M}\mathbf{a} \right\|_{\mathbf{C}^{-1}}^2$$ **Normal equations:** $$\mathbf{a}^* = \left( \mathbf{M}^\top \mathbf{C}^{-1} \mathbf{M} \right)^{-1} \mathbf{M}^\top \mathbf{C}^{-1} \mathbf{y}$$ where $\mathbf{y} = \text{vec}(\mathbf{Y}^{\text{obs}})$ and $\mathbf{C} = \text{diag}(\sigma_1^2, \ldots, \sigma_{N_{\text{pix}}^2}^2)$ **Noise covariance (diagonal):** $$\sigma_i^2 = \sigma_{\text{bkg}}^2 + \frac{\max(y_i^{\text{model}}, 0)}{t_{\text{exp}}}$$ |
| inputs | $\mathbf{M}$ $(N_{\text{pix}}^2 \times 3)$, $\mathbf{y}$ $(N_{\text{pix}}^2,)$, $\mathbf{C}$ (diagonal covariance) |
| outputs | $\mathbf{a}^* = (a_{\text{disk}}^*, a_{\text{bulge}}^*, a_{\text{PS}}^*)^\top$ $(3,)$ |
| computational_note | Iterative refinement may be applied for noise model consistency |

#### Step 4.2.5: Chi-Squared Fitness Evaluation
| Field | Value |
|-------|-------|
| step_name | Compute Fitness for Each Particle |
| formula | For each particle $p$ with nonlinear parameters $\boldsymbol{\theta}_p$: $$\mathbf{y}_p^{\text{model}} = \mathbf{M}(\boldsymbol{\theta}_p) \mathbf{a}^*(\boldsymbol{\theta}_p)$$ $$\chi^2_p = \sum_{i=1}^{N_{\text{pix}}^2} \frac{(y_i - y_{p,i}^{\text{model}})^2}{\sigma_i^2}$$ $$f_p^{(k+1)} = \chi^2_p$$ |
| inputs | $\mathbf{X}^{(k)}$ $(N_p \times D)$, forward model components |
| outputs | $\mathbf{f}^{(k+1)}$ $(N_p,)$ |
| computational_note | Embarrassingly parallel across particles |

#### Step 4.2.6: Personal Best Update
| Field | Value |
|-------|-------|
| step_name | Update Personal Best Positions |
| formula | For each particle $p = 1, \ldots, N_p$: $$\text{If } f_p^{(k+1)} < f_{\text{pbest},p}:$$ $$\quad \mathbf{P}_{\text{best},p} \leftarrow \mathbf{X}_p^{(k+1)}$$ $$\quad f_{\text{pbest},p} \leftarrow f_p^{(k+1)}$$ |
| inputs | $\mathbf{X}^{(k+1)}$ $(N_p \times D)$, $\mathbf{f}^{(k+1)}$ $(N_p,)$, $\mathbf{P}_{\text{best}}$ $(N_p \times