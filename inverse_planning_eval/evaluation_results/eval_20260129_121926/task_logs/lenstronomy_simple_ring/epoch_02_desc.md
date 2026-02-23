# Algorithmic Specification Document: Gravitational Lens Image Reconstruction

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta} \in \Omega} \mathcal{L}(\boldsymbol{\theta})$$

where the negative log-likelihood (up to constants) is:

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^{N_{\text{pix}}} \frac{\left( d_i - m_i(\boldsymbol{\theta}) \right)^2}{\sigma_{\text{bkg}}^2}$$

### data_fidelity_term
$$\mathcal{D}(\boldsymbol{\theta}) = \frac{1}{2} \| \mathbf{d} - \mathbf{m}(\boldsymbol{\theta}) \|_{\Sigma^{-1}}^2 = \frac{1}{2} \sum_{i=1}^{N_{\text{pix}}} \frac{\left( d_i - m_i(\boldsymbol{\theta}) \right)^2}{\sigma_{\text{bkg}}^2}$$

where:
- $\mathbf{d} \in \mathbb{R}^{N_{\text{pix}}}$: Observed image (flattened)
- $\mathbf{m}(\boldsymbol{\theta}) \in \mathbb{R}^{N_{\text{pix}}}$: Forward model prediction
- $\sigma_{\text{bkg}}$: Background RMS noise level
- $\Sigma = \sigma_{\text{bkg}}^2 \mathbf{I}$: Diagonal covariance matrix

### regularization_term
None (pure maximum likelihood estimation)

### constraints
Box constraints on parameter space $\Omega$:

| Parameter Group | Parameter | Lower Bound | Upper Bound |
|-----------------|-----------|-------------|-------------|
| SIE Lens | $\theta_E$ | 0.01 | 10.0 |
| SIE Lens | $e_1, e_2$ | -0.5 | 0.5 |
| SIE Lens | $x_c, y_c$ | -10 | 10 |
| Shear | $\gamma_1, \gamma_2$ | -0.2 | 0.2 |
| Source Sérsic | $R_{\text{sersic}}$ | 0.001 | 10 |
| Source Sérsic | $n_{\text{sersic}}$ | 0.5 | 5.0 |
| Source Sérsic | $e_1, e_2$ | -0.5 | 0.5 |
| Source Sérsic | $x_c, y_c$ | -10 | 10 |
| Source Sérsic | $A$ (amp) | 0 | 100 |
| Lens Light Sérsic | (same structure as Source) | — | — |

Fixed constraints:
- Shear reference point: $(\text{ra}_0, \text{dec}_0) = (0, 0)$

---

## 2. VARIABLES

### Primal Variables (Parameters to Optimize)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta_E$ | primal | scalar | float64 | 0.7 | Einstein radius of SIE lens (arcsec) |
| $e_1^{\text{lens}}$ | primal | scalar | float64 | 0.0 | First ellipticity component of SIE |
| $e_2^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Second ellipticity component of SIE |
| $x_c^{\text{lens}}$ | primal | scalar | float64 | 0.0 | SIE center x-coordinate (arcsec) |
| $y_c^{\text{lens}}$ | primal | scalar | float64 | 0.0 | SIE center y-coordinate (arcsec) |
| $\gamma_1$ | primal | scalar | float64 | 0.0 | External shear component 1 |
| $\gamma_2$ | primal | scalar | float64 | 0.0 | External shear component 2 |
| $A^{\text{src}}$ | primal | scalar | float64 | 16.0 | Source amplitude |
| $R_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 0.2 | Source Sérsic radius (arcsec) |
| $n_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 1.0 | Source Sérsic index |
| $e_1^{\text{src}}, e_2^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source ellipticity components |
| $x_c^{\text{src}}, y_c^{\text{src}}$ | primal | scalar | float64 | 0.0 | Source center (arcsec) |
| $A^{\text{ll}}$ | primal | scalar | float64 | 16.0 | Lens light amplitude |
| $R_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 0.5 | Lens light Sérsic radius |
| $n_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 2.0 | Lens light Sérsic index |
| $e_1^{\text{ll}}, e_2^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light ellipticity |
| $x_c^{\text{ll}}, y_c^{\text{ll}}$ | primal | scalar | float64 | 0.0 | Lens light center |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{d}$ | observations | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Simulated | Observed image with noise |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $N_{\text{pix}}$ | constants | scalar | int | 60 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | constants | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\sigma_{\text{bkg}}$ | constants | scalar | float64 | 0.005 | Background RMS noise |
| $t_{\text{exp}}$ | constants | scalar | float64 | 500.0 | Exposure time (seconds) |
| $\text{FWHM}$ | constants | scalar | float64 | 0.05 | PSF full-width half-maximum |
| $\mathbf{T}$ | constants | $(2, 2)$ | float64 | See preprocessing | Pixel-to-angle transformation matrix |
| $\mathbf{P}$ | constants | $(N_k, N_k)$ | float64 | Computed | PSF kernel (Gaussian) |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Coordinate System Definition

| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Define pixel-to-coordinate transformation |
| formula | $$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$ |
| inputs | $\Delta_{\text{pix}}$ (scalar) |
| outputs | $\mathbf{T}$ (shape: $2 \times 2$) |

### Step 3.2: Image Center Offset Calculation

| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Compute RA/Dec at pixel origin |
| formula | $$c_x = c_y = \frac{N_{\text{pix}} - 1}{2}$$ $$\text{RA}_0 = -(T_{00} \cdot c_x + T_{01} \cdot c_y)$$ $$\text{Dec}_0 = -(T_{10} \cdot c_x + T_{11} \cdot c_y)$$ |
| inputs | $N_{\text{pix}}$ (scalar), $\mathbf{T}$ (shape: $2 \times 2$) |
| outputs | $\text{RA}_0$, $\text{Dec}_0$ (scalars) |

### Step 3.3: PSF Kernel Generation

| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Generate Gaussian PSF kernel |
| formula | $$\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$$ $$P(x, y) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{x^2 + y^2}{2\sigma_{\text{PSF}}^2}\right)$$ Truncated at $3\sigma_{\text{PSF}}$ |
| inputs | FWHM (scalar), $\Delta_{\text{pix}}$ (scalar) |
| outputs | $\mathbf{P}$ (shape: $(N_k, N_k)$, normalized PSF kernel) |

### Step 3.4: True Model Image Generation

| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Compute noise-free model image |
| formula | $$\mathbf{m}_{\text{true}} = \mathcal{F}(\boldsymbol{\theta}_{\text{true}})$$ (See Forward Operator definition in Section 4) |
| inputs | True parameters $\boldsymbol{\theta}_{\text{true}}$, $\mathbf{T}$, $\mathbf{P}$ |
| outputs | $\mathbf{m}_{\text{true}}$ (shape: $N_{\text{pix}} \times N_{\text{pix}}$) |

### Step 3.5: Poisson Noise Addition

| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Add photon shot noise |
| formula | $$\mathbf{C} = \mathbf{m}_{\text{true}} \cdot t_{\text{exp}}$$ $$C_{ij} \leftarrow \max(C_{ij}, 0)$$ $$\tilde{C}_{ij} \sim \text{Poisson}(C_{ij})$$ $$\mathbf{m}_{\text{poisson}} = \frac{\tilde{\mathbf{C}}}{t_{\text{exp}}}$$ |
| inputs | $\mathbf{m}_{\text{true}}$ (shape: $N_{\text{pix}} \times N_{\text{pix}}$), $t_{\text{exp}}$ (scalar) |
| outputs | $\mathbf{m}_{\text{poisson}}$ (shape: $N_{\text{pix}} \times N_{\text{pix}}$) |

### Step 3.6: Gaussian Background Noise Addition

| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Add Gaussian background noise |
| formula | $$\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{bkg}}^2 \mathbf{I})$$ $$n_{ij} = \sigma_{\text{bkg}} \cdot z_{ij}, \quad z_{ij} \sim \mathcal{N}(0,1)$$ $$\mathbf{d} = \mathbf{m}_{\text{poisson}} + \mathbf{n}$$ |
| inputs | $\mathbf{m}_{\text{poisson}}$ (shape: $N_{\text{pix}} \times N_{\text{pix}}$), $\sigma_{\text{bkg}}$ (scalar) |
| outputs | $\mathbf{d}$ (shape: $N_{\text{pix}} \times N_{\text{pix}}$) — final observed image |

---

## 4. ALGORITHM STEPS

### Forward Operator Definition

The forward operator $\mathcal{F}: \boldsymbol{\theta} \mapsto \mathbf{m}$ computes the model image:

$$\mathbf{m}(\boldsymbol{\theta}) = \mathbf{P} * \left[ I_{\text{src}}(\boldsymbol{\beta}(\boldsymbol{\alpha})) + I_{\text{ll}}(\boldsymbol{\alpha}) \right]$$

where $*$ denotes 2D convolution.

#### Sub-step 4.F1: Pixel Coordinate Grid

| Field | Value |
|-------|-------|
| step_name | Generate angular coordinate grid |
| formula | For pixel $(i, j)$: $$\alpha_x^{(i,j)} = \text{RA}_0 + T_{00} \cdot i + T_{01} \cdot j$$ $$\alpha_y^{(i,j)} = \text{Dec}_0 + T_{10} \cdot i + T_{11} \cdot j$$ |
| inputs | $i, j \in \{0, \ldots, N_{\text{pix}}-1\}$, $\mathbf{T}$, $\text{RA}_0$, $\text{Dec}_0$ |
| outputs | $\boldsymbol{\alpha} = (\alpha_x, \alpha_y)$ — image plane coordinates (shape: $N_{\text{pix}} \times N_{\text{pix}} \times 2$) |

#### Sub-step 4.F2: SIE Lens Deflection

| Field | Value |
|-------|-------|
| step_name | Compute SIE deflection angles |
| formula | Shifted coordinates: $$x' = \alpha_x - x_c^{\text{lens}}, \quad y' = \alpha_y - y_c^{\text{lens}}$$ Axis ratio and orientation from ellipticity: $$\phi = \frac{1}{2}\arctan\left(\frac{e_2}{e_1}\right), \quad |e| = \sqrt{e_1^2 + e_2^2}$$ $$q = \frac{1 - |e|}{1 + |e|}$$ Deflection (SIE): $$\delta\alpha_x^{\text{SIE}} = \frac{\theta_E}{\sqrt{1-q^2}} \arctan\left(\frac{\sqrt{1-q^2} \, x''}{\psi + q^2 s}\right)$$ $$\delta\alpha_y^{\text{SIE}} = \frac{\theta_E}{\sqrt{1-q^2}} \text{arctanh}\left(\frac{\sqrt{1-q^2} \, y''}{\psi + s}\right)$$ where $(x'', y'')$ are coordinates rotated by $\phi$, $s = \sqrt{q^2 x''^2 + y''^2}$, $\psi = \sqrt{x''^2 + q^2 y''^2}$ |
| inputs | $\boldsymbol{\alpha}$, $\theta_E$, $e_1$, $e_2$, $x_c^{\text{lens}}$, $y_c^{\text{lens}}$ |
| outputs | $\delta\boldsymbol{\alpha}^{\text{SIE}}$ (shape: $N_{\text{pix}} \times N_{\text{pix}} \times 2$) |

#### Sub-step 4.F3: External Shear Deflection

| Field | Value |
|-------|-------|
| step_name | Compute shear deflection angles |
| formula | $$\delta\alpha_x^{\text{shear}} = \gamma_1 \alpha_x + \gamma_2 \alpha_y$$ $$\delta\alpha_y^{\text{shear}} = \gamma_2 \alpha_x - \gamma_1 \alpha_y$$ |
| inputs | $\boldsymbol{\alpha}$, $\gamma_1$, $\gamma_2$ |
| outputs | $\delta\boldsymbol{\alpha}^{\text{shear}}$ (shape: $N_{\text{pix}} \times N_{\text{pix}} \times 2$) |

#### Sub-step 4.F4: Total Deflection and Source Plane Mapping

| Field | Value |
|-------|-------|
| step_name | Ray-trace to source plane |
| formula | $$\boldsymbol{\beta} = \boldsymbol{\alpha} - \delta\boldsymbol{\alpha}^{\text{SIE}} - \delta\boldsymbol{\alpha}^{\text{shear}}$$ |
| inputs | $\boldsymbol{\alpha}$, $\delta\boldsymbol{\alpha}^{\text{SIE}}$, $\delta\boldsymbol{\alpha}^{\text{shear}}$ |
| outputs | $\boldsymbol{\beta}$ — source plane coordinates (shape: $N_{\text{pix}} \times N_{\text{pix}} \times 2$) |

#### Sub-step 4.F5: Source Light Evaluation (Sérsic Ellipse)

| Field | Value |
|-------|-------|
| step_name | Evaluate source surface brightness |
| formula | Shifted and rotated coordinates in source plane: $$\beta_x' = \beta_x - x_c^{\text{src}}, \quad \beta_y' = \beta_y - y_c^{\text{src}}$$ Elliptical radius: $$r_{\text{eff}} = \sqrt{q_s \beta_x'^2 + \beta_y'^2 / q_s}$$ where $q_s = (1-|e^{\text{src}}|)/(1+|e^{\text{src}}|)$ Sérsic profile: $$I_{\text{src}}(\boldsymbol{\beta}) = A^{\text{src}} \exp\left[ -b_n \left( \left(\frac{r_{\text{eff}}}{R_{\text{sersic}}^{\text{src}}}\right)^{1/n_{\text{sersic}}^{\text{src}}} - 1 \right) \right]$$ where $b_n \approx 1.9992 n - 0.3271$ (approximation for $0.5 < n < 10$) |
| inputs | $\boldsymbol{\beta}$, $A^{\text{src}}$, $R_{\text{sersic}}^{\text{src}}$, $n_{\text{sersic}}^{\text{src}}$, $e_1^{\text{src}}$, $e_2^{\text{src}}$, $x_c^{\text{src}}$, $y_c^{\text{src}}$ |
| outputs | $I_{\text{src}}$ (shape: $N_{\text{pix}} \times N_{\text{pix}}$) |

#### Sub-step 4.F6: Lens Light Evaluation (Sérsic Ellipse)

| Field | Value |
|-------|-------|
| step_name | Evaluate lens galaxy surface brightness |
| formula | Same functional form as source, evaluated at image plane: $$I_{\text{ll}}(\boldsymbol{\alpha}) = A^{\text{ll}} \exp\left[ -b_n \left( \left(\frac{r_{\text{eff}}^{\text{ll}}}{R_{\text{sersic}}^{\text{ll}}}\right)^{1/n_{\text{sersic}}^{\text{ll}}} - 1 \right) \right]$$ with elliptical radius computed from $(\alpha_x - x_c^{\text{ll}}, \alpha_y - y_c^{\text{ll}})$ |
| inputs | $\boldsymbol{\alpha}$, lens light parameters |
| outputs | $I_{\text{ll}}$ (shape: $N_{\text{pix}} \times N_{\text{pix}}$) |

#### Sub-step 4.F7: PSF Convolution

| Field | Value |
|-------|-------|
| step_name | Convolve with point spread function |
| formula | $$\mathbf{m} = \mathbf{P} * (I_{\text{src}} + I_{\text{ll}})$$ Discrete convolution: $$m_{ij} = \sum_{k,l} P_{kl} \cdot (I_{\text{src}} + I_{\text{ll}})_{i-k, j-l}$$ |
| inputs | $I_{\text{src}}$, $I_{\text{ll}}$ (each $N_{\text{pix}} \times N_{\text{pix}}$), $\mathbf{P}$ (PSF kernel) |
| outputs | $\mathbf{m}$ — model image (shape: $N_{\text{pix}} \times N_{\text{pix}}$) |

---

### Optimization Algorithm

#### Initialization

| Field | Value |
|-------|-------|
| step_name | Initialize PSO particle swarm |
| formula | For particle $p \in \{1, \ldots, N_{\text{particles}}\}$: $$\boldsymbol{\theta}^{(p)}_0 \sim \mathcal{N}(\boldsymbol{\theta}_{\text{init}}, \text{diag}(\boldsymbol{\sigma}^2))$$ Clipped to bounds: $$\theta^{(p)}_{0,k} \leftarrow \text{clip}(\theta^{(p)}_{0,k}, \theta_k^{\text{lower}}, \theta_k^{\text{upper}})$$ Initialize velocities: $$\mathbf{v}^{(p)}_0 = \mathbf{0}$$ Initialize personal best: $$\boldsymbol{\theta}^{(p)}_{\text{pbest}} = \boldsymbol{\theta}^{(p)}_0, \quad f^{(p)}_{\text{pbest}} = \mathcal{L}(\boldsymbol{\theta}^{(p)}_0)$$ Initialize global best: $$\boldsymbol{\theta}_{\text{gbest}} = \arg\min_p f^{(p)}_{\text{pbest}}$$ |
| inputs | $\boldsymbol{\theta}_{\text{init}}$, $\boldsymbol{\sigma}$ (initialization sigma), $N_{\text{particles}} = 50$ |
| outputs | $\{\boldsymbol{\theta}^{(p)}_0\}_{p=1}^{N_{\text{particles}}}$, $\{\mathbf{v}^{(p)}_0\}$, $\boldsymbol{\theta}_{\text{gbest}}$ |

---

**BEGIN ITERATION LOOP (PSO Phase)**

#### Step PSO.1: Velocity Update

| Field | Value |
|-------|-------|
| step_name | Update particle velocities |
| formula | $$\mathbf{v}^{(p)}_{t+1} = \omega \mathbf{v}^{(p)}_t + c_1 r_1 (\boldsymbol{\theta}^{(p)}_{\text{pbest}} - \boldsymbol{\theta}^{(p)}_t) + c_2 r_2 (\boldsymbol{\theta}_{\text{gbest}} - \boldsymbol{\theta}^{(p)}_t)$$ where $\omega$ is inertia weight, $c_1, c_2$ are cognitive/social coefficients, $r_1, r_2 \sim U(0,1)$ |
| inputs | $\mathbf{v}^{(p)}_t$, $\boldsymbol{\theta}^{(p)}_t$, $\boldsymbol{\theta}^{(p)}_{\text{pbest}}$, $\boldsymbol{\theta}_{\text{gbest}}$ |
| outputs | $\mathbf{v}^{(p)}_{t+1}$ |
| computational_note | Applied to all particles $p = 1, \ldots, N_{\text{particles}}$ |

#### Step PSO.2: Position Update

| Field | Value |
|-------|-------|
| step_name | Update particle positions |
| formula | $$\boldsymbol{\theta}^{(p)}_{t+1} = \boldsymbol{\theta}^{(p)}_t + \mathbf{v}^{(p)}_{t+1}$$ $$\theta^{(p)}_{t+1,k} \leftarrow \text{clip}(\theta^{(p)}_{t+1,k}, \theta_k^{\text{lower}}, \theta_k^{\text{upper}})$$ |
| inputs | $\boldsymbol{\theta}^{(p)}_t$, $\mathbf{v}^{(p)}_{t+1}$, bounds |
| outputs | $\boldsymbol{\theta}^{(p)}_{t+1}$ |

#### Step PSO.3: Fitness Evaluation

| Field | Value |
|-------|-------|
| step_name | Evaluate objective function |
| formula | $$f^{(p)}_{t+1} = \mathcal{L}(\boldsymbol{\theta}^{(p)}_{t+1}) = \frac{1}{2\sigma_{\text{bkg}}^2} \sum_{i,j} \left( d_{ij} - m_{ij}(\boldsymbol{\theta}^{(p)}_{t+1}) \right)^2$$ |
| inputs | $\boldsymbol{\theta}^{(p)}_{t+1}$, $\mathbf{d}$, forward operator $\mathcal{F}$ |
| outputs | $f^{(p)}_{t+1}$ (scalar fitness for each particle) |

#### Step PSO.4: Personal and Global Best Update

| Field | Value |
|-------|-------|
| step_name | Update best known positions |
| formula | Personal best: $$\text{if } f^{(p)}_{t+1} < f^{(p)}_{\text{pbest}}: \quad \boldsymbol{\theta}^{(p)}_{\text{pbest}} \leftarrow \boldsymbol{\theta}^{(p)}_{t+1}, \quad f^{(p)}_{\text{pbest}} \leftarrow f^{(p)}_{t+1}$$ Global best: $$\boldsymbol{\theta}_{\text{gbest}} \leftarrow \boldsymbol{\theta}^{(p^*)}_{\text{pbest}} \quad \text{where } p^* = \arg\min_p f^{(p)}_{\text{pbest}}$$ |
| inputs | $f^{(p)}_{t+1}$, $\boldsymbol{\theta}^{(p)}_{t+1}$, current bests |
| outputs | Updated $\boldsymbol{\theta}^{(p)}_{\text{pbest}}$, $\boldsymbol{\theta}_{\text{gbest}}$ |

#### Step PSO.5: Convergence Check

| Field | Value |
|-------|-------|
| step_name | Check PSO termination |
| formula | $$\text{terminate if } t \geq N_{\text{iter}}^{\text{PSO}}$$ |
| inputs | $t$, $N_{\text{iter}}^{\text{PSO}} = 10$ |
| outputs | Boolean termination flag |

**END ITERATION LOOP (PSO Phase)**

---

**BEGIN ITERATION LOOP (MCMC Phase)**

#### Step MCMC.0: Initialize Walkers

| Field | Value |
|-------|-------|
| step_name | Initialize MCMC walkers around PSO solution |
| formula | $$\boldsymbol{\theta}^{(w)}_0 = \boldsymbol{\theta}_{\text{gbest}} + \sigma_{\text{scale}} \cdot \boldsymbol{\sigma} \cdot \mathbf{z}^{(w)}$$ where $\mathbf{z}^{(w)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\sigma_{\text{scale}} = 0.1$ |
| inputs | $\boldsymbol{\theta}_{\text{gbest}}$, $\boldsymbol{\sigma}$, $N_{\text{walkers}} = 50$ |
| outputs | $\{\boldsymbol{\theta}^{(w)}_0\}_{w=1}^{N_{\text{walkers}}}$ |

#### Step MCMC.1: Stretch Move Proposal (Affine-Invariant Ensemble Sampler)

| Field | Value |
|-------|-------|
| step_name | Generate stretch move proposal |
| formula | For walker $w$, select complementary walker $w' \neq w$ uniformly at random: $$Z \sim g(z) \propto \frac{1}{\sqrt{z}} \quad \text{for }