# Algorithmic Specification Document: Gravitational Lensing Image Reconstruction

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \left[ \log \mathcal{L}(\mathbf{I}_{\text{obs}} | \boldsymbol{\theta}) + \log P(\boldsymbol{\theta}) \right]$$

Subject to box constraints on all parameters.

Equivalently (minimization form):
$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \left[ \frac{1}{2}\chi^2(\boldsymbol{\theta}) - \log P(\boldsymbol{\theta}) \right]$$

**data_fidelity_term**:
$$\chi^2(\boldsymbol{\theta}) = \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} \frac{\left( I_{\text{obs}}(i,j) - I_{\text{model}}(i,j; \boldsymbol{\theta}) \right)^2}{\sigma^2(i,j)}$$

where the noise variance is:
$$\sigma^2(i,j) = \sigma_{\text{bkg}}^2 + \frac{\max(I_{\text{model}}(i,j), 0)}{t_{\text{exp}}}$$

**regularization_term**:
Gaussian prior on lens ellipticity parameters:
$$-\log P(\boldsymbol{\theta}) = \frac{(e_1^{\text{lens}} - \mu_{e_1})^2}{2\sigma_{e_1}^2} + \frac{(e_2^{\text{lens}} - \mu_{e_2})^2}{2\sigma_{e_2}^2}$$

with $\mu_{e_1} = 0$, $\sigma_{e_1} = 0.2$, $\mu_{e_2} = 0$, $\sigma_{e_2} = 0.2$.

**constraints**:
| Constraint Type | Expression |
|-----------------|------------|
| Box (lens EPL) | $0 \leq \theta_E \leq 10$, $-0.5 \leq e_1, e_2 \leq 0.5$, $1.5 \leq \gamma \leq 2.5$, $-10 \leq c_x, c_y \leq 10$ |
| Box (shear) | $-0.5 \leq \gamma_1, \gamma_2 \leq 0.5$ |
| Box (source) | $0.001 \leq R_{\text{sersic}} \leq 10$, $0.5 \leq n_{\text{sersic}} \leq 5$, $-0.5 \leq e_1, e_2 \leq 0.5$ |
| Box (lens light) | $0.001 \leq R_{\text{sersic}} \leq 10$, $0.5 \leq n_{\text{sersic}} \leq 5$, $-0.5 \leq e_1, e_2 \leq 0.5$ |
| Box (point source) | $-10 \leq \text{ra}_k, \text{dec}_k \leq 10$ for $k = 1, \ldots, 4$ |
| Source-PS coupling | Source center coincides with point source origin in source plane |
| Image position tolerance | $\|\mathbf{x}_{\text{image}} - \mathbf{x}_{\text{predicted}}\|_2 \leq 0.004$ arcsec |

---

## 2. VARIABLES

### Primal Variables (Parameters to Optimize)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta_E$ | primal | scalar | float64 | 1.2 | Einstein radius (arcsec) |
| $\gamma$ | primal | scalar | float64 | 2.0 | Power-law slope of mass profile |
| $e_1^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens ellipticity component 2 |
| $c_x^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens center x-coordinate |
| $c_y^{\text{lens}}$ | primal | scalar | float64 | 0.0 | Lens center y-coordinate |
| $\gamma_1$ | primal | scalar | float64 | 0.0 | External shear component 1 |
| $\gamma_2$ | primal | scalar | float64 | 0.0 | External shear component 2 |
| $R_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 0.03 | Source Sérsic half-light radius |
| $n_{\text{sersic}}^{\text{src}}$ | primal | scalar | float64 | 1.0 | Source Sérsic index |
| $e_1^{\text{src}}, e_2^{\text{src}}$ | primal | (2,) | float64 | (0, 0) | Source ellipticity |
| $c_x^{\text{src}}, c_y^{\text{src}}$ | primal | (2,) | float64 | (0, 0) | Source center |
| $R_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 0.1 | Lens light Sérsic radius |
| $n_{\text{sersic}}^{\text{ll}}$ | primal | scalar | float64 | 1.0 | Lens light Sérsic index |
| $e_1^{\text{ll}}, e_2^{\text{ll}}$ | primal | (2,) | float64 | (0, 0) | Lens light ellipticity |
| $c_x^{\text{ll}}, c_y^{\text{ll}}$ | primal | (2,) | float64 | (0, 0) | Lens light center |
| $\mathbf{x}_{\text{img}}$ | primal | (4,) | float64 | $\mathbf{x}_{\text{true}} + 0.01$ | Image positions RA |
| $\mathbf{y}_{\text{img}}$ | primal | (4,) | float64 | $\mathbf{y}_{\text{true}} - 0.01$ | Image positions Dec |

### Dual Variables (PSO/MCMC Internal)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{p}_i$ | dual | $(N_{\text{params}},)$ | float64 | Sampled from $\mathcal{U}[\boldsymbol{\theta}_L, \boldsymbol{\theta}_U]$ | Particle position (PSO) |
| $\mathbf{v}_i$ | dual | $(N_{\text{params}},)$ | float64 | 0 | Particle velocity (PSO) |
| $\mathbf{p}_i^{\text{best}}$ | dual | $(N_{\text{params}},)$ | float64 | $\mathbf{p}_i$ | Personal best position |
| $\mathbf{g}^{\text{best}}$ | dual | $(N_{\text{params}},)$ | float64 | $\arg\min_i f(\mathbf{p}_i)$ | Global best position |
| $\mathbf{w}_j^{(t)}$ | dual | $(N_{\text{params}},)$ | float64 | Ball around PSO best | MCMC walker $j$ at iteration $t$ |

### Constants

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\sigma_{\text{bkg}}$ | constants | scalar | float64 | 0.5 | Background RMS noise (counts/s) |
| $t_{\text{exp}}$ | constants | scalar | float64 | 100 | Exposure time (seconds) |
| $N_x, N_y$ | constants | scalar | int | 100 | Image dimensions (pixels) |
| $\Delta_{\text{pix}}$ | constants | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\text{FWHM}$ | constants | scalar | float64 | 0.1 | PSF FWHM (arcsec) |
| $N_{\text{particles}}$ | constants | scalar | int | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | constants | scalar | int | 10 | PSO iterations |
| $N_{\text{walkers}}$ | constants | scalar | int | 50 | MCMC walkers |
| $N_{\text{burn}}$ | constants | scalar | int | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | constants | scalar | int | 10 | MCMC sampling steps |

### Observations

| Symbol | Category | Shape | Dtype | Description |
|--------|----------|-------|-------|-------------|
| $\mathbf{I}_{\text{obs}}$ | observations | $(N_x, N_y)$ | float64 | Observed (simulated) image |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Coordinate System Setup
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Coordinate Transformation Matrix |
| formula | $\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$ |
| inputs | $\Delta_{\text{pix}}$ (scalar) |
| outputs | $\mathbf{T} \in \mathbb{R}^{2 \times 2}$ |

### Step 3.2: Reference Point Calculation
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Image Center to World Coordinates |
| formula | $c_x = c_y = \frac{N_{\text{pix}} - 1}{2}$, $\begin{pmatrix} \text{RA}_0 \\ \text{Dec}_0 \end{pmatrix} = -\mathbf{T} \begin{pmatrix} c_x \\ c_y \end{pmatrix}$ |
| inputs | $N_{\text{pix}}$ (scalar), $\mathbf{T}$ (2×2) |
| outputs | $\text{RA}_0, \text{Dec}_0$ (scalars) |

### Step 3.3: PSF Construction
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Gaussian PSF Kernel |
| formula | $\text{PSF}(x, y) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{x^2 + y^2}{2\sigma_{\text{PSF}}^2}\right)$, where $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$ |
| inputs | FWHM (scalar), $\Delta_{\text{pix}}$ (scalar), truncation=5 |
| outputs | PSF kernel $\mathbf{K} \in \mathbb{R}^{N_k \times N_k}$ where $N_k = 2 \cdot \lfloor 5 \cdot \text{FWHM} / \Delta_{\text{pix}} \rfloor + 1$ |

### Step 3.4: Shear Conversion
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Polar to Cartesian Shear |
| formula | $\gamma_1 = \gamma \cos(2\phi)$, $\gamma_2 = \gamma \sin(2\phi)$ |
| inputs | $\gamma = 0.02$ (scalar), $\phi = 0.1$ (scalar) |
| outputs | $\gamma_1, \gamma_2$ (scalars) |

### Step 3.5: Lens Equation Solving
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Find Multiple Image Positions |
| formula | Solve $\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$ for $\boldsymbol{\theta}$ given source position $\boldsymbol{\beta} = (0, 0.1)$ |
| inputs | Source position $\boldsymbol{\beta}$ (2,), lens parameters, search window $= N_{\text{pix}} \cdot \Delta_{\text{pix}}$ |
| outputs | Image positions $\{\mathbf{x}_k, \mathbf{y}_k\}_{k=1}^{4}$, each (4,) |

### Step 3.6: Point Source Amplitudes
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Magnification-based Amplitudes |
| formula | $\mu_k = \left| \det\left(\frac{\partial \boldsymbol{\beta}}{\partial \boldsymbol{\theta}}\right)^{-1} \right|_{\boldsymbol{\theta}_k}$, $A_k = 100 \cdot (\mu_k + \epsilon_k)$ where $\epsilon_k \sim \mathcal{N}(0, 0.5)$ |
| inputs | Image positions (4,), lens model |
| outputs | Point amplitudes $\mathbf{A} \in \mathbb{R}^4$ |

### Step 3.7: Forward Model Image Generation
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Noise-free Image Synthesis |
| formula | $I_{\text{model}}(i,j) = \left[\mathbf{K} * \left( I_{\text{lens}}(\mathbf{r}) + I_{\text{source}}(\boldsymbol{\beta}(\mathbf{r})) + \sum_{k=1}^4 A_k \delta(\mathbf{r} - \mathbf{r}_k) \right)\right]_{i,j}$ |
| inputs | All model parameters, PSF kernel |
| outputs | $\mathbf{I}_{\text{model}} \in \mathbb{R}^{100 \times 100}$ |

### Step 3.8: Poisson Noise Addition
| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | Photon Counting Noise |
| formula | $C_{i,j} = \max(I_{\text{model}}(i,j) \cdot t_{\text{exp}}, 0)$, $\tilde{C}_{i,j} \sim \text{Poisson}(C_{i,j})$, $n_{\text{Poisson}}(i,j) = \frac{\tilde{C}_{i,j}}{t_{\text{exp}}} - I_{\text{model}}(i,j)$ |
| inputs | $\mathbf{I}_{\text{model}}$ (100×100), $t_{\text{exp}}$ (scalar) |
| outputs | $\mathbf{n}_{\text{Poisson}} \in \mathbb{R}^{100 \times 100}$ |

### Step 3.9: Background Noise Addition
| Field | Value |
|-------|-------|
| step_order | 9 |
| step_name | Gaussian Background Noise |
| formula | $n_{\text{bkg}}(i,j) \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$ |
| inputs | $\sigma_{\text{bkg}}$ (scalar), image shape |
| outputs | $\mathbf{n}_{\text{bkg}} \in \mathbb{R}^{100 \times 100}$ |

### Step 3.10: Observation Assembly
| Field | Value |
|-------|-------|
| step_order | 10 |
| step_name | Combine Model and Noise |
| formula | $\mathbf{I}_{\text{obs}} = \mathbf{I}_{\text{model}} + \mathbf{n}_{\text{bkg}} + \mathbf{n}_{\text{Poisson}}$ |
| inputs | $\mathbf{I}_{\text{model}}$, $\mathbf{n}_{\text{bkg}}$, $\mathbf{n}_{\text{Poisson}}$ (all 100×100) |
| outputs | $\mathbf{I}_{\text{obs}} \in \mathbb{R}^{100 \times 100}$ |

---

## 4. ALGORITHM STEPS

### Initialization

| Step | Field | Value |
|------|-------|-------|
| I.1 | step_name | Initialize Parameter Vector |
| | formula | $\boldsymbol{\theta}^{(0)} = [\theta_E^{(0)}, e_1^{(0)}, e_2^{(0)}, \gamma^{(0)}, c_x^{(0)}, c_y^{(0)}, \gamma_1^{(0)}, \gamma_2^{(0)}, \ldots]^T$ |
| | inputs | Initial kwargs dictionaries |
| | outputs | $\boldsymbol{\theta}^{(0)} \in \mathbb{R}^{N_{\text{params}}}$ |
| | computational_note | Parameters concatenated in fixed order: lens → source → lens_light → point_source |

| Step | Field | Value |
|------|-------|-------|
| I.2 | step_name | Initialize PSO Swarm |
| | formula | $\mathbf{p}_i^{(0)} = \boldsymbol{\theta}^{(0)} + \sigma_{\text{scale}} \cdot \boldsymbol{\sigma} \odot \mathbf{u}_i$, where $\mathbf{u}_i \sim \mathcal{U}[-1, 1]^{N_{\text{params}}}$ |
| | inputs | $\boldsymbol{\theta}^{(0)}$, $\boldsymbol{\sigma}$ (sigma params), $\sigma_{\text{scale}} = 1.0$ |
| | outputs | Particle positions $\{\mathbf{p}_i^{(0)}\}_{i=1}^{50}$ |
| | computational_note | $\odot$ denotes element-wise multiplication |

| Step | Field | Value |
|------|-------|-------|
| I.3 | step_name | Initialize Particle Velocities |
| | formula | $\mathbf{v}_i^{(0)} = \mathbf{0}$ |
| | inputs | None |
| | outputs | $\{\mathbf{v}_i^{(0)}\}_{i=1}^{50}$, each $\in \mathbb{R}^{N_{\text{params}}}$ |

| Step | Field | Value |
|------|-------|-------|
| I.4 | step_name | Evaluate Initial Fitness |
| | formula | $f_i^{(0)} = -\log \mathcal{L}(\mathbf{I}_{\text{obs}} | \mathbf{p}_i^{(0)})$ |
| | inputs | $\{\mathbf{p}_i^{(0)}\}$, $\mathbf{I}_{\text{obs}}$ |
| | outputs | $\{f_i^{(0)}\}_{i=1}^{50}$ |

| Step | Field | Value |
|------|-------|-------|
| I.5 | step_name | Initialize Personal and Global Bests |
| | formula | $\mathbf{p}_i^{\text{best}} = \mathbf{p}_i^{(0)}$, $f_i^{\text{best}} = f_i^{(0)}$, $\mathbf{g}^{\text{best}} = \mathbf{p}_{\arg\min_i f_i^{(0)}}$ |
| | inputs | $\{\mathbf{p}_i^{(0)}\}$, $\{f_i^{(0)}\}$ |
| | outputs | $\{\mathbf{p}_i^{\text{best}}\}$, $\mathbf{g}^{\text{best}}$ |

---

### Iteration Loop: PSO Phase

**BEGIN PSO_ITERATION_LOOP** (for $t = 1, \ldots, N_{\text{iter}}^{\text{PSO}}$)

| Step | Field | Value |
|------|-------|-------|
| P.1 | step_name | Update Particle Velocities |
| | formula | $\mathbf{v}_i^{(t)} = \omega \mathbf{v}_i^{(t-1)} + c_1 \mathbf{r}_1 \odot (\mathbf{p}_i^{\text{best}} - \mathbf{p}_i^{(t-1)}) + c_2 \mathbf{r}_2 \odot (\mathbf{g}^{\text{best}} - \mathbf{p}_i^{(t-1)})$ |
| | inputs | $\mathbf{v}_i^{(t-1)}$, $\mathbf{p}_i^{(t-1)}$, $\mathbf{p}_i^{\text{best}}$, $\mathbf{g}^{\text{best}}$; $\mathbf{r}_1, \mathbf{r}_2 \sim \mathcal{U}[0,1]^{N_{\text{params}}}$ |
| | outputs | $\mathbf{v}_i^{(t)}$ for all $i$ |
| | computational_note | $\omega \approx 0.7$ (inertia), $c_1 = c_2 \approx 1.5$ (cognitive/social coefficients) |

| Step | Field | Value |
|------|-------|-------|
| P.2 | step_name | Update Particle Positions |
| | formula | $\tilde{\mathbf{p}}_i^{(t)} = \mathbf{p}_i^{(t-1)} + \mathbf{v}_i^{(t)}$ |
| | inputs | $\mathbf{p}_i^{(t-1)}$, $\mathbf{v}_i^{(t)}$ |
| | outputs | $\tilde{\mathbf{p}}_i^{(t)}$ (unconstrained) |

| Step | Field | Value |
|------|-------|-------|
| P.3 | step_name | Apply Box Constraints |
| | formula | $\mathbf{p}_i^{(t)} = \max(\boldsymbol{\theta}_L, \min(\boldsymbol{\theta}_U, \tilde{\mathbf{p}}_i^{(t)}))$ |
| | inputs | $\tilde{\mathbf{p}}_i^{(t)}$, bounds $\boldsymbol{\theta}_L$, $\boldsymbol{\theta}_U$ |
| | outputs | $\mathbf{p}_i^{(t)}$ (feasible) |

| Step | Field | Value |
|------|-------|-------|
| P.4 | step_name | Evaluate Fitness |
| | formula | $f_i^{(t)} = \frac{1}{2}\chi^2(\mathbf{p}_i^{(t)}) - \log P(\mathbf{p}_i^{(t)})$ |
| | inputs | $\mathbf{p}_i^{(t)}$, $\mathbf{I}_{\text{obs}}$ |
| | outputs | $f_i^{(t)}$ |
| | computational_note | Returns $+\infty$ if constraints violated |

| Step | Field | Value |
|------|-------|-------|
| P.5 | step_name | Update Personal Bests |
| | formula | If $f_i^{(t)} < f_i^{\text{best}}$: $\mathbf{p}_i^{\text{best}} \leftarrow \mathbf{p}_i^{(t)}$, $f_i^{\text{best}} \leftarrow f_i^{(t)}$ |
| | inputs | $f_i^{(t)}$, $f_i^{\text{best}}$, $\mathbf{p}_i^{(t)}$ |
| | outputs | Updated $\mathbf{p}_i^{\text{best}}$, $f_i^{\text{best}}$ |

| Step | Field | Value |
|------|-------|-------|
| P.6 | step_name | Update Global Best |
| | formula | $\mathbf{g}^{\text{best}} \leftarrow \mathbf{p}_{\arg\min_i f_i^{\text{best}}}$ |
| | inputs | All $\{\mathbf{p}_i^{\text{best}}, f_i^{\text{best}}\}$ |
| | outputs | Updated $\mathbf{g}^{\text{best}}$ |

| Step | Field | Value |
|------|-------|-------|
| P.7 | step_name | Check PSO Termination |
| | formula | Continue if $t < N_{\text{iter}}^{\text{PSO}}$ |
| | inputs | Current iteration $t$ |
| | outputs | Boolean: continue/stop |

**END PSO_ITERATION_LOOP**

---

### Iteration Loop: MCMC Phase

| Step | Field | Value |
|------|-------|-------|
| M.0 | step_name | Initialize MCMC Walkers |
| | formula | $\mathbf{w}_j^{(0)} = \mathbf{g}^{\text{best}} + \sigma_{\text{scale}}^{\text{MCMC}} \cdot \boldsymbol{\sigma} \odot \mathbf{z}_j$, where $\mathbf{z}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| | inputs | $\mathbf{g}^{\text{best}}$ from PSO, $\sigma_{\text{scale}}^{\text{MCMC}} = 0.1$ |
| | outputs | $\{\mathbf{w}_j^{(0)}\}_{j=1}^{50}$ |

**BEGIN MCMC_ITERATION_LOOP** (for $t = 1, \ldots, N_{\text{burn}} + N_{\text{run}}$)

| Step | Field | Value |
|------|-------|-------|
| M.1 | step_name | Propose New Positions (Stretch Move) |
| | formula | $\mathbf{w}_j^* = \mathbf{w}_k^{(t-1)} + Z \cdot (\mathbf{w}_j^{(t-1)} - \mathbf{w}_k^{(t-1)})$, where $k \neq j$ drawn uniformly, $Z \sim g(z) \propto \frac{1}{\sqrt{z}}$ for $z \in [\frac{1}{a}, a]$, $a = 2$ |
| | inputs | $\{\mathbf{w}_j^{(t-1)}\}$ |
| | outputs | Proposals $\{\mathbf{w}_j^*\}$ |
| | computational_note | Affine-invariant ensemble sampler (emcee) |

| Step | Field | Value |
|------|-------|-------|
| M.2 | step_name | Compute Acceptance Probability |
| | formula | $\alpha_j = \min\left(1, Z^{N_{\text{params}}-1} \cdot \frac{\mathcal{L}(\mathbf{I}_{\text{obs}}|\mathbf{w}_j^*) P(\mathbf{w}_j^*)}{\mathcal{L}(\mathbf{I}_{\text{obs}}|\mathbf{w}_j^{(t-1)}) P(\mathbf{w}_j^{(t-1)})}\right)$ |
| | inputs | $\mathbf{w}_j^*$, $\mathbf{w}_j^{(t-1)}$, stretch factor $Z$ |
| | outputs | $\alpha_j \in [0, 1]$ |

| Step | Field | Value |
|------|-------|-------|
| M.3 | step_name | Accept/Reject Proposals |
| | formula | $\mathbf{w}_j^{(t)} = \begin{cases} \mathbf{w}_j^* & \text{if } u_j < \alpha_j \\ \mathbf{w}_j^{(t-1)} & \text{otherwise} \end{cases}$, where $u_j \sim \mathcal{U}[0,1]$ |
| | inputs | $\alpha_j$, $\mathbf{w}_j^*$, $\mathbf{w}_j^{(t-1)}$ |
| | outputs | $\mathbf{w}_j^