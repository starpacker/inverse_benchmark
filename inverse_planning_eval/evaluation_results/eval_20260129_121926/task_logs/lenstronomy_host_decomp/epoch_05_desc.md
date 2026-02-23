## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Particle Swarm Optimization for Quasar-Host Galaxy Image Decomposition
- **algorithm_family**: Metaheuristic Global Optimization (Swarm Intelligence)
- **summary**: PSO-based fitting of a parametric astronomical image model comprising a point source (quasar) and two Sérsic profile components (disk and bulge) to noisy observation data, with linear amplitude optimization at each fitness evaluation.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta} \in \Omega} \chi^2(\boldsymbol{\theta}) \quad \text{subject to} \quad \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$$

where $\Omega$ includes joint center constraints coupling disk, bulge, and point source positions.

### data_fidelity_term
$$\chi^2(\boldsymbol{\theta}) = \sum_{i=1}^{N_{\text{pix}}} \frac{\left(y_i^{\text{obs}} - y_i^{\text{model}}(\boldsymbol{\theta}, \mathbf{a}^*(\boldsymbol{\theta}))\right)^2}{\sigma_i^2}$$

where:
- $y_i^{\text{obs}}$: observed pixel value at pixel $i$
- $y_i^{\text{model}}$: forward model prediction
- $\sigma_i^2 = \sigma_{\text{bkg}}^2 + \max(y_i^{\text{obs}}, 0) / t_{\text{exp}}$ (combined background and Poisson variance)
- $\mathbf{a}^*(\boldsymbol{\theta})$: linear amplitudes solved via least-squares given shape parameters $\boldsymbol{\theta}$

### regularization_term
None (pure maximum likelihood estimation)

### constraints
$$\mathbb{I}_{\Omega}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta} \in \Omega \\ +\infty & \text{otherwise} \end{cases}$$

Box constraints:
- $-0.5 \leq e_1^{\text{disk}}, e_2^{\text{disk}} \leq 0.5$
- $0.001 \leq R_{\text{sersic}}^{\text{disk}} \leq 10$
- $0.001 \leq R_{\text{sersic}}^{\text{bulge}} \leq 10$
- $-10 \leq c_x, c_y \leq 10$

Equality constraints (joint centers):
- $c_x^{\text{disk}} = c_x^{\text{bulge}} = c_x^{\text{ps}}$
- $c_y^{\text{disk}} = c_y^{\text{bulge}} = c_y^{\text{ps}}$

---

## 2. VARIABLES

### 2.1 Primal Variables (Nonlinear Shape Parameters)

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $R_{\text{sersic}}^{\text{disk}}$ | scalar | float64 | 1.0 | Disk half-light radius (arcsec) |
| $e_1^{\text{disk}}$ | scalar | float64 | 0.0 | Disk ellipticity component 1 |
| $e_2^{\text{disk}}$ | scalar | float64 | 0.0 | Disk ellipticity component 2 |
| $R_{\text{sersic}}^{\text{bulge}}$ | scalar | float64 | 0.5 | Bulge half-light radius (arcsec) |
| $c_x$ | scalar | float64 | 0.0 | Shared center x-coordinate (arcsec) |
| $c_y$ | scalar | float64 | 0.0 | Shared center y-coordinate (arcsec) |

### 2.2 Dual Variables
None (PSO is primal-only)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}^{(k)}$ | $(N_p, d)$ | float64 | Gaussian around init | Particle positions at iteration $k$ |
| $\mathbf{V}^{(k)}$ | $(N_p, d)$ | float64 | $\mathcal{U}(-\boldsymbol{\sigma}, \boldsymbol{\sigma})$ | Particle velocities at iteration $k$ |
| $\mathbf{P}_{\text{best}}$ | $(N_p, d)$ | float64 | $\mathbf{X}^{(0)}$ | Personal best positions |
| $f_{\text{pbest}}$ | $(N_p,)$ | float64 | $\chi^2(\mathbf{X}_j^{(0)})$ | Personal best fitness values |
| $\mathbf{g}_{\text{best}}$ | $(d,)$ | float64 | $\arg\min_j f_{\text{pbest},j}$ | Global best position |
| $f_{\text{gbest}}$ | scalar | float64 | $\min_j f_{\text{pbest},j}$ | Global best fitness |
| $\mathbf{a}^*$ | $(3,)$ | float64 | Solved per evaluation | Linear amplitudes $[A_{\text{disk}}, A_{\text{bulge}}, A_{\text{ps}}]$ |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_p$ | 50 | Number of PSO particles |
| $K_{\max}$ | 50 | Maximum iterations |
| $\sigma_{\text{scale}}$ | 1.0 | Initial velocity scale factor |
| $\omega$ | 0.5 (typical) | Inertia weight |
| $c_1$ | 1.5 (typical) | Cognitive coefficient |
| $c_2$ | 1.5 (typical) | Social coefficient |
| $N_{\text{pix}}$ | 80 | Image dimension (pixels per side) |
| $\Delta_{\text{pix}}$ | 0.05 | Pixel scale (arcsec/pixel) |
| $\sigma_{\text{bkg}}$ | 0.1 | Background RMS noise (counts/s) |
| $t_{\text{exp}}$ | 100.0 | Exposure time (seconds) |
| FWHM | 0.1 | PSF full-width half-maximum (arcsec) |
| $n_{\text{disk}}$ | 1 | Disk Sérsic index (fixed) |
| $n_{\text{bulge}}$ | 4 | Bulge Sérsic index (fixed) |
| $\tau_{\text{PSF}}$ | 3 | PSF truncation radius (in FWHM units) |
| $d$ | 6 | Dimensionality of nonlinear parameter space |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{Y}^{\text{obs}}$ | $(80, 80)$ | float64 | Observed image with noise |
| $\mathbf{K}$ | $(K_{\text{size}}, K_{\text{size}})$ | float64 | PSF kernel |

---

## 3. PREPROCESSING PIPELINE

**Step P1: Coordinate System Setup**
- **step_order**: P1
- **step_name**: Compute pixel-to-angular transformation
- **formula**: 
$$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$
$$c_x^{\text{pix}} = c_y^{\text{pix}} = \frac{N_{\text{pix}} - 1}{2} = 39.5$$
$$\text{RA}_0 = -(\mathbf{T}_{00} \cdot c_x^{\text{pix}} + \mathbf{T}_{01} \cdot c_y^{\text{pix}}) = 1.975$$
$$\text{Dec}_0 = -(\mathbf{T}_{10} \cdot c_x^{\text{pix}} + \mathbf{T}_{11} \cdot c_y^{\text{pix}}) = -1.975$$
- **inputs**: $N_{\text{pix}}$ (scalar), $\Delta_{\text{pix}}$ (scalar)
- **outputs**: $\mathbf{T}$ (2×2), $\text{RA}_0$ (scalar), $\text{Dec}_0$ (scalar)

**Step P2: PSF Kernel Generation**
- **step_order**: P2
- **step_name**: Generate Gaussian PSF kernel
- **formula**:
$$\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} = \frac{0.1}{2.3548} \approx 0.04247 \text{ arcsec}$$
$$K_{\text{size}} = 2 \cdot \lfloor \tau_{\text{PSF}} \cdot \text{FWHM} / \Delta_{\text{pix}} \rfloor + 1 = 2 \cdot \lfloor 6 \rfloor + 1 = 13$$
$$\mathbf{K}_{ij} = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{(x_i^2 + y_j^2)}{2\sigma_{\text{PSF}}^2}\right)$$
where $x_i, y_j$ are angular offsets from kernel center
- **inputs**: FWHM (scalar), $\Delta_{\text{pix}}$ (scalar), $\tau_{\text{PSF}}$ (scalar)
- **outputs**: $\mathbf{K}$ (13×13), $\sigma_{\text{PSF}}$ (scalar)

**Step P3: Simulate Clean Image**
- **step_order**: P3
- **step_name**: Generate ground-truth model image
- **formula**:
$$\mathbf{Y}^{\text{clean}} = \mathbf{K} * \mathbf{I}_{\text{disk}}(\boldsymbol{\theta}^{\text{true}}) + \mathbf{K} * \mathbf{I}_{\text{bulge}}(\boldsymbol{\theta}^{\text{true}}) + \mathbf{K} * \mathbf{I}_{\text{ps}}(\boldsymbol{\theta}^{\text{true}})$$
where $*$ denotes 2D convolution
- **inputs**: $\boldsymbol{\theta}^{\text{true}}$ (parameter dict), $\mathbf{K}$ (13×13)
- **outputs**: $\mathbf{Y}^{\text{clean}}$ (80×80)

**Step P4: Add Poisson Noise**
- **step_order**: P4
- **step_name**: Simulate photon counting noise
- **formula**:
$$\mathbf{C} = \mathbf{Y}^{\text{clean}} \cdot t_{\text{exp}}$$
$$\mathbf{C}^+ = \max(\mathbf{C}, 0)$$
$$\tilde{\mathbf{C}} \sim \text{Poisson}(\mathbf{C}^+)$$
$$\boldsymbol{\eta}_{\text{Poisson}} = \frac{\tilde{\mathbf{C}}}{t_{\text{exp}}} - \mathbf{Y}^{\text{clean}}$$
- **inputs**: $\mathbf{Y}^{\text{clean}}$ (80×80), $t_{\text{exp}}$ (scalar)
- **outputs**: $\boldsymbol{\eta}_{\text{Poisson}}$ (80×80)

**Step P5: Add Background Noise**
- **step_order**: P5
- **step_name**: Add Gaussian background noise
- **formula**:
$$\boldsymbol{\eta}_{\text{bkg}} = \sigma_{\text{bkg}} \cdot \mathbf{Z}, \quad \mathbf{Z}_{ij} \sim \mathcal{N}(0, 1)$$
- **inputs**: $\sigma_{\text{bkg}}$ (scalar)
- **outputs**: $\boldsymbol{\eta}_{\text{bkg}}$ (80×80)

**Step P6: Combine to Form Observation**
- **step_order**: P6
- **step_name**: Generate final observed image
- **formula**:
$$\mathbf{Y}^{\text{obs}} = \mathbf{Y}^{\text{clean}} + \boldsymbol{\eta}_{\text{bkg}} + \boldsymbol{\eta}_{\text{Poisson}}$$
- **inputs**: $\mathbf{Y}^{\text{clean}}$, $\boldsymbol{\eta}_{\text{bkg}}$, $\boldsymbol{\eta}_{\text{Poisson}}$ (all 80×80)
- **outputs**: $\mathbf{Y}^{\text{obs}}$ (80×80)

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

**Step 1: Define Parameter Search Bounds**
- **step_order**: 1
- **step_name**: Set box constraints
- **formula**:
$$\boldsymbol{\theta}_{\text{lower}} = [0.001, -0.5, -0.5, 0.001, -10, -10]^T$$
$$\boldsymbol{\theta}_{\text{upper}} = [10, 0.5, 0.5, 10, 10, 10]^T$$
(ordering: $[R_{\text{disk}}, e_1, e_2, R_{\text{bulge}}, c_x, c_y]$)
- **inputs**: None
- **outputs**: $\boldsymbol{\theta}_{\text{lower}}$ (6,), $\boldsymbol{\theta}_{\text{upper}}$ (6,)

**Step 2: Define Initial Guesses and Spread**
- **step_order**: 2
- **step_name**: Set initial parameter values and sampling widths
- **formula**:
$$\boldsymbol{\theta}_{\text{init}} = [1.0, 0.0, 0.0, 0.5, 0.0, 0.0]^T$$
$$\boldsymbol{\sigma}_{\theta} = [0.5, 0.1, 0.1, 0.3, 0.1, 0.1]^T$$
- **inputs**: None
- **outputs**: $\boldsymbol{\theta}_{\text{init}}$ (6,), $\boldsymbol{\sigma}_{\theta}$ (6,)

**Step 3: Initialize Particle Positions**
- **step_order**: 3
- **step_name**: Sample initial swarm positions
- **formula**:
$$\mathbf{X}_j^{(0)} = \boldsymbol{\theta}_{\text{init}} + \sigma_{\text{scale}} \cdot \boldsymbol{\sigma}_{\theta} \odot \mathbf{z}_j, \quad \mathbf{z}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$
$$\mathbf{X}_j^{(0)} \leftarrow \text{clip}\left(\mathbf{X}_j^{(0)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}\right) \quad \forall j \in \{1, \ldots, N_p\}$$
- **inputs**: $\boldsymbol{\theta}_{\text{init}}$, $\boldsymbol{\sigma}_{\theta}$, $\sigma_{\text{scale}}$, $N_p$, bounds
- **outputs**: $\mathbf{X}^{(0)}$ (50×6)
- **computational_note**: $\odot$ denotes element-wise multiplication

**Step 4: Initialize Particle Velocities**
- **step_order**: 4
- **step_name**: Sample initial velocities
- **formula**:
$$\mathbf{V}_j^{(0)} = \sigma_{\text{scale}} \cdot \boldsymbol{\sigma}_{\theta} \odot \mathbf{u}_j, \quad \mathbf{u}_j \sim \mathcal{U}(-1, 1)^d$$
- **inputs**: $\boldsymbol{\sigma}_{\theta}$, $\sigma_{\text{scale}}$, $N_p$
- **outputs**: $\mathbf{V}^{(0)}$ (50×6)

**Step 5: Evaluate Initial Fitness**
- **step_order**: 5
- **step_name**: Compute χ² for all particles
- **formula**:
$$f_j^{(0)} = \chi^2(\mathbf{X}_j^{(0)}) \quad \forall j \in \{1, \ldots, N_p\}$$
(See Step 8 for $\chi^2$ computation details)
- **inputs**: $\mathbf{X}^{(0)}$, $\mathbf{Y}^{\text{obs}}$, $\mathbf{K}$
- **outputs**: $\mathbf{f}^{(0)}$ (50,)

**Step 6: Initialize Personal and Global Bests**
- **step_order**: 6
- **step_name**: Set initial best records
- **formula**:
$$\mathbf{P}_{\text{best}} = \mathbf{X}^{(0)}$$
$$f_{\text{pbest}} = \mathbf{f}^{(0)}$$
$$j^* = \arg\min_j f_j^{(0)}$$
$$\mathbf{g}_{\text{best}} = \mathbf{X}_{j^*}^{(0)}$$
$$f_{\text{gbest}} = f_{j^*}^{(0)}$$
- **inputs**: $\mathbf{X}^{(0)}$, $\mathbf{f}^{(0)}$
- **outputs**: $\mathbf{P}_{\text{best}}$ (50×6), $f_{\text{pbest}}$ (50,), $\mathbf{g}_{\text{best}}$ (6,), $f_{\text{gbest}}$ (scalar)

**Step 7: Initialize Iteration Counter**
- **step_order**: 7
- **step_name**: Set counter
- **formula**: $k \leftarrow 0$
- **inputs**: None
- **outputs**: $k$ (scalar)

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (iterate while $k < K_{\max}$)

**Step 8: Compute Forward Model for Each Particle**
- **step_order**: 8
- **step_name**: Evaluate Sérsic profiles and point source [FORWARD]
- **formula**:

For particle $j$ with parameters $\boldsymbol{\theta}_j = [R_d, e_1, e_2, R_b, c_x, c_y]$:

*Coordinate transformation to elliptical radius (disk):*
$$\phi = \frac{1}{2}\arctan\left(\frac{2 e_2}{1 - e_1^2 - e_2^2 + \sqrt{(1-e_1^2-e_2^2)^2 + 4e_2^2}}\right)$$
$$q = \frac{1 - \sqrt{e_1^2 + e_2^2}}{1 + \sqrt{e_1^2 + e_2^2}}$$

*For each pixel $(m, n)$:*
$$x_{mn} = \text{RA}_0 + \mathbf{T}_{00} \cdot m + \mathbf{T}_{01} \cdot n$$
$$y_{mn} = \text{Dec}_0 + \mathbf{T}_{10} \cdot m + \mathbf{T}_{11} \cdot n$$
$$x' = (x_{mn} - c_x)\cos\phi + (y_{mn} - c_y)\sin\phi$$
$$y' = -(x_{mn} - c_x)\sin\phi + (y_{mn} - c_y)\cos\phi$$
$$r_{\text{disk}} = \sqrt{q \cdot (x')^2 + (y')^2 / q}$$
$$r_{\text{bulge}} = \sqrt{(x_{mn} - c_x)^2 + (y_{mn} - c_y)^2}$$

*Sérsic profiles (unnormalized):*
$$b_n = 1.9992n - 0.3271 \quad \text{(approximation for } 0.5 < n < 10\text{)}$$
$$[\mathbf{S}_{\text{disk}}]_{mn} = \exp\left(-b_1 \left[\left(\frac{r_{\text{disk}}}{R_d}\right)^{1/1} - 1\right]\right), \quad b_1 \approx 1.6721$$
$$[\mathbf{S}_{\text{bulge}}]_{mn} = \exp\left(-b_4 \left[\left(\frac{r_{\text{bulge}}}{R_b}\right)^{1/4} - 1\right]\right), \quad b_4 \approx 7.6697$$

*Point source (delta function at $c_x, c_y$):*
$$[\mathbf{S}_{\text{ps}}]_{mn} = \delta(x_{mn} - c_x)\delta(y_{mn} - c_y) \quad \text{(implemented as single-pixel injection)}$$

*Convolved basis images:*
$$\mathbf{B}_1 = \mathbf{K} * \mathbf{S}_{\text{disk}}$$
$$\mathbf{B}_2 = \mathbf{K} * \mathbf{S}_{\text{bulge}}$$
$$\mathbf{B}_3 = \mathbf{K} * \mathbf{S}_{\text{ps}}$$
- **inputs**: $\mathbf{X}_j^{(k)}$, $\mathbf{K}$, coordinate grid
- **outputs**: $\mathbf{B}_1, \mathbf{B}_2, \mathbf{B}_3$ (each 80×80) for each particle

**Step 9: Solve Linear Amplitudes**
- **step_order**: 9
- **step_name**: Least-squares amplitude optimization
- **formula**:

Flatten images to vectors and form design matrix:
$$\mathbf{M} = [\text{vec}(\mathbf{B}_1) \;|\; \text{vec}(\mathbf{B}_2) \;|\; \text{vec}(\mathbf{B}_3)] \in \mathbb{R}^{6400 \times 3}$$
$$\mathbf{y} = \text{vec}(\mathbf{Y}^{\text{obs}}) \in \mathbb{R}^{6400}$$
$$\boldsymbol{\Sigma}^{-1} = \text{diag}\left(\frac{1}{\sigma_{\text{bkg}}^2 + \max(y_i, 0)/t_{\text{exp}}}\right)$$

Solve weighted least squares:
$$\mathbf{a}^* = (\mathbf{M}^T \boldsymbol{\Sigma}^{-1} \mathbf{M})^{-1} \mathbf{M}^T \boldsymbol{\Sigma}^{-1} \mathbf{y}$$
- **inputs**: $\mathbf{B}_1, \mathbf{B}_2, \mathbf{B}_3$, $\mathbf{Y}^{\text{obs}}$, $\sigma_{\text{bkg}}$, $t_{\text{exp}}$
- **outputs**: $\mathbf{a}^* = [A_{\text{disk}}^*, A_{\text{bulge}}^*, A_{\text{ps}}^*]^T$ (3,)

**Step 10: Compute Model Image**
- **step_order**: 10
- **step_name**: Reconstruct model prediction
- **formula**:
$$\mathbf{Y}_j^{\text{model}} = A_{\text{disk}}^* \mathbf{B}_1 + A_{\text{bulge}}^* \mathbf{B}_2 + A_{\text{ps}}^* \mathbf{B}_3$$
- **inputs**: $\mathbf{B}_1, \mathbf{B}_2, \mathbf{B}_3$, $\mathbf{a}^*$
- **outputs**: $\mathbf{Y}_j^{\text{model}}$ (80×80)

**Step 11: Compute Fitness (Chi-Squared)**
- **step_order**: 11
- **step_name**: Evaluate objective function
- **formula**:
$$\sigma_{mn}^2 = \sigma_{\text{bkg}}^2 + \frac{\max(Y_{mn}^{\text{obs}}, 0)}{t_{\text{exp}}}$$
$$f_j^{(k)} = \chi^2_j = \sum_{m,n} \frac{\left(Y_{mn}^{\text{obs}} - Y_{mn,j}^{\text{model}}\right)^2}{\sigma_{mn}^2}$$
- **inputs**: $\mathbf{Y}^{\text{obs}}$, $\mathbf{Y}_j^{\text{model}}$, $\sigma_{\text{bkg}}$, $t_{\text{exp}}$
- **outputs**: $f_j^{(k)}$ (scalar per particle)
- **computational_note**: Steps 8-11 are repeated for all $j \in \{1, \ldots, N_p\}$

**Step 12: Update Personal Bests**
- **step_order**: 12
- **step_name**: Compare with historical best
- **formula**:
$$\text{For } j = 1, \ldots, N_p:$$
$$\text{If } f_j^{(k)} < f_{\text{pbest},j}: \quad \mathbf{P}_{\text{best},j} \leftarrow \mathbf{X}_j^{(k)}, \quad f_{\text{pbest},j} \leftarrow f_j^{(k)}$$
- **inputs**: $\mathbf{X}^{(k)}$, $\mathbf{f}^{(k)}$, $\mathbf{P}_{\text{best}}$, $f_{\text{pbest}}$
- **outputs**: $\mathbf{P}_{\text{best}}$ (50×6), $f_{\text{pbest}}$ (50,) [updated]

**Step 13: Update Global Best**
- **step_order**: 13
- **step_name**: Find swarm-wide optimum
- **formula**:
$$j^* = \arg\min_j f_{\text{pbest},j}$$
$$\text{If } f_{\text{pbest},j^*} < f_{\text{gbest}}: \quad \mathbf{g}_{\text{best}} \leftarrow \mathbf{P}_{\text{best},j^*}, \quad f_{\text{gbest}} \leftarrow f_{\text{pbest},j^*}$$
- **inputs**: $\mathbf{P}_{\text{best}}$, $f_{\text{pbest}}$, $\mathbf{g}_{\text{best}}$, $f_{\text{gbest}}$
- **outputs**: $\mathbf{g}_{\text{best}}$ (6,), $f_{\text{gbest}}$ (scalar) [updated]

**Step 14: Generate Random Coefficients**
- **step_order**: 14
- **step_name**: Sample stochastic weights
- **formula**:
$$r_{1,j} \sim \mathcal{U}(0, 1), \quad r_{2,j} \sim \mathcal{U}(0, 1) \quad \forall j$$
- **inputs**: $N_p$
- **outputs**: $\mathbf{r}_1, \mathbf{r}_2$ (each 50,)

**Step 15: Update Velocities**
- **step_order**: 15
- **step_name**: PSO velocity equation
- **formula**:
$$\mathbf{V}_j^{(k+1)} = \omega \mathbf{V}_j^{(k)} + c_1 r