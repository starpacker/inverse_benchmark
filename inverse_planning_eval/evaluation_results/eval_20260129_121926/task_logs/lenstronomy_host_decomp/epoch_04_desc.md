## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: PSO-based Astronomical Image Model Fitting
- **algorithm_family**: Metaheuristic Optimization / Particle Swarm Optimization
- **summary**: Fits a parametric astrophysical image model (point source + Sérsic galaxy profiles) to observed data by minimizing a chi-squared objective using Particle Swarm Optimization with linear amplitude solving.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\boldsymbol{\theta}_{\text{shape}}} \left[ \min_{\boldsymbol{a}} \frac{1}{2} \sum_{i=1}^{N_{\text{pix}}} \frac{(y_i - [\mathbf{M}(\boldsymbol{\theta}_{\text{shape}}) \boldsymbol{a}]_i)^2}{\sigma_i^2} \right] + \mathbb{I}_{\mathcal{C}}(\boldsymbol{\theta}_{\text{shape}})$$

where the inner minimization over amplitudes $\boldsymbol{a}$ is solved analytically via linear least squares.

### data_fidelity_term
$$\chi^2(\boldsymbol{\theta}) = \sum_{i=1}^{N_{\text{pix}}} \frac{(y_i - \hat{y}_i(\boldsymbol{\theta}))^2}{\sigma_i^2}$$

where:
- $y_i$ is the observed pixel value at pixel $i$
- $\hat{y}_i(\boldsymbol{\theta}) = [\mathbf{H} * (I_{\text{disk}} + I_{\text{bulge}} + I_{\text{PS}})]_i$ is the predicted pixel value
- $\sigma_i^2 = \sigma_{\text{bkg}}^2 + y_i / t_{\text{exp}}$ (combined background and Poisson variance)
- $\mathbf{H}$ is the PSF convolution operator
- The noise model combines Gaussian background (L2) with Poisson statistics

### regularization_term
None (no explicit regularization; constraints enforced via indicator function)

### constraints
$$\mathbb{I}_{\mathcal{C}}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta} \in \mathcal{C} \\ +\infty & \text{otherwise} \end{cases}$$

Box constraints $\mathcal{C}$:
- Disk: $e_1, e_2 \in [-0.5, 0.5]$, $R_{\text{sersic}} \in [0.001, 10]$, $n_{\text{sersic}} \in [0.5, 5]$, $c_x, c_y \in [-10, 10]$
- Bulge: $R_{\text{sersic}} \in [0.001, 10]$, $n_{\text{sersic}} \in [0.5, 5]$, $c_x, c_y \in [-10, 10]$
- Point Source: $\alpha, \delta \in [-10, 10]$

Joint constraints:
- $c_x^{\text{disk}} = c_x^{\text{bulge}}$, $c_y^{\text{disk}} = c_y^{\text{bulge}}$
- $\alpha^{\text{PS}} = c_x^{\text{disk}}$, $\delta^{\text{PS}} = c_y^{\text{disk}}$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\theta}^{(k)}_p$ | $(N_{\text{particles}}, N_{\text{params}})$ | float64 | Sampled from $[\boldsymbol{\theta}_{\text{init}} - \boldsymbol{\sigma}, \boldsymbol{\theta}_{\text{init}} + \boldsymbol{\sigma}]$ | Particle positions (parameter vectors) |
| $\boldsymbol{v}^{(k)}_p$ | $(N_{\text{particles}}, N_{\text{params}})$ | float64 | Random within velocity bounds | Particle velocities |
| $\boldsymbol{a}$ | $(N_{\text{components}},) = (3,)$ | float64 | Linear solve | Amplitude coefficients [disk, bulge, point source] |

**Shape parameter vector $\boldsymbol{\theta}_{\text{shape}}$ composition:**

| Component | Parameters | Count |
|-----------|------------|-------|
| Disk (SERSIC_ELLIPSE) | $R_{\text{sersic}}, e_1, e_2, c_x, c_y$ (n_sersic=1 fixed) | 5 |
| Bulge (SERSIC) | $R_{\text{sersic}}, c_x, c_y$ (n_sersic=4 fixed) | 3 |
| Point Source | $\alpha, \delta$ (amplitude solved linearly) | 2 |
| **Total free** (after joint constraints) | $R_{\text{sersic}}^{\text{disk}}, e_1, e_2, R_{\text{sersic}}^{\text{bulge}}, c_x, c_y$ | 6 |

### 2.2 Dual Variables

None (PSO is primal-only metaheuristic)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\theta}^{\text{pbest}}_p$ | $(N_{\text{particles}}, N_{\text{params}})$ | float64 | $\boldsymbol{\theta}^{(0)}_p$ | Personal best position for each particle |
| $f^{\text{pbest}}_p$ | $(N_{\text{particles}},)$ | float64 | $\chi^2(\boldsymbol{\theta}^{(0)}_p)$ | Personal best fitness for each particle |
| $\boldsymbol{\theta}^{\text{gbest}}$ | $(N_{\text{params}},)$ | float64 | $\arg\min_p f^{\text{pbest}}_p$ | Global best position |
| $f^{\text{gbest}}$ | scalar | float64 | $\min_p f^{\text{pbest}}_p$ | Global best fitness |
| $\mathbf{M}$ | $(N_{\text{pix}}, N_{\text{components}})$ | float64 | Computed per evaluation | Design matrix for linear amplitude solve |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{\text{particles}}$ | 50 | Number of PSO particles |
| $N_{\text{iterations}}$ | 50 | Maximum PSO iterations |
| $N_{\text{pix}}$ | 6400 (80×80) | Total number of pixels |
| $\Delta_{\text{pix}}$ | 0.05 arcsec | Pixel scale |
| $\sigma_{\text{bkg}}$ | 0.1 | Background RMS noise |
| $t_{\text{exp}}$ | 100.0 s | Exposure time |
| FWHM | 0.1 arcsec | PSF full width at half maximum |
| $\sigma_{\text{PSF}}$ | $\text{FWHM}/(2\sqrt{2\ln 2}) \approx 0.0425$ | PSF Gaussian standard deviation |
| truncation | 3 | PSF truncation in units of FWHM |
| $\sigma_{\text{scale}}$ | 1.0 | PSO sigma scale factor |
| $n_{\text{sersic}}^{\text{disk}}$ | 1 (fixed) | Disk Sérsic index |
| $n_{\text{sersic}}^{\text{bulge}}$ | 4 (fixed) | Bulge Sérsic index |
| $w$ | 0.7298 (lenstronomy default) | PSO inertia weight |
| $c_1$ | 1.49618 (lenstronomy default) | PSO cognitive coefficient |
| $c_2$ | 1.49618 (lenstronomy default) | PSO social coefficient |
| random_seed | 42 | NumPy random seed |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(80, 80)$ | float64 | Observed image (noisy simulated data) |
| $\mathbf{H}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | PSF convolution matrix (implicit via FFT) |

---

## 3. PREPROCESSING PIPELINE

**Step 0: Coordinate System Setup**
- **formula**: 
$$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}, \quad c_x^{\text{ref}} = c_y^{\text{ref}} = \frac{N_{\text{pix}} - 1}{2} = 39.5$$
$$\alpha_0 = -(\mathbf{T}_{00} c_x^{\text{ref}} + \mathbf{T}_{01} c_y^{\text{ref}}) = 1.975, \quad \delta_0 = -(\mathbf{T}_{10} c_x^{\text{ref}} + \mathbf{T}_{11} c_y^{\text{ref}}) = -1.975$$
- **inputs**: $N_{\text{pix}}=80$, $\Delta_{\text{pix}}=0.05$
- **outputs**: $\mathbf{T} \in \mathbb{R}^{2 \times 2}$, $(\alpha_0, \delta_0) \in \mathbb{R}^2$

**Step 1: PSF Construction**
- **formula**: 
$$P(r) = \exp\left(-\frac{r^2}{2\sigma_{\text{PSF}}^2}\right), \quad \sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$$
- **inputs**: FWHM=0.1, $\Delta_{\text{pix}}$=0.05, truncation=3
- **outputs**: Gaussian PSF kernel (implicit convolution operator $\mathbf{H}$)

**Step 2: Ellipticity Conversion**
- **formula**: 
$$e_1 = \frac{1-q}{1+q}\cos(2\phi), \quad e_2 = \frac{1-q}{1+q}\sin(2\phi)$$
where $\phi = 0.3$ rad, $q = 0.6$
- **inputs**: $\phi=0.3$, $q=0.6$
- **outputs**: $e_1, e_2$ (ellipticity components for disk)

**Step 3: Ground Truth Model Construction**
- **formula**: 
$$I_{\text{disk}}(x,y) = A_{\text{disk}} \exp\left(-b_1 \left[\left(\frac{r_{\text{eff}}}{R_{\text{sersic}}^{\text{disk}}}\right)^{1/1} - 1\right]\right)$$
$$I_{\text{bulge}}(x,y) = A_{\text{bulge}} \exp\left(-b_4 \left[\left(\frac{r}{R_{\text{sersic}}^{\text{bulge}}}\right)^{1/4} - 1\right]\right)$$
$$I_{\text{PS}}(x,y) = A_{\text{PS}} \cdot \delta(x - \alpha_{\text{PS}}, y - \delta_{\text{PS}})$$
where $b_n \approx 1.9992n - 0.3271$, $r_{\text{eff}} = \sqrt{q x'^2 + y'^2/q}$ for elliptical profile
- **inputs**: kwargs_disk, kwargs_bulge, kwargs_ps (ground truth parameters)
- **outputs**: Model parameter dictionaries

**Step 4: Clean Image Simulation** [FORWARD]
- **formula**: 
$$\mathbf{y}_{\text{clean}} = \mathbf{H} * (I_{\text{disk}} + I_{\text{bulge}} + I_{\text{PS}})$$
- **inputs**: Light models, PSF, coordinate grid
- **outputs**: $\mathbf{y}_{\text{clean}} \in \mathbb{R}^{80 \times 80}$

**Step 5: Poisson Noise Addition**
- **formula**: 
$$\mathbf{y}_{\text{counts}} = \mathbf{y}_{\text{clean}} \cdot t_{\text{exp}}, \quad \mathbf{y}_{\text{counts}}[\mathbf{y}_{\text{counts}} < 0] = 0$$
$$\tilde{\mathbf{y}}_{\text{counts}} \sim \text{Poisson}(\mathbf{y}_{\text{counts}})$$
$$\mathbf{n}_{\text{Poisson}} = \frac{\tilde{\mathbf{y}}_{\text{counts}}}{t_{\text{exp}}} - \mathbf{y}_{\text{clean}}$$
- **inputs**: $\mathbf{y}_{\text{clean}}$, $t_{\text{exp}}=100$
- **outputs**: $\mathbf{n}_{\text{Poisson}} \in \mathbb{R}^{80 \times 80}$

**Step 6: Gaussian Background Noise Addition**
- **formula**: 
$$\mathbf{n}_{\text{bkg}} = \sigma_{\text{bkg}} \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
- **inputs**: $\sigma_{\text{bkg}}=0.1$
- **outputs**: $\mathbf{n}_{\text{bkg}} \in \mathbb{R}^{80 \times 80}$

**Step 7: Final Observed Image**
- **formula**: 
$$\mathbf{y} = \mathbf{y}_{\text{clean}} + \mathbf{n}_{\text{bkg}} + \mathbf{n}_{\text{Poisson}}$$
- **inputs**: $\mathbf{y}_{\text{clean}}$, $\mathbf{n}_{\text{bkg}}$, $\mathbf{n}_{\text{Poisson}}$
- **outputs**: $\mathbf{y} \in \mathbb{R}^{80 \times 80}$ (observed noisy image)

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

**Step I.1: Parameter Bounds Definition**
- **formula**: 
$$\boldsymbol{\theta}_{\text{lower}} = [0.001, -0.5, -0.5, -10, -10, 0.001, -10, -10, -10, -10]^T$$
$$\boldsymbol{\theta}_{\text{upper}} = [10, 0.5, 0.5, 10, 10, 10, 10, 10, 10, 10]^T$$
(After constraint reduction, effective dimension reduces)
- **inputs**: Fixed bounds from code
- **outputs**: $\boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}} \in \mathbb{R}^{N_{\text{params}}}$

**Step I.2: Initial Parameter Vector**
- **formula**: 
$$\boldsymbol{\theta}_{\text{init}} = [R_{\text{disk}}^{(0)}, e_1^{(0)}, e_2^{(0)}, c_x^{(0)}, c_y^{(0)}, R_{\text{bulge}}^{(0)}, \alpha^{(0)}, \delta^{(0)}]^T$$
$$= [1.0, 0, 0, 0, 0, 0.5, 0.0, 0.0]^T$$
- **inputs**: Hardcoded initial guesses
- **outputs**: $\boldsymbol{\theta}_{\text{init}}$

**Step I.3: Sigma Vector for Initialization**
- **formula**: 
$$\boldsymbol{\sigma} = \sigma_{\text{scale}} \cdot [0.5, 0.1, 0.1, 0.1, 0.1, 0.3, 0.01, 0.01]^T$$
where $\sigma_{\text{scale}} = 1.0$
- **inputs**: kwargs_source_sigma, kwargs_ps_sigma
- **outputs**: $\boldsymbol{\sigma} \in \mathbb{R}^{N_{\text{params}}}$

**Step I.4: Particle Swarm Initialization**
- **formula**: 
$$\boldsymbol{\theta}^{(0)}_p = \boldsymbol{\theta}_{\text{init}} + \boldsymbol{\sigma} \odot \mathbf{u}_p, \quad \mathbf{u}_p \sim \mathcal{U}(-1, 1)^{N_{\text{params}}}$$
$$\boldsymbol{\theta}^{(0)}_p = \text{clip}(\boldsymbol{\theta}^{(0)}_p, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$$
for $p = 1, \ldots, N_{\text{particles}}$
- **inputs**: $\boldsymbol{\theta}_{\text{init}}$, $\boldsymbol{\sigma}$, bounds
- **outputs**: $\boldsymbol{\theta}^{(0)}_p$ for all particles

**Step I.5: Initial Velocity**
- **formula**: 
$$\boldsymbol{v}^{(0)}_p \sim \mathcal{U}(-|\boldsymbol{\theta}_{\text{upper}} - \boldsymbol{\theta}_{\text{lower}}|, |\boldsymbol{\theta}_{\text{upper}} - \boldsymbol{\theta}_{\text{lower}}|)$$
- **inputs**: Bounds
- **outputs**: $\boldsymbol{v}^{(0)}_p$ for all particles

**Step I.6: Initial Fitness Evaluation**
- **formula**: 
$$f^{(0)}_p = \chi^2(\boldsymbol{\theta}^{(0)}_p) \quad \forall p$$
$$\boldsymbol{\theta}^{\text{pbest}}_p = \boldsymbol{\theta}^{(0)}_p, \quad f^{\text{pbest}}_p = f^{(0)}_p$$
$$\boldsymbol{\theta}^{\text{gbest}} = \boldsymbol{\theta}^{(0)}_{p^*}, \quad p^* = \arg\min_p f^{(0)}_p$$
- **inputs**: Initial particle positions
- **outputs**: Personal and global bests initialized

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, \ldots, N_{\text{iterations}} - 1$)

**Step L.1: Generate Random Coefficients**
- **formula**: 
$$r_1^{(k)}, r_2^{(k)} \sim \mathcal{U}(0, 1)$$
- **inputs**: None
- **outputs**: $r_1^{(k)}, r_2^{(k)} \in \mathbb{R}$
- **computational_note**: Generated independently per particle and per dimension

**Step L.2: Velocity Update**
- **formula**: 
$$\boldsymbol{v}^{(k+1)}_p = w \boldsymbol{v}^{(k)}_p + c_1 r_1^{(k)} \odot (\boldsymbol{\theta}^{\text{pbest}}_p - \boldsymbol{\theta}^{(k)}_p) + c_2 r_2^{(k)} \odot (\boldsymbol{\theta}^{\text{gbest}} - \boldsymbol{\theta}^{(k)}_p)$$
where $w = 0.7298$, $c_1 = c_2 = 1.49618$
- **inputs**: $\boldsymbol{v}^{(k)}_p$, $\boldsymbol{\theta}^{(k)}_p$, $\boldsymbol{\theta}^{\text{pbest}}_p$, $\boldsymbol{\theta}^{\text{gbest}}$
- **outputs**: $\boldsymbol{v}^{(k+1)}_p$
- **computational_note**: Velocity clamping may be applied to prevent explosion

**Step L.3: Position Update**
- **formula**: 
$$\tilde{\boldsymbol{\theta}}^{(k+1)}_p = \boldsymbol{\theta}^{(k)}_p + \boldsymbol{v}^{(k+1)}_p$$
- **inputs**: $\boldsymbol{\theta}^{(k)}_p$, $\boldsymbol{v}^{(k+1)}_p$
- **outputs**: $\tilde{\boldsymbol{\theta}}^{(k+1)}_p$ (unconstrained)

**Step L.4: Boundary Enforcement (Projection)**
- **formula**: 
$$\boldsymbol{\theta}^{(k+1)}_p = \text{clip}(\tilde{\boldsymbol{\theta}}^{(k+1)}_p, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$$
- **inputs**: $\tilde{\boldsymbol{\theta}}^{(k+1)}_p$, bounds
- **outputs**: $\boldsymbol{\theta}^{(k+1)}_p$ (feasible)

**Step L.5: Apply Joint Constraints**
- **formula**: 
$$c_x^{\text{bulge}} \leftarrow c_x^{\text{disk}}, \quad c_y^{\text{bulge}} \leftarrow c_y^{\text{disk}}$$
$$\alpha^{\text{PS}} \leftarrow c_x^{\text{disk}}, \quad \delta^{\text{PS}} \leftarrow c_y^{\text{disk}}$$
- **inputs**: $\boldsymbol{\theta}^{(k+1)}_p$
- **outputs**: $\boldsymbol{\theta}^{(k+1)}_p$ with enforced joint constraints

**Step L.6: Design Matrix Construction** [FORWARD]
- **formula**: 
$$\mathbf{M}_{:,1} = \text{vec}(\mathbf{H} * I_{\text{disk}}^{\text{unit}}(\boldsymbol{\theta}^{(k+1)}_p))$$
$$\mathbf{M}_{:,2} = \text{vec}(\mathbf{H} * I_{\text{bulge}}^{\text{unit}}(\boldsymbol{\theta}^{(k+1)}_p))$$
$$\mathbf{M}_{:,3} = \text{vec}(\mathbf{H} * I_{\text{PS}}^{\text{unit}}(\boldsymbol{\theta}^{(k+1)}_p))$$
where $I^{\text{unit}}$ denotes unit-amplitude surface brightness profiles
- **inputs**: Shape parameters $\boldsymbol{\theta}^{(k+1)}_p$, PSF $\mathbf{H}$
- **outputs**: $\mathbf{M} \in \mathbb{R}^{N_{\text{pix}} \times 3}$

**Step L.7: Linear Amplitude Solve**
- **formula**: 
$$\boldsymbol{a}^* = \arg\min_{\boldsymbol{a}} \|\mathbf{W}(\mathbf{y} - \mathbf{M}\boldsymbol{a})\|_2^2 = (\mathbf{M}^T \mathbf{W}^2 \mathbf{M})^{-1} \mathbf{M}^T \mathbf{W}^2 \mathbf{y}$$
where $\mathbf{W} = \text{diag}(1/\sigma_i)$ is the noise weighting matrix
- **inputs**: $\mathbf{M}$, $\mathbf{y}$, noise model
- **outputs**: $\boldsymbol{a}^* = [A_{\text{disk}}, A_{\text{bulge}}, A_{\text{PS}}]^T$
- **computational_note**: Solved via weighted least squares (lenstronomy's ImageLinearFit)

**Step L.8: Predicted Image Construction**
- **formula**: 
$$\hat{\mathbf{y}}_p = \mathbf{M} \boldsymbol{a}^*$$
- **inputs**: $\mathbf{M}$, $\boldsymbol{a}^*$
- **outputs**: $\hat{\mathbf{y}}_p \in \mathbb{R}^{N_{\text{pix}}}$

**Step L.9: Fitness Evaluation**
- **formula**: 
$$f^{(k+1)}_p = \chi^2(\boldsymbol{\theta}^{(k+1)}_p) = \sum_{i=1}^{N_{\text{pix}}} \frac{(y_i - \hat{y}_{p,i})^2}{\sigma_i^2}$$
- **inputs**: $\mathbf{y}$, $\hat{\mathbf{y}}_p$, noise model
- **outputs**: $f^{(k+1)}_p$ (scalar fitness)

**Step L.10: Personal Best Update**
- **formula**: 
$$\text{if } f^{(k+1)}_p < f^{\text{pbest}}_p: \quad \boldsymbol{\theta}^{\text{pbest}}_p \leftarrow \boldsymbol{\theta}^{(k+1)}_p, \quad f^{\text{pbest}}_p \leftarrow f^{(k+1)}_p$$
- **inputs**: $f^{(k+1)}_p$, $f^{\text{pbest}}_p$
- **outputs**: Updated personal bests

**Step L.11: Global Best Update**
- **formula**: 
$$p^* = \arg\min_p f^{\text{pbest}}_p$$
$$\text{if } f^{\text{pbest}}_{p^*} < f^{\text{gbest}}: \quad \boldsymbol{\theta}^{\text{gbest}} \leftarrow \boldsymbol{\theta}^{\text{pbest}}_{p^*}, \quad f^{\text{gbest}} \leftarrow f^{\text{pbest}}_{p^*}$$
- **inputs**: All personal bests
- **outputs**: Updated global best

**END_LOOP**

### 4.3 FINALIZATION

**Step F.1: Extract Best Parameters**
- **formula**: 
$$\boldsymbol{\theta}^* = \boldsymbol{\theta}^{\text{gbest}}$$
- **inputs**: Final global best
- **outputs**: kwargs_result dictionary

**Step F.2: Final Image Reconstruction**
- **formula**: 
$$\hat{\mathbf{y}}^* = \mathbf{M}(\boldsymbol{\theta}^*) \boldsymbol{a}^*(\boldsymbol{\theta}^*)$$
using ImageLinearFit.image_linear_solve
- **inputs**: $\boldsymbol{\theta}^*$
- **outputs**: image_reconstructed $\in \mathbb{R}^{80 \times 80}$

**Step F.3: Residual Computation**
- **formula**: 
$$\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}^*$$
- **inputs**: $\mathbf{y}$, $\hat{\mathbf{y}}^*$
- **outputs**: residual $\in \mathbb{R}^{80 \times 80}$

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: 
$$k \geq N_{\text{iterations}} = 50$$
- **parameters**: 
  - $N_{\text{iterations}} = 50$
  - $N_{\text{particles}} = 50$
- **check_frequency**: After each complete swarm update (once per iteration)

---

## 6. EVALUATION