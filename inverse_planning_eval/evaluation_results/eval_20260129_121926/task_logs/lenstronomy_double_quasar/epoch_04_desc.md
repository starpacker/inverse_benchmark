## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Gravitational Lens Modeling with PSO-MCMC Hybrid Optimization
- **algorithm_family**: Parametric Bayesian Inference / Hybrid Global-Local Optimization
- **summary**: A two-stage optimization pipeline that first uses Particle Swarm Optimization (PSO) for global exploration of gravitational lens model parameters, followed by Markov Chain Monte Carlo (MCMC) sampling for posterior characterization, minimizing the chi-squared residual between observed and forward-modeled lensed images.

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta} \in \Omega} \left[ -\log \mathcal{L}(\mathbf{d} | \boldsymbol{\theta}) \right] = \arg\min_{\boldsymbol{\theta} \in \Omega} \left[ \frac{1}{2} \sum_{i,j} \frac{(d_{ij} - \mathcal{F}(\boldsymbol{\theta})_{ij})^2}{\sigma_{\text{bg}}^2} \right]$$

subject to box constraints $\boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}}$

where $\boldsymbol{\theta} = \{\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\theta}_{\text{source}}, \boldsymbol{\theta}_{\text{lens\_light}}, \boldsymbol{\theta}_{\text{ps}}\}$

### data_fidelity_term

$$\mathcal{D}(\boldsymbol{\theta}) = \frac{1}{2\sigma_{\text{bg}}^2} \| \mathbf{d} - \mathcal{F}(\boldsymbol{\theta}) \|_2^2 = \frac{1}{2\sigma_{\text{bg}}^2} \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} \left( d_{ij} - \mathcal{F}(\boldsymbol{\theta})_{ij} \right)^2$$

where:
- $\mathbf{d} \in \mathbb{R}^{N_x \times N_y}$ is the observed noisy image
- $\mathcal{F}(\boldsymbol{\theta})$ is the forward model (lensed image prediction)
- $\sigma_{\text{bg}} = 0.5$ is the background RMS noise
- This is an **L2 (Gaussian) likelihood** term

### regularization_term

**None** — The algorithm uses implicit regularization through bounded parameter spaces and prior constraints rather than explicit regularization terms.

### constraints

$$\mathbb{I}_{\Omega}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}} \\ +\infty & \text{otherwise} \end{cases}$$

Additional constraints:
- Joint source-point source position constraint: source center linked to point source origin
- Image position uncertainty tolerance: $\sigma_{\text{pos}} = 0.004$ arcsec
- Source position tolerance: $\epsilon_{\text{source}} = 0.001$ arcsec
- Fixed shear center at origin: $(ra_0, dec_0) = (0, 0)$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 1.1 | Einstein radius (arcsec) |
| $\gamma_{\text{EPL}}$ | scalar | float64 | 2.0 | Power-law slope of mass profile |
| $e_1^{\text{lens}}$ | scalar | float64 | 0.0 | Lens ellipticity component 1 |
| $e_2^{\text{lens}}$ | scalar | float64 | 0.0 | Lens ellipticity component 2 |
| $x_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center x-coordinate (arcsec) |
| $y_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center y-coordinate (arcsec) |
| $\gamma_1$ | scalar | float64 | 0.0 | External shear component 1 |
| $\gamma_2$ | scalar | float64 | 0.0 | External shear component 2 |
| $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.03 | Source Sersic half-light radius |
| $n_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | scalar | float64 | 0.0 | Source ellipticity component 1 |
| $e_2^{\text{src}}$ | scalar | float64 | 0.0 | Source ellipticity component 2 |
| $x_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center x (arcsec) |
| $y_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center y (arcsec) |
| $R_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 0.1 | Lens light Sersic radius |
| $n_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 1.0 | Lens light Sersic index |
| $e_1^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light ellipticity component 1 |
| $e_2^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light ellipticity component 2 |
| $x_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center x |
| $y_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center y |
| $\mathbf{ra}_{\text{img}}$ | $(N_{\text{img}},)$ | float64 | $\mathbf{x}_{\text{img}} + 0.01$ | Point source RA positions |
| $\mathbf{dec}_{\text{img}}$ | $(N_{\text{img}},)$ | float64 | $\mathbf{y}_{\text{img}} - 0.01$ | Point source Dec positions |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| — | — | — | — | No explicit dual variables (handled internally by lenstronomy) |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{p}_i^{(k)}$ | $(N_{\text{params}},)$ | float64 | Random in bounds | PSO particle $i$ position at iteration $k$ |
| $\mathbf{v}_i^{(k)}$ | $(N_{\text{params}},)$ | float64 | 0 | PSO particle $i$ velocity at iteration $k$ |
| $\mathbf{p}_i^{\text{best}}$ | $(N_{\text{params}},)$ | float64 | $\mathbf{p}_i^{(0)}$ | Personal best position for particle $i$ |
| $\mathbf{g}^{\text{best}}$ | $(N_{\text{params}},)$ | float64 | Best of initial | Global best position |
| $\mathbf{w}_j^{(t)}$ | $(N_{\text{params}},)$ | float64 | From PSO result | MCMC walker $j$ position at step $t$ |
| $\mathbf{X}_{\text{chain}}$ | $(N_{\text{walkers}}, N_{\text{run}}, N_{\text{params}})$ | float64 | — | MCMC chain samples |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\sigma_{\text{bg}}$ | 0.5 | Background RMS noise (counts/s) |
| $t_{\text{exp}}$ | 100 | Exposure time (seconds) |
| $N_{\text{pix}}$ | 100 | Image dimension (pixels) |
| $\Delta_{\text{pix}}$ | 0.05 | Pixel scale (arcsec/pixel) |
| $\text{FWHM}$ | 0.2 | PSF full-width half-maximum (arcsec) |
| $N_{\text{particles}}$ | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | 10 | Number of PSO iterations |
| $N_{\text{burn}}$ | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | 10 | MCMC sampling steps |
| $r_{\text{walker}}$ | 4 | MCMC walker ratio |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | 1.0 | PSO sigma scale factor |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | 0.1 | MCMC sigma scale factor |
| $\sigma_{\text{pos}}$ | 0.004 | Image position uncertainty (arcsec) |
| $\epsilon_{\text{source}}$ | 0.001 | Source position tolerance (arcsec) |
| $\text{seed}$ | 42 | Random seed |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{d}$ | $(100, 100)$ | float64 | Observed noisy image (counts/s) |
| $\mathbf{x}_{\text{img}}$ | $(N_{\text{img}},)$ | float64 | True image RA positions (from lens equation solver) |
| $\mathbf{y}_{\text{img}}$ | $(N_{\text{img}},)$ | float64 | True image Dec positions |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Coordinate System Setup

**step_order**: 0  
**step_name**: Pixel-to-Angular Coordinate Transformation  
**formula**:
$$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$
$$c_x = c_y = \frac{N_{\text{pix}} - 1}{2} = 49.5$$
$$\text{RA}_0 = -(\mathbf{T}_{00} \cdot c_x + \mathbf{T}_{01} \cdot c_y) = -(-0.05 \times 49.5 + 0) = 2.475$$
$$\text{Dec}_0 = -(\mathbf{T}_{10} \cdot c_x + \mathbf{T}_{11} \cdot c_y) = -(0 + 0.05 \times 49.5) = -2.475$$
**inputs**: $\Delta_{\text{pix}} = 0.05$, $N_{\text{pix}} = 100$  
**outputs**: $\mathbf{T} \in \mathbb{R}^{2 \times 2}$, $\text{RA}_0 = 2.475$, $\text{Dec}_0 = -2.475$

### Step 1: True Model Parameter Definition

**step_order**: 1  
**step_name**: Ground Truth Parameter Specification  
**formula**:
$$\boldsymbol{\theta}_{\text{lens}}^{\text{true}} = \{\theta_E = 1.0, \gamma = 1.96, x_c = 0, y_c = 0, e_1 = 0.07, e_2 = -0.03\}$$
$$\boldsymbol{\theta}_{\text{shear}}^{\text{true}} = \{\gamma_1 = 0.01, \gamma_2 = 0.01\}$$
$$\boldsymbol{\theta}_{\text{source}}^{\text{true}} = \{\text{amp} = 160, R = 0.5, n = 7, x_c = 0.1, y_c = 0.3, e_1 = 0, e_2 = 0.1\}$$
$$\boldsymbol{\theta}_{\text{lens\_light}}^{\text{true}} = \{\text{amp} = 400, R = 1.0, n = 2, x_c = 0, y_c = 0\}$$
**inputs**: Hardcoded values  
**outputs**: True parameter dictionaries

### Step 2: Lens Equation Solving

**step_order**: 2  
**step_name**: Find Lensed Image Positions  
**formula**:
Solve $\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$ for $\boldsymbol{\theta}$ given source position $\boldsymbol{\beta} = (0.1, 0.3)$
where $\boldsymbol{\alpha}$ is the deflection angle from EPL+Shear model.
Search window: $N_{\text{pix}} \times \Delta_{\text{pix}} = 5.0$ arcsec, minimum separation: $\Delta_{\text{pix}} = 0.05$ arcsec
**inputs**: $(ra_s, dec_s) = (0.1, 0.3)$, $\boldsymbol{\theta}_{\text{lens}}^{\text{true}}$  
**outputs**: $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}} \in \mathbb{R}^{N_{\text{img}}}$ (up to 4 images)

### Step 3: Point Source Magnification

**step_order**: 3  
**step_name**: Compute Image Magnifications  
**formula**:
$$\mu_i = \frac{1}{\det(\mathbf{A}(\mathbf{x}_i, \mathbf{y}_i))} \quad \text{where} \quad \mathbf{A} = \mathbf{I} - \frac{\partial \boldsymbol{\alpha}}{\partial \boldsymbol{\theta}}$$
$$\text{amp}_i = |\mu_i| \times 100$$
**inputs**: $\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}}, \boldsymbol{\theta}_{\text{lens}}^{\text{true}}$  
**outputs**: Point source amplitudes $\mathbf{amp} \in \mathbb{R}^{N_{\text{img}}}$

### Step 4: Clean Image Simulation

**step_order**: 4  
**step_name**: Forward Model Image Generation [FORWARD]  
**formula**:
$$\mathbf{I}_{\text{clean}} = \mathbf{P} * \left[ \mathcal{L}_{\text{lens}}(\boldsymbol{\theta}_{\text{ll}}) + \sum_i \mathcal{S}(\boldsymbol{\beta}_i; \boldsymbol{\theta}_{\text{src}}) \cdot |\det(\mathbf{A}_i)|^{-1} + \sum_j \text{amp}_j \cdot \delta(\mathbf{x} - \mathbf{x}_j) \right]$$
where $\mathbf{P}$ is the Gaussian PSF with FWHM = 0.2 arcsec, $*$ denotes convolution
**inputs**: All true parameters, PSF configuration  
**outputs**: $\mathbf{I}_{\text{clean}} \in \mathbb{R}^{100 \times 100}$

### Step 5: Noise Addition

**step_order**: 5  
**step_name**: Simulate Observational Noise  
**formula**:
$$\mathbf{I}_{\text{counts}} = \mathbf{I}_{\text{clean}} \cdot t_{\text{exp}}$$
$$\mathbf{I}_{\text{counts}}[i,j] = \max(0, \mathbf{I}_{\text{counts}}[i,j])$$
$$\mathbf{N}_{\text{poisson}}[i,j] \sim \frac{\text{Poisson}(\lfloor \mathbf{I}_{\text{counts}}[i,j] \rfloor)}{t_{\text{exp}}} - \mathbf{I}_{\text{clean}}[i,j]$$
$$\mathbf{N}_{\text{bg}}[i,j] \sim \mathcal{N}(0, \sigma_{\text{bg}}^2) \quad \text{where } \sigma_{\text{bg}} = 0.5$$
$$\mathbf{d} = \mathbf{I}_{\text{clean}} + \mathbf{N}_{\text{poisson}} + \mathbf{N}_{\text{bg}}$$
**inputs**: $\mathbf{I}_{\text{clean}}$, $t_{\text{exp}} = 100$, $\sigma_{\text{bg}} = 0.5$, seed = 42  
**outputs**: $\mathbf{d} \in \mathbb{R}^{100 \times 100}$ (observed noisy image)

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Parameter bounds setup | $\boldsymbol{\theta}_{\text{lower}} = \{\theta_E^{\min}=0, \gamma^{\min}=1.5, e_1^{\min}=-0.5, ...\}$; $\boldsymbol{\theta}_{\text{upper}} = \{\theta_E^{\max}=10, \gamma^{\max}=2.5, e_1^{\max}=0.5, ...\}$ | Hardcoded bounds | $\boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}$ |
| I.2 | Initial parameter guess | $\boldsymbol{\theta}^{(0)} = \{\theta_E=1.1, \gamma=2.0, e_1=0, e_2=0, x_c=0, y_c=0, \gamma_1=0, \gamma_2=0, ...\}$ | Hardcoded initial values | $\boldsymbol{\theta}^{(0)}$ |
| I.3 | Search scale setup | $\boldsymbol{\sigma} = \{\sigma_{\theta_E}=0.1, \sigma_\gamma=0.1, \sigma_{e_1}=0.2, \sigma_{e_2}=0.2, ...\}$ | Hardcoded sigma values | $\boldsymbol{\sigma}$ |
| I.4 | PSO particle initialization | $\mathbf{p}_i^{(0)} \sim \mathcal{U}(\boldsymbol{\theta}^{(0)} - \sigma_{\text{scale}} \cdot \boldsymbol{\sigma}, \boldsymbol{\theta}^{(0)} + \sigma_{\text{scale}} \cdot \boldsymbol{\sigma})$ for $i = 1, ..., N_{\text{particles}}$ | $\boldsymbol{\theta}^{(0)}$, $\boldsymbol{\sigma}$, $\sigma_{\text{scale}}=1.0$ | $\{\mathbf{p}_i^{(0)}\}_{i=1}^{50}$ |
| I.5 | Velocity initialization | $\mathbf{v}_i^{(0)} = \mathbf{0}$ for all $i$ | — | $\{\mathbf{v}_i^{(0)}\}_{i=1}^{50}$ |
| I.6 | Best position initialization | Evaluate $\chi^2(\mathbf{p}_i^{(0)})$ for all $i$; $\mathbf{p}_i^{\text{best}} = \mathbf{p}_i^{(0)}$; $\mathbf{g}^{\text{best}} = \arg\min_i \chi^2(\mathbf{p}_i^{(0)})$ | $\{\mathbf{p}_i^{(0)}\}$ | $\mathbf{g}^{\text{best}}$, $\{\mathbf{p}_i^{\text{best}}\}$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP**: PSO Optimization (k = 1 to $N_{\text{iter}}^{\text{PSO}} = 10$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| P.1 | Velocity update | $\mathbf{v}_i^{(k)} = \omega \mathbf{v}_i^{(k-1)} + c_1 r_1 (\mathbf{p}_i^{\text{best}} - \mathbf{p}_i^{(k-1)}) + c_2 r_2 (\mathbf{g}^{\text{best}} - \mathbf{p}_i^{(k-1)})$ where $\omega, c_1, c_2$ are PSO coefficients, $r_1, r_2 \sim \mathcal{U}(0,1)$ | $\mathbf{v}_i^{(k-1)}, \mathbf{p}_i^{(k-1)}, \mathbf{p}_i^{\text{best}}, \mathbf{g}^{\text{best}}$ | $\mathbf{v}_i^{(k)}$ | Standard PSO dynamics |
| P.2 | Position update | $\mathbf{p}_i^{(k)} = \mathbf{p}_i^{(k-1)} + \mathbf{v}_i^{(k)}$ | $\mathbf{p}_i^{(k-1)}, \mathbf{v}_i^{(k)}$ | $\mathbf{p}_i^{(k)}$ | — |
| P.3 | Bounds enforcement | $\mathbf{p}_i^{(k)} = \text{clip}(\mathbf{p}_i^{(k)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$ | $\mathbf{p}_i^{(k)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}$ | $\mathbf{p}_i^{(k)}$ | Box constraint projection |
| P.4 | Forward model evaluation [FORWARD] | $\mathcal{F}(\mathbf{p}_i^{(k)}) = \mathbf{P} * [\mathcal{L}_{\text{lens}} + \mathcal{L}_{\text{source}}^{\text{lensed}} + \mathcal{L}_{\text{ps}}]$ | $\mathbf{p}_i^{(k)}$, model classes | $\mathbf{I}_{\text{model},i}^{(k)}$ | Full image synthesis |
| P.5 | Loss computation | $\chi_i^{2,(k)} = \frac{1}{\sigma_{\text{bg}}^2} \sum_{m,n} (d_{mn} - \mathcal{F}(\mathbf{p}_i^{(k)})_{mn})^2$ | $\mathbf{d}, \mathbf{I}_{\text{model},i}^{(k)}, \sigma_{\text{bg}}$ | $\chi_i^{2,(k)}$ | Chi-squared statistic |
| P.6 | Personal best update | $\mathbf{p}_i^{\text{best}} = \begin{cases} \mathbf{p}_i^{(k)} & \text{if } \chi_i^{2,(k)} < \chi^2(\mathbf{p}_i^{\text{best}}) \\ \mathbf{p}_i^{\text{best}} & \text{otherwise} \end{cases}$ | $\mathbf{p}_i^{(k)}, \chi_i^{2,(k)}, \mathbf{p}_i^{\text{best}}$ | $\mathbf{p}_i^{\text{best}}$ | — |
| P.7 | Global best update | $\mathbf{g}^{\text{best}} = \arg\min_{i} \chi^2(\mathbf{p}_i^{\text{best}})$ | $\{\mathbf{p}_i^{\text{best}}\}_{i=1}^{N_{\text{particles}}}$ | $\mathbf{g}^{\text{best}}$ | — |

**END_LOOP**: PSO Optimization

**BEGIN_LOOP**: MCMC Burn-in (t = 1 to $N_{\text{burn}} = 10$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| B.1 | Walker initialization | $\mathbf{w}_j^{(0)} \sim \mathcal{N}(\mathbf{g}^{\text{best}}, (\sigma_{\text{scale}}^{\text{MCMC}} \cdot \boldsymbol{\sigma})^2)$ for $j = 1, ..., N_{\text{walkers}}$ where $N_{\text{walkers}} = r_{\text{walker}} \times N_{\text{params}} = 4 \times N_{\text{params}}$ | $\mathbf{g}^{\text{best}}, \boldsymbol{\sigma}, \sigma_{\text{scale}}^{\text{MCMC}}=0.1$ | $\{\mathbf{w}_j^{(0)}\}$ | Initialize from PSO result |
| B.2 | Proposal generation | $\mathbf{w}_j^{\text{prop}} = \mathbf{w}_j^{(t-1)} + \mathbf{z}_j$ via affine-invariant ensemble sampler (stretch move) | $\mathbf{w}_j^{(t-1)}$, complementary walkers | $\mathbf{w}_j^{\text{prop}}$ | emcee-style sampler |
| B.3 | Log-likelihood evaluation | $\log \mathcal{L}(\mathbf{w}_j^{\text{prop}}) = -\frac{1}{2\sigma_{\text{bg}}^2} \|\mathbf{d} - \mathcal{F}(\mathbf{w}_j^{\text{prop}})\|_2^2$ | $\mathbf{w}_j^{\text{prop}}, \mathbf{d}$ | $\log \mathcal{L}_j^{\text{prop}}$ | — |
| B.4 | Acceptance decision | $\mathbf{w}_j^{(t)} = \begin{cases} \mathbf{w}_j^{\text{prop}} & \text{with prob } \min(1, e^{\log \mathcal{L}_j^{\text{prop}} - \log \mathcal{L}_j^{(t-1)}}) \\ \mathbf{w}_j^{(t-1)} & \text{otherwise} \end{cases}$ | $\mathbf{w}_j^{\text{prop}}, \mathbf{w}_j^{(t-1)}, \log \mathcal{L}$ values | $\mathbf{w}_j^{(t)}$ | Metropolis-Hastings |

**END_LOOP**: MCMC Burn-in

**BEGIN_LOOP**: MCMC Sampling (t = 1 to $N_{\text{run}} = 10$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| S.1 | Proposal generation | Same as B.2 | — | $\mathbf{w}_j^{\text{prop}}$ | — |
| S.2 | Log-likelihood evaluation | Same as B.3 | — | $\log \mathcal{L}_j^{\text{prop}}$ | — |
| S.3 | Acceptance decision | Same as B.4 | — | $\mathbf{w}_j^{(t)}$ | — |
| S.4 | Chain storage | $\mathbf{X}_{\text{chain}}[j, t, :] = \mathbf{w}_j^{(t)}$ | $\mathbf{w}_j^{(t)}$ | $\mathbf{X}_{\text{chain}}$ | Store samples |

**END_LOOP**: MCMC Sampling

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Best fit extraction | $\hat{\boldsymbol{\theta}} = \arg\max_{\mathbf{w} \in \mathbf{X}_{\text{chain}}} \log \mathcal{L}(\mathbf{w})$ | $\mathbf{X}_{\text{chain}}$ | $\hat{\boldsymbol{\theta}}$ (best fit parameters) |
| F.2 | Parameter unpacking | Extract $\hat{\boldsymbol{\theta}}_{\text{lens}}, \hat{\boldsymbol{\theta}}_{\text{source}}, \hat{\boldsymbol{\theta}}_{\text{lens\_light}}, \hat{\boldsymbol{\theta}}_{\text{ps}}$ | $\hat{\boldsymbol{\theta}}$ | Individual parameter dictionaries |
| F.3 | Model image generation | $\hat{\mathbf{I}} = \mathcal{F}(\hat{\boldsymbol{\theta}})$ | $\hat{\boldsymbol{\theta}}$ | $\hat{\mathbf{I}} \in \mathbb{R}^{100 \times 100}$ |
| F.