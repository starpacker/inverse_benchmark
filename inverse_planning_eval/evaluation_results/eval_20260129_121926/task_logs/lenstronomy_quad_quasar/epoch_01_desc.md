# Algorithmic Specification Document: Gravitational Lens Modeling and Parameter Inference

## 1. Mathematical Formulation

### 1.1 Problem Statement

The algorithm solves a **Bayesian inverse problem** for strong gravitational lensing: given an observed image $\mathbf{d} \in \mathbb{R}^{N \times N}$, estimate the posterior distribution of physical parameters $\boldsymbol{\theta}$ describing the lens mass distribution, source light, lens light, and point source properties.

### 1.2 Objective Function

The optimization maximizes the log-posterior (equivalently minimizes the negative log-posterior):

$$
\mathcal{L}(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta} | \mathbf{d}) = -\log p(\mathbf{d} | \boldsymbol{\theta}) - \log p(\boldsymbol{\theta}) + \text{const}
$$

**Expanded form:**

$$
\mathcal{L}(\boldsymbol{\theta}) = \underbrace{\frac{1}{2} \sum_{i,j} \frac{\left(d_{ij} - \hat{d}_{ij}(\boldsymbol{\theta})\right)^2}{\sigma_{ij}^2}}_{\text{Data Fidelity (Chi-squared)}} + \underbrace{\sum_{k} \frac{(\theta_k - \mu_k)^2}{2\sigma_k^2}}_{\text{Gaussian Priors}} + \underbrace{\mathcal{C}(\boldsymbol{\theta})}_{\text{Constraints}}
$$

Where:
- $d_{ij}$: observed pixel value at position $(i,j)$
- $\hat{d}_{ij}(\boldsymbol{\theta})$: predicted pixel value from forward model
- $\sigma_{ij}^2 = \sigma_{\text{bkg}}^2 + \frac{\hat{d}_{ij}}{t_{\text{exp}}}$: noise variance (background + Poisson)
- $\mathcal{C}(\boldsymbol{\theta})$: constraint penalty terms

### 1.3 Forward Model

The predicted image is:

$$
\hat{\mathbf{d}}(\boldsymbol{\theta}) = \mathbf{P} * \left[ I_{\text{lens}}(\mathbf{x}) + I_{\text{source}}(\boldsymbol{\beta}(\mathbf{x})) + \sum_{k=1}^{N_{\text{ps}}} A_k \, \delta(\mathbf{x} - \mathbf{x}_k) \right]
$$

Where:
- $\mathbf{P}$: Point Spread Function (PSF) kernel
- $*$: convolution operator
- $I_{\text{lens}}(\mathbf{x})$: lens galaxy light at image plane position $\mathbf{x}$
- $I_{\text{source}}(\boldsymbol{\beta})$: source light at source plane position $\boldsymbol{\beta}$
- $\boldsymbol{\beta}(\mathbf{x}) = \mathbf{x} - \boldsymbol{\alpha}(\mathbf{x})$: lens equation mapping
- $\boldsymbol{\alpha}(\mathbf{x})$: deflection angle field
- $A_k$, $\mathbf{x}_k$: point source amplitudes and positions

### 1.4 Component Models

#### 1.4.1 Lens Mass Model: Elliptical Power Law (EPL) + External Shear

**Deflection angle (EPL):**
$$
\boldsymbol{\alpha}_{\text{EPL}}(\mathbf{x}) = \frac{\theta_E^{2-\gamma}}{1 + q} \left( \frac{|\mathbf{x}'|}{s_{\text{scale}}} \right)^{2-\gamma} \mathbf{F}(\phi', q, \gamma)
$$

Where:
- $\theta_E$: Einstein radius
- $\gamma$: power-law slope (3D density $\rho \propto r^{-\gamma}$)
- $q = \sqrt{(1-e)/(1+e)}$: axis ratio from ellipticity $e = \sqrt{e_1^2 + e_2^2}$
- $\mathbf{x}' = \mathbf{R}(\phi_e) (\mathbf{x} - \mathbf{x}_c)$: rotated, centered coordinates

**External shear deflection:**
$$
\boldsymbol{\alpha}_{\text{shear}}(\mathbf{x}) = \begin{pmatrix} \gamma_1 x + \gamma_2 y \\ \gamma_2 x - \gamma_1 y \end{pmatrix}
$$

**Total deflection:**
$$
\boldsymbol{\alpha}(\mathbf{x}) = \boldsymbol{\alpha}_{\text{EPL}}(\mathbf{x}) + \boldsymbol{\alpha}_{\text{shear}}(\mathbf{x})
$$

#### 1.4.2 Light Profiles: Sérsic Model

**Sérsic surface brightness:**
$$
I(\mathbf{x}) = I_e \exp\left[ -b_n \left( \left(\frac{r(\mathbf{x})}{R_{\text{sersic}}}\right)^{1/n} - 1 \right) \right]
$$

Where:
- $I_e$ (or `amp`): effective intensity
- $R_{\text{sersic}}$: half-light radius
- $n$: Sérsic index
- $b_n \approx 2n - 1/3 + 4/(405n)$: normalization constant
- $r(\mathbf{x}) = \sqrt{q x'^2 + y'^2/q}$: elliptical radius (for elliptical Sérsic)

#### 1.4.3 Point Spread Function (Gaussian)

$$
P(\mathbf{x}) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left( -\frac{|\mathbf{x}|^2}{2\sigma_{\text{PSF}}^2} \right)
$$

Where: $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$

### 1.5 Prior Terms

**Gaussian priors on ellipticity:**
$$
\log p(e_1) = -\frac{(e_1 - 0)^2}{2 \times 0.2^2}, \quad \log p(e_2) = -\frac{(e_2 - 0)^2}{2 \times 0.2^2}
$$

**Box constraints (hard bounds):**
$$
\mathcal{C}_{\text{bounds}}(\boldsymbol{\theta}) = \begin{cases} 0 & \text{if } \boldsymbol{\theta}_{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}_{\text{upper}} \\ +\infty & \text{otherwise} \end{cases}
$$

### 1.6 Constraints

**Source-Point Source Joint Constraint:**
$$
\mathbf{x}_{\text{source}} = \text{centroid of point source in source plane}
$$

**Lens Equation Solver Constraint (image position likelihood):**
$$
\log p(\mathbf{x}_{\text{img}} | \boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\beta}_s) = -\frac{1}{2} \sum_{k=1}^{N_{\text{img}}} \frac{|\mathbf{x}_k - \mathbf{x}_k^{\text{pred}}(\boldsymbol{\theta}_{\text{lens}}, \boldsymbol{\beta}_s)|^2}{\sigma_{\text{pos}}^2}
$$

Where $\sigma_{\text{pos}} = 0.004$ arcsec (image position uncertainty).

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | Dtype | Init Value | Bounds |
|----------|--------|-------|-------|------------|--------|
| **Lens (EPL)** | | | | | |
| Einstein radius | $\theta_E$ | scalar | float64 | 1.2 | [0, 10] |
| Power-law slope | $\gamma$ | scalar | float64 | 2.0 | [1.5, 2.5] |
| Ellipticity 1 | $e_1^{\text{lens}}$ | scalar | float64 | 0 | [-0.5, 0.5] |
| Ellipticity 2 | $e_2^{\text{lens}}$ | scalar | float64 | 0 | [-0.5, 0.5] |
| Center x | $x_c^{\text{lens}}$ | scalar | float64 | 0 | [-10, 10] |
| Center y | $y_c^{\text{lens}}$ | scalar | float64 | 0 | [-10, 10] |
| **Lens (Shear)** | | | | | |
| Shear component 1 | $\gamma_1$ | scalar | float64 | 0 | [-0.5, 0.5] |
| Shear component 2 | $\gamma_2$ | scalar | float64 | 0 | [-0.5, 0.5] |
| **Source (Sérsic Ellipse)** | | | | | |
| Half-light radius | $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.03 | [0.001, 10] |
| Sérsic index | $n^{\text{src}}$ | scalar | float64 | 1.0 | [0.5, 5.0] |
| Ellipticity 1 | $e_1^{\text{src}}$ | scalar | float64 | 0 | [-0.5, 0.5] |
| Ellipticity 2 | $e_2^{\text{src}}$ | scalar | float64 | 0 | [-0.5, 0.5] |
| Center x | $x_c^{\text{src}}$ | scalar | float64 | 0 | [-10, 10] |
| Center y | $y_c^{\text{src}}$ | scalar | float64 | 0 | [-10, 10] |
| **Lens Light (Sérsic)** | | | | | |
| Half-light radius | $R_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 0.1 | [0.001, 10] |
| Sérsic index | $n^{\text{ll}}$ | scalar | float64 | 1.0 | [0.5, 5.0] |
| Ellipticity 1 | $e_1^{\text{ll}}$ | scalar | float64 | 0 | [-0.5, 0.5] |
| Ellipticity 2 | $e_2^{\text{ll}}$ | scalar | float64 | 0 | [-0.5, 0.5] |
| Center x | $x_c^{\text{ll}}$ | scalar | float64 | 0 | [-10, 10] |
| Center y | $y_c^{\text{ll}}$ | scalar | float64 | 0 | [-10, 10] |
| **Point Sources** | | | | | |
| Image RA positions | $\alpha_k$ | (4,) | float64 | $\alpha_k^{\text{true}} + 0.01$ | [-10, 10] |
| Image Dec positions | $\delta_k$ | (4,) | float64 | $\delta_k^{\text{true}} - 0.01$ | [-10, 10] |

### 2.2 Auxiliary/Derived Variables

| Variable | Symbol | Computation |
|----------|--------|-------------|
| Source position | $\boldsymbol{\beta}(\mathbf{x})$ | $\mathbf{x} - \boldsymbol{\alpha}(\mathbf{x})$ |
| Magnification | $\mu(\mathbf{x})$ | $\det^{-1}(\mathbf{I} - \partial\boldsymbol{\alpha}/\partial\mathbf{x})$ |
| Point source amplitudes | $A_k$ | Solved via linear optimization (multi-linear mode) |
| Source amplitude | $I_e^{\text{src}}$ | Solved via linear optimization |
| Lens light amplitude | $I_e^{\text{ll}}$ | Solved via linear optimization |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| **Data** | | | |
| Number of pixels | $N$ | 100 | Image dimension |
| Pixel scale | $\Delta_{\text{pix}}$ | 0.05 arcsec | Angular size per pixel |
| Background RMS | $\sigma_{\text{bkg}}$ | 0.5 | Background noise level |
| Exposure time | $t_{\text{exp}}$ | 100 s | Integration time |
| PSF FWHM | FWHM | 0.1 arcsec | Point spread function width |
| **PSO** | | | |
| Number of particles | $N_{\text{part}}$ | 50 | Swarm size |
| Number of iterations | $N_{\text{iter}}^{\text{PSO}}$ | 10 | PSO iterations |
| Sigma scale | $\sigma_{\text{scale}}^{\text{PSO}}$ | 1.0 | Initial dispersion scale |
| **MCMC** | | | |
| Number of walkers | $N_{\text{walk}}$ | 50 | Ensemble size |
| Burn-in steps | $N_{\text{burn}}$ | 10 | Discarded samples |
| Production steps | $N_{\text{run}}$ | 10 | Retained samples |
| Sigma scale | $\sigma_{\text{scale}}^{\text{MCMC}}$ | 0.1 | Proposal scale |
| **Priors** | | | |
| Ellipticity prior mean | $\mu_{e}$ | 0 | Prior center |
| Ellipticity prior std | $\sigma_{e}$ | 0.2 | Prior width |
| Position uncertainty | $\sigma_{\text{pos}}$ | 0.004 arcsec | Image position error |
| Source position tolerance | $\sigma_{\text{src}}$ | 0.001 arcsec | Source constraint |

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | Description |
|----------|--------|-------|-------------|
| Observed image | $\mathbf{d}$ | $(100, 100)$ | Noisy simulated observation |
| Coordinate transform | $\mathbf{T}$ | $(2, 2)$ | Pixel to angular coordinates |
| Reference RA | $\alpha_0$ | scalar | RA at pixel (0,0) |
| Reference Dec | $\delta_0$ | scalar | Dec at pixel (0,0) |

---

## 3. Preprocessing Pipeline

### Step 1: Coordinate System Setup
**Input:** $N_{\text{pix}}$, $\Delta_{\text{pix}}$  
**Output:** $\mathbf{T}$, $\alpha_0$, $\delta_0$

$$
\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}
$$

$$
c_x = c_y = \frac{N_{\text{pix}} - 1}{2}
$$

$$
\alpha_0 = -(T_{00} \cdot c_x + T_{01} \cdot c_y) = \Delta_{\text{pix}} \cdot c_x
$$

$$
\delta_0 = -(T_{10} \cdot c_x + T_{11} \cdot c_y) = -\Delta_{\text{pix}} \cdot c_y
$$

### Step 2: PSF Construction
**Input:** FWHM, $\Delta_{\text{pix}}$, truncation factor $\tau=5$  
**Output:** PSF kernel $\mathbf{P}$

$$
\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} \approx 0.0424 \text{ arcsec}
$$

$$
P_{ij} = \exp\left(-\frac{(i\Delta_{\text{pix}})^2 + (j\Delta_{\text{pix}})^2}{2\sigma_{\text{PSF}}^2}\right), \quad |i|, |j| \leq \tau \cdot \sigma_{\text{PSF}}/\Delta_{\text{pix}}
$$

### Step 3: Shear Coordinate Conversion
**Input:** Polar shear $(\phi_\gamma=0.1, \gamma=0.02)$  
**Output:** Cartesian shear $(\gamma_1, \gamma_2)$

$$
\gamma_1 = \gamma \cos(2\phi_\gamma) \approx 0.0196
$$

$$
\gamma_2 = \gamma \sin(2\phi_\gamma) \approx 0.00397
$$

### Step 4: Point Source Position Computation
**Input:** Source position $(\alpha_s, \delta_s) = (0, 0.1)$, lens parameters  
**Output:** Image positions $\{(\alpha_k, \delta_k)\}_{k=1}^{4}$

Solve lens equation: $\boldsymbol{\beta}_s = \mathbf{x}_k - \boldsymbol{\alpha}(\mathbf{x}_k)$ for all solutions $\mathbf{x}_k$.

### Step 5: Magnification and Amplitude Computation
**Input:** Image positions, lens parameters  
**Output:** Point source amplitudes $\{A_k\}$

$$
\mu_k = \left| \det\left(\mathbf{I} - \frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_k}\right) \right|^{-1}
$$

$$
\tilde{\mu}_k \sim \mathcal{N}(\mu_k, 0.5)
$$

$$
A_k = 100 \cdot \tilde{\mu}_k
$$

### Step 6: Forward Model Evaluation (Noiseless)
**Input:** All model parameters, PSF  
**Output:** Noiseless image $\hat{\mathbf{d}}$

$$
\hat{\mathbf{d}} = \mathbf{P} * \left[ I_{\text{lens}} + I_{\text{source}} + I_{\text{PS}} \right]
$$

### Step 7: Noise Addition
**Input:** Noiseless image $\hat{\mathbf{d}}$, $t_{\text{exp}}$, $\sigma_{\text{bkg}}$  
**Output:** Noisy observation $\mathbf{d}$

**7a. Poisson Noise:**
$$
\mathbf{c} = \hat{\mathbf{d}} \cdot t_{\text{exp}}, \quad c_{ij} \leftarrow \max(c_{ij}, 0)
$$

$$
\tilde{c}_{ij} \sim \text{Poisson}(c_{ij})
$$

$$
\mathbf{n}_{\text{Poisson}} = \frac{\tilde{\mathbf{c}}}{t_{\text{exp}}} - \hat{\mathbf{d}}
$$

**7b. Gaussian Background Noise:**
$$
n^{\text{bkg}}_{ij} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)
$$

**7c. Combined Observation:**
$$
\mathbf{d} = \hat{\mathbf{d}} + \mathbf{n}_{\text{Poisson}} + \mathbf{n}_{\text{bkg}}
$$

---

## 4. Algorithm Steps

### Phase 1: Particle Swarm Optimization (PSO)

#### Step 1.1: Initialization
**Input:** Initial parameters $\boldsymbol{\theta}^{(0)}$, bounds $[\boldsymbol{\theta}_L, \boldsymbol{\theta}_U]$, sigmas $\boldsymbol{\sigma}$, $N_{\text{part}}=50$  
**Output:** Particle positions $\{\mathbf{p}_i^{(0)}\}_{i=1}^{N_{\text{part}}}$, velocities $\{\mathbf{v}_i^{(0)}\}$

For each particle $i = 1, \ldots, N_{\text{part}}$:
$$
p_{i,k}^{(0)} \sim \mathcal{N}\left(\theta_k^{(0)}, (\sigma_{\text{scale}} \cdot \sigma_k)^2\right), \quad \text{clipped to } [\theta_{L,k}, \theta_{U,k}]
$$

$$
\mathbf{v}_i^{(0)} = \mathbf{0}
$$

$$
\mathbf{b}_i = \mathbf{p}_i^{(0)}, \quad f_i^{\text{best}} = \mathcal{L}(\mathbf{p}_i^{(0)})
$$

$$
\mathbf{g} = \arg\min_{\mathbf{b}_i} f_i^{\text{best}}
$$

#### Step 1.2: PSO Iteration
**Input:** Current particles $\{\mathbf{p}_i^{(t)}, \mathbf{v}_i^{(t)}, \mathbf{b}_i, f_i^{\text{best}}\}$, global best $\mathbf{g}$  
**Output:** Updated particles $\{\mathbf{p}_i^{(t+1)}, \mathbf{v}_i^{(t+1)}\}$, updated bests

For $t = 1, \ldots, N_{\text{iter}}^{\text{PSO}}$:  
For each particle $i = 1, \ldots, N_{\text{part}}$:

$$
r_1, r_2 \sim \mathcal{U}(0, 1)
$$

$$
\mathbf{v}_i^{(t+1)} = \omega \mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{b}_i - \mathbf{p}_i^{(t)}) + c_2 r_2 (\mathbf{g} - \mathbf{p}_i^{(t)})
$$

$$
\mathbf{p}_i^{(t+1)} = \mathbf{p}_i^{(t)} + \mathbf{v}_i^{(t+1)}, \quad \text{clipped to bounds}
$$

$$
f_i = \mathcal{L}(\mathbf{p}_i^{(t+1)})
$$

$$
\text{If } f_i < f_i^{\text{best}}: \quad \mathbf{b}_i \leftarrow \mathbf{p}_i^{(t+1)}, \quad f_i^{\text{best}} \leftarrow f_i
$$

$$
\mathbf{g} \leftarrow \arg\min_{\mathbf{b}_i} f_i^{\text{best}}
$$

**Standard PSO coefficients:** $\omega \approx 0.7$, $c_1 = c_2 \approx 1.5$

#### Step 1.3: PSO Output
**Output:** Best-fit parameters $\boldsymbol{\theta}^{\text{PSO}} = \mathbf{g}$

---

### Phase 2: Markov Chain Monte Carlo (MCMC)

#### Step 2.1: MCMC Initialization
**Input:** PSO result $\boldsymbol{\theta}^{\text{PSO}}$, $\boldsymbol{\sigma}$, $N_{\text{walk}}=50$  
**Output:** Initial walker positions $\{\boldsymbol{\theta}_j^{(0)}\}_{j=1}^{N_{\text{walk}}}$

For each walker $j = 1, \ldots, N_{\text{walk}}$:
$$
\theta_{j,k}^{(0)} \sim \mathcal{N}\left(\theta_k^{\text{PSO}}, (\sigma_{\text{scale}}^{\text{MCMC}} \cdot \sigma_k)^2\right)
$$

$$
\ln p_j^{(0)} = \ln p(\boldsymbol{\theta}_j^{(0)} | \mathbf{d})
$$

#### Step 2.2: MCMC Iteration (Affine-Invariant Ensemble Sampler)
**Input:** Walker ensemble $\{\boldsymbol{\theta}_j^{(t)}\}$  
**Output:** Updated ensemble $\{\boldsymbol{\theta}_j^{(t+1)}\}$

For $t = 1, \ldots, N_{\text{burn}} + N_{\text{run}}$:  
For each walker $j$:

**2.2a. Stretch Move Proposal:**
$$
k \sim \mathcal{U}(\{1, \ldots, N_{\text{walk}}\} \setminus \{j\})
$$

$$
z \sim g(z) \propto \begin{cases} 1/\sqrt{z} & \text{if } z \in [1/a, a] \\ 0 & \text{otherwise} \end{cases}, \quad a = 2
$$

$$
\boldsymbol{\theta}_j^* = \boldsymbol{\theta}_k^{(t)} + z \left( \boldsymbol{\theta}_j^{(t)} - \boldsymbol{\theta}_k^{(t)} \right)
$$

**2.2b. Acceptance Probability:**
$$
\ln p_j^* = \ln p(\boldsymbol{\theta}_j^* | \mathbf{d})
$$

$$
\alpha = \min\left(1, z^{D-1} \exp\left(\ln p_j^* - \ln p_j^{(t)}\right)\right)
$$

where $D$ = number of free parameters.

**2.2c. Accept/Reject:**
$$
u \sim \mathcal{U}(0, 1)
$$

$$
\boldsymbol{\theta}_j^{(t+1)} = \begin{cases} \boldsymbol{\theta}_j^* & \text{if } u < \alpha \\ \boldsymbol{\theta}_j^{(t)} & \text{otherwise} \end{cases}
$$

#### Step 2.3: Post-Processing
**Input:** Full chain $\{\boldsymbol{\theta}_j^{(t)}\}$ for $t \in [1, N_{\text{burn}} + N_{\text{run}}]$  
**Output:** Best-fit $\boldsymbol{\theta}^*$, posterior samples

Discard burn-in: keep $t \in [N_{\text{burn}}+1, N_{\text{burn}} + N_{\text{run}}]$

$$
\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}_j^{(t)}} \ln p(\boldsymbol{\theta}_j^{(t)} | \mathbf{d})
$$

---

## 5. Stopping Criteria

| Phase | Criterion | Condition |
|-------|-----------|-----------|
| PSO | Maximum iterations | $t = N_{\text{iter}}^{\text{PSO}} = 10$ |
| MCMC | Maximum iterations | $t = N_{\text{burn}} + N_{\text{run}} = 20$ |
| Bounds check | Hard constraint | Reject if $\boldsymbol{\theta} \notin [\boldsymbol{\theta}_L, \boldsymbol{\theta}_U]$ |

**Note:** The code uses fixed iteration counts without adaptive convergence criteria (e.g., Gelman-Rubin diagnostic).

---

## 6. Evaluation Metrics

### 6.1 Residual Root Mean Square (RMS)

$$
\text{RMS} = \sqrt{\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \left( d_{ij} - \hat{d}_{ij}(\boldsymbol{\theta}^*) \right)^2}
$$

**Input:** Observed image $\mathbf{d}$, reconstructed image $\hat{\mathbf{d}}(\boldsymbol{\theta}^*)$  
**Output:** Scalar RMS value

### 6.2 Parameter Absolute Errors

For each parameter $\theta_k$:

$$
\epsilon_{\theta_k} = \left| \theta_k^{\text{fit}} - \theta_k^{\text{true}} \right|
$$

**Reporte