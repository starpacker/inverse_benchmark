# Algorithmic Specification Document: Strong Gravitational Lens Modeling via PSO-MCMC Inference

## 1. Mathematical Formulation

### 1.1 Objective Function

The algorithm solves a Bayesian parameter estimation problem for strong gravitational lens modeling. The objective is to find parameters $\boldsymbol{\theta}$ that maximize the posterior probability:

$$\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} \, p(\boldsymbol{\theta} | \mathbf{I}^{\text{obs}}) = \arg\max_{\boldsymbol{\theta}} \, p(\mathbf{I}^{\text{obs}} | \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta})$$

Equivalently, minimize the negative log-posterior:

$$\mathcal{J}(\boldsymbol{\theta}) = -\ln p(\mathbf{I}^{\text{obs}} | \boldsymbol{\theta}) - \ln p(\boldsymbol{\theta})$$

### 1.2 Data Fidelity Term (Log-Likelihood)

Assuming Gaussian noise with pixel-dependent variance:

$$-\ln p(\mathbf{I}^{\text{obs}} | \boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^{N_{\text{pix}}} \sum_{j=1}^{N_{\text{pix}}} \frac{\left(I^{\text{obs}}_{ij} - I^{\text{model}}_{ij}(\boldsymbol{\theta})\right)^2}{\sigma_{ij}^2} + \text{const}$$

where the noise variance combines background and Poisson contributions:

$$\sigma_{ij}^2 = \sigma_{\text{bkg}}^2 + \frac{I^{\text{model}}_{ij}(\boldsymbol{\theta})}{t_{\text{exp}}}$$

### 1.3 Prior/Regularization Term

Uniform box priors on all parameters:

$$p(\boldsymbol{\theta}) = \begin{cases} 
\text{const} & \text{if } \boldsymbol{\theta}^{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}^{\text{upper}} \\
0 & \text{otherwise}
\end{cases}$$

### 1.4 Forward Model Decomposition

The model image is computed as:

$$I^{\text{model}}_{ij}(\boldsymbol{\theta}) = \mathcal{P} * \left[ I^{\text{lens-light}}_{ij} + I^{\text{source}}_{ij} + I^{\text{PS}}_{ij} \right]$$

where $\mathcal{P}$ is the PSF convolution operator and:

$$I^{\text{source}}_{ij} = S\left(\boldsymbol{\beta}(\boldsymbol{x}_{ij}; \boldsymbol{\theta}_{\text{lens}}); \boldsymbol{\theta}_{\text{source}}\right)$$

with the lens equation mapping image plane $\boldsymbol{x}$ to source plane $\boldsymbol{\beta}$:

$$\boldsymbol{\beta} = \boldsymbol{x} - \boldsymbol{\alpha}(\boldsymbol{x}; \boldsymbol{\theta}_{\text{lens}})$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable Group | Parameter | Symbol | Shape | Bounds | Initial Value |
|----------------|-----------|--------|-------|--------|---------------|
| **EPL Lens** | Einstein radius | $\theta_E$ | scalar | $[0, 10]$ | $1.1$ |
| | Power-law slope | $\gamma$ | scalar | $[1.5, 2.5]$ | $2.0$ |
| | Ellipticity component 1 | $e_1^{\text{lens}}$ | scalar | $[-0.5, 0.5]$ | $0$ |
| | Ellipticity component 2 | $e_2^{\text{lens}}$ | scalar | $[-0.5, 0.5]$ | $0$ |
| | Center x | $x_c^{\text{lens}}$ | scalar | $[-10, 10]$ | $0$ |
| | Center y | $y_c^{\text{lens}}$ | scalar | $[-10, 10]$ | $0$ |
| **External Shear** | Shear component 1 | $\gamma_1$ | scalar | $[-0.2, 0.2]$ | $0$ |
| | Shear component 2 | $\gamma_2$ | scalar | $[-0.2, 0.2]$ | $0$ |
| **Source Light** | Sérsic radius | $R_{\text{sersic}}^{\text{src}}$ | scalar | $[0.001, 10]$ | $0.03$ |
| | Sérsic index | $n_{\text{sersic}}^{\text{src}}$ | scalar | $[0.5, 5.0]$ | $1.0$ |
| | Ellipticity $(e_1, e_2)$ | $(e_1^{\text{src}}, e_2^{\text{src}})$ | $(1,), (1,)$ | $[-0.5, 0.5]$ | $(0, 0)$ |
| | Center $(x, y)$ | $(x_c^{\text{src}}, y_c^{\text{src}})$ | $(1,), (1,)$ | $[-10, 10]$ | $(0, 0)$ |
| **Lens Light** | Sérsic radius | $R_{\text{sersic}}^{\text{ll}}$ | scalar | $[0.001, 10]$ | $0.1$ |
| | Sérsic index | $n_{\text{sersic}}^{\text{ll}}$ | scalar | $[0.5, 5.0]$ | $1.0$ |
| | Center $(x, y)$ | $(x_c^{\text{ll}}, y_c^{\text{ll}})$ | $(1,), (1,)$ | $[-10, 10]$ | $(0, 0)$ |
| **Point Sources** | Image positions RA | $\{\text{ra}_k\}_{k=1}^{N_{\text{img}}}$ | $(N_{\text{img}},)$ | $[-10, 10]$ | $x_{\text{image}}^{\text{true}} + 0.01$ |
| | Image positions Dec | $\{\text{dec}_k\}_{k=1}^{N_{\text{img}}}$ | $(N_{\text{img}},)$ | $[-10, 10]$ | $y_{\text{image}}^{\text{true}} - 0.01$ |

### 2.2 Auxiliary/Derived Variables

| Variable | Symbol | Formula | Shape |
|----------|--------|---------|-------|
| Deflection angle | $\boldsymbol{\alpha}(\boldsymbol{x})$ | $\boldsymbol{\alpha} = \boldsymbol{\alpha}_{\text{EPL}} + \boldsymbol{\alpha}_{\text{shear}}$ | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ |
| Source position | $\boldsymbol{\beta}$ | $\boldsymbol{\beta} = \boldsymbol{x} - \boldsymbol{\alpha}(\boldsymbol{x})$ | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ |
| Magnification | $\mu_k$ | $\mu_k = \left[\det\left(\mathbf{I} - \frac{\partial \boldsymbol{\alpha}}{\partial \boldsymbol{x}}\right)\right]^{-1}$ | $(N_{\text{img}},)$ |
| Model image | $\mathbf{I}^{\text{model}}$ | Forward model output | $(N_{\text{pix}}, N_{\text{pix}})$ |
| Residual image | $\mathbf{R}$ | $\mathbf{R} = \mathbf{I}^{\text{obs}} - \mathbf{I}^{\text{model}}$ | $(N_{\text{pix}}, N_{\text{pix}})$ |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Background RMS | $\sigma_{\text{bkg}}$ | $0.5$ | Gaussian noise standard deviation |
| Exposure time | $t_{\text{exp}}$ | $100$ s | Integration time |
| Number of pixels | $N_{\text{pix}}$ | $100$ | Image dimension |
| Pixel scale | $\Delta_{\text{pix}}$ | $0.05$ arcsec/pixel | Angular size per pixel |
| PSF FWHM | FWHM | $0.2$ arcsec | Gaussian PSF width |
| PSF truncation | $n_{\text{trunc}}$ | $3$ | Truncation in units of $\sigma$ |
| PSO particles | $N_p$ | $50$ | Swarm size |
| PSO iterations | $N_{\text{iter}}^{\text{PSO}}$ | $10$ | PSO iterations |
| MCMC burn-in | $N_{\text{burn}}$ | $10$ | Burn-in steps |
| MCMC samples | $N_{\text{run}}$ | $10$ | Production samples |
| Walker ratio | $r_w$ | $4$ | Walkers per dimension |
| Image position uncertainty | $\sigma_{\text{pos}}$ | $0.004$ arcsec | Astrometric error |
| Source position tolerance | $\epsilon_{\text{src}}$ | $0.001$ arcsec | Lens equation solver tolerance |

### 2.4 Observed Data/Inputs

| Data | Symbol | Shape | dtype |
|------|--------|-------|-------|
| Observed image | $\mathbf{I}^{\text{obs}}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | `float64` |
| True image positions | $(x_{\text{image}}, y_{\text{image}})$ | $(N_{\text{img}},), (N_{\text{img}},)$ | `float64` |
| Coordinate transform matrix | $\mathbf{T}$ | $(2, 2)$ | `float64` |
| Reference coordinates | $(\text{RA}_0, \text{Dec}_0)$ | $(1,), (1,)$ | `float64` |

---

## 3. Preprocessing Pipeline

### Step 1: Coordinate System Initialization
**Input:** $N_{\text{pix}}$, $\Delta_{\text{pix}}$  
**Output:** $\mathbf{T}$, $\text{RA}_0$, $\text{Dec}_0$

$$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$

$$c_x = c_y = \frac{N_{\text{pix}} - 1}{2}$$

$$\text{RA}_0 = -(T_{00} \cdot c_x + T_{01} \cdot c_y)$$

$$\text{Dec}_0 = -(T_{10} \cdot c_x + T_{11} \cdot c_y)$$

### Step 2: PSF Construction
**Input:** FWHM, $\Delta_{\text{pix}}$, $n_{\text{trunc}}$  
**Output:** PSF kernel $\mathcal{P}$

$$\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$$

$$\mathcal{P}(x, y) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left(-\frac{x^2 + y^2}{2\sigma_{\text{PSF}}^2}\right)$$

Truncated at $|x|, |y| \leq n_{\text{trunc}} \cdot \sigma_{\text{PSF}}$

### Step 3: True Model Construction
**Input:** True parameters $\boldsymbol{\theta}^{\text{true}}$  
**Output:** Clean simulated image $\mathbf{I}^{\text{clean}}$

**3.1 EPL Deflection Angles:**
$$\boldsymbol{\alpha}_{\text{EPL}}(\boldsymbol{x}) = \theta_E \cdot f(\boldsymbol{x}; \gamma, e_1, e_2, x_c, y_c)$$

**3.2 Shear Deflection:**
$$\boldsymbol{\alpha}_{\text{shear}} = \begin{pmatrix} \gamma_1 (x - x_0) + \gamma_2 (y - y_0) \\ \gamma_2 (x - x_0) - \gamma_1 (y - y_0) \end{pmatrix}$$

**3.3 Sérsic Surface Brightness:**
$$I_{\text{Sérsic}}(r) = A \exp\left[-b_n \left(\left(\frac{r}{R_{\text{sersic}}}\right)^{1/n} - 1\right)\right]$$

where $b_n \approx 1.9992n - 0.3271$

### Step 4: Solve Lens Equation for Point Sources
**Input:** Source position $(\text{ra}_s, \text{dec}_s)$, lens parameters  
**Output:** Image positions $\{(x_k, y_k)\}_{k=1}^{N_{\text{img}}}$

Solve: $\boldsymbol{\beta}_s = \boldsymbol{x} - \boldsymbol{\alpha}(\boldsymbol{x})$ for $\boldsymbol{x}$

### Step 5: Compute Magnifications
**Input:** Image positions, lens parameters  
**Output:** Magnification factors $\{\mu_k\}$

$$\mu_k = \frac{1}{(1 - \kappa)^2 - \gamma_{\text{tot}}^2}$$

where $\kappa$ is convergence and $\gamma_{\text{tot}}$ is total shear at position $k$.

### Step 6: Add Noise
**Input:** $\mathbf{I}^{\text{clean}}$, $t_{\text{exp}}$, $\sigma_{\text{bkg}}$  
**Output:** $\mathbf{I}^{\text{obs}}$

$$\mathbf{I}^{\text{counts}} = \mathbf{I}^{\text{clean}} \cdot t_{\text{exp}}$$

$$I^{\text{counts}}_{ij} \leftarrow \max(0, I^{\text{counts}}_{ij})$$

$$\tilde{I}^{\text{counts}}_{ij} \sim \text{Poisson}(\lfloor I^{\text{counts}}_{ij} \rfloor)$$

$$\mathbf{I}^{\text{Poisson}} = \frac{\tilde{\mathbf{I}}^{\text{counts}}}{t_{\text{exp}}}$$

$$\mathbf{N}^{\text{bkg}}_{ij} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$$

$$\mathbf{I}^{\text{obs}} = \mathbf{I}^{\text{clean}} + (\mathbf{I}^{\text{Poisson}} - \mathbf{I}^{\text{clean}}) + \mathbf{N}^{\text{bkg}}$$

---

## 4. Algorithm Steps

### Phase 1: Particle Swarm Optimization (PSO)

#### Step 1.0: Initialization
**Input:** Parameter bounds $[\boldsymbol{\theta}^{\text{lower}}, \boldsymbol{\theta}^{\text{upper}}]$, initial guess $\boldsymbol{\theta}^{(0)}$, $\sigma_{\text{init}}$  
**Output:** Initial particle positions $\{\boldsymbol{x}_i^{(0)}\}_{i=1}^{N_p}$, velocities $\{\boldsymbol{v}_i^{(0)}\}$

$$\boldsymbol{x}_i^{(0)} \sim \mathcal{N}(\boldsymbol{\theta}^{(0)}, \text{diag}(\boldsymbol{\sigma}_{\text{init}}^2))$$

$$\boldsymbol{x}_i^{(0)} \leftarrow \text{clip}(\boldsymbol{x}_i^{(0)}, \boldsymbol{\theta}^{\text{lower}}, \boldsymbol{\theta}^{\text{upper}})$$

$$\boldsymbol{v}_i^{(0)} = \boldsymbol{0}$$

$$\boldsymbol{p}_i^{(0)} = \boldsymbol{x}_i^{(0)}, \quad \boldsymbol{g}^{(0)} = \arg\min_{\boldsymbol{x}_i} \mathcal{J}(\boldsymbol{x}_i^{(0)})$$

#### Step 1.1: Fitness Evaluation (per particle)
**Input:** Particle position $\boldsymbol{x}_i$  
**Output:** Fitness value $f_i = -\ln \mathcal{L}(\boldsymbol{x}_i)$

$$\mathbf{I}^{\text{model}} = \mathcal{F}(\boldsymbol{x}_i)$$

$$f_i = \frac{1}{2} \sum_{j,k} \frac{(I^{\text{obs}}_{jk} - I^{\text{model}}_{jk})^2}{\sigma_{jk}^2}$$

#### Step 1.2: Personal Best Update
**Input:** $\boldsymbol{x}_i^{(t)}$, $\boldsymbol{p}_i^{(t-1)}$, $f_i^{(t)}$, $f(\boldsymbol{p}_i^{(t-1)})$  
**Output:** $\boldsymbol{p}_i^{(t)}$

$$\boldsymbol{p}_i^{(t)} = \begin{cases} \boldsymbol{x}_i^{(t)} & \text{if } f_i^{(t)} < f(\boldsymbol{p}_i^{(t-1)}) \\ \boldsymbol{p}_i^{(t-1)} & \text{otherwise} \end{cases}$$

#### Step 1.3: Global Best Update
**Input:** $\{\boldsymbol{p}_i^{(t)}\}_{i=1}^{N_p}$  
**Output:** $\boldsymbol{g}^{(t)}$

$$\boldsymbol{g}^{(t)} = \arg\min_{\boldsymbol{p}_i^{(t)}} f(\boldsymbol{p}_i^{(t)})$$

#### Step 1.4: Velocity Update
**Input:** $\boldsymbol{v}_i^{(t-1)}$, $\boldsymbol{x}_i^{(t-1)}$, $\boldsymbol{p}_i^{(t)}$, $\boldsymbol{g}^{(t)}$  
**Output:** $\boldsymbol{v}_i^{(t)}$

$$\boldsymbol{v}_i^{(t)} = \omega \boldsymbol{v}_i^{(t-1)} + c_1 \boldsymbol{r}_1 \odot (\boldsymbol{p}_i^{(t)} - \boldsymbol{x}_i^{(t-1)}) + c_2 \boldsymbol{r}_2 \odot (\boldsymbol{g}^{(t)} - \boldsymbol{x}_i^{(t-1)})$$

where $\boldsymbol{r}_1, \boldsymbol{r}_2 \sim \mathcal{U}(0,1)^d$, $\omega$ is inertia weight, $c_1, c_2$ are cognitive/social coefficients.

#### Step 1.5: Position Update
**Input:** $\boldsymbol{x}_i^{(t-1)}$, $\boldsymbol{v}_i^{(t)}$  
**Output:** $\boldsymbol{x}_i^{(t)}$

$$\boldsymbol{x}_i^{(t)} = \boldsymbol{x}_i^{(t-1)} + \boldsymbol{v}_i^{(t)}$$

$$\boldsymbol{x}_i^{(t)} \leftarrow \text{clip}(\boldsymbol{x}_i^{(t)}, \boldsymbol{\theta}^{\text{lower}}, \boldsymbol{\theta}^{\text{upper}})$$

#### Step 1.6: Iterate
Repeat Steps 1.1–1.5 for $t = 1, \ldots, N_{\text{iter}}^{\text{PSO}}$

---

### Phase 2: Markov Chain Monte Carlo (MCMC) with Ensemble Sampler

#### Step 2.0: Initialization
**Input:** PSO best $\boldsymbol{g}^{\text{final}}$, $\sigma_{\text{scale}} = 0.1$, walker ratio $r_w = 4$  
**Output:** Initial walker positions $\{\boldsymbol{w}_j^{(0)}\}_{j=1}^{N_w}$

$$N_w = r_w \cdot d$$

$$\boldsymbol{w}_j^{(0)} \sim \mathcal{N}(\boldsymbol{g}^{\text{final}}, \sigma_{\text{scale}}^2 \cdot \text{diag}(\boldsymbol{\sigma}_{\text{init}}^2))$$

#### Step 2.1: Stretch Move Proposal
**Input:** Walker $\boldsymbol{w}_j^{(t-1)}$, complementary ensemble $\mathcal{S}_{-j}$  
**Output:** Proposal $\boldsymbol{w}_j'$

Select $\boldsymbol{w}_k$ uniformly from $\mathcal{S}_{-j}$

$$z \sim g(z) \propto \frac{1}{\sqrt{z}} \quad \text{for } z \in [1/a, a], \; a = 2$$

$$\boldsymbol{w}_j' = \boldsymbol{w}_k + z(\boldsymbol{w}_j^{(t-1)} - \boldsymbol{w}_k)$$

#### Step 2.2: Acceptance Probability
**Input:** $\boldsymbol{w}_j^{(t-1)}$, $\boldsymbol{w}_j'$, $z$  
**Output:** Accept/reject decision

$$\alpha = \min\left(1, z^{d-1} \frac{p(\boldsymbol{w}_j' | \mathbf{I}^{\text{obs}})}{p(\boldsymbol{w}_j^{(t-1)} | \mathbf{I}^{\text{obs}})}\right)$$

$$\boldsymbol{w}_j^{(t)} = \begin{cases} \boldsymbol{w}_j' & \text{with probability } \alpha \\ \boldsymbol{w}_j^{(t-1)} & \text{otherwise} \end{cases}$$

#### Step 2.3: Burn-in Phase
Repeat Steps 2.1–2.2 for $t = 1, \ldots, N_{\text{burn}}$, discarding samples.

#### Step 2.4: Production Phase
Repeat Steps 2.1–2.2 for $t = 1, \ldots, N_{\text{run}}$, storing samples.

#### Step 2.5: Extract Best Fit
**Input:** All stored samples $\{\boldsymbol{w}_j^{(t)}\}$  
**Output:** $\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{w}} p(\boldsymbol{w} | \mathbf{I}^{\text{obs}})$

---

## 5. Stopping Criteria

| Phase | Criterion | Condition |
|-------|-----------|-----------|
| **PSO** | Iteration limit | $t \geq N_{\text{iter}}^{\text{PSO}} = 10$ |
| **MCMC Burn-in** | Fixed iterations | $t \geq N_{\text{burn}} = 10$ |
| **MCMC Production** | Fixed iterations | $t \geq N_{\text{run}} = 10$ |
| **Bounds Check** | Feasibility | $\boldsymbol{\theta}^{\text{lower}} \leq \boldsymbol{\theta} \leq \boldsymbol{\theta}^{\text{upper}}$ |

---

## 6. Evaluation Metrics

### 6.1 Chi-Squared Statistic

$$\chi^2 = \sum_{i=1}^{N_{\text{pix}}} \sum_{j=1}^{N_{\text{pix}}} \frac{(I^{\text{obs}}_{ij} - I^{\text{model}}_{ij})^2}{\sigma_{\text{bkg}}^2}$$

### 6.2 Reduced Chi-Squared

$$\chi^2_{\nu} = \frac{\chi^2}{N_{\text{dof}}} = \frac{\chi^2}{N_{\text{pix}}^2 - 1}$$

### 6.3 Residual RMS

$$\text{RMS}_{\text{residual}} = \sqrt{\frac{1}{N_{\text{pix}}^2} \sum_{i,j} (I^{\text{obs}}_{ij} - I^{\text{model}}_{ij})^2}$$

### 6.4 Maximum Absolute Residual

$$\text{MaxRes} = \max_{i,j} |I^{\text{obs}}_{ij} - I^{\text{model}}_{ij}|$$

### 6.5 Parameter Residuals

For each parameter $\theta_k$:

$$\Delta\theta_k = \theta_k^{\text{fit}} - \theta_k^{\text{true}}$$

**Lens Parameters Evaluated:**
- $\Delta\theta_E$, $\Delta\gamma$, $\Delta e_1^{\text{lens}}$, $\Delta e_2^{\text{lens}}$, $\Delta x_c^{\text{lens}}$, $\Delta y_c^{\text{lens}}$

**Shear Parameters Evaluated:**
- $\Delta\gamma_1$, $\Delta\gamma_2$

---

## 7. Physical Model Specifications

### 7.1 Elliptical Power Law (EPL) Mass Profile

Convergence:
$$\kappa(x, y) = \frac{3 - \gamma}{2} \left(\frac{\theta_E}{\sqrt{q_m x^2 + y^2/q_m}}\right)^{\gamma - 1}$$

where $q_m$ is axis ratio derived from $(e_1, e_2)$:
$$q_m = \frac{1 - |e|}{1 + |e|}, \quad |e| = \sqrt{e_1^2 + e_2^2}$$

### 7.2 External Shear

$$\gamma_{\text{ext}} = \sqrt{\gamma_1^2 + \gamma_2^2}, \quad \phi_{\text{ext}} = \frac{1}{2}\arctan\left(\frac{\gamma_2}{\gamma_1}\right)$$

### 7.3 Sérsic Profile

$$I(R) = I_e \exp\left\{-b_n\left[\left(\frac{R}{R_e}\right)^{1/n} - 1\right]\right\}$$

$$b_n \approx 2n - \frac{1}{3} + \frac{4}{405n} + \frac{46}{25515n^2}$$