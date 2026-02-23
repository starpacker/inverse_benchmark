## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Trans-dimensional Bayesian MCMC Joint Inversion
- **algorithm_family**: Markov Chain Monte Carlo (Metropolis-Hastings with parallel tempering)
- **summary**: A trans-dimensional Bayesian inversion algorithm that jointly inverts surface wave dispersion and receiver function data to estimate subsurface shear velocity structure using multiple parallel MCMC chains with adaptive noise estimation.

## 1. OBJECTIVE FUNCTION

- **full_expression**:
$$\max_{m, \sigma} p(m, \sigma | d_{\text{SWD}}, d_{\text{RF}}) \propto p(d_{\text{SWD}}, d_{\text{RF}} | m, \sigma) \cdot p(m) \cdot p(\sigma)$$

- **data_fidelity_term** (Log-likelihood for joint targets):
$$\mathcal{L}(m, \sigma) = \mathcal{L}_{\text{SWD}}(m, \sigma_{\text{SWD}}) + \mathcal{L}_{\text{RF}}(m, \sigma_{\text{RF}})$$

For Surface Wave Dispersion (exponential/L1-like correlated noise):
$$\mathcal{L}_{\text{SWD}} = -\frac{1}{2} \mathbf{r}_{\text{SWD}}^T \mathbf{C}_{\text{SWD}}^{-1} \mathbf{r}_{\text{SWD}} - \frac{1}{2}\log|\mathbf{C}_{\text{SWD}}|$$
where $\mathbf{r}_{\text{SWD}} = d_{\text{SWD}}^{\text{obs}} - G_{\text{SWD}}(m)$ and $[\mathbf{C}_{\text{SWD}}]_{ij} = \sigma_{\text{SWD}}^2 \exp(-|i-j| / \rho_{\text{SWD}})$

For Receiver Function (Gaussian correlated noise):
$$\mathcal{L}_{\text{RF}} = -\frac{1}{2} \mathbf{r}_{\text{RF}}^T \mathbf{C}_{\text{RF}}^{-1} \mathbf{r}_{\text{RF}} - \frac{1}{2}\log|\mathbf{C}_{\text{RF}}|$$
where $\mathbf{r}_{\text{RF}} = d_{\text{RF}}^{\text{obs}} - G_{\text{RF}}(m)$ and $[\mathbf{C}_{\text{RF}}]_{ij} = \sigma_{\text{RF}}^2 \rho_{\text{RF}}^{|i-j|}$

- **regularization_term**: Implicit through prior distributions (see constraints)

- **constraints**:
$$\mathbb{I}_{[1.4, 2.1]}(v_p/v_s) \cdot \mathbb{I}_{[1, 20]}(n_{\text{layers}}) \cdot \mathbb{I}_{[2, 5]}(v_s) \cdot \mathbb{I}_{[0, 60]}(z) \cdot \mathbb{I}_{[10^{-5}, 0.05]}(\sigma_{\text{RF}}) \cdot \mathbb{I}_{[10^{-5}, 0.05]}(\sigma_{\text{SWD}})$$
Moho depth constraint: $z_{\text{Moho}} \sim \mathcal{N}(38, 4^2)$

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{h}$ | $(n_{\text{layers}},)$ | float64 | Random from prior | Layer thicknesses in km |
| $\mathbf{v}_s$ | $(n_{\text{layers}},)$ | float64 | Random from prior | Shear wave velocities in km/s |
| $v_p/v_s$ | scalar | float64 | Random from $[1.4, 2.1]$ | P-to-S velocity ratio |
| $n_{\text{layers}}$ | scalar | int | Random from $[1, 20]$ | Number of layers (trans-dimensional) |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| N/A | — | — | — | No explicit dual variables in MCMC formulation |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\rho_{\text{SWD}}$ | scalar | float64 | 0.0 | SWD noise correlation coefficient |
| $\sigma_{\text{SWD}}$ | scalar | float64 | Random from $[10^{-5}, 0.05]$ | SWD noise standard deviation |
| $\rho_{\text{RF}}$ | scalar | float64 | 0.98 | RF noise correlation coefficient |
| $\sigma_{\text{RF}}$ | scalar | float64 | Random from $[10^{-5}, 0.05]$ | RF noise standard deviation |
| $\mathbf{m}_{\text{proposed}}$ | varies | float64 | — | Proposed model state |
| $\mathcal{L}_{\text{current}}$ | scalar | float64 | Computed | Current log-likelihood |
| $\mathcal{L}_{\text{proposed}}$ | scalar | float64 | — | Proposed log-likelihood |
| $\alpha$ | scalar | float64 | — | Acceptance probability |
| $n_{\text{accepted}}$ | scalar | int | 0 | Acceptance counter per chain |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $n_{\text{chains}}$ | 5 | Number of parallel MCMC chains |
| $N_{\text{burnin}}$ | $2048 \times 32 = 65536$ | Burn-in iterations |
| $N_{\text{main}}$ | $2048 \times 16 = 32768$ | Main sampling iterations |
| $\Delta_h$ | 0.025 | Proposal step size for layer thickness |
| $\Delta_{v_s}$ | 0.025 | Proposal step size for velocity |
| $\Delta_{v_p/v_s}$ | 0.015 | Proposal step size for vp/vs ratio |
| $\Delta_{\sigma}$ | 0.005 | Proposal step size for noise parameters |
| $h_{\min}$ | 0.1 km | Minimum layer thickness |
| $\kappa$ | $10^{-5}$ | Regularization condition number (rcond) |
| $\mu_{\text{Moho}}$ | 38 km | Moho depth prior mean |
| $\sigma_{\text{Moho}}$ | 4 km | Moho depth prior std |
| $[\text{acc}_{\min}, \text{acc}_{\max}]$ | [40, 45] | Target acceptance rate range (%) |
| gauss | 1.0 | RF Gaussian filter width |
| water | 0.01 | RF water level stabilization |
| p | 6.4 s/deg | Ray parameter for RF |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{x}_{\text{SWD}}$ | $(21,)$ | float64 | Period values: $\text{linspace}(1, 41, 21)$ seconds |
| $d_{\text{SWD}}^{\text{obs}}$ | $(21,)$ | float64 | Observed Rayleigh phase velocities |
| $\mathbf{x}_{\text{RF}}$ | $(N_{\text{RF}},)$ | float64 | Time samples for receiver function |
| $d_{\text{RF}}^{\text{obs}}$ | $(N_{\text{RF}},)$ | float64 | Observed P receiver function amplitudes |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Generate synthetic SWD | $d_{\text{SWD}}^{\text{synth}} = G_{\text{SWD}}(\mathbf{h}_{\text{true}}, \mathbf{v}_{s,\text{true}}, (v_p/v_s)_{\text{true}})$ | $\mathbf{h}_{\text{true}}=[5,23,8,0]$, $\mathbf{v}_{s,\text{true}}=[2.7,3.6,3.8,4.4]$, $(v_p/v_s)_{\text{true}}=1.73$, $\mathbf{x}_{\text{SWD}}:(21,)$ | $d_{\text{SWD}}^{\text{synth}}:(21,)$ |
| 2 | Generate synthetic RF | $d_{\text{RF}}^{\text{synth}} = G_{\text{RF}}(\mathbf{h}_{\text{true}}, \mathbf{v}_{s,\text{true}}, (v_p/v_s)_{\text{true}})$ | $\mathbf{h}_{\text{true}}, \mathbf{v}_{s,\text{true}}, (v_p/v_s)_{\text{true}}$ | $d_{\text{RF}}^{\text{synth}}:(N_{\text{RF}},)$ |
| 3 | Compute SWD noise | $\boldsymbol{\epsilon}_{\text{SWD}} = \text{ExpNoise}(d_{\text{SWD}}^{\text{synth}}, \rho=0.0, \sigma=0.012)$ | $d_{\text{SWD}}^{\text{synth}}:(21,)$ | $\boldsymbol{\epsilon}_{\text{SWD}}:(21,)$ |
| 4 | Compute RF noise | $\boldsymbol{\epsilon}_{\text{RF}} = \text{GaussNoise}(d_{\text{RF}}^{\text{synth}}, \rho=0.98, \sigma=0.005)$ | $d_{\text{RF}}^{\text{synth}}:(N_{\text{RF}},)$ | $\boldsymbol{\epsilon}_{\text{RF}}:(N_{\text{RF}},)$ |
| 5 | Add noise to SWD | $d_{\text{SWD}}^{\text{obs}} = d_{\text{SWD}}^{\text{synth}} + \boldsymbol{\epsilon}_{\text{SWD}}$ | $d_{\text{SWD}}^{\text{synth}}:(21,)$, $\boldsymbol{\epsilon}_{\text{SWD}}:(21,)$ | $d_{\text{SWD}}^{\text{obs}}:(21,)$ |
| 6 | Add noise to RF | $d_{\text{RF}}^{\text{obs}} = d_{\text{RF}}^{\text{synth}} + \boldsymbol{\epsilon}_{\text{RF}}$ | $d_{\text{RF}}^{\text{synth}}:(N_{\text{RF}},)$, $\boldsymbol{\epsilon}_{\text{RF}}:(N_{\text{RF}},)$ | $d_{\text{RF}}^{\text{obs}}:(N_{\text{RF}},)$ |
| 7 | Compute true noise stats | $\boldsymbol{\sigma}_{\text{true}} = [\rho_{\text{SWD}}, \text{std}(\boldsymbol{\epsilon}_{\text{SWD}}), \rho_{\text{RF}}, \text{std}(\boldsymbol{\epsilon}_{\text{RF}})]$ | $\boldsymbol{\epsilon}_{\text{SWD}}$, $\boldsymbol{\epsilon}_{\text{RF}}$, $\rho_{\text{SWD}}=0.0$, $\rho_{\text{RF}}=0.98$ | $\boldsymbol{\sigma}_{\text{true}}:(4,)$ |
| 8 | Create target objects | $\mathcal{T}_1 = \text{RayleighDispersionPhase}(\mathbf{x}_{\text{SWD}}, d_{\text{SWD}}^{\text{obs}})$; $\mathcal{T}_2 = \text{PReceiverFunction}(\mathbf{x}_{\text{RF}}, d_{\text{RF}}^{\text{obs}})$ | Observed data, coordinates | $\mathcal{T}_{\text{joint}}$ |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize chains | For $c = 1, \ldots, n_{\text{chains}}$: allocate chain storage | $n_{\text{chains}}=5$ | Chain objects $(c_1, \ldots, c_5)$ | Parallel chain initialization |
| 2 | Sample initial layers | $n_{\text{layers}}^{(c)} \sim \mathcal{U}\{1, 20\}$ | Prior bounds | $n_{\text{layers}}^{(c)}$ for each chain | Integer uniform |
| 3 | Sample initial thicknesses | $h_i^{(c)} \sim \mathcal{U}(h_{\min}, z_{\max}/n_{\text{layers}})$ for $i=1,\ldots,n_{\text{layers}}$ with $\sum_i h_i \leq z_{\max}$ | $h_{\min}=0.1$, $z_{\max}=60$ | $\mathbf{h}^{(c)}:(n_{\text{layers}},)$ | Subject to depth constraint |
| 4 | Sample initial velocities | $v_{s,i}^{(c)} \sim \mathcal{U}(2.0, 5.0)$ for $i=1,\ldots,n_{\text{layers}}$ | Prior bounds $[2, 5]$ km/s | $\mathbf{v}_s^{(c)}:(n_{\text{layers}},)$ | — |
| 5 | Sample initial vp/vs | $(v_p/v_s)^{(c)} \sim \mathcal{U}(1.4, 2.1)$ | Prior bounds | $(v_p/v_s)^{(c)}$ | — |
| 6 | Sample initial noise SWD | $\sigma_{\text{SWD}}^{(c)} \sim \mathcal{U}(10^{-5}, 0.05)$ | Prior bounds | $\sigma_{\text{SWD}}^{(c)}$ | — |
| 7 | Sample initial noise RF | $\sigma_{\text{RF}}^{(c)} \sim \mathcal{U}(10^{-5}, 0.05)$ | Prior bounds | $\sigma_{\text{RF}}^{(c)}$ | — |
| 8 | Compute initial forward SWD | $\hat{d}_{\text{SWD}}^{(c)} = G_{\text{SWD}}(\mathbf{h}^{(c)}, \mathbf{v}_s^{(c)}, (v_p/v_s)^{(c)})$ | $\mathbf{h}^{(c)}, \mathbf{v}_s^{(c)}, (v_p/v_s)^{(c)}, \mathbf{x}_{\text{SWD}}$ | $\hat{d}_{\text{SWD}}^{(c)}:(21,)$ | [FORWARD] Surface wave modeling |
| 9 | Compute initial forward RF | $\hat{d}_{\text{RF}}^{(c)} = G_{\text{RF}}(\mathbf{h}^{(c)}, \mathbf{v}_s^{(c)}, (v_p/v_s)^{(c)})$ | $\mathbf{h}^{(c)}, \mathbf{v}_s^{(c)}, (v_p/v_s)^{(c)}, \mathbf{x}_{\text{RF}}$, $p=6.4$ | $\hat{d}_{\text{RF}}^{(c)}:(N_{\text{RF}},)$ | [FORWARD] RF modeling |
| 10 | Build SWD covariance | $[\mathbf{C}_{\text{SWD}}^{(c)}]_{ij} = (\sigma_{\text{SWD}}^{(c)})^2 \exp\left(-\frac{|i-j|}{\rho_{\text{SWD}}+\epsilon}\right)$ where $\epsilon=\kappa$ if $\rho_{\text{SWD}}=0$ | $\sigma_{\text{SWD}}^{(c)}$, $\rho_{\text{SWD}}=0$, $\kappa=10^{-5}$ | $\mathbf{C}_{\text{SWD}}^{(c)}:(21,21)$ | Regularized for $\rho=0$ |
| 11 | Build RF covariance | $[\mathbf{C}_{\text{RF}}^{(c)}]_{ij} = (\sigma_{\text{RF}}^{(c)})^2 (\rho_{\text{RF}})^{|i-j|}$ | $\sigma_{\text{RF}}^{(c)}$, $\rho_{\text{RF}}=0.98$ | $\mathbf{C}_{\text{RF}}^{(c)}:(N_{\text{RF}},N_{\text{RF}})$ | — |
| 12 | Compute initial likelihood | $\mathcal{L}^{(c)} = -\frac{1}{2}\mathbf{r}_{\text{SWD}}^T(\mathbf{C}_{\text{SWD}}^{(c)})^{-1}\mathbf{r}_{\text{SWD}} - \frac{1}{2}\log|\mathbf{C}_{\text{SWD}}^{(c)}| - \frac{1}{2}\mathbf{r}_{\text{RF}}^T(\mathbf{C}_{\text{RF}}^{(c)})^{-1}\mathbf{r}_{\text{RF}} - \frac{1}{2}\log|\mathbf{C}_{\text{RF}}^{(c)}|$ | Residuals, covariances | $\mathcal{L}_{\text{current}}^{(c)}$ | Uses rcond=$10^{-5}$ for inversion |
| 13 | Initialize counters | $n_{\text{accepted}}^{(c)} = 0$, $k^{(c)} = 0$ | — | Counters for each chain | — |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (over $k = 1, \ldots, N_{\text{burnin}} + N_{\text{main}}$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 14 | Select move type | $\tau \sim \mathcal{U}\{1, 2, 3, 4, 5\}$: 1=birth, 2=death, 3=move layer, 4=change velocity, 5=change noise | Random | $\tau$ | Trans-dimensional moves |
| 15 | **BEGIN_SUBLOOP** (birth move, $\tau=1$) | | | | |
| 16 | Check layer limit | If $n_{\text{layers}} \geq 20$: reject move | $n_{\text{layers}}$ | Continue/reject flag | — |
| 17 | Sample new layer depth | $z_{\text{new}} \sim \mathcal{U}(0, z_{\max})$ | $z_{\max}=60$ | $z_{\text{new}}$ | — |
| 18 | Sample new velocity | $v_{s,\text{new}} \sim \mathcal{U}(2.0, 5.0)$ | Prior bounds | $v_{s,\text{new}}$ | — |
| 19 | Insert layer | Update $\mathbf{h}', \mathbf{v}_s'$ by inserting at $z_{\text{new}}$; $n_{\text{layers}}' = n_{\text{layers}} + 1$ | $\mathbf{h}, \mathbf{v}_s, z_{\text{new}}, v_{s,\text{new}}$ | $\mathbf{h}', \mathbf{v}_s', n_{\text{layers}}'$ | — |
| 20 | **END_SUBLOOP** (birth) | | | | |
| 21 | **BEGIN_SUBLOOP** (death move, $\tau=2$) | | | | |
| 22 | Check layer minimum | If $n_{\text{layers}} \leq 1$: reject move | $n_{\text{layers}}$ | Continue/reject flag | — |
| 23 | Select layer to remove | $j \sim \mathcal{U}\{1, \ldots, n_{\text{layers}}\}$ | $n_{\text{layers}}$ | $j$ | — |
| 24 | Remove layer | Update $\mathbf{h}', \mathbf{v}_s'$ by removing layer $j$; $n_{\text{layers}}' = n_{\text{layers}} - 1$ | $\mathbf{h}, \mathbf{v}_s, j$ | $\mathbf{h}', \mathbf{v}_s', n_{\text{layers}}'$ | — |
| 25 | **END_SUBLOOP** (death) | | | | |
| 26 | **BEGIN_SUBLOOP** (perturb layer, $\tau=3$) | | | | |
| 27 | Select layer | $j \sim \mathcal{U}\{1, \ldots, n_{\text{layers}}\}$ | $n_{\text{layers}}$ | $j$ | — |
| 28 | Perturb thickness | $h_j' = h_j + \delta_h$ where $\delta_h \sim \mathcal{N}(0, \Delta_h^2)$, $\Delta_h = 0.025$ | $h_j$, $\Delta_h$ | $h_j'$ | — |
| 29 | Check thickness bounds | If $h_j' < h_{\min}=0.1$ or $\sum h' > z_{\max}=60$: reject | $h_j'$, $h_{\min}$, $z_{\max}$ | Valid/reject flag | — |
| 30 | **END_SUBLOOP** (perturb layer) | | | | |
| 31 | **BEGIN_SUBLOOP** (perturb velocity, $\tau=4$) | | | | |
| 32 | Select parameter | $\xi \sim \mathcal{U}\{1, 2\}$: 1=vs, 2=vpvs | Random | $\xi$ | — |
| 33 | Perturb vs (if $\xi=1$) | $j \sim \mathcal{U}\{1,\ldots,n_{\text{layers}}\}$; $v_{s,j}' = v_{s,j} + \delta_{v_s}$ where $\delta_{v_s} \sim \mathcal{N}(0, \Delta_{v_s}^2)$, $\Delta_{v_s}=0.025$ | $v_{s,j}$, $\Delta_{v_s}$ | $v_{s,j}'$ | — |
| 34 | Check vs bounds | If $v_{s,j}' \notin [2.0, 5.0]$: reject | $v_{s,j}'$ | Valid/reject flag | — |
| 35 | Perturb vpvs (if $\xi=2$) | $(v_p/v_s)' = (v_p/v_s) + \delta_{\text{vpvs}}$ where $\delta_{\text{vpvs}} \sim \mathcal{N}(0, \Delta_{v_p/v_s}^2)$, $\Delta_{v_p/v_s}=0.015$ | $(v_p/v_s)$, $\Delta_{v_p/v_s}$ | $(v_p/v_s)'$ | — |
| 36 | Check vpvs bounds | If $(v_p/v_s)' \notin [1.4, 2.1]$: reject | $(v_p/v_s)'$ | Valid/reject flag | — |
| 37 | **END_SUBLOOP** (perturb velocity) | | | | |
| 38 | **BEGIN_SUBLOOP** (perturb noise, $\tau=5$) | | | | |
| 39 | Select noise parameter | $\zeta \sim \mathcal{U}\{1, 2\}$: 1=SWD, 2=RF | Random | $\zeta$ | — |
| 40 | Perturb noise | $\sigma_{\zeta}' = \sigma_{\zeta} + \delta_\sigma$ where $\delta_\sigma \sim \mathcal{N}(0, \Delta_\sigma^2)$, $\Delta_\sigma=0.005$ | $\sigma_\zeta$, $\Delta_\sigma$ | $\sigma_\zeta'$ | — |
| 41 | Check noise bounds | If $\sigma_\zeta' \notin [10^{-5}, 0.05]$: reject | $\sigma_\zeta'$ | Valid/reject flag | — |
| 42 | **END_SUBLOOP** (perturb noise) | | | | |
| 43 | Compute proposed forward SWD | $\hat{d}_{\text{SWD}}' = G_{\text{SWD}}(\mathbf{h}', \mathbf{v}_s', (v_p/v_s)')$ | Proposed model, $\mathbf{x}_{\text{SWD}}$ | $\hat{d}_{\text{SWD}}':(21,)$ | [FORWARD] |
| 44 | Compute proposed forward RF | $\hat{d}_{\text{RF}}' = G_{\text{RF}}(\mathbf{h}', \mathbf{v}_s', (v_p/v_s)')$ | Proposed model, $\mathbf{x}_{\text{RF}}$ | $\hat{d}_{\text{RF}}':(N_{\text{RF}},)$ | [FORWARD] |
| 45 | Compute proposed residual SWD | $\mathbf{r}_{\text{SWD}}' = d_{\text{SWD}}^{\text{obs}} - \hat{d}_{\text{SWD}}'$ | $d_{\text{SWD}}^{\text{obs}}$, $\hat{d}_{\text{SWD}}'$ | $\mathbf{r}_{\text{SWD}}':(21,)$ | — |
| 46 | Compute proposed residual RF | $\mathbf{r}_{\text{RF}}' = d_{\text{RF}}^{\text{obs}} - \hat{d}_{\text{RF}}'$ | $d_{\text{RF}}^{\text{obs}}$, $\hat{d}_{\text{RF}}'$ | $\mathbf{r}_{\text{RF}}':(N_{\text{RF}},)$ | — |
| 47 | Update proposed covariances | $[\mathbf{C}_{\text{SWD}}']_{ij} = (\sigma_{\text{SWD}}')^2 \exp(-|i-j|/(\rho_{\text{SWD}}+\kappa))$; $[\mathbf{C}_{\text{RF}}']_{ij} = (\sigma_{\text{RF}}')^2 (\rho_{\text{RF}})^{|i-j|}$ | $\sigma_{\text{SWD}}'$, $\sigma_{\text{RF}}'$, $\rho_{\text{SWD}}$, $\rho_{\text{RF}}$ | $\mathbf{C}_{\text{SWD}}'$, $\mathbf{C}_{\text{RF}}'$ | — |
| 48 | Compute proposed likelihood | $\mathcal{L}' = -\frac{1}{2}(\mathbf{r}_{\text{SWD}}')^T(\mathbf{C}_{\text{SWD}}')^{-1}\mathbf{r}_{\text{SWD}}' - \frac{1}{2}\log|\mathbf{C}_{\text{SWD}}'| - \frac{1}{2}(\mathbf{r}_{\text{RF}}')^T(\mathbf{C}_{\text{RF}}')^{-1}\mathbf{r}_{\text{RF}}' - \frac{1}{2}\log|\mathbf{C}_{\text{RF}}'|$ | Residuals, covariances | $\mathcal{L}_{\text{proposed}}$ | Matrix inverse with rcond=$10^{-5}$ |
| 49 | Compute Moho prior | $z_{\text{Moho}}' = \sum_{i=1}^{n_{\text{layers}}'} h_i'$; $\log p_{\text{Moho}} = -\frac{(z_{\text{Moho}}' - 38)^2}{2 \cdot 4^2}$ | $\mathbf{h}'$, $\mu_{\text{Moho}}=38$, $\sigma_{\text{Moho}}=4$ | $\log p_{\text{Moho}}$ | — |
| 50 | Compute acceptance ratio | $\log \alpha = \mathcal{L}_{\text{proposed}} - \mathcal{L}_{\text{current}} + \log p_{\text{Moho}}' - \log p_{\text{Moho}} + \log q(\text{reverse})/q(\text{forward})$ | Likelihoods, priors | $\alpha$ | Proposal ratio for birth/death |
| 51 | Sample uniform | $u \sim \mathcal{U}(0, 1)$ | — |