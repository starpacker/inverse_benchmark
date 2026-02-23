# Algorithmic Specification Document: Bayesian Seismic Inversion via MCMC

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\hat{m} = \arg\max_{m} P(m | d_{\text{swd}}, d_{\text{rf}}) \propto P(d_{\text{swd}}, d_{\text{rf}} | m) \cdot P(m)$$

Equivalently, the MCMC sampler targets the posterior distribution:
$$P(m | d) \propto \exp\left(-\mathcal{L}(m)\right) \cdot P(m)$$

where the negative log-likelihood (misfit function) is:
$$\mathcal{L}(m) = \mathcal{L}_{\text{swd}}(m) + \mathcal{L}_{\text{rf}}(m)$$

**data_fidelity_term**:

For Surface Wave Dispersion (exponential/correlated noise model):
$$\mathcal{L}_{\text{swd}}(m) = \frac{1}{2} \mathbf{r}_{\text{swd}}^T \mathbf{C}_{\text{swd}}^{-1} \mathbf{r}_{\text{swd}} + \frac{1}{2}\log|\mathbf{C}_{\text{swd}}|$$

where $\mathbf{r}_{\text{swd}} = \mathbf{d}_{\text{swd}}^{\text{obs}} - G_{\text{swd}}(m)$

For Receiver Function (Gaussian noise model):
$$\mathcal{L}_{\text{rf}}(m) = \frac{1}{2} \mathbf{r}_{\text{rf}}^T \mathbf{C}_{\text{rf}}^{-1} \mathbf{r}_{\text{rf}} + \frac{1}{2}\log|\mathbf{C}_{\text{rf}}|$$

where $\mathbf{r}_{\text{rf}} = \mathbf{d}_{\text{rf}}^{\text{obs}} - G_{\text{rf}}(m)$

**Covariance matrices**:
$$[\mathbf{C}]_{ij} = \sigma^2 \cdot \rho^{|i-j|}$$

where $\rho$ is the correlation coefficient and $\sigma$ is the noise standard deviation.

**regularization_term**:
Implicit through prior distribution $P(m)$ (uniform priors on bounded domains).

**constraints**:
| Constraint | Expression |
|------------|------------|
| Vp/Vs ratio bounds | $1.4 \leq v_{p}/v_{s} \leq 2.1$ |
| Number of layers | $1 \leq n_{\text{layers}} \leq 20$ |
| Shear velocity bounds | $2.0 \leq v_s \leq 5.0$ km/s |
| Depth bounds | $0 \leq z \leq 60$ km |
| Minimum layer thickness | $h_i \geq 0.1$ km |
| RF noise correlation | $\rho_{\text{rf}} = 0.98$ (fixed) |
| SWD noise correlation | $\rho_{\text{swd}} = 0.0$ (fixed) |
| RF noise sigma bounds | $10^{-5} \leq \sigma_{\text{rf}} \leq 0.05$ |
| SWD noise sigma bounds | $10^{-5} \leq \sigma_{\text{swd}} \leq 0.05$ |
| Moho depth constraint | $z_{\text{moho}} \sim \mathcal{N}(38, 4^2)$ km |

---

## 2. VARIABLES

### Primal Variables (Model Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{h}$ | primal | $(n_{\text{layers}},)$ | float64 | Random within prior | Layer thicknesses (km) |
| $\mathbf{v}_s$ | primal | $(n_{\text{layers}},)$ | float64 | Random within prior | Shear wave velocities (km/s) |
| $v_p/v_s$ | primal | scalar | float64 | Random in $[1.4, 2.1]$ | P-wave to S-wave velocity ratio |
| $n_{\text{layers}}$ | primal | scalar | int | Random in $[1, 20]$ | Number of layers (trans-dimensional) |
| $\sigma_{\text{swd}}$ | primal | scalar | float64 | Random within prior | SWD noise standard deviation |
| $\sigma_{\text{rf}}$ | primal | scalar | float64 | Random within prior | RF noise standard deviation |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $n_{\text{chains}}$ | constants | scalar | int | 5 | Number of parallel MCMC chains |
| $N_{\text{burnin}}$ | constants | scalar | int | 65536 | Burn-in iterations per chain |
| $N_{\text{main}}$ | constants | scalar | int | 32768 | Main sampling iterations per chain |
| $\boldsymbol{\delta}$ | constants | $(5,)$ | float64 | $(0.025, 0.025, 0.015, 0.005, 0.005)$ | Proposal distribution widths |
| $[\alpha_{\min}, \alpha_{\max}]$ | constants | $(2,)$ | float64 | $[40, 45]$ | Target acceptance rate (%) |
| $h_{\min}$ | constants | scalar | float64 | 0.1 | Minimum layer thickness (km) |
| $\kappa$ | constants | scalar | float64 | $10^{-5}$ | Regularization cutoff for matrix inversion |
| $\rho_{\text{swd}}$ | constants | scalar | float64 | 0.0 | SWD noise correlation coefficient |
| $\rho_{\text{rf}}$ | constants | scalar | float64 | 0.98 | RF noise correlation coefficient |
| $(\mu_{\text{moho}}, \sigma_{\text{moho}})$ | constants | $(2,)$ | float64 | $(38, 4)$ | Moho depth prior (mean, std) |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{x}_{\text{swd}}$ | observations | $(N_{\text{swd}},)$ | float64 | Loaded from file | SWD periods (s) |
| $\mathbf{d}_{\text{swd}}^{\text{obs}}$ | observations | $(N_{\text{swd}},)$ | float64 | Loaded from file | Observed phase velocities (km/s) |
| $\mathbf{x}_{\text{rf}}$ | observations | $(N_{\text{rf}},)$ | float64 | Loaded from file | RF time samples (s) |
| $\mathbf{d}_{\text{rf}}^{\text{obs}}$ | observations | $(N_{\text{rf}},)$ | float64 | Loaded from file | Observed RF amplitudes |
| $p$ | observations | scalar | float64 | 6.4 | Ray parameter (s/deg) |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Configuration Loading

| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load configuration parameters |
| formula | $(\text{priors}, \text{initparams}) = \texttt{parse\_config}(\text{config.ini})$ |
| inputs | config.ini file (text) |
| outputs | priors: dict, initparams: dict |

### Step 3.2: Synthetic Forward Model Generation

| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Generate synthetic reference data |
| formula | $\mathbf{d}_{\text{swd}}^{\text{syn}} = G_{\text{swd}}(\mathbf{h}^{\text{true}}, \mathbf{v}_s^{\text{true}}, (v_p/v_s)^{\text{true}}, \mathbf{x}_{\text{swd}})$ |
| | $\mathbf{d}_{\text{rf}}^{\text{syn}} = G_{\text{rf}}(\mathbf{h}^{\text{true}}, \mathbf{v}_s^{\text{true}}, (v_p/v_s)^{\text{true}}, \mathbf{x}_{\text{rf}})$ |
| inputs | $\mathbf{h}^{\text{true}} = [5, 23, 8, 0]$ km, $\mathbf{v}_s^{\text{true}} = [2.7, 3.6, 3.8, 4.4]$ km/s, $(v_p/v_s)^{\text{true}} = 1.73$, $\mathbf{x}_{\text{swd}} \in \mathbb{R}^{21}$, $\mathbf{x}_{\text{rf}} \in \mathbb{R}^{N_{\text{rf}}}$ |
| outputs | $\mathbf{d}_{\text{swd}}^{\text{syn}} \in \mathbb{R}^{21}$, $\mathbf{d}_{\text{rf}}^{\text{syn}} \in \mathbb{R}^{N_{\text{rf}}}$ |

### Step 3.3: Generate SWD Period Vector

| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Create surface wave period sampling |
| formula | $[\mathbf{x}_{\text{swd}}]_i = 1 + \frac{40(i-1)}{20}, \quad i = 1, \ldots, 21$ |
| inputs | None |
| outputs | $\mathbf{x}_{\text{swd}} \in \mathbb{R}^{21}$, range $[1, 41]$ s |

### Step 3.4: Add Exponential Correlated Noise to SWD

| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Generate correlated noise for SWD |
| formula | $\boldsymbol{\epsilon}_{\text{swd}} = \texttt{compute\_expnoise}(\mathbf{d}_{\text{swd}}^{\text{syn}}, \rho=0.0, \sigma=0.012)$ |
| | $\mathbf{d}_{\text{swd}}^{\text{obs}} = \mathbf{d}_{\text{swd}}^{\text{syn}} + \boldsymbol{\epsilon}_{\text{swd}}$ |
| inputs | $\mathbf{d}_{\text{swd}}^{\text{syn}} \in \mathbb{R}^{21}$, $\rho_{\text{swd}}=0.0$, $\sigma_{\text{swd}}=0.012$ |
| outputs | $\boldsymbol{\epsilon}_{\text{swd}} \in \mathbb{R}^{21}$, $\mathbf{d}_{\text{swd}}^{\text{obs}} \in \mathbb{R}^{21}$ |

### Step 3.5: Add Gaussian Correlated Noise to RF

| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Generate correlated Gaussian noise for RF |
| formula | $\boldsymbol{\epsilon}_{\text{rf}} = \texttt{compute\_gaussnoise}(\mathbf{d}_{\text{rf}}^{\text{syn}}, \rho=0.98, \sigma=0.005)$ |
| | $\mathbf{d}_{\text{rf}}^{\text{obs}} = \mathbf{d}_{\text{rf}}^{\text{syn}} + \boldsymbol{\epsilon}_{\text{rf}}$ |
| inputs | $\mathbf{d}_{\text{rf}}^{\text{syn}} \in \mathbb{R}^{N_{\text{rf}}}$, $\rho_{\text{rf}}=0.98$, $\sigma_{\text{rf}}=0.005$ |
| outputs | $\boldsymbol{\epsilon}_{\text{rf}} \in \mathbb{R}^{N_{\text{rf}}}$, $\mathbf{d}_{\text{rf}}^{\text{obs}} \in \mathbb{R}^{N_{\text{rf}}}$ |

### Step 3.6: Compute True Noise Vector

| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Assemble noise parameter vector |
| formula | $\boldsymbol{\eta}^{\text{true}} = [\rho_{\text{swd}}, \text{std}(\boldsymbol{\epsilon}_{\text{swd}}), \rho_{\text{rf}}, \text{std}(\boldsymbol{\epsilon}_{\text{rf}})]^T$ |
| inputs | $\boldsymbol{\epsilon}_{\text{swd}}$, $\boldsymbol{\epsilon}_{\text{rf}}$, $\rho_{\text{swd}}=0.0$, $\rho_{\text{rf}}=0.98$ |
| outputs | $\boldsymbol{\eta}^{\text{true}} \in \mathbb{R}^{4}$ |

### Step 3.7: Compute Expected Likelihood

| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Compute reference likelihood value |
| formula | $\mathcal{L}^{\text{exp}} = \texttt{compute\_explike}([\mathbf{d}_{\text{swd}}^{\text{obs}}, \mathbf{d}_{\text{rf}}^{\text{obs}}], [\mathbf{d}_{\text{swd}}^{\text{syn}}, \mathbf{d}_{\text{rf}}^{\text{syn}}], \boldsymbol{\eta}^{\text{true}}, [\text{False}, \text{True}], \kappa)$ |
| inputs | Observed data, synthetic data, noise parameters, Gaussian flags, rcond=$10^{-5}$ |
| outputs | $\mathcal{L}^{\text{exp}}$: scalar (reference log-likelihood) |

### Step 3.8: Construct Depth-Velocity Profile

| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | Convert layer model to depth profile |
| formula | $z_i = \sum_{j=1}^{i} h_j, \quad i = 1, \ldots, n_{\text{layers}}$ |
| | $\mathbf{z}_{\text{plot}} = \texttt{repeat}([z_1, \ldots, z_{n}], 2)[1:] \cup \{150\}$ |
| | $\mathbf{v}_{s,\text{plot}} = \texttt{repeat}([v_{s,1}, \ldots, v_{s,n}], 2)$ |
| inputs | $\mathbf{h} \in \mathbb{R}^{4}$, $\mathbf{v}_s \in \mathbb{R}^{4}$ |
| outputs | $\mathbf{z}_{\text{plot}} \in \mathbb{R}^{2n}$, $\mathbf{v}_{s,\text{plot}} \in \mathbb{R}^{2n}$ (staircase representation) |

### Step 3.9: Initialize Target Objects

| Field | Value |
|-------|-------|
| step_order | 9 |
| step_name | Create observation target objects |
| formula | $\mathcal{T}_1 = \texttt{RayleighDispersionPhase}(\mathbf{x}_{\text{swd}}, \mathbf{d}_{\text{swd}}^{\text{obs}}, \boldsymbol{\epsilon}_{\text{swd}})$ |
| | $\mathcal{T}_2 = \texttt{PReceiverFunction}(\mathbf{x}_{\text{rf}}, \mathbf{d}_{\text{rf}}^{\text{obs}})$ |
| | $\mathcal{T} = \texttt{JointTarget}([\mathcal{T}_1, \mathcal{T}_2])$ |
| inputs | Observed data vectors, period/time vectors, noise estimates |
| outputs | Joint target object $\mathcal{T}$ |

### Step 3.10: Set RF Model Parameters

| Field | Value |
|-------|-------|
| step_order | 10 |
| step_name | Configure receiver function forward model |
| formula | $\mathcal{T}_2.\text{params} \leftarrow \{\text{gauss}: 1.0, \text{water}: 0.01, p: 6.4\}$ |
| inputs | Gaussian filter width, water level, ray parameter |
| outputs | Configured RF target object |

---

## 4. ALGORITHM STEPS

### Initialization

**--- BEGIN INITIALIZATION ---**

#### Step I.1: Initialize Random Number Generator

| Field | Value |
|-------|-------|
| step_name | Set random seed |
| formula | $\texttt{RNG} \leftarrow \texttt{RandomState}(\text{seed}=\text{None})$ |
| inputs | Optional seed value |
| outputs | Initialized RNG state |
| computational_note | None implies system-derived seed |

#### Step I.2: Initialize MCMC Chains

| Field | Value |
|-------|-------|
| step_name | Create parallel chain ensemble |
| formula | For $c = 1, \ldots, n_{\text{chains}}$: |
| | $n_{\text{layers}}^{(c,0)} \sim \mathcal{U}\{1, 20\}$ |
| | $\mathbf{h}^{(c,0)} \sim \mathcal{U}(0, 60/n_{\text{layers}})^{n_{\text{layers}}}$ subject to $h_i \geq h_{\min}$ |
| | $\mathbf{v}_s^{(c,0)} \sim \mathcal{U}(2, 5)^{n_{\text{layers}}}$ |
| | $(v_p/v_s)^{(c,0)} \sim \mathcal{U}(1.4, 2.1)$ |
| | $\sigma_{\text{swd}}^{(c,0)} \sim \mathcal{U}(10^{-5}, 0.05)$ |
| | $\sigma_{\text{rf}}^{(c,0)} \sim \mathcal{U}(10^{-5}, 0.05)$ |
| inputs | Prior bounds, $n_{\text{chains}}=5$ |
| outputs | Initial model states $\{m^{(c,0)}\}_{c=1}^{5}$ |
| computational_note | Uniform distributions on specified domains |

#### Step I.3: Compute Initial Likelihoods

| Field | Value |
|-------|-------|
| step_name | Evaluate initial chain likelihoods |
| formula | For $c = 1, \ldots, n_{\text{chains}}$: |
| | $\mathcal{L}^{(c,0)} = \mathcal{L}_{\text{swd}}(m^{(c,0)}) + \mathcal{L}_{\text{rf}}(m^{(c,0)})$ |
| inputs | Initial models $\{m^{(c,0)}\}$ |
| outputs | Initial likelihoods $\{\mathcal{L}^{(c,0)}\}$ |
| computational_note | Requires forward model evaluation |

#### Step I.4: Initialize Acceptance Counters

| Field | Value |
|-------|-------|
| step_name | Set up acceptance tracking |
| formula | $A_{\text{accept}}^{(c)} \leftarrow 0, \quad A_{\text{total}}^{(c)} \leftarrow 0, \quad \forall c$ |
| inputs | None |
| outputs | Acceptance counters per chain |
| computational_note | Used for acceptance rate monitoring |

**--- END INITIALIZATION ---**

### Iteration Loop

**--- BEGIN ITERATION LOOP ---**

For $k = 1, \ldots, N_{\text{burnin}} + N_{\text{main}}$:

#### Step L.1: Select Proposal Type

| Field | Value |
|-------|-------|
| step_name | Choose perturbation mechanism |
| formula | $\tau \sim \texttt{Categorical}(\{$layer birth, layer death, move interface, change $v_s$, change $v_p/v_s$, change noise$\})$ |
| inputs | Current iteration $k$ |
| outputs | Proposal type $\tau$ |
| computational_note | Trans-dimensional moves (birth/death) modify $n_{\text{layers}}$ |

#### Step L.2: Generate Proposal (Layer Property Perturbation)

| Field | Value |
|-------|-------|
| step_name | Perturb model parameters |
| formula | If $\tau = $ "change $v_s$": |
| | $j \sim \mathcal{U}\{1, n_{\text{layers}}\}$ |
| | $v_{s,j}' = v_{s,j} + \delta_1 \cdot \xi, \quad \xi \sim \mathcal{N}(0,1)$ |
| | If $\tau = $ "change $v_p/v_s$": |
| | $(v_p/v_s)' = (v_p/v_s) + \delta_2 \cdot \xi$ |
| | If $\tau = $ "move interface": |
| | $h_j' = h_j + \delta_3 \cdot \xi$ |
| | If $\tau = $ "change noise": |
| | $\sigma' = \sigma + \delta_4 \cdot \xi$ or $\delta_5 \cdot \xi$ |
| inputs | Current model $m^{(c,k-1)}$, proposal widths $\boldsymbol{\delta} = (0.025, 0.025, 0.015, 0.005, 0.005)$ |
| outputs | Proposed model $m'^{(c)}$ |
| computational_note | $\delta$ indices correspond to different parameter types |

#### Step L.3: Generate Proposal (Birth Move)

| Field | Value |
|-------|-------|
| step_name | Add new layer |
| formula | $z_{\text{new}} \sim \mathcal{U}(0, z_{\max})$ |
| | $v_{s,\text{new}} = v_s(z_{\text{new}}) + \delta_1 \cdot \xi$ (interpolated + perturbation) |
| | $n_{\text{layers}}' = n_{\text{layers}} + 1$ |
| | Insert layer at depth $z_{\text{new}}$, update $\mathbf{h}'$, $\mathbf{v}_s'$ |
| inputs | Current model, depth bounds |
| outputs | Proposed model with additional layer |
| computational_note | Birth proposal includes Jacobian term in acceptance |

#### Step L.4: Generate Proposal (Death Move)

| Field | Value |
|-------|-------|
| step_name | Remove existing layer |
| formula | $j \sim \mathcal{U}\{1, n_{\text{layers}}\}$ (select layer to remove) |
| | $n_{\text{layers}}' = n_{\text{layers}} - 1$ |
| | Merge layers $j$ and $j+1$: $h_j' = h_j + h_{j+1}$ |
| inputs | Current model |
| outputs | Proposed model with one fewer layer |
| computational_note | Death proposal includes inverse Jacobian term |

#### Step L.5: Check Prior Bounds

| Field | Value |
|-------|-------|
| step_name | Validate proposal against priors |
| formula | $\text{valid} = \mathbb{1}\left[m' \in \Omega_{\text{prior}}\right]$ |
| | where $\Omega_{\text{prior}} = \{1.4 \leq v_p/v_s \leq 2.1\} \cap \{2 \leq v_s \leq 5\} \cap \{h_i \geq 0.1\} \cap \ldots$ |
| inputs | Proposed model $m'$, prior bounds |
| outputs | Boolean validity flag |
| computational_note | If invalid, proposal is rejected immediately |

#### Step L.6: Compute Forward Model (SWD)

| Field | Value |
|-------|-------|
| step_name | Surface wave dispersion forward model |
| formula | $\mathbf{d}_{\text{swd}}^{\text{pred}} = G_{\text{swd}}(\mathbf{h}', \mathbf{v}_s', (v_p/v_s)', \mathbf{x}_{\text{swd}})$ |
| | where $G_{\text{swd}}$ solves the secular equation for Rayleigh wave phase velocities |
| inputs | Proposed model parameters, period vector |
| outputs | Predicted phase velocities $\mathbf{d}_{\text{swd}}^{\text{pred}} \in \mathbb{R}^{21}$ |
| computational_note | Uses propagator matrix method for layered media |

#### Step L.7: Compute Forward Model (RF)

| Field | Value |
|-------|-------|
| step_name | Receiver function forward model |
| formula | $\mathbf{d}_{\text{rf}}^{\text{pred}} = G_{\text{rf}}(\mathbf{h}', \mathbf{v}_s', (v_p/v_s)', \mathbf{x}_{\text{rf}}, p)$ |
| | where $G_{\text{rf}}$ computes P-to-S converted phases via reflectivity |
| inputs | Proposed model parameters, time vector, ray parameter $p=6.4$ s/deg |
| outputs | Predicted RF amplitudes $\mathbf{d}_{\text{rf}}^{\text{pred}} \in \mathbb{R}^{N_{\text{rf}}}$ |
| computational_note | Includes Gaussian filter (width=1.0) and water level (0.01) deconvolution |

#### Step L.8: Construct Covariance Matrix (SWD)

| Field | Value |
|-------|-------|
| step_name | Build SWD noise covariance |
| formula | $[\mathbf{C}_{\text{swd}}]_{ij} = (\sigma_{\text{swd}}')^2 \cdot (\rho_{\text{swd}})^{|i-j|}$ |
| | For $\rho_{\text{swd}} = 0$: $\mathbf{C}_{\text{swd}} = (\sigma_{\text{swd}}')^2 \mathbf{I}$ |
| inputs | Proposed noise $\sigma_{\text{swd}}'$, correlation $\rho_{\text{swd}}=0$, dimension $N_{\text{swd}}$ |
| outputs | $\mathbf{C}_{\text{swd}} \in \mathbb{R}^{21 \times 21}$ |
| computational_note | Diagonal matrix when correlation is zero |

#### Step L.9: Construct Covariance Matrix (RF)

| Field | Value |
|-------|-------|
| step_name | Build RF noise covariance |
| formula | $[\mathbf{C}_{\text{rf}}]_{ij} = (\sigma_{\text{rf}}')^2 \cdot (\rho_{\text{rf}})^{