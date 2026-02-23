## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: BayHunter MCMC Joint Inversion
- **algorithm_family**: Markov Chain Monte Carlo (MCMC) Bayesian Inversion / Metropolis-Hastings Sampling
- **summary**: Bayesian inversion algorithm that jointly inverts surface wave dispersion and receiver function seismic data to estimate subsurface shear-wave velocity structure using parallel-tempered MCMC sampling with trans-dimensional model parameterization.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\mathbf{m}} = \arg\max_{\mathbf{m}} P(\mathbf{m}|\mathbf{d}) = \arg\max_{\mathbf{m}} \left[ P(\mathbf{d}|\mathbf{m}) \cdot P(\mathbf{m}) \right]$$

where sampling is performed from the posterior $P(\mathbf{m}|\mathbf{d})$ via MCMC.

### data_fidelity_term
Joint log-likelihood for surface wave dispersion (SWD) and receiver function (RF):
$$\log P(\mathbf{d}|\mathbf{m}) = \log P(\mathbf{d}_{\text{SWD}}|\mathbf{m}) + \log P(\mathbf{d}_{\text{RF}}|\mathbf{m})$$

**SWD likelihood (exponentially-correlated noise):**
$$\log P(\mathbf{d}_{\text{SWD}}|\mathbf{m}) = -\frac{1}{2}\left(\mathbf{d}_{\text{SWD}} - \mathbf{G}_{\text{SWD}}(\mathbf{m})\right)^T \mathbf{C}_{\text{SWD}}^{-1} \left(\mathbf{d}_{\text{SWD}} - \mathbf{G}_{\text{SWD}}(\mathbf{m})\right) - \frac{1}{2}\log|\mathbf{C}_{\text{SWD}}|$$

where $[\mathbf{C}_{\text{SWD}}]_{ij} = \sigma_{\text{SWD}}^2 \exp\left(-\frac{|i-j|}{\ell_{\text{SWD}}}\right)$ with correlation $\rho_{\text{SWD}} = 0.0$

**RF likelihood (Gaussian-correlated noise):**
$$\log P(\mathbf{d}_{\text{RF}}|\mathbf{m}) = -\frac{1}{2}\left(\mathbf{d}_{\text{RF}} - \mathbf{G}_{\text{RF}}(\mathbf{m})\right)^T \mathbf{C}_{\text{RF}}^{-1} \left(\mathbf{d}_{\text{RF}} - \mathbf{G}_{\text{RF}}(\mathbf{m})\right) - \frac{1}{2}\log|\mathbf{C}_{\text{RF}}|$$

where $[\mathbf{C}_{\text{RF}}]_{ij} = \sigma_{\text{RF}}^2 \exp\left(-\frac{(i-j)^2}{2\ell_{\text{RF}}^2}\right)$ with correlation $\rho_{\text{RF}} = 0.98$

### regularization_term
Prior distribution $P(\mathbf{m})$:
$$\log P(\mathbf{m}) = \log P(n_{\text{lay}}) + \sum_{k=1}^{n_{\text{lay}}} \log P(v_{s,k}) + \sum_{k=1}^{n_{\text{lay}}-1} \log P(h_k) + \log P(v_p/v_s) + \log P(\boldsymbol{\sigma})$$

With Moho depth constraint:
$$\log P(z_{\text{Moho}}) = -\frac{(z_{\text{Moho}} - \mu_{\text{Moho}})^2}{2\sigma_{\text{Moho}}^2}, \quad \mu_{\text{Moho}} = 38 \text{ km}, \sigma_{\text{Moho}} = 4 \text{ km}$$

### constraints
- $n_{\text{lay}} \in [1, 20]$ (integer)
- $v_s \in [2.0, 5.0]$ km/s
- $z \in [0, 60]$ km
- $v_p/v_s \in [1.4, 2.1]$
- $h_k \geq h_{\min} = 0.1$ km (minimum layer thickness)
- $\sigma_{\text{RF}} \in [10^{-5}, 0.05]$
- $\sigma_{\text{SWD}} \in [10^{-5}, 0.05]$

## 2. VARIABLES

### Primal Variables (Model Parameters)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $n_{\text{lay}}$ | primal | scalar | int | Sampled from $U[1, 20]$ | Number of layers (trans-dimensional) |
| $\mathbf{h}$ | primal | $(n_{\text{lay}}-1,)$ | float64 | Sampled from prior | Layer thicknesses [km] |
| $\mathbf{v}_s$ | primal | $(n_{\text{lay}},)$ | float64 | Sampled from $U[2, 5]$ | Shear wave velocities [km/s] |
| $v_p/v_s$ | primal | scalar | float64 | Sampled from $U[1.4, 2.1]$ | P-to-S velocity ratio |
| $\sigma_{\text{SWD}}$ | primal | scalar | float64 | Sampled from $U[10^{-5}, 0.05]$ | SWD noise standard deviation |
| $\sigma_{\text{RF}}$ | primal | scalar | float64 | Sampled from $U[10^{-5}, 0.05]$ | RF noise standard deviation |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{x}_{\text{SWD}}$ | observations | $(21,)$ | float64 | `np.linspace(1, 41, 21)` | Periods for SWD [s] |
| $\mathbf{d}_{\text{SWD}}$ | observations | $(21,)$ | float64 | Loaded from file | Observed phase velocities [km/s] |
| $\mathbf{x}_{\text{RF}}$ | observations | $(N_{\text{RF}},)$ | float64 | Loaded from file | Time samples for RF [s] |
| $\mathbf{d}_{\text{RF}}$ | observations | $(N_{\text{RF}},)$ | float64 | Loaded from file | Observed RF amplitudes |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $N_{\text{chains}}$ | constants | scalar | int | 5 | Number of parallel MCMC chains |
| $N_{\text{burnin}}$ | constants | scalar | int | 65536 (2048×32) | Burn-in iterations per chain |
| $N_{\text{main}}$ | constants | scalar | int | 32768 (2048×16) | Main sampling iterations per chain |
| $\boldsymbol{\delta}$ | constants | $(5,)$ | float64 | $(0.025, 0.025, 0.015, 0.005, 0.005)$ | Proposal distribution widths |
| $[a_{\min}, a_{\max}]$ | constants | $(2,)$ | float64 | $[40, 45]$ | Target acceptance rate range [%] |
| $h_{\min}$ | constants | scalar | float64 | 0.1 | Minimum layer thickness [km] |
| $\kappa$ | constants | scalar | float64 | $10^{-5}$ | Regularization for matrix inversion (rcond) |
| $\rho_{\text{RF}}$ | constants | scalar | float64 | 0.98 | RF noise correlation parameter |
| $\rho_{\text{SWD}}$ | constants | scalar | float64 | 0.0 | SWD noise correlation parameter |
| $\mu_{\text{Moho}}$ | constants | scalar | float64 | 38.0 | Moho depth prior mean [km] |
| $\sigma_{\text{Moho}}$ | constants | scalar | float64 | 4.0 | Moho depth prior std [km] |
| $N_{\text{models}}^{\max}$ | constants | scalar | int | 50000 | Maximum models to store |
| $p$ | constants | scalar | float64 | 6.4 | Ray parameter for RF [s/deg] |
| $\alpha_{\text{gauss}}$ | constants | scalar | float64 | 1.0 | Gaussian filter width for RF |
| $w$ | constants | scalar | float64 | 0.01 | Water level for RF deconvolution |
| $N_{\text{threads}}$ | constants | scalar | int | 6 | Number of parallel threads |
| $\Delta t_{\text{send}}$ | constants | scalar | float64 | 1.0 | BayWatch update interval [s] |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{m}^{(k)}$ | auxiliary | variable | float64 | — | Current model state at iteration $k$ |
| $\mathbf{m}'$ | auxiliary | variable | float64 | — | Proposed model state |
| $\mathcal{L}^{(k)}$ | auxiliary | scalar | float64 | — | Current log-likelihood |
| $\mathcal{L}'$ | auxiliary | scalar | float64 | — | Proposed log-likelihood |
| $\alpha$ | auxiliary | scalar | float64 | — | Acceptance probability |
| $u$ | auxiliary | scalar | float64 | — | Uniform random sample $U[0,1]$ |

## 3. PREPROCESSING PIPELINE

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 1 | Generate true model | $\mathbf{h}_{\text{true}} = [5, 23, 8, 0]^T$, $\mathbf{v}_{s,\text{true}} = [2.7, 3.6, 3.8, 4.4]^T$, $(v_p/v_s)_{\text{true}} = 1.73$ | None | $\mathbf{h}_{\text{true}}$: $(4,)$, $\mathbf{v}_{s,\text{true}}$: $(4,)$ |
| 2 | Compute synthetic SWD | $\mathbf{d}_{\text{SWD}}^{\text{clean}} = \mathbf{G}_{\text{SWD}}(\mathbf{h}_{\text{true}}, \mathbf{v}_{s,\text{true}}, (v_p/v_s)_{\text{true}})$ | $\mathbf{x}_{\text{SWD}}$: $(21,)$, model params | $\mathbf{d}_{\text{SWD}}^{\text{clean}}$: $(21,)$ |
| 3 | Compute synthetic RF | $\mathbf{d}_{\text{RF}}^{\text{clean}} = \mathbf{G}_{\text{RF}}(\mathbf{h}_{\text{true}}, \mathbf{v}_{s,\text{true}}, (v_p/v_s)_{\text{true}}; p=6.4)$ | Model params | $\mathbf{d}_{\text{RF}}^{\text{clean}}$: $(N_{\text{RF}},)$ |
| 4 | Add SWD noise | $\boldsymbol{\epsilon}_{\text{SWD}} \sim \text{ExpNoise}(\rho=0.0, \sigma=0.012)$, $\mathbf{d}_{\text{SWD}} = \mathbf{d}_{\text{SWD}}^{\text{clean}} + \boldsymbol{\epsilon}_{\text{SWD}}$ | $\mathbf{d}_{\text{SWD}}^{\text{clean}}$: $(21,)$ | $\mathbf{d}_{\text{SWD}}$: $(21,)$, $\boldsymbol{\epsilon}_{\text{SWD}}$: $(21,)$ |
| 5 | Add RF noise | $\boldsymbol{\epsilon}_{\text{RF}} \sim \mathcal{N}(\mathbf{0}, \mathbf{C}_{\text{RF}})$ with $\rho=0.98$, $\sigma=0.005$, $\mathbf{d}_{\text{RF}} = \mathbf{d}_{\text{RF}}^{\text{clean}} + \boldsymbol{\epsilon}_{\text{RF}}$ | $\mathbf{d}_{\text{RF}}^{\text{clean}}$: $(N_{\text{RF}},)$ | $\mathbf{d}_{\text{RF}}$: $(N_{\text{RF}},)$, $\boldsymbol{\epsilon}_{\text{RF}}$: $(N_{\text{RF}},)$ |
| 6 | Compute true noise stats | $\boldsymbol{\theta}_{\text{noise}} = [\rho_{\text{SWD}}, \text{std}(\boldsymbol{\epsilon}_{\text{SWD}}), \rho_{\text{RF}}, \text{std}(\boldsymbol{\epsilon}_{\text{RF}})]^T$ | Noise realizations | $\boldsymbol{\theta}_{\text{noise}}$: $(4,)$ |
| 7 | Construct SWD target | Instantiate `RayleighDispersionPhase`$(\mathbf{x}_{\text{SWD}}, \mathbf{d}_{\text{SWD}}, \boldsymbol{\epsilon}_{\text{SWD}})$ | Data arrays | Target object |
| 8 | Construct RF target | Instantiate `PReceiverFunction`$(\mathbf{x}_{\text{RF}}, \mathbf{d}_{\text{RF}})$; set $\alpha_{\text{gauss}}=1.0$, $w=0.01$, $p=6.4$ | Data arrays | Target object |
| 9 | Create joint target | $\mathcal{T} = \mathcal{T}_{\text{SWD}} \cup \mathcal{T}_{\text{RF}}$ | Individual targets | Joint target object |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| I.1 | Initialize chains | For $c = 1, \ldots, N_{\text{chains}}$: draw $\mathbf{m}_c^{(0)} \sim P(\mathbf{m})$ | Prior bounds | $\{\mathbf{m}_c^{(0)}\}_{c=1}^{N_{\text{chains}}}$ | Random seed configurable |
| I.2 | Compute initial likelihoods | For each chain $c$: $\mathcal{L}_c^{(0)} = \log P(\mathbf{d}|\mathbf{m}_c^{(0)})$ | Initial models, observations | $\{\mathcal{L}_c^{(0)}\}_{c=1}^{N_{\text{chains}}}$ | Requires forward model evaluation |
| I.3 | Initialize acceptance counters | $A_c = 0$ for $c = 1, \ldots, N_{\text{chains}}$ | None | $\{A_c\}_{c=1}^{N_{\text{chains}}}$ | Track acceptance rates |
| I.4 | Initialize storage | Allocate arrays for posterior samples | $N_{\text{models}}^{\max}$ | Storage structures | Memory pre-allocation |

### 4.2 ITERATION LOOP

**--- BEGIN_LOOP (Burn-in Phase): $k = 1, \ldots, N_{\text{burnin}}$ ---**

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| B.1 | Propose new model | $\mathbf{m}' = \mathbf{m}^{(k-1)} + \boldsymbol{\xi}$, where $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \text{diag}(\boldsymbol{\delta}^2))$ | $\mathbf{m}^{(k-1)}$, $\boldsymbol{\delta}$ | $\mathbf{m}'$ | Trans-dimensional moves (birth/death) also possible |
| B.2 | Check prior bounds | If $\mathbf{m}' \notin \Omega_{\text{prior}}$: reject, go to B.7 | $\mathbf{m}'$, prior bounds | Accept/reject flag | Includes layer bounds, velocity bounds |
| B.3 | Compute forward models | $\mathbf{d}'_{\text{SWD}} = \mathbf{G}_{\text{SWD}}(\mathbf{m}')$, $\mathbf{d}'_{\text{RF}} = \mathbf{G}_{\text{RF}}(\mathbf{m}')$ | $\mathbf{m}'$ | $\mathbf{d}'_{\text{SWD}}$, $\mathbf{d}'_{\text{RF}}$ | Calls SynthObs routines |
| B.4 | Compute proposed likelihood | $\mathcal{L}' = \log P(\mathbf{d}_{\text{SWD}}|\mathbf{d}'_{\text{SWD}}, \sigma_{\text{SWD}}, \rho_{\text{SWD}}) + \log P(\mathbf{d}_{\text{RF}}|\mathbf{d}'_{\text{RF}}, \sigma_{\text{RF}}, \rho_{\text{RF}})$ | Predicted data, observations | $\mathcal{L}'$ | Uses correlated noise covariance |
| B.5 | Compute acceptance probability | $\alpha = \min\left(1, \exp(\mathcal{L}' - \mathcal{L}^{(k-1)})\right)$ | $\mathcal{L}'$, $\mathcal{L}^{(k-1)}$ | $\alpha$ | Metropolis-Hastings ratio |
| B.6 | Accept/reject decision | Draw $u \sim U[0,1]$; if $u < \alpha$: $\mathbf{m}^{(k)} = \mathbf{m}'$, $\mathcal{L}^{(k)} = \mathcal{L}'$, $A \leftarrow A + 1$ | $\alpha$, $u$ | $\mathbf{m}^{(k)}$, $\mathcal{L}^{(k)}$ | Update acceptance counter |
| B.7 | Reject case | Else: $\mathbf{m}^{(k)} = \mathbf{m}^{(k-1)}$, $\mathcal{L}^{(k)} = \mathcal{L}^{(k-1)}$ | Previous state | $\mathbf{m}^{(k)}$, $\mathcal{L}^{(k)}$ | Chain stays at current state |
| B.8 | Adapt proposal (optional) | If acceptance rate $\notin [a_{\min}, a_{\max}]$: adjust $\boldsymbol{\delta}$ | Acceptance stats, $[40, 45]$ | Updated $\boldsymbol{\delta}$ | Adaptive MCMC during burn-in |

**--- END_LOOP (Burn-in Phase) ---**

**--- BEGIN_LOOP (Main Sampling Phase): $k = 1, \ldots, N_{\text{main}}$ ---**

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| M.1 | Propose new model | $\mathbf{m}' = \mathbf{m}^{(k-1)} + \boldsymbol{\xi}$, $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \text{diag}(\boldsymbol{\delta}^2))$ | $\mathbf{m}^{(k-1)}$, $\boldsymbol{\delta}$ | $\mathbf{m}'$ | Fixed proposal after burn-in |
| M.2 | Check prior bounds | If $\mathbf{m}' \notin \Omega_{\text{prior}}$: reject, go to M.7 | $\mathbf{m}'$ | Accept/reject flag | Same as burn-in |
| M.3 | Compute forward models | $\mathbf{d}'_{\text{SWD}} = \mathbf{G}_{\text{SWD}}(\mathbf{m}')$, $\mathbf{d}'_{\text{RF}} = \mathbf{G}_{\text{RF}}(\mathbf{m}')$ | $\mathbf{m}'$ | Predicted data | Forward modeling |
| M.4 | Compute proposed likelihood | $\mathcal{L}' = \log P(\mathbf{d}|\mathbf{m}')$ | Predicted, observed | $\mathcal{L}'$ | Joint likelihood |
| M.5 | Compute acceptance probability | $\alpha = \min\left(1, \exp(\mathcal{L}' - \mathcal{L}^{(k-1)})\right)$ | Likelihoods | $\alpha$ | M-H acceptance |
| M.6 | Accept/reject and store | If $u < \alpha$: accept, store $\mathbf{m}^{(k)}$ in posterior samples | $\alpha$, $u$ | Posterior sample | Store accepted models |
| M.7 | Reject case | Else: $\mathbf{m}^{(k)} = \mathbf{m}^{(k-1)}$ (do not store duplicate) | — | — | Avoid duplicate storage |
| M.8 | Check storage limit | If stored models $> N_{\text{models}}^{\max}$: apply thinning | Count | — | maxmodels = 50000 |

**--- END_LOOP (Main Sampling Phase) ---**

### 4.3 FINALIZATION

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| F.1 | Combine chain outputs | $\mathcal{M}_{\text{posterior}} = \bigcup_{c=1}^{N_{\text{chains}}} \mathcal{M}_c$ | Chain samples | Combined samples | Parallel chain aggregation |
| F.2 | Outlier chain detection | Remove chains with $|\bar{\mathcal{L}}_c - \bar{\mathcal{L}}| > \text{dev} \cdot |\bar{\mathcal{L}}|$, dev = 0.05 | Chain likelihoods | Filtered samples | 5% deviation threshold |
| F.3 | Save final distribution | Store up to $N_{\text{models}}^{\max} = 100000$ models | Filtered samples | Saved distribution | `save_final_distribution` |
| F.4 | Generate diagnostic plots | Posterior distributions, data fits, velocity profiles | Samples, reference model | PDF plots | `save_plots` with refmodel |

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: 
  - Burn-in phase terminates when $k = N_{\text{burnin}} = 65536$
  - Main phase terminates when $k = N_{\text{main}} = 32768$
  - Total iterations per chain: $N_{\text{total}} = N_{\text{burnin}} + N_{\text{main}} = 98304$
- **parameters**:
  - `iter_burnin`: $65536$ ($2048 \times 32$)
  - `iter_main`: $32768$ ($2048 \times 16$)
  - `nchains`: $5$
  - `maxmodels`: $50000$ (storage), $100000$ (evaluation)
- **check_frequency**: Every iteration (implicit loop counter)

## 6. EVALUATION METRICS

| Metric Name | Formula | Description |
|-------------|---------|-------------|
| Log-likelihood | $\mathcal{L}(\mathbf{m}) = -\frac{1}{2}\sum_{t \in \{\text{SWD}, \text{RF}\}} (\mathbf{d}_t - \mathbf{G}_t(\mathbf{m}))^T \mathbf{C}_t^{-1} (\mathbf{d}_t - \mathbf{G}_t(\mathbf{m})) - \frac{1}{2}\sum_t \log|\mathbf{C}_t|$ | Posterior probability measure |
| Expected likelihood | $\mathcal{L}_{\text{exp}} = \texttt{compute\_explike}(\mathbf{d}_{\text{obs}}, \mathbf{d}_{\text{true}}, \boldsymbol{\theta}_{\text{noise}}, \kappa)$ | Theoretical best-fit likelihood given true noise |
| Acceptance rate | $r_{\text{acc}} = \frac{A}{N_{\text{iter}}} \times 100\%$ | Percentage of accepted proposals; target: $[40\%, 45\%]$ |
| Number of layers | $n_{\text{lay}}^{(k)}$ | Trans-dimensional parameter; true value = 3 |
| Chain convergence | $\text{dev}_c = \frac{|\bar{\mathcal{L}}_c - \bar{\mathcal{L}}|}{|\bar{\mathcal{L}}|}$ | Deviation from mean log-likelihood; threshold = 0.05 |
| SWD data misfit | $\chi^2_{\text{SWD}} = (\mathbf{d}_{\text{SWD}} - \mathbf{G}_{\text{SWD}}(\mathbf{m}))^T \mathbf{C}_{\text{SWD}}^{-1} (\mathbf{d}_{\text{SWD}} - \mathbf{G}_{\text{SWD}}(\mathbf{m}))$ | Weighted residual for surface waves |
| RF data misfit | $\chi^2_{\text{RF}} = (\mathbf{d}_{\text{RF}} - \mathbf{G}_{\text{RF}}(\mathbf{m}))^T \mathbf{C}_{\text{RF}}^{-1} (\mathbf{d}_{\text{RF}} - \mathbf{G}_{\text{RF}}(\mathbf{m}))$ | Weighted residual for receiver functions |