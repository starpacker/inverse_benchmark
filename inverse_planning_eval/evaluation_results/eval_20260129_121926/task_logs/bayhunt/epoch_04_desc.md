## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: BayHunter Joint Bayesian Inversion
- **algorithm_family**: Markov Chain Monte Carlo (MCMC) / Metropolis-Hastings Sampling
- **summary**: Transdimensional Bayesian MCMC inversion that jointly inverts surface wave dispersion and receiver function data to estimate subsurface shear wave velocity structure with uncertainty quantification.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\mathbf{m}^* = \arg\max_{\mathbf{m}} p(\mathbf{m} | \mathbf{d}) \propto p(\mathbf{d} | \mathbf{m}) \cdot p(\mathbf{m})$$

where the posterior is sampled via MCMC rather than directly optimized.

### data_fidelity_term
**Joint Log-Likelihood**:
$$\log p(\mathbf{d} | \mathbf{m}) = \log p(\mathbf{d}_{\text{SWD}} | \mathbf{m}) + \log p(\mathbf{d}_{\text{RF}} | \mathbf{m})$$

**Surface Wave Dispersion (non-Gaussian correlated noise)**:
$$\log p(\mathbf{d}_{\text{SWD}} | \mathbf{m}) = -\frac{1}{2} \left[ N \log(2\pi) + \log|\mathbf{C}_{\text{SWD}}| + (\mathbf{d}_{\text{SWD}} - \mathbf{G}_{\text{SWD}}(\mathbf{m}))^T \mathbf{C}_{\text{SWD}}^{-1} (\mathbf{d}_{\text{SWD}} - \mathbf{G}_{\text{SWD}}(\mathbf{m})) \right]$$

where $\mathbf{C}_{\text{SWD}}$ is the covariance matrix with correlation $\rho_{\text{SWD}}$ and standard deviation $\sigma_{\text{SWD}}$.

**Receiver Function (Gaussian correlated noise)**:
$$\log p(\mathbf{d}_{\text{RF}} | \mathbf{m}) = -\frac{1}{2} \left[ N \log(2\pi) + \log|\mathbf{C}_{\text{RF}}| + (\mathbf{d}_{\text{RF}} - \mathbf{G}_{\text{RF}}(\mathbf{m}))^T \mathbf{C}_{\text{RF}}^{-1} (\mathbf{d}_{\text{RF}} - \mathbf{G}_{\text{RF}}(\mathbf{m})) \right]$$

where $\mathbf{C}_{\text{RF}}$ is constructed with correlation $\rho_{\text{RF}} = 0.98$ and standard deviation $\sigma_{\text{RF}}$.

### regularization_term
**Prior Distribution (implicit regularization)**:
$$\log p(\mathbf{m}) = \log p(k) + \sum_{i=1}^{k} \log p(z_i) + \sum_{i=1}^{k+1} \log p(v_{s,i}) + \log p(\text{vpvs}) + \log p(\boldsymbol{\sigma})$$

with uniform priors on each parameter within specified bounds.

### constraints
$$\mathbf{m} \in \mathcal{M} = \{ \mathbf{m} : v_s \in [2, 5] \text{ km/s}, z \in [0, 60] \text{ km}, k \in [1, 20], \text{vpvs} \in [1.4, 2.1], h_{\min} \geq 0.1 \text{ km} \}$$

$$+ \mathbb{I}_{\text{moho}}(z_{\text{moho}} \sim \mathcal{N}(38, 4^2))$$

---

## 2. VARIABLES

### 2.1 Primal Variables (Model Parameters)

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $k$ | scalar | int | Sampled from $[1, 20]$ | Number of layers (transdimensional) |
| $\mathbf{h}$ | $(k,)$ | float64 | Random within prior | Layer thicknesses [km] |
| $\mathbf{v}_s$ | $(k+1,)$ | float64 | Random within prior | Shear wave velocities [km/s] |
| $\text{vpvs}$ | scalar | float64 | Random in $[1.4, 2.1]$ | Vp/Vs ratio |
| $\rho_{\text{SWD}}$ | scalar | float64 | Fixed at 0.0 | SWD noise correlation |
| $\sigma_{\text{SWD}}$ | scalar | float64 | Sampled in $[10^{-5}, 0.05]$ | SWD noise std |
| $\rho_{\text{RF}}$ | scalar | float64 | Fixed at 0.98 | RF noise correlation |
| $\sigma_{\text{RF}}$ | scalar | float64 | Sampled in $[10^{-5}, 0.05]$ | RF noise std |

### 2.2 Dual Variables
None (sampling-based method, no explicit dual formulation)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{m}_{\text{current}}$ | varies | dict | From prior | Current model state per chain |
| $\mathbf{m}_{\text{proposed}}$ | varies | dict | Perturbation | Proposed model state |
| $\mathcal{L}_{\text{current}}$ | scalar | float64 | Computed | Current log-likelihood |
| $\mathcal{L}_{\text{proposed}}$ | scalar | float64 | Computed | Proposed log-likelihood |
| $\alpha$ | scalar | float64 | Computed | Acceptance probability |
| $u$ | scalar | float64 | $\sim U(0,1)$ | Random draw for acceptance |
| chain_id | scalar | int | $\{0, \ldots, n_{\text{chains}}-1\}$ | Chain identifier |
| iter_count | scalar | int | 0 | Iteration counter |
| acceptance_rate | scalar | float64 | 0 | Running acceptance rate |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $n_{\text{chains}}$ | 5 | Number of parallel MCMC chains |
| $N_{\text{burnin}}$ | $2048 \times 32 = 65536$ | Burn-in iterations |
| $N_{\text{main}}$ | $2048 \times 16 = 32768$ | Main sampling iterations |
| $\boldsymbol{\delta}$ | $(0.025, 0.025, 0.015, 0.005, 0.005)$ | Proposal distribution widths |
| acceptance_target | $[40, 45]$ | Target acceptance rate [%] |
| $h_{\min}$ | 0.1 | Minimum layer thickness [km] |
| rcond | $10^{-5}$ | Regularization for covariance inversion |
| maxmodels | 50000 | Maximum models to store |
| $\mu_{\text{moho}}$ | 38 | Moho depth prior mean [km] |
| $\sigma_{\text{moho}}$ | 4 | Moho depth prior std [km] |
| gauss_rf | 1.0 | Gaussian width for RF computation |
| water_rf | 0.01 | Water level for RF deconvolution |
| $p$ | 6.4 | Ray parameter [s/deg] |
| nthreads | 6 | Number of parallel threads |
| dev | 0.05 | Outlier chain detection threshold |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{x}_{\text{SWD}}$ | $(21,)$ | float64 | Periods for surface wave dispersion [s] |
| $\mathbf{d}_{\text{SWD}}$ | $(21,)$ | float64 | Observed phase velocities [km/s] |
| $\mathbf{x}_{\text{RF}}$ | $(N_{\text{RF}},)$ | float64 | Time samples for receiver function [s] |
| $\mathbf{d}_{\text{RF}}$ | $(N_{\text{RF}},)$ | float64 | Observed receiver function amplitudes |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Define True Model | $\mathbf{h}_{\text{true}} = [5, 23, 8, 0]$, $\mathbf{v}_{s,\text{true}} = [2.7, 3.6, 3.8, 4.4]$, $\text{vpvs}_{\text{true}} = 1.73$ | None | $\mathbf{h}_{\text{true}}$ $(4,)$, $\mathbf{v}_{s,\text{true}}$ $(4,)$ |
| 1 | Generate SWD Grid | $\mathbf{x}_{\text{SWD}} = \text{linspace}(1, 41, 21)$ | None | $\mathbf{x}_{\text{SWD}}$ $(21,)$ |
| 2 | Compute Synthetic SWD | $\mathbf{d}_{\text{SWD,synth}} = \mathbf{G}_{\text{SWD}}(\mathbf{h}_{\text{true}}, \mathbf{v}_{s,\text{true}}, \text{vpvs}_{\text{true}})$ | $\mathbf{h}_{\text{true}}$, $\mathbf{v}_{s,\text{true}}$, $\mathbf{x}_{\text{SWD}}$ | $\mathbf{d}_{\text{SWD,synth}}$ $(21,)$ |
| 3 | Compute Synthetic RF | $\mathbf{d}_{\text{RF,synth}} = \mathbf{G}_{\text{RF}}(\mathbf{h}_{\text{true}}, \mathbf{v}_{s,\text{true}}, \text{vpvs}_{\text{true}}; p=6.4)$ | $\mathbf{h}_{\text{true}}$, $\mathbf{v}_{s,\text{true}}$ | $\mathbf{d}_{\text{RF,synth}}$ $(N_{\text{RF}},)$, $\mathbf{x}_{\text{RF}}$ |
| 4 | Add SWD Noise | $\boldsymbol{\epsilon}_{\text{SWD}} \sim \text{ExpNoise}(\rho=0.0, \sigma=0.012)$; $\mathbf{d}_{\text{SWD}} = \mathbf{d}_{\text{SWD,synth}} + \boldsymbol{\epsilon}_{\text{SWD}}$ | $\mathbf{d}_{\text{SWD,synth}}$ | $\mathbf{d}_{\text{SWD}}$ $(21,)$, $\boldsymbol{\epsilon}_{\text{SWD}}$ |
| 5 | Add RF Noise | $\boldsymbol{\epsilon}_{\text{RF}} \sim \mathcal{N}(0, \mathbf{C}_{\text{RF}})$ with $\rho=0.98$, $\sigma=0.005$; $\mathbf{d}_{\text{RF}} = \mathbf{d}_{\text{RF,synth}} + \boldsymbol{\epsilon}_{\text{RF}}$ | $\mathbf{d}_{\text{RF,synth}}$ | $\mathbf{d}_{\text{RF}}$ $(N_{\text{RF}},)$, $\boldsymbol{\epsilon}_{\text{RF}}$ |
| 6 | Compute True Noise Stats | $\boldsymbol{\sigma}_{\text{true}} = [\rho_{\text{SWD}}, \text{std}(\boldsymbol{\epsilon}_{\text{SWD}}), \rho_{\text{RF}}, \text{std}(\boldsymbol{\epsilon}_{\text{RF}})]$ | $\boldsymbol{\epsilon}_{\text{SWD}}$, $\boldsymbol{\epsilon}_{\text{RF}}$ | $\boldsymbol{\sigma}_{\text{true}}$ $(4,)$ |
| 7 | Compute Expected Likelihood | $\mathcal{L}_{\text{exp}} = \text{compute\_explike}([\mathbf{d}_{\text{SWD}}, \mathbf{d}_{\text{RF}}], [\mathbf{d}_{\text{SWD,synth}}, \mathbf{d}_{\text{RF,synth}}], \boldsymbol{\sigma}_{\text{true}})$ | Observations, synthetics, noise | $\mathcal{L}_{\text{exp}}$ scalar |
| 8 | Build Depth Profile | $\mathbf{z}_{\text{plot}} = \text{cumsum}(\mathbf{h}_{\text{true}})$; staircase representation | $\mathbf{h}_{\text{true}}$, $\mathbf{v}_{s,\text{true}}$ | $(\mathbf{z}_{\text{plot}}, \mathbf{v}_{s,\text{plot}})$ |
| 9 | Initialize Targets | $\text{target}_1 = \text{RayleighDispersionPhase}(\mathbf{x}_{\text{SWD}}, \mathbf{d}_{\text{SWD}})$; $\text{target}_2 = \text{PReceiverFunction}(\mathbf{x}_{\text{RF}}, \mathbf{d}_{\text{RF}})$ | Observed data | Joint target object |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Load priors | Parse config.ini for bounds: $v_s \in [2,5]$, $z \in [0,60]$, $k \in [1,20]$, $\text{vpvs} \in [1.4, 2.1]$ | config.ini | priors dict | File I/O |
| Set proposal widths | $\boldsymbol{\delta} = (0.025, 0.025, 0.015, 0.005, 0.005)$ for $(v_s, z, \text{vpvs}, \sigma_{\text{SWD}}, \sigma_{\text{RF}})$ | config | propdist tuple | |
| Initialize chains | For $c = 0, \ldots, n_{\text{chains}}-1$: sample $\mathbf{m}^{(c)}_0 \sim p(\mathbf{m})$ uniformly from priors | priors | $\{\mathbf{m}^{(c)}_0\}_{c=0}^{n_{\text{chains}}-1}$ | Parallel initialization |
| Compute initial likelihood | $\mathcal{L}^{(c)}_0 = \log p(\mathbf{d} | \mathbf{m}^{(c)}_0)$ for each chain | $\mathbf{m}^{(c)}_0$, data | $\{\mathcal{L}^{(c)}_0\}$ | Forward model evaluation |
| Set random seed | If `random_seed=None`, use system entropy | None | RNG state | Reproducibility |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (Burn-in Phase: $t = 1, \ldots, N_{\text{burnin}}$)

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Select move type | Choose from: (1) perturb $v_s$, (2) perturb $z$, (3) birth (add layer), (4) death (remove layer), (5) perturb vpvs, (6) perturb noise | $u \sim U(0,1)$ | move_type | Transdimensional moves |
| Generate proposal | $\mathbf{m}' = \mathbf{m}_t + \boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{\delta}^2)$ for continuous; dimension change for birth/death | $\mathbf{m}_t$, $\boldsymbol{\delta}$ | $\mathbf{m}'$ | Move-specific perturbation |
| Check prior bounds | If $\mathbf{m}' \notin \mathcal{M}$: reject immediately, set $\mathbf{m}_{t+1} = \mathbf{m}_t$ | $\mathbf{m}'$, priors | accept/reject flag | Boundary enforcement |
| Forward model SWD [FORWARD] | $\hat{\mathbf{d}}_{\text{SWD}} = \mathbf{G}_{\text{SWD}}(\mathbf{h}', \mathbf{v}'_s, \text{vpvs}'; \mathbf{x}_{\text{SWD}})$ | $\mathbf{m}'$ | $\hat{\mathbf{d}}_{\text{SWD}}$ $(21,)$ | Rayleigh wave dispersion computation |
| Forward model RF [FORWARD] | $\hat{\mathbf{d}}_{\text{RF}} = \mathbf{G}_{\text{RF}}(\mathbf{h}', \mathbf{v}'_s, \text{vpvs}'; p=6.4, \text{gauss}=1.0, \text{water}=0.01)$ | $\mathbf{m}'$ | $\hat{\mathbf{d}}_{\text{RF}}$ $(N_{\text{RF}},)$ | P-wave receiver function |
| Compute residuals | $\mathbf{r}_{\text{SWD}} = \mathbf{d}_{\text{SWD}} - \hat{\mathbf{d}}_{\text{SWD}}$; $\mathbf{r}_{\text{RF}} = \mathbf{d}_{\text{RF}} - \hat{\mathbf{d}}_{\text{RF}}$ | Observed, predicted | $\mathbf{r}_{\text{SWD}}$, $\mathbf{r}_{\text{RF}}$ | Data misfit |
| Build covariance matrices | $[\mathbf{C}_{\text{SWD}}]_{ij} = \sigma_{\text{SWD}}^2 \rho_{\text{SWD}}^{|i-j|}$; $[\mathbf{C}_{\text{RF}}]_{ij} = \sigma_{\text{RF}}^2 \rho_{\text{RF}}^{|i-j|}$ | $\rho$, $\sigma$ from $\mathbf{m}'$ | $\mathbf{C}_{\text{SWD}}$, $\mathbf{C}_{\text{RF}}$ | Exponential covariance |
| Compute proposed likelihood | $\mathcal{L}' = -\frac{1}{2}\left[\mathbf{r}_{\text{SWD}}^T \mathbf{C}_{\text{SWD}}^{-1} \mathbf{r}_{\text{SWD}} + \mathbf{r}_{\text{RF}}^T \mathbf{C}_{\text{RF}}^{-1} \mathbf{r}_{\text{RF}} + \log|\mathbf{C}_{\text{SWD}}| + \log|\mathbf{C}_{\text{RF}}|\right]$ | $\mathbf{r}$, $\mathbf{C}$ | $\mathcal{L}'$ | rcond=$10^{-5}$ for inversion |
| Compute acceptance ratio | $\alpha = \min\left(1, \exp(\mathcal{L}' - \mathcal{L}_t) \cdot \frac{p(\mathbf{m}')}{p(\mathbf{m}_t)} \cdot \frac{q(\mathbf{m}_t | \mathbf{m}')}{q(\mathbf{m}' | \mathbf{m}_t)}\right)$ | $\mathcal{L}'$, $\mathcal{L}_t$, priors | $\alpha \in [0,1]$ | Includes Jacobian for dimension changes |
| Accept/reject | Draw $u \sim U(0,1)$; if $u < \alpha$: $\mathbf{m}_{t+1} = \mathbf{m}'$, $\mathcal{L}_{t+1} = \mathcal{L}'$; else: $\mathbf{m}_{t+1} = \mathbf{m}_t$ | $\alpha$, $u$ | $\mathbf{m}_{t+1}$, $\mathcal{L}_{t+1}$ | Metropolis-Hastings step |
| Update acceptance counter | $n_{\text{accept}} \mathrel{+}= \mathbb{I}[u < \alpha]$ | accept flag | $n_{\text{accept}}$ | For adaptive tuning |

**END_LOOP** (Burn-in)

**BEGIN_LOOP** (Main Sampling Phase: $t = 1, \ldots, N_{\text{main}}$)

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| [Repeat all burn-in steps] | Same as burn-in loop | | | |
| Store accepted model | If accepted: append $\mathbf{m}_{t+1}$ to model chain storage | $\mathbf{m}_{t+1}$ | Updated storage | Limit to maxmodels=50000 |
| Store likelihood | Append $\mathcal{L}_{t+1}$ to likelihood chain | $\mathcal{L}_{t+1}$ | Updated likelihood trace | For convergence diagnostics |

**END_LOOP** (Main Sampling)

### 4.3 FINALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Aggregate chains | Combine samples from all $n_{\text{chains}}$ chains | Chain storages | Combined posterior samples | |
| Detect outlier chains | Flag chains with mean likelihood deviating $> 5\%$ from median | Likelihood traces | Valid chain indices | dev=0.05 threshold |
| Save final distribution | Export valid models (up to maxmodels=100000) | Valid samples | HDF5/pickle files | Disk I/O |
| Compute posterior statistics | Mean, median, credible intervals for each parameter | Posterior samples | Summary statistics | |
| Generate plots | Velocity profiles, marginal distributions, data fits | Samples, data | PDF figures | matplotlib backend |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: 
$$t_{\text{total}} = N_{\text{burnin}} + N_{\text{main}} = 65536 + 32768 = 98304 \text{ iterations per chain}$$
- **parameters**:
  - `iter_burnin`: $2048 \times 32 = 65536$
  - `iter_main`: $2048 \times 16 = 32768$
  - `nchains`: 5
  - Total forward model evaluations: $5 \times 98304 = 491520$
- **check_frequency**: Every iteration (implicit via loop bounds)

---

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| Log-Likelihood | $\mathcal{L}(\mathbf{m}) = \log p(\mathbf{d}_{\text{SWD}} | \mathbf{m}) + \log p(\mathbf{d}_{\text{RF}} | \mathbf{m})$ | Joint data fit quality |
| Expected Likelihood | $\mathcal{L}_{\text{exp}} = \text{compute\_explike}(\mathbf{d}_{\text{obs}}, \mathbf{d}_{\text{true}}, \boldsymbol{\sigma}_{\text{true}})$ | Theoretical optimum likelihood given true model |
| Acceptance Rate | $r_{\text{accept}} = \frac{n_{\text{accept}}}{N_{\text{iter}}} \times 100\%$ | Target: $40\%-45\%$ |
| Number of Layers Recovery | $\hat{k} = \text{mode}(\{k^{(i)}\}_{i=1}^{N_{\text{samples}}})$ vs $k_{\text{true}} = 3$ | Transdimensional accuracy |
| SWD RMS Misfit | $\text{RMS}_{\text{SWD}} = \sqrt{\frac{1}{N_{\text{SWD}}} \sum_{i=1}^{N_{\text{SWD}}} (d_{\text{SWD},i} - \hat{d}_{\text{SWD},i})^2}$ | Surface wave data fit |
| RF RMS Misfit | $\text{RMS}_{\text{RF}} = \sqrt{\frac{1}{N_{\text{RF}}} \sum_{i=1}^{N_{\text{RF}}} (d_{\text{RF},i} - \hat{d}_{\text{RF},i})^2}$ | Receiver function data fit |
| Outlier Chain Fraction | $f_{\text{outlier}} = \frac{n_{\text{outlier}}}{n_{\text{chains}}}$ | Chains failing dev=0.05 criterion |

---

## 7. REQUIRED OUTPUT FIELDS

```
algorithm_identification.algorithm_name
algorithm_identification.algorithm_family
algorithm_identification.summary
objective_function.full_expression
objective_function.data_fidelity_term
objective_function.regularization_term
objective_function.constraints
variables.primal_variables
variables.auxiliary_variables
variables.constants
variables.observations
data_preprocessing.steps[]
initialization.steps[]
iteration_loop.burnin_steps[]
iteration_loop.main_steps[]
finalization.steps[]
stopping_criterion.type
stopping_criterion.expression
stopping_criterion.parameters
evaluation.metrics[]
hyperparameters.nchains
hyperparameters.iter_burnin
hyperparameters.iter_main
hyperparameters.proposal_distribution
hyperparameters.acceptance_target
hyperparameters.rcond
hyperparameters.maxmodels
hyperparameters.outlier_dev
output.savepath
output.configfile
output.status
```