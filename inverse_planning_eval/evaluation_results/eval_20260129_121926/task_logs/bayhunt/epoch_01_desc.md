# Algorithmic Specification Document: Transdimensional Bayesian Inversion for Joint Seismological Data

## 1. Mathematical Formulation

### 1.1 Objective Function (Posterior Distribution)

The algorithm seeks to sample from the posterior distribution over Earth velocity structure models given observed seismological data:

$$
p(\mathbf{m}, k, \boldsymbol{\theta}_n | \mathbf{d}_{obs}) \propto p(\mathbf{d}_{obs} | \mathbf{m}, k, \boldsymbol{\theta}_n) \cdot p(\mathbf{m}, k) \cdot p(\boldsymbol{\theta}_n)
$$

Where:
- $\mathbf{m} = \{\mathbf{h}, \mathbf{v}_s, v_{p}/v_{s}\}$ — model parameters
- $k$ — number of layers (transdimensional parameter)
- $\boldsymbol{\theta}_n$ — noise hyperparameters
- $\mathbf{d}_{obs}$ — observed data (joint SWD and RF)

### 1.2 Joint Likelihood Function

The joint log-likelihood decomposes as:

$$
\log \mathcal{L}(\mathbf{d}_{obs} | \mathbf{m}, k, \boldsymbol{\theta}_n) = \log \mathcal{L}_{SWD} + \log \mathcal{L}_{RF}
$$

**Surface Wave Dispersion (SWD) Likelihood** — Exponentially-correlated noise model:

$$
\log \mathcal{L}_{SWD} = -\frac{1}{2} \left[ N_{swd} \log(2\pi) + \log|\mathbf{C}_{swd}| + (\mathbf{d}_{swd}^{obs} - \mathbf{d}_{swd}^{pred})^T \mathbf{C}_{swd}^{-1} (\mathbf{d}_{swd}^{obs} - \mathbf{d}_{swd}^{pred}) \right]
$$

Where the covariance matrix has elements:
$$
[\mathbf{C}_{swd}]_{ij} = \sigma_{swd}^2 \exp\left(-\frac{|x_i - x_j|}{L_{swd}}\right) \cdot \rho_{swd}^{|i-j|}
$$

**Receiver Function (RF) Likelihood** — Gaussian-correlated noise model:

$$
\log \mathcal{L}_{RF} = -\frac{1}{2} \left[ N_{rf} \log(2\pi) + \log|\mathbf{C}_{rf}| + (\mathbf{d}_{rf}^{obs} - \mathbf{d}_{rf}^{pred})^T \mathbf{C}_{rf}^{-1} (\mathbf{d}_{rf}^{obs} - \mathbf{d}_{rf}^{pred}) \right]
$$

Where:
$$
[\mathbf{C}_{rf}]_{ij} = \sigma_{rf}^2 \exp\left(-\frac{(x_i - x_j)^2}{2L_{rf}^2}\right) \cdot \rho_{rf}^{|i-j|}
$$

### 1.3 Prior Distributions

| Parameter | Prior Distribution | Bounds/Parameters |
|-----------|-------------------|-------------------|
| $k$ (number of layers) | Uniform | $k \in [1, 20]$ |
| $v_{p}/v_{s}$ | Uniform | $v_{p}/v_{s} \in [1.4, 2.1]$ |
| $v_{s,i}$ (shear velocity) | Uniform | $v_{s,i} \in [2.0, 5.0]$ km/s |
| $z_i$ (interface depth) | Uniform | $z_i \in [0, 60]$ km |
| $\rho_{swd}$ (SWD noise correlation) | Fixed | $\rho_{swd} = 0.0$ |
| $\rho_{rf}$ (RF noise correlation) | Fixed | $\rho_{rf} = 0.98$ |
| $\sigma_{swd}$ | Log-uniform | $\sigma_{swd} \in [10^{-5}, 0.05]$ |
| $\sigma_{rf}$ | Log-uniform | $\sigma_{rf} \in [10^{-5}, 0.05]$ |

**Moho Depth Prior** (informative):
$$
p(z_{moho}) = \mathcal{N}(38, 4^2) \text{ km}
$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables (Sampled by MCMC)

| Variable | Symbol | Shape | Dtype | Domain | Description |
|----------|--------|-------|-------|--------|-------------|
| Layer thicknesses | $\mathbf{h} = [h_1, ..., h_k]$ | $(k,)$ | `float64` | $h_i > h_{min} = 0.1$ km | Thickness of each layer |
| Shear velocities | $\mathbf{v}_s = [v_{s,1}, ..., v_{s,k+1}]$ | $(k+1,)$ | `float64` | $[2.0, 5.0]$ km/s | Vs in each layer |
| Vp/Vs ratio | $v_p/v_s$ | scalar | `float64` | $[1.4, 2.1]$ | Constant for all layers |
| Number of layers | $k$ | scalar | `int` | $[1, 20]$ | Transdimensional parameter |
| SWD noise sigma | $\sigma_{swd}$ | scalar | `float64` | $[10^{-5}, 0.05]$ | SWD noise standard deviation |
| RF noise sigma | $\sigma_{rf}$ | scalar | `float64` | $[10^{-5}, 0.05]$ | RF noise standard deviation |

### 2.2 Auxiliary/Derived Variables

| Variable | Symbol | Computation |
|----------|--------|-------------|
| Interface depths | $\mathbf{z} = [z_1, ..., z_k]$ | $z_i = \sum_{j=1}^{i} h_j$ |
| P-wave velocities | $\mathbf{v}_p$ | $v_{p,i} = (v_p/v_s) \cdot v_{s,i}$ |
| Predicted SWD | $\mathbf{d}_{swd}^{pred}$ | $\mathcal{F}_{swd}(\mathbf{h}, \mathbf{v}_s, v_p/v_s)$ |
| Predicted RF | $\mathbf{d}_{rf}^{pred}$ | $\mathcal{F}_{rf}(\mathbf{h}, \mathbf{v}_s, v_p/v_s)$ |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Number of chains | $N_{chains}$ | 5 | Parallel MCMC chains |
| Burn-in iterations | $N_{burnin}$ | $2048 \times 32 = 65536$ | Per chain |
| Main iterations | $N_{main}$ | $2048 \times 16 = 32768$ | Per chain |
| Proposal std (Vs) | $\delta_{vs}$ | 0.025 | Gaussian proposal width |
| Proposal std (depth) | $\delta_z$ | 0.025 | Gaussian proposal width |
| Proposal std (vpvs) | $\delta_{vpvs}$ | 0.015 | Gaussian proposal width |
| Proposal std (noise) | $\delta_{\sigma}$ | 0.005 | Gaussian proposal width |
| Target acceptance rate | $[a_{min}, a_{max}]$ | $[40\%, 45\%]$ | Adaptive proposal tuning |
| Min layer thickness | $h_{min}$ | 0.1 km | Constraint |
| Covariance regularization | rcond | $10^{-5}$ | For matrix inversion |
| Maximum models saved | $M_{max}$ | 50000 | Storage limit |
| Ray parameter (RF) | $p$ | 6.4 s/° | RF computation |
| Gaussian factor (RF) | gauss | 1.0 | RF pulse width |
| Water level (RF) | water | 0.01 | RF deconvolution |

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | Description |
|----------|--------|-------|-------------|
| SWD periods | $\mathbf{x}_{swd}$ | $(N_{swd},) = (21,)$ | Period values [1, 41] s |
| SWD observations | $\mathbf{d}_{swd}^{obs}$ | $(N_{swd},)$ | Phase velocities (km/s) |
| RF time samples | $\mathbf{x}_{rf}$ | $(N_{rf},)$ | Time axis (s) |
| RF observations | $\mathbf{d}_{rf}^{obs}$ | $(N_{rf},)$ | RF amplitudes |

---

## 3. Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING PIPELINE                      │
└─────────────────────────────────────────────────────────────┘
```

| Step | Operation | Input | Output | Formula/Description |
|------|-----------|-------|--------|---------------------|
| 1 | **Define true model** | — | $\mathbf{h}^{true}, \mathbf{v}_s^{true}, (v_p/v_s)^{true}$ | $\mathbf{h} = [5, 23, 8, 0]$ km, $\mathbf{v}_s = [2.7, 3.6, 3.8, 4.4]$ km/s, $v_p/v_s = 1.73$ |
| 2 | **Generate SWD synthetic** | $\mathbf{h}^{true}, \mathbf{v}_s^{true}$ | $\tilde{\mathbf{d}}_{swd}$ | $\tilde{\mathbf{d}}_{swd} = \mathcal{F}_{swd}(\mathbf{h}^{true}, \mathbf{v}_s^{true}, 1.73)$ |
| 3 | **Generate RF synthetic** | $\mathbf{h}^{true}, \mathbf{v}_s^{true}$ | $\tilde{\mathbf{d}}_{rf}$ | $\tilde{\mathbf{d}}_{rf} = \mathcal{F}_{rf}(\mathbf{h}^{true}, \mathbf{v}_s^{true}, 1.73; p=6.4)$ |
| 4 | **Generate SWD noise** | $\tilde{\mathbf{d}}_{swd}$ | $\boldsymbol{\epsilon}_{swd}$ | Exponentially-correlated: $\rho_{swd}=0.0$, $\sigma_{swd}=0.012$ |
| 5 | **Generate RF noise** | $\tilde{\mathbf{d}}_{rf}$ | $\boldsymbol{\epsilon}_{rf}$ | Gaussian-correlated: $\rho_{rf}=0.98$, $\sigma_{rf}=0.005$ |
| 6 | **Create observed data** | $\tilde{\mathbf{d}}, \boldsymbol{\epsilon}$ | $\mathbf{d}^{obs}$ | $\mathbf{d}_{swd}^{obs} = \tilde{\mathbf{d}}_{swd} + \boldsymbol{\epsilon}_{swd}$, $\mathbf{d}_{rf}^{obs} = \tilde{\mathbf{d}}_{rf} + \boldsymbol{\epsilon}_{rf}$ |
| 7 | **Compute true noise stats** | $\boldsymbol{\epsilon}_{swd}, \boldsymbol{\epsilon}_{rf}$ | $\boldsymbol{\theta}_n^{true}$ | $\boldsymbol{\theta}_n^{true} = [\rho_{swd}, \text{std}(\boldsymbol{\epsilon}_{swd}), \rho_{rf}, \text{std}(\boldsymbol{\epsilon}_{rf})]$ |
| 8 | **Compute expected likelihood** | $\mathbf{d}^{obs}, \tilde{\mathbf{d}}, \boldsymbol{\theta}_n^{true}$ | $\mathcal{L}^{exp}$ | Reference likelihood at true model |
| 9 | **Build depth-velocity profile** | $\mathbf{h}^{true}, \mathbf{v}_s^{true}$ | $(z_{plot}, v_{s,plot})$ | Staircase representation for visualization |
| 10 | **Initialize targets** | $\mathbf{x}, \mathbf{d}^{obs}$ | Target objects | `RayleighDispersionPhase`, `PReceiverFunction` |

---

## 4. Algorithm Steps: Transdimensional MCMC

### 4.1 Initialization Phase

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| I.1 | **Initialize $N_{chains}$ chains** | — | priors, initparams | Chain states $\{\mathbf{s}^{(c)}\}_{c=1}^{N_{chains}}$ |
| I.2 | **Sample initial model per chain** | $k^{(0)} \sim \mathcal{U}[1, 20]$ | Prior bounds | Initial layer count |
| I.3 | **Sample initial velocities** | $v_{s,i}^{(0)} \sim \mathcal{U}[2.0, 5.0]$ | Prior bounds | Initial Vs profile |
| I.4 | **Sample initial depths** | $z_i^{(0)} \sim \mathcal{U}[0, 60]$, sorted | Prior bounds | Initial interfaces |
| I.5 | **Sample initial vpvs** | $(v_p/v_s)^{(0)} \sim \mathcal{U}[1.4, 2.1]$ | Prior bounds | Initial Vp/Vs |
| I.6 | **Sample initial noise** | $\sigma^{(0)} \sim \text{LogU}[10^{-5}, 0.05]$ | Prior bounds | Initial noise params |
| I.7 | **Compute initial likelihood** | $\log\mathcal{L}^{(0)} = \log\mathcal{L}_{SWD}^{(0)} + \log\mathcal{L}_{RF}^{(0)}$ | $\mathbf{m}^{(0)}, \mathbf{d}^{obs}$ | Initial log-likelihood |

### 4.2 Main MCMC Iteration (Per Chain)

For iteration $t = 1, ..., N_{burnin} + N_{main}$:

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| M.1 | **Select move type** | $\tau \sim \text{Categorical}(\mathbf{p}_{move})$ | Move probabilities | Move type $\tau \in \{$birth, death, perturb_vs, perturb_z, perturb_vpvs, perturb_noise$\}$ |
| M.2 | **Propose new state** | $\mathbf{m}' = q(\mathbf{m}^{(t-1)}, \tau)$ | Current state, move type | Proposed model |
| M.2a | *Birth move* | $k' = k + 1$; insert layer at $z' \sim \mathcal{U}[0, z_{max}]$; $v_s' \sim \mathcal{U}[2, 5]$ | — | Add interface |
| M.2b | *Death move* | $k' = k - 1$; remove random interface | — | Remove interface |
| M.2c | *Perturb Vs* | $v_{s,i}' = v_{s,i} + \mathcal{N}(0, \delta_{vs}^2)$ | $\delta_{vs} = 0.025$ | Modified velocity |
| M.2d | *Perturb depth* | $z_i' = z_i + \mathcal{N}(0, \delta_z^2)$ | $\delta_z = 0.025$ | Modified depth |
| M.2e | *Perturb vpvs* | $(v_p/v_s)' = (v_p/v_s) + \mathcal{N}(0, \delta_{vpvs}^2)$ | $\delta_{vpvs} = 0.015$ | Modified ratio |
| M.2f | *Perturb noise* | $\sigma' = \sigma + \mathcal{N}(0, \delta_\sigma^2)$ | $\delta_\sigma = 0.005$ | Modified noise |
| M.3 | **Check prior bounds** | $\mathbf{1}[\mathbf{m}' \in \Omega]$ | $\mathbf{m}'$, bounds | Valid/invalid flag |
| M.4 | **Forward model (SWD)** | $\mathbf{d}_{swd}^{pred} = \mathcal{F}_{swd}(\mathbf{h}', \mathbf{v}_s', (v_p/v_s)')$ | Proposed model | Predicted SWD |
| M.5 | **Forward model (RF)** | $\mathbf{d}_{rf}^{pred} = \mathcal{F}_{rf}(\mathbf{h}', \mathbf{v}_s', (v_p/v_s)'; p, \text{gauss})$ | Proposed model | Predicted RF |
| M.6 | **Compute proposed likelihood** | $\log\mathcal{L}' = \log\mathcal{L}_{SWD}' + \log\mathcal{L}_{RF}'$ | $\mathbf{d}^{pred}, \mathbf{d}^{obs}, \boldsymbol{\theta}_n'$ | Proposed log-likelihood |
| M.7 | **Compute acceptance probability** | $\alpha = \min\left(1, \frac{p(\mathbf{m}')p(\mathbf{d}^{obs}|\mathbf{m}')}{p(\mathbf{m}^{(t-1)})p(\mathbf{d}^{obs}|\mathbf{m}^{(t-1)})} \cdot \frac{q(\mathbf{m}^{(t-1)}|\mathbf{m}')}{q(\mathbf{m}'|\mathbf{m}^{(t-1)})}\right)$ | Likelihoods, priors, proposals | Acceptance ratio |
| M.8 | **Accept/Reject** | $u \sim \mathcal{U}[0,1]$; if $u < \alpha$: $\mathbf{m}^{(t)} = \mathbf{m}'$ else $\mathbf{m}^{(t)} = \mathbf{m}^{(t-1)}$ | $\alpha$, random draw | Updated state |
| M.9 | **Adaptive tuning** (burn-in only) | Adjust $\boldsymbol{\delta}$ to achieve $a \in [40\%, 45\%]$ | Acceptance history | Updated proposal widths |
| M.10 | **Store sample** (main phase only) | Append $(\mathbf{m}^{(t)}, \log\mathcal{L}^{(t)})$ to chain history | Current state | Posterior samples |

### 4.3 Acceptance Ratio Details

**For fixed-dimensional moves** (perturb):
$$
\alpha = \min\left(1, \exp\left(\log\mathcal{L}' - \log\mathcal{L}^{(t-1)}\right)\right)
$$

**For birth move** ($k \to k+1$):
$$
\alpha = \min\left(1, \frac{p(k+1)}{p(k)} \cdot \frac{1}{(k+1)} \cdot \frac{\Delta z \cdot \Delta v_s}{1} \cdot \exp\left(\log\mathcal{L}' - \log\mathcal{L}^{(t-1)}\right)\right)
$$

**For death move** ($k \to k-1$):
$$
\alpha = \min\left(1, \frac{p(k-1)}{p(k)} \cdot k \cdot \frac{1}{\Delta z \cdot \Delta v_s} \cdot \exp\left(\log\mathcal{L}' - \log\mathcal{L}^{(t-1)}\right)\right)
$$

---

## 5. Stopping Criterion

The algorithm terminates when **ALL** of the following conditions are met:

| Criterion | Formula | Threshold |
|-----------|---------|-----------|
| Burn-in complete | $t > N_{burnin}$ | $N_{burnin} = 65536$ per chain |
| Main sampling complete | $t > N_{burnin} + N_{main}$ | $N_{main} = 32768$ per chain |
| All chains finished | $\forall c: t^{(c)} \geq N_{total}$ | $N_{chains} = 5$ |
| Model storage limit (soft) | $|\{\mathbf{m}_{saved}\}| \leq M_{max}$ | $M_{max} = 50000$ |

**Total iterations per chain:** $N_{total} = 65536 + 32768 = 98304$

**Total forward model evaluations:** $N_{chains} \times N_{total} = 5 \times 98304 = 491520$

---

## 6. Evaluation Metrics

### 6.1 Convergence Diagnostics

**Inter-chain variance** (outlier detection):
$$
\text{dev}_c = \frac{|\bar{\mathcal{L}}_c - \bar{\mathcal{L}}_{all}|}{\bar{\mathcal{L}}_{all}}
$$

Chain $c$ is excluded if $\text{dev}_c > 0.05$

### 6.2 Posterior Summary Statistics

**Marginal posterior mean for depth-velocity profile:**
$$
\bar{v}_s(z) = \frac{1}{M} \sum_{m=1}^{M} v_s^{(m)}(z)
$$

**Marginal posterior standard deviation:**
$$
\sigma_{v_s}(z) = \sqrt{\frac{1}{M-1} \sum_{m=1}^{M} \left(v_s^{(m)}(z) - \bar{v}_s(z)\right)^2}
$$

**Number of layers posterior:**
$$
p(k | \mathbf{d}^{obs}) \approx \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}[k^{(m)} = k]
$$

### 6.3 Data Fit Metrics

**Root Mean Square Error (SWD):**
$$
\text{RMSE}_{swd} = \sqrt{\frac{1}{N_{swd}} \sum_{i=1}^{N_{swd}} \left(d_{swd,i}^{obs} - \bar{d}_{swd,i}^{pred}\right)^2}
$$

**Root Mean Square Error (RF):**
$$
\text{RMSE}_{rf} = \sqrt{\frac{1}{N_{rf}} \sum_{i=1}^{N_{rf}} \left(d_{rf,i}^{obs} - \bar{d}_{rf,i}^{pred}\right)^2}
$$

**Expected log-likelihood at true model:**
$$
\mathcal{L}^{exp} = \log p(\mathbf{d}^{obs} | \mathbf{m}^{true}, \boldsymbol{\theta}_n^{true})
$$

---

## 7. Algorithm Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ALGORITHM FLOW                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐                                                     │
│  │ PREPROCESSING   │                                                     │
│  │ ─────────────── │                                                     │
│  │ • Load config   │                                                     │
│  │ • Generate synth│──────┐                                              │
│  │ • Add noise     │      │                                              │
│  │ • Create d_obs  │      │                                              │
│  └─────────────────┘      │                                              │
│           │               │                                              │
│           ▼               ▼                                              │
│  ┌─────────────────────────────────────┐                                │
│  │     INITIALIZE N_chains = 5         │                                │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                            │
│  │  │ C1  │ │ C2  │ │ C3  │ │ C4  │ │ C5  │  (parallel)               │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                            │
│  └─────┼───────┼───────┼───────┼───────┼────┘                           │
│        │       │       │       │       │                                 │
│        ▼       ▼       ▼       ▼       ▼                                 │
│  ╔═══════════════════════════════════════════╗                          │
│  ║     BURN-IN PHASE (65,536 iterations)     ║                          │
│  ║  ┌──────────────────────────────────────┐ ║                          │
│  ║  │ for t = 1 to N_burnin:               │ ║                          │
│  ║  │   1. Select move type τ              │ ║                          │
│  ║  │   2. Propose m' ~ q(·|m,τ)           │ ║                          │
│  ║  │   3. F_swd(m') → d_swd^pred          │ ║                          │
│  ║  │   4. F_rf(m') → d_rf^pred            │ ║                          │
│  ║  │   5. Compute log L'                  │ ║                          │
│  ║  │   6. α = min(1, L'/L × prior ratio)  │ ║                          │
│  ║  │   7. Accept with prob α              │ ║                          │
│  ║  │   8. Adapt proposal widths δ         │ ║                          │
│  ║  └──────────────────────────────────────┘ ║                          │
│  ╚═══════════════════════════════════════════╝                          │
│        │       │       │       │       │                                 │
│        ▼       ▼       ▼       ▼       ▼                                 │
│  ╔═══════════════════════════════════════════╗                          │
│  ║     MAIN PHASE (32,768 iterations)        ║                          │
│  ║  ┌──────────────────────────────────────┐ ║                          │
│  ║  │ for t = 1 to N_main:                 │ ║                          │
│  ║  │   1-7. Same as burn-in               │ ║                          │
│  ║  │   8. Store (m^(t), log L^(t))        │ ║                          │
│  ║  └──────────────────────────────────────┘ ║                          │
│  ╚═══════════════════════════════════════════╝                          │
│        │       │       │       │       │                                 │
│        └───────┴───────┼───────┴───────┘                                │
│                        ▼                                                 │
│  ┌─────────────────────────────────────────┐                            │
│  │         COMBINE CHAINS                   │                            │
│  │  • Detect outlier chains (dev > 0.05)   │                            │
│  │  • Pool accepted models (≤ 50,000)      │                            │
│  └─────────────────────────────────────────┘                            │
│                        │                                                 │
│                        ▼                                                 │
│  ┌─────────────────────────────────────────┐                            │
│  │         POSTERIOR ANALYSIS               │                            │
│  │  • Compute marginal distributions       │                            │
│  │  • Generate Vs(z) profiles              │                            │
│  │  • Compute p(k|d_obs)                   │                            │
│  │  • Save plots with reference model      │                            │
│  └─────────────────────────────────────────┘                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Summary Table

| Aspect | Specification |
|--------|---------------|