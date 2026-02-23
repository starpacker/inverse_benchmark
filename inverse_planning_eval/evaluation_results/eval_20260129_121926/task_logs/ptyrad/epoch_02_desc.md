# Algorithmic Specification Document: Ptychographic Reconstruction via Automatic Differentiation

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{x}} \mathcal{L}(\mathbf{x}) = \sum_{j \in \mathcal{B}} \mathcal{L}_{\text{data}}(I_j^{\text{meas}}, I_j^{\text{pred}}(\mathbf{x})) + \mathcal{R}(\mathbf{x})$$

where $\mathbf{x} = \{A_o, \phi_o, \boldsymbol{\theta}, \Delta z, P, \boldsymbol{\delta}\}$ is the set of optimizable parameters, $\mathcal{B}$ is a batch of scan indices, and $I_j^{\text{pred}}(\mathbf{x}) = |A(\mathbf{x}; j)|^2$ is the predicted diffraction intensity from the forward operator $A$.

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}}(I^{\text{meas}}, I^{\text{pred}}) = \text{CombinedLoss}(I^{\text{meas}}, I^{\text{pred}})$$

The specific loss function is defined by the `loss_params` configuration (e.g., Poisson negative log-likelihood, amplitude MSE, or intensity MSE).

**regularization_term**:
$$\mathcal{R}(\mathbf{x}) = \text{CombinedConstraint}(\mathbf{x})$$

Regularization is applied via the `constraint_params` configuration, potentially including smoothness constraints on object or probe.

**constraints**:
- $A_o \geq 0$ (object amplitude non-negativity, implicit via parameterization)
- $\phi_o \in \mathbb{R}$ (object phase unconstrained)
- $\boldsymbol{\theta} \in \mathbb{R}^{N_{\text{pos}} \times 2}$ or $\mathbb{R}^{1 \times 2}$ (tilt angles)
- $\Delta z > 0$ (slice thickness positive)

---

## 2. VARIABLES

### Primal Variables (Optimizable)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $A_o$ | primal | $(N_s, N_m, N_{oy}, N_{ox})$ | float32 | $\|O_{\text{init}}\|$ | Object amplitude per slice and mode |
| $\phi_o$ | primal | $(N_s, N_m, N_{oy}, N_{ox})$ | float32 | $\angle O_{\text{init}}$ | Object phase per slice and mode |
| $\boldsymbol{\theta}$ | primal | $(N_{\text{tilt}}, 2)$ | float32 | From init | Object tilt angles $(\theta_y, \theta_x)$ in mrad |
| $\Delta z$ | primal | $(N_s,)$ or scalar | float32 | From init | Slice thickness |
| $P$ | primal | $(N_p, N_{py}, N_{px}, 2)$ | float32 | From init (real view) | Probe modes as real-valued tensor |
| $\boldsymbol{\delta}$ | primal | $(N_{\text{pos}}, 2)$ | float32 | From init | Sub-pixel probe position shifts |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $H$ | constants | $(N_{py}, N_{px})$ | complex64 | Precomputed | Base Fresnel propagator |
| $w_m$ | constants | $(N_m,)$ | float32 | From init | Object mode occupancies |
| $\mathbf{r}_j$ | constants | $(N_{\text{pos}}, 2)$ | int32 | From init | Integer crop positions for each scan position |
| $\Delta x$ | constants | scalar | float32 | From init | Real-space pixel size |
| $\lambda$ | constants | scalar | float32 | From init | Wavelength |
| $k$ | constants | scalar | float32 | $2\pi/\lambda$ | Wavenumber |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $I^{\text{meas}}$ | observations | $(N_{\text{pos}}, N_{py}, N_{px})$ | float32 | From data | Measured diffraction intensities |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load parameters | $\text{params} \leftarrow \text{load\_params}(\text{path})$ | YAML path | params dict |
| 2 | Initialize object | $O_{\text{init}} \in \mathbb{C}^{N_s \times N_m \times N_{oy} \times N_{ox}}$ | init_params | Complex object tensor |
| 3 | Decompose object | $A_o = \|O_{\text{init}}\|, \quad \phi_o = \angle O_{\text{init}}$ | $O_{\text{init}}$: $(N_s, N_m, N_{oy}, N_{ox})$ | $A_o, \phi_o$: $(N_s, N_m, N_{oy}, N_{ox})$ each |
| 4 | Initialize probe | $P_{\text{complex}} \in \mathbb{C}^{N_p \times N_{py} \times N_{px}}$ | init_params | Complex probe tensor |
| 5 | Convert probe to real view | $P = \text{view\_as\_real}(P_{\text{complex}})$ | $P_{\text{complex}}$: $(N_p, N_{py}, N_{px})$ | $P$: $(N_p, N_{py}, N_{px}, 2)$ |
| 6 | Create frequency grids | $k_y[n] = \frac{n - N_{py}/2 + 0.5}{N_{py} \cdot \Delta x}, \quad k_x[m] = \frac{m - N_{px}/2 + 0.5}{N_{px} \cdot \Delta x}$ | $N_{py}, N_{px}, \Delta x$ | $(K_y, K_x)$: $(N_{py}, N_{px})$ each |
| 7 | Compute $K_z$ grid | $K_z = \sqrt{k^2 - K_x^2 - K_y^2}$ | $k, K_x, K_y$ | $K_z$: $(N_{py}, N_{px})$ |
| 8 | Initialize base propagator | $H = \exp\left(i \Delta z \cdot K_z\right)$ | $\Delta z, K_z$ | $H$: $(N_{py}, N_{px})$ complex64 |
| 9 | Create FFT frequency grids for shifting | $\nu_y = \text{fftfreq}(N_{py}), \quad \nu_x = \text{fftfreq}(N_{px})$ | $N_{py}, N_{px}$ | $(\nu_y, \nu_x)$ grids |

---

## 4. ALGORITHM STEPS

### Initialization

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Initialize optimizer | $\text{opt} \leftarrow \text{Adam}(\{(p, \eta_p) : p \in \mathbf{x}, \eta_p > 0\})$ | Learning rates $\eta_p$, parameters $\mathbf{x}$ | Optimizer state | Adam optimizer with per-parameter learning rates |
| Initialize iteration counter | $t \leftarrow 0$ | — | $t = 0$ | — |
| Prepare batch indices | $\mathcal{I} \leftarrow \text{shuffle}(\{0, 1, \ldots, N_{\text{pos}}-1\})$ | $N_{\text{pos}}$ | Permuted index array | Random permutation per epoch |

---

### Iteration Loop

**BEGIN ITERATION LOOP** (for $t = 1, 2, \ldots, N_{\text{iter}}$)

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| **4.1 Sample batch** | $\mathcal{B}_t \subseteq \mathcal{I}$ with $|\mathcal{B}_t| = B$ | Index set $\mathcal{I}$, batch size $B$ | Batch indices $\mathcal{B}_t$ | Sequential or random sampling |
| **4.2 Reconstruct complex object** | $O = A_o \odot \exp(i \phi_o)$ | $A_o, \phi_o$: $(N_s, N_m, N_{oy}, N_{ox})$ | $O$: $(N_s, N_m, N_{oy}, N_{ox})$ complex | Element-wise operations |
| **4.3 Extract object patches** | $O_j = O[:, :, r_j^y : r_j^y + N_{py}, r_j^x : r_j^x + N_{px}]$ for $j \in \mathcal{B}_t$ | $O$, crop positions $\mathbf{r}_j$ | $O_{\text{patch}}$: $(B, N_s, N_m, N_{py}, N_{px})$ complex | Batched ROI extraction |
| **4.4 Optional object pre-blur** | $O_{\text{patch}} \leftarrow G_{\sigma_o} * O_{\text{patch}}$ | $O_{\text{patch}}$, blur std $\sigma_o$ | Blurred patches | Gaussian blur with kernel size 5 |
| **4.5 Reconstruct complex probe** | $P_{\text{complex}} = \text{view\_as\_complex}(P)$ | $P$: $(N_p, N_{py}, N_{px}, 2)$ | $P_{\text{complex}}$: $(N_p, N_{py}, N_{px})$ complex | Reinterpret memory layout |
| **4.6 Compute shifted probes** | $\tilde{P}_j = \mathcal{F}^{-1}\left[\mathcal{F}[P_{\text{complex}}] \odot \exp\left(-2\pi i (\nu_y \delta_j^y + \nu_x \delta_j^x)\right)\right]$ | $P_{\text{complex}}$, shifts $\boldsymbol{\delta}_j$ | $\tilde{P}$: $(B, N_p, N_{py}, N_{px})$ complex | Sub-pixel shift via Fourier shift theorem |
| **4.7 Compute tilted propagators** | $H_j = H \odot \exp\left(i \Delta z (K_y \tan(\theta_j^y/10^3) + K_x \tan(\theta_j^x/10^3))\right)$ | $H$, tilts $\boldsymbol{\theta}_j$, $\Delta z$ | $H_{\text{tilt}}$: $(B, N_{py}, N_{px})$ or broadcast | Tilt angles in mrad converted to rad |
| **4.8 Multislice forward propagation** | For slice $s = 0, \ldots, N_s - 1$: $\psi^{(s+1)} = \mathcal{F}^{-1}\left[H_j \odot \mathcal{F}\left[\psi^{(s)} \odot O_j^{(s)}\right]\right]$ with $\psi^{(0)} = \tilde{P}_j$ | $O_{\text{patch}}, \tilde{P}, H_{\text{tilt}}$ | Exit wave $\psi_{\text{exit}}$: $(B, N_p, N_m, N_{py}, N_{px})$ | Sequential slice propagation |
| **4.9 Compute diffraction amplitude** | $\Psi_j = \mathcal{F}[\psi_{\text{exit}}]$ | Exit wave | Fourier amplitude: $(B, N_p, N_m, N_{py}, N_{px})$ | 2D FFT |
| **4.10 Incoherent mode summation** | $I_j^{\text{pred}} = \sum_{p=1}^{N_p} \sum_{m=1}^{N_m} w_m \|\Psi_j^{(p,m)}\|^2$ | $\Psi_j$, mode weights $w_m$ | $I^{\text{pred}}$: $(B, N_{py}, N_{px})$ | Sum over probe and object modes |
| **4.11 Optional detector blur** | $I_j^{\text{pred}} \leftarrow G_{\sigma_d} * I_j^{\text{pred}}$ | $I^{\text{pred}}$, blur std $\sigma_d$ | Blurred intensities | Gaussian blur with kernel size 5 |
| **4.12 Retrieve measurements** | $I_j^{\text{meas}} = I^{\text{meas}}[\mathcal{B}_t]$ | Measurement tensor, batch indices | $I^{\text{meas}}_{\text{batch}}$: $(B, N_{py}, N_{px})$ | Index selection |
| **4.13 Compute loss** | $\mathcal{L}_t = \text{CombinedLoss}(I^{\text{meas}}_{\text{batch}}, I^{\text{pred}})$ | Measured, predicted intensities | Scalar loss $\mathcal{L}_t$ | Configurable loss function |
| **4.14 Compute gradients** | $\nabla_{\mathbf{x}} \mathcal{L}_t \leftarrow \text{autograd}(\mathcal{L}_t)$ | Loss, computation graph | Gradients for all parameters | Automatic differentiation |
| **4.15 Update parameters** | $\mathbf{x} \leftarrow \text{Adam\_step}(\mathbf{x}, \nabla_{\mathbf{x}} \mathcal{L}_t, \eta)$ | Parameters, gradients, learning rates | Updated parameters | Adam update rule |
| **4.16 Apply constraints** | $\mathbf{x} \leftarrow \text{CombinedConstraint}(\mathbf{x}, t)$ | Parameters, iteration $t$ | Constrained parameters | Iteration-dependent constraints |
| **4.17 Update learning rates** | For each $p$: if $t = t_{\text{start}}^p$, enable grad; if $t = t_{\text{end}}^p$, disable grad | Iteration $t$, start/end schedules | Updated requires_grad flags | Dynamic parameter activation |
| **4.18 Convergence check** | Evaluate stopping criterion (see Section 5) | $t$, $\mathcal{L}_t$, $N_{\text{iter}}$ | Continue/stop flag | — |

**END ITERATION LOOP**

---

## 5. STOPPING CRITERION

| type | expression | parameters |
|------|------------|------------|
| iteration_limit | $t \geq N_{\text{iter}}$ | $N_{\text{iter}}$: maximum iterations from `recon_params` |

---

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| Final loss | $\mathcal{L}_{\text{final}} = \mathcal{L}_{N_{\text{iter}}}$ | Loss value at final iteration |
| Object amplitude mean | $\bar{A}_o = \frac{1}{N} \sum_{i} A_o^{(i)}$ | Mean of reconstructed object amplitude |
| Object amplitude std | $\sigma_{A_o} = \sqrt{\frac{1}{N} \sum_i (A_o^{(i)} - \bar{A}_o)^2}$ | Standard deviation of object amplitude |
| Object phase mean | $\bar{\phi}_o = \frac{1}{N} \sum_i \phi_o^{(i)}$ | Mean of reconstructed object phase |
| Object phase std | $\sigma_{\phi_o} = \sqrt{\frac{1}{N} \sum_i (\phi_o^{(i)} - \bar{\phi}_o)^2}$ | Standard deviation of object phase |
| Solver time | $T_{\text{solve}}$ (seconds) | Total wall-clock time for reconstruction |
| Overdetermined ratio | $\rho = \frac{N_{\text{pos}} \cdot N_{py} \cdot N_{px}}{\sum_p \text{numel}(p)}$ | Ratio of measurements to optimizable parameters |