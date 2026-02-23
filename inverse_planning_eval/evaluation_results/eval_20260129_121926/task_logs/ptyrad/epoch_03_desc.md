## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: PtyRAD (Ptychographic Reconstruction using Automatic Differentiation)
- **algorithm_family**: Gradient-based iterative phase retrieval / First-order optimization
- **summary**: Multislice ptychographic reconstruction that jointly recovers object transmission function (amplitude and phase), illumination probe, scan positions, slice thickness, and sample tilts by minimizing a configurable loss function via automatic differentiation and gradient descent.

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{\mathbf{a}, \boldsymbol{\phi}, \mathbf{P}, \boldsymbol{\delta}, \boldsymbol{\theta}, \Delta z} \quad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}}(\mathbf{y}, \hat{\mathbf{y}}) + \mathcal{L}_{\text{reg}}(\mathbf{a}, \boldsymbol{\phi}, \mathbf{P})$$

where $\hat{\mathbf{y}} = \mathcal{A}(\mathbf{a}, \boldsymbol{\phi}, \mathbf{P}, \boldsymbol{\delta}, \boldsymbol{\theta}, \Delta z)$ is the forward model prediction.

### data_fidelity_term

$$\mathcal{L}_{\text{data}}(\mathbf{y}, \hat{\mathbf{y}}) = \text{CombinedLoss}(\mathbf{y}, \hat{\mathbf{y}}; \texttt{loss\_params})$$

The specific form is determined by the `loss_params` configuration dictionary. Common implementations include:

- **Amplitude loss**: $\mathcal{L}_{\text{amp}} = \sum_j \| \sqrt{y_j} - \sqrt{\hat{y}_j} \|_2^2$
- **Intensity loss**: $\mathcal{L}_{\text{int}} = \sum_j \| y_j - \hat{y}_j \|_2^2$
- **Poisson negative log-likelihood**: $\mathcal{L}_{\text{poisson}} = \sum_j (\hat{y}_j - y_j \log(\hat{y}_j + \epsilon))$

### regularization_term

$$\mathcal{L}_{\text{reg}} = \text{CombinedLoss regularization terms}(\texttt{loss\_params})$$

Object-specific regularization may be applied using `_current_object_patches`. Specific regularizers are configured via YAML parameters.

### constraints

Applied via `CombinedConstraint` after each gradient update (projection-based constraints):
- Constraint activation controlled by `start_iter` parameter per constraint type
- Constraints are applied as projections, not incorporated into objective directly

---

## 2. VARIABLES

### Primal Variables (Optimized)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{a}$ (`opt_obja`) | primal | $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | float32 | $\|\texttt{init\_variables['obj']}\|$ | Object amplitude |
| $\boldsymbol{\phi}$ (`opt_objp`) | primal | $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | float32 | $\angle\texttt{init\_variables['obj']}$ | Object phase |
| $\mathbf{P}$ (`opt_probe`) | primal | $(N_{\text{pmode}}, N_{py}, N_{px}, 2)$ | float32 | `view_as_real(init_variables['probe'])` | Probe (real-valued view of complex) |
| $\boldsymbol{\delta}$ (`opt_probe_pos_shifts`) | primal | $(N_{\text{scan}}, 2)$ | float32 | `init_variables['probe_pos_shifts']` | Sub-pixel probe position shifts |
| $\boldsymbol{\theta}$ (`opt_obj_tilts`) | primal | $(N_{\text{tilt}}, 2)$ | float32 | `init_variables['obj_tilts']` | Object tilt angles (mrad) |
| $\Delta z$ (`opt_slice_thickness`) | primal | $(1,)$ or $(N_{\text{slice}}-1,)$ | float32 | `init_variables['slice_thickness']` | Inter-slice propagation distance |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}$ (`measurements`) | observations | $(N_{\text{scan}}, N_{py}, N_{px})$ | float32 | `init_variables['measurements']` | Measured diffraction intensities |

### Constants / Buffers

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{H}$ (`H`) | constants | $(N_{py}, N_{px})$ | complex64 | `init_variables['H']` | Base Fresnel propagator |
| $\mathbf{w}$ (`omode_occu`) | constants | $(N_{\text{omode}},)$ | float32 | `init_variables['omode_occu']` | Object mode occupation weights |
| $\mathbf{r}$ (`crop_pos`) | constants | $(N_{\text{scan}}, 2)$ | int32 | `init_variables['crop_pos']` | Integer crop positions for object ROI |
| $\Delta x$ (`dx`) | constants | scalar | float32 | `init_variables['dx']` | Real-space pixel size |
| $\Delta k$ (`dk`) | constants | scalar | float32 | `init_variables['dk']` | Reciprocal-space pixel size |
| $\lambda$ (`lambd`) | constants | scalar | float32 | `init_variables['lambd']` | Wavelength |
| $N_{\text{slow}}$ (`N_scan_slow`) | constants | scalar | int32 | `init_variables['N_scan_slow']` | Scan points in slow direction |
| $N_{\text{fast}}$ (`N_scan_fast`) | constants | scalar | int32 | `init_variables['N_scan_fast']` | Scan points in fast direction |
| $\sigma_{\text{det}}$ (`detector_blur_std`) | constants | scalar or None | float | `model_params['detector_blur_std']` | Detector PSF blur std |
| $\sigma_{\text{obj}}$ (`obj_preblur_std`) | constants | scalar or None | float | `model_params['obj_preblur_std']` | Object pre-blur std |
| $k$ | constants | scalar | float32 | $k = 2\pi/\lambda$ | Wavenumber |

### Auxiliary Variables (Computed)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $K_y, K_x$ (`propagator_grid`) | auxiliary | $(2, N_{py}, N_{px})$ | float32 | FFT frequency grids | Propagator frequency coordinates |
| $K_z$ (`Kz`) | auxiliary | $(N_{py}, N_{px})$ | float32 | $\sqrt{k^2 - K_x^2 - K_y^2}$ | Axial frequency component |
| $\mathbf{H}_{\text{tilt}}$ (`H_fixed_tilts_full`) | auxiliary | $(N_{\text{tilt}}, N_{py}, N_{px})$ | complex64 | Computed from $\mathbf{H}$, $\boldsymbol{\theta}$, $\Delta z$ | Pre-computed tilted propagators |
| `shift_probes_grid` | auxiliary | $(2, N_{py}, N_{px})$ | float32 | `fftfreq` grids | Grid for Fourier shift of probes |
| `shift_object_grid` | auxiliary | $(2, N_{oy}, N_{ox})$ | float32 | `fftfreq` grids | Grid for Fourier shift of object |
| `rpy_grid`, `rpx_grid` | auxiliary | $(N_{py}, N_{px})$ | int32 | `arange` meshgrid | Integer grids for ROI extraction |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula / Operation | Inputs | Outputs |
|------|------|---------------------|--------|---------|
| 1 | Load Parameters | `params = load_params(params_path, validate=True)` | `params_path: str` | `params: dict` |
| 2 | Set Device | `device = set_gpu_device(gpuid)` | `gpuid: int` | `device: torch.device` |
| 3 | Initialize Variables | `init = Initializer(params['init_params']).init_all()` | `params['init_params']: dict` | `init.init_variables: dict` containing all initial arrays |
| 4 | Initialize Loss | `loss_fn = CombinedLoss(loss_params, device)` | `loss_params: dict`, `device` | `loss_fn: callable` |
| 5 | Initialize Constraints | `constraint_fn = CombinedConstraint(constraint_params, device)` | `constraint_params: dict`, `device` | `constraint_fn: callable` |
| 6 | Decompose Object | $\mathbf{a} = \|\mathbf{O}_{\text{init}}\|$, $\boldsymbol{\phi} = \angle \mathbf{O}_{\text{init}}$ | `init_variables['obj']: complex` | `opt_obja: float32`, `opt_objp: float32` |
| 7 | Convert Probe | $\mathbf{P}_{\text{real}} = \text{view\_as\_real}(\mathbf{P}_{\text{complex}})$ | `init_variables['probe']: complex64` | `opt_probe: float32, shape (..., 2)` |
| 8 | Create Propagator Grid | $k_y^{(m)} = \frac{2\pi}{\Delta x} \cdot \text{ifftshift}\left(\frac{m - N_{py}/2 + 0.5}{N_{py}}\right)$, similarly for $k_x$ | `dx`, `Npy`, `Npx` | `propagator_grid: (2, Npy, Npx)` |
| 9 | Compute $K_z$ | $K_z = \sqrt{k^2 - K_x^2 - K_y^2}$ where $k = 2\pi/\lambda$ | `lambd`, `propagator_grid` | `Kz: (Npy, Npx)` |
| 10 | Precompute Tilted Propagator | $\mathbf{H}_{\text{tilt}}^{(i)} = \mathbf{H} \odot \exp\left(i \Delta z \left(K_y \tan(\theta_y^{(i)}/10^3) + K_x \tan(\theta_x^{(i)}/10^3)\right)\right)$ | `H`, `opt_obj_tilts`, `opt_slice_thickness`, `propagator_grid` | `H_fixed_tilts_full` |
| 11 | Create Shift Grids | `shift_probes_grid = stack([fftfreq(Npy), fftfreq(Npx)])` | `Npy`, `Npx` | `shift_probes_grid: (2, Npy, Npx)` |
| 12 | Create ROI Grids | `rpy_grid, rpx_grid = meshgrid(arange(Npy), arange(Npx))` | `Npy`, `Npx` | `rpy_grid, rpx_grid: (Npy, Npx)` |
| 13 | Compute Probe Intensity Sum | $I_P = \sum_{m,n,i,j} \|\mathbf{P}_{m,n,i,j}\|^2$ | `opt_probe` | `probe_int_sum: scalar` |
| 14 | Create Optimizer | `optimizer = create_optimizer(optimizer_params, optimizable_params)` | `optimizer_params: dict`, `optimizable_params: list` | `optimizer: torch.optim.Optimizer` |
| 15 | Prepare Batches | `indices, batches, output_path = prepare_recon(model, init, params)` | `model`, `init`, `params` | Index arrays and batch structure |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| I1 | Set Gradient Flags | For each param $p$: `requires_grad = (lr[p] != 0) and (start_iter[p] == 1)` | `lr_params`, `start_iter` | Updated `requires_grad` attributes | Controls which variables are optimized |
| I2 | Initialize Tracking | `loss_iters = []`, `iter_times = []`, `dz_iters = []`, `avg_tilt_iters = []` | — | Empty lists | For convergence monitoring |
| I3 | Record Start Time | `start_t = time_sync()` | — | `start_t: float` | Wall-clock timing |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (iteration $t = 1, 2, \ldots, N_{\text{iter}}$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L1 | Update Grad Flags | If $t \in \texttt{start\_iter}[p]$: set `requires_grad=True`; if $t \in \texttt{end\_iter}[p]$: set `requires_grad=False` | `t`, `start_iter`, `end_iter` | Updated gradient flags | Staged optimization |
| L2 | Sample Batch | $\mathcal{B}_t \subset \{0, 1, \ldots, N_{\text{scan}}-1\}$ | `batches`, `t` | `indices: (B,)` | Mini-batch selection |
| L3 | Extract Object Patches | $\mathbf{O}_{\text{ROI}}^{(j)} = (\mathbf{a}, \boldsymbol{\phi})[:, :, r_y^{(j)}:r_y^{(j)}+N_{py}, r_x^{(j)}:r_x^{(j)}+N_{px}]$ for $j \in \mathcal{B}_t$ | `opt_obja`, `opt_objp`, `crop_pos[indices]`, `rpy_grid`, `rpx_grid` | `object_patches: (B, N_omode, N_slice, Npy, Npx, 2)` | Last dim: (amp, phase) |
| L4 | Optional Object Blur | If $\sigma_{\text{obj}} > 0$: $\mathbf{O}_{\text{ROI}} \leftarrow \text{GaussianBlur}(\mathbf{O}_{\text{ROI}}; k=5, \sigma=\sigma_{\text{obj}})$ | `object_patches`, `obj_preblur_std` | `object_patches` (blurred) | Kernel size = 5 |
| L5 | Get Probes | If `shift_probes`: $\mathbf{P}^{(j)} = \mathcal{F}^{-1}\left[\mathcal{F}[\mathbf{P}] \odot \exp(-2\pi i (k_y \delta_y^{(j)} + k_x \delta_x^{(j)}))\right]$; else: $\mathbf{P}^{(j)} = \mathbf{P}$ | `opt_probe`, `opt_probe_pos_shifts[indices]`, `shift_probes_grid` | `probes: (B, N_pmode, Npy, Npx)` complex | Fourier shift theorem |
| L6 | Compute Propagators | See sub-steps L6a-L6d | — | `propagators` | Conditional logic |

**Sub-step L6a** (tilt_obj=True, change_thickness=True):
$$\mathbf{H}_{\text{prop}}^{(j)} = \exp(i \Delta z K_z) \odot \exp\left(i \Delta z \left(K_y \tan(\theta_y^{(j)}/10^3) + K_x \tan(\theta_x^{(j)}/10^3)\right)\right)$$

**Sub-step L6b** (tilt_obj=True, change_thickness=False, change_tilt=True):
$$\mathbf{H}_{\text{prop}}^{(j)} = \mathbf{H} \odot \exp\left(i \Delta z \left(K_y \tan(\theta_y^{(j)}/10^3) + K_x \tan(\theta_x^{(j)}/10^3)\right)\right)$$

**Sub-step L6c** (tilt_obj=True, change_thickness=False, change_tilt=False):
$$\mathbf{H}_{\text{prop}}^{(j)} = \mathbf{H}_{\text{tilt}}^{(j)} \quad \text{(precomputed)}$$

**Sub-step L6d** (tilt_obj=False):
$$\mathbf{H}_{\text{prop}} = \mathbf{H} \quad \text{or} \quad \exp(i \Delta z K_z) \text{ if change\_thickness}$$

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L7 | Multislice Forward | $\hat{\mathbf{y}} = \texttt{multislice\_forward\_model\_vec\_all}(\mathbf{O}_{\text{ROI}}, \mathbf{P}, \mathbf{H}_{\text{prop}}, \mathbf{w})$ | `object_patches`, `probes`, `propagators`, `omode_occu` | `dp_fwd: (B, Npy, Npx)` | See L7 expansion |

**Expansion of L7 (Multislice Forward Model)**:

**BEGIN_INNER_LOOP** (slice $s = 0, 1, \ldots, N_{\text{slice}}-1$)

| Sub-step | Formula |
|----------|---------|
| L7.1 | Form complex object: $O_s = a_s \odot \exp(i \phi_s)$ |
| L7.2 | Exit wave at slice: $\psi_s = \psi_{s-1} \odot O_s$ (for $s=0$: $\psi_0 = P \odot O_0$) |
| L7.3 | Propagate to next slice (if $s < N_{\text{slice}}-1$): $\psi_s \leftarrow \mathcal{F}^{-1}[\mathbf{H}_{\text{prop}} \odot \mathcal{F}[\psi_s]]$ |

**END_INNER_LOOP**

| Sub-step | Formula |
|----------|---------|
| L7.4 | Final exit wave: $\psi_{\text{exit}} = \psi_{N_{\text{slice}}-1}$ |
| L7.5 | Far-field propagation: $\Psi = \mathcal{F}[\psi_{\text{exit}}]$ |
| L7.6 | Intensity with mode summation: $\hat{y} = \sum_m w_m \|\Psi_m\|^2$ |

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L8 | Optional Detector Blur | If $\sigma_{\text{det}} > 0$: $\hat{\mathbf{y}} \leftarrow \text{GaussianBlur}(\hat{\mathbf{y}}; k=5, \sigma=\sigma_{\text{det}})$ | `dp_fwd`, `detector_blur_std` | `dp_fwd` (blurred) | Kernel size = 5 |
| L9 | Get Measurements | $\mathbf{y}_{\mathcal{B}} = \texttt{measurements}[\mathcal{B}_t]$ with optional padding/rescaling | `measurements`, `indices`, `meas_padded`, `meas_scale_factors` | `measurements_batch: (B, Npy, Npx)` | On-the-fly processing |
| L10 | Compute Loss | $\mathcal{L}_t = \texttt{loss\_fn}(\hat{\mathbf{y}}, \mathbf{y}_{\mathcal{B}}, \mathbf{O}_{\text{ROI}})$ | `dp_fwd`, `measurements_batch`, `_current_object_patches` | `loss: scalar` | Includes regularization |
| L11 | Zero Gradients | `optimizer.zero_grad()` | `optimizer` | Cleared gradients | — |
| L12 | Backward Pass | $\nabla_{\mathbf{x}} \mathcal{L}_t = \texttt{autograd}(\mathcal{L}_t)$ | `loss` | Gradients in `.grad` | Automatic differentiation |
| L13 | Optimizer Step | $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \cdot \texttt{optimizer\_update}(\nabla \mathcal{L}_t)$ | `optimizer`, gradients | Updated parameters | Adam/SGD/etc. per config |
| L14 | Apply Constraints | $\mathbf{x}^{(t+1)} \leftarrow \texttt{constraint\_fn}(\mathbf{x}^{(t+1)}, t)$ | `model`, `constraint_fn`, `t` | Projected parameters | Constraint-dependent activation |
| L15 | Record Loss | `loss_iters.append(loss.item())` | `loss` | Updated `loss_iters` | Monitoring |
| L16 | Record Time | `iter_times.append(time_sync() - iter_start)` | — | Updated `iter_times` | — |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| F1 | Record Total Time | `solver_t = time_sync() - start_t` | `start_t` | `solver_time: float` | — |
| F2 | Save Results | Write to `output_path` | `model`, `output_path` | Saved files | Via `recon_loop` |
| F3 | Cleanup DDP | `dist.destroy_process_group()` if initialized | — | — | Distributed cleanup |
| F4 | Close Logger | `logger.close()` if `flush_file` | `logger` | — | — |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit (primary) with optional convergence monitoring
- **expression**: 
$$t \geq N_{\text{iter}}$$
where $N_{\text{iter}}$ is specified in `params['recon_params']`

- **parameters**:
  - `N_iter`: Maximum iteration count (from `recon_params`)
  - Per-variable activation windows: `start_iter[param]`, `end_iter[param]`
  
- **check_frequency**: Every iteration (loop termination condition)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Loss per iteration | $\mathcal{L}_t$ stored in `loss_iters` | Training loss trajectory |
| Final loss | $\mathcal{L}_{\text{final}} = \mathcal{L}_{N_{\text{iter}}}$ | Converged loss value |
| Object amplitude mean | $\bar{a} = \frac{1}{N}\sum_{i} a_i$ | Mean of reconstructed amplitude |
| Object amplitude std | $\sigma_a = \sqrt{\frac{1}{N}\sum_i (a_i - \bar{a})^2}$ | Spread of amplitude values |
| Object phase mean | $\bar{\phi} = \frac{1}{N}\sum_i \phi_i$ | Mean of reconstructed phase |
| Object phase std | $\sigma_\phi = \sqrt{\frac{1}{N}\sum_i (\phi_i - \bar{\phi})^2}$ | Spread of phase values |
| Overdetermined ratio | $R = \frac{N_{\text{meas}}}{N_{\text{params}}} = \frac{\texttt{measurements.numel()}}{\sum_p \texttt{numel}(p) \cdot \mathbb{1}[\texttt{requires\_grad}(p)]}$ | Data-to-parameter ratio |
| Solver time | $T_{\text{total}} = t_{\text{end}} - t_{\text{start}}$ (seconds) | Total wall-clock time |
| Iteration time | `iter_times[t]` per iteration | Per-iteration timing |