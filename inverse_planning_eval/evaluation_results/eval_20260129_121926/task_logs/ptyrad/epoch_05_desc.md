## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: PtyRAD (Ptychographic Reconstruction via Automatic Differentiation)
- **algorithm_family**: Gradient-based optimization with automatic differentiation (first-order methods)
- **summary**: Multislice ptychographic reconstruction that minimizes a combined loss function over object amplitude/phase, probe, and auxiliary parameters using stochastic gradient descent variants with constraint enforcement.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\theta} \sum_{b \in \text{batches}} \mathcal{L}_{\text{combined}}\bigl(I_{\text{meas}}^{(b)}, I_{\text{pred}}^{(b)}(\theta), O^{(b)}\bigr) \quad \text{subject to} \quad \theta \in \mathcal{C}$$

where $\theta = \{A_O, \phi_O, P, \Delta\mathbf{r}, \boldsymbol{\alpha}, \Delta z\}$ and $\mathcal{C}$ denotes the constraint set.

### data_fidelity_term
$$\mathcal{L}_{\text{data}} = \sum_{j \in b} \left\| \sqrt{I_{\text{meas},j}} - \sqrt{I_{\text{pred},j}} \right\|_2^2$$

where $I_{\text{pred},j} = \left| \mathcal{F}\{\psi_{\text{exit},j}\} \right|^2$ (amplitude-based loss typical for ptychography; exact form determined by `loss_params`).

### regularization_term
$$\mathcal{L}_{\text{reg}} = \sum_{k} \lambda_k R_k(\theta)$$

Regularization terms are specified by `loss_params` dictionary. Each active loss type contributes a term $R_k$ weighted by $\lambda_k$. Object-specific regularization may use `_current_object_patches`.

### constraints
$$\mathcal{C} = \bigcap_{c \in \text{active\_constraints}} \mathcal{C}_c$$

Constraints are applied via `CombinedConstraint` with `start_iter` activation. Constraint types include amplitude bounds, support constraints, and probe normalization as specified in `constraint_params`.

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $A_O$ (`opt_obja`) | $(N_s, N_z, N_{oy}, N_{ox})$ | float32 | $\|O_{\text{init}}\|$ | Object amplitude per slice |
| $\phi_O$ (`opt_objp`) | $(N_s, N_z, N_{oy}, N_{ox})$ | float32 | $\angle O_{\text{init}}$ | Object phase per slice |
| $P$ (`opt_probe`) | $(N_m, N_{py}, N_{px}, 2)$ | float32 | `view_as_real(probe_init)` | Probe modes (real/imag stacked) |
| $\Delta\mathbf{r}$ (`opt_probe_pos_shifts`) | $(N_j, 2)$ | float32 | `probe_pos_shifts_init` | Sub-pixel probe position corrections |
| $\boldsymbol{\alpha}$ (`opt_obj_tilts`) | $(N_t, 2)$ | float32 | `obj_tilts_init` | Object tilt angles (mrad) |
| $\Delta z$ (`opt_slice_thickness`) | scalar or $(N_z,)$ | float32 | `slice_thickness_init` | Slice thickness |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| — | — | — | — | No explicit dual variables; gradients computed via autograd |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| `H_fixed_tilts_full` | $(N_t, N_{py}, N_{px})$ | complex64 | Computed | Pre-computed tilted propagator |
| `propagator_grid` | $(2, N_{py}, N_{px})$ | float32 | Computed | $(K_y, K_x)$ frequency grids |
| `shift_probes_grid` | $(2, N_{py}, N_{px})$ | float32 | Computed | FFT frequency grid for probe shifts |
| `shift_object_grid` | $(2, N_{oy}, N_{ox})$ | float32 | Computed | FFT frequency grid for object shifts |
| `rpy_grid`, `rpx_grid` | $(N_{py}, N_{px})$ | int32 | Computed | ROI selection grids |
| `Kz` | $(N_{py}, N_{px})$ | float32 | Computed | Axial wavenumber grid |
| `k` | scalar | float32 | $2\pi/\lambda$ | Wavenumber |
| `probe_int_sum` | scalar | float32 | $\sum\|P\|^2$ | Initial probe intensity sum |
| `_current_object_patches` | $(N_b, N_s, N_z, N_m, N_{py}, N_{px}, 2)$ | float32 | None | Cached object ROI for loss |
| `loss_iters` | list | float | `[]` | Loss history |
| `iter_times` | list | float | `[]` | Iteration timing |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $\lambda$ (`lambd`) | scalar | float32 | From params | Wavelength |
| $\delta x$ (`dx`) | scalar | float32 | From params | Real-space pixel size |
| $\delta k$ (`dk`) | scalar | float32 | From params | Reciprocal-space pixel size |
| $H$ | $(N_{py}, N_{px})$ | complex64 | From init | Base Fresnel propagator |
| `omode_occu` | $(N_m,)$ | float32 | From init | Probe mode occupancy weights |
| `crop_pos` | $(N_j, 2)$ | int32 | From init | Integer scan positions |
| `N_scan_slow`, `N_scan_fast` | scalar | int32 | From init | Scan grid dimensions |
| `detector_blur_std` | scalar or None | float32 | From params | Detector blur sigma |
| `obj_preblur_std` | scalar or None | float32 | From params | Object pre-blur sigma |
| `kernel_size` | scalar | int | 5 | Gaussian blur kernel size |
| `start_iter`, `end_iter` | dict | int | From params | Per-variable optimization windows |
| `lr_params` | dict | float | From params | Per-variable learning rates |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $I_{\text{meas}}$ (`measurements`) | $(N_j, N_{ky}, N_{kx})$ | float32 | Measured diffraction intensities |
| `meas_padded` | $(N_{ky}', N_{kx}')$ | float32 | Padded measurement template (optional) |
| `meas_padded_idx` | $(4,)$ | int32 | Padding indices $[h_1, h_2, w_1, w_2]$ |
| `meas_scale_factors` | $(2,)$ | float32 | Measurement resampling factors (optional) |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load Parameters | $\text{params} = \text{load\_params}(\text{path})$ | `params_path` | `params` dict |
| 2 | Initialize Variables | $\text{init} = \text{Initializer}(\text{init\_params}).\text{init\_all}()$ | `params['init_params']` | `init.init_variables` dict |
| 3 | Object Decomposition | $A_O^{(0)} = \|O_{\text{init}}\|, \quad \phi_O^{(0)} = \angle O_{\text{init}}$ | `init_variables['obj']` $(N_s, N_z, N_{oy}, N_{ox})$ complex | `opt_obja`, `opt_objp` each $(N_s, N_z, N_{oy}, N_{ox})$ float32 |
| 4 | Probe Real Conversion | $P_{\text{real}}^{(0)} = \text{view\_as\_real}(P_{\text{init}})$ | `init_variables['probe']` $(N_m, N_{py}, N_{px})$ complex64 | `opt_probe` $(N_m, N_{py}, N_{px}, 2)$ float32 |
| 5 | Construct Loss Function | $\mathcal{L} = \text{CombinedLoss}(\text{loss\_params})$ | `params['loss_params']` | `loss_fn` callable |
| 6 | Construct Constraint Function | $\mathcal{C} = \text{CombinedConstraint}(\text{constraint\_params})$ | `params['constraint_params']` | `constraint_fn` callable |
| 7 | Create Frequency Grids | $k_y^{(p)} = \text{fftfreq}(N_{py}), \quad k_x^{(p)} = \text{fftfreq}(N_{px})$ | $N_{py}, N_{px}$ | `shift_probes_grid` $(2, N_{py}, N_{px})$ |
| 8 | Create Propagator Grid | $K_y = \text{ifftshift}\left(\frac{2\pi}{\delta x} \cdot \frac{y - N_{py}/2 + 0.5}{N_{py}}\right)$, similarly $K_x$ | $N_{py}, N_{px}, \delta x$ | `propagator_grid` $(2, N_{py}, N_{px})$ |
| 9 | Compute Axial Wavenumber | $K_z = \sqrt{k^2 - K_x^2 - K_y^2}, \quad k = \frac{2\pi}{\lambda}$ | `propagator_grid`, $\lambda$ | `Kz` $(N_{py}, N_{px})$ |
| 10 | Pre-compute Tilted Propagator | $H_{\text{tilt}} = H \cdot \exp\left(i \Delta z \left(K_y \tan\frac{\alpha_y}{1000} + K_x \tan\frac{\alpha_x}{1000}\right)\right)$ | $H$, $\boldsymbol{\alpha}^{(0)}$, $\Delta z$, `propagator_grid` | `H_fixed_tilts_full` $(N_t, N_{py}, N_{px})$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Allocate Model | Instantiate `PtychoAD` module | `init_variables`, `model_params`, `device` | `model` object | Registers parameters and buffers |
| 2 | Set Gradient Flags | $\text{requires\_grad}[\theta_i] = (\text{lr}_i \neq 0) \land (\text{start\_iter}_i = 1)$ | `lr_params`, `start_iter` | Per-parameter gradient state | Enables selective optimization |
| 3 | Create Optimizer | $\text{optimizer} = \text{create\_optimizer}(\text{optimizer\_params}, \text{optimizable\_params})$ | `optimizer_params`, parameter list with learning rates | `optimizer` (Adam/SGD/etc.) | From `optimizer_params['type']` |
| 4 | Prepare Indices | $\text{indices} = [0, 1, \ldots, N_j - 1]$, partition into batches | $N_j$, `batch_size` | `indices` $(N_j,)$, `batches` list | Shuffled per epoch if configured |
| 5 | Initialize Tracking | `loss_iters = []`, `iter_times = []` | — | Empty lists | For convergence monitoring |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (iteration $t = 1, \ldots, N_{\text{iter}}$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 6 | Update Gradient Flags | For each $\theta_i$: set `requires_grad` based on $t \in [\text{start\_iter}_i, \text{end\_iter}_i]$ | $t$, `start_iter`, `end_iter` | Updated gradient states | Re-compile if $t \in$ `compilation_iters` |
| 7 | Sample Batch | $b = \text{batches}[t \mod |\text{batches}|]$ | `batches`, $t$ | `indices_batch` $(N_b,)$ | Cyclic batch sampling |
| 8 | Zero Gradients | $\nabla_\theta \mathcal{L} \leftarrow 0$ | `optimizer` | Cleared gradient buffers | `optimizer.zero_grad()` |
| 9 | Extract Object ROI | $O_{\text{ROI}}[j] = (A_O, \phi_O)[:, :, r_y^{(j)}:r_y^{(j)}+N_{py}, r_x^{(j)}:r_x^{(j)}+N_{px}]$ where $(r_y^{(j)}, r_x^{(j)}) = \text{crop\_pos}[j]$ | `opt_obja`, `opt_objp`, `crop_pos`, `indices_batch` | `object_patches` $(N_b, N_s, N_z, N_m, N_{py}, N_{px}, 2)$ | Last dim: (amp, phase) |
| 10 | Apply Object Pre-blur | If `obj_preblur_std` $\neq 0$: $O_{\text{ROI}} \leftarrow \text{gaussian\_blur}(O_{\text{ROI}}, \sigma=\sigma_{\text{obj}}, k=5)$ | `object_patches`, `obj_preblur_std` | `object_patches` (blurred) | Optional smoothing |
| 11 | Reconstruct Complex Probe | $P_{\mathbb{C}} = \text{view\_as\_complex}(P_{\text{real}})$ | `opt_probe` $(N_m, N_{py}, N_{px}, 2)$ | $P_{\mathbb{C}}$ $(N_m, N_{py}, N_{px})$ complex64 | Interpret real tensor as complex |
| 12 | Shift Probes | If `shift_probes`: $P_j = \mathcal{F}^{-1}\left\{\mathcal{F}\{P_{\mathbb{C}}\} \cdot \exp\left(-2\pi i (k_y \Delta r_y^{(j)} + k_x \Delta r_x^{(j)})\right)\right\}$ | $P_{\mathbb{C}}$, `opt_probe_pos_shifts`, `shift_probes_grid` | `probes` $(N_b, N_m, N_{py}, N_{px})$ | Sub-pixel Fourier shift |
| 13 | Compute Propagator | $H_j = \exp(i \Delta z K_z) \cdot \exp\left(i \Delta z \left(K_y \tan\frac{\alpha_y^{(j)}}{1000} + K_x \tan\frac{\alpha_x^{(j)}}{1000}\right)\right)$ | `opt_slice_thickness`, `opt_obj_tilts`, `Kz`, `propagator_grid` | `propagators` $(N_b, N_{py}, N_{px})$ or $(1, N_{py}, N_{px})$ | Uses pre-computed if tilts/thickness fixed |
| 14 | Multislice Forward [FORWARD] | For each position $j$, slice $n = 0, \ldots, N_z-1$: $\psi_n = P_j$ (if $n=0$) or $\psi_n = \mathcal{F}^{-1}\{H_j \cdot \mathcal{F}\{\psi_{n-1} \cdot O_n^{(j)}\}\}$; $\psi_{\text{exit}} = \psi_{N_z-1} \cdot O_{N_z-1}^{(j)}$ where $O_n^{(j)} = A_{O,n}^{(j)} \exp(i \phi_{O,n}^{(j)})$ | `object_patches`, `probes`, `propagators`, `omode_occu` | `exit_waves` $(N_b, N_m, N_{py}, N_{px})$ | `multislice_forward_model_vec_all` |
| 15 | Compute Diffraction Pattern | $I_{\text{pred},j} = \sum_m w_m \left|\mathcal{F}\{\psi_{\text{exit},j,m}\}\right|^2$ | `exit_waves`, `omode_occu` | `dp_fwd` $(N_b, N_{py}, N_{px})$ | Mode-weighted incoherent sum |
| 16 | Apply Detector Blur | If `detector_blur_std` $\neq 0$: $I_{\text{pred}} \leftarrow \text{gaussian\_blur}(I_{\text{pred}}, \sigma=\sigma_{\text{det}}, k=5)$ | `dp_fwd`, `detector_blur_std` | `dp_fwd` (blurred) | Optional PSF convolution |
| 17 | Retrieve Measurements | $I_{\text{meas},b} = \text{measurements}[\text{indices\_batch}]$; apply padding/resampling if configured | `measurements`, `indices_batch`, `meas_padded`, `meas_scale_factors` | `meas_batch` $(N_b, N_{ky}, N_{kx})$ | On-the-fly preprocessing |
| 18 | Compute Loss | $\mathcal{L} = \mathcal{L}_{\text{combined}}(I_{\text{meas},b}, I_{\text{pred},b}, O_{\text{ROI}})$ | `meas_batch`, `dp_fwd`, `_current_object_patches` | `loss` scalar | `loss_fn(dp_fwd, meas_batch, ...)` |
| 19 | Backward Pass [ADJOINT] | $\nabla_\theta \mathcal{L} = \text{autograd.backward}(\mathcal{L})$ | `loss`, all parameters with `requires_grad=True` | Gradients in `.grad` attributes | PyTorch automatic differentiation |
| 20 | Optimizer Step | $\theta \leftarrow \theta - \eta \cdot \text{update}(\nabla_\theta \mathcal{L})$ | Gradients, `optimizer` | Updated $\theta = \{A_O, \phi_O, P, \Delta\mathbf{r}, \boldsymbol{\alpha}, \Delta z\}$ | Adam/SGD/etc. with per-param LR |
| 21 | Apply Constraints | $\theta \leftarrow \text{constraint\_fn}(\theta, t)$ | $\theta$, iteration $t$ | Constrained $\theta$ | Active if $t \geq \text{start\_iter}_c$ |
| 22 | Update Propagator Cache | If `change_thickness` or `change_tilt`: recompute `H_fixed_tilts_full` | Updated $\Delta z$, $\boldsymbol{\alpha}$ | `H_fixed_tilts_full` | `init_propagator_vars()` |
| 23 | Record Loss | `loss_iters.append(loss.item())` | `loss` | Updated `loss_iters` | Convergence tracking |
| 24 | Clear Cache | `_current_object_patches = None` | — | Freed memory | Prevents memory leak |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 25 | Reconstruct Complex Object | $O_{\text{final}} = A_O \cdot \exp(i \phi_O)$ | `opt_obja`, `opt_objp` | `obj_complex` $(N_s, N_z, N_{oy}, N_{ox})$ complex | Final object reconstruction |
| 26 | Reconstruct Complex Probe | $P_{\text{final}} = \text{view\_as\_complex}(P_{\text{real}})$ | `opt_probe` | `probe_complex` $(N_m, N_{py}, N_{px})$ complex | Final probe reconstruction |
| 27 | Compute Statistics | $\bar{A}_O = \text{mean}(A_O), \quad \sigma_{A_O} = \text{std}(A_O)$, similarly for $\phi_O$ | `opt_obja`, `opt_objp` | `obj_amp_mean`, `obj_amp_std`, `obj_phase_mean`, `obj_phase_std` | Summary statistics |
| 28 | Save Results | Write `model`, `optimizer` state to `output_path` | `model`, `optimizer`, `output_path` | Saved files | Checkpoint for restart |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit (with optional convergence monitoring)
- **expression**: $t \geq N_{\text{iter}}$ where $N_{\text{iter}}$ is specified in `params['recon_params']['N_iter']`
- **parameters**: 
  - `N_iter`: Total number of iterations from `recon_params`
  - Loss history stored in `loss_iters` for convergence analysis
- **check_frequency**: Every iteration (loop termination condition)

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Final Loss | $\mathcal{L}_{\text{final}} = \texttt{loss\_iters}[-1]$ | — | Accumulated loss value | Terminal value of combined objective |
| Object Amplitude Mean | $\bar{A}_O = \frac{1}{|\Omega|} \sum_{(s,z,y,x) \in \Omega} A_O^{(s,z,y,x)}$ | — | `opt_obja` | Average transmission amplitude |
| Object Amplitude Std | $\sigma_{A_O} = \sqrt{\frac{1}{|\Omega|} \sum_{(s,z,y,x)} (A_O^{(s,z,y,x)} - \bar{A}_O)^2}$ | — | `opt_obja` | Amplitude variation measure |
| Object Phase Mean | $\bar{\phi}_O = \frac{1}{|\Omega|} \sum_{(s,z,y,x) \in \Omega} \phi_O^{(s,z,y,x)}$ | — | `opt_objp` | Average phase shift |
| Object Phase Std | $\sigma_{\phi_O} = \sqrt{\frac{1}{|\Omega|} \sum_{(s,z,y,x)} (\phi_O^{(s,z,y,x)} - \bar{\phi}_O)^2}$ | — | `opt_objp` | Phase variation measure |
| Overdetermined Ratio | $\rho = \frac{N_j \cdot N_{ky} \cdot N_{kx}}{\sum_i |\theta_i| \cdot \mathbb{1}[\text{requires\_grad}_i]}$ | `measurements.numel()` | Total optimizable parameters | Data-to-variable ratio (should be $> 1$) |
| Solver Time | $T_{\text{solve}}$ (seconds) | — | Wall-clock time | Computational efficiency metric |