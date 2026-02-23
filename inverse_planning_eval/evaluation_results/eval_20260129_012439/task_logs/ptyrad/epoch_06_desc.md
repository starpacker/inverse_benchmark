

## 0. ALGORITHM IDENTIFICATION

```json
{
  "algorithm_name": "PtyRAD - Ptychographic Reconstruction via Automatic Differentiation",
  "algorithm_family": "Gradient-based optimization with automatic differentiation",
  "summary": "Multislice ptychographic reconstruction using first-order gradient descent with PyTorch autograd to jointly optimize object amplitude/phase, probe, tilts, slice thickness, and probe positions by minimizing a combined loss between predicted and measured diffraction patterns."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{O_a, O_p, P, \boldsymbol{\theta}, \Delta z, \Delta \mathbf{s}} \mathcal{L}_{\text{data}}(\mathbf{y}, \hat{\mathbf{y}}) + \mathcal{L}_{\text{reg}}(O_a, O_p, P)$$

subject to constraints applied via `CombinedConstraint` at specified iterations.

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}}(\mathbf{y}, \hat{\mathbf{y}}) = \text{CombinedLoss}(\mathbf{y}_{\text{batch}}, \hat{\mathbf{y}}_{\text{batch}})$$

where $\hat{\mathbf{y}}_i = A(O_a, O_p, P, \boldsymbol{\theta}, \Delta z, \Delta \mathbf{s}; i)$ is the forward model output for scan position $i$. The specific loss type (L1, L2, amplitude-based) is configured via `loss_params`.

**regularization_term**:
$$\mathcal{L}_{\text{reg}} = \sum_{k} w_k \cdot R_k(O_a, O_p, P)$$

where $R_k$ are regularization terms defined in `CombinedLoss` with weights $w_k$ from configuration.

**constraints**:
$$\mathcal{C}(O_a, O_p, P, \boldsymbol{\theta}, \Delta z, \Delta \mathbf{s}; \text{iter}) = \text{CombinedConstraint applied at specified iterations}$$

Constraints include projection operations applied to variables at iterations specified by `start_iter` in `constraint_params`.

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $O_a$ (`opt_obja`) | $(N_s, N_o, N_{oy}, N_{ox})$ | float32 | $\|O_{\text{init}}\|$ | Object amplitude across slices and object modes |
| $O_p$ (`opt_objp`) | $(N_s, N_o, N_{oy}, N_{ox})$ | float32 | $\angle O_{\text{init}}$ | Object phase across slices and object modes |
| $P$ (`opt_probe`) | $(N_p, N_{py}, N_{px}, 2)$ | float32 | `init_variables['probe']` as real view | Probe modes stored as real-valued tensor (real, imag) |
| $\boldsymbol{\theta}$ (`opt_obj_tilts`) | $(N_t, 2)$ | float32 | `init_variables['obj_tilts']` | Object tilts in mrad, $N_t=1$ for global or $N_t=N_{\text{scan}}$ for per-position |
| $\Delta z$ (`opt_slice_thickness`) | $(1,)$ | float32 | `init_variables['slice_thickness']` | Slice thickness for Fresnel propagation |
| $\Delta \mathbf{s}$ (`opt_probe_pos_shifts`) | $(N_{\text{scan}}, 2)$ | float32 | `init_variables['probe_pos_shifts']` | Sub-pixel probe position shifts $(y, x)$ |

### 2.2 Dual Variables

None - this algorithm uses direct gradient descent without dual formulation.

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| `propagator_grid` | $(2, N_{py}, N_{px})$ | float32 | Computed from `dx`, probe shape | $(K_y, K_x)$ frequency grids for Fresnel propagator |
| `rpy_grid` | $(N_{py}, N_{px})$ | int32 | `meshgrid(arange(Npy), arange(Npx))` | Row indices for object ROI extraction |
| `rpx_grid` | $(N_{py}, N_{px})$ | int32 | `meshgrid(arange(Npy), arange(Npx))` | Column indices for object ROI extraction |
| `shift_probes_grid` | $(2, N_{py}, N_{px})$ | float32 | `fftfreq` grids | Frequency grids for probe shifting |
| `shift_object_grid` | $(2, N_{oy}, N_{ox})$ | float32 | `fftfreq` grids | Frequency grids for object shifting |
| `H_fixed_tilts_full` | $(N_t, N_{py}, N_{px})$ | complex64 | Precomputed | Cached tilted propagator when tilts are fixed |
| `k` | scalar | float32 | $2\pi/\lambda$ | Wavenumber |
| `Kz` | $(N_{py}, N_{px})$ | float32 | $\sqrt{k^2 - K_x^2 - K_y^2}$ | Axial frequency component |
| `probe_int_sum` | scalar | float32 | $\sum |P|^2$ | Probe intensity sum for normalization |
| `loss_iters` | list | float32 | `[]` | Loss values per iteration |
| `iter_times` | list | float32 | `[]` | Time per iteration |
| `_current_object_patches` | $(N_b, N_s, N_o, N_{py}, N_{px}, 2)$ | float32 | None | Cached object patches for loss computation |

### 2.4 Constants

| Symbol | Shape | Dtype | Value/Source | Description |
|--------|-------|-------|--------------|-------------|
| `omode_occu` | $(N_o,)$ | float32 | `init_variables['omode_occu']` | Object mode occupancy weights |
| $H$ | $(N_{py}, N_{px})$ | complex64 | `init_variables['H']` | Base Fresnel propagator |
| $\mathbf{y}$ (`measurements`) | $(N_{\text{scan}}, N_{dy}, N_{dx})$ | float32 | `init_variables['measurements']` | Measured diffraction patterns |
| `N_scan_slow` | scalar | int32 | `init_variables['N_scan_slow']` | Number of slow scan positions |
| `N_scan_fast` | scalar | int32 | `init_variables['N_scan_fast']` | Number of fast scan positions |
| `crop_pos` | $(N_{\text{scan}}, 2)$ | int32 | `init_variables['crop_pos']` | Integer crop positions for object ROI |
| `dx` | scalar | float32 | `init_variables['dx']` | Real-space pixel size |
| `dk` | scalar | float32 | `init_variables['dk']` | Reciprocal-space pixel size |
| $\lambda$ (`lambd`) | scalar | float32 | `init_variables['lambd']` | Wavelength |
| `detector_blur_std` | scalar or None | float32 | `model_params['detector_blur_std']` | Detector blur standard deviation |
| `obj_preblur_std` | scalar or None | float32 | `model_params['obj_preblur_std']` | Object pre-blur standard deviation |
| `lr_params` | dict | - | `model_params['update_params']` | Learning rates per variable |
| `start_iter` | dict | int | `update_params[key]['start_iter']` | Iteration to start optimizing each variable |
| `end_iter` | dict | int | `update_params[key]['end_iter']` | Iteration to stop optimizing each variable |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $\mathbf{y}$ | $(N_{\text{scan}}, N_{dy}, N_{dx})$ | float32 | `init_variables['measurements']` | Measured diffraction pattern intensities |
| `meas_padded` | $(N_{py}, N_{px})$ | float32 | `init_variables['on_the_fly_meas_padded']` | Padding template for on-the-fly measurement resizing |
| `meas_padded_idx` | $(4,)$ | int32 | `init_variables['on_the_fly_meas_padded_idx']` | Padding indices $(h_1, h_2, w_1, w_2)$ |
| `meas_scale_factors` | $(2,)$ | float32 | `init_variables['on_the_fly_meas_scale_factors']` | Scale factors for measurement resampling |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| 1 | Load Parameters | $\text{params} = \text{load\_params}(\text{path})$ | `params_path`: string | `params`: dict | YAML file exists and is valid |
| 2 | Set Device | $\text{device} = \text{set\_gpu\_device}(\text{gpuid})$ | `gpuid`: int | `device`: torch.device | GPU available if gpuid specified |
| 3 | Initialize Variables | $\text{init} = \text{Initializer}(\text{init\_params}).\text{init\_all}()$ | `params['init_params']`: dict | `init.init_variables`: dict containing all initial arrays | Data files exist |
| 4 | Extract Object Initial | $O_{\text{init}} = \text{init\_variables}[\text{'obj'}]$ | `init_variables`: dict | $O_{\text{init}}$: $(N_s, N_o, N_{oy}, N_{ox})$ complex64 | Object initialized |
| 5 | Compute Initial Amplitude | $O_a^{(0)} = |O_{\text{init}}|$ | $O_{\text{init}}$: complex64 | $O_a^{(0)}$: $(N_s, N_o, N_{oy}, N_{ox})$ float32 | Element-wise absolute value |
| 6 | Compute Initial Phase | $O_p^{(0)} = \angle O_{\text{init}}$ | $O_{\text{init}}$: complex64 | $O_p^{(0)}$: $(N_s, N_o, N_{oy}, N_{ox})$ float32 | Element-wise angle |
| 7 | Initialize Loss Function | $\mathcal{L} = \text{CombinedLoss}(\text{loss\_params}, \text{device})$ | `loss_params`: dict, `device` | `loss_fn`: callable | Loss parameters valid |
| 8 | Initialize Constraints | $\mathcal{C} = \text{CombinedConstraint}(\text{constraint\_params}, \text{device})$ | `constraint_params`: dict, `device` | `constraint_fn`: callable | Constraint parameters valid |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | `opt_obja` | $O_a^{(0)} = |\text{init\_variables}[\text{'obj'}]|$ | $(N_s, N_o, N_{oy}, N_{ox})$ | float32 | Preprocessing step 5 |
| 2 | `opt_objp` | $O_p^{(0)} = \angle \text{init\_variables}[\text{'obj'}]$ | $(N_s, N_o, N_{oy}, N_{ox})$ | float32 | Preprocessing step 6 |
| 3 | `opt_probe` | $P^{(0)} = \text{view\_as\_real}(\text{init\_variables}[\text{'probe'}])$ | $(N_p, N_{py}, N_{px}, 2)$ | float32 | `init_variables` |
| 4 | `opt_obj_tilts` | $\boldsymbol{\theta}^{(0)} = \text{init\_variables}[\text{'obj\_tilts'}]$ | $(N_t, 2)$ | float32 | `init_variables` |
| 5 | `opt_slice_thickness` | $\Delta z^{(0)} = \text{init\_variables}[\text{'slice\_thickness'}]$ | $(1,)$ | float32 | `init_variables` |
| 6 | `opt_probe_pos_shifts` | $\Delta \mathbf{s}^{(0)} = \text{init\_variables}[\text{'probe\_pos\_shifts'}]$ | $(N_{\text{scan}}, 2)$ | float32 | `init_variables` |
| 7 | `propagator_grid` | $K_y[j] = \text{ifftshift}\left(\frac{2\pi}{dx} \cdot \frac{j - N_{py}/2 + 0.5}{N_{py}}\right)$, $K_x[i] = \text{ifftshift}\left(\frac{2\pi}{dx} \cdot \frac{i - N_{px}/2 + 0.5}{N_{px}}\right)$ | $(2, N_{py}, N_{px})$ | float32 | Computed |
| 8 | `k` | $k = \frac{2\pi}{\lambda}$ | scalar | float32 | Computed |
| 9 | `Kz` | $K_z = \sqrt{k^2 - K_x^2 - K_y^2}$ | $(N_{py}, N_{px})$ | float32 | Computed |
| 10 | `H_fixed_tilts_full` | $H_{\text{fixed}} = H \cdot \exp\left(i \Delta z^{(0)} \left(K_y \tan\left(\frac{\theta_y}{1000}\right) + K_x \tan\left(\frac{\theta_x}{1000}\right)\right)\right)$ | $(N_t, N_{py}, N_{px})$ | complex64 | Computed |
| 11 | `optimizer` | $\text{optimizer} = \text{create\_optimizer}(\text{optimizer\_params}, \text{optimizable\_params})$ | - | - | Adam or configured optimizer |
| 12 | `indices` | $\text{indices} = \text{prepare\_recon}(\text{model}, \text{init}, \text{params})[0]$ | $(N_{\text{scan}},)$ | int64 | Scan indices |
| 13 | `batches` | $\text{batches} = \text{prepare\_recon}(\text{model}, \text{init}, \text{params})[1]$ | list of index arrays | int64 | Batch partitioning |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (iter $t = 1$ to $N_{\text{iter}}$, batch $b$ over `batches`)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 14 | Update requires_grad flags | For each param $p$: $p.\text{requires\_grad} = (\text{lr}_p \neq 0) \land (\text{start\_iter}_p \leq t) \land (t \leq \text{end\_iter}_p)$ | `start_iter`, `end_iter`, `lr_params`, iter $t$ | Updated gradient flags | Enables/disables optimization per variable |
| 15 | Get batch indices | $\mathcal{I}_b = \text{batches}[b]$ | `batches`: list, $b$: int | $\mathcal{I}_b$: $(N_b,)$ int64 | Current batch indices |
| 16 | [FORWARD] Extract object ROI grid Y | $G_y[n, j, i] = \text{rpy\_grid}[j, i] + \text{crop\_pos}[\mathcal{I}_b[n], 0]$ | `rpy_grid`: $(N_{py}, N_{px})$, `crop_pos`: $(N_{\text{scan}}, 2)$, $\mathcal{I}_b$ | $G_y$: $(N_b, N_{py}, N_{px})$ int32 | Batch-wise Y coordinates |
| 17 | [FORWARD] Extract object ROI grid X | $G_x[n, j, i] = \text{rpx\_grid}[j, i] + \text{crop\_pos}[\mathcal{I}_b[n], 1]$ | `rpx_grid`: $(N_{py}, N_{px})$, `crop_pos`: $(N_{\text{scan}}, 2)$, $\mathcal{I}_b$ | $G_x$: $(N_b, N_{py}, N_{px})$ int32 | Batch-wise X coordinates |
| 18 | [FORWARD] Stack object amp/phase | $O_{\text{stack}} = \text{stack}([O_a, O_p], \text{dim}=-1)$ | $O_a$: $(N_s, N_o, N_{oy}, N_{ox})$, $O_p$: $(N_s, N_o, N_{oy}, N_{ox})$ | $O_{\text{stack}}$: $(N_s, N_o, N_{oy}, N_{ox}, 2)$ | Combine for indexing |
| 19 | [FORWARD] Extract object patches | $O_{\text{patch}}[n, s, o, j, i, :] = O_{\text{stack}}[s, o, G_y[n,j,i], G_x[n,j,i], :]$ | $O_{\text{stack}}$, $G_y$, $G_x$ | $O_{\text{patch}}$: $(N_b, N_s, N_o, N_{py}, N_{px}, 2)$ | Advanced indexing |
| 20 | [FORWARD] Optional object preblur | If `obj_preblur_std` $\neq$ None: $O_{\text{patch}} = \text{gaussian\_blur}(O_{\text{patch}}, k=5, \sigma=\text{obj\_preblur\_std})$ | $O_{\text{patch}}$, `obj_preblur_std` | $O_{\text{patch}}$: $(N_b, N_s, N_o, N_{py}, N_{px}, 2)$ | Kernel size 5 |
| 21 | [FORWARD] Get complex probe | $P_c = \text{view\_as\_complex}(P)$ | $P$: $(N_p, N_{py}, N_{px}, 2)$ | $P_c$: $(N_p, N_{py}, N_{px})$ complex64 | Reinterpret as complex |
| 22 | [FORWARD] Compute probe shifts (if enabled) | If `shift_probes`: $P_b = \mathcal{F}^{-1}\left\{\exp\left(-2\pi i (k_y \Delta s_{n,0} + k_x \Delta s_{n,1})\right) \cdot \mathcal{F}\{P_c\}\right\}$ for $n \in \mathcal{I}_b$ | $P_c$, $\Delta \mathbf{s}[\mathcal{I}_b]$: $(N_b, 2)$, `shift_probes_grid` | $P_b$: $(N_b, N_p, N_{py}, N_{px})$ complex64 | Sub-pixel Fourier shift |
| 23 | [FORWARD] Broadcast probe (if no shifts) | If not `shift_probes`: $P_b = \text{broadcast}(P_c, (N_b, N_p, N_{py}, N_{px}))$ | $P_c$: $(N_p, N_{py}, N_{px})$ | $P_b$: $(N_b, N_p, N_{py}, N_{px})$ complex64 | Expand dims |
| 24 | [FORWARD] Compute propagator (tilted, variable dz) | If `tilt_obj` and `change_thickness`: $H_b = \exp(i \Delta z K_z) \cdot \exp\left(i \Delta z \left(K_y \tan\left(\frac{\theta_{n,0}}{1000}\right) + K_x \tan\left(\frac{\theta_{n,1}}{1000}\right)\right)\right)$ | $\Delta z$, $K_z$, $K_y$, $K_x$, $\boldsymbol{\theta}[\mathcal{I}_b]$ | $H_b$: $(N_b, N_{py}, N_{px})$ complex64 | Full propagator computation |
| 25 | [FORWARD] Compute propagator (tilted, fixed dz) | If `tilt_obj` and not `change_thickness` and `change_tilt`: $H_b = H \cdot \exp\left(i \Delta z \left(K_y \tan\left(\frac{\theta_{n,0}}{1000}\right) + K_x \tan\left(\frac{\theta_{n,1}}{1000}\right)\right)\right)$ | $H$, $\Delta z$, $K_y$, $K_x$, $\boldsymbol{\theta}[\mathcal{I}_b]$ | $H_b$: $(N_b, N_{py}, N_{px})$ complex64 | Use cached base propagator |
| 26 | [FORWARD] Use cached propagator (fixed tilts) | If `tilt_obj` and not `change_thickness` and not `change_tilt`: $H_b = H_{\text{fixed}}[\mathcal{I}_b]$ or $H_b = H_{\text{fixed}}$ (global) | $H_{\text{fixed}}$ | $H_b$: $(N_b, N_{py}, N_{px})$ or $(1, N_{py}, N_{px})$ complex64 | Precomputed |
| 27 | [FORWARD] Compute propagator (no tilt, variable dz) | If not `tilt_obj` and `change_thickness`: $H_b = \exp(i \Delta z K_z)[None,]$ | $\Delta z$, $K_z$ | $H_b$: $(1, N_{py}, N_{px})$ complex64 | Broadcast over batch |
| 28 | [FORWARD] Use base propagator (no tilt, fixed dz) | If not `tilt_obj` and not `change_thickness`: $H_b = H[None,]$ | $H$ | $H_b$: $(1, N_{py}, N_{px})$ complex64 | Broadcast over batch |
| 29 | [FORWARD] Convert object patch to complex | $O_c[n,s,o,j,i] = O_{\text{patch}}[n,s,o,j,i,0] \cdot \exp(i \cdot O_{\text{patch}}[n,s,o,j,i,1])$ | $O_{\text{patch}}$: $(N_b, N_s, N_o, N_{py}, N_{px}, 2)$ | $O_c$: $(N_b, N_s, N_o, N_{py}, N_{px})$ complex64 | Amplitude × phasor |
| 30 | [FORWARD] Multislice propagation | $\hat{\mathbf{y}}_b = \text{multislice\_forward\_model\_vec\_all}(O_c, P_b, H_b, \text{omode\_occu})$ | $O_c$, $P_b$, $H_b$, `omode_occu` | $\hat{\mathbf{y}}_b$: $(N_b, N_{py}, N_{px})$ float32 | See detailed expansion below |
| 31 | [FORWARD] Optional detector blur | If `detector_blur_std` $\neq$ None: $\hat{\mathbf{y}}_b = \text{gaussian\_blur}(\hat{\mathbf{y}}_b, k=5, \sigma=\text{detector\_blur\_std})$ | $\hat{\mathbf{y}}_b$, `detector_blur_std` | $\hat{\mathbf{y}}_b$: $(N_b, N_{py}, N_{px})$ float32 | Kernel size 5 |
| 32 | Get measurements for batch | $\mathbf{y}_b = \text{measurements}[\mathcal{I}_b]$ | `measurements`, $\mathcal{I}_b$ | $\mathbf{y}_b$: $(N_b, N_{dy}, N_{dx})$ float32 | Index into stored measurements |
| 33 | Optional measurement padding | If `meas_padded` $\neq$ None: Create canvas, insert $\mathbf{y}_b$ at indices | $\mathbf{y}_b$, `meas_padded`, `meas_padded_idx` | $\mathbf{y}_b$: $(N_b, N_{py}, N_{px})$ float32 | Zero-pad to match prediction size |
| 34 | Optional measurement resampling | If `meas_scale_factors` $\neq$ None: $\mathbf{y}_b = \text{interpolate}(\mathbf{y}_b, \text{scale\_factor}) / \prod(\text{scale\_factor})$ | $\mathbf{y}_b$, `meas_scale_factors` | $\mathbf{y}_b$: $(N_b, N_{py}', N_{px}')$ float32 | Bilinear interpolation |
| 35 | Compute loss | $\ell = \mathcal{L}(\mathbf{y}_b, \hat{\mathbf{y}}_b, O_{\text{patch}})$ | $\mathbf{y}_b$, $\hat{\mathbf{y}}_b$, $O_{\text{patch}}$, `loss_fn` | $\ell$: scalar float32 | Combined loss evaluation |
| 36 | [ADJOINT] Compute gradients | $\nabla_{O_a} \ell, \nabla_{O_p} \ell, \nabla_P \ell, \nabla_{\boldsymbol{\theta}} \ell, \nabla_{\Delta z} \ell, \nabla_{\Delta \mathbf{s}} \ell = \text{autograd.backward}(\ell)$ | $\ell$, all parameters with `requires_grad=True` | Gradients stored in `.grad` attributes | PyTorch automatic differentiation |
| 37 | Optimizer step | $\text{optimizer.step}()$ - updates all parameters with nonzero lr | Gradients, `optimizer` | Updated $O_a, O_p, P, \boldsymbol{\theta}, \Delta z, \Delta \mathb