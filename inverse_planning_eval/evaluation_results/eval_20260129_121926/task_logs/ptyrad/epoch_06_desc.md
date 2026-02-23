## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "PtyRAD Automatic Differentiation Ptychography",
  "algorithm_family": "Gradient-Based Optimization with Automatic Differentiation",
  "summary": "Reconstructs object amplitude, phase, probe, and auxiliary parameters from ptychographic diffraction measurements by minimizing a data fidelity loss using gradient descent with PyTorch autodiff through a multislice forward model."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{A}, \boldsymbol{\phi}, \mathbf{P}, \boldsymbol{\theta}} \quad \mathcal{L}_{\text{data}}(\mathbf{y}, \hat{\mathbf{y}}) + \mathcal{L}_{\text{reg}}(\mathbf{A}, \boldsymbol{\phi}, \mathbf{P})$$

subject to constraints applied via `CombinedConstraint` at specified iterations.

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}} = \texttt{CombinedLoss}(\mathbf{y}, \hat{\mathbf{y}}, \mathbf{O}_{\text{patches}})$$

where the specific form depends on `loss_params` configuration (typical choices include amplitude L2: $\sum_j \|\sqrt{\hat{y}_j} - \sqrt{y_j}\|_2^2$).

**regularization_term**:
Defined by active components in `loss_params` (algorithm-configurable; may include object smoothness, probe orthogonality, etc.)

**constraints**:
$$\mathcal{C}(\mathbf{A}, \boldsymbol{\phi}, \mathbf{P}; k) \quad \text{applied when } k \in [\texttt{start\_iter}, \texttt{end\_iter}]$$

Specific constraints defined by `constraint_params` (e.g., object support, probe power normalization, positivity).

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{A}$ (`opt_obja`) | $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | float32 | $\|\texttt{init\_variables['obj']}\|$ | Object amplitude |
| $\boldsymbol{\phi}$ (`opt_objp`) | $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | float32 | $\angle\texttt{init\_variables['obj']}$ | Object phase |
| $\mathbf{P}$ (`opt_probe`) | $(N_{\text{pmode}}, N_{py}, N_{px}, 2)$ | float32 | $\texttt{view\_as\_real}(\texttt{init\_variables['probe']})$ | Probe (real representation of complex) |
| $\boldsymbol{\theta}_{\text{tilt}}$ (`opt_obj_tilts`) | $(N_{\text{tilt}}, 2)$ | float32 | `init_variables['obj_tilts']` | Object tilts (y, x) in milliradians |
| $\Delta z$ (`opt_slice_thickness`) | $(1,)$ | float32 | `init_variables['slice_thickness']` | Slice thickness |
| $\boldsymbol{\delta}_{\text{pos}}$ (`opt_probe_pos_shifts`) | $(N_{\text{scan}}, 2)$ | float32 | `init_variables['probe_pos_shifts']` | Sub-pixel probe position shifts |

### 2.2 Dual Variables

None (gradient-based optimization, not primal-dual splitting)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| `_current_object_patches` | $(N_{\text{batch}}, N_{\text{omode}}, N_{\text{slice}}, N_{py}, N_{px}, 2)$ | float32 | None | Cached object patches for loss computation |
| `loss_iters` | list | float | `[]` | Loss history per iteration |
| `iter_times` | list | float | `[]` | Timing per iteration |
| `H_fixed_tilts_full` | $(N_{\text{tilt}}, N_{py}, N_{px})$ | complex64 | Computed | Precomputed tilted propagators (when tilts fixed) |
| `Kz` | $(N_{py}, N_{px})$ | float32 | $\sqrt{k^2 - K_x^2 - K_y^2}$ | Axial wavevector component |
| `propagator_grid` | $(2, N_{py}, N_{px})$ | float32 | Computed | $(K_y, K_x)$ grid for propagators |
| `shift_probes_grid` | $(2, N_{py}, N_{px})$ | float32 | Computed | Fourier frequency grid for probe shifts |
| `rpy_grid`, `rpx_grid` | $(N_{py}, N_{px})$ | int32 | `meshgrid(arange(Npy), arange(Npx))` | Integer grids for ROI extraction |

### 2.4 Constants

| Symbol | Value/Source | Dtype | Description |
|--------|--------------|-------|-------------|
| `dx` | `init_variables['dx']` | float32 | Real-space pixel size |
| `dk` | `init_variables['dk']` | float32 | Reciprocal-space pixel size |
| $\lambda$ (`lambd`) | `init_variables['lambd']` | float32 | Wavelength |
| $k$ | $2\pi/\lambda$ | float32 | Wavenumber |
| `H` | `init_variables['H']` | complex64 | Base Fresnel propagator |
| `omode_occu` | `init_variables['omode_occu']` | float32 | Object mode occupancy weights |
| `crop_pos` | `init_variables['crop_pos']` | int32 | $(N_{\text{scan}}, 2)$ scan positions (integer pixels) |
| `detector_blur_std` | `model_params['detector_blur_std']` | float/None | Detector PSF blur sigma |
| `obj_preblur_std` | `model_params['obj_preblur_std']` | float/None | Object pre-blur sigma |
| `start_iter` | `model_params['update_params'][*]['start_iter']` | dict | Per-variable optimization start iteration |
| `end_iter` | `model_params['update_params'][*]['end_iter']` | dict | Per-variable optimization end iteration |
| `lr_params` | `model_params['update_params'][*]['lr']` | dict | Per-variable learning rates |
| `gaussian_kernel_size` | 5 | int | Hardcoded Gaussian blur kernel size |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ (`measurements`) | $(N_{\text{scan}}, N_{dy}, N_{dx})$ | float32 | Measured diffraction intensities |
| `meas_padded` | $(N_{py}, N_{px})$ or None | float32 | Template for on-the-fly measurement padding |
| `meas_padded_idx` | $(4,)$ | int32 | Padding indices `[pad_h1, pad_h2, pad_w1, pad_w2]` |
| `meas_scale_factors` | tuple or None | float | Resampling scale factors |

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs | Assumptions |
|------|------|---------|--------|---------|-------------|
| 1 | Load Parameters | $\texttt{params} = \texttt{load\_params}(\texttt{path})$ | `params_path` | `params` (dict) | YAML file exists and is valid |
| 2 | Set Device | $\texttt{device} = \texttt{cuda:}\texttt{gpuid}$ | `gpuid` (int) | `device` | GPU available if gpuid specified |
| 3 | Initialize Variables | $\texttt{init} = \texttt{Initializer}(\texttt{init\_params}).\texttt{init\_all}()$ | `params['init_params']` | `init.init_variables` (dict with obj, probe, etc.) | Data files loadable |
| 4 | Extract Object Amplitude | $\mathbf{A}^{(0)} = \|\mathbf{O}_{\text{init}}\|$ | `init_variables['obj']` (complex) | `opt_obja` $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | — |
| 5 | Extract Object Phase | $\boldsymbol{\phi}^{(0)} = \angle \mathbf{O}_{\text{init}}$ | `init_variables['obj']` (complex) | `opt_objp` $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | — |
| 6 | Convert Probe to Real | $\mathbf{P}^{(0)} = \texttt{view\_as\_real}(\mathbf{P}_{\text{init}})$ | `init_variables['probe']` (complex64) | `opt_probe` $(N_{\text{pmode}}, N_{py}, N_{px}, 2)$ | — |
| 7 | Compute Probe Intensity Sum | $S_P = \sum \|\mathbf{P}\|^2$ | `opt_probe` | `probe_int_sum` (scalar) | For normalization |
| 8 | Create Propagator Grid | $K_y[m] = \texttt{ifftshift}\left(\frac{2\pi (m - N_{py}/2 + 0.5)}{N_{py} \cdot dx}\right)$, $K_x[n] = \texttt{ifftshift}\left(\frac{2\pi (n - N_{px}/2 + 0.5)}{N_{px} \cdot dx}\right)$ | `dx`, $N_{py}$, $N_{px}$ | `propagator_grid` $(2, N_{py}, N_{px})$ | — |
| 9 | Compute Axial Wavevector | $K_z = \sqrt{k^2 - K_x^2 - K_y^2}$ where $k = 2\pi/\lambda$ | `propagator_grid`, `lambd` | `Kz` $(N_{py}, N_{px})$ | Paraxial/Ewald sphere |
| 10 | Initialize Fixed Tilted Propagators | $\mathbf{H}_{\text{tilt}} = \mathbf{H} \odot \exp\left(i \Delta z \left(K_y \tan(\theta_y/10^3) + K_x \tan(\theta_x/10^3)\right)\right)$ | `H`, `opt_obj_tilts`, `opt_slice_thickness`, `propagator_grid` | `H_fixed_tilts_full` $(N_{\text{tilt}}, N_{py}, N_{px})$ | Tilts in mrad |
| 11 | Create Shift Grids | $k_{py}[m] = \texttt{fftfreq}(N_{py})$, $k_{px}[n] = \texttt{fftfreq}(N_{px})$ | $N_{py}$, $N_{px}$ | `shift_probes_grid` $(2, N_{py}, N_{px})$ | For sub-pixel shifts |
| 12 | Create ROI Grids | $\texttt{rpy\_grid}[m,n] = m$, $\texttt{rpx\_grid}[m,n] = n$ | $N_{py}$, $N_{px}$ | `rpy_grid`, `rpx_grid` $(N_{py}, N_{px})$ | Integer indexing |
| 13 | Initialize Loss Function | $\mathcal{L} = \texttt{CombinedLoss}(\texttt{loss\_params})$ | `params['loss_params']` | `loss_fn` | — |
| 14 | Initialize Constraints | $\mathcal{C} = \texttt{CombinedConstraint}(\texttt{constraint\_params})$ | `params['constraint_params']` | `constraint_fn` | — |
| 15 | Prepare Batches | $\texttt{batches} = \texttt{prepare\_recon}(\texttt{model}, \texttt{init}, \texttt{params})$ | model, init, params | `indices`, `batches`, `output_path` | — |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Variable | Initial Value | Shape | Dtype | Source |
|------|----------|---------------|-------|-------|--------|
| 1 | $\mathbf{A}^{(0)}$ | $\|\texttt{init\_variables['obj']}\|$ | $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | float32 | Preprocessed |
| 2 | $\boldsymbol{\phi}^{(0)}$ | $\angle\texttt{init\_variables['obj']}$ | $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | float32 | Preprocessed |
| 3 | $\mathbf{P}^{(0)}$ | $\texttt{view\_as\_real}(\texttt{init\_variables['probe']})$ | $(N_{\text{pmode}}, N_{py}, N_{px}, 2)$ | float32 | Preprocessed |
| 4 | $\boldsymbol{\theta}_{\text{tilt}}^{(0)}$ | `init_variables['obj_tilts']` | $(N_{\text{tilt}}, 2)$ | float32 | Preprocessed |
| 5 | $\Delta z^{(0)}$ | `init_variables['slice_thickness']` | $(1,)$ | float32 | Preprocessed |
| 6 | $\boldsymbol{\delta}_{\text{pos}}^{(0)}$ | `init_variables['probe_pos_shifts']` | $(N_{\text{scan}}, 2)$ | float32 | Preprocessed |
| 7 | `optimizer` | $\texttt{create\_optimizer}(\texttt{optimizer\_params}, \texttt{optimizable\_params})$ | — | — | Adam/SGD with per-variable lr |
| 8 | $k$ | $0$ | scalar | int | Iteration counter |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP (k = 1 to max_iter, controlled by recon_loop)
```

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 9 | Update Requires Grad | For each param $p$: $p.\texttt{requires\_grad} = (\texttt{lr}_p \neq 0) \land (k \geq \texttt{start\_iter}_p) \land (k \leq \texttt{end\_iter}_p \text{ or } \texttt{end\_iter}_p = \texttt{None})$ | `start_iter`, `end_iter`, `lr_params`, $k$ | Updated `requires_grad` flags | Per-variable activation schedule |
| 10 | Sample Batch | $\mathcal{B}_k \subseteq \{0, \ldots, N_{\text{scan}}-1\}$ | `batches`, $k$ | `indices` $(N_{\text{batch}},)$ int | Mini-batch selection |

```
BEGIN_SUBLOOP (Forward Model Computation)
```

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 11 | Construct Object Tensor | $\mathbf{O}_{\text{stack}}[\cdot, \cdot, \cdot, \cdot, 0] = \mathbf{A}$, $\mathbf{O}_{\text{stack}}[\cdot, \cdot, \cdot, \cdot, 1] = \boldsymbol{\phi}$ | $\mathbf{A}$, $\boldsymbol{\phi}$ | `opt_obj` $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox}, 2)$ | Stack amplitude and phase |
| 12 | Compute ROI Y Indices | $\mathbf{G}_y[j, m, n] = \texttt{rpy\_grid}[m, n] + \texttt{crop\_pos}[\mathcal{B}_k[j], 0]$ | `rpy_grid` $(N_{py}, N_{px})$, `crop_pos` $(N_{\text{scan}}, 2)$, `indices` | `obj_ROI_grid_y` $(N_{\text{batch}}, N_{py}, N_{px})$ | [FORWARD] Integer grid offset |
| 13 | Compute ROI X Indices | $\mathbf{G}_x[j, m, n] = \texttt{rpx\_grid}[m, n] + \texttt{crop\_pos}[\mathcal{B}_k[j], 1]$ | `rpx_grid`, `crop_pos`, `indices` | `obj_ROI_grid_x` $(N_{\text{batch}}, N_{py}, N_{px})$ | [FORWARD] Integer grid offset |
| 14 | Extract Object Patches | $\mathbf{O}_{\text{patch}}[j] = \mathbf{O}_{\text{stack}}[:, :, \mathbf{G}_y[j], \mathbf{G}_x[j], :]$ | `opt_obj`, `obj_ROI_grid_y`, `obj_ROI_grid_x` | `object_patches` $(N_{\text{batch}}, N_{\text{omode}}, N_{\text{slice}}, N_{py}, N_{px}, 2)$ | Advanced indexing with permute |
| 15 | Apply Object Pre-blur (conditional) | If `obj_preblur_std` $\neq$ None: $\mathbf{O}_{\text{patch}} \leftarrow \texttt{gaussian\_blur}(\mathbf{O}_{\text{patch}}, \sigma=\texttt{obj\_preblur\_std}, k=5)$ | `object_patches`, `obj_preblur_std` | `object_patches` (same shape) | Separable Gaussian, kernel size 5 |
| 16 | Get Complex Probe | $\mathbf{P}_c = \texttt{view\_as\_complex}(\mathbf{P})$ | `opt_probe` $(N_{\text{pmode}}, N_{py}, N_{px}, 2)$ | `probe` $(N_{\text{pmode}}, N_{py}, N_{px})$ complex64 | Real → Complex view |
| 17 | Compute Probe Shifts (conditional) | If `shift_probes`: $\mathbf{P}_j = \mathcal{F}^{-1}\left[\mathcal{F}[\mathbf{P}_c] \odot \exp\left(-2\pi i (k_{py} \delta_{y,j} + k_{px} \delta_{x,j})\right)\right]$ | `probe`, `opt_probe_pos_shifts`, `shift_probes_grid`, `indices` | `probes` $(N_{\text{batch}}, N_{\text{pmode}}, N_{py}, N_{px})$ complex64 | Sub-pixel Fourier shift |
| 17b | Broadcast Probe (alternative) | If not `shift_probes`: $\mathbf{P}_j = \texttt{broadcast}(\mathbf{P}_c, N_{\text{batch}})$ | `probe` | `probes` $(N_{\text{batch}}, N_{\text{pmode}}, N_{py}, N_{px})$ complex64 | No shifts applied |
| 18 | Select Tilts | If global tilt ($N_{\text{tilt}}=1$): $\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{tilt}}$; else: $\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{tilt}}[\mathcal{B}_k]$ | `opt_obj_tilts`, `indices` | `tilts` $(N_{\text{batch}} \text{ or } 1, 2)$ | Position-dependent or global |
| 19 | Compute Tilted Propagator (case: tilt+change_thickness) | $\mathbf{H}_j = \exp(i \Delta z K_z) \odot \exp\left(i \Delta z \left(K_y \tan(\theta_{y,j}/10^3) + K_x \tan(\theta_{x,j}/10^3)\right)\right)$ | $\Delta z$, `Kz`, `propagator_grid`, `tilts` | `propagators` $(N_{\text{batch}}, N_{py}, N_{px})$ complex64 | Full recomputation |
| 19b | Retrieve Fixed Propagator (case: tilt, fixed thickness) | $\mathbf{H}_j = \mathbf{H}_{\text{fixed\_tilts}}[j]$ or recompute if `change_tilt` | `H`, `H_fixed_tilts_full`, `propagator_grid`, `tilts`, $\Delta z$ | `propagators` $(N_{\text{batch}}, N_{py}, N_{px})$ complex64 | Use precomputed if tilts frozen |
| 19c | Compute Propagator (case: no tilt, change thickness) | $\mathbf{H} = \exp(i \Delta z K_z)$ broadcast | $\Delta z$, `Kz` | `propagators` $(1, N_{py}, N_{px})$ complex64 | Single propagator |
| 19d | Retrieve Base Propagator (case: no tilt, fixed thickness) | $\mathbf{H} = \mathbf{H}_{\text{base}}$ | `H` | `propagators` $(1, N_{py}, N_{px})$ complex64 | Use stored propagator |

```
BEGIN_SUBLOOP (Multislice Forward Model - inside multislice_forward_model_vec_all)
```

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 20 | Convert Patch to Complex | $\mathbf{O}^c_{\text{patch}}[j, o, s] = \mathbf{O}_{\text{patch}}[j, o, s, :, :, 0] \cdot \exp(i \cdot \mathbf{O}_{\text{patch}}[j, o, s, :, :, 1])$ | `object_patches` | `obj_complex` $(N_{\text{batch}}, N_{\text{omode}}, N_{\text{slice}}, N_{py}, N_{px})$ complex64 | Amplitude-phase to complex |
| 21 | Initialize Exit Wave | $\boldsymbol{\psi}^{(0)} = \mathbf{P}_j$ | `probes` | $\boldsymbol{\psi}$ $(N_{\text{batch}}, N_{\text{pmode}}, N_{py}, N_{px})$ complex64 | Start with probe |

```
BEGIN_SUBLOOP (Slice loop: s = 0 to N_slice - 1)
```

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 22 | Object Interaction | $\boldsymbol{\psi}_{\text{int}}[j, p, o] = \boldsymbol{\psi}[j, p] \odot \mathbf{O}^c_{\text{patch}}[j, o, s]$ | $\boldsymbol{\psi}$, `obj_complex[:,:,s]` | $\boldsymbol{\psi}_{\text{int}}$ $(N_{\text{batch}}, N_{\text{pmode}}, N_{\text{omode}}, N_{py}, N_{px})$ complex64 | Element-wise multiplication |
| 23 | Propagate (if $s < N_{\text{slice}} - 1$) | $\boldsymbol{\psi}^{(s+1)} = \mathcal{F}^{-1}\left[\mathbf{H}_j \odot \mathcal{F}[\boldsymbol{\psi}_{\text{int}}]\right]$ | $\boldsymbol{\psi}_{\text{int}}$, `propagators` | $\boldsymbol{\psi}$ updated | [FORWARD] Fresnel propagation via FFT |

```
END_SUBLOOP (Slice loop)
```

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 24 | Compute Far-field | $\boldsymbol{\Psi} = \mathcal{F}[\boldsymbol{\psi}_{\text{exit}}]$ | $\boldsymbol{\psi}_{\text{exit}}$ | $\boldsymbol{\Psi}$ $(N_{\text{batch}}, N_{\text{pmode}}, N_{\text{omode}}, N_{py}, N_{px})$ complex64 | [FORWARD] 2D FFT |
| 25 | Compute Mode Intensities | $I_{j,p,o} = \|\boldsymbol{\Psi}[j, p, o]\|^2$ | $\boldsymbol{\Psi}$ | `mode_intensities` $(N_{\text{batch}}, N_{\text{pmode}}, N_{\text{omode}}, N_{py}, N_{px})$ float32 | Element-wise abs squared |
| 26 | Incoherent Mode Summation | $\hat{y}_j = \sum_{p,o} w_o \cdot I_{j,p,o}$ | `mode_intensities`, `omode_occu` | `dp_fwd` $(N_{\text{batch}}, N_{py}, N_{px})$ float32 | Weighted sum over modes |

```
END_SUBLOOP (Multislice Forward Model)
```

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 27 | Apply Detector Blur (conditional) | If `detector_blur_std` $\neq$ None: $\hat{\mathbf{y}} \leftarrow \texttt{gaussian\_blur}(\hat{\mathbf{y}}, \sigma=\texttt{detector\_blur\_std}, k=5)$ | `dp_fwd`, `detector_blur_std` | `dp_fwd` $(N_{\text{batch}}, N_{py}, N_{px})$ | Simulates detector PSF |
| 28 | Cache Object Patches | $\texttt{\_current\_object\_patches} \leftarrow \mathbf{O}_{\text{patch}}$ | `object_patches` | Cached attribute | For object-specific loss terms |

```
END_SUBLOOP (Forward Model Computation)
```

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 29 | Retrieve Measurements | $\mathbf{y}_{\mathcal{B}} = \texttt{measurements}[\mathcal{B}_k]$ | `measurements`, `indices` | `meas_batch` $(N_{\text{batch}}, N_{dy}, N_{dx})$ float32 | Batch extraction |
| 30 | Apply Measurement Padding (conditional) | If `meas_padded` $\neq$ None: canvas $\leftarrow$ `meas_padded`; canvas$[h_1:h_2, w_1:w_2] \leftarrow \mathbf{y}_{\mathcal{B}}$; $\mathbf{y}_{\mathcal{B}} \leftarrow$ canvas | `meas_batch`, `meas_padded`, `meas_padded_idx` | `meas_batch` $(N_{\text{batch}}, N_{py}, N_{px})$ | On-the-fly padding |
| 31 | Apply Measurement Resampling (conditional) | If `meas_scale_factors` $\neq$ None: $\mathbf{y}_{\mathcal{B}} \leftarrow \texttt{interpolate}(\mathbf{y}_{\mathcal{B}}, \texttt{scale\_factor}) / \prod(\texttt{scale\_factor})$ | `meas_batch`, `meas_scale_factors` | `meas_batch` resampled | Bilinear interpolation |
| 32 | Compute Loss | $\mathcal{L}^{(k)} = \texttt{loss\_fn}(\hat{\mathbf{y}}, \mathbf{y}_{\mathcal{B}}, \texttt{\_current\_object\_patches})$ | `dp_fwd`, `meas_batch`, `_current_object_patches` | `loss` scalar | CombinedLoss evaluation |
| 33