## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: PtyRAD (Ptychographic Reconstruction via Automatic Differentiation)
- **algorithm_family**: Gradient-based iterative optimization for multislice ptychographic phase retrieval
- **summary**: Reconstructs complex-valued object transmission functions and probe wavefield from diffraction intensity measurements using automatic differentiation with a multislice forward model.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{a}, \boldsymbol{\phi}, \mathbf{P}, \boldsymbol{\theta}, \Delta z, \boldsymbol{\delta}} \quad \mathcal{L}_{\text{data}}(\mathbf{y}, \hat{\mathbf{y}}) + \mathcal{L}_{\text{reg}}(\mathbf{a}, \boldsymbol{\phi}, \mathbf{P}) + \mathcal{I}_{\mathcal{C}}(\mathbf{a}, \boldsymbol{\phi}, \mathbf{P})$$

where $\hat{\mathbf{y}} = \mathcal{A}(\mathbf{a}, \boldsymbol{\phi}, \mathbf{P}, \boldsymbol{\theta}, \Delta z, \boldsymbol{\delta})$ is the forward model output.

### data_fidelity_term
$$\mathcal{L}_{\text{data}}(\mathbf{y}, \hat{\mathbf{y}}) = \text{CombinedLoss}(\mathbf{y}, \hat{\mathbf{y}})$$

The specific loss form is configured via `loss_params` dictionary. The code uses a generic `CombinedLoss` class that supports multiple loss types activated by `state: True`.

### regularization_term
$$\mathcal{L}_{\text{reg}} = \text{CombinedLoss regularization components (configured externally)}$$

Regularization is embedded within the `CombinedLoss` object based on `loss_params`.

### constraints
$$\mathcal{I}_{\mathcal{C}}(\cdot) = \begin{cases} 0 & \text{if constraints satisfied} \\ +\infty & \text{otherwise} \end{cases}$$

Constraints are applied via `CombinedConstraint` class with iteration-dependent activation (`start_iter` parameter).

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{a}$ (`opt_obja`) | $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | `float32` | $\|\texttt{init\_variables['obj']}\|$ | Object amplitude |
| $\boldsymbol{\phi}$ (`opt_objp`) | $(N_{\text{omode}}, N_{\text{slice}}, N_{oy}, N_{ox})$ | `float32` | $\angle\texttt{init\_variables['obj']}$ | Object phase |
| $\mathbf{P}$ (`opt_probe`) | $(N_{\text{pmode}}, N_{py}, N_{px}, 2)$ | `float32` | `view_as_real(init_variables['probe'])` | Probe as real tensor (last dim: real/imag) |
| $\boldsymbol{\theta}$ (`opt_obj_tilts`) | $(N_{\text{tilt}}, 2)$ | `float32` | `init_variables['obj_tilts']` | Object tilt angles (y, x) in milliradians |
| $\Delta z$ (`opt_slice_thickness`) | $(1,)$ or $(N_{\text{slice}},)$ | `float32` | `init_variables['slice_thickness']` | Slice thickness |
| $\boldsymbol{\delta}$ (`opt_probe_pos_shifts`) | $(N_{\text{scan}}, 2)$ | `float32` | `init_variables['probe_pos_shifts']` | Sub-pixel probe position shifts |

### 2.2 Dual Variables

None explicitly defined; optimization uses standard gradient descent (no primal-dual splitting).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| `H_fixed_tilts_full` | $(N_{\text{tilt}}, N_{py}, N_{px})$ | `complex64` | Computed in `init_propagator_vars()` | Precomputed propagator with fixed tilts |
| `propagator_grid` | $(2, N_{py}, N_{px})$ | `float32` | Computed in `create_grids()` | $(K_y, K_x)$ frequency grids |
| `shift_probes_grid` | $(2, N_{py}, N_{px})$ | `float32` | `fftfreq` grids | Probe shifting frequency grid |
| `shift_object_grid` | $(2, N_{oy}, N_{ox})$ | `float32` | `fftfreq` grids | Object shifting frequency grid |
| `rpy_grid`, `rpx_grid` | $(N_{py}, N_{px})$ | `int32` | `torch.arange` | ROI selection grids |
| `Kz` | $(N_{py}, N_{px})$ | `float32` | $\sqrt{k^2 - K_x^2 - K_y^2}$ | Axial spatial frequency |
| `k` | scalar | `float32` | $2\pi/\lambda$ | Wavenumber |
| `probe_int_sum` | scalar | `float32` | $\sum\|\mathbf{P}_{\text{complex}}\|^2$ | Probe intensity sum |
| `loss_iters` | list | varies | `[]` | Loss history |
| `iter_times` | list | varies | `[]` | Iteration timing history |
| `_current_object_patches` | $(N_b, N_{\text{omode}}, N_{\text{slice}}, N_{py}, N_{px}, 2)$ | `float32` | `None` | Cached object patches for loss computation |

### 2.4 Constants

| Symbol | Value/Source | Description |
|--------|--------------|-------------|
| `detector_blur_std` | `model_params['detector_blur_std']` | Detector PSF blur standard deviation |
| `obj_preblur_std` | `model_params['obj_preblur_std']` | Object pre-blur standard deviation |
| `omode_occu` | `init_variables['omode_occu']` | Object mode occupancy weights |
| `H` | `init_variables['H']` | Base Fresnel propagator |
| `N_scan_slow`, `N_scan_fast` | `init_variables[...]` | Scan grid dimensions |
| `crop_pos` | `init_variables['crop_pos']` | Object crop positions per scan point |
| `dx` | `init_variables['dx']` | Real-space pixel size |
| `dk` | `init_variables['dk']` | Reciprocal-space pixel size |
| `lambd` ($\lambda$) | `init_variables['lambd']` | Wavelength |
| Kernel size for blur | $5$ | Hardcoded in `gaussian_blur` calls |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $\mathbf{y}$ (`measurements`) | $(N_{\text{scan}}, N_{dy}, N_{dx})$ | `float32` | `init_variables['measurements']` | Measured diffraction intensities |
| `meas_padded` | $(N_{dy}', N_{dx}')$ | `float32` | `init_variables['on_the_fly_meas_padded']` | Padded measurement template (optional) |
| `meas_padded_idx` | $(4,)$ | `int32` | `init_variables['on_the_fly_meas_padded_idx']` | Padding indices $(h_1, h_2, w_1, w_2)$ |
| `meas_scale_factors` | tuple | float | `init_variables['on_the_fly_meas_scale_factors']` | Resampling factors (optional) |

---

## 3. PREPROCESSING PIPELINE

| Step Order | Step Name | Formula | Inputs | Outputs |
|------------|-----------|---------|--------|---------|
| 0 | Parameter Loading | $\texttt{params} = \texttt{load\_params}(\texttt{path})$ | `params_path` | `params` dict |
| 1 | Device Setup | $\texttt{device} = \texttt{set\_gpu\_device}(\texttt{gpuid})$ | `gpuid` | PyTorch device |
| 2 | Variable Initialization | $\texttt{init} = \texttt{Initializer}(\texttt{init\_params}).\texttt{init\_all}()$ | `params['init_params']` | `init_variables` dict |
| 3 | Loss Function Init | $\mathcal{L} = \texttt{CombinedLoss}(\texttt{loss\_params})$ | `params['loss_params']` | `loss_fn` callable |
| 4 | Constraint Function Init | $\mathcal{C} = \texttt{CombinedConstraint}(\texttt{constraint\_params})$ | `params['constraint_params']` | `constraint_fn` callable |
| 5 | Object Decomposition | $\mathbf{a} = \|\mathbf{O}_{\text{init}}\|, \quad \boldsymbol{\phi} = \angle\mathbf{O}_{\text{init}}$ | `init_variables['obj']` (complex) | `opt_obja`, `opt_objp` (float32) |
| 6 | Probe Real View | $\mathbf{P}_{\text{real}} = \texttt{view\_as\_real}(\mathbf{P}_{\text{complex}})$ | `init_variables['probe']` (complex64) | `opt_probe` shape $(..., 2)$ |
| 7 | Grid Creation | See `create_grids()` | Object/probe shapes, `dx` | Frequency grids for shifting/propagation |
| 8 | Propagator Init | $H_{\text{fixed}} = H \cdot e^{i\Delta z(K_y\tan\theta_y + K_x\tan\theta_x)}$ | `H`, `opt_obj_tilts`, `opt_slice_thickness` | `H_fixed_tilts_full` |

### Grid Creation Details (Step 7)

**Propagator grid**:
$$y_{\text{grid}}[j] = \frac{j - N_{py}/2 + 0.5}{N_{py}}, \quad j = 0, \ldots, N_{py}-1$$
$$K_y = \texttt{ifftshift}\left(\frac{2\pi \cdot y_{\text{grid}}}{dx}\right), \quad K_x = \texttt{ifftshift}\left(\frac{2\pi \cdot x_{\text{grid}}}{dx}\right)$$

**Shift grids**:
$$k_{py} = \texttt{fftfreq}(N_{py}), \quad k_{px} = \texttt{fftfreq}(N_{px})$$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Operation | Formula | Outputs |
|------|-----------|---------|---------|
| I1 | Model Creation | `model = PtychoAD(init_variables, model_params, device)` | `model` with parameters |
| I2 | Optimizer Creation | `optimizer = create_optimizer(optimizer_params, optimizable_params)` | PyTorch optimizer |
| I3 | Batch Preparation | `indices, batches, output_path = prepare_recon(model, init, params)` | Scan indices, batch structure |
| I4 | Parameter Activation | For each param: `requires_grad = (lr ≠ 0) and (start_iter == 1)` | Gradient flags |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP (iteration n = 1 to N_iter)
```

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| L1 | Parameter Scheduling | For each param $p$: enable grad if $n \geq \texttt{start\_iter}[p]$, disable if $n > \texttt{end\_iter}[p]$ | Current iteration $n$ | Updated `requires_grad` flags | [SCHEDULING] |

```
  BEGIN_LOOP (batch b = 1 to N_batches)
```

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| L2 | Index Selection | $\mathcal{I}_b = \texttt{batches}[b]$ | Batch index | Scan indices for batch | |
| L3 | Object ROI Extraction | $\mathbf{O}_{\text{ROI}}[i] = (\mathbf{a}, \boldsymbol{\phi})[:, :, c_y[i]:c_y[i]+N_{py}, c_x[i]:c_x[i]+N_{px}]$ | `crop_pos`, indices | $(N_b, N_{\text{omode}}, N_{\text{slice}}, N_{py}, N_{px}, 2)$ | |
| L4 | Object Pre-blur (optional) | $\tilde{\mathbf{O}}_{\text{ROI}} = \texttt{gaussian\_blur}(\mathbf{O}_{\text{ROI}}, k=5, \sigma=\sigma_{\text{obj}})$ | `obj_preblur_std` | Blurred object patches | |
| L5 | Complex Object Formation | $O[i] = \tilde{a}[i] \cdot e^{i\tilde{\phi}[i]}$ | Object patches (amp, phase) | Complex object $(N_b, N_{\text{omode}}, N_{\text{slice}}, N_{py}, N_{px})$ | Implicit in forward model |
| L6 | Probe Retrieval | $\mathbf{P}_{\text{complex}} = \texttt{view\_as\_complex}(\mathbf{P})$ | `opt_probe` | Complex probe | |
| L7 | Probe Shifting (optional) | $\tilde{P}[i] = \mathcal{F}^{-1}\left[\mathcal{F}[P] \cdot e^{-2\pi i(k_y \delta_y[i] + k_x \delta_x[i])}\right]$ | `opt_probe_pos_shifts`, shift grid | Shifted probes $(N_b, N_{\text{pmode}}, N_{py}, N_{px})$ | [FORWARD] |
| L8 | Propagator Computation | See below | Tilts, thickness, grids | $H_{\text{prop}}$ $(N_b, N_{py}, N_{px})$ or $(1, N_{py}, N_{px})$ | |
| L9 | Multislice Forward Model | $\hat{y}[i] = \texttt{multislice\_forward\_model\_vec\_all}(O[i], \tilde{P}[i], H_{\text{prop}})$ | Object, probe, propagator | Predicted diffraction $(N_b, N_{dy}, N_{dx})$ | [FORWARD] |
| L10 | Detector Blur (optional) | $\hat{y}_{\text{blur}} = \texttt{gaussian\_blur}(\hat{y}, k=5, \sigma=\sigma_{\text{det}})$ | `detector_blur_std` | Blurred predictions | |
| L11 | Measurement Retrieval | $y_b = \mathbf{y}[\mathcal{I}_b]$ with optional padding/resampling | `measurements`, indices | Retrieved measurements | |
| L12 | Loss Computation | $\ell_b = \mathcal{L}(\hat{y}, y_b, \mathbf{O}_{\text{ROI}})$ | Predictions, measurements, object patches | Scalar loss | |
| L13 | Gradient Computation | $\nabla_{\theta}\ell_b = \texttt{autograd.backward}(\ell_b)$ | Loss | Gradients accumulated | [ADJOINT via AD] |
| L14 | Parameter Update | $\theta \leftarrow \theta - \eta \cdot \texttt{optimizer\_step}(\nabla_\theta\ell_b)$ | Gradients, learning rates | Updated parameters | Optimizer-specific |
| L15 | Constraint Application | $\theta \leftarrow \mathcal{C}(\theta, n)$ | Parameters, iteration | Constrained parameters | Iteration-dependent |

```
  END_LOOP (batch)
```

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| L16 | Loss Recording | `loss_iters.append(epoch_loss)` | Accumulated loss | Updated history | |
| L17 | Time Recording | `iter_times.append(iter_time)` | Timing | Updated history | |

```
END_LOOP (iteration)
```

#### Propagator Computation (Step L8) Details

**Case 1: Tilt + Variable Thickness**
$$H_{\text{prop}} = e^{i\Delta z \cdot K_z} \cdot e^{i\Delta z(K_y\tan(\theta_y/1000) + K_x\tan(\theta_x/1000))}$$

**Case 2: Tilt + Fixed Thickness (with gradient on tilts)**
$$H_{\text{prop}} = H \cdot e^{i\Delta z(K_y\tan(\theta_y/1000) + K_x\tan(\theta_x/1000))}$$

**Case 3: Tilt + Fixed Thickness (no gradient on tilts)**
$$H_{\text{prop}} = H_{\text{fixed\_tilts\_full}}[\mathcal{I}]$$

**Case 4: No Tilt + Variable Thickness**
$$H_{\text{prop}} = e^{i\Delta z \cdot K_z}$$

**Case 5: No Tilt + Fixed Thickness**
$$H_{\text{prop}} = H$$

where $K_z = \sqrt{k^2 - K_x^2 - K_y^2}$ and $k = 2\pi/\lambda$.

#### Multislice Forward Model (Step L9) [FORWARD]

The function `multislice_forward_model_vec_all` implements:
$$\psi_0 = P$$
$$\psi_{n+1} = \mathcal{F}^{-1}\left[H \cdot \mathcal{F}\left[O_n \cdot \psi_n\right]\right], \quad n = 0, \ldots, N_{\text{slice}}-1$$
$$\hat{y} = \sum_{m} w_m \left|\mathcal{F}[\psi_{N_{\text{slice}}}^{(m)}]\right|^2$$

where $w_m$ = `omode_occu` weights for incoherent mode summation.

### 4.3 FINALIZATION

| Step | Operation | Output |
|------|-----------|--------|
| F1 | Model State Extraction | Final `opt_obja`, `opt_objp`, `opt_probe`, etc. |
| F2 | Complex Object Reconstruction | $O_{\text{final}} = \mathbf{a} \cdot e^{i\boldsymbol{\phi}}$ |
| F3 | Complex Probe Reconstruction | $P_{\text{final}} = \texttt{view\_as\_complex}(\mathbf{P})$ |
| F4 | Results Saving | Saved to `output_path` via `recon_loop` |
| F5 | DDP Cleanup | `dist.destroy_process_group()` if distributed |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit (primary) with potential convergence checks in `recon_loop`
- **expression**: $n \geq N_{\text{iter}}$ where $N_{\text{iter}}$ = `params['recon_params']['n_iter']`
- **parameters**: 
  - `n_iter`: Maximum iterations (from YAML config)
  - Individual parameter `end_iter` values for selective optimization termination
- **check_frequency**: Evaluated at end of each epoch (after all batches)

---

## 6. EVALUATION METRICS

| Metric Name | Formula | Description |
|-------------|---------|-------------|
| Final Loss | $\mathcal{L}_{\text{final}} = \texttt{loss\_iters}[-1]$ | Terminal value of combined loss |
| Object Amplitude Mean | $\bar{a} = \frac{1}{N}\sum_{i} a_i$ | Mean of reconstructed amplitude |
| Object Amplitude Std | $\sigma_a = \sqrt{\frac{1}{N}\sum_i(a_i - \bar{a})^2}$ | Standard deviation of amplitude |
| Object Phase Mean | $\bar{\phi} = \frac{1}{N}\sum_i \phi_i$ | Mean of reconstructed phase |
| Object Phase Std | $\sigma_\phi = \sqrt{\frac{1}{N}\sum_i(\phi_i - \bar{\phi})^2}$ | Standard deviation of phase |
| Solver Time | $t_{\text{solve}}$ (seconds) | Total reconstruction wall time |
| Overdetermined Ratio | $\frac{N_{\text{meas}}}{\sum_p \mathbb{1}[\texttt{requires\_grad}[p]] \cdot \texttt{numel}(p)}$ | Measurement-to-variable ratio |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON MUST include:
- `algorithm_identification`: {`algorithm_name`, `algorithm_family`, `summary`}
- `objective_function`: {`full_expression`, `data_fidelity_term`, `regularization_term`, `constraints`}
- `variables`: {`primal_variables`, `dual_variables`, `auxiliary_variables`, `constants`, `observations`}
- `data_preprocessing`: list of {`step_order`, `step_name`, `formula`, `inputs`, `outputs`}
- `initialization`: list of initialization steps
- `iteration_loop`: {`outer_loop`, `inner_loop`} with step details
- `finalization`: post-processing steps
- `stopping_criterion`: {`type`, `expression`, `parameters`, `check_frequency`}
- `evaluation`: list of {`metric_name`, `formula`, `description`}
- `forward_model`: {`multislice_propagation`, `propagator_formula`, `detector_model`}
- `optimization_method`: {`optimizer_type`, `learning_rates`, `parameter_scheduling`}