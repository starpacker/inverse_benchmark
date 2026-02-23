## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Light Field Microscopy Richardson-Lucy Deconvolution (LFM-RL)
- **algorithm_family**: Expectation-Maximization / Maximum Likelihood Iterative Deconvolution
- **summary**: Reconstructs a 3D volume from a 2D light field microscopy image using Richardson-Lucy iterative deconvolution with depth-adaptive anti-aliasing filtering and sensitivity correction.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{V} = \arg\max_{V \geq 0} \mathcal{L}(V) = \arg\max_{V \geq 0} \sum_{i} \left[ y_i \log((\mathbf{H}V)_i) - (\mathbf{H}V)_i \right]$$

where $y$ is the observed light field image, $V$ is the 3D volume, and $\mathbf{H}$ is the forward projection operator.

### data_fidelity_term
**Poisson log-likelihood (negative):**
$$\mathcal{D}(V) = -\sum_{i} \left[ y_i \log((\mathbf{H}V)_i) - (\mathbf{H}V)_i \right]$$

This is equivalent to minimizing the Kullback-Leibler divergence $D_{KL}(y \| \mathbf{H}V)$.

### regularization_term
**Implicit regularization via anti-aliasing filter (when `filter_flag=True`):**
$$\mathcal{R}(V) = \text{Depth-adaptive Lanczos low-pass filtering applied per-slice}$$

No explicit penalty term in the objective; regularization is applied as a constraint projection step.

### constraints
$$V \geq 0 \quad \text{(enforced implicitly via multiplicative updates)}$$
$$\mathbb{I}_{\text{finite}}(V) = \begin{cases} 0 & \text{if all elements finite} \\ +\infty & \text{otherwise} \end{cases}$$

Non-finite values are projected to 0 after each update.

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $V^{(k)}$ (`reconVolume`) | $(N_x, N_y, N_z)$ = `volumeSize` | float32 | $\mathbf{1}$ (all ones) | Reconstructed 3D volume at iteration $k$ |

### 2.2 Dual Variables

None. The algorithm is primal-only (no dual formulation).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $V^{(0)}$ (`initVolume`) | $(N_x, N_y, N_z)$ | float32 | $\mathbf{1}$ | Initial unit volume for normalization computation |
| $\mathbf{s}_f$ (`onesForward`) | $(M_x, M_y)$ = `imgSize` | float32 | $\mathbf{H} V^{(0)}$ | Forward projection sensitivity map |
| $\mathbf{s}_b$ (`onesBack`) | $(N_x, N_y, N_z)$ | float32 | $\mathbf{H}^T \mathbf{s}_f$ | Back projection sensitivity map |
| $\hat{y}^{(k)}$ (`LFimageGuess`) | $(M_x, M_y)$ | float32 | $\mathbf{s}_f$ for $k=0$ | Predicted light field image at iteration $k$ |
| $\mathbf{e}^{(k)}_{\text{LF}}$ (`errorLFimage`) | $(M_x, M_y)$ | float32 | Computed | Weighted ratio error in image space |
| $\mathbf{e}^{(k)}_{\text{vol}}$ (`errorBack`) | $(N_x, N_y, N_z)$ | float32 | Computed | Normalized back-projected error |
| $\mathcal{K}$ (`kernelFFT`) | $(N_x, N_y, N_z)$ | complex64 | Precomputed | Depth-adaptive Lanczos filter kernels in Fourier domain |
| $\mathbf{w}$ (`widths`) | $(N_z,)$ | float32 | Computed from camera/resolution | Depth-adaptive filter widths |
| $\epsilon^{(k)}$ (`err_metric`) | scalar | float32 | Computed | Mean absolute error metric per iteration |
| `error_metrics` | $(N_{\text{iter}},)$ | list of float | Empty list | Accumulated error metrics |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `depth_range` | $[-300, 300]$ mm | Min/max reconstruction depth |
| `depth_step` | $150$ mm | Depth sampling interval |
| `new_spacing_px` | $15$ pixels | Lenslet spacing after resampling |
| `super_res_factor` | $5$ | Super-resolution factor |
| `lanczos_window_size` | $4$ | Lanczos filter window parameter $a$ |
| `filter_flag` | `True` | Enable anti-aliasing filter |
| `niter` | $100$ (default) | Maximum iteration count |
| `use_gpu` | `True` | GPU acceleration flag |
| `step` (forward) | $10$ | Block processing step for forward projection |
| `step` (backward) | $10$ | Block processing step for backward projection |
| `step` (precompute) | $8$ | Block processing step for sensitivity computation |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $y$ (`LFimage`) | $(M_x, M_y)$ | float32 | Corrected input | Normalized, geometrically corrected light field image |
| $y_{\text{raw}}$ (`raw_lenslet_image`) | $(H_{\text{raw}}, W_{\text{raw}})$ | varies | TIFF file | Raw lenslet image from microscope |
| $y_{\text{calib}}$ (`white_image`) | $(H_{\text{raw}}, W_{\text{raw}})$ | varies | TIFF file | Calibration (white) image |
| `config_dict` | dict | - | YAML file | Camera and system configuration parameters |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Parameter Derivation

| Field | Value |
|-------|-------|
| step_order | 0 |
| step_name | Geometry and Operator Computation |
| formula | $\mathbf{H}, \mathbf{H}^T = \texttt{LFM\_computeLFMatrixOperators}(\text{Camera}, \text{Resolution}, \text{LensletCenters})$ |
| inputs | `config_dict` (dict), `white_image` $(H, W)$, `depth_range` $[d_{\min}, d_{\max}]$, `depth_step` $\Delta d$, `super_res_factor` $s$, `new_spacing_px` $p$ |
| outputs | `Camera` (dict), `LensletCenters` (dict), `Resolution` (dict), `H` [FORWARD], `Ht` [ADJOINT] |

$$N_z = \left\lfloor \frac{d_{\max} - d_{\min}}{\Delta d} \right\rfloor + 1$$

### Step 1: Geometric Transformation Retrieval

| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Compute Transformation Matrix |
| formula | $\mathbf{T} = \texttt{LFM\_retrieveTransformation}(\text{LensletGridModel}, \text{NewLensletGridModel})$ |
| inputs | `LensletGridModel` $(N_{\text{lenslets}}, 2)$, `NewLensletGridModel` $(N_{\text{lenslets}}, 2)$ |
| outputs | `FixAll` (transformation struct), `trans` (formatted transform) |

### Step 2: Image Size Computation

| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Compute Output Dimensions |
| formula | $\text{imgSize} = \texttt{get\_transformed\_shape}(\text{white\_image.shape}, \mathbf{T})$ <br> $\text{imgSize} \leftarrow \text{imgSize} + (1 - \text{mod}(\text{imgSize}, 2))$ (ensure odd→even) <br> $\text{texSize} = \lceil \text{imgSize} \times \text{texScaleFactor} \rceil$ <br> $\text{texSize} \leftarrow \text{texSize} + (1 - \text{mod}(\text{texSize}, 2))$ <br> $\text{volumeSize} = [\text{texSize}_x, \text{texSize}_y, N_z]$ |
| inputs | `white_image.shape` $(H, W)$, `trans`, `resolution['texScaleFactor']`, `resolution['depths']` |
| outputs | `imgSize` $(2,)$ int32, `texSize` $(2,)$ int32, `volumeSize` $(3,)$ int32 |

### Step 3: Image Transformation and Normalization

| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Apply Geometric Correction |
| formula | $y_{\text{trans}} = \texttt{transform\_img}(y_{\text{raw}}, \mathbf{T}, \text{offset})$ |
| inputs | `lenslet_image` $(H_{\text{raw}}, W_{\text{raw}})$ float32, `trans`, `lenslet_centers['offset']` |
| outputs | `new` $(M_x, M_y)$ float32 |

### Step 4: Min-Max Normalization

| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Normalize to [0,1] |
| formula | $y = \frac{y_{\text{trans}} - \min(y_{\text{trans}})}{\max(y_{\text{trans}}) - \min(y_{\text{trans}})}$ |
| inputs | `new` $(M_x, M_y)$ |
| outputs | `LFimage` $(M_x, M_y)$ float32, values in $[0, 1]$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Initialize Volume | $V^{(0)} = \mathbf{1} \in \mathbb{R}^{N_x \times N_y \times N_z}$ | `volumeSize` | `initVolume`, `reconVolume` |
| I.2 | Compute Forward Sensitivity | $\mathbf{s}_f = \mathbf{H} V^{(0)}$ | `H` [FORWARD], `initVolume`, `lenslet_centers`, `resolution`, `imgSize`, `crange`, `step=8` | `onesForward` $(M_x, M_y)$ |
| I.3 | Compute Backward Sensitivity | $\mathbf{s}_b = \mathbf{H}^T \mathbf{s}_f$ | `Ht` [ADJOINT], `onesForward`, `lenslet_centers`, `resolution`, `texSize`, `crange`, `step=8` | `onesBack` $(N_x, N_y, N_z)$ |
| I.4 | Compute Filter Widths | $\mathbf{w} = \texttt{LFM\_computeDepthAdaptiveWidth}(\text{camera}, \text{resolution})$ | `camera`, `resolution` | `widths` $(N_z,)$ |
| I.5 | Build Lanczos Filter Kernels | $\mathcal{K} = \texttt{lanczosfft}(\text{volumeSize}, \mathbf{w}, a)$ where $a = 4$ | `volumeSize`, `widths`, `lanczos_window_size` | `kernelFFT` $(N_x, N_y, N_z)$ complex |
| I.6 | Initialize Error List | `error_metrics` $\leftarrow []$ | - | `error_metrics` (empty list) |
| I.7 | Start Timer | $t_1 = \texttt{time()}$ | - | `t1` |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** ($k = 0, 1, \ldots, N_{\text{iter}}-1$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| L.1 | Forward Projection [FORWARD] | $$\hat{y}^{(k)} = \begin{cases} \mathbf{s}_f & k = 0 \\ \mathbf{H} V^{(k)} & k > 0 \end{cases}$$ | `H`, `reconVolume`, `onesForward`, iteration index $k$ | `LFimageGuess` $(M_x, M_y)$ | GPU memory freed after step |
| L.2 | Compute Image-Space Error Ratio | $$\mathbf{e}^{(k)}_{\text{LF}} = \frac{y}{\hat{y}^{(k)}} \odot \mathbf{s}_f$$ | `LFimage`, `LFimageGuess`, `onesForward` | `errorLFimage` $(M_x, M_y)$ | Element-wise division and multiplication |
| L.3 | Sanitize Non-Finite Values | $$(\mathbf{e}^{(k)}_{\text{LF}})_i \leftarrow 0 \quad \forall i : \neg\text{isfinite}((\mathbf{e}^{(k)}_{\text{LF}})_i)$$ | `errorLFimage` | `errorLFimage` (sanitized) | Handles division by zero and overflow |
| L.4 | Compute Error Metric | $$\epsilon^{(k)} = \frac{1}{M_x M_y} \sum_{i,j} \left| (\mathbf{e}^{(k)}_{\text{LF}})_{ij} - (\mathbf{s}_f)_{ij} \right|$$ | `errorLFimage`, `onesForward` | `err_metric` (scalar) | Mean Absolute Error |
| L.5 | Store Error Metric | `error_metrics.append`$(\epsilon^{(k)})$ | `err_metric` | `error_metrics` (updated) | - |
| L.6 | Back Projection [ADJOINT] | $$\tilde{\mathbf{e}}^{(k)}_{\text{vol}} = \mathbf{H}^T \mathbf{e}^{(k)}_{\text{LF}}$$ | `Ht`, `errorLFimage`, `lenslet_centers`, `resolution`, `texSize`, `crange`, `step=10` | `errorBack` $(N_x, N_y, N_z)$ | GPU memory freed after step |
| L.7 | Normalize Back-Projected Error | $$\mathbf{e}^{(k)}_{\text{vol}} = \frac{\tilde{\mathbf{e}}^{(k)}_{\text{vol}}}{\mathbf{s}_b}$$ | `errorBack`, `onesBack` | `errorBack` (normalized) | Element-wise division |
| L.8 | Sanitize Non-Finite Values | $$(\mathbf{e}^{(k)}_{\text{vol}})_i \leftarrow 0 \quad \forall i : \neg\text{isfinite}((\mathbf{e}^{(k)}_{\text{vol}})_i)$$ | `errorBack` | `errorBack` (sanitized) | - |
| L.9 | Multiplicative Volume Update | $$V^{(k+1)} = V^{(k)} \odot \mathbf{e}^{(k)}_{\text{vol}}$$ | `reconVolume`, `errorBack` | `reconVolume` (updated) | Richardson-Lucy multiplicative update |
| L.10 | Anti-Aliasing Filter (Conditional) | **IF** `filter_flag == True`: <br> **BEGIN_INNER_LOOP** ($j = 0, \ldots, N_z - 1$) <br> $$V^{(k+1)}_{:,:,j} = \left| \mathcal{F}^{-1}\left\{ \mathcal{K}_{:,:,j} \odot \mathcal{F}\left\{ V^{(k+1)}_{:,:,j} \right\} \right\} \right|$$ <br> **END_INNER_LOOP** | `reconVolume`, `kernelFFT` | `reconVolume` (filtered) | FFT-based convolution per depth slice; `fftshift` applied to center result |
| L.11 | Final Sanitization | $$(V^{(k+1)})_i \leftarrow 0 \quad \forall i : \neg\text{isfinite}((V^{(k+1)})_i)$$ | `reconVolume` | `reconVolume` (sanitized) | GPU memory freed after step |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Record End Time | $t_2 = \texttt{time()}$; $\Delta t = t_2 - t_1$ | `t1` | Elapsed time printed |
| F.2 | Transfer to CPU | `reconVolume_np` $= \texttt{cupy.asnumpy}(V^{(N_{\text{iter}})})$ if GPU else identity | `reconVolume` | `reconVolume_np` $(N_x, N_y, N_z)$ numpy float32 |
| F.3 | Extract Center Slice | $\text{center\_slice} = V_{:,:,\lfloor N_z/2 \rfloor}$ | `recon_volume_np` | `center_slice` $(N_x, N_y)$ |
| F.4 | Normalize for Visualization | $$\text{img}_{\text{norm}} = \left\lfloor 255 \times \frac{\text{img} - \min(\text{img})}{\max(\text{img}) - \min(\text{img}) + 10^{-8}} \right\rfloor$$ | Image array | uint8 image $[0, 255]$ |

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | `iteration_limit` |
| expression | $k < N_{\text{iter}}$ where $N_{\text{iter}} = 100$ (configurable) |
| parameters | `niter = 100` (default), can be set to any positive integer |
| check_frequency | Evaluated at the start of each iteration |

**Note**: No convergence-based early stopping is implemented. The algorithm runs for exactly `niter` iterations.

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| LF Mean Absolute Error (MAE) | $$\epsilon^{(k)} = \frac{1}{M_x M_y} \sum_{i,j} \left| \left(\frac{y}{\hat{y}^{(k)}} \cdot \mathbf{s}_f\right)_{ij} - (\mathbf{s}_f)_{ij} \right|$$ | Measures deviation of weighted ratio from sensitivity map; approaches 0 as $\hat{y}^{(k)} \to y$ |
| Reconstruction Time | $$\Delta t = t_2 - t_1 \text{ (seconds)}$$ | Wall-clock time for all iterations |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include the following fields:

```
algorithm_identification:
  - algorithm_name
  - algorithm_family
  - summary

objective_function:
  - full_expression
  - data_fidelity_term
  - regularization_term
  - constraints

variables:
  - primal_variables[]
  - dual_variables[]
  - auxiliary_variables[]
  - constants[]
  - observations[]

data_preprocessing:
  - steps[] (each with: step_order, step_name, formula, inputs, outputs)

initialization:
  - steps[] (each with: step_name, formula, inputs, outputs)

iteration_loop:
  - loop_variable
  - loop_range
  - steps[] (each with: step_name, formula, inputs, outputs, computational_note)
  - inner_loops[] (if applicable)

finalization:
  - steps[] (each with: step_name, formula, inputs, outputs)

stopping_criterion:
  - type
  - expression
  - parameters
  - check_frequency

evaluation:
  - metrics[] (each with: name, formula, description)
```