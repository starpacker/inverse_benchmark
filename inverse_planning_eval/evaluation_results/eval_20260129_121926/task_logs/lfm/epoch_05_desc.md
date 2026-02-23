## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Richardson-Lucy Deconvolution for Light Field Microscopy Volume Reconstruction
- **algorithm_family**: Maximum Likelihood / Expectation-Maximization Iterative Deconvolution
- **summary**: Reconstructs a 3D volume from a 2D light field microscopy image using iterative Richardson-Lucy deconvolution with depth-adaptive Lanczos anti-aliasing filtering.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{V} = \arg\max_{V \geq 0} \sum_{p} \left[ y_p \log((\mathbf{H}V)_p) - (\mathbf{H}V)_p \right]$$

where $y$ is the observed light field image and $\mathbf{H}$ is the forward projection operator.

### data_fidelity_term
**Poisson Log-Likelihood**:
$$\mathcal{L}(V) = \sum_{p} \left[ y_p \log((\mathbf{H}V)_p) - (\mathbf{H}V)_p - \log(y_p!) \right]$$

The algorithm implicitly maximizes this via the multiplicative Richardson-Lucy update rule.

### regularization_term
**Depth-Adaptive Lanczos Anti-Aliasing Filter** (applied per iteration when `filter_flag=True`):
$$V^{(k+1)}_{:,:,j} \leftarrow \left| \mathcal{F}^{-1}\left( K_j \cdot \mathcal{F}(V^{(k+1)}_{:,:,j}) \right) \right|$$

where $K_j$ is the precomputed Lanczos kernel in Fourier domain for depth slice $j$.

### constraints
$$V \geq 0 \quad \text{(non-negativity, enforced implicitly via multiplicative updates and absolute value)}$$
$$V[\neg\text{isfinite}(V)] = 0 \quad \text{(numerical stability constraint)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $V$ (`reconVolume`) | $(N_x, N_y, N_z)$ = `volumeSize` | float32 | $V^{(0)} = \mathbf{1}$ (all ones) | Reconstructed 3D volume |

### 2.2 Dual Variables

None (multiplicative update algorithm without explicit dual formulation)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\hat{y}$ (`LFimageGuess`) | $(M_x, M_y)$ = `imgSize` | float32 | Computed per iteration | Forward projection of current volume estimate |
| $e_{\text{LF}}$ (`errorLFimage`) | $(M_x, M_y)$ | float32 | Computed per iteration | Weighted ratio image in sensor space |
| $e_{\text{back}}$ (`errorBack`) | $(N_x, N_y, N_z)$ | float32 | Computed per iteration | Backprojected and normalized error |
| $\mathbf{1}_{\text{fwd}}$ (`onesForward`) | $(M_x, M_y)$ | float32 | $\mathbf{H}(\mathbf{1})$ | Forward projection of unit volume |
| $\mathbf{1}_{\text{back}}$ (`onesBack`) | $(N_x, N_y, N_z)$ | float32 | $\mathbf{H}^T(\mathbf{1}_{\text{fwd}})$ | Backprojection of onesForward (normalization factor) |
| $K$ (`kernelFFT`) | $(N_x, N_y, N_z)$ | complex64 | Precomputed from `lanczosfft` | Depth-adaptive Lanczos filter kernels in Fourier domain |
| $w$ (`widths`) | $(N_z,)$ | float32 | From `LFM_computeDepthAdaptiveWidth` | Depth-dependent filter widths |
| `error_metrics` | list of length `niter` | float64 | Empty list | MAE convergence tracking |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `niter` | 100 (default), 1 (in main) | Maximum number of iterations |
| `filter_flag` | True | Whether to apply Lanczos anti-aliasing filter |
| `lanczos_window_size` | 4 | Lanczos kernel window parameter $a$ |
| `depth_range` | [-300, 300] | Depth range in mm |
| `depth_step` | 150 | Depth step in mm |
| `super_res_factor` | 5 | Super-resolution upsampling factor |
| `new_spacing_px` | 15 | New lenslet spacing in pixels |
| `step` (forward/back projection) | 8 (precompute), 10 (iteration) | Batch processing step size |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $y$ (`LFimage`) | $(M_x, M_y)$ | float32 | Corrected from `raw_lenslet_image` | Normalized, geometrically corrected light field image |
| `white_image` | $(H_{\text{raw}}, W_{\text{raw}})$ | uint16→float32 | `calib.tif` | Calibration/white image |
| `raw_lenslet_image` | $(H_{\text{raw}}, W_{\text{raw}})$ | uint16 | `example_fly.tif` | Raw sensor image |
| `config_dict` | dict | - | `config.yaml` | Camera and optical parameters |

---

## 3. PREPROCESSING PIPELINE

### Step P1: Load Calibration Image
- **step_order**: P1
- **step_name**: Load white image
- **formula**: $W = \text{tifffile.imread}(\texttt{calib.tif})$
- **inputs**: File path (string)
- **outputs**: $W$ with shape $(H_{\text{raw}}, W_{\text{raw}})$, dtype uint16

### Step P2: Load Configuration
- **step_order**: P2
- **step_name**: Parse YAML configuration
- **formula**: $\texttt{config\_dict} = \text{yaml.safe\_load}(\texttt{config.yaml})$
- **inputs**: File path (string)
- **outputs**: Dictionary containing camera/optical parameters

### Step P3: Load Raw Image
- **step_order**: P3
- **step_name**: Load raw lenslet image
- **formula**: $I_{\text{raw}} = \text{tifffile.imread}(\texttt{example\_fly.tif})$
- **inputs**: File path (string)
- **outputs**: $I_{\text{raw}}$ with shape $(H_{\text{raw}}, W_{\text{raw}})$, dtype uint16

### Step P4: Compute Camera Parameters
- **step_order**: P4
- **step_name**: Set camera parameters
- **formula**: $\texttt{Camera} = \text{LFM\_setCameraParams\_v2}(\texttt{config\_dict}, \texttt{new\_spacing\_px})$
- **inputs**: `config_dict`, `new_spacing_px` = 15
- **outputs**: `Camera` dictionary with optical configuration

### Step P5: Compute Geometry Parameters
- **step_order**: P5
- **step_name**: Calculate lenslet geometry
- **formula**: 
$$(\texttt{LensletCenters}, \texttt{Resolution}, \texttt{LensletGridModel}, \texttt{NewLensletGridModel}) = \text{LFM\_computeGeometryParameters}(\texttt{Camera}, W, [\text{-}300, 300], 150, 5, \text{False})$$
- **inputs**: `Camera`, $W$, `depth_range`, `depth_step`, `super_res_factor`
- **outputs**: Geometry dictionaries and grid models

### Step P6: Compute Forward/Backward Operators
- **step_order**: P6
- **step_name**: Build projection operators
- **formula**: $(\mathbf{H}, \mathbf{H}^T) = \text{LFM\_computeLFMatrixOperators}(\texttt{Camera}, \texttt{Resolution}, \texttt{LensletCenters})$
- **inputs**: `Camera`, `Resolution`, `LensletCenters`
- **outputs**: $\mathbf{H}$ [FORWARD], $\mathbf{H}^T$ [ADJOINT] projection operator structures

### Step P7: Compute Geometric Transformation
- **step_order**: P7
- **step_name**: Retrieve lenslet-to-rectified transformation
- **formula**: 
$$\texttt{FixAll} = \text{LFM\_retrieveTransformation}(\texttt{LensletGridModel}, \texttt{NewLensletGridModel})$$
$$\texttt{trans} = \text{format\_transform}(\texttt{FixAll})$$
- **inputs**: `LensletGridModel`, `NewLensletGridModel`
- **outputs**: `trans` (transformation matrix/parameters)

### Step P8: Compute Output Dimensions
- **step_order**: P8
- **step_name**: Calculate image and volume sizes
- **formula**:
$$\texttt{imgSize}_{\text{init}} = \text{get\_transformed\_shape}(W.\text{shape}, \texttt{trans})$$
$$\texttt{imgSize} = \texttt{imgSize}_{\text{init}} + (1 - \texttt{imgSize}_{\text{init}} \mod 2) \quad \text{(ensure odd)}$$
$$\texttt{texSize}_{\text{init}} = \lceil \texttt{imgSize} \times \texttt{Resolution}[\text{'texScaleFactor'}] \rceil$$
$$\texttt{texSize} = \texttt{texSize}_{\text{init}} + (1 - \texttt{texSize}_{\text{init}} \mod 2)$$
$$N_z = |\texttt{Resolution}[\text{'depths'}]|$$
$$\texttt{volumeSize} = (\texttt{texSize}[0], \texttt{texSize}[1], N_z)$$
- **inputs**: $W.\text{shape}$, `trans`, `Resolution`
- **outputs**: `imgSize` $(M_x, M_y)$, `texSize` $(N_x, N_y)$, `volumeSize` $(N_x, N_y, N_z)$

### Step P9: Transform Raw Image
- **step_order**: P9
- **step_name**: Apply geometric correction
- **formula**: 
$$I_{\text{float}} = \text{float32}(I_{\text{raw}})$$
$$I_{\text{trans}} = \text{transform\_img}(I_{\text{float}}, \texttt{trans}, \texttt{LensletCenters}[\text{'offset'}])$$
- **inputs**: $I_{\text{raw}}$, `trans`, `LensletCenters['offset']`
- **outputs**: $I_{\text{trans}}$ with shape `imgSize`

### Step P10: Normalize Corrected Image
- **step_order**: P10
- **step_name**: Min-max normalization
- **formula**: 
$$y = \frac{I_{\text{trans}} - \min(I_{\text{trans}})}{\max(I_{\text{trans}}) - \min(I_{\text{trans}})}$$
- **inputs**: $I_{\text{trans}}$
- **outputs**: $y$ (`LFimage`) with values in $[0, 1]$, shape $(M_x, M_y)$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

**Step 1**: Initialize unit volume
- **step_order**: 1
- **step_name**: Create initial volume estimate
- **formula**: $V^{(0)} = \mathbf{1}_{N_x \times N_y \times N_z}$
- **inputs**: `volumeSize` = $(N_x, N_y, N_z)$
- **outputs**: $V^{(0)}$ (`initVolume`), shape $(N_x, N_y, N_z)$, dtype float32
- **computational_note**: All elements set to 1.0

**Step 2**: Precompute forward projection of ones
- **step_order**: 2
- **step_name**: Compute normalization image (forward)
- **formula**: $\mathbf{1}_{\text{fwd}} = \mathbf{H}(V^{(0)}) = \mathbf{H}(\mathbf{1})$ [FORWARD]
- **inputs**: $\mathbf{H}$, $V^{(0)}$, `LensletCenters`, `Resolution`, `imgSize`, `Camera['range']`, step=8
- **outputs**: $\mathbf{1}_{\text{fwd}}$ (`onesForward`), shape $(M_x, M_y)$
- **computational_note**: Represents the system response to uniform illumination

**Step 3**: Precompute backward projection of onesForward
- **step_order**: 3
- **step_name**: Compute normalization volume (backward)
- **formula**: $\mathbf{1}_{\text{back}} = \mathbf{H}^T(\mathbf{1}_{\text{fwd}})$ [ADJOINT]
- **inputs**: $\mathbf{H}^T$, $\mathbf{1}_{\text{fwd}}$, `LensletCenters`, `Resolution`, `texSize`, `Camera['range']`, step=8
- **outputs**: $\mathbf{1}_{\text{back}}$ (`onesBack`), shape $(N_x, N_y, N_z)$
- **computational_note**: Used to normalize backprojected errors

**Step 4**: Compute depth-adaptive filter widths
- **step_order**: 4
- **step_name**: Calculate Lanczos kernel widths per depth
- **formula**: $w = \text{LFM\_computeDepthAdaptiveWidth}(\texttt{Camera}, \texttt{Resolution})$
- **inputs**: `Camera`, `Resolution`
- **outputs**: $w$ (`widths`), shape $(N_z,)$
- **computational_note**: Width varies with depth to match optical PSF

**Step 5**: Precompute Lanczos filter kernels in Fourier domain
- **step_order**: 5
- **step_name**: Build anti-aliasing filter bank
- **formula**: $K = \text{lanczosfft}(\texttt{volumeSize}, w, a)$ where $a = 4$
- **inputs**: `volumeSize`, $w$, `lanczos_window_size` = 4
- **outputs**: $K$ (`kernelFFT`), shape $(N_x, N_y, N_z)$, dtype complex
- **computational_note**: FFT of Lanczos kernels for efficient convolution

**Step 6**: Copy initial volume to working variable
- **step_order**: 6
- **step_name**: Initialize reconstruction buffer
- **formula**: $V \leftarrow V^{(0)}$
- **inputs**: $V^{(0)}$
- **outputs**: $V$ (`reconVolume`), shape $(N_x, N_y, N_z)$
- **computational_note**: Deep copy to preserve initial state

**Step 7**: Initialize error tracking
- **step_order**: 7
- **step_name**: Create error metrics list
- **formula**: $\texttt{error\_metrics} = []$
- **inputs**: None
- **outputs**: Empty list for convergence monitoring

---

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{iter}}-1$ where $N_{\text{iter}} = \texttt{niter}$)

**Step 8**: Compute forward projection (conditional)
- **step_order**: 8
- **step_name**: Forward project current estimate
- **formula**: 
$$\hat{y}^{(k)} = \begin{cases} \mathbf{1}_{\text{fwd}} & \text{if } k = 0 \\ \mathbf{H}(V^{(k)}) & \text{if } k > 0 \end{cases}$$
[FORWARD operator for $k > 0$]
- **inputs**: $V^{(k)}$ (or skip if $k=0$), $\mathbf{H}$, `LensletCenters`, `Resolution`, `imgSize`, `Camera['range']`, step=10
- **outputs**: $\hat{y}^{(k)}$ (`LFimageGuess`), shape $(M_x, M_y)$
- **computational_note**: Skip projection at first iteration for efficiency

**Step 9**: Compute weighted error ratio
- **step_order**: 9
- **step_name**: Calculate sensor-space error
- **formula**: 
$$e_{\text{LF}}^{(k)} = \frac{y}{\hat{y}^{(k)}} \odot \mathbf{1}_{\text{fwd}}$$
where $\odot$ denotes element-wise multiplication
- **inputs**: $y$, $\hat{y}^{(k)}$, $\mathbf{1}_{\text{fwd}}$
- **outputs**: $e_{\text{LF}}^{(k)}$ (`errorLFimage`), shape $(M_x, M_y)$
- **computational_note**: Element-wise division and multiplication

**Step 10**: Handle non-finite values in error ratio
- **step_order**: 10
- **step_name**: Numerical stability correction (sensor space)
- **formula**: 
$$e_{\text{LF}}^{(k)}[p] = 0 \quad \forall p : \neg\text{isfinite}(e_{\text{LF}}^{(k)}[p])$$
- **inputs**: $e_{\text{LF}}^{(k)}$
- **outputs**: $e_{\text{LF}}^{(k)}$ (modified in-place)
- **computational_note**: Replaces NaN, Inf, -Inf with zero

**Step 11**: Compute convergence metric
- **step_order**: 11
- **step_name**: Calculate mean absolute error
- **formula**: 
$$\epsilon^{(k)} = \frac{1}{M_x M_y} \sum_{p} \left| e_{\text{LF}}^{(k)}[p] - \mathbf{1}_{\text{fwd}}[p] \right|$$
- **inputs**: $e_{\text{LF}}^{(k)}$, $\mathbf{1}_{\text{fwd}}$
- **outputs**: $\epsilon^{(k)}$ (scalar), appended to `error_metrics`
- **computational_note**: Measures deviation from expected ratio of 1

**Step 12**: Backproject error to volume space
- **step_order**: 12
- **step_name**: Backward projection of weighted error
- **formula**: $e_{\text{back,raw}}^{(k)} = \mathbf{H}^T(e_{\text{LF}}^{(k)})$ [ADJOINT]
- **inputs**: $\mathbf{H}^T$, $e_{\text{LF}}^{(k)}$, `LensletCenters`, `Resolution`, `texSize`, `Camera['range']`, step=10
- **outputs**: $e_{\text{back,raw}}^{(k)}$ (`errorBack` before normalization), shape $(N_x, N_y, N_z)$
- **computational_note**: Distributes sensor error to volume voxels

**Step 13**: Normalize backprojected error
- **step_order**: 13
- **step_name**: Apply normalization factor
- **formula**: 
$$e_{\text{back}}^{(k)} = \frac{e_{\text{back,raw}}^{(k)}}{\mathbf{1}_{\text{back}}}$$
- **inputs**: $e_{\text{back,raw}}^{(k)}$, $\mathbf{1}_{\text{back}}$
- **outputs**: $e_{\text{back}}^{(k)}$ (`errorBack`), shape $(N_x, N_y, N_z)$
- **computational_note**: Element-wise division

**Step 14**: Handle non-finite values in backprojected error
- **step_order**: 14
- **step_name**: Numerical stability correction (volume space)
- **formula**: 
$$e_{\text{back}}^{(k)}[v] = 0 \quad \forall v : \neg\text{isfinite}(e_{\text{back}}^{(k)}[v])$$
- **inputs**: $e_{\text{back}}^{(k)}$
- **outputs**: $e_{\text{back}}^{(k)}$ (modified in-place)
- **computational_note**: Replaces NaN, Inf, -Inf with zero

**Step 15**: Multiplicative volume update
- **step_order**: 15
- **step_name**: Richardson-Lucy update
- **formula**: 
$$V^{(k+1/2)} = V^{(k)} \odot e_{\text{back}}^{(k)}$$
- **inputs**: $V^{(k)}$, $e_{\text{back}}^{(k)}$
- **outputs**: $V^{(k+1/2)}$ (intermediate), shape $(N_x, N_y, N_z)$
- **computational_note**: Element-wise multiplication preserves non-negativity

**Step 16**: Apply anti-aliasing filter (conditional)
- **step_order**: 16
- **step_name**: Depth-wise Lanczos filtering
- **formula**: If `filter_flag` = True:

**BEGIN_LOOP** (for $j = 0, 1, \ldots, N_z - 1$)
$$V^{(k+1/2)}_{:,:,j} \leftarrow \left| \mathcal{F}^{-1}\left( \text{fftshift}\left( K_{:,:,j} \odot \mathcal{F}(V^{(k+1/2)}_{:,:,j}) \right) \right) \right|$$

where:
- $\mathcal{F}$ = `fft2` (2D FFT)
- $\mathcal{F}^{-1}$ = `ifft2` (2D inverse FFT)
- $|\cdot|$ = absolute value (ensures non-negativity)

**END_LOOP**

- **inputs**: $V^{(k+1/2)}$, $K$ (`kernelFFT`)
- **outputs**: $V^{(k+1/2)}$ (filtered in-place)
- **computational_note**: Suppresses aliasing artifacts independently per depth slice

**Step 17**: Final non-finite correction
- **step_order**: 17
- **step_name**: Numerical stability correction (final)
- **formula**: 
$$V^{(k+1)}[v] = 0 \quad \forall v : \neg\text{isfinite}(V^{(k+1/2)}[v])$$
$$V^{(k+1)} = V^{(k+1/2)}$$
- **inputs**: $V^{(k+1/2)}$
- **outputs**: $V^{(k+1)}$ (`reconVolume`)
- **computational_note**: Clean up any numerical artifacts from filtering

**END_LOOP**

---

### 4.3 FINALIZATION

**Step 18**: Transfer to host memory
- **step_order**: 18
- **step_name**: GPU to CPU transfer
- **formula**: 
$$V_{\text{out}} = \begin{cases} \text{cupy.asnumpy}(V^{(N_{\text{iter}})}) & \text{if use\_gpu} \\ V^{(N_{\text{iter}})} & \text{otherwise} \end{cases}$$
- **inputs**: $V^{(N_{\text{iter}})}$ (GPU array or NumPy array)
- **outputs**: $V_{\text{out}}$ (`recon_volume_np`), NumPy array, shape $(N_x, N_y, N_z)$
- **computational_note**: Converts CuPy array to NumPy for saving/visualization

**Step 19**: Extract center slice for visualization
- **step_order**: 19
- **step_name**: Select middle depth plane
- **formula**: 
$$S_{\text{center}} = V_{\text{out}}[:, :, \lfloor N_z / 2 \rfloor]$$
- **inputs**: $V_{\text{out}}$
- **outputs**: $S_{\text{center}}$ (`center_slice`), shape $(N_x, N_y)$
- **computational_note**: Used for qualitative comparison

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k < N_{\text{iter}}$
- **parameters**: 
  - $N_{\text{iter}}$ = `niter` = 100 (default) or 1 (in main script)
- **check_frequency**: Evaluated at the start of each iteration in the for-loop

---

## 6. EVALUATION METRICS

### Metric 1: Light Field Mean Absolute Error (MAE)
- **name**: LF MAE
- **formula**: 
$$\text{MAE}^{(k)} = \frac{1}{M_x M_y} \sum_{i=1}^{M_x} \sum_{j=1}^{M_y} \left| e_{\text{LF}}^{(k)}[i,j] - \mathbf{1}_{\text{fwd}}[i,j] \right|$$

where $e_{\text{LF}}^{(k)} = \frac{y}{\mathbf{H}(V^{(k)})} \odot \mathbf{1}_{\text{fwd}}$

- **reference_data**: $\mathbf{1}_{\text{fwd}}$ (forward projection of unit volume)
- **predicted_data**: $e_{\text{LF}}^{(k)}$ (weighted ratio at iteration $k$)
- **description**: Measures convergence by quantifying how close the ratio $y / \hat{y}$ is to unity. When converged, $e_{\text{LF}} \approx \mathbf{1}_{\text{fwd}}$, yielding MAE ≈ 0.