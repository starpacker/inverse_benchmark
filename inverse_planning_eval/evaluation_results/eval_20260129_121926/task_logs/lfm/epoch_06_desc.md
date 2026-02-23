## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Richardson-Lucy Deconvolution for Light Field Microscopy",
  "algorithm_family": "Expectation-Maximization / Maximum Likelihood Iterative Deconvolution",
  "summary": "Iterative multiplicative volume reconstruction from 2D light field images using forward/backward projection operators with optional depth-adaptive Lanczos anti-aliasing filtering."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\hat{V} = \arg\max_{V \geq 0} \sum_{i} \left[ y_i \log(HV)_i - (HV)_i \right]$$

Equivalently, minimizing the Kullback-Leibler divergence:
$$\hat{V} = \arg\min_{V \geq 0} D_{KL}(y \| HV) = \arg\min_{V \geq 0} \sum_i \left[ y_i \log\frac{y_i}{(HV)_i} - y_i + (HV)_i \right]$$

**data_fidelity_term**:
$$D_{KL}(y \| HV) = \sum_i \left[ y_i \log\frac{y_i}{(HV)_i} + (HV)_i - y_i \right]$$
where $y$ is the observed light field image and $HV$ is the forward projection of volume $V$.

**regularization_term**:
Implicit regularization via depth-adaptive Lanczos low-pass filtering applied at each iteration (when `filter_flag=True`):
$$R(V) = \text{Lanczos anti-aliasing filter applied per depth slice}$$

**constraints**:
$$V \geq 0 \quad \text{(implicit via multiplicative update structure)}$$
$$V[i] = 0 \quad \text{where } V[i] \text{ is not finite (NaN/Inf handling)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $V^{(k)}$ (`reconVolume`) | $(N_x^{tex}, N_y^{tex}, N_d)$ = `volumeSize` | float32 | $\mathbf{1}$ (all ones) | Reconstructed 3D volume at iteration $k$ |

### 2.2 Dual Variables

None (Richardson-Lucy is primal-only).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $V^{(0)}$ (`initVolume`) | $(N_x^{tex}, N_y^{tex}, N_d)$ | float32 | $\mathbf{1}$ | Initial volume (all ones) |
| $\tilde{H}\mathbf{1}$ (`onesForward`) | $(N_x^{img}, N_y^{img})$ = `imgSize` | float32 | Computed once | Forward projection of ones volume |
| $\tilde{H}^T(\tilde{H}\mathbf{1})$ (`onesBack`) | $(N_x^{tex}, N_y^{tex}, N_d)$ | float32 | Computed once | Back projection of onesForward |
| $\hat{y}^{(k)}$ (`LFimageGuess`) | $(N_x^{img}, N_y^{img})$ | float32 | - | Forward projection of current volume estimate |
| $E^{(k)}_{LF}$ (`errorLFimage`) | $(N_x^{img}, N_y^{img})$ | float32 | - | Weighted ratio image |
| $E^{(k)}_{back}$ (`errorBack`) | $(N_x^{tex}, N_y^{tex}, N_d)$ | float32 | - | Normalized back-projected error |
| $\mathcal{K}_j$ (`kernelFFT[:,:,j]`) | $(N_x^{tex}, N_y^{tex}, N_d)$ | complex64 | Precomputed | FFT of depth-adaptive Lanczos kernel |
| $w_j$ (`widths`) | $(N_d,)$ | float32 | Precomputed | Depth-adaptive filter widths |
| $\epsilon^{(k)}$ (`err_metric`) | scalar | float32 | - | Mean absolute error metric per iteration |
| `error_metrics` | $(N_{iter},)$ | list[float] | Empty list | Convergence tracking array |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{iter}$ (`niter`) | 100 (default), 1 (in main) | Maximum number of iterations |
| `filter_flag` | True | Whether to apply anti-aliasing filter |
| $a$ (`lanczos_window_size`) | 4 | Lanczos kernel window parameter |
| `step` (forward) | 10 (loop), 8 (precompute) | Subsampling step for projection |
| `step` (backward) | 10 (loop), 8 (precompute) | Subsampling step for back-projection |
| `depth_range` | $[-300, 300]$ mm | Depth range for reconstruction |
| `depth_step` | $150$ mm | Depth sampling interval |
| `super_res_factor` | 5 | Super-resolution factor |
| `new_spacing_px` | 15 | Lenslet spacing in pixels |
| `use_gpu` | True | GPU acceleration flag |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $y_{raw}$ (`raw_lenslet_image`) | $(H_{raw}, W_{raw})$ | uint16/float | File: `example_fly.tif` | Raw light field lenslet image |
| $y_{white}$ (`white_image`) | $(H_{raw}, W_{raw})$ | uint16/float | File: `calib.tif` | Calibration/white image |
| `config_dict` | dict | - | File: `config.yaml` | Camera and system configuration |
| $y$ (`LFimage`) | $(N_x^{img}, N_y^{img})$ | float32 | Corrected | Corrected and normalized light field image |

---

## 3. PREPROCESSING PIPELINE

### Step P1: Load Calibration Image
- **step_order**: P1
- **step_name**: Load White Image
- **formula**: $y_{white} = \text{tifffile.imread}(\texttt{calib.tif})$
- **inputs**: File path (string)
- **outputs**: $y_{white} \in \mathbb{R}^{H_{raw} \times W_{raw}}$
- **assumptions**: File exists and is valid TIFF

### Step P2: Load Configuration
- **step_order**: P2
- **step_name**: Load YAML Configuration
- **formula**: $\texttt{config\_dict} = \text{yaml.safe\_load}(\texttt{config.yaml})$
- **inputs**: File path (string)
- **outputs**: `config_dict` (dictionary)
- **assumptions**: Valid YAML structure with required camera parameters

### Step P3: Load Raw Image
- **step_order**: P3
- **step_name**: Load Raw Lenslet Image
- **formula**: $y_{raw} = \text{tifffile.imread}(\texttt{example\_fly.tif})$
- **inputs**: File path (string)
- **outputs**: $y_{raw} \in \mathbb{R}^{H_{raw} \times W_{raw}}$
- **assumptions**: File exists and is valid TIFF

### Step P4: Set Camera Parameters
- **step_order**: P4
- **step_name**: Initialize Camera Structure
- **formula**: $\texttt{Camera} = \text{LFM\_setCameraParams\_v2}(\texttt{config\_dict}, \texttt{new\_spacing\_px}=15)$
- **inputs**: `config_dict`, `new_spacing_px`=15
- **outputs**: `Camera` (dict containing optical parameters and `range`)
- **assumptions**: Config contains required fields

### Step P5: Compute Geometry Parameters
- **step_order**: P5
- **step_name**: Compute LFM Geometry
- **formula**: 
$$(\texttt{LensletCenters}, \texttt{Resolution}, \texttt{LensletGridModel}, \texttt{NewLensletGridModel}) = \text{LFM\_computeGeometryParameters}(\texttt{Camera}, y_{white}, [-300, 300], 150, 5, \text{False})$$
- **inputs**: `Camera`, $y_{white}$, `depth_range`=$[-300, 300]$, `depth_step`=150, `super_res_factor`=5
- **outputs**: 
  - `LensletCenters`: dict with `offset` field
  - `Resolution`: dict with `depths`, `texScaleFactor` fields
  - `LensletGridModel`: $\mathbb{R}^{m \times n}$
  - `NewLensletGridModel`: $\mathbb{R}^{m \times n}$
- **assumptions**: White image provides valid lenslet pattern

### Step P6: Compute Projection Operators
- **step_order**: P6
- **step_name**: Build Forward/Backward Operators
- **formula**: $(H, H^T) = \text{LFM\_computeLFMatrixOperators}(\texttt{Camera}, \texttt{Resolution}, \texttt{LensletCenters})$
- **inputs**: `Camera`, `Resolution`, `LensletCenters`
- **outputs**: 
  - $H$: Forward projection operator (function/matrix)
  - $H^T$: Backward (adjoint) projection operator (function/matrix)
- **assumptions**: Geometry fully specifies light field projection model

### Step P7: Compute Geometric Transformation
- **step_order**: P7
- **step_name**: Retrieve Lenslet Grid Transformation
- **formula**: $\texttt{FixAll} = \text{LFM\_retrieveTransformation}(\texttt{LensletGridModel}, \texttt{NewLensletGridModel})$
- **inputs**: `LensletGridModel`, `NewLensletGridModel`
- **outputs**: `FixAll` (transformation parameters)
- **assumptions**: Grid models are compatible

### Step P8: Format Transformation
- **step_order**: P8
- **step_name**: Format Transformation Matrix
- **formula**: $T = \text{format\_transform}(\texttt{FixAll})$
- **inputs**: `FixAll`
- **outputs**: $T$ (formatted transformation structure)
- **assumptions**: None

### Step P9: Compute Transformed Image Size
- **step_order**: P9
- **step_name**: Calculate Output Image Dimensions
- **formula**: 
$$\texttt{imgSize}_{raw} = \text{get\_transformed\_shape}(y_{white}.\text{shape}, T)$$
$$\texttt{imgSize} = \texttt{imgSize}_{raw} + (1 - \texttt{imgSize}_{raw} \mod 2)$$
- **inputs**: $y_{white}$.shape, $T$
- **outputs**: $\texttt{imgSize} = (N_x^{img}, N_y^{img})$ (ensured even)
- **assumptions**: Transformation is valid

### Step P10: Compute Texture Size
- **step_order**: P10
- **step_name**: Calculate Texture Space Dimensions
- **formula**: 
$$\texttt{texSize}_{raw} = \lceil \texttt{imgSize} \times \texttt{Resolution}[\text{'texScaleFactor'}] \rceil$$
$$\texttt{texSize} = \texttt{texSize}_{raw} + (1 - \texttt{texSize}_{raw} \mod 2)$$
- **inputs**: `imgSize`, `Resolution['texScaleFactor']`
- **outputs**: $\texttt{texSize} = (N_x^{tex}, N_y^{tex})$ (ensured even, int32)
- **assumptions**: Scale factor is positive

### Step P11: Compute Volume Size
- **step_order**: P11
- **step_name**: Calculate 3D Volume Dimensions
- **formula**: 
$$N_d = \text{len}(\texttt{Resolution}[\text{'depths'}])$$
$$\texttt{volumeSize} = (N_x^{tex}, N_y^{tex}, N_d)$$
- **inputs**: `texSize`, `Resolution['depths']`
- **outputs**: $\texttt{volumeSize} = (N_x^{tex}, N_y^{tex}, N_d)$ (int32)
- **assumptions**: Depths array is non-empty

### Step P12: Transfer Raw Image to GPU
- **step_order**: P12
- **step_name**: Upload to GPU Memory
- **formula**: $\tilde{y}_{raw} = \text{cupy.array}(y_{raw}, \text{dtype='float32'})$
- **inputs**: $y_{raw} \in \mathbb{R}^{H_{raw} \times W_{raw}}$
- **outputs**: $\tilde{y}_{raw}$ (GPU array, float32)
- **assumptions**: GPU available if `use_gpu=True`

### Step P13: Apply Geometric Transformation
- **step_order**: P13
- **step_name**: Transform Lenslet Image
- **formula**: $y_{trans} = \text{transform\_img}(\tilde{y}_{raw}, T, \texttt{LensletCenters}[\text{'offset'}])$
- **inputs**: $\tilde{y}_{raw}$, $T$, offset
- **outputs**: $y_{trans} \in \mathbb{R}^{N_x^{img} \times N_y^{img}}$ (GPU array)
- **assumptions**: Transformation is invertible

### Step P14: Normalize Corrected Image
- **step_order**: P14
- **step_name**: Min-Max Normalization
- **formula**: 
$$y = \frac{y_{trans} - \min(y_{trans})}{\max(y_{trans}) - \min(y_{trans})}$$
- **inputs**: $y_{trans}$
- **outputs**: $y \in [0, 1]^{N_x^{img} \times N_y^{img}}$ (`LFimage`, GPU array, float32)
- **assumptions**: $\max(y_{trans}) \neq \min(y_{trans})$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | `xp` | `cupy` (if `use_gpu=True`) else `numpy` | - | module | Configuration |
| 2 | `mempool` | `cupy.get_default_memory_pool()` | - | object | CuPy runtime |
| 3 | `crange` | `Camera['range']` | varies | - | Preprocessing Step P4 |
| 4 | $V^{(0)}$ (`initVolume`) | $\mathbf{1}_{N_x^{tex} \times N_y^{tex} \times N_d}$ | $(N_x^{tex}, N_y^{tex}, N_d)$ | float32 | `xp.ones(volumeSize, dtype='float32')` |
| 5 | $\tilde{H}\mathbf{1}$ (`onesForward`) | $H(V^{(0)})$ | $(N_x^{img}, N_y^{img})$ | float32 | Forward projection |
| 6 | $\tilde{H}^T(\tilde{H}\mathbf{1})$ (`onesBack`) | $H^T(\tilde{H}\mathbf{1})$ | $(N_x^{tex}, N_y^{tex}, N_d)$ | float32 | Back projection |
| 7 | $y$ (`LFimage`) | `xp.asarray(LFimage)` | $(N_x^{img}, N_y^{img})$ | float32 | Ensure GPU array |
| 8 | $V^{(0)}$ (`reconVolume`) | `xp.copy(initVolume)` | $(N_x^{tex}, N_y^{tex}, N_d)$ | float32 | Copy of initVolume |
| 9 | `error_metrics` | `[]` | dynamic | list | Empty list |
| 10 | $w$ (`widths`) | $\text{LFM\_computeDepthAdaptiveWidth}(\texttt{Camera}, \texttt{Resolution})$ | $(N_d,)$ | float32 | Depth-dependent widths |
| 11 | $\mathcal{K}$ (`kernelFFT`) | $\text{lanczosfft}(\texttt{volumeSize}, w, a=4)$ | $(N_x^{tex}, N_y^{tex}, N_d)$ | complex64 | Precomputed FFT of Lanczos kernels |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{iter}-1$)

---

**Step 12: Compute Forward Projection** [FORWARD]
- **step_order**: 12
- **step_name**: Forward Project Current Volume Estimate
- **formula**: 
$$\hat{y}^{(k)} = \begin{cases} \tilde{H}\mathbf{1} & \text{if } k = 0 \\ H(V^{(k)}) = \text{LFM\_forwardProject}(H, V^{(k)}, \texttt{LensletCenters}, \texttt{Resolution}, \texttt{imgSize}, \texttt{crange}, \text{step}=10) & \text{if } k > 0 \end{cases}$$
- **inputs**: 
  - $V^{(k)}$: $(N_x^{tex}, N_y^{tex}, N_d)$
  - $H$: forward operator
  - `LensletCenters`, `Resolution`, `imgSize`, `crange`
- **outputs**: $\hat{y}^{(k)}$: $(N_x^{img}, N_y^{img})$
- **computational_note**: First iteration reuses precomputed `onesForward` to avoid redundant computation

---

**Step 13: Free GPU Memory Block 1**
- **step_order**: 13
- **step_name**: GPU Memory Cleanup
- **formula**: $\text{mempool.free\_all\_blocks}()$
- **inputs**: `mempool`
- **outputs**: None (side effect: freed GPU memory)
- **computational_note**: Only executed if `use_gpu=True`

---

**Step 14: Compute Weighted Ratio Image**
- **step_order**: 14
- **step_name**: Element-wise Ratio with Scaling
- **formula**: 
$$E^{(k)}_{LF}[i,j] = \frac{y[i,j]}{\hat{y}^{(k)}[i,j]} \cdot (\tilde{H}\mathbf{1})[i,j]$$
- **inputs**: 
  - $y$: $(N_x^{img}, N_y^{img})$
  - $\hat{y}^{(k)}$: $(N_x^{img}, N_y^{img})$
  - $\tilde{H}\mathbf{1}$: $(N_x^{img}, N_y^{img})$
- **outputs**: $E^{(k)}_{LF}$: $(N_x^{img}, N_y^{img})$
- **computational_note**: Element-wise division followed by element-wise multiplication

---

**Step 15: Handle Non-Finite Values in Error Image**
- **step_order**: 15
- **step_name**: NaN/Inf Replacement (Error Image)
- **formula**: 
$$E^{(k)}_{LF}[i,j] = \begin{cases} 0 & \text{if } E^{(k)}_{LF}[i,j] \notin \mathbb{R} \text{ (NaN or Inf)} \\ E^{(k)}_{LF}[i,j] & \text{otherwise} \end{cases}$$
- **inputs**: $E^{(k)}_{LF}$: $(N_x^{img}, N_y^{img})$
- **outputs**: $E^{(k)}_{LF}$: $(N_x^{img}, N_y^{img})$ (cleaned)
- **computational_note**: Uses `~xp.isfinite()` mask

---

**Step 16: Compute Iteration Error Metric**
- **step_order**: 16
- **step_name**: Mean Absolute Error Calculation
- **formula**: 
$$\epsilon^{(k)} = \frac{1}{N_x^{img} \cdot N_y^{img}} \sum_{i,j} \left| E^{(k)}_{LF}[i,j] - (\tilde{H}\mathbf{1})[i,j] \right|$$
- **inputs**: 
  - $E^{(k)}_{LF}$: $(N_x^{img}, N_y^{img})$
  - $\tilde{H}\mathbf{1}$: $(N_x^{img}, N_y^{img})$
- **outputs**: $\epsilon^{(k)}$: scalar (float)
- **computational_note**: Converted to Python float via `.item()`

---

**Step 17: Append Error to Tracking List**
- **step_order**: 17
- **step_name**: Store Error Metric
- **formula**: $\texttt{error\_metrics}.\text{append}(\epsilon^{(k)})$
- **inputs**: $\epsilon^{(k)}$, `error_metrics`
- **outputs**: `error_metrics` (updated list of length $k+1$)
- **computational_note**: In-place list append

---

**Step 18: Back-Project Error Image** [ADJOINT]
- **step_order**: 18
- **step_name**: Backward Projection of Weighted Ratio
- **formula**: 
$$E^{(k)}_{back,raw} = H^T(E^{(k)}_{LF}) = \text{LFM\_backwardProject}(H^T, E^{(k)}_{LF}, \texttt{LensletCenters}, \texttt{Resolution}, \texttt{texSize}, \texttt{crange}, \text{step}=10)$$
- **inputs**: 
  - $E^{(k)}_{LF}$: $(N_x^{img}, N_y^{img})$
  - $H^T$: backward operator
  - `LensletCenters`, `Resolution`, `texSize`, `crange`
- **outputs**: $E^{(k)}_{back,raw}$: $(N_x^{tex}, N_y^{tex}, N_d)$
- **computational_note**: Adjoint of forward projection operator

---

**Step 19: Free GPU Memory Block 2**
- **step_order**: 19
- **step_name**: GPU Memory Cleanup
- **formula**: $\text{mempool.free\_all\_blocks}()$
- **inputs**: `mempool`
- **outputs**: None
- **computational_note**: Only executed if `use_gpu=True`

---

**Step 20: Normalize Back-Projected Error**
- **step_order**: 20
- **step_name**: Element-wise Division by OnesBack
- **formula**: 
$$E^{(k)}_{back}[i,j,d] = \frac{E^{(k)}_{back,raw}[i,j,d]}{(\tilde{H}^T(\tilde{H}\mathbf{1}))[i,j,d]}$$
- **inputs**: 
  - $E^{(k)}_{back,raw}$: $(N_x^{tex}, N_y^{tex}, N_d)$
  - $\tilde{H}^T(\tilde{H}\mathbf{1})$: $(N_x^{tex}, N_y^{tex}, N_d)$
- **outputs**: $E^{(k)}_{back}$: $(N_x^{tex}, N_y^{tex}, N_d)$
- **computational_note**: Element-wise division

---

**Step 21: Handle Non-Finite Values in Back-Projected Error**
- **step_order**: 21
- **step_name**: NaN/Inf Replacement (Back-Projected Error)
- **formula**: 
$$E^{(k)}_{back}[i,j,d] = \begin{cases} 0 & \text{if } E^{(k)}_{back}[i,j,d] \notin \mathbb{R} \\ E^{(k)}_{back}[i,j,d] & \text{otherwise} \end{cases}$$
- **inputs**: $E^{(k)}_{back}$: $(N_x^{tex}, N_y^{tex}, N_d)$
- **outputs**: $E^{(k)}_{back}$: $(N_x^{tex}, N_y^{tex}, N_d)$ (cleaned)
- **computational_note**: Uses `~xp.isfinite()` mask

---

**Step 22: Multiplicative Volume Update**
- **step_order**: 22
- **step_name**: Richardson-Lucy Update
- **formula**: 
$$V^{(k+1)}[i,j,d] = V^{(k)}[i,j,d] \cdot E^{(k)}_{back}[i,j,d]$$
- **inputs**: 
  - $V^{(k)}$: $(N_x^{tex}, N_y^{tex}, N_d)$
  - $E^{(k)}_{back}$: $(N_x^{tex}, N_y^{tex}, N_d)$
- **outputs**: $V^{(k+1)}$: $(N_x^{tex}, N_y^{tex}, N_d)$
- **computational_note**: Element-wise multiplication (in-place update to `reconVolume`)

---

**BEGIN_CONDITIONAL** (if `filter_flag == True`)

**BEGIN_INNER_LOOP** (for $j = 0, 1, \ldots, N_d - 1$)

**Step 23: Extract Depth Slice**
- **step_order**: 23
- **step_name**: Select j-th Depth Slice
- **formula**: $S_j = V^{(k+1)}[:, :, j]$
- **inputs**: $V^{(k+1)}$: $(N_x^{tex}, N_y^{tex}, N_d)$
- **outputs**: $S_j$: $(N_x^{tex}, N_y^{tex})$
- **computational_note**: Slice extraction (view, not copy)

---

**Step 24: Compute 2D FFT of Depth Slice**
- **step_order**: 24
- **step_name**: Forward FFT
- **formula**: $\hat{S}_j = \mathcal{F}_{2D}\{S_j\} = \text{fft2}(S_j)$
- **inputs**: $S_j$: $(N_x^{tex}, N_y^{tex})$
- **outputs**: $\hat{S}_j$: $(N_x^{tex}, N_y^{tex})$ (complex64)
- **computational_note**: CuPy/NumPy FFT

---

**Step 25: Apply Frequency-Domain Lanczos Filter**
- **step_order**: 25
- **step_name**: Multiply by Precomputed Kernel
- **formula**: $\hat{S}_j^{filtered} = \mathcal{K}[:,:,j] \odot \hat{S}_j$
- **inputs**: 
  - $\hat{S}_j$: $(N_x^{tex}, N_y^{tex})$ (complex)
  - $\mathcal{K}[:,:,j]$: $(N_x^{tex}, N_y^{tex})$ (complex)
- **outputs**: $\hat{S}_j^{filtered}$: $(N_x^{tex}, N_y^{tex})$ (complex)
- **computational_note**: Element-wise complex multiplication

---

**Step 26: Compute Inverse FFT**
- **step_order**: 26
- **step_name**: Inverse FFT
- **formula**: $S_j^{filtered,raw} = \mathcal{F}_{2D}^{-1}\{\hat{S}_j^{filtered}\} = \text{ifft2}(\hat{S}_j^{filtered})$
- **inputs**: $\hat{S}_j^{filtered}$: $(N_x^{tex}, N_y^{tex})$ (complex)
- **outputs**: $S_j^{filtered,raw}$: $(N_x^{tex}, N_y^{tex})$ (complex)
- **computational_note**: CuPy/NumPy inverse FFT

---

**Step 27: Apply FFT Shift**
- **step_order**: 27
- **step_name**: Center Zero Frequency
- **formula**: $S_j^{shifted} = \text{fftshift}(S_j^{filtered,raw})$
- **inputs**: $S_j^{filtered,raw}$: $(N_x^{tex}, N_y^{tex})$
- **outputs**: $S_j^{shifted}$: $(N_x^{tex}, N_y^{tex})$
- **computational_note**: Shifts zero-frequency component to center

---

**Step 28: Take Absolute Value**
- **step_order**: 28
- **step_name**: Magnitude Extraction
- **formula**: $V^{(k+1)}[:,:,j] = |S_j^{shifted}|$
- **inputs**: $S_j^{shifted}$: $(N_x^{tex},