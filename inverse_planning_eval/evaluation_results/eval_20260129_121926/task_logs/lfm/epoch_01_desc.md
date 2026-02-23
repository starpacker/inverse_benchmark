# Algorithmic Specification Document: Light Field Microscopy Volume Reconstruction via Richardson-Lucy Deconvolution

---

## 1. Mathematical Formulation

### 1.1 Problem Statement

Given a 2D light field microscopy (LFM) observation, reconstruct the underlying 3D fluorescent volume. This is an inverse problem under a Poisson noise model.

### 1.2 Forward Imaging Model

The observed light field image $\mathbf{b} \in \mathbb{R}^{M \times N}$ is formed by projection of a 3D volume $\mathbf{x} \in \mathbb{R}^{P \times Q \times D}$ through the LFM optical system:

$$
\mathbf{b} = \mathcal{H}(\mathbf{x}) + \boldsymbol{\eta}
$$

where:
- $\mathcal{H}: \mathbb{R}^{P \times Q \times D} \rightarrow \mathbb{R}^{M \times N}$ is the forward projection operator
- $\boldsymbol{\eta}$ represents Poisson-distributed noise
- $D$ is the number of depth planes

### 1.3 Objective Function (Maximum Likelihood under Poisson Model)

The Richardson-Lucy algorithm maximizes the log-likelihood for Poisson statistics:

$$
\mathcal{L}(\mathbf{x}) = \sum_{i,j} \left[ b_{ij} \log\left(\left[\mathcal{H}(\mathbf{x})\right]_{ij}\right) - \left[\mathcal{H}(\mathbf{x})\right]_{ij} \right]
$$

subject to the non-negativity constraint:
$$
\mathbf{x} \geq 0
$$

### 1.4 Regularization Term (Anti-Aliasing Filter)

An optional depth-adaptive Lanczos low-pass filter is applied per depth slice:

$$
\mathbf{x}^{(k+1)}_{:,:,d} \leftarrow \left| \mathcal{F}^{-1}\left( \mathbf{K}_d \odot \mathcal{F}\left( \mathbf{x}^{(k+1)}_{:,:,d} \right) \right) \right|, \quad \forall d \in \{1, \ldots, D\}
$$

where:
- $\mathcal{F}$ denotes the 2D FFT
- $\mathbf{K}_d \in \mathbb{R}^{P \times Q}$ is the depth-dependent Lanczos filter kernel in Fourier domain
- $\odot$ denotes element-wise multiplication

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | Dtype | Initialization | Description |
|----------|--------|-------|-------|----------------|-------------|
| `reconVolume` | $\mathbf{x}^{(k)}$ | $(P, Q, D)$ = `volumeSize` | `float32` | $\mathbf{x}^{(0)} = \mathbf{1}_{P \times Q \times D}$ | Reconstructed 3D volume at iteration $k$ |

### 2.2 Auxiliary/Intermediate Variables

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `onesForward` | $\mathbf{f}_1$ | $(M', N')$ = `imgSize` | `float32` | $\mathcal{H}(\mathbf{1})$ - Forward projection of unity volume |
| `onesBack` | $\mathbf{g}_1$ | $(P, Q, D)$ = `volumeSize` | `float32` | $\mathcal{H}^T(\mathbf{f}_1)$ - Back projection of `onesForward` |
| `LFimageGuess` | $\hat{\mathbf{b}}^{(k)}$ | $(M', N')$ | `float32` | $\mathcal{H}(\mathbf{x}^{(k)})$ - Current forward projection estimate |
| `errorLFimage` | $\mathbf{e}^{(k)}$ | $(M', N')$ | `float32` | Ratio correction in image domain |
| `errorBack` | $\boldsymbol{\delta}^{(k)}$ | $(P, Q, D)$ | `float32` | Multiplicative correction in volume domain |
| `kernelFFT` | $\mathbf{K}$ | $(P, Q, D)$ | `complex64` | Lanczos filter kernels in Fourier domain |
| `H` | $\mathcal{H}$ | Operator | - | Forward projection operator (sparse/functional) |
| `Ht` | $\mathcal{H}^T$ | Operator | - | Backward (adjoint) projection operator |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value/Range | Type | Description |
|-----------|--------|-------------|------|-------------|
| `niter` | $K$ | Default: 100 | `int` | Maximum number of RL iterations |
| `depth_range` | $[z_{\min}, z_{\max}]$ | e.g., $[-300, 300]$ mm | `list[float]` | Axial range for reconstruction |
| `depth_step` | $\Delta z$ | e.g., 150 mm | `float` | Spacing between depth planes |
| `super_res_factor` | $s$ | e.g., 5 | `int` | Lateral super-resolution factor |
| `new_spacing_px` | $p$ | e.g., 15 | `int` | Target lenslet spacing in pixels |
| `lanczos_window_size` | $w$ | Default: 4 | `int` | Lanczos kernel support parameter |
| `filter_flag` | - | `True`/`False` | `bool` | Enable/disable anti-aliasing |
| `crange` | - | From `Camera['range']` | `array` | Valid reconstruction range indices |
| `step` (projection) | - | 8 or 10 | `int` | Sub-sampling step for projections |

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `raw_lenslet_image` | $\mathbf{b}_{\text{raw}}$ | $(H_{\text{raw}}, W_{\text{raw}})$ | `uint16`/`float` | Raw light field capture |
| `white_image` | $\mathbf{w}$ | $(H_{\text{raw}}, W_{\text{raw}})$ | `uint16`/`float` | Calibration (flat-field) image |
| `config_dict` | - | - | `dict` | Camera/optical system parameters |
| `LFimage` | $\mathbf{b}$ | $(M', N')$ = `imgSize` | `float32` | Corrected & normalized LF image |

### 2.5 Derived Geometric Parameters

| Variable | Symbol | Computation | Description |
|----------|--------|-------------|-------------|
| `ndepths` | $D$ | $\lfloor(z_{\max} - z_{\min})/\Delta z\rfloor + 1$ | Number of depth planes |
| `imgSize` | $(M', N')$ | From geometric transformation | Corrected image dimensions |
| `texSize` | $(P, Q)$ | $\lceil \texttt{imgSize} \times \texttt{texScaleFactor} \rceil$, made even | Volume lateral dimensions |
| `volumeSize` | $(P, Q, D)$ | $[\texttt{texSize}, D]$ | Full volume dimensions |

---

## 3. Preprocessing Pipeline

### Step 1: Data Loading
$$
(\mathbf{w}, \texttt{config}, \mathbf{b}_{\text{raw}}) \leftarrow \texttt{load\_lfm\_data}(\texttt{path})
$$
- **Input**: File path to data directory
- **Output**: Calibration image $\mathbf{w}$, configuration dict, raw LF image $\mathbf{b}_{\text{raw}}$

### Step 2: Camera Parameter Initialization
$$
\texttt{Camera} \leftarrow \texttt{LFM\_setCameraParams\_v2}(\texttt{config}, p)
$$
- **Input**: Configuration dictionary, new lenslet spacing $p$
- **Output**: Camera parameter structure

### Step 3: Geometry Computation
$$
(\texttt{LensletCenters}, \texttt{Resolution}, \texttt{GridModel}, \texttt{NewGridModel}) \leftarrow \texttt{LFM\_computeGeometryParameters}(\texttt{Camera}, \mathbf{w}, [z_{\min}, z_{\max}], \Delta z, s)
$$
- **Input**: Camera params, calibration image, depth range, depth step, super-res factor
- **Output**: Lenslet geometry, resolution parameters, grid transformation models

### Step 4: Projection Operator Construction
$$
(\mathcal{H}, \mathcal{H}^T) \leftarrow \texttt{LFM\_computeLFMatrixOperators}(\texttt{Camera}, \texttt{Resolution}, \texttt{LensletCenters})
$$
- **Input**: Camera, resolution, lenslet geometry
- **Output**: Forward operator $\mathcal{H}$, adjoint operator $\mathcal{H}^T$

### Step 5: Geometric Transformation Retrieval
$$
\mathbf{T} \leftarrow \texttt{LFM\_retrieveTransformation}(\texttt{GridModel}, \texttt{NewGridModel})
$$
- **Input**: Original and target grid models
- **Output**: Affine/projective transformation matrix $\mathbf{T}$

### Step 6: Output Size Computation
$$
(M', N') \leftarrow \texttt{get\_transformed\_shape}(\text{shape}(\mathbf{w}), \mathbf{T})
$$
$$
(M', N') \leftarrow (M', N') + (1 - (M', N') \mod 2) \quad \text{(ensure even)}
$$

### Step 7: Texture Size Computation
$$
(P, Q) \leftarrow \lceil (M', N') \times \texttt{texScaleFactor} \rceil
$$
$$
(P, Q) \leftarrow (P, Q) + (1 - (P, Q) \mod 2) \quad \text{(ensure even)}
$$

### Step 8: Image Geometric Correction
$$
\tilde{\mathbf{b}} \leftarrow \texttt{transform\_img}(\mathbf{b}_{\text{raw}}, \mathbf{T}, \texttt{offset})
$$
- **Input**: Raw image, transformation, offset
- **Output**: Geometrically corrected image $\tilde{\mathbf{b}}$

### Step 9: Intensity Normalization
$$
\mathbf{b} \leftarrow \frac{\tilde{\mathbf{b}} - \min(\tilde{\mathbf{b}})}{\max(\tilde{\mathbf{b}}) - \min(\tilde{\mathbf{b}})}
$$
- **Input**: Corrected image $\tilde{\mathbf{b}}$
- **Output**: Normalized LF image $\mathbf{b} \in [0, 1]$

---

## 4. Algorithm Steps: Richardson-Lucy Deconvolution with Anti-Aliasing

### Initialization Phase

**Step I-1**: Initialize volume to unity
$$
\mathbf{x}^{(0)} \leftarrow \mathbf{1}_{P \times Q \times D}
$$

**Step I-2**: Precompute normalization projections
$$
\mathbf{f}_1 \leftarrow \mathcal{H}(\mathbf{1}_{P \times Q \times D})
$$
$$
\mathbf{g}_1 \leftarrow \mathcal{H}^T(\mathbf{f}_1)
$$

**Step I-3**: Compute depth-adaptive filter widths
$$
\boldsymbol{\sigma} \leftarrow \texttt{LFM\_computeDepthAdaptiveWidth}(\texttt{Camera}, \texttt{Resolution})
$$
where $\boldsymbol{\sigma} = [\sigma_1, \ldots, \sigma_D]$ are per-depth blur widths.

**Step I-4**: Build Lanczos filter kernels in Fourier domain
$$
\mathbf{K}_d \leftarrow \texttt{lanczosfft}(\texttt{volumeSize}, \sigma_d, w), \quad \forall d \in \{1, \ldots, D\}
$$

### Iterative Phase (for $k = 0, 1, \ldots, K-1$)

**Step 1**: Forward Projection (Compute Estimate)
$$
\hat{\mathbf{b}}^{(k)} \leftarrow 
\begin{cases}
\mathbf{f}_1 & \text{if } k = 0 \\
\mathcal{H}(\mathbf{x}^{(k)}) & \text{if } k > 0
\end{cases}
$$
- **Input**: Current volume $\mathbf{x}^{(k)}$, operator $\mathcal{H}$
- **Output**: Projected estimate $\hat{\mathbf{b}}^{(k)} \in \mathbb{R}^{M' \times N'}$

**Step 2**: Compute Ratio Correction in Image Domain
$$
\mathbf{e}^{(k)}_{ij} \leftarrow \frac{b_{ij}}{\hat{b}^{(k)}_{ij}} \cdot f_{1,ij}
$$
with numerical safeguard:
$$
\mathbf{e}^{(k)}_{ij} \leftarrow 0 \quad \text{if } \mathbf{e}^{(k)}_{ij} \notin \mathbb{R} \text{ (i.e., NaN or Inf)}
$$
- **Input**: Observed $\mathbf{b}$, estimate $\hat{\mathbf{b}}^{(k)}$, normalization $\mathbf{f}_1$
- **Output**: Error image $\mathbf{e}^{(k)} \in \mathbb{R}^{M' \times N'}$

**Step 3**: Back Projection
$$
\boldsymbol{\delta}^{(k)} \leftarrow \mathcal{H}^T(\mathbf{e}^{(k)})
$$
- **Input**: Error image $\mathbf{e}^{(k)}$, adjoint operator $\mathcal{H}^T$
- **Output**: Volume correction $\boldsymbol{\delta}^{(k)} \in \mathbb{R}^{P \times Q \times D}$

**Step 4**: Normalize Back Projection
$$
\boldsymbol{\delta}^{(k)}_{pqd} \leftarrow \frac{\boldsymbol{\delta}^{(k)}_{pqd}}{g_{1,pqd}}
$$
with numerical safeguard:
$$
\boldsymbol{\delta}^{(k)}_{pqd} \leftarrow 0 \quad \text{if } \boldsymbol{\delta}^{(k)}_{pqd} \notin \mathbb{R}
$$
- **Input**: Raw back-projected correction, normalization volume $\mathbf{g}_1$
- **Output**: Normalized correction factor

**Step 5**: Multiplicative Update
$$
\mathbf{x}^{(k+1)} \leftarrow \mathbf{x}^{(k)} \odot \boldsymbol{\delta}^{(k)}
$$
- **Input**: Current volume, correction factor
- **Output**: Updated volume (pre-filtering)

**Step 6**: Anti-Aliasing Filter (Conditional)

If `filter_flag = True`:
$$
\mathbf{x}^{(k+1)}_{:,:,d} \leftarrow \left| \mathcal{F}^{-1}\left( \text{fftshift}\left( \mathbf{K}_d \odot \mathcal{F}\left( \mathbf{x}^{(k+1)}_{:,:,d} \right) \right) \right) \right|, \quad \forall d \in \{1, \ldots, D\}
$$
- **Input**: Updated volume slice, Lanczos kernel $\mathbf{K}_d$
- **Output**: Filtered volume slice

**Step 7**: Final Numerical Safeguard
$$
\mathbf{x}^{(k+1)}_{pqd} \leftarrow 0 \quad \text{if } \mathbf{x}^{(k+1)}_{pqd} \notin \mathbb{R}
$$

---

## 5. Stopping Criterion

The algorithm terminates when:

$$
k \geq K
$$

where $K = \texttt{niter}$ is the maximum number of iterations (default: 100).

**Note**: No explicit convergence-based early stopping is implemented. The iteration count is fixed.

---

## 6. Evaluation Metrics

### 6.1 Per-Iteration Error Metric (Mean Absolute Error)

$$
\text{MAE}^{(k)} = \frac{1}{M' \cdot N'} \sum_{i=1}^{M'} \sum_{j=1}^{N'} \left| e^{(k)}_{ij} - f_{1,ij} \right|
$$

where:
- $\mathbf{e}^{(k)}$ is the ratio-corrected error image at iteration $k$
- $\mathbf{f}_1$ is the baseline forward projection of unity

**Interpretation**: Measures deviation of the corrected ratio from the ideal case where $\mathbf{b} = \hat{\mathbf{b}}^{(k)}$ (perfect reconstruction would yield $\mathbf{e}^{(k)} = \mathbf{f}_1$).

### 6.2 Normalization for Visualization

For output display, images are normalized to 8-bit range:

$$
\mathbf{I}_{\text{uint8}} = \left\lfloor 255 \cdot \frac{\mathbf{I} - \min(\mathbf{I})}{\max(\mathbf{I}) - \min(\mathbf{I}) + \epsilon} \right\rfloor
$$

where $\epsilon = 10^{-8}$ prevents division by zero.

---

## 7. Complete Algorithm Summary (Pseudocode)

```
ALGORITHM: LFM Volume Reconstruction via Richardson-Lucy

INPUT: b_raw (raw LF image), w (calibration), config, hyperparameters
OUTPUT: x* (reconstructed volume), {MAE^(k)} (error history)

1. PREPROCESSING:
   Camera ← SetCameraParams(config, p)
   (Geometry, H, H^T) ← ComputeGeometry(Camera, w, z_range, Δz, s)
   T ← ComputeTransform(Geometry)
   b ← Normalize(Transform(b_raw, T))
   
2. INITIALIZATION:
   x^(0) ← 1_{P×Q×D}
   f_1 ← H(1_{P×Q×D})
   g_1 ← H^T(f_1)
   K ← BuildLanczosKernels(volumeSize, widths, w)
   
3. ITERATION (k = 0 to K-1):
   IF k = 0: b_hat ← f_1
   ELSE: b_hat ← H(x^(k))
   
   e ← (b / b_hat) ⊙ f_1
   e[~finite] ← 0
   
   MAE^(k) ← mean(|e - f_1|)
   
   δ ← H^T(e) / g_1
   δ[~finite] ← 0
   
   x^(k+1) ← x^(k) ⊙ δ
   
   IF filter_flag:
      FOR d = 1 to D:
         x^(k+1)[:,:,d] ← |IFFT(fftshift(K_d ⊙ FFT(x^(k+1)[:,:,d])))|
   
   x^(k+1)[~finite] ← 0

4. RETURN x^(K), {MAE^(0), ..., MAE^(K-1)}
```

---

## 8. Computational Complexity Notes

| Operation | Complexity per Iteration |
|-----------|-------------------------|
| Forward Projection $\mathcal{H}(\mathbf{x})$ | $O(P \cdot Q \cdot D \cdot L)$ where $L$ = lenslet count |
| Back Projection $\mathcal{H}^T(\mathbf{e})$ | $O(P \cdot Q \cdot D \cdot L)$ |
| FFT-based Filtering (per depth) | $O(P \cdot Q \cdot \log(PQ))$ |
| Total per Iteration | $O(PQD \cdot L + D \cdot PQ\log(PQ))$ |