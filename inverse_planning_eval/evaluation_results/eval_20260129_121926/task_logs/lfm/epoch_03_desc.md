## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: Light Field Microscopy Richardson-Lucy Deconvolution
- **algorithm_family**: Iterative Expectation-Maximization / Maximum Likelihood Deconvolution
- **summary**: Reconstructs a 3D volume from a 2D light field microscopy image using multiplicative Richardson-Lucy updates with depth-adaptive Lanczos anti-aliasing filtering.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{V} = \arg\max_{V \geq 0} \; \mathcal{L}(V) = \arg\max_{V \geq 0} \sum_{i} \left[ b_i \log(HV)_i - (HV)_i \right]$$

where $b$ is the observed light field image and $H$ is the forward projection operator. This corresponds to maximum likelihood estimation under Poisson noise.

### data_fidelity_term
$$\mathcal{L}(V) = \sum_{i} \left[ b_i \log(HV)_i - (HV)_i \right]$$

Poisson log-likelihood where:
- $b_i$: observed intensity at pixel $i$ in the corrected light field image
- $(HV)_i$: predicted intensity from forward projection of volume $V$

### regularization_term
Implicit regularization via depth-adaptive Lanczos anti-aliasing filter applied at each iteration:
$$R(V) = \sum_{z=1}^{N_z} \left\| V_{:,:,z} - \mathcal{F}^{-1}\left[ K_z \cdot \mathcal{F}[V_{:,:,z}] \right] \right\|^2$$

where $K_z$ is the depth-dependent Lanczos kernel in Fourier domain.

### constraints
- $V \geq 0$ (non-negativity, enforced implicitly by multiplicative update)
- $V < \infty$ (finite values enforced by zeroing non-finite elements)

---

## 2. VARIABLES

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $b$ (`LFimage`) | observations | $(H_{img}, W_{img})$ | float32 | Loaded, transformed, normalized | Corrected light field image |
| $w$ (`white_image`) | observations | $(H_{raw}, W_{raw})$ | float32 | Loaded from `calib.tif` | Calibration/white image |
| $y_{raw}$ (`raw_lenslet_image`) | observations | $(H_{raw}, W_{raw})$ | float32 | Loaded from `example_fly.tif` | Raw lenslet image |

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $V^{(k)}$ (`reconVolume`) | primal | $(T_x, T_y, N_z)$ = `volumeSize` | float32 | $V^{(0)} = \mathbf{1}$ (all ones) | Reconstructed 3D volume at iteration $k$ |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\hat{b}^{(k)}$ (`LFimageGuess`) | auxiliary | $(H_{img}, W_{img})$ | float32 | $H V^{(k)}$ | Forward projection of current estimate |
| $e_b^{(k)}$ (`errorLFimage`) | auxiliary | $(H_{img}, W_{img})$ | float32 | Computed per iteration | Error ratio in image space |
| $e_V^{(k)}$ (`errorBack`) | auxiliary | $(T_x, T_y, N_z)$ | float32 | Computed per iteration | Backprojected normalized error |
| $\mathbf{1}_H$ (`onesForward`) | auxiliary | $(H_{img}, W_{img})$ | float32 | $H \cdot \mathbf{1}_{vol}$ | Forward projection of ones volume |
| $\mathbf{1}_{H^T}$ (`onesBack`) | auxiliary | $(T_x, T_y, N_z)$ | float32 | $H^T \cdot \mathbf{1}_H$ | Backprojection of onesForward |
| $\tilde{K}$ (`kernelFFT`) | auxiliary | $(T_x, T_y, N_z)$ | complex64 | `lanczosfft(volumeSize, widths, a)` | Depth-adaptive Lanczos filter kernels in Fourier domain |
| $\sigma_z$ (`widths`) | auxiliary | $(N_z,)$ | float32 | `LFM_computeDepthAdaptiveWidth` | Depth-dependent filter widths |

### Operators (Constants)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $H$ | constants | Operator object | - | `LFM_computeLFMatrixOperators` | Forward projection operator (volume → LF image) |
| $H^T$ (`Ht`) | constants | Operator object | - | `LFM_computeLFMatrixOperators` | Backward/adjoint projection operator (LF image → volume) |

### Geometric Parameters (Constants)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| `Camera` | constants | dict | - | `LFM_setCameraParams_v2` | Camera intrinsic/extrinsic parameters |
| `LensletCenters` | constants | dict | - | `LFM_computeGeometryParameters` | Lenslet center coordinates and offset |
| `Resolution` | constants | dict | - | `LFM_computeGeometryParameters` | Resolution parameters including `texScaleFactor`, `depths` |
| `imgSize` | constants | $(2,)$ | int32 | Computed from transform | Transformed image dimensions (ensured even) |
| `texSize` | constants | $(2,)$ | int32 | $\lceil \text{imgSize} \times \text{texScaleFactor} \rceil$ (even) | Texture/volume XY dimensions |
| `volumeSize` | constants | $(3,)$ | int32 | $[\text{texSize}_x, \text{texSize}_y, N_z]$ | Full volume dimensions |
| `crange` | constants | - | - | `Camera['range']` | Projection range parameter |

### Numerical Constants
| Symbol | Category | Value | Description |
|--------|----------|-------|-------------|
| `depth_range` | constants | $[-300, 300]$ mm | Min/max reconstruction depths |
| `depth_step` | constants | $150$ mm | Depth sampling interval |
| `new_spacing_px` | constants | $15$ pixels | Lenslet spacing in pixels |
| `super_res_factor` | constants | $5$ | Super-resolution factor |
| `lanczos_window_size` ($a$) | constants | $4$ | Lanczos kernel half-width |
| `niter` | constants | $100$ (or $1$ for testing) | Maximum iterations |
| `filter_flag` | constants | `True` | Enable anti-aliasing filter |
| `step_precompute` | constants | $8$ | Subsampling for ones projection |
| `step_iterate` | constants | $10$ | Subsampling for iteration projections |
| $\epsilon_{norm}$ | constants | $10^{-8}$ | Division safety constant |

---

## 3. PREPROCESSING PIPELINE

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 1 | Load Calibration | $w = \text{imread}(\texttt{calib.tif})$ | File path | $w$: $(H_{raw}, W_{raw})$ |
| 2 | Load Config | $\text{config} = \text{yaml.load}(\texttt{config.yaml})$ | File path | `config_dict`: dict |
| 3 | Load Raw Image | $y_{raw} = \text{imread}(\texttt{example\_fly.tif})$ | File path | $y_{raw}$: $(H_{raw}, W_{raw})$ |
| 4 | Set Camera Params | $\text{Camera} = \text{LFM\_setCameraParams\_v2}(\text{config}, \text{new\_spacing\_px})$ | config_dict, $15$ | Camera: dict |
| 5 | Compute Geometry | $(\text{LC}, \text{Res}, \text{LGM}, \text{NLGM}) = \text{LFM\_computeGeometryParameters}(\text{Camera}, w, [-300,300], 150, 5, \text{False})$ | Camera, $w$, depth params | LensletCenters, Resolution, LensletGridModel, NewLensletGridModel |
| 6 | Compute Operators | $(H, H^T) = \text{LFM\_computeLFMatrixOperators}(\text{Camera}, \text{Res}, \text{LC})$ | Camera, Resolution, LensletCenters | Forward operator $H$, Adjoint operator $H^T$ |
| 7 | Compute Transform | $T = \text{format\_transform}(\text{LFM\_retrieveTransformation}(\text{LGM}, \text{NLGM}))$ | LensletGridModel, NewLensletGridModel | Affine transform $T$ |
| 8 | Compute imgSize | $\text{imgSize} = \text{get\_transformed\_shape}(w.\text{shape}, T) + (1 - \text{mod}(\cdot, 2))$ | $w.\text{shape}$, $T$ | imgSize: $(H_{img}, W_{img})$ int32 (even) |
| 9 | Compute texSize | $\text{texSize} = \lceil \text{imgSize} \times \text{texScaleFactor} \rceil + (1 - \text{mod}(\cdot, 2))$ | imgSize, texScaleFactor | texSize: $(T_x, T_y)$ int32 (even) |
| 10 | Compute volumeSize | $\text{volumeSize} = [\text{texSize}_x, \text{texSize}_y, |\text{depths}|]$ | texSize, $N_z = \text{len}(\text{depths})$ | volumeSize: $(T_x, T_y, N_z)$ int32 |
| 11 | Transform Image | $y_t = \text{transform\_img}(y_{raw}, T, \text{LC.offset})$ | $y_{raw}$, $T$, offset | $y_t$: $(H_{img}, W_{img})$ float32 |
| 12 | Normalize Image | $b = \frac{y_t - \min(y_t)}{\max(y_t) - \min(y_t)}$ | $y_t$ | $b$ (`LFimage`): $(H_{img}, W_{img})$ float32, range $[0,1]$ |
| 13 | Compute Filter Widths | $\sigma_z = \text{LFM\_computeDepthAdaptiveWidth}(\text{Camera}, \text{Res})$ | Camera, Resolution | widths: $(N_z,)$ float32 |
| 14 | Build Lanczos Kernels | $\tilde{K} = \text{lanczosfft}(\text{volumeSize}, \sigma_z, 4)$ | volumeSize, widths, $a=4$ | kernelFFT: $(T_x, T_y, N_z)$ complex64 |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| I.1 | Initialize Volume | $V^{(0)} = \mathbf{1} \in \mathbb{R}^{T_x \times T_y \times N_z}$ | volumeSize | $V^{(0)}$: float32, all ones |
| I.2 | Precompute Forward Ones | $\mathbf{1}_H = H(V^{(0)})$ with step$=8$ | $H$, $V^{(0)}$, geometry params | $\mathbf{1}_H$: $(H_{img}, W_{img})$ float32 |
| I.3 | Precompute Backward Ones | $\mathbf{1}_{H^T} = H^T(\mathbf{1}_H)$ with step$=8$ | $H^T$, $\mathbf{1}_H$, geometry params | $\mathbf{1}_{H^T}$: $(T_x, T_y, N_z)$ float32 |
| I.4 | Copy Initial Estimate | $V^{(0)}_{work} = \text{copy}(V^{(0)})$ | $V^{(0)}$ | reconVolume: $(T_x, T_y, N_z)$ float32 |
| I.5 | Initialize Error List | $\mathcal{E} = []$ | - | error_metrics: list |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** ($k = 0, 1, \ldots, N_{iter}-1$)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| L.1 | Forward Project | $\hat{b}^{(k)} = \begin{cases} \mathbf{1}_H & k=0 \\ H(V^{(k)}) & k>0 \end{cases}$ | $H$, $V^{(k)}$, step$=10$ | $\hat{b}^{(k)}$: $(H_{img}, W_{img})$ | `LFM_forwardProject` with subsampling |
| L.2 | Free GPU Memory | `mempool.free_all_blocks()` | - | - | CuPy memory management |
| L.3 | Compute Error Ratio | $e_b^{(k)} = \frac{b}{\hat{b}^{(k)}} \cdot \mathbf{1}_H$ | $b$, $\hat{b}^{(k)}$, $\mathbf{1}_H$ | $e_b^{(k)}$: $(H_{img}, W_{img})$ | Element-wise operations |
| L.4 | Handle Non-finite | $e_b^{(k)}[i] = 0 \;\forall i : \neg\text{isfinite}(e_b^{(k)}[i])$ | $e_b^{(k)}$ | $e_b^{(k)}$ (cleaned) | Zero out NaN/Inf values |
| L.5 | Compute MAE | $\epsilon^{(k)} = \frac{1}{N_{pixels}} \sum_i |e_b^{(k)}_i - (\mathbf{1}_H)_i|$ | $e_b^{(k)}$, $\mathbf{1}_H$ | $\epsilon^{(k)} \in \mathbb{R}$ | Mean absolute error metric |
| L.6 | Store Metric | $\mathcal{E}.\text{append}(\epsilon^{(k)})$ | $\epsilon^{(k)}$ | Updated $\mathcal{E}$ | - |
| L.7 | Backward Project | $e_V^{(k)} = H^T(e_b^{(k)})$ | $H^T$, $e_b^{(k)}$, step$=10$ | $e_V^{(k)}$: $(T_x, T_y, N_z)$ | `LFM_backwardProject` with subsampling |
| L.8 | Free GPU Memory | `mempool.free_all_blocks()` | - | - | CuPy memory management |
| L.9 | Normalize Error | $e_V^{(k)} = \frac{e_V^{(k)}}{\mathbf{1}_{H^T}}$ | $e_V^{(k)}$, $\mathbf{1}_{H^T}$ | $e_V^{(k)}$ (normalized) | Element-wise division |
| L.10 | Handle Non-finite | $e_V^{(k)}[i,j,z] = 0 \;\forall (i,j,z) : \neg\text{isfinite}(e_V^{(k)}[i,j,z])$ | $e_V^{(k)}$ | $e_V^{(k)}$ (cleaned) | Zero out NaN/Inf values |
| L.11 | Multiplicative Update | $V^{(k+1)} = V^{(k)} \odot e_V^{(k)}$ | $V^{(k)}$, $e_V^{(k)}$ | $V^{(k+1)}$: $(T_x, T_y, N_z)$ | Element-wise multiplication |
| L.12 | Anti-aliasing Filter | **IF** `filter_flag`: **BEGIN_INNER_LOOP** ($z = 0, \ldots, N_z-1$): $V^{(k+1)}_{:,:,z} = \left| \mathcal{F}^{-1}_{\text{shift}}\left[ \tilde{K}_{:,:,z} \odot \mathcal{F}[V^{(k+1)}_{:,:,z}] \right] \right|$ **END_INNER_LOOP** | $V^{(k+1)}$, $\tilde{K}$ | $V^{(k+1)}$ (filtered) | Per-slice 2D FFT convolution |
| L.13 | Final Non-finite Check | $V^{(k+1)}[i,j,z] = 0 \;\forall (i,j,z) : \neg\text{isfinite}(V^{(k+1)}[i,j,z])$ | $V^{(k+1)}$ | $V^{(k+1)}$ (cleaned) | - |
| L.14 | Free GPU Memory | `mempool.free_all_blocks()` | - | - | CuPy memory management |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| F.1 | Transfer to CPU | $V_{out} = \text{asnumpy}(V^{(N_{iter})})$ | $V^{(N_{iter})}$ (GPU) | $V_{out}$: numpy array $(T_x, T_y, N_z)$ |
| F.2 | Extract Center Slice | $S_{center} = V_{out}[:, :, \lfloor N_z/2 \rfloor]$ | $V_{out}$ | $S_{center}$: $(T_x, T_y)$ |

---

## 5. STOPPING CRITERION

- **type**: `iteration_limit`
- **expression**: $k \geq N_{iter}$
- **parameters**:
  - $N_{iter} = 100$ (production) or $N_{iter} = 1$ (testing)
- **check_frequency**: Evaluated at the end of each iteration (implicit in `for i in range(niter)`)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Light Field MAE | $\epsilon^{(k)} = \frac{1}{|I|} \sum_{i \in I} \left| e_b^{(k)}_i - (\mathbf{1}_H)_i \right|$ where $e_b^{(k)} = \frac{b}{\hat{b}^{(k)}} \cdot \mathbf{1}_H$ | Mean absolute deviation of the weighted error ratio from the ones-forward projection; measures convergence toward data consistency |
| Normalized Display | $\tilde{I} = \frac{I - \min(I)}{\max(I) - \min(I) + 10^{-8}} \times 255$ | Uint8 normalization for visualization |

---

## APPENDIX: KEY MATHEMATICAL DEFINITIONS

### Richardson-Lucy Update (Standard Form)
The multiplicative update implements:
$$V^{(k+1)} = V^{(k)} \cdot \frac{H^T\left( \frac{b}{H V^{(k)}} \right)}{H^T \mathbf{1}}$$

In code, this is computed as:
$$V^{(k+1)} = V^{(k)} \cdot \frac{H^T\left( \frac{b}{H V^{(k)}} \cdot H\mathbf{1} \right)}{H^T(H\mathbf{1})}$$

### Lanczos Filter Definition
The Lanczos kernel of order $a$ (lanczos_window_size$=4$):
$$L_a(x) = \begin{cases} \text{sinc}(x) \cdot \text{sinc}(x/a) & |x| < a \\ 0 & \text{otherwise} \end{cases}$$

where $\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$.

The depth-adaptive 2D separable kernel in Fourier domain:
$$\tilde{K}_z = \text{FFT}_{2D}\left[ L_a(x/\sigma_z) \cdot L_a(y/\sigma_z) \right]$$

where $\sigma_z$ is computed by `LFM_computeDepthAdaptiveWidth` based on camera geometry and depth $z$.