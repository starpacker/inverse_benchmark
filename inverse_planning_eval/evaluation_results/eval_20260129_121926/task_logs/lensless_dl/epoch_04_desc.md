## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: APGD (Accelerated Proximal Gradient Descent) / FISTA for Image Deconvolution
- **algorithm_family**: Proximal Gradient Methods, First-Order Optimization
- **summary**: Reconstructs a latent image from blurred measurements by minimizing squared error with non-negativity constraints using FISTA momentum acceleration.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{x \in \mathbb{R}^{D \times H \times W \times C}} \frac{1}{2}\|Ax - y\|_2^2 + \mathcal{I}_{\geq 0}(x)$$

where $A$ is the convolution operator with PSF $h$, $y$ is the observed measurement, and $\mathcal{I}_{\geq 0}$ is the indicator function for the non-negative orthant.

### data_fidelity_term
$$f(x) = \frac{1}{2}\|Ax - y\|_2^2 = \frac{1}{2}\sum_{d,i,j,c}\left((h * x)_{d,i,j,c} - y_{d,i,j,c}\right)^2$$

where $*$ denotes 2D convolution over spatial axes $(-3, -2)$ corresponding to height and width dimensions.

**Verification**: L2 loss confirmed by squared residual computation `residual = Ay_k - measurement` followed by gradient `A^T(residual)`.

### regularization_term
$$g(x) = \mathcal{I}_{\geq 0}(x) = \begin{cases} 0 & \text{if } x_{d,i,j,c} \geq 0 \;\forall\, d,i,j,c \\ +\infty & \text{otherwise} \end{cases}$$

### constraints
- **Non-negativity**: $x \geq 0$ (elementwise), enforced via proximal operator
- **No explicit box upper bound**: Upper bound is unconstrained

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x_k$ | $(D, H, W, C)$ | float32 | $\mathbf{0}$ (zeros_like psf) | Current primal iterate at iteration $k$ |
| $x_{k+1}$ | $(D, H, W, C)$ | float32 | Computed | Next primal iterate |

### 2.2 Dual Variables
None. This is a primal-only method.

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $y_k$ | $(D, H, W, C)$ | float32 | $\mathbf{0}$ (copy of $x_0$) | Extrapolated point for momentum |
| $t_k$ | scalar | float64 | $1.0$ | FISTA momentum parameter |
| $t_{k+1}$ | scalar | float64 | Computed | Next momentum parameter |
| $Ay_k$ | $(D, H, W, C)$ | float32 | Computed | Forward model applied to $y_k$ |
| $r_k$ | $(D, H, W, C)$ | float32 | Computed | Residual $Ay_k - y$ |
| $\nabla f(y_k)$ | $(D, H, W, C)$ | float32 | Computed | Gradient at extrapolated point |
| $z_k$ | $(D, H, W, C)$ | float32 | Computed | Pre-proximal gradient descent result |
| $H$ | $(D, H_{pad}, \lfloor W_{pad}/2\rfloor + 1, C)$ | complex64 | Computed | rFFT2 of zero-padded PSF |
| $H^*$ | $(D, H_{pad}, \lfloor W_{pad}/2\rfloor + 1, C)$ | complex64 | $\overline{H}$ (conjugate) | Adjoint filter in frequency domain |
| $L$ | scalar | float32 | Power method estimate | Lipschitz constant of $\nabla f$ |
| $\alpha$ | scalar | float32 | $1/L$ | Step size |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| `downsample` | 4 | Spatial downsampling factor |
| `n_iter` | 50 (default), 5 (in main) | Maximum iterations |
| `power_method_max_iter` | 20 | Iterations for Lipschitz estimation |
| `fft_norm` | "ortho" | FFT normalization mode |
| `t_0` | 1.0 | Initial momentum parameter |
| `dtype` | float32 | Numerical precision |

### 2.5 Observations
| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $y$ | $(D, H, W, C)$ | float32 | `data_dict["data"]` | Measured/observed image |
| $h$ | $(D, H, W, C)$ | float32 | `data_dict["psf"]` | Point spread function |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Load PSF | $h_{raw} = \text{load\_image}(\text{psf\_path})$ | psf_path: string | $h_{raw}$: $(H_{orig}, W_{orig})$ or $(H_{orig}, W_{orig}, C)$ |
| 1 | Load Measurement | $y_{raw} = \text{load\_image}(\text{data\_path})$ | data_path: string | $y_{raw}$: $(H_{orig}, W_{orig})$ or $(H_{orig}, W_{orig}, C)$ |
| 2 | Downsample PSF | $h_{ds}[i,j] = h_{raw}[i \cdot s, j \cdot s]$ where $s=4$ | $h_{raw}$, $s=4$ | $h_{ds}$: $(H_{orig}/s, W_{orig}/s, \ldots)$ |
| 3 | Downsample Measurement | $y_{ds}[i,j] = y_{raw}[i \cdot s, j \cdot s]$ where $s=4$ | $y_{raw}$, $s=4$ | $y_{ds}$: $(H_{orig}/s, W_{orig}/s, \ldots)$ |
| 4 | Normalize PSF | $h_{norm} = \frac{h_{ds} - \min(h_{ds})}{\max(h_{ds}) - \min(h_{ds})}$ | $h_{ds}$ | $h_{norm} \in [0,1]$ |
| 5 | Normalize Measurement | $y_{norm} = \frac{y_{ds} - \min(y_{ds})}{\max(y_{ds}) - \min(y_{ds})}$ | $y_{ds}$ | $y_{norm} \in [0,1]$ |
| 6 | Reshape to 4D | If 2D: $h \leftarrow h_{norm}[\text{newaxis}, :, :, \text{newaxis}]$; If 3D: $h \leftarrow h_{norm}[\text{newaxis}, :, :, :]$ | $h_{norm}$ | $h$: $(D, H, W, C)$ |
| 7 | Reshape Measurement to 4D | Same logic as Step 6 | $y_{norm}$ | $y$: $(D, H, W, C)$ |
| 8 | Compute Padded Shape | $H_{pad} = \text{next\_fast\_len}(2H - 1)$, $W_{pad} = \text{next\_fast\_len}(2W - 1)$ | $h$ shape | padded_shape: $(D, H_{pad}, W_{pad}, C)$ |
| 9 | Compute Crop Indices | $\text{start} = \lfloor(H_{pad} - H)/2\rfloor$, $\text{end} = \text{start} + H$ (similarly for $W$) | padded_shape, psf_shape | start_idx, end_idx |
| 10 | Compute PSF FFT | $H = \mathcal{F}_{2D}(\text{pad}(h))$ with ortho normalization | $h$, setup | $H$: complex frequency response |
| 11 | Compute Adjoint Filter | $H^* = \overline{H}$ (complex conjugate) | $H$ | $H^*$: adjoint frequency response |
| 12 | Estimate Lipschitz Constant | Power method: $L = \|A^T A\|_2 \approx$ largest singular value squared | $H$, $H^*$, setup | $L$: scalar |
| 13 | Compute Step Size | $\alpha = 1/L$ | $L$ | $\alpha$: scalar |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION
| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Initialize Primal | $x_0 = \mathbf{0} \in \mathbb{R}^{D \times H \times W \times C}$ | psf shape | $x_0$ | `np.zeros_like(psf)` |
| Initialize Extrapolation | $y_0 = x_0$ | $x_0$ | $y_0$ | Copy operation |
| Initialize Momentum | $t_0 = 1.0$ | None | $t_0$ | Scalar assignment |
| Compute Step Size | $\alpha = \begin{cases} 1/L & \text{if } L > 0 \\ 1.0 & \text{otherwise} \end{cases}$ | $L$ | $\alpha$ | Safeguard against $L=0$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, n_{iter}-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Forward Model [FORWARD] | $Ay_k = \mathcal{F}^{-1}_{2D}\left[\mathcal{F}_{2D}(\text{pad}(y_k)) \odot H\right]_{\text{crop}}$ | $y_k$, $H$, setup | $Ay_k$ | FFT convolution with ifftshift; ortho norm |
| 2 | Compute Residual | $r_k = Ay_k - y$ | $Ay_k$, $y$ | $r_k$ | Elementwise subtraction |
| 3 | Compute Gradient [ADJOINT] | $\nabla f(y_k) = A^T r_k = \mathcal{F}^{-1}_{2D}\left[\mathcal{F}_{2D}(\text{pad}(r_k)) \odot H^*\right]_{\text{crop}}$ | $r_k$, $H^*$, setup | $\nabla f(y_k)$ | Adjoint = correlation via conjugate |
| 4 | Gradient Descent Step | $z_k = y_k - \alpha \nabla f(y_k)$ | $y_k$, $\alpha$, $\nabla f(y_k)$ | $z_k$ | Pre-proximal iterate |
| 5 | Proximal Operator | $x_{k+1} = \text{prox}_{\alpha g}(z_k) = \max(z_k, 0)$ | $z_k$ | $x_{k+1}$ | Elementwise maximum with 0 |
| 6 | Update Momentum Parameter | $t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$ | $t_k$ | $t_{k+1}$ | FISTA momentum sequence |
| 7 | Compute Extrapolation Coefficient | $\beta_k = \frac{t_k - 1}{t_{k+1}}$ | $t_k$, $t_{k+1}$ | $\beta_k$ | Momentum coefficient |
| 8 | Extrapolation Step | $y_{k+1} = x_{k+1} + \beta_k(x_{k+1} - x_k)$ | $x_{k+1}$, $x_k$, $\beta_k$ | $y_{k+1}$ | Nesterov-type momentum |
| 9 | Update Iterates | $x_k \leftarrow x_{k+1}$, $t_k \leftarrow t_{k+1}$ | $x_{k+1}$, $t_{k+1}$ | $x_k$, $t_k$ | Prepare for next iteration |

**END_LOOP**

#### Nested Procedure: Power Method for Lipschitz Constant [INNER]

**BEGIN_LOOP** (for $j = 0, 1, \ldots, 19$)

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Initialize | $v_0 \sim \mathcal{N}(0, 1)^{D \times H \times W \times C}$, $v_0 \leftarrow v_0 / \|v_0\|_2$ | psf_shape | $v_0$ |
| 1 | Forward [FORWARD] | $u_j = A v_j$ | $v_j$, $H$ | $u_j$ |
| 2 | Adjoint [ADJOINT] | $w_j = A^T u_j$ | $u_j$, $H^*$ | $w_j$ |
| 3 | Compute Norm | $\sigma_j = \|w_j\|_2$ | $w_j$ | $\sigma_j$ |
| 4 | Normalize | $v_{j+1} = w_j / \sigma_j$ (if $\sigma_j > 0$) | $w_j$, $\sigma_j$ | $v_{j+1}$ |

**END_LOOP**

Output: $L = \sigma_{19}$ (final norm value)

### 4.3 FINALIZATION
| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Extract Result | $\hat{x} = x_{n_{iter}}$ | Final $x_k$ | $\hat{x}$ | Final iterate |
| Remove Batch Dimension | If $D=1$: $\hat{x} \leftarrow \hat{x}[0]$ | $\hat{x}$ | $\hat{x}$: $(H, W, C)$ | Squeeze singleton depth |
| Normalize for Display | $\hat{x}_{disp} = \frac{\hat{x} - \min(\hat{x})}{\max(\hat{x}) - \min(\hat{x})}$ | $\hat{x}$ | $\hat{x}_{disp} \in [0,1]$ | For visualization only |
| Clip Display | $\hat{x}_{disp} = \text{clip}(\hat{x}_{disp}, 0, 1)$ | $\hat{x}_{disp}$ | $\hat{x}_{disp}$ | Ensure valid range |
| Save Outputs | Write PNG and NPY files | $\hat{x}$, $\hat{x}_{disp}$ | Files | PNG at dpi=150 |

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k < n_{iter}$
- **parameters**: 
  - `n_iter`: 50 (default argument), 5 (value used in `__main__`)
- **check_frequency**: Every iteration (loop condition)

No convergence-based early stopping is implemented.

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| Reconstruction Min | $\min_{d,i,j,c} \hat{x}_{d,i,j,c}$ | Minimum pixel value in reconstruction |
| Reconstruction Max | $\max_{d,i,j,c} \hat{x}_{d,i,j,c}$ | Maximum pixel value in reconstruction |
| Reconstruction Mean | $\frac{1}{N}\sum_{d,i,j,c} \hat{x}_{d,i,j,c}$ where $N = D \cdot H \cdot W \cdot C$ | Average pixel value |

Note: No quantitative error metrics (PSNR, SSIM) are computed as no ground truth is available.

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON MUST include:

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
  - primal_variables (list with: symbol, shape, dtype, initialization, description)
  - dual_variables
  - auxiliary_variables (list with: symbol, shape, dtype, initialization, description)
  - constants (list with: symbol, value, description)
  - observations (list with: symbol, shape, dtype, source, description)

data_preprocessing:
  - steps (list with: step_order, step_name, formula, inputs, outputs)

initialization:
  - steps (list with: step_name, formula, inputs, outputs)

iteration_loop:
  - loop_structure: {start: 0, end: n_iter, variable: k}
  - steps (list with: step_order, step_name, formula, inputs, outputs, computational_note)
  - nested_procedures (list with: name, loop_structure, steps)

finalization:
  - steps (list with: step_name, formula, inputs, outputs)

stopping_criterion:
  - type
  - expression
  - parameters
  - check_frequency

evaluation:
  - metrics (list with: name, formula, description)

operators:
  - forward_operator: {name, formula, type: FORWARD}
  - adjoint_operator: {name, formula, type: ADJOINT}
  - proximal_operator: {name, definition, closed_form_solution}
```

**Proximal Operator Detail**:
- **Name**: Non-negativity projection
- **Definition**: $\text{prox}_{\alpha \mathcal{I}_{\geq 0}}(z) = \arg\min_{x \geq 0} \frac{1}{2}\|x - z\|_2^2$
- **Closed-form solution**: $[\text{prox}_{\alpha \mathcal{I}_{\geq 0}}(z)]_{d,i,j,c} = \max(z_{d,i,j,c}, 0)$