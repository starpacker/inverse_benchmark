## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: APGD (Accelerated Proximal Gradient Descent) / FISTA
- **algorithm_family**: First-order proximal gradient methods with Nesterov momentum
- **summary**: Non-negatively constrained image deconvolution using accelerated proximal gradient descent with FFT-based convolution operators.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{x}} \frac{1}{2} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + \mathcal{I}_{\geq 0}(\mathbf{x})$$

### data_fidelity_term
$$f(\mathbf{x}) = \frac{1}{2} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2$$

where $\mathbf{A}$ is the 2D convolution operator with PSF $\mathbf{h}$, and $\mathbf{b}$ is the measured data.

### regularization_term
None (no explicit regularization beyond the constraint)

### constraints
$$\mathcal{I}_{\geq 0}(\mathbf{x}) = \begin{cases} 0 & \text{if } x_{ijkc} \geq 0 \; \forall i,j,k,c \\ +\infty & \text{otherwise} \end{cases}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}^{(k)}$ | $(D, H, W, C)$ | float32 | $\mathbf{0}$ | Current image estimate at iteration $k$ |

### 2.2 Dual Variables

None (primal-only algorithm)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}^{(k)}$ | $(D, H, W, C)$ | float32 | $\mathbf{0}$ | Momentum-extrapolated point |
| $t^{(k)}$ | scalar | float64 | $1.0$ | FISTA momentum parameter |
| $\mathbf{A}\mathbf{y}^{(k)}$ | $(D, H, W, C)$ | float32 | — | Forward convolution of extrapolated point |
| $\mathbf{r}^{(k)}$ | $(D, H, W, C)$ | float32 | — | Residual $\mathbf{A}\mathbf{y}^{(k)} - \mathbf{b}$ |
| $\nabla f(\mathbf{y}^{(k)})$ | $(D, H, W, C)$ | float32 | — | Gradient at extrapolated point |
| $\mathbf{z}^{(k)}$ | $(D, H, W, C)$ | float32 | — | Pre-proximal gradient step result |
| $\mathbf{H}$ | $(D, H_p, \lfloor W_p/2 \rfloor + 1, C)$ | complex64 | — | FFT of zero-padded PSF |
| $\mathbf{H}^*$ | $(D, H_p, \lfloor W_p/2 \rfloor + 1, C)$ | complex64 | — | Complex conjugate of $\mathbf{H}$ |
| $L$ | scalar | float32 | — | Lipschitz constant of $\nabla f$ |
| $\alpha$ | scalar | float32 | — | Step size $1/L$ |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $n_{\text{iter}}$ | 50 (default) | Maximum number of iterations |
| $n_{\text{power}}$ | 20 | Power method iterations for Lipschitz estimation |
| $d$ | 4 (default) | Downsampling factor |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{b}$ | $(D, H, W, C)$ | float32 | Preprocessed measurement data |
| $\mathbf{h}$ | $(D, H, W, C)$ | float32 | Preprocessed point spread function |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load PSF | $\mathbf{h}_{\text{raw}} = \text{read\_image}(\text{psf\_path})$ | psf_path (string) | $\mathbf{h}_{\text{raw}}$ : $(H_0, W_0)$ or $(H_0, W_0, C)$ |
| 2 | Load Data | $\mathbf{b}_{\text{raw}} = \text{read\_image}(\text{data\_path})$ | data_path (string) | $\mathbf{b}_{\text{raw}}$ : $(H_0, W_0)$ or $(H_0, W_0, C)$ |
| 3 | Downsample PSF | $\mathbf{h}_{\text{ds}}[i,j,c] = \mathbf{h}_{\text{raw}}[d \cdot i, d \cdot j, c]$ | $\mathbf{h}_{\text{raw}}$, $d$ | $\mathbf{h}_{\text{ds}}$ : $(\lfloor H_0/d \rfloor, \lfloor W_0/d \rfloor, C)$ |
| 4 | Downsample Data | $\mathbf{b}_{\text{ds}}[i,j,c] = \mathbf{b}_{\text{raw}}[d \cdot i, d \cdot j, c]$ | $\mathbf{b}_{\text{raw}}$, $d$ | $\mathbf{b}_{\text{ds}}$ : $(\lfloor H_0/d \rfloor, \lfloor W_0/d \rfloor, C)$ |
| 5 | Normalize PSF | $\mathbf{h}_{\text{norm}} = \frac{\mathbf{h}_{\text{ds}} - \min(\mathbf{h}_{\text{ds}})}{\max(\mathbf{h}_{\text{ds}}) - \min(\mathbf{h}_{\text{ds}})}$ | $\mathbf{h}_{\text{ds}}$ | $\mathbf{h}_{\text{norm}}$ : same shape |
| 6 | Normalize Data | $\mathbf{b}_{\text{norm}} = \frac{\mathbf{b}_{\text{ds}} - \min(\mathbf{b}_{\text{ds}})}{\max(\mathbf{b}_{\text{ds}}) - \min(\mathbf{b}_{\text{ds}})}$ | $\mathbf{b}_{\text{ds}}$ | $\mathbf{b}_{\text{norm}}$ : same shape |
| 7 | Reshape PSF to 4D | $\mathbf{h} = \mathbf{h}_{\text{norm}}[\text{newaxis}, :, :, :]$ (add channel dim if 2D, then depth dim) | $\mathbf{h}_{\text{norm}}$ | $\mathbf{h}$ : $(1, H, W, C)$ |
| 8 | Reshape Data to 4D | $\mathbf{b} = \mathbf{b}_{\text{norm}}[\text{newaxis}, :, :, :]$ | $\mathbf{b}_{\text{norm}}$ | $\mathbf{b}$ : $(1, H, W, C)$ |
| 9 | Compute Padded Shape | $H_p = \text{next\_fast\_len}(2H - 1), \quad W_p = \text{next\_fast\_len}(2W - 1)$ | $H, W$ | $H_p, W_p$ (integers) |
| 10 | Compute Padding Indices | $s_0 = \lfloor(H_p - H)/2\rfloor, \; s_1 = \lfloor(W_p - W)/2\rfloor, \; e_0 = s_0 + H, \; e_1 = s_1 + W$ | $H, W, H_p, W_p$ | $s_0, s_1, e_0, e_1$ |
| 11 | Pad PSF | $\mathbf{h}_{\text{pad}} \in \mathbb{R}^{D \times H_p \times W_p \times C} = \mathbf{0}$; $\mathbf{h}_{\text{pad}}[:, s_0:e_0, s_1:e_1, :] = \mathbf{h}$ | $\mathbf{h}$, padding indices | $\mathbf{h}_{\text{pad}}$ : $(D, H_p, W_p, C)$ |
| 12 | Compute PSF FFT | $\mathbf{H} = \text{rfft2}(\mathbf{h}_{\text{pad}}, \text{axes}=(-3,-2), \text{norm}=\text{"ortho"})$ | $\mathbf{h}_{\text{pad}}$ | $\mathbf{H}$ : $(D, H_p, \lfloor W_p/2\rfloor+1, C)$ complex |
| 13 | Compute Adjoint Filter | $\mathbf{H}^* = \overline{\mathbf{H}}$ (element-wise complex conjugate) | $\mathbf{H}$ | $\mathbf{H}^*$ : same shape, complex |
| 14 | Estimate Lipschitz Constant | See Steps 15–19 (power method subroutine) | $\mathbf{H}, \mathbf{H}^*$, setup | $L$ : scalar |

**Power Method Subroutine (Steps 15–19):**

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 15 | Initialize Random Vector | $\mathbf{v}^{(0)} \sim \mathcal{N}(0,1)^{D \times H \times W \times C}$; $\mathbf{v}^{(0)} \leftarrow \mathbf{v}^{(0)} / \|\mathbf{v}^{(0)}\|_2$ | shape $(D,H,W,C)$ | $\mathbf{v}^{(0)}$ |
| 16 | BEGIN_LOOP | for $j = 0, \ldots, n_{\text{power}}-1$ | | |
| 17 | Apply $\mathbf{A}^T\mathbf{A}$ | $\mathbf{u} = \mathbf{A}^T(\mathbf{A}\mathbf{v}^{(j)})$ using FFT convolution/deconvolution | $\mathbf{v}^{(j)}, \mathbf{H}, \mathbf{H}^*$ | $\mathbf{u}$ |
| 18 | Normalize | $\sigma = \|\mathbf{u}\|_2$; $\mathbf{v}^{(j+1)} = \mathbf{u} / \sigma$ | $\mathbf{u}$ | $\mathbf{v}^{(j+1)}, \sigma$ |
| 19 | END_LOOP | | | |
| 20 | Set Lipschitz | $L = \sigma$ (final eigenvalue estimate) | $\sigma$ | $L$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize Primal | $\mathbf{x}^{(0)} = \mathbf{0} \in \mathbb{R}^{D \times H \times W \times C}$ | shape from $\mathbf{h}$ | $\mathbf{x}^{(0)}$ | Zero-filled array |
| 2 | Initialize Extrapolation | $\mathbf{y}^{(0)} = \mathbf{x}^{(0)}$ | $\mathbf{x}^{(0)}$ | $\mathbf{y}^{(0)}$ | Copy of primal |
| 3 | Initialize Momentum | $t^{(0)} = 1.0$ | — | $t^{(0)}$ | Scalar |
| 4 | Compute Step Size | $\alpha = \begin{cases} 1/L & \text{if } L > 0 \\ 1 & \text{otherwise} \end{cases}$ | $L$ | $\alpha$ | Inverse Lipschitz |

### 4.2 ITERATION LOOP

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 5 | BEGIN_LOOP | for $k = 0, \ldots, n_{\text{iter}}-1$ | | | |
| 6 | Pad Extrapolated Point | $\mathbf{y}^{(k)}_{\text{pad}} \in \mathbb{R}^{D \times H_p \times W_p \times C} = \mathbf{0}$; $\mathbf{y}^{(k)}_{\text{pad}}[:, s_0:e_0, s_1:e_1, :] = \mathbf{y}^{(k)}$ | $\mathbf{y}^{(k)}$, padding indices | $\mathbf{y}^{(k)}_{\text{pad}}$ | [FORWARD] Zero-padding |
| 7 | FFT of Padded Input | $\hat{\mathbf{y}}^{(k)} = \text{rfft2}(\mathbf{y}^{(k)}_{\text{pad}}, \text{axes}=(-3,-2), \text{norm}=\text{"ortho"})$ | $\mathbf{y}^{(k)}_{\text{pad}}$ | $\hat{\mathbf{y}}^{(k)}$ | [FORWARD] 2D real FFT |
| 8 | Frequency Multiplication | $\hat{\mathbf{w}}^{(k)} = \hat{\mathbf{y}}^{(k)} \odot \mathbf{H}$ | $\hat{\mathbf{y}}^{(k)}, \mathbf{H}$ | $\hat{\mathbf{w}}^{(k)}$ | [FORWARD] Element-wise complex multiply |
| 9 | Inverse FFT | $\mathbf{w}^{(k)}_{\text{pad}} = \text{ifftshift}(\text{irfft2}(\hat{\mathbf{w}}^{(k)}, \text{axes}=(-3,-2), s=(H_p, W_p), \text{norm}=\text{"ortho"}), \text{axes}=(-3,-2))$ | $\hat{\mathbf{w}}^{(k)}$ | $\mathbf{w}^{(k)}_{\text{pad}}$ | [FORWARD] Shift to center |
| 10 | Crop Convolution Result | $\mathbf{A}\mathbf{y}^{(k)} = \text{Re}(\mathbf{w}^{(k)}_{\text{pad}}[:, s_0:e_0, s_1:e_1, :])$ | $\mathbf{w}^{(k)}_{\text{pad}}$, indices | $\mathbf{A}\mathbf{y}^{(k)}$ | [FORWARD] Extract valid region |
| 11 | Compute Residual | $\mathbf{r}^{(k)} = \mathbf{A}\mathbf{y}^{(k)} - \mathbf{b}$ | $\mathbf{A}\mathbf{y}^{(k)}, \mathbf{b}$ | $\mathbf{r}^{(k)}$ | Data fidelity residual |
| 12 | Pad Residual | $\mathbf{r}^{(k)}_{\text{pad}} \in \mathbb{R}^{D \times H_p \times W_p \times C} = \mathbf{0}$; $\mathbf{r}^{(k)}_{\text{pad}}[:, s_0:e_0, s_1:e_1, :] = \mathbf{r}^{(k)}$ | $\mathbf{r}^{(k)}$, padding indices | $\mathbf{r}^{(k)}_{\text{pad}}$ | [ADJOINT] Zero-padding |
| 13 | FFT of Residual | $\hat{\mathbf{r}}^{(k)} = \text{rfft2}(\mathbf{r}^{(k)}_{\text{pad}}, \text{axes}=(-3,-2), \text{norm}=\text{"ortho"})$ | $\mathbf{r}^{(k)}_{\text{pad}}$ | $\hat{\mathbf{r}}^{(k)}$ | [ADJOINT] 2D real FFT |
| 14 | Adjoint Frequency Multiply | $\hat{\mathbf{g}}^{(k)} = \hat{\mathbf{r}}^{(k)} \odot \mathbf{H}^*$ | $\hat{\mathbf{r}}^{(k)}, \mathbf{H}^*$ | $\hat{\mathbf{g}}^{(k)}$ | [ADJOINT] Complex conjugate filter |
| 15 | Adjoint Inverse FFT | $\mathbf{g}^{(k)}_{\text{pad}} = \text{ifftshift}(\text{irfft2}(\hat{\mathbf{g}}^{(k)}, \text{axes}=(-3,-2), s=(H_p, W_p), \text{norm}=\text{"ortho"}), \text{axes}=(-3,-2))$ | $\hat{\mathbf{g}}^{(k)}$ | $\mathbf{g}^{(k)}_{\text{pad}}$ | [ADJOINT] Shift to center |
| 16 | Crop Gradient | $\nabla f(\mathbf{y}^{(k)}) = \text{Re}(\mathbf{g}^{(k)}_{\text{pad}}[:, s_0:e_0, s_1:e_1, :])$ | $\mathbf{g}^{(k)}_{\text{pad}}$, indices | $\nabla f(\mathbf{y}^{(k)})$ | [ADJOINT] $\mathbf{A}^T(\mathbf{A}\mathbf{y}^{(k)} - \mathbf{b})$ |
| 17 | Gradient Descent Step | $\mathbf{z}^{(k)} = \mathbf{y}^{(k)} - \alpha \nabla f(\mathbf{y}^{(k)})$ | $\mathbf{y}^{(k)}, \alpha, \nabla f(\mathbf{y}^{(k)})$ | $\mathbf{z}^{(k)}$ | Pre-proximal update |
| 18 | Proximal Operator | $\mathbf{x}^{(k+1)} = \text{prox}_{\alpha \mathcal{I}_{\geq 0}}(\mathbf{z}^{(k)}) = \max(\mathbf{z}^{(k)}, 0)$ | $\mathbf{z}^{(k)}$ | $\mathbf{x}^{(k+1)}$ | Element-wise non-negativity projection |
| 19 | Update Momentum Parameter | $t^{(k+1)} = \frac{1 + \sqrt{1 + 4(t^{(k)})^2}}{2}$ | $t^{(k)}$ | $t^{(k+1)}$ | FISTA sequence |
| 20 | Compute Momentum Coefficient | $\beta^{(k)} = \frac{t^{(k)} - 1}{t^{(k+1)}}$ | $t^{(k)}, t^{(k+1)}$ | $\beta^{(k)}$ | Extrapolation weight |
| 21 | Momentum Extrapolation | $\mathbf{y}^{(k+1)} = \mathbf{x}^{(k+1)} + \beta^{(k)}(\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)})$ | $\mathbf{x}^{(k+1)}, \mathbf{x}^{(k)}, \beta^{(k)}$ | $\mathbf{y}^{(k+1)}$ | Nesterov acceleration |
| 22 | END_LOOP | | | | |

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 23 | Extract Final Estimate | $\mathbf{x}^* = \mathbf{x}^{(n_{\text{iter}})}$ | $\mathbf{x}^{(n_{\text{iter}})}$ | $\mathbf{x}^*$ | Final iterate |
| 24 | Remove Batch Dimension | $\mathbf{x}^*_{\text{out}} = \begin{cases} \mathbf{x}^*[0,:,:,:] & \text{if } D = 1 \\ \mathbf{x}^* & \text{otherwise} \end{cases}$ | $\mathbf{x}^*$ | $\mathbf{x}^*_{\text{out}}$ : $(H, W, C)$ or $(D, H, W, C)$ | Squeeze singleton depth |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k < n_{\text{iter}}$
- **parameters**: 
  - $n_{\text{iter}} = 50$ (default, configurable)
- **check_frequency**: Evaluated at the beginning of each iteration

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Reconstruction Min | $\min_{i,j,k,c} x^*_{ijkc}$ | — | $\mathbf{x}^*_{\text{out}}$ | Minimum pixel value in reconstruction |
| Reconstruction Max | $\max_{i,j,k,c} x^*_{ijkc}$ | — | $\mathbf{x}^*_{\text{out}}$ | Maximum pixel value in reconstruction |
| Reconstruction Mean | $\frac{1}{N}\sum_{i,j,k,c} x^*_{ijkc}$ where $N = D \cdot H \cdot W \cdot C$ | — | $\mathbf{x}^*_{\text{out}}$ | Mean pixel value in reconstruction |