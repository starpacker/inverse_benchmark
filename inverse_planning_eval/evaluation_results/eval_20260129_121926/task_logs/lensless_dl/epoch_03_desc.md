## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: APGD (Accelerated Proximal Gradient Descent) / FISTA
- **algorithm_family**: First-order proximal methods, Nesterov-accelerated gradient descent
- **summary**: Reconstructs an image from a blurred measurement by solving a non-negatively constrained least-squares deconvolution problem using FFT-based convolution operators and momentum acceleration.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{x \in \mathbb{R}^{D \times H \times W \times C}} \frac{1}{2}\|Ax - y\|_2^2 \quad \text{subject to} \quad x \geq 0$$

### data_fidelity_term
$$f(x) = \frac{1}{2}\|Ax - y\|_2^2$$
where $A: \mathbb{R}^{D \times H \times W \times C} \to \mathbb{R}^{D \times H \times W \times C}$ is the convolution operator with PSF $h$, defined as $Ax = h \circledast x$ (2D convolution over spatial dimensions).

### regularization_term
None (only constraint-based regularization via non-negativity)

### constraints
- $x_{d,i,j,c} \geq 0 \quad \forall\, d \in [D], i \in [H], j \in [W], c \in [C]$ (element-wise non-negativity)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $x^{(k)}$ | primal | $(1, H, W, C)$ | float32 | $x^{(0)} = \mathbf{0}$ | Current image estimate at iteration $k$ |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $y^{(k)}$ | auxiliary | $(1, H, W, C)$ | float32 | $y^{(0)} = x^{(0)} = \mathbf{0}$ | Momentum-extrapolated point |
| $t^{(k)}$ | auxiliary | scalar | float64 | $t^{(0)} = 1.0$ | FISTA momentum coefficient |
| $\tilde{x}^{(k)}$ | auxiliary | $(1, H, W, C)$ | float32 | N/A | Pre-proximal gradient step result |
| $r^{(k)}$ | auxiliary | $(1, H, W, C)$ | float32 | N/A | Residual $Ay^{(k)} - y$ |
| $g^{(k)}$ | auxiliary | $(1, H, W, C)$ | float32 | N/A | Gradient $A^T r^{(k)}$ |

### Dual Variables
None

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $y$ | observations | $(1, H, W, C)$ | float32 | Loaded and preprocessed from file | Blurred measurement data |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $h$ | constants | $(1, H, W, C)$ | float32 | Loaded and preprocessed from file | Point spread function |
| $H$ | constants | $(1, H', \lfloor W'/2 \rfloor + 1, C)$ | complex64 | $H = \mathcal{F}_{2D}\{\text{pad}(h)\}$ | FFT of zero-padded PSF |
| $H^*$ | constants | $(1, H', \lfloor W'/2 \rfloor + 1, C)$ | complex64 | $H^* = \overline{H}$ (complex conjugate) | Adjoint operator FFT kernel |
| $L$ | constants | scalar | float32 | Power method (20 iterations) | Lipschitz constant of $\nabla f$ |
| $\alpha$ | constants | scalar | float32 | $\alpha = 1/L$ | Step size |
| $n_{\text{iter}}$ | constants | scalar | int | 5 (from main) / 50 (default) | Maximum iterations |
| $\text{downsample}$ | constants | scalar | int | 4 | Downsampling factor |
| $n_{\text{power}}$ | constants | scalar | int | 20 | Power method iterations |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Load PSF | $h_{\text{raw}} = \text{PIL.open}(\text{psf\_path})$ | psf_path (string) | $h_{\text{raw}} \in \mathbb{R}^{H_0 \times W_0 \times C_0}$ |
| 2 | Load Data | $y_{\text{raw}} = \text{PIL.open}(\text{data\_path})$ | data_path (string) | $y_{\text{raw}} \in \mathbb{R}^{H_0 \times W_0 \times C_0}$ |
| 3 | Downsample PSF | $h_{\text{ds}}[i,j,c] = h_{\text{raw}}[4i, 4j, c]$ | $h_{\text{raw}}$, factor=4 | $h_{\text{ds}} \in \mathbb{R}^{\lfloor H_0/4 \rfloor \times \lfloor W_0/4 \rfloor \times C}$ |
| 4 | Downsample Data | $y_{\text{ds}}[i,j,c] = y_{\text{raw}}[4i, 4j, c]$ | $y_{\text{raw}}$, factor=4 | $y_{\text{ds}} \in \mathbb{R}^{\lfloor H_0/4 \rfloor \times \lfloor W_0/4 \rfloor \times C}$ |
| 5 | Normalize PSF | $h_{\text{norm}} = \frac{h_{\text{ds}} - \min(h_{\text{ds}})}{\max(h_{\text{ds}}) - \min(h_{\text{ds}})}$ | $h_{\text{ds}}$ | $h_{\text{norm}} \in [0,1]^{H \times W \times C}$ |
| 6 | Normalize Data | $y_{\text{norm}} = \frac{y_{\text{ds}} - \min(y_{\text{ds}})}{\max(y_{\text{ds}}) - \min(y_{\text{ds}})}$ | $y_{\text{ds}}$ | $y_{\text{norm}} \in [0,1]^{H \times W \times C}$ |
| 7 | Reshape to 4D | $h = h_{\text{norm}}[\text{newaxis}, :, :, :]$, $y = y_{\text{norm}}[\text{newaxis}, :, :, :]$ | $h_{\text{norm}}$, $y_{\text{norm}}$ | $h, y \in \mathbb{R}^{1 \times H \times W \times C}$ |
| 8 | Compute Padded Shape | $H' = \text{next\_fast\_len}(2H - 1)$, $W' = \text{next\_fast\_len}(2W - 1)$ | $(H, W)$ | Padded spatial dimensions |
| 9 | Compute Start/End Indices | $s = \lfloor (H' - H, W' - W) / 2 \rfloor$, $e = s + (H, W)$ | Padded shape, original shape | Cropping indices |
| 10 | Zero-pad PSF | $h_{\text{pad}} \in \mathbb{R}^{1 \times H' \times W' \times C}$, $h_{\text{pad}}[:, s_0:e_0, s_1:e_1, :] = h$ | $h$, indices | $h_{\text{pad}}$ |
| 11 | Compute PSF FFT | $H = \mathcal{F}_{2D}^{\text{ortho}}(h_{\text{pad}})$ over axes $(-3, -2)$ | $h_{\text{pad}}$ | $H \in \mathbb{C}^{1 \times H' \times (\lfloor W'/2 \rfloor + 1) \times C}$ |
| 12 | Compute Adjoint Kernel | $H^* = \overline{H}$ (element-wise complex conjugate) | $H$ | $H^* \in \mathbb{C}^{1 \times H' \times (\lfloor W'/2 \rfloor + 1) \times C}$ |
| 13 | Estimate Lipschitz | Power method: see Section 4.1 | $H$, $H^*$, setup | $L \in \mathbb{R}^+$ |
| 14 | Compute Step Size | $\alpha = 1/L$ (or $1.0$ if $L = 0$) | $L$ | $\alpha \in \mathbb{R}^+$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Initialize primal | $x^{(0)} = \mathbf{0} \in \mathbb{R}^{1 \times H \times W \times C}$ | PSF shape | $x^{(0)}$ |
| I.2 | Initialize momentum point | $y^{(0)} = x^{(0)}$ | $x^{(0)}$ | $y^{(0)}$ |
| I.3 | Initialize momentum coeff | $t^{(0)} = 1.0$ | None | $t^{(0)}$ |

**Lipschitz Constant Estimation (Power Method)**

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | Initialize random vector | $v^{(0)} \sim \mathcal{N}(0, 1)^{1 \times H \times W \times C}$, $v^{(0)} \leftarrow v^{(0)} / \|v^{(0)}\|_2$ | PSF shape | $v^{(0)}$ with $\|v^{(0)}\|_2 = 1$ | |
| | **BEGIN_LOOP** ($j = 0, \ldots, 19$) | | | | |
| L.2 | Forward operator | $u^{(j)} = A v^{(j)}$ (FFT convolution) | $v^{(j)}$, $H$ | $u^{(j)}$ | |
| L.3 | Adjoint operator | $w^{(j)} = A^T u^{(j)}$ (FFT correlation) | $u^{(j)}$, $H^*$ | $w^{(j)}$ | |
| L.4 | Compute norm | $\sigma^{(j)} = \|w^{(j)}\|_2$ | $w^{(j)}$ | $\sigma^{(j)}$ | |
| L.5 | Normalize | $v^{(j+1)} = w^{(j)} / \sigma^{(j)}$ (if $\sigma^{(j)} > 0$) | $w^{(j)}$, $\sigma^{(j)}$ | $v^{(j+1)}$ | |
| | **END_LOOP** | | | | |
| L.6 | Output Lipschitz | $L = \sigma^{(19)}$ | Final $\sigma$ | $L$ | Spectral radius of $A^T A$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** ($k = 0, 1, \ldots, n_{\text{iter}} - 1$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 1 | Forward model | $Ay^{(k)} = \mathcal{F}_{2D}^{-1}\left[\mathcal{F}_{2D}(\text{pad}(y^{(k)})) \odot H\right]_{\text{crop}}$ | $y^{(k)}$, $H$, setup | $Ay^{(k)} \in \mathbb{R}^{1 \times H \times W \times C}$ | FFT convolution with ortho normalization, ifftshift applied |
| 2 | Compute residual | $r^{(k)} = Ay^{(k)} - y$ | $Ay^{(k)}$, $y$ | $r^{(k)} \in \mathbb{R}^{1 \times H \times W \times C}$ | Element-wise subtraction |
| 3 | Adjoint operator (gradient) | $g^{(k)} = A^T r^{(k)} = \mathcal{F}_{2D}^{-1}\left[\mathcal{F}_{2D}(\text{pad}(r^{(k)})) \odot H^*\right]_{\text{crop}}$ | $r^{(k)}$, $H^*$, setup | $g^{(k)} \in \mathbb{R}^{1 \times H \times W \times C}$ | FFT correlation (adjoint convolution) |
| 4 | Gradient descent step | $\tilde{x}^{(k)} = y^{(k)} - \alpha \cdot g^{(k)}$ | $y^{(k)}$, $g^{(k)}$, $\alpha$ | $\tilde{x}^{(k)} \in \mathbb{R}^{1 \times H \times W \times C}$ | $\alpha = 1/L$ |
| 5 | Proximal operator | $x^{(k+1)} = \text{prox}_{\mathcal{I}_{\geq 0}}(\tilde{x}^{(k)}) = \max(\tilde{x}^{(k)}, 0)$ | $\tilde{x}^{(k)}$ | $x^{(k+1)} \in \mathbb{R}_{\geq 0}^{1 \times H \times W \times C}$ | Element-wise maximum with 0 |
| 6 | Update momentum coeff | $t^{(k+1)} = \frac{1 + \sqrt{1 + 4(t^{(k)})^2}}{2}$ | $t^{(k)}$ | $t^{(k+1)} \in \mathbb{R}^+$ | FISTA momentum schedule |
| 7 | Momentum extrapolation | $y^{(k+1)} = x^{(k+1)} + \frac{t^{(k)} - 1}{t^{(k+1)}} \left(x^{(k+1)} - x^{(k)}\right)$ | $x^{(k+1)}$, $x^{(k)}$, $t^{(k)}$, $t^{(k+1)}$ | $y^{(k+1)} \in \mathbb{R}^{1 \times H \times W \times C}$ | Nesterov acceleration |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Extract result | $x^* = x^{(n_{\text{iter}})}$ | Final $x^{(k)}$ | $x^* \in \mathbb{R}^{1 \times H \times W \times C}$ |
| F.2 | Remove batch dim | If shape[0] = 1: $x^*_{\text{out}} = x^*[0]$ | $x^*$ | $x^*_{\text{out}} \in \mathbb{R}^{H \times W \times C}$ |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k \geq n_{\text{iter}}$
- **parameters**:
  - $n_{\text{iter}} = 5$ (as called in main block)
  - Default $n_{\text{iter}} = 50$ (function signature)
- **check_frequency**: Every iteration (end of loop body)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Reconstruction Min | $\min_{d,i,j,c} x^*_{d,i,j,c}$ | Minimum pixel value in reconstruction |
| Reconstruction Max | $\max_{d,i,j,c} x^*_{d,i,j,c}$ | Maximum pixel value in reconstruction |
| Reconstruction Mean | $\frac{1}{D \cdot H \cdot W \cdot C} \sum_{d,i,j,c} x^*_{d,i,j,c}$ | Mean pixel value in reconstruction |
| Normalized Display | $\tilde{x}^* = \text{clip}\left(\frac{x^* - \min(x^*)}{\max(x^*) - \min(x^*)}, 0, 1\right)$ | Normalized image for visualization |

---

## APPENDIX: Operator Definitions

### FFT-Based Convolution Operator $A$
$$[Ax](\mathbf{n}) = \mathcal{F}_{2D}^{-1}\left\{\mathcal{F}_{2D}\{\text{pad}(x)\} \odot H \right\}_{\text{crop}}(\mathbf{n})$$

where:
- $\mathcal{F}_{2D}^{\text{ortho}}$ denotes 2D rFFT with orthonormal normalization
- $\text{pad}(\cdot)$: zero-pad to shape $(1, H', W', C)$ centered at indices $(s_0:e_0, s_1:e_1)$
- $\text{crop}(\cdot)$: extract central region at indices $(s_0:e_0, s_1:e_1)$
- $\odot$: element-wise (Hadamard) product
- $\mathcal{F}_{2D}^{-1}$ includes `ifftshift` over spatial axes

### FFT-Based Adjoint Operator $A^T$
$$[A^T y](\mathbf{n}) = \mathcal{F}_{2D}^{-1}\left\{\mathcal{F}_{2D}\{\text{pad}(y)\} \odot H^* \right\}_{\text{crop}}(\mathbf{n})$$

where $H^* = \overline{H}$ is the element-wise complex conjugate.

### Proximal Operator for Non-negativity Indicator
$$\text{prox}_{\mathcal{I}_{\geq 0}}(z) = \arg\min_{x \geq 0} \frac{1}{2}\|x - z\|_2^2 = \max(z, 0)$$