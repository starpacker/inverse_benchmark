## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) / APGD",
  "algorithm_family": "Accelerated Proximal Gradient Descent",
  "summary": "FFT-based image deconvolution with non-negativity constraint using Nesterov momentum acceleration."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\mathbf{x} \\geq 0} \\frac{1}{2} \\|\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\|_2^2",
  "data_fidelity_term": "f(\\mathbf{x}) = \\frac{1}{2} \\|\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\|_2^2 \\quad \\text{(L2 squared norm)}",
  "regularization_term": "g(\\mathbf{x}) = \\iota_{\\geq 0}(\\mathbf{x}) = \\begin{cases} 0 & \\text{if } x_{ijkl} \\geq 0 \\ \\forall i,j,k,l \\\\ +\\infty & \\text{otherwise} \\end{cases}",
  "constraints": "\\mathbf{x} \\geq 0 \\quad \\text{(element-wise non-negativity, enforced via indicator function)}"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}^{(k)}$ | $(1, H, W, C)$ | float32 | $\mathbf{0}$ | Current image estimate at iteration $k$ |

### 2.2 Dual Variables
None (primal-only method)

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}^{(k)}$ | $(1, H, W, C)$ | float32 | $\mathbf{x}^{(0)}$ | Momentum-extrapolated point |
| $t^{(k)}$ | scalar | float64 | $1.0$ | FISTA momentum parameter |
| $\mathbf{H}$ | $(1, P_H, \lfloor P_W/2 \rfloor + 1, C)$ | complex64 | Computed from PSF | FFT of zero-padded PSF |
| $\mathbf{H}^*$ | $(1, P_H, \lfloor P_W/2 \rfloor + 1, C)$ | complex64 | $\overline{\mathbf{H}}$ | Complex conjugate of $\mathbf{H}$ (adjoint kernel) |
| $L$ | scalar | float32 | Power method estimate | Lipschitz constant of $\nabla f$ |
| $\mathbf{r}^{(k)}$ | $(1, H, W, C)$ | float32 | — | Residual $\mathbf{A}\mathbf{y}^{(k)} - \mathbf{b}$ |
| $\nabla f(\mathbf{y}^{(k)})$ | $(1, H, W, C)$ | float32 | — | Gradient at momentum point |
| $\tilde{\mathbf{x}}^{(k+1)}$ | $(1, H, W, C)$ | float32 | — | Pre-proximal gradient step result |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $n_{\text{iter}}$ | 50 (default), 5 (in main) | Maximum number of iterations |
| $n_{\text{power}}$ | 20 | Power method iterations for Lipschitz estimation |
| $d$ | 4 | Downsampling factor |
| FFT norm | `"ortho"` | Orthonormal FFT normalization ($1/\sqrt{N}$) |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{b}$ | $(1, H, W, C)$ | float32 | Preprocessed measurement data |
| $\mathbf{h}$ | $(1, H, W, C)$ | float32 | Preprocessed PSF kernel |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": "P1",
      "step_name": "Load Raw Images",
      "formula": "\\mathbf{h}_{\\text{raw}} = \\text{load}(\\text{psf\\_path}), \\quad \\mathbf{b}_{\\text{raw}} = \\text{load}(\\text{data\\_path})",
      "inputs": ["psf_path (string)", "data_path (string)"],
      "outputs": ["$\\mathbf{h}_{\\text{raw}}$ (H_raw, W_raw) or (H_raw, W_raw, C)", "$\\mathbf{b}_{\\text{raw}}$ (H_raw, W_raw) or (H_raw, W_raw, C)"],
      "assumptions": "Images are valid PNG files loadable by PIL"
    },
    {
      "step_order": "P2",
      "step_name": "Downsample Images",
      "formula": "\\mathbf{h}_{\\text{ds}}[i,j] = \\mathbf{h}_{\\text{raw}}[d \\cdot i, d \\cdot j], \\quad \\mathbf{b}_{\\text{ds}}[i,j] = \\mathbf{b}_{\\text{raw}}[d \\cdot i, d \\cdot j] \\quad \\text{where } d=4",
      "inputs": ["$\\mathbf{h}_{\\text{raw}}$", "$\\mathbf{b}_{\\text{raw}}$", "$d=4$"],
      "outputs": ["$\\mathbf{h}_{\\text{ds}}$ (H_raw/d, W_raw/d, C)", "$\\mathbf{b}_{\\text{ds}}$ (H_raw/d, W_raw/d, C)"],
      "assumptions": "Slicing with stride d"
    },
    {
      "step_order": "P3",
      "step_name": "Normalize to [0,1]",
      "formula": "\\mathbf{h}_{\\text{norm}} = \\frac{\\mathbf{h}_{\\text{ds}} - \\min(\\mathbf{h}_{\\text{ds}})}{\\max(\\mathbf{h}_{\\text{ds}}) - \\min(\\mathbf{h}_{\\text{ds}})}, \\quad \\mathbf{b}_{\\text{norm}} = \\frac{\\mathbf{b}_{\\text{ds}} - \\min(\\mathbf{b}_{\\text{ds}})}{\\max(\\mathbf{b}_{\\text{ds}}) - \\min(\\mathbf{b}_{\\text{ds}})}",
      "inputs": ["$\\mathbf{h}_{\\text{ds}}$", "$\\mathbf{b}_{\\text{ds}}$"],
      "outputs": ["$\\mathbf{h}_{\\text{norm}} \\in [0,1]$", "$\\mathbf{b}_{\\text{norm}} \\in [0,1]$"],
      "assumptions": "max > min; otherwise return unchanged"
    },
    {
      "step_order": "P4",
      "step_name": "Reshape to 4D",
      "formula": "\\mathbf{h} = \\mathbf{h}_{\\text{norm}}[\\text{newaxis}, :, :, :], \\quad \\mathbf{b} = \\mathbf{b}_{\\text{norm}}[\\text{newaxis}, :, :, :]",
      "inputs": ["$\\mathbf{h}_{\\text{norm}}$ (H, W, C)", "$\\mathbf{b}_{\\text{norm}}$ (H, W, C)"],
      "outputs": ["$\\mathbf{h}$ (1, H, W, C)", "$\\mathbf{b}$ (1, H, W, C)"],
      "assumptions": "If 2D input, first expand to 3D by adding channel axis"
    },
    {
      "step_order": "P5",
      "step_name": "Compute Padded Shape",
      "formula": "P_H = \\text{next\\_fast\\_len}(2H - 1), \\quad P_W = \\text{next\\_fast\\_len}(2W - 1)",
      "inputs": ["$H$ (PSF height)", "$W$ (PSF width)"],
      "outputs": ["$P_H$ (padded height)", "$P_W$ (padded width)"],
      "assumptions": "next_fast_len returns optimal FFT size $\\geq$ input"
    },
    {
      "step_order": "P6",
      "step_name": "Compute Padding Indices",
      "formula": "s_H = \\lfloor(P_H - H)/2\\rfloor, \\quad s_W = \\lfloor(P_W - W)/2\\rfloor, \\quad e_H = s_H + H, \\quad e_W = s_W + W",
      "inputs": ["$P_H$", "$P_W$", "$H$", "$W$"],
      "outputs": ["$s_H, s_W$ (start indices)", "$e_H, e_W$ (end indices)"],
      "assumptions": "Integer division for centering"
    },
    {
      "step_order": "P7",
      "step_name": "Compute PSF FFT",
      "formula": "\\mathbf{h}_{\\text{pad}} = \\text{ZeroPad}(\\mathbf{h}) \\in \\mathbb{R}^{1 \\times P_H \\times P_W \\times C}, \\quad \\mathbf{H} = \\mathcal{F}_{2D}^{\\text{ortho}}(\\mathbf{h}_{\\text{pad}})",
      "inputs": ["$\\mathbf{h}$ (1, H, W, C)", "padded_shape $(1, P_H, P_W, C)$"],
      "outputs": ["$\\mathbf{H}$ (1, P_H, P_W//2+1, C) complex64"],
      "assumptions": "rfft2 along axes (-3, -2) with ortho normalization"
    },
    {
      "step_order": "P8",
      "step_name": "Compute Adjoint Kernel",
      "formula": "\\mathbf{H}^* = \\overline{\\mathbf{H}} \\quad \\text{(element-wise complex conjugate)}",
      "inputs": ["$\\mathbf{H}$"],
      "outputs": ["$\\mathbf{H}^*$ (1, P_H, P_W//2+1, C) complex64"],
      "assumptions": "Adjoint of convolution is correlation"
    },
    {
      "step_order": "P9",
      "step_name": "Estimate Lipschitz Constant via Power Method",
      "formula": "\\text{Initialize: } \\mathbf{v}^{(0)} \\sim \\mathcal{N}(0,1), \\quad \\mathbf{v}^{(0)} \\leftarrow \\mathbf{v}^{(0)} / \\|\\mathbf{v}^{(0)}\\|_2",
      "inputs": ["PSF shape (1, H, W, C)"],
      "outputs": ["$\\mathbf{v}^{(0)}$ (1, H, W, C) normalized"],
      "assumptions": "Standard Gaussian initialization"
    },
    {
      "step_order": "P10",
      "step_name": "Power Method Loop (20 iterations)",
      "formula": "\\text{For } j = 0, \\ldots, 19: \\quad \\mathbf{v}^{(j+1)} = \\mathbf{A}^T \\mathbf{A} \\mathbf{v}^{(j)}, \\quad L_j = \\|\\mathbf{v}^{(j+1)}\\|_2, \\quad \\mathbf{v}^{(j+1)} \\leftarrow \\mathbf{v}^{(j+1)} / L_j",
      "inputs": ["$\\mathbf{v}^{(j)}$", "$\\mathbf{H}$", "$\\mathbf{H}^*$"],
      "outputs": ["$L = L_{19}$ (Lipschitz constant estimate)"],
      "assumptions": "Converges to largest singular value squared of A"
    }
  ]
}
```

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION
```json
{
  "steps": [
    {
      "step_order": 1,
      "variable_name": "$\\mathbf{x}^{(0)}$",
      "initial_value": "\\mathbf{0} \\in \\mathbb{R}^{1 \\times H \\times W \\times C}",
      "shape": "(1, H, W, C)",
      "dtype": "float32",
      "source": "np.zeros_like(psf)"
    },
    {
      "step_order": 2,
      "variable_name": "$\\mathbf{y}^{(0)}$",
      "initial_value": "\\mathbf{x}^{(0)}",
      "shape": "(1, H, W, C)",
      "dtype": "float32",
      "source": "x_k.copy()"
    },
    {
      "step_order": 3,
      "variable_name": "$t^{(0)}$",
      "initial_value": "1.0",
      "shape": "scalar",
      "dtype": "float64",
      "source": "t_k = 1.0"
    },
    {
      "step_order": 4,
      "variable_name": "$\\alpha$",
      "initial_value": "\\alpha = \\frac{1}{L} \\quad \\text{(if } L > 0 \\text{, else } 1.0\\text{)}",
      "shape": "scalar",
      "dtype": "float32",
      "source": "step_size = 1.0 / L"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "loop_structure": "for k = 0, 1, ..., n_iter - 1",
  "steps": [
    {
      "marker": "BEGIN_LOOP k = 0 to n_iter - 1"
    },
    {
      "step_order": 5,
      "step_name": "[FORWARD] Pad Momentum Point",
      "formula": "\\mathbf{y}_{\\text{pad}}^{(k)} = \\text{ZeroPad}(\\mathbf{y}^{(k)}) \\quad \\text{where } \\mathbf{y}_{\\text{pad}}^{(k)}[..., s_H:e_H, s_W:e_W, :] = \\mathbf{y}^{(k)}",
      "inputs": ["$\\mathbf{y}^{(k)}$ (1, H, W, C)"],
      "outputs": ["$\\mathbf{y}_{\\text{pad}}^{(k)}$ (1, P_H, P_W, C)"],
      "computational_note": "Zero-initialized array with centered insertion"
    },
    {
      "step_order": 6,
      "step_name": "[FORWARD] FFT of Padded Momentum",
      "formula": "\\hat{\\mathbf{y}}^{(k)} = \\mathcal{F}_{2D}^{\\text{ortho}}(\\mathbf{y}_{\\text{pad}}^{(k)}) = \\frac{1}{\\sqrt{P_H \\cdot P_W}} \\sum_{m,n} \\mathbf{y}_{\\text{pad}}^{(k)}[m,n] e^{-2\\pi i (\\frac{um}{P_H} + \\frac{vn}{P_W})}",
      "inputs": ["$\\mathbf{y}_{\\text{pad}}^{(k)}$ (1, P_H, P_W, C)"],
      "outputs": ["$\\hat{\\mathbf{y}}^{(k)}$ (1, P_H, P_W//2+1, C) complex64"],
      "computational_note": "rfft2 with axes=(-3,-2), norm='ortho'"
    },
    {
      "step_order": 7,
      "step_name": "[FORWARD] Frequency-Domain Multiplication",
      "formula": "\\hat{\\mathbf{z}}^{(k)} = \\hat{\\mathbf{y}}^{(k)} \\odot \\mathbf{H}",
      "inputs": ["$\\hat{\\mathbf{y}}^{(k)}$ (1, P_H, P_W//2+1, C)", "$\\mathbf{H}$ (1, P_H, P_W//2+1, C)"],
      "outputs": ["$\\hat{\\mathbf{z}}^{(k)}$ (1, P_H, P_W//2+1, C) complex64"],
      "computational_note": "Element-wise complex multiplication"
    },
    {
      "step_order": 8,
      "step_name": "[FORWARD] Inverse FFT",
      "formula": "\\mathbf{z}_{\\text{shift}}^{(k)} = \\mathcal{F}_{2D}^{-1, \\text{ortho}}(\\hat{\\mathbf{z}}^{(k)})",
      "inputs": ["$\\hat{\\mathbf{z}}^{(k)}$ (1, P_H, P_W//2+1, C)"],
      "outputs": ["$\\mathbf{z}_{\\text{shift}}^{(k)}$ (1, P_H, P_W, C)"],
      "computational_note": "irfft2 with s=(P_H, P_W), axes=(-3,-2), norm='ortho'"
    },
    {
      "step_order": 9,
      "step_name": "[FORWARD] Inverse FFT Shift",
      "formula": "(\\mathbf{A}\\mathbf{y}^{(k)})_{\\text{pad}} = \\text{ifftshift}(\\mathbf{z}_{\\text{shift}}^{(k)}, \\text{axes}=(-3,-2))",
      "inputs": ["$\\mathbf{z}_{\\text{shift}}^{(k)}$ (1, P_H, P_W, C)"],
      "outputs": ["$(\\mathbf{A}\\mathbf{y}^{(k)})_{\\text{pad}}$ (1, P_H, P_W, C)"],
      "computational_note": "Shift zero-frequency to corner"
    },
    {
      "step_order": 10,
      "step_name": "[FORWARD] Crop Forward Result",
      "formula": "\\mathbf{A}\\mathbf{y}^{(k)} = \\text{Re}\\left((\\mathbf{A}\\mathbf{y}^{(k)})_{\\text{pad}}[..., s_H:e_H, s_W:e_W, :]\\right)",
      "inputs": ["$(\\mathbf{A}\\mathbf{y}^{(k)})_{\\text{pad}}$ (1, P_H, P_W, C)"],
      "outputs": ["$\\mathbf{A}\\mathbf{y}^{(k)}$ (1, H, W, C)"],
      "computational_note": "Extract real part and crop to original size"
    },
    {
      "step_order": 11,
      "step_name": "Compute Residual",
      "formula": "\\mathbf{r}^{(k)} = \\mathbf{A}\\mathbf{y}^{(k)} - \\mathbf{b}",
      "inputs": ["$\\mathbf{A}\\mathbf{y}^{(k)}$ (1, H, W, C)", "$\\mathbf{b}$ (1, H, W, C)"],
      "outputs": ["$\\mathbf{r}^{(k)}$ (1, H, W, C)"],
      "computational_note": "Element-wise subtraction"
    },
    {
      "step_order": 12,
      "step_name": "[ADJOINT] Pad Residual",
      "formula": "\\mathbf{r}_{\\text{pad}}^{(k)} = \\text{ZeroPad}(\\mathbf{r}^{(k)})",
      "inputs": ["$\\mathbf{r}^{(k)}$ (1, H, W, C)"],
      "outputs": ["$\\mathbf{r}_{\\text{pad}}^{(k)}$ (1, P_H, P_W, C)"],
      "computational_note": "Same padding as forward operation"
    },
    {
      "step_order": 13,
      "step_name": "[ADJOINT] FFT of Padded Residual",
      "formula": "\\hat{\\mathbf{r}}^{(k)} = \\mathcal{F}_{2D}^{\\text{ortho}}(\\mathbf{r}_{\\text{pad}}^{(k)})",
      "inputs": ["$\\mathbf{r}_{\\text{pad}}^{(k)}$ (1, P_H, P_W, C)"],
      "outputs": ["$\\hat{\\mathbf{r}}^{(k)}$ (1, P_H, P_W//2+1, C) complex64"],
      "computational_note": "rfft2 with axes=(-3,-2), norm='ortho'"
    },
    {
      "step_order": 14,
      "step_name": "[ADJOINT] Frequency-Domain Multiplication with Conjugate",
      "formula": "\\hat{\\mathbf{g}}^{(k)} = \\hat{\\mathbf{r}}^{(k)} \\odot \\mathbf{H}^*",
      "inputs": ["$\\hat{\\mathbf{r}}^{(k)}$ (1, P_H, P_W//2+1, C)", "$\\mathbf{H}^*$ (1, P_H, P_W//2+1, C)"],
      "outputs": ["$\\hat{\\mathbf{g}}^{(k)}$ (1, P_H, P_W//2+1, C) complex64"],
      "computational_note": "Element-wise multiplication with conjugate kernel"
    },
    {
      "step_order": 15,
      "step_name": "[ADJOINT] Inverse FFT",
      "formula": "\\mathbf{g}_{\\text{shift}}^{(k)} = \\mathcal{F}_{2D}^{-1, \\text{ortho}}(\\hat{\\mathbf{g}}^{(k)})",
      "inputs": ["$\\hat{\\mathbf{g}}^{(k)}$ (1, P_H, P_W//2+1, C)"],
      "outputs": ["$\\mathbf{g}_{\\text{shift}}^{(k)}$ (1, P_H, P_W, C)"],
      "computational_note": "irfft2 with s=(P_H, P_W), axes=(-3,-2), norm='ortho'"
    },
    {
      "step_order": 16,
      "step_name": "[ADJOINT] Inverse FFT Shift",
      "formula": "(\\nabla f)_{\\text{pad}}^{(k)} = \\text{ifftshift}(\\mathbf{g}_{\\text{shift}}^{(k)}, \\text{axes}=(-3,-2))",
      "inputs": ["$\\mathbf{g}_{\\text{shift}}^{(k)}$ (1, P_H, P_W, C)"],
      "outputs": ["$(\\nabla f)_{\\text{pad}}^{(k)}$ (1, P_H, P_W, C)"],
      "computational_note": "Shift zero-frequency to corner"
    },
    {
      "step_order": 17,
      "step_name": "[ADJOINT] Crop Gradient",
      "formula": "\\nabla f(\\mathbf{y}^{(k)}) = \\mathbf{A}^T \\mathbf{r}^{(k)} = \\text{Re}\\left((\\nabla f)_{\\text{pad}}^{(k)}[..., s_H:e_H, s_W:e_W, :]\\right)",
      "inputs": ["$(\\nabla f)_{\\text{pad}}^{(k)}$ (1, P_H, P_W, C)"],
      "outputs": ["$\\nabla f(\\mathbf{y}^{(k)})$ (1, H, W, C)"],
      "computational_note": "Gradient is $\\mathbf{A}^T(\\mathbf{A}\\mathbf{y}^{(k)} - \\mathbf{b})$"
    },
    {
      "step_order": 18,
      "step_name": "Gradient Descent Step",
      "formula": "\\tilde{\\mathbf{x}}^{(k+1)} = \\mathbf{y}^{(k)} - \\alpha \\cdot \\nabla f(\\mathbf{y}^{(k)}) = \\mathbf{y}^{(k)} - \\frac{1}{L} \\mathbf{A}^T(\\mathbf{A}\\mathbf{y}^{(k)} - \\mathbf{b})",
      "inputs": ["$\\mathbf{y}^{(k)}$ (1, H, W, C)", "$\\nabla f(\\mathbf{y}^{(k)})$ (1, H, W, C)", "$\\alpha = 1/L$"],
      "outputs": ["$\\tilde{\\mathbf{x}}^{(k+1)}$ (1, H, W, C)"],
      "computational_note": "Element-wise: y_k - step_size * gradient"
    },
    {
      "step_order": 19,
      "step_name": "Proximal Operator (Non-negativity Projection)",
      "formula": "\\mathbf{x}^{(k+1)} = \\text{prox}_{\\alpha g}(\\tilde{\\mathbf{x}}^{(k+1)}) = \\max(\\tilde{\\mathbf{x}}^{(k+1)}, 0) = \\text{proj}_{\\geq 0}(\\tilde{\\mathbf{x}}^{(k+1)})",
      "inputs": ["$\\tilde{\\mathbf{x}}^{(k+1)}$ (1, H, W, C)"],
      "outputs": ["$\\mathbf{x}^{(k+1)}$ (1, H, W, C)"],
      "computational_note": "Element-wise maximum with zero; closed-form for indicator of non-negative orthant"
    },
    {
      "step_order": 20,
      "step_name": "Update Momentum Parameter",
      "formula": "t^{(k+1)} = \\frac{1 + \\sqrt{1 + 4 (t^{(k)})^2}}{2}",
      "inputs": ["$t^{(k)}$ (scalar)"],
      "outputs": ["$t^{(k+1)}$ (scalar)"],
      "computational_note": "FISTA momentum schedule"
    },
    {
      "step_order": 21,
      "step_name": "Compute Momentum Coefficient",
      "formula": "\\beta^{(k)} = \\frac{t^{(k)} - 1}{t^{(k+1)