

## 0. ALGORITHM IDENTIFICATION

```json
{
  "algorithm_name": "Gravitational Lens Source Reconstruction via Adam with TV Regularization",
  "algorithm_family": "First-Order Gradient-Based Optimization with Automatic Differentiation",
  "summary": "Reconstructs a source galaxy image from a gravitationally lensed observation by minimizing L2 data fidelity plus anisotropic Total Variation regularization using the Adam optimizer with non-negativity projection."
}
```

---

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{x \in \mathbb{R}^{N_s \times N_s}} \frac{1}{N_o^2} \| \mathcal{A}(x) - y \|_2^2 + \lambda_{\text{TV}} \left( \frac{1}{N_s(N_s-1)} \sum_{i,j} |x_{i,j+1} - x_{i,j}| + \frac{1}{(N_s-1)N_s} \sum_{i,j} |x_{i+1,j} - x_{i,j}| \right) + \iota_{\geq 0}(x)$$

where $\mathcal{A}: \mathbb{R}^{N_s \times N_s} \to \mathbb{R}^{N_o \times N_o}$ is the gravitational lensing forward operator.

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}}(x) = \frac{1}{N_o^2} \sum_{i=1}^{N_o} \sum_{j=1}^{N_o} \left( [\mathcal{A}(x)]_{i,j} - y_{i,j} \right)^2$$

This is an **L2 (squared) loss** verified from code: `torch.mean((reconstruction - observation)**2)`.

**regularization_term**:
$$\mathcal{R}_{\text{TV}}(x) = \lambda_{\text{TV}} \left( \underbrace{\frac{1}{N_s(N_s-1)} \sum_{i=1}^{N_s} \sum_{j=1}^{N_s-1} |x_{i,j+1} - x_{i,j}|}_{\text{horizontal TV}} + \underbrace{\frac{1}{(N_s-1)N_s} \sum_{i=1}^{N_s-1} \sum_{j=1}^{N_s} |x_{i+1,j} - x_{i,j}|}_{\text{vertical TV}} \right)$$

**constraints**:
$$\iota_{\geq 0}(x) = \begin{cases} 0 & \text{if } x_{i,j} \geq 0 \; \forall i,j \\ +\infty & \text{otherwise} \end{cases}$$

Enforced via projection: `model_source_image.clamp_(min=0)`.

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x^{(k)}$ | $(128, 128)$ | `torch.float32` | $x^{(0)} = \mathbf{0}_{128 \times 128}$ | Reconstructed source image at iteration $k$ |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $m^{(k)}$ | $(128, 128)$ | `torch.float32` | $m^{(0)} = \mathbf{0}_{128 \times 128}$ | Adam first moment estimate |
| $v^{(k)}$ | $(128, 128)$ | `torch.float32` | $v^{(0)} = \mathbf{0}_{128 \times 128}$ | Adam second moment estimate |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\hat{x}^{(k)}$ | $(128, 128)$ | `torch.float32` | Computed | Source after Adam update, before projection |
| $g^{(k)}$ | $(128, 128)$ | `torch.float32` | Computed | Gradient of total loss w.r.t. $x$ |
| $\hat{m}^{(k)}$ | $(128, 128)$ | `torch.float32` | Computed | Bias-corrected first moment |
| $\hat{v}^{(k)}$ | $(128, 128)$ | `torch.float32` | Computed | Bias-corrected second moment |
| $\tilde{y}^{(k)}$ | $(128, 128)$ | `torch.float32` | Computed | Predicted observation from forward model |
| $\mu_{\text{fine}}$ | $(N_{\text{up}}, N_{\text{up}}, 1)$ | `torch.float32` | Computed | Fine-grid source brightness before quadrature |
| $\mu$ | $(N_{\text{up}}, N_{\text{up}})$ | `torch.float32` | Computed | Integrated brightness on upsampled grid |
| $\mu_{\text{fft}}$ | $(S_0, S_1//2+1)$ | `torch.complex64` | Computed | FFT of brightness map |
| $\text{psf}_{\text{fft}}$ | $(S_0, S_1//2+1)$ | `torch.complex64` | Computed | FFT of normalized PSF |
| $\mu_{\text{conv}}$ | $(S_0, S_1)$ | `torch.float32` | Computed | Convolved brightness (real part of IFFT) |
| $\mu_{\text{clip}}$ | $(256, 256)$ | `torch.float32` | Computed | Clipped to valid region |
| $\mathcal{L}_{\text{data}}$ | scalar | `torch.float32` | Computed | Data fidelity loss |
| $\mathcal{L}_{\text{TV}}$ | scalar | `torch.float32` | Computed | TV regularization loss |
| $\mathcal{L}_{\text{total}}$ | scalar | `torch.float32` | Computed | Total loss |
| $\alpha^{(k)}$ | scalar | `torch.float32` | Computed | Effective learning rate after scheduler |

### 2.4 Constants

| Symbol | Value | Dtype | Description |
|--------|-------|-------|-------------|
| $N_s$ | $128$ | `int` | Source image dimension |
| $N_o$ | $128$ | `int` | Observation image dimension |
| $N_{\text{up}}$ | $2$ | `int` | Upsample factor |
| $\Delta_s$ | $0.04$ | `float32` | Source pixelscale (arcsec/pixel) |
| $\Delta_o$ | $0.05$ | `float32` | Observation pixelscale (arcsec/pixel) |
| $z_l$ | $0.5$ | `float32` | Lens redshift |
| $z_s$ | $1.0$ | `float32` | Source redshift |
| $x_0^{\text{lens}}$ | $0.0$ | `float32` | Lens center x (arcsec) |
| $y_0^{\text{lens}}$ | $0.0$ | `float32` | Lens center y (arcsec) |
| $q$ | $0.7$ | `float32` | Lens axis ratio |
| $\phi$ | $0.5$ | `float32` | Lens position angle (radians) |
| $R_{\text{Ein}}$ | $1.5$ | `float32` | Einstein radius (arcsec) |
| $s$ | $0.0$ | `float32` | Lens core radius (softening) |
| $x_0^{\text{src}}$ | $0.0$ | `float32` | Source center x (arcsec) |
| $y_0^{\text{src}}$ | $0.0$ | `float32` | Source center y (arcsec) |
| $\lambda_{\text{TV}}$ | $10^{-4}$ | `float32` | TV regularization weight |
| $\alpha_0$ | $0.1$ | `float32` | Initial learning rate |
| $\beta_1$ | $0.9$ | `float32` | Adam first moment decay |
| $\beta_2$ | $0.999$ | `float32` | Adam second moment decay |
| $\epsilon$ | $10^{-8}$ | `float32` | Adam numerical stability constant |
| $K_{\text{max}}$ | $100$ | `int` | Maximum iterations |
| $K_{\text{step}}$ | $500$ | `int` | Learning rate scheduler step size |
| $\gamma_{\text{lr}}$ | $0.5$ | `float32` | Learning rate decay factor |
| $h_0$ | $0.6766$ | `float32` | Hubble constant (dimensionless) |
| $\Omega_{m,0}$ | $0.30966$ | `float32` | Matter density parameter |
| $c_{\text{Mpc/s}}$ | $9.71561189 \times 10^{-15}$ | `float64` | Speed of light in Mpc/s |
| $\text{km\_to\_Mpc}$ | $3.2407792896664 \times 10^{-20}$ | `float64` | Conversion factor |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y$ | $(128, 128)$ | `torch.float32` | Observed lensed image (noisy) |

---

## 3. PREPROCESSING PIPELINE

```json
{
  "data_preprocessing": [
    {
      "step_order": 1,
      "step_name": "Load source image",
      "formula": "$I_{\\text{raw}} = \\text{np.load}(\\text{path})$",
      "inputs": {"path": "string"},
      "outputs": {"I_raw": "(H_{\\text{raw}}, W_{\\text{raw}})"},
      "assumptions": "File exists at specified path; otherwise synthetic source created"
    },
    {
      "step_order": 2,
      "step_name":