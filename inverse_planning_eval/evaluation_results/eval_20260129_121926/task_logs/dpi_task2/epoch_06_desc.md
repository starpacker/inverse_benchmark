## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Deep Probabilistic Imaging for MRI Reconstruction",
  "algorithm_family": "Variational Inference with Normalizing Flows",
  "summary": "Learns a RealNVP generative model to approximate the posterior distribution of MRI images given undersampled k-space measurements by minimizing a variational objective combining data fidelity, regularization, and entropy maximization terms."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\theta, s} \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ \mathcal{L}_{\text{data}}(G_{\theta,s}(z)) + \mathcal{L}_{\text{prior}}(G_{\theta,s}(z)) - w_{\text{logdet}} \log \left| \det \frac{\partial G_{\theta,s}}{\partial z} \right| \right]$$

where $G_{\theta,s}(z) = \text{Softplus}(f_\theta(z)) \cdot e^{s}$ is the image generator composed of RealNVP flow $f_\theta$ followed by Softplus activation and learned scaling.

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}}(x) = \frac{1}{\sigma^2 \cdot \bar{M}} \cdot \frac{1}{B \cdot N \cdot N \cdot 2} \sum_{b,i,j,c} \left( M_{i,j,c} \cdot (\mathcal{F}(x_b))_{i,j,c} - M_{i,j,c} \cdot y_{i,j,c} \right)^2$$

where $\mathcal{F}$ is the orthonormalized 2D FFT, $M$ is the undersampling mask, $y$ is the noisy k-space measurement, $\bar{M} = \text{mean}(M)$, and $c \in \{0,1\}$ indexes real/imaginary components.

**regularization_term**:

L1 Sparsity:
$$\mathcal{L}_{L1}(x) = \frac{w_{L1}}{\Phi} \cdot \frac{1}{N^2} \sum_{i,j} |x_{i,j}|$$

Total Variation:
$$\mathcal{L}_{TV}(x) = \frac{w_{TV} \cdot N}{\Phi} \cdot \left[ \frac{1}{N(N-1)} \sum_{i,j} |x_{i+1,j} - x_{i,j}| + \frac{1}{N(N-1)} \sum_{i,j} |x_{i,j+1} - x_{i,j}| \right]$$

Combined prior:
$$\mathcal{L}_{\text{prior}}(x) = \mathcal{L}_{L1}(x) + \mathcal{L}_{TV}(x)$$

**constraints**:
$$x \geq 0 \quad \text{(enforced implicitly via Softplus activation: } x = \log(1 + e^{x_{\text{raw}}}) \cdot e^s \text{)}$$

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta$ | Model-dependent | float32 | Xavier/Kaiming (RealNVP default) | RealNVP flow network parameters |
| $s$ | $(1,)$ | float32 | $\log\left(\frac{\Phi}{0.8 \cdot N^2}\right)$ | Learnable log-scale parameter |

### 2.2 Dual Variables
None (this is a primal optimization method)

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $z$ | $(B, N^2)$ | float32 | Sampled from $\mathcal{N}(0,I)$ each iteration | Latent codes for flow |
| $x_{\text{raw}}$ | $(B, N^2)$ | float32 | Computed | Raw flow output before reshaping |
| $\text{logdet}_{\text{flow}}$ | $(B,)$ | float32 | Computed | Log determinant from RealNVP |
| $x_{\text{samp}}$ | $(B, N, N)$ | float32 | Computed | Reshaped flow output |
| $\text{det}_{\text{softplus}}$ | $(B,)$ | float32 | Computed | Softplus Jacobian correction |
| $\text{det}_{\text{scale}}$ | $(1,)$ | float32 | Computed | Scale factor Jacobian correction |
| $\text{logdet}_{\text{total}}$ | $(B,)$ | float32 | Computed | Combined log determinant |
| $x$ | $(B, N, N)$ | float32 | Computed | Final generated images |
| $\hat{k}$ | $(B, N, N, 2)$ | float32 | Computed | Predicted masked k-space |

### 2.4 Constants
| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $N$ | scalar | int | 64 (default) | Image dimension (npix) |
| $B$ | scalar | int | 64 | Batch size for MC estimation |
| $\sigma$ | scalar | float32 | $5 \times 10^{-7}$ (default) | K-space noise standard deviation |
| $w_{L1}$ | scalar | float32 | 0.0 (default) | L1 regularization weight |
| $w_{TV}$ | scalar | float32 | $10^3$ (default) | TV regularization weight |
| $w_{\text{logdet}}$ | scalar | float32 | 1.0 (default) | Log determinant weight |
| $\eta$ | scalar | float32 | $10^{-5}$ (default) | Learning rate |
| $K$ | scalar | int | 10 (default) | Number of epochs |
| $n_{\text{flow}}$ | scalar | int | 16 (default) | Number of RealNVP coupling layers |
| $c_{\text{size}}$ | scalar | int | 8 | Half-width of fully-sampled center region |
| $\gamma_{\text{clip}}$ | scalar | float32 | $10^{-2}$ | Gradient clipping threshold |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y$ | $(N, N, 2)$ | float32 | Noisy masked k-space measurements |
| $M$ | $(N, N, 2)$ | float32 | K-space undersampling mask (binary, stacked for real/imag) |
| $x_{\text{true}}$ | $(N, N)$ | float32 | Ground truth image (for evaluation only) |
| $\Phi$ | scalar | float32 | Total flux: $\Phi = \sum_{i,j} (x_{\text{true}})_{i,j}$ |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Load target image | $x_{\text{raw}} \leftarrow \text{pickle.load}(\text{impath})[\text{'target'}]$ | impath (string) | $x_{\text{raw}}$: (H_orig, W_orig) | Pickle contains 'target' key |
| P2 | Resize image | $x_{\text{true}} \leftarrow \text{BilinearResize}(x_{\text{raw}}, N, N)$ | $x_{\text{raw}}$: (H_orig, W_orig), $N$: scalar | $x_{\text{true}}$: $(N, N)$ | Bilinear interpolation via PIL |
| P3 | Compute clean k-space | $k_{\text{clean}} \leftarrow \mathcal{F}_{\text{ortho}}(x_{\text{true}})$, where $(\mathcal{F}_{\text{ortho}}(x))_{u,v} = \frac{1}{N}\sum_{i,j} x_{i,j} e^{-2\pi i(ui/N + vj/N)}$ | $x_{\text{true}}$: $(N, N)$ | $k_{\text{clean}}$: $(N, N)$ complex | Orthonormalized FFT |
| P4 | Stack real/imag | $k_{\text{stacked}} \leftarrow [\text{Re}(k_{\text{clean}}), \text{Im}(k_{\text{clean}})]$ | $k_{\text{clean}}$: $(N, N)$ complex | $k_{\text{stacked}}$: $(N, N, 2)$ | Last dim: [real, imag] |
| P5 | Add k-space noise | $k_{\text{noisy}} \leftarrow k_{\text{stacked}} + \epsilon$, where $\epsilon_{i,j,c} \sim \mathcal{N}(0, \sigma^2)$ | $k_{\text{stacked}}$: $(N, N, 2)$, $\sigma$: scalar | $k_{\text{noisy}}$: $(N, N, 2)$ | IID Gaussian noise |
| P6 | Load mask | $M_{\text{raw}} \leftarrow \text{numpy.load}(\text{maskpath})$ | maskpath (string) | $M_{\text{raw}}$: (H_mask, W_mask) | Binary mask |
| P7 | Resize mask | $M_{\text{resized}} \leftarrow \text{NearestResize}(M_{\text{raw}}, N, N)$ if shape differs | $M_{\text{raw}}$ | $M_{\text{resized}}$: $(N, N)$ | Nearest neighbor preserves binary values |
| P8 | Fill center region | $M_{\text{filled}}[c_s:c_e, c_s:c_e] \leftarrow 1$, where $c_s = N/2 - c_{\text{size}}$, $c_e = N/2 + c_{\text{size}}$ | $M_{\text{resized}}$: $(N, N)$, $c_{\text{size}}=8$ | $M_{\text{filled}}$: $(N, N)$ | Center always fully sampled |
| P9 | FFT shift mask | $M_{\text{shifted}} \leftarrow \text{fftshift}(M_{\text{filled}})$ | $M_{\text{filled}}$: $(N, N)$ | $M_{\text{shifted}}$: $(N, N)$ | Moves DC to corner |
| P10 | Stack mask | $M \leftarrow [M_{\text{shifted}}, M_{\text{shifted}}]$ (stacked on last dim) | $M_{\text{shifted}}$: $(N, N)$ | $M$: $(N, N, 2)$ | Same mask for real/imag |
| P11 | Apply mask to k-space | $y \leftarrow M \odot k_{\text{noisy}}$ | $M$: $(N, N, 2)$, $k_{\text{noisy}}$: $(N, N, 2)$ | $y$: $(N, N, 2)$ | Element-wise product |
| P12 | Compute flux | $\Phi \leftarrow \sum_{i,j} (x_{\text{true}})_{i,j}$ | $x_{\text{true}}$: $(N, N)$ | $\Phi$: scalar | Used for weight normalization |
| P13 | Compute normalized weights | $w_{L1}^{\text{norm}} \leftarrow w_{L1}/\Phi$, $w_{TV}^{\text{norm}} \leftarrow w_{TV} \cdot N/\Phi$, $w_{\text{logdet}}^{\text{norm}} \leftarrow w_{\text{logdet}}/(0.5 \cdot \sum M)$ | $\Phi$, $w_{L1}$, $w_{TV}$, $w_{\text{logdet}}$, $M$, $N$ | $w_{L1}^{\text{norm}}$, $w_{TV}^{\text{norm}}$, $w_{\text{logdet}}^{\text{norm}}$: scalars | Flux-normalized weights |
| P14 | Compute mask mean | $\bar{M} \leftarrow \text{mean}(M)$ | $M$: $(N, N, 2)$ | $\bar{M}$: scalar | For loss normalization |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| I1 | $\theta$ | RealNVP default initialization (Xavier for linear layers) | Model-dependent | float32 | `realnvpfc_model.RealNVP(N², n_flow, affine=True)` |
| I2 | $s$ | $s^{(0)} = \log\left(\frac{\Phi}{0.8 \cdot N^2}\right)$ | $(1,)$ | float32 | Heuristic based on expected mean intensity |
| I3 | $M_{\text{tensor}}$ | $M$ (from preprocessing) | $(N, N, 2)$ | float32 | Converted to PyTorch tensor on device |
| I4 | $y_{\text{tensor}}$ | $y$ (from preprocessing) | $(N, N, 2)$ | float32 | Converted to PyTorch tensor on device |
| I5 | optimizer state | Adam with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ | - | - | Applied to $\{\theta, s\}$ |
| I6 | $k$ | $0$ | scalar | int | Epoch counter |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP (for k = 0 to K-1)
```

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Sample latent codes | $z^{(k)} \sim \mathcal{N}(0, I_{N^2})$, i.e., $z^{(k)}_{b,j} \sim \mathcal{N}(0,1)$ independently | $B$, $N$ | $z^{(k)}$: $(B, N^2)$ | `torch.randn(64, npix*npix)` |
| 2 | [FORWARD] RealNVP reverse pass | $(x_{\text{raw}}^{(k)}, \text{logdet}_{\text{flow}}^{(k)}) \leftarrow f_\theta^{-1}(z^{(k)})$ | $z^{(k)}$: $(B, N^2)$, $\theta$ | $x_{\text{raw}}^{(k)}$: $(B, N^2)$, $\text{logdet}_{\text{flow}}^{(k)}$: $(B,)$ | Affine coupling layers with inverse pass |
| 3 | Reshape to image | $x_{\text{samp}}^{(k)} \leftarrow \text{reshape}(x_{\text{raw}}^{(k)}, (B, N, N))$ | $x_{\text{raw}}^{(k)}$: $(B, N^2)$ | $x_{\text{samp}}^{(k)}$: $(B, N, N)$ | View operation |
| 4 | Compute scale factor | $\alpha^{(k)} \leftarrow \exp(s^{(k)})$ | $s^{(k)}$: $(1,)$ | $\alpha^{(k)}$: $(1,)$ | Exponentiate log-scale |
| 5 | Apply Softplus and scaling | $x^{(k)} \leftarrow \text{Softplus}(x_{\text{samp}}^{(k)}) \cdot \alpha^{(k)}$, where $\text{Softplus}(t) = \log(1 + e^t)$ | $x_{\text{samp}}^{(k)}$: $(B, N, N)$, $\alpha^{(k)}$: $(1,)$ | $x^{(k)}$: $(B, N, N)$ | Ensures non-negativity |
| 6 | Compute Softplus Jacobian correction | $\text{det}_{\text{softplus}}^{(k)} \leftarrow \sum_{i,j} \left[ (x_{\text{samp}}^{(k)})_{b,i,j} - \log(1 + \exp((x_{\text{samp}}^{(k)})_{b,i,j})) \right]$ for each $b$ | $x_{\text{samp}}^{(k)}$: $(B, N, N)$ | $\text{det}_{\text{softplus}}^{(k)}$: $(B,)$ | $\log(\text{sigmoid}(t)) = t - \text{Softplus}(t)$ |
| 7 | Compute scale Jacobian correction | $\text{det}_{\text{scale}}^{(k)} \leftarrow s^{(k)} \cdot N^2$ | $s^{(k)}$: $(1,)$, $N$: scalar | $\text{det}_{\text{scale}}^{(k)}$: $(1,)$ | Log determinant of diagonal scaling |
| 8 | Combine log determinants | $\text{logdet}_{\text{total}}^{(k)} \leftarrow \text{logdet}_{\text{flow}}^{(k)} + \text{det}_{\text{softplus}}^{(k)} + \text{det}_{\text{scale}}^{(k)}$ | $\text{logdet}_{\text{flow}}^{(k)}$: $(B,)$, $\text{det}_{\text{softplus}}^{(k)}$: $(B,)$, $\text{det}_{\text{scale}}^{(k)}$: $(1,)$ | $\text{logdet}_{\text{total}}^{(k)}$: $(B,)$ | Broadcasting for det_scale |
| 9 | [FORWARD] Pad image for FFT | $x_{\text{complex}}^{(k)} \leftarrow x^{(k)} + 0i$ (create complex tensor) | $x^{(k)}$: $(B, N, N)$ | $x_{\text{complex}}^{(k)}$: $(B, N, N)$ complex | Stack real with zeros for imag |
| 10 | [FORWARD] Compute 2D FFT | $\hat{k}_{\text{full}}^{(k)} \leftarrow \mathcal{F}_{\text{ortho}}(x_{\text{complex}}^{(k)})$, where $(\mathcal{F}_{\text{ortho}})_{u,v} = \frac{1}{N}\sum_{i,j} x_{i,j} e^{-2\pi i(ui/N + vj/N)}$ | $x_{\text{complex}}^{(k)}$: $(B, N, N)$ complex | $\hat{k}_{\text{full}}^{(k)}$: $(B, N, N)$ complex | `torch.fft.fft2(norm="ortho")` |
| 11 | Convert to real representation | $\hat{k}_{\text{stacked}}^{(k)} \leftarrow [\text{Re}(\hat{k}_{\text{full}}^{(k)}), \text{Im}(\hat{k}_{\text{full}}^{(k)})]$ | $\hat{k}_{\text{full}}^{(k)}$: $(B, N, N)$ complex | $\hat{k}_{\text{stacked}}^{(k)}$: $(B, N, N, 2)$ | `torch.view_as_real` |
| 12 | Apply undersampling mask | $\hat{k}^{(k)} \leftarrow \hat{k}_{\text{stacked}}^{(k)} \odot M_{\text{tensor}}$ | $\hat{k}_{\text{stacked}}^{(k)}$: $(B, N, N, 2)$, $M_{\text{tensor}}$: $(N, N, 2)$ | $\hat{k}^{(k)}$: $(B, N, N, 2)$ | Broadcasting over batch |
| 13 | Compute k-space residual | $r^{(k)} \leftarrow \hat{k}^{(k)} - y_{\text{tensor}}$ | $\hat{k}^{(k)}$: $(B, N, N, 2)$, $y_{\text{tensor}}$: $(N, N, 2)$ | $r^{(k)}$: $(B, N, N, 2)$ | Broadcasting over batch |
| 14 | Compute squared residual | $(r^2)^{(k)} \leftarrow (r^{(k)})^2$ | $r^{(k)}$: $(B, N, N, 2)$ | $(r^2)^{(k)}$: $(B, N, N, 2)$ | Element-wise square |
| 15 | Compute per-sample MSE | $\text{MSE}^{(k)}_b \leftarrow \frac{1}{N \cdot N \cdot 2} \sum_{i,j,c} (r^2)^{(k)}_{b,i,j,c}$ | $(r^2)^{(k)}$: $(B, N, N, 2)$ | $\text{MSE}^{(k)}$: $(B,)$ | Mean over spatial and channel dims |
| 16 | Compute data fidelity loss | $\mathcal{L}_{\text{data}}^{(k)} \leftarrow \frac{\text{MSE}^{(k)}}{\sigma^2 \cdot \bar{M}}$ | $\text{MSE}^{(k)}$: $(B,)$, $\sigma$, $\bar{M}$: scalars | $\mathcal{L}_{\text{data}}^{(k)}$: $(B,)$ | Per-sample loss |
| 17 | Compute L1 norm (if $w_{L1}^{\text{norm}} > 0$) | $\mathcal{L}_{L1}^{(k)} \leftarrow \frac{1}{N^2} \sum_{i,j} |x^{(k)}_{b,i,j}|$ for each $b$; else $\mathcal{L}_{L1}^{(k)} \leftarrow 0$ | $x^{(k)}$: $(B, N, N)$ | $\mathcal{L}_{L1}^{(k)}$: $(B,)$ or scalar | Mean absolute value |
| 18 | Compute TV horizontal differences | $\Delta_h^{(k)} \leftarrow x^{(k)}_{:,1:,:} - x^{(k)}_{:,:-1,:}$ | $x^{(k)}$: $(B, N, N)$ | $\Delta_h^{(k)}$: $(B, N-1, N)$ | Vertical gradient |
| 19 | Compute TV vertical differences | $\Delta_v^{(k)} \leftarrow x^{(k)}_{:,:,1:} - x^{(k)}_{:,:,:-1}$ | $x^{(k)}$: $(B, N, N)$ | $\Delta_v^{(k)}$: $(B, N, N-1)$ | Horizontal gradient |
| 20 | Compute TV loss (if $w_{TV}^{\text{norm}} > 0$) | $\mathcal{L}_{TV}^{(k)} \leftarrow \frac{1}{N(N-1)}\sum_{i,j}|\Delta_h^{(k)}_{b,i,j}| + \frac{1}{N(N-1)}\sum_{i,j}|\Delta_v^{(k)}_{b,i,j}|$; else $\mathcal{L}_{TV}^{(k)} \leftarrow 0$ | $\Delta_h^{(k)}$, $\Delta_v^{(k)}$ | $\mathcal{L}_{TV}^{(k)}$: $(B,)$ or scalar | Isotropic TV |
| 21 | Compute prior loss | $\mathcal{L}_{\text{prior}}^{(k)} \leftarrow w_{TV}^{\text{norm}} \cdot \mathcal{L}_{TV}^{(k)} + w_{L1}^{\text{norm}} \cdot \mathcal{L}_{L1}^{(k)}$ | $\mathcal{L}_{TV}^{(k)}$, $\mathcal{L}_{L1}^{(k)}$, weights | $\mathcal{L}_{\text{prior}}^{(k)}$: $(B,)$ | Weighted sum |
| 22 | Compute total loss | $\mathcal{L}^{(k)} \leftarrow \frac{1}{B}\sum_b \mathcal{L}_{\text{data},b}^{(k)} + \frac{1}{B}\sum_b \mathcal{L}_{\text{prior},b}^{(k)} - w_{\text{logdet}}^{\text{norm}} \cdot \frac{1}{B}\sum_b \text{logdet}_{\text{total},b}^{(k)}$ | All loss components | $\mathcal{L}^{(k)}$: scalar | Monte Carlo average |
| 23 | [BACKWARD] Compute gradients | $g_\theta^{(k)} \leftarrow \nabla_\theta \mathcal{L}^{(k)}$, $g_s^{(k)} \leftarrow \nabla_s \mathcal{L}^{(k)}$ | $\mathcal{L}^{(k)}$, $\theta^{(k)}$, $s^{(k)}$ | $g_\theta^{(k)}$, $g_s^{(k)}$ | Automatic differentiation |
| 24 | Clip gradients | $\tilde{g}_\theta^{(k)} \leftarrow \text{clip\_grad\_norm}(g_\theta^{(k)}, \gamma_{\text{clip}})$, $\tilde{g}_s^{(k)} \leftarrow \text{clip\_grad\_norm}(g_s^{(k)}, \gamma_{\text{clip}})$ | Gradients, $\gamma_{\text{clip}} = 10^{-2}$ | Clipped gradients | Scale if $\|g\| > \gamma$ |
| 25 | Adam update | $\theta^{(k+1)} \leftarrow \text{Adam}(\theta^{(k)}, \tilde{g}_\theta^{(k