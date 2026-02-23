## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: Deep Probabilistic Imaging for MRI Reconstruction (DPI-MRI)
- **algorithm_family**: Generative Model-based Inverse Problem Solving via Normalizing Flows
- **summary**: Learns a RealNVP normalizing flow generator that maps standard Gaussian latent codes to MRI images consistent with undersampled k-space measurements, optimizing a variational objective combining data fidelity, regularization, and entropy maximization.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\theta, s} \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ \mathcal{L}_{\text{data}}(G_\theta(z), y) + \mathcal{L}_{\text{prior}}(G_\theta(z)) - \lambda_{\text{logdet}} \log \left| \det \frac{\partial G_\theta(z)}{\partial z} \right| \right]$$

where $G_\theta(z) = e^s \cdot \text{Softplus}(f_\theta^{-1}(z))$ with $f_\theta$ being the RealNVP flow.

### data_fidelity_term
$$\mathcal{L}_{\text{data}}(x, y) = \frac{1}{\bar{M}} \cdot \frac{1}{\sigma^2} \cdot \frac{1}{N_{\text{pix}}^2 \cdot 2} \sum_{i,j,c} \left( M_{i,j,c} \cdot \mathcal{F}(x)_{i,j,c} - y_{i,j,c} \right)^2$$

where:
- $\mathcal{F}$: orthonormalized 2D FFT
- $M$: k-space undersampling mask (real/imag stacked)
- $y = M \odot (\mathcal{F}(x_{\text{true}}) + \epsilon)$: noisy masked k-space measurements
- $\bar{M} = \text{mean}(M)$: average mask value
- $c \in \{0, 1\}$: real and imaginary components

### regularization_term
$$\mathcal{L}_{\text{prior}}(x) = \lambda_{\text{L1}} \cdot \frac{1}{\Phi} \cdot \frac{1}{N_{\text{pix}}^2} \sum_{i,j} |x_{i,j}| + \lambda_{\text{TV}} \cdot \frac{N_{\text{pix}}}{\Phi} \cdot \text{TV}(x)$$

where:
$$\text{TV}(x) = \frac{1}{N_{\text{pix}}(N_{\text{pix}}-1)} \sum_{i,j} |x_{i+1,j} - x_{i,j}| + \frac{1}{N_{\text{pix}}(N_{\text{pix}}-1)} \sum_{i,j} |x_{i,j+1} - x_{i,j}|$$

and $\Phi = \sum_{i,j} x^{\text{true}}_{i,j}$ is the total flux.

### constraints
None (unconstrained optimization; positivity enforced via Softplus activation)

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta$ | primal | RealNVP params | float32 | Default PyTorch initialization for RealNVP with `n_flow` coupling layers | Parameters of RealNVP normalizing flow |
| $s$ | primal | $(1,)$ | float32 | $s^{(0)} = \log\left(\frac{\Phi}{0.8 \cdot N_{\text{pix}}^2}\right)$ | Log scale factor for image intensity |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $z$ | auxiliary | $(N_{\text{batch}}, N_{\text{pix}}^2)$ | float32 | Sampled: $z \sim \mathcal{N}(0, I)$ | Latent codes from standard normal |
| $\tilde{x}$ | auxiliary | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed from flow | Raw flow output before activation |
| $x$ | auxiliary | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Generated images after softplus and scaling |
| $\hat{k}$ | auxiliary | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | Computed | Predicted masked k-space |
| $\log|\det J_f|$ | auxiliary | $(N_{\text{batch}},)$ | float32 | Computed by flow | Log-determinant from RealNVP |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $y$ | observations | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | From preprocessing | Noisy masked k-space measurements |
| $M$ | observations | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | From file + center filling | K-space undersampling mask |
| $x^{\text{true}}$ | observations | $(N_{\text{pix}}, N_{\text{pix}})$ | float32 | From file | Ground truth image (for evaluation only) |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $N_{\text{pix}}$ | constants | scalar | int | 64 (default) | Image dimension |
| $N_{\text{batch}}$ | constants | scalar | int | 64 | Batch size for sampling |
| $N_{\text{flow}}$ | constants | scalar | int | 16 (default) | Number of RealNVP coupling layers |
| $\sigma$ | constants | scalar | float32 | $5 \times 10^{-7}$ (default) | K-space noise standard deviation |
| $\eta$ | constants | scalar | float32 | $10^{-5}$ (default) | Learning rate |
| $\lambda_{\text{logdet}}$ | constants | scalar | float32 | $\frac{1.0}{0.5 \cdot \sum_{i,j,c} M_{i,j,c}}$ | Normalized log-det weight |
| $\lambda_{\text{L1}}$ | constants | scalar | float32 | 0.0 (default) | L1 regularization weight |
| $\lambda_{\text{TV}}$ | constants | scalar | float32 | $10^3$ (default) | TV regularization weight |
| $N_{\text{epoch}}$ | constants | scalar | int | 10 (default) | Number of training epochs |
| $c_{\text{size}}$ | constants | scalar | int | 8 | Half-width of fully-sampled center region |
| $\gamma_{\text{clip}}$ | constants | scalar | float32 | $10^{-2}$ | Gradient clipping threshold |
| $\alpha_{\text{scale}}$ | constants | scalar | float32 | 0.8 | Scale initialization denominator factor |

## 3. PREPROCESSING PIPELINE

| Step Order | Step Name | Formula | Inputs | Outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load Ground Truth | $x^{\text{raw}} = \text{pickle.load}(\texttt{impath})[\text{'target'}]$ | `impath` (string) | $x^{\text{raw}}$ (arbitrary shape) |
| 2 | Resize Image | $x^{\text{true}} = \text{BilinearResize}(x^{\text{raw}}, N_{\text{pix}})$ | $x^{\text{raw}}$, $N_{\text{pix}}$ | $x^{\text{true}} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 3 | Compute Clean K-space | $k^{\text{clean}}_{i,j,0} + i \cdot k^{\text{clean}}_{i,j,1} = \frac{1}{N_{\text{pix}}} \sum_{m,n} x^{\text{true}}_{m,n} e^{-2\pi i (im/N + jn/N)}$ | $x^{\text{true}}$ | $k^{\text{clean}} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}} \times 2}$ |
| 4 | Add K-space Noise | $k^{\text{noisy}} = k^{\text{clean}} + \epsilon, \quad \epsilon_{i,j,c} \sim \mathcal{N}(0, \sigma^2)$ | $k^{\text{clean}}$, $\sigma$ | $k^{\text{noisy}} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}} \times 2}$ |
| 5 | Load Mask | $M^{\text{raw}} = \text{np.load}(\texttt{maskpath})$ | `maskpath` (string) | $M^{\text{raw}}$ (arbitrary shape) |
| 6 | Resize Mask | $M^{\text{resized}} = \text{NearestResize}(M^{\text{raw}}, N_{\text{pix}})$ if shape mismatch | $M^{\text{raw}}$, $N_{\text{pix}}$ | $M^{\text{resized}} \in \{0,1\}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 7 | Fill Center Region | $M^{\text{center}}_{i,j} = \begin{cases} 1 & \text{if } |i - N_{\text{pix}}/2| < c_{\text{size}} \land |j - N_{\text{pix}}/2| < c_{\text{size}} \\ M^{\text{resized}}_{i,j} & \text{otherwise} \end{cases}$ | $M^{\text{resized}}$, $c_{\text{size}}=8$ | $M^{\text{center}} \in \{0,1\}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 8 | FFT Shift Mask | $M^{\text{shifted}}_{i,j} = M^{\text{center}}_{(i+N/2)\mod N, (j+N/2)\mod N}$ | $M^{\text{center}}$ | $M^{\text{shifted}} \in \{0,1\}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 9 | Stack Mask | $M_{i,j,c} = M^{\text{shifted}}_{i,j}$ for $c \in \{0,1\}$ | $M^{\text{shifted}}$ | $M \in \{0,1\}^{N_{\text{pix}} \times N_{\text{pix}} \times 2}$ |
| 10 | Compute Masked Measurements | $y = M \odot k^{\text{noisy}}$ | $M$, $k^{\text{noisy}}$ | $y \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}} \times 2}$ |
| 11 | Compute Flux | $\Phi = \sum_{i,j} x^{\text{true}}_{i,j}$ | $x^{\text{true}}$ | $\Phi \in \mathbb{R}$ |
| 12 | Compute Mask Mean | $\bar{M} = \frac{1}{N_{\text{pix}}^2 \cdot 2} \sum_{i,j,c} M_{i,j,c}$ | $M$ | $\bar{M} \in \mathbb{R}$ |
| 13 | Normalize Weights | $\lambda'_{\text{L1}} = \frac{\lambda_{\text{L1}}}{\Phi}, \quad \lambda'_{\text{TV}} = \frac{\lambda_{\text{TV}} \cdot N_{\text{pix}}}{\Phi}, \quad \lambda'_{\text{logdet}} = \frac{\lambda_{\text{logdet}}}{0.5 \cdot \sum M}$ | $\lambda_{\text{L1}}, \lambda_{\text{TV}}, \lambda_{\text{logdet}}, \Phi, M$ | Normalized weights |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| I.1 | Initialize Generator | $f_\theta: \mathbb{R}^{N_{\text{pix}}^2} \to \mathbb{R}^{N_{\text{pix}}^2}$ as RealNVP with $N_{\text{flow}}$ affine coupling layers | $N_{\text{pix}}$, $N_{\text{flow}}$ | $\theta^{(0)}$ |
| I.2 | Initialize Log Scale | $s^{(0)} = \log\left(\frac{\Phi}{0.8 \cdot N_{\text{pix}}^2}\right)$ | $\Phi$, $N_{\text{pix}}$ | $s^{(0)} \in \mathbb{R}$ |
| I.3 | Initialize Optimizer | Adam optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 10^{-5}$ | $\theta^{(0)}$, $s^{(0)}$, $\eta$ | Optimizer state |
| I.4 | Transfer to Device | Move $M$, $y$, $\theta^{(0)}$, $s^{(0)}$ to GPU if available | All tensors | Device tensors |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{epoch}} - 1$)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| L.1 | Sample Latent Codes | $z^{(k)} \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$, shape $(64, N_{\text{pix}}^2)$ | - | $z^{(k)}$ | PyTorch `torch.randn` |
| L.2 | Flow Reverse Pass | $(\tilde{x}^{\text{flat}}, \log|\det J_f|) = f_\theta^{-1}(z^{(k)})$ | $z^{(k)}$, $\theta^{(k)}$ | $\tilde{x}^{\text{flat}} \in \mathbb{R}^{64 \times N_{\text{pix}}^2}$, $\log|\det J_f| \in \mathbb{R}^{64}$ | RealNVP inverse |
| L.3 | Reshape | $\tilde{x} = \text{reshape}(\tilde{x}^{\text{flat}}, (64, N_{\text{pix}}, N_{\text{pix}}))$ | $\tilde{x}^{\text{flat}}$ | $\tilde{x} \in \mathbb{R}^{64 \times N_{\text{pix}} \times N_{\text{pix}}}$ | - |
| L.4 | Apply Scale Factor | $\sigma_s = e^{s^{(k)}}$ | $s^{(k)}$ | $\sigma_s \in \mathbb{R}$ | - |
| L.5 | Apply Softplus Activation | $x = \sigma_s \cdot \text{Softplus}(\tilde{x})$ where $\text{Softplus}(u) = \log(1 + e^u)$ | $\tilde{x}$, $\sigma_s$ | $x \in \mathbb{R}^{64 \times N_{\text{pix}} \times N_{\text{pix}}}$ | Ensures positivity |
| L.6 | Softplus Jacobian | $\log|\det J_{\text{sp}}| = \sum_{i,j} \left( \tilde{x}_{i,j} - \text{Softplus}(\tilde{x}_{i,j}) \right) = \sum_{i,j} \log\left(\frac{e^{\tilde{x}_{i,j}}}{1+e^{\tilde{x}_{i,j}}}\right)$ | $\tilde{x}$ | $\log|\det J_{\text{sp}}| \in \mathbb{R}^{64}$ | Uses identity: $u - \text{Softplus}(u) = \log(\text{sigmoid}(u))$ |
| L.7 | Scale Jacobian | $\log|\det J_s| = s^{(k)} \cdot N_{\text{pix}}^2$ | $s^{(k)}$ | $\log|\det J_s| \in \mathbb{R}$ | - |
| L.8 | Total Log-Determinant | $\log|\det J| = \log|\det J_f| + \log|\det J_{\text{sp}}| + \log|\det J_s|$ | Components | $\log|\det J| \in \mathbb{R}^{64}$ | Chain rule for Jacobians |
| L.9 | Forward FFT | $K = \mathcal{F}_{\text{ortho}}(x)$, where $K_{i,j,0} + iK_{i,j,1} = \frac{1}{N}\sum_{m,n} x_{m,n} e^{-2\pi i(im/N + jn/N)}$ | $x$ | $K \in \mathbb{R}^{64 \times N_{\text{pix}} \times N_{\text{pix}} \times 2}$ | Orthonormalized FFT2 |
| L.10 | Apply Mask | $\hat{k} = M \odot K$ | $K$, $M$ | $\hat{k} \in \mathbb{R}^{64 \times N_{\text{pix}} \times N_{\text{pix}} \times 2}$ | Element-wise with broadcast |
| L.11 | Data Fidelity Loss | $\mathcal{L}_{\text{data}}^{(b)} = \frac{1}{\bar{M} \cdot \sigma^2 \cdot N_{\text{pix}}^2 \cdot 2} \sum_{i,j,c} (\hat{k}^{(b)}_{i,j,c} - y_{i,j,c})^2$ | $\hat{k}$, $y$, $\bar{M}$, $\sigma$ | $\mathcal{L}_{\text{data}} \in \mathbb{R}^{64}$ | Per-sample loss |
| L.12 | L1 Loss | $\mathcal{L}_{\text{L1}}^{(b)} = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} |x^{(b)}_{i,j}|$ | $x$ | $\mathcal{L}_{\text{L1}} \in \mathbb{R}^{64}$ | Skip if $\lambda_{\text{L1}} = 0$ |
| L.13 | TV Loss | $\mathcal{L}_{\text{TV}}^{(b)} = \frac{1}{N_{\text{pix}}(N_{\text{pix}}-1)} \left( \sum_{i,j} |x^{(b)}_{i+1,j} - x^{(b)}_{i,j}| + \sum_{i,j} |x^{(b)}_{i,j+1} - x^{(b)}_{i,j}| \right)$ | $x$ | $\mathcal{L}_{\text{TV}} \in \mathbb{R}^{64}$ | Anisotropic TV |
| L.14 | Prior Loss | $\mathcal{L}_{\text{prior}} = \lambda'_{\text{TV}} \cdot \mathcal{L}_{\text{TV}} + \lambda'_{\text{L1}} \cdot \mathcal{L}_{\text{L1}}$ | $\mathcal{L}_{\text{TV}}$, $\mathcal{L}_{\text{L1}}$, weights | $\mathcal{L}_{\text{prior}} \in \mathbb{R}^{64}$ | - |
| L.15 | Total Loss | $\mathcal{L} = \frac{1}{64} \sum_{b=1}^{64} \left( \mathcal{L}_{\text{data}}^{(b)} + \mathcal{L}_{\text{prior}}^{(b)} - \lambda'_{\text{logdet}} \cdot \log|\det J|^{(b)} \right)$ | All loss components | $\mathcal{L} \in \mathbb{R}$ | Batch average |
| L.16 | Compute Gradients | $\nabla_\theta \mathcal{L}$, $\nabla_s \mathcal{L}$ via backpropagation | $\mathcal{L}$ | Gradients | PyTorch autograd |
| L.17 | Gradient Clipping | $\nabla \leftarrow \min\left(1, \frac{\gamma_{\text{clip}}}{\|\nabla\|_2}\right) \cdot \nabla$ with $\gamma_{\text{clip}} = 0.01$ | Gradients | Clipped gradients | L2 norm clipping |
| L.18 | Adam Update | $\theta^{(k+1)}, s^{(k+1)} = \text{Adam}(\theta^{(k)}, s^{(k)}, \nabla, \eta)$ | Parameters, gradients, $\eta$ | $\theta^{(k+1)}$, $s^{(k+1)}$ | Standard Adam |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| F.1 | Sample Final Latent | $z^{\text{final}} \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$, shape $(64, N_{\text{pix}}^2)$ | - | $z^{\text{final}}$ |
| F.2 | Generate Final Images | $\tilde{x}^{\text{final}} = \text{reshape}(f_{\theta^{(N_{\text{epoch}})}}^{-1}(z^{\text{final}}), (64, N_{\text{pix}}, N_{\text{pix}}))$ | $z^{\text{final}}$, $\theta^{(N_{\text{epoch}})}$ | $\tilde{x}^{\text{final}}$ |
| F.3 | Apply Final Transform | $x^{\text{final}} = e^{s^{(N_{\text{epoch}})}} \cdot \text{Softplus}(\tilde{x}^{\text{final}})$ | $\tilde{x}^{\text{final}}$, $s^{(N_{\text{epoch}})}$ | $x^{\text{final}} \in \mathbb{R}^{64 \times N_{\text{pix}} \times N_{\text{pix}}}$ |
| F.4 | Compute Mean Reconstruction | $\bar{x} = \frac{1}{64} \sum_{b=1}^{64} x^{\text{final},(b)}$ | $x^{\text{final}}$ | $\bar{x} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ |

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k \geq N_{\text{epoch}}$
- **parameters**: 
  - $N_{\text{epoch}} = 10$ (default)
- **check_frequency**: After each complete epoch (iteration)

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Mean Squared Error | $\text{MSE} = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} (\bar{x}_{i,j} - x^{\text{true}}_{i,j})^2$ | Average squared pixel-wise error between mean reconstruction and ground truth |
| Root Mean Squared Error | $\text{RMSE} = \sqrt{\text{MSE}}$ | Square root of MSE for interpretable error magnitude |
| Peak Signal-to-Noise Ratio | $\text{PSNR} = 20 \log_{10}\left( \frac{\max_{i,j}(x^{\text{true}}_{i,j})}{\text{RMSE}} \right)$ dB | Logarithmic ratio of peak signal to reconstruction error |
| Training Loss | $\mathcal{L}^{(k)}$ stored at each epoch | Monitors optimization progress |
| K-space Data Fidelity | $\frac{1}{64}\sum_b \mathcal{L}_{\text{data}}^{(b)}$ printed at each epoch | Monitors measurement consistency |