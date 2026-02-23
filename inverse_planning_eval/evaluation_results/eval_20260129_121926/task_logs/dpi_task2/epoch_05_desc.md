## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Deep Probabilistic Imaging for MRI Reconstruction (DPI-MRI)
- **algorithm_family**: Amortized Variational Inference with Normalizing Flows
- **summary**: Learns a generative model (RealNVP normalizing flow) that maps Gaussian latent variables to MRI images consistent with undersampled k-space measurements, optimizing a variational objective combining data fidelity, regularization, and entropy maximization.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\theta, s} \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ \mathcal{L}_{\text{data}}(x(z;\theta,s)) + \mathcal{L}_{\text{prior}}(x(z;\theta,s)) - w_{\text{logdet}} \log \left| \det \frac{\partial x}{\partial z} \right| \right]$$

where $x(z;\theta,s) = \text{Softplus}(G_\theta(z)) \cdot e^s$

### data_fidelity_term
$$\mathcal{L}_{\text{data}} = \frac{1}{\sigma^2 \cdot \bar{M}} \cdot \frac{1}{N_{\text{pix}}^2 \cdot 2} \sum_{u,v,c} \left( M_{u,v,c} \cdot \hat{x}_{u,v,c} - y_{u,v,c} \right)^2$$

where:
- $\hat{x} = \mathcal{F}_{\text{2D}}(x)$ is the 2D orthonormal FFT of the reconstructed image
- $M$ is the undersampling mask (stacked for real/imaginary)
- $y$ is the observed masked k-space
- $\bar{M} = \text{mean}(M)$ is the mean of the mask
- $c \in \{0,1\}$ indexes real and imaginary components

### regularization_term
**L1 Regularization:**
$$\mathcal{L}_{\text{L1}} = \lambda_{\text{L1}} \cdot \frac{1}{N_{\text{pix}}^2} \sum_{i,j} |x_{i,j}|$$

where $\lambda_{\text{L1}} = \frac{\text{l1\_weight}}{\text{flux}}$

**Total Variation Regularization:**
$$\mathcal{L}_{\text{TV}} = \lambda_{\text{TV}} \left( \frac{1}{N_{\text{pix}}(N_{\text{pix}}-1)} \sum_{i,j} |x_{i+1,j} - x_{i,j}| + \frac{1}{N_{\text{pix}}(N_{\text{pix}}-1)} \sum_{i,j} |x_{i,j+1} - x_{i,j}| \right)$$

where $\lambda_{\text{TV}} = \frac{\text{tv\_weight} \cdot N_{\text{pix}}}{\text{flux}}$

**Log-Determinant (Entropy) Term:**
$$\mathcal{L}_{\text{logdet}} = -w_{\text{logdet}} \cdot \log \left| \det \frac{\partial x}{\partial z} \right|$$

where $w_{\text{logdet}} = \frac{\text{logdet\_weight}}{0.5 \cdot \sum_{u,v,c} M_{u,v,c}}$

### constraints
Implicit non-negativity constraint via Softplus activation:
$$x \geq 0 \quad \text{(enforced by } x = \text{Softplus}(\cdot) \cdot e^s > 0 \text{)}$$

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta$ | Model-dependent | float32 | Xavier/Kaiming (RealNVP default) | Parameters of RealNVP normalizing flow generator |
| $s$ | $(1,)$ | float32 | $\log\left(\frac{\text{flux}}{0.8 \cdot N_{\text{pix}}^2}\right)$ | Learnable log-scale factor for image intensities |

### 2.2 Dual Variables
None (unconstrained optimization via gradient descent)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $z$ | $(N_{\text{batch}}, N_{\text{pix}}^2)$ | float32 | Sampled each iteration | Latent codes from standard normal |
| $\tilde{x}$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Raw generator output before activation |
| $x$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Reconstructed images after Softplus and scaling |
| $\hat{x}$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | Computed | Predicted k-space (real/imag stacked) |
| $\log\det_{\text{flow}}$ | $(N_{\text{batch}},)$ | float32 | Computed | Log-determinant from RealNVP |
| $\log\det_{\text{softplus}}$ | $(N_{\text{batch}},)$ | float32 | Computed | Log-determinant correction for Softplus |
| $\log\det_{\text{scale}}$ | $(1,)$ | float32 | Computed | Log-determinant correction for scaling |
| $\log\det_{\text{total}}$ | $(N_{\text{batch}},)$ | float32 | Computed | Total log-determinant of transformation |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $N_{\text{pix}}$ | scalar | int | 64 (default) | Image dimension |
| $N_{\text{batch}}$ | scalar | int | 64 | Batch size for sampling |
| $N_{\text{flow}}$ | scalar | int | 16 (default) | Number of flow layers in RealNVP |
| $N_{\text{epoch}}$ | scalar | int | 10 (default) | Number of training epochs |
| $\sigma$ | scalar | float32 | $5 \times 10^{-7}$ (default) | K-space noise standard deviation |
| $\eta$ | scalar | float32 | $10^{-5}$ (default) | Learning rate |
| $\text{logdet\_weight}$ | scalar | float32 | 1.0 (default) | Weight for log-determinant term |
| $\text{l1\_weight}$ | scalar | float32 | 0.0 (default) | Weight for L1 regularization |
| $\text{tv\_weight}$ | scalar | float32 | $10^{3}$ (default) | Weight for TV regularization |
| $\text{center\_size}$ | scalar | int | 8 | Half-width of fully-sampled center region |
| $\text{grad\_clip}$ | scalar | float32 | $10^{-2}$ | Gradient clipping threshold |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $I_{\text{true}}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float32 | Ground truth MRI image |
| $y$ | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | Noisy masked k-space observations |
| $M$ | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | Undersampling mask (binary, stacked for real/imag) |
| $\text{flux}$ | scalar | float32 | Total intensity $\sum_{i,j} I_{\text{true},i,j}$ |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load ground truth image | $I_{\text{raw}} = \text{pickle.load}(\text{impath})[\text{'target'}]$ | impath: str | $I_{\text{raw}}$: $(H_{\text{orig}}, W_{\text{orig}})$ |
| 2 | Resize image | $I_{\text{true}} = \text{BilinearResize}(I_{\text{raw}}, N_{\text{pix}}, N_{\text{pix}})$ | $I_{\text{raw}}$: $(H, W)$, $N_{\text{pix}}$: scalar | $I_{\text{true}}$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 3 | Compute clean k-space | $K_{\text{clean},u,v} = \frac{1}{N_{\text{pix}}} \sum_{i,j} I_{\text{true},i,j} \exp\left(-2\pi \imath \left(\frac{ui}{N_{\text{pix}}} + \frac{vj}{N_{\text{pix}}}\right)\right)$ | $I_{\text{true}}$: $(N_{\text{pix}}, N_{\text{pix}})$ | $K_{\text{clean}}$: $(N_{\text{pix}}, N_{\text{pix}})$ complex |
| 4 | Stack real/imag | $K_{\text{stacked},u,v,0} = \Re(K_{\text{clean},u,v})$, $K_{\text{stacked},u,v,1} = \Im(K_{\text{clean},u,v})$ | $K_{\text{clean}}$: $(N_{\text{pix}}, N_{\text{pix}})$ complex | $K_{\text{stacked}}$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$ |
| 5 | Add k-space noise | $K_{\text{noisy}} = K_{\text{stacked}} + \sigma \cdot \epsilon$, where $\epsilon_{u,v,c} \sim \mathcal{N}(0,1)$ | $K_{\text{stacked}}$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$, $\sigma$: scalar | $K_{\text{noisy}}$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$ |
| 6 | Load mask | $M_{\text{raw}} = \text{np.load}(\text{maskpath})$ | maskpath: str | $M_{\text{raw}}$: $(H_M, W_M)$ |
| 7 | Resize mask | $M_{\text{resized}} = \text{NearestResize}(M_{\text{raw}}, N_{\text{pix}}, N_{\text{pix}})$ if $H_M \neq N_{\text{pix}}$ | $M_{\text{raw}}$: $(H_M, W_M)$, $N_{\text{pix}}$: scalar | $M_{\text{resized}}$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 8 | Set center region | $M_{\text{center},u,v} = 1$ for $u,v \in [N_{\text{pix}}/2 - 8, N_{\text{pix}}/2 + 8)$ | $M_{\text{resized}}$: $(N_{\text{pix}}, N_{\text{pix}})$ | $M_{\text{center}}$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 9 | FFT shift mask | $M_{\text{shifted},u,v} = M_{\text{center},(u+N_{\text{pix}}/2) \mod N_{\text{pix}}, (v+N_{\text{pix}}/2) \mod N_{\text{pix}}}$ | $M_{\text{center}}$: $(N_{\text{pix}}, N_{\text{pix}})$ | $M_{\text{shifted}}$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 10 | Stack mask | $M_{u,v,c} = M_{\text{shifted},u,v}$ for $c \in \{0,1\}$ | $M_{\text{shifted}}$: $(N_{\text{pix}}, N_{\text{pix}})$ | $M$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$ |
| 11 | Apply mask to k-space | $y = M \odot K_{\text{noisy}}$ | $M$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$, $K_{\text{noisy}}$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | $y$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$ |
| 12 | Compute flux | $\text{flux} = \sum_{i,j} I_{\text{true},i,j}$ | $I_{\text{true}}$: $(N_{\text{pix}}, N_{\text{pix}})$ | flux: scalar |
| 13 | Compute normalized weights | $\lambda_{\text{L1}} = \frac{\text{l1\_weight}}{\text{flux}}$, $\lambda_{\text{TV}} = \frac{\text{tv\_weight} \cdot N_{\text{pix}}}{\text{flux}}$, $w_{\text{logdet}} = \frac{\text{logdet\_weight}}{0.5 \cdot \sum M}$ | l1\_weight, tv\_weight, logdet\_weight, flux, $M$ | $\lambda_{\text{L1}}$, $\lambda_{\text{TV}}$, $w_{\text{logdet}}$: scalars |
| 14 | Compute mask mean | $\bar{M} = \frac{1}{N_{\text{pix}}^2 \cdot 2} \sum_{u,v,c} M_{u,v,c}$ | $M$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | $\bar{M}$: scalar |
| 15 | Transfer to device | $M_{\text{tensor}} = \text{torch.Tensor}(M).\text{to}(\text{device})$, $y_{\text{tensor}} = \text{torch.Tensor}(y).\text{to}(\text{device})$ | $M$, $y$: numpy arrays | $M_{\text{tensor}}$, $y_{\text{tensor}}$: torch tensors |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize RealNVP generator | $G_\theta = \text{RealNVP}(N_{\text{pix}}^2, N_{\text{flow}}, \text{affine=True})$ | $N_{\text{pix}}$: scalar, $N_{\text{flow}}$: scalar | $G_\theta$: nn.Module with parameters $\theta$ | Creates $N_{\text{flow}}$ affine coupling layers |
| 2 | Initialize log-scale parameter | $s^{(0)} = \log\left(\frac{\text{flux}}{0.8 \cdot N_{\text{pix}}^2}\right)$ | flux: scalar, $N_{\text{pix}}$: scalar | $s$: $(1,)$ learnable parameter | Heuristic initialization based on expected intensity |
| 3 | Initialize Adam optimizer | $\text{optimizer} = \text{Adam}(\{\theta, s\}, \text{lr}=\eta)$ | $\theta$: parameters, $s$: parameter, $\eta$: scalar | optimizer: Optimizer object | Optimizes all generator and scale parameters jointly |
| 4 | Initialize loss history | $\mathcal{H} = []$ | None | $\mathcal{H}$: empty list | Stores loss values for monitoring |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{epoch}} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 5 | Sample latent codes | $z^{(k)}_{b,d} \sim \mathcal{N}(0, 1)$ for $b \in [0, N_{\text{batch}})$, $d \in [0, N_{\text{pix}}^2)$ | $N_{\text{batch}}$, $N_{\text{pix}}$ | $z^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}^2)$ | Independent standard normal samples |
| 6 | Generate images via flow [INVERSE/GENERATIVE] | $(\tilde{x}_{\text{flat}}^{(k)}, \log\det_{\text{flow}}^{(k)}) = G_\theta^{-1}(z^{(k)})$ | $z^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}^2)$, $\theta$ | $\tilde{x}_{\text{flat}}^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}^2)$, $\log\det_{\text{flow}}^{(k)}$: $(N_{\text{batch}},)$ | RealNVP inverse mapping with Jacobian |
| 7 | Reshape to image | $\tilde{x}^{(k)}_{b,i,j} = \tilde{x}_{\text{flat},b,i \cdot N_{\text{pix}} + j}^{(k)}$ | $\tilde{x}_{\text{flat}}^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}^2)$ | $\tilde{x}^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | Reshape flattened output to spatial grid |
| 8 | Compute scale factor | $\alpha^{(k)} = \exp(s^{(k)})$ | $s^{(k)}$: $(1,)$ | $\alpha^{(k)}$: scalar | Exponentiate log-scale parameter |
| 9 | Apply Softplus and scaling | $x^{(k)}_{b,i,j} = \log(1 + \exp(\tilde{x}^{(k)}_{b,i,j})) \cdot \alpha^{(k)}$ | $\tilde{x}^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$, $\alpha^{(k)}$: scalar | $x^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | Enforces positivity and scales intensity |
| 10 | Compute Softplus log-det correction | $\log\det_{\text{softplus},b}^{(k)} = \sum_{i,j} \left( \tilde{x}^{(k)}_{b,i,j} - \log(1 + \exp(\tilde{x}^{(k)}_{b,i,j})) \right)$ | $\tilde{x}^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | $\log\det_{\text{softplus}}^{(k)}$: $(N_{\text{batch}},)$ | Jacobian of Softplus: $\frac{\partial}{\partial \tilde{x}} \text{Softplus}(\tilde{x}) = \sigma(\tilde{x})$, so $\log|\det| = \sum \log \sigma(\tilde{x}) = \sum(\tilde{x} - \text{Softplus}(\tilde{x}))$ |
| 11 | Compute scale log-det correction | $\log\det_{\text{scale}}^{(k)} = s^{(k)} \cdot N_{\text{pix}}^2$ | $s^{(k)}$: $(1,)$, $N_{\text{pix}}$: scalar | $\log\det_{\text{scale}}^{(k)}$: scalar | Log-det of uniform scaling by $\exp(s)$ |
| 12 | Compute total log-determinant | $\log\det_{\text{total},b}^{(k)} = \log\det_{\text{flow},b}^{(k)} + \log\det_{\text{softplus},b}^{(k)} + \log\det_{\text{scale}}^{(k)}$ | $\log\det_{\text{flow}}^{(k)}$, $\log\det_{\text{softplus}}^{(k)}$, $\log\det_{\text{scale}}^{(k)}$ | $\log\det_{\text{total}}^{(k)}$: $(N_{\text{batch}},)$ | Chain rule for composed transformations |
| 13 | Compute 2D FFT [FORWARD] | $\hat{x}^{(k)}_{\text{complex},b,u,v} = \frac{1}{N_{\text{pix}}} \sum_{i,j} x^{(k)}_{b,i,j} \exp\left(-2\pi \imath \left(\frac{ui}{N_{\text{pix}}} + \frac{vj}{N_{\text{pix}}}\right)\right)$ | $x^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | $\hat{x}^{(k)}_{\text{complex}}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ complex | Orthonormalized 2D FFT (norm="ortho") |
| 14 | Stack real/imaginary | $\hat{x}^{(k)}_{b,u,v,0} = \Re(\hat{x}^{(k)}_{\text{complex},b,u,v})$, $\hat{x}^{(k)}_{b,u,v,1} = \Im(\hat{x}^{(k)}_{\text{complex},b,u,v})$ | $\hat{x}^{(k)}_{\text{complex}}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ complex | $\hat{x}^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}}, 2)$ | Separate real and imaginary for loss computation |
| 15 | Apply k-space mask | $\hat{x}^{(k)}_{\text{masked},b,u,v,c} = M_{u,v,c} \cdot \hat{x}^{(k)}_{b,u,v,c}$ | $\hat{x}^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}}, 2)$, $M$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | $\hat{x}^{(k)}_{\text{masked}}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}}, 2)$ | Element-wise multiplication with broadcast |
| 16 | Compute data fidelity loss | $\mathcal{L}_{\text{data},b}^{(k)} = \frac{1}{\sigma^2 \cdot \bar{M}} \cdot \frac{1}{N_{\text{pix}}^2 \cdot 2} \sum_{u,v,c} \left( \hat{x}^{(k)}_{\text{masked},b,u,v,c} - y_{u,v,c} \right)^2$ | $\hat{x}^{(k)}_{\text{masked}}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}}, 2)$, $y$: $(N_{\text{pix}}, N_{\text{pix}}, 2)$, $\sigma$, $\bar{M}$ | $\mathcal{L}_{\text{data}}^{(k)}$: $(N_{\text{batch}},)$ | L2 loss in k-space, normalized by noise variance and mask fraction |
| 17 | Compute L1 loss | $\mathcal{L}_{\text{L1},b}^{(k)} = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} |x^{(k)}_{b,i,j}|$ | $x^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | $\mathcal{L}_{\text{L1}}^{(k)}$: $(N_{\text{batch}},)$ | Mean absolute value (sparsity prior) |
| 18 | Compute TV loss (vertical) | $\mathcal{L}_{\text{TV-v},b}^{(k)} = \frac{1}{N_{\text{pix}}(N_{\text{pix}}-1)} \sum_{i=1}^{N_{\text{pix}}-1} \sum_{j=0}^{N_{\text{pix}}-1} |x^{(k)}_{b,i,j} - x^{(k)}_{b,i-1,j}|$ | $x^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | $\mathcal{L}_{\text{TV-v}}^{(k)}$: $(N_{\text{batch}},)$ | Vertical finite differences |
| 19 | Compute TV loss (horizontal) | $\mathcal{L}_{\text{TV-h},b}^{(k)} = \frac{1}{N_{\text{pix}}(N_{\text{pix}}-1)} \sum_{i=0}^{N_{\text{pix}}-1} \sum_{j=1}^{N_{\text{pix}}-1} |x^{(k)}_{b,i,j} - x^{(k)}_{b,i,j-1}|$ | $x^{(k)}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | $\mathcal{L}_{\text{TV-h}}^{(k)}$: $(N_{\text{batch}},)$ | Horizontal finite differences |
| 20 | Combine TV loss | $\mathcal{L}_{\text{TV},b}^{(k)} = \mathcal{L}_{\text{TV-v},b}^{(k)} + \mathcal{L}_{\text{TV-h},b}^{(k)}$ | $\mathcal{L}_{\text{TV-v}}^{(k)}$, $\mathcal{L}_{\text{TV-h}}^{(k)}$: $(N_{\text{batch}},)$ | $\mathcal{L}_{\text{TV}}^{(k)}$: $(N_{\text{batch}},)$ | Anisotropic TV |
| 21 | Compute prior loss | $\mathcal{L}_{\text{prior},b}^{(k)} = \lambda_{\text{TV}} \cdot \mathcal{L}_{\text{TV},b}^{(k)} + \lambda_{\text{L1}} \cdot \mathcal{L}_{\text{L1},b}^{(k)}$ | $\mathcal{L}_{\text{TV}}^{(k)}$, $\mathcal{L}_{\text{L1}}^{(k)}$, $\lambda_{\text{TV}}$, $\lambda_{\text{L1}}$ | $\mathcal{L}_{\text{prior}}^{(k)}$: $(N_{\text{batch}},)$ | Weighted sum of regularization