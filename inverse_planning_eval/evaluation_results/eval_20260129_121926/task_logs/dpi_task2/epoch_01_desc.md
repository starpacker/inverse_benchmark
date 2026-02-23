# Algorithmic Specification Document: Deep Probabilistic Imaging for MRI Reconstruction

## 1. Mathematical Formulation

### 1.1 Inverse Problem Statement

Reconstruct an MRI image $\mathbf{x} \in \mathbb{R}^{n \times n}$ from undersampled, noisy k-space measurements $\mathbf{y} \in \mathbb{C}^{n \times n}$.

**Forward Model:**
$$\mathbf{y} = \mathbf{M} \odot \mathcal{F}(\mathbf{x}) + \boldsymbol{\epsilon}$$

where:
- $\mathcal{F}: \mathbb{R}^{n \times n} \to \mathbb{C}^{n \times n}$ is the orthonormalized 2D DFT: $\mathcal{F}(\mathbf{x}) = \frac{1}{n}\text{DFT}_{\text{2D}}(\mathbf{x})$
- $\mathbf{M} \in \{0,1\}^{n \times n}$ is the undersampling mask
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ is complex Gaussian noise
- $\odot$ denotes Hadamard (element-wise) product

### 1.2 Variational Objective Function

The algorithm minimizes the variational free energy over the generative model parameters $\theta$ and scale parameter $s$:

$$\boxed{\mathcal{L}(\theta, s) = \mathbb{E}_{\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \mathcal{L}_{\text{data}}(G_{\theta,s}(\mathbf{z})) + \mathcal{L}_{\text{prior}}(G_{\theta,s}(\mathbf{z})) - w_{\text{logdet}} \cdot \log \left| \det \frac{\partial G_{\theta,s}}{\partial \mathbf{z}} \right| \right]}$$

### 1.3 Loss Component Decomposition

#### 1.3.1 Data Fidelity Term (K-space L2 Loss)
$$\mathcal{L}_{\text{data}}(\hat{\mathbf{x}}) = \frac{1}{\sigma^2 \cdot \bar{M}} \left\| \mathbf{M} \odot \left( \mathcal{F}(\hat{\mathbf{x}}) - \mathbf{y} \right) \right\|_F^2$$

where $\bar{M} = \frac{1}{n^2}\sum_{i,j} M_{i,j}$ is the mean mask value.

#### 1.3.2 Regularization Terms
$$\mathcal{L}_{\text{prior}}(\hat{\mathbf{x}}) = \lambda_{\text{L1}} \cdot \mathcal{R}_{\text{L1}}(\hat{\mathbf{x}}) + \lambda_{\text{TV}} \cdot \mathcal{R}_{\text{TV}}(\hat{\mathbf{x}})$$

**L1 Sparsity Prior:**
$$\mathcal{R}_{\text{L1}}(\hat{\mathbf{x}}) = \frac{1}{n^2} \sum_{i,j} |\hat{x}_{i,j}|$$

**Anisotropic Total Variation:**
$$\mathcal{R}_{\text{TV}}(\hat{\mathbf{x}}) = \frac{1}{n(n-1)} \sum_{i=1}^{n-1} \sum_{j=1}^{n} |\hat{x}_{i+1,j} - \hat{x}_{i,j}| + \frac{1}{n(n-1)} \sum_{i=1}^{n} \sum_{j=1}^{n-1} |\hat{x}_{i,j+1} - \hat{x}_{i,j}|$$

#### 1.3.3 Entropy/Log-Determinant Term
$$\mathcal{L}_{\text{entropy}} = -w_{\text{logdet}} \cdot \log \left| \det \frac{\partial G_{\theta,s}}{\partial \mathbf{z}} \right|$$

### 1.4 Generative Model Structure

The image generator $G_{\theta,s}: \mathbb{R}^{n^2} \to \mathbb{R}^{n \times n}$ is composed as:

$$\boxed{G_{\theta,s}(\mathbf{z}) = s \cdot \text{softplus}\left( \text{reshape}_{n \times n}\left( f_\theta(\mathbf{z}) \right) \right)}$$

where:
- $f_\theta: \mathbb{R}^{n^2} \to \mathbb{R}^{n^2}$ is a RealNVP normalizing flow with $K$ coupling layers
- $\text{softplus}(t) = \log(1 + e^t)$ ensures non-negativity
- $s = e^{\log s}$ is a learnable scale parameter

### 1.5 Jacobian Decomposition

The log-determinant decomposes as:

$$\log \left| \det \frac{\partial G_{\theta,s}}{\partial \mathbf{z}} \right| = \underbrace{\log \left| \det \frac{\partial f_\theta}{\partial \mathbf{z}} \right|}_{\text{Flow Jacobian}} + \underbrace{\sum_{i,j} \left( \tilde{x}_{i,j} - \text{softplus}(\tilde{x}_{i,j}) \right)}_{\text{Softplus Jacobian}} + \underbrace{n^2 \cdot \log s}_{\text{Scale Jacobian}}$$

where $\tilde{\mathbf{x}} = \text{reshape}_{n \times n}(f_\theta(\mathbf{z}))$ is the pre-activation image.

**Derivation of Softplus Jacobian:**
$$\frac{\partial}{\partial t} \text{softplus}(t) = \sigma(t) = \frac{1}{1+e^{-t}}$$
$$\log \sigma(t) = \log \frac{1}{1+e^{-t}} = -\log(1+e^{-t}) = t - \log(1+e^t) = t - \text{softplus}(t)$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | dtype | Initialization | Description |
|----------|--------|-------|-------|----------------|-------------|
| Flow parameters | $\theta$ | (varies by architecture) | float32 | Default RealNVP init | RealNVP neural network weights |
| Log scale | $\log s$ | $(1,)$ | float32 | $\log\left(\frac{\Phi}{0.8 \cdot n^2}\right)$ | Learnable log of scale factor |

### 2.2 Auxiliary/Latent Variables

| Variable | Symbol | Shape | dtype | Distribution | Description |
|----------|--------|-------|-------|--------------|-------------|
| Latent code | $\mathbf{z}$ | $(B, n^2)$ | float32 | $\mathcal{N}(\mathbf{0}, \mathbf{I})$ | Samples from base distribution |
| Pre-activation image | $\tilde{\mathbf{x}}$ | $(B, n, n)$ | float32 | Deterministic | Output of flow before softplus |
| Flow log-det | $\log|\det J_f|$ | $(B,)$ | float32 | Deterministic | Jacobian from RealNVP |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Image size | $n$ | 64 | Spatial dimension (npix) |
| Flow layers | $K$ | 16 | Number of affine coupling layers |
| Learning rate | $\eta$ | $10^{-5}$ | Adam optimizer step size |
| Epochs | $N_{\text{epoch}}$ | 10 | Number of training iterations |
| Batch size | $B$ | 64 | Samples per iteration |
| Noise level | $\sigma$ | $5 \times 10^{-7}$ | K-space noise std |
| Log-det weight (raw) | $w_{\text{logdet}}^{\text{raw}}$ | 1.0 | User-specified logdet weight |
| L1 weight (raw) | $w_{\text{L1}}^{\text{raw}}$ | 0.0 | User-specified L1 weight |
| TV weight (raw) | $w_{\text{TV}}^{\text{raw}}$ | $10^3$ | User-specified TV weight |
| Gradient clip norm | - | $10^{-2}$ | Max gradient norm |
| Center mask size | - | 8 | Half-width of fully-sampled center |

**Normalized Weights (computed at runtime):**
$$\lambda_{\text{L1}} = \frac{w_{\text{L1}}^{\text{raw}}}{\Phi}, \quad \lambda_{\text{TV}} = \frac{w_{\text{TV}}^{\text{raw}} \cdot n}{\Phi}, \quad w_{\text{logdet}} = \frac{w_{\text{logdet}}^{\text{raw}}}{0.5 \cdot \|\mathbf{M}\|_1}$$

where $\Phi = \sum_{i,j} x^{\text{true}}_{i,j}$ is the total flux of the ground truth.

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | dtype | Description |
|----------|--------|-------|-------|-------------|
| Ground truth image | $\mathbf{x}^{\text{true}}$ | $(n, n)$ | float32 | True MRI image (for evaluation) |
| Noisy k-space | $\mathbf{y}$ | $(n, n, 2)$ | float32 | Real/imag stacked measurements |
| Undersampling mask | $\mathbf{M}$ | $(n, n, 2)$ | float32 | Binary mask (stacked for real/imag) |
| Total flux | $\Phi$ | scalar | float32 | $\sum_{i,j} x^{\text{true}}_{i,j}$ |

---

## 3. Preprocessing Pipeline

```
Input: impath (string), maskpath (string), n (int), σ (float)
Output: Preprocessed data dictionary
```

| Step | Operation | Input | Output | Formula/Description |
|------|-----------|-------|--------|---------------------|
| 1 | Load image | `impath` | $\mathbf{x}_{\text{raw}} \in \mathbb{R}^{h \times w}$ | Unpickle `target` field |
| 2 | Resize image | $\mathbf{x}_{\text{raw}}$ | $\mathbf{x}^{\text{true}} \in \mathbb{R}^{n \times n}$ | Bilinear interpolation to $(n, n)$ |
| 3 | Compute k-space | $\mathbf{x}^{\text{true}}$ | $\mathbf{k} \in \mathbb{R}^{n \times n \times 2}$ | $\mathbf{k} = [\Re(\mathcal{F}(\mathbf{x}^{\text{true}})), \Im(\mathcal{F}(\mathbf{x}^{\text{true}}))]$ |
| 4 | Add noise | $\mathbf{k}$ | $\mathbf{y} \in \mathbb{R}^{n \times n \times 2}$ | $\mathbf{y} = \mathbf{k} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2)$ |
| 5 | Load mask | `maskpath` | $\mathbf{M}_{\text{raw}} \in \{0,1\}^{m \times m}$ | Load numpy array |
| 6 | Resize mask | $\mathbf{M}_{\text{raw}}$ | $\mathbf{M}_{\text{resized}} \in \{0,1\}^{n \times n}$ | Nearest-neighbor interpolation (if $m \neq n$) |
| 7 | Enforce center sampling | $\mathbf{M}_{\text{resized}}$ | $\mathbf{M}_{\text{center}}$ | $M_{i,j} = 1$ for $i,j \in [\frac{n}{2}-8, \frac{n}{2}+8)$ |
| 8 | FFT shift mask | $\mathbf{M}_{\text{center}}$ | $\mathbf{M}_{\text{shifted}}$ | $\mathbf{M}_{\text{shifted}} = \text{fftshift}(\mathbf{M}_{\text{center}})$ |
| 9 | Stack mask | $\mathbf{M}_{\text{shifted}}$ | $\mathbf{M} \in \mathbb{R}^{n \times n \times 2}$ | $\mathbf{M} = [\mathbf{M}_{\text{shifted}}, \mathbf{M}_{\text{shifted}}]$ |
| 10 | Compute flux | $\mathbf{x}^{\text{true}}$ | $\Phi \in \mathbb{R}$ | $\Phi = \sum_{i,j} x^{\text{true}}_{i,j}$ |
| 11 | Apply mask to measurements | $\mathbf{y}, \mathbf{M}$ | $\mathbf{y}_{\text{masked}}$ | $\mathbf{y}_{\text{masked}} = \mathbf{M} \odot \mathbf{y}$ |

---

## 4. Algorithm Steps

### 4.1 Initialization Phase

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| I.1 | Initialize flow | $f_\theta \leftarrow \text{RealNVP}(n^2, K, \text{affine=True})$ | $n^2$, $K$ | $\theta^{(0)}$ |
| I.2 | Initialize log-scale | $\log s \leftarrow \log\left(\frac{\Phi}{0.8 \cdot n^2}\right)$ | $\Phi$, $n$ | $(\log s)^{(0)}$ |
| I.3 | Compute normalized weights | $\lambda_{\text{L1}} = \frac{w_{\text{L1}}^{\text{raw}}}{\Phi}$, $\lambda_{\text{TV}} = \frac{w_{\text{TV}}^{\text{raw}} \cdot n}{\Phi}$, $w_{\text{logdet}} = \frac{w_{\text{logdet}}^{\text{raw}}}{0.5 \cdot \|\mathbf{M}\|_1}$ | Hyperparams, $\Phi$, $\mathbf{M}$ | $\lambda_{\text{L1}}, \lambda_{\text{TV}}, w_{\text{logdet}}$ |
| I.4 | Compute mask mean | $\bar{M} = \frac{1}{n^2 \cdot 2} \sum_{i,j,c} M_{i,j,c}$ | $\mathbf{M}$ | $\bar{M}$ |
| I.5 | Initialize optimizer | $\text{Adam}(\{\theta, \log s\}, \eta)$ | $\eta$ | Optimizer state |

### 4.2 Per-Epoch Iteration (for $k = 0, 1, \ldots, N_{\text{epoch}}-1$)

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| E.1 | Sample latent codes | $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{B \times n^2})$ | $B$, $n^2$ | $\mathbf{Z} \in \mathbb{R}^{B \times n^2}$ |
| E.2 | Flow forward (reverse direction) | $(\tilde{\mathbf{X}}_{\text{flat}}, \log|\det J_f|) = f_\theta^{-1}(\mathbf{Z})$ | $\mathbf{Z}$, $\theta$ | $\tilde{\mathbf{X}}_{\text{flat}} \in \mathbb{R}^{B \times n^2}$, $\log|\det J_f| \in \mathbb{R}^B$ |
| E.3 | Reshape | $\tilde{\mathbf{X}} = \text{reshape}(\tilde{\mathbf{X}}_{\text{flat}}, (B, n, n))$ | $\tilde{\mathbf{X}}_{\text{flat}}$ | $\tilde{\mathbf{X}} \in \mathbb{R}^{B \times n \times n}$ |
| E.4 | Compute scale factor | $s = \exp(\log s)$ | $\log s$ | $s \in \mathbb{R}$ |
| E.5 | Apply activation & scale | $\hat{\mathbf{X}} = s \cdot \text{softplus}(\tilde{\mathbf{X}})$ | $\tilde{\mathbf{X}}$, $s$ | $\hat{\mathbf{X}} \in \mathbb{R}^{B \times n \times n}$ |
| E.6 | Softplus log-det | $\log|\det J_{\text{sp}}|_b = \sum_{i,j} \left( \tilde{X}_{b,i,j} - \text{softplus}(\tilde{X}_{b,i,j}) \right)$ | $\tilde{\mathbf{X}}$ | $\log|\det J_{\text{sp}}| \in \mathbb{R}^B$ |
| E.7 | Scale log-det | $\log|\det J_s| = n^2 \cdot \log s$ | $\log s$, $n$ | $\log|\det J_s| \in \mathbb{R}$ |
| E.8 | Total log-det | $\log|\det J|_b = \log|\det J_f|_b + \log|\det J_{\text{sp}}|_b + \log|\det J_s|$ | Steps E.2, E.6, E.7 | $\log|\det J| \in \mathbb{R}^B$ |
| E.9 | Forward operator (FFT) | $\hat{\mathbf{K}}_b = \mathcal{F}(\hat{\mathbf{X}}_b)$ stacked as $[\Re, \Im]$ | $\hat{\mathbf{X}}$ | $\hat{\mathbf{K}} \in \mathbb{R}^{B \times n \times n \times 2}$ |
| E.10 | Apply mask | $\hat{\mathbf{K}}_{\text{masked}} = \mathbf{M} \odot \hat{\mathbf{K}}$ | $\hat{\mathbf{K}}$, $\mathbf{M}$ | $\hat{\mathbf{K}}_{\text{masked}} \in \mathbb{R}^{B \times n \times n \times 2}$ |
| E.11 | Data fidelity loss | $\mathcal{L}_{\text{data},b} = \frac{1}{\sigma^2 \cdot \bar{M}} \cdot \frac{1}{n^2 \cdot 2} \sum_{i,j,c} \left( \hat{K}_{b,i,j,c} - y_{i,j,c} \right)^2$ | $\hat{\mathbf{K}}_{\text{masked}}$, $\mathbf{y}_{\text{masked}}$ | $\mathcal{L}_{\text{data}} \in \mathbb{R}^B$ |
| E.12 | L1 loss (if $\lambda_{\text{L1}} > 0$) | $\mathcal{R}_{\text{L1},b} = \frac{1}{n^2} \sum_{i,j} |\hat{X}_{b,i,j}|$ | $\hat{\mathbf{X}}$ | $\mathcal{R}_{\text{L1}} \in \mathbb{R}^B$ |
| E.13 | TV loss (if $\lambda_{\text{TV}} > 0$) | $\mathcal{R}_{\text{TV},b} = \frac{1}{n(n-1)} \sum_{i,j} |\hat{X}_{b,i+1,j} - \hat{X}_{b,i,j}| + \frac{1}{n(n-1)} \sum_{i,j} |\hat{X}_{b,i,j+1} - \hat{X}_{b,i,j}|$ | $\hat{\mathbf{X}}$ | $\mathcal{R}_{\text{TV}} \in \mathbb{R}^B$ |
| E.14 | Prior loss | $\mathcal{L}_{\text{prior},b} = \lambda_{\text{L1}} \cdot \mathcal{R}_{\text{L1},b} + \lambda_{\text{TV}} \cdot \mathcal{R}_{\text{TV},b}$ | Steps E.12, E.13 | $\mathcal{L}_{\text{prior}} \in \mathbb{R}^B$ |
| E.15 | Total loss | $\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \left[ \mathcal{L}_{\text{data},b} + \mathcal{L}_{\text{prior},b} - w_{\text{logdet}} \cdot \log|\det J|_b \right]$ | Steps E.11, E.14, E.8 | $\mathcal{L} \in \mathbb{R}$ |
| E.16 | Backpropagation | $\nabla_\theta \mathcal{L}, \nabla_{\log s} \mathcal{L} \leftarrow \text{autograd}(\mathcal{L})$ | $\mathcal{L}$ | Gradients |
| E.17 | Gradient clipping | $\text{clip}(\nabla, \|\cdot\|_2 \leq 10^{-2})$ | Gradients | Clipped gradients |
| E.18 | Parameter update | $\theta^{(k+1)}, (\log s)^{(k+1)} \leftarrow \text{Adam\_step}$ | Clipped gradients | Updated params |

### 4.3 Final Reconstruction Generation

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| F.1 | Sample latent codes | $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{B \times n^2})$ | $B=64$, $n^2$ | $\mathbf{Z}$ |
| F.2 | Generate images | $\hat{\mathbf{X}} = s \cdot \text{softplus}(\text{reshape}(f_\theta^{-1}(\mathbf{Z})))$ | $\mathbf{Z}$, $\theta^*$, $s^*$ | $\hat{\mathbf{X}} \in \mathbb{R}^{64 \times n \times n}$ |
| F.3 | Compute mean reconstruction | $\bar{\mathbf{x}} = \frac{1}{64} \sum_{b=1}^{64} \hat{\mathbf{X}}_b$ | $\hat{\mathbf{X}}$ | $\bar{\mathbf{x}} \in \mathbb{R}^{n \times n}$ |

---

## 5. Stopping Criterion

The algorithm terminates when:

$$k \geq N_{\text{epoch}}$$

where $N_{\text{epoch}}$ is the user-specified number of epochs (default: 10).

**Note:** No convergence-based early stopping is implemented. The algorithm runs for exactly $N_{\text{epoch}}$ iterations.

---

## 6. Evaluation Metrics

### 6.1 Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n^2} \sum_{i,j} \left( \bar{x}_{i,j} - x^{\text{true}}_{i,j} \right)^2}$$

where $\bar{\mathbf{x}}$ is the mean reconstruction.

### 6.2 Peak Signal-to-Noise Ratio (PSNR)

$$\text{PSNR} = 20 \cdot \log_{10}\left( \frac{\max_{i,j}(x^{\text{true}}_{i,j})}{\text{RMSE}} \right) \quad \text{[dB]}$$

**Edge case:** If $\text{RMSE} = 0$, then $\text{PSNR} = +\infty$.

### 6.3 Training Loss (monitored per epoch)

$$\mathcal{L}^{(k)} = \frac{1}{B} \sum_{b=1}^{B} \left[ \mathcal{L}_{\text{data},b} + \mathcal{L}_{\text{prior},b} - w_{\text{logdet}} \cdot \log|\det J|_b \right]$$

---

## 7. Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEEP PROBABILISTIC IMAGING                           │
│                         FOR MRI RECONSTRUCTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   z ~ N(0,I)  ──►  RealNVP Flow f_θ  ──►  reshape  ──►  softplus  ──►  ×s  │
│      ↓                    ↓                                           ↓    │
│   (B, n²)           log|det J_f|                                    x̂     │
│                                                                       ↓    │
│                                                              2D-FFT + Mask │
│                                                                       ↓    │
│                                                                   k̂_masked │
│                                                                       ↓    │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  L = L_data(k̂,y) + λ_L1·R_L1(x̂) + λ_TV·R_TV(x̂) - w·log|det J|  │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                       │
│                           Adam Optimizer                                   │
│                          (with grad clip)                                  │
│                                    ↓                                       │
│                            Update θ, log(s)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```