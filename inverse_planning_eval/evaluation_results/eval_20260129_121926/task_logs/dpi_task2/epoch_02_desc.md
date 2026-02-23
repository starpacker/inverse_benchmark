# Algorithmic Specification Document: Deep Probabilistic Imaging for MRI Reconstruction

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\theta, \log s} \mathcal{L} = \mathbb{E}_{z \sim \mathcal{N}(0, I_{n^2})} \left[ \mathcal{L}_{\text{data}}(x(z; \theta, s)) + \mathcal{L}_{\text{prior}}(x(z; \theta, s)) - w_{\text{logdet}} \cdot \log |J(z; \theta, s)| \right]$$

where the image is generated via:
$$x(z; \theta, s) = s \cdot \text{softplus}(G_\theta(z)), \quad s = \exp(\log s)$$

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}}(x) = \frac{1}{\sigma^2 \cdot \bar{M}} \cdot \frac{1}{B \cdot n \cdot n \cdot 2} \sum_{b,i,j,c} \left( M_{i,j,c} \cdot (\mathcal{F}(x_b))_{i,j,c} - y_{i,j,c} \right)^2$$

where $\bar{M} = \frac{1}{n \cdot n \cdot 2}\sum_{i,j,c} M_{i,j,c}$ is the mean of the mask.

**regularization_term**:
$$\mathcal{L}_{\text{prior}}(x) = w_{L1} \cdot \mathcal{L}_{L1}(x) + w_{TV} \cdot \mathcal{L}_{TV}(x)$$

where:
$$\mathcal{L}_{L1}(x) = \frac{1}{n^2} \sum_{i,j} |x_{i,j}|$$

$$\mathcal{L}_{TV}(x) = \frac{1}{n(n-1)} \sum_{i=1}^{n-1} \sum_{j=0}^{n-1} |x_{i+1,j} - x_{i,j}| + \frac{1}{n(n-1)} \sum_{i=0}^{n-1} \sum_{j=1}^{n-1} |x_{i,j+1} - x_{i,j}|$$

**constraints**:
- $x_{i,j} \geq 0 \quad \forall i,j$ (enforced implicitly via softplus activation)

---

## 2. VARIABLES

### Primal Variables (Optimized)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta$ | primal | model-dependent | float32 | RealNVP default init | Parameters of normalizing flow generator $G_\theta$ |
| $\log s$ | primal | $(1,)$ | float32 | $\log\left(\frac{\Phi}{0.8 \cdot n^2}\right)$ | Learnable log-scale factor |

### Latent Variables (Sampled)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $z$ | primal | $(B, n^2)$ | float32 | $z \sim \mathcal{N}(0, I_{n^2})$ | Latent code samples per iteration |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $y$ | observations | $(n, n, 2)$ | float32 | From preprocessing | Noisy masked k-space measurements (real/imag stacked) |
| $M$ | observations | $(n, n, 2)$ | float32 | From file | Undersampling mask (binary, fft-shifted) |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $n$ | constants | scalar | int | `npix` argument | Image dimension |
| $\sigma$ | constants | scalar | float32 | `sigma` argument | K-space noise standard deviation |
| $\Phi$ | constants | scalar | float32 | $\sum_{i,j} x^{\text{true}}_{i,j}$ | Total flux of ground truth image |
| $w_{L1}$ | constants | scalar | float32 | $\frac{\lambda_{L1}}{\Phi}$ | Normalized L1 weight |
| $w_{TV}$ | constants | scalar | float32 | $\frac{\lambda_{TV} \cdot n}{\Phi}$ | Normalized TV weight |
| $w_{\text{logdet}}$ | constants | scalar | float32 | $\frac{\lambda_{\text{logdet}}}{0.5 \cdot \sum_{i,j,c} M_{i,j,c}}$ | Normalized log-det weight |
| $B$ | constants | scalar | int | 64 | Batch size |
| $n_{\text{flow}}$ | constants | scalar | int | `n_flow` argument | Number of flow layers in RealNVP |

---

## 3. PREPROCESSING PIPELINE

### Step 1: Image Loading
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load Target Image |
| formula | $x^{\text{raw}} = \text{unpickle}(\texttt{impath})[\text{'target'}]$ |
| inputs | `impath`: string (file path) |
| outputs | $x^{\text{raw}} \in \mathbb{R}^{n_{\text{orig}} \times n_{\text{orig}}}$ |

### Step 2: Image Resizing
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Bilinear Resize |
| formula | $x^{\text{true}}_{i,j} = \sum_{k,l} w_{i,j,k,l} \cdot x^{\text{raw}}_{k,l}$ where $w$ are bilinear interpolation weights |
| inputs | $x^{\text{raw}} \in \mathbb{R}^{n_{\text{orig}} \times n_{\text{orig}}}$, target size $n$ |
| outputs | $x^{\text{true}} \in \mathbb{R}^{n \times n}$ |

### Step 3: K-space Computation
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | 2D FFT with Orthonormal Normalization |
| formula | $\tilde{k}_{u,v} = \frac{1}{n} \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} x^{\text{true}}_{i,j} \exp\left(-2\pi \imath \left(\frac{ui}{n} + \frac{vj}{n}\right)\right)$ |
| inputs | $x^{\text{true}} \in \mathbb{R}^{n \times n}$ |
| outputs | $\tilde{k} \in \mathbb{C}^{n \times n}$ |

### Step 4: Real/Imaginary Stacking
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Complex to Real Representation |
| formula | $k^{\text{clean}}_{i,j,0} = \text{Re}(\tilde{k}_{i,j}), \quad k^{\text{clean}}_{i,j,1} = \text{Im}(\tilde{k}_{i,j})$ |
| inputs | $\tilde{k} \in \mathbb{C}^{n \times n}$ |
| outputs | $k^{\text{clean}} \in \mathbb{R}^{n \times n \times 2}$ |

### Step 5: Noise Addition
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Gaussian Noise Corruption |
| formula | $k^{\text{noisy}}_{i,j,c} = k^{\text{clean}}_{i,j,c} + \epsilon_{i,j,c}, \quad \epsilon_{i,j,c} \sim \mathcal{N}(0, \sigma^2)$ |
| inputs | $k^{\text{clean}} \in \mathbb{R}^{n \times n \times 2}$, noise level $\sigma$ |
| outputs | $k^{\text{noisy}} \in \mathbb{R}^{n \times n \times 2}$ |

### Step 6: Mask Loading and Resizing
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Load and Resize Mask |
| formula | $M^{\text{raw}} = \text{load}(\texttt{maskpath})$; resize via nearest neighbor if needed |
| inputs | `maskpath`: string, target size $n$ |
| outputs | $M^{\text{resize}} \in \{0,1\}^{n \times n}$ |

### Step 7: Center Fully Sampled Region
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Force Center Sampling |
| formula | $M^{\text{center}}_{i,j} = \begin{cases} 1 & \text{if } |i - n/2| < 8 \text{ and } |j - n/2| < 8 \\ M^{\text{resize}}_{i,j} & \text{otherwise} \end{cases}$ |
| inputs | $M^{\text{resize}} \in \{0,1\}^{n \times n}$ |
| outputs | $M^{\text{center}} \in \{0,1\}^{n \times n}$ |

### Step 8: FFT Shift Mask
| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | FFT Shift |
| formula | $M^{\text{shift}}_{i,j} = M^{\text{center}}_{(i + n/2) \mod n, (j + n/2) \mod n}$ |
| inputs | $M^{\text{center}} \in \{0,1\}^{n \times n}$ |
| outputs | $M^{\text{shift}} \in \{0,1\}^{n \times n}$ |

### Step 9: Stack Mask for Real/Imag
| Field | Value |
|-------|-------|
| step_order | 9 |
| step_name | Duplicate Mask Channels |
| formula | $M_{i,j,0} = M_{i,j,1} = M^{\text{shift}}_{i,j}$ |
| inputs | $M^{\text{shift}} \in \{0,1\}^{n \times n}$ |
| outputs | $M \in \{0,1\}^{n \times n \times 2}$ |

### Step 10: Masked K-space Observation
| Field | Value |
|-------|-------|
| step_order | 10 |
| step_name | Apply Mask to K-space |
| formula | $y_{i,j,c} = M_{i,j,c} \cdot k^{\text{noisy}}_{i,j,c}$ |
| inputs | $M \in \{0,1\}^{n \times n \times 2}$, $k^{\text{noisy}} \in \mathbb{R}^{n \times n \times 2}$ |
| outputs | $y \in \mathbb{R}^{n \times n \times 2}$ |

### Step 11: Compute Flux
| Field | Value |
|-------|-------|
| step_order | 11 |
| step_name | Total Intensity |
| formula | $\Phi = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} x^{\text{true}}_{i,j}$ |
| inputs | $x^{\text{true}} \in \mathbb{R}^{n \times n}$ |
| outputs | $\Phi \in \mathbb{R}$ (scalar) |

---

## 4. ALGORITHM STEPS

### Initialization

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| I.1 | Initialize Generator | $G_\theta \leftarrow \text{RealNVP}(n^2, n_{\text{flow}}, \text{affine}=\text{True})$ | $n$, $n_{\text{flow}}$ | $\theta$ | Normalizing flow with $n_{\text{flow}}$ coupling layers |
| I.2 | Initialize Log Scale | $\log s \leftarrow \log\left(\frac{\Phi}{0.8 \cdot n^2}\right)$ | $\Phi$, $n$ | $\log s$ | Initial scale based on flux |
| I.3 | Compute Normalized Weights | $w_{L1} = \frac{\lambda_{L1}}{\Phi}$, $w_{TV} = \frac{\lambda_{TV} \cdot n}{\Phi}$, $w_{\text{logdet}} = \frac{\lambda_{\text{logdet}}}{0.5 \sum M}$ | $\lambda_{L1}, \lambda_{TV}, \lambda_{\text{logdet}}, \Phi, n, M$ | $w_{L1}, w_{TV}, w_{\text{logdet}}$ | Normalization for scale invariance |
| I.4 | Compute Mask Mean | $\bar{M} = \frac{1}{n \cdot n \cdot 2} \sum_{i,j,c} M_{i,j,c}$ | $M$ | $\bar{M}$ | For loss normalization |
| I.5 | Initialize Optimizer | $\text{opt} \leftarrow \text{Adam}(\{\theta, \log s\}, \eta)$ | Learning rate $\eta$ | Optimizer state | Adam optimizer |
| I.6 | Initialize Epoch Counter | $k \leftarrow 0$ | - | $k$ | - |

---

### Iteration Loop

**BEGIN ITERATION LOOP** (for $k = 0, 1, \ldots, K-1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | Sample Latent Codes | $z^{(b)} \sim \mathcal{N}(0, I_{n^2}) \quad \forall b \in \{1, \ldots, B\}$ | $B$, $n$ | $z \in \mathbb{R}^{B \times n^2}$ | i.i.d. standard normal |
| L.2 | Generate via Flow (Reverse) | $(u, \log|J_{\text{flow}}|) = G_\theta^{-1}(z)$ | $z$, $\theta$ | $u \in \mathbb{R}^{B \times n^2}$, $\log|J_{\text{flow}}| \in \mathbb{R}^B$ | RealNVP inverse mapping |
| L.3 | Reshape to Image | $u^{\text{img}}_{b,i,j} = u_{b, i \cdot n + j}$ | $u \in \mathbb{R}^{B \times n^2}$ | $u^{\text{img}} \in \mathbb{R}^{B \times n \times n}$ | Reshape flat vector to 2D |
| L.4 | Compute Scale Factor | $s = \exp(\log s)$ | $\log s$ | $s \in \mathbb{R}^+$ | Exponentiate log-scale |
| L.5 | Apply Softplus Activation | $\tilde{x}_{b,i,j} = \log(1 + \exp(u^{\text{img}}_{b,i,j}))$ | $u^{\text{img}}$ | $\tilde{x} \in \mathbb{R}^{B \times n \times n}$ | Ensures positivity |
| L.6 | Apply Scale | $x_{b,i,j} = s \cdot \tilde{x}_{b,i,j}$ | $\tilde{x}$, $s$ | $x \in \mathbb{R}^{B \times n \times n}$ | Scale to flux magnitude |
| L.7 | Softplus Jacobian Correction | $\Delta_{\text{sp},b} = \sum_{i,j} \left( u^{\text{img}}_{b,i,j} - \log(1 + \exp(u^{\text{img}}_{b,i,j})) \right)$ | $u^{\text{img}}$ | $\Delta_{\text{sp}} \in \mathbb{R}^B$ | Log-det contribution from softplus |
| L.8 | Scale Jacobian Correction | $\Delta_{\text{scale}} = n^2 \cdot \log s$ | $\log s$, $n$ | $\Delta_{\text{scale}} \in \mathbb{R}$ | Log-det contribution from scaling |
| L.9 | Total Log-Determinant | $\log|J|_b = \log|J_{\text{flow}}|_b + \Delta_{\text{sp},b} + \Delta_{\text{scale}}$ | $\log|J_{\text{flow}}|$, $\Delta_{\text{sp}}$, $\Delta_{\text{scale}}$ | $\log|J| \in \mathbb{R}^B$ | Combined Jacobian |
| L.10 | Forward FFT (per sample) | $\tilde{k}^{(b)}_{u,v} = \frac{1}{n} \sum_{i,j} x_{b,i,j} \exp\left(-2\pi\imath\left(\frac{ui}{n} + \frac{vj}{n}\right)\right)$ | $x$ | $\tilde{k} \in \mathbb{C}^{B \times n \times n}$ | Ortho-normalized 2D FFT |
| L.11 | Complex to Real | $k^{\text{pred}}_{b,i,j,0} = \text{Re}(\tilde{k}^{(b)}_{i,j})$, $k^{\text{pred}}_{b,i,j,1} = \text{Im}(\tilde{k}^{(b)}_{i,j})$ | $\tilde{k}$ | $k^{\text{pred}} \in \mathbb{R}^{B \times n \times n \times 2}$ | Stack real/imag |
| L.12 | Apply Mask to Prediction | $\hat{k}_{b,i,j,c} = M_{i,j,c} \cdot k^{\text{pred}}_{b,i,j,c}$ | $k^{\text{pred}}$, $M$ | $\hat{k} \in \mathbb{R}^{B \times n \times n \times 2}$ | Masked k-space |
| L.13 | Data Fidelity Loss | $\ell^{\text{data}}_b = \frac{1}{\sigma^2 \cdot \bar{M}} \cdot \frac{1}{n \cdot n \cdot 2} \sum_{i,j,c} (\hat{k}_{b,i,j,c} - y_{i,j,c})^2$ | $\hat{k}$, $y$, $\sigma$, $\bar{M}$ | $\ell^{\text{data}} \in \mathbb{R}^B$ | Per-sample data loss |
| L.14 | L1 Loss (if $w_{L1} > 0$) | $\ell^{L1}_b = \frac{1}{n^2} \sum_{i,j} |x_{b,i,j}|$ | $x$ | $\ell^{L1} \in \mathbb{R}^B$ | Sparsity term |
| L.15 | TV Loss (if $w_{TV} > 0$) | $\ell^{TV}_b = \frac{1}{n(n-1)} \sum_{i=0}^{n-2} \sum_j |x_{b,i+1,j} - x_{b,i,j}| + \frac{1}{n(n-1)} \sum_i \sum_{j=0}^{n-2} |x_{b,i,j+1} - x_{b,i,j}|$ | $x$ | $\ell^{TV} \in \mathbb{R}^B$ | Smoothness term |
| L.16 | Prior Loss | $\ell^{\text{prior}}_b = w_{TV} \cdot \ell^{TV}_b + w_{L1} \cdot \ell^{L1}_b$ | $\ell^{TV}$, $\ell^{L1}$, $w_{TV}$, $w_{L1}$ | $\ell^{\text{prior}} \in \mathbb{R}^B$ | Combined regularization |
| L.17 | Total Loss | $\mathcal{L} = \frac{1}{B} \sum_b \left( \ell^{\text{data}}_b + \ell^{\text{prior}}_b - w_{\text{logdet}} \cdot \log|J|_b \right)$ | $\ell^{\text{data}}$, $\ell^{\text{prior}}$, $\log|J|$, $w_{\text{logdet}}$ | $\mathcal{L} \in \mathbb{R}$ | Scalar loss |
| L.18 | Compute Gradients | $g_\theta = \nabla_\theta \mathcal{L}$, $g_{\log s} = \nabla_{\log s} \mathcal{L}$ | $\mathcal{L}$, $\theta$, $\log s$ | $g_\theta$, $g_{\log s}$ | Backpropagation |
| L.19 | Gradient Clipping | $g \leftarrow g \cdot \min\left(1, \frac{0.01}{\|g\|_2}\right)$ | $g_\theta$, $g_{\log s}$ | Clipped gradients | Max norm = 0.01 |
| L.20 | Adam Update | $\theta \leftarrow \theta - \eta \cdot \text{Adam}(g_\theta)$, $\log s \leftarrow \log s - \eta \cdot \text{Adam}(g_{\log s})$ | Gradients, $\eta$, optimizer state | Updated $\theta$, $\log s$ | - |
| L.21 | Increment Counter | $k \leftarrow k + 1$ | $k$ | $k$ | - |
| L.22 | Check Stopping Criterion | Continue if $k < K$ | $k$, $K$ | Boolean | Iteration limit check |

**END ITERATION LOOP**

---

### Post-Processing (Final Reconstruction)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| P.1 | Sample Final Latent | $z^{(b)} \sim \mathcal{N}(0, I_{n^2}) \quad \forall b \in \{1, \ldots, B\}$ | $B$, $n$ | $z \in \mathbb{R}^{B \times n^2}$ | Fresh samples |
| P.2 | Generate Final Images | $x^{\text{final}} = \exp(\log s) \cdot \text{softplus}(G_\theta^{-1}(z))$ | $z$, $\theta$, $\log s$ | $x^{\text{final}} \in \mathbb{R}^{B \times n \times n}$ | Apply learned model |

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $k \geq K$ |
| parameters | $K = n_{\text{epoch}}$ (default: 10) |

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Mean Reconstruction | $\bar{x}_{i,j} = \frac{1}{B} \sum_{b=1}^{B} x^{\text{final}}_{b,i,j}$ | Posterior mean estimate |
| Mean Squared Error | $\text{MSE} = \frac{1}{n^2} \sum_{i,j} (\bar{x}_{i,j} - x^{\text{true}}_{i,j})^2$ | Average squared pixel error |
| Root Mean Squared Error | $\text{RMSE} = \sqrt{\text{MSE}}$ | Standard error measure |
| Peak Signal-to-Noise Ratio | $\text{PSNR} = 20 \cdot \log_{10}\left(\frac{\max_{i,j}(x^{\text{true}}_{i,j})}{\text{RMSE}}\right)$ | Logarithmic quality metric (dB) |