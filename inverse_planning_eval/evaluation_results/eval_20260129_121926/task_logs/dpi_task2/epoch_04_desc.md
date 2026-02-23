## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Deep Probabilistic Imaging for MRI Reconstruction (DPI-MRI)
- **algorithm_family**: Variational Inference with Normalizing Flows
- **summary**: Reconstructs MRI images from undersampled k-space measurements by training a RealNVP normalizing flow generator to minimize a stochastic loss combining k-space data fidelity, image-domain regularization (L1 and TV), and a flow entropy term (log-determinant).

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{\boldsymbol{\theta}, s} \; \mathbb{E}_{\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \mathcal{L}_{\text{data}}(\mathbf{x}(\mathbf{z}; \boldsymbol{\theta}, s)) + \mathcal{L}_{\text{prior}}(\mathbf{x}(\mathbf{z}; \boldsymbol{\theta}, s)) - w_{\text{logdet}} \cdot \log \left| \det \frac{\partial \mathbf{x}}{\partial \mathbf{z}} \right| \right]$$

where $\mathbf{x}(\mathbf{z}; \boldsymbol{\theta}, s) = \text{softplus}(G_{\boldsymbol{\theta}}(\mathbf{z})) \cdot e^{s}$

### data_fidelity_term

$$\mathcal{L}_{\text{data}}(\mathbf{x}) = \frac{1}{\sigma^2 \cdot \bar{M}} \cdot \frac{1}{N_{\text{pix}}^2 \cdot 2} \sum_{u,v,c} \left( M_{u,v,c} \cdot \hat{x}_{u,v,c} - y_{u,v,c} \right)^2$$

where:
- $\hat{\mathbf{x}} = \mathcal{F}_{\text{2D}}(\mathbf{x})$ is the 2D orthonormal FFT with real/imaginary stacked on dimension $c \in \{0,1\}$
- $\bar{M} = \text{mean}(\mathbf{M})$ is the mean of the mask
- $\sigma$ is the noise standard deviation

### regularization_term

**L1 Term:**
$$\mathcal{L}_{L1}(\mathbf{x}) = w_{L1}^{\text{norm}} \cdot \frac{1}{N_{\text{pix}}^2} \sum_{i,j} |x_{i,j}|$$

where $w_{L1}^{\text{norm}} = \frac{w_{L1}}{\Phi}$ and $\Phi = \sum_{i,j} x^{\text{true}}_{i,j}$ is the flux.

**TV Term:**
$$\mathcal{L}_{TV}(\mathbf{x}) = w_{TV}^{\text{norm}} \cdot \left( \frac{1}{(N_{\text{pix}}-1) \cdot N_{\text{pix}}} \sum_{i,j} |x_{i+1,j} - x_{i,j}| + \frac{1}{N_{\text{pix}} \cdot (N_{\text{pix}}-1)} \sum_{i,j} |x_{i,j+1} - x_{i,j}| \right)$$

where $w_{TV}^{\text{norm}} = \frac{w_{TV} \cdot N_{\text{pix}}}{\Phi}$

**Log-Determinant Term (Entropy):**
$$\mathcal{L}_{\text{logdet}} = -w_{\text{logdet}}^{\text{norm}} \cdot \left( \log|\det J_{G}| + \sum_{i,j} \left( \tilde{x}_{i,j} - \log(1 + e^{\tilde{x}_{i,j}}) \right) + s \cdot N_{\text{pix}}^2 \right)$$

where:
- $\tilde{\mathbf{x}} = G_{\boldsymbol{\theta}}(\mathbf{z})$ is the raw flow output before softplus
- $w_{\text{logdet}}^{\text{norm}} = \frac{w_{\text{logdet}}}{0.5 \cdot \sum_{u,v,c} M_{u,v,c}}$

### constraints

$$\mathbf{x} \geq 0 \quad \text{(enforced implicitly via softplus activation)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\theta}$ | RealNVP params | float32 | RealNVP default init | Parameters of RealNVP normalizing flow generator |
| $s$ | $(1,)$ | float32 | $\log\left(\frac{\Phi}{0.8 \cdot N_{\text{pix}}^2}\right)$ | Learnable log-scale factor for image intensity |

### 2.2 Dual Variables

None (unconstrained optimization)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{z}$ | $(N_{\text{batch}}, N_{\text{pix}}^2)$ | float32 | $\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | Latent samples drawn each iteration |
| $\tilde{\mathbf{x}}$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | — | Raw flow output before activation |
| $\log|\det J_G|$ | $(N_{\text{batch}},)$ | float32 | — | Log-determinant from RealNVP |
| $\mathbf{x}$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | — | Generated images after softplus and scaling |
| $\hat{\mathbf{x}}$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | — | K-space of generated images |
| loss_data | $(N_{\text{batch}},)$ | float32 | — | Per-sample data fidelity loss |
| loss_prior | $(N_{\text{batch}},)$ | float32 | — | Per-sample regularization loss |
| logdet | $(N_{\text{batch}},)$ | float32 | — | Per-sample total log-determinant |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{\text{pix}}$ | 64 (default) | Image dimension |
| $N_{\text{batch}}$ | 64 | Number of latent samples per iteration |
| $N_{\text{flow}}$ | 16 (default) | Number of affine coupling layers in RealNVP |
| $N_{\text{epoch}}$ | 10 (default) | Number of training iterations |
| $\eta$ | $10^{-5}$ (default) | Learning rate for Adam optimizer |
| $\sigma$ | $5 \times 10^{-7}$ (default) | K-space noise standard deviation |
| $w_{L1}$ | 0.0 (default) | L1 regularization weight (before normalization) |
| $w_{TV}$ | $10^{3}$ (default) | TV regularization weight (before normalization) |
| $w_{\text{logdet}}$ | 1.0 (default) | Log-determinant weight (before normalization) |
| center_size | 8 | Half-width of fully-sampled center region |
| grad_clip | $10^{-2}$ | Gradient clipping max norm |
| $\beta_1, \beta_2$ | 0.9, 0.999 | Adam optimizer parameters (PyTorch defaults) |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{x}^{\text{true}}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float32 | Ground truth MRI image |
| $\mathbf{y}$ | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | Noisy, masked k-space measurements |
| $\mathbf{M}$ | $(N_{\text{pix}}, N_{\text{pix}}, 2)$ | float32 | K-space undersampling mask (binary, fftshifted) |
| $\Phi$ | scalar | float32 | Flux: $\sum_{i,j} x^{\text{true}}_{i,j}$ |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Load Image | $\mathbf{x}^{\text{true}} = \texttt{pickle.load}(\text{impath})[\text{'target'}]$ | impath (str) | $\mathbf{x}^{\text{true}}_{\text{raw}}$ (variable shape) |
| 1 | Resize Image | $\mathbf{x}^{\text{true}} = \texttt{BilinearResize}(\mathbf{x}^{\text{true}}_{\text{raw}}, N_{\text{pix}})$ | $\mathbf{x}^{\text{true}}_{\text{raw}}$, $N_{\text{pix}}$ | $\mathbf{x}^{\text{true}} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 2 | Compute K-space | $\mathbf{k} = \mathcal{F}_{\text{2D,ortho}}(\mathbf{x}^{\text{true}})$, stack as $[\Re(\mathbf{k}), \Im(\mathbf{k})]$ | $\mathbf{x}^{\text{true}}$ | $\mathbf{k} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}} \times 2}$ |
| 3 | Add Noise | $\mathbf{k}_{\text{noisy}} = \mathbf{k} + \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2)$ | $\mathbf{k}$, $\sigma$ | $\mathbf{k}_{\text{noisy}} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}} \times 2}$ |
| 4 | Load Mask | $\mathbf{M}_{\text{raw}} = \texttt{np.load}(\text{maskpath})$ | maskpath (str) | $\mathbf{M}_{\text{raw}}$ (variable shape) |
| 5 | Resize Mask | $\mathbf{M} = \texttt{NearestResize}(\mathbf{M}_{\text{raw}}, N_{\text{pix}})$ if shape mismatch | $\mathbf{M}_{\text{raw}}$, $N_{\text{pix}}$ | $\mathbf{M} \in \{0,1\}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 6 | Force Center Sampling | $M_{i,j} = 1$ for $i,j \in [N_{\text{pix}}/2 - 8, N_{\text{pix}}/2 + 8)$ | $\mathbf{M}$ | $\mathbf{M}$ (modified) |
| 7 | FFT Shift Mask | $\mathbf{M} = \texttt{fftshift}(\mathbf{M})$ | $\mathbf{M}$ | $\mathbf{M} \in \{0,1\}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 8 | Stack Mask | $\mathbf{M} = \texttt{stack}([\mathbf{M}, \mathbf{M}], \text{axis}=-1)$ | $\mathbf{M}$ | $\mathbf{M} \in \{0,1\}^{N_{\text{pix}} \times N_{\text{pix}} \times 2}$ |
| 9 | Apply Mask to K-space | $\mathbf{y} = \mathbf{M} \odot \mathbf{k}_{\text{noisy}}$ | $\mathbf{M}$, $\mathbf{k}_{\text{noisy}}$ | $\mathbf{y} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}} \times 2}$ |
| 10 | Compute Flux | $\Phi = \sum_{i,j} x^{\text{true}}_{i,j}$ | $\mathbf{x}^{\text{true}}$ | $\Phi \in \mathbb{R}$ |
| 11 | Compute Normalized Weights | $w_{L1}^{\text{norm}} = \frac{w_{L1}}{\Phi}$, $w_{TV}^{\text{norm}} = \frac{w_{TV} \cdot N_{\text{pix}}}{\Phi}$, $w_{\text{logdet}}^{\text{norm}} = \frac{w_{\text{logdet}}}{0.5 \cdot \sum \mathbf{M}}$ | $\Phi$, $\mathbf{M}$, weights | Normalized weights |
| 12 | Compute Mask Mean | $\bar{M} = \texttt{mean}(\mathbf{M})$ | $\mathbf{M}$ | $\bar{M} \in \mathbb{R}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Initialize RealNVP | $G_{\boldsymbol{\theta}} = \texttt{RealNVP}(N_{\text{pix}}^2, N_{\text{flow}}, \text{affine=True})$ | $N_{\text{pix}}$, $N_{\text{flow}}$ | $\boldsymbol{\theta}$ | Normalizing flow with $N_{\text{flow}}$ affine coupling layers |
| Initialize Log-Scale | $s^{(0)} = \log\left(\frac{\Phi}{0.8 \cdot N_{\text{pix}}^2}\right)$ | $\Phi$, $N_{\text{pix}}$ | $s$ | Learnable scalar parameter |
| Initialize Optimizer | $\texttt{opt} = \texttt{Adam}([\boldsymbol{\theta}, s], \eta)$ | $\boldsymbol{\theta}$, $s$, $\eta$ | optimizer state | Uses PyTorch default $\beta_1=0.9$, $\beta_2=0.999$ |
| Move to Device | Transfer $\mathbf{M}$, $\mathbf{y}$, model to GPU | tensors | device tensors | — |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{epoch}} - 1$)

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Sample Latent | $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{N_{\text{batch}} \times N_{\text{pix}}^2})$ | $N_{\text{batch}}$, $N_{\text{pix}}$ | $\mathbf{z} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}}^2}$ | i.i.d. standard normal |
| Flow Reverse Pass | $(\tilde{\mathbf{x}}_{\text{flat}}, \log|\det J_G|) = G_{\boldsymbol{\theta}}.\texttt{reverse}(\mathbf{z})$ | $\mathbf{z}$, $\boldsymbol{\theta}$ | $\tilde{\mathbf{x}}_{\text{flat}} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}}^2}$, $\log|\det J_G| \in \mathbb{R}^{N_{\text{batch}}}$ | [FORWARD] RealNVP inverse transform |
| Reshape | $\tilde{\mathbf{x}} = \texttt{reshape}(\tilde{\mathbf{x}}_{\text{flat}}, (N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}}))$ | $\tilde{\mathbf{x}}_{\text{flat}}$ | $\tilde{\mathbf{x}} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}}}$ | — |
| Compute Scale | $\alpha = e^{s}$ | $s$ | $\alpha \in \mathbb{R}$ | — |
| Apply Softplus & Scale | $x_{b,i,j} = \log(1 + e^{\tilde{x}_{b,i,j}}) \cdot \alpha$ | $\tilde{\mathbf{x}}$, $\alpha$ | $\mathbf{x} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}}}$ | Ensures non-negativity |
| Softplus Jacobian | $J_{\text{sp},b} = \sum_{i,j} \left( \tilde{x}_{b,i,j} - \log(1 + e^{\tilde{x}_{b,i,j}}) \right)$ | $\tilde{\mathbf{x}}$ | $J_{\text{sp}} \in \mathbb{R}^{N_{\text{batch}}}$ | Log-det of softplus Jacobian |
| Scale Jacobian | $J_{\text{sc}} = s \cdot N_{\text{pix}}^2$ | $s$, $N_{\text{pix}}$ | $J_{\text{sc}} \in \mathbb{R}$ | Log-det of uniform scaling |
| Total Log-Det | $\text{logdet}_b = \log|\det J_G|_b + J_{\text{sp},b} + J_{\text{sc}}$ | $\log|\det J_G|$, $J_{\text{sp}}$, $J_{\text{sc}}$ | $\text{logdet} \in \mathbb{R}^{N_{\text{batch}}}$ | — |
| Forward FFT [FORWARD] | $\hat{x}_{b,u,v} = \frac{1}{N_{\text{pix}}} \sum_{i,j} x_{b,i,j} \cdot e^{-2\pi \mathbf{i}(ui + vj)/N_{\text{pix}}}$, stack $[\Re, \Im]$ | $\mathbf{x}$ | $\hat{\mathbf{x}} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}} \times 2}$ | Orthonormal 2D FFT |
| Apply Mask | $\hat{\mathbf{x}}_{\text{masked}} = \mathbf{M} \odot \hat{\mathbf{x}}$ | $\hat{\mathbf{x}}$, $\mathbf{M}$ | $\hat{\mathbf{x}}_{\text{masked}}$ | Element-wise multiplication |
| Data Fidelity Loss | $\ell_{\text{data},b} = \frac{1}{\sigma^2 \cdot \bar{M}} \cdot \frac{1}{N_{\text{pix}}^2 \cdot 2} \sum_{u,v,c} (\hat{x}_{\text{masked},b,u,v,c} - y_{u,v,c})^2$ | $\hat{\mathbf{x}}_{\text{masked}}$, $\mathbf{y}$ | $\ell_{\text{data}} \in \mathbb{R}^{N_{\text{batch}}}$ | — |
| L1 Loss | $\ell_{L1,b} = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} |x_{b,i,j}|$ | $\mathbf{x}$ | $\ell_{L1} \in \mathbb{R}^{N_{\text{batch}}}$ | Computed if $w_{L1} > 0$ |
| TV Loss | $\ell_{TV,b} = \frac{1}{(N_{\text{pix}}-1) N_{\text{pix}}} \sum_{i,j} |x_{b,i+1,j} - x_{b,i,j}| + \frac{1}{N_{\text{pix}}(N_{\text{pix}}-1)} \sum_{i,j} |x_{b,i,j+1} - x_{b,i,j}|$ | $\mathbf{x}$ | $\ell_{TV} \in \mathbb{R}^{N_{\text{batch}}}$ | Computed if $w_{TV} > 0$ |
| Prior Loss | $\ell_{\text{prior},b} = w_{TV}^{\text{norm}} \cdot \ell_{TV,b} + w_{L1}^{\text{norm}} \cdot \ell_{L1,b}$ | $\ell_{TV}$, $\ell_{L1}$, weights | $\ell_{\text{prior}} \in \mathbb{R}^{N_{\text{batch}}}$ | — |
| Total Loss | $\mathcal{L} = \frac{1}{N_{\text{batch}}} \sum_b \left( \ell_{\text{data},b} + \ell_{\text{prior},b} - w_{\text{logdet}}^{\text{norm}} \cdot \text{logdet}_b \right)$ | all losses | $\mathcal{L} \in \mathbb{R}$ | Scalar loss for backprop |
| Zero Gradients | $\nabla_{\boldsymbol{\theta}} = \mathbf{0}$, $\nabla_s = 0$ | — | — | — |
| Backpropagation | Compute $\nabla_{\boldsymbol{\theta}} \mathcal{L}$, $\nabla_s \mathcal{L}$ via autodiff | $\mathcal{L}$ | gradients | — |
| Gradient Clipping | $\nabla \leftarrow \nabla \cdot \min\left(1, \frac{10^{-2}}{\|\nabla\|_2}\right)$ | gradients | clipped gradients | Max norm $= 10^{-2}$ |
| Adam Update | $\boldsymbol{\theta}^{(k+1)}, s^{(k+1)} = \texttt{Adam.step}()$ | clipped gradients, optimizer state | updated params | — |
| Record Loss | Append $\mathcal{L}$ to loss_history | $\mathcal{L}$ | loss_history | — |

**END_LOOP**

### 4.3 FINALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Sample Final Latent | $\mathbf{z}^* \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{64 \times N_{\text{pix}}^2})$ | — | $\mathbf{z}^*$ | No gradient tracking |
| Generate Final Images | $\tilde{\mathbf{x}}^* = G_{\boldsymbol{\theta}}.\texttt{reverse}(\mathbf{z}^*)[0]$ reshaped | $\mathbf{z}^*$, $\boldsymbol{\theta}$ | $\tilde{\mathbf{x}}^*$ | — |
| Apply Final Activation | $\mathbf{x}^* = \text{softplus}(\tilde{\mathbf{x}}^*) \cdot e^s$ | $\tilde{\mathbf{x}}^*$, $s$ | $\mathbf{x}^* \in \mathbb{R}^{64 \times N_{\text{pix}} \times N_{\text{pix}}}$ | Final reconstructions |
| Convert to NumPy | $\texttt{reconstructed} = \mathbf{x}^*.\texttt{cpu().numpy()}$ | $\mathbf{x}^*$ | numpy array | — |
| Compute Mean Reconstruction | $\bar{\mathbf{x}} = \frac{1}{64} \sum_{b=1}^{64} x^*_b$ | $\mathbf{x}^*$ | $\bar{\mathbf{x}} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ | — |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k < N_{\text{epoch}}$
- **parameters**: 
  - $N_{\text{epoch}} = 10$ (default)
- **check_frequency**: Every iteration (loop condition)

---

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| MSE | $\text{MSE} = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} (\bar{x}_{i,j} - x^{\text{true}}_{i,j})^2$ | Mean squared error between mean reconstruction and ground truth |
| RMSE | $\text{RMSE} = \sqrt{\text{MSE}}$ | Root mean squared error |
| PSNR | $\text{PSNR} = 20 \cdot \log_{10}\left(\frac{\max(\mathbf{x}^{\text{true}})}{\text{RMSE}}\right)$ | Peak signal-to-noise ratio in dB; $+\infty$ if MSE $= 0$ |
| Training Loss | $\mathcal{L}^{(k)}$ | Total loss at iteration $k$ (stored in loss_history) |
| K-space Loss | $\frac{1}{N_{\text{batch}}} \sum_b \ell_{\text{data},b}$ | Mean data fidelity term (printed each epoch) |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include the following fields:

1. `algorithm_identification` — object with `algorithm_name`, `algorithm_family`, `summary`
2. `objective_function` — object with `full_expression`, `data_fidelity_term`, `regularization_term`, `constraints`
3. `variables` — object with subsections: `primal_variables`, `dual_variables`, `auxiliary_variables`, `constants`, `observations`
4. `data_preprocessing` — array of step objects with `step_order`, `step_name`, `formula`, `inputs`, `outputs`
5