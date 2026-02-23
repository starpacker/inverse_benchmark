# Algorithmic Specification Document: Deep Probabilistic Imaging for Radio Interferometry

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{\theta, s} \mathcal{L}(\theta, s) = \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ \mathcal{L}_{\text{data}}(I_{\theta,s}(z)) + \mathcal{L}_{\text{prior}}(I_{\theta,s}(z)) - w_{\text{logdet}} \cdot \log \left| \det \frac{\partial I_{\theta,s}}{\partial z} \right| \right]$$

where $I_{\theta,s}(z)$ is the image generated from latent code $z$ via normalizing flow with parameters $\theta$ and scale parameter $s$.

### data_fidelity_term

$$\mathcal{L}_{\text{data}} = w_{\text{cp}} \cdot \mathcal{L}_{\text{cphase}}$$

$$\mathcal{L}_{\text{cphase}} = \frac{2}{N_{\text{cp}}} \sum_{j=1}^{N_{\text{cp}}} \frac{1 - \cos(\phi_j^{\text{true}} - \phi_j^{\text{pred}})}{\sigma_{\text{cp},j}^2}$$

where $\phi_j$ are closure phases in radians and $\sigma_{\text{cp},j}$ are closure phase uncertainties in radians.

### regularization_term

$$\mathcal{L}_{\text{prior}} = w_{\text{ce}} \cdot \mathcal{L}_{\text{cross-entropy}} + w_{\text{flux}} \cdot \mathcal{L}_{\text{flux}}$$

$$\mathcal{L}_{\text{cross-entropy}} = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} I_{i,j} \left( \ln(I_{i,j} + \epsilon) - \ln(I_{i,j}^{\text{prior}} + \epsilon) \right)$$

$$\mathcal{L}_{\text{flux}} = \left( \sum_{i,j} I_{i,j} - F_{\text{target}} \right)^2$$

### constraints

- $I_{i,j} > 0 \quad \forall i,j$ (enforced via softplus activation)
- $z \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$ (latent space prior)

---

## 2. VARIABLES

### Primal Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta$ | primal | varies | float32 | Xavier/Kaiming | RealNVP normalizing flow parameters |
| $s$ | primal | $(1,)$ | float32 | $\ln\left(\frac{F_{\text{target}}}{0.8 \cdot N_{\text{pix}}^2}\right)$ | Log-scale parameter for image intensity |
| $z$ | primal | $(N_{\text{batch}}, N_{\text{pix}}^2)$ | float32 | $\mathcal{N}(0,1)$ | Latent space samples |

### Intermediate Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\tilde{I}$ | primal | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | computed | Pre-activation image from flow |
| $I$ | primal | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | computed | Final reconstructed image |
| $V$ | primal | $(N_{\text{batch}}, 2, N_{\text{vis}})$ | float32 | computed | Complex visibilities (real, imag) |
| $\|V\|$ | primal | $(N_{\text{batch}}, N_{\text{vis}})$ | float32 | computed | Visibility amplitudes |
| $\phi^{\text{pred}}$ | primal | $(N_{\text{batch}}, N_{\text{cp}})$ | float32 | computed | Predicted closure phases |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $F$ | constants | $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ | float32 | from DFT | Discrete Fourier Transform matrix |
| $w_{\text{cp}}$ | constants | $(1,)$ | float32 | $N_{\text{cp}} / N_{\text{camp}}$ | Closure phase loss weight |
| $w_{\text{ce}}$ | constants | $(1,)$ | float32 | $1024.0$ | Cross-entropy regularization weight |
| $w_{\text{flux}}$ | constants | $(1,)$ | float32 | $1000.0$ | Flux constraint weight |
| $w_{\text{logdet}}$ | constants | $(1,)$ | float32 | $2 \cdot \lambda / N_{\text{camp}}$ | Log-determinant weight |
| $I^{\text{prior}}$ | constants | $(N_{\text{pix}}, N_{\text{pix}})$ | float32 | Gaussian | Prior image (Gaussian blob) |
| $F_{\text{target}}$ | constants | $(1,)$ | float32 | from data | Target total flux |
| $\epsilon$ | constants | $(1,)$ | float32 | $10^{-12}$ | Numerical stability constant |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $(u, v)$ | observations | $(N_{\text{vis}}, 2)$ | float64 | from UVFITS | Baseline coordinates |
| $\phi^{\text{true}}$ | observations | $(N_{\text{cp}},)$ | float32 | from data | Observed closure phases (degrees) |
| $\sigma_{\text{cp}}$ | observations | $(N_{\text{cp}},)$ | float32 | from data | Closure phase uncertainties (degrees) |
| $\mathcal{I}_1, \mathcal{I}_2, \mathcal{I}_3$ | observations | $(N_{\text{cp}},)$ | int64 | computed | Closure phase baseline indices |
| $s_1, s_2, s_3$ | observations | $(N_{\text{cp}},)$ | float32 | computed | Closure phase baseline signs |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Load Observation Data

| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load UVFITS |
| formula | $\text{obs} = \text{load\_uvfits}(\text{path})$ |
| inputs | path: string |
| outputs | obs: observation object containing $(u, v, V^{\text{obs}}, \sigma)$ |

### Step 3.2: Extract Flux Constant

| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Compute Reference Flux |
| formula | $F_{\text{target}} = \text{median}\left( \|V_{\text{APEX-ALMA}}\| \right)$ |
| inputs | obs: observation object |
| outputs | $F_{\text{target}}$: scalar float32 |

### Step 3.3: Create Prior Image

| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Generate Gaussian Prior |
| formula | $I^{\text{prior}}_{i,j} = \frac{F_{\text{target}}}{2\pi \sigma_x \sigma_y} \exp\left( -\frac{(x_i - x_0)^2}{2\sigma_x^2} - \frac{(y_j - y_0)^2}{2\sigma_y^2} \right)$ |
| inputs | $F_{\text{target}}$: scalar, $\sigma_x = \sigma_y = \text{FWHM}_{\text{prior}} / (2\sqrt{2\ln 2})$, $(x_0, y_0) = (0, 0)$ |
| outputs | $I^{\text{prior}}$: $(N_{\text{pix}}, N_{\text{pix}})$ float32 |

### Step 3.4: Compute DFT Matrix

| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Generate Fourier Matrix |
| formula | $F_{k,m} = \exp\left( -2\pi i (u_k x_m + v_k y_m) \right)$ |
| inputs | $(u, v)$: $(N_{\text{vis}}, 2)$, pixel coordinates $(x_m, y_m)$ where $x_m = (m \mod N_{\text{pix}} - N_{\text{pix}}/2) \cdot \Delta_{\text{pix}}$, $y_m = (\lfloor m / N_{\text{pix}} \rfloor - N_{\text{pix}}/2) \cdot \Delta_{\text{pix}}$ |
| outputs | $F$: $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ float32 where last dimension is (real, imag) |

### Step 3.5: Compute Closure Phase Mapping

| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Build Closure Phase Index Arrays |
| formula | For each closure phase triangle $(t_1, t_2, t_3)$: find baseline indices $\mathcal{I}_1, \mathcal{I}_2, \mathcal{I}_3$ and signs $s_1, s_2, s_3 \in \{-1, +1\}$ such that $\phi_{\text{cp}} = s_1 \phi_{12} + s_2 \phi_{23} + s_3 \phi_{31}$ |
| inputs | obs.cphase: closure phase table, obs.data: visibility table |
| outputs | $\mathcal{I}_1, \mathcal{I}_2, \mathcal{I}_3$: $(N_{\text{cp}},)$ int64, $s_1, s_2, s_3$: $(N_{\text{cp}},)$ float32 |

### Step 3.6: Extract Closure Phase Observations

| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Extract Observed Closure Phases |
| formula | $\phi^{\text{true}}_j = \text{obs.cphase}[j]$, $\sigma_{\text{cp},j} = \text{obs.sigmacp}[j]$ |
| inputs | obs: observation object |
| outputs | $\phi^{\text{true}}$: $(N_{\text{cp}},)$ float32 (degrees), $\sigma_{\text{cp}}$: $(N_{\text{cp}},)$ float32 (degrees) |

---

## 4. ALGORITHM STEPS

### Initialization

**BEGIN INITIALIZATION**

#### Step 4.1: Initialize Normalizing Flow

| Field | Value |
|-------|-------|
| step_name | Create RealNVP Model |
| formula | $g_\theta: \mathbb{R}^{N_{\text{pix}}^2} \to \mathbb{R}^{N_{\text{pix}}^2}$ with $n_{\text{flow}}$ affine coupling layers |
| inputs | $N_{\text{pix}}^2$: int, $n_{\text{flow}}$: int |
| outputs | $\theta$: model parameters |
| computational_note | RealNVP implements bijective transformation with tractable Jacobian |

#### Step 4.2: Initialize Scale Parameter

| Field | Value |
|-------|-------|
| step_name | Initialize Log-Scale |
| formula | $s^{(0)} = \ln\left( \frac{F_{\text{target}}}{0.8 \cdot N_{\text{pix}}^2} \right)$ |
| inputs | $F_{\text{target}}$: scalar, $N_{\text{pix}}$: int |
| outputs | $s$: $(1,)$ float32 |
| computational_note | Initial scale chosen so expected pixel value times number of pixels approximates target flux |

#### Step 4.3: Initialize Optimizer

| Field | Value |
|-------|-------|
| step_name | Create Adam Optimizer |
| formula | $\text{optimizer} = \text{Adam}(\{\theta, s\}, \eta)$ |
| inputs | $\theta$: flow parameters, $s$: scale parameter, $\eta$: learning rate |
| outputs | optimizer state |
| computational_note | Adam with default $\beta_1 = 0.9$, $\beta_2 = 0.999$ |

#### Step 4.4: Compute Loss Weights

| Field | Value |
|-------|-------|
| step_name | Set Hyperparameters |
| formula | $w_{\text{cp}} = \frac{N_{\text{cp}}}{N_{\text{camp}}}$, $w_{\text{flux}} = 1000$, $w_{\text{ce}} = 1024$, $w_{\text{logdet}} = \frac{2\lambda}{N_{\text{camp}}}$ |
| inputs | $N_{\text{cp}}$: int, $N_{\text{camp}}$: int, $\lambda$: logdet factor |
| outputs | $w_{\text{cp}}, w_{\text{flux}}, w_{\text{ce}}, w_{\text{logdet}}$: scalars |
| computational_note | Weights balance data fidelity, regularization, and entropy maximization |

**END INITIALIZATION**

---

### Iteration Loop

**BEGIN ITERATION LOOP** (for $k = 0, 1, \ldots, N_{\text{epoch}} - 1$)

#### Step 4.5: Sample Latent Codes

| Field | Value |
|-------|-------|
| step_name | Draw Latent Samples |
| formula | $z^{(b)} \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$ for $b = 1, \ldots, N_{\text{batch}}$ |
| inputs | $N_{\text{batch}} = 32$, $N_{\text{pix}}^2$: int |
| outputs | $z$: $(N_{\text{batch}}, N_{\text{pix}}^2)$ float32 |
| computational_note | Standard normal samples via torch.randn |

#### Step 4.6: Apply Inverse Flow

| Field | Value |
|-------|-------|
| step_name | Generate Pre-Activation Images |
| formula | $(\tilde{I}, \log|\det J_g|) = g_\theta^{-1}(z)$ where $\log|\det J_g| = \sum_{\ell=1}^{n_{\text{flow}}} \sum_i \log|s_\ell^{(i)}|$ |
| inputs | $z$: $(N_{\text{batch}}, N_{\text{pix}}^2)$, $\theta$: flow parameters |
| outputs | $\tilde{I}$: $(N_{\text{batch}}, N_{\text{pix}}^2)$, $\log|\det J_g|$: $(N_{\text{batch}},)$ |
| computational_note | RealNVP reverse pass computes $g^{-1}$ and log-determinant simultaneously |

#### Step 4.7: Apply Positivity Transform

| Field | Value |
|-------|-------|
| step_name | Softplus Activation |
| formula | $I_{i,j} = \text{softplus}(\tilde{I}_{i,j}) \cdot e^s$ where $\text{softplus}(x) = \ln(1 + e^x)$ |
| inputs | $\tilde{I}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$, $s$: scalar |
| outputs | $I$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ float32 |
| computational_note | Ensures $I > 0$ for physical validity |

#### Step 4.8: Update Log-Determinant

| Field | Value |
|-------|-------|
| step_name | Compute Full Jacobian Log-Determinant |
| formula | $\log|\det J| = \log|\det J_g| + \sum_{i,j} \left( \tilde{I}_{i,j} - \text{softplus}(\tilde{I}_{i,j}) \right) + s \cdot N_{\text{pix}}^2$ |
| inputs | $\log|\det J_g|$: $(N_{\text{batch}},)$, $\tilde{I}$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$, $s$: scalar |
| outputs | $\log|\det J|$: $(N_{\text{batch}},)$ |
| computational_note | Chain rule: $\frac{\partial}{\partial x}\text{softplus}(x) = \sigma(x)$, so $\log|\det| = x - \text{softplus}(x) = -\log(1+e^{-x})$ |

#### Step 4.9: Compute Visibilities

| Field | Value |
|-------|-------|
| step_name | Apply DFT Forward Model |
| formula | $V_{\text{real}} = I_{\text{flat}} \cdot F_{:,:,0}$, $V_{\text{imag}} = I_{\text{flat}} \cdot F_{:,:,1}$ |
| inputs | $I$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ reshaped to $(N_{\text{batch}}, N_{\text{pix}}^2)$, $F$: $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ |
| outputs | $V$: $(N_{\text{batch}}, 2, N_{\text{vis}})$ float32 |
| computational_note | Matrix multiplication implements discrete Fourier transform |

#### Step 4.10: Compute Visibility Amplitudes

| Field | Value |
|-------|-------|
| step_name | Compute Amplitudes |
| formula | $\|V_k\| = \sqrt{V_{\text{real},k}^2 + V_{\text{imag},k}^2 + \epsilon}$ where $\epsilon = 10^{-16}$ |
| inputs | $V$: $(N_{\text{batch}}, 2, N_{\text{vis}})$ |
| outputs | $\|V\|$: $(N_{\text{batch}}, N_{\text{vis}})$ float32 |
| computational_note | Small $\epsilon$ for numerical stability in gradient computation |

#### Step 4.11: Compute Closure Phases

| Field | Value |
|-------|-------|
| step_name | Compute Predicted Closure Phases |
| formula | $\phi_j^{\text{pred}} = \frac{180}{\pi} \left( s_{1,j} \cdot \text{atan2}(V_{\text{imag}}[\mathcal{I}_{1,j}], V_{\text{real}}[\mathcal{I}_{1,j}]) + s_{2,j} \cdot \text{atan2}(V_{\text{imag}}[\mathcal{I}_{2,j}], V_{\text{real}}[\mathcal{I}_{2,j}]) + s_{3,j} \cdot \text{atan2}(V_{\text{imag}}[\mathcal{I}_{3,j}], V_{\text{real}}[\mathcal{I}_{3,j}]) \right)$ |
| inputs | $V$: $(N_{\text{batch}}, 2, N_{\text{vis}})$, $\mathcal{I}_1, \mathcal{I}_2, \mathcal{I}_3$: $(N_{\text{cp}},)$, $s_1, s_2, s_3$: $(N_{\text{cp}},)$ |
| outputs | $\phi^{\text{pred}}$: $(N_{\text{batch}}, N_{\text{cp}})$ float32 (degrees) |
| computational_note | atan2 returns angle in $(-\pi, \pi]$; output converted to degrees |

#### Step 4.12: Compute Cross-Entropy Loss

| Field | Value |
|-------|-------|
| step_name | KL-Divergence from Prior |
| formula | $\mathcal{L}_{\text{ce}}^{(b)} = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} I_{i,j}^{(b)} \left( \ln(I_{i,j}^{(b)} + \epsilon) - \ln(I_{i,j}^{\text{prior}} + \epsilon) \right)$ |
| inputs | $I$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$, $I^{\text{prior}}$: $(N_{\text{pix}}, N_{\text{pix}})$, $\epsilon = 10^{-12}$ |
| outputs | $\mathcal{L}_{\text{ce}}$: $(N_{\text{batch}},)$ float32 |
| computational_note | Encourages image to be similar to Gaussian prior in relative entropy sense |

#### Step 4.13: Compute Flux Loss

| Field | Value |
|-------|-------|
| step_name | Flux Constraint Penalty |
| formula | $\mathcal{L}_{\text{flux}}^{(b)} = \left( \sum_{i,j} I_{i,j}^{(b)} - F_{\text{target}} \right)^2$ |
| inputs | $I$: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$, $F_{\text{target}}$: scalar |
| outputs | $\mathcal{L}_{\text{flux}}$: $(N_{\text{batch}},)$ float32 |
| computational_note | Soft constraint on total image flux |

#### Step 4.14: Compute Closure Phase Loss

| Field | Value |
|-------|-------|
| step_name | Closure Phase Chi-Squared |
| formula | $\mathcal{L}_{\text{cp}}^{(b)} = \frac{2}{N_{\text{cp}}} \sum_{j=1}^{N_{\text{cp}}} \frac{1 - \cos\left( \frac{\pi}{180}(\phi_j^{\text{true}} - \phi_j^{\text{pred},(b)}) \right)}{\left( \frac{\pi}{180} \sigma_{\text{cp},j} \right)^2}$ |
| inputs | $\phi^{\text{true}}$: $(N_{\text{cp}},)$ degrees, $\phi^{\text{pred}}$: $(N_{\text{batch}}, N_{\text{cp}})$ degrees, $\sigma_{\text{cp}}$: $(N_{\text{cp}},)$ degrees |
| outputs | $\mathcal{L}_{\text{cp}}$: $(N_{\text{batch}},)$ float32 |
| computational_note | Von Mises-type loss for angular data; avoids phase wrapping issues |

#### Step 4.15: Compute Total Loss

| Field | Value |
|-------|-------|
| step_name | Aggregate Loss Components |
| formula | $\mathcal{L} = \frac{1}{N_{\text{batch}}} \sum_{b=1}^{N_{\text{batch}}} \left[ w_{\text{cp}} \mathcal{L}_{\text{cp}}^{(b)} + w_{\text{ce}} \mathcal{L}_{\text{ce}}^{(b)} + w_{\text{flux}} \mathcal{L}_{\text{flux}}^{(b)} - w_{\text{logdet}} \log|\det J|^{(b)} \right]$ |
| inputs | All component losses and weights |
| outputs | $\mathcal{L}$: scalar float32 |
| computational_note | Negative log-det encourages diversity (entropy maximization) |

#### Step 4.16: Compute Gradients

| Field | Value |
|-------|-------|
| step_name | Backpropagation |
| formula | $\nabla_\theta \mathcal{L}$, $\nabla_s \mathcal{L}$ via automatic differentiation |
| inputs | $\mathcal{L}$: scalar, computational graph |
| outputs | gradients w.r.t. $\theta$ and $s$ |
| computational_note | PyTorch autograd traverses computation graph |

#### Step 4.17: Clip Gradients

| Field | Value |
|-------|-------|
| step_name | Gradient Clipping |
| formula | $\tilde{g} = g \cdot \min\left(1, \frac{c}{\|g\|_2}\right)$ where $c = 0.1$ |
| inputs | gradients $g$, clip threshold $c$ |
| outputs | clipped gradients $\tilde{g}$ |
| computational_note | Prevents exploding gradients; applied to combined parameter set |

#### Step 4.18: Update Parameters

| Field | Value |
|-------|-------|
| step_name | Adam Update |
| formula | $m_t = \beta_1 m_{t-1} + (1-\beta_1) \tilde{g}$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) \tilde{g}^2$, $\hat{m}_t = m_t / (1-\beta_1^t)$, $\hat{v}_t = v_t / (1-\beta_2^t)$, $\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon_{\text{Adam}})$ |
| inputs | $\tilde{g}$: clipped gradients, $\eta$: learning rate, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon_{\text{Adam}} = 10^{-8}$ |
| outputs | updated $\theta$, $s$ |
| computational_note | Same update applied to scale parameter $s$ |

#### Step 4.19: Convergence Check

| Field | Value |
|-------|-------|
| step_name | Check Iteration Count |
| formula | if $k < N_{\text{epoch}} - 1$: continue; else: terminate |
| inputs | $k$: current iteration, $N_{\text{epoch}}$: maximum iterations |
| outputs | continue/terminate signal |
| computational_note | Fixed iteration count; no early stopping criterion implemented |

**END ITERATION LOOP**

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $k \geq N_{\text{epoch}}$ |
| parameters | $N_{\text{epoch}} = 100$ (default), configurable via command line |

---

## 6. EVALUATION METRICS

### Metric 6.1: Mean Reconstruction

| Field | Value |
|-------|-------|
| name | Mean Image |
| formula | $\bar{I}_{i,j} = \frac{1}{N_{\text{samples}}} \sum_{b=1}^{N_{\text{samples}}} I_{i,j}^{(b)}$ |
| description | Posterior mean image computed from $N_{\text{samples}} = 32$ flow samples |

### Metric 6.