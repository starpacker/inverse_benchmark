## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Deep Probabilistic Imaging (DPI) for Radio Interferometry
- **algorithm_family**: Variational Inference with Normalizing Flows
- **summary**: Reconstructs astronomical images from interferometric closure phases by optimizing a RealNVP normalizing flow to produce image samples that minimize a weighted combination of data fidelity (closure phase), entropy regularization, flux constraint, and variational log-determinant terms.

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{\theta, \lambda} \mathcal{L}(\theta, \lambda) = \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ w_{\text{cp}} \mathcal{L}_{\text{cphase}}(x) + w_{\text{ce}} \mathcal{L}_{\text{CE}}(x) + w_{\text{flux}} \mathcal{L}_{\text{flux}}(x) - w_{\text{logdet}} \log \left| \det \frac{\partial x}{\partial z} \right| \right]$$

where $x = \text{Softplus}(G_\theta(z)) \cdot e^{\lambda}$ and $G_\theta$ is the RealNVP generator.

### data_fidelity_term

**Closure Phase Loss (von Mises-like formulation)**:
$$\mathcal{L}_{\text{cphase}}(x) = 2 \cdot \frac{1}{N_{\text{cp}}} \sum_{i=1}^{N_{\text{cp}}} \frac{1 - \cos(\phi_i^{\text{true}} - \phi_i^{\text{pred}}(x))}{\sigma_{\text{cp},i}^2}$$

where:
- $\phi_i^{\text{true}}$ is the observed closure phase in radians
- $\phi_i^{\text{pred}}(x)$ is the predicted closure phase from image $x$
- $\sigma_{\text{cp},i}$ is the closure phase uncertainty in radians

### regularization_term

**Component 1 - Cross-Entropy Regularization**:
$$\mathcal{L}_{\text{CE}}(x) = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} x_{ij} \cdot \left( \log(x_{ij} + 10^{-12}) - \log(p_{ij} + 10^{-12}) \right)$$

where $p$ is the prior image (Gaussian blob).

**Component 2 - Flux Constraint**:
$$\mathcal{L}_{\text{flux}}(x) = \left( \sum_{i,j} x_{ij} - F_0 \right)^2$$

where $F_0$ is the target total flux.

**Component 3 - Variational Log-Determinant**:
$$\mathcal{L}_{\text{logdet}}(z, x_{\text{raw}}) = \log \left| \det \frac{\partial G_\theta}{\partial z} \right| + \sum_{i,j} \left( x_{\text{raw},ij} - \text{Softplus}(x_{\text{raw},ij}) \right) + \lambda \cdot N_{\text{pix}}^2$$

### constraints

- **Positivity**: $x_{ij} \geq 0$ enforced via Softplus activation: $x = \text{Softplus}(x_{\text{raw}}) \cdot e^{\lambda}$
- **Latent space**: $z \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta$ | Model-dependent | float32 | Xavier/Kaiming (RealNVP default) | RealNVP normalizing flow parameters |
| $\lambda$ | $(1,)$ | float32 | $\log\left(\frac{F_0}{0.8 \cdot N_{\text{pix}}^2}\right)$ | Learnable log-scale factor for image intensity |

### 2.2 Dual Variables

None — this is a direct optimization approach, not a primal-dual method.

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $z$ | $(N_{\text{batch}}, N_{\text{pix}}^2)$ | float32 | $\mathcal{N}(0, I)$ per iteration | Latent samples from standard normal |
| $x_{\text{raw}}$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | $G_\theta(z)$ | Raw flow output before activation |
| $x$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Final positive image samples |
| $\mathcal{V}$ | $(N_{\text{batch}}, 2, N_{\text{vis}})$ | float32 | Computed | Complex visibility (real, imag) |
| $|\mathcal{V}|$ | $(N_{\text{batch}}, N_{\text{vis}})$ | float32 | Computed | Visibility amplitudes |
| $\phi^{\text{pred}}$ | $(N_{\text{batch}}, N_{\text{cp}})$ | float32 | Computed | Predicted closure phases in degrees |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{\text{batch}}$ | 32 | Number of latent samples per iteration |
| $N_{\text{pix}}$ | 32 (default) | Image dimension |
| $N_{\text{flow}}$ | 16 (default) | Number of RealNVP coupling layers |
| $\eta$ | $10^{-4}$ (default) | Adam learning rate |
| $N_{\text{epoch}}$ | 100 (default) | Number of optimization epochs |
| $w_{\text{flux}}$ | 1000.0 | Flux constraint weight |
| $w_{\text{ce}}$ | 1024.0 | Cross-entropy regularization weight |
| $w_{\text{logdet,base}}$ | 1.0 (default) | Log-determinant weight factor |
| $w_{\text{cp,base}}$ | $N_{\text{cp}} / N_{\text{camp}}$ | Closure phase weight (data-dependent) |
| $\epsilon_{\text{amp}}$ | $10^{-16}$ | Numerical stability for amplitude computation |
| $\epsilon_{\text{log}}$ | $10^{-12}$ | Numerical stability for logarithms |
| $\text{FOV}$ | 160 μas (default) | Field of view |
| $\text{FWHM}_{\text{prior}}$ | 50 μas (default) | Prior Gaussian FWHM |
| $\gamma_{\text{clip}}$ | 0.1 | Gradient clipping norm threshold |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\phi^{\text{true}}$ | $(N_{\text{cp}},)$ | float32 | Observed closure phases in degrees |
| $\sigma_{\text{cp}}$ | $(N_{\text{cp}},)$ | float32 | Closure phase uncertainties in degrees |
| $F_0$ | scalar | float32 | Total flux (median APEX-ALMA amplitude) |
| $\mathbf{F}$ | $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ | float32 | DFT matrix (real, imag components) |
| $p$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float32 | Prior image (Gaussian blob) |
| $I_1, I_2, I_3$ | $(N_{\text{cp}},)$ | int64 | Closure phase baseline index arrays |
| $s_1, s_2, s_3$ | $(N_{\text{cp}},)$ | float32 | Closure phase baseline sign arrays |

---

## 3. PREPROCESSING PIPELINE

**Step 0: Physical Parameter Derivation**

| Field | Value |
|-------|-------|
| step_order | 0 |
| step_name | Flux and Scale Computation |
| formula | $F_0 = \text{median}(\|V_{\text{APEX-ALMA}}\|)$, $\text{FOV}_{\text{rad}} = \text{FOV} \times \text{RADPERUAS}$, $\text{FWHM}_{\text{rad}} = \text{FWHM}_{\text{prior}} \times \text{RADPERUAS}$ |
| inputs | obs (uvfits), FOV (μas), FWHM_prior (μas) |
| outputs | $F_0$ (scalar), FOV_rad (scalar), FWHM_rad (scalar) |

**Step 1: DFT Matrix Construction**

| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Discrete Fourier Transform Matrix |
| formula | $\mathbf{F}_{mn} = \exp\left(-2\pi i (u_m x_n + v_m y_n)\right)$ where $x_n = (n \mod N_{\text{pix}} - N_{\text{pix}}/2) \cdot \Delta$, $y_n = (\lfloor n/N_{\text{pix}} \rfloor - N_{\text{pix}}/2) \cdot \Delta$, $\Delta = \text{FOV}_{\text{rad}}/N_{\text{pix}}$ |
| inputs | $(u, v)$ coordinates: $(N_{\text{vis}}, 2)$, pixel size $\Delta$, $N_{\text{pix}}$ |
| outputs | $\mathbf{F} \in \mathbb{R}^{N_{\text{pix}}^2 \times N_{\text{vis}} \times 2}$ (transposed, split into real/imag) |

**Step 2: Prior Image Construction**

| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Gaussian Prior Image |
| formula | $p_{ij} = \frac{F_0}{2\pi \sigma_{\text{prior}}^2} \exp\left(-\frac{(x_i - x_c)^2 + (y_j - y_c)^2}{2\sigma_{\text{prior}}^2}\right)$ where $\sigma_{\text{prior}} = \text{FWHM}_{\text{rad}} / (2\sqrt{2\ln 2})$ |
| inputs | $F_0$, FWHM_rad, $N_{\text{pix}}$, FOV_rad |
| outputs | $p \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ |

**Step 3: Closure Phase Index Mapping**

| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Baseline Triangle Indexing |
| formula | For each closure phase triangle $(t_1, t_2, t_3)$: $I_k[i] = j$ where baseline $j$ matches stations, $s_k[i] = +1$ if $(t_a, t_b)$ order matches, $-1$ otherwise |
| inputs | obs.cphase (triangle definitions), obs.data (baseline definitions) |
| outputs | $I_1, I_2, I_3 \in \mathbb{Z}^{N_{\text{cp}}}$, $s_1, s_2, s_3 \in \{-1, +1\}^{N_{\text{cp}}}$ |

**Step 4: Weight Computation**

| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Loss Weight Derivation |
| formula | $w_{\text{cp}} = N_{\text{cp}} / N_{\text{camp}}$, $w_{\text{logdet}} = 2 \cdot w_{\text{logdet,base}} / N_{\text{camp}}$ |
| inputs | $N_{\text{cp}}$, $N_{\text{camp}}$, $w_{\text{logdet,base}}$ |
| outputs | $w_{\text{cp}}$, $w_{\text{logdet}}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Description | Formula/Action |
|------|-------------|----------------|
| I1 | Initialize RealNVP | $G_\theta \leftarrow \text{RealNVP}(N_{\text{pix}}^2, N_{\text{flow}}, \text{affine=True})$ |
| I2 | Initialize log-scale | $\lambda \leftarrow \log\left(\frac{F_0}{0.8 \cdot N_{\text{pix}}^2}\right)$ |
| I3 | Initialize optimizer | $\text{Adam}(\{\theta, \lambda\}, \eta=10^{-4}, \beta_1=0.9, \beta_2=0.999)$ |
| I4 | Initialize history | $\mathcal{H} \leftarrow []$ |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP k = 0 to N_epoch - 1
```

**Step 1: Latent Sampling**
| Field | Value |
|-------|-------|
| step_name | Sample Latent Variables |
| formula | $z^{(k)} \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$, $z^{(k)} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}}^2}$ |
| inputs | $N_{\text{batch}}=32$, $N_{\text{pix}}^2$ |
| outputs | $z^{(k)}$ |
| computational_note | Reparameterization for backpropagation |

**Step 2: Flow Forward Pass** [FORWARD]
| Field | Value |
|-------|-------|
| step_name | Generate Raw Images via Flow |
| formula | $(x_{\text{raw}}^{(k)}, \log|\det J_G|) = G_\theta^{-1}(z^{(k)})$ (inverse = generative direction) |
| inputs | $z^{(k)}$, $\theta$ |
| outputs | $x_{\text{raw}}^{(k)} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}}^2}$, $\log|\det J_G| \in \mathbb{R}^{N_{\text{batch}}}$ |
| computational_note | RealNVP reverse pass computes both output and log-determinant |

**Step 3: Positivity Transform**
| Field | Value |
|-------|-------|
| step_name | Apply Softplus and Scale |
| formula | $x^{(k)} = \text{Softplus}(x_{\text{raw}}^{(k)}) \cdot e^{\lambda}$ where $\text{Softplus}(t) = \log(1 + e^t)$ |
| inputs | $x_{\text{raw}}^{(k)}$, $\lambda$ |
| outputs | $x^{(k)} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}}}$ |
| computational_note | Ensures non-negativity of reconstructed images |

**Step 4: Log-Determinant Correction**
| Field | Value |
|-------|-------|
| step_name | Compute Full Log-Determinant |
| formula | $\log|\det J|^{(k)} = \log|\det J_G| + \sum_{i,j}\left(x_{\text{raw},ij}^{(k)} - \text{Softplus}(x_{\text{raw},ij}^{(k)})\right) + \lambda \cdot N_{\text{pix}}^2$ |
| inputs | $\log|\det J_G|$, $x_{\text{raw}}^{(k)}$, $\lambda$, $N_{\text{pix}}$ |
| outputs | $\log|\det J|^{(k)} \in \mathbb{R}^{N_{\text{batch}}}$ |
| computational_note | Softplus Jacobian: $\frac{\partial}{\partial t}\text{Softplus}(t) = \sigma(t)$, so $\log|\det| = \sum(t - \text{Softplus}(t))$ |

**Step 5: Complex DFT** [FORWARD]
| Field | Value |
|-------|-------|
| step_name | Compute Complex Visibilities |
| formula | $\mathcal{V}_{\text{re}}^{(k)} = \text{vec}(x^{(k)}) \cdot \mathbf{F}_{:,:,0}$, $\mathcal{V}_{\text{im}}^{(k)} = \text{vec}(x^{(k)}) \cdot \mathbf{F}_{:,:,1}$, $\mathcal{V}^{(k)} = [\mathcal{V}_{\text{re}}^{(k)}; \mathcal{V}_{\text{im}}^{(k)}]$ |
| inputs | $x^{(k)}$, $\mathbf{F}$ |
| outputs | $\mathcal{V}^{(k)} \in \mathbb{R}^{N_{\text{batch}} \times 2 \times N_{\text{vis}}}$ |
| computational_note | Matrix multiplication with split real/imaginary DFT matrix |

**Step 6: Visibility Amplitude**
| Field | Value |
|-------|-------|
| step_name | Compute Visibility Amplitudes |
| formula | $|\mathcal{V}|_m^{(k)} = \sqrt{(\mathcal{V}_{\text{re},m}^{(k)})^2 + (\mathcal{V}_{\text{im},m}^{(k)})^2 + \epsilon_{\text{amp}}}$ |
| inputs | $\mathcal{V}^{(k)}$, $\epsilon_{\text{amp}}=10^{-16}$ |
| outputs | $|\mathcal{V}|^{(k)} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{vis}}}$ |
| computational_note | Epsilon prevents gradient issues at zero amplitude |

**Step 7: Closure Phase Computation**
| Field | Value |
|-------|-------|
| step_name | Compute Predicted Closure Phases |
| formula | $\phi_i^{\text{pred},(k)} = \frac{180}{\pi}\left(s_{1,i} \cdot \text{atan2}(\mathcal{V}_{\text{im},I_{1,i}}, \mathcal{V}_{\text{re},I_{1,i}}) + s_{2,i} \cdot \text{atan2}(\mathcal{V}_{\text{im},I_{2,i}}, \mathcal{V}_{\text{re},I_{2,i}}) + s_{3,i} \cdot \text{atan2}(\mathcal{V}_{\text{im},I_{3,i}}, \mathcal{V}_{\text{re},I_{3,i}})\right)$ |
| inputs | $\mathcal{V}^{(k)}$, $I_1, I_2, I_3$, $s_1, s_2, s_3$ |
| outputs | $\phi^{\text{pred},(k)} \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{cp}}}$ (degrees) |
| computational_note | Closure phase is bispectrum argument; sign accounts for baseline orientation |

**Step 8: Closure Phase Loss**
| Field | Value |
|-------|-------|
| step_name | Compute Closure Phase Data Fidelity |
| formula | $\mathcal{L}_{\text{cphase}}^{(k)} = \frac{2}{N_{\text{cp}}} \sum_{i=1}^{N_{\text{cp}}} \frac{1 - \cos\left(\frac{\pi}{180}(\phi_i^{\text{true}} - \phi_i^{\text{pred},(k)})\right)}{\left(\frac{\pi}{180}\sigma_{\text{cp},i}\right)^2}$ |
| inputs | $\phi^{\text{true}}$, $\phi^{\text{pred},(k)}$, $\sigma_{\text{cp}}$ |
| outputs | $\mathcal{L}_{\text{cphase}}^{(k)} \in \mathbb{R}^{N_{\text{batch}}}$ |
| computational_note | Von Mises-like loss; periodic in angle |

**Step 9: Cross-Entropy Loss**
| Field | Value |
|-------|-------|
| step_name | Compute Entropy Regularization |
| formula | $\mathcal{L}_{\text{CE}}^{(k)} = \frac{1}{N_{\text{pix}}^2} \sum_{i,j} x_{ij}^{(k)} \cdot \left(\log(x_{ij}^{(k)} + \epsilon_{\text{log}}) - \log(p_{ij} + \epsilon_{\text{log}})\right)$ |
| inputs | $x^{(k)}$, $p$, $\epsilon_{\text{log}}=10^{-12}$ |
| outputs | $\mathcal{L}_{\text{CE}}^{(k)} \in \mathbb{R}^{N_{\text{batch}}}$ |
| computational_note | KL-like divergence encouraging similarity to prior |

**Step 10: Flux Loss**
| Field | Value |
|-------|-------|
| step_name | Compute Flux Constraint Loss |
| formula | $\mathcal{L}_{\text{flux}}^{(k)} = \left(\sum_{i,j} x_{ij}^{(k)} - F_0\right)^2$ |
| inputs | $x^{(k)}$, $F_0$ |
| outputs | $\mathcal{L}_{\text{flux}}^{(k)} \in \mathbb{R}^{N_{\text{batch}}}$ |
| computational_note | Soft constraint on total image flux |

**Step 11: Total Loss Aggregation**
| Field | Value |
|-------|-------|
| step_name | Compute Weighted Total Loss |
| formula | $\mathcal{L}^{(k)} = \frac{1}{N_{\text{batch}}}\sum_b \left[ w_{\text{cp}} \mathcal{L}_{\text{cphase},b}^{(k)} + w_{\text{ce}} \mathcal{L}_{\text{CE},b}^{(k)} + w_{\text{flux}} \mathcal{L}_{\text{flux},b}^{(k)} - w_{\text{logdet}} \log|\det J|_b^{(k)} \right]$ |
| inputs | All component losses, weights: $w_{\text{cp}}$, $w_{\text{ce}}=1024$, $w_{\text{flux}}=1000$, $w_{\text{logdet}}$ |
| outputs | $\mathcal{L}^{(k)} \in \mathbb{R}$ (scalar) |
| computational_note | Negative log-det encourages exploration |

**Step 12: Gradient Computation**
| Field | Value |
|-------|-------|
| step_name | Backpropagation |
| formula | $g_\theta^{(k)} = \nabla_\theta \mathcal{L}^{(k)}$, $g_\lambda^{(k)} = \nabla_\lambda \mathcal{L}^{(k)}$ |
| inputs | $\mathcal{L}^{(k)}$, computational graph |
| outputs | $g_\theta^{(k)}$, $g_\lambda^{(k)}$ |
| computational_note | Automatic differentiation through all operations |

**Step 13: Gradient Clipping**
| Field | Value |
|-------|-------|
| step_name | Clip Gradient Norm |
| formula | If $\|[g_\theta; g_\lambda]\|_2 > \gamma_{\text{clip}}$: $[g_\theta; g_\lambda] \leftarrow \gamma_{\text{clip}} \cdot \frac{[g_\theta; g_\lambda]}{\|[g_\theta; g_\lambda]\|_2}$ |
| inputs | $g_\theta^{(k)}$, $g_\lambda^{(k)}$, $\gamma_{\text{clip}}=0.1$ |
| outputs | Clipped gradients |
| computational_note | Stabilizes training |

**Step 14: Adam Update**
| Field | Value |
|-------|-------|
| step_name | Parameter Update |
| formula | $m^{(k+1)} = \beta_1 m^{(k)} + (1-\beta_1) g^{(k)}$, $v^{(k+1)} = \beta_2 v^{(k)} + (1-\beta_2) (g^{(k)})^2$, $\hat{m} = m^{(k+1)}/(1-\beta_1^{k+1})$, $\hat{v} = v^{(k+1)}/(1-\beta_2^{k+1})$, $[\theta; \lambda]^{(k+1)} = [\theta; \lambda]^{(k)} - \eta \cdot \hat{m}/(\sqrt{\hat{v}} + 10^{-8})$ |
| inputs | Clipped gradients, $\eta=10^{-4}$, $\beta_1=0.9$, $\beta_2=0.999$ |
| outputs | Updated $\theta^{(k+1)}$, $\lambda^{(k+1)}$ |
| computational_note | Standard Adam optimizer |

**Step 15: History Update**
| Field | Value |
|-------|-------|
| step_name | Record Loss |
| formula | $\mathcal{H} \leftarrow \mathcal{H} \cup \{\mathcal{L}^{(k)}\}$ |
| inputs | $\mathcal{L}^{(k)}$ |
| outputs | Updated $\mathcal{H}$ |
| computational_note | For convergence monitoring |

```
END_LOOP
```

### 4.3 FINALIZATION

| Step | Description | Formula/Action |
|------|-------------|----------------|
| F1 | Generate final samples | $z_{\text{final}} \sim \mathcal{N}(0, I)$, $z_{\text{final}} \in \mathbb{R}^{32 \times N_{\text{pix}}^2}$ |
| F2 | Transform to images | $x_{\text{final}} = \text{Softplus}(G_\theta(z_{\text{final}})) \cdot e^{\lambda}$ |
| F3 | Compute mean image | $\bar{x} = \frac{1}{32}\sum_{b=1}^{32} x_{\text{final},b}$ |
| F4 | Compute std image | $\sigma_x = \sqrt{\frac{1}{32}\sum_{b=1}^{32} (x_{\text{final},b} - \bar{x})^2}$ |
| F5 | Save outputs | Store model weights, final samples, mean, std, loss history |

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $k \geq N_{\text{epoch}}$ |
| parameters | $N_{\text{epoch}} = 100$ (default) |
| check_frequency | Every iteration |

---

## 6. EVALUATION METRICS

### Metric 1: Flux Error

| Field | Value |
|-------|-------|
| name | Relative Flux Error |
| formula | $\text{FluxError} = \frac{|\sum_{i,j} \bar{x}_{ij} - F_0|}{F_0} \times 100\%$ |
| description | Percentage deviation of reconstructed total flux from target |

### Metric 2: Closure Phase Error

| Field | Value |
|-------|-------|
| name | Mean Closure Phase Residual |
| formula | $\text{CPError} = \frac{1}{N_{\text{cp}}}\sum_{i=1}^{N_{\text{cp}}} \min(|\phi_i^{\text{true}} - \phi_i^{\text{pred}}|, 360° - |\phi_i^{\text{true}} - \phi_i^{\text{pred}}|)$ |
| description | Average angular residual accounting for periodicity (degrees) |

### Metric 3: Final Loss

| Field | Value |
|-------|-------|
| name | Final Objective Value |
| formula | $\text{FinalLoss} = \mathcal{L}^{(N_{\text{epoch}}-1)}$ |
| description | Total loss at final