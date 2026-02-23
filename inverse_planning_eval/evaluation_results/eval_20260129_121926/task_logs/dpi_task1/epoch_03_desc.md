## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Deep Probabilistic Imaging (DPI) for Radio Interferometry
- **algorithm_family**: Variational Inference with Normalizing Flows
- **summary**: Learns a generative model (RealNVP normalizing flow) that maps standard Gaussian latent variables to the posterior distribution of images conditioned on interferometric closure phase observations, optimized via stochastic gradient descent on the evidence lower bound (ELBO).

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{\theta, \phi} \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ \mathcal{L}_{\text{data}}(G_\theta(z)) + \mathcal{L}_{\text{prior}}(G_\theta(z)) - \lambda_{\text{logdet}} \log \left| \det \frac{\partial G_\theta(z)}{\partial z} \right| \right]$$

where $G_\theta(z) = \text{Softplus}(f_\theta(z)) \cdot e^{\phi}$ is the image generator composed of normalizing flow $f_\theta$ and learnable log-scale $\phi$.

### data_fidelity_term

$$\mathcal{L}_{\text{data}}(x) = w_{\text{cp}} \cdot \frac{2}{N_{\text{cp}}} \sum_{i=1}^{N_{\text{cp}}} \frac{1 - \cos(\psi_i^{\text{true}} - \psi_i^{\text{pred}}(x))}{(\sigma_{\text{cp},i})^2}$$

where:
- $\psi_i^{\text{pred}}(x) = s_1^{(i)} \cdot \angle V_{j_1^{(i)}}(x) + s_2^{(i)} \cdot \angle V_{j_2^{(i)}}(x) + s_3^{(i)} \cdot \angle V_{j_3^{(i)}}(x)$
- $V_k(x) = \sum_{p=1}^{N_{\text{pix}}^2} x_p \cdot e^{-2\pi i (u_k \xi_p + v_k \eta_p)}$ (complex visibility)
- $\angle V = \text{atan2}(\text{Im}(V), \text{Re}(V))$
- $s_1^{(i)}, s_2^{(i)}, s_3^{(i)} \in \{-1, +1\}$ are baseline orientation signs
- $\sigma_{\text{cp},i}$ is closure phase uncertainty in radians

### regularization_term

$$\mathcal{L}_{\text{prior}}(x) = w_{\text{CE}} \cdot \frac{1}{N_{\text{pix}}^2} \sum_{p=1}^{N_{\text{pix}}^2} x_p \left( \log(x_p + \epsilon) - \log(x_p^{\text{prior}} + \epsilon) \right) + w_{\text{flux}} \cdot \left( \sum_{p=1}^{N_{\text{pix}}^2} x_p - F_0 \right)^2$$

where $\epsilon = 10^{-12}$ for numerical stability.

### constraints

None (unconstrained optimization; positivity enforced via Softplus activation)

---

## 2. VARIABLES

### Primal Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\theta$ | `primal` | (varies by n_flow) | float32 | RealNVP default init | Normalizing flow parameters |
| $\phi$ | `primal` | $(1,)$ | float32 | $\log\left(\frac{F_0}{0.8 \cdot N_{\text{pix}}^2}\right)$ | Learnable log-scale factor |

### Auxiliary Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $z$ | `auxiliary` | $(N_{\text{batch}}, N_{\text{pix}}^2)$ | float32 | $\mathcal{N}(0, I)$ per iteration | Latent Gaussian samples |
| $\tilde{x}$ | `auxiliary` | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Pre-activation image: $f_\theta(z)$ reshaped |
| $x$ | `auxiliary` | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Reconstructed image: $\text{Softplus}(\tilde{x}) \cdot e^\phi$ |
| $\log\det J$ | `auxiliary` | $(N_{\text{batch}},)$ | float32 | Computed | Log-determinant of full transformation Jacobian |
| $V$ | `auxiliary` | $(N_{\text{batch}}, 2, N_{\text{vis}})$ | float32 | Computed | Complex visibilities (real, imag stacked) |
| $|V|$ | `auxiliary` | $(N_{\text{batch}}, N_{\text{vis}})$ | float32 | Computed | Visibility amplitudes |
| $\psi^{\text{pred}}$ | `auxiliary` | $(N_{\text{batch}}, N_{\text{cp}})$ | float32 | Computed | Predicted closure phases (degrees) |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\psi^{\text{true}}$ | `observations` | $(N_{\text{cp}},)$ | float32 | Loaded from uvfits | Observed closure phases (degrees) |
| $\sigma_{\text{cp}}$ | `observations` | $(N_{\text{cp}},)$ | float32 | Loaded from uvfits | Closure phase uncertainties (degrees) |
| $(u_k, v_k)$ | `observations` | $(N_{\text{vis}}, 2)$ | float64 | Loaded from uvfits | Baseline coordinates in wavelengths |

### Constants

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $N_{\text{batch}}$ | `constants` | scalar | int | 32 | Batch size for Monte Carlo estimation |
| $N_{\text{pix}}$ | `constants` | scalar | int | 32 (default) | Image dimension per side |
| $N_{\text{flow}}$ | `constants` | scalar | int | 16 (default) | Number of RealNVP coupling layers |
| $\eta$ | `constants` | scalar | float32 | $10^{-4}$ (default) | Learning rate |
| $N_{\text{epoch}}$ | `constants` | scalar | int | 100 (default) | Number of optimization iterations |
| $w_{\text{CE}}$ | `constants` | scalar | float32 | 1024.0 | Cross-entropy regularization weight |
| $w_{\text{flux}}$ | `constants` | scalar | float32 | 1000.0 | Flux constraint weight |
| $w_{\text{cp}}$ | `constants` | scalar | float32 | $N_{\text{cp}} / N_{\text{camp}}$ | Closure phase data weight |
| $\lambda_{\text{logdet}}$ | `constants` | scalar | float32 | $\frac{2.0 \cdot \lambda_{\text{factor}}}{N_{\text{camp}}}$ | Log-determinant weight; $\lambda_{\text{factor}}=1.0$ default |
| $\epsilon_{\text{amp}}$ | `constants` | scalar | float32 | $10^{-16}$ | Numerical stability for amplitude computation |
| $\epsilon_{\text{log}}$ | `constants` | scalar | float32 | $10^{-12}$ | Numerical stability for logarithm |
| $c_{\text{grad}}$ | `constants` | scalar | float32 | 0.1 | Gradient clipping norm threshold |
| $F_0$ | `constants` | scalar | float32 | Data-dependent | Total flux (median APEX-ALMA amplitude) |
| FOV | `constants` | scalar | float64 | $160 \times \text{RADPERUAS}$ | Field of view in radians |
| $\text{FWHM}_{\text{prior}}$ | `constants` | scalar | float64 | $50 \times \text{RADPERUAS}$ | Prior Gaussian FWHM in radians |

### Precomputed Operators

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{F}$ | `constants` | $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ | float32 | Computed from $(u,v)$ | DFT matrix (real, imag components) |
| $x^{\text{prior}}$ | `constants` | $(N_{\text{pix}}, N_{\text{pix}})$ | float32 | Gaussian with FWHM, total flux $F_0$ | Prior mean image |
| $(j_1, j_2, j_3)^{(i)}$ | `constants` | $(N_{\text{cp}},) \times 3$ | int64 | Computed from obs structure | Closure phase baseline indices |
| $(s_1, s_2, s_3)^{(i)}$ | `constants` | $(N_{\text{cp}},) \times 3$ | float32 | Computed from obs structure | Closure phase baseline signs |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Load observation | $\text{obs} = \text{load\_uvfits}(\text{path})$ | File path | Observation object |
| 2 | Extract flux constant | $F_0 = \text{median}(\|V_{\text{APEX-ALMA}}\|)$ | obs | $F_0 \in \mathbb{R}$ |
| 3 | Construct pixel grid | $\xi_p = (p \mod N_{\text{pix}} - N_{\text{pix}}/2) \cdot \Delta$, $\eta_p = (\lfloor p/N_{\text{pix}} \rfloor - N_{\text{pix}}/2) \cdot \Delta$ where $\Delta = \text{FOV}/N_{\text{pix}}$ | FOV, $N_{\text{pix}}$ | $(\xi, \eta) \in \mathbb{R}^{N_{\text{pix}}^2 \times 2}$ |
| 4 | Compute DFT matrix | $F_{pk} = e^{-2\pi i (u_k \xi_p + v_k \eta_p)}$, stored as $\mathbf{F}[p,k,0] = \text{Re}(F_{pk})$, $\mathbf{F}[p,k,1] = \text{Im}(F_{pk})$ | $(u,v)$, $(\xi, \eta)$ | $\mathbf{F} \in \mathbb{R}^{N_{\text{pix}}^2 \times N_{\text{vis}} \times 2}$ |
| 5 | Build prior image | $x^{\text{prior}} = F_0 \cdot \mathcal{G}(\xi, \eta; 0, 0, \text{FWHM}_{\text{prior}})$ (2D Gaussian normalized to unit sum, scaled by $F_0$) | $F_0$, FWHM, grid | $x^{\text{prior}} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 6 | Compute closure phases | Extract $\psi^{\text{true}}$ with `min-cut0bl` triangles, `uv_min`=$10^8$ wavelengths | obs | $\psi^{\text{true}} \in \mathbb{R}^{N_{\text{cp}}}$, $\sigma_{\text{cp}} \in \mathbb{R}^{N_{\text{cp}}}$ |
| 7 | Build index maps | For each closure phase $i$, find visibility indices $(j_1, j_2, j_3)$ and orientation signs $(s_1, s_2, s_3)$ by matching timestamps and station pairs | obs | Index tensors (int64), Sign tensors (float32) |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Initialize flow | $f_\theta \leftarrow \text{RealNVP}(N_{\text{pix}}^2, N_{\text{flow}}, \text{affine}=\text{True})$ | $N_{\text{pix}}$, $N_{\text{flow}}$ | Flow network $f_\theta$ |
| 2 | Initialize log-scale | $\phi \leftarrow \log\left(\frac{F_0}{0.8 \cdot N_{\text{pix}}^2}\right)$ | $F_0$, $N_{\text{pix}}$ | $\phi \in \mathbb{R}$ |
| 3 | Initialize optimizer | $\text{opt} \leftarrow \text{Adam}(\{\theta, \phi\}, \eta)$ | $\eta$ | Optimizer state |
| 4 | Compute weights | $w_{\text{cp}} = N_{\text{cp}}/N_{\text{camp}}$, $\lambda_{\text{logdet}} = 2\lambda_{\text{factor}}/N_{\text{camp}}$ | Observation counts | Weight scalars |
| 5 | Move to device | Transfer $\mathbf{F}$, indices, signs, $\psi^{\text{true}}$, $\sigma_{\text{cp}}$, $x^{\text{prior}}$ to GPU | All data | Device tensors |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{epoch}} - 1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| 1 | Sample latent | $z \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$, shape $(32, N_{\text{pix}}^2)$ | — | $z$ | Reparameterization |
| 2 | Flow reverse pass | $(\tilde{x}_{\text{flat}}, \log\det J_f) = f_\theta^{-1}(z)$ | $z$, $\theta$ | $\tilde{x}_{\text{flat}} \in \mathbb{R}^{32 \times N_{\text{pix}}^2}$, $\log\det J_f \in \mathbb{R}^{32}$ | Invertible transform |
| 3 | Reshape | $\tilde{x} = \text{reshape}(\tilde{x}_{\text{flat}}, (32, N_{\text{pix}}, N_{\text{pix}}))$ | $\tilde{x}_{\text{flat}}$ | $\tilde{x}$ | — |
| 4 | Apply softplus | $x = \text{Softplus}(\tilde{x}) \cdot e^\phi = \log(1 + e^{\tilde{x}}) \cdot e^\phi$ | $\tilde{x}$, $\phi$ | $x \in \mathbb{R}^{32 \times N_{\text{pix}} \times N_{\text{pix}}}$ | Enforces positivity |
| 5 | Softplus Jacobian | $\Delta_{\text{sp}} = \sum_{p} \left( \tilde{x}_p - \log(1+e^{\tilde{x}_p}) \right) = \sum_p \log\sigma(\tilde{x}_p)$ | $\tilde{x}$ | $\Delta_{\text{sp}} \in \mathbb{R}^{32}$ | Log-det correction |
| 6 | Scale Jacobian | $\Delta_{\text{scale}} = \phi \cdot N_{\text{pix}}^2$ | $\phi$ | $\Delta_{\text{scale}} \in \mathbb{R}$ | Scalar broadcast |
| 7 | Total log-det | $\log\det J = \log\det J_f + \Delta_{\text{sp}} + \Delta_{\text{scale}}$ | Steps 2,5,6 | $\log\det J \in \mathbb{R}^{32}$ | Full Jacobian |
| 8 | Flatten image | $x_{\text{flat}} = \text{reshape}(x, (32, N_{\text{pix}}^2))$ | $x$ | $x_{\text{flat}}$ | For DFT |
| 9 | Compute visibilities | $V_{\text{re}} = x_{\text{flat}} \mathbf{F}_{:,:,0}$, $V_{\text{im}} = x_{\text{flat}} \mathbf{F}_{:,:,1}$ | $x_{\text{flat}}$, $\mathbf{F}$ | $V \in \mathbb{R}^{32 \times 2 \times N_{\text{vis}}}$ | Complex as 2-channel |
| 10 | Compute amplitudes | $\|V\|_k = \sqrt{V_{\text{re},k}^2 + V_{\text{im},k}^2 + 10^{-16}}$ | $V$ | $\|V\| \in \mathbb{R}^{32 \times N_{\text{vis}}}$ | Numerical stability |
| 11 | Extract triangle visibilities | $V^{(1)} = V_{:,:,j_1}$, $V^{(2)} = V_{:,:,j_2}$, $V^{(3)} = V_{:,:,j_3}$ | $V$, indices | Three visibility tensors | Index selection |
| 12 | Compute phases | $\alpha_m = \text{atan2}(V^{(m)}_{\text{im}}, V^{(m)}_{\text{re}})$ for $m \in \{1,2,3\}$ | $V^{(m)}$ | $\alpha_m \in \mathbb{R}^{32 \times N_{\text{cp}}}$ | Four-quadrant arctangent |
| 13 | Compute closure phase | $\psi^{\text{pred}} = (s_1 \alpha_1 + s_2 \alpha_2 + s_3 \alpha_3) \cdot \frac{180}{\pi}$ | $\alpha_m$, $s_m$ | $\psi^{\text{pred}} \in \mathbb{R}^{32 \times N_{\text{cp}}}$ | Degrees |
| 14 | Closure phase loss | $\mathcal{L}_{\text{cp}} = \frac{2}{N_{\text{cp}}} \sum_{i} \frac{1 - \cos\left(\frac{\pi}{180}(\psi^{\text{true}}_i - \psi^{\text{pred}}_i)\right)}{\left(\frac{\pi}{180}\sigma_{\text{cp},i}\right)^2}$ | $\psi^{\text{pred}}$, $\psi^{\text{true}}$, $\sigma_{\text{cp}}$ | $\mathcal{L}_{\text{cp}} \in \mathbb{R}^{32}$ | Von Mises-like |
| 15 | Cross-entropy loss | $\mathcal{L}_{\text{CE}} = \frac{1}{N_{\text{pix}}^2} \sum_p x_p \left(\log(x_p + 10^{-12}) - \log(x^{\text{prior}}_p + 10^{-12})\right)$ | $x$, $x^{\text{prior}}$ | $\mathcal{L}_{\text{CE}} \in \mathbb{R}^{32}$ | KL-like prior |
| 16 | Flux loss | $\mathcal{L}_{\text{flux}} = \left(\sum_p x_p - F_0\right)^2$ | $x$, $F_0$ | $\mathcal{L}_{\text{flux}} \in \mathbb{R}^{32}$ | Flux constraint |
| 17 | Aggregate loss | $\mathcal{L} = \frac{1}{32}\sum_b \left[ w_{\text{cp}} \mathcal{L}_{\text{cp}}^{(b)} + w_{\text{CE}} \mathcal{L}_{\text{CE}}^{(b)} + w_{\text{flux}} \mathcal{L}_{\text{flux}}^{(b)} - \lambda_{\text{logdet}} \log\det J^{(b)} \right]$ | All losses | $\mathcal{L} \in \mathbb{R}$ | Scalar objective |
| 18 | Zero gradients | $\nabla_{\theta,\phi} \leftarrow 0$ | opt | — | — |
| 19 | Backpropagation | Compute $\nabla_{\theta,\phi} \mathcal{L}$ | $\mathcal{L}$ | Gradients | Automatic differentiation |
| 20 | Gradient clipping | $\nabla \leftarrow \nabla \cdot \min\left(1, \frac{0.1}{\|\nabla\|_2}\right)$ | Gradients | Clipped gradients | Max norm = 0.1 |
| 21 | Parameter update | $(\theta, \phi) \leftarrow \text{Adam}((\theta, \phi), \nabla)$ | Clipped gradients, opt | Updated $(\theta, \phi)$ | — |
| 22 | Store loss | Append $\mathcal{L}$ to history | $\mathcal{L}$ | Loss history | — |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Sample final latent | $z \sim \mathcal{N}(0, I)$, shape $(32, N_{\text{pix}}^2)$ | — | $z$ |
| 2 | Generate images | $\tilde{x} = f_\theta^{-1}(z)$ (no gradient) | $z$, $\theta$ | $\tilde{x}$ |
| 3 | Transform to image | $x^{\text{final}} = \text{Softplus}(\tilde{x}) \cdot e^\phi$ | $\tilde{x}$, $\phi$ | $x^{\text{final}} \in \mathbb{R}^{32 \times N_{\text{pix}} \times N_{\text{pix}}}$ |
| 4 | Compute statistics | $\bar{x} = \frac{1}{32}\sum_b x^{\text{final}}_b$, $\sigma_x = \text{std}(x^{\text{final}}, \text{axis}=0)$ | $x^{\text{final}}$ | Mean and std images |

---

## 5. STOPPING CRITERION

| Property | Value |
|----------|-------|
| **type** | `iteration_limit` |
| **expression** | $k \geq N_{\text{epoch}}$ |
| **parameters** | $N_{\text{epoch}} = 100$ (default) |
| **check_frequency** | Every iteration (after step 21) |

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| **Total Flux** | $F_{\text{recon}} = \sum_{p=1}^{N_{\text{pix}}^2} \bar{x}_p$ | Sum of pixel values in mean reconstruction |
| **Flux Error (%)** | $\epsilon_F = \frac{|F_{\text{recon}} - F_0|}{F_0} \times 100$ | Relative flux error as percentage |
| **Final Loss** | $\mathcal{L}_{\text{final}} = \text{loss\_history}[-1]$ | Last recorded training loss value |
| **Mean Closure Phase Error** | $\bar{\epsilon}_\psi = \frac{1}{N_{\text{cp}}} \sum_{i=1}^{N_{\text{cp}}} \min\left(|\psi_i^{\text{true}} - \psi_i^{\text{pred}}|, 360 - |\psi_i^{\text{true}} - \psi_i^{\text{pred}}|\right)$ | Mean angular residual in degrees, accounting for $360°$ periodicity |
| **Image Mean** | $\bar{x}_{pq} = \frac{1}{N_{\text{batch}}} \sum_{b=1}^{N_{\text{batch}}} x^{\text{final}}_{b,p,q}$ | Posterior mean image estimate |
| **Image Std** | $\sigma_{x,pq} = \sqrt{\frac{1}{N_{\text{batch}}-1} \sum_{b=1}^{N_{\text{batch}}} (x^{\text{final}}_{b,p,q} - \bar{x}_{pq})^2}$ | Posterior uncertainty estimate per pixel |