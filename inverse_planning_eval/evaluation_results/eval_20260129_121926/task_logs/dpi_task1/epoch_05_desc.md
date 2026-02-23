## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Deep Probabilistic Imaging with RealNVP for Radio Interferometry
- **algorithm_family**: Variational Inference with Normalizing Flows
- **summary**: Learns a normalizing flow generative model to produce posterior image samples from interferometric closure phase observations by minimizing a variational objective combining data fidelity, prior regularization, and flow entropy terms.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\theta, \lambda} \; \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ \mathcal{L}_{\text{data}}(x(z)) + \mathcal{L}_{\text{prior}}(x(z)) - w_{\text{det}} \log \left| \det \frac{\partial x}{\partial z} \right| \right]$$

where $x(z) = \text{softplus}(G_\theta^{-1}(z)) \cdot e^\lambda$ and $G_\theta$ is the RealNVP flow.

### data_fidelity_term
$$\mathcal{L}_{\text{data}} = w_{\text{cp}} \cdot \frac{2}{N_{\text{cp}}} \sum_{i=1}^{N_{\text{cp}}} \frac{1 - \cos(\phi_i^{\text{true}} - \phi_i^{\text{pred}})}{\sigma_{\text{cp},i}^2}$$

This is a **von Mises-based angular loss** (NOT standard L2), where angles are in radians.

### regularization_term

**Cross-entropy term (KL-divergence-like)**:
$$\mathcal{L}_{\text{CE}} = w_{\text{CE}} \cdot \frac{1}{N_{\text{pix}}^2} \sum_{i,j} x_{i,j} \left( \log(x_{i,j} + \epsilon) - \log(p_{i,j} + \epsilon) \right)$$

**Flux constraint term**:
$$\mathcal{L}_{\text{flux}} = w_{\text{flux}} \cdot \left( \sum_{i,j} x_{i,j} - F_0 \right)^2$$

**Negative log-determinant (entropy) term**:
$$\mathcal{L}_{\text{det}} = -w_{\text{det}} \cdot \log \left| \det \frac{\partial x}{\partial z} \right|$$

### constraints
$$x_{i,j} > 0 \quad \forall i,j \quad \text{(enforced via softplus transformation)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta$ | Flow-dependent | float32 | Random (RealNVP default) | RealNVP normalizing flow parameters |
| $\lambda$ | $(1,)$ | float32 | $\log\left(\frac{F_0}{0.8 \cdot N_{\text{pix}}^2}\right)$ | Learnable log-scale factor for image intensity |

### 2.2 Dual Variables

None (unconstrained optimization via Adam).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $z$ | $(B, N_{\text{pix}}^2)$ | float32 | $\sim \mathcal{N}(0, I)$ per iteration | Latent samples from standard normal |
| $\tilde{x}$ | $(B, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Raw image output from flow inverse |
| $\log\det_{\text{flow}}$ | $(B,)$ | float32 | Computed | Log-determinant from RealNVP inverse |
| $x$ | $(B, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Final non-negative image samples |
| $V$ | $(B, 2, N_{\text{vis}})$ | float32 | Computed | Complex visibilities (real, imag stacked) |
| $|V|$ | $(B, N_{\text{vis}})$ | float32 | Computed | Visibility amplitudes |
| $\phi^{\text{pred}}$ | $(B, N_{\text{cp}})$ | float32 | Computed | Predicted closure phases in degrees |
| $\log\det_{\text{total}}$ | $(B,)$ | float32 | Computed | Total log-determinant of transformation |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{\text{pix}}$ | 32 (default) | Image dimension |
| $B$ | 32 | Batch size (number of samples per iteration) |
| $n_{\text{flow}}$ | 16 (default) | Number of RealNVP coupling layers |
| $\eta$ | $10^{-4}$ (default) | Adam learning rate |
| $K$ | 100 (default) | Number of optimization epochs |
| $w_{\text{flux}}$ | 1000.0 | Flux constraint weight |
| $w_{\text{CE}}$ | 1024.0 | Cross-entropy regularization weight |
| $w_{\text{cp}}$ | $\frac{N_{\text{cp}}}{N_{\text{camp}}}$ | Closure phase weight |
| $w_{\text{det}}$ | $\frac{2 \cdot \text{logdet\_factor}}{N_{\text{camp}}}$ | Log-determinant weight |
| $\epsilon$ | $10^{-12}$ | Numerical stability constant for log |
| $\epsilon_{\text{amp}}$ | $10^{-16}$ | Numerical stability for amplitude computation |
| $\text{grad\_clip}$ | 0.1 | Gradient clipping norm |
| zero_symbol | 100000 | Sentinel value for index 0 handling |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $F$ | $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ | float32 | DFT matrix (real, imag components) |
| $\phi^{\text{true}}$ | $(N_{\text{cp}},)$ | float32 | Observed closure phases in degrees |
| $\sigma_{\text{cp}}$ | $(N_{\text{cp}},)$ | float32 | Closure phase uncertainties in degrees |
| $p$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float32 | Prior Gaussian image |
| $F_0$ | scalar | float32 | Flux constant (median APEX-ALMA amplitude) |
| $I_1, I_2, I_3$ | $(N_{\text{cp}},)$ each | int64 | Closure phase visibility indices |
| $s_1, s_2, s_3$ | $(N_{\text{cp}},)$ each | float32 | Closure phase visibility signs ($\pm 1$) |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load observation | $\text{obs} = \text{load\_uvfits}(\text{obspath})$ | obspath: string | obs: ehtim.Obsdata |
| 2 | Compute flux constant | $F_0 = \text{median}(\|V_{\text{APEX-ALMA}}\|)$ | obs | $F_0$: scalar |
| 3 | Convert units | $\text{fov\_rad} = \text{fov} \times 10^{-6} \times \frac{\pi}{180 \times 3600}$, $\text{fwhm\_rad} = \text{prior\_fwhm} \times 10^{-6} \times \frac{\pi}{180 \times 3600}$ | fov, prior_fwhm (μas) | fov_rad, fwhm_rad (radians) |
| 4 | Create prior image | $p_{i,j} = F_0 \cdot \exp\left(-\frac{(x_i^2 + y_j^2)}{2(\text{fwhm\_rad}/2.355)^2}\right)$ (Gaussian) | $F_0$, fwhm_rad, $N_{\text{pix}}$, fov_rad | $p$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 5 | Extract UV coordinates | $\text{uv} = [(u_1, v_1), \ldots, (u_{N_{\text{vis}}}, v_{N_{\text{vis}}})]^T$ | obs | uv: $(N_{\text{vis}}, 2)$ |
| 6 | Compute pixel coordinates | $x_i = (i - N_{\text{pix}}/2) \cdot \Delta$, $y_j = (j - N_{\text{pix}}/2) \cdot \Delta$ where $\Delta = \text{fov\_rad}/N_{\text{pix}}$ | $N_{\text{pix}}$, fov_rad | coords: $(2, N_{\text{pix}}^2)$ |
| 7 | Build DFT matrix | $F_{k,n} = \exp\left(-2\pi i (u_k x_n + v_k y_n)\right)$, then $F_{\text{tensor}} = [\text{Re}(F^T), \text{Im}(F^T)]$ | uv, coords | $F$: $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ |
| 8 | Add closure quantities | obs.add_cphase(count='min-cut0bl', uv_min=$10^8$), obs.add_camp(debias=True) | obs | obs with cphase, camp tables |
| 9 | Build closure phase index map | For each closure phase $k$, find visibility indices $(I_{1,k}, I_{2,k}, I_{3,k})$ and signs $(s_{1,k}, s_{2,k}, s_{3,k})$ by matching telescope triplets | obs.cphase, obs.data | $I_1, I_2, I_3$: $(N_{\text{cp}},)$; $s_1, s_2, s_3$: $(N_{\text{cp}},)$ |
| 10 | Extract observations | $\phi^{\text{true}} = \text{obs.cphase['cphase']}$, $\sigma_{\text{cp}} = \text{obs.cphase['sigmacp']}$ | obs | $\phi^{\text{true}}$, $\sigma_{\text{cp}}$: $(N_{\text{cp}},)$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Initialize RealNVP flow | $G_\theta \leftarrow \text{RealNVP}(N_{\text{pix}}^2, n_{\text{flow}}, \text{affine=True})$ | $N_{\text{pix}}$, $n_{\text{flow}}$ | $\theta$ (flow parameters) |
| 2 | Initialize log-scale | $\lambda \leftarrow \log\left(\frac{F_0}{0.8 \cdot N_{\text{pix}}^2}\right)$ | $F_0$, $N_{\text{pix}}$ | $\lambda$: $(1,)$ |
| 3 | Compute weights | $w_{\text{cp}} \leftarrow \frac{N_{\text{cp}}}{N_{\text{camp}}}$, $w_{\text{det}} \leftarrow \frac{2 \cdot \text{logdet\_factor}}{N_{\text{camp}}}$ | $N_{\text{cp}}$, $N_{\text{camp}}$, logdet_factor | $w_{\text{cp}}$, $w_{\text{det}}$ |
| 4 | Initialize optimizer | $\text{optimizer} \leftarrow \text{Adam}(\{\theta, \lambda\}, \eta)$ | $\theta$, $\lambda$, $\eta$ | optimizer state |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** $k = 0, 1, \ldots, K-1$

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 5 | Sample latent vectors | $z \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$ | $B$, $N_{\text{pix}}$ | $z$: $(B, N_{\text{pix}}^2)$ | torch.randn |
| 6 | Flow inverse pass | $(\tilde{x}_{\text{flat}}, \log\det_{\text{flow}}) = G_\theta^{-1}(z)$ | $z$, $\theta$ | $\tilde{x}_{\text{flat}}$: $(B, N_{\text{pix}}^2)$; $\log\det_{\text{flow}}$: $(B,)$ | RealNVP reverse |
| 7 | Reshape raw image | $\tilde{x} \leftarrow \text{reshape}(\tilde{x}_{\text{flat}}, (B, N_{\text{pix}}, N_{\text{pix}}))$ | $\tilde{x}_{\text{flat}}$ | $\tilde{x}$: $(B, N_{\text{pix}}, N_{\text{pix}})$ | |
| 8 | Compute scale factor | $\alpha \leftarrow e^\lambda$ | $\lambda$ | $\alpha$: scalar | |
| 9 | Apply softplus and scale | $x_{b,i,j} \leftarrow \log(1 + e^{\tilde{x}_{b,i,j}}) \cdot \alpha$ | $\tilde{x}$, $\alpha$ | $x$: $(B, N_{\text{pix}}, N_{\text{pix}})$ | Ensures positivity |
| 10 | Softplus log-determinant | $\Delta_{\text{sp},b} \leftarrow \sum_{i,j} \left( \tilde{x}_{b,i,j} - \log(1 + e^{\tilde{x}_{b,i,j}}) \right)$ | $\tilde{x}$ | $\Delta_{\text{sp}}$: $(B,)$ | $\log \frac{\partial \text{softplus}}{\partial \tilde{x}} = \tilde{x} - \text{softplus}(\tilde{x})$ |
| 11 | Scale log-determinant | $\Delta_{\alpha} \leftarrow \lambda \cdot N_{\text{pix}}^2$ | $\lambda$, $N_{\text{pix}}$ | $\Delta_\alpha$: scalar | |
| 12 | Total log-determinant | $\log\det_{\text{total},b} \leftarrow \log\det_{\text{flow},b} + \Delta_{\text{sp},b} + \Delta_\alpha$ | $\log\det_{\text{flow}}$, $\Delta_{\text{sp}}$, $\Delta_\alpha$ | $\log\det_{\text{total}}$: $(B,)$ | |
| 13 | Flatten image for DFT | $x_{\text{flat}} \leftarrow \text{reshape}(x, (B, N_{\text{pix}}^2))$ | $x$ | $x_{\text{flat}}$: $(B, N_{\text{pix}}^2)$ | |
| 14 | Compute visibilities [FORWARD] | $V_{b,0,k} \leftarrow \sum_n x_{\text{flat},b,n} F_{n,k,0}$, $V_{b,1,k} \leftarrow \sum_n x_{\text{flat},b,n} F_{n,k,1}$ | $x_{\text{flat}}$, $F$ | $V$: $(B, 2, N_{\text{vis}})$ | Real/Imag parts |
| 15 | Compute visibility amplitudes | $\|V\|_{b,k} \leftarrow \sqrt{V_{b,0,k}^2 + V_{b,1,k}^2 + \epsilon_{\text{amp}}}$ | $V$ | $\|V\|$: $(B, N_{\text{vis}})$ | |
| 16 | Select visibilities for CP | $V^{(1)}_{b,:,m} \leftarrow V_{b,:,I_{1,m}}$, $V^{(2)}_{b,:,m} \leftarrow V_{b,:,I_{2,m}}$, $V^{(3)}_{b,:,m} \leftarrow V_{b,:,I_{3,m}}$ | $V$, $I_1$, $I_2$, $I_3$ | $V^{(1)}, V^{(2)}, V^{(3)}$: $(B, 2, N_{\text{cp}})$ each | Index selection |
| 17 | Compute visibility phases | $\psi^{(j)}_{b,m} \leftarrow \text{atan2}(V^{(j)}_{b,1,m}, V^{(j)}_{b,0,m})$ for $j=1,2,3$ | $V^{(1)}, V^{(2)}, V^{(3)}$ | $\psi^{(1)}, \psi^{(2)}, \psi^{(3)}$: $(B, N_{\text{cp}})$ | Radians |
| 18 | Compute closure phases | $\phi^{\text{pred}}_{b,m} \leftarrow (s_{1,m} \psi^{(1)}_{b,m} + s_{2,m} \psi^{(2)}_{b,m} + s_{3,m} \psi^{(3)}_{b,m}) \cdot \frac{180}{\pi}$ | $\psi^{(1)}, \psi^{(2)}, \psi^{(3)}$, $s_1, s_2, s_3$ | $\phi^{\text{pred}}$: $(B, N_{\text{cp}})$ | Degrees |
| 19 | Cross-entropy loss | $\mathcal{L}_{\text{CE},b} \leftarrow \frac{1}{N_{\text{pix}}^2} \sum_{i,j} x_{b,i,j} \left( \log(x_{b,i,j} + \epsilon) - \log(p_{i,j} + \epsilon) \right)$ | $x$, $p$ | $\mathcal{L}_{\text{CE}}$: $(B,)$ | |
| 20 | Flux loss | $\mathcal{L}_{\text{flux},b} \leftarrow \left( \sum_{i,j} x_{b,i,j} - F_0 \right)^2$ | $x$, $F_0$ | $\mathcal{L}_{\text{flux}}$: $(B,)$ | |
| 21 | Convert angles to radians | $\phi^{\text{true}}_{\text{rad}} \leftarrow \phi^{\text{true}} \cdot \frac{\pi}{180}$, $\phi^{\text{pred}}_{\text{rad}} \leftarrow \phi^{\text{pred}} \cdot \frac{\pi}{180}$, $\sigma_{\text{rad}} \leftarrow \sigma_{\text{cp}} \cdot \frac{\pi}{180}$ | $\phi^{\text{true}}$, $\phi^{\text{pred}}$, $\sigma_{\text{cp}}$ | Radian versions | |
| 22 | Closure phase loss | $\mathcal{L}_{\text{cp},b} \leftarrow \frac{2}{N_{\text{cp}}} \sum_{m=1}^{N_{\text{cp}}} \frac{1 - \cos(\phi^{\text{true}}_{\text{rad},m} - \phi^{\text{pred}}_{\text{rad},b,m})}{\sigma_{\text{rad},m}^2}$ | $\phi^{\text{true}}_{\text{rad}}$, $\phi^{\text{pred}}_{\text{rad}}$, $\sigma_{\text{rad}}$ | $\mathcal{L}_{\text{cp}}$: $(B,)$ | Von Mises loss |
| 23 | Data loss | $\mathcal{L}_{\text{data},b} \leftarrow w_{\text{cp}} \cdot \mathcal{L}_{\text{cp},b}$ | $\mathcal{L}_{\text{cp}}$, $w_{\text{cp}}$ | $\mathcal{L}_{\text{data}}$: $(B,)$ | |
| 24 | Prior loss | $\mathcal{L}_{\text{prior},b} \leftarrow w_{\text{CE}} \cdot \mathcal{L}_{\text{CE},b} + w_{\text{flux}} \cdot \mathcal{L}_{\text{flux},b}$ | $\mathcal{L}_{\text{CE}}$, $\mathcal{L}_{\text{flux}}$, $w_{\text{CE}}$, $w_{\text{flux}}$ | $\mathcal{L}_{\text{prior}}$: $(B,)$ | |
| 25 | Total loss | $\mathcal{L} \leftarrow \frac{1}{B}\sum_b \mathcal{L}_{\text{data},b} + \frac{1}{B}\sum_b \mathcal{L}_{\text{prior},b} - w_{\text{det}} \cdot \frac{1}{B}\sum_b \log\det_{\text{total},b}$ | $\mathcal{L}_{\text{data}}$, $\mathcal{L}_{\text{prior}}$, $\log\det_{\text{total}}$, $w_{\text{det}}$ | $\mathcal{L}$: scalar | |
| 26 | Compute gradients | $\nabla_\theta \mathcal{L}, \nabla_\lambda \mathcal{L} \leftarrow \text{backprop}(\mathcal{L})$ | $\mathcal{L}$, $\theta$, $\lambda$ | gradients | |
| 27 | Clip gradients | $\nabla \leftarrow \nabla \cdot \min\left(1, \frac{0.1}{\|\nabla\|_2}\right)$ | gradients | clipped gradients | Norm threshold 0.1 |
| 28 | Adam update | $\theta \leftarrow \theta - \eta \cdot \text{Adam}(\nabla_\theta)$, $\lambda \leftarrow \lambda - \eta \cdot \text{Adam}(\nabla_\lambda)$ | gradients, optimizer state | updated $\theta$, $\lambda$ | |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 29 | Sample final latents | $z^* \sim \mathcal{N}(0, I_{N_{\text{pix}}^2})$ | $B$, $N_{\text{pix}}$ | $z^*$: $(B, N_{\text{pix}}^2)$ |
| 30 | Generate final images | $\tilde{x}^* \leftarrow G_\theta^{-1}(z^*)$, $x^* \leftarrow \text{softplus}(\tilde{x}^*) \cdot e^\lambda$ | $z^*$, $\theta$, $\lambda$ | $x^*$: $(B, N_{\text{pix}}, N_{\text{pix}})$ |
| 31 | Compute mean image | $\bar{x} \leftarrow \frac{1}{B} \sum_{b=1}^B x^*_b$ | $x^*$ | $\bar{x}$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 32 | Compute std image | $\sigma_x \leftarrow \sqrt{\frac{1}{B} \sum_{b=1}^B (x^*_b - \bar{x})^2}$ | $x^*$, $\bar{x}$ | $\sigma_x$: $(N_{\text{pix}}, N_{\text{pix}})$ |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k \geq K$ where $K = n\_epoch$
- **parameters**: $K = 100$ (default)
- **check_frequency**: After each iteration (epoch)

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Total Flux | $F_{\text{total}} = \sum_{i,j} \bar{x}_{i,j}$ | $F_0$ (flux constant) | $\bar{x}$ (mean reconstructed image) | Total image flux |
| Flux Error | $\text{Flux Error} = \frac{|F_{\text{total}} - F_0|}{F_0} \times 100\%$ | $F_0$ | $F_{\text{total}}$ | Relative flux reconstruction error |
| Mean Closure Phase Error | $\bar{\epsilon}_{\phi} = \frac{1}{N_{\text{cp}}} \sum_{m=1}^{N_{\text{cp}}} \min(|\phi^{\text{true}}_m - \phi^{\text{pred}}_m|, 360 - |\phi^{\text{true}}_m - \phi^{\text{pred}}_m|)$ | $\phi^{\text{true}}$ (degrees) | $\phi^{\text{pred}}$ from $\bar{x}$ (degrees) | Angular closure phase residual accounting for wrap-around |
| Final Loss | $\mathcal{L}_{\text{final}} = \mathcal{L}_{k=K-1}$ | N/A | Final iteration loss | Optimization convergence indicator |