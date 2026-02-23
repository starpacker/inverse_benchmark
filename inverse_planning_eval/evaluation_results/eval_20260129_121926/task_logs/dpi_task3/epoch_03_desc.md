## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: Deep Probabilistic Imaging via Normalizing Flows (DPI-NF)
- **algorithm_family**: Variational Inference with Normalizing Flows
- **summary**: Learns a generative model of interferometric images by training a RealNVP normalizing flow to map Gaussian latent variables to geometric image parameters, minimizing an alpha-divergence between the learned posterior and the data likelihood.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\phi} \; \mathcal{L}_{\alpha}(\phi) = \begin{cases} \mathbb{E}_{z \sim \mathcal{N}(0,I)}\left[ s \cdot L(z;\phi) \right] & \text{if } \alpha = 1 \\ \sum_{i=1}^{B} w_i \cdot s \cdot L(z_i;\phi) & \text{if } \alpha \neq 1 \end{cases}$$

where:
$$L(z;\phi) = \lambda_{\text{data}}(k) \cdot \mathcal{L}_{\text{data}}(G_\psi(f_\phi(z))) - \log \left|\det \frac{\partial f_\phi}{\partial z}\right| + \frac{1}{2}\|z\|_2^2$$

### data_fidelity_term
For `data_product='cphase_logcamp'`:
$$\mathcal{L}_{\text{data}} = \frac{1}{2s}\left( w_{\text{camp}} \cdot \frac{1}{N_{\text{camp}}}\sum_{j=1}^{N_{\text{camp}}} \frac{(\log A_j^{\text{obs}} - \log A_j)^2}{\sigma_{\log A,j}^2} + w_{\text{cphase}} \cdot \frac{2}{N_{\text{cphase}}}\sum_{j=1}^{N_{\text{cphase}}} \frac{1 - \cos(\Phi_j^{\text{obs}} - \Phi_j)}{(\sigma_{\Phi,j} \cdot \pi/180)^2} \right)$$

For `data_product='vis'`:
$$\mathcal{L}_{\text{data}} = \frac{1}{2s} \cdot \frac{1}{N_{\text{vis}}}\sum_{j=1}^{N_{\text{vis}}} \frac{|V_j^{\text{obs}} - V_j|^2}{\sigma_{V,j}^2}$$

### regularization_term
Log-determinant regularization (entropy term from normalizing flow):
$$\mathcal{R}(\phi, z) = -\log \left|\det \frac{\partial f_\phi}{\partial z}\right| - \sum_{i=1}^{d} \left( -\theta_i - 2\log(1 + e^{-\theta_i}) \right)$$

where $\theta = f_\phi(z)$ are pre-sigmoid parameters.

### constraints
None (unconstrained optimization in parameter space; physical constraints enforced via sigmoid transformation and parameter ranges).

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\phi$ | primal | varies | float32 | See RealNVP init below | RealNVP network parameters |
| $z$ | primal | $(B, d)$ | float32 | $z \sim \mathcal{N}(0, I_d)$ | Latent samples |
| $\theta$ | auxiliary | $(B, d)$ | float32 | $\theta = f_\phi(z)$ | Pre-sigmoid geometric parameters |
| $p$ | auxiliary | $(B, d)$ | float32 | $p = \sigma(\theta)$ | Sigmoid-transformed parameters in $[0,1]$ |
| $x$ | auxiliary | $(B, N_{\text{pix}}, N_{\text{pix}})$ | float32 | $x = G_\psi(p)$ | Generated images |

### Dual Variables
None.

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $V^{\text{obs}}$ | observations | $(2, N_{\text{vis}})$ | float32 | Loaded from UVFITS | Complex visibilities (real, imag) |
| $\Phi^{\text{obs}}$ | observations | $(N_{\text{cphase}},)$ | float32 | Computed from obs | Closure phases in degrees |
| $\log A^{\text{obs}}$ | observations | $(N_{\text{camp}},)$ | float32 | Computed from obs | Log closure amplitudes |
| $\sigma_V$ | observations | $(N_{\text{vis}},)$ | float32 | Loaded from UVFITS | Visibility uncertainties |
| $\sigma_\Phi$ | observations | $(N_{\text{cphase}},)$ | float32 | Computed from obs | Closure phase uncertainties |
| $\sigma_{\log A}$ | observations | $(N_{\text{camp}},)$ | float32 | Computed from obs | Log closure amplitude uncertainties |

### Constants
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $N_{\text{pix}}$ | constants | scalar | int | 64 (default) | Image dimension |
| $N_{\text{flow}}$ | constants | scalar | int | 16 (default) | Number of flow blocks |
| $N_{\text{gauss}}$ | constants | scalar | int | 2 (default) | Number of nuisance Gaussians |
| $B$ | constants | scalar | int | 64 | Batch size |
| $\eta$ | constants | scalar | float32 | $10^{-4}$ | Learning rate |
| $\gamma_{\text{clip}}$ | constants | scalar | float32 | $10^{-4}$ | Gradient clipping threshold |
| $\alpha$ | constants | scalar | float32 | 1.0 (default) | Alpha-divergence parameter |
| $\beta$ | constants | scalar | float32 | 0.0 (default) | Beta parameter for adaptive alpha |
| $\lambda_0$ | constants | scalar | float32 | 4 | Start order for data weight |
| $\tau$ | constants | scalar | float32 | 2000 | Decay rate for data weight |
| $w_{\text{logdet}}$ | constants | scalar | float32 | 1.0 | Log-determinant weight |
| FOV | constants | scalar | float32 | 120.0 | Field of view in $\mu$as |
| $r_{\min}, r_{\max}$ | constants | scalar | float32 | 10.0, 40.0 | Ring radius range in $\mu$as |
| $\sigma_{\min}, \sigma_{\max}$ | constants | scalar | float32 | 1.0, 40.0 | Ring width range in $\mu$as |
| $s_{\min}, s_{\max}$ | constants | scalar | float32 | $10^{-3}$, 0.99 | Asymmetry range |
| floor$_{\min}$, floor$_{\max}$ | constants | scalar | float32 | 0.0, 1.0 | Floor fraction range |
| $\epsilon$ | constants | scalar | float32 | $10^{-4}$ | Numerical stability constant |
| $\epsilon_{\text{amp}}$ | constants | scalar | float32 | $10^{-16}$ | Amplitude stability constant |
| seqfrac | constants | scalar | float32 | 1/16 | Hidden dimension fraction |

### RealNVP Initialization
| Component | Initialization |
|-----------|----------------|
| ActNorm.loc | zeros$(1,)$, data-dependent at first forward: $-\text{mean}(\text{input})$ |
| ActNorm.log_scale_inv | zeros$(1,)$, data-dependent: $\log(\text{std}(\text{input}) + 10^{-6})$ |
| AffineCoupling.Linear.weight | $\mathcal{N}(0, 0.05)$ |
| AffineCoupling.Linear.bias | zeros |
| ZeroFC.fc.weight | zeros |
| ZeroFC.fc.bias | zeros |
| ZeroFC.scale | zeros$(d_{\text{out}},)$ |
| BatchNorm1d.eps | $10^{-2}$ |
| LeakyReLU.negative_slope | 0.01 |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Load observation | $\text{obs} = \text{load\_uvfits}(\text{path})$ | path: str | obs: ehtim.Obsdata |
| 2 | Extract flux constant | $F_0 = \text{median}(\|V_{\text{AP-AA}}\|)$ | obs | $F_0$: scalar |
| 3 | Construct prior image | $I_{\text{prior}} = \text{tophat}(F_0, \text{FWHM}/2) * \text{Gauss}(\text{res})$ | $F_0$, FWHM=80$\mu$as | $I_{\text{prior}}$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 4 | Extract UV coordinates | $\mathbf{u} = (u_j, v_j)_{j=1}^{N_{\text{vis}}}$ | obs | $\mathbf{u}$: $(N_{\text{vis}}, 2)$ |
| 5 | Compute DFT matrix | $F_{ij} = e^{-2\pi i (u_j x_i + v_j y_i)}$ via `ftmatrix` | $\mathbf{u}$, psize, npix | $F$: $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ |
| 6 | Build closure phase map | Map visibility indices to closure triangles with signs | obs.cphase, obs.data | cphase_ind$_{1,2,3}$: $(N_{\text{cp}},)$, cphase_sign$_{1,2,3}$: $(N_{\text{cp}},)$ |
| 7 | Build closure amplitude map | Map visibility indices to closure quadrangles | obs.camp, obs.data | camp_ind$_{1,2,3,4}$: $(N_{\text{ca}},)$ |
| 8 | Extract ground truth | $V^{\text{obs}}, \Phi^{\text{obs}}, \log A^{\text{obs}}$ | obs | Tensors as specified |
| 9 | Extract uncertainties | $\sigma_V, \sigma_\Phi, \sigma_{\log A}$ | obs | Tensors as specified |
| 10 | Construct pixel grid | $x_{ij} = \frac{2(j - N/2)}{N}, y_{ij} = \frac{2(i - N/2)}{N}$ | $N_{\text{pix}}$ | grid_x, grid_y: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 11 | Compute polar grid | $r_{ij} = \sqrt{x_{ij}^2 + y_{ij}^2}, \phi_{ij} = \arctan2(y_{ij}, x_{ij})$ | grid_x, grid_y | grid_r, grid_$\theta$: $(N_{\text{pix}}, N_{\text{pix}})$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Initialize geometric model | $G_\psi$ = SimpleCrescentNuisanceFloor with registered buffers | $N_{\text{pix}}$, FOV, ranges | $G_\psi$: nn.Module |
| 2 | Compute parameter dimension | $d = 4 + 6 N_{\text{gauss}} + 2$ (if not flux_flag) | $N_{\text{gauss}}$ | $d$: int |
| 3 | Initialize RealNVP | $f_\phi$ = RealNVP($d$, $N_{\text{flow}}$) with random permutations | $d$, $N_{\text{flow}}$, seed=$i$ | $f_\phi$: nn.Module |
| 4 | Compute scale factor | $s = 1/N_{\text{camp}}$ (for cphase_logcamp) | $N_{\text{camp}}$ | $s$: scalar |
| 5 | Compute weights | $w_{\text{camp}}=1, w_{\text{cphase}}=N_{\text{cp}}/N_{\text{ca}}, w_{\text{vis}}=0$ | counts | weights: scalars |
| 6 | Compute logdet scale | $\lambda_{\text{logdet}} = 2 w_{\text{logdet}} / N_{\text{camp}}$ | $w_{\text{logdet}}$, $N_{\text{camp}}$ | $\lambda_{\text{logdet}}$: scalar |
| 7 | Initialize optimizer | Adam($\phi$, lr=$\eta$) | $\eta = 10^{-4}$ | optimizer |
| 8 | Compute effective alpha | $\alpha_{\text{eff}} = \alpha$ if $\beta=0$ else $1 - \beta \cdot s$ | $\alpha$, $\beta$, $s$ | $\alpha_{\text{eff}}$: scalar |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{epoch}}-1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| 1 | Compute data weight | $\lambda_{\text{data}}(k) = \min(10^{-\lambda_0 + k/\tau}, 1)$ | $k$, $\lambda_0=4$, $\tau=2000$ | $\lambda_{\text{data}}$: scalar | Annealing schedule |
| 2 | Sample latent | $z \sim \mathcal{N}(0, I_d)$, shape $(B, d)$ | $B=64$, $d$ | $z$: $(B, d)$ | |
| 3 | Flow reverse pass | $\theta, \log|J_f| = f_\phi^{-1}(z)$ | $z$, $f_\phi$ | $\theta$: $(B, d)$, $\log|J_f|$: $(B,)$ | Invertible transform |
| 4 | Sigmoid transform | $p_i = \sigma(\theta_i) = \frac{1}{1 + e^{-\theta_i}}$ | $\theta$ | $p$: $(B, d)$ | Maps to $[0,1]^d$ |
| 5 | Sigmoid log-det | $\log|J_\sigma| = \sum_{i=1}^d \left(-\theta_i - 2\log(1 + e^{-\theta_i})\right)$ | $\theta$ | $\log|J_\sigma|$: $(B,)$ | |
| 6 | Total log-det | $\log|J| = \log|J_f| + \log|J_\sigma|$ | $\log|J_f|$, $\log|J_\sigma|$ | $\log|J|$: $(B,)$ | |
| 7 | Generate image | $x = G_\psi(p)$ | $p$, $G_\psi$ | $x$: $(B, N_{\text{pix}}, N_{\text{pix}})$ | See 4.2.1 |
| 8 | Forward model | $V, \|V\|, \Phi, \log A = \mathcal{F}(x)$ | $x$, $F$ | observables | See 4.2.2 |
| 9 | Compute data loss | $\mathcal{L}_{\text{data}}$ per formula in §1 | observables, ground truth | $\mathcal{L}_{\text{data}}$: $(B,)$ | |
| 10 | Compute log-prob | $\log q = -\log|J| - \frac{1}{2}\|z\|_2^2$ | $\log|J|$, $z$ | $\log q$: $(B,)$ | |
| 11 | Compute sample loss | $L = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \log q$ | $\lambda_{\text{data}}$, $\mathcal{L}_{\text{data}}$, $\log q$ | $L$: $(B,)$ | |
| 12 | Compute final loss | See 4.2.3 | $L$, $\alpha_{\text{eff}}$, $s$ | $\mathcal{L}$: scalar | |
| 13 | Backward pass | $\nabla_\phi \mathcal{L}$ | $\mathcal{L}$ | gradients | |
| 14 | Clip gradients | $\nabla_\phi \leftarrow \text{clip}(\nabla_\phi, \gamma_{\text{clip}})$ | gradients, $\gamma_{\text{clip}}=10^{-4}$ | clipped gradients | |
| 15 | Update parameters | $\phi \leftarrow \phi - \eta \cdot \text{Adam}(\nabla_\phi)$ | $\phi$, gradients | $\phi$ updated | |

**END_LOOP**

#### 4.2.1 Image Generation Sub-procedure

| Step | Formula |
|------|---------|
| 1 | Scale parameters: $r = r_{\min}/(\text{FOV}/2) + p_0 (r_{\max} - r_{\min})/(\text{FOV}/2)$ |
| 2 | $\sigma_w = \sigma_{\min}/(\text{FOV}/2) + p_1 (\sigma_{\max} - \sigma_{\min})/(\text{FOV}/2)$ |
| 3 | $s = s_{\min} + p_2 (s_{\max} - s_{\min})$ |
| 4 | $\eta = \frac{181\pi}{180}(2p_3 - 1)$ |
| 5 | Ring: $R_{ij} = \exp\left(-\frac{(r_{ij} - r)^2}{2\sigma_w^2}\right)$ |
| 6 | Asymmetry: $S_{ij} = 1 + s\cos(\phi_{ij} - \eta)$ |
| 7 | Crescent: $C = S \odot R$ |
| 8 | Disk: $D_{ij} = \frac{1}{2}\left(1 + \text{erf}\left(\frac{r - r_{ij}}{\sqrt{2}\sigma_w}\right)\right)$ |
| 9 | Normalize: $\hat{C} = C / (\sum C + \epsilon)$, $\hat{D} = D / (\sum D + \epsilon)$ |
| 10 | Combine: $I_{\text{cres}} = F_{\text{cres}}((1-f)\hat{C} + f\hat{D})$ |
| 11 | For each Gaussian $k$: add rotated elliptical Gaussian at $(x_k, y_k)$ with $(\sigma_{x,k}, \sigma_{y,k}, \theta_k)$ |
| 12 | Final normalize: $x = I / (\sum I + \epsilon)$ |

#### 4.2.2 Forward Operator Sub-procedure

| Step | Formula |
|------|---------|
| 1 | Flatten image: $\tilde{x} = \text{reshape}(x, (B, N_{\text{pix}}^2))$ |
| 2 | DFT: $V_{\text{re}} = \tilde{x} F_{\text{re}}$, $V_{\text{im}} = \tilde{x} F_{\text{im}}$ |
| 3 | Amplitude: $\|V_j\| = \sqrt{V_{\text{re},j}^2 + V_{\text{im},j}^2 + \epsilon_{\text{amp}}}$ |
| 4 | Phase: $\psi_j = \arctan2(V_{\text{im},j}, V_{\text{re},j})$ |
| 5 | Closure phase: $\Phi_m = (s_1 \psi_{i_1} + s_2 \psi_{i_2} + s_3 \psi_{i_3}) \cdot 180/\pi$ |
| 6 | Log closure amp: $\log A_m = \log\|V_{j_1}\| + \log\|V_{j_2}\| - \log\|V_{j_3}\| - \log\|V_{j_4}\|$ |

#### 4.2.3 Loss Aggregation

| Condition | Formula |
|-----------|---------|
| $\alpha_{\text{eff}} = 1$ (KL) | $\mathcal{L} = \frac{1}{B}\sum_{i=1}^B s \cdot L_i$ |
| $\alpha_{\text{eff}} \neq 1$ | $w_i = \text{softmax}(-(1-\alpha_{\text{eff}})L)_i$ (detached); $\mathcal{L} = \sum_{i=1}^B w_i \cdot s \cdot L_i$ |

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Save model | torch.save($\phi$, path) | $\phi$, path | checkpoint file |
| 2 | Generate evaluation samples | $z \sim \mathcal{N}(0, I)$, $x = G_\psi(\sigma(f_\phi^{-1}(z)))$ | trained $f_\phi$, $N_{\text{samples}}=50$ | imgs: $(N_{\text{samples}}, N_{\text{pix}}, N_{\text{pix}})$ |
| 3 | Compute mean image | $\bar{x} = \frac{1}{N_{\text{samples}}}\sum x_i$ | imgs | $\bar{x}$: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 4 | Compute std image | $\sigma_x = \sqrt{\frac{1}{N}\sum(x_i - \bar{x})^2}$ | imgs, $\bar{x}$ | $\sigma_x$: $(N_{\text{pix}}, N_{\text{pix}})$ |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k \geq N_{\text{epoch}}$
- **parameters**: 
  - $N_{\text{epoch}} = 10$ (default, configurable via `--n_epoch`)
- **check_frequency**: Every iteration (after parameter update)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Training Loss | $\mathcal{L}$ as defined in §1 | Final optimization objective value |
| Closure Phase $\chi^2$ | $\chi^2_\Phi = \frac{1}{N_{\text{cp}}}\sum_{j=1}^{N_{\text{cp}}} \frac{1 - \cos((\Phi_j^{\text{obs}} - \Phi_j)\pi/180)}{(\sigma_{\Phi,j}\pi/180)^2}$ | Reduced chi-squared for closure phases |
| Log Closure Amplitude $\chi^2$ | $\chi^2_{\log A} = \frac{1}{N_{\text{ca}}}\sum_{j=1}^{N_{\text{ca}}} \frac{(\log A_j^{\text{obs}} - \log A_j)^2}{\sigma_{\log A,j}^2}$ | Reduced chi-squared for log closure amplitudes |
| Total Flux Mean | $\bar{F} = \frac{1}{N_{\text{samples}}}\sum_{i=1}^{N_{\text{samples}}} \sum_{jk} x_{i,jk}$ | Mean total flux across samples |
| Total Flux Std | $\sigma_F = \sqrt{\frac{1}{N}\sum_i (F_i - \bar{F})^2}$ | Standard deviation of total flux |
| Mean Image | $\bar{x}_{jk} = \frac{1}{N_{\text{samples}}}\sum_i x_{i,jk}$ | Pixel-wise posterior mean |
| Std Image | $\sigma_{x,jk} = \sqrt{\frac{1}{N}\sum_i (x_{i,jk} - \bar{x}_{jk})^2}$ | Pixel-wise posterior standard deviation |