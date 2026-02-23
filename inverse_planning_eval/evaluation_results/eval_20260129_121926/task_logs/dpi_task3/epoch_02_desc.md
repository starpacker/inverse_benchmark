# Algorithmic Specification Document: Deep Probabilistic Imaging via Normalizing Flows

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\phi} \mathcal{L}_{\alpha}(\phi) = \begin{cases}
\mathbb{E}_{z \sim \mathcal{N}(0,I)}\left[\gamma \cdot \mathcal{D}(y, \hat{y}(z;\phi)) - \log |J_{f_\phi^{-1}}(z)| - \log p_{\text{sigmoid}}(\theta) + \frac{1}{2}\|z\|_2^2 \right] & \text{if } \alpha = 1 \\
\sum_{i=1}^{B} w_i^{(\alpha)} \cdot s \cdot \mathcal{L}_i & \text{if } \alpha \neq 1
\end{cases}$$

where $w_i^{(\alpha)} = \text{softmax}\left(-(1-\alpha)\mathcal{L}\right)_i$ are importance weights.

**data_fidelity_term**:
$$\mathcal{D}(y, \hat{y}) = \begin{cases}
\frac{\lambda_{\psi}}{N_{\psi}} \sum_{j=1}^{N_{\psi}} \frac{2(1 - \cos(\psi_j - \hat{\psi}_j))}{\sigma_{\psi,j}^2} + \frac{\lambda_{\mathcal{A}}}{N_{\mathcal{A}}} \sum_{k=1}^{N_{\mathcal{A}}} \frac{(\log \mathcal{A}_k - \log \hat{\mathcal{A}}_k)^2}{\sigma_{\mathcal{A},k}^2} & \text{(closure products)} \\
\frac{\lambda_V}{N_V} \sum_{j=1}^{N_V} \frac{|V_j - \hat{V}_j|^2}{\sigma_{V,j}^2} & \text{(visibilities)}
\end{cases}$$

**regularization_term**:
$$\mathcal{R}(\phi, z) = -\log |J_{f_\phi^{-1}}(z)| - \sum_{d=1}^{D}\left(-\theta_d - 2\log(1 + e^{-\theta_d})\right) + \frac{1}{2}\|z\|_2^2$$

where the middle term is the log-Jacobian of the sigmoid transformation.

**constraints**:
- $z \in \mathbb{R}^D$, $z \sim \mathcal{N}(0, I_D)$
- $\theta = f_\phi^{-1}(z) \in \mathbb{R}^D$ (unconstrained parameters)
- $\tilde{\theta} = \sigma(\theta) \in (0,1)^D$ (constrained parameters)
- $I(x) \geq 0$ for all pixels $x$ (image positivity, enforced by construction)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\phi$ | `primal` | varies | float32 | See network init | RealNVP network parameters |
| $z$ | `primal` | $(B, D)$ | float32 | $z \sim \mathcal{N}(0,I)$ | Latent samples |
| $\theta$ | `primal` | $(B, D)$ | float32 | $f_\phi^{-1}(z)$ | Unconstrained geometric parameters |
| $\tilde{\theta}$ | `primal` | $(B, D)$ | float32 | $\sigma(\theta)$ | Constrained parameters in $(0,1)^D$ |

### Dual Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $w^{(\alpha)}$ | `dual` | $(B,)$ | float32 | $\text{softmax}(-(1-\alpha)\mathcal{L})$ | Importance weights (detached) |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $D$ | `constants` | scalar | int | $4 + 6N_g + 2$ | Parameter dimension |
| $N_{\text{flow}}$ | `constants` | scalar | int | 16 | Number of flow blocks |
| $N_g$ | `constants` | scalar | int | 2 | Number of nuisance Gaussians |
| $N_{\text{pix}}$ | `constants` | scalar | int | 64 | Image dimension |
| $\text{FOV}$ | `constants` | scalar | float32 | 120 | Field of view (μas) |
| $\alpha$ | `constants` | scalar | float32 | 1.0 | Alpha-divergence parameter |
| $\gamma$ | `constants` | scalar | float32 | varies | Data weight (annealed) |
| $s$ | `constants` | scalar | float32 | $1/N_{\mathcal{A}}$ | Scale factor |
| $\eta$ | `constants` | scalar | float32 | $10^{-4}$ | Learning rate |
| $\epsilon$ | `constants` | scalar | float32 | $10^{-4}$ | Gradient clipping threshold |
| $F$ | `constants` | $(N_{\text{pix}}^2, N_V, 2)$ | float32 | DFT matrix | Discrete Fourier Transform matrix |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\psi$ | `observations` | $(N_\psi,)$ | float32 | from data | Observed closure phases (degrees) |
| $\log\mathcal{A}$ | `observations` | $(N_\mathcal{A},)$ | float32 | from data | Observed log closure amplitudes |
| $V$ | `observations` | $(2, N_V)$ | float32 | from data | Observed complex visibilities (real, imag) |
| $\sigma_\psi$ | `observations` | $(N_\psi,)$ | float32 | from data | Closure phase uncertainties |
| $\sigma_\mathcal{A}$ | `observations` | $(N_\mathcal{A},)$ | float32 | from data | Log closure amplitude uncertainties |
| $\sigma_V$ | `observations` | $(N_V,)$ | float32 | from data | Visibility uncertainties |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Load Observation Data
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load UVFITS |
| formula | $\text{obs} \leftarrow \text{load\_uvfits}(\text{path})$ |
| inputs | `obspath`: string |
| outputs | `obs`: observation object containing $(u,v,V,\sigma)$ |

### Step 3.2: Estimate Flux Constant
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Flux estimation |
| formula | $F_0 = \text{median}(\{|V_{ij}| : (i,j) \in \text{AP-AA baseline}\})$ |
| inputs | `obs`: observation object |
| outputs | `flux_const`: scalar float32 |

### Step 3.3: Compute DFT Matrix
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | DFT matrix construction |
| formula | $F_{pk} = \sum_{m,n} \text{pulse}(x_m, y_n) \exp\left(-2\pi i (u_k x_m + v_k y_n)\right)$, stored as $F \in \mathbb{R}^{N_{\text{pix}}^2 \times N_V \times 2}$ with $F[:,:,0] = \Re(F)$, $F[:,:,1] = \Im(F)$ |
| inputs | `simim.psize`: pixel size (rad), `uv`: $(N_V, 2)$ array |
| outputs | `dft_mat`: $(N_{\text{pix}}^2, N_V, 2)$ tensor |

### Step 3.4: Build Closure Phase Index Map
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Closure phase indexing |
| formula | For triangle $(t_1, t_2, t_3)$: find visibility indices $i_1, i_2, i_3$ and signs $s_1, s_2, s_3 \in \{-1, +1\}$ such that $\psi = s_1 \arg(V_{i_1}) + s_2 \arg(V_{i_2}) + s_3 \arg(V_{i_3})$ |
| inputs | `obs.cphase`: closure phase table, `obs.data`: visibility table |
| outputs | `cphase_ind_list`: 3 tensors of shape $(N_\psi,)$, `cphase_sign_list`: 3 tensors of shape $(N_\psi,)$ |

### Step 3.5: Build Closure Amplitude Index Map
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Closure amplitude indexing |
| formula | For quadrangle $(t_1,t_2,t_3,t_4)$: $\log\mathcal{A} = \log|V_{12}| + \log|V_{34}| - \log|V_{14}| - \log|V_{23}|$; store indices for baselines 12, 34, 14, 23 |
| inputs | `obs.camp`: closure amplitude table, `obs.data`: visibility table |
| outputs | `camp_ind_list`: 4 tensors of shape $(N_\mathcal{A},)$ |

### Step 3.6: Construct Pixel Grid
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Grid construction |
| formula | $\Delta = 1/N_{\text{pix}}$, $x_m = -1 + \Delta + 2m\Delta$ for $m=0,\ldots,N_{\text{pix}}-1$; $(G_x)_{mn} = x_n$, $(G_y)_{mn} = -x_m$, $G_r = \sqrt{G_x^2 + G_y^2}$, $G_\theta = \text{atan2}(G_y, G_x)$ |
| inputs | `npix`: int |
| outputs | `grid_x, grid_y`: $(N_{\text{pix}}, N_{\text{pix}})$, `grid_r, grid_theta`: $(N_{\text{pix}}, N_{\text{pix}})$ |

---

## 4. ALGORITHM STEPS

### Initialization

**Step I.1: Initialize Geometric Model**
| Field | Value |
|-------|-------|
| step_name | Geometric model setup |
| formula | Instantiate $G: (0,1)^D \to \mathbb{R}_{\geq 0}^{N_{\text{pix}} \times N_{\text{pix}}}$ with parameter ranges: $r \in [r_{\min}, r_{\max}]$, $\sigma_w \in [\sigma_{\min}, \sigma_{\max}]$, $s \in [s_{\min}, s_{\max}]$, $\eta \in [-\pi, \pi]$, etc. |
| inputs | `npix, fov, n_gaussian, ranges` |
| outputs | `img_converter`: nn.Module |

**Step I.2: Initialize RealNVP Flow**
| Field | Value |
|-------|-------|
| step_name | Flow network initialization |
| formula | $f_\phi = \pi_{N_{\text{flow}}} \circ \text{Flow}_{N_{\text{flow}}} \circ \cdots \circ \pi_1 \circ \text{Flow}_1$ where each $\text{Flow}_k = \text{Flip} \circ \text{Coupling}_2 \circ \text{ActNorm}_2 \circ \text{Flip} \circ \text{Coupling}_1 \circ \text{ActNorm}_1$ |
| inputs | `nparams, n_flow` |
| outputs | `params_generator`: nn.Module with parameters $\phi$ |
| computational_note | $\pi_k$ are fixed random permutations; Flip reverses dimension order |

**Step I.3: Initialize ActNorm Parameters**
| Field | Value |
|-------|-------|
| step_name | ActNorm data-dependent init |
| formula | On first forward pass with input $x$: $\mu \leftarrow \text{mean}(x)$, $\sigma \leftarrow \text{std}(x)$; set $b \leftarrow -\mu$, $\log s^{-1} \leftarrow \log(\sigma + 10^{-6})$ |
| inputs | First batch of data |
| outputs | Initialized `loc`, `log_scale_inv` parameters |

**Step I.4: Initialize Optimizer**
| Field | Value |
|-------|-------|
| step_name | Adam optimizer setup |
| formula | $\text{optimizer} \leftarrow \text{Adam}(\phi, \eta)$ with default $\beta_1=0.9$, $\beta_2=0.999$ |
| inputs | `params_generator.parameters(), lr` |
| outputs | `optimizer`: Adam instance |

**Step I.5: Compute Weight Configuration**
| Field | Value |
|-------|-------|
| step_name | Loss weight setup |
| formula | If `data_product='cphase_logcamp'`: $\lambda_\psi = N_\psi/N_\mathcal{A}$, $\lambda_\mathcal{A} = 1$, $\lambda_V = 0$, $s = 1/N_\mathcal{A}$; else: $\lambda_V = 1$, $\lambda_\psi = \lambda_\mathcal{A} = 0$, $s = 1/N_V$ |
| inputs | `data_product, obs` |
| outputs | `cphase_weight, camp_weight, vis_weight, scale_factor` |

---

### Iteration Loop

**--- BEGIN ITERATION LOOP ---** (for $k = 0, 1, \ldots, N_{\text{epoch}} - 1$)

**Step L.1: Compute Data Weight (Annealing)**
| Field | Value |
|-------|-------|
| step_name | Annealing schedule |
| formula | $\gamma_k = \min\left(10^{-k_0 + k/\tau}, 1\right)$ |
| inputs | `k`: current epoch, `start_order` $k_0$, `decay_rate` $\tau$ |
| outputs | `data_weight`: scalar |
| computational_note | Gradually increases data fidelity weight from $10^{-k_0}$ to 1 |

**Step L.2: Sample Latent Variables**
| Field | Value |
|-------|-------|
| step_name | Latent sampling |
| formula | $z^{(i)} \sim \mathcal{N}(0, I_D)$ for $i = 1, \ldots, B$ |
| inputs | `n_batch` $B$, `nparams` $D$ |
| outputs | `z_sample`: $(B, D)$ tensor |

**Step L.3: RealNVP Reverse Pass (Generate Parameters)**

**Step L.3.1: Apply Inverse Permutation**
| Field | Value |
|-------|-------|
| step_name | Inverse permutation |
| formula | For block $j = N_{\text{flow}}-1, \ldots, 0$: $z \leftarrow z[:, \pi_j^{-1}]$ |
| inputs | `z`: $(B, D)$, `inverse_orders` |
| outputs | Permuted `z` |

**Step L.3.2: Inverse Flow Block**

**Step L.3.2a: Flip Dimensions**
| Field | Value |
|-------|-------|
| step_name | Dimension flip |
| formula | $z \leftarrow z[:, D-1::-1]$ (reverse order) |
| inputs | `z`: $(B, D)$ |
| outputs | Flipped `z` |

**Step L.3.2b: Inverse Affine Coupling**
| Field | Value |
|-------|-------|
| step_name | Inverse coupling |
| formula | Split $z = [z_a, z_b]$; compute $(\log \tilde{s}, t) = \text{NN}(z_a)$, $\log s = \tanh(\log \tilde{s})$; $\theta_b = z_b \cdot e^{-\log s} - t$; output $[\theta_a, \theta_b] = [z_a, \theta_b]$; $\Delta\log|J| = -\sum_d \log s_d$ |
| inputs | `z`: $(B, D)$ |
| outputs | `theta`: $(B, D)$, `logdet_coupling`: $(B,)$ |
| computational_note | NN is MLP: Linear→LeakyReLU→BN→Linear→LeakyReLU→BN→ZeroFC |

**Step L.3.2c: Inverse ActNorm**
| Field | Value |
|-------|-------|
| step_name | Inverse activation normalization |
| formula | $\theta \leftarrow \theta \cdot e^{\log s^{-1}} - b$; $\Delta\log|J| = D \cdot \log s^{-1}$ |
| inputs | `theta`: $(B, D)$, `loc` $b$, `log_scale_inv` |
| outputs | `theta`: $(B, D)$, `logdet_actnorm`: scalar |

**Step L.3.3: Accumulate Log-Determinant**
| Field | Value |
|-------|-------|
| step_name | Jacobian accumulation |
| formula | $\log|J_{f^{-1}}| \leftarrow \log|J_{f^{-1}}| + \Delta\log|J|_{\text{coupling}} + \Delta\log|J|_{\text{actnorm}}$ |
| inputs | Running `logdet`, step contributions |
| outputs | Updated `logdet`: $(B,)$ |

**Step L.4: Apply Sigmoid Transformation**
| Field | Value |
|-------|-------|
| step_name | Sigmoid constraint |
| formula | $\tilde{\theta} = \sigma(\theta) = \frac{1}{1 + e^{-\theta}}$; $\log|J_\sigma| = \sum_{d=1}^D \left(-\theta_d - 2\log(1 + e^{-\theta_d})\right)$; $\log|J| \leftarrow \log|J| + \log|J_\sigma|$ |
| inputs | `theta`: $(B, D)$ |
| outputs | `params`: $(B, D)$ in $(0,1)^D$, updated `logdet` |

**Step L.5: Generate Image via Geometric Model**

**Step L.5.1: Map Parameters to Physical Quantities**
| Field | Value |
|-------|-------|
| step_name | Parameter scaling |
| formula | $r = \frac{r_{\min} + \tilde{\theta}_0 (r_{\max} - r_{\min})}{0.5 \cdot \text{FOV}}$, $\sigma_w = \frac{\sigma_{\min} + \tilde{\theta}_1 (\sigma_{\max} - \sigma_{\min})}{0.5 \cdot \text{FOV}}$, $s = s_{\min} + \tilde{\theta}_2(s_{\max} - s_{\min})$, $\eta = \frac{181\pi}{180}(2\tilde{\theta}_3 - 1)$ |
| inputs | `params`: $(B, D)$, range constants |
| outputs | `r, sigma, s, eta`: each $(B, 1, 1)$ |

**Step L.5.2: Compute Crescent Component**
| Field | Value |
|-------|-------|
| step_name | Ring and asymmetry |
| formula | $\text{ring} = \exp\left(-\frac{(G_r - r)^2}{2\sigma_w^2}\right)$; $S = 1 + s \cos(G_\theta - \eta)$; $\text{crescent} = S \odot \text{ring}$ |
| inputs | `grid_r, grid_theta`: $(N_{\text{pix}}, N_{\text{pix}})$, `r, sigma, s, eta` |
| outputs | `crescent`: $(B, N_{\text{pix}}, N_{\text{pix}})$ |

**Step L.5.3: Compute Disk (Floor) Component**
| Field | Value |
|-------|-------|
| step_name | Filled disk |
| formula | $\text{disk} = \frac{1}{2}\left(1 + \text{erf}\left(\frac{r - G_r}{\sqrt{2}\sigma_w}\right)\right)$ |
| inputs | `grid_r, r, sigma` |
| outputs | `disk`: $(B, N_{\text{pix}}, N_{\text{pix}})$ |

**Step L.5.4: Normalize and Combine**
| Field | Value |
|-------|-------|
| step_name | Flux normalization |
| formula | $\text{crescent} \leftarrow \frac{\text{crescent}}{\sum_{x,y}\text{crescent} + \epsilon}$; $\text{disk} \leftarrow \frac{\text{disk}}{\sum_{x,y}\text{disk} + \epsilon}$; $I_{\text{cres}} = F_c \left((1-f)\text{crescent} + f \cdot \text{disk}\right)$ where $f$ is floor fraction, $F_c$ is crescent flux |
| inputs | `crescent, disk, floor, crescent_flux` |
| outputs | `crescent`: $(B, N_{\text{pix}}, N_{\text{pix}})$ |

**Step L.5.5: Add Nuisance Gaussians**
| Field | Value |
|-------|-------|
| step_name | Gaussian nuisance |
| formula | For $k = 1, \ldots, N_g$: $x_c = G_x - \mu_{x,k}$, $y_c = G_y - \mu_{y,k}$; $x_{\text{rot}} = x_c \cos\phi_k + y_c \sin\phi_k$, $y_{\text{rot}} = -x_c \sin\phi_k + y_c \cos\phi_k$; $\mathcal{G}_k = \frac{1}{2\pi\sigma_{x,k}\sigma_{y,k}}\exp\left(-\frac{x_{\text{rot}}^2}{2\sigma_{x,k}^2} - \frac{y_{\text{rot}}^2}{2\sigma_{y,k}^2}\right)$; $\mathcal{G}_k \leftarrow \frac{A_k \mathcal{G}_k}{\sum_{x,y}\mathcal{G}_k + \epsilon}$; $I \leftarrow I + \mathcal{G}_k$ |
| inputs | Gaussian parameters from `params`, `grid_x, grid_y` |
| outputs | `img`: $(B, N_{\text{pix}}, N_{\text{pix}})$ |

**Step L.5.6: Final Normalization**
| Field | Value |
|-------|-------|
| step_name | Total flux normalization |
| formula | $I \leftarrow \frac{I}{\sum_{x,y} I + \epsilon}$ |
| inputs | `img` |
| outputs | `img`: $(B, N_{\text{pix}}, N_{\text{pix}})$, normalized to unit sum |

**Step L.6: Forward Operator (Compute Observables)**

**Step L.6.1: Compute Complex Visibilities**
| Field | Value |
|-------|-------|
| step_name | DFT application |
| formula | Flatten $I$ to $(B, N_{\text{pix}}^2)$; $\hat{V}_{\Re} = I \cdot F[:,:,0]$, $\hat{V}_{\Im} = I \cdot F[:,:,1]$; $\hat{V} = [\hat{V}_{\Re}, \hat{V}_{\Im}] \in \mathbb{R}^{B \times 2 \times N_V}$ |
| inputs | `img`: $(B, N_{\text{pix}}^2)$, `dft_mat`: $(N_{\text{pix}}^2, N_V, 2)$ |
| outputs | `vis_torch`: $(B, 2, N_V)$ |

**Step L.6.2: Compute Visibility Amplitudes**
| Field | Value |
|-------|-------|
| step_name | Amplitude extraction |
| formula | $|\hat{V}_j| = \sqrt{\hat{V}_{\Re,j}^2 + \hat{V}_{\Im,j}^2 + \epsilon}$ |
| inputs | `vis_torch` |
| outputs | `vis_amp`: $(B, N_V)$ |

**Step L.6.3: Compute Closure Phases**
| Field | Value |
|-------|-------|
| step_name | Closure phase computation |
| formula | For triangle $m$ with indices $(i_1, i_2, i_3)$ and signs $(s_1, s_2, s_3)$: $\hat{\psi}_m = \frac{180}{\pi}\left(s_1 \text{atan2}(\hat{V}_{\Im,i_1}, \hat{V}_{\Re,i_1}) + s_2 \text{atan2}(\hat{V}_{\Im,i_2}, \hat{V}_{\Re,i_2}) + s_3 \text{atan2}(\hat{V}_{\Im,i_3}, \hat{V}_{\Re,i_3})\right)$ |
| inputs | `vis_torch`, `cphase_ind_list`, `cphase_sign_list` |
| outputs | `cphase`: $(B, N_\psi)$ |

**Step L.6.4: Compute Log Closure Amplitudes**
| Field | Value |
|-------|-------|
| step_name | Log closure amplitude computation |
| formula | For quadrangle with baseline indices $(i_{12}, i_{34}, i_{14}, i_{23})$: $\log\hat{\mathcal{A}} = \log|\hat{V}_{i_{12}}| + \log|\hat{V}_{i_{34}}| - \log|\hat{V}_{i_{14}}| - \log|\hat{V}_{i_{23}}|$ |
| inputs | `vis_torch`, `camp_ind_list` |
| outputs | `logcamp`: $(B, N_\mathcal{A})$ |

**Step L.7: Compute Data Fidelity Loss**

**Step L.7.1: Closure Phase Loss**
| Field | Value |
|-------|-------|
| step_name | Phase chi-squared |
| formula | $\mathcal{L}_\psi = \frac{2}{N_\psi}\sum_{j=1}^{N_\psi} \frac{1 - \cos\left((\psi_j - \hat{\psi}_j)\frac{\pi}{180}\right)}{\left(\sigma_{\psi,j}\frac{\pi}{180}\right)^2}$ |
| inputs | `cphase_true, cphase, sigma_cphase` |
| outputs | `loss_cphase`: $(B,)$ |
| computational_note | Uses von Mises-like loss for angular data |

**Step L.7.2: Log Closure Amplitude Loss**
| Field | Value |
|-------|-------|
| step_name | Amplitude chi-squared |
| formula | $\mathcal{L}_\mathcal{A} = \frac{1}{N_\mathcal{A}}\sum_{k=1}^{N_\mathcal{A}} \frac{(\log\mathcal{A}_k - \log\hat{\mathcal{A}}_k)^2}{\sigma_{\mathcal{A},k}^2}$ |
| inputs | `logcamp_true, logcamp, sigma_logcamp