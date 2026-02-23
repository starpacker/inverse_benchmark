## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Deep Probabilistic Imaging with RealNVP Normalizing Flow
- **algorithm_family**: Variational Inference / Normalizing Flow / Alpha-Divergence Optimization
- **summary**: Learns a RealNVP normalizing flow to approximate the posterior distribution over geometric model parameters for radio interferometric image reconstruction, minimizing a weighted combination of data fidelity (closure phases and log closure amplitudes) and the negative log-density under a standard Gaussian prior.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$
\min_{\phi} \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ w(k) \cdot \mathcal{L}_{\text{data}}(G_\theta(f_\phi^{-1}(z))) - \log |\det J_{f_\phi^{-1}}(z)| - \frac{1}{2}\|z\|_2^2 \right]
$$

where the expectation is approximated via importance-weighted samples when using alpha-divergence.

### data_fidelity_term

**For `data_product='cphase_logcamp'`:**
$$
\mathcal{L}_{\text{data}}(x) = \frac{1}{2 \cdot s_f} \left[ \lambda_{\text{camp}} \cdot \frac{1}{N_{\text{camp}}} \sum_{i=1}^{N_{\text{camp}}} \frac{(\log C_i^{\text{true}} - \log C_i(x))^2}{\sigma_{\text{camp},i}^2} + \lambda_{\text{cphase}} \cdot \frac{1}{N_{\text{cphase}}} \sum_{j=1}^{N_{\text{cphase}}} \frac{2(1 - \cos((\psi_j^{\text{true}} - \psi_j(x)) \cdot \pi/180))}{(\sigma_{\text{cphase},j} \cdot \pi/180)^2} \right]
$$

**For `data_product='vis'`:**
$$
\mathcal{L}_{\text{data}}(x) = \frac{1}{2 \cdot s_f} \cdot \frac{1}{N_{\text{vis}}} \sum_{k=1}^{N_{\text{vis}}} \frac{(\Re(V_k^{\text{true}}) - \Re(V_k(x)))^2 + (\Im(V_k^{\text{true}}) - \Im(V_k(x)))^2}{\sigma_{\text{vis},k}^2}
$$

where $s_f$ is the scale factor.

### regularization_term

**Negative log-determinant of flow transformation:**
$$
\mathcal{R}_{\text{logdet}}(\phi, z) = -\log |\det J_{f_\phi^{-1}}(z)|
$$

**Standard Gaussian prior on latent space:**
$$
\mathcal{R}_{\text{prior}}(z) = -\frac{1}{2} \sum_{d=1}^{D} z_d^2
$$

### constraints
$$
\theta \in [0,1]^D \quad \text{(enforced via sigmoid transformation)}
$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\phi$ | Varies | float32 | See RealNVP init | All learnable parameters of the RealNVP normalizing flow |
| $z$ | $(N_{\text{batch}}, D)$ | float32 | $\mathcal{N}(0,I)$ | Latent samples from standard Gaussian |
| $\theta_{\text{samp}}$ | $(N_{\text{batch}}, D)$ | float32 | $f_\phi^{-1}(z)$ | Transformed samples (pre-sigmoid) |
| $\theta$ | $(N_{\text{batch}}, D)$ | float32 | $\sigma(\theta_{\text{samp}})$ | Geometric model parameters in $[0,1]^D$ |

### 2.2 Dual Variables
None (gradient-based optimization without explicit dual formulation).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| `logdet` | $(N_{\text{batch}},)$ | float32 | 0 | Accumulated log-determinant through flow |
| `det_sigmoid` | $(N_{\text{batch}},)$ | float32 | Computed | Log-determinant of sigmoid transformation |
| `loss_data` | $(N_{\text{batch}},)$ | float32 | Computed | Per-sample data fidelity loss |
| `logprob` | $(N_{\text{batch}},)$ | float32 | Computed | Negative log-probability under flow |
| `rej_weights` | $(N_{\text{batch}},)$ | float32 | Computed | Softmax importance weights for alpha-divergence |
| `data_weight` | scalar | float32 | $10^{-\text{start\_order}}$ | Annealing weight for data term |

#### [INNER] ActNorm Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| `loc` | $(1,)$ | float32 | 0 (data-dependent) | Location shift parameter |
| `log_scale_inv` | $(1,)$ | float32 | 0 (data-dependent) | Log of inverse scale |
| `initialized` | scalar | uint8 | 0 | Flag for data-dependent initialization |

#### [INNER] AffineCoupling Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| `log_s` | $(N_{\text{batch}}, D/2)$ | float32 | Computed | Log-scale from coupling network |
| `t` | $(N_{\text{batch}}, D/2)$ | float32 | Computed | Translation from coupling network |

#### [INNER] ZeroFC Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| `fc.weight` | $(\text{out}, \text{in})$ | float32 | 0 | Zero-initialized weights |
| `fc.bias` | $(\text{out},)$ | float32 | 0 | Zero-initialized bias |
| `scale` | $(\text{out},)$ | float32 | 0 | Learnable output scale |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `eps` (forward) | $10^{-16}$ | Numerical stability for amplitude computation |
| `eps` (geometric) | $10^{-4}$ | Numerical stability for normalization |
| `n_flow` | 16 (default) | Number of flow blocks |
| `n_gaussian` | 2 (default) | Number of nuisance Gaussian components |
| `n_epoch` | 10 (default) | Training iterations |
| `n_batch` | 64 | Batch size |
| `lr` | $10^{-4}$ | Learning rate |
| `clip` | $10^{-4}$ | Gradient clipping threshold |
| `start_order` | 4 | Initial annealing exponent |
| `decay_rate` | 2000 | Annealing decay rate |
| `alpha_divergence` | 1.0 (default) | Alpha parameter for Rényi divergence |
| `beta` | 0.0 (default) | Alternative importance weight parameter |
| `logdet_weight` | 1.0 | Weight for log-determinant term |
| `r_range` | $[10.0, 40.0]$ | Ring radius range (microarcseconds) |
| `asym_range` | $[10^{-3}, 0.99]$ | Asymmetry parameter range |
| `width_range` | $[1.0, 40.0]$ | Ring width range (microarcseconds) |
| `floor_range` | $[0.0, 1.0]$ | Floor parameter range |
| `flux_range` | $[0.8, 1.2]$ | Total flux scaling range |
| `crescent_flux_range` | $[10^{-3}, 2.0]$ | Crescent flux range |
| `shift_range` | $[-200.0, 200.0]$ | Gaussian center shift range |
| `sigma_range` | $[1.0, 100.0]$ | Gaussian width range |
| `gaussian_scale_range` | $[10^{-3}, 2.0]$ | Gaussian amplitude range |
| `seqfrac` | $1/16$ | Hidden dimension fraction for coupling |
| `leaky_relu_slope` | 0.01 | LeakyReLU negative slope |
| `batchnorm_eps` | $10^{-2}$ | BatchNorm epsilon |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $V^{\text{true}}$ | $(2, N_{\text{vis}})$ | float32 | Complex visibilities (real, imag) |
| $|V|^{\text{true}}$ | $(N_{\text{vis}},)$ | float32 | Visibility amplitudes |
| $\psi^{\text{true}}$ | $(N_{\text{cphase}},)$ | float32 | Closure phases (degrees) |
| $\log C^{\text{true}}$ | $(N_{\text{camp}},)$ | float32 | Log closure amplitudes |
| $\sigma_{\text{vis}}$ | $(N_{\text{vis}},)$ | float32 | Visibility uncertainties |
| $\sigma_{\text{cphase}}$ | $(N_{\text{cphase}},)$ | float32 | Closure phase uncertainties (degrees) |
| $\sigma_{\text{camp}}$ | $(N_{\text{camp}},)$ | float32 | Log closure amplitude uncertainties |
| $F$ | $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ | float32 | DFT matrix (real, imag components) |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 0 | Flux Estimation | $F_{\text{const}} = \text{median}(\{|V_{ij}|\}_{(i,j) \in \text{AA-AP}})$ | Raw visibilities | `flux_const`: scalar |
| 1 | Prior Image Construction | $I_{\text{prior}} = \text{blur}(\text{tophat}(F_{\text{const}}, \text{FWHM}/2), \text{res})$ | `flux_const`, `prior_fwhm`, resolution | `prior`: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 2 | DFT Matrix Computation | $F_{pk} = \exp(-2\pi i (u_k x_p + v_k y_p)) \cdot \text{pulse}(u_k, v_k)$, stored as $[F^{\Re}, F^{\Im}]$ | UV coordinates, pixel positions, pulse function | `dft_mat`: $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ |
| 3 | Closure Phase Index Mapping | For each triangle $(t_1, t_2, t_3)$: find visibility indices and signs based on baseline orientation | Observation metadata | `cphase_ind_list`: 3 tensors of $(N_{\text{cphase}},)$; `cphase_sign_list`: 3 tensors of $(N_{\text{cphase}},)$ |
| 4 | Closure Amplitude Index Mapping | For each quadrangle: find 4 visibility indices for $\frac{|V_{12}||V_{34}|}{|V_{14}||V_{23}|}$ | Observation metadata | `camp_ind_list`: 4 tensors of $(N_{\text{camp}},)$ |
| 5 | Pixel Grid Construction | $x_i = -1 + (2i+1)/N_{\text{pix}}$ for $i=0,\ldots,N_{\text{pix}}-1$; $r_{ij} = \sqrt{x_i^2 + y_j^2}$; $\theta_{ij} = \arctan2(y_j, x_i)$ | `npix` | `grid_x`, `grid_y`, `grid_r`, `grid_theta`: all $(N_{\text{pix}}, N_{\text{pix}})$ |
| 6 | Weight Computation | $\lambda_{\text{cphase}} = N_{\text{cphase}}/N_{\text{camp}}$, $\lambda_{\text{camp}} = 1$, $s_f = 1/N_{\text{camp}}$, $s_{\text{logdet}} = 2 \cdot w_{\text{logdet}} / N_{\text{camp}}$ | Observation counts | `cphase_weight`, `camp_weight`, `scale_factor`, `logdet_scale`: scalars |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Device Setup | Assign CUDA device if available, else CPU | `cuda` index | `device` |
| 2 | Geometric Model Init | Instantiate `SimpleCrescentNuisanceFloor_Param2Img` with parameter ranges; $D = 4 + 6 \cdot N_{\text{gauss}} + 2$ | `npix`, `fov`, `n_gaussian`, ranges | `img_converter` module, `nparams`=$D$ |
| 3 | Flow Model Init | Instantiate `RealNVP` with $N_{\text{flow}}$ blocks; each block contains 2 ActNorm + 2 AffineCoupling layers | `nparams`, `n_flow` | `params_generator` module |
| 4 | Permutation Init | For each flow block $i$: $\pi_i = \text{RandomState}(i).\text{permutation}(D)$; $\pi_i^{-1} = \text{argsort}(\pi_i)$ | `n_flow`, `nparams` | `orders`, `inverse_orders`: lists of $N_{\text{flow}}$ permutation arrays |
| 5 | Optimizer Init | $\text{Adam}(\phi, \text{lr}=10^{-4})$ | Learning rate | `optimizer` |
| 6 | Alpha-Divergence Parameter | If $\beta=0$: $\alpha = \alpha_{\text{divergence}}$; else $\alpha = 1 - \beta \cdot s_f$ | `alpha_divergence`, `beta`, `scale_factor` | `alpha_div` |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (epoch $k = 0, 1, \ldots, N_{\text{epoch}}-1$)

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Annealing Weight | $w(k) = \min(10^{-\text{start\_order} + k/\text{decay\_rate}}, 1)$ | `start_order`, `decay_rate`, $k$ | `data_weight`: scalar |
| 2 | Latent Sampling | $z \sim \mathcal{N}(0, I_D)$, shape $(N_{\text{batch}}, D)$ | `n_batch`, `nparams` | $z$: $(N_{\text{batch}}, D)$ |

**BEGIN_INNER_LOOP** (RealNVP Reverse Pass through $N_{\text{flow}}$ blocks)

For $i = N_{\text{flow}}-1, \ldots, 0$:

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 3a | Inverse Permutation | $\tilde{z} \leftarrow z[:, \pi_i^{-1}]$ | $z$, `inverse_orders[i]` | $\tilde{z}$ |
| 3b | Reverse Flow Block | See Flow Reverse below | $\tilde{z}$ | $z$, $\Delta\log|\det J|$ |

**Flow Reverse** (single block):

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| i | Reverse Permute | $u \leftarrow z[:, D-1::-1]$ (reverse indices) | $z$ | $u$ |
| ii | AffineCoupling2 Reverse | $u_a, u_b = \text{split}(u)$; $\log s, t = \text{net}(u_a)$; $\log s \leftarrow \tanh(\log s)$; $z_b = u_b / \exp(\log s) - t$; output $= [u_a, z_b]$; $\Delta\log|\det J| = -\sum \log s$ | $u$ | output, $\Delta\log|\det J|$ |
| iii | ActNorm2 Reverse | $z \leftarrow u \cdot \exp(\log s_{\text{inv}}) - \text{loc}$; $\Delta\log|\det J| = -D \cdot \log s_{\text{inv}}$ | $u$ | $z$, $\Delta\log|\det J|$ |
| iv | Reverse Permute | $u \leftarrow z[:, D-1::-1]$ | $z$ | $u$ |
| v | AffineCoupling1 Reverse | Same as step ii | $u$ | output, $\Delta\log|\det J|$ |
| vi | ActNorm1 Reverse | Same as step iii | output | $z$, $\Delta\log|\det J|$ |

**END_INNER_LOOP**

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 4 | Sigmoid Transform | $\theta = \sigma(\theta_{\text{samp}}) = \frac{1}{1 + e^{-\theta_{\text{samp}}}}$ | $\theta_{\text{samp}}$ | $\theta$: $(N_{\text{batch}}, D)$ |
| 5 | Sigmoid Log-Det | $\log|\det J_\sigma| = \sum_{d=1}^{D} \left( -\theta_{\text{samp},d} - 2 \log(1 + e^{-\theta_{\text{samp},d}}) \right)$ | $\theta_{\text{samp}}$ | `det_sigmoid`: $(N_{\text{batch}},)$ |
| 6 | Total Log-Det | $\log|\det J| \leftarrow \log|\det J| + \log|\det J_\sigma|$ | `logdet`, `det_sigmoid` | `logdet`: $(N_{\text{batch}},)$ |

**BEGIN_INNER_LOOP** (Geometric Model: Parameter → Image)

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 7a | Parameter Scaling | $r = r_{\min}/(\text{fov}/2) + \theta_0 \cdot (r_{\max} - r_{\min})/(\text{fov}/2)$; similarly for $\sigma_w$, $s$, $\eta = \frac{181\pi}{180}(2\theta_3 - 1)$, floor, crescent_flux, Gaussian parameters | $\theta$, ranges, `fov` | Scaled parameters |
| 7b | Ring Generation | $\text{ring}_{ij} = \exp\left( -\frac{(r_{ij}^{\text{grid}} - r)^2}{2\sigma_w^2} \right)$ | $r$, $\sigma_w$, `grid_r` | `ring`: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ |
| 7c | Asymmetry Modulation | $S_{ij} = 1 + s \cdot \cos(\theta_{ij}^{\text{grid}} - \eta)$; $\text{crescent} = S \odot \text{ring}$ | $s$, $\eta$, `grid_theta`, `ring` | `crescent` |
| 7d | Disk Generation | $\text{disk}_{ij} = \frac{1}{2}\left(1 + \text{erf}\left(\frac{r - r_{ij}^{\text{grid}}}{\sqrt{2}\sigma_w}\right)\right)$ | $r$, $\sigma_w$, `grid_r` | `disk` |
| 7e | Normalization | $\text{crescent} \leftarrow \text{crescent} / (\sum_{ij} \text{crescent}_{ij} + \epsilon)$; same for disk | `crescent`, `disk`, $\epsilon=10^{-4}$ | Normalized components |
| 7f | Floor Mixing | $\text{img} = F_{\text{crescent}} \cdot ((1-f) \cdot \text{crescent} + f \cdot \text{disk})$ | `crescent_flux`, `floor`, components | `img` |
| 7g | Gaussian Addition | For each $k=1,\ldots,N_{\text{gauss}}$: $x_c = x^{\text{grid}} - x_k$, $y_c = y^{\text{grid}} - y_k$; rotate by $\theta_k$; $G_k = \frac{1}{2\pi\sigma_{x,k}\sigma_{y,k}} \exp\left(-\frac{1}{2}\left(\frac{x_{\text{rot}}^2}{\sigma_{x,k}^2} + \frac{y_{\text{rot}}^2}{\sigma_{y,k}^2}\right)\right)$; normalize; $\text{img} \leftarrow \text{img} + \alpha_k \cdot G_k$ | Gaussian params, grids | `img` with Gaussians |
| 7h | Final Normalization | $\text{img} \leftarrow \text{img} / (\sum_{ij} \text{img}_{ij} + \epsilon)$ | `img` | `img`: $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ |

**END_INNER_LOOP**

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 8 | Flatten Image | $x = \text{reshape}(\text{img}, (N_{\text{batch}}, N_{\text{pix}}^2))$ | `img` | $x$: $(N_{\text{batch}}, N_{\text{pix}}^2)$ |
| 9 | DFT [FORWARD] | $V^{\Re} = x \cdot F^{\Re}$; $V^{\Im} = x \cdot F^{\Im}$; $V = [V^{\Re}, V^{\Im}]$ | $x$, `dft_mat` | $V$: $(N_{\text{batch}}, 2, N_{\text{vis}})$ |
| 10 | Visibility Amplitude | $|V_k| = \sqrt{(V_k^{\Re})^2 + (V_k^{\Im})^2 + 10^{-16}}$ | $V$ | `vis_amp`: $(N_{\text{batch}}, N_{\text{vis}})$ |
| 11 | Closure Phase | $\phi_j = \text{sign}_1 \cdot \arctan2(V^{\Im}_{i_1}, V^{\Re}_{i_1}) + \text{sign}_2 \cdot \arctan2(V^{\Im}_{i_2}, V^{\Re}_{i_2}) + \text{sign}_3 \cdot \arctan2(V^{\Im}_{i_3}, V^{\Re}_{i_3})$; $\psi_j = \phi_j \cdot 180/\pi$ | $V$, indices, signs | `cphase`: $(N_{\text{batch}}, N_{\text{cphase}})$ degrees |
| 12 | Log Closure Amplitude | $\log C_m = \log|V_{i_{12}}| + \log|V_{i_{34}}| - \log|V_{i_{14}}| - \log|V_{i_{23}}|$ | $V$, indices | `logcamp`: $(N_{\text{batch}}, N_{\text{camp}})$ |
| 13 | Closure Phase Loss | $\ell_{\text{cphase}} = \frac{2}{N_{\text{cphase}}} \sum_j \frac{1 - \cos((\psi_j^{\text{true}} - \psi_j) \cdot \pi/180)}{(\sigma_{\text{cphase},j} \cdot \pi/180)^2}$ | `cphase`, `cphase_true`, `sigma_cphase` | `loss_cphase`: $(N_{\text{batch}},)$ |
| 14 | Log Camp Loss | $\ell_{\text{camp}} = \frac{1}{N_{\text{camp}}} \sum_m \frac{(\log C_m^{\text{true}} - \log C_m)^2}{\sigma_{\text{camp},m}^2}$ | `logcamp`, `logcamp_true`, `sigma_logcamp` | `loss_camp`: $(N_{\text{batch}},)$ |
| 15 | Combined Data Loss | $\mathcal{L}_{\text{data}} = \frac{0.5}{s_f}(\lambda_{\text{camp}} \cdot \ell_{\text{camp}} + \lambda_{\text{cphase}} \cdot \ell_{\text{cphase}})$ | losses, weights, `scale_factor` | `loss_data`: $(N_{\text{batch}},)$ |
| 16 | Log-Probability | $\log p = -\log|\det J| - \frac{1}{2}\sum_d z_d^2$ | `logdet`, $z$ | `logprob`: $(N_{\text{batch}},)$ |
| 17 | Total Loss (per sample) | $L_n = w(k) \cdot \mathcal{L}_{\text{data},n} + \log p_n$ | `data_weight`, `loss_data`, `logprob` | `loss`: $(N_{\text{batch}},)$ |
| 18 | Aggregation (KL) | If $\alpha=1$: $\mathcal{L} = s_f \cdot \frac{1}{N_{\text{batch}}} \sum_n L_n$ | `loss`, `scale_factor` | `loss`: scalar |
| 19 | Aggregation (Alpha) | If $\alpha \neq 1$: $w_n = \text{softmax}(-(1-\alpha) \cdot L)_n$; $\mathcal{L} = s_f \cdot \sum_n w_n \cdot L_n$ | `loss`, `alpha_div`, `scale_factor` | `loss`: scalar |
| 20 | Gradient Computation | $\nabla_\phi \mathcal{L}$ via backpropagation | `loss` | gradients |
| 21 | Gradient Clipping | $\nabla_\phi \leftarrow \text{clip\_norm}(\nabla_\phi, \text{max\_norm}=10^{-4})$ | gradients, `clip` | clipped gradients |
| 22 | Parameter Update | $\phi \leftarrow \text{Adam\_step}(\phi, \nabla_\phi)$ | $\phi$, gradients | updated $\phi$ |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Save Model State | Store `params_generator.state_dict()` | Trained model | `model_state_dict` |
| 2 | Record Final Loss | $\mathcal{L}_{\text{final}} = \mathcal{L}$ from last epoch | Last loss | `final_loss`: scalar |
| 3 | Generate Posterior Samples | $z \sim \mathcal{N}(0,I)$; $\theta = \sigma(f_\phi^{-1}(z))$; $\text{imgs} = G_\theta(\theta)$ | Trained model, $N_{\text{samples}}$ | `sample_images`: $(N_{\text{samples}}, N_{\text{pix}}, N_{\text{pix}})$ |
| 4 | Posterior Mean | $\bar{I} = \frac{1}{N_{\text{samples}}} \sum_n \text{imgs}_n$ | `sample_images` | `mean_image`: $(N_{\text{pix}}, N_{\text{pix}})$ |
| 5 | Posterior Std | $\sigma_I = \sqrt{\frac{1}{N_{\text{samples}}} \sum_n (\text{imgs}_n - \bar{I})^2}$ | `sample_images`, `mean_image` | `std_image`: $(N_{\text{pix}}, N_{\text{pix}})$ |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k < N_{\text{epoch}}$
- **parameters**: 
  - `n_epoch`: 10 (default)
- **check_frequency**: Every iteration (after parameter update)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Closure Phase $\chi^2$ | $\chi^2_{\text{cphase}} = \frac{1}{N_{\text{cphase}}} \sum_j \frac{1 - \cos((\psi_j^{\text{true}} - \psi_j(\bar{I})) \cdot \pi/180)}{(\sigma_