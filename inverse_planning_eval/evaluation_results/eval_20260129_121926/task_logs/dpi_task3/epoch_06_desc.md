## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Deep Probabilistic Imaging via RealNVP",
  "algorithm_family": "Variational Inference with Normalizing Flows",
  "summary": "Learns a posterior distribution over geometric crescent model parameters for radio interferometric imaging by training a RealNVP normalizing flow to minimize an alpha-divergence between the approximate and true posterior."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\phi} \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ w(z;\phi) \cdot \mathcal{L}(z;\phi) \right]$$

where $\mathcal{L}(z;\phi) = \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}}(G_\phi(z)) - \log |\det J_{G_\phi}(z)| - \log p_\sigma(G_\phi(z)) + \frac{1}{2}\|z\|_2^2$

**data_fidelity_term**:
For `cphase_logcamp` mode:
$$\mathcal{L}_{\text{data}} = \frac{1}{2s} \left[ w_{\text{cp}} \sum_{i=1}^{N_{\text{cp}}} \frac{2(1 - \cos(\Psi_i^{\text{true}} - \Psi_i^{\text{pred}}))}{(\sigma_{\text{cp},i} \cdot \pi/180)^2} + w_{\text{ca}} \sum_{j=1}^{N_{\text{ca}}} \frac{(\log A_j^{\text{true}} - \log A_j^{\text{pred}})^2}{\sigma_{\text{ca},j}^2} \right]$$

For `vis` mode:
$$\mathcal{L}_{\text{data}} = \frac{w_{\text{vis}}}{2s} \sum_{k=1}^{N_{\text{vis}}} \frac{|\mathcal{V}_k^{\text{true}} - \mathcal{V}_k^{\text{pred}}|^2}{\sigma_{\text{vis},k}^2}$$

**regularization_term**:
$$\mathcal{R}(\phi, z) = -\log |\det J_{G_\phi}(z)| + \frac{1}{2}\|z\|_2^2$$

This is the negative log-prior under the normalizing flow transformation.

**constraints**:
$$\theta = \sigma(\theta_{\text{raw}}) \in (0,1)^d$$ (implicit box constraint via sigmoid)
$$I_{i,j} \geq 0$$ (images are non-negative by construction)

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\phi$ | varies | float32 | Xavier/zero init | RealNVP neural network parameters |
| $z$ | $(N_{\text{batch}}, d)$ | float32 | $\mathcal{N}(0,I)$ each iteration | Latent Gaussian samples |

### 2.2 Dual Variables
None (not a constrained optimization with explicit duals)

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_{\text{raw}}$ | $(N_{\text{batch}}, d)$ | float32 | Computed | Raw parameters from flow inverse |
| $\theta$ | $(N_{\text{batch}}, d)$ | float32 | Computed | Sigmoid-transformed parameters in $(0,1)^d$ |
| $I$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Generated images |
| $\mathcal{V}$ | $(N_{\text{batch}}, 2, N_{\text{vis}})$ | float32 | Computed | Complex visibilities (real, imag) |
| $|\mathcal{V}|$ | $(N_{\text{batch}}, N_{\text{vis}})$ | float32 | Computed | Visibility amplitudes |
| $\Psi$ | $(N_{\text{batch}}, N_{\text{cp}})$ | float32 | Computed | Closure phases in degrees |
| $\log A$ | $(N_{\text{batch}}, N_{\text{ca}})$ | float32 | Computed | Log closure amplitudes |
| $\log|\det J|$ | $(N_{\text{batch}},)$ | float32 | Computed | Log-determinant of Jacobian |
| $w$ | $(N_{\text{batch}},)$ | float32 | Computed | Importance weights for alpha-divergence |

### 2.4 Constants
| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $d$ | scalar | int | $4 + 6 \cdot N_{\text{gauss}} + 2$ | Number of geometric parameters |
| $N_{\text{flow}}$ | scalar | int | 16 (default) | Number of flow blocks |
| $N_{\text{gauss}}$ | scalar | int | 2 (default) | Number of nuisance Gaussians |
| $N_{\text{pix}}$ | scalar | int | 64 (default) | Image dimension |
| $\text{fov}$ | scalar | float | 120.0 | Field of view in microarcseconds |
| $\epsilon$ | scalar | float | $10^{-16}$ | Numerical stability constant |
| $\epsilon_{\text{img}}$ | scalar | float | $10^{-4}$ | Image normalization constant |
| $r_{\text{range}}$ | $(2,)$ | float | $[10.0, 40.0]$ | Ring radius range in $\mu$as |
| $\sigma_{\text{range}}$ | $(2,)$ | float | $[1.0, 40.0]$ | Ring width range |
| $s_{\text{range}}$ | $(2,)$ | float | $[10^{-3}, 0.99]$ | Asymmetry range |
| $\text{floor}_{\text{range}}$ | $(2,)$ | float | $[0.0, 1.0]$ | Floor fraction range |
| $\alpha$ | scalar | float | 1.0 (default) | Alpha-divergence parameter |
| $\beta$ | scalar | float | 0.0 (default) | Beta parameter |
| $\lambda_{\text{start}}$ | scalar | float | $10^{-4}$ | Initial data weight |
| $\tau$ | scalar | float | 2000 | Decay rate for data weight |
| $\eta$ | scalar | float | $10^{-4}$ | Learning rate |
| $c$ | scalar | float | $10^{-4}$ | Gradient clipping threshold |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $F$ | $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ | float32 | DFT matrix (real, imag components) |
| $\mathcal{V}^{\text{true}}$ | $(2, N_{\text{vis}})$ | float32 | Observed complex visibilities |
| $|\mathcal{V}|^{\text{true}}$ | $(N_{\text{vis}},)$ | float32 | Observed visibility amplitudes |
| $\Psi^{\text{true}}$ | $(N_{\text{cp}},)$ | float32 | Observed closure phases (degrees) |
| $\log A^{\text{true}}$ | $(N_{\text{ca}},)$ | float32 | Observed log closure amplitudes |
| $\sigma_{\text{vis}}$ | $(N_{\text{vis}},)$ | float32 | Visibility uncertainties |
| $\sigma_{\text{cp}}$ | $(N_{\text{cp}},)$ | float32 | Closure phase uncertainties |
| $\sigma_{\text{ca}}$ | $(N_{\text{ca}},)$ | float32 | Log closure amplitude uncertainties |
| $\mathcal{I}_{\text{cp}}^{(1,2,3)}$ | $(N_{\text{cp}},)$ | int64 | Closure phase visibility indices |
| $\mathcal{S}_{\text{cp}}^{(1,2,3)}$ | $(N_{\text{cp}},)$ | float32 | Closure phase signs ($\pm 1$) |
| $\mathcal{I}_{\text{ca}}^{(1,2,3,4)}$ | $(N_{\text{ca}},)$ | int64 | Closure amplitude visibility indices |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Load UVFITS | $\text{obs} = \text{load\_uvfits}(\text{path})$ | File path | obs object | Valid UVFITS format |
| P2 | Extract flux constant | $f_0 = \text{median}(|\mathcal{V}_{\text{AP-AA}}|)$ | obs | $f_0 \in \mathbb{R}$ | AP-AA baseline exists |
| P3 | Compute pixel size | $\Delta = \frac{\text{fov} \cdot \text{RADPERUAS}}{N_{\text{pix}}}$ | fov, $N_{\text{pix}}$ | $\Delta$ (radians/pixel) | — |
| P4 | Extract UV coordinates | $U = [u_1, \ldots, u_{N_{\text{vis}}}]^T$, $V = [v_1, \ldots, v_{N_{\text{vis}}}]^T$ | obs | $U, V \in \mathbb{R}^{N_{\text{vis}}}$ | — |
| P5 | Compute DFT matrix | $F_{p,k} = e^{-2\pi i (u_k x_p + v_k y_p)}$ where $(x_p, y_p)$ are pixel coordinates | $U$, $V$, $\Delta$, $N_{\text{pix}}$ | $F \in \mathbb{C}^{N_{\text{pix}}^2 \times N_{\text{vis}}}$ | — |
| P6 | Split DFT to real/imag | $F_{\text{torch}} = [\Re(F^T), \Im(F^T)]$ | $F$ | $F_{\text{torch}} \in \mathbb{R}^{N_{\text{pix}}^2 \times N_{\text{vis}} \times 2}$ | — |
| P7 | Build closure phase map | For each closure triangle $(t_1, t_2, t_3)$: find visibility indices and signs | obs.cphase | $\mathcal{I}_{\text{cp}}^{(1,2,3)}$, $\mathcal{S}_{\text{cp}}^{(1,2,3)}$ | Triangles exist in data |
| P8 | Build closure amplitude map | For each quadrangle $(t_1, t_2, t_3, t_4)$: find visibility indices | obs.camp | $\mathcal{I}_{\text{ca}}^{(1,2,3,4)}$ | Quadrangles exist |
| P9 | Extract ground truth | $\mathcal{V}^{\text{true}} = \text{obs.vis}$, $\Psi^{\text{true}} = \text{obs.cphase}$, $\log A^{\text{true}} = \text{obs.logcamp}$ | obs | Ground truth tensors | — |
| P10 | Extract uncertainties | $\sigma_{\text{vis}} = \text{obs.sigma}$, etc. | obs | Uncertainty tensors | — |
| P11 | Compute scale factor | $s = 1 / N_{\text{ca}}$ (for cphase_logcamp mode) | $N_{\text{ca}}$ | $s \in \mathbb{R}$ | — |
| P12 | Compute weights | $w_{\text{ca}} = 1$, $w_{\text{cp}} = N_{\text{cp}} / N_{\text{ca}}$ | $N_{\text{cp}}$, $N_{\text{ca}}$ | $w_{\text{ca}}, w_{\text{cp}}$ | cphase_logcamp mode |
| P13 | Build image grid | $x_j = -1 + (2j+1)/N_{\text{pix}}$ for $j=0,\ldots,N_{\text{pix}}-1$; $(G_x, G_y) = \text{meshgrid}(-x, x)$ | $N_{\text{pix}}$ | $G_x, G_y \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ | — |
| P14 | Compute polar grid | $G_r = \sqrt{G_x^2 + G_y^2}$, $G_\theta = \text{atan2}(G_y, G_x)$ | $G_x$, $G_y$ | $G_r, G_\theta \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ | — |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | ActNorm.loc | $\mathbf{0}$ | $(1,)$ | float32 | Zero initialization |
| 2 | ActNorm.log_scale_inv | $\mathbf{0}$ | $(1,)$ | float32 | Zero initialization |
| 3 | ZeroFC.weight | $\mathbf{0}$ | $(d_{\text{out}}, d_{\text{in}})$ | float32 | Zero initialization |
| 4 | ZeroFC.bias | $\mathbf{0}$ | $(d_{\text{out}},)$ | float32 | Zero initialization |
| 5 | ZeroFC.scale | $\mathbf{0}$ | $(d_{\text{out}},)$ | float32 | Zero initialization |
| 6 | Linear.weight (coupling) | $W_{ij} \sim \mathcal{N}(0, 0.05^2)$ | varies | float32 | Normal initialization |
| 7 | Linear.bias (coupling) | $\mathbf{0}$ | varies | float32 | Zero initialization |
| 8 | permutation_orders | $\pi_i = \text{RandomState}(i).\text{permutation}(d)$ for $i=0,\ldots,N_{\text{flow}}-1$ | $(N_{\text{flow}}, d)$ | int64 | Random permutation |
| 9 | inverse_orders | $\pi_i^{-1}$ such that $\pi_i^{-1}[\pi_i[j]] = j$ | $(N_{\text{flow}}, d)$ | int64 | Computed from step 8 |
| 10 | optimizer_state | Adam state with $\eta = 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ | — | — | Default Adam |
| 11 | epoch_counter | $k = 0$ | scalar | int | Zero |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{epoch}} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 12 | Compute data weight | $\lambda_{\text{data}} = \min(10^{-\lambda_{\text{start}} + k/\tau}, 1.0)$ | $k$, $\lambda_{\text{start}}=4$, $\tau=2000$ | $\lambda_{\text{data}} \in \mathbb{R}$ | Annealing schedule |
| 13 | Sample latent | $z \sim \mathcal{N}(0, I_d)$ | $N_{\text{batch}}$, $d$ | $z \in \mathbb{R}^{N_{\text{batch}} \times d}$ | torch.randn |

**BEGIN_LOOP** (RealNVP Reverse Pass: for $i = N_{\text{flow}}-1, \ldots, 0$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 14 | Apply inverse permutation | $h \leftarrow h[:, \pi_i^{-1}]$ | $h$, $\pi_i^{-1}$ | $h \in \mathbb{R}^{N_{\text{batch}} \times d}$ | Index reordering |
| 15 | Reverse permutation (flow) | $h \leftarrow h[:, d-1::-1]$ | $h$ | $h \in \mathbb{R}^{N_{\text{batch}} \times d}$ | Reverse indices |
| 16 | Split for coupling2 | $h_a = h[:, :d/2]$, $h_b = h[:, d/2:]$ | $h$ | $h_a, h_b$ | Chunk operation |
| 17 | Coupling2 network forward | $[\log s_0, t] = \text{Net}_2(h_a)$ | $h_a \in \mathbb{R}^{N_{\text{batch}} \times (d - d/2)}$ | $\log s_0, t \in \mathbb{R}^{N_{\text{batch}} \times d/2}$ | MLP with BatchNorm |
| 18 | Compute scale | $s = \exp(\tanh(\log s_0))$ | $\log s_0$ | $s \in \mathbb{R}^{N_{\text{batch}} \times d/2}$ | Stabilized exponential |
| 19 | Inverse affine coupling2 | $h_b \leftarrow h_b / s - t$ | $h_b$, $s$, $t$ | $h_b \in \mathbb{R}^{N_{\text{batch}} \times d/2}$ | — |
| 20 | Coupling2 log-det | $\delta_1 = -\sum_j \tanh(\log s_{0,j})$ | $\log s_0$ | $\delta_1 \in \mathbb{R}^{N_{\text{batch}}}$ | Per-sample sum |
| 21 | Concatenate coupling2 | $h \leftarrow [h_a, h_b]$ | $h_a$, $h_b$ | $h \in \mathbb{R}^{N_{\text{batch}} \times d}$ | — |
| 22 | ActNorm2 reverse | $h \leftarrow h \cdot e^{\gamma_2} - \mu_2$; $\delta_2 = d \cdot \gamma_2$ | $h$, $\mu_2$, $\gamma_2$ | $h$, $\delta_2$ | $\gamma_2$ = log_scale_inv |
| 23 | Reverse permutation (flow) | $h \leftarrow h[:, d-1::-1]$ | $h$ | $h \in \mathbb{R}^{N_{\text{batch}} \times d}$ | — |
| 24-29 | Coupling1 reverse | Same as steps 16-21 for coupling1 | — | $h$, $\delta_3$ | — |
| 30 | ActNorm1 reverse | $h \leftarrow h \cdot e^{\gamma_1} - \mu_1$; $\delta_4 = d \cdot \gamma_1$ | $h$, $\mu_1$, $\gamma_1$ | $h$, $\delta_4$ | — |
| 31 | Accumulate flow log-det | $\log|\det J|_{\text{flow}} \mathrel{+}= \delta_1 + \delta_2 + \delta_3 + \delta_4$ | $\delta_1, \delta_2, \delta_3, \delta_4$ | $\log|\det J|_{\text{flow}}$ | — |

**END_LOOP** (RealNVP blocks)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 32 | Store raw params | $\theta_{\text{raw}} = h$ | $h$ | $\theta_{\text{raw}} \in \mathbb{R}^{N_{\text{batch}} \times d}$ | Final flow output |
| 33 | Apply sigmoid | $\theta = \sigma(\theta_{\text{raw}}) = \frac{1}{1 + e^{-\theta_{\text{raw}}}}$ | $\theta_{\text{raw}}$ | $\theta \in (0,1)^{N_{\text{batch}} \times d}$ | Element-wise |
| 34 | Sigmoid log-det | $\log|\det J|_{\sigma} = \sum_j \left( -\theta_{\text{raw},j} - 2\log(1 + e^{-\theta_{\text{raw},j}}) \right)$ | $\theta_{\text{raw}}$ | $\log|\det J|_{\sigma} \in \mathbb{R}^{N_{\text{batch}}}$ | Softplus formulation |
| 35 | Total log-det | $\log|\det J| = \log|\det J|_{\text{flow}} + \log|\det J|_{\sigma}$ | $\log|\det J|_{\text{flow}}$, $\log|\det J|_{\sigma}$ | $\log|\det J| \in \mathbb{R}^{N_{\text{batch}}}$ | — |

**Image Generation (Geometric Model)**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 36 | Map radius | $r = \frac{r_{\min}}{0.5 \cdot \text{fov}} + \theta_0 \cdot \frac{r_{\max} - r_{\min}}{0.5 \cdot \text{fov}}$ | $\theta_0$, $r_{\text{range}}$, fov | $r \in \mathbb{R}^{N_{\text{batch}} \times 1 \times 1}$ | Broadcast to image shape |
| 37 | Map width | $\sigma_r = \frac{w_{\min}}{0.5 \cdot \text{fov}} + \theta_1 \cdot \frac{w_{\max} - w_{\min}}{0.5 \cdot \text{fov}}$ | $\theta_1$, width_range | $\sigma_r \in \mathbb{R}^{N_{\text{batch}} \times 1 \times 1}$ | — |
| 38 | Map asymmetry | $s = s_{\min} + \theta_2 \cdot (s_{\max} - s_{\min})$ | $\theta_2$, asym_range | $s \in \mathbb{R}^{N_{\text{batch}} \times 1 \times 1}$ | — |
| 39 | Map orientation | $\eta = \frac{181\pi}{180} (2\theta_3 - 1)$ | $\theta_3$ | $\eta \in \mathbb{R}^{N_{\text{batch}} \times 1 \times 1}$ | Range $(-\pi, \pi)$ |
| 40 | Compute ring | $R = \exp\left( -\frac{(G_r - r)^2}{2\sigma_r^2} \right)$ | $G_r$, $r$, $\sigma_r$ | $R \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}}}$ | Gaussian ring |
| 41 | Compute modulation | $S = 1 + s \cos(G_\theta - \eta)$ | $G_\theta$, $s$, $\eta$ | $S \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}}}$ | Brightness asymmetry |
| 42 | Compute crescent | $C = S \odot R$ | $S$, $R$ | $C \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}}}$ | Element-wise product |
| 43 | Compute disk | $D = \frac{1}{2}\left(1 + \text{erf}\left(\frac{r - G_r}{\sqrt{2}\sigma_r}\right)\right)$ | $G_r$, $r$, $\sigma_r$ | $D \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}}}$ | Filled disk |
| 44 | Normalize crescent | $\tilde{C} = C / (\sum_{i,j} C_{i,j} + \epsilon_{\text{img}})$ | $C$ | $\tilde{C}$ | Sum over spatial dims |
| 45 | Normalize disk | $\tilde{D} = D / (\sum_{i,j} D_{i,j} + \epsilon_{\text{img}})$ | $D$ | $\tilde{D}$ | — |
| 46 | Map floor & flux | $f = f_{\min} + \theta_{4+6N_g} (f_{\max} - f_{\min})$; $\phi_c = \phi_{c,\min} + \theta_{5+6N_g}(\phi_{c,\max} - \phi_{c,\min})$ | $\theta$ | $f$, $\phi_c$ | Floor and crescent flux |
| 47 | Combine crescent+disk | $I_c = \phi_c \left( (1-f)\tilde{C} + f\tilde{D} \right)$ | $\tilde{C}$, $\tilde{D}$, $f$, $\phi_c$ | $I_c \in \mathbb{R}^{N_{\text{batch}} \times N_{\text{pix}} \times N_{\text{pix}}}$ | — |

**BEGIN_LOOP** (Nuisance Gaussians: for $g = 0, \ldots, N_{\text{gauss}}-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 48 | Map Gaussian params | $x_g, y_g, \alpha_g, \sigma_{x,g}, \sigma_{y,g}, \phi_g$ from $\theta_{4+6g:10+6g}$ | $\theta$ | Gaussian parameters | Linear mappings |
| 49 | Rotate coordinates | $x' = (G_x - x_g)\cos\phi_g + (G_y - y_g)\sin\phi_g$; $y' = -(G_x - x_g)\sin\phi_g + (G_y - y_g)\cos\phi_g$ | $G_x$, $G_y$, $x_g$, $y_g$, $\phi_g$ | $x'$, $y'$ | — |
| 50 | Compute Gaussian | $\mathcal{G}_g = \frac{1}{2\pi\sigma_{x,g}\sigma_{y,g}} \exp\left(-\frac{1}{2}\left(\frac{x'^2}{\sigma_{x,g}^2} + \frac{y'^2}{\sigma_{y,g}^2}\right)\right)$ | $x'$, $y'$, $\sigma_{x,g}$, $\sigma_{y,g}$ | $\mathcal{G}_g$ | 2D elliptical Gaussian |
| 51 | Normalize Gaussian | $\tilde{\mathcal{G}}_g = \mathcal{G}_g / (\sum_{i,j}\mathcal{G}_{g,i,j} + \epsilon_{\text{img}})$ | $\mathcal{G}_g$ | $\tilde{\mathcal{G}}_g$ | — |
| 52 | Add to image | $I_c \leftarrow I