## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Deep Probabilistic Imaging via Normalizing Flows (DPI-NF)
- **algorithm_family**: Variational Inference with Normalizing Flows
- **summary**: A variational inference algorithm that learns a posterior distribution over geometric image model parameters using RealNVP normalizing flows, optimized via alpha-divergence minimization for radio interferometric imaging from closure quantities.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\phi} \mathcal{L}_{\alpha}(\phi) = \sum_{i=1}^{N_{\text{batch}}} w_i \cdot s \cdot \left[ \lambda(k) \cdot \mathcal{L}_{\text{data}}^{(i)} + \mathcal{L}_{\text{prior}}^{(i)} \right]$$

where $w_i$ are importance weights for alpha-divergence:
$$w_i = \text{softmax}\left( -\beta \cdot s \cdot \left[ \lambda(k) \cdot \mathcal{L}_{\text{data}}^{(i)} + \mathcal{L}_{\text{prior}}^{(i)} \right] \right)_i$$

For KL divergence ($\alpha = 1$): $w_i = \frac{1}{N_{\text{batch}}}$

### data_fidelity_term
**Closure Phase Loss (L2 on cosine distance)**:
$$\mathcal{L}_{\text{cphase}}^{(i)} = \frac{2}{N_{\text{cp}}} \sum_{j=1}^{N_{\text{cp}}} \frac{1 - \cos\left(\frac{\pi}{180}(\psi_j^{\text{true}} - \psi_j^{(i)})\right)}{\left(\frac{\pi}{180}\sigma_{\psi,j}\right)^2}$$

**Log Closure Amplitude Loss (L2)**:
$$\mathcal{L}_{\text{logcamp}}^{(i)} = \frac{1}{N_{\text{ca}}} \sum_{j=1}^{N_{\text{ca}}} \frac{\left(A_j^{\text{true}} - A_j^{(i)}\right)^2}{\sigma_{A,j}^2}$$

**Combined Data Loss**:
$$\mathcal{L}_{\text{data}}^{(i)} = \frac{1}{2s} \left( w_{\text{camp}} \cdot \mathcal{L}_{\text{logcamp}}^{(i)} + w_{\text{cphase}} \cdot \mathcal{L}_{\text{cphase}}^{(i)} \right)$$

### regularization_term
**Negative Log Prior (Standard Normal in Latent Space)**:
$$\mathcal{L}_{\text{prior}}^{(i)} = -\log \det\left(\frac{\partial G_\phi(z^{(i)})}{\partial z^{(i)}}\right) - \log \det\left(\frac{\partial \sigma(\theta^{(i)})}{\partial \theta^{(i)}}\right) + \frac{1}{2}\|z^{(i)}\|_2^2$$

where $G_\phi$ is the RealNVP inverse mapping and $\sigma$ is the sigmoid function.

### constraints
$$I(x, y) \geq 0 \quad \forall (x,y) \quad \text{(implicit via geometric model)}$$
$$\theta_j \in [0, 1] \quad \forall j \quad \text{(enforced via sigmoid)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\phi$ | varies | float32 | Xavier/Zero | RealNVP flow parameters (all learnable weights and biases) |
| $\phi_{\text{loc}}$ | $(1,)$ per ActNorm | float32 | Data-dependent: $-\bar{x}$ | ActNorm location parameter |
| $\phi_{\log s^{-1}}$ | $(1,)$ per ActNorm | float32 | Data-dependent: $\log(\text{std}(x)+10^{-6})$ | ActNorm log inverse scale |
| $\phi_{\text{scale}}$ | $(d_{\text{out}},)$ per ZeroFC | float32 | $\mathbf{0}$ | ZeroFC output scaling parameter |

### 2.2 Dual Variables

None (unconstrained optimization via gradient descent).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $z^{(i)}$ | $(N_{\text{batch}}, n_{\text{params}})$ | float32 | $\mathcal{N}(0, I)$ | Latent samples from base distribution |
| $\theta^{(i)}$ | $(N_{\text{batch}}, n_{\text{params}})$ | float32 | Computed | Raw parameter output from flow |
| $p^{(i)}$ | $(N_{\text{batch}}, n_{\text{params}})$ | float32 | Computed | Transformed parameters $\sigma(\theta^{(i)})$ |
| $I^{(i)}$ | $(N_{\text{batch}}, N_{\text{pix}}, N_{\text{pix}})$ | float32 | Computed | Generated images |
| $\log\det J^{(i)}$ | $(N_{\text{batch}},)$ | float32 | Computed | Cumulative log-determinant of Jacobian |
| $V^{(i)}$ | $(N_{\text{batch}}, 2, N_{\text{vis}})$ | float32 | Computed | Complex visibilities (real, imag) |
| $\psi^{(i)}$ | $(N_{\text{batch}}, N_{\text{cp}})$ | float32 | Computed | Predicted closure phases |
| $A^{(i)}$ | $(N_{\text{batch}}, N_{\text{ca}})$ | float32 | Computed | Predicted log closure amplitudes |
| $\lambda(k)$ | scalar | float32 | Computed | Epoch-dependent data weight |
| $w_i$ | $(N_{\text{batch}},)$ | float32 | Computed | Importance weights for alpha-divergence |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{\text{pix}}$ | 64 (default) | Image dimension in pixels |
| $\text{fov}$ | 120 μas (default) | Field of view |
| $N_{\text{flow}}$ | 16 (default) | Number of RealNVP flow blocks |
| $N_{\text{gaussian}}$ | 2 (default) | Number of nuisance Gaussians |
| $N_{\text{batch}}$ | 64 | Batch size |
| $N_{\text{epoch}}$ | 10 (default) | Number of training epochs |
| $\eta_{\text{lr}}$ | $10^{-4}$ (default) | Learning rate |
| $\gamma_{\text{clip}}$ | $10^{-4}$ (default) | Gradient clipping threshold |
| $\alpha$ | 1.0 (default) | Alpha-divergence parameter |
| $\beta$ | 0.0 (default) | Alternative divergence parameter |
| $k_{\text{start}}$ | 4 (default) | Start order for data weight schedule |
| $\tau_{\text{decay}}$ | 2000 (default) | Decay rate for data weight schedule |
| $w_{\text{logdet}}$ | 1.0 (default) | Log-determinant weight |
| $\epsilon$ | $10^{-16}$ | Numerical stability constant |
| $\epsilon_{\text{img}}$ | $10^{-4}$ | Image normalization constant |
| $r_{\text{range}}$ | $[10, 40]$ μas | Crescent radius range |
| $\sigma_{\text{range}}$ | $[1, 40]$ μas | Crescent width range |
| $s_{\text{range}}$ | $[10^{-3}, 0.99]$ | Asymmetry range |
| $\text{floor}_{\text{range}}$ | $[0, 1]$ | Floor fraction range |
| $\text{flux}_{\text{crescent}}$ | $[10^{-3}, 2]$ | Crescent flux range |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\psi^{\text{true}}$ | $(N_{\text{cp}},)$ | float32 | Observed closure phases (degrees) |
| $A^{\text{true}}$ | $(N_{\text{ca}},)$ | float32 | Observed log closure amplitudes |
| $\sigma_\psi$ | $(N_{\text{cp}},)$ | float32 | Closure phase uncertainties |
| $\sigma_A$ | $(N_{\text{ca}},)$ | float32 | Log closure amplitude uncertainties |
| $\mathbf{F}$ | $(N_{\text{pix}}^2, N_{\text{vis}}, 2)$ | float32 | DFT matrix (real, imag components) |
| $\mathcal{I}_{\text{cp}}$ | $3 \times (N_{\text{cp}},)$ | int64 | Closure phase baseline indices |
| $\mathcal{S}_{\text{cp}}$ | $3 \times (N_{\text{cp}},)$ | float32 | Closure phase baseline signs |
| $\mathcal{I}_{\text{ca}}$ | $4 \times (N_{\text{ca}},)$ | int64 | Closure amplitude baseline indices |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load UV Data | Extract $(u_j, v_j)$ coordinates and visibilities from UVFITS | `obspath` | $\mathbf{u} \in \mathbb{R}^{N_{\text{vis}} \times 2}$, $V^{\text{obs}} \in \mathbb{C}^{N_{\text{vis}}}$ |
| 2 | Compute Flux Prior | $F_0 = \text{median}(\|V_{AP \to AA}\|)$ | $V^{\text{obs}}$ | $F_0 \in \mathbb{R}$ |
| 3 | Build DFT Matrix | $\mathbf{F}_{kj} = \exp\left(-2\pi i (u_j x_k + v_j y_k)\right)$ where $(x_k, y_k)$ are pixel coordinates | $\mathbf{u}$, $N_{\text{pix}}$, psize | $\mathbf{F} \in \mathbb{R}^{N_{\text{pix}}^2 \times N_{\text{vis}} \times 2}$ |
| 4 | Build Pixel Grid | $x_k = -1 + (2k+1)/N_{\text{pix}}$ for $k=0,\ldots,N_{\text{pix}}-1$; $\mathbf{r} = \sqrt{x^2 + y^2}$; $\boldsymbol{\theta} = \arctan2(y, x)$ | $N_{\text{pix}}$ | Grid tensors: $\mathbf{x}, \mathbf{y}, \mathbf{r}, \boldsymbol{\theta} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ |
| 5 | Extract Closure Phases | Build triangle mapping from visibility indices with sign tracking | obs.cphase | $\mathcal{I}_{\text{cp}} = (I_1, I_2, I_3)$, $\mathcal{S}_{\text{cp}} = (s_1, s_2, s_3)$, $\psi^{\text{true}}$, $\sigma_\psi$ |
| 6 | Extract Log Closure Amplitudes | Build quadrangle mapping: $A = \log|V_{12}| + \log|V_{34}| - \log|V_{14}| - \log|V_{23}|$ | obs.logcamp | $\mathcal{I}_{\text{ca}} = (I_{12}, I_{34}, I_{14}, I_{23})$, $A^{\text{true}}$, $\sigma_A$ |
| 7 | Initialize Permutation Orders | For each flow $i$: $\pi_i = \text{RandomPermutation}(n_{\text{params}}, \text{seed}=i)$; $\pi_i^{-1} = \text{InversePermutation}(\pi_i)$ | $N_{\text{flow}}$, $n_{\text{params}}$ | $\{\pi_i\}_{i=1}^{N_{\text{flow}}}$, $\{\pi_i^{-1}\}_{i=1}^{N_{\text{flow}}}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize ActNorm (deferred) | $\phi_{\text{loc}} \leftarrow \mathbf{0}$; $\phi_{\log s^{-1}} \leftarrow \mathbf{0}$; initialized_flag $\leftarrow 0$ | None | ActNorm parameters | Data-dependent init on first forward pass |
| 2 | Initialize Linear Layers | $W \sim \mathcal{N}(0, 0.05^2)$; $b \leftarrow \mathbf{0}$ | Layer dimensions | FC weights and biases | For non-ZeroFC layers |
| 3 | Initialize ZeroFC | $W \leftarrow \mathbf{0}$; $b \leftarrow \mathbf{0}$; $\phi_{\text{scale}} \leftarrow \mathbf{0}$ | Layer dimensions | ZeroFC parameters | Ensures identity-like initialization |
| 4 | Compute Scale Factor | $s = 1 / N_{\text{ca}}$ | $N_{\text{ca}}$ | $s \in \mathbb{R}$ | For cphase_logcamp mode |
| 5 | Compute Log-det Scale | $s_{\log\det} = 2 \cdot w_{\log\det} / N_{\text{ca}}$ | $w_{\log\det}$, $N_{\text{ca}}$ | $s_{\log\det} \in \mathbb{R}$ | |
| 6 | Compute Closure Phase Weight | $w_{\text{cphase}} = N_{\text{cp}} / N_{\text{ca}}$ | $N_{\text{cp}}$, $N_{\text{ca}}$ | $w_{\text{cphase}} \in \mathbb{R}$ | Balances loss terms |
| 7 | Set Camp Weight | $w_{\text{camp}} \leftarrow 1.0$ | None | $w_{\text{camp}} \in \mathbb{R}$ | |
| 8 | Initialize Adam Optimizer | $m \leftarrow \mathbf{0}$; $v \leftarrow \mathbf{0}$; $t \leftarrow 0$ | $\eta_{\text{lr}}$ | Optimizer state | |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0$ to $N_{\text{epoch}} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 9 | Compute Data Weight | $\lambda(k) = \min\left(10^{-k_{\text{start}} + k/\tau_{\text{decay}}}, 1\right)$ | $k$, $k_{\text{start}}$, $\tau_{\text{decay}}$ | $\lambda(k) \in \mathbb{R}$ | Annealing schedule |
| 10 | Sample Latent Variables | $z^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{n_{\text{params}}})$ for $i = 1, \ldots, N_{\text{batch}}$ | $N_{\text{batch}}$, $n_{\text{params}}$ | $\mathbf{z} \in \mathbb{R}^{N_{\text{batch}} \times n_{\text{params}}}$ | torch.randn |

**BEGIN_SUBLOOP** (RealNVP Reverse Pass: for $\ell = N_{\text{flow}} - 1$ down to $0$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 11 | Apply Inverse Permutation | $\mathbf{h}^{(\ell)} \leftarrow \mathbf{h}^{(\ell+1)}[:, \pi_\ell^{-1}]$ | $\mathbf{h}^{(\ell+1)}$, $\pi_\ell^{-1}$ | $\mathbf{h}^{(\ell)}$ | Index reordering |
| 12 | Reverse Second Coupling | Split: $\mathbf{h}_a, \mathbf{h}_b = \text{chunk}(\mathbf{h}^{(\ell)}, 2)$; $(\log s_0, t) = \text{NN}_2(\mathbf{h}_a)$; $\log s = \tanh(\log s_0)$; $\mathbf{h}_b' = \mathbf{h}_b \cdot e^{-\log s} - t$; $\Delta_{\log\det} = -\sum_j \log s_j$ | $\mathbf{h}^{(\ell)}$ | Updated $\mathbf{h}^{(\ell)}$, $\Delta_{\log\det}$ | Affine inverse |
| 13 | Reverse Second ActNorm | $\mathbf{h}^{(\ell)} \leftarrow \mathbf{h}^{(\ell)} \cdot e^{\phi_{\log s^{-1}}} - \phi_{\text{loc}}$; $\Delta_{\log\det} = -n_{\text{params}} \cdot (-\phi_{\log s^{-1}})$ | $\mathbf{h}^{(\ell)}$ | Updated $\mathbf{h}^{(\ell)}$, $\Delta_{\log\det}$ | |
| 14 | Apply Internal Reverse | $\mathbf{h}^{(\ell)} \leftarrow \mathbf{h}^{(\ell)}[:, \text{arange}(n-1, -1, -1)]$ | $\mathbf{h}^{(\ell)}$ | Reversed $\mathbf{h}^{(\ell)}$ | Dimension flip |
| 15 | Reverse First Coupling | Same as Step 12 with $\text{NN}_1$ | $\mathbf{h}^{(\ell)}$ | Updated $\mathbf{h}^{(\ell)}$, $\Delta_{\log\det}$ | |
| 16 | Reverse First ActNorm | Same as Step 13 | $\mathbf{h}^{(\ell)}$ | Updated $\mathbf{h}^{(\ell)}$, $\Delta_{\log\det}$ | |
| 17 | Apply Internal Reverse | $\mathbf{h}^{(\ell)} \leftarrow \mathbf{h}^{(\ell)}[:, \text{arange}(n-1, -1, -1)]$ | $\mathbf{h}^{(\ell)}$ | Reversed $\mathbf{h}^{(\ell)}$ | |
| 18 | Accumulate Log-det | $\log\det J \leftarrow \log\det J + \Delta_{\log\det}$ | Running sum | $\log\det J \in \mathbb{R}^{N_{\text{batch}}}$ | |

**END_SUBLOOP**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 19 | Store Raw Parameters | $\boldsymbol{\theta} \leftarrow \mathbf{h}^{(0)}$ | Final $\mathbf{h}$ | $\boldsymbol{\theta} \in \mathbb{R}^{N_{\text{batch}} \times n_{\text{params}}}$ | |
| 20 | Apply Sigmoid | $p_j^{(i)} = \sigma(\theta_j^{(i)}) = \frac{1}{1 + e^{-\theta_j^{(i)}}}$ | $\boldsymbol{\theta}$ | $\mathbf{p} \in [0,1]^{N_{\text{batch}} \times n_{\text{params}}}$ | Maps to valid range |
| 21 | Sigmoid Log-det | $\Delta_{\text{sigmoid}} = \sum_j \left( -\theta_j - 2 \log(1 + e^{-\theta_j}) \right)$ | $\boldsymbol{\theta}$ | $\Delta_{\text{sigmoid}} \in \mathbb{R}^{N_{\text{batch}}}$ | Jacobian of sigmoid |
| 22 | Total Log-det | $\log\det J_{\text{total}} \leftarrow \log\det J + \Delta_{\text{sigmoid}}$ | $\log\det J$, $\Delta_{\text{sigmoid}}$ | $\log\det J_{\text{total}} \in \mathbb{R}^{N_{\text{batch}}}$ | |

**Image Generation (Geometric Model)**:

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 23 | Rescale Radius | $r^{(i)} = \frac{r_{\min}}{0.5 \cdot \text{fov}} + p_0^{(i)} \cdot \frac{r_{\max} - r_{\min}}{0.5 \cdot \text{fov}}$ | $\mathbf{p}$, ranges | $r^{(i)} \in \mathbb{R}^{N_{\text{batch}}}$ | Normalized units |
| 24 | Rescale Width | $\sigma_r^{(i)} = \frac{\sigma_{\min}}{0.5 \cdot \text{fov}} + p_1^{(i)} \cdot \frac{\sigma_{\max} - \sigma_{\min}}{0.5 \cdot \text{fov}}$ | $\mathbf{p}$, ranges | $\sigma_r^{(i)}$ | |
| 25 | Rescale Asymmetry | $s^{(i)} = s_{\min} + p_2^{(i)} \cdot (s_{\max} - s_{\min})$ | $\mathbf{p}$, ranges | $s^{(i)} \in (0,1)$ | |
| 26 | Rescale Orientation | $\eta^{(i)} = \frac{181\pi}{180}(2p_3^{(i)} - 1)$ | $\mathbf{p}$ | $\eta^{(i)} \in [-\pi, \pi]$ | |
| 27 | Compute Ring | $\text{Ring}^{(i)}_{xy} = \exp\left( -\frac{(\mathbf{r}_{xy} - r^{(i)})^2}{2(\sigma_r^{(i)})^2} \right)$ | $r^{(i)}$, $\sigma_r^{(i)}$, $\mathbf{r}$ | Ring image | Gaussian annulus |
| 28 | Compute Asymmetry Modulation | $S^{(i)}_{xy} = 1 + s^{(i)} \cos(\boldsymbol{\theta}_{xy} - \eta^{(i)})$ | $s^{(i)}$, $\eta^{(i)}$, $\boldsymbol{\theta}$ | Modulation field | |
| 29 | Compute Crescent | $\text{Crescent}^{(i)}_{xy} = S^{(i)}_{xy} \cdot \text{Ring}^{(i)}_{xy}$ | Ring, $S$ | Crescent image | |
| 30 | Compute Disk | $\text{Disk}^{(i)}_{xy} = \frac{1}{2}\left(1 + \text{erf}\left(\frac{r^{(i)} - \mathbf{r}_{xy}}{\sqrt{2}\sigma_r^{(i)}}\right)\right)$ | $r^{(i)}$, $\sigma_r^{(i)}$, $\mathbf{r}$ | Filled disk | Error function |
| 31 | Normalize Crescent | $\text{Crescent}^{(i)} \leftarrow \frac{\text{Crescent}^{(i)}}{\sum_{xy}\text{Crescent}^{(i)}_{xy} + \epsilon_{\text{img}}}$ | Crescent | Unit-flux crescent | |
| 32 | Normalize Disk | $\text{Disk}^{(i)} \leftarrow \frac{\text{Disk}^{(i)}}{\sum_{xy}\text{Disk}^{(i)}_{xy} + \epsilon_{\text{img}}}$ | Disk | Unit-flux disk | |
| 33 | Combine with Floor | $I_{\text{cres}}^{(i)} = f_{\text{cres}}^{(i)} \left[ (1 - f_{\text{floor}}^{(i)}) \text{Crescent}^{(i)} + f_{\text{floor}}^{(i)} \text{Disk}^{(i)} \right]$ | Components, floor, flux params | Combined crescent | |

**BEGIN_SUBLOOP** (Add Gaussians: for $g = 0$ to $N_{\text{gaussian}} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 34 | Extract Gaussian Params | $x_c^{(g)}, y_c^{(g)}, a^{(g)}, \sigma_x^{(g)}, \sigma_y^{(g)}, \phi^{(g)}$ from rescaled $\mathbf{p}$ | $\mathbf{p}$ | Gaussian parameters | 6 params per Gaussian |
| 35 | Compute Rotated Coords | $x' = (\mathbf{x} - x_c^{(g)})\cos\phi^{(g)} + (\mathbf{y} - y_c^{(g)})\sin\phi^{(g)}$; $y' = -(\mathbf{x} - x_c^{(g)})\sin\phi^{(g)} + (\mathbf{y} - y_c^{(g)})\cos\phi^{(g)}$ | Grid, Gaussian params | Rotated coordinates | |
| 36 | Compute Gaussian | $G^{(g)}_{xy} = \frac{1}{2\pi\sigma_x^{(g)}\sigma_y^{(g)}} \exp\left(-\frac{1}{2}\left(\frac{x'^2}{(\sigma_x^{(g)})^2} + \frac{y'^2}{(\sigma_y^{(g)})^2}\right)\right)$ | Rotated coords, widths | Gaussian image | |
| 37 | Normalize Gaussian | $G^{(g)} \leftarrow \frac{G^{(g)}}{\sum_{xy} G^{(g)}_{xy} + \epsilon_{\text{img}}}$ | $G^{(g)}$ | Unit-flux Gaussian | |
| 38 | Add to Image | $I^{(i)} \leftarrow I_{\text{cres}}^{(i)} + a^{(g)} G^{(g)}$ | Current image, Gaussian | Updated image | |

**END_SUBLOOP**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 39 | Final Image Normalization | $I^{(i)} \leftarrow \frac{I^{(i)}}{\sum_{xy} I^{(i)}_{xy} + \epsilon_{\text{img}}}$ | $I^{(i)}$ | Unit-flux image $I^{(i)} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ | For flux_flag=False |

**Forward Operator**:

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 40 | Flatten Image | $\mathbf{x}^{(i)} = \text{vec}(I^{(i)}) \in \mathbb{R}^{N_{\text{pix}}^2}$ | $I^{(i)}$ | Flattened image | |
| 41 | Compute Visibilities | $V_j^{(i)} = \sum_k x_k^{(i)} F_{kj}$; Explicitly: $V_{\text{re},j}^{(i)} = \sum_k x_k^{(i)} F_{kj,0}$; $V_{\text{im},j}^{(i)} = \sum_k x_k^{(i)} F_{kj,1}$ | $\mathbf{x}^{(i)}$, $\mathbf{F}$ | $\mathbf{V