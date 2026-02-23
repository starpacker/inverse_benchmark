# Algorithmic Specification Document: Deep Probabilistic Imaging for Interferometry

---

## 1. Mathematical Formulation

### 1.1 Problem Statement

Approximate the posterior distribution $p(\boldsymbol{\theta} \mid \mathbf{d})$ over geometric image parameters $\boldsymbol{\theta}$ given interferometric observations $\mathbf{d}$, using variational inference with a normalizing flow.

### 1.2 Complete Objective Function

**Variational Objective (ELBO-style):**

$$
\mathcal{L}(\phi) = \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ w(k) \cdot \mathcal{L}_{\text{data}}(g_\phi(z)) - \log q_\phi(g_\phi(z)) \right]
$$

where $g_\phi: \mathbb{R}^{n_\theta} \to \mathbb{R}^{n_\theta}$ is the normalizing flow with parameters $\phi$, and $w(k)$ is an epoch-dependent annealing weight.

### 1.3 Decomposition of Objective

#### 1.3.1 Data Fidelity Term

For **closure phase + log closure amplitude** mode:

$$
\mathcal{L}_{\text{data}}(\boldsymbol{\theta}) = \frac{1}{2s} \left[ w_{ca} \cdot \mathcal{L}_{lca}(\boldsymbol{\theta}) + w_{cp} \cdot \mathcal{L}_{cp}(\boldsymbol{\theta}) \right]
$$

where $s = 1/N_{ca}$ is the scale factor.

**Closure Phase Loss:**
$$
\mathcal{L}_{cp}(\boldsymbol{\theta}) = \frac{2}{N_{cp}} \sum_{i=1}^{N_{cp}} \frac{1 - \cos\left(\frac{\pi}{180}(\phi_i^{\text{obs}} - \phi_i^{\text{pred}}(\boldsymbol{\theta}))\right)}{\left(\frac{\pi}{180}\sigma_{cp,i}\right)^2}
$$

**Log Closure Amplitude Loss:**
$$
\mathcal{L}_{lca}(\boldsymbol{\theta}) = \frac{1}{N_{ca}} \sum_{i=1}^{N_{ca}} \frac{\left(\log A_i^{\text{obs}} - \log A_i^{\text{pred}}(\boldsymbol{\theta})\right)^2}{\sigma_{lca,i}^2}
$$

For **complex visibility** mode:
$$
\mathcal{L}_{\text{data}}(\boldsymbol{\theta}) = \frac{1}{2N_{vis}} \sum_{j=1}^{N_{vis}} \frac{|V_j^{\text{obs}} - V_j^{\text{pred}}(\boldsymbol{\theta})|^2}{\sigma_{vis,j}^2}
$$

#### 1.3.2 Regularization Term (Normalizing Flow Prior)

$$
-\log q_\phi(\boldsymbol{\theta}) = -\log p_Z(z) + \log \left| \det \frac{\partial g_\phi}{\partial z} \right| = \frac{1}{2}\|z\|_2^2 + \frac{n_\theta}{2}\log(2\pi) - \log |\det J_{g_\phi}(z)|
$$

The log-determinant decomposes as:
$$
\log |\det J_{g_\phi}(z)| = \underbrace{\sum_{\ell=1}^{L} \text{logdet}_\ell(z)}_{\text{flow blocks}} + \underbrace{\sum_{i=1}^{n_\theta} \left( -\tilde{\theta}_i - 2\log(1 + e^{-\tilde{\theta}_i}) \right)}_{\text{sigmoid transform}}
$$

where $\boldsymbol{\theta} = \sigma(\tilde{\boldsymbol{\theta}})$ with $\sigma$ being the element-wise sigmoid.

#### 1.3.3 α-Divergence Extension

For $\alpha \neq 1$:
$$
\mathcal{L}_\alpha = \sum_{b=1}^{B} \omega_b \cdot s \cdot \ell_b, \quad \omega_b = \text{softmax}\left( -(1-\alpha) \cdot \boldsymbol{\ell} \right)_b
$$

where $\ell_b = w(k) \cdot \mathcal{L}_{\text{data}}^{(b)} - \log q_\phi^{(b)}$ is the per-sample loss.

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Symbol | Shape | Dtype | Description | Initialization |
|--------|-------|-------|-------------|----------------|
| $\phi$ | - | `float32` | All parameters of RealNVP flow | See below |
| `ActNorm.loc` | $(1,)$ | `float32` | Location parameter | Data-dependent: $-\mu_{\text{input}}$ |
| `ActNorm.log_scale_inv` | $(1,)$ | `float32` | Log inverse scale | Data-dependent: $\log(\sigma_{\text{input}} + 10^{-6})$ |
| `ZeroFC.fc.weight` | $(d_{out}, d_{in})$ | `float32` | Linear weights | $\mathbf{0}$ |
| `ZeroFC.fc.bias` | $(d_{out},)$ | `float32` | Linear bias | $\mathbf{0}$ |
| `ZeroFC.scale` | $(d_{out},)$ | `float32` | Output scale | $\mathbf{0}$ |
| `AffineCoupling.net[0].weight` | $(h, d_{in}/2)$ | `float32` | First layer weights | $\mathcal{N}(0, 0.05)$ |
| `AffineCoupling.net[0].bias` | $(h,)$ | `float32` | First layer bias | $\mathbf{0}$ |

### 2.2 Auxiliary/Latent Variables

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $z$ | $(B, n_\theta)$ | `float32` | Latent samples from base distribution |
| $\tilde{\boldsymbol{\theta}}$ | $(B, n_\theta)$ | `float32` | Pre-sigmoid parameter samples |
| $\boldsymbol{\theta}$ | $(B, n_\theta)$ | `float32` | Geometric parameters in $[0,1]^{n_\theta}$ |
| $I$ | $(B, N_{pix}, N_{pix})$ | `float32` | Generated images |
| $V$ | $(B, 2, N_{vis})$ | `float32` | Complex visibilities (real, imag) |

### 2.3 Hyperparameters/Constants

| Symbol | Default | Description |
|--------|---------|-------------|
| $N_{pix}$ | 64 | Image dimension (pixels per side) |
| $\text{FOV}$ | 120 | Field of view (μas) |
| $n_\theta$ | $4 + 6N_G + 2$ | Number of geometric parameters |
| $N_G$ | 2 | Number of nuisance Gaussians |
| $L$ | 16 | Number of flow blocks |
| $B$ | 64 | Batch size |
| $\eta$ | $10^{-4}$ | Learning rate |
| $\gamma_{\text{clip}}$ | $10^{-4}$ | Gradient clipping threshold |
| $\alpha$ | 1.0 | α-divergence parameter |
| $\lambda_{\text{logdet}}$ | 1.0 | Log-determinant weight |
| $k_0$ | 4 | Start order for annealing |
| $\tau$ | 2000 | Decay rate for annealing |
| $w_{cp}$ | $N_{cp}/N_{ca}$ | Closure phase weight |
| $w_{ca}$ | 1.0 | Closure amplitude weight |

**Parameter Ranges (mapped from $[0,1]$):**

| Parameter | Range | Units |
|-----------|-------|-------|
| Ring radius $r$ | $[10, 40]$ | μas |
| Ring width $\sigma_r$ | $[1, 40]$ | μas |
| Asymmetry $s$ | $[10^{-3}, 0.99]$ | - |
| Asymmetry angle $\eta$ | $[-181°, 181°]$ | degrees |
| Gaussian shift $(x, y)$ | $[-200, 200]$ | μas |
| Gaussian scale | $[10^{-3}, 2]$ | - |
| Gaussian width $(\sigma_x, \sigma_y)$ | $[1, 100]$ | μas |
| Gaussian rotation $\theta$ | $[0, 90.5°]$ | degrees |
| Floor fraction | $[0, 1]$ | - |
| Crescent flux scale | $[10^{-3}, 2]$ | - |

### 2.4 Observed Data/Inputs

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{F}$ | $(N_{pix}^2, N_{vis}, 2)$ | `float32` | DFT matrix (real, imag) |
| $V^{\text{obs}}$ | $(2, N_{vis})$ | `float32` | Observed complex visibilities |
| $\phi^{\text{obs}}$ | $(N_{cp},)$ | `float32` | Observed closure phases (degrees) |
| $\log A^{\text{obs}}$ | $(N_{ca},)$ | `float32` | Observed log closure amplitudes |
| $\sigma_{vis}$ | $(N_{vis},)$ | `float32` | Visibility uncertainties |
| $\sigma_{cp}$ | $(N_{cp},)$ | `float32` | Closure phase uncertainties (degrees) |
| $\sigma_{lca}$ | $(N_{ca},)$ | `float32` | Log closure amplitude uncertainties |
| $\mathcal{I}_{cp}$ | $(3, N_{cp})$ | `int64` | Closure phase baseline indices |
| $\mathcal{S}_{cp}$ | $(3, N_{cp})$ | `float32` | Closure phase baseline signs |
| $\mathcal{I}_{ca}$ | $(4, N_{ca})$ | `int64` | Closure amplitude baseline indices |

---

## 3. Preprocessing Pipeline

### Step 1: Load Observation Data
- **Input:** UVFITS file path
- **Output:** `obs` object containing $(u,v)$ coordinates, visibilities, timestamps, station labels
- **Operation:** `obs = eh.obsdata.load_uvfits(obspath)`

### Step 2: Estimate Reference Flux
- **Input:** `obs` object
- **Output:** $F_0 \in \mathbb{R}^+$ (flux constant)
- **Operation:** 
$$F_0 = \text{median}(\{|V_{ij}| : (i,j) \in \text{baseline}(AP, AA)\})$$

### Step 3: Construct Prior Image
- **Input:** $N_{pix}$, FOV, $F_0$, FWHM$_{\text{prior}}$
- **Output:** Reference image structure `simim`
- **Operations:**
  1. Create empty square image: $I_0 \in \mathbb{R}^{N_{pix} \times N_{pix}}$
  2. Add tophat disk: radius $= \frac{\text{FWHM}_{\text{prior}}}{2}$, total flux $= F_0$
  3. Blur with circular Gaussian kernel: $\sigma = \text{resolution}(\text{obs})$

### Step 4: Compute DFT Matrix
- **Input:** $(u,v)$ coordinates, pixel size $\Delta$, image dimensions
- **Output:** $\mathbf{F} \in \mathbb{R}^{N_{pix}^2 \times N_{vis} \times 2}$
- **Operation:** 
$$F_{pq} = \exp\left(-2\pi i (u_q x_p + v_q y_p)\right) \cdot \text{pulse}(u_q, v_q)$$
where $(x_p, y_p)$ is the position of pixel $p$.

### Step 5: Compute NUFFT Trajectory
- **Input:** $(u,v)$ coordinates, pixel size $\Delta$
- **Output:** `ktraj_vis` $\in \mathbb{R}^{1 \times 2 \times N_{vis}}$
- **Operation:**
$$\text{ktraj}_{vis} = 2\pi \Delta \cdot (v, u)^T$$

### Step 6: Build Closure Phase Index Maps
- **Input:** `obs.data`, `obs.cphase`
- **Output:** $\mathcal{I}_{cp} \in \mathbb{Z}^{3 \times N_{cp}}$, $\mathcal{S}_{cp} \in \{-1, +1\}^{3 \times N_{cp}}$
- **Operation:** For each closure phase triangle $(t_1, t_2, t_3)$:
  - Find visibility indices $i_{12}, i_{23}, i_{31}$ matching baselines
  - Set sign $= +1$ if baseline direction matches, $-1$ otherwise

### Step 7: Build Closure Amplitude Index Maps
- **Input:** `obs.data`, `obs.camp`
- **Output:** $\mathcal{I}_{ca} \in \mathbb{Z}^{4 \times N_{ca}}$
- **Operation:** For each closure amplitude quadrangle $(t_1, t_2, t_3, t_4)$:
  - Find visibility indices for baselines: $(t_1t_2), (t_3t_4), (t_1t_4), (t_2t_3)$

### Step 8: Extract Ground Truth and Uncertainties
- **Input:** `obs` object
- **Output:** $V^{\text{obs}}$, $\phi^{\text{obs}}$, $\log A^{\text{obs}}$, $\sigma_{vis}$, $\sigma_{cp}$, $\sigma_{lca}$
- **Operations:**
  - $V^{\text{obs}} = [\Re(V), \Im(V)]^T$
  - $\phi^{\text{obs}} = \text{obs.cphase['cphase']}$
  - $\log A^{\text{obs}} = \text{obs.logcamp['camp']}$

---

## 4. Algorithm Steps

### 4.1 Initialization Phase

**Step I-1: Initialize Pixel Grid**
$$
x_j = -1 + \frac{2j+1}{N_{pix}}, \quad j = 0, \ldots, N_{pix}-1
$$
$$
(G_x)_{jk} = x_k, \quad (G_y)_{jk} = -x_j
$$
$$
G_r = \sqrt{G_x^2 + G_y^2}, \quad G_\phi = \text{atan2}(G_y, G_x)
$$

**Step I-2: Initialize Flow Permutations**
- For $\ell = 1, \ldots, L$:
$$\pi_\ell = \text{RandomPermutation}(n_\theta; \text{seed}=\ell)$$
$$\pi_\ell^{-1} = \text{InversePermutation}(\pi_\ell)$$

**Step I-3: Initialize Optimizer**
$$
\text{optimizer} = \text{Adam}(\phi; \eta=10^{-4}, \beta_1=0.9, \beta_2=0.999)
$$

---

### 4.2 Main Iteration Loop

**For epoch $k = 0, 1, \ldots, K-1$:**

---

**Step 1: Compute Annealing Weight**
$$
w(k) = \min\left(10^{-k_0 + k/\tau}, 1\right)
$$
- **Input:** $k$, $k_0$, $\tau$
- **Output:** $w(k) \in (0, 1]$

---

**Step 2: Sample Latent Variables**
$$
z^{(b)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{n_\theta}), \quad b = 1, \ldots, B
$$
- **Input:** $B$, $n_\theta$
- **Output:** $\mathbf{Z} \in \mathbb{R}^{B \times n_\theta}$

---

**Step 3: Apply Inverse Flow (Generative Direction)**

For each flow block $\ell = L, L-1, \ldots, 1$:

**Step 3.1: Inverse Permutation**
$$
\mathbf{h}^{(\ell)} = \mathbf{h}^{(\ell+1)}[:, \pi_\ell^{-1}]
$$

**Step 3.2: Inverse Second Coupling**
$$
\mathbf{h}_a, \mathbf{h}_b = \text{split}(\text{reverse}(\mathbf{h}^{(\ell)}))
$$
$$
(\log s, t) = \text{NN}_{\ell,2}(\mathbf{h}_a)
$$
$$
\log \tilde{s} = \tanh(\log s)
$$
$$
\mathbf{h}_b' = \mathbf{h}_b \cdot e^{-\log \tilde{s}} - t
$$
$$
\Delta_{\ell,2} = -\sum_i \log \tilde{s}_i
$$

**Step 3.3: Inverse Second ActNorm**
$$
\mathbf{h} = \mathbf{h} \cdot e^{\gamma_{\ell,2}} - \mu_{\ell,2}
$$
$$
\Delta_{\ell,2}' = -n_\theta \cdot \gamma_{\ell,2}
$$

**Step 3.4: Dimension Reversal**
$$
\mathbf{h} = \mathbf{h}[:, n_\theta-1::-1]
$$

**Step 3.5: Inverse First Coupling** (analogous to 3.2)

**Step 3.6: Inverse First ActNorm** (analogous to 3.3)

**Step 3.7: Dimension Reversal**
$$
\mathbf{h} = \mathbf{h}[:, n_\theta-1::-1]
$$

**Accumulate log-determinant:**
$$
\log |\det J| = \sum_\ell (\Delta_{\ell,1} + \Delta_{\ell,1}' + \Delta_{\ell,2} + \Delta_{\ell,2}')
$$

- **Input:** $\mathbf{Z}$, flow parameters $\phi$
- **Output:** $\tilde{\boldsymbol{\Theta}} \in \mathbb{R}^{B \times n_\theta}$, $\log |\det J| \in \mathbb{R}^B$

---

**Step 4: Apply Sigmoid Transform**
$$
\boldsymbol{\Theta} = \sigma(\tilde{\boldsymbol{\Theta}}) = \frac{1}{1 + e^{-\tilde{\boldsymbol{\Theta}}}}
$$
$$
\Delta_\sigma^{(b)} = \sum_{i=1}^{n_\theta} \left( -\tilde{\theta}_i^{(b)} - 2\log(1 + e^{-\tilde{\theta}_i^{(b)}}) \right)
$$
$$
\log |\det J|_{\text{total}} = \log |\det J| + \Delta_\sigma
$$
- **Input:** $\tilde{\boldsymbol{\Theta}}$
- **Output:** $\boldsymbol{\Theta} \in [0,1]^{B \times n_\theta}$, updated $\log |\det J|_{\text{total}}$

---

**Step 5: Map Parameters to Physical Units**

For sample $b$, define scaled parameters:
$$
r = r_{\min} + \theta_1 (r_{\max} - r_{\min})
$$
$$
\sigma_r = \sigma_{r,\min} + \theta_2 (\sigma_{r,\max} - \sigma_{r,\min})
$$
$$
s = s_{\min} + \theta_3 (s_{\max} - s_{\min})
$$
$$
\eta = \frac{181\pi}{180}(2\theta_4 - 1)
$$

(All normalized by $0.5 \cdot \text{FOV}$ for spatial parameters)

- **Input:** $\boldsymbol{\Theta}$, parameter ranges
- **Output:** Physical parameters $(r, \sigma_r, s, \eta, \ldots)$

---

**Step 6: Generate Images from Geometric Model**

**Step 6.1: Compute Crescent Ring**
$$
R_{jk} = \exp\left( -\frac{((G_r)_{jk} - r)^2}{2\sigma_r^2} \right)
$$

**Step 6.2: Apply Asymmetry**
$$
S_{jk} = 1 + s \cos((G_\phi)_{jk} - \eta)
$$
$$
C_{jk} = S_{jk} \cdot R_{jk}
$$

**Step 6.3: Compute Disk Component**
$$
D_{jk} = \frac{1}{2}\left(1 + \text{erf}\left(\frac{r - (G_r)_{jk}}{\sqrt{2}\sigma_r}\right)\right)
$$

**Step 6.4: Normalize and Combine**
$$
\tilde{C} = \frac{C}{\sum_{jk} C_{jk} + \epsilon}, \quad \tilde{D} = \frac{D}{\sum_{jk} D_{jk} + \epsilon}
$$
$$
I_{\text{crescent}} = f_c \left( (1 - f_{\text{floor}}) \tilde{C} + f_{\text{floor}} \tilde{D} \right)
$$

**Step 6.5: Add Nuisance Gaussians**

For each Gaussian $g = 1, \ldots, N_G$:
$$
x_c = G_x - x_g, \quad y_c = G_y - y_g
$$
$$
x_{\text{rot}} = x_c \cos\theta_g + y_c \sin\theta_g
$$
$$
y_{\text{rot}} = -x_c \sin\theta_g + y_c \cos\theta_g
$$
$$
\delta = \frac{1}{2}\left(\frac{x_{\text{rot}}^2}{\sigma_{x,g}^2} + \frac{y_{\text{rot}}^2}{\sigma_{y,g}^2}\right)
$$
$$
G_g = \frac{1}{2\pi\sigma_{x,g}\sigma_{y,g}} e^{-\delta}
$$
$$
\tilde{G}_g = \frac{G_g}{\sum_{jk}(G_g)_{jk} + \epsilon}
$$
$$
I = I + a_g \cdot \tilde{G}_g
$$

**Step 6.6: Final Normalization**
$$
I^{(b)} = \frac{I}{\sum_{jk} I_{jk} + \epsilon}
$$

- **Input:** Physical parameters
- **Output:** $\mathbf{I} \in \mathbb{R}^{B \times N_{pix} \times N_{pix}}$

---

**Step 7: Compute Forward Model (Visibilities)**

**Step 7.1: Flatten Image**
$$
\mathbf{x}^{(b)} = \text{vec}(I^{(b)}) \in \mathbb{R}^{N_{pix}^2}
$$

**Step 7.2: Apply DFT**
$$
V_{\text{re}}^{(b)} = \mathbf{x}^{(b)T} \mathbf{F}_{:,:,0}
$$
$$
V_{\text{im}}^{(b)} = \mathbf{x}^{(b)T} \mathbf{F}_{:,:,1}
$$

**Step 7.3: Compute Visibility Amplitudes**
$$
|V_j|^{(b)} = \sqrt{(V_{\text{re},j}^{(b)})^2 + (V_{\text{im},j}^{(b)})^2 + \epsilon}
$$

- **Input:** $\mathbf{I}$, $\mathbf{F}$
- **Output:** $\mathbf{V} \in \mathbb{R}^{B \times 2 \times N_{vis}}$, $|\mathbf{V}| \in \mathbb{R}^{B \times N_{vis}}$

---

**Step 8: Compute Closure Phases**

For closure triangle $i$ with baseline indices $(j_1, j_2, j_3)$ and signs $(s_1, s_2, s_3)$:
$$
\psi_m = \text{atan2}(V_{\text{im}, j_m}, V_{\text{re}, j_m})
$$
$$
\phi_i^{\text{pred}} = \frac{180}{\pi}(s_1 \psi_1 + s_2 \psi_2 + s_3 \psi_3)
$$

- **Input:** $\mathbf{V}$, $\mathcal{I}_{cp}$, $\mathcal{S}_{cp}$
- **Output:** $\boldsymbol{\phi}^{\text{pred}} \in \mathbb{R}^{B \times N_{cp}}$

---

**Step 9: Compute Log Closure Amplitudes**

For closure quadrangle $i$ with baseline indices $(j_{12}, j_{34}, j_{14}, j_{23})$:
$$
\log A_i^{\text{pred}} = \log |V_{j_{12}}| + \log |V_{j_{34}}| - \log |V_{j_{14}}| - \log |V_{j_{23}}|
$$

- **Input:** $|\mathbf{V}|$, $\mathcal{I}_{ca}$
- **Output:** $\log \mathbf{A}^{\text{pred}} \in \mathbb{R}^{B \times N_{ca}}$

---

**Step 10: Compute Data Loss**

**For closure quantities mode:**
$$
\mathcal{L}_{cp}^{(b)} = \frac{2}{N_{cp}} \sum_{i=1}^{N_{cp}} \frac{1 - \cos\left(\frac{\pi}{180}(\phi_i^{\text{obs}} - \phi_i^{(b),\text{pred}})\right)}{\left(\frac{\pi}{180}\sigma_{cp,i}\right)^2}
$$
$$
\mathcal{L}_{lca}^{(b)} = \frac{1}{N_{ca}} \sum_{i=1}^{N_{ca}} \frac{(\log A_i^{\text{obs}} - \log A_i^{(b),\text{pred}})^2}{\sigma_{lca,i}^2}
$$
$$
\mathcal{L}_{\text{data}}^{(b)} = \frac{N_{ca}}{2} \left( w_{ca} \cdot \mathcal{L}_{lca}^{(b)} + w_{cp} \cdot \mathcal{L}_{cp}^{(b)} \right)
$$

- **Input:** Predicted and observed closure quantities, uncertainties
- **Output:** $\mathcal{L}_{\text{data}} \in \mathbb{R}^B$

---

**Step 11: Compute Log Probability**
$$
\log p^{(b)} = -\log |\det J|_{\text{total}}^{(b)} - \frac{1}{2}\|\mathbf{z}^{(b)}\|_2^2
$$

- **Input:** $\log |\det J|_{\text{total}}$, $\mathbf{Z}$
- **Output:** $\log \mathbf{p} \in \mathbb{R}^B$

---

**Step 12: