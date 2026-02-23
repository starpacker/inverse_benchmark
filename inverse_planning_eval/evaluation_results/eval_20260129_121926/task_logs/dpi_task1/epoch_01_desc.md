# Algorithmic Specification Document: Deep Probabilistic Imaging for Radio Interferometry

## 1. Mathematical Formulation

### 1.1 Problem Statement
Reconstruct a non-negative image $\mathbf{I} \in \mathbb{R}_+^{N \times N}$ from sparse interferometric observations in the Fourier domain, using a variational inference framework with normalizing flows.

### 1.2 Complete Objective Function

The algorithm minimizes the following variational free energy over normalizing flow parameters $\boldsymbol{\theta}$ and log-scale parameter $s$:

$$
\mathcal{L}(\boldsymbol{\theta}, s) = \mathbb{E}_{\mathbf{z} \sim \mathcal{N}(0, \mathbf{I}_d)} \left[ \mathcal{L}_{\text{data}}(\mathbf{I}_{\boldsymbol{\theta},s}(\mathbf{z})) + \mathcal{L}_{\text{prior}}(\mathbf{I}_{\boldsymbol{\theta},s}(\mathbf{z})) - \lambda_{\text{det}} \log \left| \det \frac{\partial \mathbf{I}}{\partial \mathbf{z}} \right| \right]
$$

### 1.3 Decomposition of Loss Terms

#### Data Fidelity Term (Closure Phase Likelihood)
$$
\mathcal{L}_{\text{data}}(\mathbf{I}) = w_{\text{cp}} \cdot \frac{1}{N_{\text{cp}}} \sum_{k=1}^{N_{\text{cp}}} \frac{2 \left(1 - \cos(\phi_k^{\text{true}} - \phi_k^{\text{pred}}(\mathbf{I}))\right)}{\sigma_{\text{cp},k}^2}
$$

where $\phi_k^{\text{pred}}$ is the predicted closure phase and $\sigma_{\text{cp},k}$ is the uncertainty (in radians).

#### Regularization Terms
**Cross-Entropy with Prior:**
$$
\mathcal{L}_{\text{CE}}(\mathbf{I}) = w_{\text{CE}} \cdot \frac{1}{N^2} \sum_{i,j} I_{ij} \left( \log(I_{ij} + \epsilon) - \log(P_{ij} + \epsilon) \right)
$$

**Flux Constraint:**
$$
\mathcal{L}_{\text{flux}}(\mathbf{I}) = w_{\text{flux}} \cdot \left( \sum_{i,j} I_{ij} - F_0 \right)^2
$$

#### Entropy Maximization (Log-Determinant)
$$
\mathcal{L}_{\text{det}} = -\lambda_{\text{det}} \cdot \log \left| \det \frac{\partial \mathbf{I}}{\partial \mathbf{z}} \right|
$$

The full log-determinant decomposes as:
$$
\log \left| \det \frac{\partial \mathbf{I}}{\partial \mathbf{z}} \right| = \log \left| \det \mathbf{J}_{g_\theta} \right| + \sum_{i,j} \left( \tilde{I}_{ij} - \text{softplus}(\tilde{I}_{ij}) \right) + s \cdot N^2
$$

### 1.4 Combined Objective
$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{flux}} + \mathcal{L}_{\text{det}}
$$

---

## 2. Variables Taxonomy

### 2.1 Optimization (Primal) Variables

| Variable | Symbol | Shape | Dtype | Initialization |
|----------|--------|-------|-------|----------------|
| NVP flow parameters | $\boldsymbol{\theta}$ | Model-dependent | float32 | Random (PyTorch default) |
| Log-scale parameter | $s$ | $(1,)$ | float32 | $s_0 = \log\left(\frac{F_0}{0.8 \cdot N^2}\right)$ |

### 2.2 Auxiliary/Intermediate Variables

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| Latent samples | $\mathbf{z}$ | $(B, N^2)$ | float32 | $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I}_{N^2})$ |
| Pre-activation image | $\tilde{\mathbf{I}}$ | $(B, N, N)$ | float32 | $\tilde{\mathbf{I}} = g_{\boldsymbol{\theta}}(\mathbf{z})$ |
| Reconstructed image | $\mathbf{I}$ | $(B, N, N)$ | float32 | $\mathbf{I} = \text{softplus}(\tilde{\mathbf{I}}) \cdot e^s$ |
| Complex visibilities | $\mathbf{V}$ | $(B, 2, M)$ | float32 | Real/Imag stacked |
| Visibility amplitudes | $|\mathbf{V}|$ | $(B, M)$ | float32 | $\sqrt{V_r^2 + V_i^2 + \epsilon}$ |
| Closure phases | $\boldsymbol{\phi}$ | $(B, N_{\text{cp}})$ | float32 | In degrees |
| Log-determinant | $\log|\det \mathbf{J}|$ | $(B,)$ | float32 | Accumulated Jacobian |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Image dimension | $N$ | 32 | Pixels per side |
| Number of flow layers | $L$ | 16 | RealNVP depth |
| Batch size | $B$ | 32 | Samples per iteration |
| Learning rate | $\eta$ | $10^{-4}$ | Adam optimizer |
| Number of epochs | $K$ | 100 | Training iterations |
| Closure phase weight | $w_{\text{cp}}$ | $N_{\text{cp}} / N_{\text{camp}}$ | Data fidelity weight |
| Cross-entropy weight | $w_{\text{CE}}$ | 1024.0 | Prior regularization |
| Flux weight | $w_{\text{flux}}$ | 1000.0 | Flux constraint |
| Log-det weight factor | $\lambda_{\text{factor}}$ | 1.0 | User-specified |
| Log-det weight | $\lambda_{\text{det}}$ | $2 \lambda_{\text{factor}} / N_{\text{camp}}$ | Entropy weight |
| Gradient clip norm | - | 0.1 | Gradient clipping |
| Numerical epsilon | $\epsilon$ | $10^{-16}$ (amp), $10^{-12}$ (log) | Stability constants |

### 2.4 Observed Data/Inputs

| Data | Symbol | Shape | Source |
|------|--------|-------|--------|
| UV coordinates | $(\mathbf{u}, \mathbf{v})$ | $(M, 2)$ | UVFITS file |
| Complex visibilities (observed) | $\mathbf{V}^{\text{obs}}$ | $(M,)$ complex | UVFITS file |
| Visibility uncertainties | $\boldsymbol{\sigma}_V$ | $(M,)$ | UVFITS file |
| Closure phases (observed) | $\boldsymbol{\phi}^{\text{true}}$ | $(N_{\text{cp}},)$ | Computed from obs |
| Closure phase uncertainties | $\boldsymbol{\sigma}_{\text{cp}}$ | $(N_{\text{cp}},)$ | Computed from obs |
| Flux constant | $F_0$ | scalar | Median APEX-ALMA amplitude |
| Prior image | $\mathbf{P}$ | $(N, N)$ | Gaussian blob |
| Field of view | FOV | scalar (μas) | User-specified |
| Prior FWHM | FWHM | scalar (μas) | User-specified |

---

## 3. Preprocessing Pipeline

### Step 1: Load Observation Data
**Input:** UVFITS file path  
**Output:** Observation object `obs` with visibility data

$$
\text{obs} = \texttt{load\_uvfits}(\text{obspath})
$$

### Step 2: Estimate Flux Constant
**Input:** Observation object  
**Output:** Scalar flux $F_0$

$$
F_0 = \text{median}\left( \{|V_{ij}| : (i,j) \in \text{APEX-ALMA baseline}\} \right)
$$

### Step 3: Construct Prior Image
**Input:** $N$, FOV, FWHM, $F_0$  
**Output:** Prior image $\mathbf{P} \in \mathbb{R}^{N \times N}$

$$
\mathbf{P} = \text{Gaussian}(\mu=0, \sigma_x = \sigma_y = \text{FWHM}_{\text{rad}}, \text{total flux} = F_0)
$$

### Step 4: Compute DFT Matrix
**Input:** Pixel size $\Delta$, image dimensions $(N, N)$, UV coordinates  
**Output:** DFT matrix $\mathbf{F} \in \mathbb{R}^{N^2 \times M \times 2}$

Pixel coordinates:
$$
x_i = (i - N/2) \cdot \Delta, \quad y_j = (j - N/2) \cdot \Delta
$$

DFT matrix elements:
$$
F_{pq} = \exp\left(-2\pi i (u_q x_p + v_q y_p)\right)
$$

Stored as real/imaginary stacked: $\mathbf{F}[:,:,0] = \Re(\mathbf{F})$, $\mathbf{F}[:,:,1] = \Im(\mathbf{F})$

### Step 5: Build Closure Phase Index Maps
**Input:** Observation with baseline data and closure phase triangles  
**Output:** Index tensors $(\mathbf{idx}_1, \mathbf{idx}_2, \mathbf{idx}_3)$ and sign tensors $(\mathbf{s}_1, \mathbf{s}_2, \mathbf{s}_3)$

For each closure phase triangle $(t_1, t_2, t_3)$:
- Find visibility indices corresponding to baselines $(t_1 \to t_2)$, $(t_2 \to t_3)$, $(t_3 \to t_1)$
- Store sign $\pm 1$ based on baseline orientation

### Step 6: Convert Data to Tensors
**Input:** NumPy arrays  
**Output:** PyTorch tensors on specified device

| Array | Tensor Type |
|-------|-------------|
| DFT matrix | `torch.float32` |
| Closure phase indices | `torch.long` |
| Closure phase signs | `torch.float32` |
| Observed closure phases | `torch.float32` |
| Prior image | `torch.float32` |
| Uncertainties | `torch.float32` |

---

## 4. Algorithm Steps

### Initialization

**Step 0.1: Initialize Normalizing Flow**
$$
g_{\boldsymbol{\theta}}: \mathbb{R}^{N^2} \to \mathbb{R}^{N^2}
$$
RealNVP with $L$ affine coupling layers.

**Step 0.2: Initialize Log-Scale Parameter**
$$
s \leftarrow \log\left(\frac{F_0}{0.8 \cdot N^2}\right)
$$

**Step 0.3: Initialize Optimizer**
$$
\text{optimizer} = \text{Adam}(\{\boldsymbol{\theta}, s\}, \eta)
$$

### Per-Iteration Updates (for $k = 0, 1, \ldots, K-1$)

---

**Step 1: Sample Latent Variables**
$$
\mathbf{z}^{(b)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{N^2}), \quad b = 1, \ldots, B
$$
**Input:** None  
**Output:** $\mathbf{Z} \in \mathbb{R}^{B \times N^2}$

---

**Step 2: Generate Pre-Activation Images via Normalizing Flow**
$$
(\tilde{\mathbf{I}}^{(b)}, \log|\det \mathbf{J}_g^{(b)}|) = g_{\boldsymbol{\theta}}^{-1}(\mathbf{z}^{(b)})
$$
**Input:** $\mathbf{Z}$, flow parameters $\boldsymbol{\theta}$  
**Output:** $\tilde{\mathbf{I}} \in \mathbb{R}^{B \times N \times N}$, $\log|\det \mathbf{J}_g| \in \mathbb{R}^{B}$

---

**Step 3: Apply Positivity Transform and Scaling**
$$
\mathbf{I}^{(b)} = \text{softplus}(\tilde{\mathbf{I}}^{(b)}) \cdot e^s
$$
where $\text{softplus}(x) = \log(1 + e^x)$

**Input:** $\tilde{\mathbf{I}}$, $s$  
**Output:** $\mathbf{I} \in \mathbb{R}_+^{B \times N \times N}$

---

**Step 4: Accumulate Log-Determinant Correction**
$$
\log|\det \mathbf{J}|^{(b)} = \log|\det \mathbf{J}_g^{(b)}| + \sum_{i,j}\left(\tilde{I}_{ij}^{(b)} - \text{softplus}(\tilde{I}_{ij}^{(b)})\right) + s \cdot N^2
$$

**Input:** $\log|\det \mathbf{J}_g|$, $\tilde{\mathbf{I}}$, $s$  
**Output:** $\log|\det \mathbf{J}| \in \mathbb{R}^{B}$

---

**Step 5: Compute Complex Visibilities (Forward Model)**
$$
\mathbf{V}^{(b)} = \mathbf{F} \cdot \text{vec}(\mathbf{I}^{(b)})
$$

Explicitly:
$$
V_m^{(b)} = \sum_{p=1}^{N^2} I_p^{(b)} \cdot F_{pm}
$$

Split into real/imaginary:
$$
V_{r,m}^{(b)} = \sum_p I_p^{(b)} F_{pm}^{(r)}, \quad V_{i,m}^{(b)} = \sum_p I_p^{(b)} F_{pm}^{(i)}
$$

**Input:** $\mathbf{I}$, $\mathbf{F}$  
**Output:** $\mathbf{V} \in \mathbb{R}^{B \times 2 \times M}$

---

**Step 6: Compute Visibility Amplitudes**
$$
|V_m^{(b)}| = \sqrt{(V_{r,m}^{(b)})^2 + (V_{i,m}^{(b)})^2 + \epsilon}
$$

**Input:** $\mathbf{V}$  
**Output:** $|\mathbf{V}| \in \mathbb{R}^{B \times M}$

---

**Step 7: Compute Closure Phases**

For each closure triangle $k$ with visibility indices $(m_1, m_2, m_3)$ and signs $(s_1, s_2, s_3)$:

$$
\phi_k^{(b)} = \left( s_1 \cdot \text{atan2}(V_{i,m_1}^{(b)}, V_{r,m_1}^{(b)}) + s_2 \cdot \text{atan2}(V_{i,m_2}^{(b)}, V_{r,m_2}^{(b)}) + s_3 \cdot \text{atan2}(V_{i,m_3}^{(b)}, V_{r,m_3}^{(b)}) \right) \cdot \frac{180}{\pi}
$$

**Input:** $\mathbf{V}$, index/sign lists  
**Output:** $\boldsymbol{\phi} \in \mathbb{R}^{B \times N_{\text{cp}}}$ (degrees)

---

**Step 8: Compute Cross-Entropy Loss**
$$
\mathcal{L}_{\text{CE}}^{(b)} = \frac{1}{N^2} \sum_{i,j} I_{ij}^{(b)} \left( \log(I_{ij}^{(b)} + 10^{-12}) - \log(P_{ij} + 10^{-12}) \right)
$$

**Input:** $\mathbf{I}$, $\mathbf{P}$  
**Output:** $\mathcal{L}_{\text{CE}} \in \mathbb{R}^{B}$

---

**Step 9: Compute Flux Loss**
$$
\mathcal{L}_{\text{flux}}^{(b)} = \left( \sum_{i,j} I_{ij}^{(b)} - F_0 \right)^2
$$

**Input:** $\mathbf{I}$, $F_0$  
**Output:** $\mathcal{L}_{\text{flux}} \in \mathbb{R}^{B}$

---

**Step 10: Compute Closure Phase Loss**

Convert to radians:
$$
\alpha_k^{\text{true}} = \phi_k^{\text{true}} \cdot \frac{\pi}{180}, \quad \alpha_k^{(b)} = \phi_k^{(b)} \cdot \frac{\pi}{180}
$$

Von Mises-like loss:
$$
\mathcal{L}_{\text{cp}}^{(b)} = \frac{2}{N_{\text{cp}}} \sum_{k=1}^{N_{\text{cp}}} \frac{1 - \cos(\alpha_k^{\text{true}} - \alpha_k^{(b)})}{(\sigma_{\text{cp},k} \cdot \pi / 180)^2}
$$

**Input:** $\boldsymbol{\phi}$, $\boldsymbol{\phi}^{\text{true}}$, $\boldsymbol{\sigma}_{\text{cp}}$  
**Output:** $\mathcal{L}_{\text{cp}} \in \mathbb{R}^{B}$

---

**Step 11: Aggregate Total Loss**
$$
\mathcal{L}^{(b)} = w_{\text{cp}} \cdot \mathcal{L}_{\text{cp}}^{(b)} + w_{\text{CE}} \cdot \mathcal{L}_{\text{CE}}^{(b)} + w_{\text{flux}} \cdot \mathcal{L}_{\text{flux}}^{(b)} - \lambda_{\text{det}} \cdot \log|\det \mathbf{J}|^{(b)}
$$

$$
\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \mathcal{L}^{(b)}
$$

**Input:** All individual losses  
**Output:** Scalar $\mathcal{L}$

---

**Step 12: Backpropagation and Parameter Update**
$$
\mathbf{g} = \nabla_{\boldsymbol{\theta}, s} \mathcal{L}
$$

Gradient clipping:
$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{0.1}{\|\mathbf{g}\|_2}\right)
$$

Adam update:
$$
(\boldsymbol{\theta}, s) \leftarrow \text{Adam}((\boldsymbol{\theta}, s), \mathbf{g}, \eta)
$$

**Input:** $\mathcal{L}$, current parameters  
**Output:** Updated $(\boldsymbol{\theta}, s)$

---

## 5. Stopping Criterion

The algorithm terminates when:

$$
k \geq K \quad \text{(fixed iteration budget)}
$$

where $K$ is the user-specified number of epochs (default: 100).

**Note:** No convergence-based early stopping is implemented. The algorithm runs for exactly $K$ iterations.

---

## 6. Evaluation Metrics

### 6.1 Mean Reconstructed Image
$$
\bar{\mathbf{I}} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{I}^{(b)}
$$

### 6.2 Uncertainty Quantification (Standard Deviation)
$$
\boldsymbol{\sigma}_{\mathbf{I}} = \sqrt{\frac{1}{B} \sum_{b=1}^{B} (\mathbf{I}^{(b)} - \bar{\mathbf{I}})^2}
$$

### 6.3 Flux Error
$$
\text{Flux Error (\%)} = \frac{|\sum_{i,j} \bar{I}_{ij} - F_0|}{F_0} \times 100
$$

### 6.4 Mean Closure Phase Error
$$
\Delta\phi_k = \min\left( |\phi_k^{\text{true}} - \phi_k^{\text{pred}}|, \, 360° - |\phi_k^{\text{true}} - \phi_k^{\text{pred}}| \right)
$$

$$
\text{Mean CP Error} = \frac{1}{N_{\text{cp}}} \sum_{k=1}^{N_{\text{cp}}} \Delta\phi_k \quad \text{(degrees)}
$$

### 6.5 Final Training Loss
$$
\mathcal{L}_{\text{final}} = \mathcal{L}^{(K-1)}
$$

---

## Summary Table: Loss Component Weights

| Component | Symbol | Formula | Default Value |
|-----------|--------|---------|---------------|
| Closure Phase | $w_{\text{cp}}$ | $N_{\text{cp}} / N_{\text{camp}}$ | Data-dependent |
| Cross-Entropy | $w_{\text{CE}}$ | Fixed | 1024.0 |
| Flux | $w_{\text{flux}}$ | Fixed | 1000.0 |
| Log-Determinant | $\lambda_{\text{det}}$ | $2 \lambda_{\text{factor}} / N_{\text{camp}}$ | Data-dependent |