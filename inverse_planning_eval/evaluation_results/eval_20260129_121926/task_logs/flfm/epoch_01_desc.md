# Algorithmic Specification Document: 3D Richardson-Lucy Deconvolution

## 1. Mathematical Formulation

### 1.1 Problem Statement

**Objective**: Reconstruct a 3D fluorescent volume $\mathbf{x} \in \mathbb{R}^{N_z \times N_y \times N_x}$ from a 2D widefield microscopy measurement $\mathbf{y} \in \mathbb{R}^{N_y \times N_x}$.

### 1.2 Forward Model

The image formation model (widefield fluorescence microscopy) is given by:

$$\mathbf{y} = \mathcal{H}(\mathbf{x}) + \boldsymbol{\eta}$$

where the forward operator $\mathcal{H}: \mathbb{R}^{N_z \times N_y \times N_x} \rightarrow \mathbb{R}^{N_y \times N_x}$ is defined as:

$$[\mathcal{H}(\mathbf{x})]_{i,j} = \sum_{z=0}^{N_z-1} (\mathbf{x}_z * \mathbf{h}_z)_{i,j} = \sum_{z=0}^{N_z-1} \mathcal{F}^{-1}\left\{\mathcal{F}\{\mathbf{x}_z\} \cdot \mathcal{F}\{\mathbf{h}_z\}\right\}_{i,j}$$

where:
- $\mathbf{x}_z \in \mathbb{R}^{N_y \times N_x}$ is the $z$-th slice of the 3D object
- $\mathbf{h}_z \in \mathbb{R}^{N_y \times N_x}$ is the $z$-th slice of the 3D Point Spread Function (PSF)
- $*$ denotes 2D circular convolution
- $\mathcal{F}\{\cdot\}$ and $\mathcal{F}^{-1}\{\cdot\}$ denote 2D Fourier transform and its inverse
- $\boldsymbol{\eta} \sim \mathcal{N}(0, \sigma^2\mathbf{I})$ is additive Gaussian noise

### 1.3 Optimization Objective

The Richardson-Lucy algorithm solves the **Maximum Likelihood estimation** under Poisson noise assumption by iteratively minimizing the **Kullback-Leibler divergence**:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x} \geq 0} \; D_{KL}(\mathbf{y} \| \mathcal{H}(\mathbf{x}))$$

where the KL divergence (generalized I-divergence) is:

$$D_{KL}(\mathbf{y} \| \mathcal{H}(\mathbf{x})) = \sum_{i,j} \left[ y_{i,j} \log\frac{y_{i,j}}{[\mathcal{H}(\mathbf{x})]_{i,j}} - y_{i,j} + [\mathcal{H}(\mathbf{x})]_{i,j} \right]$$

**Decomposition**:
| Term | Formula | Description |
|------|---------|-------------|
| Data Fidelity | $D_{KL}(\mathbf{y} \| \mathcal{H}(\mathbf{x}))$ | KL divergence between measurement and projection |
| Constraint | $\mathbf{x} \geq 0$ | Non-negativity (enforced via projection) |
| Regularization | None (implicit via early stopping) | — |

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Symbol | Name | Shape | Dtype | Initialization | Description |
|--------|------|-------|-------|----------------|-------------|
| $\mathbf{x}^{(k)}$ | `estimate` | $(N_z, N_y, N_x)$ | `float32` | $x^{(0)}_{z,i,j} = \bar{y} = \frac{1}{N_y N_x}\sum_{i,j} y_{i,j}$ | 3D reconstructed volume at iteration $k$ |

### 2.2 Auxiliary/Intermediate Variables

| Symbol | Name | Shape | Dtype | Description |
|--------|------|-------|-------|-------------|
| $\mathbf{p}^{(k)}$ | `proj` | $(N_y, N_x)$ | `float32` | Forward projection $\mathcal{H}(\mathbf{x}^{(k)})$ |
| $\mathbf{r}^{(k)}$ | `ratio` | $(N_y, N_x)$ | `float32` | Error ratio $\mathbf{y} \oslash \mathbf{p}^{(k)}$ |
| $\mathbf{u}^{(k)}$ | `update_factor` | $(N_z, N_y, N_x)$ | `float32` | Back-projection $\mathcal{H}^T(\mathbf{r}^{(k)})$ |
| $\hat{\mathbf{x}}_z$ | `est_fft` | $(N_z, N_y, N_x/2+1)$ | `complex64` | FFT of estimate slices |
| $\hat{\mathbf{r}}$ | `err_fft` | $(N_y, N_x/2+1)$ | `complex64` | FFT of error ratio |

### 2.3 Hyperparameters/Constants

| Symbol | Name | Value | Description |
|--------|------|-------|-------------|
| $K$ | `num_iter` | 30 | Maximum number of iterations |
| $\epsilon$ | — | $10^{-8}$ | Numerical stability constant for division |
| $\sigma_0$ | `sigma0` | 2.0 | Base Gaussian width for PSF |
| $\alpha$ | — | 0.1 | PSF axial broadening coefficient |
| $\sigma_\eta$ | — | 0.1 | Measurement noise standard deviation |
| $N_{\text{beads}}$ | `num_beads` | 10 | Number of point sources in phantom |
| $I_{\text{bead}}$ | — | 100.0 | Intensity of each point source |

### 2.4 Observed Data/Inputs

| Symbol | Name | Shape | Dtype | Description |
|--------|------|-------|-------|-------------|
| $\mathbf{y}$ | `measurement` | $(N_y, N_x)$ | `float32` | 2D observed widefield image |
| $\hat{\mathbf{H}}$ | `psf_fft` | $(N_z, N_y, N_x/2+1)$ | `complex64` | Precomputed FFT of PSF |
| $\hat{\mathbf{H}}^T$ | `psft_fft` | $(N_z, N_y, N_x/2+1)$ | `complex64` | Precomputed FFT of flipped PSF |
| $\mathbf{x}^*$ | `ground_truth` | $(N_z, N_y, N_x)$ | `float32` | Ground truth (for evaluation only) |

### 2.5 Dimension Parameters

| Symbol | Name | Default Value | Description |
|--------|------|---------------|-------------|
| $N_z$ | `nz` | 20 | Number of axial slices |
| $N_y$ | `ny` | 128 | Height in pixels |
| $N_x$ | `nx` | 128 | Width in pixels |

---

## 3. Preprocessing Pipeline

### Step 1: PSF Generation
**Input**: Shape $(N_z, N_y, N_x)$, parameters $(\sigma_0, \alpha)$  
**Output**: $\mathbf{h} \in \mathbb{R}^{N_z \times N_y \times N_x}$

$$h_{z,i,j} = \frac{1}{Z} \exp\left( -\frac{(i - c_y)^2 + (j - c_x)^2}{2\sigma_z^2} \right)$$

where:
- $c_y = \lfloor N_y/2 \rfloor$, $c_x = \lfloor N_x/2 \rfloor$, $c_z = \lfloor N_z/2 \rfloor$
- $\sigma_z = \sigma_0 + \alpha |z - c_z|$
- $Z = \sum_{z,i,j} h_{z,i,j}$ (normalization constant ensuring $\sum \mathbf{h} = 1$)

### Step 2: Ground Truth Object Generation
**Input**: Shape $(N_z, N_y, N_x)$, seed = 42  
**Output**: $\mathbf{x}^* \in \mathbb{R}^{N_z \times N_y \times N_x}$

$$x^*_{z,i,j} = \begin{cases} I_{\text{bead}} & \text{if } (z,i,j) \in \mathcal{S} \\ 0 & \text{otherwise} \end{cases}$$

where $\mathcal{S}$ is a set of $N_{\text{beads}}$ randomly sampled locations with:
- $z \sim \text{Uniform}\{0, \ldots, N_z-1\}$
- $i \sim \text{Uniform}\{10, \ldots, N_y-11\}$
- $j \sim \text{Uniform}\{10, \ldots, N_x-11\}$

### Step 3: Measurement Simulation (Forward Model)
**Input**: $\mathbf{x}^*$, $\mathbf{h}$  
**Output**: $\mathbf{y} \in \mathbb{R}^{N_y \times N_x}$

$$\mathbf{y} = \max\left(0, \; \sum_{z=0}^{N_z-1} \mathcal{F}^{-1}\left\{\mathcal{F}\{\mathbf{x}^*_z\} \cdot \mathcal{F}\{\mathbf{h}_z\}\right\} + \boldsymbol{\eta}\right)$$

where $\eta_{i,j} \sim \mathcal{N}(0, \sigma_\eta^2)$

### Step 4: PSF FFT Precomputation
**Input**: $\mathbf{h} \in \mathbb{R}^{N_z \times N_y \times N_x}$  
**Output**: $\hat{\mathbf{H}}, \hat{\mathbf{H}}^T \in \mathbb{C}^{N_z \times N_y \times (N_x/2+1)}$

$$\hat{\mathbf{H}}_z = \mathcal{F}_{\text{rfft2}}\{\mathbf{h}_z\}$$

$$\hat{\mathbf{H}}^T_z = \mathcal{F}_{\text{rfft2}}\{\text{flip}_{2D}(\mathbf{h}_z)\}$$

where $\text{flip}_{2D}$ reverses both spatial dimensions (equivalent to $\mathbf{h}_z(-i,-j)$ with periodic boundary).

---

## 4. Algorithm Steps: Richardson-Lucy Deconvolution

### Initialization

| Step | Operation | Formula | Output |
|------|-----------|---------|--------|
| **I.1** | Compute measurement mean | $\bar{y} = \frac{1}{N_y N_x} \sum_{i,j} y_{i,j}$ | Scalar |
| **I.2** | Initialize estimate | $x^{(0)}_{z,i,j} = \bar{y} \quad \forall z,i,j$ | $\mathbf{x}^{(0)} \in \mathbb{R}^{N_z \times N_y \times N_x}$ |

### Main Iteration Loop

**For** $k = 0, 1, \ldots, K-1$ **do**:

| Step | Name | Mathematical Formula | Input Variables | Output Variables |
|------|------|---------------------|-----------------|------------------|
| **4.1** | Forward FFT | $\hat{\mathbf{X}}^{(k)}_z = \mathcal{F}_{\text{rfft2}}\{\mathbf{x}^{(k)}_z\}, \quad z = 0,\ldots,N_z-1$ | $\mathbf{x}^{(k)}$ | $\hat{\mathbf{X}}^{(k)}$ |
| **4.2** | Frequency-domain multiplication | $\hat{\mathbf{C}}^{(k)}_z = \hat{\mathbf{X}}^{(k)}_z \odot \hat{\mathbf{H}}_z$ | $\hat{\mathbf{X}}^{(k)}$, $\hat{\mathbf{H}}$ | $\hat{\mathbf{C}}^{(k)}$ |
| **4.3** | Inverse FFT | $\mathbf{C}^{(k)}_z = \mathcal{F}^{-1}_{\text{irfft2}}\{\hat{\mathbf{C}}^{(k)}_z\}$ | $\hat{\mathbf{C}}^{(k)}$ | $\mathbf{C}^{(k)}$ |
| **4.4** | **Forward Projection** | $\mathbf{p}^{(k)} = \sum_{z=0}^{N_z-1} \mathbf{C}^{(k)}_z$ | $\mathbf{C}^{(k)}$ | $\mathbf{p}^{(k)} \in \mathbb{R}^{N_y \times N_x}$ |
| **4.5** | **Compute Error Ratio** | $r^{(k)}_{i,j} = \frac{y_{i,j}}{p^{(k)}_{i,j} + \epsilon}$ | $\mathbf{y}$, $\mathbf{p}^{(k)}$ | $\mathbf{r}^{(k)} \in \mathbb{R}^{N_y \times N_x}$ |
| **4.6** | Error FFT | $\hat{\mathbf{r}}^{(k)} = \mathcal{F}_{\text{rfft2}}\{\mathbf{r}^{(k)}\}$ | $\mathbf{r}^{(k)}$ | $\hat{\mathbf{r}}^{(k)}$ |
| **4.7** | Back-projection (freq. domain) | $\hat{\mathbf{U}}^{(k)}_z = \hat{\mathbf{r}}^{(k)} \odot \hat{\mathbf{H}}^T_z$ | $\hat{\mathbf{r}}^{(k)}$, $\hat{\mathbf{H}}^T$ | $\hat{\mathbf{U}}^{(k)}$ |
| **4.8** | **Back-projection Inverse FFT** | $\mathbf{u}^{(k)}_z = \mathcal{F}^{-1}_{\text{irfft2}}\{\hat{\mathbf{U}}^{(k)}_z\}$ | $\hat{\mathbf{U}}^{(k)}$ | $\mathbf{u}^{(k)} \in \mathbb{R}^{N_z \times N_y \times N_x}$ |
| **4.9** | **Multiplicative Update** | $\tilde{\mathbf{x}}^{(k+1)} = \mathbf{x}^{(k)} \odot \mathbf{u}^{(k)}$ | $\mathbf{x}^{(k)}$, $\mathbf{u}^{(k)}$ | $\tilde{\mathbf{x}}^{(k+1)}$ |
| **4.10** | **Non-negativity Projection** | $x^{(k+1)}_{z,i,j} = \max(0, \tilde{x}^{(k+1)}_{z,i,j})$ | $\tilde{\mathbf{x}}^{(k+1)}$ | $\mathbf{x}^{(k+1)}$ |

**End For**

### Compact Update Formula

The complete Richardson-Lucy update can be written compactly as:

$$\boxed{\mathbf{x}^{(k+1)} = \mathcal{P}_{\geq 0}\left[ \mathbf{x}^{(k)} \odot \mathcal{H}^T\left( \frac{\mathbf{y}}{\mathcal{H}(\mathbf{x}^{(k)}) + \epsilon} \right) \right]}$$

where:
- $\mathcal{H}(\cdot)$: Forward operator (projection)
- $\mathcal{H}^T(\cdot)$: Adjoint operator (back-projection)
- $\mathcal{P}_{\geq 0}[\cdot] = \max(0, \cdot)$: Non-negativity projection
- $\odot$: Element-wise (Hadamard) product
- $\oslash$: Element-wise division

### Adjoint Operator Definition

The adjoint (back-projection) operator $\mathcal{H}^T: \mathbb{R}^{N_y \times N_x} \rightarrow \mathbb{R}^{N_z \times N_y \times N_x}$ is:

$$[\mathcal{H}^T(\mathbf{r})]_z = \mathbf{r} \star \mathbf{h}_z = \mathcal{F}^{-1}\left\{\mathcal{F}\{\mathbf{r}\} \cdot \mathcal{F}\{\text{flip}(\mathbf{h}_z)\}\right\}$$

where $\star$ denotes 2D correlation (equivalent to convolution with flipped kernel).

---

## 5. Stopping Criterion

| Criterion Type | Condition | Value |
|---------------|-----------|-------|
| **Fixed Iteration Limit** | $k \geq K$ | $K = 30$ |
| Convergence Check | Not implemented | — |
| Residual Threshold | Not implemented | — |

The algorithm terminates after exactly $K$ iterations (no early stopping based on convergence).

---

## 6. Evaluation Metrics

### 6.1 Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{N_z N_y N_x} \sum_{z=0}^{N_z-1} \sum_{i=0}^{N_y-1} \sum_{j=0}^{N_x-1} \left( x^{(K)}_{z,i,j} - x^*_{z,i,j} \right)^2$$

### 6.2 Peak Signal-to-Noise Ratio (PSNR)

$$\text{PSNR} = 20 \cdot \log_{10}\left( \frac{\max(\mathbf{x}^*)}{\sqrt{\text{MSE}}} \right) \quad \text{[dB]}$$

where $\max(\mathbf{x}^*) = \max_{z,i,j} x^*_{z,i,j}$ is the maximum intensity in the ground truth volume.

---

## 7. Algorithm Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    RICHARDSON-LUCY DECONVOLUTION                 │
├─────────────────────────────────────────────────────────────────┤
│  INPUT: y ∈ ℝ^(Ny×Nx), Ĥ ∈ ℂ^(Nz×Ny×Nx'), Ĥᵀ ∈ ℂ^(Nz×Ny×Nx')  │
│  OUTPUT: x̂ ∈ ℝ^(Nz×Ny×Nx)                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INIT: x⁽⁰⁾ ← mean(y) · 𝟙                                       │
│                                                                  │
│  FOR k = 0 TO K-1:                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ① FORWARD:    p⁽ᵏ⁾ = Σ_z F⁻¹{F{x_z⁽ᵏ⁾} · Ĥ_z}           │  │
│  │                                                           │  │
│  │  ② RATIO:      r⁽ᵏ⁾ = y ⊘ (p⁽ᵏ⁾ + ε)                     │  │
│  │                                                           │  │
│  │  ③ BACKWARD:   u_z⁽ᵏ⁾ = F⁻¹{F{r⁽ᵏ⁾} · Ĥᵀ_z}  ∀z          │  │
│  │                                                           │  │
│  │  ④ UPDATE:     x⁽ᵏ⁺¹⁾ = max(0, x⁽ᵏ⁾ ⊙ u⁽ᵏ⁾)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│  END FOR                                                         │
│                                                                  │
│  RETURN x̂ = x⁽ᴷ⁾                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathcal{F}\{\cdot\}$ | 2D Discrete Fourier Transform |
| $\mathcal{F}^{-1}\{\cdot\}$ | 2D Inverse Discrete Fourier Transform |
| $\mathcal{F}_{\text{rfft2}}\{\cdot\}$ | Real-input 2D FFT (half-spectrum) |
| $\odot$ | Element-wise (Hadamard) product |
| $\oslash$ | Element-wise division |
| $*$ | 2D circular convolution |
| $\star$ | 2D correlation |
| $\mathbf{1}$ | Tensor of ones |
| $\|\cdot\|_F$ | Frobenius norm |