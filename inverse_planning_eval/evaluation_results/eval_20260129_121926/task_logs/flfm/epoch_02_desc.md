# Algorithmic Specification Document: Richardson-Lucy 3D Deconvolution

## 1. OBJECTIVE FUNCTION

- **full_expression**:
$$\min_{x \geq 0} \; D_{\text{KL}}(y \,\|\, Hx) = \min_{x \geq 0} \sum_{i=0}^{n_y-1} \sum_{j=0}^{n_x-1} \left[ (Hx)_{ij} - y_{ij} \log\left((Hx)_{ij}\right) \right]$$

- **data_fidelity_term**:
$$\mathcal{L}(x) = \sum_{i,j} \left[ (Hx)_{ij} - y_{ij} \log\left((Hx)_{ij}\right) \right]$$
This is the negative Poisson log-likelihood. The forward operator $H$ is defined as:
$$(Hx)_{ij} = \sum_{z=0}^{n_z-1} (h_z * x_z)_{ij}$$
where $*$ denotes 2D circular convolution.

- **regularization_term**: None (unregularized Richardson-Lucy)

- **constraints**:
  - $x_{z,i,j} \geq 0 \quad \forall \, z \in [0, n_z), \; i \in [0, n_y), \; j \in [0, n_x)$

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $x^{(k)}$ | primal | $(n_z, n_y, n_x)$ | float32 | $x^{(0)}_{z,i,j} = \bar{y}$ | 3D volume estimate at iteration $k$ |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $y$ | observations | $(n_y, n_x)$ | float32 | From data | 2D measured image (noisy) |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $h$ | constants | $(n_z, n_y, n_x)$ | float32 | Generated | 3D point spread function |
| $\hat{H}$ | constants | $(n_z, n_y, \lfloor n_x/2 \rfloor + 1)$ | complex64 | $\mathcal{F}_{2D}[h]$ | Precomputed 2D FFT of PSF per z-slice |
| $\hat{H}^T$ | constants | $(n_z, n_y, \lfloor n_x/2 \rfloor + 1)$ | complex64 | $\mathcal{F}_{2D}[\text{flip}(h)]$ | Precomputed 2D FFT of flipped PSF |
| $\epsilon$ | constants | scalar | float32 | $10^{-8}$ | Numerical stability constant |
| $K$ | constants | scalar | int | User-specified | Maximum iteration count |

### Derived Quantities (per iteration)
| Symbol | Category | Shape | Dtype | Description |
|--------|----------|-------|-------|-------------|
| $\hat{y}^{(k)}$ | derived | $(n_y, n_x)$ | float32 | Forward projection $Hx^{(k)}$ |
| $r^{(k)}$ | derived | $(n_y, n_x)$ | float32 | Ratio $y / (\hat{y}^{(k)} + \epsilon)$ |
| $u^{(k)}$ | derived | $(n_z, n_y, n_x)$ | float32 | Update factor $H^T r^{(k)}$ |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Coordinate Grid Construction
- **step_order**: 1
- **step_name**: `construct_spatial_grid`
- **formula**:
$$Y_{ij} = i - \lfloor n_y/2 \rfloor, \quad X_{ij} = j - \lfloor n_x/2 \rfloor$$
$$R^2_{ij} = X_{ij}^2 + Y_{ij}^2$$
- **inputs**: $n_y$ (int), $n_x$ (int)
- **outputs**: $R^2 \in \mathbb{R}^{n_y \times n_x}$

### Step 3.2: PSF Generation
- **step_order**: 2
- **step_name**: `generate_psf`
- **formula**:
$$\sigma(z) = \sigma_0 + 0.1 \cdot |z - z_c|$$
$$\tilde{h}_{z,i,j} = \exp\left( -\frac{R^2_{ij}}{2\sigma(z)^2} \right)$$
where $z_c = \lfloor n_z / 2 \rfloor$ and $\sigma_0 = 2.0$.
- **inputs**: $R^2 \in \mathbb{R}^{n_y \times n_x}$, $n_z$ (int), $\sigma_0 = 2.0$
- **outputs**: $\tilde{h} \in \mathbb{R}^{n_z \times n_y \times n_x}$ (unnormalized PSF)

### Step 3.3: PSF Normalization
- **step_order**: 3
- **step_name**: `normalize_psf`
- **formula**:
$$h = \frac{\tilde{h}}{\sum_{z,i,j} \tilde{h}_{z,i,j}}$$
- **inputs**: $\tilde{h} \in \mathbb{R}^{n_z \times n_y \times n_x}$
- **outputs**: $h \in \mathbb{R}^{n_z \times n_y \times n_x}$ with $\sum h = 1$

### Step 3.4: Ground Truth Object Generation
- **step_order**: 4
- **step_name**: `generate_ground_truth`
- **formula**:
$$x^*_{z,i,j} = \begin{cases} 100.0 & \text{if } (z,i,j) \in \mathcal{S} \\ 0 & \text{otherwise} \end{cases}$$
where $\mathcal{S}$ is a set of 10 randomly selected voxel locations with seed 42, constrained by $z \in [0, n_z)$, $i \in [10, n_y-10)$, $j \in [10, n_x-10)$.
- **inputs**: shape $(n_z, n_y, n_x)$, random seed 42
- **outputs**: $x^* \in \mathbb{R}^{n_z \times n_y \times n_x}$ (sparse ground truth)

### Step 3.5: Forward Model Simulation
- **step_order**: 5
- **step_name**: `simulate_measurement`
- **formula**:
$$y_{\text{clean}} = \sum_{z=0}^{n_z-1} \mathcal{F}^{-1}_{2D}\left[ \mathcal{F}_{2D}[x^*_z] \odot \mathcal{F}_{2D}[h_z] \right]$$
where $\odot$ denotes element-wise multiplication.
- **inputs**: $x^* \in \mathbb{R}^{n_z \times n_y \times n_x}$, $h \in \mathbb{R}^{n_z \times n_y \times n_x}$
- **outputs**: $y_{\text{clean}} \in \mathbb{R}^{n_y \times n_x}$

### Step 3.6: Noise Addition
- **step_order**: 6
- **step_name**: `add_noise`
- **formula**:
$$\eta_{ij} \sim \mathcal{N}(0, 0.1^2)$$
$$y = \max(y_{\text{clean}} + \eta, \, 0)$$
- **inputs**: $y_{\text{clean}} \in \mathbb{R}^{n_y \times n_x}$
- **outputs**: $y \in \mathbb{R}^{n_y \times n_x}$ (non-negative noisy measurement)

### Step 3.7: PSF FFT Precomputation
- **step_order**: 7
- **step_name**: `precompute_psf_fft`
- **formula**:
$$\hat{H}_z = \mathcal{F}_{2D}[h_z] \quad \forall z$$
$$\hat{H}^T_z = \mathcal{F}_{2D}[\text{flip}_{2D}(h_z)] \quad \forall z$$
where $\text{flip}_{2D}(h_z)_{i,j} = h_z(n_y - 1 - i, n_x - 1 - j)$ (spatial reversal in both lateral dimensions).
- **inputs**: $h \in \mathbb{R}^{n_z \times n_y \times n_x}$
- **outputs**: $\hat{H} \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$, $\hat{H}^T \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$

---

## 4. ALGORITHM STEPS

### Initialization

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| 0.1 | `compute_mean_measurement` | $\bar{y} = \frac{1}{n_y n_x} \sum_{i,j} y_{ij}$ | $y$ | $\bar{y}$ (scalar) | Scalar reduction |
| 0.2 | `initialize_estimate` | $x^{(0)}_{z,i,j} = \bar{y} \quad \forall z,i,j$ | $\bar{y}$, shape | $x^{(0)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | Uniform initialization |

### Iteration Loop

**BEGIN LOOP** for $k = 0, 1, \ldots, K-1$:

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| 1.1 | `compute_estimate_fft` | $\hat{X}^{(k)}_z = \mathcal{F}_{2D}[x^{(k)}_z] \quad \forall z$ | $x^{(k)}$ | $\hat{X}^{(k)} \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$ | Batched real-to-complex FFT |
| 1.2 | `frequency_domain_convolution` | $\hat{C}^{(k)}_z = \hat{X}^{(k)}_z \odot \hat{H}_z \quad \forall z$ | $\hat{X}^{(k)}$, $\hat{H}$ | $\hat{C}^{(k)} \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$ | Element-wise complex multiply |
| 1.3 | `inverse_fft_layers` | $C^{(k)}_z = \mathcal{F}^{-1}_{2D}[\hat{C}^{(k)}_z] \quad \forall z$ | $\hat{C}^{(k)}$ | $C^{(k)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | Batched complex-to-real IFFT |
| 1.4 | `sum_over_depth` | $\hat{y}^{(k)} = \sum_{z=0}^{n_z-1} C^{(k)}_z$ | $C^{(k)}$ | $\hat{y}^{(k)} \in \mathbb{R}^{n_y \times n_x}$ | Forward projection complete |
| 2.1 | `compute_ratio` | $r^{(k)}_{ij} = \frac{y_{ij}}{\hat{y}^{(k)}_{ij} + \epsilon}$ | $y$, $\hat{y}^{(k)}$, $\epsilon$ | $r^{(k)} \in \mathbb{R}^{n_y \times n_x}$ | $\epsilon = 10^{-8}$ prevents division by zero |
| 3.1 | `compute_ratio_fft` | $\hat{R}^{(k)} = \mathcal{F}_{2D}[r^{(k)}]$ | $r^{(k)}$ | $\hat{R}^{(k)} \in \mathbb{C}^{n_y \times (n_x/2+1)}$ | 2D real FFT |
| 3.2 | `broadcast_ratio_fft` | $\hat{R}^{(k)}_{\text{exp}} = \text{expand}(\hat{R}^{(k)}, \text{axis}=0)$ | $\hat{R}^{(k)}$ | $\hat{R}^{(k)}_{\text{exp}} \in \mathbb{C}^{1 \times n_y \times (n_x/2+1)}$ | Add dimension for broadcasting |
| 3.3 | `frequency_domain_backprojection` | $\hat{U}^{(k)}_z = \hat{R}^{(k)}_{\text{exp}} \odot \hat{H}^T_z \quad \forall z$ | $\hat{R}^{(k)}_{\text{exp}}$, $\hat{H}^T$ | $\hat{U}^{(k)} \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$ | Broadcast multiply |
| 3.4 | `inverse_fft_update` | $u^{(k)}_z = \mathcal{F}^{-1}_{2D}[\hat{U}^{(k)}_z] \quad \forall z$ | $\hat{U}^{(k)}$ | $u^{(k)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | Update factor volume |
| 4.1 | `multiplicative_update` | $\tilde{x}^{(k+1)} = x^{(k)} \odot u^{(k)}$ | $x^{(k)}$, $u^{(k)}$ | $\tilde{x}^{(k+1)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | Element-wise multiply |
| 4.2 | `enforce_nonnegativity` | $x^{(k+1)}_{z,i,j} = \max(\tilde{x}^{(k+1)}_{z,i,j}, 0)$ | $\tilde{x}^{(k+1)}$ | $x^{(k+1)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | ReLU operation |
| 5.1 | `check_termination` | Continue if $k + 1 < K$ | $k$, $K$ | Boolean | Fixed iteration count |

**END LOOP**

### Output
Return $x^{(K)}$ as the final reconstruction.

---

## 5. STOPPING CRITERION

- **type**: `iteration_limit`
- **expression**:
$$k < K$$
- **parameters**:
  - $K$: Maximum number of iterations (default: 30 in main, configurable via `num_iter`)
  - No convergence-based early stopping is implemented

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Mean Squared Error (MSE) | $\text{MSE} = \frac{1}{n_z n_y n_x} \sum_{z,i,j} \left( x^{(K)}_{z,i,j} - x^*_{z,i,j} \right)^2$ | Average squared difference between reconstruction and ground truth |
| Peak Signal-to-Noise Ratio (PSNR) | $\text{PSNR} = 20 \cdot \log_{10}\left( \frac{x^*_{\max}}{\sqrt{\text{MSE}}} \right)$ | Ratio of maximum signal to reconstruction error in decibels, where $x^*_{\max} = \max_{z,i,j} x^*_{z,i,j}$ |
| Maximum Intensity Projection (MIP) | $\text{MIP}_{i,j} = \max_{z} x^{(K)}_{z,i,j}$ | 2D visualization by taking max along depth axis (used for qualitative assessment) |