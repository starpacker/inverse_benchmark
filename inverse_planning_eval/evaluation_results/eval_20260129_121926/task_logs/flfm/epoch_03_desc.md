## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Richardson-Lucy Deconvolution for 3D Fluorescence Microscopy
- **algorithm_family**: Expectation-Maximization / Maximum Likelihood Iterative Reconstruction
- **summary**: Iterative multiplicative algorithm that reconstructs a 3D fluorescent object from a 2D widefield measurement by maximizing the Poisson likelihood through alternating forward projection and back-projection operations.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{x} \in \mathbb{R}^{n_z \times n_y \times n_x}} \quad D_{\text{KL}}(\mathbf{y} \| \mathcal{H}\mathbf{x}) \quad \text{subject to} \quad \mathbf{x} \geq 0$$

where $D_{\text{KL}}$ denotes the Kullback-Leibler divergence (generalized for Poisson statistics).

### data_fidelity_term
$$D_{\text{KL}}(\mathbf{y} \| \mathcal{H}\mathbf{x}) = \sum_{i,j} \left[ y_{ij} \log\left(\frac{y_{ij}}{[\mathcal{H}\mathbf{x}]_{ij}}\right) + [\mathcal{H}\mathbf{x}]_{ij} - y_{ij} \right]$$

where:
- $\mathbf{y} \in \mathbb{R}^{n_y \times n_x}$: observed 2D measurement
- $\mathcal{H}: \mathbb{R}^{n_z \times n_y \times n_x} \to \mathbb{R}^{n_y \times n_x}$: forward operator (sum of 2D convolutions)
- $[\mathcal{H}\mathbf{x}]_{ij} = \sum_{z=0}^{n_z-1} [\mathbf{x}_z * \mathbf{h}_z]_{ij}$

### regularization_term
None (unregularized Richardson-Lucy)

### constraints
| Constraint | Mathematical Form | Description |
|------------|-------------------|-------------|
| Non-negativity | $x_{z,i,j} \geq 0 \quad \forall z,i,j$ | Enforced via ReLU projection after each update |

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{x}^{(k)}$ | primal | $(n_z, n_y, n_x)$ | float32 | $x^{(0)}_{z,i,j} = \bar{y} = \frac{1}{n_y n_x}\sum_{i,j} y_{ij}$ | 3D volume estimate at iteration $k$ |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{p}^{(k)}$ | auxiliary | $(n_y, n_x)$ | float32 | Computed | Forward projection $\mathcal{H}\mathbf{x}^{(k)}$ |
| $\mathbf{r}^{(k)}$ | auxiliary | $(n_y, n_x)$ | float32 | Computed | Ratio $\mathbf{y} / (\mathbf{p}^{(k)} + \epsilon)$ |
| $\mathbf{u}^{(k)}$ | auxiliary | $(n_z, n_y, n_x)$ | float32 | Computed | Back-projected update factor $\mathcal{H}^T \mathbf{r}^{(k)}$ |
| $\hat{\mathbf{x}}_z$ | auxiliary | $(n_y, \lfloor n_x/2 \rfloor + 1)$ | complex64 | Computed | 2D rFFT of $\mathbf{x}_z$ |
| $\hat{\mathbf{r}}$ | auxiliary | $(n_y, \lfloor n_x/2 \rfloor + 1)$ | complex64 | Computed | 2D rFFT of ratio $\mathbf{r}$ |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}$ | observations | $(n_y, n_x)$ | float32 | From simulation | 2D widefield measurement |

### Constants (Operators)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{h}$ | constants | $(n_z, n_y, n_x)$ | float32 | See §3 | 3D point spread function |
| $\hat{\mathbf{h}}$ | constants | $(n_z, n_y, \lfloor n_x/2 \rfloor + 1)$ | complex64 | $\hat{\mathbf{h}} = \text{rFFT2}(\mathbf{h})$ | Precomputed PSF FFT |
| $\hat{\mathbf{h}}^T$ | constants | $(n_z, n_y, \lfloor n_x/2 \rfloor + 1)$ | complex64 | $\hat{\mathbf{h}}^T = \text{rFFT2}(\text{flip}(\mathbf{h}))$ | Precomputed adjoint PSF FFT |
| $\mathbf{g}$ | constants | $(n_z, n_y, n_x)$ | float32 | See §3 | Ground truth (for evaluation only) |

### Numerical Constants
| Symbol | Category | Value | Description |
|--------|----------|-------|-------------|
| $n_z$ | constants | 20 | Number of z-slices |
| $n_y$ | constants | 128 | Image height |
| $n_x$ | constants | 128 | Image width |
| $\sigma_0$ | constants | 2.0 | Base PSF Gaussian sigma (pixels) |
| $\alpha_\sigma$ | constants | 0.1 | PSF sigma growth rate per z-distance |
| $N_{\text{beads}}$ | constants | 10 | Number of point sources in ground truth |
| $I_{\text{bead}}$ | constants | 100.0 | Intensity of each bead |
| $\sigma_{\text{noise}}$ | constants | 0.1 | Additive Gaussian noise standard deviation |
| $\epsilon$ | constants | $10^{-8}$ | Division stabilization constant |
| $K_{\max}$ | constants | 30 | Maximum iterations (main script) |
| seed | constants | 42 | Random seed for reproducibility |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Define grid centers | $z_c = \lfloor n_z/2 \rfloor, \; y_c = \lfloor n_y/2 \rfloor, \; x_c = \lfloor n_x/2 \rfloor$ | $(n_z, n_y, n_x)$ | $(z_c, y_c, x_c) = (10, 64, 64)$ |
| 2 | Construct spatial coordinates | $Y_{ij} = i - y_c, \quad X_{ij} = j - x_c \quad \forall i \in [0, n_y), j \in [0, n_x)$ | $(y_c, x_c)$ | $\mathbf{Y}, \mathbf{X} \in \mathbb{R}^{n_y \times n_x}$ |
| 3 | Compute squared radius | $R^2_{ij} = X_{ij}^2 + Y_{ij}^2$ | $\mathbf{X}, \mathbf{Y}$ | $\mathbf{R}^2 \in \mathbb{R}^{n_y \times n_x}$ |
| 4 | Generate PSF per z-slice | $h_{z,i,j} = \exp\left( -\frac{R^2_{ij}}{2\sigma_z^2} \right)$ where $\sigma_z = \sigma_0 + \alpha_\sigma |z - z_c|$ | $\mathbf{R}^2, \sigma_0=2.0, \alpha_\sigma=0.1$ | $\tilde{\mathbf{h}} \in \mathbb{R}^{n_z \times n_y \times n_x}$ |
| 5 | Normalize PSF | $\mathbf{h} = \tilde{\mathbf{h}} / \sum_{z,i,j} \tilde{h}_{z,i,j}$ | $\tilde{\mathbf{h}}$ | $\mathbf{h} \in \mathbb{R}^{n_z \times n_y \times n_x}$, $\|\mathbf{h}\|_1 = 1$ |
| 6 | Initialize ground truth | $\mathbf{g} = \mathbf{0} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | Shape | Zero tensor |
| 7 | Set random seed | torch.manual_seed(42) | seed=42 | Deterministic RNG |
| 8 | Place point sources | For $b = 1, \ldots, N_{\text{beads}}$: sample $(z_b, y_b, x_b)$ uniformly with $z_b \in [0, n_z)$, $y_b \in [10, n_y-10)$, $x_b \in [10, n_x-10)$; set $g_{z_b, y_b, x_b} = I_{\text{bead}}$ | $N_{\text{beads}}=10$, $I_{\text{bead}}=100.0$ | $\mathbf{g}$ with 10 non-zero entries |
| 9 | Simulate forward model | $\tilde{y}_{ij} = \sum_{z=0}^{n_z-1} [\mathbf{g}_z * \mathbf{h}_z]_{ij} = \sum_{z=0}^{n_z-1} \mathcal{F}^{-1}\left[\mathcal{F}[\mathbf{g}_z] \odot \mathcal{F}[\mathbf{h}_z]\right]_{ij}$ | $\mathbf{g}, \mathbf{h}$ | $\tilde{\mathbf{y}} \in \mathbb{R}^{n_y \times n_x}$ |
| 10 | Add measurement noise | $\mathbf{y} = \max(0, \tilde{\mathbf{y}} + \sigma_{\text{noise}} \cdot \boldsymbol{\eta})$ where $\eta_{ij} \sim \mathcal{N}(0,1)$ | $\tilde{\mathbf{y}}, \sigma_{\text{noise}}=0.1$ | $\mathbf{y} \in \mathbb{R}_{\geq 0}^{n_y \times n_x}$ |
| 11 | Precompute PSF FFT | $\hat{\mathbf{h}}_z = \text{rFFT2}(\mathbf{h}_z) \quad \forall z$ | $\mathbf{h}$ | $\hat{\mathbf{h}} \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$ |
| 12 | Precompute adjoint PSF FFT | $\hat{\mathbf{h}}^T_z = \text{rFFT2}(\text{flip}_{xy}(\mathbf{h}_z)) \quad \forall z$ | $\mathbf{h}$ | $\hat{\mathbf{h}}^T \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$ |

**Note on flip operation**: $[\text{flip}_{xy}(\mathbf{h}_z)]_{i,j} = h_{z, n_y-1-i, n_x-1-j}$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Compute measurement mean | $\bar{y} = \frac{1}{n_y n_x} \sum_{i,j} y_{ij}$ | $\mathbf{y} \in \mathbb{R}^{n_y \times n_x}$ | $\bar{y} \in \mathbb{R}$ |
| I.2 | Initialize estimate | $x^{(0)}_{z,i,j} = \bar{y} \quad \forall z,i,j$ | $\bar{y}$, shape $(n_z, n_y, n_x)$ | $\mathbf{x}^{(0)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ |
| I.3 | Initialize iteration counter | $k \leftarrow 0$ | — | $k = 0$ |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP (k = 0, 1, ..., K_max - 1)
```

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| L.1 | Compute estimate FFT | $\hat{\mathbf{x}}^{(k)}_z = \text{rFFT2}(\mathbf{x}^{(k)}_z) \quad \forall z \in [0, n_z)$ | $\mathbf{x}^{(k)}$ | $\hat{\mathbf{x}}^{(k)} \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$ | Batched over z |
| L.2 | Frequency-domain convolution | $\hat{\mathbf{c}}^{(k)}_z = \hat{\mathbf{x}}^{(k)}_z \odot \hat{\mathbf{h}}_z \quad \forall z$ | $\hat{\mathbf{x}}^{(k)}, \hat{\mathbf{h}}$ | $\hat{\mathbf{c}}^{(k)} \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$ | Element-wise product |
| L.3 | Inverse FFT per layer | $\mathbf{c}^{(k)}_z = \text{irFFT2}(\hat{\mathbf{c}}^{(k)}_z) \quad \forall z$ | $\hat{\mathbf{c}}^{(k)}$ | $\mathbf{c}^{(k)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | Batched over z |
| L.4 | Forward projection (sum) | $p^{(k)}_{ij} = \sum_{z=0}^{n_z-1} c^{(k)}_{z,i,j}$ | $\mathbf{c}^{(k)}$ | $\mathbf{p}^{(k)} \in \mathbb{R}^{n_y \times n_x}$ | $\mathbf{p}^{(k)} = \mathcal{H}\mathbf{x}^{(k)}$ |
| L.5 | Compute ratio | $r^{(k)}_{ij} = \frac{y_{ij}}{p^{(k)}_{ij} + \epsilon}$ | $\mathbf{y}, \mathbf{p}^{(k)}, \epsilon=10^{-8}$ | $\mathbf{r}^{(k)} \in \mathbb{R}^{n_y \times n_x}$ | Stabilized division |
| L.6 | Ratio FFT | $\hat{\mathbf{r}}^{(k)} = \text{rFFT2}(\mathbf{r}^{(k)})$ | $\mathbf{r}^{(k)}$ | $\hat{\mathbf{r}}^{(k)} \in \mathbb{C}^{n_y \times (n_x/2+1)}$ | 2D FFT |
| L.7 | Broadcast and multiply | $\hat{\mathbf{u}}^{(k)}_z = \hat{\mathbf{r}}^{(k)} \odot \hat{\mathbf{h}}^T_z \quad \forall z$ | $\hat{\mathbf{r}}^{(k)}, \hat{\mathbf{h}}^T$ | $\hat{\mathbf{u}}^{(k)} \in \mathbb{C}^{n_z \times n_y \times (n_x/2+1)}$ | Broadcast $(n_y, \cdot) \to (n_z, n_y, \cdot)$ |
| L.8 | Inverse FFT for update | $\mathbf{u}^{(k)}_z = \text{irFFT2}(\hat{\mathbf{u}}^{(k)}_z) \quad \forall z$ | $\hat{\mathbf{u}}^{(k)}$ | $\mathbf{u}^{(k)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | $\mathbf{u}^{(k)} = \mathcal{H}^T \mathbf{r}^{(k)}$ |
| L.9 | Multiplicative update | $\tilde{x}^{(k+1)}_{z,i,j} = x^{(k)}_{z,i,j} \cdot u^{(k)}_{z,i,j}$ | $\mathbf{x}^{(k)}, \mathbf{u}^{(k)}$ | $\tilde{\mathbf{x}}^{(k+1)} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | Element-wise multiplication |
| L.10 | Non-negativity projection | $x^{(k+1)}_{z,i,j} = \max(0, \tilde{x}^{(k+1)}_{z,i,j})$ | $\tilde{\mathbf{x}}^{(k+1)}$ | $\mathbf{x}^{(k+1)} \in \mathbb{R}_{\geq 0}^{n_z \times n_y \times n_x}$ | ReLU operation |
| L.11 | Increment counter | $k \leftarrow k + 1$ | $k$ | $k$ | — |

```
END_LOOP
```

**Forward Operator Definition**:
$$\mathcal{H}: \mathbb{R}^{n_z \times n_y \times n_x} \to \mathbb{R}^{n_y \times n_x}, \quad [\mathcal{H}\mathbf{x}]_{ij} = \sum_{z=0}^{n_z-1} \sum_{m,n} x_{z,m,n} \cdot h_{z, i-m, j-n}$$

**Adjoint Operator Definition**:
$$\mathcal{H}^T: \mathbb{R}^{n_y \times n_x} \to \mathbb{R}^{n_z \times n_y \times n_x}, \quad [\mathcal{H}^T\mathbf{r}]_{z,i,j} = \sum_{m,n} r_{m,n} \cdot h_{z, m-i, n-j}$$

Equivalently: $[\mathcal{H}^T\mathbf{r}]_z = \mathbf{r} \star \mathbf{h}_z = \mathbf{r} * \text{flip}(\mathbf{h}_z)$, where $\star$ denotes cross-correlation.

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Return solution | $\mathbf{x}^* = \mathbf{x}^{(K_{\max})}$ | Final iterate | Reconstructed 3D volume |

---

## 5. STOPPING CRITERION

| Property | Value |
|----------|-------|
| **type** | iteration_limit |
| **expression** | $k < K_{\max}$ |
| **parameters** | $K_{\max} = 30$ (as invoked in main); function default $K_{\max} = 20$ |
| **check_frequency** | Every iteration (loop condition) |

**Note**: No convergence-based early stopping is implemented. The algorithm runs for exactly $K_{\max}$ iterations.

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Mean Squared Error (MSE) | $$\text{MSE} = \frac{1}{n_z \cdot n_y \cdot n_x} \sum_{z,i,j} \left( x^*_{z,i,j} - g_{z,i,j} \right)^2$$ | Average squared difference between reconstruction $\mathbf{x}^*$ and ground truth $\mathbf{g}$ |
| Peak Signal-to-Noise Ratio (PSNR) | $$\text{PSNR} = 20 \cdot \log_{10}\left( \frac{\max_{z,i,j}(g_{z,i,j})}{\sqrt{\text{MSE}}} \right) \text{ [dB]}$$ | Logarithmic ratio of peak ground truth intensity to RMS error; higher is better |

**Visualization outputs** (saved to disk):
- `measurement.png`: Grayscale image of observed 2D measurement $\mathbf{y}$
- `result_comparison.png`: Side-by-side maximum intensity projections (MIP) along z-axis:
  - Left: $\text{MIP}(\mathbf{g}) = \max_z g_{z,i,j}$
  - Right: $\text{MIP}(\mathbf{x}^*) = \max_z x^*_{z,i,j}$ with PSNR annotation