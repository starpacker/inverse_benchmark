## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Richardson-Lucy Deconvolution for 3D Microscopy
- **algorithm_family**: Expectation-Maximization / Multiplicative Update Methods
- **summary**: Iterative maximum-likelihood deconvolution algorithm that reconstructs a 3D fluorescence volume from a 2D widefield measurement using multiplicative updates derived from Poisson noise statistics.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{x} = \arg\max_{x \geq 0} \sum_{i,j} \left[ y_{i,j} \log(\mathbf{H}x)_{i,j} - (\mathbf{H}x)_{i,j} \right]$$

Equivalently, minimize the Kullback-Leibler divergence:
$$\hat{x} = \arg\min_{x \geq 0} D_{KL}(y \| \mathbf{H}x) = \arg\min_{x \geq 0} \sum_{i,j} \left[ y_{i,j} \log\frac{y_{i,j}}{(\mathbf{H}x)_{i,j}} - y_{i,j} + (\mathbf{H}x)_{i,j} \right]$$

### data_fidelity_term
$$\mathcal{D}(x) = -\sum_{i=0}^{N_y-1}\sum_{j=0}^{N_x-1} \left[ y_{i,j} \log\left(\sum_{z=0}^{N_z-1}(x_z * h_z)_{i,j}\right) - \sum_{z=0}^{N_z-1}(x_z * h_z)_{i,j} \right]$$

where $*$ denotes 2D circular convolution.

### regularization_term
None (unregularized Richardson-Lucy)

### constraints
$$\mathcal{I}_{x \geq 0}(x) = \begin{cases} 0 & \text{if } x_{z,i,j} \geq 0 \; \forall z,i,j \\ +\infty & \text{otherwise} \end{cases}$$

Enforced via ReLU projection after each multiplicative update.

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x^{(k)}$ | $(N_z, N_y, N_x)$ | float32 | $x^{(0)}_{z,i,j} = \bar{y} = \frac{1}{N_y N_x}\sum_{i,j} y_{i,j}$ | 3D volume estimate at iteration $k$ |

### 2.2 Dual Variables

None (Richardson-Lucy is a primal-only algorithm)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $p^{(k)}$ | $(N_y, N_x)$ | float32 | Computed | Forward projection $(\mathbf{H}x^{(k)})$ |
| $r^{(k)}$ | $(N_y, N_x)$ | float32 | Computed | Ratio/error image $y / (p^{(k)} + \epsilon)$ |
| $u^{(k)}$ | $(N_z, N_y, N_x)$ | float32 | Computed | Backprojected update factor $\mathbf{H}^T r^{(k)}$ |
| $\hat{x}^{(k)}$ | $(N_z, N_y, \lfloor N_x/2\rfloor + 1)$ | complex64 | Computed | FFT of estimate (rFFT along last two dims) |
| $\hat{r}^{(k)}$ | $(N_y, \lfloor N_x/2\rfloor + 1)$ | complex64 | Computed | FFT of ratio image |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $N_z$ | scalar | int | 20 | Number of z-slices |
| $N_y$ | scalar | int | 128 | Image height |
| $N_x$ | scalar | int | 128 | Image width |
| $K$ | scalar | int | 30 | Maximum iterations |
| $\epsilon$ | scalar | float32 | $10^{-8}$ | Division stabilization constant |
| $\sigma_0$ | scalar | float32 | 2.0 | Base PSF width parameter |
| $\alpha$ | scalar | float32 | 0.1 | PSF width growth rate with defocus |
| $\sigma_{noise}$ | scalar | float32 | 0.1 | Additive noise standard deviation |
| $\hat{h}$ | $(N_z, N_y, \lfloor N_x/2\rfloor + 1)$ | complex64 | Precomputed | FFT of PSF |
| $\hat{h}^T$ | $(N_z, N_y, \lfloor N_x/2\rfloor + 1)$ | complex64 | Precomputed | FFT of spatially-flipped PSF |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y$ | $(N_y, N_x)$ | float32 | 2D widefield measurement image |

---

## 3. PREPROCESSING PIPELINE

### Step P1: Generate Coordinate Grids
- **step_order**: P1
- **step_name**: Coordinate Grid Generation
- **formula**: 
$$i_y = \{0, 1, \ldots, N_y-1\} - \lfloor N_y/2 \rfloor$$
$$i_x = \{0, 1, \ldots, N_x-1\} - \lfloor N_x/2 \rfloor$$
$$Y_{i,j} = (i_y)_i, \quad X_{i,j} = (i_x)_j$$
$$R^2_{i,j} = X_{i,j}^2 + Y_{i,j}^2$$
- **inputs**: $N_y$ (scalar), $N_x$ (scalar)
- **outputs**: $R^2$ (shape: $(N_y, N_x)$)

### Step P2: Generate Depth-Dependent PSF
- **step_order**: P2
- **step_name**: PSF Layer Generation
- **formula**: For each $z \in \{0, 1, \ldots, N_z-1\}$:
$$d_z = |z - \lfloor N_z/2 \rfloor|$$
$$\sigma_z = \sigma_0 + \alpha \cdot d_z = 2.0 + 0.1 \cdot d_z$$
$$h_z[i,j] = \exp\left(-\frac{R^2_{i,j}}{2\sigma_z^2}\right)$$
- **inputs**: $R^2$ (shape: $(N_y, N_x)$), $N_z$ (scalar), $\sigma_0=2.0$, $\alpha=0.1$
- **outputs**: $h_{raw}$ (shape: $(N_z, N_y, N_x)$, unnormalized PSF)

### Step P3: Normalize PSF
- **step_order**: P3
- **step_name**: PSF Energy Normalization
- **formula**: 
$$h_{z,i,j} = \frac{h_{raw,z,i,j}}{\sum_{z'=0}^{N_z-1}\sum_{i'=0}^{N_y-1}\sum_{j'=0}^{N_x-1} h_{raw,z',i',j'}}$$
- **inputs**: $h_{raw}$ (shape: $(N_z, N_y, N_x)$)
- **outputs**: $h$ (shape: $(N_z, N_y, N_x)$, normalized PSF with $\sum h = 1$)

### Step P4: Compute PSF FFT
- **step_order**: P4
- **step_name**: Forward PSF FFT
- **formula**: 
$$\hat{h}[z, m, n] = \text{rFFT2D}(h[z,:,:])$$
$$\hat{h}_{z,m,n} = \sum_{i=0}^{N_y-1}\sum_{j=0}^{N_x-1} h_{z,i,j} \exp\left(-2\pi \mathbf{i}\left(\frac{mi}{N_y} + \frac{nj}{N_x}\right)\right)$$
- **inputs**: $h$ (shape: $(N_z, N_y, N_x)$)
- **outputs**: $\hat{h}$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor + 1)$, complex64)

### Step P5: Compute Adjoint PSF FFT
- **step_order**: P5
- **step_name**: Flipped PSF FFT for Adjoint Operation
- **formula**: 
$$h^{flip}_{z,i,j} = h_{z, (N_y - i) \mod N_y, (N_x - j) \mod N_x}$$
$$\hat{h}^T[z, m, n] = \text{rFFT2D}(h^{flip}[z,:,:])$$
- **inputs**: $h$ (shape: $(N_z, N_y, N_x)$)
- **outputs**: $\hat{h}^T$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor + 1)$, complex64)

### Step P6: Generate Ground Truth Object
- **step_order**: P6
- **step_name**: Sparse Bead Phantom Generation
- **formula**: 
$$x^{gt}_{z,i,j} = \begin{cases} 100.0 & \text{if } (z,i,j) \in \mathcal{B} \\ 0 & \text{otherwise} \end{cases}$$
where $\mathcal{B}$ is a set of 10 random voxel locations with $z \in [0, N_z)$, $i \in [10, N_y-10)$, $j \in [10, N_x-10)$, using random seed 42.
- **inputs**: Shape $(N_z, N_y, N_x)$, seed=42
- **outputs**: $x^{gt}$ (shape: $(N_z, N_y, N_x)$)

### Step P7: Simulate Forward Measurement
- **step_order**: P7
- **step_name**: Forward Convolution Model
- **formula**: 
$$y^{clean}_{i,j} = \sum_{z=0}^{N_z-1} \mathcal{F}^{-1}\left[\mathcal{F}(x^{gt}_z) \odot \mathcal{F}(h_z)\right]_{i,j}$$
where $\mathcal{F}$ denotes 2D rFFT and $\mathcal{F}^{-1}$ denotes 2D irFFT.
- **inputs**: $x^{gt}$ (shape: $(N_z, N_y, N_x)$), $h$ (shape: $(N_z, N_y, N_x)$)
- **outputs**: $y^{clean}$ (shape: $(N_y, N_x)$)

### Step P8: Add Noise and Clip
- **step_order**: P8
- **step_name**: Noise Injection and Non-negativity Clipping
- **formula**: 
$$\eta_{i,j} \sim \mathcal{N}(0, 1)$$
$$y_{i,j} = \max(0, y^{clean}_{i,j} + \sigma_{noise} \cdot \eta_{i,j}) = \max(0, y^{clean}_{i,j} + 0.1 \cdot \eta_{i,j})$$
- **inputs**: $y^{clean}$ (shape: $(N_y, N_x)$), $\sigma_{noise} = 0.1$
- **outputs**: $y$ (shape: $(N_y, N_x)$)

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Compute Measurement Mean | $\bar{y} = \frac{1}{N_y \cdot N_x}\sum_{i=0}^{N_y-1}\sum_{j=0}^{N_x-1} y_{i,j}$ | $y$ (shape: $(N_y, N_x)$) | $\bar{y}$ (scalar) | Global mean for initialization |
| 2 | Initialize Estimate | $x^{(0)}_{z,i,j} = \bar{y} \quad \forall z \in [0, N_z), i \in [0, N_y), j \in [0, N_x)$ | $\bar{y}$ (scalar), shape $(N_z, N_y, N_x)$ | $x^{(0)}$ (shape: $(N_z, N_y, N_x)$) | Uniform initialization |
| 3 | Set Iteration Counter | $k \leftarrow 0$ | — | $k$ (scalar, int) | — |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, K-1$ where $K=30$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 4 | [FORWARD] Compute Estimate FFT | $\hat{x}^{(k)}_{z,m,n} = \sum_{i=0}^{N_y-1}\sum_{j=0}^{N_x-1} x^{(k)}_{z,i,j} \exp\left(-2\pi\mathbf{i}\left(\frac{mi}{N_y} + \frac{nj}{N_x}\right)\right)$ | $x^{(k)}$ (shape: $(N_z, N_y, N_x)$) | $\hat{x}^{(k)}$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor+1)$) | 2D rFFT applied to each z-slice |
| 5 | [FORWARD] Frequency-Domain Multiplication | $\hat{c}^{(k)}_{z,m,n} = \hat{x}^{(k)}_{z,m,n} \cdot \hat{h}_{z,m,n}$ | $\hat{x}^{(k)}$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor+1)$), $\hat{h}$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor+1)$) | $\hat{c}^{(k)}$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor+1)$) | Element-wise complex multiplication |
| 6 | [FORWARD] Inverse FFT per Layer | $c^{(k)}_{z,i,j} = \text{irFFT2D}(\hat{c}^{(k)}_{z,:,:})$ | $\hat{c}^{(k)}$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor+1)$) | $c^{(k)}$ (shape: $(N_z, N_y, N_x)$) | Convolved layers in spatial domain |
| 7 | [FORWARD] Sum Over Z | $p^{(k)}_{i,j} = \sum_{z=0}^{N_z-1} c^{(k)}_{z,i,j}$ | $c^{(k)}$ (shape: $(N_z, N_y, N_x)$) | $p^{(k)}$ (shape: $(N_y, N_x)$) | Forward projection complete: $p^{(k)} = \mathbf{H}x^{(k)}$ |
| 8 | Compute Stabilized Ratio | $r^{(k)}_{i,j} = \frac{y_{i,j}}{p^{(k)}_{i,j} + \epsilon} = \frac{y_{i,j}}{p^{(k)}_{i,j} + 10^{-8}}$ | $y$ (shape: $(N_y, N_x)$), $p^{(k)}$ (shape: $(N_y, N_x)$), $\epsilon=10^{-8}$ | $r^{(k)}$ (shape: $(N_y, N_x)$) | Prevents division by zero |
| 9 | [ADJOINT] Compute Ratio FFT | $\hat{r}^{(k)}_{m,n} = \sum_{i=0}^{N_y-1}\sum_{j=0}^{N_x-1} r^{(k)}_{i,j} \exp\left(-2\pi\mathbf{i}\left(\frac{mi}{N_y} + \frac{nj}{N_x}\right)\right)$ | $r^{(k)}$ (shape: $(N_y, N_x)$) | $\hat{r}^{(k)}$ (shape: $(N_y, \lfloor N_x/2\rfloor+1)$) | 2D rFFT of error ratio |
| 10 | [ADJOINT] Broadcast and Multiply | $\hat{u}^{(k)}_{z,m,n} = \hat{r}^{(k)}_{m,n} \cdot \hat{h}^T_{z,m,n}$ | $\hat{r}^{(k)}$ (shape: $(N_y, \lfloor N_x/2\rfloor+1)$), $\hat{h}^T$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor+1)$) | $\hat{u}^{(k)}$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor+1)$) | Broadcasts 2D ratio to all z-slices |
| 11 | [ADJOINT] Inverse FFT for Update Factor | $u^{(k)}_{z,i,j} = \text{irFFT2D}(\hat{u}^{(k)}_{z,:,:})$ | $\hat{u}^{(k)}$ (shape: $(N_z, N_y, \lfloor N_x/2\rfloor+1)$) | $u^{(k)}$ (shape: $(N_z, N_y, N_x)$) | Backprojection complete: $u^{(k)} = \mathbf{H}^T r^{(k)}$ |
| 12 | Multiplicative Update | $\tilde{x}^{(k+1)}_{z,i,j} = x^{(k)}_{z,i,j} \cdot u^{(k)}_{z,i,j}$ | $x^{(k)}$ (shape: $(N_z, N_y, N_x)$), $u^{(k)}$ (shape: $(N_z, N_y, N_x)$) | $\tilde{x}^{(k+1)}$ (shape: $(N_z, N_y, N_x)$) | Element-wise multiplication |
| 13 | Non-negativity Projection | $x^{(k+1)}_{z,i,j} = \max(0, \tilde{x}^{(k+1)}_{z,i,j})$ | $\tilde{x}^{(k+1)}$ (shape: $(N_z, N_y, N_x)$) | $x^{(k+1)}$ (shape: $(N_z, N_y, N_x)$) | ReLU enforces physical constraint |
| 14 | Increment Counter | $k \leftarrow k + 1$ | $k$ (scalar) | $k$ (scalar) | — |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 15 | Output Reconstruction | $\hat{x} = x^{(K)}$ | $x^{(K)}$ (shape: $(N_z, N_y, N_x)$) | $\hat{x}$ (shape: $(N_z, N_y, N_x)$) | Final reconstructed volume |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k < K$ where $K = 30$
- **parameters**: 
  - $K = 30$ (maximum iterations, from `num_iterations`)
- **check_frequency**: Evaluated at the end of each iteration (step 14)

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Mean Squared Error (MSE) | $\text{MSE} = \frac{1}{N_z \cdot N_y \cdot N_x} \sum_{z=0}^{N_z-1}\sum_{i=0}^{N_y-1}\sum_{j=0}^{N_x-1} \left(\hat{x}_{z,i,j} - x^{gt}_{z,i,j}\right)^2$ | $x^{gt}$ (shape: $(N_z, N_y, N_x)$) | $\hat{x}$ (shape: $(N_z, N_y, N_x)$) | Average squared voxel-wise error |
| Peak Signal-to-Noise Ratio (PSNR) | $\text{PSNR} = 20 \cdot \log_{10}\left(\frac{\max(x^{gt})}{\sqrt{\text{MSE}}}\right) \text{ dB}$ | $x^{gt}$ (shape: $(N_z, N_y, N_x)$) | $\hat{x}$ (shape: $(N_z, N_y, N_x)$) | Logarithmic quality measure; higher is better |