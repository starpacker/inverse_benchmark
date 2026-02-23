## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Richardson-Lucy Deconvolution",
  "algorithm_family": "Expectation-Maximization / Maximum Likelihood",
  "summary": "Iterative multiplicative deconvolution algorithm for 3D fluorescence microscopy reconstruction that maximizes the Poisson likelihood through alternating forward projection, ratio computation, and backward projection steps."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\hat{x} = \arg\max_{x \geq 0} \; p(y | x) = \arg\min_{x \geq 0} \; D_{KL}(y \| Hx)$$

where $D_{KL}$ is the Kullback-Leibler divergence (generalized for continuous case):
$$D_{KL}(y \| Hx) = \sum_{i,j} \left[ (Hx)_{ij} - y_{ij} \log\left((Hx)_{ij}\right) \right]$$

**data_fidelity_term**:
$$\mathcal{L}(x) = \sum_{i=1}^{n_y} \sum_{j=1}^{n_x} \left[ (Hx)_{ij} - y_{ij} \log\left((Hx)_{ij} + \epsilon\right) \right]$$

where the forward operator $H$ is defined as:
$$(Hx)_{ij} = \sum_{z=0}^{n_z-1} \left( x_z * h_z \right)_{ij}$$

with $*$ denoting 2D convolution and $x_z, h_z$ being the $z$-th slices of the 3D volume and PSF respectively.

**regularization_term**: None (unregularized Richardson-Lucy)

**constraints**:
$$\mathcal{I}_{\geq 0}(x) = \begin{cases} 0 & \text{if } x_{z,i,j} \geq 0 \; \forall z,i,j \\ +\infty & \text{otherwise} \end{cases}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x^{(k)}$ | $(n_z, n_y, n_x)$ | float32 | $\bar{y} \cdot \mathbf{1}_{n_z \times n_y \times n_x}$ | 3D volume estimate at iteration $k$ |

### 2.2 Dual Variables

None (Richardson-Lucy is a primal-only algorithm)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\hat{X}^{(k)}$ | $(n_z, n_y, n_x//2+1)$ | complex64 | Computed | FFT of estimate along spatial dims |
| $P^{(k)}$ | $(n_z, n_y, n_x//2+1)$ | complex64 | Computed | Element-wise product $\hat{X}^{(k)} \odot \hat{H}$ |
| $L^{(k)}$ | $(n_z, n_y, n_x)$ | float32 | Computed | Convolved layers before summation |
| $p^{(k)}$ | $(n_y, n_x)$ | float32 | Computed | Forward projection $(Hx^{(k)})$ |
| $r^{(k)}$ | $(n_y, n_x)$ | float32 | Computed | Ratio $y / (p^{(k)} + \epsilon)$ |
| $\hat{R}^{(k)}$ | $(n_y, n_x//2+1)$ | complex64 | Computed | FFT of ratio |
| $\hat{R}_{3D}^{(k)}$ | $(n_z, n_y, n_x//2+1)$ | complex64 | Computed | Broadcasted ratio FFT |
| $Q^{(k)}$ | $(n_z, n_y, n_x//2+1)$ | complex64 | Computed | Product $\hat{R}_{3D}^{(k)} \odot \hat{H}^T$ |
| $u^{(k)}$ | $(n_z, n_y, n_x)$ | float32 | Computed | Update factor (back-projection) |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $n_z$ | scalar | int | 20 | Number of z-slices |
| $n_y$ | scalar | int | 128 | Height in pixels |
| $n_x$ | scalar | int | 128 | Width in pixels |
| $\epsilon$ | scalar | float32 | $10^{-8}$ | Numerical stability constant |
| $K$ | scalar | int | 30 | Maximum iterations |
| $\sigma_0$ | scalar | float32 | 2.0 | Base PSF width |
| $\alpha$ | scalar | float32 | 0.1 | PSF z-broadening rate |
| $\sigma_{noise}$ | scalar | float32 | 0.1 | Additive noise std dev |
| $\hat{H}$ | $(n_z, n_y, n_x//2+1)$ | complex64 | Precomputed | FFT of PSF |
| $\hat{H}^T$ | $(n_z, n_y, n_x//2+1)$ | complex64 | Precomputed | FFT of z-flipped PSF |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y$ | $(n_y, n_x)$ | float32 | 2D measured image (non-negative) |
| $h$ | $(n_z, n_y, n_x)$ | float32 | 3D point spread function |

---

## 3. PREPROCESSING PIPELINE

### Step P1: Coordinate Grid Generation
- **step_order**: P1
- **step_name**: Generate spatial coordinate grids
- **formula**: 
  $$Y_{ij} = i - \lfloor n_y/2 \rfloor, \quad X_{ij} = j - \lfloor n_x/2 \rfloor$$
  $$R^2_{ij} = X_{ij}^2 + Y_{ij}^2$$
- **inputs**: $n_y$ (scalar), $n_x$ (scalar)
- **outputs**: $R^2$ (shape: $(n_y, n_x)$, dtype: float32)
- **assumptions**: Origin centered at $(n_y//2, n_x//2)$

### Step P2: PSF Layer Generation
- **step_order**: P2
- **step_name**: Generate depth-dependent Gaussian PSF
- **formula**: For each $z \in \{0, 1, \ldots, n_z-1\}$:
  $$\sigma_z = \sigma_0 + \alpha \cdot |z - \lfloor n_z/2 \rfloor|$$
  $$\tilde{h}_{z,i,j} = \exp\left( -\frac{R^2_{ij}}{2\sigma_z^2} \right)$$
- **inputs**: $R^2$ (shape: $(n_y, n_x)$), $\sigma_0 = 2.0$, $\alpha = 0.1$, $n_z = 20$
- **outputs**: $\tilde{h}$ (shape: $(n_z, n_y, n_x)$, dtype: float32)
- **assumptions**: Gaussian approximation to optical PSF

### Step P3: PSF Normalization
- **step_order**: P3
- **step_name**: Normalize PSF to unit sum
- **formula**: 
  $$h = \frac{\tilde{h}}{\sum_{z,i,j} \tilde{h}_{z,i,j}}$$
- **inputs**: $\tilde{h}$ (shape: $(n_z, n_y, n_x)$)
- **outputs**: $h$ (shape: $(n_z, n_y, n_x)$, dtype: float32)
- **assumptions**: Energy conservation in imaging system

### Step P4: Ground Truth Object Generation
- **step_order**: P4
- **step_name**: Generate sparse bead phantom
- **formula**: Initialize $o^{GT} = \mathbf{0}_{n_z \times n_y \times n_x}$. For $b = 1, \ldots, 10$:
  $$z_b \sim \mathcal{U}\{0, n_z-1\}, \; i_b \sim \mathcal{U}\{10, n_y-11\}, \; j_b \sim \mathcal{U}\{10, n_x-11\}$$
  $$o^{GT}_{z_b, i_b, j_b} = 100.0$$
- **inputs**: Shape $(n_z, n_y, n_x)$, random seed = 42
- **outputs**: $o^{GT}$ (shape: $(n_z, n_y, n_x)$, dtype: float32)
- **assumptions**: Sparse point sources with fixed intensity

### Step P5: Forward Model Simulation
- **step_order**: P5
- **step_name**: Simulate measurement via layer-wise convolution
- **formula**: 
  $$\tilde{y} = \sum_{z=0}^{n_z-1} \mathcal{F}^{-1}\left[ \mathcal{F}[o^{GT}_z] \odot \mathcal{F}[h_z] \right]$$
  where $\mathcal{F}$ denotes 2D real FFT (rfft2) and $\mathcal{F}^{-1}$ denotes inverse.
- **inputs**: $o^{GT}$ (shape: $(n_z, n_y, n_x)$), $h$ (shape: $(n_z, n_y, n_x)$)
- **outputs**: $\tilde{y}$ (shape: $(n_y, n_x)$, dtype: float32)
- **assumptions**: Incoherent imaging, linear shift-invariant per z-slice

### Step P6: Noise Addition
- **step_order**: P6
- **step_name**: Add Gaussian noise and enforce non-negativity
- **formula**: 
  $$\eta_{ij} \sim \mathcal{N}(0, \sigma_{noise}^2), \quad \sigma_{noise} = 0.1$$
  $$y_{ij} = \max(0, \tilde{y}_{ij} + \eta_{ij})$$
- **inputs**: $\tilde{y}$ (shape: $(n_y, n_x)$), $\sigma_{noise} = 0.1$
- **outputs**: $y$ (shape: $(n_y, n_x)$, dtype: float32)
- **assumptions**: Additive white Gaussian noise approximation

### Step P7: PSF FFT Precomputation
- **step_order**: P7
- **step_name**: Precompute PSF Fourier transforms
- **formula**: 
  $$\hat{H}_z = \mathcal{F}[h_z] \quad \text{for each } z$$
  $$\hat{H}^T_z = \mathcal{F}[\text{flip}_{xy}(h_z)] \quad \text{for each } z$$
  where $\text{flip}_{xy}$ reverses both spatial dimensions.
- **inputs**: $h$ (shape: $(n_z, n_y, n_x)$)
- **outputs**: $\hat{H}$ (shape: $(n_z, n_y, n_x//2+1)$), $\hat{H}^T$ (shape: $(n_z, n_y, n_x//2+1)$), both complex64
- **assumptions**: Correlation implemented as convolution with flipped kernel

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| I1 | $\bar{y}$ | $\bar{y} = \frac{1}{n_y n_x}\sum_{i,j} y_{ij}$ | scalar | float32 | Computed from observation |
| I2 | $x^{(0)}$ | $x^{(0)}_{z,i,j} = \bar{y} \quad \forall z,i,j$ | $(n_z, n_y, n_x)$ | float32 | Uniform initialization |
| I3 | $k$ | $k = 0$ | scalar | int | Loop counter |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP: for k = 0 to K-1
```

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | [FORWARD] Compute estimate FFT | $\hat{X}^{(k)}_z = \mathcal{F}[x^{(k)}_z]$ for all $z$ simultaneously | $x^{(k)}$ ($(n_z, n_y, n_x)$) | $\hat{X}^{(k)}$ ($(n_z, n_y, n_x//2+1)$, complex64) | torch.fft.rfft2 along dims (-2,-1) |
| 2 | [FORWARD] Frequency-domain convolution | $P^{(k)}_{z,i,j} = \hat{X}^{(k)}_{z,i,j} \cdot \hat{H}_{z,i,j}$ | $\hat{X}^{(k)}$ ($(n_z, n_y, n_x//2+1)$), $\hat{H}$ ($(n_z, n_y, n_x//2+1)$) | $P^{(k)}$ ($(n_z, n_y, n_x//2+1)$, complex64) | Element-wise complex multiplication |
| 3 | [FORWARD] Inverse FFT to spatial domain | $L^{(k)}_z = \mathcal{F}^{-1}[P^{(k)}_z]$ for all $z$ | $P^{(k)}$ ($(n_z, n_y, n_x//2+1)$) | $L^{(k)}$ ($(n_z, n_y, n_x)$, float32) | torch.fft.irfft2 with output size $(n_y, n_x)$ |
| 4 | [FORWARD] Sum over z-dimension | $p^{(k)}_{i,j} = \sum_{z=0}^{n_z-1} L^{(k)}_{z,i,j}$ | $L^{(k)}$ ($(n_z, n_y, n_x)$) | $p^{(k)}$ ($(n_y, n_x)$, float32) | Forward projection complete |
| 5 | Compute stabilized ratio | $r^{(k)}_{i,j} = \frac{y_{i,j}}{p^{(k)}_{i,j} + \epsilon}$ | $y$ ($(n_y, n_x)$), $p^{(k)}$ ($(n_y, n_x)$), $\epsilon = 10^{-8}$ | $r^{(k)}$ ($(n_y, n_x)$, float32) | Division with numerical stabilization |
| 6 | [ADJOINT] Compute ratio FFT | $\hat{R}^{(k)} = \mathcal{F}[r^{(k)}]$ | $r^{(k)}$ ($(n_y, n_x)$) | $\hat{R}^{(k)}$ ($(n_y, n_x//2+1)$, complex64) | 2D rfft2 |
| 7 | [ADJOINT] Broadcast to 3D | $\hat{R}^{(k)}_{3D} = \text{unsqueeze}_0(\hat{R}^{(k)})$ | $\hat{R}^{(k)}$ ($(n_y, n_x//2+1)$) | $\hat{R}^{(k)}_{3D}$ ($(1, n_y, n_x//2+1)$, complex64) | Add singleton z-dimension for broadcasting |
| 8 | [ADJOINT] Frequency-domain correlation | $Q^{(k)}_{z,i,j} = \hat{R}^{(k)}_{3D,0,i,j} \cdot \hat{H}^T_{z,i,j}$ | $\hat{R}^{(k)}_{3D}$ ($(1, n_y, n_x//2+1)$), $\hat{H}^T$ ($(n_z, n_y, n_x//2+1)$) | $Q^{(k)}$ ($(n_z, n_y, n_x//2+1)$, complex64) | Broadcasting over z; implements $\text{Corr}(r, h_z) = \text{Conv}(r, \text{flip}(h_z))$ |
| 9 | [ADJOINT] Inverse FFT for update factor | $u^{(k)}_z = \mathcal{F}^{-1}[Q^{(k)}_z]$ for all $z$ | $Q^{(k)}$ ($(n_z, n_y, n_x//2+1)$) | $u^{(k)}$ ($(n_z, n_y, n_x)$, float32) | torch.fft.irfft2 with output size $(n_y, n_x)$ |
| 10 | Multiplicative update | $\tilde{x}^{(k+1)}_{z,i,j} = x^{(k)}_{z,i,j} \cdot u^{(k)}_{z,i,j}$ | $x^{(k)}$ ($(n_z, n_y, n_x)$), $u^{(k)}$ ($(n_z, n_y, n_x)$) | $\tilde{x}^{(k+1)}$ ($(n_z, n_y, n_x)$, float32) | Element-wise multiplication |
| 11 | Non-negativity projection | $x^{(k+1)}_{z,i,j} = \max(0, \tilde{x}^{(k+1)}_{z,i,j})$ | $\tilde{x}^{(k+1)}$ ($(n_z, n_y, n_x)$) | $x^{(k+1)}$ ($(n_z, n_y, n_x)$, float32) | torch.relu; enforces constraint $x \geq 0$ |
| 12 | Increment counter | $k \leftarrow k + 1$ | $k$ (scalar) | $k$ (scalar) | Loop bookkeeping |

```
END_LOOP
```

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| F1 | Extract final estimate | $\hat{x} = x^{(K)}$ | $x^{(K)}$ ($(n_z, n_y, n_x)$) | $\hat{x}$ ($(n_z, n_y, n_x)$, float32) | Copy final iterate as reconstruction |

---

## 5. STOPPING CRITERION

```json
{
  "type": "iteration_limit",
  "expression": "k \\geq K",
  "parameters": {
    "K": 30,
    "description": "Fixed number of iterations"
  },
  "check_frequency": "After each complete iteration (steps 1-12)"
}
```

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| MSE | $\text{MSE} = \frac{1}{n_z n_y n_x} \sum_{z=0}^{n_z-1} \sum_{i=0}^{n_y-1} \sum_{j=0}^{n_x-1} \left( \hat{x}_{z,i,j} - o^{GT}_{z,i,j} \right)^2$ | $o^{GT}$ ($(n_z, n_y, n_x)$) | $\hat{x}$ ($(n_z, n_y, n_x)$) | Mean squared error over entire 3D volume |
| PSNR | $\text{PSNR} = 20 \cdot \log_{10}\left( \frac{\max_{z,i,j}(o^{GT}_{z,i,j})}{\sqrt{\text{MSE}}} \right)$ | $o^{GT}$ ($(n_z, n_y, n_x)$) | $\hat{x}$ ($(n_z, n_y, n_x)$) | Peak signal-to-noise ratio in decibels |