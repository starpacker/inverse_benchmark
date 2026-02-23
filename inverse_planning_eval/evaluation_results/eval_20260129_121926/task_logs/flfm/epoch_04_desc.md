## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Richardson-Lucy Deconvolution for 3D Widefield Microscopy
- **algorithm_family**: Expectation-Maximization / Maximum Likelihood Estimation
- **summary**: Iterative multiplicative deconvolution algorithm that reconstructs a 3D fluorescent volume from a single 2D widefield microscopy measurement using slice-wise convolution with a depth-varying point spread function.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\hat{\mathbf{x}} = \arg\max_{\mathbf{x} \geq 0} \sum_{i,j} \left[ y_{ij} \log\left( (H\mathbf{x})_{ij} \right) - (H\mathbf{x})_{ij} \right]$$

Equivalently, the algorithm minimizes the Kullback-Leibler divergence (I-divergence):
$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x} \geq 0} D_{KL}(\mathbf{y} \| H\mathbf{x}) = \arg\min_{\mathbf{x} \geq 0} \sum_{i,j} \left[ y_{ij} \log\frac{y_{ij}}{(H\mathbf{x})_{ij}} - y_{ij} + (H\mathbf{x})_{ij} \right]$$

### data_fidelity_term
**Poisson Log-Likelihood / KL-Divergence**:
$$\mathcal{L}(\mathbf{x}) = D_{KL}(\mathbf{y} \| H\mathbf{x}) = \sum_{i=0}^{N_y-1} \sum_{j=0}^{N_x-1} \left[ y_{ij} \log\frac{y_{ij}}{(H\mathbf{x})_{ij}} - y_{ij} + (H\mathbf{x})_{ij} \right]$$

where $(H\mathbf{x})_{ij} = \sum_{z=0}^{N_z-1} (x_z * h_z)_{ij}$ is the forward projection.

### regularization_term
**None** — The algorithm is unregularized Richardson-Lucy.

### constraints
$$\mathbf{x} \in \mathcal{C} = \{\mathbf{x} \in \mathbb{R}^{N_z \times N_y \times N_x} : x_{z,i,j} \geq 0 \;\forall\; z,i,j\}$$

Enforced via projection: $\mathbf{x} \leftarrow \max(\mathbf{x}, 0)$ (ReLU operation).

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}^{(k)}$ | $(N_z, N_y, N_x)$ | float32 | $x^{(0)}_{z,i,j} = \bar{y} = \frac{1}{N_y N_x}\sum_{i,j} y_{ij}$ | 3D volume estimate at iteration $k$ |

### 2.2 Dual Variables

None — Richardson-Lucy is a primal-only algorithm.

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{p}^{(k)}$ | $(N_y, N_x)$ | float32 | Computed | Forward projection $H\mathbf{x}^{(k)}$ |
| $\mathbf{r}^{(k)}$ | $(N_y, N_x)$ | float32 | Computed | Ratio image $y_{ij} / (p^{(k)}_{ij} + \epsilon)$ |
| $\mathbf{u}^{(k)}$ | $(N_z, N_y, N_x)$ | float32 | Computed | Multiplicative update factor $H^T \mathbf{r}^{(k)}$ |
| $\hat{\mathbf{x}}_z$ | $(N_y, \lfloor N_x/2 \rfloor + 1)$ | complex64 | Computed | FFT of estimate slice $z$ |
| $\hat{\mathbf{r}}$ | $(N_y, \lfloor N_x/2 \rfloor + 1)$ | complex64 | Computed | FFT of ratio image |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_z$ | 20 | Number of z-slices |
| $N_y$ | 128 | Image height |
| $N_x$ | 128 | Image width |
| $K$ | 30 | Maximum iterations (num_iterations) |
| $\epsilon$ | $10^{-8}$ | Numerical stability constant for division |
| $\sigma_0$ | 2.0 | Base PSF Gaussian width |
| $\alpha$ | 0.1 | PSF width growth rate per z-distance |
| $N_{\text{beads}}$ | 10 | Number of point sources in ground truth |
| $I_{\text{bead}}$ | 100.0 | Intensity of each bead |
| $\sigma_{\text{noise}}$ | 0.1 | Noise standard deviation |
| seed | 42 | Random seed for reproducibility |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(N_y, N_x)$ | float32 | 2D widefield measurement image |
| $\hat{\mathbf{h}}$ | $(N_z, N_y, \lfloor N_x/2 \rfloor + 1)$ | complex64 | Precomputed FFT of PSF |
| $\hat{\tilde{\mathbf{h}}}$ | $(N_z, N_y, \lfloor N_x/2 \rfloor + 1)$ | complex64 | Precomputed FFT of flipped PSF |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Define Coordinate System
**Formula**:
$$z_c = \lfloor N_z / 2 \rfloor, \quad y_c = \lfloor N_y / 2 \rfloor, \quad x_c = \lfloor N_x / 2 \rfloor$$
$$Y_{i,j} = i - y_c, \quad X_{i,j} = j - x_c, \quad R^2_{i,j} = X_{i,j}^2 + Y_{i,j}^2$$
**Inputs**: $N_z, N_y, N_x$
**Outputs**: $z_c=10$, $y_c=64$, $x_c=64$, $R^2 \in \mathbb{R}^{N_y \times N_x}$

### Step 1: Generate Depth-Varying PSF
**Formula**:
$$h_{z,i,j} = \exp\left( -\frac{R^2_{i,j}}{2\sigma_z^2} \right), \quad \sigma_z = \sigma_0 + \alpha |z - z_c|$$
**Inputs**: $R^2$ (shape $(N_y, N_x)$), $\sigma_0=2.0$, $\alpha=0.1$, $z_c=10$
**Outputs**: Unnormalized PSF $\tilde{\mathbf{h}} \in \mathbb{R}^{N_z \times N_y \times N_x}$

### Step 2: Normalize PSF
**Formula**:
$$h_{z,i,j} = \frac{\tilde{h}_{z,i,j}}{\sum_{z'}\sum_{i'}\sum_{j'} \tilde{h}_{z',i',j'}}$$
**Inputs**: $\tilde{\mathbf{h}}$ (shape $(N_z, N_y, N_x)$)
**Outputs**: Normalized PSF $\mathbf{h}$ with $\sum_{z,i,j} h_{z,i,j} = 1$

### Step 3: Generate Ground Truth Object
**Formula**:
$$x^{\text{gt}}_{z,i,j} = \sum_{b=1}^{N_{\text{beads}}} I_{\text{bead}} \cdot \delta(z - z_b)\delta(i - i_b)\delta(j - j_b)$$
where $(z_b, i_b, j_b)$ are uniformly sampled with $z_b \in [0, N_z)$, $i_b \in [10, N_y-10)$, $j_b \in [10, N_x-10)$.
**Inputs**: seed=42, $N_{\text{beads}}=10$, $I_{\text{bead}}=100.0$
**Outputs**: $\mathbf{x}^{\text{gt}} \in \mathbb{R}^{N_z \times N_y \times N_x}$

### Step 4: Simulate Forward Measurement
**Formula**:
$$\tilde{y}_{i,j} = \sum_{z=0}^{N_z-1} \mathcal{F}^{-1}\left[ \mathcal{F}[x^{\text{gt}}_z] \odot \mathcal{F}[h_z] \right]_{i,j}$$
**Inputs**: $\mathbf{x}^{\text{gt}}$, $\mathbf{h}$
**Outputs**: Clean measurement $\tilde{\mathbf{y}} \in \mathbb{R}^{N_y \times N_x}$

### Step 5: Add Noise and Apply Non-negativity
**Formula**:
$$y_{i,j} = \max\left(0, \tilde{y}_{i,j} + \sigma_{\text{noise}} \cdot n_{i,j}\right), \quad n_{i,j} \sim \mathcal{N}(0,1)$$
**Inputs**: $\tilde{\mathbf{y}}$, $\sigma_{\text{noise}}=0.1$
**Outputs**: Noisy measurement $\mathbf{y} \in \mathbb{R}_{\geq 0}^{N_y \times N_x}$

### Step 6: Precompute PSF FFTs
**Formula**:
$$\hat{h}_{z} = \mathcal{F}_{\text{2D}}[h_z], \quad \hat{\tilde{h}}_{z} = \mathcal{F}_{\text{2D}}[\text{flip}(h_z)]$$
where $\text{flip}(h_z)_{i,j} = h_{z, N_y-1-i, N_x-1-j}$.
**Inputs**: $\mathbf{h}$ (shape $(N_z, N_y, N_x)$)
**Outputs**: $\hat{\mathbf{h}}, \hat{\tilde{\mathbf{h}}} \in \mathbb{C}^{N_z \times N_y \times (N_x/2+1)}$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Compute Mean Intensity | $\bar{y} = \frac{1}{N_y N_x} \sum_{i,j} y_{i,j}$ | $\mathbf{y}$ | $\bar{y} \in \mathbb{R}$ |
| I.2 | Initialize Estimate | $x^{(0)}_{z,i,j} = \bar{y} \quad \forall z,i,j$ | $\bar{y}$, shape | $\mathbf{x}^{(0)} \in \mathbb{R}^{N_z \times N_y \times N_x}$ |
| I.3 | Set Iteration Counter | $k \leftarrow 0$ | — | $k=0$ |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP: for k = 0 to K-1
```

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | **[FORWARD] Compute FFT of Estimate** | $\hat{x}^{(k)}_z = \mathcal{F}_{\text{2D}}[x^{(k)}_z]$ for all $z$ | $\mathbf{x}^{(k)}$ | $\hat{\mathbf{x}}^{(k)} \in \mathbb{C}^{N_z \times N_y \times (N_x/2+1)}$ | rfft2 along spatial dims |
| L.2 | **[FORWARD] Frequency-Domain Product** | $\hat{c}^{(k)}_z = \hat{x}^{(k)}_z \odot \hat{h}_z$ | $\hat{\mathbf{x}}^{(k)}$, $\hat{\mathbf{h}}$ | $\hat{\mathbf{c}}^{(k)}$ | Element-wise complex mult |
| L.3 | **[FORWARD] Inverse FFT** | $c^{(k)}_z = \mathcal{F}^{-1}_{\text{2D}}[\hat{c}^{(k)}_z]$ | $\hat{\mathbf{c}}^{(k)}$ | $\mathbf{c}^{(k)} \in \mathbb{R}^{N_z \times N_y \times N_x}$ | irfft2 |
| L.4 | **[FORWARD] Sum Over Z** | $p^{(k)}_{i,j} = \sum_{z=0}^{N_z-1} c^{(k)}_{z,i,j}$ | $\mathbf{c}^{(k)}$ | $\mathbf{p}^{(k)} \in \mathbb{R}^{N_y \times N_x}$ | Forward projection complete |
| L.5 | Compute Ratio | $r^{(k)}_{i,j} = \frac{y_{i,j}}{p^{(k)}_{i,j} + \epsilon}$ | $\mathbf{y}$, $\mathbf{p}^{(k)}$, $\epsilon=10^{-8}$ | $\mathbf{r}^{(k)} \in \mathbb{R}^{N_y \times N_x}$ | Stabilized division |
| L.6 | **[ADJOINT] Compute FFT of Ratio** | $\hat{r}^{(k)} = \mathcal{F}_{\text{2D}}[\mathbf{r}^{(k)}]$ | $\mathbf{r}^{(k)}$ | $\hat{r}^{(k)} \in \mathbb{C}^{N_y \times (N_x/2+1)}$ | Broadcast for z |
| L.7 | **[ADJOINT] Frequency-Domain Correlation** | $\hat{u}^{(k)}_z = \hat{r}^{(k)} \odot \hat{\tilde{h}}_z$ | $\hat{r}^{(k)}$, $\hat{\tilde{\mathbf{h}}}$ | $\hat{\mathbf{u}}^{(k)} \in \mathbb{C}^{N_z \times N_y \times (N_x/2+1)}$ | Correlation via flipped PSF |
| L.8 | **[ADJOINT] Inverse FFT** | $u^{(k)}_z = \mathcal{F}^{-1}_{\text{2D}}[\hat{u}^{(k)}_z]$ | $\hat{\mathbf{u}}^{(k)}$ | $\mathbf{u}^{(k)} \in \mathbb{R}^{N_z \times N_y \times N_x}$ | Adjoint complete |
| L.9 | Multiplicative Update | $\tilde{x}^{(k+1)}_{z,i,j} = x^{(k)}_{z,i,j} \cdot u^{(k)}_{z,i,j}$ | $\mathbf{x}^{(k)}$, $\mathbf{u}^{(k)}$ | $\tilde{\mathbf{x}}^{(k+1)}$ | RL update rule |
| L.10 | Enforce Non-negativity | $x^{(k+1)}_{z,i,j} = \max(0, \tilde{x}^{(k+1)}_{z,i,j})$ | $\tilde{\mathbf{x}}^{(k+1)}$ | $\mathbf{x}^{(k+1)}$ | Projection onto $\mathbb{R}_{\geq 0}$ |
| L.11 | Increment Counter | $k \leftarrow k + 1$ | $k$ | $k$ | — |

```
END_LOOP
```

**Computational Note**: Steps L.1–L.4 implement the forward operator $H: \mathbb{R}^{N_z \times N_y \times N_x} \to \mathbb{R}^{N_y \times N_x}$. Steps L.6–L.8 implement the adjoint operator $H^T: \mathbb{R}^{N_y \times N_x} \to \mathbb{R}^{N_z \times N_y \times N_x}$. The correlation is computed as convolution with the spatially-flipped PSF: $\text{Corr}(r, h_z) = \text{Conv}(r, \text{flip}(h_z))$.

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Return Estimate | $\hat{\mathbf{x}} = \mathbf{x}^{(K)}$ | $\mathbf{x}^{(K)}$ | Reconstructed volume $\hat{\mathbf{x}} \in \mathbb{R}^{N_z \times N_y \times N_x}$ |
| F.2 | Compute MSE | $\text{MSE} = \frac{1}{N_z N_y N_x} \sum_{z,i,j} (\hat{x}_{z,i,j} - x^{\text{gt}}_{z,i,j})^2$ | $\hat{\mathbf{x}}$, $\mathbf{x}^{\text{gt}}$ | $\text{MSE} \in \mathbb{R}$ |
| F.3 | Compute PSNR | $\text{PSNR} = 20 \log_{10}\left( \frac{\max(\mathbf{x}^{\text{gt}})}{\sqrt{\text{MSE}}} \right)$ | $\text{MSE}$, $\mathbf{x}^{\text{gt}}$ | $\text{PSNR} \in \mathbb{R}$ (dB) |
| F.4 | Generate MIP | $\text{MIP}_{i,j} = \max_z \hat{x}_{z,i,j}$ | $\hat{\mathbf{x}}$ | Maximum intensity projection |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k \geq K$ where $K = 30$
- **parameters**:
  - `num_iter` ($K$): 30 (from `num_iterations` in main)
  - No convergence threshold is implemented
- **check_frequency**: Evaluated at the end of each iteration

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Mean Squared Error (MSE) | $\text{MSE} = \frac{1}{N_z N_y N_x} \sum_{z=0}^{N_z-1} \sum_{i=0}^{N_y-1} \sum_{j=0}^{N_x-1} \left( \hat{x}_{z,i,j} - x^{\text{gt}}_{z,i,j} \right)^2$ | Average squared reconstruction error |
| Peak Signal-to-Noise Ratio (PSNR) | $\text{PSNR} = 20 \log_{10}\left( \frac{x^{\text{gt}}_{\max}}{\sqrt{\text{MSE}}} \right) = 10 \log_{10}\left( \frac{(x^{\text{gt}}_{\max})^2}{\text{MSE}} \right)$ | Logarithmic ratio of peak signal to RMS error, in decibels |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON MUST include all of the following fields:

```
algorithm_identification:
  - algorithm_name
  - algorithm_family
  - summary

objective_function:
  - full_expression
  - data_fidelity_term
  - regularization_term
  - constraints

variables:
  - primal_variables (list with symbol, shape, dtype, initialization, description)
  - dual_variables
  - auxiliary_variables
  - constants (list with symbol, value, description)
  - observations

data_preprocessing:
  - steps (list with step_order, step_name, formula, inputs, outputs)

initialization:
  - steps (list with step_order, step_name, formula, inputs, outputs)

iteration_loop:
  - begin_condition
  - steps (list with step_order, step_name, formula, inputs, outputs, operator_type)
  - end_condition

finalization:
  - steps (list with step_order, step_name, formula, inputs, outputs)

stopping_criterion:
  - type
  - expression
  - parameters
  - check_frequency

evaluation:
  - metrics (list with name, formula, description)
```