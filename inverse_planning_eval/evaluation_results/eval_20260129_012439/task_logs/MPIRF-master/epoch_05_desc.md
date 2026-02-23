## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Regularized Kaczmarz Algorithm (Algebraic Reconstruction Technique)
- **algorithm_family**: Iterative Row-Action Methods / Algebraic Reconstruction Techniques
- **summary**: Reconstructs Magnetic Particle Imaging (MPI) concentration distributions from frequency-domain measurements by iteratively projecting onto hyperplanes defined by system matrix rows with Tikhonov regularization.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{x} \in \mathbb{C}^N} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + \lambda \|\mathbf{x}\|_2^2$$

Equivalently, via augmented system formulation:
$$\min_{\mathbf{x}, \mathbf{r}} \left\| \begin{bmatrix} \mathbf{A} \\ \sqrt{\lambda}\mathbf{I} \end{bmatrix} \mathbf{x} - \begin{bmatrix} \mathbf{b} \\ \mathbf{0} \end{bmatrix} + \begin{bmatrix} \sqrt{\lambda}\mathbf{r} \\ \mathbf{0} \end{bmatrix} \right\|_2^2$$

### data_fidelity_term
$$\mathcal{D}(\mathbf{x}) = \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 = \sum_{m=0}^{M-1} \left| \sum_{n=0}^{N-1} A_{m,n} x_n - b_m \right|^2$$

### regularization_term
$$\mathcal{R}(\mathbf{x}) = \lambda \|\mathbf{x}\|_2^2 = \lambda \sum_{n=0}^{N-1} |x_n|^2$$

### constraints
$$\mathbf{x} \in \mathbb{C}^N \text{ (unconstrained during optimization; real part extracted post-convergence)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}$ | $(N,)$ where $N = x_n \times y_n$ | `complex128` | $\mathbf{x}^{(0)} = \mathbf{0}$ | Reconstructed concentration in vectorized pixel space |

### 2.2 Dual Variables

None (Kaczmarz is a primal method)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{r}$ | $(M,)$ | `complex128` | $\mathbf{r}^{(0)} = \mathbf{0}$ | Regularization residual vector tracking cumulative corrections |
| $\mathbf{e}$ | $(M,)$ | `float64` | Computed from $\mathbf{A}$ | Row energy: $e_m = \|\mathbf{a}_m\|_2$ |
| $\beta$ | scalar | `complex128` | N/A | Step size for current row projection |
| $k$ | scalar | `int` | N/A | Current row index |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda$ | `1e-6` (default) or `0` (main execution) | Tikhonov regularization parameter |
| $L$ | `20` (default) or `50` (main execution) | Maximum number of outer iterations |
| $M$ | `A.shape[0]` | Number of frequency components (equations) |
| $N$ | `A.shape[1]` = $x_n \times y_n$ | Number of pixels (unknowns) |
| $\pi$ | `3.1416` | Mathematical constant |
| $k_B$ | `1.3806488e-23` J/K | Boltzmann constant |
| $T_{abs}$ | `273.15` K | Absolute zero offset |
| $\mu_0$ | $4\pi \times 10^{-7}$ H/m | Vacuum permeability |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{A}$ | $(M, N)$ | `complex128` | System matrix (frequency domain) |
| $\mathbf{b}$ | $(M,)$ | `complex128` | Frequency-domain voltage measurements |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Physical Parameter Computation
**step_order**: 3.1  
**step_name**: Compute magnetic particle parameters  
**formula**:
$$T_t = T_{temp} + 273.15$$
$$V = \frac{\pi d^3}{6}$$
$$m_m = M_s \cdot V$$
$$b_{coeff} = \frac{\mu_0 \cdot m_m}{k_B \cdot T_t}$$
**inputs**: $T_{temp}=20.0$ K, $d=30\times10^{-9}$ m, $M_s=8\times10^5$ A/m  
**outputs**: $T_t$ (scalar), $V$ (scalar), $m_m$ (scalar), $b_{coeff}$ (scalar)

### Step 3.2: Scanner Gradient Normalization
**step_order**: 3.2  
**step_name**: Compute normalized gradient and amplitude  
**formula**:
$$g_x = \frac{G_x}{\mu_0}, \quad g_y = \frac{G_y}{\mu_0}$$
$$a_x = \frac{A_y}{\mu_0}, \quad a_y = \frac{A_x}{\mu_0}$$
$$\mathbf{g} = \begin{bmatrix} g_x \\ g_y \end{bmatrix}$$
**inputs**: $G_x=2.0$ T/m, $G_y=2.0$ T/m, $A_x=12\times10^{-3}$ T, $A_y=12\times10^{-3}$ T  
**outputs**: $g_x$, $g_y$, $a_x$, $a_y$, $\mathbf{g} \in \mathbb{R}^{2\times1}$

### Step 3.3: Spatial Grid Generation
**step_order**: 3.3  
**step_name**: Generate spatial coordinates  
**formula**:
$$x_{max} = \frac{a_x}{g_x}, \quad y_{max} = \frac{a_y}{g_y}$$
$$\mathbf{x}_{seq} = \{-x_{max}, -x_{max}+\Delta s, \ldots, x_{max}\}, \quad \Delta s = 10^{-4}$$
$$\mathbf{y}_{seq} = \{-y_{max}, -y_{max}+\Delta s, \ldots, y_{max}\}$$
$$x_n = |\mathbf{y}_{seq}|, \quad y_n = |\mathbf{x}_{seq}|$$
**inputs**: $a_x$, $a_y$, $g_x$, $g_y$, $\Delta s = 10^{-4}$  
**outputs**: $\mathbf{x}_{seq}$, $\mathbf{y}_{seq}$, $x_n$, $y_n$, $x_{max}$, $y_{max}$

### Step 3.4: Time Sequence Generation
**step_order**: 3.4  
**step_name**: Generate temporal sampling points  
**formula**:
$$f_n = \text{round}(T_R \cdot f_s)$$
$$\mathbf{t}_{seq} = \left\{0, \frac{T_R}{f_n}, \frac{2T_R}{f_n}, \ldots, T_R\right\}$$
$$F_n = |\mathbf{t}_{seq}|$$
**inputs**: $T_R = 6.528\times10^{-4}$ s, $f_s = 2.5\times10^6$ Hz  
**outputs**: $\mathbf{t}_{seq} \in \mathbb{R}^{F_n}$, $F_n$

### Step 3.5: Drive Field Computation
**step_order**: 3.5  
**step_name**: Compute drive field and its time derivative  
**formula**:
$$H_{d,x}(t) = -a_x \cos\left(2\pi f_x t + \frac{\pi}{2}\right)$$
$$H_{d,y}(t) = -a_y \cos\left(2\pi f_y t + \frac{\pi}{2}\right)$$
$$\dot{H}_{d,x}(t) = 2\pi f_x a_x \sin\left(2\pi f_x t + \frac{\pi}{2}\right)$$
$$\dot{H}_{d,y}(t) = 2\pi f_y a_y \sin\left(2\pi f_y t + \frac{\pi}{2}\right)$$
$$\mathbf{H}_d = \begin{bmatrix} H_{d,x}(\mathbf{t}_{seq}) \\ H_{d,y}(\mathbf{t}_{seq}) \end{bmatrix}, \quad \dot{\mathbf{H}}_d = \begin{bmatrix} \dot{H}_{d,x}(\mathbf{t}_{seq}) \\ \dot{H}_{d,y}(\mathbf{t}_{seq}) \end{bmatrix}$$
**inputs**: $a_x$, $a_y$, $f_x = 2.5\times10^6/102$ Hz, $f_y = 2.5\times10^6/96$ Hz, $\mathbf{t}_{seq}$  
**outputs**: $\mathbf{H}_d \in \mathbb{R}^{2 \times F_n}$, $\dot{\mathbf{H}}_d \in \mathbb{R}^{2 \times F_n}$

### Step 3.6: Phantom Generation
**step_order**: 3.6  
**step_name**: Generate P-shaped phantom image  
**formula**:
$$C_{img}[i,j] = \begin{cases} c_0 & \text{if } (i,j) \in \Omega_P \\ 0 & \text{otherwise} \end{cases}$$
where $\Omega_P$ is defined by:
$$\Omega_{outer} = \left[\frac{14 x_n}{121}, \frac{105 x_n}{121}\right) \times \left[\frac{29 y_n}{121}, \frac{90 y_n}{121}\right)$$
$$\Omega_{hole1} = \left[\frac{29 x_n}{121}, \frac{60 x_n}{121}\right) \times \left[\frac{44 y_n}{121}, \frac{75 y_n}{121}\right)$$
$$\Omega_{hole2} = \left[\frac{74 x_n}{121}, \frac{105 x_n}{121}\right) \times \left[\frac{44 y_n}{121}, \frac{90 y_n}{121}\right)$$
$$\Omega_P = \Omega_{outer} \setminus (\Omega_{hole1} \cup \Omega_{hole2})$$
**inputs**: $x_n$, $y_n$, $c_0 = 5\times10^7$  
**outputs**: $\mathbf{C}_{phantom} \in \mathbb{R}^{x_n \times y_n}$

### Step 3.7: Selection Field Grid
**step_order**: 3.7  
**step_name**: Compute static selection field at each pixel  
**formula**:
$$\mathbf{G}_{sc}[i,j,0] = g_x \cdot x_{pos}(j), \quad \mathbf{G}_{sc}[i,j,1] = g_y \cdot y_{pos}(i)$$
where:
$$x_{pos}(j) = j \cdot \Delta s - x_{max}$$
$$y_{pos}(i) = -i \cdot \Delta s + y_{max}$$
**inputs**: $g_x$, $g_y$, $x_{max}$, $y_{max}$, $\Delta s$, $x_n$, $y_n$  
**outputs**: $\mathbf{G}_{sc} \in \mathbb{R}^{x_n \times y_n \times 2}$

### Step 3.8: System Matrix Construction (Time Domain)
**step_order**: 3.8  
**step_name**: Compute time-domain system matrix via Langevin model  
**formula**:
For each time index $i \in \{0, \ldots, F_n-1\}$:

**3.8.1** Compute coefficient:
$$\mathbf{coeff}[i] = s_c \cdot m_m \cdot b_{coeff} \cdot \dot{\mathbf{H}}_d[:,i]$$

**3.8.2** Compute total field:
$$\mathbf{G}_s[i,j,k,:] = \mathbf{H}_d[:,i] - \mathbf{G}_{sc}[j,k,:]$$

**3.8.3** Compute field magnitude:
$$|\mathbf{H}|[j,k] = \sqrt{\mathbf{G}_s[i,j,k,0]^2 + \mathbf{G}_s[i,j,k,1]^2}$$

**3.8.4** Compute Langevin derivative:
$$\xi = b_{coeff} \cdot |\mathbf{H}|$$
$$L'(\xi) = \begin{cases} \frac{1}{\xi^2} - \frac{1}{\sinh^2(\xi)} & \text{if } |\xi| \geq 10^{-6} \\ \frac{1}{3} & \text{if } |\xi| < 10^{-6} \end{cases}$$

**3.8.5** Compute system matrix element:
$$\mathbf{S}_{temp}[i,j,k,:] = \Delta c \cdot \mathbf{coeff}[i] \cdot L'(\xi[j,k])$$

**inputs**: $\mathbf{H}_d$, $\dot{\mathbf{H}}_d$, $\mathbf{G}_{sc}$, $b_{coeff}$, $m_m$, $s_c=1.0$, $\Delta c = 0.05$  
**outputs**: $\mathbf{S}_{temp} \in \mathbb{R}^{F_n \times x_n \times y_n \times 2}$

### Step 3.9: Measurement Simulation (Time Domain)
**step_order**: 3.9  
**step_name**: Simulate voltage measurements  
**formula**:
$$u_x[i] = \sum_{j,k} \mathbf{C}_{phantom}[j,k] \cdot \mathbf{coeff}_x[i] \cdot L'(\xi[j,k])$$
$$u_y[i] = \sum_{j,k} \mathbf{C}_{phantom}[j,k] \cdot \mathbf{coeff}_y[i] \cdot L'(\xi[j,k])$$
**inputs**: $\mathbf{C}_{phantom}$, $\mathbf{coeff}$, $L'(\xi)$  
**outputs**: $\mathbf{u}_{temp} \in \mathbb{R}^{2 \times F_n}$

### Step 3.10: Frequency Domain Transformation - System Matrix
**step_order**: 3.10  
**step_name**: Transform system matrix to frequency domain  
**formula**:
$$\mathbf{S}_{reshape} = \text{reshape}(\mathbf{S}_{temp}, (F_n, x_n \cdot y_n, 2)) / \Delta c$$
$$\tilde{\mathbf{S}}_x = \mathcal{F}\left\{1000 \cdot \mathbf{S}_{reshape}[:,:,0]^T\right\}$$
$$\tilde{\mathbf{S}}_y = \mathcal{F}\left\{1000 \cdot \mathbf{S}_{reshape}[:,:,1]^T\right\}$$
$$\mathbf{A} = \left(\tilde{\mathbf{S}}_x + \tilde{\mathbf{S}}_y\right)^T$$
where $\mathcal{F}$ denotes FFT along the time dimension.
**inputs**: $\mathbf{S}_{temp}$, $\Delta c$  
**outputs**: $\mathbf{A} \in \mathbb{C}^{F_n \times N}$ where $N = x_n \cdot y_n$

### Step 3.11: Frequency Domain Transformation - Measurements
**step_order**: 3.11  
**step_name**: Transform measurements to frequency domain  
**formula**:
$$\tilde{\mathbf{u}} = \mathcal{F}\left\{1000 \cdot \mathbf{u}_{temp}^T\right\}^T$$
$$\mathbf{b} = \tilde{\mathbf{u}}[:,0] + \tilde{\mathbf{u}}[:,1]$$
**inputs**: $\mathbf{u}_{temp}$  
**outputs**: $\mathbf{b} \in \mathbb{C}^{F_n}$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

**Step 1**: Initialize reconstruction vector
- **step_order**: 1
- **step_name**: Zero-initialize primal variable
- **formula**: $\mathbf{x}^{(0)} = \mathbf{0}_N$
- **inputs**: $N = x_n \times y_n$
- **outputs**: $\mathbf{x} \in \mathbb{C}^N$
- **computational_note**: Dtype inherited from $\mathbf{b}$

**Step 2**: Initialize residual vector
- **step_order**: 2
- **step_name**: Zero-initialize residual
- **formula**: $\mathbf{r}^{(0)} = \mathbf{0}_M$
- **inputs**: $M$
- **outputs**: $\mathbf{r} \in \mathbb{C}^M$
- **computational_note**: Dtype inherited from $\mathbf{x}$

**Step 3**: Precompute row energies
- **step_order**: 3
- **step_name**: Compute row norms
- **formula**: $e_m = \|\mathbf{a}_m\|_2 = \sqrt{\sum_{n=0}^{N-1} |A_{m,n}|^2}, \quad \forall m \in \{0, \ldots, M-1\}$
- **inputs**: $\mathbf{A} \in \mathbb{C}^{M \times N}$
- **outputs**: $\mathbf{e} \in \mathbb{R}^M$
- **computational_note**: Uses `np.linalg.norm` per row

**Step 4**: Initialize row index cycle
- **step_order**: 4
- **step_name**: Create sequential row ordering
- **formula**: $\mathbf{k}_{cycle} = [0, 1, 2, \ldots, M-1]$
- **inputs**: $M$
- **outputs**: $\mathbf{k}_{cycle} \in \mathbb{Z}^M$
- **computational_note**: Deterministic sequential sweep

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (Outer: $\ell = 0, 1, \ldots, L-1$)

**BEGIN_LOOP** (Inner: $m = 0, 1, \ldots, M-1$)

**Step 5**: Select row index
- **step_order**: 5
- **step_name**: Get current row index
- **formula**: $k = \mathbf{k}_{cycle}[m]$
- **inputs**: $\mathbf{k}_{cycle}$, $m$
- **outputs**: $k$
- **computational_note**: Sequential access

**Step 6**: Check row validity
- **step_order**: 6
- **step_name**: Skip zero-energy rows
- **formula**: $\text{valid} = (e_k > 0)$
- **inputs**: $\mathbf{e}$, $k$
- **outputs**: Boolean flag
- **computational_note**: If false, skip Steps 7-10

**Step 7**: Compute inner product
- **step_order**: 7
- **step_name**: Forward projection
- **formula**: $\langle \mathbf{a}_k, \mathbf{x} \rangle = \sum_{n=0}^{N-1} A_{k,n} \cdot x_n$
- **inputs**: $\mathbf{A}[k,:]$, $\mathbf{x}$
- **outputs**: $p_k \in \mathbb{C}$
- **computational_note**: [FORWARD] Complex dot product

**Step 8**: Compute step size numerator
- **step_order**: 8
- **step_name**: Compute residual term
- **formula**: $\nu_k = b_k - p_k - \sqrt{\lambda} \cdot r_k$
- **inputs**: $\mathbf{b}[k]$, $p_k$, $\lambda$, $\mathbf{r}[k]$
- **outputs**: $\nu_k \in \mathbb{C}$
- **computational_note**: Regularized residual

**Step 9**: Compute step size
- **step_order**: 9
- **step_name**: Compute projection coefficient
- **formula**: $\beta = \frac{\nu_k}{e_k^2 + \lambda}$
- **inputs**: $\nu_k$, $e_k$, $\lambda$
- **outputs**: $\beta \in \mathbb{C}$
- **computational_note**: Denominator is regularized squared norm

**Step 10**: Update primal variable
- **step_order**: 10
- **step_name**: Kaczmarz update
- **formula**: $\mathbf{x} \leftarrow \mathbf{x} + \beta \cdot \overline{\mathbf{a}_k}$
- **inputs**: $\mathbf{x}$, $\beta$, $\mathbf{A}[k,:]$
- **outputs**: $\mathbf{x}$ (updated)
- **computational_note**: [ADJOINT] $\overline{\mathbf{a}_k}$ denotes complex conjugate of row $k$

**Step 11**: Update residual
- **step_order**: 11
- **step_name**: Regularization residual update
- **formula**: $r_k \leftarrow r_k + \sqrt{\lambda} \cdot \beta$
- **inputs**: $\mathbf{r}[k]$, $\lambda$, $\beta$
- **outputs**: $\mathbf{r}[k]$ (updated)
- **computational_note**: Tracks regularization contribution

**END_LOOP** (Inner)

**END_LOOP** (Outer)

### 4.3 FINALIZATION

**Step 12**: Extract real part
- **step_order**: 12
- **step_name**: Discard imaginary component
- **formula**: $\mathbf{x}_{real} = \Re(\mathbf{x})$
- **inputs**: $\mathbf{x} \in \mathbb{C}^N$
- **outputs**: $\mathbf{x}_{real} \in \mathbb{R}^N$
- **computational_note**: Physical concentration is real-valued

**Step 13**: Reshape to image
- **step_order**: 13
- **step_name**: Vector to matrix conversion
- **formula**: $\mathbf{C}_{recon} = \text{reshape}(\mathbf{x}_{real}, (x_n, y_n))$
- **inputs**: $\mathbf{x}_{real}$, $x_n$, $y_n$
- **outputs**: $\mathbf{C}_{recon} \in \mathbb{R}^{x_n \times y_n}$
- **computational_note**: Row-major ordering

**Step 14**: Crop border pixels
- **step_order**: 14
- **step_name**: Remove boundary artifacts
- **formula**: $\mathbf{C}_{crop} = \mathbf{C}_{recon}[1:-1, 1:-1]$
- **inputs**: $\mathbf{C}_{recon}$
- **outputs**: $\mathbf{C}_{crop} \in \mathbb{R}^{(x_n-2) \times (y_n-2)}$
- **computational_note**: Removes 1-pixel border on all sides

**Step 15**: Normalize output
- **step_order**: 15
- **step_name**: Scale to unit maximum
- **formula**: 
$$\mathbf{C}_{norm} = \begin{cases} \frac{\mathbf{C}_{crop}}{\max(\mathbf{C}_{crop})} & \text{if } \max(\mathbf{C}_{crop}) > 0 \\ \mathbf{C}_{crop} & \text{otherwise} \end{cases}$$
- **inputs**: $\mathbf{C}_{crop}$
- **outputs**: $\mathbf{C}_{norm} \in \mathbb{R}^{(x_n-2) \times (y_n-2)}$
- **computational_note**: Output range $[0, 1]$ if valid

---

## 5. STOPPING CRITERION

- **type**: `iteration_limit`
- **expression**: $\ell < L$ where $L$ is the maximum iteration count
- **parameters**: 
  - $L = 50$ (as called in main execution)
  - $L = 20$ (function default)
- **check_frequency**: Evaluated at the start of each outer iteration

---

## 6. EVALUATION METRICS

### Metric 1: Peak Signal-to-Noise Ratio (PSNR)

- **name**: PSNR
- **formula**:
$$\text{MSE} = \frac{1}{(x_n-2)(y_n-2)} \sum_{i,j} \left(\hat{C}_{norm}[i,j] - C_{norm}[i,j]\right)^2$$
$$\text{PSNR} = \begin{cases} 20 \log_{10}\left(\frac{R}{\sqrt{\text{MSE}}}\right) & \text{if MSE} > 0 \\ +\infty & \text{if MSE} = 0 \end{cases}$$
where $R = \max(\hat{C}_{norm}) - \min(\hat{C}_{norm})$ is the data range (defaults to 1.0 if zero)
- **reference_data**: $\hat{C}_{norm}$ (normalized, cropped ground truth phantom)
- **predicted_data**: $C_{norm}$ (normalized reconstruction)
- **description**: Measures reconstruction fidelity in decibels

### Metric 2: Structural Similarity Index (SSIM)

- **name**: SSIM (Simplified)
- **formula**:
$$\mu_x = \frac{1}{N_{px}}\sum_{i,j} \hat{C}_{norm}[i,j], \quad \mu_y = \frac{1}{N_{px}}\sum_{i,j} C_{norm}[i,j]$$
$$\sigma_x^2 = \frac{1}{N_{px}}\sum_{i,j} \left(\hat{C}_{norm}[i,j] - \mu_x\right)^2, \quad \sigma_y^2 = \frac{1}{N_{px}}\sum_{i,j} \left(C_{norm}[i,j] - \mu_y\right)^2$$
$$\sigma_{xy} = \frac{1}{N_{px}}\sum_{i,j} \left(\hat{C}_{norm}[i,j] - \mu_x\right)\left(C_{norm}[i,j] - \mu_y\right)$$
$$c_1 = (0.01 \cdot R)^2, \quad c_2 = (0.03 \cdot R)^2$$
$$\text{SSIM} = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
- **reference_data**: $\hat{C}_{norm}$ (normalized, cropped ground truth phantom)
- **predicted_data**: $C_{norm}$ (normalized reconstruction)
- **description**: Measures perceptual similarity based on luminance, contrast, and structure

### Metric 3: Mean Squared Error (MSE)

- **name**: MSE
- **formula**:
$$\text{MSE} = \frac{1}{N_{px}} \|\hat{C}_{norm} - C_{norm}\|_F^2$$
where $N_{px} = (x_n - 2)(y_n - 2)$
- **reference_data**: $\hat{C}_{norm}$