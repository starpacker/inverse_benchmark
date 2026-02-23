# Algorithmic Specification Document: Magnetic Particle Imaging (MPI) Reconstruction Pipeline

---

## 1. Mathematical Formulation

### 1.1 Overall Problem Statement

The MPI reconstruction solves the linear inverse problem of estimating spatial particle concentration $C(\mathbf{r})$ from time-domain voltage measurements $u(t)$.

**Forward Model (Continuous):**
$$u(t) = \int_{\Omega} C(\mathbf{r}) \cdot S(\mathbf{r}, t) \, d\mathbf{r}$$

where $S(\mathbf{r}, t)$ is the system function encoding the particle response.

**Discrete Linear System:**
$$\mathbf{b} = \mathbf{A}\mathbf{x}$$

where:
- $\mathbf{x} \in \mathbb{C}^{N}$: vectorized concentration image ($N = N_x \times N_y$ pixels)
- $\mathbf{b} \in \mathbb{C}^{M}$: frequency-domain measurements ($M$ frequency bins)
- $\mathbf{A} \in \mathbb{C}^{M \times N}$: system matrix in frequency domain

### 1.2 Inverse Problem Formulation

**Regularized Least Squares (Tikhonov-type):**
$$\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + \lambda \|\mathbf{x}\|_2^2$$

Solved via the **Regularized Kaczmarz Algorithm** with auxiliary residual variable.

### 1.3 Physics Model Components

**1.3.1 Langevin Function and Derivative:**
$$\mathcal{L}(\xi) = \coth(\xi) - \frac{1}{\xi}$$

$$\mathcal{L}'(\xi) = \frac{1}{\xi^2} - \frac{1}{\sinh^2(\xi)}$$

with $\lim_{\xi \to 0} \mathcal{L}'(\xi) = \frac{1}{3}$

**1.3.2 Magnetic Field Model:**
$$\mathbf{H}(\mathbf{r}, t) = \mathbf{H}_{\text{drive}}(t) - \mathbf{H}_{\text{select}}(\mathbf{r})$$

Drive field (2D Lissajous trajectory):
$$H_{\text{drive},x}(t) = -A_x \cos(2\pi f_x t + \frac{\pi}{2})$$
$$H_{\text{drive},y}(t) = -A_y \cos(2\pi f_y t + \frac{\pi}{2})$$

Selection field (linear gradient):
$$\mathbf{H}_{\text{select}}(\mathbf{r}) = \begin{pmatrix} G_x \cdot x \\ G_y \cdot y \end{pmatrix}$$

**1.3.3 System Function:**
$$S(\mathbf{r}, t) = p_s \cdot m \cdot \beta \cdot \dot{\mathbf{H}}_{\text{drive}}(t) \cdot \mathcal{L}'(\beta \|\mathbf{H}(\mathbf{r}, t)\|)$$

where:
- $\beta = \frac{\mu_0 m}{k_B T}$ (Langevin parameter)
- $m = M_s \cdot V$ (particle magnetic moment)
- $p_s$ = coil sensitivity

---

## 2. Variables Taxonomy

### 2.1 Physical Constants

| Symbol | Name | Value | Units |
|--------|------|-------|-------|
| $\pi$ | Pi | 3.1416 | - |
| $k_B$ | Boltzmann constant | $1.3806488 \times 10^{-23}$ | J/K |
| $\mu_0$ | Vacuum permeability | $4\pi \times 10^{-7}$ | T·m/A |
| $T_0$ | Absolute zero offset | 273.15 | K |

### 2.2 Hyperparameters / Configuration Parameters

| Symbol | Code Variable | Default Value | Type | Description |
|--------|--------------|---------------|------|-------------|
| $T_{\text{amb}}$ | `temperature` | 20.0 | float | Ambient temperature (°C) |
| $d$ | `diameter` | $30 \times 10^{-9}$ | float | Particle diameter (m) |
| $M_s$ | `mag_saturation` | $8 \times 10^5$ | float | Saturation magnetization (A/m) |
| $C_0$ | `concentration` | $5 \times 10^7$ | float | Reference concentration |
| $G_x^{\text{raw}}$ | `select_gradient_x` | 2.0 | float | Selection gradient x (T/m) |
| $G_y^{\text{raw}}$ | `select_gradient_y` | 2.0 | float | Selection gradient y (T/m) |
| $f_x$ | `drive_freq_x` | $2.5\times10^6/102$ | float | Drive frequency x (Hz) |
| $f_y$ | `drive_freq_y` | $2.5\times10^6/96$ | float | Drive frequency y (Hz) |
| $A_x^{\text{raw}}$ | `drive_amp_x` | $12 \times 10^{-3}$ | float | Drive amplitude x (T) |
| $A_y^{\text{raw}}$ | `drive_amp_y` | $12 \times 10^{-3}$ | float | Drive amplitude y (T) |
| $T_R$ | `repetition_time` | $6.528 \times 10^{-4}$ | float | Repetition time (s) |
| $f_s$ | `sample_freq` | $2.5 \times 10^6$ | float | Sampling frequency (Hz) |
| $\Delta C$ | `delta_concentration` | $50 \times 10^{-3}$ | float | System matrix normalization |
| $\lambda$ | `lambd` | $1 \times 10^{-6}$ | float | Tikhonov regularization parameter |
| $L$ | `iterations` | 20 | int | Number of Kaczmarz iterations |

### 2.3 Derived Constants (Computed Once)

| Symbol | Formula | Code Variable | Shape | dtype |
|--------|---------|---------------|-------|-------|
| $T$ | $T_{\text{amb}} + T_0$ | `Tt` | scalar | float64 |
| $V$ | $\frac{\pi d^3}{6}$ | `volume` | scalar | float64 |
| $m$ | $M_s \cdot V$ | `mm` | scalar | float64 |
| $\beta$ | $\frac{\mu_0 m}{k_B T}$ | `b_coeff` | scalar | float64 |
| $G_x$ | $G_x^{\text{raw}} / \mu_0$ | `gx` | scalar | float64 |
| $G_y$ | $G_y^{\text{raw}} / \mu_0$ | `gy` | scalar | float64 |
| $A_x$ | $A_y^{\text{raw}} / \mu_0$ | `ax` | scalar | float64 |
| $A_y$ | $A_x^{\text{raw}} / \mu_0$ | `ay` | scalar | float64 |
| $x_{\max}$ | $A_x / G_x$ | `xmax` | scalar | float64 |
| $y_{\max}$ | $A_y / G_y$ | `ymax` | scalar | float64 |
| $N_x$ | $\text{len}(y_{\text{seq}})$ | `xn` | scalar | int |
| $N_y$ | $\text{len}(x_{\text{seq}})$ | `yn` | scalar | int |
| $N_t$ | $\text{len}(t_{\text{seq}})$ | `fn` | scalar | int |

### 2.4 Primal / Optimization Variables

| Symbol | Code Variable | Shape | dtype | Initialization |
|--------|---------------|-------|-------|----------------|
| $\mathbf{x}$ | `x` | $(N_x \cdot N_y,)$ | complex128 | $\mathbf{0}$ |

### 2.5 Auxiliary / Dual Variables

| Symbol | Code Variable | Shape | dtype | Initialization |
|--------|---------------|-------|-------|----------------|
| $\mathbf{r}$ | `residual` | $(M,)$ | complex128 | $\mathbf{0}$ |

### 2.6 Observed Data / Inputs

| Symbol | Code Variable | Shape | dtype | Description |
|--------|---------------|-------|-------|-------------|
| $\mathbf{A}$ | `system_matrix` | $(N_t, N_x \cdot N_y)$ | complex128 | FFT of system matrix |
| $\mathbf{b}$ | `measurements` | $(N_t,)$ | complex128 | FFT of voltage signal |
| $\mathbf{C}_{\text{true}}$ | `phantom_image` | $(N_x, N_y)$ | float64 | Ground truth phantom |

---

## 3. Preprocessing Pipeline

### Step 1: Compute Physical Parameters
**Input:** Raw scanner/particle parameters  
**Output:** Derived physical constants

| # | Operation | Formula |
|---|-----------|---------|
| 1.1 | Absolute temperature | $T = T_{\text{amb}} + 273.15$ |
| 1.2 | Particle volume | $V = \frac{\pi d^3}{6}$ |
| 1.3 | Magnetic moment | $m = M_s \cdot V$ |
| 1.4 | Langevin parameter | $\beta = \frac{\mu_0 m}{k_B T}$ |

### Step 2: Normalize Field Parameters
**Input:** Raw gradient/amplitude values (in Tesla)  
**Output:** Field values in A/m

| # | Operation | Formula |
|---|-----------|---------|
| 2.1 | Selection gradient | $G_x = G_x^{\text{raw}}/\mu_0, \quad G_y = G_y^{\text{raw}}/\mu_0$ |
| 2.2 | Drive amplitude | $A_x = A_y^{\text{raw}}/\mu_0, \quad A_y = A_x^{\text{raw}}/\mu_0$ |

### Step 3: Construct Spatial Grid
**Input:** $A_x, A_y, G_x, G_y$, step size $\Delta s = 10^{-4}$  
**Output:** Coordinate arrays and grid dimensions

| # | Operation | Formula |
|---|-----------|---------|
| 3.1 | FOV bounds | $x_{\max} = A_x/G_x, \quad y_{\max} = A_y/G_y$ |
| 3.2 | x-coordinates | $x_j = -x_{\max} + j \cdot \Delta s, \quad j = 0, 1, \ldots, N_y-1$ |
| 3.3 | y-coordinates | $y_i = y_{\max} - i \cdot \Delta s, \quad i = 0, 1, \ldots, N_x-1$ |
| 3.4 | Grid dimensions | $N_y = \lfloor 2x_{\max}/\Delta s \rfloor + 1, \quad N_x = \lfloor 2y_{\max}/\Delta s \rfloor + 1$ |

### Step 4: Construct Temporal Grid
**Input:** $T_R$, $f_s$  
**Output:** Time array $\mathbf{t}$

| # | Operation | Formula |
|---|-----------|---------|
| 4.1 | Number of samples | $N_t = \text{round}(T_R \cdot f_s) + 1$ |
| 4.2 | Time samples | $t_n = n \cdot \frac{T_R}{N_t-1}, \quad n = 0, 1, \ldots, N_t-1$ |

### Step 5: Compute Drive Field and Derivative
**Input:** $A_x, A_y, f_x, f_y, \mathbf{t}$  
**Output:** $\mathbf{H}_{\text{drive}}(t)$, $\dot{\mathbf{H}}_{\text{drive}}(t)$

| # | Operation | Formula |
|---|-----------|---------|
| 5.1 | Drive field x | $H_{d,x}(t) = -A_x \cos(2\pi f_x t + \frac{\pi}{2})$ |
| 5.2 | Drive field y | $H_{d,y}(t) = -A_y \cos(2\pi f_y t + \frac{\pi}{2})$ |
| 5.3 | Derivative x | $\dot{H}_{d,x}(t) = 2\pi f_x A_x \sin(2\pi f_x t + \frac{\pi}{2})$ |
| 5.4 | Derivative y | $\dot{H}_{d,y}(t) = 2\pi f_y A_y \sin(2\pi f_y t + \frac{\pi}{2})$ |

### Step 6: Compute Selection Field Grid
**Input:** Spatial coordinates, $G_x, G_y$  
**Output:** $\mathbf{H}_{\text{select}}(\mathbf{r}) \in \mathbb{R}^{N_x \times N_y \times 2}$

| # | Operation | Formula |
|---|-----------|---------|
| 6.1 | Selection field | $\mathbf{G}_{\text{sc}}[i, j, :] = \begin{pmatrix} G_x \cdot x_j \\ G_y \cdot y_i \end{pmatrix}$ |

### Step 7: Generate Ground Truth Phantom
**Input:** Grid dimensions, concentration $C_0$  
**Output:** $\mathbf{C}_{\text{true}} \in \mathbb{R}^{N_x \times N_y}$

| # | Operation | Description |
|---|-----------|-------------|
| 7.1 | Initialize | $\mathbf{C}_{\text{true}} = \mathbf{0}_{N_x \times N_y}$ |
| 7.2 | Outer rectangle | $\mathbf{C}_{\text{true}}[\lfloor\frac{14}{121}N_x\rfloor:\lfloor\frac{105}{121}N_x\rfloor, \lfloor\frac{29}{121}N_y\rfloor:\lfloor\frac{90}{121}N_y\rfloor] = 1$ |
| 7.3 | Upper hole | $\mathbf{C}_{\text{true}}[\lfloor\frac{29}{121}N_x\rfloor:\lfloor\frac{60}{121}N_x\rfloor, \lfloor\frac{44}{121}N_y\rfloor:\lfloor\frac{75}{121}N_y\rfloor] = 0$ |
| 7.4 | Lower hole | $\mathbf{C}_{\text{true}}[\lfloor\frac{74}{121}N_x\rfloor:\lfloor\frac{105}{121}N_x\rfloor, \lfloor\frac{44}{121}N_y\rfloor:\lfloor\frac{90}{121}N_y\rfloor] = 0$ |
| 7.5 | Scale | $\mathbf{C}_{\text{true}} \leftarrow \mathbf{C}_{\text{true}} \cdot C_0$ |

---

## 4. Algorithm Steps

### 4.1 Forward Operator: System Matrix and Measurement Computation

**Objective:** Compute $\mathbf{A} \in \mathbb{C}^{N_t \times N}$ and $\mathbf{b} \in \mathbb{C}^{N_t}$

#### Algorithm: Forward Model Construction

```
ALGORITHM ForwardOperator
INPUT: config_data (containing all preprocessing outputs)
OUTPUT: A (system_matrix_freq), b (voltage_freq)

INITIALIZE:
    aux_signal_temp ← zeros(N_t, N_x, N_y, 2)    [float64]
    voltage_temp ← zeros(2, N_t)                  [float64]
```

**Per Time-Point Loop** ($n = 0, 1, \ldots, N_t - 1$):

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 4.1.1 | Coefficient computation | $\gamma_n = p_s \cdot m \cdot \beta \cdot \dot{\mathbf{H}}_d(t_n)$ | $p_s, m, \beta, \dot{\mathbf{H}}_d$ | $\gamma_n \in \mathbb{R}^2$ |
| 4.1.2 | Total field | $\mathbf{H}_{i,j,n} = \mathbf{H}_d(t_n) - \mathbf{G}_{\text{sc}}[i,j]$ | $\mathbf{H}_d, \mathbf{G}_{\text{sc}}$ | $\mathbf{H} \in \mathbb{R}^{N_x \times N_y \times 2}$ |
| 4.1.3 | Field magnitude | $\|\mathbf{H}\|_{i,j} = \sqrt{H_{i,j,x}^2 + H_{i,j,y}^2}$ | $\mathbf{H}$ | $\|\mathbf{H}\| \in \mathbb{R}^{N_x \times N_y}$ |
| 4.1.4 | Langevin argument | $\xi_{i,j} = \beta \cdot \|\mathbf{H}\|_{i,j}$ | $\beta, \|\mathbf{H}\|$ | $\xi \in \mathbb{R}^{N_x \times N_y}$ |
| 4.1.5 | Langevin derivative | $\mathcal{L}'_{i,j} = \begin{cases} \frac{1}{\xi_{i,j}^2} - \frac{1}{\sinh^2(\xi_{i,j})} & \|\xi_{i,j}\| \geq 10^{-6} \\ \frac{1}{3} & \text{otherwise} \end{cases}$ | $\xi$ | $\mathcal{L}' \in \mathbb{R}^{N_x \times N_y}$ |
| 4.1.6 | System matrix entry | $S_{n,i,j} = \Delta C \cdot \gamma_n \cdot \mathcal{L}'_{i,j}$ | $\Delta C, \gamma_n, \mathcal{L}'$ | `aux_signal_temp[n,:,:,:]` |
| 4.1.7 | Voltage integration | $u_n^{(c)} = \sum_{i,j} C_{\text{true}}[i,j] \cdot \gamma_{n,c} \cdot \mathcal{L}'_{i,j}, \quad c \in \{x,y\}$ | $C_{\text{true}}, \gamma_n, \mathcal{L}'$ | `voltage_temp[:,n]` |

**Post-Processing (FFT Domain Transformation):**

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 4.1.8 | Reshape system matrix | $\tilde{\mathbf{S}} = \text{reshape}(\mathbf{S}, (N_t, N, 2)) / \Delta C$ | `aux_signal_temp` | $\tilde{\mathbf{S}} \in \mathbb{R}^{N_t \times N \times 2}$ |
| 4.1.9 | FFT x-component | $\hat{\mathbf{S}}_x = \mathcal{F}\{\tilde{\mathbf{S}}[:,:,0]^T \cdot 1000\}$ | $\tilde{\mathbf{S}}$ | $\hat{\mathbf{S}}_x \in \mathbb{C}^{N \times N_t}$ |
| 4.1.10 | FFT y-component | $\hat{\mathbf{S}}_y = \mathcal{F}\{\tilde{\mathbf{S}}[:,:,1]^T \cdot 1000\}$ | $\tilde{\mathbf{S}}$ | $\hat{\mathbf{S}}_y \in \mathbb{C}^{N \times N_t}$ |
| 4.1.11 | Combine components | $\mathbf{A} = (\hat{\mathbf{S}}_x + \hat{\mathbf{S}}_y)^T$ | $\hat{\mathbf{S}}_x, \hat{\mathbf{S}}_y$ | $\mathbf{A} \in \mathbb{C}^{N_t \times N}$ |
| 4.1.12 | FFT voltage | $\hat{\mathbf{u}} = \mathcal{F}\{\mathbf{u}^T \cdot 1000\}^T$ | `voltage_temp` | $\hat{\mathbf{u}} \in \mathbb{C}^{N_t \times 2}$ |
| 4.1.13 | Sum voltage components | $\mathbf{b} = \hat{\mathbf{u}}[:,0] + \hat{\mathbf{u}}[:,1]$ | $\hat{\mathbf{u}}$ | $\mathbf{b} \in \mathbb{C}^{N_t}$ |

---

### 4.2 Inverse Operator: Regularized Kaczmarz Algorithm

**Objective:** Solve $\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + \lambda\|\mathbf{x}\|_2^2$

#### Algorithm: Regularized Kaczmarz (Algebraic Reconstruction Technique)

```
ALGORITHM RegularizedKaczmarz
INPUT: A ∈ ℂ^(M×N), b ∈ ℂ^M, λ ∈ ℝ₊, L ∈ ℤ₊
OUTPUT: x ∈ ℂ^N

INITIALIZE:
    x ← 0_N                           [complex128]
    r ← 0_M                           [complex128]
    ε_k ← ||A[k,:]||₂  for k=0,...,M-1  [float64]
```

**Main Iteration** (for $\ell = 0, 1, \ldots, L-1$):

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 4.2.1 | Row selection | $k = m \mod M$ | iteration index $m$ | row index $k$ |
| 4.2.2 | Check energy | Skip if $\varepsilon_k = 0$ | $\varepsilon_k$ | - |
| 4.2.3 | Compute residual | $\rho_k = b_k - \langle \mathbf{a}_k, \mathbf{x} \rangle$ | $b_k, \mathbf{a}_k, \mathbf{x}$ | $\rho_k \in \mathbb{C}$ |
| 4.2.4 | Compute step size | $\beta_k = \frac{\rho_k - \sqrt{\lambda} \cdot r_k}{\varepsilon_k^2 + \lambda}$ | $\rho_k, r_k, \varepsilon_k, \lambda$ | $\beta_k \in \mathbb{C}$ |
| 4.2.5 | Update primal | $\mathbf{x} \leftarrow \mathbf{x} + \beta_k \cdot \bar{\mathbf{a}}_k$ | $\mathbf{x}, \beta_k, \mathbf{a}_k$ | $\mathbf{x} \in \mathbb{C}^N$ |
| 4.2.6 | Update auxiliary | $r_k \leftarrow r_k + \sqrt{\lambda} \cdot \beta_k$ | $r_k, \lambda, \beta_k$ | $r_k \in \mathbb{C}$ |

where $\mathbf{a}_k$ denotes the $k$-th row of $\mathbf{A}$ and $\bar{\mathbf{a}}_k$ is its complex conjugate.

**Complete per-iteration update equations:**

$$\beta_k^{(\ell)} = \frac{b_k - \mathbf{a}_k \cdot \mathbf{x}^{(\ell)} - \sqrt{\lambda} r_k^{(\ell)}}{\|\mathbf{a}_k\|_2^2 + \lambda}$$

$$\mathbf{x}^{(\ell+1)} = \mathbf{x}^{(\ell)} + \beta_k^{(\ell)} \bar{\mathbf{a}}_k$$

$$r_k^{(\ell+1)} = r_k^{(\ell)} + \sqrt{\lambda} \beta_k^{(\ell)}$$

---

### 4.3 Post-Processing: Reconstruction Finalization

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 4.3.1 | Extract real part | $\mathbf{C}_{\text{rec}} = \text{Re}(\text{reshape}(\mathbf{x}, (N_x, N_y)))$ | $\mathbf{x}$ | $\mathbf{C}_{\text{rec}} \in \mathbb{R}^{N_x \times N_y}$ |
| 4.3.2 | Border cropping | $\mathbf{C}_{\text{crop}} = \mathbf{C}_{\text{rec}}[1:N_x-1, 1:N_y-1]$ | $\mathbf{C}_{\text{rec}}$ | $\mathbf{C}_{\text{crop}} \in \mathbb{R}^{(N_x-2) \times (N_y-2)}$ |
| 4.3.3 | Normalization | $\hat{\mathbf{C}} = \frac{\mathbf{C}_{\text{crop}}}{\max(\mathbf{C}_{\text{crop}})}$ | $\mathbf{C}_{\text{crop}}$ | $\hat{\mathbf{C}} \in [0,1]^{(N_x-2) \times (N_y-2)}$ |

---

## 5. Stopping Criterion

The algorithm terminates based on a **fixed iteration count**:

$$\text{STOP when } \ell = L$$

where $L$ is the maximum number of outer Kaczmarz iterations (default: $L = 20$).

**Note:** Each outer iteration $\ell$ performs $M$ sequential row updates (one full sweep through all equations).

**Total elementary updates:** $L \times M$

**No convergence-based early stopping is implemented.**

---

## 6. Evaluation Metrics

### 6.1 Pre-Evaluation Processing

Both ground truth and reconstruction undergo identical normalization:

$$\mathbf{I}_{\text{gt}} = \frac{\mathbf{C}_{\text{true}}[1:N_x-1, 1:N_y-1]}{\max(\mathbf{C}_{\text{true}}[1:N_x-1, 1:N_y-1])}$$

$$\mathbf{I}_{\text{rec}} = \hat{\mathbf{C}}$$

Data range: $R = \max(\mathbf{I}_{\text{gt}}) - \min(\mathbf{I}_{\text{gt}})$

### 6.2 Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{(N_x-2)(N_y-2)} \sum_{i,j} \left( I_{\text{gt}}[i,j] - I_{\text{rec}}[i,j] \right)^2$$

### 6.3 Peak Signal-to-Noise Ratio (PSNR)

$$\text{PSNR} = 20 \cdot \log_{10}\left( \frac{R}{\sqrt{\text{MSE}}} \right) \quad \text{[dB]}$$

Special case: $\text{PSNR} = +\infty$ if $\text{MSE} = 0$

### 6.4 Structural Similarity Index (SSIM)

**Component statistics:**
$$\mu_x = \mathbb{E}[\mathbf{I}_{\text{gt}}], \quad \mu_y = \mathbb{E}[\mathbf{I}_{\text{rec}}]$$

$$\sigma_x^2 = \text{Var}[\mathbf{I}_{\text{gt}}], \quad \sigma_y^2 = \text{Var}[\mathbf{I}_{\text{rec}}]$$

$$\sigma_{xy} = \mathbb{E}[(\mathbf{I}_{\text{gt}} - \mu_x)(\mathbf{I}_{\text{rec}} - \mu_y)]$$

**Stabilization constants:**
$$C_1 = (0.01 \cdot R)^2, \quad C_2 = (0.03 \cdot R)^2$$

**SSIM formula:**
$$\text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y