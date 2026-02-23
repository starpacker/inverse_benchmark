## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: Regularized Kaczmarz Algorithm for Magnetic Particle Imaging (MPI) Reconstruction
- **algorithm_family**: Algebraic Reconstruction Techniques (ART) / Row-Action Methods
- **summary**: Iteratively reconstructs spatial particle concentration from frequency-domain MPI measurements by cycling through rows of the system matrix with Tikhonov-type regularization.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{x \in \mathbb{C}^N} \|Ax - b\|_2^2 + \lambda \|x\|_2^2$$

where $A \in \mathbb{C}^{M \times N}$ is the frequency-domain system matrix, $b \in \mathbb{C}^M$ is the measured voltage signal in frequency domain, and $x \in \mathbb{C}^N$ represents the vectorized particle concentration distribution.

### data_fidelity_term
$$\mathcal{D}(x) = \|Ax - b\|_2^2 = \sum_{m=0}^{M-1} |a_m^\top x - b_m|^2$$

where $a_m^\top$ denotes the $m$-th row of $A$.

### regularization_term
$$\mathcal{R}(x) = \lambda \|x\|_2^2$$

**Note**: In the main execution, `lambd=0` is used, effectively solving the unregularized problem.

### constraints
None (unconstrained optimization; non-negativity is implicitly handled by taking the real part post-reconstruction)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $x$ | primal | $(N,)$ where $N = x_n \cdot y_n$ | `complex128` | $x = \mathbf{0}_N$ | Vectorized concentration distribution |

### Dual/Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $r$ | auxiliary | $(M,)$ | `complex128` | $r = \mathbf{0}_M$ | Residual vector for regularization |
| $\beta$ | auxiliary | scalar | `complex128` | Computed per iteration | Step size for row update |
| $E$ | auxiliary | $(M,)$ | `float64` | $E_m = \|a_m\|_2$ | Precomputed row norms |

### Constants
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\lambda$ | constants | scalar | `float64` | `0` (main), `1e-6` (default) | Tikhonov regularization parameter |
| $L$ | constants | scalar | `int` | `50` (main), `20` (default) | Maximum iteration count |
| $\pi$ | constants | scalar | `float64` | `3.1416` | Pi approximation |
| $k_B$ | constants | scalar | `float64` | `1.3806488 \times 10^{-23}$ | Boltzmann constant (J/K) |
| $T_0$ | constants | scalar | `float64` | `273.15$ | Absolute zero offset (K) |
| $\mu_0$ | constants | scalar | `float64` | $4\pi \times 10^{-7}$ | Vacuum permeability (H/m) |
| $T$ | constants | scalar | `float64` | `20.0` | Temperature (°C) |
| $d$ | constants | scalar | `float64` | `30e-9` | Particle diameter (m) |
| $M_s$ | constants | scalar | `float64` | `8e5` | Saturation magnetization (A/m) |
| $C_0$ | constants | scalar | `float64` | `5e7` | Phantom concentration |
| $G_x$ | constants | scalar | `float64` | `2.0` | Selection gradient x (T/m) |
| $G_y$ | constants | scalar | `float64` | `2.0` | Selection gradient y (T/m) |
| $f_x$ | constants | scalar | `float64` | `2500000/102 \approx 24509.8$ | Drive frequency x (Hz) |
| $f_y$ | constants | scalar | `float64$ | `2500000/96 \approx 26041.7$ | Drive frequency y (Hz) |
| $A_x^{drive}$ | constants | scalar | `float64` | `12e-3` | Drive amplitude x (T) |
| $A_y^{drive}$ | constants | scalar | `float64` | `12e-3` | Drive amplitude y (T) |
| $T_R$ | constants | scalar | `float64` | `6.528e-4` | Repetition time (s) |
| $f_s$ | constants | scalar | `float64` | `2.5e6` | Sampling frequency (Hz) |
| $\Delta C$ | constants | scalar | `float64` | `50e-3` | Reference concentration for system matrix |
| $\kappa$ | constants | scalar | `float64` | `1.0` | Coil sensitivity |
| $\Delta s$ | constants | scalar | `float64` | `1e-4` | Spatial step size (m) |
| $c_1^{SSIM}$ | constants | scalar | `float64` | $(0.01 \cdot R)^2$ | SSIM stability constant 1 |
| $c_2^{SSIM}$ | constants | scalar | `float64` | $(0.03 \cdot R)^2$ | SSIM stability constant 2 |

### Observations
| Symbol | Category | Shape | Dtype | Description |
|--------|----------|-------|-------|-------------|
| $b$ | observations | $(M,)$ | `complex128` | FFT of simulated voltage measurements |
| $A$ | observations | $(M, N)$ | `complex128` | Frequency-domain system matrix |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Temperature Conversion | $T_t = T + T_0 = 20.0 + 273.15 = 293.15$ | $T$, $T_0$ | $T_t$ (scalar) |
| 2 | Particle Volume | $V = \frac{\pi d^3}{6} = \frac{3.1416 \cdot (30\times10^{-9})^3}{6}$ | $d$, $\pi$ | $V$ (scalar) |
| 3 | Magnetic Moment | $m = M_s \cdot V$ | $M_s$, $V$ | $m$ (scalar) |
| 4 | Langevin Coefficient | $\beta_{coef} = \frac{\mu_0 \cdot m}{k_B \cdot T_t}$ | $\mu_0$, $m$, $k_B$, $T_t$ | $\beta_{coef}$ (scalar) |
| 5 | Gradient Normalization | $g_x = \frac{G_x}{\mu_0}, \quad g_y = \frac{G_y}{\mu_0}$ | $G_x$, $G_y$, $\mu_0$ | $\mathbf{g} = [g_x, g_y]^\top$ (2,1) |
| 6 | Amplitude Normalization | $a_x = \frac{A_y^{drive}}{\mu_0}, \quad a_y = \frac{A_x^{drive}}{\mu_0}$ | $A_x^{drive}$, $A_y^{drive}$, $\mu_0$ | $a_x$, $a_y$ (scalars) |
| 7 | Spatial Extent | $x_{max} = \frac{a_x}{g_x}, \quad y_{max} = \frac{a_y}{g_y}$ | $a_x$, $a_y$, $g_x$, $g_y$ | $x_{max}$, $y_{max}$ (scalars) |
| 8 | Spatial Grid | $\mathbf{x} = \{-x_{max}, -x_{max}+\Delta s, \ldots, x_{max}\}$, $\mathbf{y} = \{-y_{max}, \ldots, y_{max}\}$ | $x_{max}$, $y_{max}$, $\Delta s$ | $x_n = |\mathbf{y}|$, $y_n = |\mathbf{x}|$ |
| 9 | Time Grid | $\mathbf{t} = \{0, \frac{T_R}{f_n}, \ldots, T_R\}$ where $f_n = \text{round}(T_R \cdot f_s)$ | $T_R$, $f_s$ | $\mathbf{t}$ shape $(f_n+1,)$ |
| 10 | Drive Field X | $H_x^{drive}(t) = -a_x \cos(2\pi f_x t + \frac{\pi}{2})$ | $a_x$, $f_x$, $\mathbf{t}$ | $\mathbf{H}_x^{drive}$ shape $(f_n,)$ |
| 11 | Drive Field Y | $H_y^{drive}(t) = -a_y \cos(2\pi f_y t + \frac{\pi}{2})$ | $a_y$, $f_y$, $\mathbf{t}$ | $\mathbf{H}_y^{drive}$ shape $(f_n,)$ |
| 12 | Drive Field Derivative X | $\dot{H}_x^{drive}(t) = 2\pi f_x a_x \sin(2\pi f_x t + \frac{\pi}{2})$ | $a_x$, $f_x$, $\mathbf{t}$ | $\dot{\mathbf{H}}_x^{drive}$ shape $(f_n,)$ |
| 13 | Drive Field Derivative Y | $\dot{H}_y^{drive}(t) = 2\pi f_y a_y \sin(2\pi f_y t + \frac{\pi}{2})$ | $a_y$, $f_y$, $\mathbf{t}$ | $\dot{\mathbf{H}}_y^{drive}$ shape $(f_n,)$ |
| 14 | Phantom Generation | $C_{phantom}[i,j] = \begin{cases} C_0 & \text{P-shape region} \\ 0 & \text{otherwise} \end{cases}$ | $C_0$, $x_n$, $y_n$ | $C_{phantom}$ shape $(x_n, y_n)$ |
| 15 | Selection Field Grid | $\mathbf{H}^{sel}[i,j] = \begin{bmatrix} g_x \cdot x_j \\ g_y \cdot y_i \end{bmatrix}$ where $x_j = j\Delta s - x_{max}$, $y_i = y_{max} - i\Delta s$ | $\mathbf{g}$, grid | $\mathbf{G}_{sc}$ shape $(x_n, y_n, 2)$ |
| 16 | Total Field Computation | $\mathbf{H}(\mathbf{r}, t) = \mathbf{H}^{drive}(t) - \mathbf{H}^{sel}(\mathbf{r})$ | $\mathbf{H}^{drive}$, $\mathbf{G}_{sc}$ | $\mathbf{G}_s$ shape $(x_n, y_n, 2)$ per time |
| 17 | Field Magnitude | $|\mathbf{H}| = \sqrt{H_x^2 + H_y^2}$ | $\mathbf{G}_s$ | $|\mathbf{H}|$ shape $(x_n, y_n)$ |
| 18 | Langevin Derivative | $L'(\xi) = \begin{cases} \frac{1}{\xi^2} - \frac{1}{\sinh^2(\xi)} & |\xi| \geq 10^{-6} \\ \frac{1}{3} & |\xi| < 10^{-6} \end{cases}$ where $\xi = \beta_{coef}|\mathbf{H}|$ | $\beta_{coef}$, $|\mathbf{H}|$ | $L'$ shape $(x_n, y_n)$ |
| 19 | System Matrix (Time) | $S_{aux}[t, i, j, c] = \Delta C \cdot \kappa \cdot m \cdot \beta_{coef} \cdot \dot{H}_c^{drive}(t) \cdot L'[i,j]$ | All above | $S_{aux}$ shape $(f_n, x_n, y_n, 2)$ |
| 20 | Voltage Simulation | $u_c(t) = \sum_{i,j} C_{phantom}[i,j] \cdot \kappa \cdot m \cdot \beta_{coef} \cdot \dot{H}_c^{drive}(t) \cdot L'[i,j]$ | $C_{phantom}$, etc. | $\mathbf{u}$ shape $(2, f_n)$ |
| 21 | Reshape System Matrix | $\tilde{S} = \text{reshape}(S_{aux}/\Delta C, (f_n, N, 2))$ | $S_{aux}$ | $\tilde{S}$ shape $(f_n, N, 2)$ |
| 22 | FFT System Matrix | $A = \mathcal{F}\{\tilde{S}_{:,:,0}^\top\} + \mathcal{F}\{\tilde{S}_{:,:,1}^\top\}$, scaled by 1000 | $\tilde{S}$ | $A$ shape $(f_n, N)$ |
| 23 | FFT Measurements | $b = \mathcal{F}\{u_0 \cdot 1000\} + \mathcal{F}\{u_1 \cdot 1000\}$ | $\mathbf{u}$ | $b$ shape $(f_n,)$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Extract Dimensions | $M = A.\text{shape}[0], \quad N = A.\text{shape}[1]$ | $A$ | $M$, $N$ |
| I.2 | Initialize Solution | $x^{(0)} = \mathbf{0}_N \in \mathbb{C}^N$ | $N$ | $x^{(0)}$ |
| I.3 | Initialize Residual | $r^{(0)} = \mathbf{0}_M \in \mathbb{C}^M$ | $M$ | $r^{(0)}$ |
| I.4 | Compute Row Norms | $E_m = \|a_m\|_2 = \sqrt{\sum_{n=0}^{N-1} |A_{m,n}|^2}$ for $m = 0, \ldots, M-1$ | $A$ | $\mathbf{E}$ shape $(M,)$ |
| I.5 | Initialize Row Index | $\mathcal{K} = [0, 1, 2, \ldots, M-1]$ | $M$ | $\mathcal{K}$ shape $(M,)$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (outer: $\ell = 0, \ldots, L-1$)

**BEGIN_LOOP** (inner: $m = 0, \ldots, M-1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | Get Row Index | $k = \mathcal{K}[m]$ | $\mathcal{K}$, $m$ | $k$ | Sequential row access |
| L.2 | Check Row Energy | If $E_k = 0$: skip to next $m$ | $E_k$ | — | Avoid division by zero |
| L.3 | Compute Inner Product | $\langle a_k, x \rangle = \sum_{n=0}^{N-1} A_{k,n} \cdot x_n$ | $A[k,:]$, $x$ | $\langle a_k, x \rangle$ | Complex dot product |
| L.4 | Compute Numerator | $\nu = b_k - \langle a_k, x \rangle - \sqrt{\lambda} \cdot r_k$ | $b_k$, $\langle a_k, x \rangle$, $r_k$, $\lambda$ | $\nu$ | |
| L.5 | Compute Denominator | $\delta = E_k^2 + \lambda$ | $E_k$, $\lambda$ | $\delta$ | |
| L.6 | Compute Step Size | $\beta = \frac{\nu}{\delta}$ | $\nu$, $\delta$ | $\beta$ | |
| L.7 | Update Solution | $x \leftarrow x + \beta \cdot \bar{a}_k$ | $x$, $\beta$, $A[k,:]^*$ | $x$ | $\bar{a}_k$ = conjugate of row $k$ |
| L.8 | Update Residual | $r_k \leftarrow r_k + \sqrt{\lambda} \cdot \beta$ | $r_k$, $\lambda$, $\beta$ | $r_k$ | |

**END_LOOP** (inner)

**END_LOOP** (outer)

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Extract Real Part | $\tilde{x} = \text{Re}(x)$ | $x$ | $\tilde{x} \in \mathbb{R}^N$ |
| F.2 | Reshape to Image | $C_{recon} = \text{reshape}(\tilde{x}, (x_n, y_n))$ | $\tilde{x}$, $x_n$, $y_n$ | $C_{recon}$ shape $(x_n, y_n)$ |
| F.3 | Crop Borders | $C_{crop} = C_{recon}[1:-1, 1:-1]$ | $C_{recon}$ | $C_{crop}$ shape $(x_n-2, y_n-2)$ |
| F.4 | Normalize | $C_{norm} = \begin{cases} \frac{C_{crop}}{\max(C_{crop})} & \max(C_{crop}) > 0 \\ C_{crop} & \text{otherwise} \end{cases}$ | $C_{crop}$ | $C_{norm}$ |

---

## 5. STOPPING CRITERION

| Property | Value |
|----------|-------|
| **type** | `iteration_limit` |
| **expression** | $\ell < L$ where $L = 50$ (main execution), $L = 20$ (default) |
| **parameters** | `iterations=50` (outer loop), `M` row sweeps per iteration (inner loop) |
| **check_frequency** | After each complete sweep through all $M$ rows |

**Note**: No explicit convergence check is implemented; algorithm terminates after exactly $L \times M$ row updates.

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| **MSE** | $\text{MSE} = \frac{1}{|\Omega|}\sum_{(i,j) \in \Omega}(C_{gt}[i,j] - C_{pred}[i,j])^2$ | Mean Squared Error between normalized cropped ground truth and reconstruction |
| **PSNR** | $\text{PSNR} = 20 \log_{10}\left(\frac{R}{\sqrt{\text{MSE}}}\right)$ where $R = \max(C_{gt}) - \min(C_{gt})$ | Peak Signal-to-Noise Ratio in dB; returns $\infty$ if MSE = 0 |
| **SSIM** | $\text{SSIM} = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$ | Structural Similarity Index |

where:
- $\mu_x = \text{mean}(C_{gt})$, $\mu_y = \text{mean}(C_{pred})$
- $\sigma_x^2 = \text{var}(C_{gt})$, $\sigma_y^2 = \text{var}(C_{pred})$
- $\sigma_{xy} = \text{mean}((C_{gt} - \mu_x)(C_{pred} - \mu_y))$
- $c_1 = (0.01 \cdot R)^2$, $c_2 = (0.03 \cdot R)^2$
- $R = \max(C_{gt}) - \min(C_{gt})$; if $R = 0$, set $R = 1$