# Algorithmic Specification Document: Magnetic Particle Imaging Reconstruction

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{x} \in \mathbb{C}^N} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + \lambda \|\mathbf{r}\|_2^2$$

where the residual vector $\mathbf{r}$ is implicitly coupled to $\mathbf{x}$ through the augmented system formulation.

**data_fidelity_term**:
$$\mathcal{D}(\mathbf{x}) = \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 = \sum_{m=0}^{M-1} \left| \sum_{n=0}^{N-1} A_{m,n} x_n - b_m \right|^2$$

**regularization_term**:
$$\mathcal{R}(\mathbf{r}) = \lambda \|\mathbf{r}\|_2^2 = \lambda \sum_{m=0}^{M-1} |r_m|^2$$

**constraints**:
- None (unconstrained optimization; non-negativity enforced via post-processing by taking real part)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{x}$ | primal | $(N,)$ | complex128 | $\mathbf{0}$ | Vectorized particle concentration image |
| $\mathbf{r}$ | primal | $(M,)$ | complex128 | $\mathbf{0}$ | Auxiliary residual vector for regularization |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{A}$ | constants | $(M, N)$ | complex128 | Computed | Frequency-domain system matrix |
| $\mathbf{b}$ | constants | $(M,)$ | complex128 | Computed | Frequency-domain measurement vector |
| $\lambda$ | constants | scalar | float64 | $10^{-6}$ (default) | Tikhonov regularization parameter |
| $L$ | constants | scalar | int | 20 (default) | Maximum iterations |
| $M$ | constants | scalar | int | $F_n$ | Number of frequency components (equations) |
| $N$ | constants | scalar | int | $X_n \cdot Y_n$ | Number of spatial pixels (unknowns) |
| $\|\mathbf{a}_k\|$ | constants | $(M,)$ | float64 | Precomputed | Row-wise $\ell_2$ norms of $\mathbf{A}$ |

### Physical Constants
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\pi$ | constants | scalar | float64 | 3.1416 | Mathematical constant |
| $k_B$ | constants | scalar | float64 | $1.3806488 \times 10^{-23}$ | Boltzmann constant (J/K) |
| $T_0$ | constants | scalar | float64 | 273.15 | Absolute zero offset (K) |
| $\mu_0$ | constants | scalar | float64 | $4\pi \times 10^{-7}$ | Vacuum permeability (H/m) |

### Scanner Parameters
| Symbol | Category | Shape | Dtype | Default | Description |
|--------|----------|-------|-------|---------|-------------|
| $T$ | constants | scalar | float64 | 20.0 | Temperature (°C) |
| $d$ | constants | scalar | float64 | $30 \times 10^{-9}$ | Particle diameter (m) |
| $M_s$ | constants | scalar | float64 | $8 \times 10^5$ | Saturation magnetization (A/m) |
| $c_0$ | constants | scalar | float64 | $5 \times 10^7$ | Particle concentration |
| $G_x, G_y$ | constants | scalar | float64 | 2.0, 2.0 | Selection field gradients (T/m) |
| $f_x, f_y$ | constants | scalar | float64 | $\frac{2.5\times10^6}{102}, \frac{2.5\times10^6}{96}$ | Drive frequencies (Hz) |
| $A_x^{drive}, A_y^{drive}$ | constants | scalar | float64 | $12 \times 10^{-3}$ | Drive amplitudes (T) |
| $T_R$ | constants | scalar | float64 | $6.528 \times 10^{-4}$ | Repetition time (s) |
| $f_s$ | constants | scalar | float64 | $2.5 \times 10^6$ | Sampling frequency (Hz) |
| $\Delta c$ | constants | scalar | float64 | $50 \times 10^{-3}$ | Calibration concentration |

### Intermediate Variables
| Symbol | Category | Shape | Dtype | Description |
|--------|----------|-------|-------|-------------|
| $\mathbf{C}_{phantom}$ | observations | $(X_n, Y_n)$ | float64 | Ground truth phantom image |
| $\mathbf{G}_{sc}$ | constants | $(X_n, Y_n, 2)$ | float64 | Selection field at each spatial position |
| $\mathbf{H}_{drive}(t)$ | constants | $(2, F_n)$ | float64 | Time-varying drive field |
| $\dot{\mathbf{H}}_{drive}(t)$ | constants | $(2, F_n)$ | float64 | Time derivative of drive field |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Physical Parameter Computation
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Compute particle magnetic moment and Langevin coefficient |
| formula | $$T_t = T + T_0$$ $$V = \frac{\pi d^3}{6}$$ $$m_m = M_s \cdot V$$ $$\beta = \frac{\mu_0 \cdot m_m}{k_B \cdot T_t}$$ |
| inputs | $T$ (scalar), $d$ (scalar), $M_s$ (scalar) |
| outputs | $T_t$ (scalar): absolute temperature; $m_m$ (scalar): magnetic moment; $\beta$ (scalar): Langevin coefficient |

### Step 3.2: Scanner Field Parameters
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Normalize field parameters |
| formula | $$g_x = \frac{G_x}{\mu_0}, \quad g_y = \frac{G_y}{\mu_0}$$ $$a_x = \frac{A_y^{drive}}{\mu_0}, \quad a_y = \frac{A_x^{drive}}{\mu_0}$$ |
| inputs | $G_x, G_y, A_x^{drive}, A_y^{drive}$ (scalars), $\mu_0$ (scalar) |
| outputs | $g_x, g_y, a_x, a_y$ (scalars): normalized gradient and amplitude |

### Step 3.3: Spatial Grid Construction
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Define spatial field of view and discretization |
| formula | $$x_{max} = \frac{a_x}{g_x}, \quad y_{max} = \frac{a_y}{g_y}$$ $$\Delta s = 10^{-4}$$ $$\mathbf{x}_{seq} = \{-x_{max}, -x_{max}+\Delta s, \ldots, x_{max}\}$$ $$\mathbf{y}_{seq} = \{-y_{max}, -y_{max}+\Delta s, \ldots, y_{max}\}$$ $$X_n = |\mathbf{y}_{seq}|, \quad Y_n = |\mathbf{x}_{seq}|$$ |
| inputs | $a_x, a_y, g_x, g_y$ (scalars) |
| outputs | $\mathbf{x}_{seq}$ ($(Y_n,)$), $\mathbf{y}_{seq}$ ($(X_n,)$), $X_n, Y_n$ (scalars) |

### Step 3.4: Temporal Sequence Construction
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Define time sampling |
| formula | $$F_n = \text{round}(T_R \cdot f_s)$$ $$\Delta t = \frac{T_R}{F_n}$$ $$\mathbf{t}_{seq} = \{0, \Delta t, 2\Delta t, \ldots, T_R\}$$ |
| inputs | $T_R$ (scalar), $f_s$ (scalar) |
| outputs | $\mathbf{t}_{seq}$ ($(F_n,)$): time sequence; $F_n$ (scalar): number of time samples |

### Step 3.5: Drive Field Computation
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Compute oscillating drive field and its derivative |
| formula | $$H_{drive,x}(t) = -a_x \cos(2\pi f_x t + \frac{\pi}{2})$$ $$H_{drive,y}(t) = -a_y \cos(2\pi f_y t + \frac{\pi}{2})$$ $$\dot{H}_{drive,x}(t) = 2\pi f_x a_x \sin(2\pi f_x t + \frac{\pi}{2})$$ $$\dot{H}_{drive,y}(t) = 2\pi f_y a_y \sin(2\pi f_y t + \frac{\pi}{2})$$ |
| inputs | $a_x, a_y, f_x, f_y$ (scalars), $\mathbf{t}_{seq}$ ($(F_n,)$) |
| outputs | $\mathbf{H}_{drive}$ ($(2, F_n)$), $\dot{\mathbf{H}}_{drive}$ ($(2, F_n)$) |

### Step 3.6: Selection Field Grid
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Compute selection field at each spatial position |
| formula | For each pixel $(i, j)$ where $i \in [0, X_n)$, $j \in [0, Y_n)$: $$y_{pos} = y_{max} - i \cdot \Delta s$$ $$x_{pos} = -x_{max} + j \cdot \Delta s$$ $$G_{sc}[i, j, 0] = g_x \cdot x_{pos}$$ $$G_{sc}[i, j, 1] = g_y \cdot y_{pos}$$ |
| inputs | $g_x, g_y, x_{max}, y_{max}$ (scalars), $X_n, Y_n$ (scalars), $\Delta s$ (scalar) |
| outputs | $\mathbf{G}_{sc}$ ($(X_n, Y_n, 2)$): selection field components |

### Step 3.7: Phantom Generation
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Generate P-shaped phantom mask |
| formula | Initialize $\mathbf{C}_{img} = \mathbf{0}_{X_n \times Y_n}$. Define index bounds: $$i_1 = \lfloor X_n \cdot \frac{14}{121} \rfloor, \quad i_2 = \lfloor X_n \cdot \frac{105}{121} \rfloor$$ $$j_1 = \lfloor Y_n \cdot \frac{29}{121} \rfloor, \quad j_2 = \lfloor Y_n \cdot \frac{90}{121} \rfloor$$ Set $C_{img}[i_1:i_2, j_1:j_2] = 1$ Apply cutouts (see code for exact indices). Finally: $$\mathbf{C}_{phantom} = c_0 \cdot \mathbf{C}_{img}$$ |
| inputs | $X_n, Y_n$ (scalars), $c_0$ (scalar) |
| outputs | $\mathbf{C}_{phantom}$ ($(X_n, Y_n)$): ground truth concentration |

### Step 3.8: Forward Model - System Matrix Construction
| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | Compute time-domain sensitivity and transform to frequency domain |
| formula | For each time index $i \in [0, F_n)$ and spatial position $(p, q)$: $$\mathbf{H}_{total}[p,q] = \mathbf{H}_{drive}[:,i] - \mathbf{G}_{sc}[p,q,:]$$ $$|\mathbf{H}| = \sqrt{H_{total,x}^2 + H_{total,y}^2}$$ $$\xi = \beta \cdot |\mathbf{H}|$$ $$L'(\xi) = \begin{cases} \frac{1}{3} & \text{if } |\xi| < 10^{-6} \\ \frac{1}{\xi^2} - \frac{1}{\sinh^2(\xi)} & \text{otherwise} \end{cases}$$ $$S_{time}[i, p, q, :] = \kappa \cdot \dot{\mathbf{H}}_{drive}[:,i] \cdot L'(\xi) \cdot \Delta c$$ where $\kappa = m_m \cdot \beta$ (coil sensitivity = 1). Reshape and FFT: $$\tilde{\mathbf{S}}_{x} = \mathcal{F}\{S_{time}[:,:,:,0]^T\} \cdot 1000$$ $$\tilde{\mathbf{S}}_{y} = \mathcal{F}\{S_{time}[:,:,:,1]^T\} \cdot 1000$$ $$\mathbf{A} = (\tilde{\mathbf{S}}_{x} + \tilde{\mathbf{S}}_{y})^T / \Delta c$$ |
| inputs | $\mathbf{H}_{drive}$, $\dot{\mathbf{H}}_{drive}$, $\mathbf{G}_{sc}$, $\beta$, $m_m$, $\Delta c$ |
| outputs | $\mathbf{A}$ ($(F_n, X_n \cdot Y_n)$, complex128): frequency-domain system matrix |

### Step 3.9: Forward Model - Measurement Simulation
| Field | Value |
|-------|-------|
| step_order | 9 |
| step_name | Simulate voltage measurements from phantom |
| formula | For each time index $i$: $$u_x(t_i) = \sum_{p,q} C_{phantom}[p,q] \cdot \kappa \cdot \dot{H}_{drive,x}(t_i) \cdot L'(\xi[p,q])$$ $$u_y(t_i) = \sum_{p,q} C_{phantom}[p,q] \cdot \kappa \cdot \dot{H}_{drive,y}(t_i) \cdot L'(\xi[p,q])$$ Transform to frequency domain: $$\tilde{\mathbf{u}} = \mathcal{F}\{\mathbf{u}_x\} \cdot 1000 + \mathcal{F}\{\mathbf{u}_y\} \cdot 1000$$ |
| inputs | $\mathbf{C}_{phantom}$, $\dot{\mathbf{H}}_{drive}$, $L'(\xi)$, $\kappa$ |
| outputs | $\mathbf{b}$ ($(F_n,)$, complex128): frequency-domain measurements |

---

## 4. ALGORITHM STEPS

### Initialization

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Initialize solution vector | $\mathbf{x}^{(0)} = \mathbf{0}_N$ | $N$ | $\mathbf{x}^{(0)}$ ($(N,)$, complex128) | Zero initialization |
| Initialize residual vector | $\mathbf{r}^{(0)} = \mathbf{0}_M$ | $M$ | $\mathbf{r}^{(0)}$ ($(M,)$, complex128) | Auxiliary variable for regularization |
| Precompute row norms | $e_m = \|\mathbf{a}_m\|_2 = \sqrt{\sum_{n=0}^{N-1} |A_{m,n}|^2}$ for $m = 0, \ldots, M-1$ | $\mathbf{A}$ | $\mathbf{e}$ ($(M,)$, float64) | One-time $O(MN)$ computation |
| Define row index cycle | $\mathbf{k} = [0, 1, 2, \ldots, M-1]$ | $M$ | $\mathbf{k}$ ($(M,)$, int) | Sequential row access |

---

### Iteration Loop

**BEGIN LOOP** over $\ell = 0, 1, \ldots, L-1$:

**BEGIN INNER LOOP** over $m = 0, 1, \ldots, M-1$:

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Get row index | $k = \mathbf{k}[m]$ | $\mathbf{k}$, $m$ | $k$ (scalar) | Index mapping |
| Check row energy | If $e_k \leq 0$: skip to next $m$ | $e_k$ | — | Avoid division by zero |
| Compute inner product | $\langle \mathbf{a}_k, \mathbf{x} \rangle = \sum_{n=0}^{N-1} A_{k,n} \cdot x_n$ | $\mathbf{A}[k,:]$, $\mathbf{x}$ | $\alpha$ (complex scalar) | $O(N)$ dot product |
| Compute update coefficient numerator | $\nu = b_k - \alpha - \sqrt{\lambda} \cdot r_k$ | $b_k$, $\alpha$, $\lambda$, $r_k$ | $\nu$ (complex scalar) | Residual term |
| Compute update coefficient denominator | $\delta = e_k^2 + \lambda$ | $e_k$, $\lambda$ | $\delta$ (real scalar) | Regularized denominator |
| Compute step size | $\gamma = \frac{\nu}{\delta}$ | $\nu$, $\delta$ | $\gamma$ (complex scalar) | Kaczmarz step |
| Update solution | $x_n \leftarrow x_n + \gamma \cdot \overline{A_{k,n}}$ for all $n$ | $\mathbf{x}$, $\gamma$, $\mathbf{A}[k,:]$ | $\mathbf{x}$ (updated) | Conjugate of row elements |
| Update residual | $r_k \leftarrow r_k + \sqrt{\lambda} \cdot \gamma$ | $r_k$, $\lambda$, $\gamma$ | $r_k$ (updated) | Regularization accumulator |

**END INNER LOOP**

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Convergence check | None (fixed iterations) | — | — | No early termination |

**END LOOP**

---

### Post-Processing Steps

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Reshape to image | $\mathbf{C}_{raw} = \text{reshape}(\mathbf{x}, (X_n, Y_n))$ | $\mathbf{x}$, $X_n$, $Y_n$ | $\mathbf{C}_{raw}$ ($(X_n, Y_n)$, complex128) | Column-major reshape |
| Extract real part | $\mathbf{C}_{real} = \Re(\mathbf{C}_{raw})$ | $\mathbf{C}_{raw}$ | $\mathbf{C}_{real}$ ($(X_n, Y_n)$, float64) | Physical concentration is real |
| Crop borders | $\mathbf{C}_{crop} = \mathbf{C}_{real}[1:X_n-1, 1:Y_n-1]$ | $\mathbf{C}_{real}$ | $\mathbf{C}_{crop}$ ($(X_n-2, Y_n-2)$) | Remove boundary artifacts |
| Normalize | $\mathbf{C}_{norm} = \begin{cases} \frac{\mathbf{C}_{crop}}{\max(\mathbf{C}_{crop})} & \text{if } \max(\mathbf{C}_{crop}) > 0 \\ \mathbf{C}_{crop} & \text{otherwise} \end{cases}$ | $\mathbf{C}_{crop}$ | $\mathbf{C}_{norm}$ ($(X_n-2, Y_n-2)$) | Scale to [0, 1] |

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $\ell < L$ where $\ell$ is current outer iteration index |
| parameters | $L = 20$ (default), configurable via `iterations` argument |

---

## 6. EVALUATION METRICS

### Mean Squared Error (MSE)
| Field | Value |
|-------|-------|
| name | MSE |
| formula | $$\text{MSE} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} \left( \hat{C}_{norm}[i,j] - C_{norm}[i,j] \right)^2$$ where $\Omega$ is the cropped image domain, $\hat{C}_{norm}$ is normalized ground truth, $C_{norm}$ is normalized reconstruction |
| description | Average squared pixel-wise reconstruction error |

### Peak Signal-to-Noise Ratio (PSNR)
| Field | Value |
|-------|-------|
| name | PSNR |
| formula | $$\text{PSNR} = \begin{cases} +\infty & \text{if } \text{MSE} = 0 \\ 20 \cdot \log_{10}\left(\frac{R}{\sqrt{\text{MSE}}}\right) & \text{otherwise} \end{cases}$$ where $R = \max(\hat{C}_{norm}) - \min(\hat{C}_{norm})$ is the dynamic range (default to 1.0 if zero) |
| description | Logarithmic ratio of signal power to noise power (dB) |

### Structural Similarity Index (SSIM)
| Field | Value |
|-------|-------|
| name | SSIM |
| formula | $$\text{SSIM} = \frac{(2\mu_{\hat{C}}\mu_C + c_1)(2\sigma_{\hat{C}C} + c_2)}{(\mu_{\hat{C}}^2 + \mu_C^2 + c_1)(\sigma_{\hat{C}}^2 + \sigma_C^2 + c_2)}$$ where: $$\mu_{\hat{C}} = \frac{1}{|\Omega|}\sum_{(i,j)}\hat{C}_{norm}[i,j], \quad \mu_C = \frac{1}{|\Omega|}\sum_{(i,j)}C_{norm}[i,j]$$ $$\sigma_{\hat{C}}^2 = \frac{1}{|\Omega|}\sum_{(i,j)}(\hat{C}_{norm}[i,j] - \mu_{\hat{C}})^2$$ $$\sigma_C^2 = \frac{1}{|\Omega|}\sum_{(i,j)}(C_{norm}[i,j] - \mu_C)^2$$ $$\sigma_{\hat{C}C} = \frac{1}{|\Omega|}\sum_{(i,j)}(\hat{C}_{norm}[i,j] - \mu_{\hat{C}})(C_{norm}[i,j] - \mu_C)$$ $$c_1 = (0.01 \cdot R)^2, \quad c_2 = (0.03 \cdot R)^2$$ |
| description | Perceptual similarity measure combining luminance, contrast, and structure; range $[-1, 1]$ with 1 indicating identical images |