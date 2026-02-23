## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Regularized Kaczmarz Algorithm for Magnetic Particle Imaging Reconstruction
- **algorithm_family**: Algebraic Reconstruction Technique (ART) / Row-Action Methods
- **summary**: Iteratively reconstructs particle concentration distribution from frequency-domain MPI measurements by sequentially projecting onto hyperplanes defined by system matrix rows with Tikhonov regularization.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{x \in \mathbb{C}^N} \left\| \mathbf{A}x - b \right\|_2^2 + \lambda \|x\|_2^2$$

Equivalently formulated as the augmented least-squares problem:
$$\min_{x} \left\| \begin{bmatrix} \mathbf{A} \\ \sqrt{\lambda}\mathbf{I} \end{bmatrix} x - \begin{bmatrix} b \\ \mathbf{0} \end{bmatrix} \right\|_2^2$$

### data_fidelity_term
$$\mathcal{D}(x) = \|\mathbf{A}x - b\|_2^2 = \sum_{m=0}^{M-1} \left| \sum_{n=0}^{N-1} A_{m,n} x_n - b_m \right|^2$$

where $\mathbf{A} \in \mathbb{C}^{M \times N}$ is the frequency-domain system matrix and $b \in \mathbb{C}^M$ is the frequency-domain measurement vector.

### regularization_term
$$\mathcal{R}(x) = \lambda \|x\|_2^2 = \lambda \sum_{n=0}^{N-1} |x_n|^2$$

This is **Tikhonov (L2) regularization** enforcing solution smoothness.

### constraints
$$x \in \mathbb{C}^N \quad \text{(unconstrained during iteration; real part extracted post-convergence)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x$ | $(N,)$ where $N = x_n \cdot y_n$ | `complex128` (inherits from $b$) | $x^{(0)} = \mathbf{0}_N$ | Vectorized concentration distribution in frequency domain |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $r$ (`residual`) | $(M,)$ | `complex128` (inherits from $x$) | $r^{(0)} = \mathbf{0}_M$ | Auxiliary residual for regularized augmented system; tracks accumulated regularization corrections |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $E$ (`energy`) | $(M,)$ | `float64` | $E_m = \|a_m\|_2$ | Precomputed L2-norm of each row of $\mathbf{A}$ |
| $\beta$ | scalar | `complex128` | Computed per row | Step size for row projection |
| `row_index_cycle` | $(M,)$ | `int64` | $[0, 1, \ldots, M-1]$ | Sequential row ordering |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda$ (`lambd`) | `1e-6` (default) or `0` (main execution) | Tikhonov regularization parameter |
| $L$ (`iterations`) | `20` (default) or `50` (main execution) | Number of outer iteration sweeps |
| $\pi$ (`PI`) | `3.1416` | Mathematical constant |
| $k_B$ (`KB`) | `1.3806488e-23` J/K | Boltzmann constant |
| $T_0$ (`TDT`) | `273.15` K | Absolute zero offset |
| $\mu_0$ (`U0`) | $4\pi \times 10^{-7}$ H/m | Vacuum permeability |
| `temperature` | `20.0` °C | Sample temperature |
| `diameter` | `30e-9` m | Particle diameter |
| `mag_saturation` | `8e5` A/m | Saturation magnetization |
| `concentration` | `5e7` | Phantom concentration scale |
| `select_gradient_x` | `2.0` T/m | Selection field gradient (x) |
| `select_gradient_y` | `2.0` T/m | Selection field gradient (y) |
| `drive_freq_x` | `2500000.0 / 102.0` Hz | Drive field frequency (x) |
| `drive_freq_y` | `2500000.0 / 96.0` Hz | Drive field frequency (y) |
| `drive_amp_x` | `12e-3` T | Drive field amplitude (x) |
| `drive_amp_y` | `12e-3` T | Drive field amplitude (y) |
| `repetition_time` | `6.528e-4` s | Signal acquisition period |
| `sample_freq` | `2.5e6` Hz | Sampling frequency |
| `delta_concentration` | `50e-3` | System matrix calibration concentration |
| `coil_sensitivity` | `1.0` | Receive coil sensitivity factor |
| `step` | `1e-4` m | Spatial grid step size |
| Langevin singularity threshold | `1e-6` | Threshold for small-argument approximation |
| SSIM constants: $c_1 = (0.01 \cdot \text{range})^2$, $c_2 = (0.03 \cdot \text{range})^2$ | Derived | SSIM stability constants |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $b$ (`measurements`/`voltage_freq`) | $(M,)$ where $M = f_n$ | `complex128` | FFT of simulated induced voltage; observation vector |
| $\mathbf{A}$ (`system_matrix`/`system_matrix_freq`) | $(M, N)$ | `complex128` | FFT-transformed system matrix relating concentration to voltage |
| `phantom_image` | $(x_n, y_n)$ | `float64` | Ground truth particle concentration distribution |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Physical Parameter Derivation

| Parameter | Formula | Inputs | Output |
|-----------|---------|--------|--------|
| Absolute temperature | $T_t = T_{\text{celsius}} + 273.15$ | `temperature` | `Tt` (K) |
| Particle volume | $V = \frac{\pi d^3}{6}$ | `diameter` | `volume` (m³) |
| Magnetic moment | $m_m = M_s \cdot V$ | `mag_saturation`, `volume` | `mm` |
| Langevin coefficient | $\beta_{\text{coeff}} = \frac{\mu_0 m_m}{k_B T_t}$ | `U0`, `mm`, `KB`, `Tt` | `b_coeff` |
| Normalized gradients | $g_x = \frac{G_x}{\mu_0}, \quad g_y = \frac{G_y}{\mu_0}$ | `select_gradient_x/y`, `U0` | `gx`, `gy` |
| Normalized amplitudes | $a_x = \frac{A_y}{\mu_0}, \quad a_y = \frac{A_x}{\mu_0}$ | `drive_amp_x/y`, `U0` | `ax`, `ay` |

### Step 1: Spatial Grid Construction

| Substep | Formula | Output Shape |
|---------|---------|--------------|
| Field of view | $x_{\max} = \frac{a_x}{g_x}, \quad y_{\max} = \frac{a_y}{g_y}$ | scalars |
| Spatial sequences | $\texttt{x\_seq} = [-x_{\max}, -x_{\max}+\Delta, \ldots, x_{\max}]$ | $(y_n,)$ |
| | $\texttt{y\_seq} = [-y_{\max}, -y_{\max}+\Delta, \ldots, y_{\max}]$ | $(x_n,)$ |
| Grid dimensions | $x_n = \text{len}(\texttt{y\_seq}), \quad y_n = \text{len}(\texttt{x\_seq})$ | scalars |

### Step 2: Temporal Sequence Construction

| Formula | Output |
|---------|--------|
| $f_n = \text{round}(T_{\text{rep}} \cdot f_s)$ | Number of time samples |
| $t_k = k \cdot \frac{T_{\text{rep}}}{f_n}, \quad k = 0, 1, \ldots, f_n$ | `t_sequence`: $(f_n+1,)$ |

### Step 3: Drive Field Computation

| Component | Formula | Shape |
|-----------|---------|-------|
| $H_x^{\text{drive}}(t)$ | $-a_x \cos(2\pi f_x t + \frac{\pi}{2})$ | $(f_n,)$ |
| $\dot{H}_x^{\text{drive}}(t)$ | $2\pi f_x a_x \sin(2\pi f_x t + \frac{\pi}{2})$ | $(f_n,)$ |
| $H_y^{\text{drive}}(t)$ | $-a_y \cos(2\pi f_y t + \frac{\pi}{2})$ | $(f_n,)$ |
| $\dot{H}_y^{\text{drive}}(t)$ | $2\pi f_y a_y \sin(2\pi f_y t + \frac{\pi}{2})$ | $(f_n,)$ |
| Stacked | $\mathbf{H}^{\text{drive}} = [H_x^{\text{drive}}; H_y^{\text{drive}}]$ | $(2, f_n)$ |
| Stacked derivative | $\dot{\mathbf{H}}^{\text{drive}} = [\dot{H}_x^{\text{drive}}; \dot{H}_y^{\text{drive}}]$ | $(2, f_n)$ |

### Step 4: Selection Field Grid

$$G_{\text{sc}}[i, j, :] = \begin{bmatrix} g_x \cdot x_{\text{pos}} \\ g_y \cdot y_{\text{pos}} \end{bmatrix}$$

where:
- $x_{\text{pos}} = j \cdot \Delta - x_{\max}$
- $y_{\text{pos}} = -i \cdot \Delta + y_{\max}$

Output shape: $(x_n, y_n, 2)$

### Step 5: Phantom Generation (P-Shape)

$$C_{\text{img}}[i,j] = \begin{cases}
C_0 & \text{if } (i,j) \in \Omega_{\text{outer}} \setminus \Omega_{\text{inner}} \\
0 & \text{otherwise}
\end{cases}$$

where $\Omega_{\text{outer}}$ and $\Omega_{\text{inner}}$ are defined by fractional index bounds:
- Outer: $[\frac{14}{121}x_n : \frac{105}{121}x_n] \times [\frac{29}{121}y_n : \frac{90}{121}y_n]$
- Inner cutouts create P-shape

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Operation | Formula | Output |
|------|-----------|---------|--------|
| I.1 | Initialize solution | $x^{(0)} = \mathbf{0} \in \mathbb{C}^N$ | `x`: $(N,)$ |
| I.2 | Initialize residual | $r^{(0)} = \mathbf{0} \in \mathbb{C}^M$ | `residual`: $(M,)$ |
| I.3 | Compute row norms | $E_m = \|\mathbf{a}_m\|_2 = \sqrt{\sum_{n=0}^{N-1} |A_{m,n}|^2}$ for $m = 0, \ldots, M-1$ | `energy`: $(M,)$ |
| I.4 | Build row index sequence | $\text{idx} = [0, 1, \ldots, M-1]$ | `row_index_cycle`: $(M,)$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (outer: $\ell = 0, \ldots, L-1$)

**BEGIN_LOOP** (inner: $m = 0, \ldots, M-1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| 4.2.1 | Row Selection | $k = \text{idx}[m]$ | `row_index_cycle`, $m$ | $k$ | Sequential ordering |
| 4.2.2 | Skip Zero Rows | If $E_k = 0$: skip to next $m$ | `energy[k]` | — | Guard against degenerate rows |
| 4.2.3 | Forward Projection [FORWARD] | $\langle \mathbf{a}_k, x \rangle = \sum_{n=0}^{N-1} A_{k,n} \cdot x_n$ | $\mathbf{A}[k,:]$, $x$ | `dot_prod` | Row-vector inner product |
| 4.2.4 | Residual Computation | $\text{num} = b_k - \langle \mathbf{a}_k, x \rangle - \sqrt{\lambda} \cdot r_k$ | $b[k]$, `dot_prod`, `residual[k]`, $\lambda$ | `numerator` | Augmented system residual |
| 4.2.5 | Normalization Factor | $\text{den} = E_k^2 + \lambda$ | `energy[k]`, $\lambda$ | `denominator` | Regularized row energy |
| 4.2.6 | Step Size Computation | $\beta = \frac{\text{num}}{\text{den}}$ | `numerator`, `denominator` | $\beta$ | Kaczmarz relaxation parameter |
| 4.2.7 | Primal Update | $x \leftarrow x + \beta \cdot \overline{\mathbf{a}_k}$ | $x$, $\beta$, $\mathbf{A}[k,:]^*$ | $x$ | Conjugate for complex domain |
| 4.2.8 | Auxiliary Residual Update | $r_k \leftarrow r_k + \sqrt{\lambda} \cdot \beta$ | `residual[k]`, $\lambda$, $\beta$ | `residual[k]` | Tracks regularization contribution |

**END_LOOP** (inner)

**END_LOOP** (outer)

### 4.3 FINALIZATION

| Step | Operation | Formula | Output |
|------|-----------|---------|--------|
| F.1 | Extract real part | $\tilde{C} = \text{Re}(x)$ | Real-valued vector |
| F.2 | Reshape to image | $C_{\text{2D}} = \text{reshape}(\tilde{C}, (x_n, y_n))$ | $(x_n, y_n)$ array |
| F.3 | Crop borders | $C_{\text{crop}} = C_{\text{2D}}[1:-1, 1:-1]$ | $(x_n-2, y_n-2)$ array |
| F.4 | Normalize | $C_{\text{norm}} = \frac{C_{\text{crop}}}{\max(C_{\text{crop}})}$ if $\max > 0$, else $C_{\text{crop}}$ | Final reconstruction |

---

## 5. STOPPING CRITERION

- **type**: `iteration_limit`
- **expression**: Terminate when $\ell = L$ (outer loop counter reaches `iterations`)
- **parameters**:
  - $L = 50$ (main execution value)
  - Default: $L = 20$
- **check_frequency**: After each complete sweep through all $M$ rows

---

## 6. EVALUATION METRICS

| Metric | Formula | Description |
|--------|---------|-------------|
| **MSE** | $\text{MSE} = \frac{1}{|\Omega|}\sum_{(i,j) \in \Omega} \left( C^{\text{gt}}_{\text{norm}}[i,j] - C^{\text{recon}}_{\text{norm}}[i,j] \right)^2$ | Mean Squared Error over cropped domain |
| **PSNR** | $\text{PSNR} = 20 \log_{10}\left(\frac{\text{range}}{\sqrt{\text{MSE}}}\right)$ where $\text{range} = \max(C^{\text{gt}}_{\text{norm}}) - \min(C^{\text{gt}}_{\text{norm}})$ | Peak Signal-to-Noise Ratio (dB) |
| **SSIM** | $\text{SSIM} = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$ | Structural Similarity Index |

where:
- $\mu_x = \text{mean}(C^{\text{gt}}_{\text{norm}})$, $\mu_y = \text{mean}(C^{\text{recon}}_{\text{norm}})$
- $\sigma_x^2 = \text{var}(C^{\text{gt}}_{\text{norm}})$, $\sigma_y^2 = \text{var}(C^{\text{recon}}_{\text{norm}})$
- $\sigma_{xy} = \text{mean}\left((C^{\text{gt}}_{\text{norm}} - \mu_x)(C^{\text{recon}}_{\text{norm}} - \mu_y)\right)$
- $c_1 = (0.01 \cdot \text{range})^2$, $c_2 = (0.03 \cdot \text{range})^2$

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include:

1. `algorithm_identification`: {`algorithm_name`, `algorithm_family`, `summary`}
2. `objective_function`: {`full_expression`, `data_fidelity_term`, `regularization_term`, `constraints`}
3. `variables`: {`primal`, `dual`, `auxiliary`, `constants`, `observations`}
4. `data_preprocessing`: Array of {`step_order`, `step_name`, `formula`, `inputs`, `outputs`}
5. `initialization`: Array of {`step_name`, `formula`, `outputs`}
6. `iteration_loop`: {`outer_loop`, `inner_loop`, `steps`: Array of {`step_name`, `formula`, `inputs`, `outputs`}}
7. `finalization`: Array of {`step_name`, `formula`, `outputs`}
8. `stopping_criterion`: {`type`, `expression`, `parameters`, `check_frequency`}
9. `evaluation`: Array of {`metric_name`, `formula`, `description`}
10. `forward_model`: {`time_domain_formula`, `frequency_domain_transform`, `langevin_derivative`}
11. `system_matrix_construction`: {`sensitivity_formula`, `fft_processing`}