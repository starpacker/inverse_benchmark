## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Regularized Kaczmarz Algorithm for MPI Reconstruction",
  "algorithm_family": "Algebraic Reconstruction Technique (ART) / Row-Action Methods",
  "summary": "Iteratively reconstructs magnetic particle concentration from frequency-domain voltage measurements using row-by-row projections with Tikhonov regularization."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{x} \in \mathbb{R}^N} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + \lambda \|\mathbf{x}\|_2^2$$

**data_fidelity_term**:
$$\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 = \sum_{m=0}^{M-1} |[\mathbf{A}\mathbf{x}]_m - b_m|^2$$
where $\mathbf{A} \in \mathbb{C}^{M \times N}$ is the frequency-domain system matrix and $\mathbf{b} \in \mathbb{C}^M$ is the frequency-domain voltage measurement.

**regularization_term**:
$$\lambda \|\mathbf{x}\|_2^2 = \lambda \sum_{n=0}^{N-1} |x_n|^2$$
Tikhonov (L2) regularization with parameter $\lambda$.

**constraints**:
None explicit. Implicit non-negativity enforced by taking real part in post-processing.

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}$ | $(N,)$ where $N = x_n \cdot y_n$ | `complex128` | $\mathbf{x}^{(0)} = \mathbf{0}_N$ | Vectorized particle concentration estimate |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{r}$ | $(M,)$ where $M = f_n$ | `complex128` | $\mathbf{r}^{(0)} = \mathbf{0}_M$ | Residual vector for regularization |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{e}$ | $(M,)$ | `float64` | $e_m = \|\mathbf{a}_m\|_2$ | Row energy (L2 norm of each row of $\mathbf{A}$) |
| $\beta$ | scalar | `complex128` | Computed per iteration | Step size for row update |
| $\text{dot\_prod}$ | scalar | `complex128` | Computed per iteration | Inner product $\langle \mathbf{a}_k, \mathbf{x} \rangle$ |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $\pi$ | 3.1416 | Mathematical constant |
| $k_B$ | $1.3806488 \times 10^{-23}$ J/K | Boltzmann constant |
| $T_{dt}$ | 273.15 K | Temperature offset (Celsius to Kelvin) |
| $\mu_0$ | $4\pi \times 10^{-7}$ T·m/A | Vacuum permeability |
| $T$ | 20.0 °C | Temperature |
| $d$ | $30 \times 10^{-9}$ m | Particle diameter |
| $M_s$ | $8 \times 10^5$ A/m | Saturation magnetization |
| $c_0$ | $5 \times 10^7$ particles/m³ | Concentration |
| $G_x^{sel}$ | 2.0 T/m | Selection gradient (x) |
| $G_y^{sel}$ | 2.0 T/m | Selection gradient (y) |
| $f_x^{drive}$ | $2500000/102$ Hz | Drive frequency (x) |
| $f_y^{drive}$ | $2500000/96$ Hz | Drive frequency (y) |
| $A_x^{drive}$ | $12 \times 10^{-3}$ T | Drive amplitude (x) |
| $A_y^{drive}$ | $12 \times 10^{-3}$ T | Drive amplitude (y) |
| $T_{rep}$ | $6.528 \times 10^{-4}$ s | Repetition time |
| $f_s$ | $2.5 \times 10^6$ Hz | Sampling frequency |
| $\Delta c$ | $50 \times 10^{-3}$ | Delta concentration for system matrix |
| $\text{step}$ | $1 \times 10^{-4}$ m | Spatial grid step size |
| $L_{iter}$ | 50 | Number of outer iterations |
| $\lambda$ | 0 (code default in main) | Regularization parameter |
| $\xi$ | 1.0 | Coil sensitivity |
| scaling factor | 1000 | FFT scaling multiplier |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{b}$ | $(M,)$ | `complex128` | Frequency-domain voltage measurements |
| $\mathbf{A}$ | $(M, N)$ | `complex128` | Frequency-domain system matrix |
| $\mathbf{C}_{phantom}$ | $(x_n, y_n)$ | `float64` | Ground truth phantom image |

## 3. PREPROCESSING PIPELINE

### Step P1: Compute Physical Parameters
**step_order**: P1  
**step_name**: Particle Magnetic Moment and Langevin Coefficient  
**formula**:
$$T_t = T + T_{dt} = 20.0 + 273.15 = 293.15 \text{ K}$$
$$V = \frac{\pi d^3}{6} = \frac{3.1416 \cdot (30 \times 10^{-9})^3}{6}$$
$$m_m = M_s \cdot V$$
$$\beta_{coeff} = \frac{\mu_0 \cdot m_m}{k_B \cdot T_t}$$
**inputs**: $T$ (scalar), $d$ (scalar), $M_s$ (scalar)  
**outputs**: $m_m$ (scalar), $\beta_{coeff}$ (scalar)  
**assumptions**: Particle is spherical; Langevin model applies

### Step P2: Compute Scanner Gradient Parameters
**step_order**: P2  
**step_name**: Gradient and Amplitude Normalization  
**formula**:
$$g_x = \frac{G_x^{sel}}{\mu_0}, \quad g_y = \frac{G_y^{sel}}{\mu_0}$$
$$\mathbf{g} = \begin{bmatrix} g_x \\ g_y \end{bmatrix}$$
$$a_x = \frac{A_y^{drive}}{\mu_0}, \quad a_y = \frac{A_x^{drive}}{\mu_0}$$
**inputs**: $G_x^{sel}$, $G_y^{sel}$, $A_x^{drive}$, $A_y^{drive}$, $\mu_0$  
**outputs**: $g_x$, $g_y$, $a_x$, $a_y$, $\mathbf{g}$ (shape $(2,1)$)  
**assumptions**: None

### Step P3: Compute Field of View and Grid Dimensions
**step_order**: P3  
**step_name**: Spatial Grid Setup  
**formula**:
$$x_{max} = \frac{a_x}{g_x}, \quad y_{max} = \frac{a_y}{g_y}$$
$$\mathbf{x}_{seq} = \{-x_{max}, -x_{max}+\text{step}, \ldots, x_{max}\}$$
$$\mathbf{y}_{seq} = \{-y_{max}, -y_{max}+\text{step}, \ldots, y_{max}\}$$
$$x_n = |\mathbf{y}_{seq}|, \quad y_n = |\mathbf{x}_{seq}|$$
**inputs**: $a_x$, $a_y$, $g_x$, $g_y$, step  
**outputs**: $x_{max}$, $y_{max}$, $\mathbf{x}_{seq}$ (shape $(y_n,)$), $\mathbf{y}_{seq}$ (shape $(x_n,)$), $x_n$ (int), $y_n$ (int)  
**assumptions**: Uniform spatial sampling

### Step P4: Compute Time Sequence
**step_order**: P4  
**step_name**: Temporal Sampling Points  
**formula**:
$$f_n = \text{round}(T_{rep} \cdot f_s)$$
$$\Delta t = \frac{T_{rep}}{f_n}$$
$$\mathbf{t}_{seq} = \{0, \Delta t, 2\Delta t, \ldots, T_{rep}\}$$
$$f_{n,len} = |\mathbf{t}_{seq}|$$
**inputs**: $T_{rep}$, $f_s$  
**outputs**: $\mathbf{t}_{seq}$ (shape $(f_{n,len},)$), $f_{n,len}$ (int)  
**assumptions**: Uniform temporal sampling

### Step P5: Compute Drive Field and Derivative
**step_order**: P5  
**step_name**: Drive Field Strength  
**formula**:
$$H_x^{drive}(t) = -a_x \cos(2\pi f_x^{drive} t + \frac{\pi}{2})$$
$$\dot{H}_x^{drive}(t) = a_x \sin(2\pi f_x^{drive} t + \frac{\pi}{2}) \cdot 2\pi f_x^{drive}$$
$$H_y^{drive}(t) = -a_y \cos(2\pi f_y^{drive} t + \frac{\pi}{2})$$
$$\dot{H}_y^{drive}(t) = a_y \sin(2\pi f_y^{drive} t + \frac{\pi}{2}) \cdot 2\pi f_y^{drive}$$
$$\mathbf{H}^{drive} = \begin{bmatrix} H_x^{drive}(\mathbf{t}_{seq}) \\ H_y^{drive}(\mathbf{t}_{seq}) \end{bmatrix}, \quad \dot{\mathbf{H}}^{drive} = \begin{bmatrix} \dot{H}_x^{drive}(\mathbf{t}_{seq}) \\ \dot{H}_y^{drive}(\mathbf{t}_{seq}) \end{bmatrix}$$
**inputs**: $\mathbf{t}_{seq}$, $a_x$, $a_y$, $f_x^{drive}$, $f_y^{drive}$  
**outputs**: $\mathbf{H}^{drive}$ (shape $(2, f_{n,len})$), $\dot{\mathbf{H}}^{drive}$ (shape $(2, f_{n,len})$)  
**assumptions**: Sinusoidal drive fields with 90° phase offset

### Step P6: Generate Phantom Image
**step_order**: P6  
**step_name**: P-Shape Ground Truth  
**formula**:
$$\mathbf{C}_{img}[i,j] = 0 \quad \forall i,j$$
$$\mathbf{C}_{img}\left[\lfloor\frac{14 x_n}{121}\rfloor:\lfloor\frac{105 x_n}{121}\rfloor, \lfloor\frac{29 y_n}{121}\rfloor:\lfloor\frac{90 y_n}{121}\rfloor\right] = 1$$
$$\mathbf{C}_{img}\left[\lfloor\frac{29 x_n}{121}\rfloor:\lfloor\frac{60 x_n}{121}\rfloor, \lfloor\frac{44 y_n}{121}\rfloor:\lfloor\frac{75 y_n}{121}\rfloor\right] = 0$$
$$\mathbf{C}_{img}\left[\lfloor\frac{74 x_n}{121}\rfloor:\lfloor\frac{105 x_n}{121}\rfloor, \lfloor\frac{44 y_n}{121}\rfloor:\lfloor\frac{90 y_n}{121}\rfloor\right] = 0$$
$$\mathbf{C}_{phantom} = \mathbf{C}_{img} \cdot c_0$$
**inputs**: $x_n$, $y_n$, $c_0$  
**outputs**: $\mathbf{C}_{phantom}$ (shape $(x_n, y_n)$)  
**assumptions**: P-shape phantom with uniform concentration

### Step P7: Compute Selection Field Grid
**step_order**: P7  
**step_name**: Static Selection Field at Each Pixel  
**formula**:
For each pixel $(i, j)$ where $i \in [0, x_n)$, $j \in [0, y_n)$:
$$y_{pos} = y_{max} - i \cdot \text{step}$$
$$x_{pos} = j \cdot \text{step} - x_{max}$$
$$\mathbf{G}_{sc}[i, j, 0] = g_x \cdot x_{pos}$$
$$\mathbf{G}_{sc}[i, j, 1] = g_y \cdot y_{pos}$$
**inputs**: $x_n$, $y_n$, $x_{max}$, $y_{max}$, $g_x$, $g_y$, step  
**outputs**: $\mathbf{G}_{sc}$ (shape $(x_n, y_n, 2)$)  
**assumptions**: Linear selection gradient

### Step P8: Compute System Matrix (Time Domain)
**step_order**: P8  
**step_name**: System Matrix Construction Loop  
**formula**:
For each time index $i \in [0, f_{n,len})$:
$$\mathbf{coeff}_{base} = \xi \cdot m_m \cdot \beta_{coeff} \cdot \dot{\mathbf{H}}^{drive}[:, i]$$

$$\mathbf{H}^{drive}_i = \text{tile}(\mathbf{H}^{drive}[:, i], (x_n, y_n, 1)) \quad \text{shape: } (x_n, y_n, 2)$$

$$\mathbf{G}_s = \mathbf{H}^{drive}_i - \mathbf{G}_{sc}$$

$$|\mathbf{H}|[p,q] = \sqrt{\mathbf{G}_s[p,q,0]^2 + \mathbf{G}_s[p,q,1]^2}$$

$$\xi_{val} = \beta_{coeff} \cdot |\mathbf{H}|$$

For Langevin derivative with singularity handling:
$$L'(\xi) = \begin{cases} \frac{1}{\xi^2} - \frac{1}{\sinh^2(\xi)} & \text{if } |\xi| \geq 10^{-6} \\ \frac{1}{3} & \text{if } |\xi| < 10^{-6} \end{cases}$$

$$\mathbf{DLF}[p,q,c] = L'(\xi_{val}[p,q]) \quad \forall c \in \{0, 1\}$$

$$\mathbf{S}_{system}[i, p, q, c] = \Delta c \cdot \mathbf{coeff}_{base}[c] \cdot \mathbf{DLF}[p, q, c]$$
**inputs**: $\mathbf{H}^{drive}$, $\dot{\mathbf{H}}^{drive}$, $\mathbf{G}_{sc}$, $\xi$, $m_m$, $\beta_{coeff}$, $\Delta c$  
**outputs**: $\mathbf{S}_{temp}$ (shape $(f_{n,len}, x_n, y_n, 2)$)  
**assumptions**: Equilibrium magnetization model

### Step P9: Compute Voltage Measurements (Time Domain)
**step_order**: P9  
**step_name**: Simulated Voltage Signal  
**formula**:
For each time index $i \in [0, f_{n,len})$:
$$\mathbf{P}_{pic}[p, q, c] = \mathbf{C}_{phantom}[p, q] \quad \forall c \in \{0, 1\}$$
$$\mathbf{S}_{meas}[p, q, c] = \mathbf{P}_{pic}[p, q, c] \cdot \mathbf{coeff}_{base}[c] \cdot \mathbf{DLF}[p, q, c]$$
$$u_{temp}[0, i] = \sum_{p,q} \mathbf{S}_{meas}[p, q, 0]$$
$$u_{temp}[1, i] = \sum_{p,q} \mathbf{S}_{meas}[p, q, 1]$$
**inputs**: $\mathbf{C}_{phantom}$, $\mathbf{coeff}_{base}$, $\mathbf{DLF}$  
**outputs**: $\mathbf{u}_{temp}$ (shape $(2, f_{n,len})$)  
**assumptions**: Ideal receive coil response

### Step P10: Transform System Matrix to Frequency Domain
**step_order**: P10  
**step_name**: System Matrix FFT  
**formula**:
$$\mathbf{S}_{reshaped} = \text{reshape}(\mathbf{S}_{temp}, (f_{n,len}, x_n \cdot y_n, 2)) / \Delta c$$
$$\mathbf{S}_x = \mathbf{S}_{reshaped}[:, :, 0]^T \cdot 1000 \quad \text{shape: } (N, f_{n,len})$$
$$\mathbf{S}_y = \mathbf{S}_{reshaped}[:, :, 1]^T \cdot 1000$$
$$\tilde{\mathbf{S}}_x = \mathcal{F}\{\mathbf{S}_x\} \quad \text{(FFT along last axis)}$$
$$\tilde{\mathbf{S}}_y = \mathcal{F}\{\mathbf{S}_y\}$$
$$\mathbf{A} = (\tilde{\mathbf{S}}_x + \tilde{\mathbf{S}}_y)^T \quad \text{shape: } (M, N)$$
**inputs**: $\mathbf{S}_{temp}$, $\Delta c$  
**outputs**: $\mathbf{A}$ (shape $(M, N)$) where $M = f_{n,len}$, $N = x_n \cdot y_n$  
**assumptions**: FFT along time axis

### Step P11: Transform Voltage to Frequency Domain
**step_order**: P11  
**step_name**: Measurement FFT  
**formula**:
$$\mathbf{u}_t = \mathbf{u}_{temp}^T \quad \text{shape: } (f_{n,len}, 2)$$
$$\tilde{\mathbf{u}} = \mathcal{F}\{\mathbf{u}_t^T \cdot 1000\}^T \quad \text{shape: } (f_{n,len}, 2)$$
$$\mathbf{b} = \tilde{\mathbf{u}}[:, 0] + \tilde{\mathbf{u}}[:, 1] \quad \text{shape: } (M,)$$
**inputs**: $\mathbf{u}_{temp}$  
**outputs**: $\mathbf{b}$ (shape $(M,)$)  
**assumptions**: Sum x and y channel contributions

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $\mathbf{x}$ | $\mathbf{0}_N$ | $(N,)$ | `complex128` (dtype of $\mathbf{b}$) | Zero vector |
| 2 | $\mathbf{r}$ | $\mathbf{0}_M$ | $(M,)$ | `complex128` (dtype of $\mathbf{x}$) | Zero vector |
| 3 | $e_m$ for $m \in [0, M)$ | $e_m = \|\mathbf{a}_m\|_2 = \sqrt{\sum_{n=0}^{N-1} |A_{m,n}|^2}$ | $(M,)$ | `float64` | Row L2 norms of $\mathbf{A}$ |
| 4 | row_index_cycle | $[0, 1, 2, \ldots, M-1]$ | $(M,)$ | `int64` | Sequential indices |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (Outer loop $l = 0, 1, \ldots, L_{iter}-1$ where $L_{iter} = 50$)

**BEGIN_LOOP** (Inner loop $m = 0, 1, \ldots, M-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 5 | Get Row Index | $k = \text{row\_index\_cycle}[m]$ | row_index_cycle (shape $(M,)$), $m$ (scalar) | $k$ (scalar int) | Direct array lookup |
| 6 | Check Row Energy | If $e_k \leq 0$: skip to next $m$ | $e_k$ (scalar) | boolean | Skip zero-energy rows |
| 7 | Compute Inner Product | $\text{dot\_prod} = \sum_{n=0}^{N-1} A_{k,n} \cdot x_n = \mathbf{a}_k \cdot \mathbf{x}$ | $\mathbf{A}[k, :]$ (shape $(N,)$), $\mathbf{x}$ (shape $(N,)$) | dot_prod (scalar complex) | Complex dot product |
| 8 | Compute Numerator | $\text{num} = b_k - \text{dot\_prod} - \sqrt{\lambda} \cdot r_k$ | $b_k$ (scalar), dot_prod (scalar), $\lambda$ (scalar), $r_k$ (scalar) | num (scalar complex) | Regularized residual term |
| 9 | Compute Denominator | $\text{denom} = e_k^2 + \lambda$ | $e_k$ (scalar), $\lambda$ (scalar) | denom (scalar float) | Row energy squared plus regularization |
| 10 | Compute Step Size | $\beta = \frac{\text{num}}{\text{denom}}$ | num (scalar), denom (scalar) | $\beta$ (scalar complex) | Division |
| 11 | Update Solution | $\mathbf{x} \leftarrow \mathbf{x} + \beta \cdot \overline{\mathbf{a}_k}$ | $\mathbf{x}$ (shape $(N,)$), $\beta$ (scalar), $\mathbf{A}[k, :]$ (shape $(N,)$) | $\mathbf{x}$ (shape $(N,)$) | [FORWARD] Add scaled conjugate row |
| 12 | Update Residual | $r_k \leftarrow r_k + \sqrt{\lambda} \cdot \beta$ | $r_k$ (scalar), $\lambda$ (scalar), $\beta$ (scalar) | $r_k$ (scalar) | Regularization residual accumulation |

**END_LOOP** (Inner loop over rows)

**END_LOOP** (Outer loop over iterations)

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 13 | Reshape to Image | $\mathbf{C}_{recon} = \text{reshape}(\text{Re}(\mathbf{x}), (x_n, y_n))$ | $\mathbf{x}$ (shape $(N,)$), $x_n$, $y_n$ | $\mathbf{C}_{recon}$ (shape $(x_n, y_n)$) | Take real part, reshape to 2D |
| 14 | Crop Borders | $\mathbf{C}_{cropped} = \mathbf{C}_{recon}[1:-1, 1:-1]$ | $\mathbf{C}_{recon}$ (shape $(x_n, y_n)$) | $\mathbf{C}_{cropped}$ (shape $(x_n-2, y_n-2)$) | Remove 1-pixel border |
| 15 | Compute Maximum | $v_{max} = \max(\mathbf{C}_{cropped})$ | $\mathbf{C}_{cropped}$ | $v_{max}$ (scalar) | Find maximum value |
| 16 | Normalize Output | $\mathbf{C}_{norm} = \begin{cases} \mathbf{C}_{cropped} / v_{max} & \text{if } v_{max} > 0 \\ \mathbf{C}_{cropped} & \text{otherwise} \end{cases}$ | $\mathbf{C}_{cropped}$, $v_{max}$ | $\mathbf{C}_{norm}$ (shape $(x_n-2, y_n-2)$) | Normalize to [0, 1] |

## 5. STOPPING CRITERION

```json
{
  "type": "iteration_limit",
  "expression": "l < L_{iter}",
  "parameters": {
    "L_iter": 50,
    "total_row_updates": "L_iter × M = 50 × f_{n,len}"
  },
  "check_frequency": "After each complete sweep through all M rows"
}
```

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| MSE | $\text{MSE} = \frac{1}{|\Omega|}\sum_{(i,j) \in \Omega} (\hat{C}_{i,j} - C_{i,j}^{pred})^2$ where $\hat{C} = \mathbf{C}_{phantom}[1:-1,1:-1] / \max(\mathbf{C}_{phantom}[1:-1,1:-1])$ | $\hat{C}$ (normalized cropped phantom) | $\mathbf{C}_{norm}$ | Mean squared error |
| PSNR | $\text{PSNR} = 20 \log_{10}\left(\frac{R}{\sqrt{\text{MSE}}}\right)$ where $R = \max(\hat{C}) - \min(\hat{C})$; if MSE = 0, PSNR = $\infty$ | $\hat{C}$ | $\mathbf{C}_{norm}$ | Peak signal-to-noise ratio in dB |
| SSIM | $\text{SSIM} = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$ where $\mu_x = \text{mean}(\hat{C})$, $\mu_y = \text{mean}(\mathbf{C}_{norm})$, $\sigma_x^2 = \text{var}(\hat{C})$, $\sigma_y^2 = \text{var}(\mathbf{C}_{norm})$, $\sigma_{xy} = \text{mean}((\hat{C} - \mu_x)(\mathbf{C}_{norm} - \mu_y))$, $c_1 = (0.01 \cdot R)^2$, $c_2 = (0.03 \cdot R)^2$ | $\hat{C}$ | $\mathbf{C}_{norm}$ | Structural similarity index |