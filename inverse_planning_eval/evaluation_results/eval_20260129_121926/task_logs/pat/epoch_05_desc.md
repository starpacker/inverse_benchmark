## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Delay-and-Sum Backprojection with Linear Spectral Unmixing
- **algorithm_family**: Direct Reconstruction / Analytical Inversion
- **summary**: Photoacoustic tomography reconstruction using time-of-flight delay-and-sum backprojection followed by linear spectral unmixing for chromophore concentration estimation and oxygen saturation mapping.

---

## 1. OBJECTIVE FUNCTION

- **full_expression**: This algorithm solves two sequential linear inverse problems:
  1. Image Reconstruction (per wavelength): $\hat{p}_\lambda = \mathcal{B}(y_\lambda)$ where $\mathcal{B}$ is the backprojection operator
  2. Spectral Unmixing: $\hat{C} = \arg\min_C \|S - EC\|_2^2$ with closed-form solution $\hat{C} = E^\dagger S$

- **data_fidelity_term**: 
  - Reconstruction: Implicit L2 matching via adjoint operator (backprojection)
  - Unmixing: $\|S - EC\|_2^2$ where $S \in \mathbb{R}^{n_\lambda \times N_{pixels}}$, $E \in \mathbb{R}^{n_\lambda \times 2}$, $C \in \mathbb{R}^{2 \times N_{pixels}}$

- **regularization_term**: None (direct inversion methods)

- **constraints**: 
  - sO2 box constraint: $\text{sO}_2 \in [0, 1]$ enforced via $\text{sO}_2 = \text{clip}(\text{sO}_2, 0, 1)$
  - Masking constraint: $\text{sO}_2[\mathbf{r}] = 0$ where $(c_{\text{Hb}} + c_{\text{HbO}_2})[\mathbf{r}] \leq 0.1 \cdot \max(c_{\text{Hb}} + c_{\text{HbO}_2})$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $p_\lambda[z,y,x]$ | $(n_\lambda, n_z, n_y, n_x)$ | float64 | zeros | Reconstructed initial pressure distribution per wavelength |
| $c_{\text{Hb}}[z,y,x]$ | $(n_z, n_y, n_x)$ | float64 | Computed | Deoxyhemoglobin concentration map |
| $c_{\text{HbO}_2}[z,y,x]$ | $(n_z, n_y, n_x)$ | float64 | Computed | Oxyhemoglobin concentration map |
| $\text{sO}_2[z,y,x]$ | $(n_z, n_y, n_x)$ | float64 | zeros | Oxygen saturation map |

### 2.2 Dual Variables

None (non-iterative algorithm)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $d_{i}[z,y,x]$ | $(n_z, n_y, n_x)$ | float64 | Computed per detector | Euclidean distance from pixel to detector $i$ |
| $\tau_i[z,y,x]$ | $(n_z, n_y, n_x)$ | int | Computed per detector | Sample index corresponding to time-of-flight |
| $M_{\text{valid}}[z,y,x]$ | $(n_z, n_y, n_x)$ | bool | Computed | Valid sample index mask |
| $M_{\text{sO}_2}[z,y,x]$ | $(n_z, n_y, n_x)$ | bool | Computed | Mask for sO2 calculation threshold |
| $E$ | $(n_\lambda, 2)$ | float64 | Computed | Extinction coefficient matrix |
| $E^\dagger$ | $(2, n_\lambda)$ | float64 | Computed | Moore-Penrose pseudoinverse of $E$ |
| $S$ | $(n_\lambda, N_{pixels})$ | float64 | Reshaped | Flattened reconstruction matrix |
| $C$ | $(2, N_{pixels})$ | float64 | Computed | Flattened concentration matrix |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $n_x$ | 100 | Number of pixels in x-direction |
| $n_y$ | 100 | Number of pixels in y-direction |
| $n_z$ | 1 | Number of pixels in z-direction |
| $L_x$ | 0.025 m | Field of view in x |
| $L_y$ | 0.025 m | Field of view in y |
| $f_{\text{low}}$ | 5000 Hz | Bandpass filter low cutoff |
| $f_{\text{high}}$ | 7×10⁶ Hz | Bandpass filter high cutoff |
| $n_{\text{order}}$ | 4 | Butterworth filter order |
| $\lambda_{\text{ref}}$ | [700, 730, 760, 800, 850, 900] nm | Reference wavelengths for spectra |
| $\epsilon_{\text{Hb,ref}}$ | [100, 80, 60, 40, 30, 20] | Reference Hb absorption coefficients |
| $\epsilon_{\text{HbO}_2\text{,ref}}$ | [30, 40, 50, 60, 70, 80] | Reference HbO2 absorption coefficients |
| $\theta_{\text{mask}}$ | 0.1 | sO2 masking threshold fraction |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y_{\text{raw}}[\lambda, i, t]$ | $(n_\lambda, n_{\text{det}}, n_t)$ | float64 | Raw photoacoustic time series |
| $\mathbf{g}_i$ | $(n_{\text{det}}, 3)$ | float64 | Detector positions (x, y, z) |
| $f_s$ | scalar | float64 | Sampling frequency |
| $c$ | scalar | float64 | Speed of sound in medium |
| $\boldsymbol{\lambda}$ | $(n_\lambda,)$ | float64 | Wavelengths used in acquisition |

---

## 3. PREPROCESSING PIPELINE

### Step P1: Bandpass Filter Design
- **step_order**: P1
- **step_name**: Butterworth Bandpass Filter Coefficient Computation
- **formula**: 
$$f_{\text{nyq}} = \frac{f_s}{2}, \quad \omega_{\text{low}} = \frac{f_{\text{low}}}{f_{\text{nyq}}}, \quad \omega_{\text{high}} = \frac{f_{\text{high}}}{f_{\text{nyq}}}$$
$$H(z) = \frac{\sum_{k=0}^{n_{\text{order}}} b_k z^{-k}}{\sum_{k=0}^{n_{\text{order}}} a_k z^{-k}}$$
where $(b, a)$ are Butterworth bandpass coefficients for normalized frequencies $[\omega_{\text{low}}, \omega_{\text{high}}]$
- **inputs**: $f_s$ (scalar), $f_{\text{low}}=5000$ Hz, $f_{\text{high}}=7\times10^6$ Hz, $n_{\text{order}}=4$
- **outputs**: $b$ (filter numerator), $a$ (filter denominator)

### Step P2: Zero-Phase Bandpass Filtering
- **step_order**: P2
- **step_name**: Forward-Backward Butterworth Filtering
- **formula**: 
$$y_{\text{filt}}[\lambda, i, t] = \text{filtfilt}(b, a, y_{\text{raw}}[\lambda, i, :])$$
Zero-phase filtering: forward filter then time-reverse and filter again
- **inputs**: $y_{\text{raw}}$ $(n_\lambda, n_{\text{det}}, n_t)$, $b$, $a$
- **outputs**: $y_{\text{filt}}$ $(n_\lambda, n_{\text{det}}, n_t)$

### Step P3: Hilbert Transform and Imaginary Part Extraction
- **step_order**: P3
- **step_name**: Analytic Signal Computation via Hilbert Transform
- **formula**: 
$$y_a[\lambda, i, t] = y_{\text{filt}}[\lambda, i, t] + j \cdot \mathcal{H}\{y_{\text{filt}}[\lambda, i, :]\}(t)$$
$$y[\lambda, i, t] = \text{Im}(y_a[\lambda, i, t]) = \mathcal{H}\{y_{\text{filt}}[\lambda, i, :]\}(t)$$
where $\mathcal{H}\{\cdot\}$ is the Hilbert transform operator
- **inputs**: $y_{\text{filt}}$ $(n_\lambda, n_{\text{det}}, n_t)$
- **outputs**: $y$ $(n_\lambda, n_{\text{det}}, n_t)$ — preprocessed signal

### Step P4: Reconstruction Grid Generation
- **step_order**: P4
- **step_name**: Spatial Coordinate Grid Construction
- **formula**: 
$$x_k = -\frac{L_x}{2} + k \cdot \frac{L_x}{n_x - 1}, \quad k = 0, \ldots, n_x-1$$
$$y_j = -\frac{L_y}{2} + j \cdot \frac{L_y}{n_y - 1}, \quad j = 0, \ldots, n_y-1$$
$$z_m = 0, \quad m = 0$$
$$X[m,j,k], Y[m,j,k], Z[m,j,k] = \text{meshgrid}(z, y, x, \text{indexing='ij'})$$
- **inputs**: $L_x=0.025$, $L_y=0.025$, $n_x=100$, $n_y=100$, $n_z=1$
- **outputs**: $X, Y, Z$ each $(n_z, n_y, n_x)$

### Step P5: Distance-to-Sample Conversion Factor
- **step_order**: P5
- **step_name**: Spatial Sampling Interval Computation
- **formula**: 
$$\Delta l = \frac{c}{f_s}$$
- **inputs**: $c$ (speed of sound), $f_s$ (sampling frequency)
- **outputs**: $\Delta l$ (scalar) — distance per sample

### Step P6: Absorption Spectra Interpolation
- **step_order**: P6
- **step_name**: Linear Interpolation of Extinction Coefficients
- **formula**: 
$$\epsilon_{\text{Hb}}(\lambda) = \text{interp1d}(\lambda_{\text{ref}}, \epsilon_{\text{Hb,ref}}, \text{linear})(\lambda)$$
$$\epsilon_{\text{HbO}_2}(\lambda) = \text{interp1d}(\lambda_{\text{ref}}, \epsilon_{\text{HbO}_2\text{,ref}}, \text{linear})(\lambda)$$
- **inputs**: $\boldsymbol{\lambda}$ $(n_\lambda,)$, reference data from Section 2.4
- **outputs**: $\epsilon_{\text{Hb}}$ $(n_\lambda,)$, $\epsilon_{\text{HbO}_2}$ $(n_\lambda,)$

### Step P7: Extinction Coefficient Matrix Assembly
- **step_order**: P7
- **step_name**: Build Spectral Mixing Matrix
- **formula**: 
$$E = \begin{bmatrix} \epsilon_{\text{Hb}}(\lambda_1) & \epsilon_{\text{HbO}_2}(\lambda_1) \\ \vdots & \vdots \\ \epsilon_{\text{Hb}}(\lambda_{n_\lambda}) & \epsilon_{\text{HbO}_2}(\lambda_{n_\lambda}) \end{bmatrix} \in \mathbb{R}^{n_\lambda \times 2}$$
- **inputs**: $\epsilon_{\text{Hb}}$ $(n_\lambda,)$, $\epsilon_{\text{HbO}_2}$ $(n_\lambda,)$
- **outputs**: $E$ $(n_\lambda, 2)$

### Step P8: Pseudoinverse Computation
- **step_order**: P8
- **step_name**: Moore-Penrose Pseudoinverse of Extinction Matrix
- **formula**: 
$$E^\dagger = (E^T E)^{-1} E^T \in \mathbb{R}^{2 \times n_\lambda}$$
- **inputs**: $E$ $(n_\lambda, 2)$
- **outputs**: $E^\dagger$ $(2, n_\lambda)$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize Reconstruction Volume | $p[\lambda, z, y, x] = 0 \quad \forall \lambda, z, y, x$ | $n_\lambda$, $n_z$, $n_y$, $n_x$ | $p$ $(n_\lambda, n_z, n_y, n_x)$ | Zero-initialized accumulator |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (wavelength_loop: $i_\lambda = 0, \ldots, n_\lambda - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 2 | Extract Wavelength Signal | $y^{(\lambda)} = y[i_\lambda, :, :]$ | $y$ $(n_\lambda, n_{\text{det}}, n_t)$, $i_\lambda$ | $y^{(\lambda)}$ $(n_{\text{det}}, n_t)$ | Slice for current wavelength |

**BEGIN_LOOP** (detector_loop: $i = 0, \ldots, n_{\text{det}} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 3 | Extract Detector Position | $\mathbf{g} = (g_x^{(i)}, g_y^{(i)}, g_z^{(i)}) = \mathbf{g}_i$ | $\mathbf{g}_i$ $(n_{\text{det}}, 3)$, $i$ | $\mathbf{g}$ $(3,)$ | Current detector coordinates |
| 4 | Compute Distance Field | $d[z,y,x] = \sqrt{(X[z,y,x] - g_x)^2 + (Y[z,y,x] - g_y)^2 + (Z[z,y,x] - g_z)^2}$ | $X, Y, Z$ $(n_z, n_y, n_x)$, $\mathbf{g}$ | $d$ $(n_z, n_y, n_x)$ | Euclidean distance from each voxel to detector |
| 5 | Convert Distance to Sample Index | $\tau[z,y,x] = \left\lfloor \frac{d[z,y,x]}{\Delta l} \right\rfloor$ | $d$ $(n_z, n_y, n_x)$, $\Delta l$ | $\tau$ $(n_z, n_y, n_x)$ int | Time-of-flight to sample index |
| 6 | Compute Valid Index Mask | $M_{\text{valid}}[z,y,x] = (0 \leq \tau[z,y,x]) \land (\tau[z,y,x] < n_t)$ | $\tau$ $(n_z, n_y, n_x)$, $n_t$ | $M_{\text{valid}}$ $(n_z, n_y, n_x)$ bool | Bounds checking |
| 7 | Backproject Detector Signal | $p[i_\lambda, z, y, x] \mathrel{+}= y^{(\lambda)}[i, \tau[z,y,x]] \cdot M_{\text{valid}}[z,y,x]$ | $p$, $y^{(\lambda)}$, $\tau$, $M_{\text{valid}}$, $i_\lambda$, $i$ | $p$ $(n_\lambda, n_z, n_y, n_x)$ | Accumulate signal at time-of-flight sample |

**END_LOOP** (detector_loop)

**END_LOOP** (wavelength_loop)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 8 | Flatten Reconstruction for Unmixing | $S[\lambda, n] = p[\lambda, \text{unravel}(n)]$ where $n = z \cdot n_y \cdot n_x + y \cdot n_x + x$ | $p$ $(n_\lambda, n_z, n_y, n_x)$ | $S$ $(n_\lambda, N_{\text{pixels}})$ where $N_{\text{pixels}} = n_z \cdot n_y \cdot n_x$ | Reshape to 2D matrix |
| 9 | Linear Spectral Unmixing | $C = E^\dagger \cdot S$ | $E^\dagger$ $(2, n_\lambda)$, $S$ $(n_\lambda, N_{\text{pixels}})$ | $C$ $(2, N_{\text{pixels}})$ | Matrix multiplication |
| 10 | Reshape Concentrations | $c_{\text{Hb}}[z,y,x] = C[0, \text{ravel}(z,y,x)]$, $c_{\text{HbO}_2}[z,y,x] = C[1, \text{ravel}(z,y,x)]$ | $C$ $(2, N_{\text{pixels}})$ | $c_{\text{Hb}}, c_{\text{HbO}_2}$ each $(n_z, n_y, n_x)$ | Reshape to spatial grid |
| 11 | Compute Total Hemoglobin | $c_{\text{total}}[z,y,x] = c_{\text{Hb}}[z,y,x] + c_{\text{HbO}_2}[z,y,x]$ | $c_{\text{Hb}}, c_{\text{HbO}_2}$ | $c_{\text{total}}$ $(n_z, n_y, n_x)$ | Sum of concentrations |
| 12 | Compute sO2 Threshold Mask | $M_{\text{sO}_2}[z,y,x] = c_{\text{total}}[z,y,x] > \theta_{\text{mask}} \cdot \max(c_{\text{total}})$ | $c_{\text{total}}$, $\theta_{\text{mask}}=0.1$ | $M_{\text{sO}_2}$ $(n_z, n_y, n_x)$ bool | Identify valid voxels |
| 13 | Initialize sO2 Map | $\text{sO}_2[z,y,x] = 0 \quad \forall z, y, x$ | $n_z, n_y, n_x$ | $\text{sO}_2$ $(n_z, n_y, n_x)$ | Zero initialization |
| 14 | Compute Oxygen Saturation | $\text{sO}_2[z,y,x] = \begin{cases} \frac{c_{\text{HbO}_2}[z,y,x]}{c_{\text{total}}[z,y,x]} & \text{if } M_{\text{sO}_2}[z,y,x] = \text{True} \\ 0 & \text{otherwise} \end{cases}$ | $c_{\text{HbO}_2}$, $c_{\text{total}}$, $M_{\text{sO}_2}$ | $\text{sO}_2$ $(n_z, n_y, n_x)$ | Masked division |
| 15 | Clip sO2 to Valid Range | $\text{sO}_2[z,y,x] = \max(0, \min(1, \text{sO}_2[z,y,x]))$ | $\text{sO}_2$ | $\text{sO}_2$ $(n_z, n_y, n_x)$ | Box constraint enforcement |

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 16 | Compute Mean Reconstruction | $\bar{p}[z,y,x] = \frac{1}{n_\lambda} \sum_{\lambda=0}^{n_\lambda-1} p[\lambda, z, y, x]$ | $p$ $(n_\lambda, n_z, n_y, n_x)$ | $\bar{p}$ $(n_z, n_y, n_x)$ | Wavelength-averaged image |
| 17 | Extract 2D Slice for Display | $\bar{p}_{\text{2D}}[y,x] = \bar{p}[0, y, x]$, $\text{sO}_{2,\text{2D}}[y,x] = \text{sO}_2[0, y, x]$ | $\bar{p}$, $\text{sO}_2$ | $\bar{p}_{\text{2D}}$, $\text{sO}_{2,\text{2D}}$ each $(n_y, n_x)$ | First z-slice |
| 18 | Compute Mean sO2 in ROI | $\overline{\text{sO}_2} = \frac{1}{|R|} \sum_{(y,x) \in R} \text{sO}_{2,\text{2D}}[y,x]$ where $R = \{(y,x) : \text{sO}_{2,\text{2D}}[y,x] > 0\}$ | $\text{sO}_{2,\text{2D}}$ | $\overline{\text{sO}_2}$ (scalar) | ROI mean |

---

## 5. STOPPING CRITERION

- **type**: single_pass (non-iterative algorithm)
- **expression**: Algorithm terminates after single pass through all wavelengths and detectors
- **parameters**: 
  - Total detector iterations: $n_{\text{det}}$ per wavelength
  - Total wavelength iterations: $n_\lambda$
- **check_frequency**: N/A (deterministic completion)

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Mean sO2 | $\overline{\text{sO}_2} = \frac{\sum_{(y,x): \text{sO}_2[y,x]>0} \text{sO}_2[y,x]}{\sum_{(y,x)} \mathbb{1}[\text{sO}_2[y,x]>0]}$ | N/A | $\text{sO}_{2,\text{2D}}$ | Mean oxygen saturation over non-zero voxels |
| Reconstruction Dynamic Range | $[\min_{\lambda,z,y,x} p[\lambda,z,y,x], \max_{\lambda,z,y,x} p[\lambda,z,y,x]]$ | N/A | $p$ | Min/max of reconstructed pressure distribution |