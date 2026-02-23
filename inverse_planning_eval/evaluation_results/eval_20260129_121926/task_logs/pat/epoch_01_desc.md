# Algorithmic Specification Document: Photoacoustic Tomography Reconstruction with Spectral Unmixing

## 1. Mathematical Formulation

### 1.1 Overall Problem Statement

The algorithm solves the **photoacoustic tomography (PAT) image reconstruction** problem followed by **linear spectral unmixing** for blood oxygen saturation (sO₂) estimation.

#### 1.1.1 Forward Model (Photoacoustic Signal Generation)

The photoacoustic forward problem models signal acquisition as:

$$y_{d}(t) = \int_{\Omega} p_0(\mathbf{r}) \cdot \delta\left(t - \frac{|\mathbf{r} - \mathbf{r}_d|}{c}\right) d\mathbf{r}$$

Where:
- $y_d(t)$ = time-domain signal at detector $d$
- $p_0(\mathbf{r})$ = initial pressure distribution at position $\mathbf{r}$
- $\mathbf{r}_d$ = position of detector $d$
- $c$ = speed of sound in medium
- $\Omega$ = imaging domain

**Discretized Forward Model:**
$$y_{d,k} = \sum_{i,j,l} p_0[i,j,l] \cdot \mathbb{1}\left[k = \left\lfloor\frac{\sqrt{(x_i - x_d)^2 + (y_j - y_d)^2 + (z_l - z_d)^2}}{c/f_s}\right\rfloor\right]$$

#### 1.1.2 Inverse Problem (Delay-and-Sum Reconstruction)

The reconstruction uses **Delay-and-Sum (DAS) Backprojection**:

$$\hat{p}_0^{(\lambda)}[i,j,l] = \sum_{d=1}^{N_{\text{det}}} y_d^{(\lambda)}\left[k_{d,i,j,l}\right]$$

Where the delay index is:
$$k_{d,i,j,l} = \left\lfloor\frac{\|\mathbf{r}_{i,j,l} - \mathbf{r}_d\|}{\Delta l}\right\rfloor, \quad \Delta l = \frac{c}{f_s}$$

#### 1.1.3 Spectral Unmixing Problem

**Linear Mixing Model:**
$$\mathbf{s}(\mathbf{r}) = \mathbf{E} \cdot \mathbf{c}(\mathbf{r}) + \boldsymbol{\epsilon}$$

Expanded form for $N_\lambda$ wavelengths:
$$\begin{bmatrix} \hat{p}_0^{(\lambda_1)}(\mathbf{r}) \\ \hat{p}_0^{(\lambda_2)}(\mathbf{r}) \\ \vdots \\ \hat{p}_0^{(\lambda_{N_\lambda})}(\mathbf{r}) \end{bmatrix} = \begin{bmatrix} \varepsilon_{\text{Hb}}(\lambda_1) & \varepsilon_{\text{HbO}_2}(\lambda_1) \\ \varepsilon_{\text{Hb}}(\lambda_2) & \varepsilon_{\text{HbO}_2}(\lambda_2) \\ \vdots & \vdots \\ \varepsilon_{\text{Hb}}(\lambda_{N_\lambda}) & \varepsilon_{\text{HbO}_2}(\lambda_{N_\lambda}) \end{bmatrix} \begin{bmatrix} c_{\text{Hb}}(\mathbf{r}) \\ c_{\text{HbO}_2}(\mathbf{r}) \end{bmatrix}$$

**Least-Squares Solution:**
$$\hat{\mathbf{c}} = \mathbf{E}^{\dagger} \mathbf{s} = (\mathbf{E}^T\mathbf{E})^{-1}\mathbf{E}^T \mathbf{s}$$

#### 1.1.4 Oxygen Saturation Calculation

$$\text{sO}_2(\mathbf{r}) = \frac{c_{\text{HbO}_2}(\mathbf{r})}{c_{\text{Hb}}(\mathbf{r}) + c_{\text{HbO}_2}(\mathbf{r})} \cdot \mathbb{1}\left[c_{\text{Hb}}(\mathbf{r}) + c_{\text{HbO}_2}(\mathbf{r}) > \tau\right]$$

Where threshold: $\tau = 0.1 \cdot \max_{\mathbf{r}}\left(c_{\text{Hb}}(\mathbf{r}) + c_{\text{HbO}_2}(\mathbf{r})\right)$

---

## 2. Variables Taxonomy

### 2.1 Observed Data / Inputs

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `raw_data` | $\mathbf{Y}_{\text{raw}}$ | $(N_\lambda, N_{\text{det}}, N_t)$ | `float64` | Raw time-series PA signals |
| `geometry` | $\mathbf{R}_{\text{det}}$ | $(N_{\text{det}}, 3)$ | `float64` | Detector positions $(x_d, y_d, z_d)$ |
| `wavelengths` | $\boldsymbol{\lambda}$ | $(N_\lambda,)$ | `float64` | Illumination wavelengths [nm] |
| `fs` | $f_s$ | scalar | `float64` | Sampling frequency [Hz] |
| `speed_of_sound` | $c$ | scalar | `float64` | Speed of sound [m/s] |

### 2.2 Hyperparameters / Constants

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| `lp_filter` | $f_{\text{high}}$ | $7 \times 10^6$ Hz | Bandpass filter upper cutoff |
| `hp_filter` | $f_{\text{low}}$ | $5 \times 10^3$ Hz | Bandpass filter lower cutoff |
| `order` | $n_{\text{order}}$ | 4 | Butterworth filter order |
| `n_pixels` | $(N_x, N_y, N_z)$ | $(100, 100, 1)$ | Reconstruction grid dimensions |
| `field_of_view` | $(L_x, L_y, L_z)$ | $(0.025, 0.025, 0)$ m | Physical field of view |
| `sO2_threshold_ratio` | $\alpha_\tau$ | 0.1 | Threshold ratio for sO₂ mask |

### 2.3 Intermediate / Auxiliary Variables

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `filtered_data` | $\mathbf{Y}_{\text{filt}}$ | $(N_\lambda, N_{\text{det}}, N_t)$ | `float64` | Bandpass filtered signals |
| `analytic_signal` | $\mathbf{Y}_a$ | $(N_\lambda, N_{\text{det}}, N_t)$ | `complex128` | Analytic signal (Hilbert transform) |
| `processed_data` | $\mathbf{Y}$ | $(N_\lambda, N_{\text{det}}, N_t)$ | `float64` | Preprocessed signals $= \text{Im}(\mathbf{Y}_a)$ |
| `dist` | $D_{d,i,j,l}$ | $(N_z, N_y, N_x)$ | `float64` | Distance from voxel to detector |
| `sample_idx` | $k_{d,i,j,l}$ | $(N_z, N_y, N_x)$ | `int` | Time sample index for each voxel |
| `E` | $\mathbf{E}$ | $(N_\lambda, 2)$ | `float64` | Absorption spectra matrix |
| `E_inv` | $\mathbf{E}^\dagger$ | $(2, N_\lambda)$ | `float64` | Pseudoinverse of $\mathbf{E}$ |

### 2.4 Output / Primal Variables

| Variable | Symbol | Shape | Dtype | Initialization |
|----------|--------|-------|-------|----------------|
| `reconstruction` | $\hat{\mathbf{P}}_0$ | $(N_\lambda, N_z, N_y, N_x)$ | `float64` | `np.zeros(...)` |
| `concentrations` | $\mathbf{C}$ | $(2, N_z, N_y, N_x)$ | `float64` | Computed from unmixing |
| `so2` | $\mathbf{S}_{\text{O}_2}$ | $(N_z, N_y, N_x)$ | `float64` | `np.zeros(...)` |

---

## 3. Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Data Loading
**Input:** HDF5 file path  
**Output:** $\mathbf{Y}_{\text{raw}} \in \mathbb{R}^{N_\lambda \times N_{\text{det}} \times N_t}$, metadata

### Step 2: Butterworth Bandpass Filter Design
**Input:** $f_{\text{low}}, f_{\text{high}}, f_s, n_{\text{order}}$  
**Output:** Filter coefficients $(b, a)$

$$f_{\text{nyq}} = \frac{f_s}{2}$$

$$\omega_{\text{low}} = \frac{f_{\text{low}}}{f_{\text{nyq}}}, \quad \omega_{\text{high}} = \frac{f_{\text{high}}}{f_{\text{nyq}}}$$

Transfer function (Butterworth bandpass):
$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega^2 - \omega_0^2}{\omega \cdot BW}\right)^{2n}}$$

### Step 3: Zero-Phase Bandpass Filtering
**Input:** $\mathbf{Y}_{\text{raw}}$, $(b, a)$  
**Output:** $\mathbf{Y}_{\text{filt}} \in \mathbb{R}^{N_\lambda \times N_{\text{det}} \times N_t}$

$$\mathbf{Y}_{\text{filt}} = \text{filtfilt}(b, a, \mathbf{Y}_{\text{raw}}, \text{axis}=-1)$$

The `filtfilt` operation applies forward-backward filtering:
$$\mathbf{Y}_{\text{filt}} = \mathcal{F}^{-1}\left\{|H(\omega)|^2 \cdot \mathcal{F}\{\mathbf{Y}_{\text{raw}}\}\right\}$$

### Step 4: Hilbert Transform (Envelope Detection)
**Input:** $\mathbf{Y}_{\text{filt}}$  
**Output:** $\mathbf{Y} \in \mathbb{R}^{N_\lambda \times N_{\text{det}} \times N_t}$ (imaginary part of analytic signal)

Analytic signal:
$$\mathbf{Y}_a[n] = \mathbf{Y}_{\text{filt}}[n] + j \cdot \mathcal{H}\{\mathbf{Y}_{\text{filt}}\}[n]$$

Where Hilbert transform:
$$\mathcal{H}\{y\}(t) = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{y(\tau)}{t - \tau} d\tau$$

Preprocessed output:
$$\mathbf{Y} = \text{Im}(\mathbf{Y}_a)$$

### Step 5: Absorption Spectra Interpolation
**Input:** $\boldsymbol{\lambda}$ (wavelength array)  
**Output:** $\varepsilon_{\text{Hb}}(\boldsymbol{\lambda})$, $\varepsilon_{\text{HbO}_2}(\boldsymbol{\lambda})$

Reference data points:
| $\lambda$ [nm] | 700 | 730 | 760 | 800 | 850 | 900 |
|----------------|-----|-----|-----|-----|-----|-----|
| $\varepsilon_{\text{Hb}}$ | 100 | 80 | 60 | 40 | 30 | 20 |
| $\varepsilon_{\text{HbO}_2}$ | 30 | 40 | 50 | 60 | 70 | 80 |

Linear interpolation:
$$\varepsilon(\lambda) = \varepsilon(\lambda_i) + \frac{\varepsilon(\lambda_{i+1}) - \varepsilon(\lambda_i)}{\lambda_{i+1} - \lambda_i}(\lambda - \lambda_i)$$

---

## 4. Algorithm Steps

### 4.1 Delay-and-Sum Backprojection Reconstruction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ALGORITHM 1: DELAY-AND-SUM BACKPROJECTION                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Objective:** Reconstruct initial pressure distribution $\hat{\mathbf{P}}_0$ from measured signals $\mathbf{Y}$

---

#### Step 1: Grid Initialization
**Name:** Spatial Grid Construction  
**Input:** $(N_x, N_y, N_z)$, $(L_x, L_y, L_z)$  
**Output:** Coordinate arrays $\mathbf{X}, \mathbf{Y}, \mathbf{Z}$

$$x_i = -\frac{L_x}{2} + \frac{i \cdot L_x}{N_x - 1}, \quad i = 0, \ldots, N_x - 1$$

$$y_j = -\frac{L_y}{2} + \frac{j \cdot L_y}{N_y - 1}, \quad j = 0, \ldots, N_y - 1$$

$$z_l = 0 \quad \text{(single slice)}$$

---

#### Step 2: Spatial Resolution Computation
**Name:** Delay Step Calculation  
**Input:** $c$, $f_s$  
**Output:** $\Delta l$ (spatial resolution per sample)

$$\Delta l = \frac{c}{f_s}$$

---

#### Step 3: Reconstruction Array Initialization
**Name:** Output Buffer Initialization  
**Input:** $(N_\lambda, N_z, N_y, N_x)$  
**Output:** $\hat{\mathbf{P}}_0 \leftarrow \mathbf{0}$

$$\hat{P}_0^{(\lambda)}[l, j, i] = 0, \quad \forall \lambda, l, j, i$$

---

#### Step 4: Backprojection Loop
**Name:** Delay-and-Sum Accumulation  
**Input:** $\mathbf{Y}^{(\lambda)}$, $\mathbf{R}_{\text{det}}$, $\mathbf{X}, \mathbf{Y}, \mathbf{Z}$, $\Delta l$  
**Output:** $\hat{\mathbf{P}}_0^{(\lambda)}$

**For** $\lambda = 1$ to $N_\lambda$:  
&emsp;**For** $d = 1$ to $N_{\text{det}}$:  
&emsp;&emsp;**4.1** Compute distance field:
$$D[l, j, i] = \sqrt{(x_i - x_d)^2 + (y_j - y_d)^2 + (z_l - z_d)^2}$$

&emsp;&emsp;**4.2** Compute sample indices:
$$k[l, j, i] = \left\lfloor \frac{D[l, j, i]}{\Delta l} \right\rfloor$$

&emsp;&emsp;**4.3** Create validity mask:
$$M[l, j, i] = \mathbb{1}\left[0 \leq k[l, j, i] < N_t\right]$$

&emsp;&emsp;**4.4** Accumulate contributions:
$$\hat{P}_0^{(\lambda)}[l, j, i] \mathrel{+}= Y_d^{(\lambda)}\left[k[l, j, i]\right] \cdot M[l, j, i]$$

---

### 4.2 Linear Spectral Unmixing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ALGORITHM 2: LINEAR SPECTRAL UNMIXING                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Step 1: Absorption Matrix Construction
**Name:** Endmember Matrix Assembly  
**Input:** $\varepsilon_{\text{Hb}}(\boldsymbol{\lambda})$, $\varepsilon_{\text{HbO}_2}(\boldsymbol{\lambda})$  
**Output:** $\mathbf{E} \in \mathbb{R}^{N_\lambda \times 2}$

$$\mathbf{E} = \begin{bmatrix} \varepsilon_{\text{Hb}}(\lambda_1) & \varepsilon_{\text{HbO}_2}(\lambda_1) \\ \vdots & \vdots \\ \varepsilon_{\text{Hb}}(\lambda_{N_\lambda}) & \varepsilon_{\text{HbO}_2}(\lambda_{N_\lambda}) \end{bmatrix}$$

---

#### Step 2: Signal Matrix Reshaping
**Name:** Vectorization  
**Input:** $\hat{\mathbf{P}}_0 \in \mathbb{R}^{N_\lambda \times N_z \times N_y \times N_x}$  
**Output:** $\mathbf{S} \in \mathbb{R}^{N_\lambda \times (N_z \cdot N_y \cdot N_x)}$

$$\mathbf{S} = \text{reshape}(\hat{\mathbf{P}}_0, [N_\lambda, N_z \cdot N_y \cdot N_x])$$

---

#### Step 3: Pseudoinverse Computation
**Name:** Moore-Penrose Pseudoinverse  
**Input:** $\mathbf{E}$  
**Output:** $\mathbf{E}^\dagger \in \mathbb{R}^{2 \times N_\lambda}$

$$\mathbf{E}^\dagger = (\mathbf{E}^T \mathbf{E})^{-1} \mathbf{E}^T$$

---

#### Step 4: Concentration Estimation
**Name:** Least-Squares Solution  
**Input:** $\mathbf{E}^\dagger$, $\mathbf{S}$  
**Output:** $\mathbf{C} \in \mathbb{R}^{2 \times N_z \times N_y \times N_x}$

$$\mathbf{C}_{\text{flat}} = \mathbf{E}^\dagger \mathbf{S}$$

$$\mathbf{C} = \text{reshape}(\mathbf{C}_{\text{flat}}, [2, N_z, N_y, N_x])$$

Where:
- $\mathbf{C}[0] = c_{\text{Hb}}$ (deoxygenated hemoglobin)
- $\mathbf{C}[1] = c_{\text{HbO}_2}$ (oxygenated hemoglobin)

---

### 4.3 Oxygen Saturation Calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ALGORITHM 3: sO₂ COMPUTATION                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Step 1: Total Hemoglobin Computation
**Name:** Total Hb Calculation  
**Input:** $\mathbf{C}$  
**Output:** $\mathbf{T}_{\text{Hb}}$

$$T_{\text{Hb}}[l, j, i] = c_{\text{Hb}}[l, j, i] + c_{\text{HbO}_2}[l, j, i]$$

---

#### Step 2: Threshold Mask Generation
**Name:** ROI Masking  
**Input:** $\mathbf{T}_{\text{Hb}}$, $\alpha_\tau = 0.1$  
**Output:** Binary mask $\mathbf{M}$

$$\tau = \alpha_\tau \cdot \max(\mathbf{T}_{\text{Hb}})$$

$$M[l, j, i] = \mathbb{1}\left[T_{\text{Hb}}[l, j, i] > \tau\right]$$

---

#### Step 3: sO₂ Computation
**Name:** Saturation Ratio  
**Input:** $\mathbf{C}$, $\mathbf{M}$  
**Output:** $\mathbf{S}_{\text{O}_2}$

$$S_{\text{O}_2}[l, j, i] = \begin{cases} \frac{c_{\text{HbO}_2}[l, j, i]}{T_{\text{Hb}}[l, j, i]} & \text{if } M[l, j, i] = 1 \\ 0 & \text{otherwise} \end{cases}$$

---

#### Step 4: Value Clipping
**Name:** Range Constraint  
**Input:** $\mathbf{S}_{\text{O}_2}$  
**Output:** $\mathbf{S}_{\text{O}_2}$ (clipped)

$$S_{\text{O}_2}[l, j, i] = \text{clip}(S_{\text{O}_2}[l, j, i], 0, 1)$$

---

## 5. Stopping Criterion

This algorithm is **non-iterative** (direct reconstruction). The pipeline terminates upon completion of:

| Stage | Termination Condition |
|-------|----------------------|
| Preprocessing | All $N_t$ samples filtered and transformed |
| Reconstruction | All $N_\lambda \times N_{\text{det}}$ detector-wavelength pairs backprojected |
| Spectral Unmixing | Matrix multiplication completed for all $N_z \cdot N_y \cdot N_x$ voxels |
| sO₂ Calculation | Ratio computed for all masked voxels |

**No convergence criterion** is required as this is a closed-form solution (backprojection + linear algebra).

---

## 6. Evaluation Metrics

### 6.1 Mean sO₂ in Region of Interest

$$\bar{S}_{\text{O}_2} = \frac{1}{|\Omega_{\text{ROI}}|} \sum_{(l,j,i) \in \Omega_{\text{ROI}}} S_{\text{O}_2}[l, j, i]$$

Where:
$$\Omega_{\text{ROI}} = \{(l, j, i) : S_{\text{O}_2}[l, j, i] > 0\}$$

### 6.2 Reconstruction Dynamic Range

$$P_{\max} = \max_{l,j,i,\lambda} \hat{P}_0^{(\lambda)}[l, j, i]$$

$$P_{\min} = \min_{l,j,i,\lambda} \hat{P}_0^{(\lambda)}[l, j, i]$$

**Dynamic Range:** $[P_{\min}, P_{\max}]$

### 6.3 Implicit Quality Metrics (from reconstruction)

| Metric | Formula |
|--------|---------|
| **Signal-to-Noise Ratio (implicit)** | Depends on preprocessing filter design |
| **Spatial Resolution** | $\Delta x = \frac{L_x}{N_x - 1}$, $\Delta y = \frac{L_y}{N_y - 1}$ |
| **Temporal-to-Spatial Mapping** | $\Delta l = \frac{c}{f_s}$ |

---

## 7. Complete Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE ALGORITHM FLOWCHART                        │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌───────────────┐
                            │  HDF5 Input   │
                            │  Y_raw, meta  │
                            └───────┬───────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   PREPROCESSING               │
                    │   ─────────────               │
                    │   1. Butterworth BP Filter    │
                    │   2. Zero-phase filtering     │
                    │   3. Hilbert transform        │
                    │   4. Extract Im(analytic)     │
                    └───────────────┬───────────────┘
                                    │
                            Y ∈ ℝ^(Nλ×Ndet×Nt)
                                    │
                    ┌───────────────▼───────────────┐
                    │   RECONSTRUCTION              │
                    │   ──────────────              │
                    │   Delay-and-Sum Backproj.     │
                    │   ∀λ,d: accumulate delayed    │
                    │   signal contributions        │
                    └───────────────┬───────────────┘
                                    │
                          P̂₀ ∈ ℝ^(Nλ×Nz×Ny×Nx)
                                    │
                    ┌───────────────▼───────────────┐
                    │   SPECTRAL UNMIXING           │
                    │   ─────────────────           │
                    │   C = E† · S                  │
                    │   (Moore-Penrose solution)    │
                    └───────────────┬───────────────┘
                                    │
                          C ∈ ℝ^(2×Nz×Ny×Nx)
                          [c_Hb, c_HbO2]
                                    │
                    ┌───────────────▼───────────────┐
                    │   sO₂ CALCULATION             │
                    │   ───────────────             │
                    │   sO₂ = HbO₂/(Hb+HbO₂)       │
                    │   with thresholding & clip    │
                    └───────────────┬───────────────┘
                                    │
                          sO₂ ∈ [0,1]^(Nz×Ny×Nx)
                                    │
                    ┌───────────────▼───────────────┐
                    │   EVALUATION                  │
                    │   ──────────                  │
                    │   • Mean sO₂ in ROI           │
                    │   • Reconstruction range      │
                    │   • Visualization output      │
                    └───────────────────────────────┘
```

---

## 8. Notation Reference

| Symbol | Description |
|--------|-------------|
| $N_\lambda$ | Number of wavelengths |
| $N_{\text{det}}$ | Number of detectors |
| $N_t$ | Number of time samples |
| $N_x, N_y, N_z$ | Reconstruction grid dimensions |
| $L_x, L_y, L_z$ | Physical field of view [m] |
| $f_s$ | Sampling frequency [Hz] |
| $c$ | Speed of sound [m/s] |
| $\Delta l$ | Spatial resolution per sample [m] |
| $\mathbf{E}$ | Absorption endmember matrix |
| $\mathbf{E}^\dagger$ | Moore-Penrose pseudoinverse of $\mathbf{E}$ |
| $\varepsilon_{\text{Hb}}(\lambda)$ | Hb absorption coefficient at wavelength $\lambda$ |
| $\varepsilon_{\text{HbO}_2}(\lambda)$ | HbO₂ absorption coefficient at wavelength $\lambda$ |
| $\mathcal{H}\{\cdot\}$ | Hilbert transform operator |
| $\mathbb{1}[\cdot]$ | Indicator function |