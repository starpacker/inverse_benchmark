## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Delay-and-Sum Backprojection with Linear Spectral Unmixing",
  "algorithm_family": "Direct Reconstruction / Tomographic Imaging",
  "summary": "Reconstructs initial pressure distribution from multi-wavelength photoacoustic time-series using delay-and-sum backprojection, then performs linear spectral unmixing via pseudo-inverse to estimate hemoglobin concentrations and oxygen saturation."
}
```

---

## 1. OBJECTIVE FUNCTION

**full_expression**:
The Delay-and-Sum (DAS) backprojection is a direct inversion method that approximates the adjoint of the forward operator. The implicit inverse problem is:

$$\hat{p}_0 = \arg\min_{p_0} \|y - \mathcal{A}p_0\|_2^2$$

where $\mathcal{A}$ is the photoacoustic forward operator. DAS computes $\hat{p}_0 \approx \mathcal{A}^* y$ (adjoint application).

For spectral unmixing, the objective is:

$$\hat{C} = \arg\min_{C} \|S - E \cdot C\|_F^2$$

with closed-form solution $\hat{C} = E^\dagger S$ where $E^\dagger$ is the Moore-Penrose pseudo-inverse.

**data_fidelity_term**:
- DAS Backprojection: $\sum_{d=1}^{N_{\text{det}}} \sum_{\mathbf{r}} \left| y_d(t_{\mathbf{r},d}) - p_0(\mathbf{r}) \right|^2$ where $t_{\mathbf{r},d} = \|\mathbf{r} - \mathbf{r}_d\| / c$
- Spectral Unmixing: $\|S - E \cdot C\|_F^2$ (Frobenius norm, L2)

**regularization_term**: None (direct method without explicit regularization)

**constraints**: None explicitly enforced during reconstruction; sO2 output clipped: $\text{sO2} \in [0, 1]$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\hat{p}_0^{(\lambda)}$ | $(N_z, N_y, N_x)$ | float64 | zeros | Reconstructed initial pressure at wavelength $\lambda$ |
| $C$ | $(2, N_z, N_y, N_x)$ | float64 | computed | Chromophore concentrations: $C[0]=[\text{Hb}]$, $C[1]=[\text{HbO}_2]$ |
| $\text{sO}_2$ | $(N_z, N_y, N_x)$ | float64 | zeros | Oxygen saturation map |

### 2.2 Dual Variables

None (direct reconstruction method)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $X$ | $(N_z, N_y, N_x)$ | float64 | meshgrid | x-coordinates of reconstruction grid |
| $Y$ | $(N_z, N_y, N_x)$ | float64 | meshgrid | y-coordinates of reconstruction grid |
| $Z$ | $(N_z, N_y, N_x)$ | float64 | meshgrid | z-coordinates of reconstruction grid |
| $d_{i,\mathbf{r}}$ | $(N_z, N_y, N_x)$ | float64 | computed per detector | Distance from detector $i$ to each voxel |
| $\tau_{i,\mathbf{r}}$ | $(N_z, N_y, N_x)$ | int | computed per detector | Sample index for time-of-flight lookup |
| $M_{\text{valid}}$ | $(N_z, N_y, N_x)$ | bool | computed per detector | Valid sample index mask |
| $E$ | $(N_\lambda, 2)$ | float64 | from spectra | Extinction coefficient matrix |
| $E^\dagger$ | $(2, N_\lambda)$ | float64 | computed | Pseudo-inverse of extinction matrix |
| $S$ | $(N_\lambda, N_z \cdot N_y \cdot N_x)$ | float64 | reshaped reconstruction | Flattened spectral measurements |
| $H_{\text{total}}$ | $(N_z, N_y, N_x)$ | float64 | computed | Total hemoglobin: $[\text{Hb}] + [\text{HbO}_2]$ |
| $M_{\text{sO2}}$ | $(N_z, N_y, N_x)$ | bool | computed | Mask for valid sO2 computation |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_x$ | 100 | Number of pixels in x-direction |
| $N_y$ | 100 | Number of pixels in y-direction |
| $N_z$ | 1 | Number of pixels in z-direction |
| $L_x$ | 0.025 m | Field of view in x-direction |
| $L_y$ | 0.025 m | Field of view in y-direction |
| $L_z$ | 0 m | Field of view in z-direction |
| $f_{\text{low}}$ | 5000 Hz | Bandpass filter low cutoff |
| $f_{\text{high}}$ | 7×10⁶ Hz | Bandpass filter high cutoff |
| $n_{\text{order}}$ | 4 | Butterworth filter order |
| $\lambda_{\text{ref}}$ | [700, 730, 760, 800, 850, 900] nm | Reference wavelengths for spectra |
| $\epsilon_{\text{Hb,ref}}$ | [100, 80, 60, 40, 30, 20] | Hb extinction at reference wavelengths |
| $\epsilon_{\text{HbO2,ref}}$ | [30, 40, 50, 60, 70, 80] | HbO₂ extinction at reference wavelengths |
| $\theta_{\text{mask}}$ | 0.1 | Threshold fraction for sO2 mask |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y_{\text{raw}}$ | $(N_\lambda, N_{\text{det}}, N_t)$ | float64 | Raw photoacoustic time-series data |
| $\mathbf{g}$ | $(N_{\text{det}}, 3)$ | float64 | Detector positions (x, y, z) |
| $f_s$ | scalar | float64 | Sampling frequency (Hz) |
| $c$ | scalar | float64 | Speed of sound in medium (m/s) |
| $\boldsymbol{\lambda}$ | $(N_\lambda,)$ | float64 | Wavelengths used in acquisition |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Load HDF5 Data | $y_{\text{raw}}, f_s, \mathbf{g}, c, \boldsymbol{\lambda} \leftarrow \text{HDF5\_read}(\texttt{filename})$ | filename (string) | $y_{\text{raw}}$: $(N_\lambda, N_{\text{det}}, N_t)$, $\mathbf{g}$: $(N_{\text{det}}, 3)$, $f_s$: scalar, $c$: scalar, $\boldsymbol{\lambda}$: $(N_\lambda,)$ | File exists, valid PATATO format |
| P2 | Squeeze Batch Dimension | $y_{\text{raw}} \leftarrow y_{\text{raw}}[0]$ if $y_{\text{raw}}.\text{ndim}=4$ and $y_{\text{raw}}.\text{shape}[0]=1$ | $y_{\text{raw}}$: $(1, N_\lambda, N_{\text{det}}, N_t)$ | $y_{\text{raw}}$: $(N_\lambda, N_{\text{det}}, N_t)$ | Batch dimension equals 1 |
| P3 | Compute Nyquist Frequency | $f_{\text{nyq}} = 0.5 \cdot f_s$ | $f_s$: scalar | $f_{\text{nyq}}$: scalar | — |
| P4 | Compute Normalized Cutoffs | $\omega_{\text{low}} = \frac{f_{\text{low}}}{f_{\text{nyq}}}, \quad \omega_{\text{high}} = \frac{f_{\text{high}}}{f_{\text{nyq}}}$ | $f_{\text{low}}=5000$, $f_{\text{high}}=7\times10^6$, $f_{\text{nyq}}$ | $\omega_{\text{low}}$: scalar, $\omega_{\text{high}}$: scalar | $0 < \omega_{\text{low}} < \omega_{\text{high}} < 1$ |
| P5 | Design Butterworth Filter | $(b, a) = \text{butter}(n_{\text{order}}, [\omega_{\text{low}}, \omega_{\text{high}}], \texttt{btype='band'})$ | $n_{\text{order}}=4$, $\omega_{\text{low}}$, $\omega_{\text{high}}$ | $b$: $(2 \cdot n_{\text{order}} + 1,)$, $a$: $(2 \cdot n_{\text{order}} + 1,)$ | Filter is stable |
| P6 | Apply Zero-Phase Bandpass Filter | $y_{\text{filt}}[i_\lambda, i_d, :] = \text{filtfilt}(b, a, y_{\text{raw}}[i_\lambda, i_d, :])$ for all $i_\lambda, i_d$ | $y_{\text{raw}}$: $(N_\lambda, N_{\text{det}}, N_t)$, $b$, $a$ | $y_{\text{filt}}$: $(N_\lambda, N_{\text{det}}, N_t)$ | Signal length > filter order |
| P7 | Compute Analytic Signal | $y_{\text{analytic}}[i_\lambda, i_d, :] = y_{\text{filt}}[i_\lambda, i_d, :] + j \cdot \mathcal{H}\{y_{\text{filt}}[i_\lambda, i_d, :]\}$ | $y_{\text{filt}}$: $(N_\lambda, N_{\text{det}}, N_t)$ | $y_{\text{analytic}}$: $(N_\lambda, N_{\text{det}}, N_t)$, complex128 | $\mathcal{H}$ is Hilbert transform |
| P8 | Extract Imaginary Part | $y = \text{Im}(y_{\text{analytic}})$ | $y_{\text{analytic}}$: $(N_\lambda, N_{\text{det}}, N_t)$ | $y$: $(N_\lambda, N_{\text{det}}, N_t)$, float64 | — |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $x_s$ | $x_s[i] = -\frac{L_x}{2} + i \cdot \frac{L_x}{N_x - 1}$ for $i \in [0, N_x-1]$ | $(N_x,)$ | float64 | Uniform grid |
| 2 | $y_s$ | $y_s[j] = -\frac{L_y}{2} + j \cdot \frac{L_y}{N_y - 1}$ for $j \in [0, N_y-1]$ | $(N_y,)$ | float64 | Uniform grid |
| 3 | $z_s$ | $z_s = [0.0]$ | $(1,)$ | float64 | Single z-plane |
| 4 | $(Z, Y, X)$ | $Z, Y, X = \text{meshgrid}(z_s, y_s, x_s, \texttt{indexing='ij'})$ | each $(N_z, N_y, N_x)$ | float64 | 3D grid coordinates |
| 5 | $\Delta l$ | $\Delta l = \frac{c}{f_s}$ | scalar | float64 | Spatial sample spacing |
| 6 | $\hat{p}_0$ | $\hat{p}_0 = \mathbf{0}$ | $(N_\lambda, N_z, N_y, N_x)$ | float64 | Zero initialization |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP: wavelength_loop (i_λ from 0 to N_λ - 1)
```

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 7 | Extract wavelength signal | $y^{(\lambda)} = y[i_\lambda, :, :]$ | $y$: $(N_\lambda, N_{\text{det}}, N_t)$, $i_\lambda$: scalar | $y^{(\lambda)}$: $(N_{\text{det}}, N_t)$ | Slice along wavelength axis |

```
  BEGIN_LOOP: detector_loop (i_d from 0 to N_det - 1)
```

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 8 | Extract detector position | $\mathbf{r}_d = \mathbf{g}[i_d, :]$ | $\mathbf{g}$: $(N_{\text{det}}, 3)$, $i_d$: scalar | $\mathbf{r}_d$: $(3,)$ | $\mathbf{r}_d = (x_d, y_d, z_d)$ |
| 9 | Compute distance field | $d_{i_d}[k,j,i] = \sqrt{(X[k,j,i] - x_d)^2 + (Y[k,j,i] - y_d)^2 + (Z[k,j,i] - z_d)^2}$ | $X, Y, Z$: each $(N_z, N_y, N_x)$, $\mathbf{r}_d$: $(3,)$ | $d_{i_d}$: $(N_z, N_y, N_x)$ | Euclidean distance to detector |
| 10 | Compute sample indices | $\tau_{i_d}[k,j,i] = \lfloor d_{i_d}[k,j,i] / \Delta l \rfloor$ | $d_{i_d}$: $(N_z, N_y, N_x)$, $\Delta l$: scalar | $\tau_{i_d}$: $(N_z, N_y, N_x)$, int | Integer truncation for lookup |
| 11 | Compute valid mask | $M_{\text{valid}}[k,j,i] = (\tau_{i_d}[k,j,i] \geq 0) \land (\tau_{i_d}[k,j,i] < N_t)$ | $\tau_{i_d}$: $(N_z, N_y, N_x)$, $N_t$: scalar | $M_{\text{valid}}$: $(N_z, N_y, N_x)$, bool | Bounds check |
| 12 | Accumulate backprojection | $\hat{p}_0[i_\lambda, k, j, i] \mathrel{+}= y^{(\lambda)}[i_d, \tau_{i_d}[k,j,i]]$ for all $(k,j,i)$ where $M_{\text{valid}}[k,j,i] = \text{True}$ | $\hat{p}_0$: $(N_\lambda, N_z, N_y, N_x)$, $y^{(\lambda)}$: $(N_{\text{det}}, N_t)$, $\tau_{i_d}$, $M_{\text{valid}}$ | $\hat{p}_0$: $(N_\lambda, N_z, N_y, N_x)$ | [ADJOINT] Time-of-flight lookup and sum |

```
  END_LOOP: detector_loop
```

```
END_LOOP: wavelength_loop
```

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 13 | Interpolate Hb spectrum | $\epsilon_{\text{Hb}}(\lambda_i) = \text{interp1d}(\lambda_{\text{ref}}, \epsilon_{\text{Hb,ref}}, \texttt{kind='linear'})(\lambda_i)$ for each $\lambda_i \in \boldsymbol{\lambda}$ | $\lambda_{\text{ref}}$: $(6,)$, $\epsilon_{\text{Hb,ref}}$: $(6,)$, $\boldsymbol{\lambda}$: $(N_\lambda,)$ | $\epsilon_{\text{Hb}}$: $(N_\lambda,)$ | Linear interpolation with extrapolation |
| 14 | Interpolate HbO2 spectrum | $\epsilon_{\text{HbO2}}(\lambda_i) = \text{interp1d}(\lambda_{\text{ref}}, \epsilon_{\text{HbO2,ref}}, \texttt{kind='linear'})(\lambda_i)$ for each $\lambda_i \in \boldsymbol{\lambda}$ | $\lambda_{\text{ref}}$: $(6,)$, $\epsilon_{\text{HbO2,ref}}$: $(6,)$, $\boldsymbol{\lambda}$: $(N_\lambda,)$ | $\epsilon_{\text{HbO2}}$: $(N_\lambda,)$ | Linear interpolation with extrapolation |
| 15 | Build extinction matrix | $E = \begin{bmatrix} \epsilon_{\text{Hb}}(\lambda_1) & \epsilon_{\text{HbO2}}(\lambda_1) \\ \vdots & \vdots \\ \epsilon_{\text{Hb}}(\lambda_{N_\lambda}) & \epsilon_{\text{HbO2}}(\lambda_{N_\lambda}) \end{bmatrix}$ | $\epsilon_{\text{Hb}}$: $(N_\lambda,)$, $\epsilon_{\text{HbO2}}$: $(N_\lambda,)$ | $E$: $(N_\lambda, 2)$ | Vertical stack of spectra |
| 16 | Compute pseudo-inverse | $E^\dagger = (E^T E)^{-1} E^T$ | $E$: $(N_\lambda, 2)$ | $E^\dagger$: $(2, N_\lambda)$ | Moore-Penrose pseudo-inverse |
| 17 | Flatten reconstruction | $S = \text{reshape}(\hat{p}_0, (N_\lambda, N_z \cdot N_y \cdot N_x))$ | $\hat{p}_0$: $(N_\lambda, N_z, N_y, N_x)$ | $S$: $(N_\lambda, N_z \cdot N_y \cdot N_x)$ | Vectorize spatial dimensions |
| 18 | Linear unmixing | $C_{\text{flat}} = E^\dagger \cdot S$ | $E^\dagger$: $(2, N_\lambda)$, $S$: $(N_\lambda, N_z \cdot N_y \cdot N_x)$ | $C_{\text{flat}}$: $(2, N_z \cdot N_y \cdot N_x)$ | Matrix multiplication |
| 19 | Reshape concentrations | $C = \text{reshape}(C_{\text{flat}}, (2, N_z, N_y, N_x))$ | $C_{\text{flat}}$: $(2, N_z \cdot N_y \cdot N_x)$ | $C$: $(2, N_z, N_y, N_x)$ | $C[0]=[\text{Hb}]$, $C[1]=[\text{HbO}_2]$ |
| 20 | Extract Hb concentration | $[\text{Hb}] = C[0, :, :, :]$ | $C$: $(2, N_z, N_y, N_x)$ | $[\text{Hb}]$: $(N_z, N_y, N_x)$ | — |
| 21 | Extract HbO2 concentration | $[\text{HbO}_2] = C[1, :, :, :]$ | $C$: $(2, N_z, N_y, N_x)$ | $[\text{HbO}_2]$: $(N_z, N_y, N_x)$ | — |
| 22 | Compute total hemoglobin | $H_{\text{total}} = [\text{Hb}] + [\text{HbO}_2]$ | $[\text{Hb}]$, $[\text{HbO}_2]$: each $(N_z, N_y, N_x)$ | $H_{\text{total}}$: $(N_z, N_y, N_x)$ | Element-wise addition |
| 23 | Compute sO2 mask | $M_{\text{sO2}}[k,j,i] = (H_{\text{total}}[k,j,i] > 0.1 \cdot \max(H_{\text{total}}))$ | $H_{\text{total}}$: $(N_z, N_y, N_x)$, $\theta_{\text{mask}}=0.1$ | $M_{\text{sO2}}$: $(N_z, N_y, N_x)$, bool | Threshold mask |
| 24 | Initialize sO2 | $\text{sO}_2 = \mathbf{0}$ | — | $\text{sO}_2$: $(N_z, N_y, N_x)$ | Zero array |
| 25 | Compute sO2 values | $\text{sO}_2[k,j,i] = \frac{[\text{HbO}_2][k,j,i]}{H_{\text{total}}[k,j,i]}$ for all $(k,j,i)$ where $M_{\text{sO2}}[k,j,i] = \text{True}$ | $[\text{HbO}_2]$, $H_{\text{total}}$, $M_{\text{sO2}}$: each $(N_z, N_y, N_x)$ | $\text{sO}_2$: $(N_z, N_y, N_x)$ | Element-wise division where valid |
| 26 | Clip sO2 to valid range | $\text{sO}_2 = \text{clip}(\text{sO}_2, 0, 1)$ | $\text{sO}_2$: $(N_z, N_y, N_x)$ | $\text{sO}_2$: $(N_z, N_y, N_x)$ | Enforce physiological bounds |

---

## 5. STOPPING CRITERION

```json
{
  "type": "iteration_limit",
  "expression": "Deterministic termination after complete iteration over all wavelengths and detectors: $i_\\lambda = N_\\lambda - 1$ AND $i_d = N_{\\text{det}} - 1$",
  "parameters": {
    "N_lambda": "Number of wavelengths (from data)",
    "N_det": "Number of detectors (from geometry)"
  },
  "check_frequency": "Single pass - no convergence check required"
}
```

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Mean sO2 in ROI | $\bar{\text{sO}}_2 = \frac{1}{|R|} \sum_{(k,j,i) \in R} \text{sO}_2[k,j,i]$ where $R = \{(k,j,i) : \text{sO}_2[k,j,i] > 0\}$ | Ground truth sO2 (if available) | $\text{sO}_2[0,:,:]$: $(N_y, N_x)$ | Average oxygen saturation over valid pixels |
| Reconstruction Dynamic Range | $[\min(\hat{p}_0), \max(\hat{p}_0)]$ | — | $\hat{p}_0$: $(N_\lambda, N_z, N_y, N_x)$ | Range of reconstructed initial pressure values |
| Mean Reconstruction (Display) | $\bar{p}_0 = \frac{1}{N_\lambda} \sum_{i_\lambda=0}^{N_\lambda-1} \hat{p}_0[i_\lambda, 0, :, :]$ | — | $\bar{p}_0$: $(N_y, N_x)$ | Wavelength-averaged reconstruction for visualization |