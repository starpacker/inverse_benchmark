# Algorithmic Specification Document: Photoacoustic Tomography Reconstruction with Spectral Unmixing

## 1. OBJECTIVE FUNCTION

### 1.1 Reconstruction Stage (Delay-and-Sum Backprojection)

**full_expression**: Direct inversion method (non-iterative); implicitly solves the adjoint of the forward model

$$\hat{p}_\lambda(\mathbf{r}) = \sum_{d=1}^{N_{\text{det}}} y_\lambda\left(d, \frac{\|\mathbf{r} - \mathbf{r}_d\|}{c}\right)$$

where the forward model is:

$$y(d, t) = \int_{\Omega} p(\mathbf{r}) \, \delta\left(t - \frac{\|\mathbf{r} - \mathbf{r}_d\|}{c}\right) d\mathbf{r}$$

**data_fidelity_term**: Not explicitly minimized; backprojection is the adjoint operator

**regularization_term**: None

**constraints**: None (unconstrained direct method)

---

### 1.2 Spectral Unmixing Stage (Linear Least Squares)

**full_expression**:

$$\min_{\mathbf{C}} \|\mathbf{S} - \mathbf{E}\mathbf{C}\|_F^2$$

**data_fidelity_term**:

$$\mathcal{L}(\mathbf{C}) = \sum_{i=1}^{N_\lambda} \sum_{j=1}^{N_{\text{vox}}} \left( S_{ij} - \sum_{k=1}^{2} E_{ik} C_{kj} \right)^2$$

**regularization_term**: None

**constraints**: None (post-hoc clipping applied to derived sO₂)

**closed_form_solution**:

$$\mathbf{C} = \mathbf{E}^+ \mathbf{S}$$

where $\mathbf{E}^+ = (\mathbf{E}^\top \mathbf{E})^{-1} \mathbf{E}^\top$ is the Moore-Penrose pseudoinverse.

---

## 2. VARIABLES

### 2.1 Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{Y}_{\text{raw}}$ | observations | $(N_\lambda, N_{\text{det}}, N_t)$ | float64 | Loaded from HDF5 | Raw photoacoustic time-series signals |
| $\boldsymbol{\lambda}$ | observations | $(N_\lambda,)$ | float64 | Loaded from HDF5 | Optical wavelengths used for acquisition |
| $\mathbf{G}$ | observations | $(N_{\text{det}}, 3)$ | float64 | Loaded from HDF5 | Detector positions in 3D space $(x, y, z)$ |

### 2.2 Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $f_s$ | constants | scalar | float64 | Loaded from HDF5 | Sampling frequency (Hz) |
| $c$ | constants | scalar | float64 | Loaded from HDF5 | Speed of sound in medium (m/s) |
| $f_{\text{low}}$ | constants | scalar | float64 | $5 \times 10^3$ | High-pass cutoff frequency (Hz) |
| $f_{\text{high}}$ | constants | scalar | float64 | $7 \times 10^6$ | Low-pass cutoff frequency (Hz) |
| $n_{\text{order}}$ | constants | scalar | int | $4$ | Butterworth filter order |
| $L_x, L_y$ | constants | scalar | float64 | $0.025$ | Field of view dimensions (m) |
| $N_x, N_y, N_z$ | constants | scalar | int | $(100, 100, 1)$ | Reconstruction grid dimensions |
| $\Delta l$ | constants | scalar | float64 | $c / f_s$ | Spatial sampling interval (m) |

### 2.3 Derived Constants (Absorption Spectra)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\boldsymbol{\epsilon}_{\text{Hb}}$ | constants | $(N_\lambda,)$ | float64 | Interpolated from reference | Hb extinction coefficients at wavelengths $\boldsymbol{\lambda}$ |
| $\boldsymbol{\epsilon}_{\text{HbO}_2}$ | constants | $(N_\lambda,)$ | float64 | Interpolated from reference | HbO₂ extinction coefficients at wavelengths $\boldsymbol{\lambda}$ |
| $\mathbf{E}$ | constants | $(N_\lambda, 2)$ | float64 | $[\boldsymbol{\epsilon}_{\text{Hb}}, \boldsymbol{\epsilon}_{\text{HbO}_2}]$ | Extinction coefficient matrix |

### 2.4 Intermediate Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{Y}_{\text{filt}}$ | intermediate | $(N_\lambda, N_{\text{det}}, N_t)$ | float64 | Computed | Bandpass-filtered signals |
| $\mathbf{Y}$ | intermediate | $(N_\lambda, N_{\text{det}}, N_t)$ | float64 | Computed | Preprocessed signals (Hilbert imaginary) |

### 2.5 Primal Variables (Outputs)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\hat{\mathbf{P}}$ | primal | $(N_\lambda, N_z, N_y, N_x)$ | float64 | $\mathbf{0}$ | Reconstructed initial pressure distribution |
| $\mathbf{S}$ | primal | $(N_\lambda, N_{\text{vox}})$ | float64 | Reshaped from $\hat{\mathbf{P}}$ | Flattened reconstruction for unmixing |
| $\mathbf{C}$ | primal | $(2, N_z, N_y, N_x)$ | float64 | Computed | Chromophore concentrations $[C_{\text{Hb}}, C_{\text{HbO}_2}]$ |
| $\mathbf{sO}_2$ | primal | $(N_z, N_y, N_x)$ | float64 | $\mathbf{0}$ | Oxygen saturation map |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Butterworth Filter Design

| Field | Value |
|-------|-------|
| **step_order** | 1 |
| **step_name** | Butterworth Bandpass Filter Coefficient Computation |
| **formula** | Normalized cutoff frequencies: $$\omega_{\text{low}} = \frac{f_{\text{low}}}{f_s / 2}, \quad \omega_{\text{high}} = \frac{f_{\text{high}}}{f_s / 2}$$ Transfer function (Butterworth bandpass, order $n$): $$H(z) = \frac{B(z)}{A(z)} = \frac{\sum_{k=0}^{2n} b_k z^{-k}}{\sum_{k=0}^{2n} a_k z^{-k}}$$ Coefficients $(b_k, a_k)$ computed via bilinear transform of analog prototype. |
| **inputs** | $f_{\text{low}}$: scalar, $f_{\text{high}}$: scalar, $f_s$: scalar, $n_{\text{order}}$: scalar |
| **outputs** | $\mathbf{b}$: $(2n_{\text{order}}+1,)$, $\mathbf{a}$: $(2n_{\text{order}}+1,)$ |

### Step 3.2: Zero-Phase Bandpass Filtering

| Field | Value |
|-------|-------|
| **step_order** | 2 |
| **step_name** | Zero-Phase Digital Filtering (filtfilt) |
| **formula** | Forward-backward filtering to achieve zero phase distortion: $$\mathbf{Y}_{\text{filt}}[\lambda, d, :] = \text{reverse}\left( H\left( \text{reverse}\left( H\left( \mathbf{Y}_{\text{raw}}[\lambda, d, :] \right) \right) \right) \right)$$ where $H(\cdot)$ denotes causal IIR filtering: $$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$ |
| **inputs** | $\mathbf{Y}_{\text{raw}}$: $(N_\lambda, N_{\text{det}}, N_t)$, $\mathbf{b}$: $(2n_{\text{order}}+1,)$, $\mathbf{a}$: $(2n_{\text{order}}+1,)$ |
| **outputs** | $\mathbf{Y}_{\text{filt}}$: $(N_\lambda, N_{\text{det}}, N_t)$ |

### Step 3.3: Hilbert Transform and Imaginary Part Extraction

| Field | Value |
|-------|-------|
| **step_order** | 3 |
| **step_name** | Analytic Signal Computation via Hilbert Transform |
| **formula** | Analytic signal: $$\mathbf{Y}_a[\lambda, d, :] = \mathbf{Y}_{\text{filt}}[\lambda, d, :] + j \cdot \mathcal{H}\{\mathbf{Y}_{\text{filt}}[\lambda, d, :]\}$$ where the Hilbert transform is: $$\mathcal{H}\{x\}[n] = \mathcal{F}^{-1}\left\{ X[k] \cdot H[k] \right\}$$ with $$H[k] = \begin{cases} 0 & k = 0 \\ 2 & 1 \leq k \leq N/2 - 1 \\ 0 & k = N/2 \\ 0 & k > N/2 \end{cases} - 1$$ Preprocessed signal (imaginary part): $$\mathbf{Y}[\lambda, d, :] = \text{Im}\left( \mathbf{Y}_a[\lambda, d, :] \right) = \mathcal{H}\{\mathbf{Y}_{\text{filt}}[\lambda, d, :]\}$$ |
| **inputs** | $\mathbf{Y}_{\text{filt}}$: $(N_\lambda, N_{\text{det}}, N_t)$ |
| **outputs** | $\mathbf{Y}$: $(N_\lambda, N_{\text{det}}, N_t)$ |

### Step 3.4: Absorption Spectra Interpolation

| Field | Value |
|-------|-------|
| **step_order** | 4 |
| **step_name** | Linear Interpolation of Extinction Coefficients |
| **formula** | Reference data points: $$\boldsymbol{\lambda}_{\text{ref}} = [700, 730, 760, 800, 850, 900] \text{ nm}$$ $$\boldsymbol{\epsilon}_{\text{Hb,ref}} = [100, 80, 60, 40, 30, 20]$$ $$\boldsymbol{\epsilon}_{\text{HbO}_2\text{,ref}} = [30, 40, 50, 60, 70, 80]$$ Linear interpolation for each wavelength $\lambda_i$: $$\epsilon(\lambda_i) = \epsilon[\lambda_j] + \frac{\lambda_i - \lambda_j}{\lambda_{j+1} - \lambda_j} \left( \epsilon[\lambda_{j+1}] - \epsilon[\lambda_j] \right)$$ where $\lambda_j \leq \lambda_i < \lambda_{j+1}$. Matrix assembly: $$\mathbf{E} = \begin{bmatrix} \epsilon_{\text{Hb}}(\lambda_1) & \epsilon_{\text{HbO}_2}(\lambda_1) \\ \vdots & \vdots \\ \epsilon_{\text{Hb}}(\lambda_{N_\lambda}) & \epsilon_{\text{HbO}_2}(\lambda_{N_\lambda}) \end{bmatrix}$$ |
| **inputs** | $\boldsymbol{\lambda}$: $(N_\lambda,)$, $\boldsymbol{\lambda}_{\text{ref}}$: $(6,)$, $\boldsymbol{\epsilon}_{\text{Hb,ref}}$: $(6,)$, $\boldsymbol{\epsilon}_{\text{HbO}_2\text{,ref}}$: $(6,)$ |
| **outputs** | $\mathbf{E}$: $(N_\lambda, 2)$ |

### Step 3.5: Reconstruction Grid Generation

| Field | Value |
|-------|-------|
| **step_order** | 5 |
| **step_name** | Cartesian Grid Coordinate Generation |
| **formula** | 1D coordinate vectors: $$x_i = -\frac{L_x}{2} + \frac{i}{N_x - 1} L_x, \quad i = 0, \ldots, N_x - 1$$ $$y_j = -\frac{L_y}{2} + \frac{j}{N_y - 1} L_y, \quad j = 0, \ldots, N_y - 1$$ $$z_k = 0, \quad k = 0$$ 3D coordinate arrays via meshgrid (ij-indexing): $$\mathbf{X}[k,j,i] = x_i, \quad \mathbf{Y}[k,j,i] = y_j, \quad \mathbf{Z}[k,j,i] = z_k$$ |
| **inputs** | $L_x, L_y$: scalars, $N_x, N_y, N_z$: scalars |
| **outputs** | $\mathbf{X}$: $(N_z, N_y, N_x)$, $\mathbf{Y}$: $(N_z, N_y, N_x)$, $\mathbf{Z}$: $(N_z, N_y, N_x)$ |

---

## 4. ALGORITHM STEPS

### Initialization

| Field | Value |
|-------|-------|
| **step_name** | Initialize Reconstruction Array |
| **formula** | $$\hat{\mathbf{P}} \leftarrow \mathbf{0} \in \mathbb{R}^{N_\lambda \times N_z \times N_y \times N_x}$$ |
| **inputs** | $N_\lambda, N_z, N_y, N_x$: scalars |
| **outputs** | $\hat{\mathbf{P}}$: $(N_\lambda, N_z, N_y, N_x)$ |
| **computational_note** | Allocate zero-initialized array |

---

### BEGIN ITERATION LOOP

**Loop structure**: Nested loops over wavelengths and detectors

$$\text{for } \lambda = 1, \ldots, N_\lambda \text{ do}$$
$$\quad \text{for } d = 1, \ldots, N_{\text{det}} \text{ do}$$

---

#### Step 4.1: Distance Computation

| Field | Value |
|-------|-------|
| **step_name** | Compute Euclidean Distance from Detector to All Grid Points |
| **formula** | $$\mathbf{D}^{(d)}[k,j,i] = \sqrt{(\mathbf{X}[k,j,i] - G_{d,x})^2 + (\mathbf{Y}[k,j,i] - G_{d,y})^2 + (\mathbf{Z}[k,j,i] - G_{d,z})^2}$$ |
| **inputs** | $\mathbf{X}, \mathbf{Y}, \mathbf{Z}$: $(N_z, N_y, N_x)$, $\mathbf{G}[d,:]$: $(3,)$ |
| **outputs** | $\mathbf{D}^{(d)}$: $(N_z, N_y, N_x)$ |
| **computational_note** | Vectorized over all grid points |

#### Step 4.2: Time-to-Sample Index Mapping

| Field | Value |
|-------|-------|
| **step_name** | Convert Distance to Sample Index |
| **formula** | Travel time from grid point to detector: $$\tau[k,j,i] = \frac{\mathbf{D}^{(d)}[k,j,i]}{c}$$ Sample index (integer truncation): $$\mathbf{I}^{(d)}[k,j,i] = \left\lfloor \frac{\mathbf{D}^{(d)}[k,j,i]}{\Delta l} \right\rfloor = \left\lfloor \frac{\mathbf{D}^{(d)}[k,j,i] \cdot f_s}{c} \right\rfloor$$ |
| **inputs** | $\mathbf{D}^{(d)}$: $(N_z, N_y, N_x)$, $\Delta l$: scalar |
| **outputs** | $\mathbf{I}^{(d)}$: $(N_z, N_y, N_x)$, dtype=int |
| **computational_note** | Floor operation via integer casting |

#### Step 4.3: Validity Mask Generation

| Field | Value |
|-------|-------|
| **step_name** | Compute Valid Index Mask |
| **formula** | $$\mathbf{M}^{(d)}[k,j,i] = \begin{cases} 1 & \text{if } 0 \leq \mathbf{I}^{(d)}[k,j,i] < N_t \\ 0 & \text{otherwise} \end{cases}$$ |
| **inputs** | $\mathbf{I}^{(d)}$: $(N_z, N_y, N_x)$, $N_t$: scalar |
| **outputs** | $\mathbf{M}^{(d)}$: $(N_z, N_y, N_x)$, dtype=bool |
| **computational_note** | Boolean mask for bounds checking |

#### Step 4.4: Delay-and-Sum Accumulation

| Field | Value |
|-------|-------|
| **step_name** | Backproject Signal to Grid Points |
| **formula** | For all valid grid points: $$\hat{\mathbf{P}}[\lambda, k, j, i] \mathrel{+}= \mathbf{Y}[\lambda, d, \mathbf{I}^{(d)}[k,j,i]] \cdot \mathbf{M}^{(d)}[k,j,i]$$ Equivalently (vectorized): $$\hat{\mathbf{P}}[\lambda][\mathbf{M}^{(d)}] \mathrel{+}= \mathbf{Y}[\lambda, d, \mathbf{I}^{(d)}[\mathbf{M}^{(d)}]]$$ |
| **inputs** | $\hat{\mathbf{P}}[\lambda]$: $(N_z, N_y, N_x)$, $\mathbf{Y}[\lambda, d, :]$: $(N_t,)$, $\mathbf{I}^{(d)}$: $(N_z, N_y, N_x)$, $\mathbf{M}^{(d)}$: $(N_z, N_y, N_x)$ |
| **outputs** | $\hat{\mathbf{P}}[\lambda]$: $(N_z, N_y, N_x)$ (updated in-place) |
| **computational_note** | Indexed accumulation; each detector contributes additively |

---

### END ITERATION LOOP (Reconstruction)

---

### Step 4.5: Flatten Reconstruction for Unmixing

| Field | Value |
|-------|-------|
| **step_name** | Reshape Reconstruction to Matrix Form |
| **formula** | $$\mathbf{S} = \text{reshape}(\hat{\mathbf{P}}, (N_\lambda, N_{\text{vox}}))$$ where $N_{\text{vox}} = N_z \times N_y \times N_x$ |
| **inputs** | $\hat{\mathbf{P}}$: $(N_\lambda, N_z, N_y, N_x)$ |
| **outputs** | $\mathbf{S}$: $(N_\lambda, N_{\text{vox}})$ |
| **computational_note** | Row-major flattening of spatial dimensions |

### Step 4.6: Pseudoinverse Computation

| Field | Value |
|-------|-------|
| **step_name** | Compute Moore-Penrose Pseudoinverse of Extinction Matrix |
| **formula** | $$\mathbf{E}^+ = (\mathbf{E}^\top \mathbf{E})^{-1} \mathbf{E}^\top$$ Expanded: $$\mathbf{E}^+ = \begin{bmatrix} \mathbf{e}_{\text{Hb}}^\top \\ \mathbf{e}_{\text{HbO}_2}^\top \end{bmatrix} \left( \begin{bmatrix} \|\mathbf{e}_{\text{Hb}}\|^2 & \mathbf{e}_{\text{Hb}}^\top \mathbf{e}_{\text{HbO}_2} \\ \mathbf{e}_{\text{Hb}}^\top \mathbf{e}_{\text{HbO}_2} & \|\mathbf{e}_{\text{HbO}_2}\|^2 \end{bmatrix}^{-1} \right)^\top$$ |
| **inputs** | $\mathbf{E}$: $(N_\lambda, 2)$ |
| **outputs** | $\mathbf{E}^+$: $(2, N_\lambda)$ |
| **computational_note** | SVD-based computation in numpy.linalg.pinv |

### Step 4.7: Linear Spectral Unmixing

| Field | Value |
|-------|-------|
| **step_name** | Solve Least Squares for Chromophore Concentrations |
| **formula** | $$\mathbf{C}_{\text{flat}} = \mathbf{E}^+ \mathbf{S}$$ Component-wise: $$C_{\text{Hb}}^{(v)} = \sum_{i=1}^{N_\lambda} E^+_{1,i} \cdot S_{i,v}$$ $$C_{\text{HbO}_2}^{(v)} = \sum_{i=1}^{N_\lambda} E^+_{2,i} \cdot S_{i,v}$$ |
| **inputs** | $\mathbf{E}^+$: $(2, N_\lambda)$, $\mathbf{S}$: $(N_\lambda, N_{\text{vox}})$ |
| **outputs** | $\mathbf{C}_{\text{flat}}$: $(2, N_{\text{vox}})$ |
| **computational_note** | Matrix multiplication |

### Step 4.8: Reshape Concentrations

| Field | Value |
|-------|-------|
| **step_name** | Reshape Concentration Vector to Spatial Grid |
| **formula** | $$\mathbf{C} = \text{reshape}(\mathbf{C}_{\text{flat}}, (2, N_z, N_y, N_x))$$ $$C_{\text{Hb}} = \mathbf{C}[0], \quad C_{\text{HbO}_2} = \mathbf{C}[1]$$ |
| **inputs** | $\mathbf{C}_{\text{flat}}$: $(2, N_{\text{vox}})$ |
| **outputs** | $\mathbf{C}$: $(2, N_z, N_y, N_x)$ |
| **computational_note** | Inverse of flatten operation |

### Step 4.9: Total Hemoglobin and Threshold Mask

| Field | Value |
|-------|-------|
| **step_name** | Compute Total Hemoglobin and Signal Mask |
| **formula** | Total hemoglobin: $$C_{\text{total}}[k,j,i] = C_{\text{Hb}}[k,j,i] + C_{\text{HbO}_2}[k,j,i]$$ Threshold: $$\theta = 0.1 \times \max_{k,j,i}\left( C_{\text{total}}[k,j,i] \right)$$ Validity mask: $$\mathbf{M}_{\text{sO}_2}[k,j,i] = \begin{cases} 1 & \text{if } C_{\text{total}}[k,j,i] > \theta \\ 0 & \text{otherwise} \end{cases}$$ |
| **inputs** | $C_{\text{Hb}}$: $(N_z, N_y, N_x)$, $C_{\text{HbO}_2}$: $(N_z, N_y, N_x)$ |
| **outputs** | $C_{\text{total}}$: $(N_z, N_y, N_x)$, $\mathbf{M}_{\text{sO}_2}$: $(N_z, N_y, N_x)$, $\theta$: scalar |
| **computational_note** | Threshold at 10% of maximum total hemoglobin |

### Step 4.10: Oxygen Saturation Calculation

| Field | Value |
|-------|-------|
| **step_name** | Compute sO₂ with Masking and Clipping |
| **formula** | Initialize: $$\mathbf{sO}_2 \leftarrow \mathbf{0} \in \mathbb{R}^{N_z \times N_y \times N_x}$$ Masked division: $$\mathbf{sO}_2[k,j,i] = \begin{cases} \frac{C_{\text{HbO}_2}[k,j,i]}{C_{\text{total}}[k,j,i]} & \text{if } \mathbf{M}_{\text{sO}_2}[k,j,i] = 1 \\ 0 & \text{otherwise} \end{cases}$$ Clipping to physiological range: $$\mathbf{sO}_2[k,j,i] \leftarrow \min\left(\max\left(\mathbf{sO}_2[k,j,i], 0\right), 1\right)$$ |
| **inputs** | $C_{\text{HbO}_2}$: $(N_z, N_y, N_x)$, $C_{\text{total}}$: $(N_z, N_y, N_x)$, $\mathbf{M}_{\text{sO}_2}$: $(N_z, N_y, N_x)$ |
| **outputs** | $\mathbf{sO}_2$: $(N_z, N_y, N_x)$ |
| **computational_note** | Avoids division by zero; enforces $\mathbf{sO}_2 \in [0, 1]$ |

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| **type** | iteration_limit (deterministic, single-pass algorithm) |
| **expression** | Reconstruction loop terminates when: $$\lambda = N_\lambda \text{ AND } d = N_{\text{det}}$$ Total iterations: $N_{\text{iter}} = N_\lambda \times N_{\text{det}}$ |
| **parameters** | $N_\lambda$: number of wavelengths (data-dependent) | 
| | $N_{\text{det}}$: number of detectors (data-dependent) |

**Note**: This is a direct reconstruction algorithm, not an iterative optimization. The loop structure visits each wavelength-detector pair exactly once.

---

## 6. EVALUATION METRICS

### Metric 6.1: Mean sO₂ in Region of Interest

| Field | Value |
|-------|-------|
| **name** | Mean Oxygen Saturation (ROI) |
| **formula** | Define valid voxel set: $$\mathcal{V} = \{(k,j,i) : \mathbf{sO}_2[k,j,i] > 0\}$$ Mean sO₂: $$\overline{\text{sO}_2} = \begin{cases} \frac{1}{|\mathcal{V}|} \sum_{(k,j,i) \in \mathcal{V}}