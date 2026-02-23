## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Photoacoustic Tomography Delay-and-Sum Backprojection with Linear Spectral Unmixing
- **algorithm_family**: Direct Reconstruction (Backprojection) with Linear Algebraic Unmixing
- **summary**: Reconstructs initial pressure distributions from multi-wavelength photoacoustic time-series data using delay-and-sum backprojection, then performs linear spectral unmixing to estimate hemoglobin concentrations and oxygen saturation.

---

## 1. OBJECTIVE FUNCTION

### full_expression
This algorithm implements direct reconstruction rather than iterative optimization. The underlying inverse problems are:

**Reconstruction (implicit least-squares via adjoint):**
$$\hat{p}_0 = \mathcal{A}^\dagger y \approx \mathcal{A}^T y$$

**Spectral Unmixing (explicit least-squares):**
$$\min_{\mathbf{C}} \|\mathbf{S} - \mathbf{E}\mathbf{C}\|_F^2$$

### data_fidelity_term
$$\|\mathbf{S} - \mathbf{E}\mathbf{C}\|_F^2 = \sum_{i,j}\left(S_{ij} - \sum_k E_{ik}C_{kj}\right)^2$$

where $\mathbf{S} \in \mathbb{R}^{n_{wl} \times n_{voxels}}$ is the reshaped reconstruction, $\mathbf{E} \in \mathbb{R}^{n_{wl} \times 2}$ is the absorption coefficient matrix, $\mathbf{C} \in \mathbb{R}^{2 \times n_{voxels}}$ is the concentration matrix.

### regularization_term
None

### constraints
None

---

## 2. VARIABLES

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}_{raw}$ | observations | $(n_{wl}, n_{det}, n_t)$ | float64 | Loaded from HDF5 | Raw photoacoustic time-series signals |
| $\mathbf{G}$ | observations | $(n_{det}, 3)$ | float64 | Loaded from HDF5 | Detector positions in 3D space (x, y, z) |
| $\boldsymbol{\lambda}$ | observations | $(n_{wl},)$ | float64 | Loaded from HDF5 | Illumination wavelengths in nm |
| $f_s$ | observations | scalar | float64 | Loaded from HDF5 | Sampling frequency in Hz |
| $c$ | observations | scalar | float64 | Loaded from HDF5 | Speed of sound in medium (m/s) |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $f_{low}$ | constants | scalar | float64 | $5 \times 10^3$ | Highpass filter cutoff frequency (Hz) |
| $f_{high}$ | constants | scalar | float64 | $7 \times 10^6$ | Lowpass filter cutoff frequency (Hz) |
| $n_{order}$ | constants | scalar | int | $4$ | Butterworth filter order |
| $n_x$ | constants | scalar | int | $100$ | Number of pixels in x-direction |
| $n_y$ | constants | scalar | int | $100$ | Number of pixels in y-direction |
| $n_z$ | constants | scalar | int | $1$ | Number of pixels in z-direction |
| $L_x$ | constants | scalar | float64 | $0.025$ | Field of view in x-direction (m) |
| $L_y$ | constants | scalar | float64 | $0.025$ | Field of view in y-direction (m) |
| $L_z$ | constants | scalar | float64 | $0.0$ | Field of view in z-direction (m) |
| $\boldsymbol{\lambda}_{ref}$ | constants | $(6,)$ | float64 | $[700, 730, 760, 800, 850, 900]$ | Reference wavelengths for absorption spectra (nm) |
| $\boldsymbol{\epsilon}_{Hb,ref}$ | constants | $(6,)$ | float64 | $[100, 80, 60, 40, 30, 20]$ | Reference Hb absorption coefficients |
| $\boldsymbol{\epsilon}_{HbO_2,ref}$ | constants | $(6,)$ | float64 | $[30, 40, 50, 60, 70, 80]$ | Reference HbO₂ absorption coefficients |
| $\tau_{mask}$ | constants | scalar | float64 | $0.1$ | Threshold fraction for sO₂ mask |

### Auxiliary Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{b}, \mathbf{a}$ | auxiliary | $(n_{order}+1,)$ each | float64 | Computed from Butterworth design | Filter coefficients (numerator, denominator) |
| $\mathbf{y}_{filt}$ | auxiliary | $(n_{wl}, n_{det}, n_t)$ | float64 | Zero-initialized | Bandpass filtered signals |
| $\mathbf{y}_{proc}$ | auxiliary | $(n_{wl}, n_{det}, n_t)$ | float64 | Zero-initialized | Preprocessed signals (Hilbert imaginary) |
| $\Delta l$ | auxiliary | scalar | float64 | $c / f_s$ | Spatial sampling interval (m) |
| $\mathbf{X}, \mathbf{Y}, \mathbf{Z}$ | auxiliary | $(n_z, n_y, n_x)$ each | float64 | Meshgrid | Voxel coordinate grids |
| $\mathbf{D}^{(d)}$ | auxiliary | $(n_z, n_y, n_x)$ | float64 | Computed per detector | Distance from detector $d$ to each voxel |
| $\mathbf{I}^{(d)}$ | auxiliary | $(n_z, n_y, n_x)$ | int64 | $\lfloor \mathbf{D}^{(d)} / \Delta l \rfloor$ | Sample index map for detector $d$ |
| $\mathbf{E}$ | auxiliary | $(n_{wl}, 2)$ | float64 | Interpolated from reference | Absorption coefficient matrix $[\boldsymbol{\epsilon}_{Hb}, \boldsymbol{\epsilon}_{HbO_2}]$ |
| $\mathbf{E}^+$ | auxiliary | $(2, n_{wl})$ | float64 | Moore-Penrose pseudoinverse of $\mathbf{E}$ | Pseudoinverse for unmixing |

### Primal Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{P}$ | primal | $(n_{wl}, n_z, n_y, n_x)$ | float64 | Zero-initialized | Reconstructed initial pressure distributions |
| $\mathbf{C}$ | primal | $(2, n_z, n_y, n_x)$ | float64 | Computed via unmixing | Concentration maps: $\mathbf{C}[0]=C_{Hb}$, $\mathbf{C}[1]=C_{HbO_2}$ |
| $\mathbf{sO_2}$ | primal | $(n_z, n_y, n_x)$ | float64 | Zero-initialized | Oxygen saturation map |

---

## 3. PREPROCESSING PIPELINE

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 1 | Data Loading | $\mathbf{y}_{raw}, f_s, \mathbf{G}, \boldsymbol{\lambda}, c \leftarrow \text{HDF5\_Load}(\text{filename})[0:1]$ | filename: string | $\mathbf{y}_{raw}$: $(1, n_{wl}, n_{det}, n_t)$ or $(n_{wl}, n_{det}, n_t)$; $f_s$: scalar; $\mathbf{G}$: $(n_{det}, 3)$; $\boldsymbol{\lambda}$: $(n_{wl},)$; $c$: scalar |
| 2 | Singleton Removal | $\mathbf{y}_{raw} \leftarrow \mathbf{y}_{raw}[0]$ if $\mathbf{y}_{raw}.\text{ndim}=4$ and $\mathbf{y}_{raw}.\text{shape}[0]=1$ | $\mathbf{y}_{raw}$: $(1, n_{wl}, n_{det}, n_t)$ | $\mathbf{y}_{raw}$: $(n_{wl}, n_{det}, n_t)$ |
| 3 | Butterworth Filter Design | $\omega_{low} = \frac{f_{low}}{0.5 f_s}, \quad \omega_{high} = \frac{f_{high}}{0.5 f_s}$; $\mathbf{b}, \mathbf{a} = \text{butter}(n_{order}, [\omega_{low}, \omega_{high}], \text{btype}=\text{'band'})$ | $f_{low}=5000$, $f_{high}=7\times10^6$, $f_s$, $n_{order}=4$ | $\mathbf{b}$: $(n_{order}+1,)$; $\mathbf{a}$: $(n_{order}+1,)$ |
| 4 | Bandpass Filtering | $\mathbf{y}_{filt}[w, d, :] = \text{filtfilt}(\mathbf{b}, \mathbf{a}, \mathbf{y}_{raw}[w, d, :])$ | $\mathbf{y}_{raw}$: $(n_{wl}, n_{det}, n_t)$; $\mathbf{b}, \mathbf{a}$ | $\mathbf{y}_{filt}$: $(n_{wl}, n_{det}, n_t)$ |
| 5 | Hilbert Transform | $\mathbf{y}_{analytic}[w, d, :] = \mathcal{H}\{\mathbf{y}_{filt}[w, d, :]\} = \mathbf{y}_{filt}[w, d, :] + j \cdot \text{HT}\{\mathbf{y}_{filt}[w, d, :]\}$ | $\mathbf{y}_{filt}$: $(n_{wl}, n_{det}, n_t)$ | $\mathbf{y}_{analytic}$: $(n_{wl}, n_{det}, n_t)$ complex |
| 6 | Imaginary Extraction | $\mathbf{y}_{proc} = \text{Im}(\mathbf{y}_{analytic})$ | $\mathbf{y}_{analytic}$: $(n_{wl}, n_{det}, n_t)$ complex | $\mathbf{y}_{proc}$: $(n_{wl}, n_{det}, n_t)$ real |
| 7 | Grid Construction | $x_i = -\frac{L_x}{2} + \frac{i \cdot L_x}{n_x - 1}, \quad i=0,\ldots,n_x-1$; $y_j = -\frac{L_y}{2} + \frac{j \cdot L_y}{n_y - 1}, \quad j=0,\ldots,n_y-1$; $z_k = 0$ (single plane); $(\mathbf{Z}, \mathbf{Y}, \mathbf{X}) = \text{meshgrid}(\mathbf{z}, \mathbf{y}, \mathbf{x}, \text{indexing}=\text{'ij'})$ | $n_x, n_y, n_z, L_x, L_y$ | $\mathbf{X}, \mathbf{Y}, \mathbf{Z}$: $(n_z, n_y, n_x)$ each |
| 8 | Spatial Sampling Interval | $\Delta l = \frac{c}{f_s}$ | $c$: scalar; $f_s$: scalar | $\Delta l$: scalar |
| 9 | Absorption Spectra Interpolation | $\epsilon_{Hb}(\lambda) = \text{interp1d}(\boldsymbol{\lambda}_{ref}, \boldsymbol{\epsilon}_{Hb,ref}, \text{kind}=\text{'linear'})(\boldsymbol{\lambda})$; $\epsilon_{HbO_2}(\lambda) = \text{interp1d}(\boldsymbol{\lambda}_{ref}, \boldsymbol{\epsilon}_{HbO_2,ref}, \text{kind}=\text{'linear'})(\boldsymbol{\lambda})$ | $\boldsymbol{\lambda}$: $(n_{wl},)$; reference arrays | $\boldsymbol{\epsilon}_{Hb}$: $(n_{wl},)$; $\boldsymbol{\epsilon}_{HbO_2}$: $(n_{wl},)$ |
| 10 | Absorption Matrix Construction | $\mathbf{E} = \begin{bmatrix} \boldsymbol{\epsilon}_{Hb} & \boldsymbol{\epsilon}_{HbO_2} \end{bmatrix}$ | $\boldsymbol{\epsilon}_{Hb}, \boldsymbol{\epsilon}_{HbO_2}$: $(n_{wl},)$ each | $\mathbf{E}$: $(n_{wl}, 2)$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 1 | Initialize Reconstruction | $\mathbf{P} = \mathbf{0} \in \mathbb{R}^{n_{wl} \times n_z \times n_y \times n_x}$ | $n_{wl}, n_z, n_y, n_x$ | $\mathbf{P}$: $(n_{wl}, n_z, n_y, n_x)$ |

### 4.2 ITERATION LOOP (Delay-and-Sum Backprojection)

**Note**: This is a deterministic traversal (not iterative optimization), structured as nested loops.

---

**BEGIN_LOOP** (Wavelength loop: $w = 0, \ldots, n_{wl}-1$)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| 2.1 | Extract Wavelength Signal | $\mathbf{s}^{(w)} = \mathbf{y}_{proc}[w, :, :]$ | $\mathbf{y}_{proc}$, $w$ | $\mathbf{s}^{(w)}$: $(n_{det}, n_t)$ | Slice operation |

---

**BEGIN_LOOP** (Detector loop: $d = 0, \ldots, n_{det}-1$)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| 2.2 | Extract Detector Position | $\mathbf{g}^{(d)} = \mathbf{G}[d, :] = (g_x^{(d)}, g_y^{(d)}, g_z^{(d)})$ | $\mathbf{G}$, $d$ | $\mathbf{g}^{(d)}$: $(3,)$ | |
| 2.3 | Compute Distance Map | $D^{(d)}_{k,j,i} = \sqrt{(X_{k,j,i} - g_x^{(d)})^2 + (Y_{k,j,i} - g_y^{(d)})^2 + (Z_{k,j,i} - g_z^{(d)})^2}$ | $\mathbf{X}, \mathbf{Y}, \mathbf{Z}$, $\mathbf{g}^{(d)}$ | $\mathbf{D}^{(d)}$: $(n_z, n_y, n_x)$ | Euclidean distance |
| 2.4 | Compute Sample Index Map | $I^{(d)}_{k,j,i} = \left\lfloor \frac{D^{(d)}_{k,j,i}}{\Delta l} \right\rfloor$ | $\mathbf{D}^{(d)}$, $\Delta l$ | $\mathbf{I}^{(d)}$: $(n_z, n_y, n_x)$ int | Time-of-flight to sample index |
| 2.5 | Compute Valid Mask | $M^{(d)}_{k,j,i} = \mathbb{1}\left[0 \leq I^{(d)}_{k,j,i} < n_t\right]$ | $\mathbf{I}^{(d)}$, $n_t$ | $\mathbf{M}^{(d)}$: $(n_z, n_y, n_x)$ bool | Bounds check |
| 2.6 | Backproject Signal | $P_{w,k,j,i} \leftarrow P_{w,k,j,i} + M^{(d)}_{k,j,i} \cdot s^{(w)}_{d, I^{(d)}_{k,j,i}}$ | $\mathbf{P}$, $\mathbf{s}^{(w)}$, $\mathbf{I}^{(d)}$, $\mathbf{M}^{(d)}$ | $\mathbf{P}$: $(n_{wl}, n_z, n_y, n_x)$ | Accumulate contributions |

**END_LOOP** (Detector loop)

**END_LOOP** (Wavelength loop)

---

### 4.3 FINALIZATION (Spectral Unmixing and sO₂ Calculation)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| 3.1 | Reshape Reconstruction | $\mathbf{S} = \text{reshape}(\mathbf{P}, (n_{wl}, n_z \cdot n_y \cdot n_x))$ | $\mathbf{P}$: $(n_{wl}, n_z, n_y, n_x)$ | $\mathbf{S}$: $(n_{wl}, n_{vox})$ where $n_{vox} = n_z n_y n_x$ | Flatten spatial dims |
| 3.2 | Compute Pseudoinverse | $\mathbf{E}^+ = (\mathbf{E}^T \mathbf{E})^{-1} \mathbf{E}^T$ | $\mathbf{E}$: $(n_{wl}, 2)$ | $\mathbf{E}^+$: $(2, n_{wl})$ | Moore-Penrose pseudoinverse |
| 3.3 | Solve Unmixing | $\mathbf{C}_{flat} = \mathbf{E}^+ \mathbf{S}$ | $\mathbf{E}^+$: $(2, n_{wl})$; $\mathbf{S}$: $(n_{wl}, n_{vox})$ | $\mathbf{C}_{flat}$: $(2, n_{vox})$ | Least-squares solution |
| 3.4 | Reshape Concentrations | $\mathbf{C} = \text{reshape}(\mathbf{C}_{flat}, (2, n_z, n_y, n_x))$ | $\mathbf{C}_{flat}$: $(2, n_{vox})$ | $\mathbf{C}$: $(2, n_z, n_y, n_x)$ | $\mathbf{C}[0]=C_{Hb}$, $\mathbf{C}[1]=C_{HbO_2}$ |
| 3.5 | Compute Total Hemoglobin | $H_{total} = C_{Hb} + C_{HbO_2} = \mathbf{C}[0] + \mathbf{C}[1]$ | $\mathbf{C}$: $(2, n_z, n_y, n_x)$ | $\mathbf{H}_{total}$: $(n_z, n_y, n_x)$ | |
| 3.6 | Compute sO₂ Mask | $M^{sO_2}_{k,j,i} = \mathbb{1}\left[H_{total,k,j,i} > \tau_{mask} \cdot \max(\mathbf{H}_{total})\right]$ where $\tau_{mask} = 0.1$ | $\mathbf{H}_{total}$ | $\mathbf{M}^{sO_2}$: $(n_z, n_y, n_x)$ bool | Threshold: 10% of max |
| 3.7 | Initialize sO₂ | $\mathbf{sO_2} = \mathbf{0} \in \mathbb{R}^{n_z \times n_y \times n_x}$ | dimensions | $\mathbf{sO_2}$: $(n_z, n_y, n_x)$ | |
| 3.8 | Compute sO₂ | $sO_{2,k,j,i} = \begin{cases} \frac{C_{HbO_2,k,j,i}}{H_{total,k,j,i}} & \text{if } M^{sO_2}_{k,j,i} = 1 \\ 0 & \text{otherwise} \end{cases}$ | $\mathbf{C}[1]$, $\mathbf{H}_{total}$, $\mathbf{M}^{sO_2}$ | $\mathbf{sO_2}$: $(n_z, n_y, n_x)$ | Masked division |
| 3.9 | Clip sO₂ | $\mathbf{sO_2} = \text{clip}(\mathbf{sO_2}, 0, 1)$ | $\mathbf{sO_2}$ | $\mathbf{sO_2}$: $(n_z, n_y, n_x)$ | Enforce physical bounds |

---

## 5. STOPPING CRITERION

- **type**: deterministic_completion (not iterative optimization)
- **expression**: Algorithm terminates after complete traversal of all wavelengths and detectors:
  $$\text{Complete when } w = n_{wl} - 1 \text{ AND } d = n_{det} - 1$$
- **parameters**: 
  - Total wavelength iterations: $n_{wl}$ (data-dependent)
  - Total detector iterations per wavelength: $n_{det}$ (data-dependent)
  - Total backprojection operations: $n_{wl} \times n_{det}$
- **check_frequency**: Loops complete deterministically based on data dimensions

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Mean sO₂ | $\overline{sO_2} = \frac{1}{|\mathcal{V}|}\sum_{(k,j,i) \in \mathcal{V}} sO_{2,k,j,i}$ where $\mathcal{V} = \{(k,j,i) : sO_{2,k,j,i} > 0\}$ | Mean oxygen saturation over valid (non-zero) voxels |
| Reconstruction Range | $[\min(\mathbf{P}), \max(\mathbf{P})]$ | Dynamic range of reconstructed pressure distribution |
| Mean Reconstruction Image | $\bar{P}_{k,j,i} = \frac{1}{n_{wl}}\sum_{w=0}^{n_{wl}-1} P_{w,k,j,i}$ | Wavelength-averaged reconstruction for visualization |