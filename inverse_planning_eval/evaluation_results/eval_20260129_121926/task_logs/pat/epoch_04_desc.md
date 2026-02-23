## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Delay-and-Sum Photoacoustic Tomography with Spectral Unmixing
- **algorithm_family**: Direct Reconstruction (Backprojection) + Linear Spectral Unmixing
- **summary**: Reconstructs initial pressure distribution from photoacoustic time-series data using delay-and-sum backprojection, then performs linear spectral unmixing to estimate hemoglobin concentrations and oxygen saturation.

---

## 1. OBJECTIVE FUNCTION

### full_expression
This algorithm performs direct reconstruction (non-iterative), not optimization-based inversion. The implicit model being inverted is:

$$
\hat{p}_0 = \arg\min_{p_0} \|y - \mathcal{A}p_0\|_2^2
$$

where the solution is approximated via backprojection:

$$
\hat{p}_0(\mathbf{r}) = \sum_{d=1}^{N_{\text{det}}} y_d\left(\frac{\|\mathbf{r} - \mathbf{r}_d\|}{c}\right)
$$

For spectral unmixing:
$$
\hat{C} = \arg\min_C \|S - EC\|_F^2 \quad \Rightarrow \quad \hat{C} = E^\dagger S
$$

### data_fidelity_term
**Backprojection** (implicit L2 matching via adjoint operation):
$$
\mathcal{F}_{\text{DAS}}(p_0, y) = \text{Backprojection is the adjoint of forward model, not explicit minimization}
$$

**Spectral Unmixing** (L2 norm):
$$
\mathcal{F}_{\text{unmix}}(C) = \|S - EC\|_F^2 = \sum_{i,j}\left(S_{ij} - \sum_k E_{ik}C_{kj}\right)^2
$$

### regularization_term
None (direct reconstruction and pseudoinverse solution)

### constraints
None explicitly enforced during reconstruction. Post-hoc clipping applied to sO2:
$$
\text{sO}_2 \in [0, 1] \quad \text{via} \quad \text{sO}_2 = \text{clip}(\text{sO}_2, 0, 1)
$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\hat{p}_0$ (reconstruction) | $(N_{\lambda}, N_z, N_y, N_x)$ | float64 | `np.zeros((n_wl, nz, ny, nx))` | Reconstructed initial pressure distribution per wavelength |
| $C$ (concentrations) | $(2, N_z, N_y, N_x)$ | float64 | Computed via pseudoinverse | Chromophore concentrations: $C[0]=[\text{Hb}]$, $C[1]=[\text{HbO}_2]$ |
| $\text{sO}_2$ | $(N_z, N_y, N_x)$ | float64 | `np.zeros_like(hbo2)` | Oxygen saturation map |

### 2.2 Dual Variables

None (direct method, not optimization-based)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $y_{\text{raw}}$ | $(N_{\lambda}, N_{\text{det}}, N_t)$ | float64 | Loaded from HDF5 | Raw photoacoustic time-series data |
| $y_{\text{filt}}$ | $(N_{\lambda}, N_{\text{det}}, N_t)$ | float64 | After bandpass | Bandpass filtered signal |
| $y_{\text{analytic}}$ | $(N_{\lambda}, N_{\text{det}}, N_t)$ | complex128 | After Hilbert | Analytic signal from Hilbert transform |
| $y$ (processed_data) | $(N_{\lambda}, N_{\text{det}}, N_t)$ | float64 | $\text{Im}(y_{\text{analytic}})$ | Preprocessed signal (imaginary part of analytic signal) |
| $d_{i,\mathbf{r}}$ (dist) | $(N_z, N_y, N_x)$ | float64 | Per detector | Distance from detector $i$ to each voxel |
| $k_{i,\mathbf{r}}$ (sample_idx) | $(N_z, N_y, N_x)$ | int64 | $\lfloor d_{i,\mathbf{r}} / \Delta l \rfloor$ | Time sample index for delay |
| $E$ | $(N_{\lambda}, 2)$ | float64 | From absorption spectra | Extinction coefficient matrix |
| $S$ | $(N_{\lambda}, N_z \cdot N_y \cdot N_x)$ | float64 | Reshaped reconstruction | Flattened spectral measurements |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_x$ | 100 | Number of pixels in x-direction |
| $N_y$ | 100 | Number of pixels in y-direction |
| $N_z$ | 1 | Number of pixels in z-direction |
| $L_x$ | 0.025 m | Field of view in x |
| $L_y$ | 0.025 m | Field of view in y |
| $L_z$ | 0 m | Field of view in z |
| $f_{\text{LP}}$ | $7 \times 10^6$ Hz | Lowpass filter cutoff |
| $f_{\text{HP}}$ | $5 \times 10^3$ Hz | Highpass filter cutoff |
| filter_order | 4 | Butterworth filter order |
| $\lambda_{\text{ref}}$ | [700, 730, 760, 800, 850, 900] nm | Reference wavelengths for absorption spectra |
| $\epsilon_{\text{Hb,ref}}$ | [100, 80, 60, 40, 30, 20] | Reference Hb extinction coefficients |
| $\epsilon_{\text{HbO}_2,\text{ref}}$ | [30, 40, 50, 60, 70, 80] | Reference HbO₂ extinction coefficients |
| $\theta_{\text{mask}}$ | 0.1 | sO₂ mask threshold (fraction of max) |

### 2.5 Observations

| Symbol | Shape | Source | Description |
|--------|-------|--------|-------------|
| $y_{\text{raw}}$ | $(N_{\lambda}, N_{\text{det}}, N_t)$ | HDF5 file | Raw time-series photoacoustic signals |
| $\mathbf{r}_d$ (geometry) | $(N_{\text{det}}, 3)$ | HDF5 file | Detector positions in 3D space |
| $f_s$ | scalar | HDF5 file | Sampling frequency |
| $c$ (speed_of_sound) | scalar | HDF5 file | Speed of sound in medium |
| $\boldsymbol{\lambda}$ (wavelengths) | $(N_{\lambda},)$ | HDF5 file | Illumination wavelengths |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Parameter Derivation

| Parameter | Formula | Inputs | Outputs |
|-----------|---------|--------|---------|
| Nyquist frequency | $f_{\text{nyq}} = 0.5 \cdot f_s$ | $f_s$ | $f_{\text{nyq}}$ |
| Normalized lowcut | $\omega_{\text{low}} = \frac{f_{\text{HP}}}{f_{\text{nyq}}}$ | $f_{\text{HP}}, f_{\text{nyq}}$ | $\omega_{\text{low}}$ |
| Normalized highcut | $\omega_{\text{high}} = \frac{f_{\text{LP}}}{f_{\text{nyq}}}$ | $f_{\text{LP}}, f_{\text{nyq}}$ | $\omega_{\text{high}}$ |
| Spatial sampling | $\Delta l = \frac{c}{f_s}$ | $c, f_s$ | $\Delta l$ |

### Step 1: Data Loading

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | load_hdf5 | $y_{\text{raw}}, \mathbf{r}_d, f_s, c, \boldsymbol{\lambda} \leftarrow \text{HDF5}(\text{filename})$ | filename: str | $y_{\text{raw}}$: $(N_{\lambda}, N_{\text{det}}, N_t)$, geometry: $(N_{\text{det}}, 3)$ |
| 2 | squeeze_batch | $y_{\text{raw}} \leftarrow y_{\text{raw}}[0]$ if $y_{\text{raw}}.\text{ndim}=4$ and $y_{\text{raw}}.\text{shape}[0]=1$ | $y_{\text{raw}}$: $(1, N_{\lambda}, N_{\text{det}}, N_t)$ | $y_{\text{raw}}$: $(N_{\lambda}, N_{\text{det}}, N_t)$ |

### Step 2: Bandpass Filtering

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 3 | design_butterworth | $(b, a) = \text{butter}(n=4, [\omega_{\text{low}}, \omega_{\text{high}}], \text{btype}=\text{'band'})$ | $\omega_{\text{low}}, \omega_{\text{high}}$, order=4 | $b, a$: filter coefficients |
| 4 | apply_filtfilt | $y_{\text{filt}}[i,j,:] = \text{filtfilt}(b, a, y_{\text{raw}}[i,j,:])$ | $y_{\text{raw}}$: $(N_{\lambda}, N_{\text{det}}, N_t)$, $b, a$ | $y_{\text{filt}}$: $(N_{\lambda}, N_{\text{det}}, N_t)$ |

The filtfilt operation applies zero-phase filtering:
$$
y_{\text{filt}} = \mathcal{H}^{-1}\left(\mathcal{H}(y_{\text{raw}}) \cdot |H(e^{j\omega})|^2\right)
$$

where $H(z)$ is the Butterworth transfer function.

### Step 3: Hilbert Transform (Envelope Detection)

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 5 | hilbert_transform | $y_{\text{analytic}}[i,j,:] = y_{\text{filt}}[i,j,:] + j \cdot \mathcal{H}\{y_{\text{filt}}[i,j,:]\}$ | $y_{\text{filt}}$: $(N_{\lambda}, N_{\text{det}}, N_t)$ | $y_{\text{analytic}}$: $(N_{\lambda}, N_{\text{det}}, N_t)$, complex128 |
| 6 | extract_imaginary | $y = \text{Im}(y_{\text{analytic}})$ | $y_{\text{analytic}}$ | $y$: $(N_{\lambda}, N_{\text{det}}, N_t)$, float64 |

where the Hilbert transform is:
$$
\mathcal{H}\{y(t)\} = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{y(\tau)}{t - \tau} d\tau
$$

### Step 4: Reconstruction Grid Setup

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 7 | create_x_coords | $x_i = -\frac{L_x}{2} + i \cdot \frac{L_x}{N_x - 1}, \quad i = 0, \ldots, N_x-1$ | $L_x, N_x$ | xs: $(N_x,)$ |
| 8 | create_y_coords | $y_j = -\frac{L_y}{2} + j \cdot \frac{L_y}{N_y - 1}, \quad j = 0, \ldots, N_y-1$ | $L_y, N_y$ | ys: $(N_y,)$ |
| 9 | create_z_coords | $z_k = 0.0$ | — | zs: $(1,)$ = [0.0] |
| 10 | meshgrid | $(Z, Y, X) = \text{meshgrid}(\text{zs}, \text{ys}, \text{xs}, \text{indexing}=\text{'ij'})$ | zs, ys, xs | $X, Y, Z$: each $(N_z, N_y, N_x)$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| init_reconstruction | $\hat{p}_0 \leftarrow \mathbf{0} \in \mathbb{R}^{N_{\lambda} \times N_z \times N_y \times N_x}$ | $N_{\lambda}, N_z, N_y, N_x$ | reconstruction: $(N_{\lambda}, 1, 100, 100)$ | Zero-initialized accumulator |
| compute_spatial_sampling | $\Delta l = \frac{c}{f_s}$ | $c, f_s$ | $\Delta l$: scalar | Distance per time sample |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (wavelength_loop): $i_{\lambda} = 0, \ldots, N_{\lambda} - 1$

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| select_wavelength_data | $y^{(i_{\lambda})} = y[i_{\lambda}, :, :]$ | $y$, $i_{\lambda}$ | $y^{(i_{\lambda})}$: $(N_{\text{det}}, N_t)$ | Extract single wavelength |

**BEGIN_LOOP** (detector_loop): $i_d = 0, \ldots, N_{\text{det}} - 1$

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| get_detector_position | $\mathbf{r}_d = \text{geometry}[i_d, :]$ | geometry, $i_d$ | $\mathbf{r}_d$: $(3,)$ | [FORWARD] Detector coordinates |
| compute_distances | $d(\mathbf{r}) = \sqrt{(X - r_{d,x})^2 + (Y - r_{d,y})^2 + (Z - r_{d,z})^2}$ | $X, Y, Z$, $\mathbf{r}_d$ | dist: $(N_z, N_y, N_x)$ | Euclidean distance field |
| compute_delay_indices | $k(\mathbf{r}) = \lfloor d(\mathbf{r}) / \Delta l \rfloor$ | dist, $\Delta l$ | sample_idx: $(N_z, N_y, N_x)$, int | Time sample index for each voxel |
| compute_validity_mask | $M(\mathbf{r}) = \mathbb{1}[0 \leq k(\mathbf{r}) < N_t]$ | sample_idx, $N_t$ | valid_mask: $(N_z, N_y, N_x)$, bool | Valid index mask |
| backproject_detector | $\hat{p}_0^{(i_{\lambda})}[\mathbf{r}] \mathrel{+}= M(\mathbf{r}) \cdot y^{(i_{\lambda})}[i_d, k(\mathbf{r})]$ | $y^{(i_{\lambda})}$, sample_idx, valid_mask | reconstruction updated | [ADJOINT] Accumulate backprojected signal |

**END_LOOP** (detector_loop)

**END_LOOP** (wavelength_loop)

**Backprojection Formula** (complete):
$$
\hat{p}_0^{(\lambda)}(\mathbf{r}) = \sum_{d=1}^{N_{\text{det}}} y_d^{(\lambda)}\left[k_d(\mathbf{r})\right] \cdot \mathbb{1}\left[0 \leq k_d(\mathbf{r}) < N_t\right]
$$

where:
$$
k_d(\mathbf{r}) = \left\lfloor \frac{\|\mathbf{r} - \mathbf{r}_d\|_2}{\Delta l} \right\rfloor = \left\lfloor \frac{\|\mathbf{r} - \mathbf{r}_d\|_2 \cdot f_s}{c} \right\rfloor
$$

### 4.3 FINALIZATION

#### Spectral Unmixing

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| interpolate_hb_spectrum | $\epsilon_{\text{Hb}}(\lambda) = \text{interp1d}(\lambda_{\text{ref}}, \epsilon_{\text{Hb,ref}}, \text{linear})(\boldsymbol{\lambda})$ | $\boldsymbol{\lambda}$, reference data | hb: $(N_{\lambda},)$ | Linear interpolation/extrapolation |
| interpolate_hbo2_spectrum | $\epsilon_{\text{HbO}_2}(\lambda) = \text{interp1d}(\lambda_{\text{ref}}, \epsilon_{\text{HbO}_2,\text{ref}}, \text{linear})(\boldsymbol{\lambda})$ | $\boldsymbol{\lambda}$, reference data | hbo2: $(N_{\lambda},)$ | Linear interpolation/extrapolation |
| build_extinction_matrix | $E = \begin{bmatrix} \epsilon_{\text{Hb}}(\lambda_1) & \epsilon_{\text{HbO}_2}(\lambda_1) \\ \vdots & \vdots \\ \epsilon_{\text{Hb}}(\lambda_{N_{\lambda}}) & \epsilon_{\text{HbO}_2}(\lambda_{N_{\lambda}}) \end{bmatrix}$ | hb, hbo2 | $E$: $(N_{\lambda}, 2)$ | Vertical stack |
| reshape_reconstruction | $S = \text{reshape}(\hat{p}_0, (N_{\lambda}, N_z \cdot N_y \cdot N_x))$ | reconstruction | $S$: $(N_{\lambda}, N_{\text{voxels}})$ | Flatten spatial dimensions |
| compute_pseudoinverse | $E^{\dagger} = (E^T E)^{-1} E^T$ | $E$ | $E^{\dagger}$: $(2, N_{\lambda})$ | Moore-Penrose pseudoinverse |
| solve_unmixing | $C = E^{\dagger} S$ | $E^{\dagger}$, $S$ | $C$: $(2, N_{\text{voxels}})$ | Least-squares solution |
| reshape_concentrations | $\text{concentrations} = \text{reshape}(C, (2, N_z, N_y, N_x))$ | $C$ | concentrations: $(2, 1, 100, 100)$ | Restore spatial shape |

#### sO₂ Calculation

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| extract_hb | $[\text{Hb}] = \text{concentrations}[0]$ | concentrations | hb: $(N_z, N_y, N_x)$ | Deoxyhemoglobin |
| extract_hbo2 | $[\text{HbO}_2] = \text{concentrations}[1]$ | concentrations | hbo2: $(N_z, N_y, N_x)$ | Oxyhemoglobin |
| compute_total_hb | $[\text{Hb}]_{\text{total}} = [\text{Hb}] + [\text{HbO}_2]$ | hb, hbo2 | total_hb: $(N_z, N_y, N_x)$ | Total hemoglobin |
| compute_mask | $M_{\text{sO}_2} = \mathbb{1}\left[[\text{Hb}]_{\text{total}} > 0.1 \cdot \max([\text{Hb}]_{\text{total}})\right]$ | total_hb | mask: $(N_z, N_y, N_x)$, bool | Threshold at 10% of max |
| init_so2 | $\text{sO}_2 \leftarrow \mathbf{0} \in \mathbb{R}^{N_z \times N_y \times N_x}$ | shape | so2: $(N_z, N_y, N_x)$ | Zero initialization |
| compute_so2_masked | $\text{sO}_2[M_{\text{sO}_2}] = \frac{[\text{HbO}_2][M_{\text{sO}_2}]}{[\text{Hb}]_{\text{total}}[M_{\text{sO}_2}]}$ | hbo2, total_hb, mask | so2 updated | Oxygen saturation |
| clip_so2 | $\text{sO}_2 = \text{clip}(\text{sO}_2, 0, 1)$ | so2 | so2: $(N_z, N_y, N_x)$ | Enforce physical bounds |

**Complete sO₂ formula**:
$$
\text{sO}_2(\mathbf{r}) = \begin{cases}
\text{clip}\left(\frac{[\text{HbO}_2](\mathbf{r})}{[\text{Hb}](\mathbf{r}) + [\text{HbO}_2](\mathbf{r})}, 0, 1\right) & \text{if } [\text{Hb}]_{\text{total}}(\mathbf{r}) > 0.1 \cdot \max_{\mathbf{r}'}[\text{Hb}]_{\text{total}}(\mathbf{r}') \\
0 & \text{otherwise}
\end{cases}
$$

---

## 5. STOPPING CRITERION

- **type**: single_pass (non-iterative)
- **expression**: Algorithm terminates after single pass through all wavelengths and detectors:
$$
\text{Complete when } i_{\lambda} = N_{\lambda} - 1 \text{ and } i_d = N_{\text{det}} - 1
$$
- **parameters**: 
  - $N_{\lambda}$: number of wavelengths (data-dependent)
  - $N_{\text{det}}$: number of detectors (data-dependent)
- **check_frequency**: N/A (deterministic single pass)

---

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| mean_so2 | $\bar{\text{sO}}_2 = \frac{1}{|\{\mathbf{r}: \text{sO}_2(\mathbf{r}) > 0\}|} \sum_{\mathbf{r}: \text{sO}_2(\mathbf{r}) > 0} \text{sO}_2(\mathbf{r})$ | Mean oxygen saturation over valid ROI |
| reconstruction_range | $[\min_{\mathbf{r}, \lambda} \hat{p}_0^{(\lambda)}(\mathbf{r}), \max_{\mathbf{r}, \lambda} \hat{p}_0^{(\lambda)}(\mathbf{r})]$ | Dynamic range of reconstructed image |
| recon_image | $\bar{p}_0(\mathbf{r}) = \frac{1}{N_{\lambda}} \sum_{\lambda} \hat{p}_0^{(\lambda)}(\mathbf{r})$ | Wavelength-averaged reconstruction |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include:

```
algorithm_identification:
  - algorithm_name
  - algorithm_family
  - summary

objective_function:
  - full_expression
  - data_fidelity_term
  - regularization_term
  - constraints

variables:
  - primal_variables
  - dual_variables
  - auxiliary_variables
  - constants
  - observations

data_preprocessing:
  - step_order
  - step_name
  - formula
  - inputs
  - outputs

initialization:
  - step_name
  - formula
  - outputs

iteration_loop:
  - loop_structure (nested: wavelength_loop > detector_loop)
  - step_name
  - formula
  - inputs
  - outputs
  - computational_note

finalization:
  - spectral_unmixing_steps
  - so2_calculation_steps

stopping_criterion:
  - type
  - expression
  - parameters

evaluation:
  - metric_name
  - formula
  - description

outputs:
  - reconstruction: shape (n_wl, nz, ny, nx)
  - concentrations: shape (2, nz, ny, nx)
  - so2: shape (nz, ny, nx)
  - mean_so2: scalar
```