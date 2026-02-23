## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Closed-Loop Adaptive Optics with Integral Control
- **algorithm_family**: Linear-Quadratic Control / Wavefront Reconstruction
- **summary**: Iterative integral controller that minimizes residual wavefront error by computing DM commands from Shack-Hartmann slope measurements using a pseudo-inverse reconstructor calibrated via push-pull interaction matrix.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{u}} \| \mathbf{s}_{\text{res}} \|_2^2 = \min_{\mathbf{u}} \| \mathbf{s}_{\text{meas}}(\mathbf{u}) - \mathbf{s}_{\text{ref}} \|_2^2$$

where $\mathbf{u} \in \mathbb{R}^{N_{\text{act}}}$ are deformable mirror actuator commands.

### data_fidelity_term
$$\mathcal{D}(\mathbf{u}) = \| \mathbf{s}_{\text{meas}}(\mathbf{u}) - \mathbf{s}_{\text{ref}} \|_2^2 = \sum_{i=1}^{N_{\text{meas}}} \left( s_{\text{meas},i}(\mathbf{u}) - s_{\text{ref},i} \right)^2$$

where:
- $\mathbf{s}_{\text{meas}}(\mathbf{u}) \in \mathbb{R}^{N_{\text{meas}}}$: measured slopes from Shack-Hartmann WFS given DM commands
- $\mathbf{s}_{\text{ref}} \in \mathbb{R}^{N_{\text{meas}}}$: reference slopes (flat wavefront)
- $N_{\text{meas}} = 2 \times N_{\text{valid}}$: number of slope measurements (x and y concatenated)

### regularization_term
Implicit regularization via truncated SVD with threshold $\tau = 10^{-3}$:
$$\mathcal{R}_{\text{implicit}}(\mathbf{R}) = \text{Truncation of singular values } \sigma_i < \tau \cdot \sigma_{\max}$$

### constraints
None (unconstrained actuator commands)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{u}^{(k)}$ | $(N_{\text{act}},)$ | float64 | $\mathbf{0}$ | DM actuator commands at iteration $k$ |

where $N_{\text{act}} = (N_{\text{sub}}+1)^2 \times \text{coupling\_factor}$ (number of valid actuators from DM object)

### 2.2 Dual Variables

None (not a primal-dual algorithm)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{s}_{\text{meas}}^{(k)}$ | $(N_{\text{meas}},)$ | float64 | Computed | Measured slopes at iteration $k$ |
| $\mathbf{s}_{\text{res}}^{(k)}$ | $(N_{\text{meas}},)$ | float64 | Computed | Residual slopes: $\mathbf{s}_{\text{meas}}^{(k)} - \mathbf{s}_{\text{ref}}$ |
| $\Delta\mathbf{u}^{(k)}$ | $(N_{\text{act}},)$ | float64 | Computed | Command increment: $\mathbf{R} \cdot \mathbf{s}_{\text{res}}^{(k)}$ |
| $\text{SR}^{(k)}$ | scalar | float64 | Computed | Strehl ratio at iteration $k$ (percent) |
| $I_{\text{spots}}$ | $(N_{\text{sub}}^2, N_{\text{pix}}, N_{\text{pix}})$ | float64 | Computed | WFS spot intensity patterns |
| $\mathbf{c}_x, \mathbf{c}_y$ | $(N_{\text{valid}},)$ | float64 | Computed | Centroid coordinates per subaperture |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N_{\text{sub}}$ | 20 | Number of subapertures per dimension |
| $N_{\text{modes}}$ | 20 | Number of KL modes for reconstruction |
| $\delta$ | $10^{-8}$ m | Push-pull stroke for calibration |
| $g$ | 0.4 | Integral controller gain |
| $K_{\max}$ | 20 | Maximum number of iterations |
| $\tau$ | $10^{-3}$ | SVD truncation threshold |
| $D$ | 8.0 m | Telescope diameter |
| $r_0$ | 0.15 m | Fried parameter |
| $L_0$ | 25 m | Outer scale |
| $v_{\text{wind}}$ | 10 m/s | Wind speed |
| $\alpha_{\text{coupling}}$ | 0.35 | DM mechanical coupling |
| $\rho_{\text{light}}$ | 0.5 | WFS light ratio threshold |
| $N_{\text{pix}}$ | $6 \times N_{\text{sub}} = 120$ | Pupil resolution |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\phi_{\text{atm}}^{(k)}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Atmospheric phase screen at iteration $k$ |
| $\mathbf{M}_{\text{KL}}$ | $(N_{\text{act}}, N_{\text{modes}})$ | float64 | Mode-to-command matrix (KL basis) |
| $\mathbf{D}_{\text{IM}}$ | $(N_{\text{meas}}, N_{\text{modes}})$ | float64 | Interaction matrix |
| $\mathbf{R}$ | $(N_{\text{act}}, N_{\text{meas}})$ | float64 | Final reconstructor matrix |
| $\mathbf{s}_{\text{ref}}$ | $(N_{\text{meas}},)$ | float64 | Reference slopes (flat wavefront) |
| $I_{\text{ref}}$ | $(2N_{\text{pix}}, 2N_{\text{pix}})$ | float64 | Reference PSF (diffraction-limited) |

---

## 3. PREPROCESSING PIPELINE

### Step P1: Telescope Initialization
- **step_order**: P1
- **step_name**: Initialize Telescope
- **formula**: 
$$N_{\text{pix}} = 6 \times N_{\text{sub}}, \quad \text{resolution} = N_{\text{pix}} \times N_{\text{pix}}$$
- **inputs**: $N_{\text{sub}} = 20$, $D = 8.0$ m
- **outputs**: Telescope object $\mathcal{T}$ with pupil mask $(N_{\text{pix}}, N_{\text{pix}})$

### Step P2: Source-Telescope Coupling
- **step_order**: P2
- **step_name**: Couple NGS to Telescope
- **formula**: 
$$E_{\text{src}} = A \cdot \exp(i \cdot 0), \quad \text{where } A = 10^{-0.4 \times m / 2.5}$$
- **inputs**: Magnitude $m = 8$, optical band = 'I'
- **outputs**: Source object $\mathcal{S}$ coupled to $\mathcal{T}$

### Step P3: Atmosphere Initialization
- **step_order**: P3
- **step_name**: Initialize Atmosphere
- **formula**: 
$$\phi_{\text{atm}} \sim \mathcal{N}(0, C_\phi), \quad C_\phi(r) = 6.88 \left(\frac{r}{r_0}\right)^{5/3} \left[1 - \left(\frac{r}{L_0}\right)^{1/3}\right]$$
- **inputs**: $r_0 = 0.15$ m, $L_0 = 25$ m, $v = 10$ m/s
- **outputs**: Atmosphere object $\mathcal{A}$ with phase screen $(N_{\text{pix}}, N_{\text{pix}})$

### Step P4: Deformable Mirror Initialization
- **step_order**: P4
- **step_name**: Initialize DM
- **formula**: 
$$\text{Influence function: } f_j(\mathbf{r}) = \exp\left(-\frac{\|\mathbf{r} - \mathbf{r}_j\|^2}{2\sigma_{\text{IF}}^2}\right), \quad \sigma_{\text{IF}} \propto \alpha_{\text{coupling}}$$
- **inputs**: $N_{\text{sub}} = 20$, $\alpha_{\text{coupling}} = 0.35$
- **outputs**: DM object $\mathcal{M}$ with $N_{\text{act}}$ actuators

### Step P5: WFS Initialization
- **step_order**: P5
- **step_name**: Initialize Shack-Hartmann WFS
- **formula**: 
$$\text{Valid subaperture: } \text{valid}_i = \mathbb{1}\left[\frac{\text{illuminated pixels in subap } i}{\text{total pixels in subap } i} > \rho_{\text{light}}\right]$$
- **inputs**: $N_{\text{sub}} = 20$, $\rho_{\text{light}} = 0.5$
- **outputs**: WFS object $\mathcal{W}$ with $N_{\text{valid}}$ valid subapertures, $N_{\text{meas}} = 2 N_{\text{valid}}$

### Step P6: Compute Reference PSF
- **step_order**: P6
- **step_name**: Compute Diffraction-Limited PSF
- **formula**: 
$$I_{\text{ref}} = \left| \mathcal{F}\{ P \cdot E_{\text{src}} \} \right|^2$$
where $P$ is the pupil mask and $\mathcal{F}$ is the 2D FFT
- **inputs**: Telescope $\mathcal{T}$, Source $\mathcal{S}$, flat OPD
- **outputs**: $I_{\text{ref}} \in \mathbb{R}^{2N_{\text{pix}} \times 2N_{\text{pix}}}$

### Step P7: Compute Reference Slopes
- **step_order**: P7
- **step_name**: Measure Reference Slopes
- **formula**: See Step P9 centroiding formula with flat wavefront $\phi = 0$
- **inputs**: WFS $\mathcal{W}$, Telescope $\mathcal{T}$, Source $\mathcal{S}$
- **outputs**: $\mathbf{s}_{\text{ref}} \in \mathbb{R}^{N_{\text{meas}}}$

### Step P8: Compute KL Modal Basis
- **step_order**: P8
- **step_name**: Compute KL Modes
- **formula**: 
$$\mathbf{M}_{\text{KL}} = \text{KL\_basis}(\mathcal{T}, \mathcal{A}, \mathcal{M})[:, :N_{\text{modes}}]$$
Karhunen-Loève decomposition of atmospheric covariance projected onto DM actuator space
- **inputs**: $\mathcal{T}, \mathcal{A}, \mathcal{M}$, $N_{\text{modes}} = 20$
- **outputs**: $\mathbf{M}_{\text{KL}} \in \mathbb{R}^{N_{\text{act}} \times N_{\text{modes}}}$

### Step P9: Build Interaction Matrix (Push-Pull)
- **step_order**: P9
- **step_name**: Calibrate Interaction Matrix
- **formula**: 
For each mode $j = 1, \ldots, N_{\text{modes}}$:
$$\mathbf{u}_{\text{push}} = +\delta \cdot \mathbf{M}_{\text{KL}}[:, j]$$
$$\mathbf{u}_{\text{pull}} = -\delta \cdot \mathbf{M}_{\text{KL}}[:, j]$$
$$\mathbf{s}_{\text{push}} = \text{Centroid}(\text{WFS}(\mathbf{u}_{\text{push}}))$$
$$\mathbf{s}_{\text{pull}} = \text{Centroid}(\text{WFS}(\mathbf{u}_{\text{pull}}))$$
$$\mathbf{D}_{\text{IM}}[:, j] = \frac{\mathbf{s}_{\text{push}} - \mathbf{s}_{\text{pull}}}{2\delta}$$

**Centroiding formula**:
$$E_i = \text{lenslet\_em\_field}(\phi)_i \in \mathbb{C}^{N_p \times N_p}$$
$$I_i = \left| \mathcal{F}\{E_i\} \right|^2$$
$$c_{x,i} = \frac{\sum_{m,n} I_i[m,n] \cdot X[m,n]}{\sum_{m,n} I_i[m,n]}, \quad c_{y,i} = \frac{\sum_{m,n} I_i[m,n] \cdot Y[m,n]}{\sum_{m,n} I_i[m,n]}$$
where $X[m,n] = n - N_p/2$, $Y[m,n] = m - N_p/2$
- **inputs**: $\mathbf{M}_{\text{KL}}$, $\delta = 10^{-8}$, WFS $\mathcal{W}$
- **outputs**: $\mathbf{D}_{\text{IM}} \in \mathbb{R}^{N_{\text{meas}} \times N_{\text{modes}}}$

### Step P10: Compute Reconstructor via Truncated SVD
- **step_order**: P10
- **step_name**: SVD Pseudo-Inverse
- **formula**: 
$$\mathbf{D}_{\text{IM}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$
$$\sigma_i^{+} = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > \tau \\ 0 & \text{otherwise} \end{cases}$$
$$\mathbf{R}_{\text{modal}} = \mathbf{V} \boldsymbol{\Sigma}^{+} \mathbf{U}^T \in \mathbb{R}^{N_{\text{modes}} \times N_{\text{meas}}}$$
- **inputs**: $\mathbf{D}_{\text{IM}}$, $\tau = 10^{-3}$
- **outputs**: $\mathbf{R}_{\text{modal}} \in \mathbb{R}^{N_{\text{modes}} \times N_{\text{meas}}}$

### Step P11: Convert Modal to Zonal Reconstructor
- **step_order**: P11
- **step_name**: Modal-to-Zonal Conversion
- **formula**: 
$$\mathbf{R} = \mathbf{M}_{\text{KL}} \cdot \mathbf{R}_{\text{modal}}$$
- **inputs**: $\mathbf{M}_{\text{KL}} \in \mathbb{R}^{N_{\text{act}} \times N_{\text{modes}}}$, $\mathbf{R}_{\text{modal}} \in \mathbb{R}^{N_{\text{modes}} \times N_{\text{meas}}}$
- **outputs**: $\mathbf{R} \in \mathbb{R}^{N_{\text{act}} \times N_{\text{meas}}}$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize DM Commands | $\mathbf{u}^{(0)} = \mathbf{0}$ | $N_{\text{act}}$ | $\mathbf{u}^{(0)} \in \mathbb{R}^{N_{\text{act}}}$ | Zero vector allocation |
| 2 | Initialize Strehl History | $\text{SR\_history} = []$ | — | Empty list | Storage for convergence tracking |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** ($k = 1, \ldots, K_{\max}$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 3 | Update Atmosphere | $\phi_{\text{atm}}^{(k)} = \text{shift}(\phi_{\text{atm}}^{(k-1)}, v_{\text{wind}} \cdot \Delta t)$ | $\phi_{\text{atm}}^{(k-1)}$, $v_{\text{wind}}$, $\Delta t = 1/1000$ s | $\phi_{\text{atm}}^{(k)} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ | Taylor frozen flow |
| 4 | Compute Residual Phase | $\phi_{\text{res}}^{(k)} = \phi_{\text{atm}}^{(k)} - \phi_{\text{DM}}(\mathbf{u}^{(k-1)})$ | $\phi_{\text{atm}}^{(k)}$, $\mathbf{u}^{(k-1)}$ | $\phi_{\text{res}}^{(k)} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ | [FORWARD] DM surface: $\phi_{\text{DM}} = 2 \sum_j u_j f_j(\mathbf{r})$ |
| 5 | Compute Lenslet E-fields | $E_i^{(k)} = \text{crop}_i\left( P \cdot \exp(i \phi_{\text{res}}^{(k)}) \right)$ for $i = 1, \ldots, N_{\text{sub}}^2$ | $\phi_{\text{res}}^{(k)}$, pupil $P$ | $\mathbf{E}^{(k)} \in \mathbb{C}^{N_{\text{sub}}^2 \times N_p \times N_p}$ | Extract subaperture fields |
| 6 | Compute Spot Intensities | $I_i^{(k)} = \left\| \mathcal{F}\{E_i^{(k)}\} \right\|^2$ for all valid $i$ | $\mathbf{E}^{(k)}$ | $\mathbf{I}^{(k)} \in \mathbb{R}^{N_{\text{sub}}^2 \times N_p \times N_p}$ | FFT per subaperture |
| 7 | Compute X-Centroids | $c_{x,i}^{(k)} = \frac{\sum_{m,n} I_i^{(k)}[m,n] \cdot (n - N_p/2)}{\sum_{m,n} I_i^{(k)}[m,n]}$ for valid $i$ | $\mathbf{I}^{(k)}$, coordinate grid $X$ | $\mathbf{c}_x^{(k)} \in \mathbb{R}^{N_{\text{valid}}}$ | Center of gravity |
| 8 | Compute Y-Centroids | $c_{y,i}^{(k)} = \frac{\sum_{m,n} I_i^{(k)}[m,n] \cdot (m - N_p/2)}{\sum_{m,n} I_i^{(k)}[m,n]}$ for valid $i$ | $\mathbf{I}^{(k)}$, coordinate grid $Y$ | $\mathbf{c}_y^{(k)} \in \mathbb{R}^{N_{\text{valid}}}$ | Center of gravity |
| 9 | Concatenate Slopes | $\mathbf{s}_{\text{meas}}^{(k)} = [\mathbf{c}_x^{(k)}; \mathbf{c}_y^{(k)}]$ | $\mathbf{c}_x^{(k)}$, $\mathbf{c}_y^{(k)}$ | $\mathbf{s}_{\text{meas}}^{(k)} \in \mathbb{R}^{N_{\text{meas}}}$ | Vertical concatenation |
| 10 | Compute Residual Slopes | $\mathbf{s}_{\text{res}}^{(k)} = \mathbf{s}_{\text{meas}}^{(k)} - \mathbf{s}_{\text{ref}}$ | $\mathbf{s}_{\text{meas}}^{(k)}$, $\mathbf{s}_{\text{ref}}$ | $\mathbf{s}_{\text{res}}^{(k)} \in \mathbb{R}^{N_{\text{meas}}}$ | Remove reference offset |
| 11 | Compute Command Increment | $\Delta\mathbf{u}^{(k)} = \mathbf{R} \cdot \mathbf{s}_{\text{res}}^{(k)}$ | $\mathbf{R} \in \mathbb{R}^{N_{\text{act}} \times N_{\text{meas}}}$, $\mathbf{s}_{\text{res}}^{(k)}$ | $\Delta\mathbf{u}^{(k)} \in \mathbb{R}^{N_{\text{act}}}$ | Matrix-vector product |
| 12 | Update DM Commands | $\mathbf{u}^{(k)} = \mathbf{u}^{(k-1)} - g \cdot \Delta\mathbf{u}^{(k)}$ | $\mathbf{u}^{(k-1)}$, $\Delta\mathbf{u}^{(k)}$, $g = 0.4$ | $\mathbf{u}^{(k)} \in \mathbb{R}^{N_{\text{act}}}$ | Integral control law |
| 13 | Compute Science PSF | $I_{\text{sci}}^{(k)} = \left\| \mathcal{F}\{ P \cdot \exp(i \phi_{\text{res,updated}}^{(k)}) \} \right\|^2$ | Updated $\phi_{\text{res}}^{(k)}$ with $\mathbf{u}^{(k)}$ | $I_{\text{sci}}^{(k)} \in \mathbb{R}^{2N_{\text{pix}} \times 2N_{\text{pix}}}$ | Science camera image |
| 14 | Compute OTF | $\text{OTF}^{(k)} = \left\| \mathcal{F}_{\text{shift}}\{ \mathcal{F}\{ I_{\text{sci}}^{(k)} \} \} \right\|$ | $I_{\text{sci}}^{(k)}$ | $\text{OTF}^{(k)} \in \mathbb{R}^{2N_{\text{pix}} \times 2N_{\text{pix}}}$ | Absolute value of shifted FFT |
| 15 | Compute Strehl Ratio | $\text{SR}^{(k)} = 100 \times \frac{\sum_{m,n} \text{OTF}^{(k)}[m,n]}{\sum_{m,n} \text{OTF}_{\text{ref}}[m,n]}$ | $\text{OTF}^{(k)}$, $\text{OTF}_{\text{ref}}$ | $\text{SR}^{(k)} \in \mathbb{R}$ (percent) | OTF-based Strehl |
| 16 | Store Strehl | $\text{SR\_history}.\text{append}(\text{SR}^{(k)})$ | $\text{SR}^{(k)}$ | Updated history list | Tracking |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 17 | Extract Final PSF | $I_{\text{final}} = I_{\text{sci}}^{(K_{\max})}$ | $I_{\text{sci}}^{(K_{\max})}$ | $I_{\text{final}} \in \mathbb{R}^{2N_{\text{pix}} \times 2N_{\text{pix}}}$ | Copy final science image |
| 18 | Extract Final Commands | $\mathbf{u}_{\text{final}} = \mathbf{u}^{(K_{\max})}$ | $\mathbf{u}^{(K_{\max})}$ | $\mathbf{u}_{\text{final}} \in \mathbb{R}^{N_{\text{act}}}$ | Copy final DM shape |
| 19 | Convert History to Array | $\mathbf{SR} = \text{array}(\text{SR\_history})$ | SR\_history | $\mathbf{SR} \in \mathbb{R}^{K_{\max}}$ | For evaluation |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: 
$$k \geq K_{\max}$$
- **parameters**: 
  - $K_{\max} = 20$ (maximum iterations)
- **check_frequency**: After each complete iteration (step 16)

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Strehl Ratio | $\text{SR} = 100 \times \frac{\sum_{m,n} \|\mathcal{F}_{\text{shift}}\{\mathcal{F}\{I_{\text{sci}}\}\|[m,n]}{\sum_{m,n} \|\mathcal{F}_{\text{shift}}\{\mathcal{F}\{I_{\text{ref}}\}\}\|[m,n]}$ | $I_{\text{ref}}$ (diffraction-limited PSF) | $I_{\text{sci}}$ (corrected PSF) | Quality metric in percent; 100% = perfect correction |
| Initial Strehl | $\text{SR}_{\text{init}} = \text{SR}^{(1)}$ | $I_{\text{ref}}$ | $I_{\text{sci}}^{(1)}$ | Performance before convergence |
| Final Strehl | $\text{SR}_{\text{final}} = \text{SR}^{(K_{\max})}$ | $I_{\text{ref}}$ | $I_{\text{sci}}^{(K_{\max})}$ | Performance at termination |
| Mean Strehl | $\overline{\text{SR}} = \frac{1}{K_{\max}} \sum_{k=1}^{K_{\max}} \text{SR}^{(k)}$ | — | $\mathbf{SR}$ | Average performance |
| DM RMS | $\text{RMS}_{\text{DM}} = \sqrt{\frac{1}{N_{\text{act}}} \sum_{j=1}^{N_{\text{act}}} (u_{\text{final},j})^2}$ | — | $\mathbf{u}_{\text{final}}$ | Actuator stroke magnitude (meters) |