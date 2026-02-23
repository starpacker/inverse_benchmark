## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: Closed-Loop Adaptive Optics Integral Controller with Shack-Hartmann Wavefront Sensing
- **algorithm_family**: Linear Least-Squares Inversion with Integral Feedback Control
- **summary**: Iteratively corrects atmospheric wavefront aberrations using a Shack-Hartmann wavefront sensor and deformable mirror via modal reconstruction and integral control law.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{u}} \|\boldsymbol{\phi}_{\text{res}}\|^2 \quad \text{where} \quad \boldsymbol{\phi}_{\text{res}} = \boldsymbol{\phi}_{\text{atm}} - \mathbf{N}\mathbf{u}$$

Equivalently, in the slope domain:
$$\min_{\mathbf{u}} \|\mathbf{s} - \mathbf{D}\mathbf{u}\|^2$$

### data_fidelity_term
$$\mathcal{L}(\mathbf{u}) = \|\mathbf{s}_{\text{meas}} - \mathbf{s}_{\text{ref}}\|^2 = \|\mathbf{D}(\boldsymbol{\phi}_{\text{atm}} - \mathbf{N}\mathbf{u})\|^2$$

where:
- $\mathbf{s}_{\text{meas}} \in \mathbb{R}^{n_{\text{meas}}}$: measured WFS slopes
- $\mathbf{s}_{\text{ref}} \in \mathbb{R}^{n_{\text{meas}}}$: reference slopes (flat wavefront)
- $\mathbf{D} \in \mathbb{R}^{n_{\text{meas}} \times n_{\text{act}}}$: interaction matrix
- $\mathbf{N} \in \mathbb{R}^{n_{\text{pix}} \times n_{\text{act}}}$: DM influence functions

### regularization_term
Implicit Tikhonov regularization via SVD truncation:
$$\mathbf{R} = \mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^T \quad \text{where} \quad \sigma_i^{+} = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > \tau \\ 0 & \text{otherwise} \end{cases}$$

with threshold $\tau = 10^{-3}$.

### constraints
None (unconstrained actuator commands)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{u}^{(k)}$ | primal | $(n_{\text{act}},)$ | float64 | $\mathbf{0}$ | DM actuator commands at iteration $k$ |
| $\Delta\mathbf{u}^{(k)}$ | primal | $(n_{\text{act}},)$ | float64 | computed | Command increment at iteration $k$ |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{s}^{(k)}$ | auxiliary | $(n_{\text{meas}},)$ | float64 | computed | Residual slopes at iteration $k$: $\mathbf{s}_{\text{meas}} - \mathbf{s}_{\text{ref}}$ |
| $\mathbf{c}_x, \mathbf{c}_y$ | auxiliary | $(n_{\text{valid}},)$ | float64 | computed | Centroid positions in x and y |
| $\mathbf{I}_i$ | auxiliary | $(n_{\text{pix}}, n_{\text{pix}})$ | float64 | computed | Intensity pattern in subaperture $i$ |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\boldsymbol{\phi}_{\text{atm}}$ | observations | $(n_{\text{pix\_pupil}}, n_{\text{pix\_pupil}})$ | float64 | Kolmogorov phase screen | Atmospheric phase aberration |
| $\text{PSF}^{(k)}$ | observations | $(2n_{\text{pix\_pupil}}, 2n_{\text{pix\_pupil}})$ | float64 | computed | Science camera image at iteration $k$ |

### Constants (Calibration Matrices)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{D}$ | constants | $(n_{\text{meas}}, n_{\text{modes}})$ | float64 | Push-pull calibration | Interaction matrix (modal) |
| $\mathbf{R}_{\text{modal}}$ | constants | $(n_{\text{modes}}, n_{\text{meas}})$ | float64 | Truncated SVD pseudo-inverse | Modal reconstructor |
| $\mathbf{M}_{\text{KL}}$ | constants | $(n_{\text{act}}, n_{\text{modes}})$ | float64 | KL basis computation | Mode-to-command matrix |
| $\mathbf{R}$ | constants | $(n_{\text{act}}, n_{\text{meas}})$ | float64 | $\mathbf{M}_{\text{KL}} \mathbf{R}_{\text{modal}}$ | Final zonal reconstructor |
| $\mathbf{s}_{\text{ref}}$ | constants | $(n_{\text{meas}},)$ | float64 | WFS on flat wavefront | Reference slopes |
| $\text{PSF}_{\text{ref}}$ | constants | $(2n_{\text{pix\_pupil}}, 2n_{\text{pix\_pupil}})$ | float64 | Diffraction-limited propagation | Reference PSF |

### Numerical Constants (Hardcoded Parameters)
| Symbol | Category | Value | Description |
|--------|----------|-------|-------------|
| $n_{\text{subap}}$ | constants | 20 | Number of WFS subapertures per dimension |
| $n_{\text{modes}}$ | constants | 20 | Number of controlled KL modes |
| $n_{\text{pix\_pupil}}$ | constants | $6 \times n_{\text{subap}} = 120$ | Pupil resolution (pixels) |
| $D_{\text{tel}}$ | constants | 8.0 m | Telescope diameter |
| $T_s$ | constants | 0.001 s | Sampling time (1/1000) |
| $\epsilon_{\text{obs}}$ | constants | 0.0 | Central obstruction ratio |
| $r_0$ | constants | 0.15 m | Fried parameter |
| $L_0$ | constants | 25 m | Outer scale |
| $v_{\text{wind}}$ | constants | 10 m/s | Wind speed |
| $\theta_{\text{wind}}$ | constants | 0° | Wind direction |
| $h_{\text{layer}}$ | constants | 0 m | Turbulence layer altitude |
| $\alpha_{\text{mech}}$ | constants | 0.35 | DM mechanical coupling |
| $\rho_{\text{light}}$ | constants | 0.5 | WFS light ratio threshold |
| $\delta$ | constants | $10^{-8}$ m | Push-pull stroke amplitude |
| $\tau$ | constants | $10^{-3}$ | SVD truncation threshold |
| $g$ | constants | 0.4 | Integral control gain |
| $K_{\text{max}}$ | constants | 20 | Maximum number of iterations |

---

## 3. PREPROCESSING PIPELINE

| Step Order | Step Name | Formula | Inputs | Outputs |
|------------|-----------|---------|--------|---------|
| 3.1 | Telescope Initialization | $n_{\text{pix}} = 6 \times n_{\text{subap}}$; create pupil mask $\mathbf{P} \in \{0,1\}^{n_{\text{pix}} \times n_{\text{pix}}}$ | $n_{\text{subap}}=20$, $D=8.0$ | `tel` object with pupil |
| 3.2 | Source Initialization | NGS at infinity: $\mathbf{E}_{\text{src}} = A \cdot \mathbf{P}$ with magnitude 8, I-band | coordinates $[0,0]$ | `ngs` object |
| 3.3 | Atmosphere Initialization | Kolmogorov phase screen: $\Phi_{\phi}(\mathbf{f}) = 0.023 r_0^{-5/3} \|\mathbf{f}\|^{-11/3}$ | $r_0=0.15$, $L_0=25$ | `atm` object with phase screen |
| 3.4 | DM Initialization | Gaussian influence functions: $\mathbf{N}_i(\mathbf{r}) = \exp\left(-\frac{\|\mathbf{r}-\mathbf{r}_i\|^2}{2\sigma^2}\right)$ with $\sigma$ from coupling 0.35 | $n_{\text{act}} = (n_{\text{subap}}+1)^2 = 441$ | `dm` object |
| 3.5 | WFS Initialization | Shack-Hartmann: $n_{\text{valid}}$ subapertures where light ratio $> 0.5$ | $n_{\text{subap}}=20$ | `wfs` object, $n_{\text{meas}} = 2 \times n_{\text{valid}}$ |
| 3.6 | Science Camera Initialization | Detector array size: $2 \times n_{\text{pix\_pupil}} = 240$ | — | `sci_cam` object |
| 3.7 | Reference PSF Computation | $\text{PSF}_{\text{ref}} = \|\mathcal{F}\{\mathbf{P} \cdot \mathbf{E}_{\text{src}}\}\|^2$ | `tel` with reset OPD | $\text{PSF}_{\text{ref}} \in \mathbb{R}^{240 \times 240}$ |
| 3.8 | Reference Slopes Computation | $\mathbf{s}_{\text{ref}} = \text{SH}_{\text{measure}}(\boldsymbol{\phi}=\mathbf{0})$ via diffractive model | flat wavefront | $\mathbf{s}_{\text{ref}} \in \mathbb{R}^{n_{\text{meas}}}$ |
| 3.9 | KL Basis Computation | $\mathbf{M}_{\text{KL}} = \text{compute\_KL\_basis}(\text{tel}, \text{atm}, \text{dm}, \text{lim}=0)$ truncated to $n_{\text{modes}}$ columns | telescope, atmosphere, DM | $\mathbf{M}_{\text{KL}} \in \mathbb{R}^{441 \times 20}$ |
| 3.10 | Interaction Matrix (Push) | $\mathbf{s}_{\text{push},i} = \text{SH}_{\text{measure}}(\mathbf{N} \mathbf{M}_{\text{KL},i} \delta)$ | mode $i$, stroke $\delta=10^{-8}$ | slopes for push |
| 3.11 | Interaction Matrix (Pull) | $\mathbf{s}_{\text{pull},i} = \text{SH}_{\text{measure}}(-\mathbf{N} \mathbf{M}_{\text{KL},i} \delta)$ | mode $i$, stroke $-\delta$ | slopes for pull |
| 3.12 | Interaction Matrix Column | $\mathbf{D}_{:,i} = \frac{\mathbf{s}_{\text{push},i} - \mathbf{s}_{\text{pull},i}}{2\delta}$ | push/pull slopes | column $i$ of $\mathbf{D}$ |
| 3.13 | SVD Decomposition | $\mathbf{D} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | $\mathbf{D} \in \mathbb{R}^{n_{\text{meas}} \times n_{\text{modes}}}$ | $\mathbf{U}, \boldsymbol{\Sigma}, \mathbf{V}$ |
| 3.14 | Truncated Pseudo-inverse | $\boldsymbol{\Sigma}^+_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > 10^{-3} \\ 0 & \text{otherwise} \end{cases}$ | singular values | $\boldsymbol{\Sigma}^+ \in \mathbb{R}^{n_{\text{modes}} \times n_{\text{modes}}}$ |
| 3.15 | Modal Reconstructor | $\mathbf{R}_{\text{modal}} = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^T$ | SVD components | $\mathbf{R}_{\text{modal}} \in \mathbb{R}^{n_{\text{modes}} \times n_{\text{meas}}}$ |
| 3.16 | Zonal Reconstructor | $\mathbf{R} = \mathbf{M}_{\text{KL}} \mathbf{R}_{\text{modal}}$ | modal reconstructor, KL basis | $\mathbf{R} \in \mathbb{R}^{n_{\text{act}} \times n_{\text{meas}}}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 4.1.1 | Initialize DM Commands | $\mathbf{u}^{(0)} = \mathbf{0}$ | — | $\mathbf{u}^{(0)} \in \mathbb{R}^{n_{\text{act}}}$ |
| 4.1.2 | Initialize Strehl History | $\mathcal{S} = []$ | — | empty list |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** ($k = 0, 1, \ldots, K_{\text{max}}-1$ where $K_{\text{max}} = 20$)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| 4.2.1 | Atmosphere Update | $\boldsymbol{\phi}_{\text{atm}}^{(k+1)} = \text{atm.update()}$ — translates phase screen by $v_{\text{wind}} \cdot T_s$ | $\boldsymbol{\phi}_{\text{atm}}^{(k)}$, wind parameters | $\boldsymbol{\phi}_{\text{atm}}^{(k+1)}$ | Frozen-flow Taylor hypothesis |
| 4.2.2 | Optical Propagation (WFS path) | $\boldsymbol{\phi}_{\text{res}} = \boldsymbol{\phi}_{\text{atm}}^{(k+1)} - \sum_{j} u_j^{(k)} \mathbf{N}_j$ | atmosphere, DM commands | residual phase at WFS | DM operates in reflection |
| 4.2.3 | **BEGIN_SUBBLOCK: Shack-Hartmann Slope Measurement** | | | | |
| 4.2.3.1 | Lenslet Field Extraction | $\mathbf{E}_i = \mathbf{P}_i \cdot \exp(j\boldsymbol{\phi}_{\text{res},i})$ for subaperture $i$ | residual phase | complex field cube $\mathbf{E} \in \mathbb{C}^{n_{\text{subap}}^2 \times n_{\text{pix\_sa}} \times n_{\text{pix\_sa}}}$ | $n_{\text{pix\_sa}} = 6$ pixels per subaperture |
| 4.2.3.2 | Spot Formation (FFT) | $\tilde{\mathbf{E}}_i = \mathcal{F}\{\mathbf{E}_i\}$ | lenslet fields | complex spectrum |
| 4.2.3.3 | Intensity Computation | $\mathbf{I}_i = |\tilde{\mathbf{E}}_i|^2$ | complex spectrum | intensity patterns |
| 4.2.3.4 | Centroid Grid Setup | $X_{mn} = m - \lfloor n_{\text{pix}}/2 \rfloor$, $Y_{mn} = n - \lfloor n_{\text{pix}}/2 \rfloor$ | pixel count | coordinate grids $\mathbf{X}, \mathbf{Y}$ |
| 4.2.3.5 | Center-of-Gravity Centroiding | $c_{x,i} = \frac{\sum_{m,n} I_{i,mn} X_{mn}}{\sum_{m,n} I_{i,mn}}$, $c_{y,i} = \frac{\sum_{m,n} I_{i,mn} Y_{mn}}{\sum_{m,n} I_{i,mn}}$ | intensity $\mathbf{I}_i$, grids | centroids $(c_{x,i}, c_{y,i})$ | Only for valid subapertures |
| 4.2.3.6 | Slope Vector Assembly | $\mathbf{s}_{\text{meas}} = [c_{x,1}, \ldots, c_{x,n_{\text{valid}}}, c_{y,1}, \ldots, c_{y,n_{\text{valid}}}]^T$ | all centroids | $\mathbf{s}_{\text{meas}} \in \mathbb{R}^{2n_{\text{valid}}}$ |
| 4.2.3 | **END_SUBBLOCK** | | | | |
| 4.2.4 | Residual Slope Computation | $\mathbf{s}^{(k)} = \mathbf{s}_{\text{meas}} - \mathbf{s}_{\text{ref}}$ | measured slopes, reference slopes | $\mathbf{s}^{(k)} \in \mathbb{R}^{n_{\text{meas}}}$ | Reference subtraction |
| 4.2.5 | Wavefront Reconstruction | $\Delta\mathbf{u}^{(k)} = \mathbf{R} \mathbf{s}^{(k)}$ | reconstructor, residual slopes | $\Delta\mathbf{u}^{(k)} \in \mathbb{R}^{n_{\text{act}}}$ | Matrix-vector product |
| 4.2.6 | Integral Control Update | $\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} - g \cdot \Delta\mathbf{u}^{(k)}$ | previous commands, increment, $g=0.4$ | $\mathbf{u}^{(k+1)} \in \mathbb{R}^{n_{\text{act}}}$ | Leaky integrator with gain |
| 4.2.7 | Science Path Propagation | $\text{PSF}^{(k)} = |\mathcal{F}\{\mathbf{P} \cdot \exp(j\boldsymbol{\phi}_{\text{res}})\}|^2$ | corrected residual phase | science camera image | Full aperture propagation |
| 4.2.8 | **BEGIN_SUBBLOCK: Strehl Computation** | | | | |
| 4.2.8.1 | OTF Computation (Measured) | $\text{OTF}^{(k)} = |\mathcal{F}_{\text{shift}}\{\mathcal{F}\{\text{PSF}^{(k)}\}\}|$ | current PSF | OTF magnitude |
| 4.2.8.2 | OTF Computation (Reference) | $\text{OTF}_{\text{ref}} = |\mathcal{F}_{\text{shift}}\{\mathcal{F}\{\text{PSF}_{\text{ref}}\}\}|$ | reference PSF | reference OTF magnitude |
| 4.2.8.3 | Strehl Ratio | $\text{SR}^{(k)} = 100 \times \frac{\sum_{m,n} \text{OTF}^{(k)}_{mn}}{\sum_{m,n} \text{OTF}_{\text{ref},mn}}$ | both OTFs | Strehl in percent |
| 4.2.8 | **END_SUBBLOCK** | | | | |
| 4.2.9 | Record Strehl | $\mathcal{S}.\text{append}(\text{SR}^{(k)})$ | Strehl value | updated history |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 4.3.1 | Extract Final PSF | $\text{PSF}_{\text{final}} = \text{PSF}^{(K_{\text{max}}-1)}$ | last iteration PSF | final PSF array |
| 4.3.2 | Extract Final Commands | $\mathbf{u}_{\text{final}} = \mathbf{u}^{(K_{\text{max}})}$ | last DM state | final command vector |
| 4.3.3 | Convert Strehl History | $\mathbf{SR} = \text{np.array}(\mathcal{S})$ | Strehl list | numpy array |

---

## 5. STOPPING CRITERION

| Property | Value |
|----------|-------|
| **type** | iteration_limit |
| **expression** | $k < K_{\text{max}}$ where $K_{\text{max}} = 20$ |
| **parameters** | `n_iter = 20` |
| **check_frequency** | Every iteration (at loop condition) |

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| **Strehl Ratio (OTF method)** | $\text{SR} = 100 \times \frac{\sum_{m,n} |\mathcal{F}_{\text{shift}}\{\mathcal{F}\{\text{PSF}\}\}|_{mn}}{\sum_{m,n} |\mathcal{F}_{\text{shift}}\{\mathcal{F}\{\text{PSF}_{\text{ref}}\}\}|_{mn}}$ | Ratio of integrated OTF to diffraction-limited OTF, expressed as percentage |
| **Initial Strehl** | $\text{SR}_{\text{init}} = \text{SR}^{(0)}$ | Strehl ratio at first iteration (before significant correction) |
| **Final Strehl** | $\text{SR}_{\text{final}} = \text{SR}^{(K_{\text{max}}-1)}$ | Strehl ratio at last iteration |
| **Mean Strehl** | $\overline{\text{SR}} = \frac{1}{K_{\text{max}}} \sum_{k=0}^{K_{\text{max}}-1} \text{SR}^{(k)}$ | Average Strehl ratio over all iterations |
| **Max Strehl** | $\text{SR}_{\max} = \max_k \text{SR}^{(k)}$ | Maximum achieved Strehl ratio |
| **Min Strehl** | $\text{SR}_{\min} = \min_k \text{SR}^{(k)}$ | Minimum Strehl ratio during loop |
| **DM RMS Command** | $\text{RMS}_{\mathbf{u}} = \sqrt{\frac{1}{n_{\text{act}}} \sum_{j=1}^{n_{\text{act}}} u_{\text{final},j}^2}$ | Root-mean-square of final DM actuator commands (in meters) |