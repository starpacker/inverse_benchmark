# Algorithmic Specification Document: Closed-Loop Adaptive Optics with Shack-Hartmann Wavefront Sensor

---

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{u}} \quad \mathbb{E}\left[ \| \boldsymbol{\phi}_{\text{res}}(\mathbf{u}) \|^2 \right]$$

where the residual wavefront is:
$$\boldsymbol{\phi}_{\text{res}} = \boldsymbol{\phi}_{\text{atm}} - \mathbf{N}_{\text{DM}} \mathbf{u}$$

This objective is implicitly minimized via integral control feedback.

**data_fidelity_term**:
$$\mathcal{D}(\mathbf{u}) = \| \mathbf{s}_{\text{meas}} - \mathbf{s}_{\text{ref}} \|^2$$

where $\mathbf{s}_{\text{meas}} = \mathbf{G}(\boldsymbol{\phi}_{\text{atm}} - \mathbf{N}_{\text{DM}} \mathbf{u})$ represents the WFS slope measurement operator applied to the residual phase.

**regularization_term**:
Implicit regularization through SVD truncation during reconstructor computation:
$$\mathcal{R}(\mathbf{R}) = \text{truncation of singular values below } \tau$$

**constraints**:
- None explicitly enforced (DM stroke limits not implemented in code)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{u}^{(k)}$ | primal | $(N_{\text{act}},)$ | float64 | $\mathbf{0}$ | DM actuator command vector at iteration $k$ |
| $\Delta\mathbf{u}^{(k)}$ | primal | $(N_{\text{act}},)$ | float64 | computed | Command update at iteration $k$ |

### Dual Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| N/A | — | — | — | — | No explicit dual variables |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{R}$ | constants | $(N_{\text{act}}, N_{\text{sig}})$ | float64 | SVD-based | Zonal reconstructor matrix |
| $\mathbf{R}_{\text{modal}}$ | constants | $(N_{\text{modes}}, N_{\text{sig}})$ | float64 | SVD-based | Modal reconstructor matrix |
| $\mathbf{B}$ | constants | $(N_{\text{act}}, N_{\text{modes}})$ | float64 | KL basis | Mode-to-command matrix (KL basis) |
| $\mathbf{M}$ | constants | $(N_{\text{sig}}, N_{\text{modes}})$ | float64 | push-pull | Interaction matrix |
| $\mathbf{s}_{\text{ref}}$ | constants | $(N_{\text{sig}},)$ | float64 | calibration | Reference slopes (flat wavefront) |
| $\text{PSF}_{\text{ref}}$ | constants | $(2N_{\text{pix}}, 2N_{\text{pix}})$ | float64 | diffraction | Reference diffraction-limited PSF |
| $g$ | constants | scalar | float64 | 0.4 | Integral gain |
| $\delta$ | constants | scalar | float64 | $10^{-8}$ | Calibration stroke amplitude [m] |
| $\tau$ | constants | scalar | float64 | $10^{-3}$ | SVD truncation threshold |
| $N_{\text{modes}}$ | constants | scalar | int | 20 | Number of controlled modes |
| $N_{\text{sub}}$ | constants | scalar | int | 20 | Number of subapertures per side |
| $N_{\text{act}}$ | constants | scalar | int | $(N_{\text{sub}}+1)^2$ | Number of DM actuators |
| $N_{\text{sig}}$ | constants | scalar | int | $2 \cdot N_{\text{valid}}$ | Number of WFS signals (2× valid subapertures) |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{s}^{(k)}$ | observations | $(N_{\text{sig}},)$ | float64 | measured | Raw WFS slope measurement at iteration $k$ |
| $\boldsymbol{\phi}_{\text{atm}}^{(k)}$ | observations | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | simulated | Atmospheric phase screen at iteration $k$ |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Reference PSF Computation
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Compute Diffraction-Limited Reference PSF |
| formula | $\text{PSF}_{\text{ref}} = \left\| \mathcal{F}\left\{ P \cdot e^{i \cdot 0} \right\} \right\|^2$ where $P$ is the pupil function |
| inputs | Pupil function $P$ : $(N_{\text{pix}}, N_{\text{pix}})$ |
| outputs | $\text{PSF}_{\text{ref}}$ : $(2N_{\text{pix}}, 2N_{\text{pix}})$ |

### Step 3.2: Reference Slopes Computation
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Compute Reference Slopes (Zero Phase) |
| formula | $\mathbf{s}_{\text{ref}} = \mathcal{W}(\boldsymbol{\phi} = 0)$ where $\mathcal{W}$ is the WFS operator |
| inputs | Zero phase: $(N_{\text{pix}}, N_{\text{pix}})$ |
| outputs | $\mathbf{s}_{\text{ref}}$ : $(N_{\text{sig}},)$ |

### Step 3.3: KL Modal Basis Computation
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Compute Karhunen-Loève Modal Basis |
| formula | $\mathbf{B}_{\text{full}} = \text{KL}(\text{tel}, \text{atm}, \text{DM})$; $\mathbf{B} = \mathbf{B}_{\text{full}}[:, :N_{\text{modes}}]$ |
| inputs | Telescope, atmosphere, DM parameters |
| outputs | $\mathbf{B}$ : $(N_{\text{act}}, N_{\text{modes}})$ |

### Step 3.4: Interaction Matrix Calibration (Push-Pull)
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Build Interaction Matrix via Push-Pull |
| formula | For each mode $i \in [0, N_{\text{modes}})$: $$\mathbf{s}_{\text{push},i} = \mathcal{W}(\mathbf{N}_{\text{DM}} \cdot \mathbf{B}_{:,i} \cdot \delta)$$ $$\mathbf{s}_{\text{pull},i} = \mathcal{W}(\mathbf{N}_{\text{DM}} \cdot (-\mathbf{B}_{:,i}) \cdot \delta)$$ $$\mathbf{M}_{:,i} = \frac{\mathbf{s}_{\text{push},i} - \mathbf{s}_{\text{pull},i}}{2\delta}$$ |
| inputs | $\mathbf{B}$ : $(N_{\text{act}}, N_{\text{modes}})$, $\delta$ : scalar |
| outputs | $\mathbf{M}$ : $(N_{\text{sig}}, N_{\text{modes}})$ |

### Step 3.5: SVD Decomposition of Interaction Matrix
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Singular Value Decomposition |
| formula | $\mathbf{M} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$ where $\mathbf{U} \in \mathbb{R}^{N_{\text{sig}} \times r}$, $\boldsymbol{\Sigma} = \text{diag}(\sigma_1, \ldots, \sigma_r)$, $\mathbf{V} \in \mathbb{R}^{N_{\text{modes}} \times r}$ |
| inputs | $\mathbf{M}$ : $(N_{\text{sig}}, N_{\text{modes}})$ |
| outputs | $\mathbf{U}$ : $(N_{\text{sig}}, r)$, $\boldsymbol{\sigma}$ : $(r,)$, $\mathbf{V}^T$ : $(r, N_{\text{modes}})$ |

### Step 3.6: Truncated Pseudo-Inverse Computation
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Compute Truncated Modal Reconstructor |
| formula | $$\sigma_i^{-1} = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > \tau \\ 0 & \text{otherwise} \end{cases}$$ $$\mathbf{R}_{\text{modal}} = \mathbf{V} \cdot \text{diag}(\boldsymbol{\sigma}^{-1}) \cdot \mathbf{U}^T$$ |
| inputs | $\mathbf{U}$, $\boldsymbol{\sigma}$, $\mathbf{V}^T$, $\tau = 10^{-3}$ |
| outputs | $\mathbf{R}_{\text{modal}}$ : $(N_{\text{modes}}, N_{\text{sig}})$ |

### Step 3.7: Zonal Reconstructor Assembly
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Convert Modal to Zonal Reconstructor |
| formula | $\mathbf{R} = \mathbf{B} \cdot \mathbf{R}_{\text{modal}}$ |
| inputs | $\mathbf{B}$ : $(N_{\text{act}}, N_{\text{modes}})$, $\mathbf{R}_{\text{modal}}$ : $(N_{\text{modes}}, N_{\text{sig}})$ |
| outputs | $\mathbf{R}$ : $(N_{\text{act}}, N_{\text{sig}})$ |

---

## 4. ALGORITHM STEPS

### Initialization

| Field | Value |
|-------|-------|
| step_name | Initialize DM Commands |
| formula | $\mathbf{u}^{(0)} = \mathbf{0}$ |
| inputs | $N_{\text{act}}$ : scalar |
| outputs | $\mathbf{u}^{(0)}$ : $(N_{\text{act}},)$ |
| computational_note | All actuator commands set to zero (flat mirror) |

---

### Iteration Loop

**BEGIN LOOP** for $k = 0, 1, \ldots, N_{\text{iter}} - 1$:

---

#### Step 4.1: Atmosphere Evolution
| Field | Value |
|-------|-------|
| step_name | Update Atmospheric Phase Screen |
| formula | $\boldsymbol{\phi}_{\text{atm}}^{(k)} = \text{atm.update}()$ — frozen-flow turbulence advection |
| inputs | Previous atmosphere state, wind velocity $\mathbf{v}$, time step $\Delta t$ |
| outputs | $\boldsymbol{\phi}_{\text{atm}}^{(k)}$ : $(N_{\text{pix}}, N_{\text{pix}})$ |
| computational_note | Simulates Taylor frozen flow hypothesis |

---

#### Step 4.2: Residual Phase Computation
| Field | Value |
|-------|-------|
| step_name | Compute Residual Wavefront Phase |
| formula | $\boldsymbol{\phi}_{\text{res}}^{(k)} = \boldsymbol{\phi}_{\text{atm}}^{(k)} - \mathbf{N}_{\text{DM}}(\mathbf{u}^{(k)})$ |
| inputs | $\boldsymbol{\phi}_{\text{atm}}^{(k)}$ : $(N_{\text{pix}}, N_{\text{pix}})$, $\mathbf{u}^{(k)}$ : $(N_{\text{act}},)$ |
| outputs | $\boldsymbol{\phi}_{\text{res}}^{(k)}$ : $(N_{\text{pix}}, N_{\text{pix}})$ |
| computational_note | $\mathbf{N}_{\text{DM}}$ is DM influence function interpolation |

---

#### Step 4.3: Lenslet Electric Field Extraction
| Field | Value |
|-------|-------|
| step_name | Extract Subaperture Electric Fields |
| formula | For each valid subaperture $j$: $$E_j = P_j \cdot \exp(i \boldsymbol{\phi}_{\text{res},j}^{(k)})$$ where $P_j$ is the subaperture pupil mask |
| inputs | $\boldsymbol{\phi}_{\text{res}}^{(k)}$ : $(N_{\text{pix}}, N_{\text{pix}})$ |
| outputs | $\mathbf{E}_{\text{cube}}$ : $(N_{\text{sub}}^2, N_{\text{pix/sub}}, N_{\text{pix/sub}})$ |
| computational_note | Each subaperture extracts a local patch of the wavefront |

---

#### Step 4.4: Spot Formation via FFT
| Field | Value |
|-------|-------|
| step_name | Compute WFS Spot Intensities |
| formula | For each subaperture $j$: $$\tilde{E}_j = \mathcal{F}_{2D}\{E_j\}$$ $$I_j = |\tilde{E}_j|^2$$ |
| inputs | $\mathbf{E}_{\text{cube}}$ : $(N_{\text{sub}}^2, N_{\text{pix/sub}}, N_{\text{pix/sub}})$ |
| outputs | $\mathbf{I}_{\text{cube}}$ : $(N_{\text{sub}}^2, N_{\text{pix/sub}}, N_{\text{pix/sub}})$ |
| computational_note | FFT simulates Fraunhofer diffraction at focal plane of lenslet |

---

#### Step 4.5: Center of Gravity Centroiding
| Field | Value |
|-------|-------|
| step_name | Compute Spot Centroids |
| formula | For each valid subaperture $j$: $$c_{x,j} = \frac{\sum_{m,n} I_j[m,n] \cdot X[m,n]}{\sum_{m,n} I_j[m,n]}$$ $$c_{y,j} = \frac{\sum_{m,n} I_j[m,n] \cdot Y[m,n]}{\sum_{m,n} I_j[m,n]}$$ where $X[m,n] = m - N_{\text{pix/sub}}/2$, $Y[m,n] = n - N_{\text{pix/sub}}/2$ |
| inputs | $\mathbf{I}_{\text{cube}}$ : $(N_{\text{sub}}^2, N_{\text{pix/sub}}, N_{\text{pix/sub}})$ |
| outputs | $\mathbf{c}_x$ : $(N_{\text{valid}},)$, $\mathbf{c}_y$ : $(N_{\text{valid}},)$ |
| computational_note | Only computed for subapertures passing illumination threshold |

---

#### Step 4.6: Slope Vector Assembly
| Field | Value |
|-------|-------|
| step_name | Concatenate Slopes into Measurement Vector |
| formula | $\mathbf{s}^{(k)} = [\mathbf{c}_x ; \mathbf{c}_y]$ (vertical concatenation) |
| inputs | $\mathbf{c}_x$ : $(N_{\text{valid}},)$, $\mathbf{c}_y$ : $(N_{\text{valid}},)$ |
| outputs | $\mathbf{s}^{(k)}$ : $(N_{\text{sig}},)$ where $N_{\text{sig}} = 2 \cdot N_{\text{valid}}$ |
| computational_note | Standard WFS signal vector format |

---

#### Step 4.7: Residual Slope Computation
| Field | Value |
|-------|-------|
| step_name | Subtract Reference Slopes |
| formula | $\tilde{\mathbf{s}}^{(k)} = \mathbf{s}^{(k)} - \mathbf{s}_{\text{ref}}$ |
| inputs | $\mathbf{s}^{(k)}$ : $(N_{\text{sig}},)$, $\mathbf{s}_{\text{ref}}$ : $(N_{\text{sig}},)$ |
| outputs | $\tilde{\mathbf{s}}^{(k)}$ : $(N_{\text{sig}},)$ |
| computational_note | Removes static aberrations and WFS bias |

---

#### Step 4.8: Command Update Reconstruction
| Field | Value |
|-------|-------|
| step_name | Compute DM Command Correction |
| formula | $\Delta\mathbf{u}^{(k)} = \mathbf{R} \cdot \tilde{\mathbf{s}}^{(k)}$ |
| inputs | $\mathbf{R}$ : $(N_{\text{act}}, N_{\text{sig}})$, $\tilde{\mathbf{s}}^{(k)}$ : $(N_{\text{sig}},)$ |
| outputs | $\Delta\mathbf{u}^{(k)}$ : $(N_{\text{act}},)$ |
| computational_note | Matrix-vector multiplication |

---

#### Step 4.9: Integral Control Update
| Field | Value |
|-------|-------|
| step_name | Apply Integral Controller |
| formula | $\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} - g \cdot \Delta\mathbf{u}^{(k)}$ |
| inputs | $\mathbf{u}^{(k)}$ : $(N_{\text{act}},)$, $\Delta\mathbf{u}^{(k)}$ : $(N_{\text{act}},)$, $g$ : scalar |
| outputs | $\mathbf{u}^{(k+1)}$ : $(N_{\text{act}},)$ |
| computational_note | Negative sign because slopes indicate wavefront error to be corrected |

---

#### Step 4.10: Science Path PSF Computation
| Field | Value |
|-------|-------|
| step_name | Compute Corrected Science PSF |
| formula | $$\boldsymbol{\phi}_{\text{sci}}^{(k)} = \boldsymbol{\phi}_{\text{atm}}^{(k)} - \mathbf{N}_{\text{DM}}(\mathbf{u}^{(k+1)})$$ $$\text{PSF}^{(k)} = \left| \mathcal{F}_{2D}\left\{ P \cdot \exp(i \boldsymbol{\phi}_{\text{sci}}^{(k)}) \right\} \right|^2$$ |
| inputs | $\boldsymbol{\phi}_{\text{atm}}^{(k)}$, $\mathbf{u}^{(k+1)}$, pupil $P$ |
| outputs | $\text{PSF}^{(k)}$ : $(2N_{\text{pix}}, 2N_{\text{pix}})$ |
| computational_note | Evaluated on higher-resolution science detector |

---

#### Step 4.11: Strehl Ratio Evaluation
| Field | Value |
|-------|-------|
| step_name | Compute Strehl Ratio via OTF Method |
| formula | $$\text{OTF}^{(k)} = \left| \mathcal{F}_{2D}\{\text{PSF}^{(k)}\} \right|$$ $$\text{OTF}_{\text{ref}} = \left| \mathcal{F}_{2D}\{\text{PSF}_{\text{ref}}\} \right|$$ $$\text{SR}^{(k)} = \frac{\sum_{m,n} \text{OTF}^{(k)}[m,n]}{\sum_{m,n} \text{OTF}_{\text{ref}}[m,n]} \times 100$$ |
| inputs | $\text{PSF}^{(k)}$, $\text{PSF}_{\text{ref}}$ |
| outputs | $\text{SR}^{(k)}$ : scalar (percentage) |
| computational_note | OTF method provides robust Strehl estimation |

---

#### Step 4.12: Convergence Check
| Field | Value |
|-------|-------|
| step_name | Check Iteration Limit |
| formula | Continue if $k < N_{\text{iter}} - 1$ |
| inputs | $k$ : scalar, $N_{\text{iter}}$ : scalar |
| outputs | Boolean: continue/terminate |
| computational_note | No explicit convergence threshold on residuals |

---

**END LOOP**

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $k \geq N_{\text{iter}}$ |
| parameters | $N_{\text{iter}} = 20$ (configurable) |

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Strehl Ratio (OTF) | $\text{SR} = \frac{\sum \left\|\mathcal{F}\{\text{PSF}\}\right\|}{\sum \left\|\mathcal{F}\{\text{PSF}_{\text{ref}}\}\right\|} \times 100$ | Image quality metric as percentage of diffraction limit |
| Initial Strehl | $\text{SR}^{(0)}$ | Strehl ratio at first iteration (before effective correction) |
| Final Strehl | $\text{SR}^{(N_{\text{iter}}-1)}$ | Strehl ratio at last iteration |
| Mean Strehl | $\overline{\text{SR}} = \frac{1}{N_{\text{iter}}} \sum_{k=0}^{N_{\text{iter}}-1} \text{SR}^{(k)}$ | Average Strehl over all iterations |
| Max Strehl | $\max_k \text{SR}^{(k)}$ | Peak achieved Strehl ratio |
| Min Strehl | $\min_k \text{SR}^{(k)}$ | Minimum Strehl ratio (indicates worst performance) |
| DM RMS Command | $\text{RMS}_{\mathbf{u}} = \sqrt{\frac{1}{N_{\text{act}}} \sum_{j=1}^{N_{\text{act}}} (u_j^{(\text{final})})^2}$ | Root-mean-square of final DM actuator commands [m] |