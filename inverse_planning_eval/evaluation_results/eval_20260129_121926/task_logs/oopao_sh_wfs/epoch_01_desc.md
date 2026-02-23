# Algorithmic Specification Document
## Adaptive Optics Closed-Loop Control with Shack-Hartmann Wavefront Sensing

---

## 1. Mathematical Formulation

### 1.1 Overall Objective

The goal is to find deformable mirror (DM) commands $\mathbf{u} \in \mathbb{R}^{N_{\text{act}}}$ that minimize the residual wavefront error, thereby maximizing the Strehl ratio. The implicit optimization problem is:

$$
\min_{\mathbf{u}} \; \| \phi_{\text{atm}}(\mathbf{r}) - \phi_{\text{DM}}(\mathbf{r}; \mathbf{u}) \|^2
$$

where:
- $\phi_{\text{atm}}(\mathbf{r})$ is the atmospheric phase aberration at pupil coordinate $\mathbf{r}$
- $\phi_{\text{DM}}(\mathbf{r}; \mathbf{u})$ is the DM-induced phase correction

### 1.2 Wavefront Sensing Model

The Shack-Hartmann WFS measures local wavefront slopes. For subaperture $i$:

$$
\mathbf{s}_i = \begin{pmatrix} s_{x,i} \\ s_{y,i} \end{pmatrix} = \begin{pmatrix} \frac{\partial \phi}{\partial x} \bigg|_i \\ \frac{\partial \phi}{\partial y} \bigg|_i \end{pmatrix}
$$

The slopes are computed via **Center of Gravity (CoG) centroiding** on FFT-formed spots:

$$
s_{x,i} = \frac{\sum_{p,q} I_i(p,q) \cdot X(p,q)}{\sum_{p,q} I_i(p,q)}, \quad s_{y,i} = \frac{\sum_{p,q} I_i(p,q) \cdot Y(p,q)}{\sum_{p,q} I_i(p,q)}
$$

where:
- $I_i(p,q) = \left| \mathcal{F}\{ E_i(x,y) \} \right|^2$ is the spot intensity (power spectrum of lenslet electric field)
- $X(p,q), Y(p,q)$ are coordinate grids centered at pixel $N_{\text{pix}}/2$

### 1.3 Linear Wavefront Reconstruction Model

The relationship between modal coefficients and measured slopes is linearized:

$$
\mathbf{s} = \mathbf{D} \cdot \mathbf{a} + \mathbf{n}
$$

where:
- $\mathbf{s} \in \mathbb{R}^{N_{\text{meas}}}$ is the slope measurement vector (concatenated $[s_x; s_y]$)
- $\mathbf{D} \in \mathbb{R}^{N_{\text{meas}} \times N_{\text{modes}}}$ is the **interaction matrix**
- $\mathbf{a} \in \mathbb{R}^{N_{\text{modes}}}$ is the modal coefficient vector
- $\mathbf{n}$ is measurement noise

### 1.4 Interaction Matrix Calibration (Push-Pull Method)

Each column of $\mathbf{D}$ is estimated via finite difference:

$$
\mathbf{D}_{:,j} = \frac{\mathbf{s}^{(+)}_j - \mathbf{s}^{(-)}_j}{2 \delta}
$$

where:
- $\mathbf{s}^{(+)}_j = \mathcal{S}(\mathbf{M}_{:,j} \cdot \delta)$ — slopes with positive push of mode $j$
- $\mathbf{s}^{(-)}_j = \mathcal{S}(-\mathbf{M}_{:,j} \cdot \delta)$ — slopes with negative pull of mode $j$
- $\delta$ is the stroke amplitude (calibration poke)
- $\mathbf{M} \in \mathbb{R}^{N_{\text{act}} \times N_{\text{modes}}}$ is the **mode-to-command** (M2C) matrix (KL basis)

### 1.5 Reconstructor Computation (Truncated SVD)

The modal reconstructor is computed via regularized pseudo-inverse:

$$
\mathbf{D} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \quad \text{(SVD)}
$$

$$
\mathbf{R}_{\text{modal}} = \mathbf{V} \mathbf{\Sigma}^{-1}_{\text{trunc}} \mathbf{U}^T
$$

where truncated inverse singular values:

$$
(\Sigma^{-1}_{\text{trunc}})_{ii} = \begin{cases} 
1/\sigma_i & \text{if } \sigma_i > \tau \\
0 & \text{otherwise}
\end{cases}
$$

The **final zonal reconstructor** (slopes → actuator commands):

$$
\mathbf{R} = \mathbf{M} \cdot \mathbf{R}_{\text{modal}} \in \mathbb{R}^{N_{\text{act}} \times N_{\text{meas}}}
$$

### 1.6 Closed-Loop Control Law (Integral Controller)

The discrete-time integral control law:

$$
\mathbf{u}^{[k]} = \mathbf{u}^{[k-1]} - g \cdot \mathbf{R} \cdot \left( \mathbf{s}^{[k]} - \mathbf{s}_{\text{ref}} \right)
$$

where:
- $\mathbf{u}^{[k]} \in \mathbb{R}^{N_{\text{act}}}$ is the DM command at iteration $k$
- $g \in (0, 1]$ is the loop gain
- $\mathbf{s}_{\text{ref}}$ is the reference slope vector (flat wavefront calibration)

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | Type | Initialization |
|----------|--------|-------|------|----------------|
| DM actuator commands | $\mathbf{u}$ | $(N_{\text{act}},)$ | `float64` | $\mathbf{0}$ |
| where | $N_{\text{act}} = (N_{\text{sub}}+1)^2$ | scalar | `int` | $21^2 = 441$ |

### 2.2 Auxiliary/Dual Variables

| Variable | Symbol | Shape | Type | Description |
|----------|--------|-------|------|-------------|
| Interaction matrix | $\mathbf{D}$ | $(N_{\text{meas}}, N_{\text{modes}})$ | `float64` | Calibrated via push-pull |
| Modal reconstructor | $\mathbf{R}_{\text{modal}}$ | $(N_{\text{modes}}, N_{\text{meas}})$ | `float64` | Truncated SVD pseudo-inverse |
| Final reconstructor | $\mathbf{R}$ | $(N_{\text{act}}, N_{\text{meas}})$ | `float64` | $\mathbf{M} \cdot \mathbf{R}_{\text{modal}}$ |
| KL mode basis | $\mathbf{M}$ | $(N_{\text{act}}, N_{\text{modes}})$ | `float64` | Mode-to-command matrix |
| SVD components | $\mathbf{U}, \boldsymbol{\sigma}, \mathbf{V}$ | Various | `float64` | From `np.linalg.svd` |
| Reference slopes | $\mathbf{s}_{\text{ref}}$ | $(N_{\text{meas}},)$ | `float64` | Flat wavefront calibration |
| Measured slopes | $\mathbf{s}^{[k]}$ | $(N_{\text{meas}},)$ | `float64` | Per-iteration measurement |
| Command increment | $\Delta\mathbf{u}^{[k]}$ | $(N_{\text{act}},)$ | `float64` | $\mathbf{R} \cdot (\mathbf{s}^{[k]} - \mathbf{s}_{\text{ref}})$ |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Number of subapertures | $N_{\text{sub}}$ | 20 | Per dimension |
| Number of modes | $N_{\text{modes}}$ | 20 | KL modes for control |
| Calibration stroke | $\delta$ | $10^{-8}$ m | Push-pull amplitude |
| SVD threshold | $\tau$ | $10^{-3}$ | Singular value cutoff |
| Loop gain | $g$ | 0.4 | Integral controller gain |
| Number of iterations | $N_{\text{iter}}$ | 20 | Closed-loop iterations |
| Telescope diameter | $D$ | 8.0 m | Primary mirror diameter |
| Pupil resolution | $N_{\text{pix}}$ | $6 \times N_{\text{sub}} = 120$ | Pixels across pupil |
| Fried parameter | $r_0$ | 0.15 m | Atmospheric coherence length |
| Outer scale | $L_0$ | 25 m | Turbulence outer scale |
| Wind speed | $v$ | 10 m/s | Atmospheric wind |
| Light ratio | $\ell$ | 0.5 | WFS subaperture illumination threshold |
| Mechanical coupling | $\alpha$ | 0.35 | DM inter-actuator coupling |

### 2.4 Observed Data/Inputs

| Data | Symbol | Shape | Description |
|------|--------|-------|-------------|
| Atmospheric phase | $\phi_{\text{atm}}^{[k]}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | Time-evolving turbulence |
| Science PSF | $\text{PSF}^{[k]}$ | $(2N_{\text{pix}}, 2N_{\text{pix}})$ | Focal plane image |
| Reference PSF | $\text{PSF}_{\text{ref}}$ | $(2N_{\text{pix}}, 2N_{\text{pix}})$ | Diffraction-limited |
| Lenslet electric field | $E_i(x,y)$ | $(N_{\text{pix}}/N_{\text{sub}}, N_{\text{pix}}/N_{\text{sub}})$ | Per-subaperture field |

### 2.5 Derived Dimensions

| Quantity | Formula | Value |
|----------|---------|-------|
| $N_{\text{act}}$ | $(N_{\text{sub}} + 1)^2$ | 441 |
| $N_{\text{valid}}$ | Number of illuminated subapertures | $\leq N_{\text{sub}}^2$ |
| $N_{\text{meas}}$ | $2 \times N_{\text{valid}}$ | WFS signal length |

---

## 3. Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘
```

| Step | Operation | Input | Output | Formula/Description |
|------|-----------|-------|--------|---------------------|
| **3.1** | Initialize Telescope | $D, N_{\text{pix}}, \Delta t$ | `tel` object | Pupil function $P(\mathbf{r})$ |
| **3.2** | Initialize Source | Band, magnitude, coords | `ngs` object | Guide star at infinity |
| **3.3** | Initialize Atmosphere | $r_0, L_0, v, \theta$ | `atm` object | Kolmogorov phase screens |
| **3.4** | Initialize DM | $N_{\text{sub}}, \alpha$ | `dm` object | Influence functions |
| **3.5** | Initialize WFS | $N_{\text{sub}}, \ell$ | `wfs` object | Valid subaperture mask |
| **3.6** | Initialize Science Camera | $2N_{\text{pix}}$ | `sci_cam` object | Focal plane detector |
| **3.7** | Compute Reference PSF | `tel`, `ngs`, `sci_cam` | $\text{PSF}_{\text{ref}}$ | $\text{PSF}_{\text{ref}} = \|\mathcal{F}\{P \cdot e^{i \cdot 0}\}\|^2$ |
| **3.8** | Compute Reference Slopes | `wfs`, `tel`, `ngs` | $\mathbf{s}_{\text{ref}}$ | CoG with flat wavefront |
| **3.9** | Compute KL Basis | `tel`, `atm`, `dm` | $\mathbf{M}_{\text{full}}$ | Karhunen-Loève decomposition |
| **3.10** | Truncate KL Basis | $\mathbf{M}_{\text{full}}, N_{\text{modes}}$ | $\mathbf{M}$ | $\mathbf{M} = \mathbf{M}_{\text{full}}[:, :N_{\text{modes}}]$ |
| **3.11** | Calibrate Interaction Matrix | $\mathbf{M}, \delta$, WFS | $\mathbf{D}$ | Push-pull (see §4.1) |
| **3.12** | Compute Reconstructor | $\mathbf{D}, \tau, \mathbf{M}$ | $\mathbf{R}$ | Truncated SVD (see §4.2) |

---

## 4. Algorithm Steps

### 4.1 Interaction Matrix Calibration

```
┌─────────────────────────────────────────────────────────────────┐
│           ALGORITHM: Interaction Matrix Calibration              │
└─────────────────────────────────────────────────────────────────┘
```

**Inputs:** $\mathbf{M} \in \mathbb{R}^{N_{\text{act}} \times N_{\text{modes}}}$, $\delta$, WFS system
**Output:** $\mathbf{D} \in \mathbb{R}^{N_{\text{meas}} \times N_{\text{modes}}}$

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Initialize matrix | $\mathbf{D} \leftarrow \mathbf{0}_{N_{\text{meas}} \times N_{\text{modes}}}$ | Dimensions | $\mathbf{D}$ |
| 2 | **For** $j = 1, \ldots, N_{\text{modes}}$: | | | |
| 2a | Apply positive push | $\mathbf{u}_{\text{push}} = \mathbf{M}_{:,j} \cdot \delta$ | $\mathbf{M}_{:,j}, \delta$ | DM state |
| 2b | Measure push slopes | $\mathbf{s}^{(+)} = \mathcal{S}_{\text{WFS}}(\mathbf{u}_{\text{push}})$ | DM state, WFS | $\mathbf{s}^{(+)}$ |
| 2c | Apply negative pull | $\mathbf{u}_{\text{pull}} = -\mathbf{M}_{:,j} \cdot \delta$ | $\mathbf{M}_{:,j}, \delta$ | DM state |
| 2d | Measure pull slopes | $\mathbf{s}^{(-)} = \mathcal{S}_{\text{WFS}}(\mathbf{u}_{\text{pull}})$ | DM state, WFS | $\mathbf{s}^{(-)}$ |
| 2e | Compute IM column | $\mathbf{D}_{:,j} = \frac{\mathbf{s}^{(+)} - \mathbf{s}^{(-)}}{2\delta}$ | $\mathbf{s}^{(+)}, \mathbf{s}^{(-)}, \delta$ | $\mathbf{D}_{:,j}$ |
| 3 | Reset DM | $\mathbf{u} \leftarrow \mathbf{0}$ | — | DM state |

---

### 4.2 Reconstructor Computation

```
┌─────────────────────────────────────────────────────────────────┐
│             ALGORITHM: Reconstructor via Truncated SVD           │
└─────────────────────────────────────────────────────────────────┘
```

**Inputs:** $\mathbf{D} \in \mathbb{R}^{N_{\text{meas}} \times N_{\text{modes}}}$, $\tau$, $\mathbf{M}$
**Output:** $\mathbf{R} \in \mathbb{R}^{N_{\text{act}} \times N_{\text{meas}}}$

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | SVD decomposition | $\mathbf{D} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$ | $\mathbf{D}$ | $\mathbf{U}, \boldsymbol{\sigma}, \mathbf{V}$ |
| 2 | Initialize inverse | $\boldsymbol{\sigma}^{-1} \leftarrow \mathbf{0}$ | — | $\boldsymbol{\sigma}^{-1}$ |
| 3 | Truncated inversion | $\sigma^{-1}_i = \begin{cases} 1/\sigma_i & \sigma_i > \tau \\ 0 & \text{else} \end{cases}$ | $\boldsymbol{\sigma}, \tau$ | $\boldsymbol{\sigma}^{-1}$ |
| 4 | Modal reconstructor | $\mathbf{R}_{\text{modal}} = \mathbf{V} \cdot \text{diag}(\boldsymbol{\sigma}^{-1}) \cdot \mathbf{U}^T$ | $\mathbf{U}, \boldsymbol{\sigma}^{-1}, \mathbf{V}$ | $\mathbf{R}_{\text{modal}}$ |
| 5 | Zonal reconstructor | $\mathbf{R} = \mathbf{M} \cdot \mathbf{R}_{\text{modal}}$ | $\mathbf{M}, \mathbf{R}_{\text{modal}}$ | $\mathbf{R}$ |

---

### 4.3 Slope Measurement (Diffractive WFS Model)

```
┌─────────────────────────────────────────────────────────────────┐
│          ALGORITHM: Shack-Hartmann Slope Measurement             │
└─────────────────────────────────────────────────────────────────┘
```

**Inputs:** Wavefront phase $\phi$, WFS geometry
**Output:** $\mathbf{s} \in \mathbb{R}^{N_{\text{meas}}}$

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Extract lenslet fields | $E_i = \text{crop}_i(E_{\text{pupil}})$ | $E_{\text{pupil}}$ | $\{E_i\}_{i=1}^{N_{\text{sub}}^2}$ |
| 2 | FFT spot formation | $\tilde{E}_i = \mathcal{F}\{E_i\}$ | $E_i$ | $\tilde{E}_i$ |
| 3 | Intensity computation | $I_i = |\tilde{E}_i|^2$ | $\tilde{E}_i$ | $I_i$ |
| 4 | Define coordinate grids | $X_{pq} = p - N_{\text{pix}}/2$, $Y_{pq} = q - N_{\text{pix}}/2$ | $N_{\text{pix}}$ | $X, Y$ |
| 5 | **For** each valid subaperture $i$: | | | |
| 5a | Compute flux | $F_i = \sum_{p,q} I_i(p,q)$ | $I_i$ | $F_i$ |
| 5b | Compute $x$-centroid | $s_{x,i} = \frac{1}{F_i} \sum_{p,q} I_i(p,q) \cdot X_{pq}$ | $I_i, X, F_i$ | $s_{x,i}$ |
| 5c | Compute $y$-centroid | $s_{y,i} = \frac{1}{F_i} \sum_{p,q} I_i(p,q) \cdot Y_{pq}$ | $I_i, Y, F_i$ | $s_{y,i}$ |
| 6 | Concatenate slopes | $\mathbf{s} = [s_{x,1}, \ldots, s_{x,N_{\text{valid}}}, s_{y,1}, \ldots, s_{y,N_{\text{valid}}}]^T$ | $\{s_{x,i}, s_{y,i}\}$ | $\mathbf{s}$ |

---

### 4.4 Main Closed-Loop Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│           ALGORITHM: Closed-Loop AO with Integral Control        │
└─────────────────────────────────────────────────────────────────┘
```

**Inputs:** System data (from preprocessing), $N_{\text{iter}}$, $g$
**Outputs:** Strehl history $\{\text{SR}^{[k]}\}$, final DM commands $\mathbf{u}^{[N_{\text{iter}}]}$, final PSF

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| **Init** | Initialize DM | $\mathbf{u}^{[0]} \leftarrow \mathbf{0}$ | — | $\mathbf{u}^{[0]}$ |
| **Init** | Initialize history | $\mathcal{H} \leftarrow []$ | — | $\mathcal{H}$ |
| | **For** $k = 1, \ldots, N_{\text{iter}}$: | | | |
| 1 | Evolve atmosphere | $\phi_{\text{atm}}^{[k]} = \text{atm.update}()$ | $\phi_{\text{atm}}^{[k-1]}$, $v$, $\Delta t$ | $\phi_{\text{atm}}^{[k]}$ |
| 2 | Forward propagation (sensing) | $\phi_{\text{res}}^{[k]} = \phi_{\text{atm}}^{[k]} - \phi_{\text{DM}}(\mathbf{u}^{[k-1]})$ | $\phi_{\text{atm}}^{[k]}$, $\mathbf{u}^{[k-1]}$ | $\phi_{\text{res}}^{[k]}$ |
| 3 | Measure slopes | $\mathbf{s}^{[k]} = \mathcal{S}_{\text{WFS}}(\phi_{\text{res}}^{[k]})$ | $\phi_{\text{res}}^{[k]}$ | $\mathbf{s}^{[k]}$ |
| 4 | Compute residual slopes | $\Delta\mathbf{s}^{[k]} = \mathbf{s}^{[k]} - \mathbf{s}_{\text{ref}}$ | $\mathbf{s}^{[k]}$, $\mathbf{s}_{\text{ref}}$ | $\Delta\mathbf{s}^{[k]}$ |
| 5 | Reconstruct command update | $\Delta\mathbf{u}^{[k]} = \mathbf{R} \cdot \Delta\mathbf{s}^{[k]}$ | $\mathbf{R}$, $\Delta\mathbf{s}^{[k]}$ | $\Delta\mathbf{u}^{[k]}$ |
| 6 | **Integral control update** | $\mathbf{u}^{[k]} = \mathbf{u}^{[k-1]} - g \cdot \Delta\mathbf{u}^{[k]}$ | $\mathbf{u}^{[k-1]}$, $g$, $\Delta\mathbf{u}^{[k]}$ | $\mathbf{u}^{[k]}$ |
| 7 | Forward propagation (science) | $\text{PSF}^{[k]} = |\mathcal{F}\{P \cdot e^{i\phi_{\text{res}}^{[k]}}\}|^2$ | $\phi_{\text{res}}^{[k]}$ (with updated DM) | $\text{PSF}^{[k]}$ |
| 8 | Evaluate Strehl | $\text{SR}^{[k]} = \mathcal{E}(\text{PSF}^{[k]}, \text{PSF}_{\text{ref}})$ | $\text{PSF}^{[k]}$, $\text{PSF}_{\text{ref}}$ | $\text{SR}^{[k]}$ |
| 9 | Record history | $\mathcal{H}.\text{append}(\text{SR}^{[k]})$ | $\text{SR}^{[k]}$ | $\mathcal{H}$ |
| **End** | Return results | — | — | $\mathcal{H}$, $\mathbf{u}^{[N_{\text{iter}}]}$, $\text{PSF}^{[N_{\text{iter}}]}$ |

---

### 4.5 Forward Operator

```
┌─────────────────────────────────────────────────────────────────┐
│                    FORWARD OPERATOR: A(u)                        │
└─────────────────────────────────────────────────────────────────┘
```

**Input:** DM commands $\mathbf{u}$ (optional)
**Output:** Science PSF

$$
\mathcal{A}(\mathbf{u}) = \left| \mathcal{F}\left\{ P(\mathbf{r}) \cdot \exp\left( i \left[ \phi_{\text{atm}}(\mathbf{r}) - \sum_{j=1}^{N_{\text{act}}} u_j \cdot h_j(\mathbf{r}) \right] \right) \right\} \right|^2
$$

where $h_j(\mathbf{r})$ is the influence function of actuator $j$.

---

## 5. Stopping Criterion

The algorithm uses a **fixed iteration count** stopping criterion:

$$
\text{Terminate when: } k = N_{\text{iter}}
$$

| Criterion | Type | Value | Condition |
|-----------|------|-------|-----------|
| Maximum iterations | Hard limit | $N_{\text{iter}} = 20$ | $k \geq N_{\text{iter}}$ |

**Note:** No convergence-based criterion (e.g., $\|\mathbf{s}^{[k]}\| < \epsilon$ or $|\text{SR}^{[k]} - \text{SR}^{[k-1]}| < \epsilon$) is implemented.

---

## 6. Evaluation Metrics

### 6.1 Strehl Ratio (OTF Method)

$$
\text{SR} = 100 \times \frac{\sum_{p,q} |\text{OTF}(p,q)|}{\sum_{p,q} |\text{OTF}_{\text{ref}}(p,q)|} \quad [\%]
$$

where:
$$
\text{OTF} = \left| \mathcal{F}_{\text{shift}}\{ \mathcal{F}\{ \text{PSF} \} \} \right|
$$

| Metric | Formula | Units |
|--------|---------|-------|
| Initial Strehl | $\text{SR}^{[1]}$ | % |
| Final Strehl | $\text{SR}^{[N_{\text{iter}}]}$ | % |
| Mean Strehl | $\bar{\text{SR}} = \frac{1}{N_{\text{iter}}} \sum_{k=1}^{N_{\text{iter}}} \text{SR}^{[k]}$ | % |
| Max Strehl | $\max_k \text{SR}^{[k]}$ | % |
| Min Strehl | $\min_k \text{SR}^{[k]}$ | % |

### 6.2 DM Command RMS

$$
\text{RMS}_{\text{DM}} = \sqrt{\frac{1}{N_{\text{act}}} \sum_{j=1}^{N_{\text{act}}} (u_j^{[N_{\text{iter}}]})^2} \quad [\text{m}]
$$

### 6.3 Summary Statistics Table

| Metric | Symbol | Formula |
|--------|--------|---------|
| Initial Strehl | $\text{SR}_0$ | $\text{SR}^{[1]}$ |
| Final Strehl | $\text{SR}_f$ | $\text{SR}^{[N_{\text{iter}}]}$ |
| Mean