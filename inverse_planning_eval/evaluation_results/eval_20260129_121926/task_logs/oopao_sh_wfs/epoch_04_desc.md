## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Closed-Loop Integral Controller for Shack-Hartmann Adaptive Optics
- **algorithm_family**: Feedback Control / Linear-Quadratic Wavefront Reconstruction
- **summary**: An integral control algorithm that iteratively corrects atmospheric wavefront aberrations by measuring residual slopes with a Shack-Hartmann WFS and applying corrections via a deformable mirror using a pre-calibrated modal reconstructor.

---

## 1. OBJECTIVE FUNCTION

### full_expression
The algorithm implicitly minimizes the residual wavefront error through integral feedback control:
$$\min_{\mathbf{u}} \| \boldsymbol{\phi}_{\text{atm}} - \mathbf{N} \mathbf{u} \|^2$$

where $\boldsymbol{\phi}_{\text{atm}}$ is the atmospheric phase, $\mathbf{N}$ is the DM influence function matrix, and $\mathbf{u}$ is the vector of DM actuator commands. The control law provides an approximate solution via iterative correction.

### data_fidelity_term
$$\mathcal{L}(\mathbf{u}) = \| \mathbf{s}_{\text{meas}}(\mathbf{u}) - \mathbf{s}_{\text{ref}} \|_2^2$$

where $\mathbf{s}_{\text{meas}}$ are the measured WFS slopes under DM correction $\mathbf{u}$, and $\mathbf{s}_{\text{ref}}$ are reference slopes from a flat wavefront.

### regularization_term
Implicit regularization via SVD truncation in reconstructor computation:
$$\mathcal{R}(\mathbf{R}) = \mathbb{I}_{[\sigma_i > \tau]}(\sigma_i)$$

where singular values $\sigma_i \leq \tau = 10^{-3}$ are zeroed in the pseudoinverse.

### constraints
- Box constraint on DM commands: Implicitly bounded by physical actuator stroke limits (not explicitly enforced in code)
- Stability constraint: Gain $g \in (0, 1)$ for closed-loop stability

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{u}^{(k)}$ | $(n_{\text{act}},)$ | float64 | $\mathbf{0}$ | DM actuator commands at iteration $k$; $n_{\text{act}} = (n_{\text{sub}}+1)^2$ |
| $\mathbf{c}_{\text{modal}}$ | $(n_{\text{modes}},)$ | float64 | Computed | Modal coefficients for calibration |

### 2.2 Dual Variables

None (not a primal-dual algorithm).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{s}_{\text{meas}}^{(k)}$ | $(2 n_{\text{valid}},)$ | float64 | Measured | Raw WFS slopes at iteration $k$ |
| $\Delta\mathbf{u}^{(k)}$ | $(n_{\text{act}},)$ | float64 | Computed | Command increment: $\mathbf{R} \cdot (\mathbf{s}_{\text{meas}}^{(k)} - \mathbf{s}_{\text{ref}})$ |
| $\text{SR}^{(k)}$ | scalar | float64 | Computed | Strehl ratio at iteration $k$ |
| $I_{\text{spots}}$ | $(n_{\text{sub}}^2, n_{\text{pix}}, n_{\text{pix}})$ | float64 | Computed | WFS spot intensity patterns |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $n_{\text{sub}}$ | 20 | Number of WFS subapertures per dimension |
| $n_{\text{modes}}$ | 20 | Number of KL modes for reconstruction |
| $\delta$ | $10^{-8}$ m | Push-pull stroke for calibration |
| $g$ | 0.4 | Integral controller gain |
| $K$ | 20 | Maximum number of iterations |
| $\tau$ | $10^{-3}$ | SVD truncation threshold |
| $D$ | 8.0 m | Telescope diameter |
| $n_{\text{pix}}$ | $6 \cdot n_{\text{sub}} = 120$ | Pupil resolution |
| $r_0$ | 0.15 m | Fried parameter |
| $L_0$ | 25 m | Outer scale |
| $v_{\text{wind}}$ | 10 m/s | Wind speed |
| $\kappa$ | 0.35 | DM mechanical coupling coefficient |
| $\rho_{\text{light}}$ | 0.5 | WFS light ratio threshold |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{s}_{\text{ref}}$ | $(2 n_{\text{valid}},)$ | float64 | Reference slopes from flat wavefront |
| $\text{PSF}_{\text{ref}}$ | $(2 n_{\text{pix}}, 2 n_{\text{pix}})$ | float64 | Diffraction-limited reference PSF |
| $\boldsymbol{\phi}_{\text{atm}}^{(k)}$ | $(n_{\text{pix}}, n_{\text{pix}})$ | float64 | Atmospheric phase screen at iteration $k$ |

---

## 3. PREPROCESSING PIPELINE

### Step 0: System Initialization

| Property | Value |
|----------|-------|
| step_order | 0 |
| step_name | Initialize Optical Components |
| formula | $n_{\text{pix}} = 6 \cdot n_{\text{sub}}$, $n_{\text{act}} = (n_{\text{sub}}+1)^2$ |
| inputs | $n_{\text{sub}}$, $D$, $r_0$, $L_0$ |
| outputs | tel, ngs, atm, dm, wfs, sci_cam objects |

### Step 1: Reference PSF Computation

| Property | Value |
|----------|-------|
| step_order | 1 |
| step_name | Compute Diffraction-Limited PSF |
| formula | $\text{PSF}_{\text{ref}} = \left\| \mathcal{F}\{ P(\mathbf{r}) \} \right\|^2$ where $P(\mathbf{r})$ is the pupil function |
| inputs | tel.OPD = 0 (reset) |
| outputs | $\text{PSF}_{\text{ref}} \in \mathbb{R}^{2n_{\text{pix}} \times 2n_{\text{pix}}}$ |

### Step 2: Reference Slopes Computation

| Property | Value |
|----------|-------|
| step_order | 2 |
| step_name | Measure Reference Slopes |
| formula | $\mathbf{s}_{\text{ref}} = \text{CoG}(\mathcal{F}\{ E_{\text{lenslet}} \})$ with flat wavefront |
| inputs | tel with zero OPD |
| outputs | $\mathbf{s}_{\text{ref}} \in \mathbb{R}^{2 n_{\text{valid}}}$ |

### Step 3: KL Basis Computation

| Property | Value |
|----------|-------|
| step_order | 3 |
| step_name | Compute KL Modal Basis |
| formula | $\mathbf{M}_{2C} = \text{KL\_basis}(\text{tel}, \text{atm}, \text{dm})$, then $\mathbf{B} = \mathbf{M}_{2C}[:, :n_{\text{modes}}]$ |
| inputs | tel, atm, dm, $n_{\text{modes}}$ |
| outputs | $\mathbf{B} \in \mathbb{R}^{n_{\text{act}} \times n_{\text{modes}}}$ (basis_modes) |

### Step 4: Interaction Matrix Calibration

| Property | Value |
|----------|-------|
| step_order | 4 |
| step_name | Build Interaction Matrix via Push-Pull |
| formula | For mode $i$: $\mathbf{D}_{:,i} = \frac{\mathbf{s}^{+}_i - \mathbf{s}^{-}_i}{2\delta}$ where $\mathbf{s}^{+}_i = \text{WFS}(\mathbf{B}_{:,i} \cdot \delta)$ and $\mathbf{s}^{-}_i = \text{WFS}(-\mathbf{B}_{:,i} \cdot \delta)$ |
| inputs | $\mathbf{B}$, $\delta = 10^{-8}$ |
| outputs | $\mathbf{D} \in \mathbb{R}^{n_{\text{meas}} \times n_{\text{modes}}}$ (interaction_matrix) |

### Step 5: Reconstructor Computation via SVD

| Property | Value |
|----------|-------|
| step_order | 5 |
| step_name | Compute Modal Reconstructor |
| formula | $\mathbf{D} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$, then $\mathbf{R}_{\text{modal}} = \mathbf{V} \boldsymbol{\Sigma}^{\dagger} \mathbf{U}^T$ where $\Sigma^{\dagger}_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > \tau \\ 0 & \text{otherwise} \end{cases}$ with $\tau = 10^{-3}$ |
| inputs | $\mathbf{D}$ |
| outputs | $\mathbf{R}_{\text{modal}} \in \mathbb{R}^{n_{\text{modes}} \times n_{\text{meas}}}$ |

### Step 6: Modal-to-Zonal Conversion

| Property | Value |
|----------|-------|
| step_order | 6 |
| step_name | Convert to Zonal Reconstructor |
| formula | $\mathbf{R} = \mathbf{B} \cdot \mathbf{R}_{\text{modal}}$ |
| inputs | $\mathbf{B}$, $\mathbf{R}_{\text{modal}}$ |
| outputs | $\mathbf{R} \in \mathbb{R}^{n_{\text{act}} \times n_{\text{meas}}}$ (final_reconstructor) |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Property | Value |
|----------|-------|
| step_name | Initialize DM Commands |
| formula | $\mathbf{u}^{(0)} = \mathbf{0} \in \mathbb{R}^{n_{\text{act}}}$ |
| inputs | $n_{\text{act}}$ |
| outputs | $\mathbf{u}^{(0)}$ |
| computational_note | DM starts in flat configuration |

| Property | Value |
|----------|-------|
| step_name | Initialize Strehl History |
| formula | $\text{SR\_history} = []$ (empty list) |
| inputs | None |
| outputs | strehl_history container |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, K-1$ where $K = 20$)

---

**Step 1: Atmosphere Update** [FORWARD]

| Property | Value |
|----------|-------|
| step_name | Evolve Atmospheric Phase Screen |
| formula | $\boldsymbol{\phi}_{\text{atm}}^{(k+1)} = \text{atm.update}()$ (frozen-flow Taylor hypothesis) |
| inputs | atm state, $v_{\text{wind}}$, $\Delta t$ |
| outputs | $\boldsymbol{\phi}_{\text{atm}}^{(k+1)}$ |
| computational_note | Phase screen shifts by $v_{\text{wind}} \cdot \Delta t$ per iteration |

---

**Step 2: Optical Propagation to WFS** [FORWARD]

| Property | Value |
|----------|-------|
| step_name | Propagate Through Optical Train |
| formula | $E_{\text{WFS}} = P(\mathbf{r}) \cdot \exp\left(i \left( \boldsymbol{\phi}_{\text{atm}}^{(k)} - \mathbf{N} \mathbf{u}^{(k)} \right) \right)$ |
| inputs | $\boldsymbol{\phi}_{\text{atm}}^{(k)}$, $\mathbf{u}^{(k)}$, pupil $P$ |
| outputs | Electric field at WFS lenslet plane |
| computational_note | DM subtracts phase; $\mathbf{N}$ is influence function matrix |

---

**Step 3: WFS Spot Formation** [FORWARD]

| Property | Value |
|----------|-------|
| step_name | Form Lenslet Spots via FFT |
| formula | For each subaperture $j$: $I_j(\mathbf{k}) = \left\| \mathcal{F}\{ E_{\text{lenslet},j}(\mathbf{r}) \} \right\|^2$ |
| inputs | Cube of lenslet electric fields $E_{\text{lenslet}} \in \mathbb{C}^{n_{\text{sub}}^2 \times n_{\text{pix/sub}} \times n_{\text{pix/sub}}}$ |
| outputs | $I_{\text{spots}} \in \mathbb{R}^{n_{\text{sub}}^2 \times n_{\text{pix/sub}} \times n_{\text{pix/sub}}}$ |
| computational_note | FFT performed with `np.fft.fft2(cube_em, axes=[1,2])` |

---

**Step 4: Centroid Computation (Center of Gravity)**

| Property | Value |
|----------|-------|
| step_name | Compute Spot Centroids |
| formula | For valid subaperture $j$: $$c_{x,j} = \frac{\sum_{m,n} I_j(m,n) \cdot X(m,n)}{\sum_{m,n} I_j(m,n)}, \quad c_{y,j} = \frac{\sum_{m,n} I_j(m,n) \cdot Y(m,n)}{\sum_{m,n} I_j(m,n)}$$ where $X(m,n) = m - n_{\text{pix}}/2$, $Y(m,n) = n - n_{\text{pix}}/2$ |
| inputs | $I_{\text{spots}}$, valid subaperture mask |
| outputs | $\mathbf{s}_{\text{raw}} = [c_{x,1}, \ldots, c_{x,n_v}, c_{y,1}, \ldots, c_{y,n_v}]^T \in \mathbb{R}^{2 n_{\text{valid}}}$ |
| computational_note | Only valid subapertures (light ratio > 0.5) are processed |

---

**Step 5: Residual Slope Computation**

| Property | Value |
|----------|-------|
| step_name | Compute Residual Slopes |
| formula | $\mathbf{s}^{(k)} = \mathbf{s}_{\text{raw}}^{(k)} - \mathbf{s}_{\text{ref}}$ |
| inputs | $\mathbf{s}_{\text{raw}}^{(k)}$, $\mathbf{s}_{\text{ref}}$ |
| outputs | $\mathbf{s}^{(k)} \in \mathbb{R}^{2 n_{\text{valid}}}$ |
| computational_note | Reference subtraction removes static aberrations |

---

**Step 6: Wavefront Reconstruction**

| Property | Value |
|----------|-------|
| step_name | Reconstruct DM Command Increment |
| formula | $\Delta \mathbf{u}^{(k)} = \mathbf{R} \cdot \mathbf{s}^{(k)}$ where $\mathbf{R} = \mathbf{B} \mathbf{V} \boldsymbol{\Sigma}^{\dagger} \mathbf{U}^T$ |
| inputs | $\mathbf{R} \in \mathbb{R}^{n_{\text{act}} \times 2n_{\text{valid}}}$, $\mathbf{s}^{(k)}$ |
| outputs | $\Delta \mathbf{u}^{(k)} \in \mathbb{R}^{n_{\text{act}}}$ |
| computational_note | Matrix-vector multiplication: `np.matmul(reconstructor, slopes_meas)` |

---

**Step 7: Integral Control Update**

| Property | Value |
|----------|-------|
| step_name | Update DM Commands |
| formula | $\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} - g \cdot \Delta \mathbf{u}^{(k)}$ |
| inputs | $\mathbf{u}^{(k)}$, $\Delta \mathbf{u}^{(k)}$, $g = 0.4$ |
| outputs | $\mathbf{u}^{(k+1)}$ |
| computational_note | Negative sign because slopes measure residual error |

---

**Step 8: Science Path Propagation** [FORWARD]

| Property | Value |
|----------|-------|
| step_name | Compute Science PSF |
| formula | $\text{PSF}^{(k)} = \left\| \mathcal{F}\left\{ P(\mathbf{r}) \cdot \exp\left(i \left( \boldsymbol{\phi}_{\text{atm}}^{(k)} - \mathbf{N} \mathbf{u}^{(k+1)} \right) \right) \right\} \right\|^2$ |
| inputs | Updated $\mathbf{u}^{(k+1)}$, $\boldsymbol{\phi}_{\text{atm}}^{(k)}$ |
| outputs | $\text{PSF}^{(k)} \in \mathbb{R}^{2n_{\text{pix}} \times 2n_{\text{pix}}}$ |
| computational_note | Science camera has 2× resolution of pupil |

---

**Step 9: Strehl Ratio Evaluation**

| Property | Value |
|----------|-------|
| step_name | Compute Strehl Ratio |
| formula | $\text{SR}^{(k)} = 100 \times \frac{\sum_{m,n} \left\| \mathcal{F}\{\text{PSF}^{(k)}\}(m,n) \right\|}{\sum_{m,n} \left\| \mathcal{F}\{\text{PSF}_{\text{ref}}\}(m,n) \right\|}$ |
| inputs | $\text{PSF}^{(k)}$, $\text{PSF}_{\text{ref}}$ |
| outputs | $\text{SR}^{(k)} \in [0, 100]$ (percent) |
| computational_note | OTF-based Strehl; uses `np.fft.fftshift(np.fft.fft2(...))` |

---

**END_LOOP**

### 4.3 FINALIZATION

| Property | Value |
|----------|-------|
| step_name | Extract Final Outputs |
| formula | $\text{PSF}_{\text{final}} = \text{PSF}^{(K-1)}$, $\mathbf{u}_{\text{final}} = \mathbf{u}^{(K)}$ |
| inputs | Final iteration state |
| outputs | final_psf, final_dm_commands, strehl_history array |

---

## 5. STOPPING CRITERION

| Property | Value |
|----------|-------|
| type | iteration_limit |
| expression | $k < K$ where $K = 20$ |
| parameters | $K = 20$ (n_iter) |
| check_frequency | Evaluated at end of each iteration |

---

## 6. EVALUATION METRICS

| Metric Name | Formula | Description |
|-------------|---------|-------------|
| Strehl Ratio (OTF) | $\text{SR} = 100 \times \frac{\sum \|\text{OTF}\|}{\sum \|\text{OTF}_{\text{ref}}\|}$ where $\text{OTF} = \mathcal{F}\{\text{PSF}\}$ | Ratio of optical transfer function integrals; 100% = perfect correction |
| Initial Strehl | $\text{SR}^{(0)}$ | Strehl at first iteration |
| Final Strehl | $\text{SR}^{(K-1)}$ | Strehl at last iteration |
| Mean Strehl | $\bar{\text{SR}} = \frac{1}{K} \sum_{k=0}^{K-1} \text{SR}^{(k)}$ | Average Strehl over all iterations |
| Max Strehl | $\max_k \text{SR}^{(k)}$ | Best achieved Strehl |
| Min Strehl | $\min_k \text{SR}^{(k)}$ | Worst achieved Strehl |
| DM RMS Command | $\text{RMS}_u = \sqrt{\frac{1}{n_{\text{act}}} \sum_{i=1}^{n_{\text{act}}} (u_{\text{final},i})^2}$ | Root-mean-square of final DM actuator commands |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include the following fields:

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
  - dm_commands_init
  - history_containers

iteration_loop:
  - atmosphere_update
  - optical_propagation
  - spot_formation
  - centroid_computation
  - residual_slopes
  - reconstruction
  - control_update
  - science_evaluation
  - strehl_computation

finalization:
  - final_psf
  - final_dm_commands
  - strehl_history

stopping_criterion:
  - type
  - expression
  - parameters
  - check_frequency

evaluation:
  - strehl_ratio
  - initial_strehl
  - final_strehl
  - mean_strehl
  - max_strehl
  - min_strehl
  - dm_rms
  - n_iterations
```