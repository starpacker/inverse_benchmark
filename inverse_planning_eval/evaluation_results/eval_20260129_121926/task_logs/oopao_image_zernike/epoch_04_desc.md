## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Zernike Wavefront Decomposition with Fourier Optics Forward Model
- **algorithm_family**: Linear Least Squares Projection / Fourier Optics Image Formation
- **summary**: Decomposes atmospheric optical path difference (OPD) maps into Zernike polynomial coefficients via pseudoinverse projection, with a companion forward model computing point spread functions through Fourier propagation of complex electric fields.

---

## 1. OBJECTIVE FUNCTION

### full_expression
The inverse model solves the linear least squares problem:
$$\hat{\mathbf{c}} = \arg\min_{\mathbf{c}} \|\boldsymbol{\phi}_{\text{masked}} - \mathbf{Z}\mathbf{c}\|_2^2$$

where the closed-form solution is:
$$\hat{\mathbf{c}} = \mathbf{Z}^{\dagger} \boldsymbol{\phi}_{\text{masked}}$$

### data_fidelity_term
$$f(\mathbf{c}) = \|\boldsymbol{\phi}_{\text{masked}} - \mathbf{Z}\mathbf{c}\|_2^2 = \sum_{p \in \mathcal{P}} \left(\phi_p - \sum_{j=1}^{J} Z_{p,j} c_j\right)^2$$

where $\mathcal{P} = \{p : \text{pupil}(p) = 1\}$ is the set of valid pupil pixels.

### regularization_term
None (unregularized least squares)

### constraints
None (unconstrained minimization)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{c}$ | $(J,)$ | float64 | Computed via $\mathbf{Z}^{\dagger}\boldsymbol{\phi}$ | Zernike coefficient vector |
| $\boldsymbol{\phi}_{\text{rec}}$ | $(N, N)$ | float64 | Computed via $\mathbf{Z}_{\text{full}} \mathbf{c}$ | Reconstructed OPD map |

### 2.2 Dual Variables

None (closed-form solution, no dual formulation required)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\phi}_{\text{masked}}$ | $(N_{\text{pupil}},)$ | float64 | Extracted from OPD at pupil pixels | Vectorized OPD inside pupil |
| $\mathbf{E}$ | $(N, N)$ | complex128 | $A \cdot e^{i\phi}$ | Complex electric field |
| $\mathbf{E}_{\text{pad}}$ | $(N_{\text{pad}}, N_{\text{pad}})$ | complex128 | Zero-padded $\mathbf{E}$ | Padded electric field |
| $\tilde{\mathbf{E}}$ | $(N_{\text{pad}}, N_{\text{pad}})$ | complex128 | FFT result | Focal plane complex amplitude |
| $\text{diff}$ | $(N, N)$ | float64 | $(\boldsymbol{\phi} - \boldsymbol{\phi}_{\text{rec}}) \odot \text{pupil}$ | Residual error map |
| rmse_history | $(n_{\text{iter}},)$ | float64 | Empty list | RMSE values per iteration |
| all_coeffs | list of $(J,)$ | float64 | Empty list | Coefficient history |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N$ (resolution) | 120 | Telescope resolution in pixels |
| $D$ (diameter) | 8 m | Telescope diameter |
| $\Delta t$ (sampling_time) | 0.001 s | Temporal sampling |
| $\epsilon_{\text{obstruction}}$ | 0.0 | Central obstruction ratio |
| $J$ (n_zernike_modes) | 100 | Number of Zernike modes |
| $n_{\text{iter}}$ (n_iterations) | 10 | Number of atmospheric snapshots |
| $r_0$ | 0.15 m | Fried parameter |
| $L_0$ | 25 m | Outer scale |
| $v_{\text{wind}}$ | 10 m/s | Wind speed |
| $\theta_{\text{wind}}$ | 0° | Wind direction |
| $h$ (altitude) | 0 m | Turbulence layer altitude |
| zero_padding | 4 | FFT zero-padding factor |
| $m_{\text{star}}$ (magnitude) | 10 | Source magnitude |
| opt_band | 'I' | Optical band |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $\boldsymbol{\phi}_{\text{atm}}$ | $(N, N)$ | float64 | atm.OPD | Atmospheric OPD from simulation |
| pupil | $(N, N)$ | float64 | tel.pupil | Binary telescope pupil mask |
| $\lambda$ | scalar | float64 | ngs.wavelength | Source wavelength |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Telescope and Source Initialization
- **step_order**: 0
- **step_name**: Initialize optical system
- **formula**: 
  - Pixel size: $\delta = D / N$
  - Pupil: Binary mask from Telescope object
- **inputs**: resolution $(N)$, diameter $(D)$, central_obstruction
- **outputs**: tel.pupil $(N \times N)$, tel.pixelSize $(\delta)$, ngs.wavelength $(\lambda)$

### Step 1: Coordinate Grid Generation
- **step_order**: 1
- **step_name**: Create spatial coordinates
- **formula**:
$$x_{i,j} = \left(j - \frac{N}{2}\right) \cdot \delta, \quad y_{i,j} = \left(i - \frac{N}{2}\right) \cdot \delta$$
- **inputs**: $N$, $\delta$
- **outputs**: $\mathbf{X}, \mathbf{Y}$ each $(N \times N)$

### Step 2: Explicit Zernike Basis Generation (Demonstration)
- **step_order**: 2
- **step_name**: Generate explicit Zernike modes
- **formula**: For mode $(n, m)$ at normalized coordinates $(r, \theta)$:
$$r = \frac{\sqrt{x^2 + y^2}}{D/2}, \quad \theta = \arctan2(y, x)$$

Radial polynomial:
$$R_n^{|m|}(r) = \sum_{k=0}^{(n-|m|)/2} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+|m|}{2}-k\right)! \left(\frac{n-|m|}{2}-k\right)!} r^{n-2k}$$

Full Zernike mode:
$$Z_n^m(r, \theta) = \begin{cases}
\sqrt{n+1} \cdot R_n^0(r) & m = 0 \\
\sqrt{2(n+1)} \cdot R_n^m(r) \cdot \cos(m\theta) & m > 0 \\
\sqrt{2(n+1)} \cdot R_n^{|m|}(r) \cdot \sin(|m|\theta) & m < 0
\end{cases}$$

Noll index mapping (j → (n,m)):
| j | n | m |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 1 | 1 |
| 3 | 1 | -1 |
| 4 | 2 | 0 |
| 5 | 2 | -2 |
| 6 | 2 | 2 |

- **inputs**: $\mathbf{X}, \mathbf{Y}, D$, Noll indices
- **outputs**: zernike_basis_2d $(6 \times N \times N)$

### Step 3: Full Zernike Basis via OOPAO
- **step_order**: 3
- **step_name**: Compute Zernike basis and pseudoinverse
- **formula**:
$$\mathbf{Z}^{\dagger} = (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T$$
(computed via numpy.linalg.pinv)
- **inputs**: tel, $J$
- **outputs**: Z.modes $(N_{\text{pupil}} \times J)$, Z.modesFullRes $(N^2 \times J)$, $\mathbf{Z}^{\dagger}$ $(J \times N_{\text{pupil}})$

### Step 4: Atmosphere Initialization
- **step_order**: 4
- **step_name**: Initialize turbulence model
- **formula**: Kolmogorov/von Kármán phase screen generation (internal to OOPAO)
- **inputs**: tel, $r_0$, $L_0$, $v_{\text{wind}}$, $\theta_{\text{wind}}$, $h$
- **outputs**: atm object with OPD generation capability

### Step 5: Demonstration Phase Map
- **step_order**: 5
- **step_name**: Create test aberration
- **formula**:
$$\phi_{\text{demo}} = 0.5 \cdot Z_4 + 0.5 \cdot Z_6$$
where $Z_4$ is defocus and $Z_6$ is astigmatism.

OPD conversion:
$$\text{OPD} = \frac{\phi \cdot \lambda}{2\pi}$$
- **inputs**: zernike_basis_2d, $\lambda$
- **outputs**: phase_map $(N \times N)$, opd_map $(N \times N)$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I.1 | Initialize history arrays | rmse_history = [], all_coeffs = [], all_opd_original = [], all_opd_reconstructed = [] | — | Empty lists |
| I.2 | Extract operators | $\mathbf{Z}^{\dagger}$, $\mathbf{Z}_{\text{full}}$, pupil from data_dict | data_dict | Operators ready |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $i = 0, 1, \ldots, n_{\text{iter}}-1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | Update atmosphere | atm.update() | atm state | New turbulence realization | [FORWARD] Evolves phase screen |
| L.2 | Extract current OPD | $\boldsymbol{\phi} = \text{atm.OPD.copy()}$ | atm | $\boldsymbol{\phi}$ $(N \times N)$ | Snapshot of turbulence |
| L.3 | Mask extraction | $\boldsymbol{\phi}_{\text{masked}} = \boldsymbol{\phi}[\text{pupil} = 1]$ | $\boldsymbol{\phi}$, pupil | $\boldsymbol{\phi}_{\text{masked}}$ $(N_{\text{pupil}},)$ | Vectorize valid pixels |
| L.4 | Zernike projection | $\mathbf{c} = \mathbf{Z}^{\dagger} \boldsymbol{\phi}_{\text{masked}}$ | $\mathbf{Z}^{\dagger}$, $\boldsymbol{\phi}_{\text{masked}}$ | $\mathbf{c}$ $(J,)$ | Closed-form LS solution |
| L.5 | OPD reconstruction | $\boldsymbol{\phi}_{\text{rec}} = \text{squeeze}(\mathbf{Z}_{\text{full}} \mathbf{c})$ | $\mathbf{Z}_{\text{full}}$, $\mathbf{c}$ | $\boldsymbol{\phi}_{\text{rec}}$ $(N \times N)$ | [FORWARD] Synthesis |
| L.6 | Compute residual | $\text{diff} = (\boldsymbol{\phi} - \boldsymbol{\phi}_{\text{rec}}) \odot \text{pupil}$ | $\boldsymbol{\phi}$, $\boldsymbol{\phi}_{\text{rec}}$, pupil | diff $(N \times N)$ | Masked error |
| L.7 | Compute RMSE | $\text{RMSE}_i = \text{std}(\text{diff}[\text{pupil}=1])$ | diff, pupil | scalar | Standard deviation of residual |
| L.8 | Store results | Append $\text{RMSE}_i$, $\mathbf{c}$, $\boldsymbol{\phi}$, $\boldsymbol{\phi}_{\text{rec}}$ to histories | All computed values | Updated lists | — |

**END_LOOP**

#### Forward Operator (PSF Computation) [FORWARD]

**BEGIN_SUBROUTINE** forward_operator($\phi$, tel)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| F.1 | Get pupil amplitude | $A = \text{tel.pupil}$ | tel | $A$ $(N \times N)$ binary | Aperture function |
| F.2 | Complex field | $\mathbf{E} = A \odot e^{i\phi}$ | $A$, $\phi$ | $\mathbf{E}$ $(N \times N)$ complex | Electric field at pupil |
| F.3 | Compute padded size | $N_{\text{pad}} = N \times 4$, $w = (N_{\text{pad}} - N)/2$ | $N$, zero_padding=4 | $N_{\text{pad}}$, $w$ | Padding parameters |
| F.4 | Zero padding | $\mathbf{E}_{\text{pad}} = \text{pad}(\mathbf{E}, w)$ | $\mathbf{E}$, $w$ | $\mathbf{E}_{\text{pad}}$ $(N_{\text{pad}} \times N_{\text{pad}})$ | Increase sampling |
| F.5 | Fourier propagation | $\tilde{\mathbf{E}} = \mathcal{F}_{\text{shift}}\{\mathcal{F}\{\mathcal{F}^{-1}_{\text{shift}}\{\mathbf{E}_{\text{pad}}\}\}\}$ | $\mathbf{E}_{\text{pad}}$ | $\tilde{\mathbf{E}}$ $(N_{\text{pad}} \times N_{\text{pad}})$ complex | fftshift(fft2(ifftshift(·))) |
| F.6 | Intensity | $\text{PSF} = |\tilde{\mathbf{E}}|^2$ | $\tilde{\mathbf{E}}$ | PSF $(N_{\text{pad}} \times N_{\text{pad}})$ | Squared modulus |
| F.7 | Normalize | $\text{PSF} = \text{PSF} / \max(\text{PSF})$ | PSF | PSF normalized | Peak = 1 |

**END_SUBROUTINE**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Convert history to array | rmse_history = np.array(rmse_history) | list | $(n_{\text{iter}},)$ array |
| F.2 | Compute final RMSE | final_rmse = rmse_history[-1] | rmse_history | scalar |
| F.3 | Compute mean RMSE | mean_rmse = np.mean(rmse_history) | rmse_history | scalar |
| F.4 | Package results | Create results_dict with all histories and statistics | All stored data | Dictionary |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: 
$$i < n_{\text{iter}}$$
where $n_{\text{iter}} = 10$
- **parameters**: 
  - n_iterations = 10 (number of atmospheric snapshots to process)
- **check_frequency**: After each complete iteration

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| RMSE (per iteration) | $\text{RMSE}_i = \sqrt{\frac{1}{N_{\text{pupil}}}\sum_{p \in \mathcal{P}}(\phi_p - \phi_{\text{rec},p})^2}$ (computed as std) | Root mean square error of OPD reconstruction |
| Mean RMSE | $\bar{\text{RMSE}} = \frac{1}{n_{\text{iter}}}\sum_{i=1}^{n_{\text{iter}}} \text{RMSE}_i$ | Average RMSE across all iterations |
| Final RMSE | $\text{RMSE}_{n_{\text{iter}}}$ | RMSE of last iteration |
| Min RMSE | $\min_i \text{RMSE}_i$ | Best reconstruction quality |
| Max RMSE | $\max_i \text{RMSE}_i$ | Worst reconstruction quality |
| Std RMSE | $\sqrt{\frac{1}{n_{\text{iter}}}\sum_i(\text{RMSE}_i - \bar{\text{RMSE}})^2}$ | Variability of RMSE |
| PSF Peak | $\max(\text{PSF})$ | Maximum PSF value (normalized to 1) |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON MUST include:

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
  - steps[]

iteration_loop:
  - loop_variable
  - loop_range
  - steps[]
  - subroutines[]

finalization:
  - steps[]

stopping_criterion:
  - type
  - expression
  - parameters
  - check_frequency

evaluation:
  - metrics[]
    - name
    - formula
    - description

outputs:
  - rmse_history
  - all_coeffs
  - all_opd_original
  - all_opd_reconstructed
  - final_rmse
  - mean_rmse
  - psf
```