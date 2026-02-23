## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: Zernike Modal Wavefront Decomposition
- **algorithm_family**: Linear Least Squares / Modal Projection
- **summary**: Decomposes atmospheric optical path difference (OPD) into Zernike polynomial coefficients via pseudoinverse projection, with iterative evaluation over evolving atmospheric turbulence.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{c}} \|\boldsymbol{\phi}_{\text{pupil}} - \mathbf{Z}\mathbf{c}\|_2^2$$

where $\boldsymbol{\phi}_{\text{pupil}} \in \mathbb{R}^{N_{\text{valid}}}$ is the OPD vector restricted to valid pupil pixels, $\mathbf{Z} \in \mathbb{R}^{N_{\text{valid}} \times J}$ is the Zernike mode matrix, and $\mathbf{c} \in \mathbb{R}^{J}$ are the Zernike coefficients.

### data_fidelity_term
$$\mathcal{D}(\mathbf{c}) = \|\boldsymbol{\phi}_{\text{pupil}} - \mathbf{Z}\mathbf{c}\|_2^2 = \sum_{k \in \mathcal{P}} \left(\phi_k - \sum_{j=1}^{J} Z_{kj} c_j\right)^2$$

where $\mathcal{P} = \{k : \text{pupil}[k] = 1\}$ is the set of valid pupil pixel indices.

### regularization_term
None (unregularized least squares)

### constraints
None (unconstrained optimization)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{c}^{(i)}$ | primal | $(J,)$ | float64 | Computed at each iteration via $\mathbf{Z}^+ \boldsymbol{\phi}_{\text{pupil}}$ | Zernike coefficient vector at iteration $i$ |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\boldsymbol{\phi}_{\text{rec}}^{(i)}$ | auxiliary | $(N_{\text{res}}, N_{\text{res}})$ | float64 | Computed from $\mathbf{Z}_{\text{full}} \mathbf{c}^{(i)}$ | Reconstructed OPD at iteration $i$ |
| $\mathbf{r}^{(i)}$ | auxiliary | $(N_{\text{res}}, N_{\text{res}})$ | float64 | $(\boldsymbol{\phi}^{(i)} - \boldsymbol{\phi}_{\text{rec}}^{(i)}) \odot \mathbf{P}$ | Residual error map |
| $\text{RMSE}^{(i)}$ | auxiliary | scalar | float64 | - | Root mean square error at iteration $i$ |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\boldsymbol{\phi}^{(i)}$ | observations | $(N_{\text{res}}, N_{\text{res}})$ | float64 | From `atm.OPD` at each iteration | Current atmospheric OPD [m] |
| $\boldsymbol{\phi}_{\text{pupil}}^{(i)}$ | observations | $(N_{\text{valid}},)$ | float64 | $\boldsymbol{\phi}^{(i)}[\mathbf{P} = 1]$ | OPD restricted to pupil pixels |

### Constants (Operators/Matrices)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{Z}$ | constants | $(N_{\text{valid}}, J)$ | float64 | `Zernike(telObject=tel, J=100).modes` | Zernike mode matrix (pupil pixels only) |
| $\mathbf{Z}^+$ | constants | $(J, N_{\text{valid}})$ | float64 | `np.linalg.pinv(Z.modes)` | Moore-Penrose pseudoinverse of $\mathbf{Z}$ |
| $\mathbf{Z}_{\text{full}}$ | constants | $(N_{\text{res}}^2, J)$ | float64 | `Z.modesFullRes` | Full-resolution Zernike modes |
| $\mathbf{P}$ | constants | $(N_{\text{res}}, N_{\text{res}})$ | bool/float64 | From `tel.pupil` | Binary telescope pupil mask |

### Constants (Numerical Parameters)
| Symbol | Category | Value | Description |
|--------|----------|-------|-------------|
| $N_{\text{res}}$ | constants | 120 | Telescope resolution [pixels] |
| $D$ | constants | 8 | Telescope diameter [m] |
| $\Delta t$ | constants | 0.001 | Sampling time [s] (1/1000) |
| $\epsilon_{\text{obs}}$ | constants | 0.0 | Central obstruction ratio |
| $J$ | constants | 100 | Number of Zernike modes |
| $N_{\text{iter}}$ | constants | 10 | Number of iterations |
| $r_0$ | constants | 0.15 | Fried parameter [m] |
| $L_0$ | constants | 25 | Outer scale [m] |
| $v_{\text{wind}}$ | constants | 10 | Wind speed [m/s] |
| $\theta_{\text{wind}}$ | constants | 0 | Wind direction [deg] |
| $h$ | constants | 0 | Altitude [m] |
| $\lambda$ | constants | From `opt_band='I'` | Source wavelength [m] |
| $m_{\text{mag}}$ | constants | 10 | Source magnitude |
| $N_{\text{pad}}$ | constants | 4 | Zero-padding factor for PSF |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Telescope Initialization | $\mathbf{P}[i,j] = \begin{cases} 1 & \text{if } \sqrt{x_i^2 + y_j^2} \leq D/2 \\ 0 & \text{otherwise} \end{cases}$ | $N_{\text{res}}=120$, $D=8$, $\epsilon_{\text{obs}}=0$ | $\mathbf{P} \in \{0,1\}^{120 \times 120}$, `tel.pixelSize` $= D/N_{\text{res}}$ |
| 2 | Source Initialization | $\lambda = \lambda_{\text{band}}(\text{'I'})$ | `opt_band='I'`, `magnitude=10` | `ngs` object with wavelength $\lambda$ |
| 3 | Coordinate Grid | $x_{ij} = (j - N_{\text{res}}/2) \cdot \Delta_{\text{pix}}$, $y_{ij} = (i - N_{\text{res}}/2) \cdot \Delta_{\text{pix}}$ | $N_{\text{res}}$, $\Delta_{\text{pix}} = D/N_{\text{res}}$ | $\mathbf{X}, \mathbf{Y} \in \mathbb{R}^{120 \times 120}$ |
| 4 | Explicit Zernike Basis (6 modes) | $Z_j^{\text{2D}}(x,y) = N_{nm} R_n^{|m|}(\rho) \Theta_m(\theta)$ where $\rho = \frac{2\sqrt{x^2+y^2}}{D}$, $\theta = \arctan2(y,x)$ | $(n,m) \in \{(0,0),(1,1),(1,-1),(2,0),(2,-2),(2,2)\}$, $\mathbf{X}$, $\mathbf{Y}$, $D$ | $\mathbf{Z}_{\text{2D}} \in \mathbb{R}^{6 \times 120 \times 120}$ |
| 5 | Zernike Radial Polynomial | $R_n^m(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2}-k\right)! \left(\frac{n-m}{2}-k\right)!} \rho^{n-2k}$ | $n$, $m$, $\rho$ | $R_n^m(\rho)$ |
| 6 | Zernike Normalization | $N_{nm} = \begin{cases} \sqrt{n+1} & m=0 \\ \sqrt{2(n+1)} & m \neq 0 \end{cases}$ | $n$, $m$ | Normalization factor |
| 7 | Zernike Angular Part | $\Theta_m(\theta) = \begin{cases} 1 & m=0 \\ \cos(m\theta) & m > 0 \\ \sin(|m|\theta) & m < 0 \end{cases}$ | $m$, $\theta$ | Angular component |
| 8 | OOPAO Zernike Basis | Via `Zernike(telObject=tel, J=100).computeZernike(tel)` | `tel`, $J=100$ | $\mathbf{Z} \in \mathbb{R}^{N_{\text{valid}} \times 100}$, $\mathbf{Z}_{\text{full}} \in \mathbb{R}^{14400 \times 100}$ |
| 9 | Pseudoinverse Computation | $\mathbf{Z}^+ = (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T$ (via SVD) | $\mathbf{Z}$ | $\mathbf{Z}^+ \in \mathbb{R}^{100 \times N_{\text{valid}}}$ |
| 10 | Atmosphere Initialization | Von Kármán turbulence with $\Phi(\mathbf{k}) \propto (|\mathbf{k}|^2 + k_0^2)^{-11/6}$, $k_0 = 2\pi/L_0$ | $r_0=0.15$, $L_0=25$, $v=10$, $\theta=0$, $h=0$ | `atm` object |
| 11 | Demo Phase Map | $\boldsymbol{\phi}_{\text{demo}} = 0.5 \cdot Z_4^{\text{2D}} + 0.5 \cdot Z_6^{\text{2D}}$ | $\mathbf{Z}_{\text{2D}}$ (defocus at index 3, astigmatism at index 5) | $\boldsymbol{\phi}_{\text{demo}} \in \mathbb{R}^{120 \times 120}$ [rad] |
| 12 | Phase to OPD Conversion | $\text{OPD} = \boldsymbol{\phi} \cdot \frac{\lambda}{2\pi}$ | $\boldsymbol{\phi}_{\text{demo}}$, $\lambda$ | $\text{OPD}_{\text{demo}}$ [m] |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I1 | Initialize Storage | $\text{rmse\_history} \leftarrow []$, $\text{all\_coeffs} \leftarrow []$, $\text{all\_opd\_original} \leftarrow []$, $\text{all\_opd\_reconstructed} \leftarrow []$ | - | Empty lists |
| I2 | Set Iteration Counter | $i \leftarrow 0$ | - | $i = 0$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $i = 0, 1, \ldots, N_{\text{iter}}-1$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| L1 | Update Atmosphere | $\boldsymbol{\phi}^{(i)} \leftarrow \texttt{atm.update()}$; return `atm.OPD.copy()` | `atm` object, $\Delta t$ | $\boldsymbol{\phi}^{(i)} \in \mathbb{R}^{120 \times 120}$ | Translates phase screen by $v \cdot \Delta t$ |
| L2 | Extract Pupil Pixels | $\boldsymbol{\phi}_{\text{pupil}}^{(i)} = \boldsymbol{\phi}^{(i)}[\mathcal{P}]$ where $\mathcal{P} = \{k : \mathbf{P}[k] = 1\}$ | $\boldsymbol{\phi}^{(i)}$, $\mathbf{P}$ | $\boldsymbol{\phi}_{\text{pupil}}^{(i)} \in \mathbb{R}^{N_{\text{valid}}}$ | Boolean indexing |
| L3 | Zernike Projection | $\mathbf{c}^{(i)} = \mathbf{Z}^+ \boldsymbol{\phi}_{\text{pupil}}^{(i)}$ | $\mathbf{Z}^+ \in \mathbb{R}^{100 \times N_{\text{valid}}}$, $\boldsymbol{\phi}_{\text{pupil}}^{(i)}$ | $\mathbf{c}^{(i)} \in \mathbb{R}^{100}$ | Matrix-vector product |
| L4 | Wavefront Reconstruction | $\boldsymbol{\phi}_{\text{rec}}^{(i)} = \text{reshape}(\mathbf{Z}_{\text{full}} \mathbf{c}^{(i)}, (N_{\text{res}}, N_{\text{res}}))$ | $\mathbf{Z}_{\text{full}} \in \mathbb{R}^{14400 \times 100}$, $\mathbf{c}^{(i)}$ | $\boldsymbol{\phi}_{\text{rec}}^{(i)} \in \mathbb{R}^{120 \times 120}$ | `np.squeeze` applied |
| L5 | Compute Residual | $\mathbf{r}^{(i)} = (\boldsymbol{\phi}^{(i)} - \boldsymbol{\phi}_{\text{rec}}^{(i)}) \odot \mathbf{P}$ | $\boldsymbol{\phi}^{(i)}$, $\boldsymbol{\phi}_{\text{rec}}^{(i)}$, $\mathbf{P}$ | $\mathbf{r}^{(i)} \in \mathbb{R}^{120 \times 120}$ | Element-wise multiplication |
| L6 | Compute RMSE | $\text{RMSE}^{(i)} = \sqrt{\frac{1}{N_{\text{valid}}} \sum_{k \in \mathcal{P}} (r_k^{(i)})^2}$ implemented as `np.std(diff[pupil==1])` | $\mathbf{r}^{(i)}$, $\mathbf{P}$ | $\text{RMSE}^{(i)} \in \mathbb{R}$ | Standard deviation (assumes zero mean) |
| L7 | Store Results | Append $\text{RMSE}^{(i)}$ to `rmse_history`; append $\mathbf{c}^{(i)}$, $\boldsymbol{\phi}^{(i)}$, $\boldsymbol{\phi}_{\text{rec}}^{(i)}$ to respective lists | All computed quantities | Updated storage lists | Deep copies made |
| L8 | Increment Counter | $i \leftarrow i + 1$ | $i$ | $i$ | - |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F1 | Compute Summary Statistics | $\bar{\text{RMSE}} = \frac{1}{N_{\text{iter}}} \sum_{i=0}^{N_{\text{iter}}-1} \text{RMSE}^{(i)}$, $\text{RMSE}_{\text{final}} = \text{RMSE}^{(N_{\text{iter}}-1)}$ | `rmse_history` | `mean_rmse`, `final_rmse` |
| F2 | Package Results | Construct `results_dict` with all stored arrays | All lists, summary statistics | `results_dict` |

---

### 4.4 FORWARD OPERATOR (Auxiliary Model)

The forward operator maps phase to PSF:

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F.1 | Amplitude | $A = \mathbf{P}$ | Pupil mask $\mathbf{P}$ | $A \in \{0,1\}^{120 \times 120}$ |
| F.2 | Complex Field | $E = A \odot \exp(i \boldsymbol{\phi})$ | $A$, $\boldsymbol{\phi}$ | $E \in \mathbb{C}^{120 \times 120}$ |
| F.3 | Zero Padding | $E_{\text{pad}} = \text{pad}(E, \text{width}=(N_{\text{pad}}-1) \cdot N_{\text{res}}/2)$ | $E$, $N_{\text{pad}}=4$ | $E_{\text{pad}} \in \mathbb{C}^{480 \times 480}$ |
| F.4 | FFT Propagation | $\tilde{E} = \mathcal{F}_{\text{shift}}\{\mathcal{F}\{\mathcal{F}_{\text{shift}}^{-1}\{E_{\text{pad}}\}\}\}$ | $E_{\text{pad}}$ | $\tilde{E} \in \mathbb{C}^{480 \times 480}$ |
| F.5 | PSF Computation | $\text{PSF} = |\tilde{E}|^2$ | $\tilde{E}$ | $\text{PSF} \in \mathbb{R}_{\geq 0}^{480 \times 480}$ |
| F.6 | Normalization | $\text{PSF}_{\text{norm}} = \frac{\text{PSF}}{\max(\text{PSF})}$ | PSF | $\text{PSF}_{\text{norm}} \in [0,1]^{480 \times 480}$ |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $i \geq N_{\text{iter}}$ where $N_{\text{iter}} = 10$
- **parameters**: 
  - `n_iterations = 10` (fixed iteration count)
- **check_frequency**: After each complete iteration (step L8)

---

## 6. EVALUATION METRICS

| Metric | Formula | Description |
|--------|---------|-------------|
| Iteration RMSE | $\text{RMSE}^{(i)} = \text{std}\left(\{r_k^{(i)} : k \in \mathcal{P}\}\right) = \sqrt{\frac{1}{N_{\text{valid}}-1} \sum_{k \in \mathcal{P}} (r_k^{(i)} - \bar{r}^{(i)})^2}$ | Standard deviation of residual within pupil at iteration $i$ |
| Mean RMSE | $\bar{\text{RMSE}} = \frac{1}{N_{\text{iter}}} \sum_{i=0}^{N_{\text{iter}}-1} \text{RMSE}^{(i)}$ | Average reconstruction error across all iterations |
| Final RMSE | $\text{RMSE}_{\text{final}} = \text{RMSE}^{(N_{\text{iter}}-1)}$ | Reconstruction error at last iteration |
| Min RMSE | $\text{RMSE}_{\min} = \min_{i} \text{RMSE}^{(i)}$ | Best reconstruction error achieved |
| Max RMSE | $\text{RMSE}_{\max} = \max_{i} \text{RMSE}^{(i)}$ | Worst reconstruction error |
| Std RMSE | $\sigma_{\text{RMSE}} = \sqrt{\frac{1}{N_{\text{iter}}-1} \sum_{i=0}^{N_{\text{iter}}-1} (\text{RMSE}^{(i)} - \bar{\text{RMSE}})^2}$ | Variability of reconstruction error across iterations |

**Note**: All RMSE values are reported in nanometers via multiplication by $10^9$.