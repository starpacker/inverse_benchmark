# Algorithmic Specification Document: Zernike-Based Wavefront Decomposition and PSF Computation

---

## 1. OBJECTIVE FUNCTION

### full_expression
The inverse model solves an unconstrained least-squares problem for Zernike coefficient estimation:

$$\min_{\mathbf{c}} \| \boldsymbol{\phi}_{\text{masked}} - \mathbf{Z} \mathbf{c} \|_2^2$$

where the closed-form solution is:

$$\mathbf{c}^* = \mathbf{Z}^{+} \boldsymbol{\phi}_{\text{masked}}$$

### data_fidelity_term
$$\mathcal{L}(\mathbf{c}) = \| \boldsymbol{\phi}_{\text{masked}} - \mathbf{Z} \mathbf{c} \|_2^2 = \sum_{p \in \mathcal{P}} \left( \phi_p - \sum_{j=1}^{J} Z_{p,j} c_j \right)^2$$

where $\mathcal{P} = \{p : \text{pupil}(p) = 1\}$ is the set of valid pupil pixels.

### regularization_term
None (direct pseudoinverse solution)

### constraints
None (unconstrained least-squares)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{c}$ | primal | $(J,)$ | float64 | Computed via pseudoinverse | Zernike coefficient vector |
| $\boldsymbol{\phi}_{\text{rec}}$ | primal | $(N, N)$ | float64 | Zero | Reconstructed OPD map |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{Z}$ | constants | $(P, J)$ | float64 | Precomputed | Zernike mode matrix (masked pixels) |
| $\mathbf{Z}^{+}$ | constants | $(J, P)$ | float64 | $\text{pinv}(\mathbf{Z})$ | Moore-Penrose pseudoinverse |
| $\mathbf{Z}_{\text{full}}$ | constants | $(N^2, J)$ | float64 | Precomputed | Full-resolution Zernike modes |
| $\mathbf{A}$ | constants | $(N, N)$ | binary | Telescope pupil | Pupil amplitude mask |
| $N$ | constants | scalar | int | 120 | Grid resolution |
| $D$ | constants | scalar | float64 | 8.0 | Telescope diameter [m] |
| $J$ | constants | scalar | int | 100 | Number of Zernike modes |
| $\lambda$ | constants | scalar | float64 | Source-dependent | Wavelength [m] |
| $K$ | constants | scalar | int | 4 | Zero-padding factor |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\boldsymbol{\phi}_{\text{atm}}$ | observations | $(N, N)$ | float64 | Atmosphere model | Current atmospheric OPD [m] |
| $\boldsymbol{\phi}_{\text{masked}}$ | observations | $(P,)$ | float64 | Extracted from $\boldsymbol{\phi}_{\text{atm}}$ | OPD values at valid pupil pixels |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Coordinate Grid Generation
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Coordinate Grid Construction |
| formula | $$x_{i,j} = \left(j - \frac{N}{2}\right) \cdot \Delta_{\text{pix}}, \quad y_{i,j} = \left(i - \frac{N}{2}\right) \cdot \Delta_{\text{pix}}$$ where $\Delta_{\text{pix}} = \frac{D}{N}$ |
| inputs | $N$ (scalar), $D$ (scalar) |
| outputs | $\mathbf{X} \in \mathbb{R}^{N \times N}$, $\mathbf{Y} \in \mathbb{R}^{N \times N}$ |

### Step 3.2: Polar Coordinate Transformation
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Cartesian to Normalized Polar |
| formula | $$\rho_{i,j} = \frac{2}{D}\sqrt{x_{i,j}^2 + y_{i,j}^2}, \quad \theta_{i,j} = \arctan2(y_{i,j}, x_{i,j})$$ |
| inputs | $\mathbf{X} \in \mathbb{R}^{N \times N}$, $\mathbf{Y} \in \mathbb{R}^{N \times N}$, $D$ (scalar) |
| outputs | $\boldsymbol{\rho} \in \mathbb{R}^{N \times N}$, $\boldsymbol{\theta} \in \mathbb{R}^{N \times N}$ |

### Step 3.3: Zernike Radial Polynomial Computation
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Radial Polynomial $R_n^m(\rho)$ |
| formula | $$R_n^m(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2}-k\right)! \left(\frac{n-m}{2}-k\right)!} \rho^{n-2k}$$ Valid only when $(n - m) \mod 2 = 0$; otherwise $R_n^m = 0$ |
| inputs | $n$ (scalar), $m$ (scalar), $\boldsymbol{\rho} \in \mathbb{R}^{N \times N}$ |
| outputs | $\mathbf{R}_n^m \in \mathbb{R}^{N \times N}$ |

### Step 3.4: Zernike Mode Assembly
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Full Zernike Mode $Z_n^m(\rho, \theta)$ |
| formula | $$Z_n^m(\rho, \theta) = \begin{cases} \sqrt{n+1} \cdot R_n^0(\rho) & m = 0 \\ \sqrt{2(n+1)} \cdot R_n^m(\rho) \cdot \cos(m\theta) & m > 0 \\ \sqrt{2(n+1)} \cdot R_n^{|m|}(\rho) \cdot \sin(|m|\theta) & m < 0 \end{cases}$$ Applied only where $\rho \leq 1$ (inside pupil) |
| inputs | $n$ (scalar), $m$ (scalar), $\boldsymbol{\rho}$, $\boldsymbol{\theta}$, pupil mask |
| outputs | $\mathbf{Z}_j \in \mathbb{R}^{N \times N}$ for Noll index $j$ |

### Step 3.5: Noll Index to (n, m) Mapping
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Noll Index Conversion |
| formula | For Noll index $j$: $n = \lfloor \sqrt{2j - 1} + 0.5 \rfloor - 1$, then $m$ is derived from ordering convention. Explicit mapping used: $j=1 \to (0,0)$, $j=2 \to (1,1)$, $j=3 \to (1,-1)$, $j=4 \to (2,0)$, $j=5 \to (2,-2)$, $j=6 \to (2,2)$ |
| inputs | $j$ (Noll index) |
| outputs | $(n, m)$ tuple |

### Step 3.6: Zernike Basis Matrix Construction
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Basis Matrix Assembly |
| formula | $$\mathbf{Z}_{p,j} = Z_j(\rho_p, \theta_p) \quad \forall p \in \mathcal{P}, \; j \in \{1, \ldots, J\}$$ |
| inputs | $J$ Zernike mode images, pupil mask $\mathbf{A}$ |
| outputs | $\mathbf{Z} \in \mathbb{R}^{P \times J}$ where $P = \sum_{i,j} A_{i,j}$ |

### Step 3.7: Pseudoinverse Computation
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Moore-Penrose Pseudoinverse |
| formula | $$\mathbf{Z}^{+} = (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T$$ (computed via SVD-based `np.linalg.pinv`) |
| inputs | $\mathbf{Z} \in \mathbb{R}^{P \times J}$ |
| outputs | $\mathbf{Z}^{+} \in \mathbb{R}^{J \times P}$ |

### Step 3.8: Phase Map to OPD Conversion
| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | Phase to OPD |
| formula | $$\text{OPD} = \frac{\phi \cdot \lambda}{2\pi}$$ |
| inputs | $\boldsymbol{\phi} \in \mathbb{R}^{N \times N}$ [rad], $\lambda$ [m] |
| outputs | $\mathbf{OPD} \in \mathbb{R}^{N \times N}$ [m] |

---

## 4. ALGORITHM STEPS

### Initialization

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Initialize telescope | $\mathbf{A} \leftarrow \text{Telescope}(N, D, \Delta t, \epsilon)$ | Resolution $N$, diameter $D$, sampling time $\Delta t$, central obstruction $\epsilon$ | Pupil mask $\mathbf{A} \in \{0,1\}^{N \times N}$ | OOPAO Telescope object |
| Initialize atmosphere | $\text{atm} \leftarrow \text{Atmosphere}(r_0, L_0, v, \theta_w, h)$ | Fried parameter $r_0$, outer scale $L_0$, wind speed $v$, wind direction $\theta_w$, altitude $h$ | Atmosphere object with OPD generator | Kolmogorov/von Kármán statistics |
| Compute Zernike basis | $\mathbf{Z}, \mathbf{Z}^{+}, \mathbf{Z}_{\text{full}} \leftarrow \text{ZernikeBasis}(J, \mathbf{A})$ | Number of modes $J$, pupil $\mathbf{A}$ | Basis matrices | See preprocessing steps 3-7 |
| Initialize storage | $\mathcal{H}_{\text{RMSE}} \leftarrow []$, $\mathcal{C} \leftarrow []$ | None | Empty lists | For history tracking |

---

### Iteration Loop

**BEGIN LOOP** for $i = 1, \ldots, N_{\text{iter}}$

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Update atmosphere | $\boldsymbol{\phi}_{\text{atm}}^{(i)} \leftarrow \text{atm.update()}$ | Atmosphere state at $i-1$ | Current OPD map $\boldsymbol{\phi}_{\text{atm}}^{(i)} \in \mathbb{R}^{N \times N}$ | Temporal evolution via frozen-flow |
| Extract masked OPD | $\boldsymbol{\phi}_{\text{masked}}^{(i)} = \boldsymbol{\phi}_{\text{atm}}^{(i)}[\mathbf{A} = 1]$ | $\boldsymbol{\phi}_{\text{atm}}^{(i)}$, $\mathbf{A}$ | $\boldsymbol{\phi}_{\text{masked}}^{(i)} \in \mathbb{R}^{P}$ | Flatten valid pixels |
| Zernike projection | $\mathbf{c}^{(i)} = \mathbf{Z}^{+} \boldsymbol{\phi}_{\text{masked}}^{(i)}$ | $\mathbf{Z}^{+} \in \mathbb{R}^{J \times P}$, $\boldsymbol{\phi}_{\text{masked}}^{(i)}$ | Coefficients $\mathbf{c}^{(i)} \in \mathbb{R}^{J}$ | Matrix-vector product |
| Wavefront reconstruction | $\boldsymbol{\phi}_{\text{rec}}^{(i)} = \text{reshape}\left(\mathbf{Z}_{\text{full}} \mathbf{c}^{(i)}, (N, N)\right)$ | $\mathbf{Z}_{\text{full}} \in \mathbb{R}^{N^2 \times J}$, $\mathbf{c}^{(i)}$ | $\boldsymbol{\phi}_{\text{rec}}^{(i)} \in \mathbb{R}^{N \times N}$ | Full-resolution synthesis |
| Compute residual | $\mathbf{R}^{(i)} = \left(\boldsymbol{\phi}_{\text{atm}}^{(i)} - \boldsymbol{\phi}_{\text{rec}}^{(i)}\right) \odot \mathbf{A}$ | Original OPD, reconstructed OPD, pupil | $\mathbf{R}^{(i)} \in \mathbb{R}^{N \times N}$ | Element-wise product with mask |
| Compute RMSE | $\text{RMSE}^{(i)} = \sqrt{\frac{1}{P} \sum_{p \in \mathcal{P}} \left(R_p^{(i)}\right)^2}$ | $\mathbf{R}^{(i)}$, pupil mask | $\text{RMSE}^{(i)} \in \mathbb{R}$ | Standard deviation over valid pixels |
| Store results | $\mathcal{H}_{\text{RMSE}}.\text{append}(\text{RMSE}^{(i)})$, $\mathcal{C}.\text{append}(\mathbf{c}^{(i)})$ | Current RMSE, coefficients | Updated history lists | For post-processing |
| Convergence check | Continue to next iteration | $i$, $N_{\text{iter}}$ | Boolean | Fixed iteration count (no early stopping) |

**END LOOP**

---

### Forward Operator (PSF Computation)

| step_name | formula | inputs | outputs | computational_note |
|-----------|---------|--------|---------|-------------------|
| Complex field construction | $\mathbf{E} = \mathbf{A} \odot \exp(i \boldsymbol{\phi})$ | Pupil $\mathbf{A}$, phase $\boldsymbol{\phi}$ [rad] | $\mathbf{E} \in \mathbb{C}^{N \times N}$ | Element-wise operations |
| Zero padding | $\mathbf{E}_{\text{pad}} = \text{pad}(\mathbf{E}, w)$ where $w = \frac{N(K-1)}{2}$ | $\mathbf{E}$, padding factor $K=4$ | $\mathbf{E}_{\text{pad}} \in \mathbb{C}^{NK \times NK}$ | Symmetric zero-padding |
| Focal plane propagation | $\tilde{\mathbf{E}} = \mathcal{F}_{\text{shift}}\left\{\mathcal{F}\left\{\mathcal{F}_{\text{shift}}^{-1}\{\mathbf{E}_{\text{pad}}\}\right\}\right\}$ | $\mathbf{E}_{\text{pad}}$ | $\tilde{\mathbf{E}} \in \mathbb{C}^{NK \times NK}$ | `fftshift(fft2(ifftshift(...)))` |
| Intensity computation | $\mathbf{I} = |\tilde{\mathbf{E}}|^2$ | $\tilde{\mathbf{E}}$ | $\mathbf{I} \in \mathbb{R}^{NK \times NK}$ | Squared modulus |
| Normalization | $\text{PSF} = \frac{\mathbf{I}}{\max(\mathbf{I})}$ | $\mathbf{I}$ | $\text{PSF} \in [0, 1]^{NK \times NK}$ | Peak normalization |

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $i \geq N_{\text{iter}}$ |
| parameters | $N_{\text{iter}} = 10$ (default) |

---

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| Instantaneous RMSE | $\text{RMSE}^{(i)} = \text{std}\left(\mathbf{R}^{(i)}[\mathbf{A}=1]\right) = \sqrt{\frac{1}{P}\sum_{p \in \mathcal{P}} \left(\phi_{\text{atm},p}^{(i)} - \phi_{\text{rec},p}^{(i)}\right)^2}$ | Root-mean-square residual error at iteration $i$ [m] |
| Mean RMSE | $\overline{\text{RMSE}} = \frac{1}{N_{\text{iter}}} \sum_{i=1}^{N_{\text{iter}}} \text{RMSE}^{(i)}$ | Average RMSE over all iterations [m] |
| Final RMSE | $\text{RMSE}_{\text{final}} = \text{RMSE}^{(N_{\text{iter}})}$ | RMSE at last iteration [m] |
| Minimum RMSE | $\text{RMSE}_{\min} = \min_{i} \text{RMSE}^{(i)}$ | Best achieved RMSE [m] |
| Maximum RMSE | $\text{RMSE}_{\max} = \max_{i} \text{RMSE}^{(i)}$ | Worst RMSE [m] |
| RMSE Standard Deviation | $\sigma_{\text{RMSE}} = \sqrt{\frac{1}{N_{\text{iter}}} \sum_{i=1}^{N_{\text{iter}}} \left(\text{RMSE}^{(i)} - \overline{\text{RMSE}}\right)^2}$ | Variability of RMSE across iterations [m] |
| PSF Peak Value | $\text{PSF}_{\max} = \max_{u,v} \text{PSF}(u,v)$ | Maximum intensity in PSF (equals 1 after normalization) |