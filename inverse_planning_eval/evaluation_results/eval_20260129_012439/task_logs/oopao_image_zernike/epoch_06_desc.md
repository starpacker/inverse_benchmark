

## 0. ALGORITHM IDENTIFICATION

```json
{
  "algorithm_name": "Zernike Modal Decomposition with FFT-based PSF Forward Model",
  "algorithm_family": "Linear Least Squares / Modal Projection",
  "summary": "Decomposes atmospheric optical path difference (OPD) into Zernike polynomial coefficients via pseudoinverse projection, with a Fourier-optics forward model for PSF computation."
}
```

---

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{c}} \|\boldsymbol{\phi}_{\text{masked}} - \mathbf{Z} \mathbf{c}\|_2^2$$

where $\boldsymbol{\phi}_{\text{masked}} \in \mathbb{R}^{N_{\text{pupil}}}$ is the OPD restricted to valid pupil pixels, $\mathbf{Z} \in \mathbb{R}^{N_{\text{pupil}} \times J}$ is the Zernike basis matrix, and $\mathbf{c} \in \mathbb{R}^{J}$ is the coefficient vector.

**data_fidelity_term**:
$$\mathcal{D}(\mathbf{c}) = \|\boldsymbol{\phi}_{\text{masked}} - \mathbf{Z} \mathbf{c}\|_2^2 = \sum_{p \in \text{pupil}} \left(\phi_p - \sum_{j=1}^{J} Z_{p,j} c_j\right)^2$$

This is an **L2 (squared Euclidean)** data fidelity term.

**regularization_term**: None (unregularized least squares)

**constraints**: None (unconstrained optimization)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{c}$ | $(J,)$ | float64 | Computed via $\mathbf{Z}^+ \boldsymbol{\phi}_{\text{masked}}$ | Zernike coefficient vector |

### 2.2 Dual Variables

None (direct solution method, no dual formulation)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\phi}_{\text{masked}}$ | $(N_{\text{pupil}},)$ | float64 | Extracted from OPD | OPD values at valid pupil pixels |
| $\boldsymbol{\phi}_{\text{rec}}$ | $(N, N)$ | float64 | Computed | Reconstructed OPD on full grid |
| $\mathbf{E}$ | $(N, N)$ | complex128 | Computed | Complex electric field |
| $\mathbf{E}_{\text{pad}}$ | $(N_{\text{pad}}, N_{\text{pad}})$ | complex128 | Computed | Zero-padded electric field |
| $\mathbf{U}$ | $(N_{\text{pad}}, N_{\text{pad}})$ | complex128 | Computed | Complex amplitude in focal plane |
| $\text{PSF}$ | $(N_{\text{pad}}, N_{\text{pad}})$ | float64 | Computed | Point spread function |
| $\text{diff}$ | $(N, N)$ | float64 | Computed | Residual OPD map |
| $\text{rmse}$ | scalar | float64 | Computed | Root mean square error |
| $\text{rmse\_history}$ | $(K,)$ | float64 | Empty list | RMSE values across iterations |
| $\text{all\_coeffs}$ | list of $(J,)$ | float64 | Empty list | Stored coefficients per iteration |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $N$ | 120 | Telescope resolution (pixels) |
| $D$ | 8 m | Telescope diameter |
| $\Delta t$ | 0.001 s | Sampling time |
| $\epsilon_{\text{central}}$ | 0.0 | Central obstruction ratio |
| $J$ | 100 | Number of Zernike modes |
| $K$ | 10 | Number of iterations |
| $r_0$ | 0.15 m | Fried parameter |
| $L_0$ | 25 m | Outer scale |
| $v_{\text{wind}}$ | 10 m/s | Wind speed |
| $\theta_{\text{wind}}$ | 0° | Wind direction |
| $h$ | 0 m | Turbulence altitude |
| $\lambda$ | Source wavelength (I-band) | Observation wavelength |
| $N_{\text{pad}}$ | $4N = 480$ | Zero-padded array size |
| $\text{pad\_width}$ | $(N_{\text{pad}} - N)/2 = 180$ | Padding on each side |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\boldsymbol{\phi}_{\text{atm}}$ | $(N, N)$ | float64 | Atmospheric OPD from simulation (updated each iteration) |
| $\mathbf{P}$ | $(N, N)$ | float64 | Binary telescope pupil mask |
| $\mathbf{Z}$ | $(N_{\text{pupil}}, J)$ | float64 | Zernike basis matrix (modes at pupil pixels) |
| $\mathbf{Z}_{\text{full}}$ | $(N^2, J)$ | float64 | Zernike basis on full resolution grid |
| $\mathbf{Z}^+$ | $(J, N_{\text{pupil}})$ | float64 | Moore-Penrose pseudoinverse of $\mathbf{Z}$ |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Initialize Telescope | $\mathbf{P} \in \{0,1\}^{N \times N}$, pixel size $= D/N$ | $N=120$, $D=8$, $\epsilon=0$ | $\mathbf{P}$ $(120, 120)$, pixel_size $(scalar)$ | Circular aperture |
| P2 | Initialize Source | $\lambda = \lambda_{\text{I-band}}$ | opt_band='I', magnitude=10 | $\lambda$ (scalar) | NGS at infinity |
| P3 | Create Coordinate Grid | $y_{i,j} = (i - N/2) \cdot \text{pixel\_size}$, $x_{i,j} = (j - N/2) \cdot \text{pixel\_size}$ | $N$, pixel_size | $\mathbf{X}, \mathbf{Y}$ $(120, 120)$ each | Origin at grid center |
| P4 | Compute Zernike Basis (OOPAO) | $\mathbf{Z} = \text{Zernike}(\text{tel}, J).\text{modes}$ | tel, $J=100$ | $\mathbf{Z}$ $(N_{\text{pupil}}, 100)$ | Noll-ordered Zernike polynomials |
| P5 | Compute Pseudoinverse | $\mathbf{Z}^+ = (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T$ | $\mathbf{Z}$ $(N_{\text{pupil}}, J)$ | $\mathbf{Z}^+$ $(J, N_{\text{pupil}})$ | Full column rank assumed |
| P6 | Store Full-Resolution Modes | $\mathbf{Z}_{\text{full}} = \text{Zernike.modesFullRes}$ | Zernike object | $\mathbf{Z}_{\text{full}}$ $(N^2, J)$ | Modes defined on full grid |
| P7 | Initialize Atmosphere | Von Kármán spectrum with $r_0, L_0$ | tel, $r_0=0.15$, $L_0=25$, $v=10$, $\theta=0$, $h=0$ | atm object | Single-layer turbulence |
| P8 | Generate Demo Phase Map | $\boldsymbol{\phi}_{\text{demo}} = 0.5 \cdot Z_4 + 0.5 \cdot Z_6$ | Explicit Zernike modes (defocus + astigmatism) | $\boldsymbol{\phi}_{\text{demo}}$ $(120, 120)$ | For forward model demonstration |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | rmse_history | $[]$ (empty list) | $(0,)$ | float64 | Hardcoded |
| 2 | all_coeffs | $[]$ (empty list) | list | float64 | Hardcoded |
| 3 | all_opd_original | $[]$ (empty list) | list | float64 | Hardcoded |
| 4 | all_opd_reconstructed | $[]$ (empty list) | list | float64 | Hardcoded |
| 5 | iteration_index $i$ | $0$ | scalar | int | Hardcoded |

### 4.2 ITERATION LOOP

**--- BEGIN_LOOP (i = 0 to K-1, where K=10) ---**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 6 | Update Atmosphere | $\boldsymbol{\phi}_{\text{atm}}^{(i)} = \text{atm.update()}$ | atm object, $i$ | $\boldsymbol{\phi}_{\text{atm}}^{(i)}$ $(120, 120)$ | Evolves phase screen by $\Delta t \cdot v_{\text{wind}}$ |
| 7 | Copy Current OPD | $\boldsymbol{\phi}_{\text{current}} = \text{copy}(\boldsymbol{\phi}_{\text{atm}}^{(i)})$ | $\boldsymbol{\phi}_{\text{atm}}^{(i)}$ $(120, 120)$ | $\boldsymbol{\phi}_{\text{current}}$ $(120, 120)$ | Deep copy to preserve original |
| 8 | Extract Pupil Indices | $\mathcal{I} = \{(i,j) : P_{i,j} = 1\}$ | $\mathbf{P}$ $(120, 120)$ | $\mathcal{I}$ (index set, $N_{\text{pupil}}$ elements) | np.where(tel.pupil == 1) |
| 9 | Mask OPD to Pupil | $\boldsymbol{\phi}_{\text{masked}} = \boldsymbol{\phi}_{\text{current}}[\mathcal{I}]$ | $\boldsymbol{\phi}_{\text{current}}$ $(120, 120)$, $\mathcal{I}$ | $\boldsymbol{\phi}_{\text{masked}}$ $(N_{\text{pupil}},)$ | Vectorized extraction |
| 10 | **[FORWARD] Zernike Projection** | $\mathbf{c}^{(i)} = \mathbf{Z}^+ \boldsymbol{\phi}_{\text{masked}}$ | $\mathbf{Z}^+$ $(J, N_{\text{pupil}})$, $\boldsymbol{\phi}_{\text{masked}}$ $(N_{\text{pupil}},)$ | $\mathbf{c}^{(i)}$ $(J,)$ | Matrix-vector multiplication; closed-form LS solution |
| 11 | **[FORWARD] OPD Reconstruction** | $\boldsymbol{\phi}_{\text{rec}}^{(i)} = \text{reshape}(\mathbf{Z}_{\text{full}} \mathbf{c}^{(i)}, (N, N))$ | $\mathbf{Z}_{\text{full}}$ $(N^2, J)$, $\mathbf{c}^{(i)}$ $(J,)$ | $\boldsymbol{\phi}_{\text{rec}}^{(i)}$ $(120, 120)$ | np.squeeze after matmul |
| 12 | Compute Residual Map | $\mathbf{R}^{(i)} = (\boldsymbol{\phi}_{\text{current}} - \boldsymbol{\phi}_{\text{rec}}^{(i)}) \odot \mathbf{P}$ | $\boldsymbol{\phi}_{\text{current}}$, $\boldsymbol{\phi}_{\text{rec}}^{(i)}$, $\mathbf{P}$ (all $(120, 120)$) | $\mathbf{R}^{(i)}$ $(120, 120)$ | Element-wise subtraction and masking |
| 13 | Extract Residual in Pupil | $\mathbf{r}^{(i)} = \mathbf{R}^{(i)}[\mathcal{I}]$ | $\mathbf{R}^{(i)}$ $(120, 120)$, $\mathcal{I}$ | $\mathbf{r}^{(i)}$ $(N_{\text{pupil}},)$ | Values where pupil=1 |
| 14 | Compute RMSE | $\text{rmse}^{(i)} = \sqrt{\frac{1}{N_{\text{pupil}}} \sum_{p} (r_p^{(i)})^2}$ | $\mathbf{r}^{(i)}$ $(N_{\text{pupil}},)$ | $\text{rmse}^{(i)}$ (scalar) | np.std computes sample std |
| 15 | Append RMSE to History | $\text{rmse\_history} \leftarrow \text{rmse\_history} \cup \{\text{rmse}^{(i)}\}$ | rmse_history, $\text{rmse}^{(i)}$ | rmse_history (updated) | List append |
| 16 | Store Coefficients | $\text{all\_coeffs} \leftarrow \text{all\_coeffs} \cup \{\mathbf{c}^{(i)}\}$ | all_coeffs, $\mathbf{c}^{(i)}$ | all_coeffs (updated) | Deep copy stored |
| 17 | Store Original OPD | $\text{all\_opd\_original} \leftarrow \text{all\_opd\_original} \cup \{\boldsymbol{\phi}_{\text{current}}\}$ | all_opd_original, $\boldsymbol{\phi}_{\text{current}}$ | all_opd_original (updated) | Deep copy stored |
| 18 | Store Reconstructed OPD | $\text{all\_opd\_reconstructed} \leftarrow \text{all\_opd\_reconstructed} \cup \{\boldsymbol{\phi}_{\text{rec}}^{(i)}\}$ | all_opd_reconstructed, $\boldsymbol{\phi}_{\text{rec}}^{(i)}$ | all_opd_reconstructed (updated) | Deep copy stored |
| 19 | Increment Iteration | $i \leftarrow i + 1$ | $i$ | $i$ | Loop counter |

**--- END_LOOP ---**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 20 | Convert RMSE History | $\text{rmse\_array} = \text{np.array}(\text{rmse\_history})$ | rmse_history (list) | rmse_array $(K,)$ | Type conversion |
| 21 | Compute Final RMSE | $\text{rmse}_{\text{final}} = \text{rmse\_array}[K-1]$ | rmse_array $(K,)$ | $\text{rmse}_{\text{final}}$ (scalar) | Last element |
| 22 | Compute Mean RMSE | $\text{rmse}_{\text{mean}} = \frac{1}{K} \sum_{i=0}^{K-1} \text{rmse}^{(i)}$ | rmse_array $(K,)$ | $\text{rmse}_{\text{mean}}$ (scalar) | np.mean |

**Forward Model (PSF Computation) - Executed Separately:**

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| F1 | Get Pupil Amplitude | $\mathbf{A} = \mathbf{P}$ | $\mathbf{P}$ $(120, 120)$ | $\mathbf{A}$ $(120, 120)$ | Binary amplitude mask |
| F2 | Compute Complex Field | $\mathbf{E} = \mathbf{A} \odot \exp(i \boldsymbol{\phi})$ | $\mathbf{A}$, $\boldsymbol{\phi}$ (both $(120, 120)$) | $\mathbf{E}$ $(120, 120)$ complex | Element-wise operations |
| F3 | Compute Pad Width | $w = (N_{\text{pad}} - N) / 2 = 180$ | $N=120$, $N_{\text{pad}}=480$ | $w = 180$ | Integer division |
| F4 | Zero-Pad Electric Field | $\mathbf{E}_{\text{pad}} = \text{pad}(\mathbf{E}, w)$ | $\mathbf{E}$ $(120, 120)$, $w=180$ | $\mathbf{E}_{\text{pad}}$ $(480, 480)$ | np.pad with zeros |
| F5 | Apply ifftshift | $\mathbf{E}_{\text{shift}} = \text{ifftshift}(\mathbf{E}_{\text{pad}})$ | $\mathbf{E}_{\text{pad}}$ $(480, 480)$ | $\mathbf{E}_{\text{shift}}$ $(480, 480)$ | Shift zero-frequency to corner |
| F6 | **