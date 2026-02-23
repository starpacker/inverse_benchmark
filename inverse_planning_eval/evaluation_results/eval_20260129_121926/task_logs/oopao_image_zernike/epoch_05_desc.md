## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Zernike Modal Decomposition via Pseudoinverse
- **algorithm_family**: Linear Least Squares / Modal Wavefront Reconstruction
- **summary**: Decomposes atmospheric optical path difference (OPD) maps into Zernike polynomial coefficients using the Moore-Penrose pseudoinverse, then reconstructs the OPD for residual error analysis.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{c}} \|\boldsymbol{\phi}_{\text{masked}} - \mathbf{Z}\mathbf{c}\|_2^2$$

where the solution is obtained analytically via:
$$\mathbf{c}^* = \mathbf{Z}^{\dagger} \boldsymbol{\phi}_{\text{masked}}$$

### data_fidelity_term
$$\|\boldsymbol{\phi}_{\text{masked}} - \mathbf{Z}\mathbf{c}\|_2^2 = \sum_{p \in \mathcal{P}} \left(\phi_p - \sum_{j=1}^{J} Z_{p,j} c_j\right)^2$$

where $\mathcal{P}$ is the set of pixels inside the pupil.

### regularization_term
None (unconstrained least squares)

### constraints
None

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{c}$ | $(J,)$ | float64 | Computed from $\mathbf{Z}^{\dagger} \boldsymbol{\phi}_{\text{masked}}$ | Zernike coefficient vector |

### 2.2 Dual Variables

None (direct solution, no dual formulation)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\phi}_{\text{masked}}$ | $(N_{\text{pupil}},)$ | float64 | Extracted each iteration | OPD values at valid pupil pixels |
| $\hat{\boldsymbol{\phi}}$ | $(N_{\text{res}}, N_{\text{res}})$ | float64 | Computed each iteration | Reconstructed OPD map |
| $\mathbf{r}_{\text{diff}}$ | $(N_{\text{res}}, N_{\text{res}})$ | float64 | Computed each iteration | Residual map (original - reconstructed) × pupil |
| $\sigma_{\text{RMSE}}^{(i)}$ | scalar | float64 | Computed each iteration | Root mean square error at iteration $i$ |
| `rmse_history` | $(N_{\text{iter}},)$ | float64 | Empty list | History of RMSE values |
| `all_coeffs` | list of $(J,)$ | float64 | Empty list | Stored coefficients per iteration |

### 2.4 Constants

| Symbol | Value | Shape | Description |
|--------|-------|-------|-------------|
| $N_{\text{res}}$ | 120 | scalar | Telescope resolution in pixels |
| $D$ | 8 | scalar | Telescope diameter [m] |
| $\Delta t$ | 0.001 | scalar | Sampling time [s] |
| $\epsilon_{\text{obstruction}}$ | 0.0 | scalar | Central obstruction ratio |
| $J$ | 100 | scalar | Number of Zernike modes |
| $N_{\text{iter}}$ | 10 | scalar | Number of atmospheric realizations |
| $r_0$ | 0.15 | scalar | Fried parameter [m] |
| $L_0$ | 25 | scalar | Outer scale [m] |
| $v_{\text{wind}}$ | 10 | scalar | Wind speed [m/s] |
| $\theta_{\text{wind}}$ | 0 | scalar | Wind direction [deg] |
| $h$ | 0 | scalar | Turbulence altitude [m] |
| $\lambda$ | Source wavelength | scalar | Wavelength from 'I' band [m] |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $\boldsymbol{\Phi}_{\text{atm}}^{(i)}$ | $(N_{\text{res}}, N_{\text{res}})$ | float64 | `atm.OPD` | Atmospheric OPD at iteration $i$ [m] |
| $\mathbf{P}$ | $(N_{\text{res}}, N_{\text{res}})$ | float64 | `tel.pupil` | Binary telescope pupil mask |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Telescope Initialization | $\mathbf{P}[u,v] = \begin{cases} 1 & \text{if } \sqrt{(u-N_{\text{res}}/2)^2 + (v-N_{\text{res}}/2)^2} \cdot \delta_{\text{pix}} \leq D/2 \\ 0 & \text{otherwise} \end{cases}$ | $N_{\text{res}}, D, \epsilon_{\text{obstruction}}$ | $\mathbf{P} \in \{0,1\}^{N_{\text{res}} \times N_{\text{res}}}$, $\delta_{\text{pix}} = D/N_{\text{res}}$ |
| 2 | Coordinate Grid Generation | $x[u,v] = (v - N_{\text{res}}/2) \cdot \delta_{\text{pix}}$, $y[u,v] = (u - N_{\text{res}}/2) \cdot \delta_{\text{pix}}$ | $N_{\text{res}}, \delta_{\text{pix}}$ | $\mathbf{x}, \mathbf{y} \in \mathbb{R}^{N_{\text{res}} \times N_{\text{res}}}$ |
| 3 | Zernike Basis Computation | $Z_j[u,v] = \begin{cases} \sqrt{n+1} R_n^{|m|}(\rho) & m=0 \\ \sqrt{2(n+1)} R_n^{|m|}(\rho) \cos(m\theta) & m>0 \\ \sqrt{2(n+1)} R_n^{|m|}(\rho) \sin(\|m\|\theta) & m<0 \end{cases}$ where $\rho = \frac{2\sqrt{x^2+y^2}}{D}$, $\theta = \arctan2(y,x)$, and $R_n^m(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2}-k\right)! \left(\frac{n-m}{2}-k\right)!} \rho^{n-2k}$ | $\mathbf{x}, \mathbf{y}, D, J$ | $\mathbf{Z}_{\text{modes}} \in \mathbb{R}^{N_{\text{pupil}} \times J}$, $\mathbf{Z}_{\text{fullres}} \in \mathbb{R}^{N_{\text{res}}^2 \times J}$ |
| 4 | Pseudoinverse Computation | $\mathbf{Z}^{\dagger} = (\mathbf{Z}_{\text{modes}}^T \mathbf{Z}_{\text{modes}})^{-1} \mathbf{Z}_{\text{modes}}^T$ (via SVD-based pinv) | $\mathbf{Z}_{\text{modes}} \in \mathbb{R}^{N_{\text{pupil}} \times J}$ | $\mathbf{Z}^{\dagger} \in \mathbb{R}^{J \times N_{\text{pupil}}}$ |
| 5 | Atmosphere Initialization | Initialize Kolmogorov/von Kármán turbulence phase screens with parameters $(r_0, L_0, v_{\text{wind}}, \theta_{\text{wind}}, h)$ | $r_0, L_0, v_{\text{wind}}, \theta_{\text{wind}}, h, \mathbf{P}$ | `atm` object with OPD generation capability |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize RMSE History | $\texttt{rmse\_history} \leftarrow []$ | — | `rmse_history`: empty list | Allocate storage |
| 2 | Initialize Coefficient Storage | $\texttt{all\_coeffs} \leftarrow []$ | — | `all_coeffs`: empty list | Allocate storage |
| 3 | Initialize OPD Storage | $\texttt{all\_opd\_original} \leftarrow []$, $\texttt{all\_opd\_reconstructed} \leftarrow []$ | — | `all_opd_original`, `all_opd_reconstructed`: empty lists | Allocate storage |
| 4 | Set Iteration Counter | $i \leftarrow 0$ | — | $i = 0$ | Loop initialization |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $i = 0, 1, \ldots, N_{\text{iter}}-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 5 | Update Atmosphere | $\boldsymbol{\Phi}_{\text{atm}}^{(i)} \leftarrow \texttt{atm.update()}$ | `atm` object | $\boldsymbol{\Phi}_{\text{atm}}^{(i)} \in \mathbb{R}^{N_{\text{res}} \times N_{\text{res}}}$ | Evolves turbulence by $\Delta t$ |
| 6 | Copy Current OPD | $\boldsymbol{\phi}_{\text{current}} \leftarrow \text{copy}(\boldsymbol{\Phi}_{\text{atm}}^{(i)})$ | $\boldsymbol{\Phi}_{\text{atm}}^{(i)}$ | $\boldsymbol{\phi}_{\text{current}} \in \mathbb{R}^{N_{\text{res}} \times N_{\text{res}}}$ | Deep copy for storage |
| 7 | Extract Pupil Pixels | $\boldsymbol{\phi}_{\text{masked}} \leftarrow \boldsymbol{\phi}_{\text{current}}[\mathbf{P} = 1]$ | $\boldsymbol{\phi}_{\text{current}}, \mathbf{P}$ | $\boldsymbol{\phi}_{\text{masked}} \in \mathbb{R}^{N_{\text{pupil}}}$ | $N_{\text{pupil}} = \sum_{u,v} P[u,v]$ |
| 8 | Zernike Projection | $\mathbf{c} = \mathbf{Z}^{\dagger} \boldsymbol{\phi}_{\text{masked}}$ | $\mathbf{Z}^{\dagger} \in \mathbb{R}^{J \times N_{\text{pupil}}}$, $\boldsymbol{\phi}_{\text{masked}} \in \mathbb{R}^{N_{\text{pupil}}}$ | $\mathbf{c} \in \mathbb{R}^{J}$ | Matrix-vector product; least-squares solution |
| 9 | OPD Reconstruction | $\hat{\boldsymbol{\phi}}_{\text{vec}} = \mathbf{Z}_{\text{fullres}} \mathbf{c}$, then $\hat{\boldsymbol{\phi}} = \text{reshape}(\hat{\boldsymbol{\phi}}_{\text{vec}}, (N_{\text{res}}, N_{\text{res}}))$ | $\mathbf{Z}_{\text{fullres}} \in \mathbb{R}^{N_{\text{res}}^2 \times J}$, $\mathbf{c}$ | $\hat{\boldsymbol{\phi}} \in \mathbb{R}^{N_{\text{res}} \times N_{\text{res}}}$ | Squeeze removes singleton dimension |
| 10 | Compute Residual Map | $\mathbf{r}_{\text{diff}} = (\boldsymbol{\phi}_{\text{current}} - \hat{\boldsymbol{\phi}}) \odot \mathbf{P}$ | $\boldsymbol{\phi}_{\text{current}}, \hat{\boldsymbol{\phi}}, \mathbf{P}$ | $\mathbf{r}_{\text{diff}} \in \mathbb{R}^{N_{\text{res}} \times N_{\text{res}}}$ | Element-wise product masks outside pupil |
| 11 | Compute RMSE | $\sigma_{\text{RMSE}}^{(i)} = \sqrt{\frac{1}{N_{\text{pupil}}} \sum_{(u,v): P[u,v]=1} \left(r_{\text{diff}}[u,v] - \bar{r}\right)^2}$ where $\bar{r} = \frac{1}{N_{\text{pupil}}} \sum_{(u,v): P[u,v]=1} r_{\text{diff}}[u,v]$ | $\mathbf{r}_{\text{diff}}, \mathbf{P}$ | $\sigma_{\text{RMSE}}^{(i)} \in \mathbb{R}$ | Standard deviation of residual within pupil |
| 12 | Append RMSE | $\texttt{rmse\_history}.\text{append}(\sigma_{\text{RMSE}}^{(i)})$ | $\sigma_{\text{RMSE}}^{(i)}$, `rmse_history` | Updated `rmse_history` | Storage |
| 13 | Store Coefficients | $\texttt{all\_coeffs}.\text{append}(\text{copy}(\mathbf{c}))$ | $\mathbf{c}$, `all_coeffs` | Updated `all_coeffs` | Storage |
| 14 | Store Original OPD | $\texttt{all\_opd\_original}.\text{append}(\boldsymbol{\phi}_{\text{current}})$ | $\boldsymbol{\phi}_{\text{current}}$ | Updated `all_opd_original` | Storage |
| 15 | Store Reconstructed OPD | $\texttt{all\_opd\_reconstructed}.\text{append}(\hat{\boldsymbol{\phi}})$ | $\hat{\boldsymbol{\phi}}$ | Updated `all_opd_reconstructed` | Storage |
| 16 | Increment Counter | $i \leftarrow i + 1$ | $i$ | $i$ | Loop control |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 17 | Convert RMSE to Array | $\boldsymbol{\sigma}_{\text{RMSE}} = \text{array}(\texttt{rmse\_history})$ | `rmse_history` | $\boldsymbol{\sigma}_{\text{RMSE}} \in \mathbb{R}^{N_{\text{iter}}}$ | Type conversion |
| 18 | Compute Final RMSE | $\sigma_{\text{final}} = \boldsymbol{\sigma}_{\text{RMSE}}[N_{\text{iter}}-1]$ | $\boldsymbol{\sigma}_{\text{RMSE}}$ | $\sigma_{\text{final}} \in \mathbb{R}$ | Last iteration error |
| 19 | Compute Mean RMSE | $\bar{\sigma}_{\text{RMSE}} = \frac{1}{N_{\text{iter}}} \sum_{i=0}^{N_{\text{iter}}-1} \sigma_{\text{RMSE}}^{(i)}$ | $\boldsymbol{\sigma}_{\text{RMSE}}$ | $\bar{\sigma}_{\text{RMSE}} \in \mathbb{R}$ | Average over all realizations |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $i < N_{\text{iter}}$
- **parameters**: $N_{\text{iter}} = 10$
- **check_frequency**: After each iteration (Step 16)

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| RMSE (per iteration) | $\sigma_{\text{RMSE}}^{(i)} = \text{std}\left(\mathbf{r}_{\text{diff}}[\mathbf{P}=1]\right) = \sqrt{\frac{1}{N_{\text{pupil}}} \sum_{p \in \mathcal{P}} \left(r_p - \bar{r}\right)^2}$ | $\boldsymbol{\phi}_{\text{current}}$ (original OPD) | $\hat{\boldsymbol{\phi}}$ (reconstructed OPD) | Standard deviation of fitting residual within pupil [m] |
| Mean RMSE | $\bar{\sigma}_{\text{RMSE}} = \frac{1}{N_{\text{iter}}} \sum_{i=0}^{N_{\text{iter}}-1} \sigma_{\text{RMSE}}^{(i)}$ | All original OPDs | All reconstructed OPDs | Average fitting error over all atmosphere samples |
| Min RMSE | $\sigma_{\min} = \min_i \sigma_{\text{RMSE}}^{(i)}$ | — | — | Best fitting performance |
| Max RMSE | $\sigma_{\max} = \max_i \sigma_{\text{RMSE}}^{(i)}$ | — | — | Worst fitting performance |
| Std RMSE | $\text{std}(\boldsymbol{\sigma}_{\text{RMSE}})$ | — | — | Variability of fitting error across realizations |