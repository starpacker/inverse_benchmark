## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: AMICO-NODDI (Accelerated Microstructure Imaging via Convex Optimization for NODDI)
- **algorithm_family**: Dictionary-based Sparse Reconstruction with L1 Regularization
- **summary**: Fits the NODDI biophysical model to diffusion MRI data by solving a two-stage sparse regression: non-negative least squares for isotropic (CSF) compartment followed by LASSO for white matter compartment selection from a precomputed rotatable kernel dictionary.

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{x_{\text{iso}}, \mathbf{x}_{\text{wm}}} \left[ \frac{1}{2n}\|\mathbf{y}_{\text{wm}} - \mathbf{A}_{\text{wm}}\mathbf{x}_{\text{wm}}\|_2^2 + \alpha \|\mathbf{x}_{\text{wm}}\|_1 \right] + \mathbb{I}_{x_{\text{iso}} \geq 0} + \mathbb{I}_{\mathbf{x}_{\text{wm}} \geq 0}$$

where $\mathbf{y}_{\text{wm}} = \max(\mathbf{y} - x_{\text{iso}}\mathbf{a}_{\text{iso}}, 0)$ and $x_{\text{iso}} = \arg\min_{x \geq 0} \|\mathbf{y} - x\mathbf{a}_{\text{iso}}\|_2^2$

### data_fidelity_term

**Stage 1 (Isotropic):**
$$\|\mathbf{y} - x_{\text{iso}}\mathbf{a}_{\text{iso}}\|_2^2 = \sum_{i=1}^{n_S} (y_i - x_{\text{iso}} \cdot a_{\text{iso},i})^2$$

**Stage 2 (White Matter):**
$$\frac{1}{2n}\|\mathbf{y}_{\text{wm}} - \mathbf{A}_{\text{wm}}\mathbf{x}_{\text{wm}}\|_2^2 = \frac{1}{2n}\sum_{i=1}^{n_S}\left(y_{\text{wm},i} - \sum_{j=1}^{N_{\text{atoms}}} A_{\text{wm},ij} x_{\text{wm},j}\right)^2$$

### regularization_term

$$\alpha \|\mathbf{x}_{\text{wm}}\|_1 = \alpha \sum_{j=1}^{N_{\text{atoms}}} |x_{\text{wm},j}|$$

where $\alpha = \frac{\lambda_1}{n} = \frac{0.5}{n_S}$

### constraints

- $x_{\text{iso}} \geq 0$ (enforced via NNLS)
- $x_{\text{wm},j} \geq 0 \; \forall j$ (enforced via `positive=True` in LASSO)
- $y_{\text{wm},i} \geq 0 \; \forall i$ (enforced via clipping after isotropic subtraction)

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x_{\text{iso}}$ | $(1,)$ | float64 | NNLS solution | Isotropic (CSF) volume fraction coefficient |
| $\mathbf{x}_{\text{wm}}$ | $(N_{\text{atoms}}-1,)$ | float64 | Zero | White matter atom coefficients after LASSO |
| $\mathbf{x}_{\text{wm}}^{\text{scaled}}$ | $(N_{\text{atoms}}-1,)$ | float64 | LASSO output | Normalized white matter coefficients |

### 2.2 Dual Variables

None explicitly tracked (LASSO solver handles internally).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}_{\text{wm}}$ | $(n_S,)$ | float64 | Computed | Residual signal after isotropic subtraction |
| $\mathbf{A}_{\text{wm}}^{\text{scaled}}$ | $(n_S, N_{\text{atoms}}-1)$ | float32 | Computed | Column-normalized white matter dictionary |
| $\text{idx}_{\text{dir}}$ | $(1,)$ | int | KDTree query | Index of closest direction in 500-point sphere |
| $\text{sum}_{x_{\text{wm}}}$ | $(1,)$ | float64 | Computed | Sum of WM coefficients plus epsilon |
| $\text{results}$ | $(n_{\text{voxels}}, 3)$ | float64 | Zero | Accumulated [NDI, ODI, FWF] per voxel |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda_1$ | $5 \times 10^{-1}$ | LASSO regularization weight |
| $\gamma$ | $2.675987 \times 10^8$ rad/(s·T) | Gyromagnetic ratio |
| $d_{\text{Par}}$ | $1.7 \times 10^{-3}$ mm²/s | Parallel diffusivity |
| $d_{\text{Iso}}$ | $3.0 \times 10^{-3}$ mm²/s | Isotropic (CSF) diffusivity |
| $N_{\text{dirs}}$ | 500 | Number of directions on unit sphere |
| $l_{\max}$ | 12 | Maximum spherical harmonic order |
| $N_{\text{SH}}$ | 91 | Number of SH coefficients: $(l_{\max}+1)(l_{\max}+2)/2$ |
| IC_VFs | $[0.1, 0.181, ..., 0.99]$ (12 values) | `np.linspace(0.1, 0.99, 12)` |
| IC_ODs | $[0.03, 0.06, 0.09, ..., 0.99]$ (12 values) | `np.hstack([0.03, 0.06], np.linspace(0.09, 0.99, 10))` |
| $N_{\text{atoms}}$ | 145 | Total atoms: $12 \times 12 + 1$ (WM + ISO) |
| $\epsilon$ | $10^{-16}$ | Numerical stability constant |
| LASSO max_iter | 1000 | Maximum LASSO iterations |
| b0_thr | 10 s/mm² | Threshold for b=0 classification |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{Y}$ | $(n_x, n_y, n_z, n_S)$ | float64 | 4D diffusion-weighted image data |
| $\mathbf{M}$ | $(n_x, n_y, n_z)$ | bool | Binary brain mask |
| $\mathbf{D}$ | $(n_x, n_y, n_z, 3)$ | float64 | Principal diffusion directions from DTI |
| scheme.raw | $(n_S, 4\text{ or }7)$ | float64 | Acquisition parameters [gx, gy, gz, b] or [gx, gy, gz, G, Δ, δ, TE] |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load DWI Data | $\mathbf{Y} \leftarrow \texttt{nib.load(dwi\_file).get\_fdata()}$ | dwi_file: str | $\mathbf{Y}$: $(n_x, n_y, n_z, n_S)$ |
| 2 | Load/Create Mask | $\mathbf{M} \leftarrow \texttt{nib.load(mask\_file).get\_fdata()} > 0$ | mask_file: str | $\mathbf{M}$: $(n_x, n_y, n_z)$ bool |
| 3 | Parse Scheme | $\mathbf{b}_i = (\gamma \delta_i G_i)^2 (\Delta_i - \delta_i/3) \times 10^{-6}$ | scheme_file: str | scheme.b: $(n_S,)$, scheme.shells: list |
| 4 | Normalize Gradients | $\mathbf{g}_i \leftarrow \mathbf{g}_i / (\|\mathbf{g}_i\|_2 + 10^{-16})$ | scheme.raw: $(n_S, 3)$ | scheme.raw[:,:3]: $(n_S, 3)$ normalized |
| 5 | DTI Fit | $\mathbf{D} \leftarrow \texttt{dti.TensorModel(gtab).fit(Y, mask).evecs[..., 0]}$ | $\mathbf{Y}$, $\mathbf{M}$, gtab | $\mathbf{D}$: $(n_x, n_y, n_z, 3)$ |
| 6 | Precompute SH Matrices | $\mathbf{Y}_{\text{SH}} = \texttt{real\_sh\_descoteaux}(l_{\max}, \theta, \phi)$; $\mathbf{Y}_{\text{inv}} = (\mathbf{Y}_{\text{SH}}^T \mathbf{Y}_{\text{SH}})^{-1} \mathbf{Y}_{\text{SH}}^T$ | GRAD_500, $l_{\max}$ | AUX['fit']: $(N_{\text{SH}}, 500)$ |
| 7 | Precompute Rotation Vectors | $\texttt{AUX['Ylm\_rot'][i]} = Y_{lm}(\theta_i, \phi_i)$ for $i \in [0, 499]$ | GRAD_500 | AUX['Ylm_rot']: $(500,)$ object array |
| 8 | Compute SH Constants | $c_l = \sqrt{4\pi/(2l+1)}$ for $l \in \{0,2,4,...,l_{\max}\}$ | $l_{\max}$ | AUX['const']: $(N_{\text{SH}},)$ |
| 9 | Create High-Res Scheme | Expand each shell to 500 directions using GRAD_500 | scheme.shells | scheme_high: $(500 \times n_{\text{shells}}, 7)$ |
| 10 | Generate IC Kernels | $S_{\text{IC}}(\kappa) = \frac{1}{2} e^{L_\perp} \sum_{i=0}^{6} \text{LGI}_i(L_\perp - L_\parallel) \cdot c_i(\kappa) \cdot P_{2i}^0(\cos\theta)$ | $d_{\text{Par}}$, $\kappa$, protocol | signal_ic: $(500 \times n_{\text{shells}},)$ |
| 11 | Generate EC Kernels | $S_{\text{EC}} = \exp\left(-b[(d_w^{(0)} - d_w^{(1)})\cos^2\theta + d_w^{(1)}]\right)$ | $d_{\text{Par}}$, $\kappa$, $v_{\text{IC}}$ | signal_ec: $(500 \times n_{\text{shells}},)$ |
| 12 | Combine WM Signal | $S_{\text{WM}} = v_{\text{IC}} \cdot S_{\text{IC}} + (1 - v_{\text{IC}}) \cdot S_{\text{EC}}$ | signal_ic, signal_ec, $v_{\text{IC}}$ | signal: $(500 \times n_{\text{shells}},)$ |
| 13 | Generate ISO Kernel | $S_{\text{ISO}} = \exp(-b \cdot d_{\text{Iso}})$ | $d_{\text{Iso}}$, protocol | signal_iso: $(500 \times n_{\text{shells}},)$ |
| 14 | Rotate WM Kernels to SH | $K_{lm}^{(r)} = c_l \cdot K_{l,m=0} \cdot Y_{lm}(\theta_r, \phi_r)$ for each rotation $r$ | signal, AUX | KRlm: $(500, N_{\text{SH}} \times n_{\text{shells}})$ |
| 15 | Rotate ISO Kernel to SH | $K_{lm}^{\text{ISO}} = \texttt{AUX['fit']} \cdot S_{\text{ISO}}$ (no rotation needed) | signal_iso, AUX | lm_iso: $(N_{\text{SH}} \times n_{\text{shells}},)$ |
| 16 | Resample WM to Scheme | $K_{\text{resampled}}[r, \text{idx}] = \mathbf{Y}_{\text{out}} \cdot K_{lm}^{(r)}$ | KRlm, Ylm_out, idx_out | KERNELS['wm']: $(N_{\text{atoms}}-1, 500, n_S)$ |
| 17 | Resample ISO to Scheme | $K_{\text{ISO,resampled}}[\text{idx}] = \mathbf{Y}_{\text{out}} \cdot K_{lm}^{\text{ISO}}$ | lm_iso, Ylm_out, idx_out | KERNELS['iso']: $(n_S,)$ |
| 18 | Compute Normalization | $\text{norm}_j = 1 / \|\mathbf{K}_{\text{wm}}[j, 0, \text{dwi\_idx}]\|_2$ | KERNELS['wm'] | KERNELS['norms']: $(n_{\text{dwi}}, N_{\text{atoms}}-1)$ |
| 19 | Build KDTree | $\text{tree} = \texttt{KDTree(GRAD\_500)}$ | GRAD_500 | tree: KDTree object |
| 20 | Extract Masked Voxels | $\mathbf{y}_{\text{data}} = \mathbf{Y}[\text{mask\_indices}]$; $\mathbf{d}_{\text{dirs}} = \mathbf{D}[\text{mask\_indices}]$ | $\mathbf{Y}$, $\mathbf{D}$, $\mathbf{M}$ | y_data: $(n_{\text{voxels}}, n_S)$, d_dirs: $(n_{\text{voxels}}, 3)$ |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize Results Array | $\text{results} \leftarrow \mathbf{0}_{n_{\text{voxels}} \times 3}$ | $n_{\text{voxels}}$ | results: $(n_{\text{voxels}}, 3)$ | Stores [NDI, ODI, FWF] |
| 2 | Set Regularization | $\alpha \leftarrow \lambda_1 / n_S = 0.5 / n_S$ | $\lambda_1 = 0.5$, $n_S$ | $\alpha$: scalar | LASSO penalty weight |
| 3 | Initialize Output Maps | $\text{ndi\_map}, \text{odi\_map}, \text{fwf\_map} \leftarrow \mathbf{0}_{n_x \times n_y \times n_z}$ | data.shape[:3] | ndi_map, odi_map, fwf_map: $(n_x, n_y, n_z)$ | Zero-initialized |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $i = 0$ to $n_{\text{voxels}} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 4 | Extract Voxel Signal | $\mathbf{y} \leftarrow \mathbf{y}_{\text{data}}[i, :]$ | y_data, $i$ | $\mathbf{y}$: $(n_S,)$ | Raw DWI signal |
| 5 | Extract Voxel Direction | $\mathbf{d} \leftarrow \mathbf{d}_{\text{dirs}}[i, :]$ | d_dirs, $i$ | $\mathbf{d}$: $(3,)$ | Principal direction |
| 6 | Normalize Direction | $\mathbf{d} \leftarrow \mathbf{d} / (\|\mathbf{d}\|_2 + 10^{-16})$ | $\mathbf{d}$ | $\mathbf{d}$: $(3,)$ | Unit vector |
| 7 | Query Positive Direction | $(\text{dist}, \text{idx}_+) \leftarrow \texttt{tree.query}(\mathbf{d})$ | tree, $\mathbf{d}$ | dist: scalar, idx_+: int | Nearest neighbor |
| 8 | Query Negative Direction | $(\text{dist}_-, \text{idx}_-) \leftarrow \texttt{tree.query}(-\mathbf{d})$ | tree, $\mathbf{d}$ | dist_-: scalar, idx_-: int | Antipodal check |
| 9 | Select Best Direction | $\text{idx}_{\text{dir}} \leftarrow \begin{cases} \text{idx}_- & \text{if dist}_- < \text{dist} \\ \text{idx}_+ & \text{otherwise} \end{cases}$ | dist, dist_-, idx_+, idx_- | idx_dir: int | Handles sign ambiguity |
| 10 | Select WM Dictionary | $\mathbf{A}_{\text{wm}} \leftarrow \texttt{KERNELS['wm']}[:, \text{idx}_{\text{dir}}, :]^T$ | KERNELS['wm'], idx_dir | $\mathbf{A}_{\text{wm}}$: $(n_S, N_{\text{atoms}}-1)$ | Rotated WM atoms |
| 11 | Select ISO Dictionary | $\mathbf{A}_{\text{iso}} \leftarrow \texttt{KERNELS['iso']}[:, \text{None}]$ | KERNELS['iso'] | $\mathbf{A}_{\text{iso}}$: $(n_S, 1)$ | Column vector |
| 12 | Solve NNLS for ISO | $(x_{\text{iso}}, \_) \leftarrow \arg\min_{x \geq 0} \|\mathbf{y} - x \mathbf{a}_{\text{iso}}\|_2^2$ via `scipy.optimize.nnls` | $\mathbf{A}_{\text{iso}}$, $\mathbf{y}$ | $x_{\text{iso}}$: scalar | Closed-form NNLS |
| 13 | Compute WM Residual | $\mathbf{y}_{\text{wm}} \leftarrow \mathbf{y} - x_{\text{iso}} \cdot \mathbf{A}_{\text{iso}}[:, 0]$ | $\mathbf{y}$, $x_{\text{iso}}$, $\mathbf{A}_{\text{iso}}$ | $\mathbf{y}_{\text{wm}}$: $(n_S,)$ | Subtract ISO contribution |
| 14 | Clip Negative Values | $\mathbf{y}_{\text{wm}} \leftarrow \max(\mathbf{y}_{\text{wm}}, 0)$ | $\mathbf{y}_{\text{wm}}$ | $\mathbf{y}_{\text{wm}}$: $(n_S,)$ | Enforce non-negativity |
| 15 | Get Normalization Factors | $\mathbf{n} \leftarrow \texttt{KERNELS['norms']}[0, :]$ | KERNELS['norms'] | $\mathbf{n}$: $(N_{\text{atoms}}-1,)$ | Column norms |
| 16 | Scale WM Dictionary | $\mathbf{A}_{\text{wm}}^{\text{scaled}} \leftarrow \mathbf{A}_{\text{wm}} \odot \mathbf{n}$ | $\mathbf{A}_{\text{wm}}$, $\mathbf{n}$ | $\mathbf{A}_{\text{wm}}^{\text{scaled}}$: $(n_S, N_{\text{atoms}}-1)$ | Element-wise column scaling |
| 17 | Solve LASSO for WM | $\mathbf{x}_{\text{wm}}^{\text{scaled}} \leftarrow \arg\min_{\mathbf{x} \geq 0} \frac{1}{2n_S}\|\mathbf{y}_{\text{wm}} - \mathbf{A}_{\text{wm}}^{\text{scaled}}\mathbf{x}\|_2^2 + \alpha\|\mathbf{x}\|_1$ | $\mathbf{A}_{\text{wm}}^{\text{scaled}}$, $\mathbf{y}_{\text{wm}}$, $\alpha$ | $\mathbf{x}_{\text{wm}}^{\text{scaled}}$: $(N_{\text{atoms}}-1,)$ | sklearn LASSO with positive=True |
| 18 | Unscale WM Coefficients | $\mathbf{x}_{\text{wm}} \leftarrow \mathbf{x}_{\text{wm}}^{\text{scaled}} \odot \mathbf{n}$ | $\mathbf{x}_{\text{wm}}^{\text{scaled}}$, $\mathbf{n}$ | $\mathbf{x}_{\text{wm}}$: $(N_{\text{atoms}}-1,)$ | Recover original scale |
| 19 | Sum WM Coefficients | $\text{sum}_{\text{wm}} \leftarrow \sum_{j=1}^{N_{\text{atoms}}-1} x_{\text{wm},j} + 10^{-16}$ | $\mathbf{x}_{\text{wm}}$ | sum_wm: scalar | Avoid division by zero |
| 20 | Compute NDI | $\text{NDI} \leftarrow \frac{\sum_{j=1}^{N_{\text{atoms}}-1} x_{\text{wm},j} \cdot \text{ICVF}_j}{\text{sum}_{\text{wm}}}$ | $\mathbf{x}_{\text{wm}}$, KERNELS['icvf'], sum_wm | NDI: scalar | Weighted average ICVF |
| 21 | Compute Mean Kappa | $\bar{\kappa} \leftarrow \frac{\sum_{j=1}^{N_{\text{atoms}}-1} x_{\text{wm},j} \cdot \kappa_j}{\text{sum}_{\text{wm}}}$ | $\mathbf{x}_{\text{wm}}$, KERNELS['kappa'], sum_wm | $\bar{\kappa}$: scalar | Weighted average kappa |
| 22 | Compute ODI | $\text{ODI} \leftarrow \frac{2}{\pi} \arctan\left(\frac{1}{\bar{\kappa}}\right)$ | $\bar{\kappa}$ | ODI: scalar | Convert kappa to ODI |
| 23 | Compute Total Sum | $\text{total} \leftarrow \text{sum}_{\text{wm}} + x_{\text{iso}}$ | sum_wm, $x_{\text{iso}}$ | total: scalar | Normalization factor |
| 24 | Compute FWF | $\text{FWF} \leftarrow \frac{x_{\text{iso}}}{\text{total}}$ | $x_{\text{iso}}$, total | FWF: scalar | Free water fraction |
| 25 | Store Results | $\text{results}[i, :] \leftarrow [\text{NDI}, \text{ODI}, \text{FWF}]$ | NDI, ODI, FWF, $i$ | results: $(n_{\text{voxels}}, 3)$ | Accumulate |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 26 | Reconstruct NDI Map | $\text{ndi\_map}[\text{mask\_indices}] \leftarrow \text{results}[:, 0]$ | results, mask_indices | ndi_map: $(n_x, n_y, n_z)$ | Scatter to volume |
| 27 | Reconstruct ODI Map | $\text{odi\_map}[\text{mask\_indices}] \leftarrow \text{results}[:, 1]$ | results, mask_indices | odi_map: $(n_x, n_y, n_z)$ | Scatter to volume |
| 28 | Reconstruct FWF Map | $\text{fwf\_map}[\text{mask\_indices}] \leftarrow \text{results}[:, 2]$ | results, mask_indices | fwf_map: $(n_x, n_y, n_z)$ | Scatter to volume |
| 29 | Save Results | Write ndi_map, odi_map, fwf_map to NIfTI files | Maps, affine | Files on disk | Output persistence |

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $i < n_{\text{voxels}}$ (outer loop); LASSO internal: $k < 1000$ or convergence (coordinate descent tolerance)
- **parameters**: 
  - Outer loop: $n_{\text{voxels}}$ (determined by mask)
  - LASSO: `max_iter=1000`, default tolerance $10^{-4}$
- **check_frequency**: After each voxel (outer); after each coordinate update (inner LASSO)

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| RMSE | $\text{RMSE} = \sqrt{\frac{1}{N_{\text{mask}}} \sum_{v \in \text{mask}} (\text{NDI}_v^{\text{pred}} - \text{NDI}_v^{\text{GT}})^2}$ | GT_NDI (ground truth map) | ndi_map | Root mean squared error of NDI |
| PSNR | $\text{PSNR} = 20 \log_{10}\left(\frac{1}{\text{RMSE}}\right)$ dB | GT_NDI | ndi_map | Peak signal-to-noise ratio (assuming max value 1.0) |
| Mean NDI (mask) | $\bar{\text{NDI}} = \frac{1}{N_{\text{mask}}} \sum_{v \in \text{mask}} \text{NDI}_v$ | None | ndi_map[mask] | Average NDI within mask |
| Mean ODI (mask) | $\bar{\text{ODI}} = \frac{1}{N_{\text{mask}}} \sum_{v \in \text{mask}} \text{ODI}_v$ | None | odi_map[mask] | Average ODI within mask |
| Mean NDI (center) | $\bar{\text{NDI}}_{\text{center}} = \frac{1}{N_{\text{center}}} \sum_{v \in \text{center}} \text{NDI}_v$ where center: $(x - n_x/2)^2 + (y - n_y/2)^2 < (n_x/3)^2$ | None | ndi_map[center_mask] | NDI in crossing fiber region |
| Mean NDI (background) | $\bar{\text{NDI}}_{\text{bg}} = \frac{1}{N_{\text{bg}}} \sum_{v \in \text{mask} \land \neg\text{center}} \text{NDI}_v$ | None | ndi_map[mask & ~center_mask] | NDI outside crossing region |