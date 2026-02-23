## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: AMICO-NODDI (Accelerated Microstructure Imaging via Convex Optimization for NODDI)
- **algorithm_family**: Dictionary-based linear inverse problem with sparsity regularization
- **summary**: Estimates neurite density (NDI), orientation dispersion (ODI), and free water fraction (FWF) from diffusion MRI by solving a sparse non-negative least squares problem using precomputed NODDI signal dictionaries.

---

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{x}_{\text{wm}} \geq 0} \frac{1}{2} \|\mathbf{y}_{\text{wm}} - \mathbf{A}_{\text{wm}} \mathbf{x}_{\text{wm}}\|_2^2 + \alpha \|\mathbf{x}_{\text{wm}}\|_1$$

subject to prior isotropic fit:
$$\min_{x_{\text{iso}} \geq 0} \|\mathbf{y} - x_{\text{iso}} \mathbf{a}_{\text{iso}}\|_2^2$$

**data_fidelity_term**:
$$\frac{1}{2}\|\mathbf{y}_{\text{wm}} - \mathbf{A}_{\text{wm}} \mathbf{x}_{\text{wm}}\|_2^2$$
where $\mathbf{y}_{\text{wm}} = \mathbf{y} - x_{\text{iso}} \mathbf{a}_{\text{iso}}$ is the residual signal after isotropic subtraction.

**regularization_term**:
$$\alpha \|\mathbf{x}_{\text{wm}}\|_1, \quad \alpha = \frac{\lambda_1}{n_S}$$

**constraints**:
- $\mathbf{x}_{\text{wm}} \geq 0$ (non-negativity on white matter atom coefficients)
- $x_{\text{iso}} \geq 0$ (non-negativity on isotropic coefficient)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $x_{\text{iso}}$ | primal | scalar | float64 | NNLS solution | Isotropic (CSF) compartment coefficient |
| $\mathbf{x}_{\text{wm}}$ | primal | $(n_{\text{atoms}}-1,)$ | float64 | LASSO solution | White matter dictionary atom coefficients |

### Dual Variables
None (closed-form solvers used)

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}$ | auxiliary | $(n_S,)$ | float64 | From data | Measured DWI signal for current voxel |
| $\mathbf{y}_{\text{wm}}$ | auxiliary | $(n_S,)$ | float64 | $\max(0, \mathbf{y} - x_{\text{iso}}\mathbf{a}_{\text{iso}})$ | Residual signal after isotropic removal |
| $\mathbf{A}_{\text{wm}}$ | auxiliary | $(n_S, n_{\text{atoms}}-1)$ | float32 | From kernels | Dictionary matrix for WM compartment |
| $\mathbf{a}_{\text{iso}}$ | auxiliary | $(n_S,)$ | float32 | From kernels | Isotropic signal kernel |
| $\mathbf{d}$ | auxiliary | $(3,)$ | float64 | DTI primary eigenvector | Principal diffusion direction |
| $i_{\text{dir}}$ | auxiliary | scalar | int | KDTree query | Index of closest direction in 500-grid |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\lambda_1$ | constants | scalar | float | $5 \times 10^{-1}$ | LASSO regularization weight |
| $\alpha$ | constants | scalar | float | $\lambda_1 / n_S$ | Normalized LASSO parameter |
| $d_{\parallel}$ | constants | scalar | float | $1.7 \times 10^{-3}$ mm²/s | Parallel diffusivity |
| $d_{\text{iso}}$ | constants | scalar | float | $3.0 \times 10^{-3}$ mm²/s | Isotropic diffusivity |
| $\gamma$ | constants | scalar | float | $2.675987 \times 10^{8}$ rad/(s·T) | Gyromagnetic ratio |
| $l_{\max}$ | constants | scalar | int | $12$ | Maximum SH order |
| $n_{\text{dirs}}$ | constants | scalar | int | $500$ | Number of directions on unit sphere |
| $b_0^{\text{thr}}$ | constants | scalar | float | $10$ s/mm² | b=0 classification threshold |
| $\text{IC\_VFs}$ | constants | $(12,)$ | float | $\text{linspace}(0.1, 0.99, 12)$ | Intracellular volume fraction grid |
| $\text{IC\_ODs}$ | constants | $(12,)$ | float | $[0.03, 0.06] \cup \text{linspace}(0.09, 0.99, 10)$ | Orientation dispersion grid |
| $n_{\text{atoms}}$ | constants | scalar | int | $12 \times 12 + 1 = 145$ | Total dictionary atoms (144 WM + 1 ISO) |
| $G_{\max}$ | constants | scalar | float | $0.04$ T/m | Maximum gradient strength for protocol estimation |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{D}$ | observations | $(n_x, n_y, n_z, n_S)$ | float64 | NIfTI load | 4D DWI volume |
| $\mathbf{M}$ | observations | $(n_x, n_y, n_z)$ | bool | NIfTI load | Brain mask |
| $\mathcal{S}$ | observations | $(n_S, 4\text{ or }7)$ | float64 | Text file | Acquisition scheme |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Load DWI | $\mathbf{D} \leftarrow \text{nib.load}(\text{file})$ | dwi_file: str | $\mathbf{D}$: $(n_x, n_y, n_z, n_S)$, affine: $(4,4)$ |
| 2 | Load Mask | $\mathbf{M} \leftarrow \text{nib.load}(\text{file}) > 0$ | mask_file: str | $\mathbf{M}$: $(n_x, n_y, n_z)$ bool |
| 3 | Parse Scheme | $b_i = (\gamma \delta_i G_i)^2 (\Delta_i - \delta_i/3) \times 10^{-6}$ | scheme_file: str | $\mathcal{S}$: Scheme object with shells |
| 4 | Normalize Gradients | $\mathbf{g}_i \leftarrow \mathbf{g}_i / (\|\mathbf{g}_i\| + 10^{-16})$ | raw gradients: $(n_S, 3)$ | normalized gradients: $(n_S, 3)$ |
| 5 | DTI Fit | $\mathbf{V} = \text{evecs}(\text{TensorModel.fit}(\mathbf{D}, \mathbf{M}))$ | $\mathbf{D}$, $\mathbf{M}$, gtab | $\mathbf{V}$: $(n_x, n_y, n_z, 3)$ primary eigenvectors |
| 6 | Precompute SH Matrices | $\mathbf{Y} = \text{real\_sh\_descoteaux}(l_{\max}, \theta, \phi)$; $\mathbf{Y}^{\dagger} = (\mathbf{Y}^T\mathbf{Y})^{-1}\mathbf{Y}^T$ | $l_{\max}=12$, GRAD\_500: $(500,3)$ | $\mathbf{Y}^{\dagger}$: $(n_{SH}, 500)$, $\mathbf{Y}_{\text{rot}}$: $(500,)$ of $(n_{SH},)$ |
| 7 | Create HR Scheme | Stack 500 directions per shell with shell parameters | $\mathcal{S}$ | $\mathcal{S}_{\text{HR}}$: $(500 \times n_{\text{shells}}, 7)$ |
| 8 | Generate IC Signal | $E_{\text{IC}} = \frac{1}{2} e^{L_\perp} \sum_{i=0}^{6} \text{lgi}_i \cdot c_i \cdot \sqrt{\frac{i+0.25}{\pi}} P_{2i}^0(\cos\theta)$ | $d_\parallel$, $\kappa$, $\mathcal{S}_{\text{HR}}$ | $\mathbf{s}_{\text{IC}}$: $(500 \times n_{\text{shells}},)$ |
| 9 | Generate EC Signal | $E_{\text{EC}} = \exp(-b[(d_w^{(0)} - d_w^{(1)})\cos^2\theta + d_w^{(1)}])$ | $d_\parallel$, $\kappa$, $v_{\text{IC}}$, $\mathcal{S}_{\text{HR}}$ | $\mathbf{s}_{\text{EC}}$: $(500 \times n_{\text{shells}},)$ |
| 10 | Generate ISO Signal | $E_{\text{ISO}} = \exp(-b \cdot d_{\text{iso}})$ | $d_{\text{iso}}$, $\mathcal{S}_{\text{HR}}$ | $\mathbf{s}_{\text{ISO}}$: $(500 \times n_{\text{shells}},)$ |
| 11 | Combine WM Signal | $\mathbf{s}_{\text{WM}} = v_{\text{IC}} \cdot \mathbf{s}_{\text{IC}} + (1 - v_{\text{IC}}) \cdot \mathbf{s}_{\text{EC}}$ | $\mathbf{s}_{\text{IC}}$, $\mathbf{s}_{\text{EC}}$, $v_{\text{IC}}$ | $\mathbf{s}_{\text{WM}}$: $(500 \times n_{\text{shells}},)$ |
| 12 | Rotate to SH | $\mathbf{K}_{lm} = \mathbf{Y}^{\dagger} \mathbf{s}$; $\mathbf{K}_{lm}^{(r)} = c_l \cdot K_{l,0} \cdot Y_{lm}(\theta_r, \phi_r)$ | $\mathbf{s}$, AUX | $\mathbf{K}_{lm}^{(r)}$: $(500, n_{SH} \times n_{\text{shells}})$ |
| 13 | Resample to Scheme | $K_r[i_{\text{out}}] = \mathbf{Y}_{\text{out}} \cdot \mathbf{K}_{lm}^{(r)}$ | $\mathbf{K}_{lm}^{(r)}$, $\mathbf{Y}_{\text{out}}$ | KERNELS['wm']: $(144, 500, n_S)$, KERNELS['iso']: $(n_S,)$ |
| 14 | Compute Norms | $\text{norm}_j = 1 / \|\mathbf{K}_{\text{wm}}[j, 0, \text{dwi\_idx}]\|_2$ | KERNELS['wm'] | KERNELS['norms']: $(n_{\text{dwi}}, 144)$ |
| 15 | Build KDTree | $\mathcal{T} = \text{KDTree}(\text{GRAD\_500})$ | GRAD\_500: $(500,3)$ | $\mathcal{T}$: spatial index |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| I1 | Extract Masked Voxels | $\mathbf{Y}_{\text{data}} = \mathbf{D}[\text{mask\_indices}]$; $\mathbf{d}_{\text{dirs}} = \mathbf{V}[\text{mask\_indices}]$ | $\mathbf{D}$, $\mathbf{V}$, $\mathbf{M}$ | $\mathbf{Y}_{\text{data}}$: $(n_v, n_S)$, $\mathbf{d}_{\text{dirs}}$: $(n_v, 3)$ |
| I2 | Initialize Results | $\mathbf{R} = \mathbf{0}_{n_v \times 3}$ | $n_v$ | $\mathbf{R}$: $(n_v, 3)$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (over voxels $i = 0, \ldots, n_v - 1$)

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L1 | Extract Signal | $\mathbf{y} = \mathbf{Y}_{\text{data}}[i, :]$ | $\mathbf{Y}_{\text{data}}$, $i$ | $\mathbf{y}$: $(n_S,)$ | |
| L2 | Normalize Direction | $\mathbf{d} = \mathbf{d}_{\text{dirs}}[i] / (\|\mathbf{d}_{\text{dirs}}[i]\| + 10^{-16})$ | $\mathbf{d}_{\text{dirs}}$, $i$ | $\mathbf{d}$: $(3,)$ | |
| L3 | Find Closest Direction | $i_{\text{dir}} = \arg\min_j \min(\|\mathbf{d} - \mathbf{g}_j\|, \|\mathbf{d} + \mathbf{g}_j\|)$ | $\mathbf{d}$, $\mathcal{T}$ | $i_{\text{dir}}$: int | Antipodal symmetry |
| L4 | Select Dictionary | $\mathbf{A}_{\text{wm}} = \text{KERNELS['wm']}[:, i_{\text{dir}}, :]^T$; $\mathbf{a}_{\text{iso}} = \text{KERNELS['iso']}$ | KERNELS, $i_{\text{dir}}$ | $\mathbf{A}_{\text{wm}}$: $(n_S, 144)$, $\mathbf{a}_{\text{iso}}$: $(n_S,)$ | |
| L5 | NNLS Isotropic Fit | $x_{\text{iso}} = \arg\min_{x \geq 0} \|\mathbf{y} - x \mathbf{a}_{\text{iso}}\|_2^2$ | $\mathbf{y}$, $\mathbf{a}_{\text{iso}}$ | $x_{\text{iso}}$: scalar | scipy.optimize.nnls |
| L6 | Compute Residual | $\mathbf{y}_{\text{wm}} = \max(0, \mathbf{y} - x_{\text{iso}} \mathbf{a}_{\text{iso}})$ | $\mathbf{y}$, $x_{\text{iso}}$, $\mathbf{a}_{\text{iso}}$ | $\mathbf{y}_{\text{wm}}$: $(n_S,)$ | Elementwise max |
| L7 | Scale Dictionary | $\tilde{\mathbf{A}}_{\text{wm}} = \mathbf{A}_{\text{wm}} \odot \text{norms}$ | $\mathbf{A}_{\text{wm}}$, KERNELS['norms'][0,:] | $\tilde{\mathbf{A}}_{\text{wm}}$: $(n_S, 144)$ | Column-wise scaling |
| L8 | LASSO Solve | $\tilde{\mathbf{x}}_{\text{wm}} = \arg\min_{\mathbf{x} \geq 0} \frac{1}{2n_S}\|\mathbf{y}_{\text{wm}} - \tilde{\mathbf{A}}_{\text{wm}}\mathbf{x}\|_2^2 + \alpha\|\mathbf{x}\|_1$ | $\mathbf{y}_{\text{wm}}$, $\tilde{\mathbf{A}}_{\text{wm}}$, $\alpha$ | $\tilde{\mathbf{x}}_{\text{wm}}$: $(144,)$ | sklearn Lasso, max_iter=1000 |
| L9 | Unscale Coefficients | $\mathbf{x}_{\text{wm}} = \tilde{\mathbf{x}}_{\text{wm}} \odot \text{norms}$ | $\tilde{\mathbf{x}}_{\text{wm}}$, norms | $\mathbf{x}_{\text{wm}}$: $(144,)$ | |
| L10 | Compute NDI | $\text{NDI} = \frac{\sum_j x_{\text{wm}}^{(j)} \cdot v_{\text{IC}}^{(j)}}{\sum_j x_{\text{wm}}^{(j)} + 10^{-16}}$ | $\mathbf{x}_{\text{wm}}$, KERNELS['icvf'] | NDI: scalar | Weighted average |
| L11 | Compute ODI | $\bar{\kappa} = \frac{\sum_j x_{\text{wm}}^{(j)} \cdot \kappa^{(j)}}{\sum_j x_{\text{wm}}^{(j)} + 10^{-16}}$; $\text{ODI} = \frac{2}{\pi}\arctan\left(\frac{1}{\bar{\kappa}}\right)$ | $\mathbf{x}_{\text{wm}}$, KERNELS['kappa'] | ODI: scalar | |
| L12 | Compute FWF | $\text{FWF} = \frac{x_{\text{iso}}}{\sum_j x_{\text{wm}}^{(j)} + x_{\text{iso}}}$ | $x_{\text{iso}}$, $\mathbf{x}_{\text{wm}}$ | FWF: scalar | |
| L13 | Store Results | $\mathbf{R}[i, :] = [\text{NDI}, \text{ODI}, \text{FWF}]$ | NDI, ODI, FWF | $\mathbf{R}$: updated | |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| F1 | Reconstruct NDI Map | $\text{NDI\_map}[\text{mask\_indices}] = \mathbf{R}[:, 0]$ | $\mathbf{R}$, mask\_indices | NDI\_map: $(n_x, n_y, n_z)$ |
| F2 | Reconstruct ODI Map | $\text{ODI\_map}[\text{mask\_indices}] = \mathbf{R}[:, 1]$ | $\mathbf{R}$, mask\_indices | ODI\_map: $(n_x, n_y, n_z)$ |
| F3 | Reconstruct FWF Map | $\text{FWF\_map}[\text{mask\_indices}] = \mathbf{R}[:, 2]$ | $\mathbf{R}$, mask\_indices | FWF\_map: $(n_x, n_y, n_z)$ |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit (inner LASSO solver) + single_pass (outer voxel loop)
- **expression**: 
  - Outer loop: $i < n_v$ (single pass over all masked voxels)
  - Inner LASSO: $k < 1000$ OR $\|\nabla \mathcal{L}\|_\infty < \epsilon_{\text{tol}}$ (sklearn default tolerance)
- **parameters**:
  - `max_iter = 1000` (LASSO maximum iterations)
  - `tol = 1e-4` (sklearn Lasso default convergence tolerance)
- **check_frequency**: Every LASSO iteration (inner); after each voxel (outer)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| RMSE | $\text{RMSE} = \sqrt{\frac{1}{|\mathcal{M}|}\sum_{v \in \mathcal{M}}(\text{NDI}_{\text{est}}(v) - \text{NDI}_{\text{gt}}(v))^2}$ | Root mean squared error of NDI within mask $\mathcal{M}$ |
| PSNR | $\text{PSNR} = 20 \log_{10}\left(\frac{1}{\text{RMSE}}\right)$ dB | Peak signal-to-noise ratio (assuming max NDI = 1) |
| Mean NDI | $\bar{\text{NDI}} = \frac{1}{|\mathcal{M}|}\sum_{v \in \mathcal{M}}\text{NDI}(v)$ | Mean neurite density index in mask |
| Mean ODI | $\bar{\text{ODI}} = \frac{1}{|\mathcal{M}|}\sum_{v \in \mathcal{M}}\text{ODI}(v)$ | Mean orientation dispersion index in mask |
| Regional Mean | $\bar{\text{NDI}}_{\mathcal{R}} = \frac{1}{|\mathcal{R}|}\sum_{v \in \mathcal{R}}\text{NDI}(v)$ | Mean NDI in region $\mathcal{R}$ (center vs background) |