# Algorithmic Specification Document: AMICO-NODDI Microstructure Imaging

## 1. Mathematical Formulation

### 1.1 Overall Objective

The algorithm solves a **dictionary-based linear inverse problem** to estimate neurite microstructure parameters from diffusion-weighted MRI data. For each voxel, we solve:

$$\min_{\mathbf{x} \geq 0} \frac{1}{2}\|\mathbf{y} - \mathbf{A}\mathbf{x}\|_2^2 + \lambda\|\mathbf{x}\|_1$$

where:
- $\mathbf{y} \in \mathbb{R}^{n_S}$: measured DWI signal (normalized)
- $\mathbf{A} \in \mathbb{R}^{n_S \times n_{\text{atoms}}}$: forward dictionary matrix
- $\mathbf{x} \in \mathbb{R}^{n_{\text{atoms}}}$: atom coefficients (non-negative)
- $\lambda$: sparsity regularization parameter

### 1.2 NODDI Biophysical Model

The diffusion signal is modeled as a three-compartment mixture:

$$S(\mathbf{g}, b) = v_{\text{ic}} \cdot S_{\text{ic}}(\mathbf{g}, b) + (1 - v_{\text{ic}}) \cdot S_{\text{ec}}(\mathbf{g}, b) + v_{\text{iso}} \cdot S_{\text{iso}}(b)$$

#### 1.2.1 Intracellular Compartment (Watson-Distributed Cylinders)

$$S_{\text{ic}} = \frac{1}{2} \sum_{\ell=0}^{6} L_\ell(\Lambda_\perp - \Lambda_\parallel) \cdot c_\ell(\kappa) \cdot \sqrt{\frac{2\ell + 0.5}{\pi}} \cdot P_{2\ell}^0(\cos\theta) \cdot e^{\Lambda_\perp}$$

where:
- $\Lambda_\parallel = -q^2 \cdot (Δ - \delta/3) \cdot d_\parallel$ (parallel attenuation)
- $\Lambda_\perp = 0$ (perpendicular attenuation for stick model)
- $q = \gamma \cdot \delta \cdot G$ (q-vector magnitude)
- $\cos\theta = \mathbf{g} \cdot \mathbf{n}$ (angle between gradient and fiber direction)
- $P_{2\ell}^0$: Legendre polynomials of order $2\ell$
- $L_\ell$: Legendre-Gaussian integral coefficients
- $c_\ell(\kappa)$: Watson spherical harmonic coefficients

**Watson SH Coefficients:**
$$c_0 = 2\sqrt{\pi}$$
$$c_1 = \sqrt{5} \cdot \frac{3\sqrt{\kappa} - (3 + 2\kappa) \cdot D(\sqrt{\kappa})}{\kappa} \cdot \frac{e^\kappa}{\text{erfi}(\sqrt{\kappa})}$$

where $D(x) = \frac{\sqrt{\pi}}{2} \cdot \frac{\text{erfi}(x)}{e^{x^2}}$ is the Dawson function.

**Legendre-Gaussian Integrals** (for $x = \Lambda_\perp - \Lambda_\parallel > 0.05$):
$$I_0(x) = \frac{\sqrt{\pi} \cdot \text{erf}(\sqrt{x})}{\sqrt{x}}$$
$$I_n(x) = \frac{-e^{-x} + (n - 0.5) \cdot I_{n-1}(x)}{x}, \quad n \geq 1$$

#### 1.2.2 Extracellular Compartment (Hindered Diffusion)

$$S_{\text{ec}} = \exp\left(-b \cdot \left[(D_\parallel^{\text{app}} - D_\perp^{\text{app}}) \cos^2\theta + D_\perp^{\text{app}}\right]\right)$$

**Apparent Diffusivities (Watson-averaged):**
$$D_\parallel^{\text{app}} = \frac{-(d_\parallel - d_\perp) + 2d_\perp\kappa + (d_\parallel - d_\perp) \cdot \frac{\sqrt{\kappa}}{D(\sqrt{\kappa})}}{2\kappa}$$

$$D_\perp^{\text{app}} = \frac{(d_\parallel - d_\perp) + 2(d_\parallel + d_\perp)\kappa - (d_\parallel - d_\perp) \cdot \frac{\sqrt{\kappa}}{D(\sqrt{\kappa})}}{4\kappa}$$

where:
- $d_\parallel$: intrinsic parallel diffusivity
- $d_\perp = d_\parallel \cdot (1 - v_{\text{ic}})$: intrinsic perpendicular diffusivity (tortuosity)

#### 1.2.3 Isotropic Compartment (Free Water)

$$S_{\text{iso}} = \exp(-b \cdot d_{\text{iso}})$$

### 1.3 Dictionary Construction

The forward operator $\mathbf{A}$ is constructed by discretizing the parameter space:

$$\mathbf{A} = [\mathbf{A}_{\text{WM}} \mid \mathbf{A}_{\text{ISO}}]$$

where columns of $\mathbf{A}_{\text{WM}}$ correspond to signals for parameter combinations $(\kappa_i, v_{\text{ic},j})$.

### 1.4 Spherical Harmonic Rotation Framework

For efficient computation across orientations, signals are represented in SH basis:

$$\mathbf{K}^{(\text{SH})} = \mathbf{Y}^+ \cdot \mathbf{K}^{(\text{sphere})}$$

where $\mathbf{Y}^+ = (\mathbf{Y}^T\mathbf{Y})^{-1}\mathbf{Y}^T$ is the pseudo-inverse of the SH basis matrix.

**Rotation operation:**
$$K_{\ell m}^{(\text{rot})} = \sqrt{\frac{4\pi}{2\ell+1}} \cdot K_{\ell 0} \cdot Y_{\ell m}(\theta_{\text{dir}}, \phi_{\text{dir}})$$

**Resampling to acquisition scheme:**
$$\mathbf{K}^{(\text{acq})} = \mathbf{Y}_{\text{acq}} \cdot \mathbf{K}^{(\text{SH,rot})}$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| WM atom coefficients | $\mathbf{x}_{\text{wm}}$ | $(n_{\text{atoms}}-1,)$ | float64 | Non-negative weights for WM dictionary atoms |
| ISO coefficient | $x_{\text{iso}}$ | scalar | float64 | Non-negative weight for isotropic compartment |

### 2.2 Derived Output Variables

| Variable | Symbol | Shape | Dtype | Formula |
|----------|--------|-------|-------|---------|
| NDI map | $\text{NDI}$ | $(N_x, N_y, N_z)$ | float32 | $\frac{\sum_i x_{\text{wm},i} \cdot v_{\text{ic},i}}{\sum_i x_{\text{wm},i}}$ |
| ODI map | $\text{ODI}$ | $(N_x, N_y, N_z)$ | float32 | $\frac{2}{\pi} \arctan\left(\frac{1}{\bar{\kappa}}\right)$ where $\bar{\kappa} = \frac{\sum_i x_{\text{wm},i} \cdot \kappa_i}{\sum_i x_{\text{wm},i}}$ |
| FWF map | $\text{FWF}$ | $(N_x, N_y, N_z)$ | float32 | $\frac{x_{\text{iso}}}{\sum_i x_{\text{wm},i} + x_{\text{iso}}}$ |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Parallel diffusivity | $d_\parallel$ | $1.7 \times 10^{-3}$ mm²/s | Intrinsic intra-axonal diffusivity |
| Isotropic diffusivity | $d_{\text{iso}}$ | $3.0 \times 10^{-3}$ mm²/s | Free water diffusivity |
| Gyromagnetic ratio | $\gamma$ | $2.675987 \times 10^8$ rad/(s·T) | Proton gyromagnetic ratio |
| SH maximum order | $\ell_{\max}$ | 12 | Maximum spherical harmonic degree |
| Number of SH coefficients | $n_{\text{SH}}$ | $\frac{(\ell_{\max}+1)(\ell_{\max}+2)}{2} = 91$ | - |
| Number of sphere directions | $n_{\text{dirs}}$ | 500 | High-resolution sphere sampling |
| LASSO regularization | $\lambda$ | $5 \times 10^{-1}$ | Sparsity penalty weight |
| b0 threshold | $b_0^{\text{thr}}$ | 10 s/mm² | Threshold for identifying b=0 volumes |
| IC volume fractions grid | $\{v_{\text{ic},j}\}$ | `linspace(0.1, 0.99, 12)` | 12 values |
| IC orientation dispersion grid | $\{\text{OD}_i\}$ | `[0.03, 0.06] ∪ linspace(0.09, 0.99, 10)` | 12 values |
| Concentration parameters | $\kappa_i$ | $\frac{1}{\tan(\text{OD}_i \cdot \pi/2)}$ | Derived from OD grid |

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | Dtype | Source |
|----------|--------|-------|-------|--------|
| DWI data | $\mathbf{D}$ | $(N_x, N_y, N_z, n_S)$ | float32/64 | NIfTI file |
| Brain mask | $\mathbf{M}$ | $(N_x, N_y, N_z)$ | bool | NIfTI file |
| Gradient directions | $\mathbf{G}$ | $(n_S, 3)$ | float64 | Scheme file columns 0-2 |
| b-values | $\mathbf{b}$ | $(n_S,)$ | float64 | Computed or from scheme |
| Gradient strengths | $G$ | $(n_S,)$ | float64 | Scheme file column 3 (version 1) |
| Pulse duration | $\delta$ | $(n_S,)$ | float64 | Scheme file column 5 (version 1) |
| Diffusion time | $\Delta$ | $(n_S,)$ | float64 | Scheme file column 4 (version 1) |
| Principal diffusion directions | $\mathbf{V}_1$ | $(N_x, N_y, N_z, 3)$ | float64 | From DTI fit |

### 2.5 Intermediate/Auxiliary Variables

| Variable | Shape | Description |
|----------|-------|-------------|
| `KERNELS['wm']` | $(n_{\text{atoms}}-1, 500, n_S)$ | Rotated WM dictionary for all 500 directions |
| `KERNELS['iso']` | $(n_S,)$ | Isotropic kernel (direction-independent) |
| `KERNELS['icvf']` | $(n_{\text{atoms}}-1,)$ | Volume fraction labels for each atom |
| `KERNELS['kappa']` | $(n_{\text{atoms}}-1,)$ | Concentration parameter labels for each atom |
| `KERNELS['norms']` | $(n_{\text{dwi}}, n_{\text{atoms}}-1)$ | Normalization factors for atoms |
| `AUX['fit']` | $(n_{\text{SH}}, 500)$ | SH fitting matrix $\mathbf{Y}^+$ |
| `AUX['Ylm_rot']` | $(500,)$ array of $(n_{\text{SH}},)$ | SH basis evaluated at each sphere direction |
| `AUX['const']` | $(n_{\text{SH}},)$ | Rotation constants $\sqrt{4\pi/(2\ell+1)}$ |
| `AUX['idx_m0']` | $(n_{\text{SH}},)$ | Indices of $m=0$ coefficients |

---

## 3. Preprocessing Pipeline

| Step | Operation | Input | Output | Formula/Description |
|------|-----------|-------|--------|---------------------|
| 1 | Load DWI | `dwi_file` | $\mathbf{D} \in \mathbb{R}^{N_x \times N_y \times N_z \times n_S}$ | `nib.load().get_fdata()` |
| 2 | Load mask | `mask_file` | $\mathbf{M} \in \{0,1\}^{N_x \times N_y \times N_z}$ | Binarize: $\mathbf{M} = (\text{mask\_data} > 0)$ |
| 3 | Parse scheme | `scheme_file` | `Scheme` object | Extract $\mathbf{G}, \mathbf{b}, \delta, \Delta$ |
| 4 | Compute b-values (v1) | $G, \delta, \Delta$ | $\mathbf{b}$ | $b = (\gamma \cdot \delta \cdot G)^2 \cdot (\Delta - \delta/3) \times 10^{-6}$ |
| 5 | Normalize gradients | $\mathbf{G}$ | $\hat{\mathbf{G}}$ | $\hat{\mathbf{g}}_i = \mathbf{g}_i / (\|\mathbf{g}_i\|_2 + 10^{-16})$ |
| 6 | Identify shells | $\mathbf{b}$ | `shells` list | Round b-values to nearest 100, group by unique values |
| 7 | DTI fitting | $\mathbf{D}, \mathbf{M}, \mathbf{b}, \hat{\mathbf{G}}$ | $\mathbf{V}_1$ | DIPY tensor model: extract primary eigenvector |
| 8 | Precompute SH matrices | $\ell_{\max}$, `GRAD_500` | `AUX` dictionary | $\mathbf{Y}^+ = (\mathbf{Y}^T\mathbf{Y})^{-1}\mathbf{Y}^T$ for 500-direction sphere |

---

## 4. Algorithm Steps

### 4.1 Dictionary Generation (`forward_operator`)

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 1 | Create high-res scheme | Replicate each shell to 500 directions | `scheme` | `scheme_high` |
| 2 | Initialize compartment models | - | `scheme_high` | `noddi_ic`, `noddi_ec`, `noddi_iso` |
| 3 | **For each** $(\kappa_i, v_{\text{ic},j})$: | | | |
| 3a | Compute IC signal | $S_{\text{ic}} = f_{\text{ic}}(d_\parallel, \kappa_i)$ (see §1.2.1) | $d_\parallel, \kappa_i$ | $\mathbf{s}_{\text{ic}} \in \mathbb{R}^{500 \cdot n_{\text{shells}}}$ |
| 3b | Compute EC signal | $S_{\text{ec}} = f_{\text{ec}}(d_\parallel, \kappa_i, v_{\text{ic},j})$ (see §1.2.2) | $d_\parallel, \kappa_i, v_{\text{ic},j}$ | $\mathbf{s}_{\text{ec}} \in \mathbb{R}^{500 \cdot n_{\text{shells}}}$ |
| 3c | Combine compartments | $\mathbf{s} = v_{\text{ic},j} \cdot \mathbf{s}_{\text{ic}} + (1 - v_{\text{ic},j}) \cdot \mathbf{s}_{\text{ec}}$ | $\mathbf{s}_{\text{ic}}, \mathbf{s}_{\text{ec}}, v_{\text{ic},j}$ | $\mathbf{s} \in \mathbb{R}^{500 \cdot n_{\text{shells}}}$ |
| 3d | Project to SH | $\mathbf{K}^{(\text{SH})}_s = \mathbf{Y}^+ \cdot \mathbf{s}[\text{shell } s]$ for each shell | $\mathbf{s}$, `AUX['fit']` | $\mathbf{K}^{(\text{SH})} \in \mathbb{R}^{n_{\text{SH}} \cdot n_{\text{shells}}}$ |
| 3e | Rotate to all directions | $K_{\ell m}^{(d)} = \sqrt{\frac{4\pi}{2\ell+1}} \cdot K_{\ell 0} \cdot Y_{\ell m}(\theta_d, \phi_d)$ | $\mathbf{K}^{(\text{SH})}$, `AUX` | $\mathbf{KR} \in \mathbb{R}^{500 \times n_{\text{SH}} \cdot n_{\text{shells}}}$ |
| 3f | Resample to acquisition | $\mathbf{K}^{(\text{acq})}[d, :] = \mathbf{Y}_{\text{acq}} \cdot \mathbf{KR}[d, :]$ | $\mathbf{KR}$, $\mathbf{Y}_{\text{acq}}$ | `KERNELS['wm'][idx]` $\in \mathbb{R}^{500 \times n_S}$ |
| 3g | Compute normalization | $\text{norm} = \|\mathbf{K}^{(\text{acq})}[0, \text{dwi\_idx}]\|_2^{-1}$ | `KERNELS['wm'][idx]` | `KERNELS['norms'][:, idx]` |
| 4 | Compute ISO signal | $S_{\text{iso}} = \exp(-\mathbf{b} \cdot d_{\text{iso}})$ | $\mathbf{b}, d_{\text{iso}}$ | `KERNELS['iso']` $\in \mathbb{R}^{n_S}$ |

### 4.2 Voxelwise Inversion (`run_inversion`)

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 1 | Extract masked voxels | $\mathbf{Y} = \mathbf{D}[\mathbf{M}]$ | $\mathbf{D}, \mathbf{M}$ | $\mathbf{Y} \in \mathbb{R}^{n_{\text{vox}} \times n_S}$ |
| 2 | Build KD-tree | - | `GRAD_500` | `tree` |
| 3 | **For each voxel** $i = 1, \ldots, n_{\text{vox}}$: | | | |
| 3a | Normalize direction | $\hat{\mathbf{v}} = \mathbf{v}_i / (\|\mathbf{v}_i\|_2 + 10^{-16})$ | $\mathbf{V}_1[i]$ | $\hat{\mathbf{v}} \in \mathbb{R}^3$ |
| 3b | Find nearest sphere direction | $d^* = \arg\min_d \min(\|\hat{\mathbf{v}} - \mathbf{g}_d\|, \|\hat{\mathbf{v}} + \mathbf{g}_d\|)$ | $\hat{\mathbf{v}}$, `tree` | direction index $d^*$ |
| 3c | Extract dictionary columns | $\mathbf{A}_{\text{wm}} = \text{KERNELS['wm']}[:, d^*, :]^T$ | `KERNELS`, $d^*$ | $\mathbf{A}_{\text{wm}} \in \mathbb{R}^{n_S \times (n_{\text{atoms}}-1)}$ |
| 3d | Fit isotropic (NNLS) | $x_{\text{iso}} = \arg\min_{x \geq 0} \|\mathbf{y}_i - x \cdot \mathbf{a}_{\text{iso}}\|_2^2$ | $\mathbf{y}_i$, `KERNELS['iso']` | $x_{\text{iso}} \in \mathbb{R}_{\geq 0}$ |
| 3e | Compute residual | $\mathbf{y}_{\text{wm}} = \max(\mathbf{y}_i - x_{\text{iso}} \cdot \mathbf{a}_{\text{iso}}, 0)$ | $\mathbf{y}_i$, $x_{\text{iso}}$ | $\mathbf{y}_{\text{wm}} \in \mathbb{R}^{n_S}$ |
| 3f | Scale dictionary | $\tilde{\mathbf{A}}_{\text{wm}} = \mathbf{A}_{\text{wm}} \odot \mathbf{n}$ (column-wise) | $\mathbf{A}_{\text{wm}}$, `KERNELS['norms']` | $\tilde{\mathbf{A}}_{\text{wm}}$ |
| 3g | **LASSO solve** | $\tilde{\mathbf{x}}_{\text{wm}} = \arg\min_{\mathbf{x} \geq 0} \frac{1}{2}\|\mathbf{y}_{\text{wm}} - \tilde{\mathbf{A}}_{\text{wm}}\mathbf{x}\|_2^2 + \alpha\|\mathbf{x}\|_1$ | $\mathbf{y}_{\text{wm}}, \tilde{\mathbf{A}}_{\text{wm}}, \alpha = \lambda/n_S$ | $\tilde{\mathbf{x}}_{\text{wm}}$ |
| 3h | Unscale coefficients | $\mathbf{x}_{\text{wm}} = \tilde{\mathbf{x}}_{\text{wm}} \odot \mathbf{n}$ | $\tilde{\mathbf{x}}_{\text{wm}}$, norms | $\mathbf{x}_{\text{wm}}$ |
| 3i | Compute NDI | $\text{NDI}_i = \frac{\sum_k x_{\text{wm},k} \cdot v_{\text{ic},k}}{\sum_k x_{\text{wm},k} + \epsilon}$ | $\mathbf{x}_{\text{wm}}$, `KERNELS['icvf']` | $\text{NDI}_i \in [0,1]$ |
| 3j | Compute weighted $\kappa$ | $\bar{\kappa}_i = \frac{\sum_k x_{\text{wm},k} \cdot \kappa_k}{\sum_k x_{\text{wm},k} + \epsilon}$ | $\mathbf{x}_{\text{wm}}$, `KERNELS['kappa']` | $\bar{\kappa}_i$ |
| 3k | Compute ODI | $\text{ODI}_i = \frac{2}{\pi} \arctan\left(\frac{1}{\bar{\kappa}_i}\right)$ | $\bar{\kappa}_i$ | $\text{ODI}_i \in [0,1]$ |
| 3l | Compute FWF | $\text{FWF}_i = \frac{x_{\text{iso}}}{\sum_k x_{\text{wm},k} + x_{\text{iso}}}$ | $x_{\text{iso}}, \mathbf{x}_{\text{wm}}$ | $\text{FWF}_i \in [0,1]$ |
| 4 | Reconstruct maps | Place results back into 3D volume at mask locations | results, $\mathbf{M}$ | NDI, ODI, FWF maps |

---

## 5. Stopping Criteria

### 5.1 LASSO Solver (sklearn.linear_model.Lasso)
- **Maximum iterations**: 1000
- **Convergence tolerance**: Default (`tol=1e-4` on duality gap)
- **Positivity constraint**: Enforced (`positive=True`)

### 5.2 NNLS Solver (scipy.optimize.nnls)
- Built-in convergence based on KKT conditions

### 5.3 Overall Algorithm
- **Iteration over voxels**: Terminates when all $n_{\text{vox}}$ masked voxels are processed
- **No global iteration**: Single-pass dictionary fitting per voxel

---

## 6. Evaluation Metrics

### 6.1 Root Mean Square Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{M}|} \sum_{v \in \mathcal{M}} \left(\text{NDI}_v^{\text{est}} - \text{NDI}_v^{\text{GT}}\right)^2}$$

where $\mathcal{M}$ is the set of masked voxel indices.

### 6.2 Peak Signal-to-Noise Ratio (PSNR)

$$\text{PSNR} = 20 \cdot \log_{10}\left(\frac{1}{\text{RMSE}}\right) \text{ dB}$$

(assuming signal range $[0, 1]$)

### 6.3 Regional Statistics

$$\bar{\text{NDI}}_{\text{center}} = \frac{1}{|\mathcal{C}|} \sum_{v \in \mathcal{C}} \text{NDI}_v$$

$$\bar{\text{NDI}}_{\text{background}} = \frac{1}{|\mathcal{M} \setminus \mathcal{C}|} \sum_{v \in \mathcal{M} \setminus \mathcal{C}} \text{NDI}_v$$

where $\mathcal{C} = \{(x,y,z) : (x - N_x/2)^2 + (y - N_y/2)^2 < (N_x/3)^2\}$ defines the phantom center region.

---

## 7. Summary of Key Equations

| Component | Equation |
|-----------|----------|
| **b-value** | $b = (\gamma \delta G)^2 (\Delta - \delta/3)$ |
| **q-vector** | $q = \gamma \delta G$ |
| **Isotropic signal** | $S_{\text{iso}} = e^{-b \cdot d_{\text{iso}}}$ |
| **LASSO objective** | $\min_{\mathbf{x} \geq 0} \frac{1}{2}\|\mathbf{y} - \mathbf{A}\mathbf{x}\|_2^2 + \alpha\|\mathbf{x}\|_1$ |
| **NDI estimation** | $\text{NDI} = \frac{\mathbf{x}^T \mathbf{v}_{\text{ic}}}{\mathbf{1}^T \mathbf{x}}$ |
| **ODI estimation** | $\text{ODI} = \frac{2}{\pi}\arctan\left(\frac{\mathbf{1}^T \mathbf{x}}{\mathbf{x}^T \boldsymbol{\kappa}}\right)$ |
| **FWF estimation** | $\text{FWF} = \frac{x_{\text{iso}}}{\mathb