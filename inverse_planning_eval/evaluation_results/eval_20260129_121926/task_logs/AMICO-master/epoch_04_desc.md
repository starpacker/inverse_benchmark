## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: AMICO-NODDI (Accelerated Microstructure Imaging via Convex Optimization for NODDI)
- **algorithm_family**: Dictionary-based sparse reconstruction with LASSO regularization
- **summary**: Estimates neurite density (NDI), orientation dispersion (ODI), and free water fraction (FWF) from diffusion MRI by solving a non-negative LASSO problem using pre-computed biophysical model kernels rotated via spherical harmonics.

---

## 1. OBJECTIVE FUNCTION

### full_expression
For each voxel, the optimization problem is decomposed into two sequential subproblems:

**Subproblem 1 (Isotropic fitting)**:
$$\hat{x}_{\text{iso}} = \arg\min_{x_{\text{iso}} \geq 0} \left\| \mathbf{y} - x_{\text{iso}} \mathbf{a}_{\text{iso}} \right\|_2^2$$

**Subproblem 2 (White matter fitting with LASSO)**:
$$\hat{\mathbf{x}}_{\text{wm}} = \arg\min_{\mathbf{x}_{\text{wm}} \geq 0} \left\| \tilde{\mathbf{y}} - \tilde{\mathbf{A}}_{\text{wm}} \mathbf{x}_{\text{wm}} \right\|_2^2 + \lambda \|\mathbf{x}_{\text{wm}}\|_1$$

where $\tilde{\mathbf{y}} = \max(0, \mathbf{y} - \hat{x}_{\text{iso}} \mathbf{a}_{\text{iso}})$ and $\tilde{\mathbf{A}}_{\text{wm}} = \mathbf{A}_{\text{wm}} \odot \mathbf{N}$ (column-wise scaling by norms).

### data_fidelity_term
$$\mathcal{L}_{\text{data}} = \left\| \tilde{\mathbf{y}} - \tilde{\mathbf{A}}_{\text{wm}} \mathbf{x}_{\text{wm}} \right\|_2^2$$

This is an **L2 (squared Euclidean)** loss derived from the `Lasso` scikit-learn implementation.

### regularization_term
$$\mathcal{R}(\mathbf{x}_{\text{wm}}) = \lambda \|\mathbf{x}_{\text{wm}}\|_1 = \lambda \sum_{j=1}^{N_{\text{atoms}}} |x_{\text{wm},j}|$$

where $\lambda = \frac{\lambda_1}{n_S}$ with $\lambda_1 = 0.5$ and $n_S$ = number of measurements.

### constraints
$$\mathbf{x}_{\text{wm}} \geq 0 \quad \text{(enforced via } \texttt{positive=True} \text{ in LASSO)}$$
$$x_{\text{iso}} \geq 0 \quad \text{(enforced via NNLS)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}_{\text{wm}}$ | $(N_{\text{atoms}}-1,)$ | float64 | zeros (implicit in LASSO) | Sparse coefficients for white matter dictionary atoms |
| $x_{\text{iso}}$ | scalar | float64 | 0 (implicit in NNLS) | Coefficient for isotropic (CSF) compartment |

### 2.2 Dual Variables
None explicitly maintained (LASSO uses coordinate descent internally).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\tilde{\mathbf{y}}$ | $(n_S,)$ | float64 | Computed | Residual signal after isotropic subtraction: $\max(0, \mathbf{y} - \hat{x}_{\text{iso}} \mathbf{a}_{\text{iso}})$ |
| $\mathbf{x}_{\text{wm,scaled}}$ | $(N_{\text{atoms}}-1,)$ | float64 | Computed | Scaled WM coefficients from LASSO: $\mathbf{x}_{\text{wm,scaled}} = \texttt{lasso.coef\_}$ |
| `idx_dir` | scalar | int | Computed | Index of nearest direction in 500-direction lookup table |
| `results` | $(n_{\text{voxels}}, 3)$ | float64 | zeros | Accumulated [NDI, ODI, FWF] per voxel |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\gamma$ | $2.675987 \times 10^8$ rad/(s·T) | Gyromagnetic ratio for hydrogen |
| $d_{\parallel}$ | $1.7 \times 10^{-3}$ mm²/s | Parallel diffusivity for intracellular/extracellular compartments |
| $d_{\text{iso}}$ | $3.0 \times 10^{-3}$ mm²/s | Isotropic diffusivity (free water/CSF) |
| $\lambda_1$ | $5 \times 10^{-1}$ | LASSO regularization base parameter |
| $l_{\max}$ | 12 | Maximum spherical harmonic order |
| $n_{\text{dirs}}$ | 500 | Number of directions in high-resolution scheme |
| $N_{\text{IC\_VFs}}$ | 12 | Number of intracellular volume fraction samples |
| $N_{\text{IC\_ODs}}$ | 12 | Number of orientation dispersion samples |
| $N_{\text{atoms}}$ | 145 | Total dictionary atoms: $12 \times 12 + 1$ |
| IC_VFs | $[0.1, 0.181, ..., 0.99]$ | `np.linspace(0.1, 0.99, 12)` |
| IC_ODs | $[0.03, 0.06, 0.09, ..., 0.99]$ | `np.hstack([0.03, 0.06], np.linspace(0.09, 0.99, 10))` |
| `b0_thr` | 10 s/mm² | Threshold for identifying b=0 volumes |
| `max_iter` | 1000 | Maximum LASSO iterations |
| `_REQUIRED_PRECISION` | $10^{-7}$ | Numerical precision threshold |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(n_S,)$ | float64 | Single voxel DWI signal across all measurements |
| $\mathbf{Y}$ | $(n_x, n_y, n_z, n_S)$ | float64 | Full 4D DWI volume |
| $\mathbf{d}$ | $(3,)$ | float64 | Principal diffusion direction from DTI fit |
| mask | $(n_x, n_y, n_z)$ | bool | Brain mask |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Parameter Derivation
- **formula**: 
  - $\kappa_i = \frac{1}{\tan(\text{OD}_i \cdot \pi / 2)}$ for each orientation dispersion value
- **inputs**: IC_ODs array, shape $(12,)$
- **outputs**: IC_KAPPAs array, shape $(12,)$

### Step 1: Scheme Loading and Normalization
- **formula**: 
  - $\mathbf{g}_i \leftarrow \frac{\mathbf{g}_i}{\|\mathbf{g}_i\|_2 + 10^{-16}}$ for each gradient direction
  - $b = (\gamma \cdot \delta \cdot G)^2 \cdot (\Delta - \delta/3) \times 10^{-6}$ (if STEJSKALTANNER version)
- **inputs**: raw scheme file, shape $(n_S, 4)$ or $(n_S, 7)$
- **outputs**: Scheme object with normalized directions, b-values, shell structure

### Step 2: Shell Identification
- **formula**: 
  - $b_{\text{rounded}} = \text{round}(b, -2)$ (round to nearest 100)
  - Group indices by unique rounded b-values
- **inputs**: b-values array, shape $(n_S,)$
- **outputs**: `shells` list with indices, mean b-values, gradient directions per shell

### Step 3: DTI Fitting for Principal Directions
- **formula**: 
  - Fit tensor model: $\mathbf{D} = \text{argmin} \|\log(\mathbf{S}/S_0) + \mathbf{B}\text{vec}(\mathbf{D})\|_2^2$
  - $\mathbf{d} = \mathbf{v}_1$ (first eigenvector of $\mathbf{D}$)
- **inputs**: DWI data $(n_x, n_y, n_z, n_S)$, gradient table
- **outputs**: Principal directions `dirs`, shape $(n_x, n_y, n_z, 3)$

### Step 4: Spherical Harmonic Basis Computation
- **formula**: 
  - $Y_{lm}(\theta, \phi)$ = real spherical harmonics (Descoteaux basis) up to order $l_{\max}=12$
  - $n_{\text{SH}} = \frac{(l_{\max}+1)(l_{\max}+2)}{2} = 91$
- **inputs**: 500 gradient directions (GRAD_500), $l_{\max}=12$
- **outputs**: `Y_high` $(500, 91)$, `Y_inv` = $(Y^T Y)^{-1} Y^T$ shape $(91, 500)$

### Step 5: Rotation Matrix Precomputation
- **formula**: 
  - For each direction $i \in [0, 499]$: store $Y_{lm}(\theta_i, \phi_i)$ as rotation lookup
  - $c_l = \sqrt{\frac{4\pi}{2l+1}}$ for each even $l$
  - $\text{idx}_{m=0} = \frac{l^2 + l + 2}{2} - 1$
- **inputs**: GRAD_500, $l_{\max}$
- **outputs**: AUX dictionary with `fit`, `Ylm_rot`, `const`, `idx_m0`

### Step 6: High-Resolution Scheme Creation
- **formula**: 
  - Create $(500 \cdot n_{\text{shells}}, 7)$ array replicating GRAD_500 for each shell
  - Copy shell-specific timing parameters (G, Δ, δ, TE)
- **inputs**: Original scheme, GRAD_500
- **outputs**: `scheme_high` with 500 directions per shell

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

#### Step 1: Extract Masked Voxels
- **formula**: 
  - `mask_indices` = $\{(x,y,z) : \text{mask}[x,y,z] = \text{True}\}$
  - `y_data` = $\mathbf{Y}[\text{mask\_indices}]$, shape $(n_{\text{voxels}}, n_S)$
  - `d_dirs` = $\text{dirs}[\text{mask\_indices}]$, shape $(n_{\text{voxels}}, 3)$
- **inputs**: data $(n_x, n_y, n_z, n_S)$, mask $(n_x, n_y, n_z)$, dirs $(n_x, n_y, n_z, 3)$
- **outputs**: y_data, d_dirs, mask_indices, $n_{\text{voxels}}$

#### Step 2: Initialize KD-Tree for Direction Lookup
- **formula**: 
  - Build spatial index over GRAD_500 (500 unit vectors on sphere)
- **inputs**: GRAD_500, shape $(500, 3)$
- **outputs**: KDTree object `tree`

#### Step 3: Initialize Results Array
- **formula**: 
  - `results` $\leftarrow \mathbf{0}_{n_{\text{voxels}} \times 3}$
- **inputs**: $n_{\text{voxels}}$
- **outputs**: results array for [NDI, ODI, FWF]

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (over voxels $i = 0, 1, \ldots, n_{\text{voxels}}-1$)

#### Step 1: Extract Voxel Data [FORWARD]
- **formula**: 
  - $\mathbf{y} = \texttt{y\_data}[i, :]$
  - $\mathbf{d} = \texttt{d\_dirs}[i, :]$
- **inputs**: y_data, d_dirs, index $i$
- **outputs**: signal $\mathbf{y} \in \mathbb{R}^{n_S}$, direction $\mathbf{d} \in \mathbb{R}^3$

#### Step 2: Normalize Direction
- **formula**: 
  - $\hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2 + 10^{-16}}$
- **inputs**: $\mathbf{d}$
- **outputs**: $\hat{\mathbf{d}}$ (unit vector)

#### Step 3: Find Nearest Dictionary Direction
- **formula**: 
  - $(\text{dist}_+, \text{idx}_+) = \texttt{tree.query}(\hat{\mathbf{d}})$
  - $(\text{dist}_-, \text{idx}_-) = \texttt{tree.query}(-\hat{\mathbf{d}})$
  - $\text{idx\_dir} = \begin{cases} \text{idx}_- & \text{if } \text{dist}_- < \text{dist}_+ \\ \text{idx}_+ & \text{otherwise} \end{cases}$
- **inputs**: $\hat{\mathbf{d}}$, tree
- **outputs**: `idx_dir` (integer index into 500-direction table)
- **computational_note**: Accounts for antipodal symmetry of diffusion directions

#### Step 4: Extract Direction-Specific Kernels [FORWARD]
- **formula**: 
  - $\mathbf{A}_{\text{wm}} = \texttt{kernels['wm']}[:, \text{idx\_dir}, :]^T \in \mathbb{R}^{n_S \times (N_{\text{atoms}}-1)}$
  - $\mathbf{a}_{\text{iso}} = \texttt{kernels['iso']}[:, \text{None}] \in \mathbb{R}^{n_S \times 1}$
- **inputs**: kernels dictionary, idx_dir
- **outputs**: $\mathbf{A}_{\text{wm}}$, $\mathbf{a}_{\text{iso}}$

#### Step 5: Isotropic Coefficient Estimation (NNLS)
- **formula**: 
  - $\hat{x}_{\text{iso}} = \arg\min_{x \geq 0} \|\mathbf{y} - x \cdot \mathbf{a}_{\text{iso}}\|_2^2$
  - Solved via `scipy.optimize.nnls`
- **inputs**: $\mathbf{a}_{\text{iso}}$, $\mathbf{y}$
- **outputs**: $\hat{x}_{\text{iso}}$ (scalar, non-negative)

#### Step 6: Compute Residual Signal
- **formula**: 
  - $\tilde{\mathbf{y}} = \mathbf{y} - \hat{x}_{\text{iso}} \cdot \mathbf{a}_{\text{iso}}$
  - $\tilde{y}_j \leftarrow \max(0, \tilde{y}_j) \quad \forall j$
- **inputs**: $\mathbf{y}$, $\hat{x}_{\text{iso}}$, $\mathbf{a}_{\text{iso}}$
- **outputs**: $\tilde{\mathbf{y}} \in \mathbb{R}^{n_S}_{\geq 0}$

#### Step 7: Scale Dictionary by Precomputed Norms
- **formula**: 
  - $\mathbf{N} = \texttt{kernels['norms']}[0, :] \in \mathbb{R}^{N_{\text{atoms}}-1}$
  - $\tilde{\mathbf{A}}_{\text{wm}} = \mathbf{A}_{\text{wm}} \odot \mathbf{N}$ (column-wise multiplication)
- **inputs**: $\mathbf{A}_{\text{wm}}$, norms
- **outputs**: $\tilde{\mathbf{A}}_{\text{wm}} \in \mathbb{R}^{n_S \times (N_{\text{atoms}}-1)}$
- **computational_note**: Precomputed as $N_j = 1/\|\mathbf{a}_j[\text{dwi\_idx}]\|_2$

#### Step 8: LASSO Sparse Regression
- **formula**: 
  - $\alpha = \frac{\lambda_1}{n_S} = \frac{0.5}{n_S}$
  - $\hat{\mathbf{x}}_{\text{wm,scaled}} = \arg\min_{\mathbf{x} \geq 0} \frac{1}{2n_S}\|\tilde{\mathbf{y}} - \tilde{\mathbf{A}}_{\text{wm}}\mathbf{x}\|_2^2 + \alpha \|\mathbf{x}\|_1$
  - Solved via scikit-learn `Lasso(alpha=α, positive=True, fit_intercept=False, max_iter=1000)`
- **inputs**: $\tilde{\mathbf{A}}_{\text{wm}}$, $\tilde{\mathbf{y}}$, $\alpha$
- **outputs**: $\hat{\mathbf{x}}_{\text{wm,scaled}} \in \mathbb{R}^{N_{\text{atoms}}-1}_{\geq 0}$

#### Step 9: Rescale Coefficients
- **formula**: 
  - $\hat{\mathbf{x}}_{\text{wm}} = \hat{\mathbf{x}}_{\text{wm,scaled}} \odot \mathbf{N}$
- **inputs**: $\hat{\mathbf{x}}_{\text{wm,scaled}}$, $\mathbf{N}$
- **outputs**: $\hat{\mathbf{x}}_{\text{wm}} \in \mathbb{R}^{N_{\text{atoms}}-1}_{\geq 0}$

#### Step 10: Compute NDI (Neurite Density Index)
- **formula**: 
  - $S_{\text{wm}} = \sum_{j=1}^{N_{\text{atoms}}-1} \hat{x}_{\text{wm},j} + 10^{-16}$
  - $\text{NDI} = \frac{\sum_{j=1}^{N_{\text{atoms}}-1} \hat{x}_{\text{wm},j} \cdot \text{ICVF}_j}{S_{\text{wm}}}$
- **inputs**: $\hat{\mathbf{x}}_{\text{wm}}$, `kernels['icvf']`
- **outputs**: NDI $\in [0, 1]$

#### Step 11: Compute ODI (Orientation Dispersion Index)
- **formula**: 
  - $\bar{\kappa} = \frac{\sum_{j=1}^{N_{\text{atoms}}-1} \hat{x}_{\text{wm},j} \cdot \kappa_j}{S_{\text{wm}}}$
  - $\text{ODI} = \frac{2}{\pi} \arctan\left(\frac{1}{\bar{\kappa}}\right)$
- **inputs**: $\hat{\mathbf{x}}_{\text{wm}}$, `kernels['kappa']`, $S_{\text{wm}}$
- **outputs**: ODI $\in [0, 1]$

#### Step 12: Compute FWF (Free Water Fraction)
- **formula**: 
  - $S_{\text{total}} = S_{\text{wm}} + \hat{x}_{\text{iso}}$
  - $\text{FWF} = \frac{\hat{x}_{\text{iso}}}{S_{\text{total}}}$
- **inputs**: $\hat{x}_{\text{iso}}$, $S_{\text{wm}}$
- **outputs**: FWF $\in [0, 1]$

#### Step 13: Store Results
- **formula**: 
  - $\texttt{results}[i, :] = [\text{NDI}, \text{ODI}, \text{FWF}]$
- **inputs**: NDI, ODI, FWF, index $i$
- **outputs**: Updated results array

**END_LOOP**

---

### 4.3 FINALIZATION

#### Step 1: Reconstruct 3D Maps
- **formula**: 
  - Initialize: $\text{ndi\_map}, \text{odi\_map}, \text{fwf\_map} \leftarrow \mathbf{0}_{n_x \times n_y \times n_z}$
  - $\text{ndi\_map}[\text{mask\_indices}] = \texttt{results}[:, 0]$
  - $\text{odi\_map}[\text{mask\_indices}] = \texttt{results}[:, 1]$
  - $\text{fwf\_map}[\text{mask\_indices}] = \texttt{results}[:, 2]$
- **inputs**: results $(n_{\text{voxels}}, 3)$, mask_indices, original shape
- **outputs**: ndi_map, odi_map, fwf_map (each $(n_x, n_y, n_z)$)

#### Step 2: Save NIfTI Outputs
- **formula**: 
  - Write `NODDI_NDI.nii.gz`, `NODDI_ODI.nii.gz`, `NODDI_FWF.nii.gz`
- **inputs**: 3D maps, affine matrix
- **outputs**: NIfTI files on disk

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: 
  - Outer loop: $i < n_{\text{voxels}}$ (process all masked voxels exactly once)
  - Inner LASSO: convergence or `max_iter = 1000` iterations
- **parameters**: 
  - `max_iter = 1000` (LASSO)
  - No explicit convergence tolerance exposed (sklearn default: `tol=1e-4`)
- **check_frequency**: Per LASSO solve (inner), never terminates early (outer)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| RMSE | $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i \in \text{mask}} (\text{NDI}_i - \text{GT\_NDI}_i)^2}$ | Root mean squared error against ground truth NDI |
| PSNR | $\text{PSNR} = 20 \log_{10}\left(\frac{1}{\text{RMSE}}\right)$ dB | Peak signal-to-noise ratio (assumes max NDI = 1) |
| Mean NDI | $\bar{\text{NDI}} = \frac{1}{N}\sum_{i \in \text{mask}} \text{NDI}_i$ | Average neurite density in mask |
| Mean ODI | $\bar{\text{ODI}} = \frac{1}{N}\sum_{i \in \text{mask}} \text{ODI}_i$ | Average orientation dispersion in mask |
| Center NDI | $\bar{\text{NDI}}_{\text{center}} = \frac{1}{|C|}\sum_{i \in C} \text{NDI}_i$ where $C = \{(x,y): (x-S_x/2)^2 + (y-S_y/2)^2 < (S_x/3)^2\}$ | Mean NDI in circular phantom center |
| Background NDI | $\bar{\text{NDI}}_{\text{bg}} = \frac{1}{|\text{mask} \setminus C|}\sum_{i \in \text{mask} \setminus C} \text{NDI}_i$ | Mean NDI outside phantom center |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON extraction MUST include:

1. `algorithm_identification`: {algorithm_name, algorithm_family, summary}
2. `objective_function`: {full_expression, data_fidelity_term, regularization_term, constraints}
3. `variables`: {primal_variables, dual_variables, auxiliary_variables, constants, observations}
4. `data_preprocessing`: [{step_order, step_name, formula, inputs, outputs}]
5. `initialization`: [{step_name, formula, inputs, outputs}]
6. `iteration_loop`: {loop_type: "voxel_parallel", steps: [{step_name, formula, inputs, outputs, computational_note}]}
7. `finalization`: [{step_name, formula, inputs, outputs}]
8. `stopping_criterion`: {type, expression, parameters, check_frequency}
9. `evaluation`: [{name, formula, description}]
10. `kernel_generation`: {model: "NODDI", compartments: ["intracellular", "extracellular", "isotropic"], rotation_method: "spherical_harmonics", n_atoms: 145}
11. `biophysical_parameters`: {d_parallel, d_isotropic, IC_VFs, IC_ODs, IC_KAPPAs}
12. `output_maps`: ["NDI", "ODI", "FWF"]