# Algorithmic Specification Document: NODDI-AMICO Microstructure Imaging

## 1. OBJECTIVE FUNCTION

### full_expression
The algorithm solves a sequential convex optimization problem for each voxel:

**Stage 1 (Isotropic Component):**
$$\min_{x_{\text{iso}}} \| \mathbf{y} - \mathbf{A}_{\text{iso}} x_{\text{iso}} \|_2^2 \quad \text{s.t.} \quad x_{\text{iso}} \geq 0$$

**Stage 2 (White Matter Components):**
$$\min_{\mathbf{x}_{\text{wm}}} \frac{1}{2} \| \mathbf{y}_{\text{wm}} - \tilde{\mathbf{A}}_{\text{wm}} \tilde{\mathbf{x}}_{\text{wm}} \|_2^2 + \alpha \| \tilde{\mathbf{x}}_{\text{wm}} \|_1 \quad \text{s.t.} \quad \tilde{\mathbf{x}}_{\text{wm}} \geq 0$$

### data_fidelity_term
$$\mathcal{L}_{\text{data}} = \frac{1}{2} \| \mathbf{y}_{\text{wm}} - \tilde{\mathbf{A}}_{\text{wm}} \tilde{\mathbf{x}}_{\text{wm}} \|_2^2$$

where:
$$\mathbf{y}_{\text{wm}} = \max(\mathbf{y} - x_{\text{iso}}^* \mathbf{a}_{\text{iso}}, 0)$$

### regularization_term
$$\mathcal{R}(\tilde{\mathbf{x}}_{\text{wm}}) = \alpha \| \tilde{\mathbf{x}}_{\text{wm}} \|_1 = \alpha \sum_{j=1}^{N_{\text{atoms}}} |\tilde{x}_{\text{wm},j}|$$

where:
$$\alpha = \frac{\lambda_1}{N_{\text{meas}}} = \frac{0.5}{N_{\text{meas}}}$$

### constraints
| Constraint | Expression |
|------------|------------|
| Isotropic non-negativity | $x_{\text{iso}} \geq 0$ |
| WM coefficients non-negativity | $\tilde{x}_{\text{wm},j} \geq 0 \quad \forall j \in \{1, \ldots, N_{\text{atoms}}\}$ |
| Signal residual non-negativity | $y_{\text{wm},i} = \max(y_i - x_{\text{iso}} a_{\text{iso},i}, 0)$ |

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $x_{\text{iso}}$ | primal | $(1,)$ | float64 | NNLS solution | Isotropic (CSF) volume coefficient |
| $\tilde{\mathbf{x}}_{\text{wm}}$ | primal | $(N_{\text{atoms}},)$ | float64 | zeros | Scaled white matter dictionary coefficients |
| $\mathbf{x}_{\text{wm}}$ | primal | $(N_{\text{atoms}},)$ | float64 | derived | Unscaled WM coefficients: $x_{\text{wm},j} = \tilde{x}_{\text{wm},j} \cdot n_j$ |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}$ | observations | $(N_{\text{meas}},)$ | float32 | loaded | Raw DWI signal for single voxel |
| $\mathbf{y}_{\text{wm}}$ | observations | $(N_{\text{meas}},)$ | float32 | computed | Residual signal after isotropic removal |

### Constants (Dictionary/Kernels)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{A}_{\text{wm}}$ | constants | $(N_{\text{meas}}, N_{\text{atoms}})$ | float32 | forward model | White matter kernel dictionary |
| $\tilde{\mathbf{A}}_{\text{wm}}$ | constants | $(N_{\text{meas}}, N_{\text{atoms}})$ | float32 | $A_{\text{wm},ij} \cdot n_j$ | Column-normalized WM dictionary |
| $\mathbf{a}_{\text{iso}}$ | constants | $(N_{\text{meas}},)$ | float32 | forward model | Isotropic signal kernel |
| $\mathbf{n}$ | constants | $(N_{\text{atoms}},)$ | float32 | computed | Column normalization factors: $n_j = \|\mathbf{a}_{\text{wm},j}[\text{dwi\_idx}]\|_2^{-1}$ |
| $\boldsymbol{\nu}_{\text{ic}}$ | constants | $(N_{\text{atoms}},)$ | float32 | linspace | Intracellular volume fractions per atom |
| $\boldsymbol{\kappa}$ | constants | $(N_{\text{atoms}},)$ | float32 | computed | Watson concentration parameters per atom |

### Algorithm Parameters
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\lambda_1$ | constants | scalar | float64 | $5 \times 10^{-1}$ | LASSO regularization weight |
| $d_{\parallel}$ | constants | scalar | float64 | $1.7 \times 10^{-3}$ mm²/s | Parallel diffusivity |
| $d_{\text{iso}}$ | constants | scalar | float64 | $3.0 \times 10^{-3}$ mm²/s | Isotropic (CSF) diffusivity |
| $\ell_{\max}$ | constants | scalar | int | 12 | Maximum spherical harmonic order |
| $N_{\text{dirs}}$ | constants | scalar | int | 500 | Number of directions on sphere |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Load DWI Data
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | load_dwi_data |
| formula | $\mathbf{D} \leftarrow \text{NIfTI\_load}(\texttt{dwi\_file})$ |
| inputs | dwi_file: string path |
| outputs | $\mathbf{D}$: $(N_x, N_y, N_z, N_{\text{meas}})$, float32 |

### Step 3.2: Load Binary Mask
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | load_mask |
| formula | $\mathbf{M} \leftarrow \mathbb{1}[\text{NIfTI\_load}(\texttt{mask\_file}) > 0]$ |
| inputs | mask_file: string path |
| outputs | $\mathbf{M}$: $(N_x, N_y, N_z)$, bool |

### Step 3.3: Parse Acquisition Scheme
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | parse_scheme |
| formula | For STEJSKALTANNER format: $b_i = (\gamma \delta_i G_i)^2 (\Delta_i - \delta_i/3) \times 10^{-6}$ where $\gamma = 2.675987 \times 10^8$ rad/(s·T) |
| inputs | scheme_file: $(N_{\text{meas}}, 7)$ with columns $[\mathbf{g}, G, \Delta, \delta, \text{TE}]$ |
| outputs | $\mathbf{b}$: $(N_{\text{meas}},)$ s/mm², $\mathbf{g}$: $(N_{\text{meas}}, 3)$ unit vectors |

### Step 3.4: Gradient Direction Normalization
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | normalize_gradients |
| formula | $\hat{\mathbf{g}}_i = \frac{\mathbf{g}_i}{\|\mathbf{g}_i\|_2 + 10^{-16}}$ |
| inputs | $\mathbf{g}$: $(N_{\text{meas}}, 3)$ |
| outputs | $\hat{\mathbf{g}}$: $(N_{\text{meas}}, 3)$ unit vectors |

### Step 3.5: Shell Identification
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | identify_shells |
| formula | $\tilde{b}_i = \text{round}(b_i, -2)$; shells $\mathcal{S}_k = \{i : \tilde{b}_i = B_k, b_i > b_{\text{thr}}\}$ where $b_{\text{thr}} = 10$ s/mm² |
| inputs | $\mathbf{b}$: $(N_{\text{meas}},)$ |
| outputs | $\{\mathcal{S}_k\}_{k=1}^{N_{\text{shells}}}$: index sets, $\bar{b}_k = \text{mean}(\{b_i : i \in \mathcal{S}_k\})$ |

### Step 3.6: DTI Fit for Principal Directions
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | dti_fit |
| formula | Solve $\ln(S_i/S_0) = -b_i \hat{\mathbf{g}}_i^T \mathbf{D} \hat{\mathbf{g}}_i$ for diffusion tensor $\mathbf{D}$; extract principal eigenvector $\mathbf{e}_1 = \text{eigvec}(\mathbf{D}, \lambda_{\max})$ |
| inputs | $\mathbf{D}$: $(N_x, N_y, N_z, N_{\text{meas}})$, scheme |
| outputs | $\mathbf{V}$: $(N_x, N_y, N_z, 3)$ principal diffusion directions |

### Step 3.7: Precompute Spherical Harmonic Basis
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | precompute_sh_basis |
| formula | $Y_{\ell m}(\theta, \phi) = \sqrt{\frac{(2\ell+1)(\ell-m)!}{4\pi(\ell+m)!}} P_\ell^m(\cos\theta) \cdot \begin{cases} \sqrt{2}\cos(m\phi) & m > 0 \\ 1 & m = 0 \\ \sqrt{2}\sin(|m|\phi) & m < 0 \end{cases}$ |
| inputs | GRAD_500: $(500, 3)$ Cartesian coordinates |
| outputs | $\mathbf{Y}$: $(500, N_{\text{SH}})$ where $N_{\text{SH}} = (\ell_{\max}+1)(\ell_{\max}+2)/2 = 91$ |

### Step 3.8: Compute SH Fitting Matrix
| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | compute_sh_fit_matrix |
| formula | $\mathbf{Y}^{\dagger} = (\mathbf{Y}^T \mathbf{Y})^{-1} \mathbf{Y}^T$ |
| inputs | $\mathbf{Y}$: $(500, N_{\text{SH}})$ |
| outputs | $\mathbf{Y}^{\dagger}$: $(N_{\text{SH}}, 500)$ pseudoinverse |

### Step 3.9: Precompute Rotation Constants
| Field | Value |
|-------|-------|
| step_order | 9 |
| step_name | precompute_rotation_constants |
| formula | For each $(\ell, m)$: $c_{\ell m} = \sqrt{\frac{4\pi}{2\ell+1}}$, $\text{idx}_{m=0}(\ell) = \frac{\ell^2 + \ell + 2}{2} - 1$ |
| inputs | $\ell_{\max} = 12$ |
| outputs | $\mathbf{c}$: $(N_{\text{SH}},)$, $\text{idx}_{m=0}$: $(N_{\text{SH}},)$ |

---

## 4. ALGORITHM STEPS

### INITIALIZATION

#### Step 4.1: Generate High-Resolution Scheme
| Field | Value |
|-------|-------|
| step_name | create_hr_scheme |
| formula | For each shell $k$: replicate parameters across 500 directions. $\mathbf{G}_{\text{HR}} = [\text{GRAD\_500}; \ldots]_{N_{\text{shells}} \times 500}$ |
| inputs | scheme with $N_{\text{shells}}$ shells |
| outputs | scheme_HR with $500 \cdot N_{\text{shells}}$ measurements |

#### Step 4.2: Define NODDI Parameter Grid
| Field | Value |
|-------|-------|
| step_name | define_parameter_grid |
| formula | $\nu_{\text{ic}} \in \{0.1, 0.181, \ldots, 0.99\}$ (12 values, linspace); OD $\in \{0.03, 0.06, 0.09, \ldots, 0.99\}$ (12 values); $\kappa = \cot(\text{OD} \cdot \pi/2)$ |
| inputs | None |
| outputs | $\boldsymbol{\nu}_{\text{ic}}$: $(12,)$, $\boldsymbol{\kappa}$: $(12,)$, $N_{\text{atoms}} = 144$ |

#### Step 4.3: Generate Intracellular Signal Kernel
| Field | Value |
|-------|-------|
| step_name | generate_ic_signal |
| formula | Watson-distributed cylinders with Neuman PGSE: $E_{\text{IC}}(\mathbf{q}) = \frac{1}{2} e^{L_{\perp}} \sum_{n=0}^{6} \mathcal{I}_n(L_{\perp} - L_{\parallel}) \cdot c_n(\kappa) \cdot \sqrt{\frac{n+0.25}{\pi}} P_{2n}^0(\cos\theta_{\mathbf{q}})$ |
| inputs | $d_{\parallel}$, $\kappa$, gradient parameters |
| outputs | $\mathbf{s}_{\text{IC}}$: $(500 \cdot N_{\text{shells}},)$ |

**Sub-formula 4.3a: Parallel Attenuation**
$$L_{\parallel} = -q^2 \tau d_{\parallel}, \quad q = \gamma \delta G, \quad \tau = \Delta - \delta/3$$

**Sub-formula 4.3b: Perpendicular Attenuation**
$$L_{\perp} = 0 \quad \text{(GPD approximation for restricted perpendicular diffusion)}$$

**Sub-formula 4.3c: Legendre-Gaussian Integrals**
For $x = L_{\perp} - L_{\parallel} > 0.05$:
$$\mathcal{I}_0(x) = \frac{\sqrt{\pi} \text{erf}(\sqrt{x})}{\sqrt{x}}, \quad \mathcal{I}_n(x) = \frac{-e^{-x} + (n-0.5)\mathcal{I}_{n-1}(x)}{x}$$

For $x \leq 0.05$ (Taylor expansion):
$$\mathcal{I}_0(x) \approx 2 - \frac{2x}{3} + \frac{x^2}{5} - \frac{x^3}{21} + \frac{x^4}{108}$$

**Sub-formula 4.3d: Watson SH Coefficients**
$$c_0 = 2\sqrt{\pi}, \quad c_1 = \sqrt{5} \cdot \frac{3\sqrt{\kappa} - (3+2\kappa)D(\sqrt{\kappa})}{\kappa} \cdot \frac{e^{\kappa}}{\text{erfi}(\sqrt{\kappa})}$$
where Dawson function: $D(z) = \frac{\sqrt{\pi}}{2} \cdot \frac{\text{erfi}(z)}{e^{z^2}}$

#### Step 4.4: Generate Extracellular Signal Kernel
| Field | Value |
|-------|-------|
| step_name | generate_ec_signal |
| formula | Anisotropic Gaussian with Watson-averaged diffusivities: $E_{\text{EC}}(\mathbf{q}) = \exp\left(-b[(d_{\parallel}^w - d_{\perp}^w)\cos^2\theta_{\mathbf{q}} + d_{\perp}^w]\right)$ |
| inputs | $d_{\parallel}$, $\kappa$, $\nu_{\text{ic}}$, gradient parameters |
| outputs | $\mathbf{s}_{\text{EC}}$: $(500 \cdot N_{\text{shells}},)$ |

**Sub-formula 4.4a: Tortuosity-Constrained Perpendicular Diffusivity**
$$d_{\perp} = d_{\parallel}(1 - \nu_{\text{ic}})$$

**Sub-formula 4.4b: Watson-Averaged Diffusivities**
$$d_{\parallel}^w = \frac{-(d_{\parallel}-d_{\perp}) + 2d_{\perp}\kappa + (d_{\parallel}-d_{\perp})\frac{\sqrt{\kappa}}{D(\sqrt{\kappa})}}{2\kappa}$$
$$d_{\perp}^w = \frac{(d_{\parallel}-d_{\perp}) + 2(d_{\parallel}+d_{\perp})\kappa - (d_{\parallel}-d_{\perp})\frac{\sqrt{\kappa}}{D(\sqrt{\kappa})}}{4\kappa}$$

#### Step 4.5: Generate Isotropic Signal Kernel
| Field | Value |
|-------|-------|
| step_name | generate_iso_signal |
| formula | $E_{\text{ISO}}(\mathbf{q}) = \exp(-b \cdot d_{\text{iso}})$ where $b = q^2 \tau$ |
| inputs | $d_{\text{iso}} = 3.0 \times 10^{-3}$ mm²/s |
| outputs | $\mathbf{s}_{\text{ISO}}$: $(500 \cdot N_{\text{shells}},)$ |

#### Step 4.6: Combine IC and EC for Each Atom
| Field | Value |
|-------|-------|
| step_name | combine_compartments |
| formula | $\mathbf{s}_j = \nu_{\text{ic},j} \mathbf{s}_{\text{IC},j} + (1 - \nu_{\text{ic},j}) \mathbf{s}_{\text{EC},j}$ |
| inputs | $\mathbf{s}_{\text{IC},j}$, $\mathbf{s}_{\text{EC},j}$, $\nu_{\text{ic},j}$ |
| outputs | $\mathbf{s}_j$: $(500 \cdot N_{\text{shells}},)$ combined signal for atom $j$ |

#### Step 4.7: Rotate Kernel to SH Representation
| Field | Value |
|-------|-------|
| step_name | rotate_kernel_to_sh |
| formula | For each shell $k$: $\mathbf{f}_{k} = \mathbf{Y}^{\dagger} \mathbf{s}_{j,k}$. For each rotation direction $i$: $\mathbf{K}_{ij,k} = c_{\ell m} \cdot f_{k,\text{idx}_{m=0}} \cdot Y_{\ell m}(\theta_i, \phi_i)$ |
| inputs | $\mathbf{s}_j$: $(500 \cdot N_{\text{shells}},)$, AUX structures |
| outputs | $\mathbf{K}_j^{\text{rot}}$: $(500, N_{\text{SH}} \cdot N_{\text{shells}})$ rotated SH coefficients |

#### Step 4.8: Resample Kernel to Acquisition Directions
| Field | Value |
|-------|-------|
| step_name | resample_kernel |
| formula | For each rotation $i$: $K_{ij}[\text{dwi\_idx}] = \mathbf{Y}_{\text{out}} \mathbf{K}_{ij}^{\text{rot}}$ where $\mathbf{Y}_{\text{out}}$ is SH basis at acquisition directions |
| inputs | $\mathbf{K}_j^{\text{rot}}$: $(500, N_{\text{SH}} \cdot N_{\text{shells}})$ |
| outputs | $\mathbf{K}_j$: $(500, N_{\text{meas}})$ resampled kernel |

#### Step 4.9: Compute Normalization Factors
| Field | Value |
|-------|-------|
| step_name | compute_norms |
| formula | $n_j = \frac{1}{\|\mathbf{K}_{j}[0, \text{dwi\_idx}]\|_2}$ (norm of first rotation, DWI indices only) |
| inputs | $\mathbf{K}_j$: $(500, N_{\text{meas}})$ |
| outputs | $\mathbf{n}$: $(N_{\text{atoms}},)$ |

#### Step 4.10: Build KD-Tree for Direction Lookup
| Field | Value |
|-------|-------|
| step_name | build_direction_tree |
| formula | $\mathcal{T} = \text{KDTree}(\text{GRAD\_500})$ |
| inputs | GRAD_500: $(500, 3)$ |
| outputs | $\mathcal{T}$: spatial index structure |

---

### ITERATION LOOP

**BEGIN ITERATION** (for each voxel $v \in \{1, \ldots, N_{\text{voxels}}\}$ where $\mathbf{M}_v = \text{True}$)

#### Step 4.11: Extract Voxel Data
| Field | Value |
|-------|-------|
| step_name | extract_voxel |
| formula | $\mathbf{y} = \mathbf{D}[\mathbf{idx}_v, :]$, $\mathbf{d} = \mathbf{V}[\mathbf{idx}_v, :]$ |
| inputs | $\mathbf{D}$, $\mathbf{V}$, voxel index $v$ |
| outputs | $\mathbf{y}$: $(N_{\text{meas}},)$, $\mathbf{d}$: $(3,)$ |

#### Step 4.12: Normalize Direction Vector
| Field | Value |
|-------|-------|
| step_name | normalize_direction |
| formula | $\hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2 + 10^{-16}}$ |
| inputs | $\mathbf{d}$: $(3,)$ |
| outputs | $\hat{\mathbf{d}}$: $(3,)$ unit vector |

#### Step 4.13: Find Closest Dictionary Direction
| Field | Value |
|-------|-------|
| step_name | find_closest_direction |
| formula | $i^* = \arg\min_{i} \min(\|\hat{\mathbf{d}} - \mathbf{g}_i\|_2, \|\hat{\mathbf{d}} + \mathbf{g}_i\|_2)$ (antipodal symmetry) |
| inputs | $\hat{\mathbf{d}}$, $\mathcal{T}$ |
| outputs | $i^*$: rotation index |

#### Step 4.14: Select Rotated Dictionary
| Field | Value |
|-------|-------|
| step_name | select_dictionary |
| formula | $\mathbf{A}_{\text{wm}} = \mathbf{K}_{\text{wm}}[:, i^*, :]^T$, $\mathbf{a}_{\text{iso}} = \mathbf{K}_{\text{iso}}$ |
| inputs | $\mathbf{K}_{\text{wm}}$: $(N_{\text{atoms}}, 500, N_{\text{meas}})$, $i^*$ |
| outputs | $\mathbf{A}_{\text{wm}}$: $(N_{\text{meas}}, N_{\text{atoms}})$, $\mathbf{a}_{\text{iso}}$: $(N_{\text{meas}},)$ |

#### Step 4.15: Solve NNLS for Isotropic Component
| Field | Value |
|-------|-------|
| step_name | nnls_isotropic |
| formula | $x_{\text{iso}}^* = \arg\min_{x \geq 0} \|\mathbf{y} - \mathbf{a}_{\text{iso}} x\|_2^2$ (closed-form NNLS) |
| inputs | $\mathbf{y}$, $\mathbf{a}_{\text{iso}}$ |
| outputs | $x_{\text{iso}}^*$: scalar |
| computational_note | Uses scipy.optimize.nnls |

#### Step 4.16: Compute White Matter Residual
| Field | Value |
|-------|-------|
| step_name | compute_wm_residual |
| formula | $y_{\text{wm},i} = \max(y_i - x_{\text{iso}}^* a_{\text{iso},i}, 0)$ |
| inputs | $\mathbf{y}$, $x_{\text{iso}}^*$, $\mathbf{a}_{\text{iso}}$ |
| outputs | $\mathbf{y}_{\text{wm}}$: $(N_{\text{meas}},)$ |

#### Step 4.17: Scale Dictionary Columns
| Field | Value |
|-------|-------|
| step_name | scale_dictionary |
| formula | $\tilde{A}_{\text{wm},ij} = A_{\text{wm},ij} \cdot n_j$ |
| inputs | $\mathbf{A}_{\text{wm}}$: $(N_{\text{meas}}, N_{\text{atoms}})$, $\mathbf{n}$: $(N_{\text{atoms}},)$ |
| outputs | $\tilde{\mathbf{A}}_{\text{wm}}$: $(N_{\text{meas}}, N_{\text{atoms}})$ |

#### Step 4.18: Solve LASSO for WM Coefficients
| Field | Value |
|-------|-------|
| step_name | lasso_wm |
| formula | $\tilde{\mathbf{x}}_{\text{wm}}^* = \arg\min_{\tilde{\mathbf{x}} \geq 0} \frac{1}{2N_{\text{meas}}} \|\mathbf{y}_{\text{wm}} - \tilde{\mathbf{A}}_{\text{wm}} \tilde{\mathbf{x}}\|_2^2 + \alpha \|\tilde{\mathbf{x}}\|_1$ where $\alpha = \lambda_1 / N_{\text{meas}}$ |
| inputs | $\mathbf{y}_{\text{wm}}$, $\tilde{\mathbf{A}}_{\text{wm}}$, $\alpha$ |
| outputs | $\tilde{\mathbf{x}}_{\text{wm}}^*$: $(N_{\text{atoms}},)$ |
| computational_note | Uses sklearn.linear_model.Lasso with positive=True, max