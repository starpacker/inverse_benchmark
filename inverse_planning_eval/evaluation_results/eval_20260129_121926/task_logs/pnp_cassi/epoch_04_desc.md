## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: GAP-TV (Generalized Alternating Projection with Total Variation)
- **algorithm_family**: Proximal Splitting / Plug-and-Play Methods
- **summary**: Reconstructs hyperspectral datacubes from coded aperture snapshot spectral imaging (CASSI) measurements by alternating between a linear data-consistency projection and TV-based denoising via Chambolle's dual algorithm.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{X}} \frac{1}{2}\|\mathbf{y} - \mathcal{A}(\mathbf{X})\|_2^2 + \tau \sum_{c=1}^{n_C} \|\nabla \mathbf{X}_{:,:,c}\|_1$$

subject to $\mathbf{X} \in \mathbb{R}^{r \times c \times n_C}$

### data_fidelity_term
$$\frac{1}{2}\|\mathbf{y} - \mathcal{A}(\mathbf{X})\|_2^2 = \frac{1}{2}\sum_{i,j}\left(y_{i,j} - \sum_{c=1}^{n_C} \Phi_{i,j,c} X_{i,j,c}\right)^2$$

where $\mathcal{A}(\mathbf{X}) = \sum_{c=1}^{n_C} \mathbf{X}_{:,:,c} \odot \boldsymbol{\Phi}_{:,:,c}$ (element-wise product summed over spectral channels).

### regularization_term
**Isotropic Total Variation** (applied channel-wise after spatial shift-back):
$$\text{TV}(\mathbf{X}) = \tau \sum_{c=1}^{n_C} \sum_{i,j} \sqrt{(X_{i+1,j,c} - X_{i,j,c})^2 + (X_{i,j+1,c} - X_{i,j,c})^2}$$

where $\tau = \texttt{tv\_weight} = 6$.

### constraints
None (unconstrained optimization; implicit box constraint $[0,1]$ from data normalization but not enforced during iteration).

---

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}$ | $(r, c + (n_C-1) \cdot s, n_C)$ | float64 | $\mathbf{X}^{(0)} = \mathcal{A}^T(\mathbf{y})$ | Shifted spectral datacube in measurement domain |
| $\mathbf{X}_{\text{img}}$ | $(r, c, n_C)$ | float64 | Derived from $\mathbf{X}$ via shift_back | Spectral datacube in image domain (denoising target) |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}_1$ | $(r, c + (n_C-1) \cdot s)$ | float64 | $\mathbf{0}$ | Accumulated measurement residual (GAP dual) |

**[INNER: TV Denoiser Dual Variables]**
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{p}_1$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Horizontal dual variable for TV |
| $\mathbf{p}_2$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Vertical dual variable for TV |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}_b$ | $(r, c + (n_C-1) \cdot s)$ | float64 | Computed | Forward projection $\mathcal{A}(\mathbf{X})$ |
| $\boldsymbol{\Phi}_{\text{sum}}$ | $(r, c + (n_C-1) \cdot s)$ | float64 | Precomputed | Sum of mask: $\sum_c \Phi_{:,:,c}$, with zeros replaced by 1 |

**[INNER: TV Denoiser Auxiliary Variables]**
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{z}$ | $(r, c, n_C)$ | float64 | Computed | $\text{div}(\mathbf{p}) - \lambda_{\text{TV}} \mathbf{x}$ |
| $\mathbf{z}_1$ | $(r, c, n_C)$ | float64 | Computed | Horizontal gradient of $\mathbf{z}$ |
| $\mathbf{z}_2$ | $(r, c, n_C)$ | float64 | Computed | Vertical gradient of $\mathbf{z}$ |
| $\text{divp}$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Divergence of $(\mathbf{p}_1, \mathbf{p}_2)$ |
| $\text{denom}$ | $(r, c, n_C)$ | float64 | Computed | Normalization denominator (tiled from 2D) |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda$ | 1 | GAP step size parameter |
| $\tau$ (tv_weight) | 6 | TV regularization weight |
| $K_{\text{outer}}$ (iter_max) | 5 (main), 20 (default) | Maximum outer iterations |
| $K_{\text{TV}}$ (tv_iter_max) | 5 | Maximum TV denoiser iterations |
| $\delta t$ (dt) | 0.25 | Chambolle algorithm step size |
| $s$ (step) | 1 | Spectral dispersion step size |
| PIXEL_MAX | 1.0 | Maximum pixel value for PSNR |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ (meas) | $(r, c + (n_C-1) \cdot s)$ | float64 | 2D compressed measurement |
| $\boldsymbol{\Phi}$ (mask_3d) | $(r, c + (n_C-1) \cdot s, n_C)$ | float64 | 3D coded aperture sensing matrix |
| $\mathbf{X}^*$ (truth) | $(r, c, n_C)$ | float64 | Ground truth datacube (for evaluation only) |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Load Ground Truth | $\mathbf{X}^* \leftarrow \texttt{loadmat}(\texttt{matfile})[\texttt{'img'}]$ | matfile path | $\mathbf{X}^* \in \mathbb{R}^{r \times c \times n_C}$ |
| 1 | Normalize Truth | $\mathbf{X}^* \leftarrow \mathbf{X}^* / 255$ if $\max(\mathbf{X}^*) > 1$ | $\mathbf{X}^*$ | $\mathbf{X}^* \in [0,1]^{r \times c \times n_C}$ |
| 2 | Load Base Mask | $\mathbf{M}_{256} \leftarrow \texttt{loadmat}(\texttt{maskfile})[\texttt{'mask'}]$ | maskfile path | $\mathbf{M}_{256} \in \mathbb{R}^{r \times c}$ |
| 3 | Construct 3D Mask | $\Phi_{:,i:i+c,i} \leftarrow \mathbf{M}_{256}$ for $i = 0, \ldots, n_C-1$ | $\mathbf{M}_{256}$, shape $(r,c,n_C)$ | $\boldsymbol{\Phi} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| 4 | Shift Ground Truth | $X^*_{\text{shift}_{:,i \cdot s : i \cdot s + c, i}} \leftarrow X^*_{:,:,i}$ for $i = 0, \ldots, n_C-1$ | $\mathbf{X}^*$ | $\mathbf{X}^*_{\text{shift}} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| 5 | Simulate Measurement | $\mathbf{y} = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_{:,:,i} \odot \mathbf{X}^*_{\text{shift}_{:,:,i}}$ | $\boldsymbol{\Phi}$, $\mathbf{X}^*_{\text{shift}}$ | $\mathbf{y} \in \mathbb{R}^{r \times (c+(n_C-1)s)}$ |
| 6 | Compute Mask Sum | $\Phi_{\text{sum}} = \sum_{i=1}^{n_C} \Phi_{:,:,i}$; replace zeros with 1 | $\boldsymbol{\Phi}$ | $\boldsymbol{\Phi}_{\text{sum}} \in \mathbb{R}^{r \times (c+(n_C-1)s)}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_name | formula | inputs | outputs |
|-----------|---------|--------|---------|
| Initialize Primal | $\mathbf{X}^{(0)} = \mathcal{A}^T(\mathbf{y}) = \boldsymbol{\Phi} \odot \text{tile}(\mathbf{y}, n_C)$ | $\mathbf{y}$, $\boldsymbol{\Phi}$ | $\mathbf{X}^{(0)} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| Initialize Dual | $\mathbf{y}_1^{(0)} = \mathbf{0}$ | — | $\mathbf{y}_1^{(0)} \in \mathbb{R}^{r \times (c+(n_C-1)s)}$ |
| Initialize Metrics | $\texttt{psnr\_all} = []$ | — | Empty list |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, \ldots, K_{\text{outer}} - 1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Forward Projection [FORWARD] | $\mathbf{y}_b^{(k)} = \mathcal{A}(\mathbf{X}^{(k)}) = \sum_{i=1}^{n_C} \mathbf{X}^{(k)}_{:,:,i} \odot \boldsymbol{\Phi}_{:,:,i}$ | $\mathbf{X}^{(k)}$, $\boldsymbol{\Phi}$ | $\mathbf{y}_b^{(k)}$ | Sum over spectral axis |
| 2 | Dual Update | $\mathbf{y}_1^{(k+1)} = \mathbf{y}_1^{(k)} + (\mathbf{y} - \mathbf{y}_b^{(k)})$ | $\mathbf{y}_1^{(k)}$, $\mathbf{y}$, $\mathbf{y}_b^{(k)}$ | $\mathbf{y}_1^{(k+1)}$ | Accumulate measurement residual |
| 3 | Compute Scaled Residual | $\mathbf{r}^{(k)} = \frac{\mathbf{y}_1^{(k+1)} - \mathbf{y}_b^{(k)}}{\boldsymbol{\Phi}_{\text{sum}}}$ | $\mathbf{y}_1^{(k+1)}$, $\mathbf{y}_b^{(k)}$, $\boldsymbol{\Phi}_{\text{sum}}$ | $\mathbf{r}^{(k)}$ | Element-wise division |
| 4 | Backward Projection [ADJOINT] | $\mathcal{A}^T(\mathbf{r}^{(k)}) = \boldsymbol{\Phi} \odot \text{tile}(\mathbf{r}^{(k)}, n_C)$ | $\mathbf{r}^{(k)}$, $\boldsymbol{\Phi}$ | $\mathcal{A}^T(\mathbf{r}^{(k)})$ | Broadcast and multiply |
| 5 | Primal Update (GAP) | $\mathbf{X}^{(k+\frac{1}{2})} = \mathbf{X}^{(k)} + \lambda \cdot \mathcal{A}^T(\mathbf{r}^{(k)})$ | $\mathbf{X}^{(k)}$, $\lambda$, $\mathcal{A}^T(\mathbf{r}^{(k)})$ | $\mathbf{X}^{(k+\frac{1}{2})}$ | GAP data-consistency step |
| 6 | Shift Back | $\mathbf{X}_{\text{img}}^{(k+\frac{1}{2})} = \texttt{shift\_back}(\mathbf{X}^{(k+\frac{1}{2})}, s)$: roll channel $i$ by $-i \cdot s$, crop to $(r, c, n_C)$ | $\mathbf{X}^{(k+\frac{1}{2})}$, $s$ | $\mathbf{X}_{\text{img}}^{(k+\frac{1}{2})}$ | Transform to image domain |
| 7 | TV Denoising | $\mathbf{X}_{\text{img}}^{(k+1)} = \texttt{TV\_denoiser}(\mathbf{X}_{\text{img}}^{(k+\frac{1}{2})}, \tau, K_{\text{TV}})$ | $\mathbf{X}_{\text{img}}^{(k+\frac{1}{2})}$, $\tau$, $K_{\text{TV}}$ | $\mathbf{X}_{\text{img}}^{(k+1)}$ | See nested loop below |
| 8 | Shift Forward | $\mathbf{X}^{(k+1)} = \texttt{shift}(\mathbf{X}_{\text{img}}^{(k+1)}, s)$: place channel $i$ at columns $[i \cdot s, i \cdot s + c)$ | $\mathbf{X}_{\text{img}}^{(k+1)}$, $s$ | $\mathbf{X}^{(k+1)}$ | Transform to measurement domain |
| 9 | Compute PSNR | $\text{PSNR}^{(k)} = 20 \log_{10}\left(\frac{1.0}{\sqrt{\text{MSE}(\mathbf{X}^*, \mathbf{X}_{\text{img}}^{(k+1)})}}\right)$ | $\mathbf{X}^*$, $\mathbf{X}_{\text{img}}^{(k+1)}$ | $\text{PSNR}^{(k)}$ | Append to psnr_all |

**END_LOOP**

---

**[INNER: TV Denoiser - Chambolle's Algorithm]**

Given input $\mathbf{x} \in \mathbb{R}^{r \times c \times n_C}$, regularization parameter $\lambda_{\text{TV}}$, and $K_{\text{TV}}$ iterations.

**Problem Solved**:
$$\mathbf{u}^* = \arg\min_{\mathbf{u}} \frac{\lambda_{\text{TV}}}{2}\|\mathbf{u} - \mathbf{x}\|_2^2 + \|\nabla \mathbf{u}\|_{2,1}$$

**Closed-form via dual**: $\mathbf{u}^* = \mathbf{x} - \frac{1}{\lambda_{\text{TV}}} \text{div}(\mathbf{p}^*)$

**Index arrays** (Neumann boundary conditions):
- $\text{idx}[i] = \min(i+1, r-1)$ for $i = 0, \ldots, r-1$
- $\text{iux}[i] = \max(i-1, 0)$ for $i = 0, \ldots, r-1$  
- $\text{ir}[j] = \min(j+1, c-1)$ for $j = 0, \ldots, c-1$
- $\text{il}[j] = \max(j-1, 0)$ for $j = 0, \ldots, c-1$

**BEGIN_LOOP** (for $t = 0, \ldots, K_{\text{TV}} - 1$)

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Compute z | $\mathbf{z} = \text{divp}^{(t)} - \lambda_{\text{TV}} \mathbf{x}$ | $\text{divp}^{(t)}$, $\mathbf{x}$, $\lambda_{\text{TV}}$ | $\mathbf{z}$ |
| 2 | Horizontal Gradient | $z_1[i,j,c] = z[i, \text{ir}[j], c] - z[i,j,c]$ | $\mathbf{z}$, ir | $\mathbf{z}_1$ |
| 3 | Vertical Gradient | $z_2[i,j,c] = z[\text{idx}[i], j, c] - z[i,j,c]$ | $\mathbf{z}$, idx | $\mathbf{z}_2$ |
| 4 | Compute Denominator | $d_{i,j} = 1 + \delta t \sqrt{\sum_{c=1}^{n_C}(z_1[i,j,c]^2 + z_2[i,j,c]^2)}$; tile to 3D | $\mathbf{z}_1$, $\mathbf{z}_2$, $\delta t$ | $\mathbf{d} \in \mathbb{R}^{r \times c \times n_C}$ |
| 5 | Update p1 | $\mathbf{p}_1^{(t+1)} = \frac{\mathbf{p}_1^{(t)} + \delta t \cdot \mathbf{z}_1}{\mathbf{d}}$ | $\mathbf{p}_1^{(t)}$, $\mathbf{z}_1$, $\mathbf{d}$, $\delta t$ | $\mathbf{p}_1^{(t+1)}$ |
| 6 | Update p2 | $\mathbf{p}_2^{(t+1)} = \frac{\mathbf{p}_2^{(t)} + \delta t \cdot \mathbf{z}_2}{\mathbf{d}}$ | $\mathbf{p}_2^{(t)}$, $\mathbf{z}_2$, $\mathbf{d}$, $\delta t$ | $\mathbf{p}_2^{(t+1)}$ |
| 7 | Compute Divergence | $\text{divp}^{(t+1)}[i,j,c] = p_1^{(t+1)}[i,j,c] - p_1^{(t+1)}[i, \text{il}[j], c] + p_2^{(t+1)}[i,j,c] - p_2^{(t+1)}[\text{iux}[i], j, c]$ | $\mathbf{p}_1^{(t+1)}$, $\mathbf{p}_2^{(t+1)}$, il, iux | $\text{divp}^{(t+1)}$ |

**END_LOOP**

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 8 | Primal Recovery | $\mathbf{u} = \mathbf{x} - \frac{\text{divp}^{(K_{\text{TV}})}}{\lambda_{\text{TV}}}$ | $\mathbf{x}$, $\text{divp}^{(K_{\text{TV}})}$, $\lambda_{\text{TV}}$ | $\mathbf{u}$ (denoised output) |

---

### 4.3 FINALIZATION

| step_name | formula | inputs | outputs |
|-----------|---------|--------|---------|
| Extract Final Image | $\hat{\mathbf{X}} = \mathbf{X}_{\text{img}}^{(K_{\text{outer}})}$ | Final iterate | $\hat{\mathbf{X}} \in \mathbb{R}^{r \times c \times n_C}$ |
| Compute Final PSNR | $\text{PSNR}_{\text{final}} = 20 \log_{10}\left(\frac{1.0}{\sqrt{\frac{1}{r \cdot c \cdot n_C}\sum_{i,j,c}(\hat{X}_{i,j,c} - X^*_{i,j,c})^2}}\right)$ | $\hat{\mathbf{X}}$, $\mathbf{X}^*$ | Scalar |
| Save Reconstruction | Write $\hat{\mathbf{X}}$ to `recon_result.mat` | $\hat{\mathbf{X}}$ | .mat file |
| Save Channel Grid | Plot first 9 channels as $3 \times 3$ grayscale image grid | $\hat{\mathbf{X}}$ | `recon_channels.png` |
| Save PSNR Curve | Plot $\texttt{psnr\_all}$ vs iteration index | psnr_all | `psnr_curve.png` |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k < K_{\text{outer}}$ where $K_{\text{outer}} = \texttt{iter\_max}$
- **parameters**: 
  - $K_{\text{outer}} = 5$ (as configured in main)
  - $K_{\text{TV}} = 5$ (inner TV iterations)
- **check_frequency**: Evaluated at the end of each outer iteration

---

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| PSNR | $\text{PSNR}(\mathbf{X}^*, \hat{\mathbf{X}}) = 20 \log_{10}\left(\frac{\text{PIXEL\_MAX}}{\sqrt{\text{MSE}}}\right)$ where $\text{MSE} = \frac{1}{N}\sum_{i,j,c}(X^*_{i,j,c} - \hat{X}_{i,j,c})^2$, $\text{PIXEL\_MAX} = 1.0$. Returns 100 if MSE = 0. | Peak Signal-to-Noise Ratio in dB |
| Convergence Curve | Sequence $\{\text{PSNR}^{(k)}\}_{k=0}^{K_{\text{outer}}-1}$ | PSNR at each iteration for monitoring convergence |

---

## 7. REQUIRED OUTPUT FIELDS

```
algorithm_identification.algorithm_name
algorithm_identification.algorithm_family
objective_function.full_expression
objective_function.data_fidelity_term
objective_function.regularization_term
variables.primal_variables
variables.dual_variables
variables.auxiliary_variables
variables.constants
variables.observations
data_preprocessing.steps[]
initialization.primal_init
initialization.dual_init
iteration_loop.outer_loop.steps[]
iteration_loop.inner_loop_tv.steps[]
finalization.output_extraction
stopping_criterion.type
stopping_criterion.max_iterations_outer
stopping_criterion.max_iterations_inner_tv
evaluation.psnr.formula
evaluation