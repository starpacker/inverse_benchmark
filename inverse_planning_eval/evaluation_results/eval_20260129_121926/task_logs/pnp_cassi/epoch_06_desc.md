## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "GAP-TV (Generalized Alternating Projection with Total Variation)",
  "algorithm_family": "Proximal Splitting / ADMM-like Methods",
  "summary": "Reconstructs a 3D spectral datacube from a single 2D coded aperture snapshot measurement using alternating data-consistency projection and TV denoising."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{X}} \frac{1}{2}\|\mathbf{A}(\mathbf{X}) - \mathbf{Y}\|_2^2 + \tau \cdot \text{TV}(\mathbf{X}_{\text{img}})$$

subject to $\mathbf{X}_{\text{img}} = \mathcal{S}^{-1}(\mathbf{X})$ (shift-back constraint)

**data_fidelity_term**:
$$\frac{1}{2}\|\mathbf{A}(\mathbf{X}) - \mathbf{Y}\|_2^2 = \frac{1}{2}\left\|\sum_{c=1}^{n_C} \mathbf{X}_{:,:,c} \odot \boldsymbol{\Phi}_{:,:,c} - \mathbf{Y}\right\|_2^2$$

where $\odot$ denotes element-wise multiplication.

**regularization_term**:
$$\text{TV}(\mathbf{X}_{\text{img}}) = \sum_{i,j,c} \sqrt{(\mathbf{X}_{\text{img}}[i+1,j,c] - \mathbf{X}_{\text{img}}[i,j,c])^2 + (\mathbf{X}_{\text{img}}[i,j+1,c] - \mathbf{X}_{\text{img}}[i,j,c])^2}$$

(Isotropic Total Variation with channel-wise gradient magnitude summed across spatial dimensions)

**constraints**:
- Implicit non-negativity from physical imaging model
- No explicit box constraints enforced in code

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}$ | $(r, c + s(n_C-1), n_C)$ | float64 | $\mathbf{A}^T(\mathbf{Y})$ | Shifted spectral datacube in measurement domain |
| $\mathbf{X}_{\text{img}}$ | $(r, c, n_C)$ | float64 | Derived from $\mathbf{X}$ | Spectral datacube in image domain |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}_1$ | $(r, c + s(n_C-1))$ | float64 | $\mathbf{0}$ | Accumulated measurement residual (dual for data constraint) |
| $\mathbf{p}_1$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Dual variable for horizontal TV gradient |
| $\mathbf{p}_2$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Dual variable for vertical TV gradient |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}_b$ | $(r, c + s(n_C-1))$ | float64 | Computed | Forward projection of current estimate |
| $\boldsymbol{\Phi}_{\text{sum}}$ | $(r, c + s(n_C-1))$ | float64 | $\sum_{c=1}^{n_C}\boldsymbol{\Phi}_{:,:,c}$ | Sum of mask across channels (0 replaced by 1) |
| $\mathbf{z}$ | $(r, c, n_C)$ | float64 | Computed | Gradient descent variable in TV denoiser |
| $\mathbf{z}_1$ | $(r, c, n_C)$ | float64 | Computed | Horizontal finite difference of $\mathbf{z}$ |
| $\mathbf{z}_2$ | $(r, c, n_C)$ | float64 | Computed | Vertical finite difference of $\mathbf{z}$ |
| $\text{divp}$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Divergence of dual variables $(p_1, p_2)$ |
| $\text{denom}$ | $(r, c, n_C)$ | float64 | Computed | Normalization denominator for dual update |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda$ | 1 | GAP step size parameter |
| $\tau$ | 6 | TV regularization weight (`tv_weight`) |
| $K_{\text{outer}}$ | 5 | Maximum outer iterations (`iter_max` in main, `n_iter`) |
| $K_{\text{TV}}$ | 5 | Maximum TV denoiser iterations (`tv_iter_max`) |
| $\delta t$ | 0.25 | TV denoiser step size |
| $s$ | 1 | Dispersion step size |
| $r$ | 256 | Image height (rows) |
| $c$ | 256 | Image width (columns) |
| $n_C$ | Loaded from data | Number of spectral channels |

### 2.5 Observations
| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $\mathbf{Y}$ | $(r, c + s(n_C-1))$ | float64 | `meas` | 2D compressed measurement |
| $\boldsymbol{\Phi}$ | $(r, c + s(n_C-1), n_C)$ | float64 | `mask_3d` | 3D shifted coded aperture mask |
| $\mathbf{X}^{\text{true}}$ | $(r, c, n_C)$ | float64 | `truth` | Ground truth spectral datacube |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Load Ground Truth | $\mathbf{X}^{\text{true}} \leftarrow \text{loadmat}(\texttt{matfile})[\texttt{'img'}]$ | File path | $\mathbf{X}^{\text{true}} \in \mathbb{R}^{r \times c \times n_C}$ | MAT file contains 'img' key |
| P2 | Normalize Truth | $\mathbf{X}^{\text{true}} \leftarrow \mathbf{X}^{\text{true}} / 255$ if $\max(\mathbf{X}^{\text{true}}) > 1$ else $\mathbf{X}^{\text{true}}$ | $\mathbf{X}^{\text{true}}$ | $\mathbf{X}^{\text{true}} \in [0,1]^{r \times c \times n_C}$ | Values originally in [0,255] |
| P3 | Load Base Mask | $\mathbf{M}_{256} \leftarrow \text{loadmat}(\texttt{maskfile})[\texttt{'mask'}]$ | File path | $\mathbf{M}_{256} \in \mathbb{R}^{r \times c}$ | 2D binary mask |
| P4 | Initialize Expanded Mask | $\boldsymbol{\Phi} \leftarrow \mathbf{0}^{r \times (c + s(n_C-1)) \times n_C}$ | $r, c, n_C, s$ | $\boldsymbol{\Phi}$ | - |
| P5 | Populate Shifted Mask | For $i = 0, \ldots, n_C-1$: $\boldsymbol{\Phi}_{:, i:i+c, i} \leftarrow \mathbf{M}_{256}$ | $\mathbf{M}_{256}, \boldsymbol{\Phi}$ | $\boldsymbol{\Phi} \in \mathbb{R}^{r \times (c+s(n_C-1)) \times n_C}$ | Dispersion shifts mask per channel |
| P6 | Initialize Shifted Truth | $\mathbf{X}^{\text{shift}} \leftarrow \mathbf{0}^{r \times (c + s(n_C-1)) \times n_C}$ | $r, c, n_C, s$ | $\mathbf{X}^{\text{shift}}$ | - |
| P7 | Populate Shifted Truth | For $i = 0, \ldots, n_C-1$: $\mathbf{X}^{\text{shift}}_{:, is:is+c, i} \leftarrow \mathbf{X}^{\text{true}}_{:,:,i}$ | $\mathbf{X}^{\text{true}}, s$ | $\mathbf{X}^{\text{shift}}$ | Simulates dispersion |
| P8 | Generate Measurement | $\mathbf{Y} = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_{:,:,i} \odot \mathbf{X}^{\text{shift}}_{:,:,i}$ | $\boldsymbol{\Phi}, \mathbf{X}^{\text{shift}}$ | $\mathbf{Y} \in \mathbb{R}^{r \times (c+s(n_C-1))}$ | Noiseless simulation |
| P9 | Compute Mask Sum | $\boldsymbol{\Phi}_{\text{sum}} = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_{:,:,i}$; then $(\boldsymbol{\Phi}_{\text{sum}})_{jk} \leftarrow 1$ wherever $(\boldsymbol{\Phi}_{\text{sum}})_{jk} = 0$ | $\boldsymbol{\Phi}$ | $\boldsymbol{\Phi}_{\text{sum}} \in \mathbb{R}^{r \times (c+s(n_C-1))}$ | Prevents division by zero |
| P10 | Generate TV Index Arrays | $\text{idx} = [1,2,\ldots,r-1,r-1]$; $\text{iux} = [0,0,1,\ldots,r-2]$; $\text{ir} = [1,2,\ldots,c-1,c-1]$; $\text{il} = [0,0,1,\ldots,c-2]$ | $r, c$ | Index arrays for finite differences | Neumann boundary conditions |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $\mathbf{X}$ | $\mathbf{X}^{(0)} = \mathbf{A}^T(\mathbf{Y}) = \mathbf{Y}_{:,:,\text{newaxis}} \odot \boldsymbol{\Phi}$ where $\mathbf{Y}$ is broadcast to shape $(r, c+s(n_C-1), n_C)$ | $(r, c+s(n_C-1), n_C)$ | float64 | `backward_operator(meas, Phi)` |
| 2 | $\mathbf{y}_1$ | $\mathbf{y}_1^{(0)} = \mathbf{0}$ | $(r, c+s(n_C-1))$ | float64 | Zero initialization |
| 3 | $\text{psnr\_all}$ | $[] $ (empty list) | Variable | list | Metric storage |

### 4.2 ITERATION LOOP

```
BEGIN_LOOP k = 0 to K_outer - 1
```

| step_order | step_name | formula | inputs (shapes) | outputs (shapes) | computational_note |
|------------|-----------|---------|-----------------|------------------|-------------------|
| 4 | Forward Projection [FORWARD] | $\mathbf{y}_b^{(k)} = \sum_{i=1}^{n_C} \mathbf{X}^{(k)}_{:,:,i} \odot \boldsymbol{\Phi}_{:,:,i}$ | $\mathbf{X}^{(k)}$ $(r, c+s(n_C-1), n_C)$, $\boldsymbol{\Phi}$ $(r, c+s(n_C-1), n_C)$ | $\mathbf{y}_b^{(k)}$ $(r, c+s(n_C-1))$ | `np.sum(x * Phi, axis=2)` |
| 5 | Dual Variable Update | $\mathbf{y}_1^{(k+1)} = \mathbf{y}_1^{(k)} + (\mathbf{Y} - \mathbf{y}_b^{(k)})$ | $\mathbf{y}_1^{(k)}$ $(r, c+s(n_C-1))$, $\mathbf{Y}$ $(r, c+s(n_C-1))$, $\mathbf{y}_b^{(k)}$ $(r, c+s(n_C-1))$ | $\mathbf{y}_1^{(k+1)}$ $(r, c+s(n_C-1))$ | Accumulates measurement residual |
| 6 | Compute Scaled Residual | $\mathbf{R}^{(k)} = \frac{\mathbf{y}_1^{(k+1)} - \mathbf{y}_b^{(k)}}{\boldsymbol{\Phi}_{\text{sum}}}$ | $\mathbf{y}_1^{(k+1)}$ $(r, c+s(n_C-1))$, $\mathbf{y}_b^{(k)}$ $(r, c+s(n_C-1))$, $\boldsymbol{\Phi}_{\text{sum}}$ $(r, c+s(n_C-1))$ | $\mathbf{R}^{(k)}$ $(r, c+s(n_C-1))$ | Element-wise division |
| 7 | Backward Projection [ADJOINT] | $\mathbf{B}^{(k)} = \mathbf{R}^{(k)}_{:,:,\text{newaxis}} \odot \boldsymbol{\Phi}$ where $\mathbf{R}^{(k)}$ is broadcast via `np.repeat(..., n_C, axis=2)` | $\mathbf{R}^{(k)}$ $(r, c+s(n_C-1))$, $\boldsymbol{\Phi}$ $(r, c+s(n_C-1), n_C)$ | $\mathbf{B}^{(k)}$ $(r, c+s(n_C-1), n_C)$ | `backward_operator` |
| 8 | Primal Update (GAP Step) | $\tilde{\mathbf{X}}^{(k)} = \mathbf{X}^{(k)} + \lambda \cdot \mathbf{B}^{(k)}$ | $\mathbf{X}^{(k)}$ $(r, c+s(n_C-1), n_C)$, $\mathbf{B}^{(k)}$ $(r, c+s(n_C-1), n_C)$, $\lambda=1$ | $\tilde{\mathbf{X}}^{(k)}$ $(r, c+s(n_C-1), n_C)$ | Data consistency update |
| 9 | Shift Back to Image Domain | For $i = 0, \ldots, n_C-1$: $\mathbf{X}_{\text{img}}^{(k)}[:,:,i] = \tilde{\mathbf{X}}^{(k)}[:, 0:c, i]$ after rolling $\tilde{\mathbf{X}}^{(k)}[:,:,i]$ by $-si$ along axis 1 | $\tilde{\mathbf{X}}^{(k)}$ $(r, c+s(n_C-1), n_C)$ | $\mathbf{X}_{\text{img}}^{(k)}$ $(r, c, n_C)$ | `shift_back` function; reverses dispersion |

```
    BEGIN_LOOP (TV Denoiser) t = 0 to K_TV - 1
```

| step_order | step_name | formula | inputs (shapes) | outputs (shapes) | computational_note |
|------------|-----------|---------|-----------------|------------------|-------------------|
| 10 | Initialize TV Dual (first outer iter only) | $\mathbf{p}_1^{(0)} = \mathbf{0}$, $\mathbf{p}_2^{(0)} = \mathbf{0}$, $\text{divp}^{(0)} = \mathbf{0}$ | - | $\mathbf{p}_1, \mathbf{p}_2, \text{divp}$ each $(r, c, n_C)$ | Reset each call to TV_denoiser |
| 11 | Compute Gradient Input | $\mathbf{z}^{(t)} = \text{divp}^{(t)} - \tau \cdot \mathbf{X}_{\text{img}}^{(k)}$ | $\text{divp}^{(t)}$ $(r, c, n_C)$, $\mathbf{X}_{\text{img}}^{(k)}$ $(r, c, n_C)$, $\tau=6$ | $\mathbf{z}^{(t)}$ $(r, c, n_C)$ | Scaled input |
| 12 | Horizontal Finite Difference | $\mathbf{z}_1^{(t)}[i,j,c] = \mathbf{z}^{(t)}[i, \text{ir}[j], c] - \mathbf{z}^{(t)}[i,j,c]$ where $\text{ir}[j] = \min(j+1, c-1)$ | $\mathbf{z}^{(t)}$ $(r, c, n_C)$, $\text{ir}$ | $\mathbf{z}_1^{(t)}$ $(r, c, n_C)$ | Forward difference, Neumann BC |
| 13 | Vertical Finite Difference | $\mathbf{z}_2^{(t)}[i,j,c] = \mathbf{z}^{(t)}[\text{idx}[i], j, c] - \mathbf{z}^{(t)}[i,j,c]$ where $\text{idx}[i] = \min(i+1, r-1)$ | $\mathbf{z}^{(t)}$ $(r, c, n_C)$, $\text{idx}$ | $\mathbf{z}_2^{(t)}$ $(r, c, n_C)$ | Forward difference, Neumann BC |
| 14 | Compute Gradient Magnitude | $\mathbf{G}^{(t)}[i,j] = \sqrt{\sum_{c=1}^{n_C}\left((\mathbf{z}_1^{(t)}[i,j,c])^2 + (\mathbf{z}_2^{(t)}[i,j,c])^2\right)}$ | $\mathbf{z}_1^{(t)}$ $(r, c, n_C)$, $\mathbf{z}_2^{(t)}$ $(r, c, n_C)$ | $\mathbf{G}^{(t)}$ $(r, c)$ | `np.sqrt(np.sum(z1**2 + z2**2, 2))` |
| 15 | Compute Denominator (2D) | $\text{denom}_{2D}^{(t)} = 1 + \delta t \cdot \mathbf{G}^{(t)}$ | $\mathbf{G}^{(t)}$ $(r, c)$, $\delta t = 0.25$ | $\text{denom}_{2D}^{(t)}$ $(r, c)$ | Scalar addition and multiplication |
| 16 | Broadcast Denominator to 3D | $\text{denom}^{(t)} = \text{tile}(\text{denom}_{2D}^{(t)}[:,:,\text{newaxis}], (1,1,n_C))$ | $\text{denom}_{2D}^{(t)}$ $(r, c)$ | $\text{denom}^{(t)}$ $(r, c, n_C)$ | `np.tile` |
| 17 | Update Dual Variable $\mathbf{p}_1$ | $\mathbf{p}_1^{(t+1)} = \frac{\mathbf{p}_1^{(t)} + \delta t \cdot \mathbf{z}_1^{(t)}}{\text{denom}^{(t)}}$ | $\mathbf{p}_1^{(t)}$ $(r, c, n_C)$, $\mathbf{z}_1^{(t)}$ $(r, c, n_C)$, $\text{denom}^{(t)}$ $(r, c, n_C)$, $\delta t = 0.25$ | $\mathbf{p}_1^{(t+1)}$ $(r, c, n_C)$ | Element-wise |
| 18 | Update Dual Variable $\mathbf{p}_2$ | $\mathbf{p}_2^{(t+1)} = \frac{\mathbf{p}_2^{(t)} + \delta t \cdot \mathbf{z}_2^{(t)}}{\text{denom}^{(t)}}$ | $\mathbf{p}_2^{(t)}$ $(r, c, n_C)$, $\mathbf{z}_2^{(t)}$ $(r, c, n_C)$, $\text{denom}^{(t)}$ $(r, c, n_C)$, $\delta t = 0.25$ | $\mathbf{p}_2^{(t+1)}$ $(r, c, n_C)$ | Element-wise |
| 19 | Compute Divergence | $\text{divp}^{(t+1)}[i,j,c] = \mathbf{p}_1^{(t+1)}[i,j,c] - \mathbf{p}_1^{(t+1)}[i, \text{il}[j], c] + \mathbf{p}_2^{(t+1)}[i,j,c] - \mathbf{p}_2^{(t+1)}[\text{iux}[i], j, c]$ | $\mathbf{p}_1^{(t+1)}$ $(r, c, n_C)$, $\mathbf{p}_2^{(t+1)}$ $(r, c, n_C)$, $\text{il}, \text{iux}$ | $\text{divp}^{(t+1)}$ $(r, c, n_C)$ | Backward difference (adjoint of gradient) |

```
    END_LOOP (TV Denoiser)
```

| step_order | step_name | formula | inputs (shapes) | outputs (shapes) | computational_note |
|------------|-----------|---------|-----------------|------------------|-------------------|
| 20 | TV Denoised Output | $\mathbf{X}_{\text{img}}^{(k+1)} = \mathbf{X}_{\text{img}}^{(k)} - \frac{\text{divp}^{(K_{TV})}}{\tau}$ | $\mathbf{X}_{\text{img}}^{(k)}$ $(r, c, n_C)$, $\text{divp}^{(K_{TV})}$ $(r, c, n_C)$, $\tau=6$ | $\mathbf{X}_{\text{img}}^{(k+1)}$ $(r, c, n_C)$ | Primal recovery from dual |
| 21 | Shift Forward to Measurement Domain | For $i = 0, \ldots, n_C-1$: $\mathbf{X}^{(k+1)}[:, is:is+c, i] = \mathbf{X}_{\text{img}}^{(k+1)}[:,:,i]$; other entries are 0 | $\mathbf{X}_{\text{img}}^{(k+1)}$ $(r, c, n_C)$, $s=1$ | $\mathbf{X}^{(k+1)}$ $(r, c+s(n_C-1), n_C)$ | `shift` function; re-applies dispersion |
| 22 | Compute PSNR Metric | $\text{PSNR}^{(k)} = 20 \log_{10}\left(\frac{1.0}{\sqrt{\frac{1}{r \cdot c \cdot n_C}\sum_{i,j,c}(\mathbf{X}^{\text{true}}_{i,j,c} - \mathbf{X}_{\text{img}}^{(k+1)}[i,j,c])^2}}\right)$ | $\mathbf{X}^{\text{true}}$ $(r, c, n_C)$, $\mathbf{X}_{\text{img}}^{(k+1)}$ $(r, c, n_C)$ | $\text{PSNR}^{(k)}$ (scalar) | Returns 100 if MSE=0 |
| 23 | Store Metric | $\text{psnr\_all}.\text{append}(\text{PSNR}^{(k)})$ | $\text{PSNR}^{(k)}$ | Updated list | - |

```
END_LOOP
```

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs (shapes) | outputs (shapes) | computational_note |
|------------|-----------|---------|-----------------|------------------|-------------------|
| 24 | Final Reconstruction | $\hat{\mathbf{X}} = \mathbf{X}_{\text{img}}^{(K_{\text{outer}})}$ | $\mathbf{X}_{\text{img}}^{(K_{\text{outer}})}$ $(r, c, n_C)$ | $\hat{\mathbf{X}}$ $(r, c, n_C)$ | Output in image domain |
| 25 | Compute Final PSNR | $\text{PSNR}_{\text{final}} = 20 \log_{10}\left(\frac{1.0}{\sqrt{\text{MSE}(\mathbf{X}^{\text{true}}, \hat{\mathbf{X}})}}\right)$ | $\mathbf{X}^{\text{true}}$ $(r, c, n_C)$, $\hat{\mathbf{X}}$ $(r, c, n_C)$ | $\text{PSNR}_{\text{final}}$ (scalar) | Evaluated in `evaluate_results` |
| 26 | Save Reconstruction | Write $\hat{\mathbf{X}}$ to `recon_result.mat` | $\hat{\mathbf{X}}$ | MAT file | `scipy.io.savemat` |

## 5. STOPPING CRITERION

```json
{
  "type": "iteration_limit",
  "expression": "k < K_{\\text{outer}}",
  "parameters": {
    "K_outer": 5,
    "K_TV": 5
  },