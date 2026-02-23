## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: GAP-TV (Generalized Alternating Projection with Total Variation)
- **algorithm_family**: Proximal Splitting / Plug-and-Play Methods
- **summary**: Reconstructs hyperspectral images from coded aperture snapshot spectral imaging (CASSI) measurements by alternating between a data projection step and a Total Variation denoising step using Chambolle's dual algorithm.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{X}} \; \text{TV}(\mathbf{X}_{\text{img}}) \quad \text{subject to} \quad \mathbf{y} = \mathcal{A}(\mathbf{X})$$

Solved via augmented Lagrangian relaxation:
$$\min_{\mathbf{X}} \; \frac{1}{2\lambda}\|\mathbf{y} - \mathcal{A}(\mathbf{X})\|_2^2 + \tau_{\text{TV}} \cdot \text{TV}(\mathbf{X}_{\text{img}})$$

### data_fidelity_term
$$\mathcal{D}(\mathbf{X}) = \frac{1}{2\lambda}\|\mathbf{y} - \mathcal{A}(\mathbf{X})\|_2^2 = \frac{1}{2\lambda}\sum_{i,j}\left(y_{i,j} - \sum_{c=1}^{n_C} \Phi_{i,j,c} \cdot X_{i,j,c}\right)^2$$

### regularization_term
**Isotropic Total Variation** (applied in image domain after shift_back):
$$\text{TV}(\mathbf{X}_{\text{img}}) = \sum_{i,j,c} \sqrt{(X_{\text{img},i+1,j,c} - X_{\text{img},i,j,c})^2 + (X_{\text{img},i,j+1,c} - X_{\text{img},i,j,c})^2}$$

### constraints
$$\mathbf{X}_{\text{img}} \in [0, 1]^{r \times c \times n_C} \quad \text{(implicit via normalized input)}$$

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}$ | $(r, c + s(n_C-1), n_C)$ | float64 | $\mathcal{A}^T(\mathbf{y})$ | Reconstructed cube in measurement domain |
| $\mathbf{X}_{\text{img}}$ | $(r, c, n_C)$ | float64 | Derived from $\mathbf{X}$ via shift_back | Reconstructed cube in image domain |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}_1$ | $(r, c + s(n_C-1))$ | float64 | $\mathbf{0}$ | Dual variable for data fidelity constraint |
| $\mathbf{p}_1$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | TV dual variable (horizontal gradient) |
| $\mathbf{p}_2$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | TV dual variable (vertical gradient) |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{y}_b$ | $(r, c + s(n_C-1))$ | float64 | Computed | Forward projection $\mathcal{A}(\mathbf{X})$ |
| $\boldsymbol{\Phi}_{\text{sum}}$ | $(r, c + s(n_C-1))$ | float64 | $\sum_{c=1}^{n_C} \Phi_{:,:,c}$ | Mask sum with zero-protection |
| $\mathbf{z}$ | $(r, c, n_C)$ | float64 | Computed | TV intermediate: $\text{div}(\mathbf{p}) - \tau_{\text{TV}} \cdot \mathbf{X}_{\text{img}}$ |
| $\mathbf{z}_1$ | $(r, c, n_C)$ | float64 | Computed | Horizontal difference of $\mathbf{z}$ |
| $\mathbf{z}_2$ | $(r, c, n_C)$ | float64 | Computed | Vertical difference of $\mathbf{z}$ |
| $\text{div}\mathbf{p}$ | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Divergence of dual variable $(\mathbf{p}_1, \mathbf{p}_2)$ |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda$ | 1 | GAP step size parameter |
| $\tau_{\text{TV}}$ | 6 | TV regularization weight |
| $K_{\text{outer}}$ | 5 | Maximum outer iterations |
| $K_{\text{TV}}$ | 5 | Maximum TV denoiser iterations |
| $\delta t$ | 0.25 | TV dual update step size |
| $s$ | 1 | Spectral dispersion step size |
| $r$ | 256 | Image height (from data) |
| $c$ | 256 | Image width (from data) |
| $n_C$ | 28 | Number of spectral channels (from data) |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(r, c + s(n_C-1))$ | float64 | Compressed CASSI measurement |
| $\boldsymbol{\Phi}$ | $(r, c + s(n_C-1), n_C)$ | float64 | 3D coded aperture mask |
| $\mathbf{X}_{\text{truth}}$ | $(r, c, n_C)$ | float64 | Ground truth hyperspectral cube |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| P1 | Load ground truth | $\mathbf{X}_{\text{truth}} = \texttt{loadmat}(\texttt{matfile})[\texttt{'img'}]$ | matfile path | $\mathbf{X}_{\text{truth}} \in \mathbb{R}^{r \times c \times n_C}$ |
| P2 | Normalize truth | $\mathbf{X}_{\text{truth}} \leftarrow \mathbf{X}_{\text{truth}} / 255.0 \quad \text{if } \max(\mathbf{X}_{\text{truth}}) > 1$ | $\mathbf{X}_{\text{truth}}$ | $\mathbf{X}_{\text{truth}} \in [0,1]^{r \times c \times n_C}$ |
| P3 | Load 2D mask | $\mathbf{M}_{256} = \texttt{loadmat}(\texttt{maskfile})[\texttt{'mask'}] \in \{0,1\}^{r \times c}$ | maskfile path | $\mathbf{M}_{256} \in \mathbb{R}^{r \times c}$ |
| P4 | Initialize 3D mask | $\boldsymbol{\Phi} = \mathbf{0} \in \mathbb{R}^{r \times (c + s(n_C-1)) \times n_C}$ | $r, c, n_C, s$ | $\boldsymbol{\Phi}$ |
| P5 | Construct shifted mask | $\boldsymbol{\Phi}_{:, i:i+c, i} \leftarrow \mathbf{M}_{256} \quad \forall i \in \{0, 1, \ldots, n_C-1\}$ | $\mathbf{M}_{256}, \boldsymbol{\Phi}$ | $\boldsymbol{\Phi} \in \mathbb{R}^{r \times (c+s(n_C-1)) \times n_C}$ |
| P6 | Shift truth for simulation | $\mathbf{X}_{\text{shift}} \in \mathbb{R}^{r \times (c+s(n_C-1)) \times n_C}$, $(\mathbf{X}_{\text{shift}})_{:, is:is+c, i} = (\mathbf{X}_{\text{truth}})_{:,:,i}$ | $\mathbf{X}_{\text{truth}}, s$ | $\mathbf{X}_{\text{shift}}$ |
| P7 | Simulate measurement | $\mathbf{y} = \sum_{i=0}^{n_C-1} \boldsymbol{\Phi}_{:,:,i} \odot (\mathbf{X}_{\text{shift}})_{:,:,i}$ | $\boldsymbol{\Phi}, \mathbf{X}_{\text{shift}}$ | $\mathbf{y} \in \mathbb{R}^{r \times (c+s(n_C-1))}$ |
| P8 | Compute mask sum | $(\boldsymbol{\Phi}_{\text{sum}})_{i,j} = \max\left(\sum_{c=0}^{n_C-1} \Phi_{i,j,c}, 1\right)$ | $\boldsymbol{\Phi}$ | $\boldsymbol{\Phi}_{\text{sum}} \in \mathbb{R}^{r \times (c+s(n_C-1))}$ |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize primal via adjoint | $\mathbf{X}^{(0)} = \mathcal{A}^T(\mathbf{y}) = \boldsymbol{\Phi} \odot \text{tile}(\mathbf{y}, n_C)$ where $\text{tile}(\mathbf{y}, n_C)_{i,j,c} = y_{i,j}$ | $\mathbf{y} \in \mathbb{R}^{r \times (c+s(n_C-1))}$, $\boldsymbol{\Phi}$ | $\mathbf{X}^{(0)} \in \mathbb{R}^{r \times (c+s(n_C-1)) \times n_C}$ | [ADJOINT] Backward operator |
| 2 | Initialize dual variable | $\mathbf{y}_1^{(0)} = \mathbf{0}$ | — | $\mathbf{y}_1^{(0)} \in \mathbb{R}^{r \times (c+s(n_C-1))}$ | Zero initialization |
| 3 | Initialize PSNR tracker | $\texttt{psnr\_all} = []$ | — | Empty list | For convergence monitoring |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (outer loop $k = 0, 1, \ldots, K_{\text{outer}}-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 4 | Forward projection | $\mathbf{y}_b^{(k)} = \mathcal{A}(\mathbf{X}^{(k)}) = \sum_{c=0}^{n_C-1} \boldsymbol{\Phi}_{:,:,c} \odot \mathbf{X}^{(k)}_{:,:,c}$ | $\mathbf{X}^{(k)}, \boldsymbol{\Phi}$ | $\mathbf{y}_b^{(k)} \in \mathbb{R}^{r \times (c+s(n_C-1))}$ | [FORWARD] Measurement operator |
| 5 | Dual variable update | $\mathbf{y}_1^{(k+1)} = \mathbf{y}_1^{(k)} + (\mathbf{y} - \mathbf{y}_b^{(k)})$ | $\mathbf{y}_1^{(k)}, \mathbf{y}, \mathbf{y}_b^{(k)}$ | $\mathbf{y}_1^{(k+1)}$ | Accumulate residual |
| 6 | Compute normalized residual | $\mathbf{r}^{(k)} = \frac{\mathbf{y}_1^{(k+1)} - \mathbf{y}_b^{(k)}}{\boldsymbol{\Phi}_{\text{sum}}}$ (elementwise) | $\mathbf{y}_1^{(k+1)}, \mathbf{y}_b^{(k)}, \boldsymbol{\Phi}_{\text{sum}}$ | $\mathbf{r}^{(k)} \in \mathbb{R}^{r \times (c+s(n_C-1))}$ | Protected division |
| 7 | Primal update (GAP step) | $\mathbf{X}^{(k+\frac{1}{2})} = \mathbf{X}^{(k)} + \lambda \cdot \mathcal{A}^T(\mathbf{r}^{(k)})$ where $\mathcal{A}^T(\mathbf{r})_{i,j,c} = \Phi_{i,j,c} \cdot r_{i,j}$ | $\mathbf{X}^{(k)}, \mathbf{r}^{(k)}, \boldsymbol{\Phi}, \lambda=1$ | $\mathbf{X}^{(k+\frac{1}{2})}$ | [ADJOINT] Backward operator |
| 8 | Shift back to image domain | $(\mathbf{X}_{\text{img}}^{(k)})_{:,:,i} = (\mathbf{X}^{(k+\frac{1}{2})})_{:, 0:c, i}$ after rolling: $(\mathbf{X}^{(k+\frac{1}{2})})_{:,:,i} \leftarrow \text{roll}((\mathbf{X}^{(k+\frac{1}{2})})_{:,:,i}, -si, \text{axis}=1)$ | $\mathbf{X}^{(k+\frac{1}{2})}, s$ | $\mathbf{X}_{\text{img}}^{(k)} \in \mathbb{R}^{r \times c \times n_C}$ | Reverse dispersion |

**BEGIN_LOOP** (TV denoiser inner loop $t = 0, 1, \ldots, K_{\text{TV}}-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 9 | Compute TV gradient input | $\mathbf{z}^{(t)} = \text{div}\mathbf{p}^{(t)} - \tau_{\text{TV}} \cdot \mathbf{X}_{\text{img}}^{(k)}$ | $\text{div}\mathbf{p}^{(t)}, \mathbf{X}_{\text{img}}^{(k)}, \tau_{\text{TV}}=6$ | $\mathbf{z}^{(t)} \in \mathbb{R}^{r \times c \times n_C}$ | Initialization: $\text{div}\mathbf{p}^{(0)} = \mathbf{0}$ |
| 10 | Horizontal difference | $\mathbf{z}_1^{(t)}_{i,j,c} = \mathbf{z}^{(t)}_{i,j_+,c} - \mathbf{z}^{(t)}_{i,j,c}$ where $j_+ = \min(j+1, c-1)$ | $\mathbf{z}^{(t)}$ | $\mathbf{z}_1^{(t)}$ | Neumann boundary (replicate) |
| 11 | Vertical difference | $\mathbf{z}_2^{(t)}_{i,j,c} = \mathbf{z}^{(t)}_{i_+,j,c} - \mathbf{z}^{(t)}_{i,j,c}$ where $i_+ = \min(i+1, r-1)$ | $\mathbf{z}^{(t)}$ | $\mathbf{z}_2^{(t)}$ | Neumann boundary (replicate) |
| 12 | Compute gradient magnitude | $\mathbf{D}^{(t)}_{i,j} = 1 + \delta t \cdot \sqrt{\sum_{c=0}^{n_C-1}\left((\mathbf{z}_1^{(t)}_{i,j,c})^2 + (\mathbf{z}_2^{(t)}_{i,j,c})^2\right)}$ | $\mathbf{z}_1^{(t)}, \mathbf{z}_2^{(t)}, \delta t = 0.25$ | $\mathbf{D}^{(t)} \in \mathbb{R}^{r \times c}$ | Coupled across channels |
| 13 | Tile denominator | $\tilde{\mathbf{D}}^{(t)}_{i,j,c} = \mathbf{D}^{(t)}_{i,j} \quad \forall c$ | $\mathbf{D}^{(t)}$ | $\tilde{\mathbf{D}}^{(t)} \in \mathbb{R}^{r \times c \times n_C}$ | Broadcast to 3D |
| 14 | Update dual $\mathbf{p}_1$ | $\mathbf{p}_1^{(t+1)} = \frac{\mathbf{p}_1^{(t)} + \delta t \cdot \mathbf{z}_1^{(t)}}{\tilde{\mathbf{D}}^{(t)}}$ | $\mathbf{p}_1^{(t)}, \mathbf{z}_1^{(t)}, \tilde{\mathbf{D}}^{(t)}, \delta t = 0.25$ | $\mathbf{p}_1^{(t+1)}$ | Initialization: $\mathbf{p}_1^{(0)} = \mathbf{0}$ |
| 15 | Update dual $\mathbf{p}_2$ | $\mathbf{p}_2^{(t+1)} = \frac{\mathbf{p}_2^{(t)} + \delta t \cdot \mathbf{z}_2^{(t)}}{\tilde{\mathbf{D}}^{(t)}}$ | $\mathbf{p}_2^{(t)}, \mathbf{z}_2^{(t)}, \tilde{\mathbf{D}}^{(t)}, \delta t = 0.25$ | $\mathbf{p}_2^{(t+1)}$ | Initialization: $\mathbf{p}_2^{(0)} = \mathbf{0}$ |
| 16 | Compute divergence | $(\text{div}\mathbf{p}^{(t+1)})_{i,j,c} = (\mathbf{p}_1^{(t+1)})_{i,j,c} - (\mathbf{p}_1^{(t+1)})_{i,j_-,c} + (\mathbf{p}_2^{(t+1)})_{i,j,c} - (\mathbf{p}_2^{(t+1)})_{i_-,j,c}$ where $j_- = \max(j-1, 0)$, $i_- = \max(i-1, 0)$ | $\mathbf{p}_1^{(t+1)}, \mathbf{p}_2^{(t+1)}$ | $\text{div}\mathbf{p}^{(t+1)}$ | Adjoint of gradient operator |

**END_LOOP** (TV inner loop)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 17 | TV denoised output | $\tilde{\mathbf{X}}_{\text{img}}^{(k)} = \mathbf{X}_{\text{img}}^{(k)} - \frac{\text{div}\mathbf{p}^{(K_{\text{TV}})}}{\tau_{\text{TV}}}$ | $\mathbf{X}_{\text{img}}^{(k)}, \text{div}\mathbf{p}^{(K_{\text{TV}})}, \tau_{\text{TV}}=6$ | $\tilde{\mathbf{X}}_{\text{img}}^{(k)}$ | Chambolle's closed-form |
| 18 | Shift forward to measurement domain | $\mathbf{X}^{(k+1)} \in \mathbb{R}^{r \times (c+s(n_C-1)) \times n_C}$, $(\mathbf{X}^{(k+1)})_{:, is:is+c, i} = (\tilde{\mathbf{X}}_{\text{img}}^{(k)})_{:,:,i} \quad \forall i$ | $\tilde{\mathbf{X}}_{\text{img}}^{(k)}, s$ | $\mathbf{X}^{(k+1)}$ | Apply dispersion |
| 19 | Compute iteration PSNR | $\text{PSNR}^{(k)} = 20 \log_{10}\left(\frac{1}{\sqrt{\frac{1}{rcn_C}\sum_{i,j,c}(\tilde{\mathbf{X}}_{\text{img},i,j,c}^{(k)} - X_{\text{truth},i,j,c})^2}}\right)$ | $\tilde{\mathbf{X}}_{\text{img}}^{(k)}, \mathbf{X}_{\text{truth}}$ | $\text{PSNR}^{(k)}$ | Append to psnr_all |

**END_LOOP** (outer loop)

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 20 | Extract final reconstruction | $\mathbf{X}_{\text{final}} = \tilde{\mathbf{X}}_{\text{img}}^{(K_{\text{outer}}-1)}$ | $\tilde{\mathbf{X}}_{\text{img}}^{(K_{\text{outer}}-1)}$ | $\mathbf{X}_{\text{final}} \in \mathbb{R}^{r \times c \times n_C}$ | Return image domain result |
| 21 | Compute final PSNR | $\text{PSNR}_{\text{final}} = 20 \log_{10}\left(\frac{1}{\sqrt{\text{MSE}(\mathbf{X}_{\text{final}}, \mathbf{X}_{\text{truth}})}}\right)$ | $\mathbf{X}_{\text{final}}, \mathbf{X}_{\text{truth}}$ | $\text{PSNR}_{\text{final}}$ | Final quality metric |
| 22 | Save reconstruction | Write $\mathbf{X}_{\text{final}}$ to recon_result.mat | $\mathbf{X}_{\text{final}}$ | .mat file | Persistent storage |

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| **type** | iteration_limit |
| **expression** | $k < K_{\text{outer}}$ (outer loop), $t < K_{\text{TV}}$ (TV inner loop) |
| **parameters** | $K_{\text{outer}} = 5$, $K_{\text{TV}} = 5$ |
| **check_frequency** | Evaluated at end of each iteration |

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| PSNR | $\text{PSNR} = 20 \log_{10}\left(\frac{\text{MAX}_I}{\sqrt{\text{MSE}}}\right) = 20 \log_{10}\left(\frac{1.0}{\sqrt{\frac{1}{N}\sum_{i,j,c}(X_{\text{ref},i,j,c} - X_{\text{pred},i,j,c})^2}}\right)$ where $N = r \cdot c \cdot n_C$, $\text{MAX}_I = 1.0$ | $\mathbf{X}_{\text{truth}} \in \mathbb{R}^{r \times c \times n_C}$ | $\mathbf{X}_{\text{final}} \in \mathbb{R}^{r \times c \times n_C}$ | Peak Signal-to-Noise Ratio; higher is better; returns 100 dB if MSE = 0 |
| Convergence Curve | $\{\text{PSNR}^{(k)}\}_{k=0}^{K_{\text{outer}}-1}$ | $\mathbf{X}_{\text{truth}}$ | $\{\tilde{\mathbf{X}}_{\text{img}}^{(k)}\}$ | Per-iteration PSNR trajectory; saved as psnr_curve.png |