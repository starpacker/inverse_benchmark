## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: GAP-TV (Generalized Alternating Projection with Total Variation)
- **algorithm_family**: Proximal Splitting / Plug-and-Play Methods for Inverse Problems
- **summary**: Reconstructs a 3D hyperspectral cube from a 2D coded aperture snapshot spectral imager (CASSI) measurement using alternating data-consistency projection and TV denoising.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{X} \in \mathbb{R}^{r \times c \times n_C}} \frac{1}{2}\|\mathbf{y} - \mathbf{A}(\mathbf{X})\|_2^2 + \tau \sum_{i=1}^{n_C} \|\nabla \mathbf{X}_{:,:,i}\|_1$$

where $\mathbf{A}(\mathbf{X}) = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_{:,:,i} \odot \mathcal{S}_i(\mathbf{X}_{:,:,i})$ is the CASSI forward operator combining masking and spectral dispersion.

### data_fidelity_term
$$\mathcal{D}(\mathbf{X}) = \frac{1}{2}\|\mathbf{y} - \mathbf{A}(\mathbf{X})\|_2^2$$

where:
- $\mathbf{y} \in \mathbb{R}^{r \times (c + s(n_C-1))}$ is the compressed 2D measurement
- $\mathbf{A}: \mathbb{R}^{r \times c \times n_C} \rightarrow \mathbb{R}^{r \times (c+s(n_C-1))}$ is the CASSI forward operator

### regularization_term
$$\mathcal{R}(\mathbf{X}) = \tau \sum_{i=1}^{n_C} \text{TV}(\mathbf{X}_{:,:,i}) = \tau \sum_{i=1}^{n_C} \sum_{j,k} \sqrt{(\nabla_h \mathbf{X}_{:,:,i})_{j,k}^2 + (\nabla_v \mathbf{X}_{:,:,i})_{j,k}^2}$$

where $\nabla_h$, $\nabla_v$ are horizontal and vertical finite difference operators, and $\tau$ is the TV weight.

### constraints
None

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\tilde{\mathbf{X}}$ | primal | $(r, c+s(n_C-1), n_C)$ | float64 | $\tilde{\mathbf{X}}^{(0)} = \mathbf{A}^T(\mathbf{y}) = \boldsymbol{\Phi} \odot \text{tile}(\mathbf{y}, n_C)$ | Reconstruction in shifted (measurement) domain |
| $\mathbf{X}$ | primal | $(r, c, n_C)$ | float64 | Derived via $\mathcal{S}^{-1}(\tilde{\mathbf{X}}^{(0)})$ | Reconstruction in image domain (final output) |

### Dual/Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}_1$ | auxiliary | $(r, c+s(n_C-1))$ | float64 | $\mathbf{0}$ | Accumulated residual for GAP update |
| $\mathbf{p}_1$ | dual | $(r, c, n_C)$ | float64 | $\mathbf{0}$ (per TV call) | Dual variable for horizontal gradient in TV |
| $\mathbf{p}_2$ | dual | $(r, c, n_C)$ | float64 | $\mathbf{0}$ (per TV call) | Dual variable for vertical gradient in TV |
| $\text{divp}$ | auxiliary | $(r, c, n_C)$ | float64 | $\mathbf{0}$ (per TV call) | Divergence of dual variable $(\mathbf{p}_1, \mathbf{p}_2)$ |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}$ | observations | $(r, c+s(n_C-1))$ | float64 | Loaded from simulation | Compressed 2D CASSI measurement |
| $\mathbf{X}^{gt}$ | observations | $(r, c, n_C)$ | float64 | Loaded from `.mat` file, normalized to $[0,1]$ | Ground truth hyperspectral cube |

### Constants (Operators and Masks)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\boldsymbol{\Phi}$ | constants | $(r, c+s(n_C-1), n_C)$ | float64 | Constructed from 2D mask via channel-wise shifting | 3D coded aperture sensing matrix |
| $\boldsymbol{\Phi}_{\text{sum}}$ | constants | $(r, c+s(n_C-1))$ | float64 | $\sum_{i=1}^{n_C} \boldsymbol{\Phi}_{:,:,i}$; zeros replaced by 1 | Sum of mask across channels (for normalization) |
| $\boldsymbol{\Phi}_{2D}$ | constants | $(r, c)$ | float64 | Loaded from `mask256.mat` | Base 2D coded aperture pattern |

### Algorithm Parameters (Hardcoded Constants)
| Symbol | Category | Value | Description |
|--------|----------|-------|-------------|
| $\lambda$ | constants | 1 | GAP step size parameter |
| $K_{\text{outer}}$ | constants | 5 | Maximum outer iterations |
| $\tau$ | constants | 6 | TV regularization weight (`tv_weight`) |
| $K_{\text{TV}}$ | constants | 5 | Maximum TV inner iterations (`tv_iter_max`) |
| $s$ | constants | 1 | Dispersion step size |
| $\delta t$ | constants | 0.25 | Chambolle TV algorithm step size |
| $\text{PIXEL\_MAX}$ | constants | 1.0 | Maximum pixel value for PSNR calculation |
| $r$ | constants | 256 | Image height (from data) |
| $c$ | constants | 256 | Image width (from data) |
| $n_C$ | constants | Inferred from data | Number of spectral channels |

---

## 3. PREPROCESSING PIPELINE

| Step Order | Step Name | Formula | Inputs | Outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load Ground Truth | $\mathbf{X}^{gt} \leftarrow \text{loadmat}(\texttt{matfile})[\texttt{'img'}]$ | File path | $\mathbf{X}^{gt}$: $(r, c, n_C)$ |
| 2 | Normalize Truth | $\mathbf{X}^{gt} \leftarrow \begin{cases} \mathbf{X}^{gt}/255 & \text{if } \max(\mathbf{X}^{gt}) > 1 \\ \mathbf{X}^{gt} & \text{otherwise} \end{cases}$ | $\mathbf{X}^{gt}$ | $\mathbf{X}^{gt} \in [0,1]^{r \times c \times n_C}$ |
| 3 | Load 2D Mask | $\boldsymbol{\Phi}_{2D} \leftarrow \text{loadmat}(\texttt{maskfile})[\texttt{'mask'}]$ | File path | $\boldsymbol{\Phi}_{2D}$: $(r, c)$ |
| 4 | Initialize 3D Mask | $\boldsymbol{\Phi} \leftarrow \mathbf{0}^{r \times (c+s(n_C-1)) \times n_C}$ | $r, c, n_C, s$ | $\boldsymbol{\Phi}$: $(r, c+s(n_C-1), n_C)$ |
| 5 | Construct 3D Mask | $\boldsymbol{\Phi}_{:, i:i+c, i} \leftarrow \boldsymbol{\Phi}_{2D}, \quad \forall i \in \{0, \ldots, n_C-1\}$ | $\boldsymbol{\Phi}_{2D}$, $\boldsymbol{\Phi}$ | $\boldsymbol{\Phi}$: shifted mask per channel |
| 6 | Construct Shifted Truth | $\tilde{\mathbf{X}}^{gt}_{:, i \cdot s : i \cdot s + c, i} \leftarrow \mathbf{X}^{gt}_{:,:,i}, \quad \forall i \in \{0, \ldots, n_C-1\}$ | $\mathbf{X}^{gt}$ | $\tilde{\mathbf{X}}^{gt}$: $(r, c+s(n_C-1), n_C)$ |
| 7 | Simulate Measurement | $\mathbf{y} = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_{:,:,i} \odot \tilde{\mathbf{X}}^{gt}_{:,:,i}$ | $\boldsymbol{\Phi}$, $\tilde{\mathbf{X}}^{gt}$ | $\mathbf{y}$: $(r, c+s(n_C-1))$ |
| 8 | Compute Mask Sum | $\boldsymbol{\Phi}_{\text{sum}} = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_{:,:,i}$; $(\boldsymbol{\Phi}_{\text{sum}})_{jk} \leftarrow 1$ if $(\boldsymbol{\Phi}_{\text{sum}})_{jk} = 0$ | $\boldsymbol{\Phi}$ | $\boldsymbol{\Phi}_{\text{sum}}$: $(r, c+s(n_C-1))$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| I.1 | Backprojection Initialization | $\tilde{\mathbf{X}}^{(0)} = \mathbf{A}^T(\mathbf{y}) = \boldsymbol{\Phi} \odot \text{tile}(\mathbf{y}, n_C)$ where $\text{tile}(\mathbf{y}, n_C)_{:,:,i} = \mathbf{y} \; \forall i$ | $\mathbf{y}$, $\boldsymbol{\Phi}$ | $\tilde{\mathbf{X}}^{(0)}$: $(r, c+s(n_C-1), n_C)$ |
| I.2 | Initialize Auxiliary | $\mathbf{y}_1^{(0)} = \mathbf{0}$ | — | $\mathbf{y}_1^{(0)}$: $(r, c+s(n_C-1))$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, K_{\text{outer}}-1$)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| L.1 | Forward Projection | $\mathbf{y}_b^{(k)} = \mathbf{A}(\tilde{\mathbf{X}}^{(k)}) = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_{:,:,i} \odot \tilde{\mathbf{X}}^{(k)}_{:,:,i}$ | $\tilde{\mathbf{X}}^{(k)}$, $\boldsymbol{\Phi}$ | $\mathbf{y}_b^{(k)}$: $(r, c+s(n_C-1))$ | Element-wise multiply then sum |
| L.2 | Residual Accumulation | $\mathbf{y}_1^{(k+1)} = \mathbf{y}_1^{(k)} + (\mathbf{y} - \mathbf{y}_b^{(k)})$ | $\mathbf{y}_1^{(k)}$, $\mathbf{y}$, $\mathbf{y}_b^{(k)}$ | $\mathbf{y}_1^{(k+1)}$ | GAP dual update |
| L.3 | Compute Correction Term | $\mathbf{r}^{(k)} = \frac{\mathbf{y}_1^{(k+1)} - \mathbf{y}_b^{(k)}}{\boldsymbol{\Phi}_{\text{sum}}}$ | $\mathbf{y}_1^{(k+1)}$, $\mathbf{y}_b^{(k)}$, $\boldsymbol{\Phi}_{\text{sum}}$ | $\mathbf{r}^{(k)}$: $(r, c+s(n_C-1))$ | Element-wise division |
| L.4 | Backprojection Update | $\tilde{\mathbf{X}}^{(k+\frac{1}{2})} = \tilde{\mathbf{X}}^{(k)} + \lambda \cdot \mathbf{A}^T(\mathbf{r}^{(k)})$ where $\mathbf{A}^T(\mathbf{r}) = \boldsymbol{\Phi} \odot \text{tile}(\mathbf{r}, n_C)$ | $\tilde{\mathbf{X}}^{(k)}$, $\mathbf{r}^{(k)}$, $\boldsymbol{\Phi}$, $\lambda$ | $\tilde{\mathbf{X}}^{(k+\frac{1}{2})}$ | GAP primal update |
| L.5 | Shift Back (Undisperse) | $\mathbf{X}^{(k+\frac{1}{2})}_{:,:,i} = \tilde{\mathbf{X}}^{(k+\frac{1}{2})}_{:, 0:c, i}$ after rolling channel $i$ by $-s \cdot i$ along axis 1 | $\tilde{\mathbf{X}}^{(k+\frac{1}{2})}$ | $\mathbf{X}^{(k+\frac{1}{2})}$: $(r, c, n_C)$ | Transform to image domain |
| L.6 | **TV Denoising (Inner Loop)** | $\mathbf{X}^{(k+1)} = \text{prox}_{\tau^{-1} \text{TV}}(\mathbf{X}^{(k+\frac{1}{2})})$ | $\mathbf{X}^{(k+\frac{1}{2})}$, $\tau$, $K_{\text{TV}}$ | $\mathbf{X}^{(k+1)}$ | See sub-block below |
| L.7 | Shift Forward (Disperse) | $\tilde{\mathbf{X}}^{(k+1)}_{:, i \cdot s : i \cdot s + c, i} = \mathbf{X}^{(k+1)}_{:,:,i}, \quad \forall i$ | $\mathbf{X}^{(k+1)}$ | $\tilde{\mathbf{X}}^{(k+1)}$: $(r, c+s(n_C-1), n_C)$ | Transform to measurement domain |
| L.8 | Compute PSNR | $\text{PSNR}^{(k+1)} = 20 \log_{10}\left(\frac{1.0}{\sqrt{\text{MSE}(\mathbf{X}^{gt}, \mathbf{X}^{(k+1)})}}\right)$ | $\mathbf{X}^{gt}$, $\mathbf{X}^{(k+1)}$ | $\text{PSNR}^{(k+1)}$ | Monitoring metric |

**END_LOOP**

---

#### L.6 SUB-BLOCK: TV Denoising (Chambolle's Algorithm)

Solves: $\mathbf{u}^* = \arg\min_{\mathbf{u}} \frac{\tau}{2}\|\mathbf{u} - \mathbf{f}\|_2^2 + \|\nabla \mathbf{u}\|_1$

Dual formulation: $\mathbf{u}^* = \mathbf{f} - \frac{1}{\tau}\text{div}(\mathbf{p}^*)$ where $\mathbf{p}^*$ maximizes the dual.

**Inputs**: $\mathbf{f} = \mathbf{X}^{(k+\frac{1}{2})}$, $\tau$ (tv_weight), $K_{\text{TV}}$ (tv_iter_max)

**BEGIN_LOOP** (for $t = 0, 1, \ldots, K_{\text{TV}}-1$)

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| TV.1 | Initialize (at $t=0$) | $\mathbf{p}_1^{(0)} = \mathbf{0}$, $\mathbf{p}_2^{(0)} = \mathbf{0}$, $\text{divp}^{(0)} = \mathbf{0}$ | — | Dual variables |
| TV.2 | Compute Primal Estimate | $\mathbf{z}^{(t)} = \text{divp}^{(t)} - \tau \cdot \mathbf{f}$ | $\text{divp}^{(t)}$, $\mathbf{f}$, $\tau$ | $\mathbf{z}^{(t)}$: $(r, c, n_C)$ |
| TV.3 | Horizontal Gradient | $\mathbf{z}_1^{(t)}_{j,k,i} = \mathbf{z}^{(t)}_{j, k+1, i} - \mathbf{z}^{(t)}_{j,k,i}$ with Neumann BC: $\mathbf{z}^{(t)}_{j, c, i} = \mathbf{z}^{(t)}_{j, c-1, i}$ | $\mathbf{z}^{(t)}$ | $\mathbf{z}_1^{(t)}$ |
| TV.4 | Vertical Gradient | $\mathbf{z}_2^{(t)}_{j,k,i} = \mathbf{z}^{(t)}_{j+1, k, i} - \mathbf{z}^{(t)}_{j,k,i}$ with Neumann BC: $\mathbf{z}^{(t)}_{r, k, i} = \mathbf{z}^{(t)}_{r-1, k, i}$ | $\mathbf{z}^{(t)}$ | $\mathbf{z}_2^{(t)}$ |
| TV.5 | Gradient Magnitude (Isotropic) | $\mathbf{g}^{(t)}_{j,k} = \sqrt{\sum_{i=1}^{n_C}\left[(\mathbf{z}_1^{(t)}_{j,k,i})^2 + (\mathbf{z}_2^{(t)}_{j,k,i})^2\right]}$ | $\mathbf{z}_1^{(t)}$, $\mathbf{z}_2^{(t)}$ | $\mathbf{g}^{(t)}$: $(r, c)$ |
| TV.6 | Compute Denominator | $\mathbf{d}^{(t)}_{j,k,i} = 1 + \delta t \cdot \mathbf{g}^{(t)}_{j,k}$ (broadcast to 3D) | $\mathbf{g}^{(t)}$, $\delta t = 0.25$ | $\mathbf{d}^{(t)}$: $(r, c, n_C)$ |
| TV.7 | Update Dual Variable $\mathbf{p}_1$ | $\mathbf{p}_1^{(t+1)} = \frac{\mathbf{p}_1^{(t)} + \delta t \cdot \mathbf{z}_1^{(t)}}{\mathbf{d}^{(t)}}$ | $\mathbf{p}_1^{(t)}$, $\mathbf{z}_1^{(t)}$, $\mathbf{d}^{(t)}$ | $\mathbf{p}_1^{(t+1)}$ |
| TV.8 | Update Dual Variable $\mathbf{p}_2$ | $\mathbf{p}_2^{(t+1)} = \frac{\mathbf{p}_2^{(t)} + \delta t \cdot \mathbf{z}_2^{(t)}}{\mathbf{d}^{(t)}}$ | $\mathbf{p}_2^{(t)}$, $\mathbf{z}_2^{(t)}$, $\mathbf{d}^{(t)}$ | $\mathbf{p}_2^{(t+1)}$ |
| TV.9 | Compute Divergence | $\text{divp}^{(t+1)}_{j,k,i} = (\mathbf{p}_1^{(t+1)}_{j,k,i} - \mathbf{p}_1^{(t+1)}_{j,k-1,i}) + (\mathbf{p}_2^{(t+1)}_{j,k,i} - \mathbf{p}_2^{(t+1)}_{j-1,k,i})$ with BC: $\mathbf{p}_1^{(t+1)}_{j,-1,i} = 0$, $\mathbf{p}_2^{(t+1)}_{-1,k,i} = 0$ | $\mathbf{p}_1^{(t+1)}$, $\mathbf{p}_2^{(t+1)}$ | $\text{divp}^{(t+1)}$ |

**END_LOOP**

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| TV.10 | Recover Primal Solution | $\mathbf{u} = \mathbf{f} - \frac{1}{\tau}\text{divp}^{(K_{\text{TV}})}$ | $\mathbf{f}$, $\text{divp}^{(K_{\text{TV}})}$, $\tau$ | $\mathbf{u}$: $(r, c, n_C)$ |

**Output**: $\mathbf{X}^{(k+1)} = \mathbf{u}$

---

### 4.3 FINALIZATION

| Step | Step Name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| F.1 | Extract Final Image | $\mathbf{X}^{*} = \mathbf{X}^{(K_{\text{outer}})}$ | Final iterate | Reconstructed hyperspectral cube |
| F.2 | Save Results | Write $\mathbf{X}^{*}$ to `recon_result.mat` | $\mathbf{X}^{*}$ | File output |

---

## 5. STOPPING CRITERION

| Property | Value |
|----------|-------|
| **type** | iteration_limit |
| **expression** | $k \geq K_{\text{outer}}$ where $K_{\text{outer}} = 5$ |
| **parameters** | `iter_max = 5` (outer loop), `tv_iter_max = 5` (inner TV loop) |
| **check_frequency** | After each complete outer iteration |

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| **PSNR** | $\text{PSNR}(\mathbf{X}^{gt}, \hat{\mathbf{X}}) = 20 \log_{10}\left(\frac{\text{PIXEL\_MAX}}{\sqrt{\frac{1}{N}\sum_{j,k,i}(\mathbf{X}^{gt}_{j,k,i} - \hat{\mathbf{X}}_{j,k,i})^2}}\right)$ where $\text{PIXEL\_MAX} = 1.0$ and $N = r \times c \times n_C$. Returns 100 dB if MSE = 0. | Peak Signal-to-Noise Ratio between ground truth and reconstruction |
| **MSE** | $\text{MSE}(\mathbf{X}^{gt}, \hat{\mathbf{X}}) = \frac{1}{N}\|\mathbf{X}^{gt} - \hat{\mathbf{X}}\|_F^2$ | Mean Squared Error (intermediate for PSNR) |