# Algorithmic Specification Document: GAP-TV for CASSI Reconstruction

## 1. Mathematical Formulation

### 1.1 Problem Statement
**Coded Aperture Snapshot Spectral Imaging (CASSI) Reconstruction** via Generalized Alternating Projection with Total Variation (GAP-TV).

### 1.2 Forward Model
The CASSI measurement model with spectral dispersion:
$$
\mathbf{y} = \mathbf{A}(\mathbf{x}) = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_i \odot \mathcal{S}_i(\mathbf{x}_i)
$$

where:
- $\mathbf{x} \in \mathbb{R}^{r \times c \times n_C}$: 3D hyperspectral image cube
- $\mathcal{S}_i$: spectral dispersion operator (spatial shift by $i \cdot \Delta$ pixels)
- $\boldsymbol{\Phi} \in \mathbb{R}^{r \times c' \times n_C}$: 3D coded aperture mask, $c' = c + (n_C-1)\Delta$
- $\mathbf{y} \in \mathbb{R}^{r \times c'}$: 2D compressed measurement
- $\odot$: element-wise (Hadamard) product

### 1.3 Inverse Problem Formulation
$$
\min_{\mathbf{x}} \quad \underbrace{\frac{1}{2}\|\mathbf{y} - \mathbf{A}(\mathbf{x})\|_2^2}_{\text{Data Fidelity}} + \underbrace{\frac{1}{\lambda_{TV}} \text{TV}(\mathbf{x})}_{\text{Regularization}}
$$

where the **anisotropic Total Variation** is:
$$
\text{TV}(\mathbf{x}) = \sum_{i=1}^{n_C} \sum_{j,k} \sqrt{|\nabla_h x_i[j,k]|^2 + |\nabla_v x_i[j,k]|^2}
$$

with discrete gradients:
$$
\nabla_h x[j,k] = x[j,k+1] - x[j,k], \quad \nabla_v x[j,k] = x[j+1,k] - x[j,k]
$$

---

## 2. Variables Taxonomy

| Category | Symbol | Description | Shape | Dtype | Initialization |
|----------|--------|-------------|-------|-------|----------------|
| **Primal Variables** |||||
| | $\tilde{\mathbf{x}}^{(k)}$ | Reconstruction in shifted domain | $(r, c', n_C)$ | float64 | $\mathbf{A}^T(\mathbf{y})$ |
| | $\mathbf{x}^{(k)}$ | Reconstruction in image domain | $(r, c, n_C)$ | float64 | $\mathcal{S}^{-1}(\tilde{\mathbf{x}}^{(0)})$ |
| **Auxiliary Variables** |||||
| | $\mathbf{y}_1^{(k)}$ | Accumulated measurement residual | $(r, c')$ | float64 | $\mathbf{0}$ |
| | $\mathbf{p}_1, \mathbf{p}_2$ | Dual variables (TV denoiser) | $(r, c, n_C)$ | float64 | $\mathbf{0}$ |
| | $\text{divp}$ | Divergence of dual variable | $(r, c, n_C)$ | float64 | $\mathbf{0}$ |
| **Hyperparameters** |||||
| | $\lambda$ | GAP step size | scalar | float | 1.0 |
| | $\lambda_{TV}$ | TV regularization weight | scalar | float | 6.0 |
| | $K$ | Max GAP iterations | scalar | int | 20 |
| | $T$ | Max TV iterations | scalar | int | 5 |
| | $\Delta$ | Dispersion step size | scalar | int | 1 |
| | $\tau$ | TV gradient descent step | scalar | float | 0.25 (fixed) |
| **Observed Data/Inputs** |||||
| | $\mathbf{y}$ | Compressed measurement | $(r, c')$ | float64 | loaded |
| | $\boldsymbol{\Phi}$ | 3D coded aperture mask | $(r, c', n_C)$ | float64 | loaded |
| | $\mathbf{x}^*$ | Ground truth (evaluation only) | $(r, c, n_C)$ | float64 | loaded |

---

## 3. Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘
```

| Step | Operation | Input | Output | Formula |
|------|-----------|-------|--------|---------|
| **P1** | Load ground truth | `matfile` | $\mathbf{x}^* \in \mathbb{R}^{r \times c \times n_C}$ | $\mathbf{x}^* = \texttt{loadmat}(\cdot)[\texttt{'img'}]$ |
| **P2** | Normalize intensity | $\mathbf{x}^*$ | $\mathbf{x}^* \in [0,1]$ | $\mathbf{x}^* \leftarrow \mathbf{x}^* / 255$ if $\max(\mathbf{x}^*) > 1$ |
| **P3** | Load 2D mask | `maskfile` | $\mathbf{M} \in \mathbb{R}^{r \times c}$ | $\mathbf{M} = \texttt{loadmat}(\cdot)[\texttt{'mask'}]$ |
| **P4** | Construct 3D shifted mask | $\mathbf{M}, \Delta, n_C$ | $\boldsymbol{\Phi} \in \mathbb{R}^{r \times c' \times n_C}$ | $\boldsymbol{\Phi}[:, i\Delta : i\Delta+c, i] = \mathbf{M}, \; i=0,\ldots,n_C-1$ |
| **P5** | Apply spectral dispersion | $\mathbf{x}^*, \Delta$ | $\tilde{\mathbf{x}}^* \in \mathbb{R}^{r \times c' \times n_C}$ | $\tilde{\mathbf{x}}^*[:, i\Delta : i\Delta+c, i] = \mathbf{x}^*[:,:,i]$ |
| **P6** | Simulate measurement | $\tilde{\mathbf{x}}^*, \boldsymbol{\Phi}$ | $\mathbf{y} \in \mathbb{R}^{r \times c'}$ | $\mathbf{y} = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_i \odot \tilde{\mathbf{x}}^*_i$ |

---

## 4. Algorithm Steps

### 4.1 Main Algorithm: GAP-TV

```
┌─────────────────────────────────────────────────────────────────┐
│     GENERALIZED ALTERNATING PROJECTION WITH TV (GAP-TV)         │
└─────────────────────────────────────────────────────────────────┘
```

#### Initialization
| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **I1** | Adjoint initialization | $\tilde{\mathbf{x}}^{(0)} = \mathbf{A}^T(\mathbf{y}) = \mathbf{y}_{ext} \odot \boldsymbol{\Phi}$ | $\mathbf{y}, \boldsymbol{\Phi}$ | $\tilde{\mathbf{x}}^{(0)}$ |
| | | where $\mathbf{y}_{ext}[j,k,i] = \mathbf{y}[j,k] \; \forall i$ | | |
| **I2** | Residual initialization | $\mathbf{y}_1^{(0)} = \mathbf{0}$ | — | $\mathbf{y}_1^{(0)}$ |
| **I3** | Compute mask sum | $\boldsymbol{\Phi}_{sum} = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_i$ | $\boldsymbol{\Phi}$ | $\boldsymbol{\Phi}_{sum}$ |
| **I4** | Prevent division by zero | $\boldsymbol{\Phi}_{sum}[j,k] \leftarrow 1$ if $\boldsymbol{\Phi}_{sum}[j,k] = 0$ | $\boldsymbol{\Phi}_{sum}$ | $\boldsymbol{\Phi}_{sum}$ |

#### Per-Iteration Updates (for $k = 0, 1, \ldots, K-1$)

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **G1** | Forward projection | $\bar{\mathbf{y}}^{(k)} = \mathbf{A}(\tilde{\mathbf{x}}^{(k)}) = \sum_{i=1}^{n_C} \boldsymbol{\Phi}_i \odot \tilde{\mathbf{x}}^{(k)}_i$ | $\tilde{\mathbf{x}}^{(k)}, \boldsymbol{\Phi}$ | $\bar{\mathbf{y}}^{(k)}$ |
| **G2** | Accumulate residual | $\mathbf{y}_1^{(k+1)} = \mathbf{y}_1^{(k)} + (\mathbf{y} - \bar{\mathbf{y}}^{(k)})$ | $\mathbf{y}_1^{(k)}, \mathbf{y}, \bar{\mathbf{y}}^{(k)}$ | $\mathbf{y}_1^{(k+1)}$ |
| **G3** | GAP data update | $\tilde{\mathbf{x}}^{(k+\frac{1}{2})} = \tilde{\mathbf{x}}^{(k)} + \lambda \cdot \mathbf{A}^T\left(\frac{\mathbf{y}_1^{(k+1)} - \bar{\mathbf{y}}^{(k)}}{\boldsymbol{\Phi}_{sum}}\right)$ | $\tilde{\mathbf{x}}^{(k)}, \mathbf{y}_1^{(k+1)}, \bar{\mathbf{y}}^{(k)}, \boldsymbol{\Phi}_{sum}$ | $\tilde{\mathbf{x}}^{(k+\frac{1}{2})}$ |
| **G4** | Inverse dispersion | $\mathbf{x}^{(k+\frac{1}{2})} = \mathcal{S}^{-1}(\tilde{\mathbf{x}}^{(k+\frac{1}{2})})$ | $\tilde{\mathbf{x}}^{(k+\frac{1}{2})}, \Delta$ | $\mathbf{x}^{(k+\frac{1}{2})}$ |
| **G5** | TV denoising | $\mathbf{x}^{(k+1)} = \text{TV-Denoise}(\mathbf{x}^{(k+\frac{1}{2})}; \lambda_{TV}, T)$ | $\mathbf{x}^{(k+\frac{1}{2})}, \lambda_{TV}, T$ | $\mathbf{x}^{(k+1)}$ |
| **G6** | Forward dispersion | $\tilde{\mathbf{x}}^{(k+1)} = \mathcal{S}(\mathbf{x}^{(k+1)})$ | $\mathbf{x}^{(k+1)}, \Delta$ | $\tilde{\mathbf{x}}^{(k+1)}$ |

#### Dispersion Operators

**Forward Dispersion** $\mathcal{S}: \mathbb{R}^{r \times c \times n_C} \rightarrow \mathbb{R}^{r \times c' \times n_C}$:
$$
[\mathcal{S}(\mathbf{x})]_{j,k,i} = \begin{cases} \mathbf{x}_{j, k-i\Delta, i} & \text{if } i\Delta \leq k < i\Delta + c \\ 0 & \text{otherwise} \end{cases}
$$

**Inverse Dispersion** $\mathcal{S}^{-1}: \mathbb{R}^{r \times c' \times n_C} \rightarrow \mathbb{R}^{r \times c \times n_C}$:
$$
[\mathcal{S}^{-1}(\tilde{\mathbf{x}})]_{j,k,i} = \tilde{\mathbf{x}}_{j, k+i\Delta, i} \quad \text{(via circular shift and crop)}
$$

---

### 4.2 Subroutine: TV Denoiser (Chambolle's Algorithm)

**Problem solved:**
$$
\mathbf{u}^* = \arg\min_{\mathbf{u}} \frac{1}{2}\|\mathbf{u} - \mathbf{f}\|_2^2 + \frac{1}{\lambda_{TV}}\text{TV}(\mathbf{u})
$$

**Dual formulation:**
$$
\mathbf{u}^* = \mathbf{f} - \frac{1}{\lambda_{TV}} \text{div}(\mathbf{p}^*)
$$

where $\mathbf{p}^* = (\mathbf{p}_1^*, \mathbf{p}_2^*)$ solves the dual problem.

#### Initialization
| Step | Formula |
|------|---------|
| **T-I1** | $\mathbf{p}_1^{(0)} = \mathbf{0} \in \mathbb{R}^{r \times c \times n_C}$ |
| **T-I2** | $\mathbf{p}_2^{(0)} = \mathbf{0} \in \mathbb{R}^{r \times c \times n_C}$ |
| **T-I3** | $\text{divp}^{(0)} = \mathbf{0} \in \mathbb{R}^{r \times c \times n_C}$ |
| **T-I4** | Define index arrays for boundary handling (Neumann BC) |

#### Per-Iteration Updates (for $t = 0, 1, \ldots, T-1$)

| Step | Name | Formula |
|------|------|---------|
| **T1** | Compute intermediate | $\mathbf{z}^{(t)} = \text{divp}^{(t)} - \lambda_{TV} \cdot \mathbf{f}$ |
| **T2** | Horizontal gradient | $\mathbf{z}_1^{(t)} = \nabla_h \mathbf{z}^{(t)}$ where $[\nabla_h \mathbf{z}]_{j,k,i} = \mathbf{z}_{j,k+1,i} - \mathbf{z}_{j,k,i}$ |
| **T3** | Vertical gradient | $\mathbf{z}_2^{(t)} = \nabla_v \mathbf{z}^{(t)}$ where $[\nabla_v \mathbf{z}]_{j,k,i} = \mathbf{z}_{j+1,k,i} - \mathbf{z}_{j,k,i}$ |
| **T4** | Gradient magnitude | $\mathbf{g}^{(t)}_{j,k} = \sqrt{\sum_{i=1}^{n_C} \left([\mathbf{z}_1^{(t)}]_{j,k,i}^2 + [\mathbf{z}_2^{(t)}]_{j,k,i}^2\right)}$ |
| **T5** | Denominator | $\mathbf{d}^{(t)}_{j,k,i} = 1 + \tau \cdot \mathbf{g}^{(t)}_{j,k}$ (broadcast across $i$) |
| **T6** | Update $\mathbf{p}_1$ | $\mathbf{p}_1^{(t+1)} = \frac{\mathbf{p}_1^{(t)} + \tau \cdot \mathbf{z}_1^{(t)}}{\mathbf{d}^{(t)}}$ |
| **T7** | Update $\mathbf{p}_2$ | $\mathbf{p}_2^{(t+1)} = \frac{\mathbf{p}_2^{(t)} + \tau \cdot \mathbf{z}_2^{(t)}}{\mathbf{d}^{(t)}}$ |
| **T8** | Compute divergence | $\text{divp}^{(t+1)} = \nabla_h^* \mathbf{p}_1^{(t+1)} + \nabla_v^* \mathbf{p}_2^{(t+1)}$ |

where adjoint operators (with Neumann boundary conditions):
$$
[\nabla_h^* \mathbf{p}]_{j,k,i} = \mathbf{p}_{j,k,i} - \mathbf{p}_{j,k-1,i}, \quad [\nabla_v^* \mathbf{p}]_{j,k,i} = \mathbf{p}_{j,k,i} - \mathbf{p}_{j-1,k,i}
$$

#### Output
$$
\mathbf{u}^* = \mathbf{f} - \frac{1}{\lambda_{TV}} \text{divp}^{(T)}
$$

---

## 5. Stopping Criterion

| Level | Criterion | Condition |
|-------|-----------|-----------|
| **GAP Outer Loop** | Fixed iteration count | $k \geq K$ (default $K=20$) |
| **TV Inner Loop** | Fixed iteration count | $t \geq T$ (default $T=5$) |

> **Note:** No adaptive convergence criterion (e.g., relative change threshold) is implemented. Termination is purely iteration-based.

---

## 6. Evaluation Metrics

### 6.1 Peak Signal-to-Noise Ratio (PSNR)

$$
\text{PSNR}(\mathbf{x}^*, \hat{\mathbf{x}}) = 20 \cdot \log_{10}\left(\frac{\text{MAX}_I}{\sqrt{\text{MSE}}}\right) \; \text{[dB]}
$$

where:
$$
\text{MSE} = \frac{1}{r \cdot c \cdot n_C} \sum_{j,k,i} \left(\mathbf{x}^*_{j,k,i} - \hat{\mathbf{x}}_{j,k,i}\right)^2
$$

and $\text{MAX}_I = 1.0$ (assuming normalized intensity range $[0, 1]$).

**Special case:** If $\text{MSE} = 0$, return $\text{PSNR} = 100$ dB.

---

## 7. Algorithm Flow Diagram

```
                    ┌──────────────────┐
                    │  Load & Preprocess│
                    │   (P1-P6)        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Initialize      │
                    │  x̃⁽⁰⁾ = Aᵀ(y)    │
                    │  y₁⁽⁰⁾ = 0       │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │     FOR k = 0 to K-1        │
              │  ┌────────────────────────┐ │
              │  │ G1: ȳ = A(x̃)          │ │
              │  │ G2: y₁ += (y - ȳ)     │ │
              │  │ G3: x̃ += λAᵀ((y₁-ȳ)/Φₛ)│ │
              │  │ G4: x = S⁻¹(x̃)        │ │
              │  │ G5: x = TV_Denoise(x) │◄─┼─── Inner Loop (T iters)
              │  │ G6: x̃ = S(x)          │ │
              │  │ Compute PSNR          │ │
              │  └────────────────────────┘ │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Output x⁽ᴷ⁾     │
                    │  Save Results    │
                    └──────────────────┘
```

---

## 8. Computational Complexity

| Operation | Per-Iteration Cost |
|-----------|-------------------|
| Forward $\mathbf{A}(\cdot)$ | $\mathcal{O}(r \cdot c' \cdot n_C)$ |
| Adjoint $\mathbf{A}^T(\cdot)$ | $\mathcal{O}(r \cdot c' \cdot n_C)$ |
| Dispersion $\mathcal{S}, \mathcal{S}^{-1}$ | $\mathcal{O}(r \cdot c \cdot n_C)$ |
| TV Denoiser (per inner iter) | $\mathcal{O}(r \cdot c \cdot n_C)$ |
| **Total per GAP iteration** | $\mathcal{O}(r \cdot c' \cdot n_C + T \cdot r \cdot c \cdot n_C)$ |