# Algorithmic Specification Document: Differentiable Metrology for Lens Parameter Recovery

## 1. Mathematical Formulation

### 1.1 Problem Statement

The objective is to recover optical lens parameters from deflectometry measurements by solving an inverse ray-tracing problem.

### 1.2 Objective Function

The optimization problem is formulated as an unconstrained nonlinear least-squares minimization:

$$\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \left\| \mathbf{p}_{ij}^{\text{sim}}(\boldsymbol{\theta}) - \mathbf{p}_{ij}^{\text{cap}} \right\|_2^2$$

Where:
- $\boldsymbol{\theta} = \{c_0, c_1, d_1, \mathbf{o}, \theta_x, \theta_y\}$ is the parameter vector
- $\mathbf{p}_{ij}^{\text{sim}}(\boldsymbol{\theta}) \in \mathbb{R}^2$ is the simulated ray-display intersection point at pixel $(i,j)$
- $\mathbf{p}_{ij}^{\text{cap}} \in \mathbb{R}^2$ is the captured/measured intersection point
- $\mathcal{V} = \{(i,j) : \text{valid\_cap}_{ij} = 1\}$ is the set of valid pixel indices
- $|\mathcal{V}|$ is the cardinality of the valid pixel set

### 1.3 Forward Model (Ray Tracing)

The forward operator $\mathcal{F}: \boldsymbol{\theta} \mapsto \mathbf{P}^{\text{sim}}$ is defined by the differentiable ray-tracing operation:

$$\mathbf{p}_{ij}^{\text{sim}}(\boldsymbol{\theta}) = \Pi_{xy}\left[ \mathcal{T}_{\text{ray}}(\mathbf{r}_{ij}^{\text{cam}}; \boldsymbol{\theta}) \right]$$

Where:
- $\mathbf{r}_{ij}^{\text{cam}}$ is the ray originating from camera pixel $(i,j)$
- $\mathcal{T}_{\text{ray}}$ is the ray-tracing operator through the lens system
- $\Pi_{xy}$ projects the 3D intersection point to 2D display coordinates (first two components)

### 1.4 Decomposition of Loss Terms

| Term | Formula | Description |
|------|---------|-------------|
| **Data Fidelity** | $\mathcal{L}_{\text{data}} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \left\| \mathbf{p}_{ij}^{\text{sim}} - \mathbf{p}_{ij}^{\text{cap}} \right\|_2^2$ | L2 displacement error |
| **Regularization** | $\mathcal{R}(\boldsymbol{\theta}) = 0$ | None explicitly applied (implicit via Adam) |
| **Total** | $\mathcal{L} = \mathcal{L}_{\text{data}}$ | — |

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Symbol | Code Reference | Shape | Dtype | Initialization | Physical Meaning |
|--------|----------------|-------|-------|----------------|------------------|
| $c_0$ | `surfaces[0].c` | $(1,)$ | `float32` | $0.0$ | Curvature of 1st lens surface (mm⁻¹) |
| $c_1$ | `surfaces[1].c` | $(1,)$ | `float32` | $0.0$ | Curvature of 2nd lens surface (mm⁻¹) |
| $d_1$ | `surfaces[1].d` | $(1,)$ | `float32` | $3.0$ | Lens thickness (mm) |
| $\mathbf{o}$ | `lensgroup.origin` | $(3,)$ | `float32` | Computed from `_compute_mount_geometry` | Lens 3D origin position (mm) |
| $\theta_x$ | `lensgroup.theta_x` | $(1,)$ | `float32` | $0.0$ | Lens tilt angle about X-axis (rad) |
| $\theta_y$ | `lensgroup.theta_y` | $(1,)$ | `float32` | $0.0$ | Lens tilt angle about Y-axis (rad) |

**Parameter Vector:**
$$\boldsymbol{\theta} = [c_0, c_1, d_1, o_x, o_y, o_z, \theta_x, \theta_y]^T \in \mathbb{R}^8$$

### 2.2 Auxiliary/Intermediate Variables

| Symbol | Code Reference | Shape | Dtype | Description |
|--------|----------------|-------|-------|-------------|
| $\mathbf{P}^{\text{sim}}$ | `ps` (from `forward_operator`) | $(N_{\text{cam}}, H, W, 2)$ | `float32` | Simulated intersection coordinates |
| $\mathbf{I}^{\text{sim}}$ | `I` (from render mode) | $(N_{\text{cam}}, H, W)$ | `float32` | Rendered images |
| $\mathbf{b}$ | `func_yref_y` output | $(N_{\text{cam}}, H, W, 2)$ | `float32` | Residual: $\mathbf{b}_{ij} = \mathbf{p}_{ij}^{\text{cap}} - \mathbf{p}_{ij}^{\text{sim}}$ |
| $a$, $b$, $\psi$ | Fringe solver outputs | $(N_{\text{cam}}, \cdot, H, W)$ | `float32` | Fringe pattern decomposition |
| $C$ | Calibration matrix | — | `float32` | Camera calibration data |

### 2.3 Hyperparameters/Constants

| Symbol | Code Reference | Value | Description |
|--------|----------------|-------|-------------|
| $N_{\text{iter}}$ | `maxit` | $5$ | Maximum optimization iterations |
| $(H, W)$ | `filmsize` | $(768, 768)$ | Cropped image dimensions (pixels) |
| $(H_0, W_0)$ | Original size | $(2048, 2048)$ | Original image dimensions |
| $\mathbf{T}$ | `Ts` | $[70, 100, 110]$ | Sinusoid periods (pixels) |
| $t$ | Period index | $0$ | Selected period index |
| $\alpha$ | Rotation angle `angle` | $0.0$ | Lens rotation angle (rad) |
| $\boldsymbol{\delta}_{\text{crop}}$ | `crop_offset` | $((2048-768)/2, (2048-768)/2) = (640, 640)$ | Crop offset |
| $\boldsymbol{\delta}_{\text{origin}}$ | `origin_shift` | $(0, 0, 0)$ | Origin shift vector |
| $\boldsymbol{\tau}$ | `texture_shift` | $(0.0, 1.1106231)$ | Display texture shift |
| Optimizer | `option` | `'Adam'` | Optimization algorithm |
| $R$ | Regularization matrix | `'I'` (Identity) | Regularization type |

### 2.4 Observed Data/Inputs

| Symbol | Code Reference | Shape | Dtype | Source |
|--------|----------------|-------|-------|--------|
| $\mathbf{I}^{\text{raw}}$ | `imgs` | $(N_{\text{orient}}, N_{\text{cam}}, N_{\text{phase}}, H_0, W_0)$ | `float32` | `data_new.npz['imgs']` |
| $\mathbf{I}^{\text{ref}}$ | `refs` | $(N_{\text{orient}}, N_{\text{cam}}, N_{\text{phase}}, H_0, W_0)$ | `float32` | `data_new.npz['refs']` |
| $\mathbf{P}^{\text{cap}}$ | `ps_cap` | $(N_{\text{cam}}, H, W, 2)$ | `float32` | Solved from phase-shifting |
| $\mathbf{M}$ | `valid_cap` | $(N_{\text{cam}}, H, W)$ | `bool` | Valid pixel mask |
| $\mathbf{I}_0$ | `I0` | $(N_{\text{cam}}, H, W)$ | `float32` | Preprocessed measurement images |
| Lens file | `lens_name` | — | — | `'ThorLabs/LE1234-A.txt'` |

---

## 3. Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Step | Operation | Input | Output | Formula/Description |
|------|-----------|-------|--------|---------------------|
| **3.1** | Initialize DiffMetrology | Calibration paths | `DM` object | Load camera calibration, rotation, LUT |
| **3.2** | Set film size | — | `filmsize` | $\mathbf{f} = (768, 768)$ |
| **3.3** | Compute crop offset | `filmsize` | `crop_offset` | $\boldsymbol{\delta} = \lfloor (2048 - \mathbf{f}) / 2 \rfloor = (640, 640)$ |
| **3.4** | Load raw data | NPZ file | `imgs`, `refs` | $\mathbf{I}^{\text{raw}}, \mathbf{I}^{\text{ref}} \leftarrow \text{load}(\texttt{data\_new.npz})$ |
| **3.5** | Crop images | `imgs`, `refs` | Cropped arrays | $\mathbf{I}_{\text{crop}} = \mathbf{I}[\ldots, \delta_0:\delta_0+H, \delta_1:\delta_1+W]$ |
| **3.6** | Solve intersections | `imgs`, `refs`, `Ts` | `ps_cap`, `valid_cap`, `C` | Phase-shifting deflectometry solver |
| **3.7** | Load display texture | Sinusoid images | Texture | $\mathbf{T}_{\text{disp}} = \sum_x \frac{\mathbf{I}_x}{\max(\mathbf{I}_x)}$ |
| **3.8** | Compute mount geometry | Calibration `C` | `origin` | $\mathbf{o} \leftarrow \texttt{\_compute\_mount\_geometry}(C)$ |
| **3.9** | Solve fringe parameters | `imgs` | $a, b, \psi$ | Fringe analysis: $I = a + b\cos(\psi)$ |
| **3.10** | Compute reference image | `imgs`, $a$, `valid_cap` | `I0` | See formula below |

**Step 3.10 Detail - Reference Image Computation:**

$$\mathbf{I}_{\text{sub}} = \mathbf{I}^{\text{raw}}[0, x, \ldots] - a[:, 0, \ldots]$$

$$\mathbf{I}_{\text{sum}} = \sum_x \mathbf{I}_{\text{sub}}^{(x)}$$

$$\mathbf{I}_0 = \mathbf{M} \cdot |X| \cdot \frac{\mathbf{I}_{\text{sum}} - \min(\mathbf{I}_{\text{sum}})}{\max(\mathbf{I}_{\text{sum}}) - \min(\mathbf{I}_{\text{sum}})}$$

Where $X = \{0\}$ (single phase index), $|X| = 1$, and $\mathbf{M}$ is `valid_cap`.

---

## 4. Algorithm Steps

### 4.1 Initialization Phase

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **4.1.1** | Initialize curvatures | $c_0^{(0)} = 0, \quad c_1^{(0)} = 0$ | — | $c_0, c_1$ |
| **4.1.2** | Initialize thickness | $d_1^{(0)} = 3.0$ | — | $d_1$ |
| **4.1.3** | Initialize tilt angles | $\theta_x^{(0)} = 0, \quad \theta_y^{(0)} = 0$ | — | $\theta_x, \theta_y$ |
| **4.1.4** | Set origin | $\mathbf{o}^{(0)} \leftarrow \texttt{computed}$ | Calibration | $\mathbf{o}$ |
| **4.1.5** | Update lens group | $\texttt{lensgroup.update()}$ | All params | Internal state |

### 4.2 Main Optimization Loop (Adam)

**For** $k = 0, 1, \ldots, N_{\text{iter}}-1$ **do:**

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **4.2.1** | Forward pass | $\mathbf{P}^{(k)} = \mathcal{F}(\boldsymbol{\theta}^{(k)})$ | $\boldsymbol{\theta}^{(k)}$ | $\mathbf{P}^{\text{sim},(k)}$ |
| **4.2.2** | Compute loss | $\mathcal{L}^{(k)} = \frac{1}{|\mathcal{V}|} \sum_{(i,j)\in\mathcal{V}} \|\mathbf{p}_{ij}^{(k)} - \mathbf{p}_{ij}^{\text{cap}}\|_2^2$ | $\mathbf{P}^{(k)}, \mathbf{P}^{\text{cap}}$ | $\mathcal{L}^{(k)}$ |
| **4.2.3** | Compute residual | $\mathbf{b}_{ij}^{(k)} = \mathbf{M}_{ij} \cdot (\mathbf{p}_{ij}^{\text{cap}} - \mathbf{p}_{ij}^{(k)})$ | $\mathbf{P}^{(k)}, \mathbf{P}^{\text{cap}}, \mathbf{M}$ | $\mathbf{b}^{(k)}$ |
| **4.2.4** | Handle NaN | $\mathbf{b}_{ij}^{(k)} \leftarrow 0 \text{ where } \text{isnan}(\mathbf{b}_{ij}^{(k)})$ | $\mathbf{b}^{(k)}$ | $\mathbf{b}^{(k)}$ (cleaned) |
| **4.2.5** | Backward pass | $\mathbf{g}^{(k)} = \nabla_{\boldsymbol{\theta}} \mathcal{L}^{(k)}$ | $\mathcal{L}^{(k)}$ | Gradient $\mathbf{g}^{(k)}$ |
| **4.2.6** | Adam update | $\boldsymbol{\theta}^{(k+1)} = \text{Adam}(\boldsymbol{\theta}^{(k)}, \mathbf{g}^{(k)})$ | $\boldsymbol{\theta}^{(k)}, \mathbf{g}^{(k)}$ | $\boldsymbol{\theta}^{(k+1)}$ |
| **4.2.7** | Update lens | $\texttt{lensgroup.update()}$ | $\boldsymbol{\theta}^{(k+1)}$ | Internal state |

**End For**

### 4.3 Adam Update Equations (Standard)

$$m^{(k+1)} = \beta_1 m^{(k)} + (1-\beta_1) \mathbf{g}^{(k)}$$

$$v^{(k+1)} = \beta_2 v^{(k)} + (1-\beta_2) (\mathbf{g}^{(k)})^2$$

$$\hat{m} = \frac{m^{(k+1)}}{1 - \beta_1^{k+1}}, \quad \hat{v} = \frac{v^{(k+1)}}{1 - \beta_2^{k+1}}$$

$$\boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

Default PyTorch values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta$ (learning rate, set internally by `DM.solve`).

---

## 5. Stopping Criterion

| Criterion | Formula | Value |
|-----------|---------|-------|
| **Maximum Iterations** | $k \geq N_{\text{iter}}$ | $N_{\text{iter}} = 5$ |

**Note:** No explicit convergence tolerance is specified in the code. The algorithm terminates solely based on iteration count.

---

## 6. Evaluation Metrics

### 6.1 Primary Metric: Mean Displacement Error

$$E = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \sqrt{\sum_{d=1}^{2} \left( p_{ij,d}^{\text{sim}} - p_{ij,d}^{\text{cap}} \right)^2}$$

**Code:** 
```python
E = torch.sqrt(torch.sum(T[valid_cap, ...]**2, axis=-1)).mean()
```

**Units:** Reported in micrometers (µm) via multiplication by $10^3$ (assuming base units are mm).

### 6.2 Recovered Parameters: Radius of Curvature

$$R_i = \frac{1}{c_i} \quad \text{for } i \in \{0, 1\}$$

### 6.3 Image-Domain Metrics (Visualization)

| Metric | Formula | Description |
|--------|---------|-------------|
| Measurement Image | $\mathbf{I}_0$ | Normalized captured image |
| Modeled Image | $\mathbf{I}^{\text{sim}} = \mathcal{F}_{\text{render}}(\boldsymbol{\theta})$ | Ray-traced rendered image |
| Error Map | $\mathbf{E}_{\text{img}} = \mathbf{I}_0 - \mathbf{I}^{\text{sim}}$ | Pixel-wise difference |

### 6.4 Loss History

$$\mathcal{L}^{(k)} \text{ for } k = 0, 1, \ldots, N_{\text{iter}}-1$$

Stored in list `ls` returned by `run_inversion`.

---

## 7. Algorithm Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ALGORITHM FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────────┐     ┌───────────────────────┐   │
│  │ Raw Images   │────▶│  Preprocessing   │────▶│  P_cap, valid_cap     │   │
│  │ imgs, refs   │     │  (Phase-shift)   │     │  (Measured Points)    │   │
│  └──────────────┘     └──────────────────┘     └───────────┬───────────┘   │
│                                                             │               │
│  ┌──────────────┐     ┌──────────────────┐                 │               │
│  │ θ⁽⁰⁾ = 0    │────▶│  Forward Model   │────▶ P_sim⁽ᵏ⁾   │               │
│  │ (Initial)    │     │  Ray Tracing     │        │        │               │
│  └──────────────┘     └──────────────────┘        │        │               │
│         ▲                                         ▼        ▼               │
│         │              ┌──────────────────────────────────────┐            │
│         │              │     Loss: L = mean(‖P_sim - P_cap‖²) │            │
│         │              └──────────────────┬───────────────────┘            │
│         │                                 │                                 │
│         │              ┌──────────────────▼───────────────────┐            │
│         │              │     Gradient: g = ∇_θ L              │            │
│         │              └──────────────────┬───────────────────┘            │
│         │                                 │                                 │
│         │              ┌──────────────────▼───────────────────┐            │
│         └──────────────│     Adam Update: θ⁽ᵏ⁺¹⁾              │            │
│                        └──────────────────────────────────────┘            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STOP when k ≥ maxit                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: θ* = {c₀*, c₁*, d₁*, o*, θₓ*, θᵧ*}, E (error in µm)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Notation Reference Table

| Symbol | Meaning | Units |
|--------|---------|-------|
| $c_i$ | Surface curvature | mm⁻¹ |
| $R_i = 1/c_i$ | Radius of curvature | mm |
| $d_i$ | Surface separation / thickness | mm |
| $\mathbf{o}$ | Lens group origin | mm |
| $\theta_x, \theta_y$ | Tilt angles | rad |
| $\mathbf{p}_{ij}$ | 2D intersection point at pixel $(i,j)$ | mm |
| $\mathbf{M}$ | Valid pixel mask | binary |
| $\mathcal{V}$ | Set of valid pixel indices | — |
| $\mathcal{F}$ | Forward operator (ray trace) | — |
| $\mathcal{L}$ | Loss function | mm² |
| $E$ | Mean displacement error | mm (reported in µm) |