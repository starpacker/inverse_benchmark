## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Differentiable Deflectometry Lens Parameter Estimation
- **algorithm_family**: Nonlinear Least Squares / First-Order Gradient Optimization (Adam)
- **summary**: Recovers lens geometric parameters (surface curvatures, thickness, position, tilt) by minimizing the squared displacement between ray-traced intersection points and measured deflectometry observations via Adam optimization.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{c}_0, \mathbf{c}_1, d, \mathbf{o}, \theta_x, \theta_y} \quad \mathcal{L}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \left\| \mathbf{p}_{ij}(\boldsymbol{\theta}) - \hat{\mathbf{p}}_{ij} \right\|_2^2$$

where $\boldsymbol{\theta} = \{c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y\}$ is the parameter vector.

### data_fidelity_term
$$\mathcal{L}_{\text{data}}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \sum_{k \in \{x,y\}} \left( p_{ij,k}(\boldsymbol{\theta}) - \hat{p}_{ij,k} \right)^2$$

where:
- $\mathbf{p}_{ij}(\boldsymbol{\theta}) \in \mathbb{R}^2$: predicted 2D intersection point on display for pixel $(i,j)$ via differentiable ray tracing
- $\hat{\mathbf{p}}_{ij} \in \mathbb{R}^2$: measured intersection point from phase-shifting deflectometry
- $\mathcal{V} = \{(i,j) : \text{valid\_cap}[i,j] = 1\}$: set of valid pixel indices

### regularization_term
None

### constraints
None (unconstrained optimization)

---

## 2. VARIABLES

### Primal Variables (Optimized)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $c_0$ | primal | $(1,)$ | float32 | $0.0$ | Curvature of lens surface 0 (reciprocal of radius) |
| $c_1$ | primal | $(1,)$ | float32 | $0.0$ | Curvature of lens surface 1 |
| $d$ | primal | $(1,)$ | float32 | $3.0$ | Lens thickness (distance between surfaces) |
| $\mathbf{o}$ | primal | $(3,)$ | float32 | Computed via `_compute_mount_geometry(C)` | Lens group origin position $[o_x, o_y, o_z]$ |
| $\theta_x$ | primal | $(1,)$ | float32 | $0.0$ | Lens tilt angle about x-axis |
| $\theta_y$ | primal | $(1,)$ | float32 | $0.0$ | Lens tilt angle about y-axis |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\hat{\mathbf{p}}$ (`ps_cap`) | observations | $(N_{\text{cam}}, H, W, 2)$ | float32 | Solved from fringe analysis | Measured intersection points on display |
| $\mathbf{I}_{\text{imgs}}$ | observations | $(N_{\text{cam}}, N_{\text{phase}}, H, W)$ | float32 | Loaded from `data_new.npz` | Raw phase-shifted measurement images |
| $\mathbf{I}_{\text{refs}}$ | observations | $(N_{\text{cam}}, N_{\text{phase}}, H, W)$ | float32 | Loaded from `data_new.npz` | Reference images (no lens) |

### Auxiliary Variables

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathcal{M}$ (`valid_cap`) | auxiliary | $(N_{\text{cam}}, H, W)$ | bool | From `solve_for_intersections` | Binary validity mask for each pixel |
| $\mathbf{p}$ (`ps`) | auxiliary | $(N_{\text{cam}}, H, W, 2)$ | float32 | Computed each forward pass | Current predicted intersection points |
| $\mathbf{C}$ | auxiliary | $(N_{\text{cam}}, 3, 3)$ | float32 | From `solve_for_intersections` | Camera calibration matrices |
| $\mathbf{I}_0$ | auxiliary | $(N_{\text{cam}}, H, W)$ | float32 | Preprocessed from measurements | Normalized reference images for visualization |
| $\mathbf{b}$ | auxiliary | $(N_{\text{cam}}, H, W, 2)$ | float32 | $\mathcal{M} \odot (\hat{\mathbf{p}} - \mathbf{p})$ | Residual vector (NaN set to 0) |

### Constants

| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $H_{\text{film}}$ | constants | scalar | int | $768$ | Film height after cropping |
| $W_{\text{film}}$ | constants | scalar | int | $768$ | Film width after cropping |
| $H_{\text{orig}}$ | constants | scalar | int | $2048$ | Original image height |
| $W_{\text{orig}}$ | constants | scalar | int | $2048$ | Original image width |
| $\mathbf{T}_s$ | constants | $(3,)$ | int | $[70, 100, 110]$ | Sinusoid periods for phase shifting |
| $t$ | constants | scalar | int | $0$ | Period index selector |
| $\alpha$ | constants | scalar | float | $0.0$ | Rotation angle (radians) |
| $\mathbf{s}_{\text{origin}}$ | constants | $(3,)$ | float32 | $[0.0, 0.0, 0.0]$ | Initial origin shift |
| $\text{scale}$ | constants | scalar | float | $1.0$ | DiffMetrology scale factor |
| $\mathbf{s}_{\text{texture}}$ | constants | $(2,)$ | float32 | $[0.0, 1.1106231]$ | Texture shift for display (LE1234-A specific) |
| $N_{\text{iter}}$ (`maxit`) | constants | scalar | int | $5$ | Maximum optimization iterations |

---

## 3. PREPROCESSING PIPELINE

| Step Order | Step Name | Formula | Inputs | Outputs |
|------------|-----------|---------|--------|---------|
| 1 | Device Initialization | $\text{device} \leftarrow \texttt{dm.init()}$ | None | `device`: torch.device |
| 2 | DiffMetrology Construction | $\text{DM} \leftarrow \texttt{DiffMetrology}(\text{calibration}, \text{rotation}, \text{LUT}, \mathbf{s}_{\text{origin}}, \text{scale})$ | Paths, constants | `DM`: DiffMetrology object |
| 3 | Compute Crop Offset | $\mathbf{c}_{\text{off}} = \left\lfloor \frac{(2048, 2048) - (768, 768)}{2} \right\rfloor = (640, 640)$ | $H_{\text{orig}}, W_{\text{orig}}, H_{\text{film}}, W_{\text{film}}$ | `crop_offset`: $(2,)$ int |
| 4 | Update Camera Film Size | $\forall \text{cam}: \text{cam.filmsize} \leftarrow (768, 768)$, $\text{cam.crop\_offset} \leftarrow \mathbf{c}_{\text{off}}$ | `crop_offset` | Modified camera objects |
| 5 | Load Lens Geometry | $\text{DM.lensgroup} \leftarrow \texttt{load\_file}(\text{`ThorLabs/LE1234-A.txt'})$ | Lens file path | Lens surfaces with ground truth parameters |
| 6 | Crop Measurements | $\mathbf{I}'[..., i, j] = \mathbf{I}[..., i + 640, j + 640]$ for $i,j \in [0, 768)$ | `imgs`: $(N_c, N_p, 2048, 2048)$, `refs` | `imgs`: $(N_c, N_p, 768, 768)$, `refs`: same |
| 7 | Solve Intersections | $(\hat{\mathbf{p}}, \mathcal{M}, \mathbf{C}) \leftarrow \texttt{solve\_for\_intersections}(\mathbf{I}_{\text{imgs}}, \mathbf{I}_{\text{refs}}, \mathbf{T}_s[0:])$ | Cropped images, periods | `ps_cap`, `valid_cap`, `C` |
| 8 | Load Display Texture | $\mathbf{I}_{\text{tex}} = \sum_x \frac{\text{imread}(x)}{\max(\text{imread}(x))}$ for $x \in \{0\}$ | Sinusoid image path | Normalized grayscale texture |
| 9 | Set Texture | $\texttt{DM.set\_texture}(\mathbf{I}_{\text{tex}})$ | `ims` | Display texture configured |
| 10 | Set Texture Shift | $\text{DM.screen.texture\_shift} \leftarrow [0.0, 1.1106231]$ | Hardcoded constant | Screen texture shift set |
| 11 | Compute Mount Geometry | $\mathbf{o}_{\text{init}} \leftarrow \texttt{\_compute\_mount\_geometry}(\mathbf{C})$ | Calibration `C` | Initial origin estimate |
| 12 | Set Lens Origin | $\text{DM.lensgroup.origin} \leftarrow \mathbf{o}_{\text{init}}$; $\texttt{update()}$ | Origin estimate | Lens group positioned |
| 13 | Fringe Analysis | $(a, b, \psi) \leftarrow \texttt{FR.solve}(\mathbf{I}_{\text{imgs}})$ | Measurement images | Fringe parameters |
| 14 | Compute Reference Image | $\mathbf{I}_0 = \mathcal{M} \cdot \frac{\sum_x (\mathbf{I}_{\text{imgs}}[0,x] - a[:,0]) - \min(\cdot)}{\max(\cdot) - \min(\cdot)}$ | Images, fringe DC, mask | Normalized reference: $(N_c, H, W)$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| I.1 | Reset Surface 0 Curvature | $c_0 \leftarrow 0.0$ | None | `surfaces[0].c` | Flat initial guess |
| I.2 | Reset Surface 1 Curvature | $c_1 \leftarrow 0.0$ | None | `surfaces[1].c` | Flat initial guess |
| I.3 | Set Initial Thickness | $d \leftarrow 3.0$ | None | `surfaces[1].d` | Fixed initial thickness [mm] |
| I.4 | Reset X-Tilt | $\theta_x \leftarrow 0.0$ | None | `lensgroup.theta_x` | Zero tilt |
| I.5 | Reset Y-Tilt | $\theta_y \leftarrow 0.0$ | None | `lensgroup.theta_y` | Zero tilt |
| I.6 | Update Lens Group | $\texttt{lensgroup.update()}$ | Modified parameters | Internal transforms updated | Recomputes ray transfer matrices |
| I.7 | Define Differentiable Parameters | $\mathcal{D} \leftarrow \{$`lensgroup.surfaces[0].c`, `lensgroup.surfaces[1].c`, `lensgroup.surfaces[1].d`, `lensgroup.origin`, `lensgroup.theta_x`, `lensgroup.theta_y`$\}$ | Parameter names | Differentiable parameter set | 6 parameter groups |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (iteration $k = 1, \ldots, N_{\text{iter}}$)

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| L.1 | Forward Ray Tracing | $\mathbf{p}^{(k)} = \texttt{trace}(\text{with\_element}=\text{True}, \text{mask}=\mathcal{M}, \text{angles}=\alpha)[0][..., 0:2]$ | Current $\boldsymbol{\theta}^{(k-1)}$, $\mathcal{M}$, $\alpha$ | `ps`: $(N_c, H, W, 2)$ | Differentiable ray-surface intersection |
| L.2 | Apply Validity Mask | $\mathbf{p}_{\text{valid}}^{(k)} = \mathbf{p}^{(k)}[\mathcal{M}]$ | `ps`, `valid_cap` | Valid predictions: $(|\mathcal{V}|, 2)$ | Index selection |
| L.3 | Compute Squared Residuals | $\mathbf{r}^{(k)} = \sum_{d \in \{x,y\}} (\mathbf{p}_{\text{valid}}^{(k)} - \hat{\mathbf{p}}_{\text{valid}})_d^2$ | Predictions, observations | Residual vector: $(|\mathcal{V}|,)$ | Per-pixel squared error |
| L.4 | Compute Loss | $\mathcal{L}^{(k)} = \frac{1}{|\mathcal{V}|} \sum_i r_i^{(k)}$ | Residuals | Scalar loss | Mean squared displacement |
| L.5 | Compute Jacobian Residual | $\mathbf{b}^{(k)} = \mathcal{M}[..., \text{None}] \odot (\hat{\mathbf{p}} - \mathbf{p}^{(k)})$; $\mathbf{b}^{(k)}[\text{NaN}] \leftarrow 0$ | Predictions, observations, mask | Residual tensor: $(N_c, H, W, 2)$ | NaN handling for numerical stability |
| L.6 | Backward Pass | $\nabla_{\boldsymbol{\theta}} \mathcal{L}^{(k)} \leftarrow \texttt{autograd}(\mathcal{L}^{(k)}, \mathcal{D})$ | Loss, differentiable params | Gradients for each parameter | Automatic differentiation |
| L.7 | Adam Update | $\boldsymbol{\theta}^{(k)} \leftarrow \texttt{Adam}(\boldsymbol{\theta}^{(k-1)}, \nabla_{\boldsymbol{\theta}} \mathcal{L}^{(k)})$ | Previous params, gradients | Updated parameters | Adam optimizer with default hyperparameters |
| L.8 | Update Lens Group | $\texttt{lensgroup.update()}$ | New parameters | Internal state updated | Refresh geometric transforms |
| L.9 | Record Loss | $\mathbf{l}[k] \leftarrow \mathcal{L}^{(k)}$ | Current loss | Loss history | For convergence monitoring |

**END_LOOP**

### 4.3 FINALIZATION

| Step | Step Name | Formula | Inputs | Outputs | Computational Note |
|------|-----------|---------|--------|---------|-------------------|
| F.1 | Return Results | Return $(\mathbf{l}, \text{DM})$ | Loss history, updated DM | Final outputs | Loss list and optimized model |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k \geq N_{\text{iter}}$
- **parameters**: 
  - $N_{\text{iter}} = 5$ (default `maxit` value)
- **check_frequency**: After each complete iteration (steps L.1–L.9)

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Mean Squared Displacement Loss | $\mathcal{L} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \|\mathbf{p}_{ij} - \hat{\mathbf{p}}_{ij}\|_2^2$ | Training objective (units: mm²) |
| Root Mean Displacement Error | $E = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \|\mathbf{p}_{ij} - \hat{\mathbf{p}}_{ij}\|_2$ | Mean Euclidean displacement (units: mm); reported as $E \times 10^3$ in μm |
| Radius of Curvature | $R_i = \frac{1}{c_i}$ for $i \in \{0, 1\}$ | Physical lens parameter derived from optimized curvature |
| Image Residual | $\Delta \mathbf{I} = \mathbf{I}_0 - \mathbf{I}_{\text{rendered}}$ | Per-pixel intensity difference for visual inspection |