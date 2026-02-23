## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Differentiable Ray Tracing Lens Metrology Inversion",
  "algorithm_family": "Gradient-Based Optimization (Adam) for Inverse Problems",
  "summary": "Recovers lens geometric parameters (curvatures, thickness, tilts, position) by minimizing the L2 discrepancy between measured and ray-traced intersection points on a calibrated display using automatic differentiation and the Adam optimizer."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{\theta}} \quad \mathcal{L}(\mathbf{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \left\| \mathbf{p}_{i,j}(\mathbf{\theta}) - \hat{\mathbf{p}}_{i,j} \right\|_2^2$$

where $\mathbf{\theta} = \{c_0, c_1, d_1, \mathbf{o}, \theta_x, \theta_y\}$ is the parameter vector, $\mathcal{V}$ is the set of valid pixel indices, and $|\mathcal{V}|$ is its cardinality.

**data_fidelity_term**:
$$\mathcal{D}(\mathbf{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \sum_{k \in \{0,1\}} \left( p_{i,j,k}(\mathbf{\theta}) - \hat{p}_{i,j,k} \right)^2$$

where $p_{i,j,k}$ is the $k$-th coordinate (x or y) of the predicted intersection point at valid pixel $(i,j)$, and $\hat{p}_{i,j,k}$ is the corresponding measured coordinate.

**regularization_term**: None

**constraints**: None (unconstrained optimization)

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $c_0$ | $(1,)$ | float32 | $0.0$ | Curvature of lens surface 0 (reciprocal of radius) |
| $c_1$ | $(1,)$ | float32 | $0.0$ | Curvature of lens surface 1 |
| $d_1$ | $(1,)$ | float32 | $3.0$ | Thickness/distance parameter of lens surface 1 [mm] |
| $\mathbf{o}$ | $(3,)$ | float32 | Computed from `_compute_mount_geometry(C)` | Lens group origin position $[o_x, o_y, o_z]$ [mm] |
| $\theta_x$ | $(1,)$ | float32 | $0.0$ | Lens tilt angle about x-axis [rad] |
| $\theta_y$ | $(1,)$ | float32 | $0.0$ | Lens tilt angle about y-axis [rad] |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{m}_{c_0}$ | $(1,)$ | float32 | $0.0$ | Adam first moment estimate for $c_0$ |
| $\mathbf{v}_{c_0}$ | $(1,)$ | float32 | $0.0$ | Adam second moment estimate for $c_0$ |
| $\mathbf{m}_{c_1}$ | $(1,)$ | float32 | $0.0$ | Adam first moment estimate for $c_1$ |
| $\mathbf{v}_{c_1}$ | $(1,)$ | float32 | $0.0$ | Adam second moment estimate for $c_1$ |
| $\mathbf{m}_{d_1}$ | $(1,)$ | float32 | $0.0$ | Adam first moment estimate for $d_1$ |
| $\mathbf{v}_{d_1}$ | $(1,)$ | float32 | $0.0$ | Adam second moment estimate for $d_1$ |
| $\mathbf{m}_{\mathbf{o}}$ | $(3,)$ | float32 | $\mathbf{0}$ | Adam first moment estimate for origin |
| $\mathbf{v}_{\mathbf{o}}$ | $(3,)$ | float32 | $\mathbf{0}$ | Adam second moment estimate for origin |
| $\mathbf{m}_{\theta_x}$ | $(1,)$ | float32 | $0.0$ | Adam first moment estimate for $\theta_x$ |
| $\mathbf{v}_{\theta_x}$ | $(1,)$ | float32 | $0.0$ | Adam second moment estimate for $\theta_x$ |
| $\mathbf{m}_{\theta_y}$ | $(1,)$ | float32 | $0.0$ | Adam first moment estimate for $\theta_y$ |
| $\mathbf{v}_{\theta_y}$ | $(1,)$ | float32 | $0.0$ | Adam second moment estimate for $\theta_y$ |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{P}$ | $(N_{cam}, H, W, 2)$ | float32 | Computed | Predicted intersection points on display |
| $\mathcal{L}^{(k)}$ | $(1,)$ | float32 | Computed | Loss value at iteration $k$ |
| $\mathbf{g}_{\theta}$ | Same as $\theta$ | float32 | Computed | Gradient of loss w.r.t. each parameter |
| $\hat{\mathbf{m}}_{\theta}$ | Same as $\theta$ | float32 | Computed | Bias-corrected first moment |
| $\hat{\mathbf{v}}_{\theta}$ | Same as $\theta$ | float32 | Computed | Bias-corrected second moment |
| $\mathbf{ls}$ | $(K_{max},)$ | float32 | Empty list | Loss history |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $H$ | $768$ | Film height after cropping [pixels] |
| $W$ | $768$ | Film width after cropping [pixels] |
| $N_{cam}$ | $2$ | Number of cameras |
| $\mathbf{s}_{offset}$ | $[640, 640]$ | Crop offset from original 2048×2048 image |
| $\alpha$ | Default Adam (typically $0.001$) | Learning rate |
| $\beta_1$ | Default Adam ($0.9$) | Exponential decay rate for first moment |
| $\beta_2$ | Default Adam ($0.999$) | Exponential decay rate for second moment |
| $\epsilon$ | Default Adam ($10^{-8}$) | Numerical stability constant |
| $K_{max}$ | $5$ | Maximum number of iterations |
| $\phi$ | $0.0$ | Rotation angle [rad] |
| $\mathbf{T}$ | $[70, 100, 110]$ | Sinusoid periods for phase shifting |
| $t$ | $0$ | Index into period array (uses $T=70$) |
| $\mathbf{s}_{tex}$ | $[0.0, 1.1106231]$ | Texture shift on screen |
| $\mathbf{o}_{shift}$ | $[0.0, 0.0, 0.0]$ | Origin shift |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $\mathbf{I}_{raw}$ | $(N_{orient}, N_{phase}, N_{cam}, 2048, 2048)$ | float32 | `data_new.npz['imgs']` | Raw captured images |
| $\mathbf{R}_{raw}$ | $(N_{orient}, N_{phase}, N_{cam}, 2048, 2048)$ | float32 | `data_new.npz['refs']` | Reference images without lens |
| $\hat{\mathbf{P}}$ | $(N_{cam}, H, W, 2)$ | float32 | Phase unwrapping | Measured intersection points (ps_cap) |
| $\mathbf{V}$ | $(N_{cam}, H, W)$ | bool | Phase unwrapping | Valid pixel mask (valid_cap) |
| $\mathbf{I}_0$ | $(N_{cam}, H, W)$ | float32 | Preprocessed | Normalized reference images for visualization |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| P1 | Initialize DiffMetrology | $\text{DM} = \texttt{DiffMetrology}(\text{calibration}, \text{rotation}, \text{lut}, \mathbf{o}_{shift}, 1.0, \text{device})$ | Calibration files | DM object | Calibration files exist and are valid |
| P2 | Compute crop offset | $\mathbf{s}_{offset} = \lfloor (2048 - 768) / 2 \rfloor \cdot \mathbf{1} = [640, 640]$ | Original size $(2048)$, target size $(768)$ | $\mathbf{s}_{offset}: (2,)$ | Integer division |
| P3 | Crop raw images | $\mathbf{I} = \mathbf{I}_{raw}[..., s_0:s_0+H, s_1:s_1+W]$ | $\mathbf{I}_{raw}: (N_o, N_p, N_c, 2048, 2048)$, $\mathbf{s}_{offset}$ | $\mathbf{I}: (N_o, N_p, N_c, 768, 768)$ | — |
| P4 | Crop reference images | $\mathbf{R} = \mathbf{R}_{raw}[..., s_0:s_0+H, s_1:s_1+W]$ | $\mathbf{R}_{raw}: (N_o, N_p, N_c, 2048, 2048)$, $\mathbf{s}_{offset}$ | $\mathbf{R}: (N_o, N_p, N_c, 768, 768)$ | — |
| P5 | Solve for intersection points | $(\hat{\mathbf{P}}, \mathbf{V}, \mathbf{C}) = \texttt{solve\_for\_intersections}(\mathbf{I}, \mathbf{R}, [T_t, ...])$ | $\mathbf{I}, \mathbf{R}, \mathbf{T}[t:]$ | $\hat{\mathbf{P}}: (N_c, H, W, 2)$, $\mathbf{V}: (N_c, H, W)$, $\mathbf{C}$ | Phase unwrapping via fringe analysis |
| P6 | Load texture image | $\mathbf{I}_{tex}^{(x)} = \text{grayscale}(\text{imread}(\text{path}/x))$ for $x \in \{0\}$ | Sinusoid images | $\mathbf{I}_{tex}^{(x)}$ per image | Images are RGB |
| P7 | Normalize texture | $\tilde{\mathbf{I}}_{tex}^{(x)} = \mathbf{I}_{tex}^{(x)} / \max(\mathbf{I}_{tex}^{(x)})$ | $\mathbf{I}_{tex}^{(x)}$ | $\tilde{\mathbf{I}}_{tex}^{(x)}$ | Max > 0 |
| P8 | Sum textures | $\mathbf{I}_{tex} = \sum_x \tilde{\mathbf{I}}_{tex}^{(x)}$ | All $\tilde{\mathbf{I}}_{tex}^{(x)}$ | $\mathbf{I}_{tex}$ | — |
| P9 | Set texture | $\text{DM.set\_texture}(\mathbf{I}_{tex})$ | $\mathbf{I}_{tex}$ | DM updated | — |
| P10 | Set texture shift | $\text{DM.scene.screen.texture\_shift} = \mathbf{s}_{tex}$ | $\mathbf{s}_{tex} = [0, 1.1106231]$ | DM updated | Hardcoded for LE1234-A |
| P11 | Compute mount geometry | $\mathbf{o} = \texttt{\_compute\_mount\_geometry}(\mathbf{C})$ | Calibration matrix $\mathbf{C}$ | $\mathbf{o}: (3,)$ | — |
| P12 | Set lens origin | $\text{DM.scene.lensgroup.origin} = \mathbf{o}$ | $\mathbf{o}$ | DM updated | — |
| P13 | Solve fringe parameters | $(\mathbf{a}, \mathbf{b}, \boldsymbol{\psi}) = \texttt{Fringe.solve}(\mathbf{I})$ | $\mathbf{I}$ | DC offset $\mathbf{a}$, amplitude $\mathbf{b}$, phase $\boldsymbol{\psi}$ | — |
| P14 | Extract subset images | $\mathbf{I}_{sub} = \mathbf{I}[0, \{0\}, ...]$ | $\mathbf{I}$, indices $\{0\}$ | $\mathbf{I}_{sub}: (1, H, W)$ | — |
| P15 | Subtract DC offset | $\mathbf{I}_{sub} \leftarrow \mathbf{I}_{sub} - \mathbf{a}[:, 0, ...]$ | $\mathbf{I}_{sub}$, $\mathbf{a}$ | $\mathbf{I}_{sub}$ | — |
| P16 | Sum subset images | $\mathbf{I}_{sum} = \sum_{x} \mathbf{I}_{sub}[x]$ | $\mathbf{I}_{sub}$ | $\mathbf{I}_{sum}: (H, W)$ | — |
| P17 | Apply validity mask | $\mathbf{I}_{sum} \leftarrow \mathbf{V} \odot \mathbf{I}_{sum}$ | $\mathbf{V}$, $\mathbf{I}_{sum}$ | $\mathbf{I}_{sum}$ | — |
| P18 | Normalize to $\mathbf{I}_0$ | $\mathbf{I}_0 = \mathbf{V} \cdot |\{0\}| \cdot \frac{\mathbf{I}_{sum} - \min(\mathbf{I}_{sum})}{\max(\mathbf{I}_{sum}) - \min(\mathbf{I}_{sum})}$ | $\mathbf{I}_{sum}$, $\mathbf{V}$ | $\mathbf{I}_0: (N_c, H, W)$ | — |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $c_0$ | $0.0$ | $(1,)$ | float32 | Hardcoded |
| 2 | $c_1$ | $0.0$ | $(1,)$ | float32 | Hardcoded |
| 3 | $d_1$ | $3.0$ | $(1,)$ | float32 | Hardcoded |
| 4 | $\theta_x$ | $0.0$ | $(1,)$ | float32 | Hardcoded |
| 5 | $\theta_y$ | $0.0$ | $(1,)$ | float32 | Hardcoded |
| 6 | $\mathbf{o}$ | From P11 | $(3,)$ | float32 | `_compute_mount_geometry` |
| 7 | $k$ | $0$ | scalar | int | Iteration counter |
| 8 | $\mathbf{ls}$ | $[\,]$ | dynamic | list | Empty loss history |
| 9 | $\mathbf{m}_{\theta}, \mathbf{v}_{\theta}$ | $\mathbf{0}$ | Same as $\theta$ | float32 | Adam initialization (all parameters) |
| 10 | `lensgroup.update()` | — | — | — | Propagates parameter changes to ray tracer |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 1, \ldots, K_{max}$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 11 | [FORWARD] Ray trace through lens | $\mathbf{P}^{(k)} = \texttt{stack}(\text{DM.trace}(\text{with\_element}=\text{True}, \text{mask}=\mathbf{V}, \text{angles}=\phi))[..., 0:2]$ | $c_0, c_1, d_1, \mathbf{o}, \theta_x, \theta_y, \mathbf{V}, \phi$ | $\mathbf{P}^{(k)}: (N_c, H, W, 2)$ | Differentiable ray tracing returns 3D intersection points; extract x,y coordinates |
| 12 | Apply validity mask to residual | $\mathbf{r}^{(k)}_{i,j} = \begin{cases} \mathbf{P}^{(k)}_{i,j} - \hat{\mathbf{P}}_{i,j} & \text{if } V_{i,j} = 1 \\ \mathbf{0} & \text{otherwise} \end{cases}$ | $\mathbf{P}^{(k)}$, $\hat{\mathbf{P}}$, $\mathbf{V}$ | $\mathbf{r}^{(k)}: (N_c, H, W, 2)$ | Masked difference |
| 13 | Compute per-pixel squared error | $e^{(k)}_{i,j} = \sum_{d=0}^{1} (r^{(k)}_{i,j,d})^2 = (r^{(k)}_{i,j,0})^2 + (r^{(k)}_{i,j,1})^2$ | $\mathbf{r}^{(k)}$ | $\mathbf{e}^{(k)}: (N_c, H, W)$ | Sum over x,y coordinates |
| 14 | Compute mean loss | $\mathcal{L}^{(k)} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} e^{(k)}_{i,j}$ | $\mathbf{e}^{(k)}$, $\mathbf{V}$ | $\mathcal{L}^{(k)}$: scalar | $|\mathcal{V}| = \sum_{i,j} V_{i,j}$ |
| 15 | Append loss to history | $\mathbf{ls} \leftarrow \mathbf{ls} \cup \{\mathcal{L}^{(k)}\}$ | $\mathcal{L}^{(k)}$, $\mathbf{ls}$ | $\mathbf{ls}$ updated | — |
| 16 | [ADJOINT] Backpropagate | $\mathbf{g}_{\theta}^{(k)} = \nabla_{\theta} \mathcal{L}^{(k)}$ for each $\theta \in \{c_0, c_1, d_1, \mathbf{o}, \theta_x, \theta_y\}$ | $\mathcal{L}^{(k)}$, computational graph | $\mathbf{g}_{c_0}, \mathbf{g}_{c_1}, \mathbf{g}_{d_1}, \mathbf{g}_{\mathbf{o}}, \mathbf{g}_{\theta_x}, \mathbf{g}_{\theta_y}$ | PyTorch automatic differentiation |
| 17 | Update first moment (all params) | $\mathbf{m}_{\theta}^{(k)} = \beta_1 \mathbf{m}_{\theta}^{(k-1)} + (1 - \beta_1) \mathbf{g}_{\theta}^{(k)}$ | $\mathbf{m}_{\theta}^{(k-1)}$, $\mathbf{g}_{\theta}^{(k)}$, $\beta_1$ | $\mathbf{m}_{\theta}^{(k)}$ | Applied to each parameter |
| 18 | Update second moment (all params) | $\mathbf{v}_{\theta}^{(k)} = \beta_2 \mathbf{v}_{\theta}^{(k-1)} + (1 - \beta_2) (\mathbf{g}_{\theta}^{(k)})^2$ | $\mathbf{v}_{\theta}^{(k-1)}$, $\mathbf{g}_{\theta}^{(k)}$, $\beta_2$ | $\mathbf{v}_{\theta}^{(k)}$ | Element-wise square |
| 19 | Bias-correct first moment | $\hat{\mathbf{m}}_{\theta}^{(k)} = \mathbf{m}_{\theta}^{(k)} / (1 - \beta_1^k)$ | $\mathbf{m}_{\theta}^{(k)}$, $\beta_1$, $k$ | $\hat{\mathbf{m}}_{\theta}^{(k)}$ | — |
| 20 | Bias-correct second moment | $\hat{\mathbf{v}}_{\theta}^{(k)} = \mathbf{v}_{\theta}^{(k)} / (1 - \beta_2^k)$ | $\mathbf{v}_{\theta}^{(k)}$, $\beta_2$, $k$ | $\hat{\mathbf{v}}_{\theta}^{(k)}$ | — |
| 21 | Update parameters | $\theta^{(k)} = \theta^{(k-1)} - \alpha \cdot \hat{\mathbf{m}}_{\theta}^{(k)} / (\sqrt{\hat{\mathbf{v}}_{\theta}^{(k)}} + \epsilon)$ | $\theta^{(k-1)}$, $\hat{\mathbf{m}}_{\theta}^{(k)}$, $\hat{\mathbf{v}}_{\theta}^{(k)}$, $\alpha$, $\epsilon$ | $\theta^{(k)}$ | Applied to: $c_0, c_1, d_1, \mathbf{o}, \theta_x, \theta_y$ |
| 22 | Update lens group state | `lensgroup.update()` | Updated parameters | Internal ray tracer state | Recomputes transformation matrices |
| 23 | Increment iteration | $k \leftarrow k + 1$ | $k$ | $k$ | — |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 24 | Compute final intersection points | $\mathbf{P}^{(final)} = \texttt{forward\_operator}(\text{DM}, \phi, \mathbf{V}, \text{'trace'})$ | Final parameters, $\mathbf{V}$, $\phi$ | $\mathbf{P}^{(final)}: (N_c, H, W, 2)$ | — |
| 25 | Compute displacement vectors | $\mathbf{T} = \mathbf{P}^{(final)} - \hat{\mathbf{P}}$ | $\mathbf{P}^{(final)}$, $\hat{\mathbf{P}}$ | $\mathbf{T}: (N_c, H, W, 2)$ | — |
| 26 | Compute per-pixel error magnitude | $E_{i,j} = \sqrt{T_{i,j,0}^2 + T_{i,j,1}^2}$ for $(i,j) \in \mathcal{V}$ | $\mathbf{T}$, $\mathbf{V}$ | $\mathbf{E}: (|\mathcal{V}|,)$ | Only valid pixels |
| 27 | Compute mean error | $\bar{E} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} E_{i,j}$ | $\mathbf{E}$ | $\bar{E}$: scalar [mm] | — |
| 28 | Convert to micrometers | $\bar{E}_{\mu m} = \bar{E} \times 10^3$ | $\bar{E}$ | $\bar{E}_{\mu m}$: scalar [μm] | — |
| 29 | Extract final radii of curvature | $R_i = 1 / c_i^{(final)}$ for $i \in \{0, 1\}$ | $c_0^{(final)}, c_1^{(final)}$ | $R_0, R_1$: scalars [mm] | Printed output |
| 30 | Render final images | $\mathbf{I}^{(final)} = \mathbf{V} \odot \texttt{stack}(\text{DM.render}(...))$ | Final parameters, $\mathbf{V}$ | $\mathbf{I}^{(final)}: (N_c, H, W)$ | For visualization |
| 31 | Replace NaN in rendered | $I^{(final)}_{i,j} \leftarrow 0$ where $\text{isnan}(I^{(final)}_{i,j})$ | $\mathbf{I}^{(final)}$ | $\mathbf{I}^{(final)}$ cleaned | — |
| 32 | Compute image residual | $\Delta \mathbf{I} = \mathbf{I}_0 - \mathbf{I}^{(final)}$ | $\mathbf{I}_0$, $\mathbf{I}^{(final)}$ | $\Delta \mathbf{I}: (N_c, H, W)$ | For visualization |

## 5. STOPPING CRITERION

```json
{
  "type": "iteration_limit",
  "expression": "k \\geq K_{max}",
  "parameters": {
    "K_max": 5
  },
  "check_frequency": "After each complete iteration (steps 11-22)"
}
```

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Mean Displacement Error | $\bar{E} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \sqrt{\sum_{d=0}^{1}(P^{(final)}_{i,j,d} - \hat{P}_{i,j,d})^2}$ | $\hat{\mathbf{P}}$ (measured intersection points) | $\mathbf{P}^{(final)}$ (predicted intersection points) | Mean Euclidean distance between predicted and measured display intersection points [mm], reported in [μm] |
| Radius of Curvature Surface 0 | $R_0 = 1 / c_0^{(final)}$ | Ground truth from lens file | $c_0^{(final)}$ | Recovered radius of curvature for first lens surface [mm] |
| Radius of Curvature Surface 1 | $R_1 = 1 / c_1^{(final)}$ | Ground truth from lens file | $c_1^{(final)}$ | Recovered radius of curvature for second lens surface [mm] |
| Lens Thickness | $d_1^{(final)}$ | Ground truth from lens file | $d_1^{(final)}$ | Recovered lens thickness parameter [mm] |
| Image Residual | $\Delta \mathbf{I} = \mathbf{I}_0 - \mathbf{I}^{(final)}$ | $\mathbf{I}_0$ (captured normalized images) | $\mathbf{I}^{(final)}$ (rendered images) | Pixel-wise intensity difference for visual assessment |