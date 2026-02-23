## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Differentiable Ray Tracing Lens Metrology Inversion
- **algorithm_family**: First-order gradient descent (Adam optimizer) with differentiable physics forward model
- **summary**: Recovers optical lens parameters (curvatures, thickness, position, tilt angles) by minimizing the squared displacement error between ray-traced and measured intersection points on a display screen using automatic differentiation through a ray tracing engine.

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$
\min_{\boldsymbol{\theta}} \quad \mathcal{L}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \left\| \mathbf{p}_{ij}(\boldsymbol{\theta}) - \hat{\mathbf{p}}_{ij} \right\|_2^2
$$

where $\boldsymbol{\theta} = (c_0, c_1, d_1, \mathbf{o}, \theta_x, \theta_y)$ is the parameter vector.

### data_fidelity_term

$$
\mathcal{D}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \sum_{k=1}^{2} \left( p_{ij,k}(\boldsymbol{\theta}) - \hat{p}_{ij,k} \right)^2
$$

where:
- $\mathbf{p}_{ij}(\boldsymbol{\theta}) \in \mathbb{R}^2$: predicted 2D intersection point at pixel $(i,j)$ from ray tracing [FORWARD]
- $\hat{\mathbf{p}}_{ij} \in \mathbb{R}^2$: measured intersection point from fringe analysis
- $\mathcal{V}$: set of valid pixel indices where `valid_cap[i,j] = True`
- $|\mathcal{V}|$: cardinality of valid pixel set

**Verification**: L2-squared loss confirmed by `(ps[valid_cap, ...] - ps_cap[valid_cap, ...]) ** 2` with `sum` over last axis and `mean` over valid pixels.

### regularization_term

None (no explicit regularization in objective function)

### constraints

$$
\mathcal{L}(\boldsymbol{\theta}) + \iota_{\Omega}(\boldsymbol{\theta})
$$

where the implicit constraint set $\Omega$ is defined by the physical validity of ray tracing (rays must intersect surfaces). Invalid configurations produce NaN values which are handled by:
$$
\text{NaN} \rightarrow 0 \quad \text{(replacement)}
$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $c_0$ | $(1,)$ | `torch.float32` | $0.00$ | Curvature of lens surface 0 (1/radius) |
| $c_1$ | $(1,)$ | `torch.float32` | $0.00$ | Curvature of lens surface 1 (1/radius) |
| $d_1$ | $(1,)$ | `torch.float32` | $3.00$ | Thickness of lens at surface 1 |
| $\mathbf{o}$ | $(3,)$ | `torch.float32` | Computed from `_compute_mount_geometry` | Lens group origin position (3D) |
| $\theta_x$ | $(1,)$ | `torch.float32` | $0.00$ | Lens X-axis tilt angle |
| $\theta_y$ | $(1,)$ | `torch.float32` | $0.00$ | Lens Y-axis tilt angle |

### 2.2 Dual Variables

None (first-order method, no explicit dual variables)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{m}_\theta$ | same as $\boldsymbol{\theta}$ | `torch.float32` | $\mathbf{0}$ | Adam first moment estimate |
| $\mathbf{v}_\theta$ | same as $\boldsymbol{\theta}$ | `torch.float32` | $\mathbf{0}$ | Adam second moment estimate |
| $t$ | scalar | `int` | $0$ | Adam iteration counter |
| `ls` | list of length `maxit` | `float` | `[]` | Loss history |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `maxit` | $5$ | Maximum number of iterations |
| `filmsize` | $[768, 768]$ | Cropped image dimensions (pixels) |
| `crop_offset` | $[(2048-768)/2, (2048-768)/2] = [640, 640]$ | Crop offset from original image |
| `origin_shift` | $[0.0, 0.0, 0.0]$ | Initial origin shift |
| `scale` | $1.0$ | DiffMetrology scale factor |
| `angle` | $0.0$ | Rotation angle for measurement |
| `Ts` | $[70, 100, 110]$ | Sinusoid periods for fringe analysis |
| `t` (index) | $0$ | Selected period index |
| `xs` | $[0]$ | Pattern indices |
| `texture_shift` | $[0.0, 1.1106231]$ | Screen texture shift (hardcoded for LE1234-A) |
| $\beta_1$ | $0.9$ (Adam default) | First moment decay rate |
| $\beta_2$ | $0.999$ (Adam default) | Second moment decay rate |
| $\epsilon$ | $10^{-8}$ (Adam default) | Numerical stability constant |
| $\alpha$ | Adam default | Learning rate |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `imgs` | $(N_{\text{views}}, N_{\text{patterns}}, H, W)$ → cropped to $(N, M, 768, 768)$ | `numpy.ndarray` | Raw captured images |
| `refs` | same as `imgs` | `numpy.ndarray` | Reference images (no lens) |
| $\hat{\mathbf{p}}$ (`ps_cap`) | $(N_{\text{cams}}, H, W, 2)$ | `torch.Tensor` | Measured intersection points from fringe analysis |
| `valid_cap` | $(N_{\text{cams}}, H, W)$ | `torch.Tensor` (bool) | Valid pixel mask |
| $\mathbf{C}$ | $(N_{\text{cams}}, 3, 4)$ | `numpy.ndarray` | Camera calibration matrices |
| `I0` | $(N_{\text{cams}}, H, W)$ | `torch.Tensor` | Preprocessed reference images for visualization |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 0 | Initialize DiffMetrology | $\text{DM} = \text{DiffMetrology}(\text{calibration}, \text{rotation}, \text{LUT}, \mathbf{o}_{\text{shift}}, s)$ | calibration_path, rotation_path, lut_path, origin_shift=$(0,0,0)$, scale=$1.0$ | DM object |
| 1 | Crop configuration | $\text{filmsize} = [768, 768]$; $\text{offset} = \lfloor(2048 - \text{filmsize})/2\rfloor = [640, 640]$ | original filmsize $2048 \times 2048$ | filmsize, crop_offset |
| 2 | Load lens file | Parse lens specification from `ThorLabs/{lens_name}.txt` | lens_name string | DM.scene.lensgroup surfaces populated |
| 3 | Load measurements | $\text{imgs}, \text{refs} \leftarrow \text{load}(\text{data\_path}/\text{measurement}/\text{lens\_name}/\text{data\_new.npz})$ | data_path, lens_name | imgs: $(N,M,2048,2048)$, refs: $(N,M,2048,2048)$ |
| 4 | Crop images | $\text{imgs}_{\text{crop}} = \text{imgs}[..., 640:1408, 640:1408]$; same for refs | imgs, refs, crop_offset, filmsize | imgs: $(N,M,768,768)$, refs: $(N,M,768,768)$ |
| 5 | Solve intersections | $(\hat{\mathbf{p}}, \text{valid}, \mathbf{C}) = \text{DM.solve\_for\_intersections}(\text{imgs}, \text{refs}, [70])$ | cropped imgs, refs, $T_s[0:] = [70, 100, 110]$ | ps_cap: $(N_c, H, W, 2)$, valid_cap: $(N_c, H, W)$, C |
| 6 | Load display texture | $\mathbf{I}_{\text{tex}} = \text{grayscale}(\text{imread}(\text{sinusoid\_path}/0.png))$; $\mathbf{I}_{\text{tex}} \leftarrow \mathbf{I}_{\text{tex}} / \max(\mathbf{I}_{\text{tex}})$ | sinusoid image path | normalized texture $(H_s, W_s)$ |
| 7 | Set texture shift | $\text{DM.scene.screen.texture\_shift} = [0.0, 1.1106231]$ | hardcoded values | texture_shift tensor |
| 8 | Compute mount geometry | $\mathbf{o} = \text{DM.\_compute\_mount\_geometry}(\mathbf{C})$ | camera matrices C | origin: $(3,)$ |
| 9 | Compute reference images | $\mathbf{a}, \mathbf{b}, \boldsymbol{\psi} = \text{FR.solve}(\text{imgs})$; $\mathbf{I}_{\text{sub}} = \text{imgs}[0,0,...] - \mathbf{a}[:,0,...]$; $\mathbf{I}_0 = \text{valid} \cdot \frac{\mathbf{I}_{\text{sub}} - \min(\mathbf{I}_{\text{sub}})}{\max(\mathbf{I}_{\text{sub}}) - \min(\mathbf{I}_{\text{sub}})}$ | imgs, valid_cap | I0: $(N_c, H, W)$ normalized |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Initialize surface 0 curvature | $c_0 \leftarrow 0.00$ | — | $c_0 \in \mathbb{R}$ |
| 2 | Initialize surface 1 curvature | $c_1 \leftarrow 0.00$ | — | $c_1 \in \mathbb{R}$ |
| 3 | Initialize lens thickness | $d_1 \leftarrow 3.00$ | — | $d_1 \in \mathbb{R}$ |
| 4 | Initialize X-tilt | $\theta_x \leftarrow 0.00$ | — | $\theta_x \in \mathbb{R}$ |
| 5 | Initialize Y-tilt | $\theta_y \leftarrow 0.00$ | — | $\theta_y \in \mathbb{R}$ |
| 6 | Update lens geometry | $\text{DM.scene.lensgroup.update()}$ | all parameters | internal state updated |
| 7 | Define differentiable parameters | $\text{diff\_names} = [c_0, c_1, d_1, \mathbf{o}, \theta_x, \theta_y]$ | — | parameter name list |
| 8 | Initialize Adam state | $\mathbf{m} \leftarrow \mathbf{0}$, $\mathbf{v} \leftarrow \mathbf{0}$, $t \leftarrow 0$ | — | Adam moment vectors |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 1, \ldots, \text{maxit}$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Forward ray tracing [FORWARD] | $\mathbf{p}^{(k)} = \text{DM.trace}(\text{with\_element=True}, \text{mask=valid\_cap}, \text{angles=}0.0)[0][..., 0:2]$ | $\boldsymbol{\theta}^{(k-1)}$, valid_cap, angle | $\mathbf{p}^{(k)} \in \mathbb{R}^{N_c \times H \times W \times 2}$ | Differentiable ray tracing through lens system |
| 2 | Compute residual | $\mathbf{r}^{(k)} = \text{valid\_cap}[..., \text{None}] \cdot (\hat{\mathbf{p}} - \mathbf{p}^{(k)})$; $\mathbf{r}^{(k)}[\text{NaN}] \leftarrow 0$ | $\mathbf{p}^{(k)}$, $\hat{\mathbf{p}}$, valid_cap | $\mathbf{r}^{(k)} \in \mathbb{R}^{N_c \times H \times W \times 2}$ | NaN handling for invalid rays |
| 3 | Compute loss | $\mathcal{L}^{(k)} = \frac{1}{|\mathcal{V}|} \sum_{(i,j)\in\mathcal{V}} \|\mathbf{p}^{(k)}_{ij} - \hat{\mathbf{p}}_{ij}\|_2^2$ | $\mathbf{p}^{(k)}$, $\hat{\mathbf{p}}$, valid_cap | $\mathcal{L}^{(k)} \in \mathbb{R}$ | Mean squared displacement |
| 4 | Compute gradient [ADJOINT] | $\mathbf{g}^{(k)} = \nabla_{\boldsymbol{\theta}} \mathcal{L}^{(k)}$ via automatic differentiation | $\mathcal{L}^{(k)}$, computational graph | $\mathbf{g}^{(k)}$ same shape as $\boldsymbol{\theta}$ | PyTorch autograd backpropagation |
| 5 | Adam first moment update | $\mathbf{m}^{(k)} = \beta_1 \mathbf{m}^{(k-1)} + (1-\beta_1)\mathbf{g}^{(k)}$ | $\mathbf{m}^{(k-1)}$, $\mathbf{g}^{(k)}$, $\beta_1=0.9$ | $\mathbf{m}^{(k)}$ |  |
| 6 | Adam second moment update | $\mathbf{v}^{(k)} = \beta_2 \mathbf{v}^{(k-1)} + (1-\beta_2)(\mathbf{g}^{(k)})^2$ | $\mathbf{v}^{(k-1)}$, $\mathbf{g}^{(k)}$, $\beta_2=0.999$ | $\mathbf{v}^{(k)}$ | Element-wise square |
| 7 | Bias correction | $\hat{\mathbf{m}}^{(k)} = \frac{\mathbf{m}^{(k)}}{1-\beta_1^k}$; $\hat{\mathbf{v}}^{(k)} = \frac{\mathbf{v}^{(k)}}{1-\beta_2^k}$ | $\mathbf{m}^{(k)}$, $\mathbf{v}^{(k)}$, $k$ | $\hat{\mathbf{m}}^{(k)}$, $\hat{\mathbf{v}}^{(k)}$ |  |
| 8 | Parameter update | $\boldsymbol{\theta}^{(k)} = \boldsymbol{\theta}^{(k-1)} - \alpha \frac{\hat{\mathbf{m}}^{(k)}}{\sqrt{\hat{\mathbf{v}}^{(k)}} + \epsilon}$ | $\boldsymbol{\theta}^{(k-1)}$, $\hat{\mathbf{m}}^{(k)}$, $\hat{\mathbf{v}}^{(k)}$, $\alpha$, $\epsilon=10^{-8}$ | $\boldsymbol{\theta}^{(k)}$ | Element-wise division |
| 9 | Update lens geometry | $\text{DM.scene.lensgroup.update()}$ | $\boldsymbol{\theta}^{(k)}$ | internal geometry | Propagate parameters to ray tracer |
| 10 | Record loss | $\text{ls.append}(\mathcal{L}^{(k)})$ | $\mathcal{L}^{(k)}$ | ls list | For convergence monitoring |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Extract optimized curvatures | $R_i = 1/c_i^{(\text{final})}$ for $i \in \{0, 1\}$ | $c_0^{(\text{final})}$, $c_1^{(\text{final})}$ | Radius of curvature (mm) |
| 2 | Final forward pass | $\mathbf{p}^{(\text{final})} = \text{forward\_operator}(\text{DM}, 0.0, \text{valid\_cap}, \text{'trace'})$ | optimized DM | predicted intersections |
| 3 | Render final images | $\mathbf{I}^{(\text{final})} = \text{valid\_cap} \cdot \text{DM.render}(\text{with\_element=True}, \text{angles=}0.0)$; $\mathbf{I}[\text{NaN}] \leftarrow 0$ | optimized DM, valid_cap | rendered images $(N_c, H, W)$ |
| 4 | Generate spot diagram | $\text{DM.spot\_diagram}(\hat{\mathbf{p}}, \mathbf{p}^{(\text{final})}, \text{valid\_cap})$ | measurements, predictions, mask | visualization saved to file |
| 5 | Generate comparison images | Plot $[\mathbf{I}_0, \mathbf{I}^{(\text{final})}, \mathbf{I}_0 - \mathbf{I}^{(\text{final})}]$ for each camera | I0, $\mathbf{I}^{(\text{final})}$ | 2×3 subplot figure |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: 
$$
k \geq \text{maxit}
$$
where $\text{maxit} = 5$

- **parameters**:
  - `maxit`: 5 (hardcoded default, configurable via function argument)
  
- **check_frequency**: Evaluated at the end of each iteration $k$

---

## 6. EVALUATION METRICS

| name | formula | description |
|------|---------|-------------|
| Mean Displacement Error | $E = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \sqrt{\sum_{k=1}^{2} (p_{ij,k} - \hat{p}_{ij,k})^2}$ | Mean Euclidean distance between predicted and measured intersection points over valid pixels (units: mm, displayed as μm via $\times 10^3$) |
| MSE Loss | $\mathcal{L} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \|\mathbf{p}_{ij} - \hat{\mathbf{p}}_{ij}\|_2^2$ | Mean squared displacement (optimization objective) |
| Recovered Radius of Curvature | $R_i = 1/c_i$ for surface $i$ | Physical lens parameter (mm) |
| Recovered Thickness | $d_1$ | Lens thickness at surface 1 (mm) |
| Image Residual | $\mathbf{E}_{\text{img}} = \mathbf{I}_0 - \mathbf{I}^{(\text{final})}$ | Pixel-wise difference between measured and rendered images |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include the following fields:

```
algorithm_identification:
  - algorithm_name
  - algorithm_family
  - summary

objective_function:
  - full_expression
  - data_fidelity_term
  - regularization_term
  - constraints

variables:
  - primal_variables (list with symbol, shape, dtype, initialization, description)
  - dual_variables
  - auxiliary_variables
  - constants
  - observations

data_preprocessing:
  - steps (list with step_order, step_name, formula, inputs, outputs)

initialization:
  - steps (list with step_order, step_name, formula, inputs, outputs)

iteration_loop:
  - steps (list with step_order, step_name, formula, inputs, outputs, computational_note)
  - loop_bounds (start, end, variable)

finalization:
  - steps (list with step_order, step_name, formula, inputs, outputs)

stopping_criterion:
  - type
  - expression
  - parameters
  - check_frequency

evaluation:
  - metrics (list with name, formula, description)
  - final_values (initial_error, final_error)
```