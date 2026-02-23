## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: Differentiable Metrology Lens Parameter Estimation
- **algorithm_family**: First-Order Gradient-Based Optimization (Adam)
- **summary**: Recovers lens geometric parameters (curvatures, thickness, position, tilts) by minimizing the squared displacement error between measured and ray-traced intersection points on a display screen using automatic differentiation and Adam optimization.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\boldsymbol{\theta}} \quad \mathcal{L}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \left\| \mathbf{p}_{i,j}^{\text{sim}}(\boldsymbol{\theta}) - \mathbf{p}_{i,j}^{\text{cap}} \right\|_2^2$$

where $\boldsymbol{\theta} = \{c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y\}$ and $\mathcal{V} = \{(i,j) : \text{valid\_cap}_{i,j} = 1\}$.

### data_fidelity_term
$$\mathcal{L}_{\text{data}}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \left[ \left( p_{i,j,x}^{\text{sim}} - p_{i,j,x}^{\text{cap}} \right)^2 + \left( p_{i,j,y}^{\text{sim}} - p_{i,j,y}^{\text{cap}} \right)^2 \right]$$

This is an **L2 squared loss** (mean squared error) over valid pixels.

### regularization_term
None

### constraints
None (unconstrained optimization)

---

## 2. VARIABLES

### 2.1 Primal Variables (Optimization Parameters)

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $c_0$ | $(1,)$ | float32 | $0.0$ | Curvature of lens surface 0 (inverse of radius) |
| $c_1$ | $(1,)$ | float32 | $0.0$ | Curvature of lens surface 1 |
| $d$ | $(1,)$ | float32 | $3.0$ | Thickness of lens (distance between surfaces) |
| $\mathbf{o}$ | $(3,)$ | float32 | Computed from `_compute_mount_geometry` | Lens origin position $(o_x, o_y, o_z)$ |
| $\theta_x$ | $(1,)$ | float32 | $0.0$ | Lens tilt angle about x-axis |
| $\theta_y$ | $(1,)$ | float32 | $0.0$ | Lens tilt angle about y-axis |

### 2.2 Dual Variables
None (first-order method without explicit duals)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{p}^{\text{sim}}$ | $(N_{\text{cam}}, H, W, 2)$ | float32 | Computed | Simulated ray-display intersection points |
| $\mathbf{b}$ | $(N_{\text{cam}}, H, W, 2)$ | float32 | Computed | Residual vector: $\mathbf{p}^{\text{cap}} - \mathbf{p}^{\text{sim}}$ |
| $\mathbf{m}_t$ | Per-parameter | float32 | $\mathbf{0}$ | Adam first moment estimate |
| $\mathbf{v}_t$ | Per-parameter | float32 | $\mathbf{0}$ | Adam second moment estimate |
| $\mathbf{I}^{\text{render}}$ | $(N_{\text{cam}}, H, W)$ | float32 | Computed | Rendered images (for evaluation only) |
| $\mathbf{E}$ | scalar | float32 | Computed | Mean displacement error |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\text{filmsize}$ | $(768, 768)$ | Cropped image dimensions in pixels |
| $\text{crop\_offset}$ | $(640, 640)$ | Offset for cropping from $2048 \times 2048$ |
| $\mathbf{T}_s$ | $[70, 100, 110]$ | Periods of sinusoidal fringe patterns |
| $t$ | $0$ | Index into period array |
| $\text{angle}$ | $0.0$ | Rotation angle for measurement |
| $\text{texture\_shift}$ | $[0.0, 1.1106231]$ | Display texture shift for LE1234-A lens |
| $\text{origin\_shift}$ | $[0.0, 0.0, 0.0]$ | Initial origin shift |
| $\text{scale}$ | $1.0$ | Scaling factor |
| $\text{maxit}$ | $5$ | Maximum optimization iterations |
| $\text{xs}$ | $[0]$ | Pattern indices for texture loading |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{imgs}$ | $(N_{\text{patterns}}, N_{\text{cam}}, H, W)$ | float32 | Raw captured fringe images |
| $\mathbf{refs}$ | $(N_{\text{patterns}}, N_{\text{cam}}, H, W)$ | float32 | Reference fringe images |
| $\mathbf{p}^{\text{cap}}$ | $(N_{\text{cam}}, H, W, 2)$ | float32 | Measured ray-display intersection points |
| $\text{valid\_cap}$ | $(N_{\text{cam}}, H, W)$ | bool | Validity mask for intersection points |
| $\mathbf{C}$ | Variable | float32 | Calibration matrix from intersection solving |
| $\mathbf{I}_0$ | $(N_{\text{cam}}, H, W)$ | float32 | Preprocessed measurement images for visualization |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Initialize DiffMetrology | $\text{DM} \leftarrow \text{DiffMetrology}(\text{calibration\_path}, \text{rotation\_path}, \text{lut\_path}, \mathbf{o}_{\text{shift}}, s, \text{device})$ | calibration files, origin_shift: $(3,)$, scale: scalar | DM object |
| 2 | Set film size | $\text{filmsize} \leftarrow (768, 768)$ | None | filmsize: $(2,)$ |
| 3 | Compute crop offset | $\text{crop\_offset} = \left\lfloor \frac{2048 - \text{filmsize}}{2} \right\rfloor = (640, 640)$ | filmsize: $(2,)$ | crop_offset: $(2,)$ int |
| 4 | Configure cameras | For each camera $k$: $\text{cam}_k.\text{filmsize} \leftarrow \text{filmsize}$, $\text{cam}_k.\text{crop\_offset} \leftarrow \text{crop\_offset}$ | filmsize, crop_offset | Updated camera objects |
| 5 | Load lens file | $\text{DM.scene.lensgroup.load\_file}(\text{lens\_name})$ | lens_name string | Lens surfaces with ground truth parameters |
| 6 | Load measurement data | $\mathbf{imgs}, \mathbf{refs} \leftarrow \text{load}(\text{data\_path}/\text{measurement}/\text{lens\_name}/\text{data\_new.npz})$ | data_path, lens_name | imgs: $(N_p, N_c, 2048, 2048)$, refs: $(N_p, N_c, 2048, 2048)$ |
| 7 | Crop images | $\mathbf{imgs}_{\text{crop}} = \mathbf{imgs}[\ldots, 640:1408, 640:1408]$, $\mathbf{refs}_{\text{crop}} = \mathbf{refs}[\ldots, 640:1408, 640:1408]$ | imgs, refs, crop_offset, filmsize | imgs: $(N_p, N_c, 768, 768)$, refs: $(N_p, N_c, 768, 768)$ |
| 8 | Solve for intersections | $\mathbf{p}^{\text{cap}}, \text{valid\_cap}, \mathbf{C} \leftarrow \text{DM.solve\_for\_intersections}(\mathbf{imgs}, \mathbf{refs}, \mathbf{T}_s[t:])$ | imgs, refs, periods | ps_cap: $(N_c, H, W, 2)$, valid_cap: $(N_c, H, W)$, C |
| 9 | Load texture pattern | $\mathbf{im} \leftarrow \text{imread}(\text{sinusoid\_path}/0.\text{png})$; $\mathbf{im}_{\text{gray}} = \text{mean}(\mathbf{im}, \text{axis}=-1)$; $\mathbf{im}_{\text{norm}} = \mathbf{im}_{\text{gray}} / \max(\mathbf{im}_{\text{gray}})$ | sinusoid_path | im_norm: $(H_{\text{tex}}, W_{\text{tex}})$ |
| 10 | Set display texture | $\text{DM.set\_texture}(\mathbf{im}_{\text{norm}})$ | im_norm | Updated DM texture |
| 11 | Set texture shift | $\text{DM.scene.screen.texture\_shift} \leftarrow [0.0, 1.1106231]$ | None | Updated screen parameters |
| 12 | Compute mount geometry | $\mathbf{o} \leftarrow \text{DM.\_compute\_mount\_geometry}(\mathbf{C})$ | C | origin: $(3,)$ |
| 13 | Set lens origin | $\text{DM.scene.lensgroup.origin} \leftarrow \mathbf{o}$; $\text{DM.scene.lensgroup.update}()$ | origin | Updated lensgroup |
| 14 | Solve fringe parameters | $a^{\text{cap}}, b^{\text{cap}}, \psi^{\text{cap}} \leftarrow \text{Fringe.solve}(\mathbf{imgs})$ | imgs | a_cap, b_cap, psi_cap |
| 15 | Extract sub-images | $\mathbf{imgs}_{\text{sub}} = \mathbf{imgs}[0, \text{xs}, \ldots] - a^{\text{cap}}[:, 0, \ldots]$ | imgs, xs, a_cap | imgs_sub |
| 16 | Sum sub-images | $\mathbf{imgs}_{\text{sum}} = \sum_x \mathbf{imgs}_{\text{sub}}[x]$ | imgs_sub | imgs_sum: $(H, W)$ |
| 17 | Apply validity mask | $\mathbf{imgs}_{\text{masked}} = \text{valid\_cap} \odot \mathbf{imgs}_{\text{sum}}$ | imgs_sum, valid_cap | imgs_masked |
| 18 | Normalize to I0 | $\mathbf{I}_0 = \text{valid\_cap} \cdot |\text{xs}| \cdot \frac{\mathbf{imgs}_{\text{masked}} - \min(\mathbf{imgs}_{\text{masked}})}{\max(\mathbf{imgs}_{\text{masked}}) - \min(\mathbf{imgs}_{\text{masked}})}$ | imgs_masked, valid_cap, xs | I0: $(N_c, H, W)$ normalized to $[0, 1]$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize surface 0 curvature | $c_0 \leftarrow 0.0$ | None | $c_0$: $(1,)$ | Perturbed from ground truth |
| 2 | Initialize surface 1 curvature | $c_1 \leftarrow 0.0$ | None | $c_1$: $(1,)$ | Perturbed from ground truth |
| 3 | Initialize lens thickness | $d \leftarrow 3.0$ | None | $d$: $(1,)$ | Units: mm |
| 4 | Initialize x-tilt | $\theta_x \leftarrow 0.0$ | None | $\theta_x$: $(1,)$ | Radians |
| 5 | Initialize y-tilt | $\theta_y \leftarrow 0.0$ | None | $\theta_y$: $(1,)$ | Radians |
| 6 | Update lens group | $\text{DM.scene.lensgroup.update}()$ | $c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y$ | Updated internal transforms | Propagates parameters to ray tracer |
| 7 | Define differentiable parameters | $\text{diff\_names} \leftarrow [\text{'lensgroup.surfaces[0].c'}, \text{'lensgroup.surfaces[1].c'}, \text{'lensgroup.surfaces[1].d'}, \text{'lensgroup.origin'}, \text{'lensgroup.theta\_x'}, \text{'lensgroup.theta\_y'}]$ | None | diff_names: list of 6 strings | Specifies which parameters to optimize |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, \text{maxit}-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 8 | Forward ray trace | $\mathbf{p}^{\text{sim}} = \text{DM.trace}(\text{with\_element}=\text{True}, \text{mask}=\text{valid\_cap}, \text{angles}=\text{angle})[0][\ldots, 0:2]$ | $c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y$, valid_cap, angle | $\mathbf{p}^{\text{sim}}$: $(N_c, H, W, 2)$ | [FORWARD] Differentiable ray tracing through lens system |
| 9 | Stack traced points | $\mathbf{p}^{\text{sim}} \leftarrow \text{torch.stack}(\mathbf{p}^{\text{sim}})$ | $\mathbf{p}^{\text{sim}}$ list | $\mathbf{p}^{\text{sim}}$: $(N_c, H, W, 2)$ tensor | Converts list to tensor |
| 10 | Extract valid points | $\mathbf{p}^{\text{sim}}_{\mathcal{V}} = \mathbf{p}^{\text{sim}}[\text{valid\_cap}, \ldots]$, $\mathbf{p}^{\text{cap}}_{\mathcal{V}} = \mathbf{p}^{\text{cap}}[\text{valid\_cap}, \ldots]$ | $\mathbf{p}^{\text{sim}}$, $\mathbf{p}^{\text{cap}}$, valid_cap | $\mathbf{p}^{\text{sim}}_{\mathcal{V}}$: $(|\mathcal{V}|, 2)$, $\mathbf{p}^{\text{cap}}_{\mathcal{V}}$: $(|\mathcal{V}|, 2)$ | Boolean indexing |
| 11 | Compute squared differences | $\mathbf{d}^2 = \sum_{j=1}^{2} \left( p^{\text{sim}}_{\mathcal{V},j} - p^{\text{cap}}_{\mathcal{V},j} \right)^2$ | $\mathbf{p}^{\text{sim}}_{\mathcal{V}}$, $\mathbf{p}^{\text{cap}}_{\mathcal{V}}$ | $\mathbf{d}^2$: $(|\mathcal{V}|,)$ | Sum over x,y coordinates |
| 12 | Compute loss | $\mathcal{L} = \frac{1}{|\mathcal{V}|} \sum_{i=1}^{|\mathcal{V}|} d^2_i$ | $\mathbf{d}^2$ | $\mathcal{L}$: scalar | Mean squared displacement error |
| 13 | Compute residual vector | $\mathbf{b} = \text{valid\_cap}[\ldots, \text{None}] \odot (\mathbf{p}^{\text{cap}} - \mathbf{p}^{\text{sim}})$ | $\mathbf{p}^{\text{cap}}$, $\mathbf{p}^{\text{sim}}$, valid_cap | $\mathbf{b}$: $(N_c, H, W, 2)$ | Masked residual for Jacobian computation |
| 14 | Handle NaN values | $\mathbf{b}[\text{isnan}(\mathbf{b})] \leftarrow 0.0$ | $\mathbf{b}$ | $\mathbf{b}$: $(N_c, H, W, 2)$ | Ensures numerical stability |
| 15 | Backward pass (autodiff) | $\nabla_{\boldsymbol{\theta}} \mathcal{L} = \text{autograd.backward}(\mathcal{L}, \boldsymbol{\theta})$ | $\mathcal{L}$, $\boldsymbol{\theta}$ | $\nabla_{c_0}\mathcal{L}, \nabla_{c_1}\mathcal{L}, \nabla_d\mathcal{L}, \nabla_{\mathbf{o}}\mathcal{L}, \nabla_{\theta_x}\mathcal{L}, \nabla_{\theta_y}\mathcal{L}$ | [ADJOINT] PyTorch automatic differentiation |
| 16 | Adam update: first moment | $\mathbf{m}_{k+1} = \beta_1 \mathbf{m}_k + (1 - \beta_1) \nabla_{\boldsymbol{\theta}} \mathcal{L}$ | $\mathbf{m}_k$, $\nabla_{\boldsymbol{\theta}} \mathcal{L}$, $\beta_1$ | $\mathbf{m}_{k+1}$ | Default $\beta_1 = 0.9$ |
| 17 | Adam update: second moment | $\mathbf{v}_{k+1} = \beta_2 \mathbf{v}_k + (1 - \beta_2) (\nabla_{\boldsymbol{\theta}} \mathcal{L})^2$ | $\mathbf{v}_k$, $\nabla_{\boldsymbol{\theta}} \mathcal{L}$, $\beta_2$ | $\mathbf{v}_{k+1}$ | Default $\beta_2 = 0.999$, element-wise square |
| 18 | Bias correction: first moment | $\hat{\mathbf{m}}_{k+1} = \frac{\mathbf{m}_{k+1}}{1 - \beta_1^{k+1}}$ | $\mathbf{m}_{k+1}$, $\beta_1$, $k$ | $\hat{\mathbf{m}}_{k+1}$ | Corrects initialization bias |
| 19 | Bias correction: second moment | $\hat{\mathbf{v}}_{k+1} = \frac{\mathbf{v}_{k+1}}{1 - \beta_2^{k+1}}$ | $\mathbf{v}_{k+1}$, $\beta_2$, $k$ | $\hat{\mathbf{v}}_{k+1}$ | Corrects initialization bias |
| 20 | Parameter update | $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha \frac{\hat{\mathbf{m}}_{k+1}}{\sqrt{\hat{\mathbf{v}}_{k+1}} + \epsilon}$ | $\boldsymbol{\theta}_k$, $\hat{\mathbf{m}}_{k+1}$, $\hat{\mathbf{v}}_{k+1}$, $\alpha$, $\epsilon$ | $\boldsymbol{\theta}_{k+1} = \{c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y\}_{k+1}$ | Default $\epsilon = 10^{-8}$, $\alpha$ from solver |
| 21 | Update lens group state | $\text{DM.scene.lensgroup.update}()$ | $\boldsymbol{\theta}_{k+1}$ | Updated internal transforms | Propagates new parameters |
| 22 | Record loss | $\text{ls}.\text{append}(\mathcal{L})$ | $\mathcal{L}$ | ls: list | Loss history tracking |

**END_LOOP**

### 4.3 FINALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 23 | Final forward trace | $\mathbf{p}^{\text{final}} = \text{forward\_operator}(\text{DM}, \text{angle}, \text{valid\_cap}, \text{mode}=\text{'trace'})$ | Final $\boldsymbol{\theta}$, valid_cap, angle | $\mathbf{p}^{\text{final}}$: $(N_c, H, W, 2)$ | Ray trace with optimized parameters |
| 24 | Compute displacement vector | $\mathbf{T} = \mathbf{p}^{\text{final}} - \mathbf{p}^{\text{cap}}$ | $\mathbf{p}^{\text{final}}$, $\mathbf{p}^{\text{cap}}$ | $\mathbf{T}$: $(N_c, H, W, 2)$ | Residual displacement |
| 25 | Compute Euclidean displacement | $\mathbf{E}_{\text{pix}} = \sqrt{\sum_{j=1}^{2} T_{\mathcal{V},j}^2}$ | $\mathbf{T}[\text{valid\_cap}, \ldots]$ | $\mathbf{E}_{\text{pix}}$: $(|\mathcal{V}|,)$ | Per-pixel displacement magnitude |
| 26 | Compute mean error | $E = \frac{1}{|\mathcal{V}|} \sum_{i=1}^{|\mathcal{V}|} E_{\text{pix},i}$ | $\mathbf{E}_{\text{pix}}$ | $E$: scalar (mm) | Mean displacement error |
| 27 | Convert to micrometers | $E_{\mu m} = E \times 10^3$ | $E$ | $E_{\mu m}$: scalar (μm) | Unit conversion for display |
| 28 | Render final images | $\mathbf{I}^{\text{render}} = \text{valid\_cap} \odot \text{DM.render}(\text{with\_element}=\text{True}, \text{angles}=\text{angle})$ | Final $\boldsymbol{\theta}$, valid_cap, angle | $\mathbf{I}^{\text{render}}$: $(N_c, H, W)$ | For visualization |
| 29 | Handle NaN in rendered | $\mathbf{I}^{\text{render}}[\text{isnan}(\mathbf{I}^{\text{render}})] \leftarrow 0.0$ | $\mathbf{I}^{\text{render}}$ | $\mathbf{I}^{\text{render}}$: $(N_c, H, W)$ | Clean up invalid pixels |
| 30 | Extract final curvatures | $R_0 = 1/c_0^{\text{final}}$, $R_1 = 1/c_1^{\text{final}}$ | $c_0^{\text{final}}$, $c_1^{\text{final}}$ | $R_0, R_1$: scalars (mm) | Radius of curvature |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $k < \text{maxit}$
- **parameters**: 
  - $\text{maxit} = 5$ (default)
- **check_frequency**: Evaluated at the end of each iteration

---

## 6. EVALUATION METRICS

| name | formula | reference_data | predicted_data | description |
|------|---------|----------------|----------------|-------------|
| Mean Displacement Error | $E = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \sqrt{(p_{i,x}^{\text{sim}} - p_{i,x}^{\text{cap}})^2 + (p_{i,y}^{\text{sim}} - p_{i,y}^{\text{cap}})^2}$ | $\mathbf{p}^{\text{cap}}$: measured intersection points | $\mathbf{p}^{\text{sim}}$: simulated intersection points | Mean Euclidean distance between measured and simulated ray-display intersections over valid pixels; reported in μm |
| Loss (MSE) | $\mathcal{L} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \left\| \mathbf{p}_i^{\text{sim}} - \mathbf{p}_i^{\text{cap}} \right\|_2^2$ | $\mathbf{p}^{\text{cap}}$ | $\mathbf{p}^{\text{sim}}$ | Mean squared displacement error used as optimization objective |
| Radius of Curvature | $R_s = 1 / c_s$ for surface $s \in \{0, 1\}$ | Ground truth from lens file | Optimized $c_0, c_1$ | Recovered lens surface radii of curvature |
| Image Residual | $\mathbf{I}^{\text{error}} = \mathbf{I}_0 - \mathbf{I}^{\text{render}}$ | $\mathbf{I}_0$: preprocessed measurement | $\mathbf{I}^{\text{render}}$: rendered from parameters | Pixel-wise difference between measurement and model for visualization |