# Algorithmic Specification Document: Deflectometry-Based Lens Parameter Recovery

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \left\| \mathbf{p}_i(\boldsymbol{\theta}) - \hat{\mathbf{p}}_i \right\|_2^2$$

where $\boldsymbol{\theta} = \{c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y\}$ is the parameter vector.

**data_fidelity_term**:
$$\mathcal{D}(\boldsymbol{\theta}) = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \sum_{k=1}^{2} \left( p_{i,k}(\boldsymbol{\theta}) - \hat{p}_{i,k} \right)^2$$

where:
- $\mathbf{p}_i(\boldsymbol{\theta}) = [p_{i,1}, p_{i,2}]^\top \in \mathbb{R}^2$ denotes predicted 2D intersection coordinates on display plane for pixel $i$
- $\hat{\mathbf{p}}_i = [\hat{p}_{i,1}, \hat{p}_{i,2}]^\top \in \mathbb{R}^2$ denotes measured intersection coordinates
- $\mathcal{V} = \{i : v_i = 1\}$ is the set of valid pixel indices
- $|\mathcal{V}|$ is the cardinality of the valid set

**regularization_term**: None

**constraints**: None (unconstrained optimization)

---

## 2. VARIABLES

### Primal Variables (Optimization Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $c_0$ | primal | $(1,)$ | float32 | $0.0$ | Curvature of first lens surface (inverse radius) |
| $c_1$ | primal | $(1,)$ | float32 | $0.0$ | Curvature of second lens surface (inverse radius) |
| $d$ | primal | $(1,)$ | float32 | $3.0$ | Lens thickness (distance between surfaces) |
| $\mathbf{o}$ | primal | $(3,)$ | float32 | computed from geometry | Lens group origin position $[o_x, o_y, o_z]^\top$ |
| $\theta_x$ | primal | $(1,)$ | float32 | $0.0$ | Lens tilt angle about X-axis |
| $\theta_y$ | primal | $(1,)$ | float32 | $0.0$ | Lens tilt angle about Y-axis |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\hat{\mathbf{P}}$ | observations | $(N_{cam}, H, W, 2)$ | float32 | from measurement | Measured intersection points on display |
| $\mathbf{V}$ | observations | $(N_{cam}, H, W)$ | bool | from measurement | Valid pixel mask |
| $\mathbf{I}_{raw}$ | observations | $(N_{cam}, N_T, H_{orig}, W_{orig})$ | float32 | from file | Raw fringe pattern images |
| $\mathbf{I}_{ref}$ | observations | $(N_{cam}, N_T, H_{orig}, W_{orig})$ | float32 | from file | Reference fringe pattern images |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\phi$ | constants | $(1,)$ | float32 | $0.0$ | Rotation angle for measurement |
| $\mathbf{f}$ | constants | $(2,)$ | int32 | $[768, 768]$ | Film size (cropped region of interest) |
| $\mathbf{T}$ | constants | $(3,)$ | int32 | $[70, 100, 110]$ | Periods of sinusoidal fringe patterns |
| $\boldsymbol{\delta}$ | constants | $(2,)$ | float32 | $[0.0, 1.1106231]$ | Texture shift on display screen |
| $H_{orig}$ | constants | scalar | int32 | $2048$ | Original image height |
| $W_{orig}$ | constants | scalar | int32 | $2048$ | Original image width |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Initialize Metrology System
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | System Initialization |
| formula | Load calibration matrices, rotation parameters, gamma LUT from files |
| inputs | calibration_path, rotation_path, lut_path, origin_shift $\in \mathbb{R}^3$ |
| outputs | DiffMetrology object $\mathcal{M}$ with cameras, screen, lensgroup |

### Step 3.2: Compute Crop Offset
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Crop Offset Calculation |
| formula | $\boldsymbol{\delta}_{crop} = \left\lfloor \frac{H_{orig} - f_0}{2} \right\rfloor, \left\lfloor \frac{W_{orig} - f_1}{2} \right\rfloor$ |
| inputs | $H_{orig} = 2048$, $W_{orig} = 2048$, $\mathbf{f} = [768, 768]$ |
| outputs | $\boldsymbol{\delta}_{crop} \in \mathbb{Z}^2$ with shape $(2,)$ |

### Step 3.3: Crop Images
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Region of Interest Extraction |
| formula | $\mathbf{I}_{crop}[..., i, j] = \mathbf{I}_{orig}[..., \delta_{crop,0} + i, \delta_{crop,1} + j]$ for $i \in [0, f_0)$, $j \in [0, f_1)$ |
| inputs | $\mathbf{I}_{raw} \in \mathbb{R}^{N_{cam} \times N_T \times H_{orig} \times W_{orig}}$, $\mathbf{I}_{ref} \in \mathbb{R}^{N_{cam} \times N_T \times H_{orig} \times W_{orig}}$ |
| outputs | $\mathbf{I}_{raw,crop} \in \mathbb{R}^{N_{cam} \times N_T \times H \times W}$, $\mathbf{I}_{ref,crop} \in \mathbb{R}^{N_{cam} \times N_T \times H \times W}$ |

### Step 3.4: Fringe Analysis - Solve for Intersections
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Phase Demodulation and Intersection Computation |
| formula | Phase extraction via $N$-step algorithm: $\psi = \arctan\left(\frac{\sum_{n} I_n \sin(2\pi n / N)}{\sum_{n} I_n \cos(2\pi n / N)}\right)$; Intersection points: $\hat{\mathbf{P}} = \mathcal{F}_{intersect}(\psi, \mathbf{T}, \mathcal{M})$ |
| inputs | $\mathbf{I}_{raw,crop}$, $\mathbf{I}_{ref,crop}$, $\mathbf{T}$ |
| outputs | $\hat{\mathbf{P}} \in \mathbb{R}^{N_{cam} \times H \times W \times 2}$ (measured intersection points), $\mathbf{V} \in \{0,1\}^{N_{cam} \times H \times W}$ (validity mask), $\mathbf{C}$ (calibration matrix) |

### Step 3.5: Load and Set Display Texture
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Texture Initialization |
| formula | $\mathbf{S} = \frac{\mathbf{I}_{pattern}}{\max(\mathbf{I}_{pattern})}$ where $\mathbf{I}_{pattern}$ is grayscale sinusoid image |
| inputs | Sinusoid image file at period $T_0 = 70$ |
| outputs | Normalized texture $\mathbf{S} \in [0,1]^{H_s \times W_s}$ |

### Step 3.6: Compute Mount Geometry
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Origin Estimation |
| formula | $\mathbf{o}_{init} = \mathcal{F}_{geometry}(\mathbf{C})$ (geometric triangulation from calibration) |
| inputs | Calibration matrix $\mathbf{C}$ |
| outputs | Initial origin estimate $\mathbf{o}_{init} \in \mathbb{R}^3$ |

### Step 3.7: Prepare Reference Images for Visualization
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Reference Image Normalization |
| formula | $\mathbf{I}_0 = \mathbf{V} \odot \frac{\mathbf{I}_{sub} - \min(\mathbf{I}_{sub})}{\max(\mathbf{I}_{sub}) - \min(\mathbf{I}_{sub})}$ where $\mathbf{I}_{sub} = \mathbf{I}_{raw,crop}[0, 0, ...] - \mathbf{a}$ and $\mathbf{a}$ is DC component from fringe analysis |
| inputs | $\mathbf{I}_{raw,crop}$, $\mathbf{V}$, DC component $\mathbf{a}$ |
| outputs | Normalized reference images $\mathbf{I}_0 \in \mathbb{R}^{N_{cam} \times H \times W}$ |

---

## 4. ALGORITHM STEPS

### Initialization

**--- BEGIN INITIALIZATION ---**

#### Step 4.1: Initialize Lens Parameters
| Field | Value |
|-------|-------|
| step_name | Parameter Initialization |
| formula | $c_0 \leftarrow 0.0$, $c_1 \leftarrow 0.0$, $d \leftarrow 3.0$, $\theta_x \leftarrow 0.0$, $\theta_y \leftarrow 0.0$ |
| inputs | None |
| outputs | Initial parameter vector $\boldsymbol{\theta}^{(0)} = \{c_0, c_1, d, \mathbf{o}, \theta_x, \theta_y\}$ |
| computational_note | Origin $\mathbf{o}$ retained from preprocessing step 3.6 |

#### Step 4.2: Define Optimization Variable Set
| Field | Value |
|-------|-------|
| step_name | Variable Registration |
| formula | $\mathcal{D} = \{$`lensgroup.surfaces[0].c`, `lensgroup.surfaces[1].c`, `lensgroup.surfaces[1].d`, `lensgroup.origin`, `lensgroup.theta_x`, `lensgroup.theta_y`$\}$ |
| inputs | Parameter names |
| outputs | Differentiable parameter set $\mathcal{D}$ with 6 parameter groups |
| computational_note | Total degrees of freedom: $1 + 1 + 1 + 3 + 1 + 1 = 8$ scalar parameters |

#### Step 4.3: Initialize Adam Optimizer State
| Field | Value |
|-------|-------|
| step_name | Optimizer State Initialization |
| formula | $\mathbf{m}^{(0)} \leftarrow \mathbf{0}$, $\mathbf{v}^{(0)} \leftarrow \mathbf{0}$, $t \leftarrow 0$ |
| inputs | Parameter dimension |
| outputs | First moment estimate $\mathbf{m}^{(0)}$, second moment estimate $\mathbf{v}^{(0)}$, iteration counter $t$ |
| computational_note | Adam default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ |

**--- END INITIALIZATION ---**

### Iteration Loop

**--- BEGIN ITERATION LOOP ---**

For $t = 1, 2, \ldots, t_{max}$:

#### Step 4.4: Update Lens Group Geometry
| Field | Value |
|-------|-------|
| step_name | Geometry Update |
| formula | Recompute lens surface positions and transformation matrices from current $\boldsymbol{\theta}^{(t-1)}$ |
| inputs | Current parameters $\boldsymbol{\theta}^{(t-1)}$ |
| outputs | Updated lens group state |
| computational_note | Involves rotation matrix construction from $\theta_x$, $\theta_y$ and translation by $\mathbf{o}$ |

#### Step 4.5: Forward Ray Tracing
| Field | Value |
|-------|-------|
| step_name | Forward Model Evaluation |
| formula | $\mathbf{P}^{(t)} = \mathcal{T}(\boldsymbol{\theta}^{(t-1)}, \phi, \mathbf{V})$ where $\mathcal{T}$ performs ray tracing through lens system and returns 2D intersection coordinates on display plane |
| inputs | Parameters $\boldsymbol{\theta}^{(t-1)}$, rotation angle $\phi$, validity mask $\mathbf{V}$ |
| outputs | Predicted intersection points $\mathbf{P}^{(t)} \in \mathbb{R}^{N_{cam} \times H \times W \times 2}$ |
| computational_note | Ray tracing includes: ray generation from cameras, refraction at lens surfaces via Snell's law, intersection with display plane |

#### Step 4.6: Compute Loss
| Field | Value |
|-------|-------|
| step_name | Loss Evaluation |
| formula | $\mathcal{L}^{(t)} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \sum_{k=1}^{2} \left( P^{(t)}_{i,k} - \hat{P}_{i,k} \right)^2$ |
| inputs | Predicted points $\mathbf{P}^{(t)}$, measured points $\hat{\mathbf{P}}$, validity mask $\mathbf{V}$ |
| outputs | Scalar loss value $\mathcal{L}^{(t)} \in \mathbb{R}$ |
| computational_note | Mean computed over valid pixels only |

#### Step 4.7: Compute Residual Vector
| Field | Value |
|-------|-------|
| step_name | Residual Computation |
| formula | $\mathbf{r}^{(t)}_{i,k} = \begin{cases} \hat{P}_{i,k} - P^{(t)}_{i,k} & \text{if } V_i = 1 \text{ and } P^{(t)}_{i,k} \neq \text{NaN} \\ 0 & \text{otherwise} \end{cases}$ |
| inputs | Predicted points $\mathbf{P}^{(t)}$, measured points $\hat{\mathbf{P}}$, validity mask $\mathbf{V}$ |
| outputs | Residual tensor $\mathbf{r}^{(t)} \in \mathbb{R}^{N_{cam} \times H \times W \times 2}$ |
| computational_note | NaN handling prevents gradient corruption |

#### Step 4.8: Compute Gradients via Backpropagation
| Field | Value |
|-------|-------|
| step_name | Gradient Computation |
| formula | $\mathbf{g}^{(t)} = \nabla_{\boldsymbol{\theta}} \mathcal{L}^{(t)} = \frac{2}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \left( \mathbf{P}^{(t)}_i - \hat{\mathbf{P}}_i \right)^\top \frac{\partial \mathbf{P}^{(t)}_i}{\partial \boldsymbol{\theta}}$ |
| inputs | Loss $\mathcal{L}^{(t)}$, computational graph |
| outputs | Gradient vector $\mathbf{g}^{(t)} \in \mathbb{R}^{|\boldsymbol{\theta}|}$ |
| computational_note | Automatic differentiation through ray tracing operations |

#### Step 4.9: Adam Moment Updates
| Field | Value |
|-------|-------|
| step_name | First and Second Moment Estimation |
| formula | $\mathbf{m}^{(t)} = \beta_1 \mathbf{m}^{(t-1)} + (1 - \beta_1) \mathbf{g}^{(t)}$, $\mathbf{v}^{(t)} = \beta_2 \mathbf{v}^{(t-1)} + (1 - \beta_2) (\mathbf{g}^{(t)})^2$ |
| inputs | Previous moments $\mathbf{m}^{(t-1)}$, $\mathbf{v}^{(t-1)}$, gradient $\mathbf{g}^{(t)}$ |
| outputs | Updated moments $\mathbf{m}^{(t)}$, $\mathbf{v}^{(t)}$ |
| computational_note | Element-wise squaring for $(\mathbf{g}^{(t)})^2$; $\beta_1 = 0.9$, $\beta_2 = 0.999$ |

#### Step 4.10: Bias Correction
| Field | Value |
|-------|-------|
| step_name | Bias-Corrected Moment Estimates |
| formula | $\hat{\mathbf{m}}^{(t)} = \frac{\mathbf{m}^{(t)}}{1 - \beta_1^t}$, $\hat{\mathbf{v}}^{(t)} = \frac{\mathbf{v}^{(t)}}{1 - \beta_2^t}$ |
| inputs | Moments $\mathbf{m}^{(t)}$, $\mathbf{v}^{(t)}$, iteration $t$ |
| outputs | Bias-corrected estimates $\hat{\mathbf{m}}^{(t)}$, $\hat{\mathbf{v}}^{(t)}$ |
| computational_note | Corrects initialization bias toward zero |

#### Step 4.11: Parameter Update
| Field | Value |
|-------|-------|
| step_name | Adam Parameter Update |
| formula | $\boldsymbol{\theta}^{(t)} = \boldsymbol{\theta}^{(t-1)} - \alpha \frac{\hat{\mathbf{m}}^{(t)}}{\sqrt{\hat{\mathbf{v}}^{(t)}} + \epsilon}$ |
| inputs | Previous parameters $\boldsymbol{\theta}^{(t-1)}$, corrected moments $\hat{\mathbf{m}}^{(t)}$, $\hat{\mathbf{v}}^{(t)}$ |
| outputs | Updated parameters $\boldsymbol{\theta}^{(t)}$ |
| computational_note | $\alpha$ is learning rate (framework default), $\epsilon = 10^{-8}$ for numerical stability |

#### Step 4.12: Convergence Check
| Field | Value |
|-------|-------|
| step_name | Termination Condition Evaluation |
| formula | Continue if $t < t_{max}$ |
| inputs | Current iteration $t$, maximum iterations $t_{max}$ |
| outputs | Boolean: continue or terminate |
| computational_note | No explicit convergence threshold; fixed iteration count only |

**--- END ITERATION LOOP ---**

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $t \geq t_{max}$ |
| parameters | $t_{max} = 5$ (default value from `maxit` parameter) |

---

## 6. EVALUATION METRICS

### Metric 6.1: Mean Displacement Error
| Field | Value |
|-------|-------|
| name | Mean Euclidean Displacement Error |
| formula | $E = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \sqrt{\sum_{k=1}^{2} \left( P_{i,k} - \hat{P}_{i,k} \right)^2}$ |
| description | Average Euclidean distance between predicted and measured intersection points over valid pixels; reported in micrometers ($E \times 10^3$ μm) |

### Metric 6.2: Recovered Radius of Curvature
| Field | Value |
|-------|-------|
| name | Lens Surface Radius of Curvature |
| formula | $R_j = \frac{1}{c_j}$ for surface $j \in \{0, 1\}$ |
| description | Physical radius of curvature derived from optimized curvature parameters; compared against ground truth values |

### Metric 6.3: Pixel-wise Image Residual
| Field | Value |
|-------|-------|
| name | Rendered Image Error |
| formula | $\mathbf{E}_{img} = \mathbf{I}_0 - \mathbf{I}_{rendered}$ where $\mathbf{I}_{rendered} = \mathbf{V} \odot \mathcal{R}(\boldsymbol{\theta}, \phi)$ |
| description | Difference between measured reference images and forward-rendered images using optimized parameters; visualized with colormap range $[-1, 1]$ |

### Metric 6.4: Loss History
| Field | Value |
|-------|-------|
| name | Optimization Loss Trajectory |
| formula | $\{\mathcal{L}^{(1)}, \mathcal{L}^{(2)}, \ldots, \mathcal{L}^{(t_{max})}\}$ |
| description | Sequence of loss values across iterations; returned as list `ls` from optimization routine |