## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: BPM-FISTA (Beam Propagation Method Inversion with FISTA Acceleration)
- **algorithm_family**: Proximal Gradient Methods / FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
- **summary**: Reconstructs a 3D refractive index perturbation from optical diffraction tomography measurements by minimizing L1 data fidelity with composite regularization (TV, box constraints, sparsity) using nested FISTA optimization.

---

## 1. OBJECTIVE FUNCTION

### 1.1 Full Expression

$$\min_{\Delta n} \quad \frac{1}{N_{\text{angles}}} \sum_{i=1}^{N_{\text{angles}}} \left\| \mathcal{F}_{\Delta n}(u_{\text{in}}^{(i)}) - u_{\text{out}}^{(i)} \right\|_1 + \tau \, \text{TV}(\Delta n) + \lambda_s \|\Delta n\|_1 + \mathcal{I}_{[a,b]}(\Delta n) + \mathcal{I}_{\Omega}(\Delta n)$$

where $\mathcal{F}_{\Delta n}$ is the BPM forward operator, $\|\cdot\|_1$ denotes the mean absolute value (L1 norm averaged over pixels), and $\mathcal{I}$ denotes indicator functions.

### 1.2 Data Fidelity Term

$$\mathcal{D}(\Delta n) = \frac{1}{N_{\text{angles}}} \sum_{i=1}^{N_{\text{angles}}} \frac{1}{N_y N_x} \sum_{j,k} \left| [\mathcal{F}_{\Delta n}(u_{\text{in}}^{(i)})]_{j,k} - [u_{\text{out}}^{(i)}]_{j,k} \right|$$

**Verification**: The code computes `r.abs().mean()` where `r = u - sub_u_out`, confirming **L1 loss** (mean absolute error).

### 1.3 Regularization Terms

**Total Variation (Isotropic 3D)**:
$$\text{TV}(\Delta n) = \sum_{i,j,k} \sqrt{(\nabla_z \Delta n)_{i,j,k}^2 + (\nabla_y \Delta n)_{i,j,k}^2 + (\nabla_x \Delta n)_{i,j,k}^2}$$

**L1 Sparsity** (when `sparse_param` is not None):
$$\mathcal{R}_{\text{sparse}}(\Delta n) = \lambda_s \|\Delta n\|_1$$

### 1.4 Constraints

**Box Constraint** (applied within ROI $\Omega$):
$$\mathcal{I}_{[a,b]}(\Delta n) = \begin{cases} 0 & \text{if } a \leq \Delta n_{i,j,k} \leq b \quad \forall (i,j,k) \in \Omega \\ +\infty & \text{otherwise} \end{cases}$$

where $a = \texttt{value\_range\_param[0]}$, $b = \texttt{value\_range\_param[1]}$.

**ROI Constraint**: The box constraint and TV regularization are applied only within the region of interest $\Omega = [s_0:e_0, s_1:e_1, s_2:e_2]$.

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Delta n$ (`delta_ri`) | $(N_z, N_y, N_x)$ | float32 | Zeros | Refractive index perturbation |
| $x^{(k)}$ (`x_1`, `x_2`) | $(N_z, N_y, N_x)$ | float32 | Zeros | FISTA primal iterates |
| $s^{(k)}$ (`s`) | $(N_z, N_y, N_x)$ | float32 | Zeros | FISTA extrapolated point |
| $z^{(k)}$ (`z`) | $(N_z, N_y, N_x)$ | float32 | N/A (computed) | Gradient descent intermediate |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g$ [INNER] | $(N_z, N_y, N_x, 3)$ | float32 | $\nabla z$ | TV dual variable (gradient field) |
| $d$ [INNER] | $(N_z, N_y, N_x, 3)$ | float32 | $g^{(0)}$ | TV FISTA extrapolated dual |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $q^{(k)}$ (`q_1`, `q_2`) | scalar | float64 | $1.0$ | FISTA momentum coefficient |
| $u$ | $(N_{\text{batch}}, N_y, N_x)$ | complex64 | $u_{\text{in}}$ | Propagating optical field |
| $r$ | $(N_{\text{batch}}, N_y, N_x)$ | complex64 | N/A | Residual field |
| $S$ (`s_fields`) | $(N_{\text{batch}}, N_z, N_y, N_x)$ | complex64 | N/A | Stored intermediate fields for adjoint |
| `grad` | $(N_z, N_y, N_x)$ | float32 | Zeros | Accumulated gradient |
| `loss` | scalar | float64 | $0.0$ | Accumulated loss |

**[INNER] TV-FISTA Auxiliary Variables**:

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g_1, g_2$ | $(N_z, N_y, N_x, 3)$ | float32 | $\nabla z$ | Previous/current dual iterates |
| $q_1^{\text{tv}}, q_2^{\text{tv}}$ | scalar | float64 | $1.0$ | Inner FISTA momentum |
| $x^{\text{tv}}$ | $(N_z, N_y, N_x)$ | float32 | $z$ | Inner primal variable |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `wavelength` | $0.532$ | Wavelength in micrometers |
| `camera_pixel_size` | $4.5$ | Camera pixel size in micrometers |
| `magnification` | $100$ | Optical magnification |
| `n_medium` | $1.33$ | Refractive index of medium |
| `pixelsize` | $4.5/100 = 0.045$ | Effective pixel size (μm) |
| $k_0$ | $2\pi/\lambda$ | Vacuum wavenumber |
| $k_m$ | $2\pi n_{\text{medium}}/\lambda$ | Medium wavenumber |
| `step_size` | $0.0005$ | Gradient descent step size |
| $\tau$ (`tv_param[0]`) | $1.75 \times 10^{-6}$ | TV regularization weight |
| `tv_steps` (`tv_param[1]`) | $50$ | Number of inner TV-FISTA iterations |
| $\gamma$ | $1/(12\tau)$ | TV dual step size |
| `min_val` | $0$ | Lower box constraint |
| `max_val` | $100$ | Upper box constraint |
| `n_iter` | $1$ (configurable) | Outer iteration count |
| `batch_size` | $2$ | Gradient computation batch size |
| `crop_size` | $[1004, 1004]$ | Field crop dimensions |
| `domain_size` | $[502, 502, 502]$ | Reconstruction volume size |
| `region_z` | $256$ | Active axial slices |
| `bg_z` | $93$ | Background propagation depth |

### 2.5 Observations

| Symbol | Shape | Dtype | Source | Description |
|--------|-------|-------|--------|-------------|
| $u_{\text{in}}^{\text{raw}}$ | $(N_{\text{angles}}, N_y, N_x)$ | complex64/128 | `u_in.npy` | Raw input fields |
| $u_{\text{out}}^{\text{raw}}$ | $(N_{\text{angles}}, N_y, N_x)$ | complex64/128 | `u_out.npy` | Raw output fields |
| $k_{\text{scan}}$ | $(N_{\text{angles}}, 2)$ | float64 | `k_samp.npy` | Illumination k-vectors |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Physical Parameter Derivation

| Parameter | Formula | Value |
|-----------|---------|-------|
| `pixelsize` | $\texttt{camera\_pixel\_size} / \texttt{magnification}$ | $0.045$ μm |
| $k_0$ | $2\pi / \lambda$ | $\approx 11.81$ rad/μm |
| $k_m$ | $2\pi n_{\text{medium}} / \lambda$ | $\approx 15.71$ rad/μm |
| `spec_pixel_size` | $2\pi / (\texttt{pixelsize} \times \texttt{crop\_size}[0])$ | $\approx 0.139$ rad/μm |
| `resolution` | $\texttt{pixelsize} \times \texttt{crop\_size}[0] / \texttt{domain\_size}$ | $(0.09, 0.09, 0.09)$ μm |

### Step 1: Angular Spectrum Kernel Construction

**Inputs**: `domain_size` $(3,)$, `spec_pixel_size` (scalar), `resolution` $(3,)$, $k_m$ (scalar)

**Formula**:
$$K_x[j] = \left(\text{linspace}(-N_y/2+1, N_y/2, N_y)[j] - 1\right) \cdot \Delta k$$
$$K_y[i] = \left(\text{linspace}(-N_x/2+1, N_x/2, N_x)[i] - 1\right) \cdot \Delta k$$
$$K_z = \sqrt{k_m^2 - K_x^2 - K_y^2 + 0j}$$
$$K_z[k_m^2 - K_x^2 - K_y^2 < 0] = 0 \quad \text{(evanescent suppression)}$$
$$P = \text{fftshift}\left(\exp(i K_z \cdot \Delta z)\right)$$

**Outputs**: `p_kernel` $(N_y, N_x)$ complex64 [FORWARD propagation kernel]

### Step 2: Oblique Illumination Cosine Factor

**Inputs**: $k_{\text{scan}}$ $(N_{\text{angles}}, 2)$, `spec_pixel_size`, $k_m$

**Formula**:
$$\theta_i = \arcsin\left(\sqrt{(k_{\text{scan},i,0} \cdot \Delta k / k_m)^2 + (k_{\text{scan},i,1} \cdot \Delta k / k_m)^2}\right)$$
$$\text{cosFactor}_i = \cos(\theta_i)$$

**Outputs**: `bpm_cosFactor` $(N_{\text{angles}}, 1)$ float32

### Step 3: Field Back-Propagation to Volume Boundaries

**Inputs**: $u_{\text{in}}^{\text{raw}}$, $u_{\text{out}}^{\text{raw}}$, `p_kernel`, `region_z`, `bg_z`

**Formula**:
$$u_{\text{inlet}} = \mathcal{F}^{-1}\left\{\mathcal{F}\{u_{\text{in}}^{\text{raw}}\} \cdot P^{*(N_z - N_{\text{bg}})}\right\}$$
$$u_{\text{outlet}} = \mathcal{F}^{-1}\left\{\mathcal{F}\{u_{\text{out}}^{\text{raw}}\} \cdot P^{N_{\text{bg}}}\right\}$$

where $P^*$ denotes complex conjugate (backward propagation), $N_z = \texttt{region\_z} = 256$, $N_{\text{bg}} = \texttt{bg\_z} = 93$.

**Outputs**: `u_inlet` $(N_{\text{angles}}, N_y, N_x)$, `u_outlet` $(N_{\text{angles}}, N_y, N_x)$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Operation | Formula | Output |
|------|-----------|---------|--------|
| I1 | Initialize primal | $\Delta n^{(0)} = \mathbf{0}_{N_z \times N_y \times N_x}$ | `init_delta_ri` |
| I2 | Initialize extrapolated point | $s^{(0)} = \Delta n^{(0)}$ | `s` |
| I3 | Initialize previous iterate | $x^{(0)} = \Delta n^{(0)}$ | `x_1` |
| I4 | Initialize momentum | $q^{(0)} = 1$ | `q_1` |
| I5 | Initialize loss history | $\mathcal{L} = [\,]$ | `loss_history` |
| I6 | Precompute oblique factor | $\alpha_i = k_0 \Delta z / \text{cosFactor}_i$ | `ol_factor` $(N_{\text{angles}}, 1)$ |
| I7 | Precompute backprop kernel | $P^{\dagger} = P^*$ | `bp_kernel` |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (Outer FISTA: $k = 0, 1, \ldots, N_{\text{iter}}-1$)

#### Step 1: Batched Gradient Computation

**BEGIN_LOOP** (Batch loop: $b = 0, \ldots, \lceil N_{\text{angles}}/B \rceil - 1$)

##### Step 1.1: BPM Forward Pass [FORWARD]

**BEGIN_LOOP** (Slice loop: $m = 0, \ldots, N_z - 1$)

| Substep | Formula | Output |
|---------|---------|--------|
| 1.1.a | $\tilde{u}^{(m)} = \mathcal{F}^{-1}\{\mathcal{F}\{u^{(m-1)}\} \cdot P\}$ | Propagated field |
| 1.1.b | $S_{:,m,:,:} = \tilde{u}^{(m)}$ | Store for adjoint |
| 1.1.c | $u^{(m)} = \tilde{u}^{(m)} \odot \exp(i \alpha \Delta n_m)$ | Phase-modulated field |

**END_LOOP** (Slice)

##### Step 1.2: Residual and Loss

$$r = u^{(N_z)} - u_{\text{outlet}}$$
$$\mathcal{L}_b = \frac{1}{N_y N_x} \sum_{j,k} |r_{j,k}|$$

##### Step 1.3: Adjoint Gradient [ADJOINT]

**BEGIN_LOOP** (Reverse slice loop: $m = N_z - 1, \ldots, 0$)

| Substep | Formula | Output |
|---------|---------|--------|
| 1.3.a | $r \leftarrow r \odot \exp(-i \alpha \Delta n_m)$ | Undo phase modulation |
| 1.3.b | $g_m = g_m + \text{Re}\left\{\sum_{\text{batch}} -i \alpha \cdot S_{:,m}^* \odot r\right\}$ | Accumulate gradient |
| 1.3.c | $r \leftarrow \mathcal{F}^{-1}\{\mathcal{F}\{r\} \cdot P^{\dagger}\}$ | Backpropagate residual |

**END_LOOP** (Reverse slice)

**END_LOOP** (Batch)

##### Step 1.4: Normalize Gradient

$$\nabla \mathcal{L} = \frac{g}{N_{\text{angles}}}$$

#### Step 2: Gradient Descent Update

$$z^{(k)} = s^{(k)} - \eta \nabla \mathcal{L}$$

**Inputs**: $s^{(k)}$, $\nabla \mathcal{L}$, $\eta = 0.0005$  
**Outputs**: $z^{(k)}$

#### Step 3: Proximal Operator (Composite Regularization)

$$x^{(k+1)} = \text{prox}_{\mathcal{R}}(z^{(k)})$$

The proximal operator is computed via sequential application:

##### Step 3.1: TV-FISTA Denoising [INNER]

**BEGIN_LOOP** (TV inner: $t = 0, \ldots, 49$)

| Substep | Formula | Output |
|---------|---------|--------|
| 3.1.a | $v = z - \tau \cdot \text{div}(d^{(t)})$ | Primal update |
| 3.1.b | $\tilde{v} = \Pi_{[a,b]}(v)$ | Box projection |
| 3.1.c | $\tilde{g} = d^{(t)} + \gamma \nabla \tilde{v}$ | Dual gradient step |
| 3.1.d | $g^{(t+1)} = \Pi_{\|\cdot\| \leq 1}(\tilde{g})$ | Dual projection |
| 3.1.e | $x^{\text{tv}} = \Pi_{[a,b]}(z - \tau \cdot \text{div}(g^{(t+1)}))$ | Primal from new dual |
| 3.1.f | $q^{(t+1)} = \frac{1 + \sqrt{1 + 4(q^{(t)})^2}}{2}$ | Momentum update |
| 3.1.g | $d^{(t+1)} = g^{(t+1)} + \frac{q^{(t)} - 1}{q^{(t+1)}}(g^{(t+1)} - g^{(t)})$ | Dual extrapolation |

**END_LOOP** (TV inner)

**Operator Definitions**:

**Gradient operator** $\nabla: \mathbb{R}^{N_z \times N_y \times N_x} \to \mathbb{R}^{N_z \times N_y \times N_x \times 3}$:
$$[\nabla x]_{i,j,k,0} = x_{i+1,j,k} - x_{i,j,k} \quad (i < N_z - 1), \quad 0 \text{ otherwise}$$
$$[\nabla x]_{i,j,k,1} = x_{i,j+1,k} - x_{i,j,k} \quad (j < N_y - 1), \quad 0 \text{ otherwise}$$
$$[\nabla x]_{i,j,k,2} = x_{i,j,k+1} - x_{i,j,k} \quad (k < N_x - 1), \quad 0 \text{ otherwise}$$

**Divergence operator** $\text{div}: \mathbb{R}^{N_z \times N_y \times N_x \times 3} \to \mathbb{R}^{N_z \times N_y \times N_x}$ (negative adjoint of $\nabla$):
$$[\text{div}(g)]_{i,j,k} = -\left([\nabla^* g]_{i,j,k,0} + [\nabla^* g]_{i,j,k,1} + [\nabla^* g]_{i,j,k,2}\right)$$

with boundary conditions:
- Dimension 0: $g_{0,:,:,0}$ at start, $-g_{-2,:,:,0}$ at end, $g_{i,:,:,0} - g_{i-1,:,:,0}$ interior
- Analogous for dimensions 1 and 2

**Dual projection** (pointwise onto unit ball):
$$[\Pi_{\|\cdot\| \leq 1}(g)]_{i,j,k,:} = \frac{g_{i,j,k,:}}{\max(1, \|g_{i,j,k,:}\|_2)}$$

##### Step 3.2: Box Constraint Projection

$$[\Pi_{[a,b]}(x)]_{i,j,k} = \begin{cases} \text{clamp}(x_{i,j,k}, a, b) & (i,j,k) \in \Omega \\ x_{i,j,k} & \text{otherwise} \end{cases}$$

##### Step 3.3: Soft Thresholding (if `sparse_param` is not None)

$$[\mathcal{S}_{\lambda}(x)]_{i,j,k} = \text{sign}(x_{i,j,k}) \max(|x_{i,j,k}| - \lambda, 0)$$

where $\lambda = \texttt{sparse\_param} \times \eta$.

**Note**: In current config, `sparse_param = [None]`, so this step is identity.

#### Step 4: FISTA Momentum Update

$$q^{(k+1)} = \frac{1 + \sqrt{1 + 4(q^{(k)})^2}}{2}$$

$$s^{(k+1)} = x^{(k+1)} + \frac{q^{(k)} - 1}{q^{(k+1)}}(x^{(k+1)} - x^{(k)})$$

**Inputs**: $x^{(k+1)}$, $x^{(k)}$, $q^{(k)}$  
**Outputs**: $s^{(k+1)}$, $q^{(k+1)}$

#### Step 5: Update State Variables

$$x^{(k)} \leftarrow x^{(k+1)}, \quad q^{(k)} \leftarrow q^{(k+1)}$$

**END_LOOP** (Outer FISTA)

### 4.3 FINALIZATION

| Step | Operation | Formula/Action |
|------|-----------|----------------|
| F1 | Extract solution | $\Delta n^* = s^{(N_{\text{iter}})}$ |
| F2 | Transfer to CPU | `delta_ri = s.cpu().numpy()` |
| F3 | Save output | Write to `io/output.npy` as float32 array |

---

## 5. STOPPING CRITERION

- **type**: `iteration_limit`
- **expression**: $k < N_{\text{iter}}$
- **parameters**: 
  - `n_iter` = 1 (outer loop, configurable)
  - `tv_steps` = 50 (inner TV loop, fixed)
- **check_frequency**: Every outer iteration

---

## 6. EVALUATION METRICS

| Metric Name | Formula | Description |
|-------------|---------|-------------|
| `roi_min` | $\min_{(i,j,k) \in \Omega} \Delta n^*_{i,j,k}$ | Minimum RI in ROI |
| `roi_max` | $\max_{(i,j,k) \in \Omega} \Delta n^*_{i,j,k}$ | Maximum RI in ROI |
| `roi_mean` | $\frac{1}{|\Omega|} \sum_{(i,j,k) \in \Omega} \Delta n^*_{i,j,k}$ | Mean RI in ROI |
| `roi_std` | $\sqrt{\frac{1}{|\Omega|} \sum_{(i,j,k) \in \Omega} (\Delta n^*_{i,j,k} - \bar{\Delta n})^2}$ | Standard deviation in ROI |
| `initial_loss` | $\mathcal{L}^{(0)}$ | Loss at first iteration |
| `final_loss` | $\mathcal{L}^{(N_{\text{iter}}-1)}$ | Loss at last iteration |
| `num_iterations` | $|\texttt{loss\_history}|$ | Total iterations completed |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON extraction MUST include:

```
algorithm_identification:
  - algorithm_name
  - algorithm_family
  - summary

objective_function:
  - full_expression
  - data_fidelity_term
  - regularization_terms (array)
  - constraints (array)

variables:
  - primal_variables (array of {symbol, shape, dtype, initialization, description})
  - dual_variables (array)
  - auxiliary_variables (array)
  - constants (array of {symbol, value, description})
  - observations (array)

data_preprocessing:
  - steps (array of {step_order, step_name, formula, inputs, outputs})

initialization:
  - steps (array of {step_id, operation, formula, output})

iteration_loop:
  - outer_loop:
      - loop_variable
      - range
      - steps (array, with nested loops marked)
  - inner_loops (array with same structure)

finalization:
  - steps (array)

stopping_criterion:
  - type
  - expression
  - parameters
  - check_frequency

evaluation:
  - metrics (array of {name, formula, description})
```