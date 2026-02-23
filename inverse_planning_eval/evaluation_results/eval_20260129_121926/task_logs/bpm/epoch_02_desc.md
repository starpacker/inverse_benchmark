# Algorithmic Specification Document: BPM Optical Diffraction Tomography Reconstruction

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\Delta n \in \mathcal{C}} \quad \mathcal{L}(\Delta n) = \frac{1}{N_{\text{illum}}} \sum_{i=1}^{N_{\text{illum}}} \left\| \mathcal{F}_{\Delta n}(u_{\text{in}}^{(i)}) - u_{\text{out}}^{(i)} \right\|_1 + \tau \cdot \text{TV}(\Delta n) + \lambda_s \|\Delta n\|_1$$

where $\mathcal{C} = \{\Delta n : \Delta n_{\min} \leq \Delta n[\mathbf{r}] \leq \Delta n_{\max}, \; \forall \mathbf{r} \in \text{ROI}\}$

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}}(\Delta n) = \frac{1}{N_{\text{illum}}} \sum_{i=1}^{N_{\text{illum}}} \frac{1}{N_y N_x} \sum_{p,q} \left| [\mathcal{F}_{\Delta n}(u_{\text{in}}^{(i)})]_{p,q} - [u_{\text{out}}^{(i)}]_{p,q} \right|$$

where $\mathcal{F}_{\Delta n}$ is the BPM forward operator defined in Section 4.

**regularization_term**:
$$\mathcal{R}(\Delta n) = \tau \cdot \text{TV}(\Delta n) + \lambda_s \|\Delta n\|_1$$

where:
$$\text{TV}(\Delta n) = \sum_{m,p,q} \|\nabla \Delta n[m,p,q]\|_2$$
$$\nabla \Delta n[m,p,q] = \begin{pmatrix} \Delta n[m+1,p,q] - \Delta n[m,p,q] \\ \Delta n[m,p+1,q] - \Delta n[m,p,q] \\ \Delta n[m,p,q+1] - \Delta n[m,p,q] \end{pmatrix}$$

**constraints**:
1. Box constraint on ROI: $\Delta n_{\min} \leq \Delta n[m,p,q] \leq \Delta n_{\max}$ for $(m,p,q) \in \text{ROI}$
2. ROI defined by: $m \in [s_0, e_0)$, $p \in [s_1, e_1)$, $q \in [s_2, e_2)$

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\Delta n$ | primal | $(N_z, N_y, N_x)$ | float32 | $\mathbf{0}$ | Refractive index perturbation |
| $\mathbf{x}^{(t)}$ | primal | $(N_z, N_y, N_x)$ | float32 | $\mathbf{0}$ | FISTA iterate at outer iteration $t$ |
| $\mathbf{s}^{(t)}$ | primal | $(N_z, N_y, N_x)$ | float32 | $\mathbf{0}$ | FISTA momentum variable |
| $\mathbf{g}^{(k)}$ | primal | $(N_z, N_y, N_x, 3)$ | float32 | $\nabla \mathbf{z}$ | TV dual variable at inner iteration $k$ |

### Dual/Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $q^{(t)}$ | dual | scalar | float64 | $1$ | Outer FISTA momentum parameter |
| $q_{\text{tv}}^{(k)}$ | dual | scalar | float64 | $1$ | Inner TV-FISTA momentum parameter |
| $u$ | dual | $(N_{\text{batch}}, N_y, N_x)$ | complex64 | $u_{\text{in}}$ | Propagating field during forward pass |
| $r$ | dual | $(N_{\text{batch}}, N_y, N_x)$ | complex64 | $u - u_{\text{out}}$ | Residual/adjoint field |
| $S$ | dual | $(N_{\text{batch}}, N_z, N_y, N_x)$ | complex64 | computed | Stored intermediate fields |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $k_0$ | constants | scalar | float64 | $2\pi/\lambda$ | Vacuum wavenumber |
| $k_m$ | constants | scalar | float64 | $2\pi n_m/\lambda$ | Medium wavenumber |
| $\Delta z$ | constants | scalar | float64 | $\delta_{\text{cam}} N_{\text{crop}} / (M \cdot N_z)$ | Axial voxel size |
| $H$ | constants | $(N_y, N_x)$ | complex64 | computed | Angular spectrum propagation kernel |
| $\boldsymbol{\eta}$ | constants | $(N_{\text{illum}}, 1)$ | float32 | computed | Oblique illumination correction factors |
| $\tau$ | constants | scalar | float64 | `tv_param[0]` | TV regularization weight |
| $K_{\text{tv}}$ | constants | scalar | int | `tv_param[1]` | Number of TV inner iterations |
| $\gamma$ | constants | scalar | float64 | $1/(12\tau)$ | TV dual step size |
| $\alpha$ | constants | scalar | float64 | `step_size` | Gradient descent step size |
| $\lambda_s$ | constants | scalar | float64 | `sparse_param` | Sparsity parameter (or None) |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $u_{\text{in}}^{\text{raw}}$ | observations | $(N_{\text{illum}}, N_y, N_x)$ | complex64 | loaded | Raw input fields |
| $u_{\text{out}}^{\text{raw}}$ | observations | $(N_{\text{illum}}, N_y, N_x)$ | complex64 | loaded | Raw output fields |
| $\mathbf{k}_{\text{scan}}$ | observations | $(N_{\text{illum}}, 2)$ | float64 | loaded | Scan k-vector samples |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Physical Parameter Computation
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | compute_physical_parameters |
| formula | $$k_0 = \frac{2\pi}{\lambda}, \quad k_m = \frac{2\pi n_m}{\lambda}, \quad \delta = \frac{\delta_{\text{cam}}}{M}$$ $$\Delta k = \frac{2\pi}{\delta \cdot N_{\text{crop}}}, \quad \boldsymbol{\Delta r} = \frac{\delta \cdot N_{\text{crop}}}{(N_z, N_y, N_x)}$$ |
| inputs | $\lambda$ (scalar), $\delta_{\text{cam}}$ (scalar), $M$ (scalar), $n_m$ (scalar), $N_{\text{crop}}$ (scalar), $(N_z, N_y, N_x)$ (3-tuple) |
| outputs | $k_0$ (scalar), $k_m$ (scalar), $\Delta k$ (scalar), $\boldsymbol{\Delta r} = (\Delta z, \Delta y, \Delta x)$ (3-tuple) |

### Step 3.2: Angular Spectrum Kernel Construction
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | angular_spectrum_kernel |
| formula | $$K_x[p] = \left(\left\lfloor -\frac{N_y}{2} + 1 \right\rfloor + p - 1\right) \cdot \Delta k, \quad p = 0, \ldots, N_y-1$$ $$K_y[q] = \left(\left\lfloor -\frac{N_x}{2} + 1 \right\rfloor + q - 1\right) \cdot \Delta k, \quad q = 0, \ldots, N_x-1$$ $$K_z[p,q] = \begin{cases} \sqrt{k_m^2 - K_x[p]^2 - K_y[q]^2} & \text{if } K_x[p]^2 + K_y[q]^2 \leq k_m^2 \\ 0 & \text{otherwise} \end{cases}$$ $$\tilde{H}[p,q] = \exp(i \cdot K_z[p,q] \cdot \Delta z)$$ $$H = \text{fftshift}(\tilde{H})$$ |
| inputs | $(N_z, N_y, N_x)$ (3-tuple), $\Delta k$ (scalar), $\Delta z$ (scalar), $k_m$ (scalar) |
| outputs | $H$ (shape: $N_y \times N_x$, dtype: complex64) |

### Step 3.3: Oblique Illumination Factor Computation
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | compute_oblique_factors |
| formula | $$\tilde{k}_i = \mathbf{k}_{\text{scan}}[i,:] \cdot \frac{\Delta k}{k_m}, \quad i = 0, \ldots, N_{\text{illum}}-1$$ $$\theta_i = \arcsin\left(\sqrt{\tilde{k}_i[0]^2 + \tilde{k}_i[1]^2}\right)$$ $$\eta_i = \cos(\theta_i)$$ |
| inputs | $\mathbf{k}_{\text{scan}}$ (shape: $N_{\text{illum}} \times 2$), $\Delta k$ (scalar), $k_m$ (scalar) |
| outputs | $\boldsymbol{\eta}$ (shape: $N_{\text{illum}} \times 1$, dtype: float32) |

### Step 3.4: Field Propagation to Volume Boundaries
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | propagate_fields_to_boundaries |
| formula | $$u_{\text{inlet}} = \mathcal{F}^{-1}\left\{ \mathcal{F}\{u_{\text{in}}^{\text{raw}}\} \cdot \left(H^*\right)^{N_z - N_{\text{bg}}} \right\}$$ $$u_{\text{outlet}} = \mathcal{F}^{-1}\left\{ \mathcal{F}\{u_{\text{out}}^{\text{raw}}\} \cdot H^{N_{\text{bg}}} \right\}$$ where $\mathcal{F}$ denotes 2D FFT applied to each illumination, $H^*$ is complex conjugate, and exponentiation is element-wise. |
| inputs | $u_{\text{in}}^{\text{raw}}$ (shape: $N_{\text{illum}} \times N_y \times N_x$), $u_{\text{out}}^{\text{raw}}$ (shape: $N_{\text{illum}} \times N_y \times N_x$), $H$ (shape: $N_y \times N_x$), $N_z$ (scalar), $N_{\text{bg}}$ (scalar) |
| outputs | $u_{\text{inlet}}$ (shape: $N_{\text{illum}} \times N_y \times N_x$, dtype: complex64), $u_{\text{outlet}}$ (shape: $N_{\text{illum}} \times N_y \times N_x$, dtype: complex64) |

---

## 4. ALGORITHM STEPS

### Initialization

| Field | Value |
|-------|-------|
| step_name | initialize_primal_variables |
| formula | $$\mathbf{x}^{(0)} = \mathbf{0} \in \mathbb{R}^{N_z \times N_y \times N_x}$$ $$\mathbf{s}^{(0)} = \mathbf{x}^{(0)}$$ $$q^{(0)} = 1$$ |
| inputs | $(N_z, N_y, N_x)$ |
| outputs | $\mathbf{x}^{(0)}$, $\mathbf{s}^{(0)}$, $q^{(0)}$ |
| computational_note | All arrays initialized on GPU device |

---

### Iteration Loop

**BEGIN LOOP** for $t = 0, 1, \ldots, T-1$:

#### Step 4.1: Batch Partitioning
| Field | Value |
|-------|-------|
| step_name | partition_batches |
| formula | $$N_{\text{batches}} = \left\lceil \frac{N_{\text{illum}}}{B} \right\rceil$$ $$\text{For batch } b: \quad i_{\text{start}}^{(b)} = b \cdot B, \quad i_{\text{end}}^{(b)} = \min((b+1) \cdot B, N_{\text{illum}})$$ |
| inputs | $N_{\text{illum}}$ (scalar), $B$ (batch size, scalar) |
| outputs | $N_{\text{batches}}$ (scalar), index ranges for each batch |
| computational_note | Enables memory-efficient processing on GPU |

#### Step 4.2: Forward BPM Propagation (per batch)
| Field | Value |
|-------|-------|
| step_name | forward_bpm_propagation |
| formula | $$\phi_m^{(i)} = \frac{k_0 \cdot \Delta z}{\eta_i} \cdot \mathbf{s}^{(t)}[m,:,:], \quad m = 0, \ldots, N_z-1$$ $$u^{(i)}_{-1} = u_{\text{inlet}}[i,:,:]$$ For $m = 0, \ldots, N_z - 1$: $$\tilde{u}_m^{(i)} = \mathcal{F}^{-1}\left\{ \mathcal{F}\{u^{(i)}_{m-1}\} \cdot H \right\}$$ $$S[i, m, :, :] = \tilde{u}_m^{(i)}$$ $$u^{(i)}_m = \tilde{u}_m^{(i)} \cdot \exp\left(i \cdot \phi_m^{(i)}\right)$$ Output field: $\hat{u}_{\text{out}}^{(i)} = u^{(i)}_{N_z - 1}$ |
| inputs | $\mathbf{s}^{(t)}$ (shape: $N_z \times N_y \times N_x$), $u_{\text{inlet}}$ (batch slice), $H$, $k_0$, $\Delta z$, $\boldsymbol{\eta}$ (batch slice) |
| outputs | $S$ (shape: $B_{\text{actual}} \times N_z \times N_y \times N_x$), $\hat{u}_{\text{out}}$ (shape: $B_{\text{actual}} \times N_y \times N_x$) |
| computational_note | $S$ stores intermediate fields for adjoint computation |

#### Step 4.3: Residual Computation
| Field | Value |
|-------|-------|
| step_name | compute_residual |
| formula | $$r^{(i)} = \hat{u}_{\text{out}}^{(i)} - u_{\text{outlet}}[i,:,:]$$ $$\mathcal{L}_{\text{batch}} = \frac{1}{B_{\text{actual}}} \sum_{i} \frac{1}{N_y N_x} \sum_{p,q} |r^{(i)}[p,q]|$$ |
| inputs | $\hat{u}_{\text{out}}$ (shape: $B_{\text{actual}} \times N_y \times N_x$), $u_{\text{outlet}}$ (batch slice) |
| outputs | $r$ (shape: $B_{\text{actual}} \times N_y \times N_x$), $\mathcal{L}_{\text{batch}}$ (scalar) |
| computational_note | Loss computed as mean absolute error |

#### Step 4.4: Adjoint BPM Backpropagation
| Field | Value |
|-------|-------|
| step_name | adjoint_backpropagation |
| formula | $$H_{\text{adj}} = H^*$$ Initialize: $\nabla_{\text{batch}} = \mathbf{0} \in \mathbb{R}^{N_z \times N_y \times N_x}$ For $m = N_z - 1, \ldots, 0$: $$r \leftarrow r \cdot \exp\left(-i \cdot \phi_m^{(i)}\right)$$ $$\nabla_{\text{batch}}[m,:,:] \mathrel{+}= \sum_{i} \text{Re}\left\{ -i \cdot \frac{k_0 \Delta z}{\eta_i} \cdot S[i,m,:,:]^* \cdot r^{(i)} \right\}$$ $$r^{(i)} \leftarrow \mathcal{F}^{-1}\left\{ \mathcal{F}\{r^{(i)}\} \cdot H_{\text{adj}} \right\}$$ |
| inputs | $r$ (shape: $B_{\text{actual}} \times N_y \times N_x$), $S$ (stored fields), $\mathbf{s}^{(t)}$, $H$, $k_0$, $\Delta z$, $\boldsymbol{\eta}$ |
| outputs | $\nabla_{\text{batch}}$ (shape: $N_z \times N_y \times N_x$) |
| computational_note | Adjoint method avoids storing full computational graph |

#### Step 4.5: Gradient Aggregation
| Field | Value |
|-------|-------|
| step_name | aggregate_gradient |
| formula | $$\nabla \mathcal{L} = \frac{1}{N_{\text{illum}}} \sum_{b=0}^{N_{\text{batches}}-1} \nabla_{\text{batch}}^{(b)}$$ $$\mathcal{L}^{(t)} = \frac{1}{N_{\text{batches}}} \sum_{b=0}^{N_{\text{batches}}-1} \mathcal{L}_{\text{batch}}^{(b)}$$ |
| inputs | All $\nabla_{\text{batch}}^{(b)}$, all $\mathcal{L}_{\text{batch}}^{(b)}$ |
| outputs | $\nabla \mathcal{L}$ (shape: $N_z \times N_y \times N_x$), $\mathcal{L}^{(t)}$ (scalar) |
| computational_note | Gradient normalized by total number of illuminations |

#### Step 4.6: Gradient Descent Step
| Field | Value |
|-------|-------|
| step_name | gradient_descent_update |
| formula | $$\mathbf{z} = \mathbf{s}^{(t)} - \alpha \cdot \nabla \mathcal{L}$$ |
| inputs | $\mathbf{s}^{(t)}$ (shape: $N_z \times N_y \times N_x$), $\nabla \mathcal{L}$ (shape: $N_z \times N_y \times N_x$), $\alpha$ (scalar) |
| outputs | $\mathbf{z}$ (shape: $N_z \times N_y \times N_x$) |
| computational_note | No gradient tracking during this update |

#### Step 4.7: Proximal Operator (Regularization)

**Step 4.7.1: TV Denoising via FISTA (Inner Loop)**

| Field | Value |
|-------|-------|
| step_name | tv_fista_initialization |
| formula | $$\mathbf{g}^{(0)} = \nabla \mathbf{z}$$ $$\mathbf{d}^{(0)} = \mathbf{g}^{(0)}$$ $$q_{\text{tv}}^{(0)} = 1$$ where gradient operator: $$[\nabla \mathbf{z}]_{m,p,q,0} = \begin{cases} \mathbf{z}[m+1,p,q] - \mathbf{z}[m,p,q] & m < N_z - 1 \\ 0 & m = N_z - 1 \end{cases}$$ $$[\nabla \mathbf{z}]_{m,p,q,1} = \begin{cases} \mathbf{z}[m,p+1,q] - \mathbf{z}[m,p,q] & p < N_y - 1 \\ 0 & p = N_y - 1 \end{cases}$$ $$[\nabla \mathbf{z}]_{m,p,q,2} = \begin{cases} \mathbf{z}[m,p,q+1] - \mathbf{z}[m,p,q] & q < N_x - 1 \\ 0 & q = N_x - 1 \end{cases}$$ |
| inputs | $\mathbf{z}$ (shape: $N_z \times N_y \times N_x$) |
| outputs | $\mathbf{g}^{(0)}$, $\mathbf{d}^{(0)}$ (shape: $N_z \times N_y \times N_x \times 3$), $q_{\text{tv}}^{(0)}$ (scalar) |

**BEGIN INNER LOOP** for $k = 0, 1, \ldots, K_{\text{tv}} - 1$:

| Field | Value |
|-------|-------|
| step_name | tv_inner_iteration |
| formula | **Divergence operator** (negative adjoint of gradient): $$[\text{div}(\mathbf{g})]_{m,p,q} = -\Big( D_0[\mathbf{g}]_{m,p,q} + D_1[\mathbf{g}]_{m,p,q} + D_2[\mathbf{g}]_{m,p,q} \Big)$$ where: $$D_0[\mathbf{g}]_{m,p,q} = \begin{cases} \mathbf{g}[0,p,q,0] & m = 0 \\ \mathbf{g}[m,p,q,0] - \mathbf{g}[m-1,p,q,0] & 0 < m < N_z - 1 \\ -\mathbf{g}[N_z-2,p,q,0] & m = N_z - 1 \end{cases}$$ (analogous definitions for $D_1$, $D_2$ along dimensions 1, 2) **TV iteration**: $$\mathbf{w}_1 = \mathbf{z} - \tau \cdot \text{div}(\mathbf{d}^{(k)})$$ $$\mathbf{w}_2 = \text{clamp}_{\text{ROI}}(\mathbf{w}_1; \Delta n_{\min}, \Delta n_{\max})$$ $$\mathbf{w}_3 = \nabla \mathbf{w}_2$$ $$\mathbf{w}_4 = \mathbf{d}^{(k)} + \gamma \cdot \mathbf{w}_3$$ **Projection onto unit ball**: $$\nu_{m,p,q} = \|\mathbf{w}_4[m,p,q,:]\|_2$$ $$\tilde{\nu}_{m,p,q} = \max(\nu_{m,p,q}, 1)$$ $$\mathbf{g}^{(k+1)}[m,p,q,:] = \frac{\mathbf{w}_4[m,p,q,:]}{\tilde{\nu}_{m,p,q}}$$ **Primal update**: $$\mathbf{x}_{\text{tv}} = \text{clamp}_{\text{ROI}}(\mathbf{z} - \tau \cdot \text{div}(\mathbf{g}^{(k+1)}); \Delta n_{\min}, \Delta n_{\max})$$ **FISTA momentum**: $$q_{\text{tv}}^{(k+1)} = \frac{1 + \sqrt{1 + 4(q_{\text{tv}}^{(k)})^2}}{2}$$ $$\mathbf{d}^{(k+1)} = \mathbf{g}^{(k+1)} + \frac{q_{\text{tv}}^{(k)} - 1}{q_{\text{tv}}^{(k+1)}} \left( \mathbf{g}^{(k+1)} - \mathbf{g}^{(k)} \right)$$ |
| inputs | $\mathbf{z}$, $\mathbf{d}^{(k)}$, $\mathbf{g}^{(k)}$, $q_{\text{tv}}^{(k)}$, $\tau$, $\gamma$, ROI bounds |
| outputs | $\mathbf{g}^{(k+1)}$, $\mathbf{d}^{(k+1)}$, $q_{\text{tv}}^{(k+1)}$, $\mathbf{x}_{\text{tv}}$ |

**END INNER LOOP**

where clamping operation:
$$[\text{clamp}_{\text{ROI}}(\mathbf{w}; a, b)]_{m,p,q} = \begin{cases} \max(\min(\mathbf{w}[m,p,q], b), a) & (m,p,q) \in \text{ROI} \\ \mathbf{w}[m,p,q] & \text{otherwise} \end{cases}$$

**Step 4.7.2: Sparse Regularization (Soft Thresholding)**

| Field | Value |
|-------|-------|
| step_name | sparse_soft_thresholding |
| formula | $$\mathbf{x}^{(t+1)} = \begin{cases} \text{sign}(\mathbf{x}_{\text{tv}}) \cdot \max(|\mathbf{x}_{\text{tv}}| - \lambda_s \alpha, 0) & \text{if } \lambda_s \neq \text{None} \\ \mathbf{x}_{\text{tv}} & \text{if } \lambda_s = \text{None} \end{cases}$$ |
| inputs | $\mathbf{x}_{\text{tv}}$ (shape: $N_z \times N_y \times N_x$), $\lambda_s$ (scalar or None), $\alpha$ (step size) |
| outputs | $\mathbf{x}^{(t+1)}$ (shape: $N_z \times N_y \times N_x$) |
| computational_note | Applied element-wise; threshold is $\lambda_s \cdot \alpha$ |

#### Step 4.8: Outer FISTA Momentum Update
| Field | Value |
|-------|-------|
| step_name | outer_fista_momentum |
| formula | $$q^{(t+1)} = \frac{1 + \sqrt{1 + 4(q^{(t)})^2}}{2}$$ $$\mathbf{s}^{(t+1)} = \mathbf{x}^{(t+1)} + \frac{q^{(t)} - 1}{q^{(t+1)}} \left( \mathbf{x}^{(t+1)} - \mathbf{x}^{(t)} \right)$$ |
| inputs | $\mathbf{x}^{(t+1)}$, $\mathbf{x}^{(t)}$, $q^{(t)}$ |
| outputs | $\mathbf{s}^{(t+1)}$, $q^{(t+1)}$ |
| computational_note | Nesterov acceleration with restart-free momentum sequence |

#### Step 4.9: Convergence Check
| Field | Value |
|-------|-------|
| step_name | check_convergence |
| formula | $$\text{continue} = (t + 1 < T)$$ |
| inputs | $t$ (current iteration), $T$ (max iterations) |
| outputs | Boolean continue flag |
| computational_note | Fixed iteration count; no early stopping implemented |

**END LOOP**

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $t \geq T$ where $T = $ `n_iter` |
| parameters | $T$: maximum number of outer iterations (configured via `reconstruction_config['n_iter']`) |

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| RO