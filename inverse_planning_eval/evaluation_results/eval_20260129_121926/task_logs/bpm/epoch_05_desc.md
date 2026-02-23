## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: FISTA-BPM Optical Tomography Inversion
- **algorithm_family**: Accelerated Proximal Gradient Descent with Adjoint-State Method
- **summary**: Reconstructs 3D refractive index perturbation from holographic measurements using beam propagation forward model, adjoint gradient computation, and FISTA acceleration with nested TV-FISTA regularization.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\Delta n} \quad \frac{1}{N_{\text{batch}}} \sum_{b=1}^{N_{\text{batch}}} \left| \mathcal{F}_{\text{BPM}}(\Delta n; u_{\text{in}}^{(b)}) - u_{\text{out}}^{(b)} \right|_1 + \tau \|\nabla \Delta n\|_{2,1} + \lambda_s \|\Delta n\|_1 + \mathcal{I}_{[v_{\min}, v_{\max}]}(\Delta n|_{\Omega})$$

### data_fidelity_term
$$\mathcal{L}_{\text{data}}(\Delta n) = \frac{1}{N_{\text{batch}}} \sum_{b=1}^{N_{\text{batch}}} \text{mean}\left( \left| \mathcal{F}_{\text{BPM}}(\Delta n; u_{\text{in}}^{(b)}) - u_{\text{out}}^{(b)} \right| \right)$$

where $\mathcal{F}_{\text{BPM}}$ is the beam propagation forward operator computing output field from input field through refractive index volume.

### regularization_term
**Total Variation (isotropic, 3D):**
$$R_{\text{TV}}(\Delta n) = \tau \sum_{i,j,k} \sqrt{(\nabla_z \Delta n)_{i,j,k}^2 + (\nabla_y \Delta n)_{i,j,k}^2 + (\nabla_x \Delta n)_{i,j,k}^2}$$

**Sparsity (L1):**
$$R_{\text{sparse}}(\Delta n) = \lambda_s \|\Delta n\|_1 \quad \text{(when sparse\_param} \neq \text{None)}$$

### constraints
$$\mathcal{I}_{[v_{\min}, v_{\max}]}(\Delta n|_{\Omega}) = \begin{cases} 0 & \text{if } v_{\min} \leq \Delta n_{i,j,k} \leq v_{\max}, \; \forall (i,j,k) \in \Omega \\ +\infty & \text{otherwise} \end{cases}$$

where $\Omega = [s_0:e_0, s_1:e_1, s_2:e_2]$ is the region of interest.

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\Delta n$ | $(N_z, N_y, N_x)$ | float32 | $\mathbf{0}$ | Refractive index perturbation volume |
| $x^{(k)}$ | $(N_z, N_y, N_x)$ | float32 | $\mathbf{0}$ | Current iterate of FISTA outer loop |
| $s^{(k)}$ | $(N_z, N_y, N_x)$ | float32 | $\mathbf{0}$ | Momentum-extrapolated point for outer FISTA |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $g^{(t)}$ | $(N_z, N_y, N_x, 3)$ | float32 | $\nabla z$ | Dual variable for TV inner FISTA |
| $d^{(t)}$ | $(N_z, N_y, N_x, 3)$ | float32 | $\nabla z$ | Momentum-extrapolated dual for TV inner FISTA |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $q_1^{\text{outer}}$ | scalar | float64 | $1.0$ | FISTA momentum parameter (outer loop) |
| $q_2^{\text{outer}}$ | scalar | float64 | computed | Updated momentum parameter (outer loop) |
| $q_1^{\text{inner}}$ | scalar | float64 | $1.0$ | FISTA momentum parameter (TV inner loop) |
| $q_2^{\text{inner}}$ | scalar | float64 | computed | Updated momentum parameter (TV inner loop) |
| $u^{(m)}$ | $(B, N_y, N_x)$ | complex64 | $u_{\text{in}}$ | Propagating field at slice $m$ |
| $S$ | $(B, N_z, N_y, N_x)$ | complex64 | allocated | Stored forward fields for adjoint |
| $r$ | $(B, N_y, N_x)$ | complex64 | computed | Residual/adjoint field |
| $\nabla_{\Delta n} \mathcal{L}$ | $(N_z, N_y, N_x)$ | float32 | computed | Gradient of loss w.r.t. $\Delta n$ |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda_0$ | $0.532$ μm | Wavelength in vacuum |
| $p_{\text{cam}}$ | $4.5$ μm | Camera pixel size |
| $M$ | $100$ | Magnification |
| $n_m$ | $1.33$ | Refractive index of medium |
| $k_0$ | $2\pi / \lambda_0$ | Wavenumber in vacuum |
| $k_m$ | $2\pi n_m / \lambda_0$ | Wavenumber in medium |
| $\alpha$ | $0.0005$ | Step size (outer gradient descent) |
| $\tau$ | $1.75 \times 10^{-6}$ | TV regularization weight |
| $T_{\text{TV}}$ | $50$ | Number of TV inner FISTA iterations |
| $\gamma$ | $1/(12\tau)$ | Dual step size for TV FISTA |
| $v_{\min}$ | $0$ | Lower bound for value range |
| $v_{\max}$ | $100$ | Upper bound for value range |
| $\lambda_s$ | None | Sparsity parameter (disabled) |
| $N_{\text{iter}}$ | $1$ | Number of outer iterations |
| $B$ | $2$ | Batch size |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $u_{\text{in}}^{\text{raw}}$ | $(N_{\text{angles}}, N_y^{\text{crop}}, N_x^{\text{crop}})$ | complex64 | Raw input fields from file |
| $u_{\text{out}}^{\text{raw}}$ | $(N_{\text{angles}}, N_y^{\text{crop}}, N_x^{\text{crop}})$ | complex64 | Raw output fields from file |
| $\mathbf{k}_{\text{scan}}$ | $(N_{\text{angles}}, 2)$ | float32 | Scan wave vectors |

---

## 3. PREPROCESSING PIPELINE

**Step 3.1: Compute Physical Parameters**

- **step_order**: 1
- **step_name**: Compute pixel size and wavenumbers
- **formula**: 
$$p = \frac{p_{\text{cam}}}{M}, \quad k_0 = \frac{2\pi}{\lambda_0}, \quad k_m = \frac{2\pi n_m}{\lambda_0}$$
- **inputs**: $p_{\text{cam}}$ (scalar), $M$ (scalar), $\lambda_0$ (scalar), $n_m$ (scalar)
- **outputs**: $p$ (scalar), $k_0$ (scalar), $k_m$ (scalar)

**Step 3.2: Compute Spectral Pixel Size**

- **step_order**: 2
- **step_name**: Compute spectral domain sampling
- **formula**: 
$$\Delta k = \frac{2\pi}{p \cdot N_{\text{crop}}}$$
- **inputs**: $p$ (scalar), $N_{\text{crop}} = 1004$ (scalar)
- **outputs**: $\Delta k$ (scalar)

**Step 3.3: Compute Resolution**

- **step_order**: 3
- **step_name**: Compute voxel resolution
- **formula**: 
$$\Delta r_d = \frac{p \cdot N_{\text{crop}}}{N_d}, \quad d \in \{z, y, x\}$$
- **inputs**: $p$ (scalar), $N_{\text{crop}}$ (scalar), $(N_z, N_y, N_x)$ domain size
- **outputs**: $(\Delta z, \Delta y, \Delta x)$ resolution tuple

**Step 3.4: Compute Oblique Factor**

- **step_order**: 4
- **step_name**: Compute illumination angle correction
- **formula**: 
$$\cos\theta_b = \cos\left(\arcsin\left(\sqrt{\left(\frac{k_{\text{scan},b,0} \cdot \Delta k}{k_m}\right)^2 + \left(\frac{k_{\text{scan},b,1} \cdot \Delta k}{k_m}\right)^2}\right)\right)$$
- **inputs**: $\mathbf{k}_{\text{scan}}$ $(N_{\text{angles}}, 2)$, $\Delta k$ (scalar), $k_m$ (scalar)
- **outputs**: $\cos\boldsymbol{\theta}$ $(N_{\text{angles}}, 1)$

**Step 3.5: Generate Angular Spectrum Kernel**

- **step_order**: 5
- **step_name**: Compute propagation kernel
- **formula**: 
$$K_x[i] = \left(\text{linspace}(-N_x/2+1, N_x/2, N_x)[i] - 1\right) \cdot \Delta k$$
$$K_y[j] = \left(\text{linspace}(-N_y/2+1, N_y/2, N_y)[j] - 1\right) \cdot \Delta k$$
$$K_z[j,i] = \begin{cases} \sqrt{k_m^2 - K_x[i]^2 - K_y[j]^2} & \text{if } k_m^2 \geq K_x[i]^2 + K_y[j]^2 \\ 0 & \text{otherwise} \end{cases}$$
$$P[j,i] = \text{fftshift}\left(\exp(i \cdot K_z[j,i] \cdot \Delta z)\right)$$
- **inputs**: $(N_z, N_y, N_x)$, $\Delta k$ (scalar), $\Delta z$ (scalar), $k_m$ (scalar)
- **outputs**: $P$ $(N_y, N_x)$ complex64

**Step 3.6: Propagate Input Field to Inlet Plane**

- **step_order**: 6
- **step_name**: Backpropagate input field
- **formula**: 
$$u_{\text{inlet}} = \mathcal{F}^{-1}\left\{\mathcal{F}\{u_{\text{in}}^{\text{raw}}\} \cdot P^{*(N_z - N_{\text{bg}})}\right\}$$
where $P^{*n}$ denotes element-wise $n$-th power of conjugate.
- **inputs**: $u_{\text{in}}^{\text{raw}}$ $(N_{\text{angles}}, N_y, N_x)$, $P$ $(N_y, N_x)$, $N_z$ (scalar), $N_{\text{bg}}=93$ (scalar)
- **outputs**: $u_{\text{inlet}}$ $(N_{\text{angles}}, N_y, N_x)$ complex64

**Step 3.7: Propagate Output Field to Outlet Plane**

- **step_order**: 7
- **step_name**: Forward propagate output field
- **formula**: 
$$u_{\text{outlet}} = \mathcal{F}^{-1}\left\{\mathcal{F}\{u_{\text{out}}^{\text{raw}}\} \cdot P^{N_{\text{bg}}}\right\}$$
- **inputs**: $u_{\text{out}}^{\text{raw}}$ $(N_{\text{angles}}, N_y, N_x)$, $P$ $(N_y, N_x)$, $N_{\text{bg}}=93$ (scalar)
- **outputs**: $u_{\text{outlet}}$ $(N_{\text{angles}}, N_y, N_x)$ complex64

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

**Step 1: Initialize Primal Variable**

- **step_order**: 1
- **step_name**: Zero initialization of refractive index
- **formula**: $\Delta n^{(0)} = \mathbf{0}_{N_z \times N_y \times N_x}$
- **inputs**: $(N_z, N_y, N_x)$ = $(256, 502, 502)$
- **outputs**: $\Delta n^{(0)}$ $(256, 502, 502)$
- **computational_note**: Allocated on GPU as float32

**Step 2: Initialize Momentum Point**

- **step_order**: 2
- **step_name**: Initialize extrapolated point
- **formula**: $s^{(0)} = \Delta n^{(0)}$
- **inputs**: $\Delta n^{(0)}$ $(N_z, N_y, N_x)$
- **outputs**: $s^{(0)}$ $(N_z, N_y, N_x)$

**Step 3: Initialize FISTA Parameters**

- **step_order**: 3
- **step_name**: Initialize momentum scalar
- **formula**: $q_1^{\text{outer}} = 1.0$, $x^{(0)} = \Delta n^{(0)}$
- **inputs**: None
- **outputs**: $q_1^{\text{outer}}$ (scalar), $x^{(0)}$ $(N_z, N_y, N_x)$

---

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (iteration $k = 0, \ldots, N_{\text{iter}}-1$)

**Step 4: Initialize Gradient Accumulator**

- **step_order**: 4
- **step_name**: Zero gradient buffer
- **formula**: $\nabla_{\Delta n} \mathcal{L} = \mathbf{0}_{N_z \times N_y \times N_x}$
- **inputs**: $(N_z, N_y, N_x)$
- **outputs**: $\nabla_{\Delta n} \mathcal{L}$ $(N_z, N_y, N_x)$

**Step 5: Compute Oblique Phase Factor**

- **step_order**: 5
- **step_name**: Per-angle phase accumulation rate
- **formula**: 
$$\phi_b = \frac{k_0 \cdot \Delta z}{\cos\theta_b}$$
- **inputs**: $k_0$ (scalar), $\Delta z$ (scalar), $\cos\boldsymbol{\theta}$ $(N_{\text{angles}}, 1)$
- **outputs**: $\boldsymbol{\phi}$ $(N_{\text{angles}}, 1)$

**BEGIN_LOOP** (batch $i = 0, \ldots, \lceil N_{\text{angles}}/B \rceil - 1$)

**Step 6: Extract Batch**

- **step_order**: 6
- **step_name**: Slice batch from data
- **formula**: 
$$b_{\text{start}} = i \cdot B, \quad b_{\text{end}} = \min((i+1) \cdot B, N_{\text{angles}})$$
$$u_{\text{in}}^{(\text{batch})} = u_{\text{inlet}}[b_{\text{start}}:b_{\text{end}}]$$
$$u_{\text{out}}^{(\text{batch})} = u_{\text{outlet}}[b_{\text{start}}:b_{\text{end}}]$$
$$\boldsymbol{\phi}^{(\text{batch})} = \boldsymbol{\phi}[b_{\text{start}}:b_{\text{end}}]$$
- **inputs**: $u_{\text{inlet}}$, $u_{\text{outlet}}$, $\boldsymbol{\phi}$, $i$, $B$
- **outputs**: $u_{\text{in}}^{(\text{batch})}$, $u_{\text{out}}^{(\text{batch})}$, $\boldsymbol{\phi}^{(\text{batch})}$ (shapes depend on actual batch size)

**Step 7: Allocate Field Storage**

- **step_order**: 7
- **step_name**: Allocate forward field buffer
- **formula**: $S = \mathbf{0}_{B_{\text{actual}} \times N_z \times N_y \times N_x}$ (complex64)
- **inputs**: $B_{\text{actual}}$, $(N_z, N_y, N_x)$
- **outputs**: $S$ $(B_{\text{actual}}, N_z, N_y, N_x)$

**Step 8: Initialize Forward Field**

- **step_order**: 8
- **step_name**: Copy input to propagating field
- **formula**: $u = u_{\text{in}}^{(\text{batch})}$
- **inputs**: $u_{\text{in}}^{(\text{batch})}$ $(B_{\text{actual}}, N_y, N_x)$
- **outputs**: $u$ $(B_{\text{actual}}, N_y, N_x)$

**BEGIN_LOOP** [FORWARD] (slice $m = 0, \ldots, N_z - 1$)

**Step 9: Angular Spectrum Propagation**

- **step_order**: 9
- **step_name**: Propagate field one slice
- **formula**: 
$$u \leftarrow \mathcal{F}^{-1}\left\{\mathcal{F}\{u\} \odot P\right\}$$
where $P$ is broadcast to batch dimension.
- **inputs**: $u$ $(B_{\text{actual}}, N_y, N_x)$, $P$ $(N_y, N_x)$
- **outputs**: $u$ $(B_{\text{actual}}, N_y, N_x)$
- **computational_note**: 2D FFT along last two dimensions

**Step 10: Store Propagated Field**

- **step_order**: 10
- **step_name**: Cache field before phase modulation
- **formula**: $S[:, m, :, :] = u$
- **inputs**: $u$ $(B_{\text{actual}}, N_y, N_x)$, $m$
- **outputs**: $S$ (updated in-place)

**Step 11: Phase Modulation**

- **step_order**: 11
- **step_name**: Apply refractive index phase shift
- **formula**: 
$$u \leftarrow u \odot \exp\left(i \cdot \boldsymbol{\phi}^{(\text{batch})} \cdot s^{(k)}[m, :, :]\right)$$
where $\boldsymbol{\phi}^{(\text{batch})}$ is $(B_{\text{actual}}, 1, 1)$ broadcast.
- **inputs**: $u$ $(B_{\text{actual}}, N_y, N_x)$, $\boldsymbol{\phi}^{(\text{batch})}$ $(B_{\text{actual}}, 1)$, $s^{(k)}[m]$ $(N_y, N_x)$
- **outputs**: $u$ $(B_{\text{actual}}, N_y, N_x)$

**END_LOOP** (slice $m$)

**Step 12: Compute Residual**

- **step_order**: 12
- **step_name**: Field prediction error
- **formula**: $r = u - u_{\text{out}}^{(\text{batch})}$
- **inputs**: $u$ $(B_{\text{actual}}, N_y, N_x)$, $u_{\text{out}}^{(\text{batch})}$ $(B_{\text{actual}}, N_y, N_x)$
- **outputs**: $r$ $(B_{\text{actual}}, N_y, N_x)$ complex64

**BEGIN_LOOP** [ADJOINT] (slice $m = N_z - 1, \ldots, 0$)

**Step 13: Undo Phase Modulation**

- **step_order**: 13
- **step_name**: Reverse phase shift on adjoint field
- **formula**: 
$$r \leftarrow r \odot \exp\left(-i \cdot \boldsymbol{\phi}^{(\text{batch})} \cdot s^{(k)}[m, :, :]\right)$$
- **inputs**: $r$ $(B_{\text{actual}}, N_y, N_x)$, $\boldsymbol{\phi}^{(\text{batch})}$, $s^{(k)}[m]$
- **outputs**: $r$ $(B_{\text{actual}}, N_y, N_x)$

**Step 14: Compute Batch Gradient Contribution**

- **step_order**: 14
- **step_name**: Gradient from adjoint-state formula
- **formula**: 
$$G_m^{(\text{batch})} = -i \cdot \boldsymbol{\phi}^{(\text{batch})} \odot S[:, m, :, :]^* \odot r$$
$$\nabla_{\Delta n} \mathcal{L}[m] \leftarrow \nabla_{\Delta n} \mathcal{L}[m] + \sum_{b=0}^{B_{\text{actual}}-1} \text{Re}\left(G_m^{(\text{batch})}[b]\right)$$
- **inputs**: $\boldsymbol{\phi}^{(\text{batch})}$, $S[:, m]$, $r$
- **outputs**: $\nabla_{\Delta n} \mathcal{L}[m]$ (accumulated)
- **computational_note**: Sum over batch dimension, take real part

**Step 15: Backpropagate Adjoint Field**

- **step_order**: 15
- **step_name**: Adjoint angular spectrum propagation
- **formula**: 
$$r \leftarrow \mathcal{F}^{-1}\left\{\mathcal{F}\{r\} \odot P^*\right\}$$
- **inputs**: $r$ $(B_{\text{actual}}, N_y, N_x)$, $P^*$ $(N_y, N_x)$
- **outputs**: $r$ $(B_{\text{actual}}, N_y, N_x)$

**END_LOOP** (slice $m$)

**END_LOOP** (batch $i$)

**Step 16: Normalize Gradient**

- **step_order**: 16
- **step_name**: Average over all angles
- **formula**: 
$$\nabla_{\Delta n} \mathcal{L} \leftarrow \frac{1}{N_{\text{angles}}} \nabla_{\Delta n} \mathcal{L}$$
- **inputs**: $\nabla_{\Delta n} \mathcal{L}$ $(N_z, N_y, N_x)$, $N_{\text{angles}}$
- **outputs**: $\nabla_{\Delta n} \mathcal{L}$ $(N_z, N_y, N_x)$

**Step 17: Gradient Descent Step**

- **step_order**: 17
- **step_name**: Compute pre-proximal point
- **formula**: $z = s^{(k)} - \alpha \cdot \nabla_{\Delta n} \mathcal{L}$
- **inputs**: $s^{(k)}$ $(N_z, N_y, N_x)$, $\alpha = 0.0005$, $\nabla_{\Delta n} \mathcal{L}$ $(N_z, N_y, N_x)$
- **outputs**: $z$ $(N_z, N_y, N_x)$

**Step 18: Apply Combined Proximal Operator (TV-FISTA + Clamp + Sparse)**

*This step expands to the TV-FISTA inner loop:*

**Step 18.1: Initialize TV Dual Variable**

- **step_order**: 18.1
- **step_name**: Compute initial gradient
- **formula**: 
$$g^{(0)}[:-1, :, :, 0] = z[1:, :, :] - z[:-1, :, :]$$
$$g^{(0)}[:, :-1, :, 1] = z[:, 1:, :] - z[:, :-1, :]$$
$$g^{(0)}[:, :, :-1, 2] = z[:, :, 1:] - z[:, :, :-1]$$
$$d^{(0)} = g^{(0)}, \quad q_1^{\text{inner}} = 1$$
- **inputs**: $z$ $(N_z, N_y, N_x)$
- **outputs**: $g^{(0)}$, $d^{(0)}$ $(N_z, N_y, N_x, 3)$, $q_1^{\text{inner}}$

**BEGIN_LOOP** (TV iteration $t = 0, \ldots, T_{\text{TV}} - 1$)

**Step 18.2: Compute Divergence**

- **step_order**: 18.2
- **step_name**: Negative divergence of dual
- **formula**: 
$$(\text{div } d)_{i,j,k} = -\Big[(d_{i,j,k,0} - d_{i-1,j,k,0}) + (d_{i,j,k,1} - d_{i,j-1,k,1}) + (d_{i,j,k,2} - d_{i,j,k-1,2})\Big]$$
with boundary conditions: $d_{-1,\cdot,\cdot,0} = 0$, $d_{N_z-1,\cdot,\cdot,0}$ reflected, etc.
- **inputs**: $d^{(t)}$ $(N_z, N_y, N_x, 3)$
- **outputs**: $\text{div } d^{(t)}$ $(N_z, N_y, N_x)$

**Step 18.3: Primal Update with Clamp**

- **step_order**: 18.3
- **step_name**: Value-range projection
- **formula**: 
$$\tilde{x} = z - \tau \cdot \text{div } d^{(t)}$$
$$\tilde{x}[s_0:e_0, s_1:e_1, s_2:e_2] = \text{clamp}(\tilde{x}[s_0:e_0, s_1:e_1, s_2:e_2], v_{\min}, v_{\max})$$
- **inputs**: $z$, $\tau$, $\text{div } d^{(t)}$, ROI bounds, $v_{\min}=0$, $v_{\max}=100$
- **outputs**: $\tilde{x}$ $(N_z, N_y, N_x)$

**Step 18.4: Compute Gradient of Updated Primal**

- **step_order**: 18.4
- **step_name**: Forward differences
- **formula**: $\nabla \tilde{x}$ computed as in Step 18.1
- **inputs**: $\tilde{x}$ $(N_z, N_y, N_x)$
- **outputs**: $\nabla \tilde{x}$ $(N_z, N_y, N_x, 3)$

**Step 18.5: Dual Gradient Step**

- **step_order**: 18.5
- **step_name**: Dual ascent
- **formula**: $\hat{g} = d^{(t)} + \gamma \cdot \nabla \tilde{x}$, where $\gamma = 1/(12\tau)$
- **inputs**: $d^{(t)}$, $\gamma$, $\nabla \tilde{x}$
- **outputs**: $\hat{g}$ $(N_z, N_y, N_x, 3)$

**Step 18.6: Project Dual onto Unit Ball**

- **step_order**: 18.6
- **step_name**: Pointwise $\ell_2$ projection
- **formula**: 
$$\|\hat{g}\|_{i,j,k} = \sqrt{\hat{g}_{i,j,k,0}^2 + \hat{g}_{i,j,k,1}^2 + \hat{g}_{i,j,k,2}^2}$$
$$g^{(t+1)}_{i,j,k} = \frac{\hat{g}_{i,j,k}}{\max(1, \|\hat{g}\|_{i,j,k})}$$
- **inputs**: $\hat{g}$ $(N_z, N_y, N_x, 3)$
- **outputs**: $g^{(t+1)}$ $(N_z, N_y, N_x, 3)$

**Step 18.7: Update Primal with Projected Dual**

- **step_order**: 18.7
- **step_name**: Final primal for this TV iteration
- **formula**: 
$$x^{\text{TV}} = \text{clamp}_{\Omega}\left(z - \tau \cdot \text{div } g^