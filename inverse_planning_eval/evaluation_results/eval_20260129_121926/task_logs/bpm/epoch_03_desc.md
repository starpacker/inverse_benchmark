## 0. ALGORITHM IDENTIFICATION
- **algorithm_name**: BPM-FISTA Inversion for Optical Diffraction Tomography
- **algorithm_family**: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) with Adjoint-Based Gradient Descent
- **summary**: Reconstructs 3D refractive index perturbation from holographic measurements using Beam Propagation Method (BPM) forward model with FISTA-accelerated proximal gradient descent, incorporating total variation, value range constraints, and optional sparsity regularization.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\Delta n} \quad \frac{1}{N_{\text{illum}}} \sum_{i=1}^{N_{\text{illum}}} \left| \mathcal{F}_{\text{BPM}}(\Delta n; u_{\text{in}}^{(i)}) - u_{\text{out}}^{(i)} \right|_{\text{mean}} + \tau \|\nabla \Delta n\|_{2,1} + \lambda_s \|\Delta n\|_1 + \mathcal{I}_{[v_{\min}, v_{\max}]}(\Delta n|_{\text{ROI}})$$

### data_fidelity_term
$$\mathcal{L}_{\text{data}}(\Delta n) = \frac{1}{N_{\text{illum}}} \sum_{i=1}^{N_{\text{illum}}} \text{mean}\left( \left| \mathcal{F}_{\text{BPM}}(\Delta n; u_{\text{in}}^{(i)}) - u_{\text{out}}^{(i)} \right| \right)$$

where $\mathcal{F}_{\text{BPM}}$ is the split-step Beam Propagation Method forward operator and $|\cdot|$ denotes element-wise complex magnitude.

### regularization_term
$$\mathcal{R}(\Delta n) = \tau \|\nabla \Delta n\|_{2,1} + \lambda_s \|\Delta n\|_1$$

where:
- $\|\nabla \Delta n\|_{2,1} = \sum_{i,j,k} \sqrt{(\partial_z \Delta n)^2 + (\partial_y \Delta n)^2 + (\partial_x \Delta n)^2}$ (isotropic TV)
- $\|\Delta n\|_1 = \sum_{i,j,k} |\Delta n_{i,j,k}|$ (L1 sparsity)

### constraints
$$\Delta n_{i,j,k} \in [v_{\min}, v_{\max}] \quad \forall (i,j,k) \in \text{ROI}$$

where ROI is defined by indices $[s_0:e_0, s_1:e_1, s_2:e_2]$.

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\Delta n$ (`delta_ri`) | primal | $(N_z, N_y, N_x)$ | float32 | Zero tensor | Refractive index perturbation |
| $x^{(k)}$ (`x_1`, `x_2`) | primal | $(N_z, N_y, N_x)$ | float32 | Zero tensor | FISTA primal iterate |
| $s^{(k)}$ (`s`) | primal | $(N_z, N_y, N_x)$ | float32 | Zero tensor | FISTA extrapolated point |

### Dual Variables (TV sub-problem)
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $g$ | dual | $(N_z, N_y, N_x, 3)$ | float32 | $\nabla z$ | TV dual variable (gradient field) |
| $d$ | dual | $(N_z, N_y, N_x, 3)$ | float32 | $g^{(0)}$ | TV FISTA extrapolated dual |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $q^{(k)}$ (`q_1`, `q_2`) | auxiliary | scalar | float64 | $q^{(0)} = 1$ | FISTA momentum parameter (outer loop) |
| $\tilde{q}^{(k)}$ | auxiliary | scalar | float64 | $\tilde{q}^{(0)} = 1$ | FISTA momentum parameter (TV inner loop) |
| $u$ | auxiliary | $(N_{\text{batch}}, N_y, N_x)$ | complex64 | $u_{\text{in}}$ | Propagating field in forward pass |
| $r$ | auxiliary | $(N_{\text{batch}}, N_y, N_x)$ | complex64 | $u_{\text{pred}} - u_{\text{out}}$ | Residual / adjoint field |
| $S$ (`s_fields`) | auxiliary | $(N_{\text{batch}}, N_z, N_y, N_x)$ | complex64 | Computed during forward | Stored intermediate fields for adjoint |
| $\nabla_{\Delta n}\mathcal{L}$ (`grad`) | auxiliary | $(N_z, N_y, N_x)$ | float32 | Zero tensor | Gradient of loss w.r.t. $\Delta n$ |

### Constants
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\lambda$ (`wavelength`) | constants | scalar | float64 | 0.532 µm | Optical wavelength |
| $p_{\text{cam}}$ (`camera_pixel_size`) | constants | scalar | float64 | 4.5 µm | Camera pixel size |
| $M$ (`magnification`) | constants | scalar | float64 | 100 | Optical magnification |
| $n_m$ (`n_medium`) | constants | scalar | float64 | 1.33 | Background refractive index |
| $k_m$ | constants | scalar | float64 | $2\pi n_m / \lambda$ | Medium wavenumber |
| $k_0$ | constants | scalar | float64 | $2\pi / \lambda$ | Vacuum wavenumber |
| $p$ (`pixelsize`) | constants | scalar | float64 | $p_{\text{cam}} / M = 0.045$ µm | Object-space pixel size |
| $\Delta k$ (`spec_pixel_size`) | constants | scalar | float64 | $2\pi / (p \cdot N_{\text{crop}})$ | Spectral pixel size |
| $\Delta z$ (`resolution[0]`) | constants | scalar | float64 | $p \cdot N_{\text{crop}} / N_z$ | Axial step size |
| $\alpha$ (`step_size`) | constants | scalar | float64 | 0.0005 | Gradient descent step size |
| $\tau$ (`tv_param[0]`) | constants | scalar | float64 | $1.75 \times 10^{-6}$ | TV regularization weight |
| $N_{\text{TV}}$ (`tv_param[1]`) | constants | scalar | int | 50 | TV inner iterations |
| $\gamma$ | constants | scalar | float64 | $1/(12\tau)$ | TV dual step size |
| $v_{\min}$ (`value_range_param[0]`) | constants | scalar | float64 | 0 | Minimum allowed value |
| $v_{\max}$ (`value_range_param[1]`) | constants | scalar | float64 | 100 | Maximum allowed value |
| $\lambda_s$ (`sparse_param[0]`) | constants | scalar | float64 | None (disabled) | Sparsity parameter |
| $N_{\text{iter}}$ (`n_iter`) | constants | scalar | int | 1 | Outer iterations |
| $N_{\text{batch}}$ (`batch_size`) | constants | scalar | int | 2 | Batch size for gradient |
| $N_z$ (`region_z`) | constants | scalar | int | 256 | Axial domain size |
| $(N_y, N_x)$ | constants | tuple | int | (502, 502) | Lateral domain size |
| $z_{\text{bg}}$ (`bg_z`) | constants | scalar | int | 93 | Background plane index |
| $N_{\text{crop}}$ (`crop_size[0]`) | constants | scalar | int | 1004 | Crop dimension |

### Observations
| Symbol | Category | Shape | Dtype | Description |
|--------|----------|-------|-------|-------------|
| $u_{\text{in}}^{\text{raw}}$ | observations | $(N_{\text{illum}}, N_y, N_x)$ | complex64 | Raw input fields from file |
| $u_{\text{out}}^{\text{raw}}$ | observations | $(N_{\text{illum}}, N_y, N_x)$ | complex64 | Raw output fields from file |
| $\mathbf{k}_{\text{scan}}$ | observations | $(N_{\text{illum}}, 2)$ | float64 | Illumination k-vectors |

---

## 3. PREPROCESSING PIPELINE

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Compute physical constants | $k_m = \frac{2\pi n_m}{\lambda}, \quad k_0 = \frac{2\pi}{\lambda}, \quad p = \frac{p_{\text{cam}}}{M}$ | $\lambda, n_m, p_{\text{cam}}, M$ | $k_m, k_0, p$ (scalars) |
| 2 | Compute spectral pixel size | $\Delta k = \frac{2\pi}{p \cdot N_{\text{crop}}}$ | $p$, $N_{\text{crop}}=1004$ | $\Delta k$ (scalar) |
| 3 | Compute resolution | $\Delta r_i = \frac{p \cdot N_{\text{crop}}}{N_i}$ for $i \in \{z, y, x\}$ | $p$, $N_{\text{crop}}$, domain_size | $(\Delta z, \Delta y, \Delta x)$ (3-vector) |
| 4 | Load raw data | Load from files | File paths | $u_{\text{in}}^{\text{raw}} \in \mathbb{C}^{N_{\text{illum}} \times N_y \times N_x}$, $u_{\text{out}}^{\text{raw}} \in \mathbb{C}^{N_{\text{illum}} \times N_y \times N_x}$, $\mathbf{k}_{\text{scan}} \in \mathbb{R}^{N_{\text{illum}} \times 2}$ |
| 5 | Compute oblique factor | $\cos\theta_i = \cos\left(\arcsin\sqrt{(\tilde{k}_{x,i})^2 + (\tilde{k}_{y,i})^2}\right)$ where $\tilde{k}_i = \frac{\mathbf{k}_{\text{scan},i} \cdot \Delta k}{k_m}$ | $\mathbf{k}_{\text{scan}}$, $\Delta k$, $k_m$ | $\cos\theta \in \mathbb{R}^{N_{\text{illum}} \times 1}$ |
| 6 | Construct spectral grid | $K_x[j] = (j - N_y/2) \cdot \Delta k$ for $j \in [1, N_y]$, similarly $K_y$ | $N_y, N_x, \Delta k$ | $K_x, K_y \in \mathbb{R}^{N_y \times N_x}$ (meshgrid) |
| 7 | Compute propagation kernel | $K_z = \sqrt{k_m^2 - K_x^2 - K_y^2}$ (set to 0 where argument $<0$); $P = \exp(i K_z \Delta z)$; apply fftshift | $K_x, K_y, k_m, \Delta z$ | $P \in \mathbb{C}^{N_y \times N_x}$ |
| 8 | Propagate inlet field | $u_{\text{inlet}} = \mathcal{F}^{-1}\left[\mathcal{F}[u_{\text{in}}^{\text{raw}}] \cdot P^{*(N_z - z_{\text{bg}})}\right]$ | $u_{\text{in}}^{\text{raw}}$, $P$, $N_z=256$, $z_{\text{bg}}=93$ | $u_{\text{inlet}} \in \mathbb{C}^{N_{\text{illum}} \times N_y \times N_x}$ |
| 9 | Propagate outlet field | $u_{\text{outlet}} = \mathcal{F}^{-1}\left[\mathcal{F}[u_{\text{out}}^{\text{raw}}] \cdot P^{z_{\text{bg}}}\right]$ | $u_{\text{out}}^{\text{raw}}$, $P$, $z_{\text{bg}}$ | $u_{\text{outlet}} \in \mathbb{C}^{N_{\text{illum}} \times N_y \times N_x}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|------|---------|--------|---------|
| 1 | Initialize primal variable | $\Delta n^{(0)} = \mathbf{0}$ | domain_size $(N_z, N_y, N_x)$ | $\Delta n^{(0)} \in \mathbb{R}^{256 \times 502 \times 502}$ |
| 2 | Initialize extrapolated point | $s^{(0)} = \Delta n^{(0)}$ | $\Delta n^{(0)}$ | $s^{(0)}$ |
| 3 | Initialize FISTA parameter | $q^{(0)} = 1$ | — | $q^{(0)}$ (scalar) |
| 4 | Initialize previous iterate | $x^{(0)} = \Delta n^{(0)}$ | $\Delta n^{(0)}$ | $x^{(0)}$ |
| 5 | Compute ROI bounds | $(s_0, e_0, s_1, e_1, s_2, e_2) = (0, 256, 60, 442, 60, 442)$ | ROI config `[None, None, 60, -60, 60, -60]` | ROI tuple |
| 6 | Precompute oblique factor tensor | $\omega_i = \frac{k_0 \Delta z}{\cos\theta_i}$ | $k_0, \Delta z, \cos\theta$ | $\omega \in \mathbb{R}^{N_{\text{illum}} \times 1}$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (for $k = 0, 1, \ldots, N_{\text{iter}}-1$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 1 | Initialize gradient accumulator | $G = \mathbf{0}$ | — | $G \in \mathbb{R}^{N_z \times N_y \times N_x}$ | |
| 2 | Initialize loss accumulator | $L_{\text{total}} = 0$ | — | $L_{\text{total}}$ (scalar) | |

**BEGIN_LOOP** (batch loop: for $b = 0, \ldots, \lceil N_{\text{illum}} / N_{\text{batch}} \rceil - 1$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 3 | Extract batch | $\tilde{u}_{\text{in}} = u_{\text{inlet}}[b \cdot N_{\text{batch}} : (b+1) \cdot N_{\text{batch}}]$, similarly $\tilde{u}_{\text{out}}$, $\tilde{\omega}$ | $u_{\text{inlet}}, u_{\text{outlet}}, \omega$, batch indices | $\tilde{u}_{\text{in}}, \tilde{u}_{\text{out}} \in \mathbb{C}^{N_b \times N_y \times N_x}$, $\tilde{\omega} \in \mathbb{R}^{N_b \times 1}$ | $N_b = $ actual batch size |
| 4 | Initialize field storage | $S = \mathbf{0}$ | — | $S \in \mathbb{C}^{N_b \times N_z \times N_y \times N_x}$ | Stores fields for adjoint |
| 5 | Initialize propagating field | $u = \tilde{u}_{\text{in}}$ | $\tilde{u}_{\text{in}}$ | $u \in \mathbb{C}^{N_b \times N_y \times N_x}$ | |

**BEGIN_LOOP** (forward propagation: for $m = 0, \ldots, N_z - 1$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 6a | Angular spectrum propagation | $u \leftarrow \mathcal{F}^{-1}[\mathcal{F}[u] \cdot P]$ | $u$, $P$ | $u$ | FFT-based propagation |
| 6b | Store field | $S[:, m, :, :] = u$ | $u$ | $S$ | For adjoint computation |
| 6c | Phase modulation | $u \leftarrow u \cdot \exp(i \tilde{\omega} \cdot s^{(k)}[m, :, :])$ | $u$, $\tilde{\omega}$, $s^{(k)}[m]$ | $u$ | Thin-slice approximation |

**END_LOOP** (forward propagation)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 7 | Compute residual | $r = u - \tilde{u}_{\text{out}}$ | $u$, $\tilde{u}_{\text{out}}$ | $r \in \mathbb{C}^{N_b \times N_y \times N_x}$ | |
| 8 | Accumulate batch loss | $L_{\text{total}} \leftarrow L_{\text{total}} + \text{mean}(|r|)$ | $r$, $L_{\text{total}}$ | $L_{\text{total}}$ | Mean absolute error |

**BEGIN_LOOP** (backward propagation: for $m = N_z - 1, \ldots, 0$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 9a | Undo phase modulation | $r \leftarrow r \cdot \exp(-i \tilde{\omega} \cdot s^{(k)}[m, :, :])$ | $r$, $\tilde{\omega}$, $s^{(k)}[m]$ | $r$ | Adjoint of forward phase |
| 9b | Compute batch gradient | $\tilde{G}_m = -i \tilde{\omega} \cdot S[:, m]^* \cdot r$ | $\tilde{\omega}$, $S[:, m]$, $r$ | $\tilde{G}_m \in \mathbb{C}^{N_b \times N_y \times N_x}$ | |
| 9c | Accumulate gradient | $G[m] \leftarrow G[m] + \text{Re}(\sum_{\text{batch}} \tilde{G}_m)$ | $\tilde{G}_m$, $G[m]$ | $G[m]$ | Real part, sum over batch |
| 9d | Backpropagate adjoint | $r \leftarrow \mathcal{F}^{-1}[\mathcal{F}[r] \cdot P^*]$ | $r$, $P$ | $r$ | Conjugate kernel |

**END_LOOP** (backward propagation)

**END_LOOP** (batch loop)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| 10 | Normalize gradient | $G \leftarrow G / N_{\text{illum}}$ | $G$, $N_{\text{illum}}$ | $G$ | Average over illuminations |
| 11 | Compute average loss | $L = L_{\text{total}} / N_{\text{batches}}$ | $L_{\text{total}}$, $N_{\text{batches}}$ | $L$ (scalar) | |
| 12 | Gradient descent step | $z = s^{(k)} - \alpha \cdot G$ | $s^{(k)}$, $G$, $\alpha = 0.0005$ | $z \in \mathbb{R}^{N_z \times N_y \times N_x}$ | |
| 13 | Apply proximal operator | $x^{(k+1)} = \text{prox}_{\text{TV+box+sparse}}(z)$ | $z$ | $x^{(k+1)}$ | See sub-block 4.2.1 |
| 14 | Update FISTA parameter | $q^{(k+1)} = \frac{1 + \sqrt{1 + 4(q^{(k)})^2}}{2}$ | $q^{(k)}$ | $q^{(k+1)}$ | |
| 15 | FISTA extrapolation | $s^{(k+1)} = x^{(k+1)} + \frac{q^{(k)} - 1}{q^{(k+1)}} (x^{(k+1)} - x^{(k)})$ | $x^{(k+1)}$, $x^{(k)}$, $q^{(k)}$, $q^{(k+1)}$ | $s^{(k+1)}$ | |
| 16 | Update previous iterate | $x^{(k)} \leftarrow x^{(k+1)}$, $q^{(k)} \leftarrow q^{(k+1)}$ | — | — | |

**END_LOOP** (outer iteration)

---

#### 4.2.1 TV-FISTA PROXIMAL SUB-BLOCK

Input: $z$ (gradient-stepped variable)
Output: $x_{\text{prox}}$ (regularized variable)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| T1 | Initialize dual | $g^{(0)} = \nabla z$ | $z$ | $g^{(0)} \in \mathbb{R}^{N_z \times N_y \times N_x \times 3}$ | Discrete gradient |
| T2 | Initialize extrapolated dual | $d^{(0)} = g^{(0)}$ | $g^{(0)}$ | $d^{(0)}$ | |
| T3 | Initialize momentum | $\tilde{q}^{(0)} = 1$ | — | $\tilde{q}^{(0)}$ | |

**BEGIN_LOOP** (TV iteration: for $t = 0, \ldots, N_{\text{TV}}-1$ where $N_{\text{TV}} = 50$)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| T4 | Compute divergence | $v = z - \tau \cdot \text{div}(d^{(t)})$ | $z$, $d^{(t)}$, $\tau = 1.75 \times 10^{-6}$ | $v$ | $\text{div} = -\nabla^T$ (adjoint) |
| T5 | Apply box constraint | $v_{\text{ROI}} \leftarrow \text{clamp}(v_{\text{ROI}}, v_{\min}, v_{\max})$ | $v$, $v_{\min}=0$, $v_{\max}=100$, ROI | $v$ | Only within ROI |
| T6 | Compute gradient | $w = \nabla v$ | $v$ | $w \in \mathbb{R}^{N_z \times N_y \times N_x \times 3}$ | |
| T7 | Dual gradient step | $\hat{g} = d^{(t)} + \gamma \cdot w$ | $d^{(t)}$, $w$, $\gamma = 1/(12\tau)$ | $\hat{g}$ | |
| T8 | Project onto unit ball | $g^{(t+1)}_{i,j,k} = \frac{\hat{g}_{i,j,k}}{\max(1, \|\hat{g}_{i,j,k}\|_2)}$ | $\hat{g}$ | $g^{(t+1)}$ | Component-wise L2 projection |
| T9 | Compute primal update | $\tilde{x} = z - \tau \cdot \text{div}(g^{(t+1)})$ | $z$, $g^{(t+1)}$, $\tau$ | $\tilde{x}$ | |
| T10 | Apply box constraint | $x^{(t+1)} = \text{clamp}(\tilde{x}_{\text{ROI}}, v_{\min}, v_{\max})$ | $\tilde{x}$, ROI, bounds | $x^{(t+1)}$ | |
| T11 | Update FISTA momentum | $\tilde{q}^{(t+1)} = \frac{1 + \sqrt{1 + 4(\tilde{q}^{(t)})^2}}{2}$ | $\tilde{q}^{(t)}$ | $\tilde{q}^{(t+1)}$ | |
| T12 | Dual extrapolation | $d^{(t+1)} = g^{(t+1)} + \frac{\tilde{q}^{(t)} - 1}{\tilde{q}^{(t+1)}} (g^{(t+1)} - g^{(t)})$ | $g^{(t+1)}$, $g^{(t)}$, $\tilde{q}^{(t)}$, $\tilde{q}^{(t+1)}$ | $d^{(t+1)}$ | |
| T13 | Update previous dual | $g^{(t)} \leftarrow g^{(t+1)}$, $\tilde{q}^{(t)} \leftarrow \tilde{q}^{(t+1)}$ | — | — | |

**END_LOOP** (TV iteration)

| Step | Name | Formula | Inputs | Outputs | Computational Note |
|------|------|---------|--------|---------|-------------------|
| T14 | Apply sparsity (soft threshold) | $x_{\text{prox}} = \text{sign}(x^{(N_{\text{TV}})}) \cdot \max(|x^{(N_{\text{TV}})}| - \lambda_s \alpha, 0)$ | $x^{(N_{\text{TV}})}$, $\lambda_s$, $\alpha$ | $x_{\text{prox}}$ | Skipped if $\lambda_s = $ None |

**Discrete Gradient Operator** $\nabla$:
$$(\nabla x)_{i,j,k,0} = \begin{cases} x_{i+1,j,k} - x_{i,j,k} & i < N_z - 1 \\ 0 & i = N_z - 1 \end{cases}$$
$$(\nabla x)_{i,j,k,1} = \begin{cases} x_{i,j+1,k} - x_{i,j,k} & j < N_y - 1 \\ 0 & j = N_y - 1 \end{cases}$$
$$(\nabla x)_{i,j,k,2} = \begin{cases} x_{i,j,k+1} - x_{i,j,k} & k < N_x - 1 \\ 0 & k = N_x - 1 \end{cases}$$

**Discrete Divergence Operator** $\text{div} = -\nabla^T$:
$$(\text{div}\, g)_{i,j,k} = -\left[(g_{i,j,k,0} - g_{i-1,j,k,0})_{\text{bd}} + (g_{i,j,k,1} - g_{i,j-1,k,1})_{\text{bd}} + (g_{i,j,k,2} - g_{i,j,k-1,2})_{\text{bd}}\right]$$

with boundary conditions:
- At $i=0$: use $g_{0,j,k,0}$; at $i=N_z-1$: use $-g_{N_z-2,j,k,0}$; else: $g_{i,j,k,0} - g_{i-1,j,k,0}$
- Similar for other dimensions.

---

### 4.3 FINALIZATION

| Step | Name | Formula | Inputs | Outputs |
|------|---