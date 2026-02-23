# Algorithmic Specification Document: Beam Propagation Method (BPM) Inversion for Optical Diffraction Tomography

---

## 1. Mathematical Formulation

### 1.1 Objective Function

The algorithm solves the following regularized inverse problem:

$$
\min_{\delta n} \quad \underbrace{\frac{1}{N_{\text{illum}}} \sum_{i=1}^{N_{\text{illum}}} \left\| \mathcal{F}(\delta n; u_{\text{in}}^{(i)}) - u_{\text{out}}^{(i)} \right\|_1}_{\text{Data Fidelity}} + \underbrace{\tau \cdot \text{TV}_{3D}(\delta n)}_{\text{TV Regularization}} + \underbrace{\lambda_s \|\delta n\|_1}_{\text{Sparsity}} + \underbrace{\mathcal{I}_{[n_{\min}, n_{\max}]}(\delta n)}_{\text{Box Constraint}}
$$

Where:
- $\delta n \in \mathbb{R}^{N_z \times N_y \times N_x}$ is the refractive index perturbation
- $\mathcal{F}$ is the BPM forward operator
- $\text{TV}_{3D}(\delta n) = \sum_{i,j,k} \|\nabla \delta n_{i,j,k}\|_2$ is the isotropic 3D total variation
- $\mathcal{I}_{[n_{\min}, n_{\max}]}$ is the indicator function for box constraints

### 1.2 Forward Model (BPM with Angular Spectrum Method)

The forward operator $\mathcal{F}$ propagates an input field through $N_z$ layers:

$$
u^{(m+1)} = \underbrace{\mathcal{P}_{\Delta z}\{u^{(m)}\}}_{\text{Free-space propagation}} \cdot \underbrace{\exp\left(i \cdot \frac{k_0 \Delta z}{\cos\theta_i} \cdot \delta n_m\right)}_{\text{Phase modulation}}
$$

**Free-space propagation kernel (Angular Spectrum Method):**

$$
\mathcal{P}_{\Delta z}\{u\} = \mathcal{F}^{-1}\left\{ \mathcal{F}\{u\} \cdot H(k_x, k_y) \right\}
$$

$$
H(k_x, k_y) = \exp\left(i \cdot k_z \cdot \Delta z\right), \quad k_z = \sqrt{k_m^2 - k_x^2 - k_y^2}
$$

Where:
- $k_m = \frac{2\pi}{\lambda} n_{\text{medium}}$ is the wavenumber in medium
- $k_0 = \frac{2\pi}{\lambda}$ is the vacuum wavenumber
- Evanescent waves are suppressed: $k_z = 0$ when $k_x^2 + k_y^2 > k_m^2$

**Oblique illumination correction factor:**

$$
\cos\theta_i = \cos\left(\arcsin\sqrt{\tilde{k}_{x,i}^2 + \tilde{k}_{y,i}^2}\right)
$$

where $\tilde{k}_{x,i}, \tilde{k}_{y,i}$ are normalized transverse wavevectors of the $i$-th illumination.

### 1.3 Data Fidelity Term (Per-Batch)

$$
\mathcal{L}(\delta n) = \frac{1}{N_{\text{illum}}} \sum_{i=1}^{N_{\text{illum}}} \left| \mathcal{F}(\delta n; u_{\text{in}}^{(i)}) - u_{\text{out}}^{(i)} \right|
$$

The code uses Mean Absolute Error (L1 loss) averaged over spatial dimensions and batch.

### 1.4 TV Regularization (Isotropic 3D)

$$
\text{TV}_{3D}(\delta n) = \sum_{i,j,k} \sqrt{(\delta n_{i+1,j,k} - \delta n_{i,j,k})^2 + (\delta n_{i,j+1,k} - \delta n_{i,j,k})^2 + (\delta n_{i,j,k+1} - \delta n_{i,j,k})^2}
$$

Solved via Chambolle-Pock primal-dual formulation with FISTA acceleration.

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Symbol | Code Variable | Shape | Dtype | Initialization | Description |
|--------|---------------|-------|-------|----------------|-------------|
| $\delta n$ | `delta_ri`, `init_delta_ri` | $(N_z, N_y, N_x)$ | `float32` | $\mathbf{0}$ | Refractive index perturbation |
| $x^{(k)}$ | `x_1`, `x_2` | $(N_z, N_y, N_x)$ | `float32` | $\mathbf{0}$ | FISTA primal iterate |
| $s^{(k)}$ | `s` | $(N_z, N_y, N_x)$ | `float32` | $\mathbf{0}$ | FISTA extrapolated point |

### 2.2 Auxiliary/Dual Variables

| Symbol | Code Variable | Shape | Dtype | Description |
|--------|---------------|-------|-------|-------------|
| $g$ | `g`, `g_1`, `g_2` | $(N_z, N_y, N_x, 3)$ | `float32` | Dual variable for TV (gradient field) |
| $d$ | `d` | $(N_z, N_y, N_x, 3)$ | `float32` | FISTA momentum for TV inner loop |
| $q^{(k)}$ | `q_1`, `q_2` | scalar | `float64` | FISTA momentum coefficient |
| $u^{(m)}$ | `u` | $(B, N_y, N_x)$ | `cfloat` | Propagating field at layer $m$ |
| $S^{(m)}$ | `s_fields` | $(B, N_z, N_y, N_x)$ | `cfloat` | Stored forward fields for adjoint |
| $r$ | `r` | $(B, N_y, N_x)$ | `cfloat` | Residual/adjoint field |

### 2.3 Hyperparameters/Constants

| Symbol | Code Variable | Value/Source | Description |
|--------|---------------|--------------|-------------|
| $\alpha$ | `step_size` | `0.0005` | Gradient descent step size |
| $\tau$ | `tv_param[0]` | `1.75e-6` | TV regularization weight |
| $K_{\text{TV}}$ | `tv_param[1]` | `50` | TV inner loop iterations |
| $\gamma$ | `gamma` | $\frac{1}{12\tau}$ | TV dual step size |
| $n_{\min}$ | `value_range_param[0]` | `0` | Lower bound for $\delta n$ |
| $n_{\max}$ | `value_range_param[1]` | `100` | Upper bound for $\delta n$ |
| $\lambda_s$ | `sparse_param[0]` | `None` | Sparsity (L1) weight |
| $N_{\text{iter}}$ | `n_iter` | `1` (config) | Outer loop iterations |
| $B$ | `batch_size` | `2` | Mini-batch size |
| $\lambda$ | `wavelength` | `0.532` μm | Optical wavelength |
| $n_m$ | `n_medium` | `1.33` | Medium refractive index |
| $M$ | `magnification` | `100` | Optical magnification |
| $p_{\text{cam}}$ | `camera_pixel_size` | `4.5` μm | Camera pixel size |

### 2.4 Observed Data/Inputs

| Symbol | Code Variable | Shape | Source | Description |
|--------|---------------|-------|--------|-------------|
| $u_{\text{in}}$ | `u_in` | $(N_{\text{illum}}, N_y', N_x')$ | `u_in.npy` | Measured input fields |
| $u_{\text{out}}$ | `u_out` | $(N_{\text{illum}}, N_y', N_x')$ | `u_out.npy` | Measured output fields |
| $\mathbf{k}_{\text{scan}}$ | `k_scan_samp` | $(N_{\text{illum}}, 2)$ | `k_samp.npy` | Illumination k-vectors |

### 2.5 Derived/Computed Constants

| Symbol | Code Variable | Formula | Description |
|--------|---------------|---------|-------------|
| $p$ | `pixelsize` | $p_{\text{cam}} / M$ | Object-space pixel size |
| $k_m$ | `km` | $\frac{2\pi}{\lambda} n_m$ | Wavenumber in medium |
| $k_0$ | `k0` | $\frac{2\pi}{\lambda}$ | Vacuum wavenumber |
| $\Delta k$ | `spec_pixel_size` | $\frac{2\pi}{p \cdot N_{\text{crop}}}$ | Spectral pixel size |
| $\Delta z$ | `resolution[0]` | $\frac{p \cdot N_{\text{crop}}}{N_z}$ | Axial voxel size |
| $H$ | `p_kernel` | See §1.2 | Propagation kernel |

---

## 3. Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Step | Operation | Input | Output | Formula |
|------|-----------|-------|--------|---------|
| **P1** | Load raw data | Files | $(u_{\text{in}}, u_{\text{out}}, \mathbf{k}_{\text{scan}})$ | `np.load()` |
| **P2** | Compute physical parameters | Config | $(k_0, k_m, p, \Delta k)$ | $k_m = \frac{2\pi n_m}{\lambda}, \; p = \frac{p_{\text{cam}}}{M}, \; \Delta k = \frac{2\pi}{p \cdot N_{\text{crop}}}$ |
| **P3** | Compute resolution | $(p, N_{\text{crop}}, N_{\text{domain}})$ | $\boldsymbol{\Delta} = (\Delta z, \Delta y, \Delta x)$ | $\boldsymbol{\Delta} = \frac{p \cdot N_{\text{crop}}}{\mathbf{N}_{\text{domain}}}$ |
| **P4** | Compute oblique factors | $\mathbf{k}_{\text{scan}}$ | $\cos\theta$ | $\cos\theta_i = \cos\left(\arcsin\sqrt{\tilde{k}_{x,i}^2 + \tilde{k}_{y,i}^2}\right), \; \tilde{k} = \frac{\mathbf{k}_{\text{scan}} \cdot \Delta k}{k_m}$ |
| **P5** | Build propagation kernel | $(N_{\text{domain}}, \Delta k, k_m, \Delta z)$ | $H(k_x, k_y)$ | See Algorithm A1 below |
| **P6** | Back-propagate input field | $(u_{\text{in}}, H, N_z^{\text{region}}, N_z^{\text{bg}})$ | $u_{\text{inlet}}$ | $u_{\text{inlet}} = \mathcal{F}^{-1}\left\{\mathcal{F}\{u_{\text{in}}\} \cdot H^{*(N_z^{\text{region}} - N_z^{\text{bg}})}\right\}$ |
| **P7** | Forward-propagate output field | $(u_{\text{out}}, H, N_z^{\text{bg}})$ | $u_{\text{outlet}}$ | $u_{\text{outlet}} = \mathcal{F}^{-1}\left\{\mathcal{F}\{u_{\text{out}}\} \cdot H^{N_z^{\text{bg}}}\right\}$ |

### Algorithm A1: Angular Spectrum Kernel Construction

```
INPUT: domain_size = (Nz, Ny, Nx), Δk, Δz, km
OUTPUT: H (propagation kernel), C_ol (obliquity correction)

1.  kx ← linspace(-Nx/2+1, Nx/2, Nx) - 1) × Δk
2.  ky ← linspace(-Ny/2+1, Ny/2, Ny) - 1) × Δk
3.  [Ky, Kx] ← meshgrid(ky, kx)
4.  K² ← Kx² + Ky²
5.  Kz ← sqrt(km² - K² + 0j)           // Complex square root
6.  Kz[K² > km²] ← 0                   // Suppress evanescent waves
7.  H ← exp(i · Kz · Δz)
8.  H ← fftshift(H)
9.  RETURN H
```

---

## 4. Algorithm Steps

### 4.1 Main Algorithm: FISTA-Accelerated Gradient Descent

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALGORITHM: BPM INVERSION WITH FISTA                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Initialization:**

| Variable | Formula | Value |
|----------|---------|-------|
| $x^{(0)}$ | - | $\mathbf{0} \in \mathbb{R}^{N_z \times N_y \times N_x}$ |
| $s^{(0)}$ | $s^{(0)} = x^{(0)}$ | $\mathbf{0}$ |
| $q^{(0)}$ | - | $1$ |

**Per-Iteration Updates ($k = 0, 1, \ldots, N_{\text{iter}}-1$):**

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| **S1** | Gradient Computation | $\nabla \mathcal{L} = \texttt{compute\_gradient}(s^{(k)}, u_{\text{inlet}}, u_{\text{outlet}})$ | $s^{(k)}, u_{\text{inlet}}, u_{\text{outlet}}$ | $\nabla \mathcal{L}, \ell^{(k)}$ |
| **S2** | Gradient Step | $z = s^{(k)} - \alpha \cdot \nabla \mathcal{L}$ | $s^{(k)}, \nabla \mathcal{L}, \alpha$ | $z$ |
| **S3** | Proximal/Regularization | $x^{(k+1)} = \texttt{prox}_{\text{reg}}(z)$ | $z$ | $x^{(k+1)}$ |
| **S4** | Momentum Update | $q^{(k+1)} = \frac{1 + \sqrt{1 + 4(q^{(k)})^2}}{2}$ | $q^{(k)}$ | $q^{(k+1)}$ |
| **S5** | Extrapolation | $s^{(k+1)} = x^{(k+1)} + \frac{q^{(k)} - 1}{q^{(k+1)}}(x^{(k+1)} - x^{(k)})$ | $x^{(k+1)}, x^{(k)}, q^{(k)}, q^{(k+1)}$ | $s^{(k+1)}$ |

---

### 4.2 Subroutine: Batched Gradient Computation (Adjoint Method)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ALGORITHM: ADJOINT GRADIENT COMPUTATION                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Input:** $\delta n$, $u_{\text{in}}^{(1:N)}$, $u_{\text{out}}^{(1:N)}$, $H$, $\cos\theta$, $\Delta z$, $k_0$

**Output:** $\nabla_{\delta n} \mathcal{L}$, $\ell$

```
INITIALIZE:
    grad ← zeros(Nz, Ny, Nx)
    total_loss ← 0
    Φ_i ← k₀·Δz / cos(θᵢ)  for each illumination i    // Oblique phase factor

FOR each batch b = 1, ..., ⌈N/B⌉:
    
    // === FORWARD PASS (store intermediate fields) ===
    FOR each sample i in batch:
        u ← u_in^(i)
        FOR m = 0, ..., Nz-1:
            u ← FFT⁻¹{FFT{u} · H}                     // Free-space propagation
            S^(i)_m ← u                               // Store field before modulation
            u ← u · exp(i·Φᵢ·δnₘ)                    // Phase modulation
        END FOR
        r^(i) ← u - u_out^(i)                        // Residual
        total_loss ← total_loss + mean(|r^(i)|)
    END FOR
    
    // === BACKWARD PASS (adjoint propagation) ===
    FOR each sample i in batch:
        r ← r^(i)
        FOR m = Nz-1, ..., 0:
            r ← r · exp(-i·Φᵢ·δnₘ)                   // Conjugate phase
            ∂L/∂δnₘ ← Re{-i·Φᵢ·conj(S^(i)_m)·r}      // Layer gradient
            grad_m ← grad_m + sum over batch(∂L/∂δnₘ)
            r ← FFT⁻¹{FFT{r} · H*}                   // Adjoint propagation
        END FOR
    END FOR
    
END FOR

grad ← grad / N
avg_loss ← total_loss / num_batches
RETURN grad, avg_loss
```

**Gradient Formula (per layer $m$, per illumination $i$):**

$$
\frac{\partial \mathcal{L}}{\partial \delta n_m} = \text{Re}\left\{ -i \cdot \frac{k_0 \Delta z}{\cos\theta_i} \cdot \overline{S_m^{(i)}} \cdot r_m^{(i)} \right\}
$$

where:
- $S_m^{(i)}$ is the stored forward field at layer $m$ (before phase modulation)
- $r_m^{(i)}$ is the adjoint field at layer $m$

---

### 4.3 Subroutine: Composite Proximal Operator (TV + Box + Sparsity)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 ALGORITHM: FISTA TV DENOISING                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Input:** $z$ (gradient-updated iterate), $\tau$ (TV weight), $K$ (inner iterations), $[n_{\min}, n_{\max}]$, $\lambda_s$ (sparsity), ROI

**Output:** $x$ (regularized iterate)

**Operator Definitions:**

$$
\nabla: \mathbb{R}^{N_z \times N_y \times N_x} \to \mathbb{R}^{N_z \times N_y \times N_x \times 3}
$$

$$
(\nabla x)_{i,j,k} = \begin{pmatrix} x_{i+1,j,k} - x_{i,j,k} \\ x_{i,j+1,k} - x_{i,j,k} \\ x_{i,j,k+1} - x_{i,j,k} \end{pmatrix}
$$

$$
\text{div} = -\nabla^* \quad \text{(negative adjoint)}
$$

**Projection onto unit ball:**

$$
\Pi(g)_{i,j,k} = \frac{g_{i,j,k}}{\max(1, \|g_{i,j,k}\|_2)}
$$

**Algorithm:**

```
INITIALIZE:
    g⁽⁰⁾ ← ∇z
    d⁽⁰⁾ ← g⁽⁰⁾
    t⁽⁰⁾ ← 1
    γ ← 1/(12τ)

FOR k = 0, ..., K-1:
    
    // Step T1: Primal update
    v ← z - τ·div(d⁽ᵏ⁾)
    
    // Step T2: Box constraint (on ROI only)
    v[ROI] ← clamp(v[ROI], n_min, n_max)
    
    // Step T3: Dual gradient step
    g⁽ᵏ⁺¹⁾ ← Π(d⁽ᵏ⁾ + γ·∇v)
    
    // Step T4: Primal from new dual
    x ← clamp(z - τ·div(g⁽ᵏ⁺¹⁾), n_min, n_max)  // on ROI
    
    // Step T5: FISTA momentum
    t⁽ᵏ⁺¹⁾ ← (1 + √(1 + 4(t⁽ᵏ⁾)²)) / 2
    d⁽ᵏ⁺¹⁾ ← g⁽ᵏ⁺¹⁾ + ((t⁽ᵏ⁾-1)/t⁽ᵏ⁺¹⁾)(g⁽ᵏ⁺¹⁾ - g⁽ᵏ⁾)

END FOR

// Step T6: Soft thresholding (sparsity)
IF λ_s is not None:
    x ← sign(x) · max(|x| - λ_s·α, 0)
END IF

RETURN x
```

---

## 5. Stopping Criterion

| Criterion | Condition | Code Location |
|-----------|-----------|---------------|
| **Primary** | Fixed iteration count | `for iteration in range(n_iter)` |
| | $k \geq N_{\text{iter}}$ | where $N_{\text{iter}}$ = `reconstruction_config['n_iter']` |
| **TV Inner Loop** | Fixed iteration count | `for i in range(step_num)` |
| | $j \geq K_{\text{TV}}$ | where $K_{\text{TV}}$ = `tv_param[1]` |

**No convergence-based early stopping is implemented.** The algorithm runs for exactly $N_{\text{iter}}$ outer iterations and $K_{\text{TV}}$ inner TV iterations.

---

## 6. Evaluation Metrics

### 6.1 Optimization Loss (Data Fidelity)

$$
\mathcal{L}^{(k)} = \frac{1}{N_{\text{batches}}} \sum_{b=1}^{N_{\text{batches}}} \left( \frac{1}{|B_b|} \sum_{i \in B_b} \text{mean}_{y,x}\left| \mathcal{F}(\delta n; u_{\text{in}}^{(i)}) - u_{\text{out}}^{(i)} \right| \right)
$$

**Code:** `batch_loss = r.abs().mean().item()`

### 6.2 Reconstruction Statistics (ROI-based)

Let $\Omega_{\text{ROI}} = [s_0:e_0, s_1:e_1, s_2:e_2]$ define the region of interest.

| Metric | Formula | Code Variable |
|--------|---------|---------------|
| **ROI Min** | $\min_{(i,j,k) \in \Omega} \delta n_{i,j,k}$ | `vmin` |
| **ROI Max** | $\max_{(i,j,k) \in \Omega} \delta n_{i,j,k}$ | `vmax` |
| **ROI Mean** | $\bar{\delta n} = \frac{1}{|\Omega|}\sum_{(i,j,k) \in \Omega} \delta n_{i,j,k}$ | `vmean` |
| **ROI Std** | $\sigma = \sqrt{\frac{1}{|\Omega|}\sum_{(i,j,k) \in \Omega} (\delta n_{i,j,k} - \bar{\delta n})^2}$ | `vstd` |
| **Initial Loss** | $\mathcal{L}^{(0)}$ | `initial_loss` |
| **Final Loss** | $\mathcal{L}^{(N_{\text{iter}}-1)}$ | `final_loss` |

---

## 7. Complete Algorithm Summary (Block Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BPM INVERSION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐│
│  │ Raw Data    │───▶│ Preprocessing    │───▶│ u_inlet, u_outlet, H, cosθ ││
│  │ u_in, u_out │    │ (Steps P1-P7)    │    │                             ││
│  └─────────────┘    └──────────────────┘    └──────────────┬──────────────┘│
│                                                            │               │
│                                                            ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    FISTA OUTER LOOP (k iterations)                   │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │ S1: Batched Gradient via Adjoint Method                       │  │  │
│  │  │     ∇L = Σ_batch Re{-i·Φ·conj(S_m)·r_m}                       │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │ S2: Gradient Descent Step                                     │  │  │
│  │  │     z = s⁽ᵏ⁾ - α·∇L                                           │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │ S3: Proximal Operator (FISTA TV Inner Loop)                   │  │  │
│  │  │     x⁽ᵏ⁺¹⁾ = prox_TV(prox_box(prox_sparse(z)))                │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │ S4-S5: FISTA Momentum Update                                  │  │  │
│  │  │     q⁽ᵏ⁺¹⁾ = (1+√(1+4q²))/2                                   │  │  │
│  │  │     s⁽ᵏ⁺¹⁾ = x⁽ᵏ⁺¹⁾ + (q⁽ᵏ⁾-1)/q⁽ᵏ⁺¹⁾·(x⁽ᵏ⁺¹⁾-x⁽