# Algorithmic Specification Document: Multislice Ptychographic Reconstruction via Automatic Differentiation

## 1. Mathematical Formulation

### 1.1 Complete Objective Function

The algorithm solves the following inverse problem:

$$\min_{\mathbf{x}} \mathcal{L}(\mathbf{x}) = \mathcal{L}_{\text{data}}(\mathbf{y}, \mathbf{A}(\mathbf{x})) + \mathcal{L}_{\text{reg}}(\mathbf{x})$$

subject to constraint updates applied periodically during optimization.

**Latent Variable Vector:**
$$\mathbf{x} = \{A, \phi, P, \boldsymbol{\theta}, \Delta z, \delta\mathbf{r}\}$$

where the **data fidelity term** is:
$$\mathcal{L}_{\text{data}}(\mathbf{y}, \hat{\mathbf{y}}) = \sum_{i \in \mathcal{B}} \ell(y_i, \hat{y}_i)$$

with $\hat{y}_i = \mathbf{A}_i(\mathbf{x})$ being the forward model prediction for scan position $i$, and $\ell(\cdot, \cdot)$ defined by `CombinedLoss` (configurable loss types).

---

### 1.2 Forward Model $\mathbf{A}(\mathbf{x})$: Multislice Ptychography

For each scan position index $i$:

#### Step 1: Complex Object Construction
$$O^{(s,m)}[n_y, n_x] = A^{(s,m)}[n_y, n_x] \cdot \exp\bigl(j\,\phi^{(s,m)}[n_y, n_x]\bigr)$$

where $s \in \{1, \ldots, N_s\}$ (slice index), $m \in \{1, \ldots, N_m\}$ (object mode index).

#### Step 2: Object Patch Extraction (ROI Selection)
$$O_i^{(s,m)}[p_y, p_x] = O^{(s,m)}\bigl[p_y + c_y^{(i)},\; p_x + c_x^{(i)}\bigr]$$

where $(c_y^{(i)}, c_x^{(i)}) = \texttt{crop\_pos}[i]$ are integer pixel coordinates, and $p_y \in \{0, \ldots, N_{py}-1\}$, $p_x \in \{0, \ldots, N_{px}-1\}$.

#### Step 3: Optional Object Pre-blur
If $\sigma_{\text{obj}} > 0$:
$$\tilde{O}_i = G_{\sigma_{\text{obj}}} * O_i$$
where $G_\sigma$ is a Gaussian kernel with standard deviation $\sigma$.

#### Step 4: Probe Retrieval with Sub-pixel Shifts
$$P_i^{(p,m)}[k_y, k_x] = \mathcal{F}^{-1}\Bigl[\mathcal{F}[P^{(p,m)}] \cdot \exp\bigl(-j2\pi(k_y \delta r_y^{(i)} + k_x \delta r_x^{(i)})\bigr)\Bigr]$$

where $(k_y, k_x) = \texttt{fftfreq}(N_{py}) \times \texttt{fftfreq}(N_{px})$, and $\delta\mathbf{r}^{(i)} = (\delta r_y^{(i)}, \delta r_x^{(i)})$ is the sub-pixel shift for position $i$.

If no shifts: $P_i = P$ (broadcast).

#### Step 5: Fresnel Propagator Construction
$$H_i[K_y, K_x] = \exp\Bigl(j\,\Delta z \cdot K_z[K_y, K_x]\Bigr) \cdot \exp\Bigl(j\,\Delta z\,(K_y \tan\theta_y^{(i)} + K_x \tan\theta_x^{(i)})\Bigr)$$

where:
- $K_z = \sqrt{k^2 - K_x^2 - K_y^2}$, with $k = 2\pi/\lambda$
- $K_y[n] = \texttt{ifftshift}\bigl(2\pi \cdot (n - N_{py}/2 + 0.5)/(N_{py} \cdot dx)\bigr)$
- $K_x[n] = \texttt{ifftshift}\bigl(2\pi \cdot (n - N_{px}/2 + 0.5)/(N_{px} \cdot dx)\bigr)$
- $\theta_y^{(i)}, \theta_x^{(i)}$ are tilts in radians (stored in mrad, divided by 1000)

#### Step 6: Multislice Wave Propagation
Initialize: $\psi_i^{(0)} = P_i$

For $s = 1, \ldots, N_s$:
$$\psi_i^{(s)} = \mathcal{F}^{-1}\Bigl[H_i \cdot \mathcal{F}\bigl[\psi_i^{(s-1)} \odot \tilde{O}_i^{(s)}\bigr]\Bigr]$$

where $\odot$ denotes element-wise multiplication.

#### Step 7: Exit Wave to Diffraction Pattern
$$\hat{I}_i = \sum_{m=1}^{N_m} w_m \cdot \Bigl|\mathcal{F}\bigl[\psi_i^{(N_s, m)}\bigr]\Bigr|^2$$

where $w_m = \texttt{omode\_occu}[m]$ are mode occupancy weights.

#### Step 8: Optional Detector Blur
If $\sigma_{\text{det}} > 0$:
$$\hat{y}_i = G_{\sigma_{\text{det}}} * \hat{I}_i$$
else $\hat{y}_i = \hat{I}_i$.

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | Dtype | Initialization | Description |
|----------|--------|-------|-------|----------------|-------------|
| `opt_obja` | $A$ | $(N_m, N_s, N_{oy}, N_{ox})$ | `float32` | $\|O_{\text{init}}\|$ | Object amplitude |
| `opt_objp` | $\phi$ | $(N_m, N_s, N_{oy}, N_{ox})$ | `float32` | $\angle O_{\text{init}}$ | Object phase |
| `opt_probe` | $P$ | $(N_p, N_m, N_{py}, N_{px}, 2)$ | `float32` | `view_as_real(P_init)` | Probe (real view of complex) |
| `opt_obj_tilts` | $\boldsymbol{\theta}$ | $(N_{\text{tilt}}, 2)$ | `float32` | From config | Object tilts (mrad) |
| `opt_slice_thickness` | $\Delta z$ | scalar or $(N_s,)$ | `float32` | From config | Slice thickness |
| `opt_probe_pos_shifts` | $\delta\mathbf{r}$ | $(N, 2)$ | `float32` | Zeros or from config | Sub-pixel probe shifts |

### 2.2 Auxiliary/Buffer Variables (Non-optimized)

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `H` | $H_0$ | $(N_{py}, N_{px})$ | `complex64` | Base Fresnel propagator |
| `omode_occu` | $\mathbf{w}$ | $(N_m,)$ | `float32` | Object mode weights |
| `crop_pos` | $\mathbf{c}$ | $(N, 2)$ | `int32` | Integer scan positions |
| `propagator_grid` | $(K_y, K_x)$ | $(2, N_{py}, N_{px})$ | `float32` | Frequency grids |
| `shift_probes_grid` | $(k_y, k_x)$ | $(2, N_{py}, N_{px})$ | `float32` | Shift frequency grids |
| `Kz` | $K_z$ | $(N_{py}, N_{px})$ | `float32` | $z$-component of wavevector |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `lr_params[key]` | $\eta_{\text{key}}$ | Learning rate for each variable |
| `start_iter[key]` | $t_{\text{start}}^{\text{key}}$ | Iteration to begin optimizing variable |
| `end_iter[key]` | $t_{\text{end}}^{\text{key}}$ | Iteration to stop optimizing variable |
| `detector_blur_std` | $\sigma_{\text{det}}$ | Detector PSF blur std |
| `obj_preblur_std` | $\sigma_{\text{obj}}$ | Object regularization blur std |
| `dx` | $dx$ | Real-space pixel size |
| `dk` | $dk$ | Reciprocal-space pixel size |
| `lambd` | $\lambda$ | Wavelength |
| `optimizer_params` | — | Optimizer type and settings (e.g., Adam) |

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `measurements` | $\mathbf{y}$ | $(N, N_{ky}, N_{kx})$ | `float32` | Measured diffraction patterns |
| `meas_padded` | $\mathbf{y}_{\text{pad}}$ | $(N_{ky}', N_{kx}')$ | `float32` | Padding template (optional) |
| `meas_scale_factors` | $(s_y, s_x)$ | tuple | `float` | Resampling factors (optional) |

---

## 3. Preprocessing Pipeline

| Step | Operation | Input | Output |
|------|-----------|-------|--------|
| 1 | Load parameters | `params_path` (YAML) | `params` dict |
| 2 | Set device | `gpuid` | `device` (cuda/cpu) |
| 3 | Initialize variables | `params['init_params']` | `init.init_variables` dict containing $O_{\text{init}}, P_{\text{init}}, \mathbf{c}, H_0$, etc. |
| 4 | Decompose initial object | $O_{\text{init}} \in \mathbb{C}$ | $A = \|O_{\text{init}}\|$, $\phi = \angle O_{\text{init}}$ |
| 5 | Convert probe to real view | $P_{\text{init}} \in \mathbb{C}^{N_p \times N_m \times N_{py} \times N_{px}}$ | `view_as_real(P)` → $(N_p, N_m, N_{py}, N_{px}, 2)$ |
| 6 | Build frequency grids | $N_{py}, N_{px}, dx$ | $K_y, K_x, k_y, k_x$ grids |
| 7 | Compute probe intensity sum | $P$ | $\sum\|P\|^2$ (for normalization reference) |
| 8 | Initialize loss function | `loss_params` | `loss_fn: CombinedLoss` |
| 9 | Initialize constraint function | `constraint_params` | `constraint_fn: CombinedConstraint` |
| 10 | Create optimizer | `optimizer_params`, variable groups with LRs | `optimizer` (e.g., Adam) |
| 11 | Prepare batch indices | $N$, batch size | `batches`: list of index arrays |

---

## 4. Algorithm Steps (Main Iterative Procedure)

### Initialization

1. **Initialize model**: Create `PtychoAD` with all variables on device
2. **Initialize optimizer**: `optimizer = create_optimizer(params, model.optimizable_params)`
3. **Compute fixed propagator** (if tilts/thickness not optimized):
   $$H_{\text{fixed}} = H_0 \cdot \exp\bigl(j\,\Delta z\,(K_y \tan\theta_y + K_x \tan\theta_x)\bigr)$$

### Per-Iteration Update (for iteration $t = 1, \ldots, T$)

| Step | Name | Formula / Operation | Inputs | Outputs |
|------|------|---------------------|--------|---------|
| 4.1 | **Sample batch** | $\mathcal{B}_t \subseteq \{1, \ldots, N\}$ | Batch schedule | Index set $\mathcal{B}_t$ |
| 4.2 | **Update requires_grad** | For each var $v$: `v.requires_grad = (lr[v] ≠ 0) and (start[v] ≤ t ≤ end[v])` | $t$, schedule | Grad flags |
| 4.3 | **Extract object patches** | $O_i = O[:, :, c_y^{(i)}:c_y^{(i)}+N_{py}, c_x^{(i)}:c_x^{(i)}+N_{px}]$ for $i \in \mathcal{B}_t$ | $A, \phi, \mathbf{c}$ | $O_{\mathcal{B}_t}$ |
| 4.4 | **Optional object blur** | $\tilde{O}_i = G_{\sigma_{\text{obj}}} * O_i$ | $O_i, \sigma_{\text{obj}}$ | $\tilde{O}_i$ |
| 4.5 | **Get probes** | $P_i = \text{imshift}(P, \delta\mathbf{r}^{(i)})$ or broadcast | $P, \delta\mathbf{r}$ | $P_{\mathcal{B}_t}$ |
| 4.6 | **Compute propagators** | $H_i = \exp(j\Delta z K_z)\exp(j\Delta z(K_y\tan\theta_y^{(i)} + K_x\tan\theta_x^{(i)}))$ | $\Delta z, \boldsymbol{\theta}, K_y, K_x, K_z$ | $H_{\mathcal{B}_t}$ |
| 4.7 | **Multislice forward** | $\hat{y}_i = \texttt{multislice\_forward}(\tilde{O}_i, P_i, H_i, \mathbf{w})$ | Patches, probes, propagators | $\hat{\mathbf{y}}_{\mathcal{B}_t}$ |
| 4.8 | **Optional detector blur** | $\hat{y}_i \leftarrow G_{\sigma_{\text{det}}} * \hat{y}_i$ | $\hat{y}_i, \sigma_{\text{det}}$ | Blurred $\hat{y}_i$ |
| 4.9 | **Get measurements** | $y_i$ for $i \in \mathcal{B}_t$, with optional padding/resampling | Indices | $\mathbf{y}_{\mathcal{B}_t}$ |
| 4.10 | **Compute loss** | $\mathcal{L}_t = \texttt{loss\_fn}(\hat{\mathbf{y}}_{\mathcal{B}_t}, \mathbf{y}_{\mathcal{B}_t}, O_{\mathcal{B}_t})$ | Predictions, measurements | Scalar loss |
| 4.11 | **Backward pass** | $\nabla_{\mathbf{x}} \mathcal{L}_t$ via autograd | $\mathcal{L}_t$ | Gradients |
| 4.12 | **Optimizer step** | $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \cdot \texttt{optimizer\_update}(\nabla \mathcal{L}_t)$ | Gradients, LRs | Updated $\mathbf{x}$ |
| 4.13 | **Apply constraints** | $\mathbf{x}^{(t+1)} \leftarrow \texttt{constraint\_fn}(\mathbf{x}^{(t+1)}, t)$ | Variables, iteration | Constrained vars |
| 4.14 | **Log metrics** | Store $\mathcal{L}_t$, timing, $\Delta z$, tilts | Iteration data | Logs |

### Multislice Forward Model Detail (Step 4.7)

```
function multislice_forward(O, P, H, w):
    # O: (batch, modes, slices, Ny, Nx, 2) — [amp, phase] stacked
    # P: (batch, probe_modes, obj_modes, Ny, Nx) complex
    # H: (batch, Ny, Nx) complex propagator
    # w: (modes,) occupancy weights
    
    O_complex = O[...,0] * exp(j * O[...,1])  # Reconstruct complex object
    ψ = P  # Initialize wave
    
    for s = 1 to N_slices:
        ψ = ψ ⊙ O_complex[:,:,s,:,:]  # Multiply by object slice
        ψ = ifft2(H * fft2(ψ))         # Propagate
    
    I = Σ_m w[m] * |fft2(ψ[:,:,m,:,:])|²  # Incoherent mode sum
    return I
```

---

## 5. Stopping Criterion

The algorithm terminates when **any** of the following conditions are met:

1. **Iteration limit**: $t \geq T_{\max}$ (specified in `recon_params`)
2. **Convergence** (if implemented in `recon_loop`):
   $$|\mathcal{L}_{t} - \mathcal{L}_{t-1}| < \epsilon_{\text{tol}}$$
3. **Manual interruption** or distributed process termination

The code explicitly checks:
```python
compilation_iters = {1} ∪ {start_iter[v] : v ∈ variables} ∪ {end_iter[v] : v ∈ variables}
```
to trigger recompilation when gradient requirements change.

---

## 6. Evaluation Metrics

### 6.1 Loss Function (Data Fidelity)
Defined by `CombinedLoss` — common choices include:

**Amplitude Loss (L2)**:
$$\mathcal{L}_{\text{amp}} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \bigl\|\sqrt{\hat{y}_i} - \sqrt{y_i}\bigr\|_2^2$$

**Intensity Loss (L2)**:
$$\mathcal{L}_{\text{int}} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \|\hat{y}_i - y_i\|_2^2$$

**Poisson Loss**:
$$\mathcal{L}_{\text{poisson}} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \sum_{k} \bigl[\hat{y}_i^{(k)} - y_i^{(k)} \log(\hat{y}_i^{(k)} + \epsilon)\bigr]$$

### 6.2 Reconstruction Quality Metrics (Post-hoc)

**Object Amplitude Statistics**:
$$\bar{A} = \frac{1}{N_{\text{pix}}} \sum_{n} A[n], \quad \sigma_A = \sqrt{\frac{1}{N_{\text{pix}}} \sum_n (A[n] - \bar{A})^2}$$

**Object Phase Statistics**:
$$\bar{\phi} = \frac{1}{N_{\text{pix}}} \sum_{n} \phi[n], \quad \sigma_\phi = \sqrt{\frac{1}{N_{\text{pix}}} \sum_n (\phi[n] - \bar{\phi})^2}$$

### 6.3 Overdetermination Ratio
$$R_{\text{OD}} = \frac{N \cdot N_{ky} \cdot N_{kx}}{\sum_{v : \text{grad}=\text{True}} \text{numel}(v)}$$

This measures the ratio of measurements to free parameters (reported during initialization).

---

## 7. Summary of Key Equations

| Component | Equation |
|-----------|----------|
| Complex object | $O = A \cdot e^{j\phi}$ |
| Wave after slice $s$ | $\psi^{(s)} = \mathcal{F}^{-1}[H \cdot \mathcal{F}[\psi^{(s-1)} \odot O^{(s)}]]$ |
| Fresnel propagator | $H = \exp(j\Delta z \sqrt{k^2 - K_x^2 - K_y^2}) \cdot \exp(j\Delta z(K_y\tan\theta_y + K_x\tan\theta_x))$ |
| Diffraction pattern | $\hat{y} = \sum_m w_m |\mathcal{F}[\psi^{(N_s,m)}]|^2$ |
| Gradient update | $\mathbf{x} \leftarrow \mathbf{x} - \eta \cdot \text{Adam}(\nabla_\mathbf{x} \mathcal{L})$ |
| Sub-pixel shift | $P_{\text{shifted}} = \mathcal{F}^{-1}[\mathcal{F}[P] \cdot e^{-j2\pi(k_y \delta r_y + k_x \delta r_x)}]$ |