# Algorithmic Specification Document: FPM-INR (Fourier Ptychographic Microscopy via Implicit Neural Representations)

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\Theta} \mathcal{L}(\Theta) = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \ell_{\text{SmoothL1}}\left( \sqrt{I_k^{\text{obs}}}, \left| \mathcal{F}^{-1}\left[ P_k \odot \mathcal{S}_k\left[ \mathcal{F}[f_\Theta(\mathbf{0})] \right] \right] \right| \right)$$

where $\Theta = \{\mathbf{D}_{xy}^{(r)}, \mathbf{D}_{xy}^{(i)}, \mathbf{D}_z^{(r)}, \mathbf{D}_z^{(i)}, \mathbf{W}^{(r)}, \mathbf{W}^{(i)}\}$ are all learnable parameters.

**data_fidelity_term**:
$$\ell_{\text{SmoothL1}}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| < 1 \\ |y - \hat{y}| - 0.5 & \text{otherwise} \end{cases}$$

Applied element-wise and averaged over all pixels and LED indices.

**regularization_term**: None (implicit regularization via neural network parameterization)

**constraints**: None (unconstrained optimization)

---

## 2. VARIABLES

### Primal Variables (Learnable Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{D}_{xy}^{(r)}$ | primal | $(M_x, M_y, F)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4})$ | XY feature grid for real component |
| $\mathbf{D}_{xy}^{(i)}$ | primal | $(M_x, M_y, F)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4})$ | XY feature grid for imaginary component |
| $\mathbf{D}_z^{(r)}$ | primal | $(M_z, F)$ | float32 | $\mathcal{N}(0, 1)$ | Z feature vector for real component |
| $\mathbf{D}_z^{(i)}$ | primal | $(M_z, F)$ | float32 | $\mathcal{N}(0, 1)$ | Z feature vector for imaginary component |
| $\mathbf{W}_\ell^{(r)}, \mathbf{b}_\ell^{(r)}$ | primal | varies | float32 | PyTorch default | Renderer MLP weights/biases (real), $\ell \in \{0, \ldots, L\}$ |
| $\mathbf{W}_\ell^{(i)}, \mathbf{b}_\ell^{(i)}$ | primal | varies | float32 | PyTorch default | Renderer MLP weights/biases (imag), $\ell \in \{0, \ldots, L\}$ |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $I_k^{\text{obs}}$ | observations | $(M, N)$ | float32 | Loaded from file | Measured intensity for LED $k$ |
| $\mathbf{I}_{\text{sum}}$ | observations | $(M, N, K)$ | float32 | $I_k^{\text{obs}} / \max(I)$ | Normalized intensity stack |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $P_0$ | constants | $(M, N)$ | float32 | Computed | Binary pupil support mask |
| $k_{zz}$ | constants | $(M, N)$ | complex64 | Computed | Angular spectrum propagation kernel |
| $\mathbf{p}_k = (u_k, v_k)$ | constants | $(K, 2)$ | int32 | Computed | LED positions in Fourier space |
| $k_0$ | constants | scalar | float32 | $2\pi/\lambda$ | Wave number |
| $k_{\max}$ | constants | scalar | float32 | $\text{NA} \cdot k_0$ | Maximum spatial frequency |
| $\Delta_p$ | constants | scalar | float32 | Loaded | Effective pixel size |
| $\gamma$ | constants | scalar | int32 | 3 | Magnification factor (MAGimg) |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Load Raw Data
- **step_order**: 1
- **step_name**: Load intensity measurements
- **formula**: 
$$\mathbf{I}_{\text{raw}} \leftarrow \text{LoadMAT}(\texttt{path})[:,\, :,\, 0:N_{\text{modes}}^2]$$
- **inputs**: File path (string)
- **outputs**: $\mathbf{I}_{\text{raw}} \in \mathbb{R}^{M \times N \times K}$

### Step 3.2: Compute Wave Parameters
- **step_order**: 2
- **step_name**: Compute wave number and maximum frequency
- **formula**:
$$k_0 = \frac{2\pi}{\lambda}, \quad k_{\max} = \text{NA} \cdot k_0, \quad \Delta_p = \frac{\text{pixel\_size}}{\text{mag}}$$
- **inputs**: $\lambda$ (wavelength), NA (numerical aperture), pixel_size, mag
- **outputs**: $k_0$, $k_{\max}$, $\Delta_p$ (scalars)

### Step 3.3: Compute High-Resolution Grid Dimensions
- **step_order**: 3
- **step_name**: Compute upsampled dimensions
- **formula**:
$$M_H = \gamma \cdot M, \quad N_H = \gamma \cdot N$$
- **inputs**: $M$, $N$ (low-res dimensions), $\gamma$ (magnification factor)
- **outputs**: $M_H$, $N_H$ (high-res dimensions)

### Step 3.4: Compute Fourier Frequency Axes
- **step_order**: 4
- **step_name**: Compute frequency coordinates for high-resolution grid
- **formula**:
$$F_{xx}[n] = \frac{2\pi (n - N_H/2)}{N \cdot \Delta_p}, \quad n \in \{0, \ldots, N_H - 1\}$$
$$F_{yy}[m] = \frac{2\pi (m - M_H/2)}{M \cdot \Delta_p}, \quad m \in \{0, \ldots, M_H - 1\}$$
- **inputs**: $M_H$, $N_H$, $M$, $N$, $\Delta_p$
- **outputs**: $F_{xx} \in \mathbb{R}^{N_H}$, $F_{yy} \in \mathbb{R}^{M_H}$

### Step 3.5: Compute LED Illumination Angles
- **step_order**: 5
- **step_name**: Extract and sort LED numerical apertures
- **formula**:
$$u_k = -\text{NA}_x[k], \quad v_k = -\text{NA}_y[k]$$
$$\text{NA}_{\text{illu}}[k] = \sqrt{u_k^2 + v_k^2}$$
$$\sigma = \text{argsort}(\text{NA}_{\text{illu}})$$
$$u \leftarrow u[\sigma], \quad v \leftarrow v[\sigma]$$
- **inputs**: $\text{NA}_x$, $\text{NA}_y \in \mathbb{R}^K$
- **outputs**: Sorted $u$, $v \in \mathbb{R}^K$, permutation $\sigma$

### Step 3.6: Compute LED Positions in Fourier Space
- **step_order**: 6
- **step_name**: Map LED angles to frequency indices
- **formula**:
$$\mathbf{p}_k[0] = \arg\min_n |F_{xx}[n] - k_0 \cdot u_k|$$
$$\mathbf{p}_k[1] = \arg\min_m |F_{yy}[m] - k_0 \cdot v_k|$$
- **inputs**: $F_{xx}$, $F_{yy}$, $k_0$, $u$, $v$
- **outputs**: $\mathbf{p} \in \mathbb{Z}^{K \times 2}$ (LED positions)

### Step 3.7: Normalize Intensities
- **step_order**: 7
- **step_name**: Reorder and normalize intensity stack
- **formula**:
$$\mathbf{I}_{\text{sum}} = \frac{\mathbf{I}_{\text{raw}}[:,:,\sigma]}{\max(\mathbf{I}_{\text{raw}})}$$
- **inputs**: $\mathbf{I}_{\text{raw}}$, permutation $\sigma$
- **outputs**: $\mathbf{I}_{\text{sum}} \in \mathbb{R}^{M \times N \times K}$

### Step 3.8: Compute Angular Spectrum Propagation Kernel
- **step_order**: 8
- **step_name**: Compute $k_z$ for defocus propagation
- **formula**:
$$k_{xx}[m,n] = \frac{\pi}{\Delta_p}\left(\frac{2n}{N} - 1\right), \quad k_{yy}[m,n] = \frac{\pi}{\Delta_p}\left(\frac{2m}{M} - 1\right)$$
$$k_{rr} = \sqrt{k_{xx}^2 + k_{yy}^2}$$
$$k_{zz}[m,n] = \begin{cases} \sqrt{k_0^2 - k_{rr}[m,n]^2} & \text{if } k_0^2 > k_{rr}[m,n]^2 \\ 0 & \text{otherwise} \end{cases}$$
- **inputs**: $k_0$, $\Delta_p$, $M$, $N$
- **outputs**: $k_{zz} \in \mathbb{C}^{M \times N}$

### Step 3.9: Compute Pupil Support Mask
- **step_order**: 9
- **step_name**: Binary aperture mask
- **formula**:
$$F_x[m,n] = \frac{2\pi(n - N/2)}{N \cdot \Delta_p}, \quad F_y[m,n] = \frac{2\pi(m - M/2)}{M \cdot \Delta_p}$$
$$P_0[m,n] = \begin{cases} 1 & \text{if } F_x^2 + F_y^2 \leq k_{\max}^2 \\ 0 & \text{otherwise} \end{cases}$$
- **inputs**: $k_{\max}$, $\Delta_p$, $M$, $N$
- **outputs**: $P_0 \in \{0,1\}^{M \times N}$

### Step 3.10: Compute Bilinear Interpolation Indices
- **step_order**: 10
- **step_name**: Precompute interpolation coordinates for feature sampling
- **formula**:
$$\xi_x[p] = \frac{p + 0.5}{W_H}, \quad \xi_y[q] = \frac{q + 0.5}{H_H}, \quad p \in \{0,\ldots,W_H-1\}, q \in \{0,\ldots,H_H-1\}$$
$$s_x = \xi_x \cdot M_x, \quad s_y = \xi_y \cdot M_y$$
$$x_0 = \lfloor s_x \rfloor, \quad x_1 = \min(x_0 + 1, M_x - 1)$$
$$y_0 = \lfloor s_y \rfloor, \quad y_1 = \min(y_0 + 1, M_y - 1)$$
$$w_x = s_x - x_0, \quad w_y = s_y - y_0$$
- **inputs**: $W_H$, $H_H$ (output dimensions), $M_x$, $M_y$ (grid dimensions)
- **outputs**: $x_0, x_1, y_0, y_1 \in \mathbb{Z}^{W_H \cdot H_H}$, $w_x, w_y \in \mathbb{R}^{W_H \cdot H_H}$

---

## 4. ALGORITHM STEPS

### Initialization

**BEGIN INITIALIZATION**

#### Step I.1: Initialize Feature Grids
- **step_name**: Initialize learnable XY feature tensors
- **formula**:
$$\mathbf{D}_{xy}^{(r)}[i,j,f] \sim \mathcal{U}(-10^{-4}, 10^{-4})$$
$$\mathbf{D}_{xy}^{(i)}[i,j,f] \sim \mathcal{U}(-10^{-4}, 10^{-4})$$
- **inputs**: $M_x$, $M_y$, $F$ (feature dimension)
- **outputs**: $\mathbf{D}_{xy}^{(r)}, \mathbf{D}_{xy}^{(i)} \in \mathbb{R}^{M_x \times M_y \times F}$
- **computational_note**: Shape $(M_x, M_y, F) = (\texttt{num\_modes}, \texttt{num\_modes}, 32)$

#### Step I.2: Initialize Z Feature Vectors
- **step_name**: Initialize learnable Z feature vectors
- **formula**:
$$\mathbf{D}_z^{(r)}[i,f] \sim \mathcal{N}(0, 1)$$
$$\mathbf{D}_z^{(i)}[i,f] \sim \mathcal{N}(0, 1)$$
- **inputs**: $M_z = 5$, $F$
- **outputs**: $\mathbf{D}_z^{(r)}, \mathbf{D}_z^{(i)} \in \mathbb{R}^{M_z \times F}$
- **computational_note**: $M_z = 5$ is hardcoded

#### Step I.3: Initialize Renderer MLPs
- **step_name**: Initialize MLP weights
- **formula**: For each layer $\ell \in \{0, \ldots, L\}$:
$$\mathbf{W}_\ell \in \mathbb{R}^{d_{\ell+1} \times d_\ell}, \quad \mathbf{b}_\ell \in \mathbb{R}^{d_{\ell+1}}$$
where $d_0 = F$, $d_\ell = H$ for $\ell \in \{1, \ldots, L-1\}$, $d_L = 1$
- **inputs**: $F = 32$ (input dim), $H = 32$ (hidden dim), $L = 2$ (num layers)
- **outputs**: $\{\mathbf{W}_\ell^{(r)}, \mathbf{b}_\ell^{(r)}\}_{\ell=0}^L$, $\{\mathbf{W}_\ell^{(i)}, \mathbf{b}_\ell^{(i)}\}_{\ell=0}^L$
- **computational_note**: PyTorch default initialization (Kaiming uniform)

#### Step I.4: Initialize Optimizer
- **step_name**: Configure Adam optimizer
- **formula**:
$$\mathbf{m}_0 = \mathbf{0}, \quad \mathbf{v}_0 = \mathbf{0}, \quad t = 0$$
- **inputs**: Learning rate $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$
- **outputs**: Optimizer state
- **computational_note**: Applied to all parameters with `requires_grad=True`

#### Step I.5: Initialize Learning Rate Scheduler
- **step_name**: Configure StepLR scheduler
- **formula**:
$$\eta_e = \eta_0 \cdot \gamma_{\text{lr}}^{\lfloor e / s \rfloor}$$
where $\gamma_{\text{lr}} = 0.1$, $s = 6$ (step size)
- **inputs**: Initial learning rate $\eta_0$, decay factor $\gamma_{\text{lr}}$, step size $s$
- **outputs**: Scheduler state
- **computational_note**: Learning rate decays by factor 10 every 6 epochs

**END INITIALIZATION**

---

### Iteration Loop

**BEGIN ITERATION LOOP** (for epoch $e = 0, 1, \ldots, E-1$)

#### Step 4.1: Set Defocus Value
- **step_name**: Define defocus parameter
- **formula**:
$$\Delta z = 0$$
- **inputs**: None
- **outputs**: $\Delta z \in \mathbb{R}$
- **computational_note**: Single defocus plane at $z = 0$

#### Step 4.2: Iterate Over LEDs
**BEGIN LED LOOP** (for $k = 0, 1, \ldots, K-1$)

##### Step 4.2.1: Zero Gradients
- **step_name**: Reset gradient accumulators
- **formula**:
$$\nabla_\Theta \mathcal{L} \leftarrow \mathbf{0}$$
- **inputs**: Current parameters $\Theta$
- **outputs**: Zeroed gradients
- **computational_note**: `model.zero_grad()`

##### Step 4.2.2: Compute Defocus Mask
- **step_name**: Angular spectrum propagation phase
- **formula**:
$$H_{\text{df}}[m,n] = \exp\left(i \cdot k_{zz}[m,n] \cdot \Delta z\right)$$
- **inputs**: $k_{zz} \in \mathbb{C}^{M \times N}$, $\Delta z$
- **outputs**: $H_{\text{df}} \in \mathbb{C}^{M \times N}$
- **computational_note**: Complex exponential for defocus

##### Step 4.2.3: Compute Spectrum Mask
- **step_name**: Combine pupil and defocus
- **formula**:
$$P_k = P_0 \odot |H_{\text{df}}| \odot \exp\left(i \cdot P_0 \odot \angle H_{\text{df}}\right)$$
- **inputs**: $P_0$, $H_{\text{df}}$
- **outputs**: $P_k \in \mathbb{C}^{M \times N}$
- **computational_note**: Element-wise operations

##### Step 4.2.4: Sample XY Features (Bilinear Interpolation)
- **step_name**: Interpolate feature grid to output resolution
- **formula**: For component $c \in \{r, i\}$:
$$\mathbf{F}_{xy}^{(c)}[p] = \mathbf{D}_{xy}^{(c)}[y_0[p], x_0[p]] \cdot (1 - w_x[p])(1 - w_y[p])$$
$$\quad + \mathbf{D}_{xy}^{(c)}[y_0[p], x_1[p]] \cdot w_x[p](1 - w_y[p])$$
$$\quad + \mathbf{D}_{xy}^{(c)}[y_1[p], x_0[p]] \cdot (1 - w_x[p]) w_y[p]$$
$$\quad + \mathbf{D}_{xy}^{(c)}[y_1[p], x_1[p]] \cdot w_x[p] \cdot w_y[p]$$
- **inputs**: $\mathbf{D}_{xy}^{(c)}$, indices $(x_0, x_1, y_0, y_1)$, weights $(w_x, w_y)$
- **outputs**: $\mathbf{F}_{xy}^{(c)} \in \mathbb{R}^{W_H \cdot H_H \times F}$
- **computational_note**: Standard bilinear interpolation

##### Step 4.2.5: Sample Z Features (Linear Interpolation)
- **step_name**: Interpolate z-feature vector
- **formula**: 
$$\tilde{z} = (M_z - 1) \cdot \frac{\Delta z - z_{\min}}{z_{\max} - z_{\min}}$$
$$z_0 = \text{clamp}(\lfloor \tilde{z} \rfloor, 0, M_z - 1), \quad z_1 = \text{clamp}(z_0 + 1, 0, M_z - 1)$$
$$w_z = \tilde{z} - \lfloor \tilde{z} \rfloor$$
$$\mathbf{F}_z^{(c)} = \mathbf{D}_z^{(c)}[z_0] \cdot (1 - w_z) + \mathbf{D}_z^{(c)}[z_1] \cdot w_z$$
- **inputs**: $\mathbf{D}_z^{(c)}$, $\Delta z$, $z_{\min} = 0$, $z_{\max} = 1$
- **outputs**: $\mathbf{F}_z^{(c)} \in \mathbb{R}^{F}$
- **computational_note**: With $\Delta z = 0$, $z_{\min} = 0$, $z_{\max} = 1$: $\tilde{z} = 0$

##### Step 4.2.6: Combine XY and Z Features
- **step_name**: Element-wise product of spatial and depth features
- **formula**:
$$\mathbf{F}^{(c)}[p, f] = \mathbf{F}_{xy}^{(c)}[p, f] \cdot \mathbf{F}_z^{(c)}[f]$$
- **inputs**: $\mathbf{F}_{xy}^{(c)} \in \mathbb{R}^{W_H \cdot H_H \times F}$, $\mathbf{F}_z^{(c)} \in \mathbb{R}^{F}$
- **outputs**: $\mathbf{F}^{(c)} \in \mathbb{R}^{W_H \cdot H_H \times F}$
- **computational_note**: Broadcasting over spatial dimension

##### Step 4.2.7: Render via MLP
- **step_name**: Pass features through renderer network
- **formula**: For each component $c \in \{r, i\}$:
$$\mathbf{h}_0^{(c)} = \mathbf{F}^{(c)}$$
$$\mathbf{h}_{\ell+1}^{(c)} = \text{ReLU}\left(\mathbf{h}_\ell^{(c)} \mathbf{W}_\ell^{(c)\top} + \mathbf{b}_\ell^{(c)}\right), \quad \ell = 0, \ldots, L-1$$
$$\mathbf{o}^{(c)} = \mathbf{h}_L^{(c)} \mathbf{W}_L^{(c)\top} + \mathbf{b}_L^{(c)}$$
where $\text{ReLU}(x) = \max(0, x)$
- **inputs**: $\mathbf{F}^{(c)} \in \mathbb{R}^{W_H \cdot H_H \times F}$, MLP parameters
- **outputs**: $\mathbf{o}^{(c)} \in \mathbb{R}^{W_H \cdot H_H \times 1}$
- **computational_note**: Two hidden layers with ReLU activation

##### Step 4.2.8: Reshape to Image
- **step_name**: Reshape MLP output to spatial grid
- **formula**:
$$A[m,n] = \mathbf{o}^{(r)}[m \cdot H_H + n]$$
$$\Phi[m,n] = \mathbf{o}^{(i)}[m \cdot H_H + n]$$
- **inputs**: $\mathbf{o}^{(r)}, \mathbf{o}^{(i)} \in \mathbb{R}^{W_H \cdot H_H}$
- **outputs**: $A, \Phi \in \mathbb{R}^{W_H \times H_H}$ (amplitude and phase)
- **computational_note**: Reshape operation with potential upsampling

##### Step 4.2.9: Apply Bilinear Upsampling
- **step_name**: Upsample to full resolution if downsampled
- **formula**:
$$A_{\text{up}} = \text{BilinearUpsample}(A, \text{scale}=\text{ds\_factor})$$
$$\Phi_{\text{up}} = \text{BilinearUpsample}(\Phi, \text{scale}=\text{ds\_factor})$$
- **inputs**: $A, \Phi$, ds_factor $= 1$
- **outputs**: $A_{\text{up}}, \Phi_{\text{up}} \in \mathbb{R}^{M_H \times N_H}$
- **computational_note**: When ds_factor $= 1$, this is identity

##### Step 4.2.10: Form Complex Image
- **step_name**: Combine amplitude and phase
- **formula**:
$$O_{\text{img}}[m,n] = A_{\text{up}}[m,n] \cdot \exp\left(i \cdot \Phi_{\text{up}}[m,n]\right)$$
- **inputs**: $A_{\text{up}}, \Phi_{\text{up}} \in \mathbb{R}^{M_H \times N_H}$
- **outputs**: $O_{\text{img}} \in \mathbb{C}^{M_H \times N_H}$
- **computational_note**: Euler's formula for complex representation

##### Step 4.2.11: Compute Object Spectrum
- **step_name**: 2D FFT with centering
- **formula**:
$$\tilde{O} = \text{fftshift}\left(\text{FFT2}(O_{\text{img}})\right)$$
- **inputs**: $O_{\text{img}} \in \mathbb{C}^{M_H \times N_H}$
- **outputs**: $\tilde{O} \in \mathbb{C}^{M_H \times N_H}$
- **computational_note**: Centered Fourier transform

##### Step 4.2.12: Pad Spectrum to High-Resolution Grid
- **step_name**: Zero-pad for frequency embedding
- **formula**:
$$\Delta_x = \frac{\gamma \cdot M - M_H}{2}, \quad \Delta_y = \frac{\gamma \cdot N - N_H}{2}$$
$$\tilde{O}_{\text{pad}}[m,n] = \begin{cases} \tilde{O}[m - \Delta_x, n - \Delta_y] & \text{if } \Delta_x \leq m < M_H + \Delta_x \text{ and } \Delta_y \leq n < N_H + \Delta_y \\ 0 & \text{otherwise} \end{cases}$$
- **inputs**: $\tilde{O} \in \mathbb{C}^{M_H \times N_H}$, $\gamma$
- **outputs**: $\tilde{O}_{\text{pad}} \in \mathbb{C}^{\gamma M \times \gamma N}$
- **computational_note**: When $M_H = \gamma M$, no padding needed

##### Step 4.2.13: Extract Sub-Aperture
- **step_name**: Crop spectrum region for LED $k$
- **formula**:
$$x_0^{(k)} = v_k - M/2, \quad x_1^{(k)} = v_k + M/2$$
$$y_0^{(k)} = u_k - N/2, \quad y_1^{(k)} = u_k + N/2$$
$$\tilde{O}_{\text{sub}}^{(k)} = \tilde{O}_{\text{pad}}[x_0^{(k)}:x_1^{(k)}, y_0^{(k)}:y_1^{(k)}]$$
- **inputs