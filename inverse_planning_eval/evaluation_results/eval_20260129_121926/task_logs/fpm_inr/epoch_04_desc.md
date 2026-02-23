## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: FPM-INR (Fourier Ptychographic Microscopy via Implicit Neural Representation)
- **algorithm_family**: Neural Network-based Inverse Problem Solver / Implicit Neural Representation
- **summary**: Reconstructs a high-resolution complex image (amplitude and phase) from multiple low-resolution intensity measurements by optimizing a factorized 3D feature tensor decoded through an MLP, using an FPM forward model with LED-specific sub-aperture extraction.

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{\boldsymbol{\theta}} \sum_{k=1}^{K} \mathcal{L}_{\text{SmoothL1}}\left( \sqrt{I_k^{\text{obs}}}, \left| \mathcal{F}^{-1}\left[ P_k \odot \mathcal{S}_k\left[ \mathcal{F}[f_{\boldsymbol{\theta}}(z)] \right] \right] \right| \right)$$

where $\boldsymbol{\theta}$ denotes all trainable neural network parameters.

### data_fidelity_term

**Smooth L1 Loss (Huber Loss)**:
$$\mathcal{L}_{\text{SmoothL1}}(\hat{y}, y) = \begin{cases} \frac{1}{2}(\hat{y} - y)^2 / \beta & \text{if } |\hat{y} - y| < \beta \\ |\hat{y} - y| - \frac{\beta}{2} & \text{otherwise} \end{cases}$$

with $\beta = 1.0$ (PyTorch default). The loss compares:
- $\hat{y} = \sqrt{I_k^{\text{obs}}}$: Square root of observed intensity for LED $k$
- $y = |o_k^{\text{sub}}|$: Magnitude of predicted sub-aperture image

### regularization_term

**None** (no explicit regularization term). Implicit regularization arises from:
1. Low-rank factorization of the feature tensor
2. MLP decoder architecture (bandwidth-limited representation)
3. Bilinear interpolation smoothness

### constraints

**None** (unconstrained optimization on neural network parameters). The model output is unconstrained:
- Amplitude $A(x,y) \in \mathbb{R}$ (no non-negativity enforced)
- Phase $\phi(x,y) \in \mathbb{R}$ (unwrapped)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{D}^{\text{real}}$ | $(x_{\text{mode}}, y_{\text{mode}}, n_f)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4})$ | XY feature tensor for real component |
| $\mathbf{D}^{\text{imag}}$ | $(x_{\text{mode}}, y_{\text{mode}}, n_f)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4})$ | XY feature tensor for imaginary component |
| $\mathbf{Z}^{\text{real}}$ | $(z_{\text{mode}}, n_f)$ | float32 | $\mathcal{N}(0, 1)$ | Z feature tensor for real component |
| $\mathbf{Z}^{\text{imag}}$ | $(z_{\text{mode}}, n_f)$ | float32 | $\mathcal{N}(0, 1)$ | Z feature tensor for imaginary component |
| $\mathbf{W}_0^{\text{real}}$ | $(n_f, h_{\text{dim}})$ | float32 | PyTorch default | First layer weights, real renderer |
| $\mathbf{b}_0^{\text{real}}$ | $(h_{\text{dim}},)$ | float32 | PyTorch default | First layer bias, real renderer |
| $\mathbf{W}_1^{\text{real}}$ | $(h_{\text{dim}}, h_{\text{dim}})$ | float32 | PyTorch default | Second layer weights, real renderer |
| $\mathbf{b}_1^{\text{real}}$ | $(h_{\text{dim}},)$ | float32 | PyTorch default | Second layer bias, real renderer |
| $\mathbf{W}_2^{\text{real}}$ | $(h_{\text{dim}}, 1)$ | float32 | PyTorch default | Output layer weights, real renderer |
| $\mathbf{b}_2^{\text{real}}$ | $(1,)$ | float32 | PyTorch default | Output layer bias, real renderer |
| $\mathbf{W}_i^{\text{imag}}, \mathbf{b}_i^{\text{imag}}$ | (same as real) | float32 | PyTorch default | Imaginary renderer parameters |

### 2.2 Dual Variables

None (first-order gradient descent, no dual formulation).

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}_0, \mathbf{y}_0$ | $(W_{\text{ds}} \cdot H_{\text{ds}},)$ | int64 | Computed from grid | Lower interpolation indices for XY |
| $\mathbf{x}_1, \mathbf{y}_1$ | $(W_{\text{ds}} \cdot H_{\text{ds}},)$ | int64 | $\mathbf{x}_0 + 1$, $\mathbf{y}_0 + 1$ (clamped) | Upper interpolation indices for XY |
| $\boldsymbol{\alpha}$ | $(W_{\text{ds}} \cdot H_{\text{ds}}, 2)$ | float32 | Computed from grid | Bilinear interpolation weights (lerp_weights) |
| $\text{img\_ampli}$ | $(B, W, H)$ | float32 | Forward pass | Predicted amplitude image |
| $\text{img\_phase}$ | $(B, W, H)$ | float32 | Forward pass | Predicted phase image |
| $\text{img\_complex}$ | $(B, W, H)$ | complex64 | Computed | Complex image $A \cdot e^{i\phi}$ |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $n_f$ | 32 | Number of features per spatial location |
| $h_{\text{dim}}$ | 32 | Hidden dimension of MLP renderer |
| $n_{\text{layers}}$ | 2 | Number of hidden layers in renderer |
| $z_{\text{mode}}$ | 5 | Number of depth modes |
| $z_{\min}$ | 0.0 | Minimum defocus value |
| $z_{\max}$ | 1.0 | Maximum defocus value |
| $x_{\text{mode}}$ | num_modes (512) | Number of X modes in feature tensor |
| $y_{\text{mode}}$ | num_modes (512) | Number of Y modes in feature tensor |
| MAGimg | 3 | Magnification factor (high-res / low-res) |
| $\lambda_r$ | 0.632 μm | Red wavelength |
| $\lambda_g$ | 0.5126 μm | Green wavelength |
| $\lambda_b$ | 0.471 μm | Blue wavelength |
| $\eta$ | $10^{-3}$ | Initial learning rate |
| $\gamma$ | 0.1 | Learning rate decay factor |
| lr_decay_step | 6 | Epochs between LR decay |
| $\beta_{\text{SL1}}$ | 1.0 | Smooth L1 transition parameter |
| led_batch_size | 1 | LEDs processed per iteration |
| cur_ds | 1 | Current downsampling factor |
| $\epsilon_{\text{init}}$ | $2 \times 10^{-4}$ | Feature tensor initialization scale |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $I_k^{\text{obs}}$ | $(M, N)$ | float32 | Observed intensity for LED $k$, normalized by $\max(I)$ |
| $\mathbf{I}_{\text{sum}}$ | $(M, N, K)$ | float32 | Stack of all observed intensities (reordered by NA) |
| $P_0$ | $(M, N)$ | float32 | Binary pupil support mask |
| $k_{zz}$ | $(M, N)$ | complex64 | Angular spectrum propagation kernel |
| ledpos_true | $(K, 2)$ | int64 | LED positions in frequency domain |

---

## 3. PREPROCESSING PIPELINE

### Step 0: Physical Parameter Derivation

| Parameter | Formula |
|-----------|---------|
| $k_0$ | $\frac{2\pi}{\lambda}$ |
| $D_{\text{pixel}}$ | $\frac{\text{pixel\_size}}{\text{mag}}$ |
| $k_{\max}$ | $\text{NA} \cdot k_0$ |
| $MM$ | $M \cdot \text{MAGimg}$ |
| $NN$ | $N \cdot \text{MAGimg}$ |

**Inputs**: wavelength $\lambda$, pixel_size, mag, NA from .mat file  
**Outputs**: $k_0$, $D_{\text{pixel}}$, $k_{\max}$, $MM$, $NN$

### Step 1: Frequency Grid Construction

$$F_{xx,1}[n] = \frac{n - NN/2}{N \cdot D_{\text{pixel}}} \cdot 2\pi, \quad n \in [0, NN)$$

$$F_{yy,1}[m] = \frac{m - MM/2}{M \cdot D_{\text{pixel}}} \cdot 2\pi, \quad m \in [0, MM)$$

**Inputs**: $M$, $N$, $MM$, $NN$, $D_{\text{pixel}}$  
**Outputs**: $F_{xx,1} \in \mathbb{R}^{NN}$, $F_{yy,1} \in \mathbb{R}^{MM}$

### Step 2: LED Position Ordering

$$u_k = -\text{NAx}_k, \quad v_k = -\text{NAy}_k$$
$$\text{NA}_{\text{illu},k} = \sqrt{u_k^2 + v_k^2}$$
$$\text{order} = \text{argsort}(\text{NA}_{\text{illu}})$$

**Inputs**: NAx, NAy from calibration data  
**Outputs**: Reordered $u$, $v$, order indices

### Step 3: LED Position Mapping to Frequency Domain

$$\text{ledpos}[k, 0] = \arg\min_n |F_{xx,1}[n] - k_0 \cdot u_k|$$
$$\text{ledpos}[k, 1] = \arg\min_m |F_{yy,1}[m] - k_0 \cdot v_k|$$

**Inputs**: $F_{xx,1}$, $F_{yy,1}$, $u$, $v$, $k_0$  
**Outputs**: ledpos_true $\in \mathbb{Z}^{K \times 2}$

### Step 4: Intensity Normalization

$$I_{\text{sum}}[:,:,k] = \frac{I_{\text{low}}[:,:,\text{order}[k]]}{\max(I_{\text{low}})}$$

**Inputs**: $I_{\text{low}} \in \mathbb{R}^{M \times N \times K}$, order  
**Outputs**: $I_{\text{sum}} \in \mathbb{R}^{M \times N \times K}$ (values in $[0,1]$)

### Step 5: Angular Spectrum Kernel

$$k_{xx}[m,n], k_{yy}[m,n] = \text{meshgrid}\left(\text{linspace}\left(-\frac{\pi}{D_{\text{pixel}}}, \frac{\pi}{D_{\text{pixel}}}, M\right), \text{linspace}(\cdot, N)\right)$$

$$k_{rr} = \sqrt{k_{xx}^2 + k_{yy}^2}$$

$$k_{zz}[m,n] = \begin{cases} \sqrt{k_0^2 - k_{rr}^2} & \text{if } k_0^2 - k_{rr}^2 > 0 \\ i\sqrt{k_{rr}^2 - k_0^2} & \text{otherwise} \end{cases}$$

Equivalently: $k_{zz} = |k_{zz}| \cdot e^{i \cdot \arg(k_{zz})}$ where the square root is taken over complex domain.

**Inputs**: $M$, $N$, $D_{\text{pixel}}$, $k_0$  
**Outputs**: $k_{zz} \in \mathbb{C}^{M \times N}$

### Step 6: Pupil Support Mask

$$F_{x2}[m,n] = \left(\frac{n - N/2}{N \cdot D_{\text{pixel}}} \cdot 2\pi\right)^2$$
$$F_{y2}[m,n] = \left(\frac{m - M/2}{M \cdot D_{\text{pixel}}} \cdot 2\pi\right)^2$$
$$P_0[m,n] = \mathbb{1}\left[F_{x2} + F_{y2} \leq k_{\max}^2\right]$$

**Inputs**: $M$, $N$, $D_{\text{pixel}}$, $k_{\max}$  
**Outputs**: $P_0 \in \{0,1\}^{M \times N}$

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Operation | Formula / Description |
|------|-----------|----------------------|
| I.1 | Initialize XY feature tensors | $\mathbf{D}^{\text{real}}, \mathbf{D}^{\text{imag}} \sim 2 \times 10^{-4} \cdot \mathcal{U}(0,1) - 10^{-4}$ |
| I.2 | Initialize Z feature tensors | $\mathbf{Z}^{\text{real}}, \mathbf{Z}^{\text{imag}} \sim \mathcal{N}(0, 1)$ |
| I.3 | Initialize MLP weights | PyTorch default (Kaiming uniform for weights, uniform for biases) |
| I.4 | Create coordinate grids | Compute $\mathbf{x}_0, \mathbf{y}_0, \mathbf{x}_1, \mathbf{y}_1, \boldsymbol{\alpha}$ via `create_coords` |
| I.5 | Initialize optimizer | Adam with $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ |
| I.6 | Initialize scheduler | StepLR with step_size=6, gamma=0.1 |
| I.7 | (Optional) JIT compile | `torch.jit.trace` (Windows) or `torch.compile` (Linux) |

**Coordinate Grid Creation** (`create_coords`):

$$\text{xs}_i = \frac{i + 0.5}{W_{\text{ds}}}, \quad \text{ys}_j = \frac{j + 0.5}{H_{\text{ds}}}, \quad i \in [0, W_{\text{ds}}), j \in [0, H_{\text{ds}})$$

$$\text{xy}_{ij} = [\text{ys}_j, \text{xs}_i]^T$$

$$[\mathbf{x}_0, \mathbf{y}_0] = \lfloor \text{xy} \cdot [x_{\text{mode}}, y_{\text{mode}}] \rfloor$$

$$\boldsymbol{\alpha} = \text{xy} \cdot [x_{\text{mode}}, y_{\text{mode}}] - [\mathbf{x}_0, \mathbf{y}_0]$$

with clamping: $\mathbf{x}_0 \in [0, x_{\text{mode}}-1]$, $\mathbf{x}_1 \in [0, x_{\text{mode}}-1]$

### 4.2 ITERATION LOOP

```
BEGIN_LOOP (epoch = 0 to num_epochs - 1)
    Set led_indices = [0, 1, ..., K-1]
    Set dzs = [0.0]  # Single defocus plane

    BEGIN_LOOP (dz in dzs)
        BEGIN_LOOP (it = 0 to K // led_batch_size - 1)
```

| Step | Name | Formula | Inputs | Outputs | Note |
|------|------|---------|--------|---------|------|
| L.1 | Zero gradients | `model.zero_grad()` | — | — | Clear accumulated gradients |
| L.2 | Compute defocus mask | $\text{dfmask} = \exp(i \cdot k_{zz} \cdot dz)$ | $k_{zz} \in \mathbb{C}^{M \times N}$, $dz \in \mathbb{R}$ | $\text{dfmask} \in \mathbb{C}^{M \times N}$ | [FORWARD] Angular spectrum propagation |
| L.3 | Select LED batch | $\text{led\_num} = [\text{led\_indices}[it \cdot B_{\text{led}} : (it+1) \cdot B_{\text{led}}]]$ | led_indices, $it$, $B_{\text{led}}=1$ | led_num (list of indices) | Sequential LED processing |
| L.4 | Compute spectrum mask | $\text{spectrum\_mask} = P_0 \cdot |\text{dfmask}| \cdot \exp(i \cdot P_0 \cdot \angle\text{dfmask})$ | $P_0$, dfmask | $\text{spectrum\_mask} \in \mathbb{C}^{B \times |\text{led\_num}| \times M \times N}$ | Pupil-modulated defocus |
| L.5 | Sample XY features (real) | $\mathbf{f}_{xy}^{\text{real}} = \mathbf{D}[\mathbf{y}_0, \mathbf{x}_0](1-\alpha_x)(1-\alpha_y) + \mathbf{D}[\mathbf{y}_0, \mathbf{x}_1]\alpha_x(1-\alpha_y) + \mathbf{D}[\mathbf{y}_1, \mathbf{x}_0](1-\alpha_x)\alpha_y + \mathbf{D}[\mathbf{y}_1, \mathbf{x}_1]\alpha_x\alpha_y$ | $\mathbf{D}^{\text{real}}$, indices, $\boldsymbol{\alpha}$ | $\mathbf{f}_{xy}^{\text{real}} \in \mathbb{R}^{W_{\text{ds}} H_{\text{ds}} \times n_f}$ | Bilinear interpolation |
| L.6 | Normalize Z coordinate | $\tilde{z} = (z_{\text{mode}} - 1) \cdot \frac{dz - z_{\min}}{z_{\max} - z_{\min}}$ | $dz$, $z_{\min}$, $z_{\max}$, $z_{\text{mode}}$ | $\tilde{z} \in \mathbb{R}$ | Map defocus to index space |
| L.7 | Sample Z features (real) | $z_0 = \text{clamp}(\lfloor \tilde{z} \rfloor, 0, z_{\text{mode}}-1)$, $z_1 = \text{clamp}(z_0+1, 0, z_{\text{mode}}-1)$, $\mathbf{f}_z^{\text{real}} = \mathbf{Z}[z_0](1-\alpha_z) + \mathbf{Z}[z_1]\alpha_z$ where $\alpha_z = \tilde{z} - \lfloor \tilde{z} \rfloor$ | $\mathbf{Z}^{\text{real}}$, $\tilde{z}$ | $\mathbf{f}_z^{\text{real}} \in \mathbb{R}^{n_f}$ | Linear interpolation in Z |
| L.8 | Feature fusion | $\mathbf{f}^{\text{real}} = \mathbf{f}_{xy}^{\text{real}} \odot \mathbf{f}_z^{\text{real}}$ | $\mathbf{f}_{xy}$, $\mathbf{f}_z$ | $\mathbf{f}^{\text{real}} \in \mathbb{R}^{B_z \times W_{\text{ds}} H_{\text{ds}} \times n_f}$ | Element-wise multiplication (factorized representation) |
| L.9 | MLP decode (real) | $\text{img\_real} = \sigma(\sigma(\mathbf{f} \mathbf{W}_0 + \mathbf{b}_0) \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$ where $\sigma = \text{ReLU}$ | $\mathbf{f}^{\text{real}}$, MLP params | $\text{img\_real} \in \mathbb{R}^{B \times 1 \times W_{\text{ds}} \times H_{\text{ds}}}$ | 2-hidden-layer MLP |
| L.10 | Upsample (real) | $\text{img\_ampli} = \text{BilinearUpsample}(\text{img\_real}, \text{scale}=\text{cur\_ds})$ | img_real | $\text{img\_ampli} \in \mathbb{R}^{B \times W \times H}$ | Spatial upsampling |
| L.11 | Steps L.5-L.10 for imaginary | (Same operations with $\mathbf{D}^{\text{imag}}$, $\mathbf{Z}^{\text{imag}}$, imag MLP) | — | $\text{img\_phase} \in \mathbb{R}^{B \times W \times H}$ | Phase component |
| L.12 | Form complex image | $\text{img\_complex} = \text{img\_ampli} \cdot \exp(i \cdot \text{img\_phase})$ | img_ampli, img_phase | $\text{img\_complex} \in \mathbb{C}^{B \times W \times H}$ | Complex representation |
| L.13 | Compute sub-aperture bounds | $x_0^{(k)} = v_k - M/2$, $x_1^{(k)} = v_k + M/2$, $y_0^{(k)} = u_k - N/2$, $y_1^{(k)} = u_k + N/2$ | ledpos_true, $M$, $N$ | $x_0, x_1, y_0, y_1 \in \mathbb{Z}$ | LED-specific frequency window |
| L.14 | Prepare observation | $\hat{I}_k = \sqrt{I_{\text{sum}}[:,:,k]}$ | $I_{\text{sum}}$ | $\hat{I}_k \in \mathbb{R}^{M \times N}$ | Amplitude domain target |
| L.15 | [FORWARD] FFT | $O = \text{fftshift}(\text{fft2}(\text{img\_complex}))$ | img_complex | $O \in \mathbb{C}^{B \times MM \times NN}$ | Centered Fourier spectrum |
| L.16 | [FORWARD] Zero-pad spectrum | $O_{\text{pad}} = \text{pad}(O, (\text{pad}_y, \text{pad}_y, \text{pad}_x, \text{pad}_x))$ where $\text{pad}_x = (M \cdot \text{MAGimg} \cdot \text{MAGimg} - MM)/2$ | $O$, MAGimg | $O_{\text{pad}} \in \mathbb{C}^{B \times MM \cdot \text{MAGimg} \times NN \cdot \text{MAGimg}}$ | Embed in larger spectrum |
| L.17 | [FORWARD] Extract sub-apertures | $O_{\text{sub}}^{(k)} = O_{\text{pad}}[:, x_0^{(k)}:x_1^{(k)}, y_0^{(k)}:y_1^{(k)}]$ | $O_{\text{pad}}$, bounds | $O_{\text{sub}} \in \mathbb{C}^{B \times |\text{led\_num}| \times M \times N}$ | LED-specific spectral slice |
| L.18 | [FORWARD] Apply pupil mask | $O_{\text{masked}} = O_{\text{sub}} \odot \text{spectrum\_mask}$ | $O_{\text{sub}}$, spectrum_mask | $O_{\text{masked}} \in \mathbb{C}^{B \times |\text{led\_num}| \times M \times N}$ | Bandlimit enforcement |
| L.19 | [FORWARD] IFFT | $o_{\text{sub}} = \text{ifft2}(\text{ifftshift}(O_{\text{masked}}))$ | $O_{\text{masked}}$ | $o_{\text{sub}} \in \mathbb{C}^{B \times |\text{led\_num}| \times M \times N}$ | Spatial domain sub-image |
| L.20 | [FORWARD] Compute intensity | $\tilde{I}_k = |o_{\text{sub}}|$ | $o_{\text{sub}}$ | $\tilde{I}_k \in \mathbb{R}^{B \times |\text{led\_num}| \times M \times N}$ | Predicted amplitude |
| L.21 | Compute loss | $\mathcal{L} = \text{SmoothL1Loss}(\hat{I}_k, \tilde{I}_k)$ | $\hat{I}_k$, $\tilde{I}_k$ | $\mathcal{L} \in \mathbb{R}$ | Data fidelity |
| L.22 | Compute MSE (for PSNR) | $\mathcal{L}_{\text{MSE}} = \text{MSELoss}(\hat{I}_k, \tilde{I}_k)$ | $\hat{I}_k$, $\tilde{I}_k$ | $\mathcal{L}_{\text{MSE}} \in \mathbb{R}$ | For monitoring |
| L.23 | Backpropagation | $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ via autograd | $\mathcal{L}$, $\boldsymbol{\theta}$ | Gradients stored in tensors | Automatic differentiation |
| L.24 | Optimizer step | $\boldsymbol{\theta} \leftarrow \text{Adam}(\boldsymbol{\theta}, \nabla_{\boldsymbol{\theta}} \mathcal{L})$ | Gradients, optimizer state | Updated $\boldsymbol{\theta}$ | Adam update with $\eta$, $\beta_1=0.9$, $\beta_2=0.999$ |

```
        END_LOOP (it)
    END_LOOP (dz)

    L.25: Scheduler step: η ← η × γ (if epoch % lr_decay_step == 0)
    L.26: Store current reconstruction: final_amplitude ← img_ampli[0], final_phase ← img_phase[0]
    L.27: Save visualization (every epoch)

END_LOOP (epoch)
```

### 4.3 FINALIZATION

| Step | Operation | Description |
|------|-----------|-------------|
| F.1 | Extract final amplitude | `final_amplitude = img_ampli[0].float().cpu().detach().numpy()` |
| F.2 | Extract final phase | `final_phase = img_phase[0].float().cpu().detach().numpy()` |
| F.3 | Compute statistics | Mean, std, min, max for amplitude and phase |
| F.4 | Save model parameters | Extract trainable parameters and save via `torch.save` |
| F.5 | Save results | Save amplitude and phase as .npy files |

---

## 5. STOPPING CRITERION

- **type**: iteration_limit
- **expression**: $\text{epoch} < \text{num\_epochs}$
- **parameters**: 
  - `num_epochs = 5` (default)
  - Inner loop: `it < ID_len // led_batch_size` (process all LEDs per epoch)
- **check_frequency**: Evaluated at the end of each epoch and each LED batch iteration

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Smooth L1 Loss | $\mathcal{L}_{\text{SL1}} = \frac{1}{N}\sum_{i}\begin{cases} \frac{1}{2}(y_i - \hat{y}_i)^2 & |y_i - \hat{y}_i| < 1 \\ |y_i - \hat{y}_i| - 0.5 & \text{otherwise}\end{cases}$ | Primary optimization objective |
| PSNR | $\text{PSNR} = 10 \cdot \log_{10}\left