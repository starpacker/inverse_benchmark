## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: FPM-INR (Fourier Ptychographic Microscopy via Implicit Neural Representations)
- **algorithm_family**: Neural Inverse Problems / Coordinate-based Neural Representation Optimization
- **summary**: Reconstructs high-resolution complex images from low-resolution FPM intensity measurements by optimizing an implicit neural representation through gradient descent on a physics-based forward model.

---

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\Theta} \sum_{t=1}^{T} \sum_{\ell=1}^{L} \mathcal{L}_{\text{SmoothL1}}\left( \sqrt{I^{(\ell)}}, \left| \mathcal{F}^{-1}\left[ P \cdot H_{dz} \cdot \text{Crop}_\ell\left( \text{Pad}\left( \mathcal{F}[f_\Theta(z_t)] \right) \right) \right] \right| \right)$$

where $\Theta = \{\mathbf{D}_{xy}^{(r)}, \mathbf{D}_{xy}^{(i)}, \mathbf{D}_z^{(r)}, \mathbf{D}_z^{(i)}, \mathbf{W}_{\text{MLP}}^{(r)}, \mathbf{W}_{\text{MLP}}^{(i)}\}$

### data_fidelity_term
**Smooth L1 Loss (Huber Loss)**:
$$\mathcal{L}_{\text{SmoothL1}}(\hat{y}, y) = \begin{cases} 0.5(y - \hat{y})^2 / \beta & \text{if } |y - \hat{y}| < \beta \\ |y - \hat{y}| - 0.5\beta & \text{otherwise} \end{cases}$$

where $\beta = 1.0$ (PyTorch default), $\hat{y} = \sqrt{I^{(\ell)}}$ (measured amplitude), $y = |\mathbf{o}^{(\ell)}|$ (predicted amplitude)

### regularization_term
None (implicit regularization through neural network architecture)

### constraints
None

---

## 2. VARIABLES

### Primal Variables (Optimizable Neural Network Parameters)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{D}_{xy}^{(r)}$ | primal | $(x_{\text{mode}}, y_{\text{mode}}, n_f)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4}) \cdot 2$ | XY feature tensor for real/amplitude component |
| $\mathbf{D}_{xy}^{(i)}$ | primal | $(x_{\text{mode}}, y_{\text{mode}}, n_f)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4}) \cdot 2$ | XY feature tensor for imaginary/phase component |
| $\mathbf{D}_z^{(r)}$ | primal | $(z_{\text{dim}}, n_f)$ | float32 | $\mathcal{N}(0, 1)$ | Z-axis feature tensor for real component |
| $\mathbf{D}_z^{(i)}$ | primal | $(z_{\text{dim}}, n_f)$ | float32 | $\mathcal{N}(0, 1)$ | Z-axis feature tensor for imaginary component |
| $\mathbf{W}_1^{(r)}, \mathbf{b}_1^{(r)}$ | primal | $(n_f, h_{\text{dim}}), (h_{\text{dim}},)$ | float32 | PyTorch default (Kaiming uniform) | MLP layer 1 weights/bias (real) |
| $\mathbf{W}_2^{(r)}, \mathbf{b}_2^{(r)}$ | primal | $(h_{\text{dim}}, h_{\text{dim}}), (h_{\text{dim}},)$ | float32 | PyTorch default | MLP layer 2 weights/bias (real) |
| $\mathbf{W}_3^{(r)}, \mathbf{b}_3^{(r)}$ | primal | $(h_{\text{dim}}, 1), (1,)$ | float32 | PyTorch default | MLP output layer weights/bias (real) |
| $\mathbf{W}_1^{(i)}, \mathbf{b}_1^{(i)}$ | primal | $(n_f, h_{\text{dim}}), (h_{\text{dim}},)$ | float32 | PyTorch default | MLP layer 1 weights/bias (imaginary) |
| $\mathbf{W}_2^{(i)}, \mathbf{b}_2^{(i)}$ | primal | $(h_{\text{dim}}, h_{\text{dim}}), (h_{\text{dim}},)$ | float32 | PyTorch default | MLP layer 2 weights/bias (imaginary) |
| $\mathbf{W}_3^{(i)}, \mathbf{b}_3^{(i)}$ | primal | $(h_{\text{dim}}, 1), (1,)$ | float32 | PyTorch default | MLP output layer weights/bias (imaginary) |

### Observations

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{I} \in \mathbb{R}^{M \times N \times L}$ | observations | $(M, N, L)$ | float32 | Loaded from .mat file, normalized by max, sorted by NA | Low-resolution intensity measurements |

### Auxiliary Variables (Computed During Forward Pass)

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{f}_{xy}$ | auxiliary | $(M \cdot N / d_s^2, n_f)$ | float32 | Computed via bilinear interpolation | Interpolated XY features |
| $\mathbf{f}_z$ | auxiliary | $(B, n_f)$ | float32 | Computed via linear interpolation | Interpolated Z features |
| $\mathbf{f}$ | auxiliary | $(B, M \cdot N / d_s^2, n_f)$ | float32 | $\mathbf{f} = \mathbf{f}_{xy} \odot \mathbf{f}_z$ | Combined feature tensor |
| $A(x,y)$ | auxiliary | $(B, MM, MM)$ | float32 | MLP output + upsample | Reconstructed amplitude |
| $\phi(x,y)$ | auxiliary | $(B, MM, MM)$ | float32 | MLP output + upsample | Reconstructed phase |
| $\mathbf{O}$ | auxiliary | $(B, MM, NN)$ | complex64 | $\mathcal{F}[A \cdot e^{i\phi}]$ | High-resolution spectrum |
| $\mathbf{O}^{(\ell)}_{\text{sub}}$ | auxiliary | $(B, L_b, M, N)$ | complex64 | Cropped from padded spectrum | Sub-aperture spectra |
| $\mathbf{o}^{(\ell)}$ | auxiliary | $(B, L_b, M, N)$ | complex64 | $\mathcal{F}^{-1}[\mathbf{O}^{(\ell)}_{\text{sub}} \cdot P \cdot H]$ | Low-resolution fields |

### Constants

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $n_f$ | constants | scalar | int | 32 | Number of features |
| $h_{\text{dim}}$ | constants | scalar | int | 32 | MLP hidden dimension |
| $n_{\text{layers}}$ | constants | scalar | int | 2 | Number of MLP hidden layers |
| $z_{\text{dim}}$ | constants | scalar | int | 5 | Z-axis feature dimension |
| $z_{\min}$ | constants | scalar | float | 0.0 | Minimum z value |
| $z_{\max}$ | constants | scalar | float | 1.0 | Maximum z value |
| $x_{\text{mode}}, y_{\text{mode}}$ | constants | scalar | int | 512 | Feature grid modes |
| $d_s$ | constants | scalar | int | 1 | Downsampling factor |
| MAGimg | constants | scalar | int | 3 | Image magnification factor |
| $\lambda$ | constants | scalar | float | 0.632 (red), 0.5126 (green), 0.471 (blue) | Wavelength in micrometers |
| $k_0$ | constants | scalar | float | $2\pi/\lambda$ | Wavenumber |
| $\Delta_{\text{pix}}$ | constants | scalar | float | From data file | Camera pixel size |
| $D_{\text{pix}}$ | constants | scalar | float | $\Delta_{\text{pix}}/\text{mag}$ | Effective pixel size |
| NA | constants | scalar | float | From data file | Numerical aperture |
| $k_{\max}$ | constants | scalar | float | $\text{NA} \cdot k_0$ | Maximum spatial frequency |
| $\eta$ | constants | scalar | float | $10^{-3}$ | Initial learning rate |
| $\gamma$ | constants | scalar | float | 0.1 | Learning rate decay factor |
| $T_{\text{decay}}$ | constants | scalar | int | 6 | Learning rate decay step |
| $L_b$ | constants | scalar | int | 1 | LED batch size |

### Precomputed Operators

| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $P$ | constants | $(1, 1, M, N)$ | float32 | Binary mask: $P[k_x, k_y] = \mathbb{1}[k_x^2 + k_y^2 \leq k_{\max}^2]$ | Pupil support function |
| $k_z$ | constants | $(1, M, N)$ | complex64 | $k_z = \sqrt{k_0^2 - k_x^2 - k_y^2}$ | Z-component of wave vector |
| $(u^{(\ell)}, v^{(\ell)})$ | constants | $(L, 2)$ | int | Computed from NA calibration | LED positions in frequency space |
| $\mathbf{w}_{\text{lerp}}$ | constants | $(M \cdot N / d_s^2, 2)$ | float32 | Fractional parts of scaled coordinates | Bilinear interpolation weights |

---

## 3. PREPROCESSING PIPELINE

| Step Order | Step Name | Formula | Inputs | Outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load Raw Data | $\mathbf{I}_{\text{raw}} \leftarrow \text{loadmat}(\text{path})$ | File path | $\mathbf{I}_{\text{raw}} \in \mathbb{R}^{M \times N \times L_{\text{total}}}$ |
| 2 | Crop Measurements | $\mathbf{I} \leftarrow \mathbf{I}_{\text{raw}}[0:n_{\text{modes}}, 0:n_{\text{modes}}, :]$ | $\mathbf{I}_{\text{raw}}$, $n_{\text{modes}}=512$ | $\mathbf{I} \in \mathbb{R}^{M \times N \times L}$ |
| 3 | Compute Wavenumber | $k_0 = \frac{2\pi}{\lambda}$ | $\lambda$ (color-dependent) | $k_0 \in \mathbb{R}$ |
| 4 | Compute High-Res Dimensions | $MM = M \cdot \text{MAGimg}, \quad NN = N \cdot \text{MAGimg}$ | $M, N$, MAGimg=3 | $MM, NN \in \mathbb{Z}$ |
| 5 | Compute Frequency Axes | $F_{xx}[\ell] = \frac{\ell - NN/2}{N \cdot D_{\text{pix}}} \cdot 2\pi, \quad F_{yy}[m] = \frac{m - MM/2}{M \cdot D_{\text{pix}}} \cdot 2\pi$ | $M, N, D_{\text{pix}}$ | $F_{xx} \in \mathbb{R}^{NN}, F_{yy} \in \mathbb{R}^{MM}$ |
| 6 | Sort LEDs by NA | $\text{order} = \text{argsort}(\sqrt{u^2 + v^2})$; $u \leftarrow u[\text{order}], v \leftarrow v[\text{order}]$ | NA calibration $(u, v)$ | Sorted $(u, v)$ |
| 7 | Compute LED Positions | $\text{ledpos}[\ell, 0] = \arg\min_j |F_{xx}[j] - k_0 u[\ell]|$; $\text{ledpos}[\ell, 1] = \arg\min_j |F_{yy}[j] - k_0 v[\ell]|$ | $F_{xx}, F_{yy}, k_0, u, v$ | $\text{ledpos} \in \mathbb{Z}^{L \times 2}$ |
| 8 | Normalize Intensities | $\mathbf{I}_{\text{sum}} = \mathbf{I}[:,:,\text{order}] / \max(\mathbf{I})$ | $\mathbf{I}$, order | $\mathbf{I}_{\text{sum}} \in [0, 1]^{M \times N \times L}$ |
| 9 | Compute Angular Spectrum | $k_{rr} = \sqrt{k_{xx}^2 + k_{yy}^2}$; $k_z = \sqrt{k_0^2 - k_{rr}^2} \cdot \mathbb{1}[k_0^2 > k_{rr}^2]$ | $k_0$, frequency grids | $k_z \in \mathbb{C}^{M \times N}$ |
| 10 | Compute Pupil Support | $P[m, n] = \mathbb{1}\left[\left(\frac{n - N/2}{N \cdot D_{\text{pix}}} \cdot 2\pi\right)^2 + \left(\frac{m - M/2}{M \cdot D_{\text{pix}}} \cdot 2\pi\right)^2 \leq k_{\max}^2\right]$ | $M, N, D_{\text{pix}}, k_{\max}$ | $P \in \{0, 1\}^{M \times N}$ |
| 11 | Create Interpolation Coordinates | $\mathbf{xy} = \begin{bmatrix} \frac{0.5 + j}{w} \\ \frac{0.5 + i}{h} \end{bmatrix}_{i,j}$; $\mathbf{s} = \mathbf{xy} \cdot [x_{\text{mode}}, y_{\text{mode}}]^T$; $\mathbf{w}_{\text{lerp}} = \mathbf{s} - \lfloor \mathbf{s} \rfloor$ | Grid dimensions | Index arrays $(x_0, y_0, x_1, y_1)$, weights $\mathbf{w}_{\text{lerp}}$ |
| 12 | Transfer to Device | $P, k_z, \mathbf{I}_{\text{sum}} \leftarrow \text{to\_device}(\cdot)$ | CPU tensors | GPU tensors |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| I.1 | Initialize Feature Tensors | $\mathbf{D}_{xy}^{(r)}, \mathbf{D}_{xy}^{(i)} \sim 2 \cdot 10^{-4} \cdot \mathcal{U}(0,1) - 10^{-4}$; $\mathbf{D}_z^{(r)}, \mathbf{D}_z^{(i)} \sim \mathcal{N}(0,1)$ | Dimensions | Initialized tensors | Random initialization |
| I.2 | Initialize MLP Weights | $\mathbf{W}_k \sim \mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k = 1/\text{fan\_in}$ | Layer dimensions | MLP parameters | Kaiming uniform |
| I.3 | Setup Adam Optimizer | $m_0 = 0, v_0 = 0, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ | $\eta = 10^{-3}$ | Optimizer state | |
| I.4 | Setup LR Scheduler | StepLR with step\_size=6, $\gamma=0.1$ | Optimizer | Scheduler | |
| I.5 | Optional JIT/Compile | Model compilation (platform-dependent) | Model | Optimized model | Windows: JIT trace; Linux: torch.compile |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (epoch = 0 to $E-1$ where $E = 5$)

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| L.1 | Set LED Indices | $\text{led\_indices} = [0, 1, \ldots, L-1]$ | $L$ | Index list | Sequential ordering |
| L.2 | Set Defocus Values | $\mathbf{dz} = [0.0]$ | - | $\mathbf{dz} \in \mathbb{R}^1$ | Single focal plane |

**BEGIN_INNER_LOOP** (dz in dzs)

**BEGIN_LED_LOOP** (it = 0 to $\lfloor L/L_b \rfloor - 1$)

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| L.3 | Zero Gradients | $\nabla_\Theta \mathcal{L} \leftarrow 0$ | Model | Zeroed gradients | |
| L.4 | Compute Defocus Mask | $H_{dz}[m,n] = \exp(i \cdot k_z[m,n] \cdot dz)$ | $k_z$, $dz$ | $H_{dz} \in \mathbb{C}^{M \times N}$ | Angular spectrum propagation |
| L.5 | Select LED Batch | $\ell_{\text{batch}} = \text{led\_indices}[it \cdot L_b : (it+1) \cdot L_b]$ | it, $L_b$ | LED indices | |
| L.6 | Compute Spectrum Mask | $S = P \cdot |H_{dz}| \cdot \exp(i \cdot P \cdot \angle H_{dz})$ | $P$, $H_{dz}$ | $S \in \mathbb{C}^{B \times L_b \times M \times N}$ | Combined pupil and defocus |
| L.7 | Normalize Z Coordinate | $\tilde{z} = (z_{\text{dim}} - 1) \cdot \frac{dz - z_{\min}}{z_{\max} - z_{\min}}$ | $dz$, $z_{\min}$, $z_{\max}$, $z_{\text{dim}}$ | $\tilde{z} \in [0, z_{\text{dim}}-1]$ | |
| L.8 | Interpolate Z Features | $z_0 = \text{clamp}(\lfloor \tilde{z} \rfloor, 0, z_{\text{dim}}-1)$; $z_1 = \text{clamp}(z_0 + 1, 0, z_{\text{dim}}-1)$; $w_z = \tilde{z} - \lfloor \tilde{z} \rfloor$; $\mathbf{f}_z = (1-w_z) \mathbf{D}_z[z_0] + w_z \mathbf{D}_z[z_1]$ | $\mathbf{D}_z$, $\tilde{z}$ | $\mathbf{f}_z \in \mathbb{R}^{B \times n_f}$ | Linear interpolation |
| L.9 | Interpolate XY Features | $\mathbf{f}_{xy} = (1-w_x)(1-w_y)\mathbf{D}_{xy}[y_0, x_0] + w_x(1-w_y)\mathbf{D}_{xy}[y_0, x_1] + (1-w_x)w_y\mathbf{D}_{xy}[y_1, x_0] + w_x w_y \mathbf{D}_{xy}[y_1, x_1]$ | $\mathbf{D}_{xy}$, indices, weights | $\mathbf{f}_{xy} \in \mathbb{R}^{HW/d_s^2 \times n_f}$ | Bilinear interpolation |
| L.10 | Combine Features | $\mathbf{f} = \mathbf{f}_{xy} \odot \mathbf{f}_z$ (broadcast and element-wise product) | $\mathbf{f}_{xy}$, $\mathbf{f}_z$ | $\mathbf{f} \in \mathbb{R}^{B \times HW/d_s^2 \times n_f}$ | Multiplicative combination |
| L.11 | MLP Forward Pass | $\mathbf{h}_1 = \text{ReLU}(\mathbf{f} \mathbf{W}_1 + \mathbf{b}_1)$; $\mathbf{h}_2 = \text{ReLU}(\mathbf{h}_1 \mathbf{W}_2 + \mathbf{b}_2)$; $\mathbf{out} = \mathbf{h}_2 \mathbf{W}_3 + \mathbf{b}_3$ | $\mathbf{f}$, MLP params | $\mathbf{out} \in \mathbb{R}^{B \times HW/d_s^2 \times 1}$ | Two hidden layers |
| L.12 | Reshape Output | $\mathbf{img}_{\text{low}} = \text{reshape}(\mathbf{out}, (B, 1, w/d_s, h/d_s))$ | $\mathbf{out}$ | $\mathbf{img}_{\text{low}}$ | To 2D grid |
| L.13 | Bilinear Upsample | $A = \text{Upsample}(\mathbf{img}_{\text{low}}^{(r)}, d_s)$; $\phi = \text{Upsample}(\mathbf{img}_{\text{low}}^{(i)}, d_s)$ | Low-res images, $d_s=1$ | $A, \phi \in \mathbb{R}^{B \times MM \times MM}$ | Scale factor = 1 (identity) |
| L.14 | Form Complex Image | $\mathbf{E} = A \cdot \exp(i \phi)$ | $A$, $\phi$ | $\mathbf{E} \in \mathbb{C}^{B \times MM \times MM}$ | Complex field |
| L.15 | Compute High-Res Spectrum | $\mathbf{O} = \text{fftshift}(\text{FFT2}(\mathbf{E}))$ | $\mathbf{E}$ | $\mathbf{O} \in \mathbb{C}^{B \times MM \times NN}$ | Centered spectrum |
| L.16 | Pad Spectrum | $\mathbf{O}_{\text{pad}} = \text{ZeroPad}(\mathbf{O}, p_x, p_y)$ where $p_x = p_y = \frac{M \cdot \text{MAGimg} \cdot \text{MAGimg} - MM}{2}$ | $\mathbf{O}$ | $\mathbf{O}_{\text{pad}} \in \mathbb{C}^{B \times MM \cdot \text{MAGimg} \times NN \cdot \text{MAGimg}}$ | Zero-padding for extraction |
| L.17 | Compute Crop Coordinates | $x_0^{(\ell)} = v^{(\ell)} - M/2$; $x_1^{(\ell)} = v^{(\ell)} + M/2$; $y_0^{(\ell)} = u^{(\ell)} - N/2$; $y_1^{(\ell)} = u^{(\ell)} + N/2$ | LED positions | Crop indices | |
| L.18 | Extract Sub-Apertures | $\mathbf{O}^{(\ell)}_{\text{sub}} = \mathbf{O}_{\text{pad}}[:, x_0^{(\ell)}:x_1^{(\ell)}, y_0^{(\ell)}:y_1^{(\ell)}]$ | $\mathbf{O}_{\text{pad}}$, crop indices | $\mathbf{O}_{\text{sub}} \in \mathbb{C}^{B \times L_b \times M \times N}$ | Stack for all LEDs in batch |
| L.19 | Apply Spectrum Mask | $\mathbf{O}'_{\text{sub}} = \mathbf{O}_{\text{sub}} \odot S$ | $\mathbf{O}_{\text{sub}}$, $S$ | $\mathbf{O}'_{\text{sub}} \in \mathbb{C}^{B \times L_b \times M \times N}$ | Pupil + defocus |
| L.20 | Inverse FFT | $\mathbf{o} = \text{IFFT2}(\text{ifftshift}(\mathbf{O}'_{\text{sub}}))$ | $\mathbf{O}'_{\text{sub}}$ | $\mathbf{o} \in \mathbb{C}^{B \times L_b \times M \times N}$ | Low-res field |
| L.21 | Compute Predicted Intensity | $\hat{I}^{1/2} = |\mathbf{o}|$ | $\mathbf{o}$ | $\hat{I}^{1/2} \in \mathbb{R}^{B \times L_b \times M \times N}$ | Amplitude |
| L.22 | Prepare Target | $I^{1/2}_{\text{cap}} = \sqrt{\mathbf{I}_{\text{sum}}[:,:,\ell_{\text{batch}}]}$ (permuted) | $\mathbf{I}_{\text{sum}}$ | $I^{1/2}_{\text{cap}} \in \mathbb{R}^{B \times L_b \times M \times N}$ | |
| L.23 | Compute Loss | $\mathcal{L} = \text{SmoothL1Loss}(I^{1/2}_{\text{cap}}, \hat{I}^{1/2})$ | Predicted, target | $\mathcal{L} \in \mathbb{R}$ | |
| L.24 | Compute MSE | $\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum(I^{1/2}_{\text{cap}} - \hat{I}^{1/2})^2$ | Predicted, target | $\mathcal{L}_{\text{MSE}} \in \mathbb{R}$ | For PSNR |
| L.25 | Backpropagation | $\nabla_\Theta \mathcal{L} = \text{autograd}(\mathcal{L}, \Theta)$ | $\mathcal{L}$ | Gradients | |
| L.26 | Compute PSNR | $\text{PSNR} = -10 \log_{10}(\mathcal{L}_{\text{MSE}})$ | $\mathcal{L}_{\text{MSE}}$ | PSNR in dB | |
| L.27 | Adam Update | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$; $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$; $\hat{m}_t = m_t/(1-\beta_1^t)$; $\hat{v}_t = v_t/(1-\beta_2^t)$; $\theta_t = \theta_{t-1} - \eta \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$ | Gradients, state | Updated $\Theta$ | |

**END_LED_LOOP**

**END_INNER_LOOP**

| Step | Step Name | Formula | Inputs | Outputs | Note |
|------|-----------|---------|--------|---------|------|
| L.28 | LR Scheduler Step | $\eta_{\text{new}} = \eta \cdot \gamma^{\lfloor \text{epoch}/T_{\text{decay}} \rfloor}$ | epoch | Updated $\eta$ | Step every 6 epochs |
| L.29 | Store Results | $A_{\text{final}}, \phi_{\text{final}} \leftarrow A[0], \phi[0]$ | Current outputs | NumPy arrays | Detach and convert |
| L.30 | Save