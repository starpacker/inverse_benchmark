## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: FPM-INR (Fourier Ptychographic Microscopy with Implicit Neural Representations)
- **algorithm_family**: Neural-Network-Based Inverse Problems / Coordinate-Based Neural Representations
- **summary**: Reconstructs a high-resolution complex image (amplitude and phase) from multiple low-resolution intensity measurements captured under varying LED illumination angles by optimizing an implicit neural representation through gradient descent on a Smooth L1 data fidelity loss.

---

## 1. OBJECTIVE FUNCTION

### full_expression

$$\min_{\Theta} \sum_{t=1}^{T} \mathcal{L}_{\text{SmoothL1}}\left( \sqrt{I_t}, \left| \mathcal{F}^{-1}\left\{ P \cdot H_{\text{df}} \cdot \text{Crop}_t\left( \text{Pad}\left( \mathcal{F}\{ f_\Theta(0) \} \right) \right) \right\} \right| \right)$$

where $f_\Theta: \mathbb{R} \to \mathbb{C}^{MM \times NN}$ is the neural implicit representation parameterized by $\Theta$.

### data_fidelity_term

$$\mathcal{L}_{\text{SmoothL1}}(y, \hat{y}) = \begin{cases} 0.5(y - \hat{y})^2 / \beta & \text{if } |y - \hat{y}| < \beta \\ |y - \hat{y}| - 0.5\beta & \text{otherwise} \end{cases}$$

with $\beta = 1.0$ (PyTorch default), applied element-wise and averaged over all elements.

### regularization_term

None (implicit regularization through neural network parameterization)

### constraints

None (unconstrained optimization over neural network parameters)

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{D}_{\text{real}}^{xy}$ | $(x_{\text{mode}}, y_{\text{mode}}, n_f)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4})$ | XY feature tensor for real/amplitude network |
| $\mathbf{D}_{\text{real}}^{z}$ | $(z_{\text{dim}}, n_f)$ | float32 | $\mathcal{N}(0, 1)$ | Z feature vector for real/amplitude network |
| $\mathbf{W}_{\text{real}}^{(l)}, \mathbf{b}_{\text{real}}^{(l)}$ | varies | float32 | PyTorch default | MLP weights/biases for real renderer |
| $\mathbf{D}_{\text{imag}}^{xy}$ | $(x_{\text{mode}}, y_{\text{mode}}, n_f)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4})$ | XY feature tensor for imaginary/phase network |
| $\mathbf{D}_{\text{imag}}^{z}$ | $(z_{\text{dim}}, n_f)$ | float32 | $\mathcal{N}(0, 1)$ | Z feature vector for imaginary/phase network |
| $\mathbf{W}_{\text{imag}}^{(l)}, \mathbf{b}_{\text{imag}}^{(l)}$ | varies | float32 | PyTorch default | MLP weights/biases for imaginary renderer |

### 2.2 Dual Variables

None

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}_0, \mathbf{y}_0$ | $(MM \cdot NN,)$ | int64 | Computed from coords | Lower interpolation indices for XY grid |
| $\mathbf{x}_1, \mathbf{y}_1$ | $(MM \cdot NN,)$ | int64 | Computed from coords | Upper interpolation indices for XY grid |
| $\mathbf{w}_{\text{lerp}}$ | $(MM \cdot NN, 2)$ | float32 | Computed from coords | Bilinear interpolation weights |
| $z_0, z_1$ | $(B,)$ | int64 | Computed per forward | Z interpolation indices |
| $\mathbf{w}_z$ | $(B, 1)$ | float32 | Computed per forward | Z interpolation weights |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\lambda$ | 0.632 μm (red), 0.5126 μm (green), 0.471 μm (blue) | Wavelength |
| $k_0$ | $2\pi / \lambda$ | Wavenumber |
| MAGimg | 3 | High-res magnification factor |
| $n_f$ | 32 | Number of features per coordinate |
| $z_{\text{dim}}$ | 5 | Number of depth levels |
| $z_{\min}$ | 0.0 | Minimum defocus parameter |
| $z_{\max}$ | 1.0 | Maximum defocus parameter |
| $x_{\text{mode}}, y_{\text{mode}}$ | 512 | Feature grid resolution |
| $\eta$ | $10^{-3}$ | Initial learning rate |
| $\gamma$ | 0.1 | Learning rate decay factor |
| lr_decay_step | 6 | Epochs between LR decay |
| led_batch_size | 1 | LEDs processed per iteration |
| ds_factor | 1 | Downsampling factor |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{I}$ | $(M, N, T)$ | float32 | Measured intensity images (normalized) |
| $\mathbf{P}_0$ | $(1, 1, M, N)$ | float32 | Binary pupil support mask |
| $k_{zz}$ | $(1, M, N)$ | complex64 | Angular spectrum propagation kernel |
| ledpos_true | $(T, 2)$ | int64 | LED positions in frequency space |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load Raw Data | $\mathbf{I}_{\text{raw}} = \text{load}(\text{path})$ | File path | $\mathbf{I}_{\text{raw}} \in \mathbb{R}^{M_0 \times N_0 \times T_0}$ |
| 2 | Crop to num_modes | $\mathbf{I} = \mathbf{I}_{\text{raw}}[0:M, 0:N, :]$ where $M = N = \text{num\_modes}$ | $\mathbf{I}_{\text{raw}}$ | $\mathbf{I} \in \mathbb{R}^{M \times N \times T}$ |
| 3 | Compute Wavenumber | $k_0 = \frac{2\pi}{\lambda}$ | $\lambda$ | $k_0 \in \mathbb{R}$ |
| 4 | Compute High-Res Dimensions | $MM = M \cdot \text{MAGimg}, \quad NN = N \cdot \text{MAGimg}$ | $M, N, \text{MAGimg}$ | $MM, NN \in \mathbb{Z}$ |
| 5 | Compute Frequency Axes (High-Res) | $F_{xx1}[n] = \frac{(n - NN/2)}{N \cdot D_{\text{pixel}}} \cdot 2\pi, \quad n \in [0, NN)$ | $NN, N, D_{\text{pixel}}$ | $F_{xx1} \in \mathbb{R}^{NN}$ |
| 6 | Compute LED NA Order | $\text{NA}_{\text{illu}}[t] = \sqrt{u_t^2 + v_t^2}$; $\text{order} = \text{argsort}(\text{NA}_{\text{illu}})$ | NAx, NAy | order $\in \mathbb{Z}^T$ |
| 7 | Reorder LED Positions | $u = -\text{NAx}[\text{order}], \quad v = -\text{NAy}[\text{order}]$ | NAx, NAy, order | $u, v \in \mathbb{R}^T$ |
| 8 | Compute LED Frequency Positions | $\text{ledpos}[t, 0] = \arg\min_n |F_{xx1}[n] - k_0 u_t|$, $\text{ledpos}[t, 1] = \arg\min_m |F_{yy1}[m] - k_0 v_t|$ | $F_{xx1}, F_{yy1}, u, v, k_0$ | ledpos $\in \mathbb{Z}^{T \times 2}$ |
| 9 | Normalize and Reorder Intensities | $\mathbf{I}_{\text{sum}} = \frac{\mathbf{I}[:,:,\text{order}]}{\max(\mathbf{I})}$ | $\mathbf{I}$, order | $\mathbf{I}_{\text{sum}} \in [0,1]^{M \times N \times T}$ |
| 10 | Compute Angular Spectrum Grid | $k_{xx}[m,n] = \frac{(m - M/2)\pi}{M \cdot D_{\text{pixel}}/2}, \quad k_{rr} = \sqrt{k_{xx}^2 + k_{yy}^2}$ | $M, N, D_{\text{pixel}}$ | $k_{rr} \in \mathbb{R}^{M \times N}$ |
| 11 | Compute Propagation Kernel | $k_{zz}[m,n] = \begin{cases} \sqrt{k_0^2 - k_{rr}^2} & k_0^2 > k_{rr}^2 \\ 0 & \text{otherwise} \end{cases}$ | $k_0, k_{rr}$ | $k_{zz} \in \mathbb{C}^{1 \times M \times N}$ |
| 12 | Compute Pupil Support | $P_0[m,n] = \mathbb{1}\left[ F_{x2}[m,n] + F_{y2}[m,n] \leq k_{\max}^2 \right]$ where $k_{\max} = \text{NA} \cdot k_0$ | $k_{\max}, M, N, D_{\text{pixel}}$ | $\mathbf{P}_0 \in \{0,1\}^{1 \times 1 \times M \times N}$ |
| 13 | Create XY Coordinate Grid | $\mathbf{xy}[i] = \left(\frac{i \mod NN + 0.5}{NN}, \frac{\lfloor i/NN \rfloor + 0.5}{MM}\right) \cdot (x_{\text{mode}}, y_{\text{mode}})$ | $MM, NN, x_{\text{mode}}, y_{\text{mode}}$ | $\mathbf{xy} \in \mathbb{R}^{MM \cdot NN \times 2}$ |
| 14 | Compute Interpolation Indices | $\mathbf{x}_0 = \lfloor \mathbf{xy}[:,0] \rfloor.\text{clamp}(0, x_{\text{mode}}-1)$, $\mathbf{x}_1 = (\mathbf{x}_0 + 1).\text{clamp}(0, x_{\text{mode}}-1)$ | $\mathbf{xy}$ | $\mathbf{x}_0, \mathbf{x}_1, \mathbf{y}_0, \mathbf{y}_1$ |
| 15 | Compute Interpolation Weights | $\mathbf{w}_{\text{lerp}} = \mathbf{xy} - \lfloor \mathbf{xy} \rfloor$ | $\mathbf{xy}$ | $\mathbf{w}_{\text{lerp}} \in [0,1)^{MM \cdot NN \times 2}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Initialize XY Feature Tensors | $\mathbf{D}_{\text{real}}^{xy}, \mathbf{D}_{\text{imag}}^{xy} \sim 2 \times 10^{-4} \cdot \mathcal{N}(0,1)$ | $x_{\text{mode}}, y_{\text{mode}}, n_f$ | $\mathbf{D}_{\text{real}}^{xy}, \mathbf{D}_{\text{imag}}^{xy} \in \mathbb{R}^{x_{\text{mode}} \times y_{\text{mode}} \times n_f}$ |
| 2 | Initialize Z Feature Vectors | $\mathbf{D}_{\text{real}}^{z}, \mathbf{D}_{\text{imag}}^{z} \sim \mathcal{N}(0,1)$ | $z_{\text{dim}}, n_f$ | $\mathbf{D}_{\text{real}}^{z}, \mathbf{D}_{\text{imag}}^{z} \in \mathbb{R}^{z_{\text{dim}} \times n_f}$ |
| 3 | Initialize MLP Renderers | Architecture: Linear$(n_f, 32) \to$ ReLU $\to$ Linear$(32, 32) \to$ ReLU $\to$ Linear$(32, 1)$ | $n_f$ | $\{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}$ for real and imag |
| 4 | Initialize Adam Optimizer | $\mathbf{m}_0 = \mathbf{0}, \quad \mathbf{v}_0 = \mathbf{0}, \quad t = 0$ | All trainable $\Theta$ | Optimizer state |
| 5 | Set Defocus Parameter | $dz = 0.0$ | - | $dz \in \mathbb{R}$ |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (epoch $e = 0, \ldots, E-1$ where $E = \text{num\_epochs}$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 6 | Generate LED Index List | $\mathcal{L} = [0, 1, \ldots, T-1]$ | $T$ | $\mathcal{L}$ | Sequential order |
| 7 | Normalize Z Coordinate | $\tilde{z} = (z_{\text{dim}} - 1) \cdot \frac{dz - z_{\min}}{z_{\max} - z_{\min}}$ | $dz, z_{\min}, z_{\max}, z_{\text{dim}}$ | $\tilde{z} = 0.0$ | For $dz=0$ |
| 8 | Compute Z Interpolation Indices | $z_0 = \lfloor \tilde{z} \rfloor.\text{clamp}(0, z_{\text{dim}}-1)$, $z_1 = (z_0+1).\text{clamp}(0, z_{\text{dim}}-1)$ | $\tilde{z}, z_{\text{dim}}$ | $z_0, z_1$ | |
| 9 | Compute Z Interpolation Weight | $w_z = \tilde{z} - \lfloor \tilde{z} \rfloor$ | $\tilde{z}$ | $w_z \in [0,1)$ | |

**BEGIN_LOOP** (LED iteration $\ell = 0, \ldots, T-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 10 | Zero Gradients | $\nabla_\Theta \mathcal{L} = \mathbf{0}$ | - | Cleared gradients | |
| 11 | Compute Defocus Mask | $H_{\text{df}} = \exp(i \cdot k_{zz} \cdot dz) \in \mathbb{C}^{1 \times 1 \times M \times N}$ | $k_{zz}, dz$ | $H_{\text{df}}$ | [FORWARD] |
| 12 | Compute Spectrum Mask | $\mathbf{S} = \mathbf{P}_0 \odot H_{\text{df}}$ | $\mathbf{P}_0, H_{\text{df}}$ | $\mathbf{S} \in \mathbb{C}^{1 \times 1 \times M \times N}$ | |
| 13 | Bilinear XY Feature Sampling (Real) | $\mathbf{F}_{xy}^{\text{real}}[i] = \mathbf{D}^{xy}_{\text{real}}[\mathbf{y}_0[i], \mathbf{x}_0[i]] (1-w_x)(1-w_y) + \mathbf{D}^{xy}_{\text{real}}[\mathbf{y}_0[i], \mathbf{x}_1[i]] w_x(1-w_y) + \mathbf{D}^{xy}_{\text{real}}[\mathbf{y}_1[i], \mathbf{x}_0[i]] (1-w_x)w_y + \mathbf{D}^{xy}_{\text{real}}[\mathbf{y}_1[i], \mathbf{x}_1[i]] w_x w_y$ | $\mathbf{D}^{xy}_{\text{real}}, \mathbf{x}_0, \mathbf{x}_1, \mathbf{y}_0, \mathbf{y}_1, \mathbf{w}_{\text{lerp}}$ | $\mathbf{F}_{xy}^{\text{real}} \in \mathbb{R}^{MM \cdot NN \times n_f}$ | $w_x = \mathbf{w}_{\text{lerp}}[:,0]$, $w_y = \mathbf{w}_{\text{lerp}}[:,1]$ |
| 14 | Linear Z Feature Interpolation (Real) | $\mathbf{F}_z^{\text{real}} = \mathbf{D}_{\text{real}}^z[z_0] (1-w_z) + \mathbf{D}_{\text{real}}^z[z_1] w_z$ | $\mathbf{D}_{\text{real}}^z, z_0, z_1, w_z$ | $\mathbf{F}_z^{\text{real}} \in \mathbb{R}^{1 \times n_f}$ | |
| 15 | Feature Multiplication (Real) | $\mathbf{F}^{\text{real}} = \mathbf{F}_{xy}^{\text{real}} \odot \mathbf{F}_z^{\text{real}}$ | $\mathbf{F}_{xy}^{\text{real}}, \mathbf{F}_z^{\text{real}}$ | $\mathbf{F}^{\text{real}} \in \mathbb{R}^{1 \times MM \cdot NN \times n_f}$ | Broadcast multiplication |
| 16 | MLP Forward Pass (Real) | $\mathbf{a} = \mathbf{W}^{(3)} \cdot \text{ReLU}(\mathbf{W}^{(2)} \cdot \text{ReLU}(\mathbf{W}^{(1)} \mathbf{F}^{\text{real}} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)}) + \mathbf{b}^{(3)}$ | $\mathbf{F}^{\text{real}}, \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}_{\text{real}}$ | $\mathbf{a} \in \mathbb{R}^{1 \times 1 \times MM \times NN}$ | Amplitude output |
| 17 | Bilinear XY Feature Sampling (Imag) | Same as Step 13 with $\mathbf{D}^{xy}_{\text{imag}}$ | $\mathbf{D}^{xy}_{\text{imag}}, \ldots$ | $\mathbf{F}_{xy}^{\text{imag}} \in \mathbb{R}^{MM \cdot NN \times n_f}$ | |
| 18 | Linear Z Feature Interpolation (Imag) | Same as Step 14 with $\mathbf{D}_{\text{imag}}^z$ | $\mathbf{D}_{\text{imag}}^z, z_0, z_1, w_z$ | $\mathbf{F}_z^{\text{imag}} \in \mathbb{R}^{1 \times n_f}$ | |
| 19 | Feature Multiplication (Imag) | $\mathbf{F}^{\text{imag}} = \mathbf{F}_{xy}^{\text{imag}} \odot \mathbf{F}_z^{\text{imag}}$ | $\mathbf{F}_{xy}^{\text{imag}}, \mathbf{F}_z^{\text{imag}}$ | $\mathbf{F}^{\text{imag}} \in \mathbb{R}^{1 \times MM \cdot NN \times n_f}$ | |
| 20 | MLP Forward Pass (Imag) | Same MLP structure as Step 16 with imag weights | $\mathbf{F}^{\text{imag}}, \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}_{\text{imag}}$ | $\boldsymbol{\phi} \in \mathbb{R}^{1 \times 1 \times MM \times NN}$ | Phase output |
| 21 | Upsample (if ds_factor > 1) | $\mathbf{a}, \boldsymbol{\phi} = \text{Upsample}(\mathbf{a}, \boldsymbol{\phi})$ | $\mathbf{a}, \boldsymbol{\phi}$ | $\mathbf{a}, \boldsymbol{\phi} \in \mathbb{R}^{1 \times MM \times NN}$ | Bilinear; identity when ds_factor=1 |
| 22 | Form Complex Image | $\mathbf{O}_{\text{img}} = \mathbf{a} \odot \exp(i \boldsymbol{\phi})$ | $\mathbf{a}, \boldsymbol{\phi}$ | $\mathbf{O}_{\text{img}} \in \mathbb{C}^{1 \times MM \times NN}$ | |
| 23 | Compute High-Res Spectrum | $\mathbf{O} = \text{fftshift}(\text{FFT2}(\mathbf{O}_{\text{img}}))$ | $\mathbf{O}_{\text{img}}$ | $\mathbf{O} \in \mathbb{C}^{1 \times MM \times NN}$ | [FORWARD] |
| 24 | Zero-Pad Spectrum | $\mathbf{O}_{\text{pad}} = \text{Pad}(\mathbf{O}, p_x, p_y)$ where $p_x = p_y = \frac{M \cdot \text{MAGimg} \cdot \text{MAGimg} - MM}{2}$ | $\mathbf{O}, M, \text{MAGimg}$ | $\mathbf{O}_{\text{pad}} \in \mathbb{C}^{1 \times MM \cdot \text{MAGimg} \times NN \cdot \text{MAGimg}}$ | Padding depends on mag |
| 25 | Extract LED Position | $(u_\ell, v_\ell) = \text{ledpos}[\ell]$ | ledpos, $\ell$ | $u_\ell, v_\ell \in \mathbb{Z}$ | |
| 26 | Compute Crop Bounds | $x_0 = v_\ell - M/2, \; x_1 = v_\ell + M/2, \; y_0 = u_\ell - N/2, \; y_1 = u_\ell + N/2$ | $u_\ell, v_\ell, M, N$ | $x_0, x_1, y_0, y_1$ | |
| 27 | Extract Sub-Aperture | $\mathbf{O}_{\text{sub}} = \mathbf{O}_{\text{pad}}[:, x_0:x_1, y_0:y_1]$ | $\mathbf{O}_{\text{pad}}, x_0, x_1, y_0, y_1$ | $\mathbf{O}_{\text{sub}} \in \mathbb{C}^{1 \times 1 \times M \times N}$ | |
| 28 | Apply Spectrum Mask | $\mathbf{O}_{\text{masked}} = \mathbf{O}_{\text{sub}} \odot \mathbf{S}$ | $\mathbf{O}_{\text{sub}}, \mathbf{S}$ | $\mathbf{O}_{\text{masked}} \in \mathbb{C}^{1 \times 1 \times M \times N}$ | |
| 29 | Inverse FFT | $\mathbf{o}_{\text{sub}} = \text{IFFT2}(\text{ifftshift}(\mathbf{O}_{\text{masked}}))$ | $\mathbf{O}_{\text{masked}}$ | $\mathbf{o}_{\text{sub}} \in \mathbb{C}^{1 \times 1 \times M \times N}$ | [FORWARD] |
| 30 | Compute Predicted Intensity | $\hat{\mathbf{I}}_\ell = |\mathbf{o}_{\text{sub}}|$ | $\mathbf{o}_{\text{sub}}$ | $\hat{\mathbf{I}}_\ell \in \mathbb{R}^{1 \times 1 \times M \times N}$ | |
| 31 | Extract Measured Amplitude | $\mathbf{I}_\ell^{\text{cap}} = \sqrt{\mathbf{I}_{\text{sum}}[:,:,\ell]}$ | $\mathbf{I}_{\text{sum}}, \ell$ | $\mathbf{I}_\ell^{\text{cap}} \in \mathbb{R}^{1 \times 1 \times M \times N}$ | |
| 32 | Compute Smooth L1 Loss | $\mathcal{L} = \frac{1}{MN}\sum_{m,n} \text{SmoothL1}(\mathbf{I}_\ell^{\text{cap}}[m,n], \hat{\mathbf{I}}_\ell[m,n])$ | $\mathbf{I}_\ell^{\text{cap}}, \hat{\mathbf{I}}_\ell$ | $\mathcal{L} \in \mathbb{R}$ | |
| 33 | Compute MSE (for PSNR) | $\mathcal{L}_{\text{MSE}} = \frac{1}{MN}\sum_{m,n} (\mathbf{I}_\ell^{\text{cap}}[m,n] - \hat{\mathbf{I}}_\ell[m,n])^2$ | $\mathbf{I}_\ell^{\text{cap}}, \hat{\mathbf{I}}_\ell$ | $\mathcal{L}_{\text{MSE}} \in \mathbb{R}$ | |
| 34 | Backpropagation | $\nabla_\Theta \mathcal{L} = \text{autograd}(\mathcal{L}, \Theta)$ | $\mathcal{L}, \Theta$ | $\nabla_\Theta \mathcal{L}$ | [ADJOINT] |
| 35 | Adam Update | $t \gets t+1$; $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla_\Theta$; $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\nabla_\Theta^2$; $\hat{\mathbf{m}}_t = \mathbf{m}_t/(1-\beta_1^t)$; $\hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_2^t)$; $\Theta \gets \Theta - \eta \hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$ | $\nabla_\Theta, \mathbf{m}_{t-1}, \mathbf{v}_{t-1}, \eta, \beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$ | $\Theta, \mathb