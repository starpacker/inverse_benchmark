# Algorithmic Specification Document: Fourier Ptychographic Microscopy via Implicit Neural Representations (FPM-INR)

---

## 1. Mathematical Formulation

### 1.1 Problem Statement
Reconstruct a high-resolution complex-valued image $O \in \mathbb{C}^{MM \times NN}$ from a set of $L$ low-resolution intensity measurements $\{I^{(l)}\}_{l=1}^{L} \in \mathbb{R}^{M \times N}$, where each measurement corresponds to a different LED illumination angle.

### 1.2 Complete Objective Function

$$
\min_{\Theta} \mathcal{L}(\Theta) = \frac{1}{L} \sum_{l=1}^{L} \mathcal{L}_{\text{SmoothL1}}\left( \sqrt{I_{\text{obs}}^{(l)}}, \; \left| \hat{o}^{(l)}(\Theta) \right| \right)
$$

where the Smooth L1 loss is defined as:
$$
\mathcal{L}_{\text{SmoothL1}}(x, y) = \begin{cases}
0.5(x-y)^2 & \text{if } |x-y| < 1 \\
|x-y| - 0.5 & \text{otherwise}
\end{cases}
$$

### 1.3 Decomposition

| Term | Formula | Description |
|------|---------|-------------|
| **Data Fidelity** | $\sum_{l=1}^{L} \mathcal{L}_{\text{SmoothL1}}\left(\sqrt{I_{\text{obs}}^{(l)}}, \|\hat{o}^{(l)}\|\right)$ | Amplitude matching in sub-aperture space |
| **Implicit Regularization** | Neural network architecture constraints | Spectral bias from INR representation |

### 1.4 Neural Representation of Complex Image

The complex image is parameterized as:
$$
O(x, y; \Theta) = A(x, y; \Theta_A) \cdot \exp\left(i \cdot \Phi(x, y; \Theta_\Phi)\right)
$$

where $A$ (amplitude) and $\Phi$ (phase) are each represented by a factorized neural field:

$$
A(x, y) = f_{\theta_r}\left( \mathbf{F}_{xy}^{(r)} \odot \mathbf{F}_z^{(r)} \right), \quad \Phi(x, y) = f_{\theta_i}\left( \mathbf{F}_{xy}^{(i)} \odot \mathbf{F}_z^{(i)} \right)
$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables ($\Theta$)

| Variable | Symbol | Shape | Dtype | Initialization |
|----------|--------|-------|-------|----------------|
| Spatial feature tensor (real) | $\mathbf{F}^{(r)} \in \mathbb{R}^{X_m \times Y_m \times C}$ | `(x_mode, y_mode, num_feats)` | `float32` | $\mathcal{U}(-10^{-4}, 10^{-4})$ |
| Spatial feature tensor (imag) | $\mathbf{F}^{(i)} \in \mathbb{R}^{X_m \times Y_m \times C}$ | `(x_mode, y_mode, num_feats)` | `float32` | $\mathcal{U}(-10^{-4}, 10^{-4})$ |
| Z-axis features (real) | $\mathbf{F}_z^{(r)} \in \mathbb{R}^{Z_m \times C}$ | `(z_dim, num_feats)` | `float32` | $\mathcal{N}(0, 1)$ |
| Z-axis features (imag) | $\mathbf{F}_z^{(i)} \in \mathbb{R}^{Z_m \times C}$ | `(z_dim, num_feats)` | `float32` | $\mathcal{N}(0, 1)$ |
| Renderer MLP weights (real) | $\theta_r = \{W_k^{(r)}, b_k^{(r)}\}_{k=1}^{K}$ | Layer-dependent | `float32` | PyTorch default |
| Renderer MLP weights (imag) | $\theta_i = \{W_k^{(i)}, b_k^{(i)}\}_{k=1}^{K}$ | Layer-dependent | `float32` | PyTorch default |

### 2.2 Auxiliary/Intermediate Variables

| Variable | Symbol | Shape | Description |
|----------|--------|-------|-------------|
| Bilinear interpolation weights | $\mathbf{w}_{\text{lerp}} \in [0,1]^{N_{pix} \times 2}$ | `(x_dim*y_dim, 2)` | Fractional pixel coordinates |
| Grid corner indices | $(x_0, y_0, x_1, y_1)$ | `(x_dim*y_dim,)` each | Integer indices for corners |
| Defocus mask | $D(k_x, k_y; dz)$ | `(1, L_b, M, N)` | $\exp(i k_z \cdot dz)$ |
| Spectrum mask | $P_{\text{eff}}$ | `(1, L_b, M, N)` | $P_0 \odot D$ |
| Sub-aperture spectrum | $\tilde{O}_{\text{sub}}^{(l)}$ | `(B, L_b, M, N)` | Extracted spectral region |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Number of epochs | $N_{\text{epoch}}$ | 5 | Total optimization iterations |
| Feature dimension | $C$ | 32 | INR latent feature size |
| Spatial modes | $X_m = Y_m$ | 512 | Feature grid resolution |
| Z-axis modes | $Z_m$ | 5 | Depth discretization |
| Depth range | $[z_{\min}, z_{\max}]$ | $[0.0, 1.0]$ | Normalized depth bounds |
| Learning rate | $\eta$ | $10^{-3}$ | Adam optimizer step size |
| LR decay step | $T_{\text{decay}}$ | 6 | Epochs before decay |
| LR decay factor | $\gamma$ | 0.1 | Multiplicative decay |
| Downsampling factor | $ds$ | 1 | Spatial downsampling |
| LED batch size | $L_b$ | 1 | LEDs per mini-batch |
| MLP hidden dim | $H$ | 32 | Renderer hidden units |
| MLP layers | $K$ | 2 | Renderer depth |
| Magnification | $MAG$ | 3 | Resolution enhancement factor |

### 2.4 Observed Data/Inputs

| Data | Symbol | Shape | Description |
|------|--------|-------|-------------|
| Low-res intensity images | $\mathbf{I} \in \mathbb{R}_+^{M \times N \times L}$ | `(M, N, ID_len)` | Raw measurements |
| LED illumination NAs | $(u_l, v_l)_{l=1}^{L}$ | `(ID_len, 2)` | Normalized angles |
| LED frequency positions | $(p_x^{(l)}, p_y^{(l)})$ | `(ID_len, 2)` | Pixel indices in spectrum |
| Pupil support | $P_0 \in \{0,1\}^{M \times N}$ | `(1, 1, M, N)` | Circular aperture mask |
| Angular spectrum kernel | $k_z \in \mathbb{C}^{M \times N}$ | `(1, M, N)` | Propagation phase |
| Wavelength | $\lambda$ | scalar | 0.632 (r), 0.5126 (g), 0.471 (b) μm |
| Numerical aperture | $NA$ | scalar | From calibration data |

---

## 3. Preprocessing Pipeline

| Step | Operation | Input | Output | Formula |
|------|-----------|-------|--------|---------|
| 1 | **Load Data** | `.mat` file | $\mathbf{I}_{\text{raw}}, \mathbf{NA}$ | File I/O |
| 2 | **Crop Modes** | $\mathbf{I}_{\text{raw}}$ | $\mathbf{I} \in \mathbb{R}^{M \times N \times L}$ | $\mathbf{I} = \mathbf{I}_{\text{raw}}[0:n_m, 0:n_m, :]$ |
| 3 | **Set Wavenumber** | $\lambda$ | $k_0$ | $k_0 = \frac{2\pi}{\lambda}$ |
| 4 | **Compute HR Dimensions** | $M, N, MAG$ | $MM, NN$ | $MM = M \cdot MAG, \; NN = N \cdot MAG$ |
| 5 | **Build Frequency Grid** | $M, N, D_{\text{pixel}}$ | $F_{xx}, F_{yy}$ | $F_{xx}[j] = \frac{j - N/2}{N \cdot D_{\text{pixel}}} \cdot 2\pi$ |
| 6 | **Sort LEDs by NA** | $(u, v)$ | ordered indices | $\text{order} = \text{argsort}\left(\sqrt{u^2 + v^2}\right)$ |
| 7 | **Compute LED Positions** | $F_{xx}, F_{yy}, u_l, v_l, k_0$ | $(p_x^{(l)}, p_y^{(l)})$ | $p_x^{(l)} = \arg\min_j |F_{xx}[j] - k_0 u_l|$ |
| 8 | **Normalize Intensities** | $\mathbf{I}$ | $\mathbf{I}_{\text{sum}}$ | $\mathbf{I}_{\text{sum}} = \frac{\mathbf{I}[\text{order}]}{\max(\mathbf{I})}$ |
| 9 | **Compute $k_z$ Kernel** | $k_{xx}, k_{yy}, k_0$ | $k_z \in \mathbb{C}^{M \times N}$ | $k_z = \sqrt{k_0^2 - k_{xx}^2 - k_{yy}^2}$ (complex) |
| 10 | **Build Pupil Mask** | $F_x, F_y, k_{\max}$ | $P_0$ | $P_0[i,j] = \mathbb{1}\left[F_x[i]^2 + F_y[j]^2 \leq k_{\max}^2\right]$ |
| 11 | **Transfer to Device** | CPU tensors | GPU tensors | `.to(device)` |

---

## 4. Algorithm Steps

### 4.1 Initialization Phase

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| I.1 | **Initialize Spatial Features** | $\mathbf{F}^{(r,i)}_{x,y,c} \sim \mathcal{U}(-10^{-4}, 10^{-4})$ | $(X_m, Y_m, C)$ | $\mathbf{F}^{(r)}, \mathbf{F}^{(i)}$ |
| I.2 | **Initialize Z Features** | $\mathbf{F}_z^{(r,i)}_{z,c} \sim \mathcal{N}(0, 1)$ | $(Z_m, C)$ | $\mathbf{F}_z^{(r)}, \mathbf{F}_z^{(i)}$ |
| I.3 | **Build Coordinate Grid** | $\mathbf{xy}_{n} = \left(\frac{n_x + 0.5}{X}, \frac{n_y + 0.5}{Y}\right)$ | $(X, Y)$ dims | $\mathbf{xy} \in [0,1]^{XY \times 2}$ |
| I.4 | **Compute Interpolation Indices** | $\mathbf{s} = \mathbf{xy} \cdot (X_m, Y_m)^T$ | $\mathbf{xy}$ | $x_0, y_0, x_1, y_1, \mathbf{w}_{\text{lerp}}$ |
| I.5 | **Initialize Optimizer** | Adam($\Theta$, $\eta=10^{-3}$) | $\Theta$ | optimizer state |
| I.6 | **Initialize Scheduler** | StepLR($T=6$, $\gamma=0.1$) | optimizer | scheduler state |

### 4.2 Main Iterative Procedure

**For** $e = 1, \ldots, N_{\text{epoch}}$:

&emsp;**For** $l = 1, \ldots, L$ (LED index):

| Step | Name | Mathematical Formula | Input Variables | Output Variables |
|------|------|---------------------|-----------------|------------------|
| M.1 | **Compute Defocus Mask** | $D = \exp\left(i \cdot k_z \cdot dz\right)$ | $k_z$, $dz$ | $D \in \mathbb{C}^{M \times N}$ |
| M.2 | **Construct Spectrum Mask** | $P_{\text{eff}} = P_0 \odot D$ | $P_0$, $D$ | $P_{\text{eff}} \in \mathbb{C}^{M \times N}$ |
| M.3 | **Bilinear Sample XY Features** | $\mathbf{F}_{xy} = \sum_{(i,j) \in \{0,1\}^2} w_{ij} \cdot \mathbf{F}[y_i, x_j]$ | $\mathbf{F}$, indices, weights | $\mathbf{F}_{xy} \in \mathbb{R}^{N_{pix} \times C}$ |
| M.4 | **Normalize Z Coordinate** | $\tilde{z} = (Z_m - 1) \cdot \frac{dz - z_{\min}}{z_{\max} - z_{\min}}$ | $dz$, bounds | $\tilde{z} \in [0, Z_m-1]$ |
| M.5 | **Linear Interpolate Z Features** | $\mathbf{F}_z = (1-w_z)\mathbf{F}_z[z_0] + w_z \mathbf{F}_z[z_1]$ | $\mathbf{F}_z$, $\tilde{z}$ | $\mathbf{F}_z \in \mathbb{R}^{1 \times C}$ |
| M.6 | **Combine Features** | $\mathbf{F}_{\text{comb}} = \mathbf{F}_{xy} \odot \mathbf{F}_z$ | $\mathbf{F}_{xy}$, $\mathbf{F}_z$ | $\mathbf{F}_{\text{comb}} \in \mathbb{R}^{N_{pix} \times C}$ |
| M.7 | **Render Amplitude** | $A = f_{\theta_r}(\mathbf{F}_{\text{comb}}^{(r)})$ | $\mathbf{F}_{\text{comb}}^{(r)}$, $\theta_r$ | $A \in \mathbb{R}^{MM \times NN}$ |
| M.8 | **Render Phase** | $\Phi = f_{\theta_i}(\mathbf{F}_{\text{comb}}^{(i)})$ | $\mathbf{F}_{\text{comb}}^{(i)}$, $\theta_i$ | $\Phi \in \mathbb{R}^{MM \times NN}$ |
| M.9 | **Form Complex Image** | $O = A \cdot \exp(i\Phi)$ | $A$, $\Phi$ | $O \in \mathbb{C}^{MM \times NN}$ |
| M.10 | **Compute Spectrum (FFT)** | $\tilde{O} = \text{fftshift}(\text{fft2}(O))$ | $O$ | $\tilde{O} \in \mathbb{C}^{MM \times NN}$ |
| M.11 | **Zero-Pad Spectrum** | $\tilde{O}_{\text{pad}} = \text{pad}(\tilde{O}, p)$ where $p = \frac{MAG \cdot M - MM}{2}$ | $\tilde{O}$ | $\tilde{O}_{\text{pad}} \in \mathbb{C}^{MAG \cdot M \times MAG \cdot N}$ |
| M.12 | **Extract Sub-Aperture** | $\tilde{O}_{\text{sub}}^{(l)} = \tilde{O}_{\text{pad}}\left[p_y^{(l)} - \frac{M}{2} : p_y^{(l)} + \frac{M}{2}, \; p_x^{(l)} - \frac{N}{2} : p_x^{(l)} + \frac{N}{2}\right]$ | $\tilde{O}_{\text{pad}}$, $(p_x^{(l)}, p_y^{(l)})$ | $\tilde{O}_{\text{sub}}^{(l)} \in \mathbb{C}^{M \times N}$ |
| M.13 | **Apply Spectrum Mask** | $\tilde{O}_{\text{masked}}^{(l)} = \tilde{O}_{\text{sub}}^{(l)} \odot P_{\text{eff}}$ | $\tilde{O}_{\text{sub}}^{(l)}$, $P_{\text{eff}}$ | $\tilde{O}_{\text{masked}}^{(l)} \in \mathbb{C}^{M \times N}$ |
| M.14 | **Inverse FFT** | $\hat{o}^{(l)} = \text{ifft2}(\text{ifftshift}(\tilde{O}_{\text{masked}}^{(l)}))$ | $\tilde{O}_{\text{masked}}^{(l)}$ | $\hat{o}^{(l)} \in \mathbb{C}^{M \times N}$ |
| M.15 | **Compute Predicted Amplitude** | $\hat{I}_{\text{amp}}^{(l)} = |\hat{o}^{(l)}|$ | $\hat{o}^{(l)}$ | $\hat{I}_{\text{amp}}^{(l)} \in \mathbb{R}_+^{M \times N}$ |
| M.16 | **Compute Observed Amplitude** | $I_{\text{amp}}^{(l)} = \sqrt{I_{\text{sum}}[:,:,l]}$ | $I_{\text{sum}}$ | $I_{\text{amp}}^{(l)} \in \mathbb{R}_+^{M \times N}$ |
| M.17 | **Compute Loss** | $\mathcal{L} = \text{SmoothL1}(I_{\text{amp}}^{(l)}, \hat{I}_{\text{amp}}^{(l)})$ | $I_{\text{amp}}^{(l)}$, $\hat{I}_{\text{amp}}^{(l)}$ | $\mathcal{L} \in \mathbb{R}$ |
| M.18 | **Backpropagation** | $\nabla_\Theta \mathcal{L}$ via autograd | $\mathcal{L}$, $\Theta$ | gradients |
| M.19 | **Parameter Update** | $\Theta \leftarrow \text{Adam}(\Theta, \nabla_\Theta \mathcal{L}, \eta)$ | $\Theta$, gradients | updated $\Theta$ |

&emsp;**End For** (LED)

&emsp;| M.20 | **Learning Rate Decay** | $\eta \leftarrow \gamma \cdot \eta$ if $e \mod T_{\text{decay}} = 0$ | $\eta$, $e$ | updated $\eta$ |

**End For** (epoch)

### 4.3 Bilinear Interpolation Detail (Steps M.3)

$$
\mathbf{F}_{xy}[n] = \mathbf{F}[y_0[n], x_0[n]] \cdot (1 - w_x[n])(1 - w_y[n]) + \mathbf{F}[y_0[n], x_1[n]] \cdot w_x[n](1 - w_y[n]) + \mathbf{F}[y_1[n], x_0[n]] \cdot (1 - w_x[n])w_y[n] + \mathbf{F}[y_1[n], x_1[n]] \cdot w_x[n] \cdot w_y[n]
$$

where $w_x = \mathbf{w}_{\text{lerp}}[:, 0]$, $w_y = \mathbf{w}_{\text{lerp}}[:, 1]$.

### 4.4 MLP Renderer Detail (Steps M.7, M.8)

For $k = 1, \ldots, K$:
$$
\mathbf{h}^{(0)} = \mathbf{F}_{\text{comb}}
$$
$$
\mathbf{h}^{(k)} = \text{ReLU}\left(W_k \mathbf{h}^{(k-1)} + b_k\right)
$$
$$
\text{output} = W_{K+1} \mathbf{h}^{(K)} + b_{K+1}
$$

---

## 5. Stopping Criterion

| Criterion | Condition | Value |
|-----------|-----------|-------|
| **Primary** | Maximum epochs reached | $e = N_{\text{epoch}} = 5$ |
| **Implicit** | All LED batches processed per epoch | $l = L$ (full pass) |

*Note: No explicit convergence threshold on loss or gradient norm is implemented.*

---

## 6. Evaluation Metrics

### 6.1 Training Metrics (Computed Per Iteration)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Smooth L1 Loss** | $\mathcal{L} = \frac{1}{MN}\sum_{i,j} \text{SmoothL1}(I_{\text{amp}}[i,j], \hat{I}_{\text{amp}}[i,j])$ | Primary optimization objective |
| **MSE** | $\text{MSE} = \frac{1}{MN}\sum_{i,j} \left(I_{\text{amp}}[i,j] - \hat{I}_{\text{amp}}[i,j]\right)^2$ | Mean squared error |
| **PSNR** | $\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1}{\text{MSE}}\right)$ | Peak signal-to-noise ratio (dB) |

### 6.2 Final Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Amplitude Mean** | $\bar{A} = \frac{1}{MM \cdot NN}\sum_{i,j} A[i,j]$ | Mean reconstructed amplitude |
| **Amplitude Std** | $\sigma_A = \sqrt{\frac{1}{MM \cdot NN}\sum_{i,j}(A[i,j] - \bar{A})^2}$ | Amplitude standard deviation |
| **Amplitude Range** | $[A_{\min}, A_{\max}]$ | Min/max amplitude values |
| **Phase Mean** | $\bar{\Phi} = \frac{1}{MM \cdot NN}\sum_{i,j} \Phi[i,j]$ | Mean reconstructed phase |
| **Phase Std** | $\sigma_\Phi = \sqrt{\frac{1}{MM \cdot NN}\sum_{i,j}(\Phi[i,j] - \bar{\Phi})^2}$ | Phase standard deviation |
| **Phase Range** | $[\Phi_{\min}, \Phi_{\max}]$ | Min/max phase values |

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FPM-INR RECONSTRUCTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  REPRESENTATION:                                                            │
│    O(x,y) = A(x,y)·exp(iΦ(x,y))                                            │
│    A = MLP(F_xy^(r) ⊙ F_z^(r))                                             │
│    Φ = MLP(F_xy^(i) ⊙ F_z^(i))                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  FORWARD MODEL (per LED l):                                                 │
│    Õ = FFTshift(FFT2(O))           → High-res spectrum                     │
│    Õ_sub =