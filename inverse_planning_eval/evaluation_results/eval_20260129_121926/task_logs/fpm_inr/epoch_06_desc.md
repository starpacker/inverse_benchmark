## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "FPM-INR (Fourier Ptychographic Microscopy via Implicit Neural Representations)",
  "algorithm_family": "Neural Network-based Inverse Problem Solver / Implicit Neural Representations",
  "summary": "Reconstructs high-resolution complex images from low-resolution Fourier ptychographic microscopy measurements by optimizing implicit neural representations (learnable feature tensors + MLP decoders) using gradient descent on a smooth L1 intensity matching loss."
}
```

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\theta} \sum_{l=1}^{L} \mathcal{L}_{\text{SmoothL1}}\left( \sqrt{I_l^{\text{meas}}}, \left| \mathcal{F}^{-1}\left[ P \odot H_{\text{df}} \odot \text{Crop}_l\left( \text{ZeroPad}\left( \mathcal{F}\left[ f_\theta \right] \right) \right) \right] \right| \right)$$

where $f_\theta(x,y) = A_\theta(x,y) \exp\left(i \Phi_\theta(x,y)\right)$ is the neural network parameterized complex image.

**data_fidelity_term**:
$$\mathcal{L}_{\text{SmoothL1}}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 / \beta & \text{if } |y - \hat{y}| < \beta \\ |y - \hat{y}| - \frac{\beta}{2} & \text{otherwise} \end{cases}$$

with $\beta = 1.0$ (PyTorch default). This is applied element-wise and averaged.

**regularization_term**: None (implicit regularization through neural network architecture)

**constraints**: None (unconstrained optimization over network parameters)

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{D}^{\text{real}}$ | $(x_{\text{mode}}, y_{\text{mode}}, n_f)$ | float32 | $\mathcal{U}(-10^{-4}, 10^{-4})$ | Feature tensor for amplitude network |
| $\mathbf{D}^{\text{imag}}$ | $(x_{\text{mode}}, y_{\text{mode}}, n_f)$ | float32 | $\mathcal{N}(0, 2\times10^{-4})$ | Feature tensor for phase network |
| $\mathbf{Z}^{\text{real}}$ | $(z_{\text{dim}}, n_f)$ | float32 | $\mathcal{N}(0, 1)$ | Z-axis feature tensor for amplitude |
| $\mathbf{Z}^{\text{imag}}$ | $(z_{\text{dim}}, n_f)$ | float32 | $\mathcal{N}(0, 1)$ | Z-axis feature tensor for phase |
| $\mathbf{W}_1^{\text{real}}$ | $(n_f, h_{\text{dim}})$ | float32 | Kaiming uniform | First layer weights for amplitude MLP |
| $\mathbf{b}_1^{\text{real}}$ | $(h_{\text{dim}},)$ | float32 | Uniform | First layer bias for amplitude MLP |
| $\mathbf{W}_2^{\text{real}}$ | $(h_{\text{dim}}, h_{\text{dim}})$ | float32 | Kaiming uniform | Second layer weights for amplitude MLP |
| $\mathbf{b}_2^{\text{real}}$ | $(h_{\text{dim}},)$ | float32 | Uniform | Second layer bias for amplitude MLP |
| $\mathbf{W}_3^{\text{real}}$ | $(h_{\text{dim}}, 1)$ | float32 | Kaiming uniform | Output layer weights for amplitude MLP |
| $\mathbf{b}_3^{\text{real}}$ | $(1,)$ | float32 | Uniform | Output layer bias for amplitude MLP |
| $\mathbf{W}_1^{\text{imag}}$ | $(n_f, h_{\text{dim}})$ | float32 | Kaiming uniform | First layer weights for phase MLP |
| $\mathbf{b}_1^{\text{imag}}$ | $(h_{\text{dim}},)$ | float32 | Uniform | First layer bias for phase MLP |
| $\mathbf{W}_2^{\text{imag}}$ | $(h_{\text{dim}}, h_{\text{dim}})$ | float32 | Kaiming uniform | Second layer weights for phase MLP |
| $\mathbf{b}_2^{\text{imag}}$ | $(h_{\text{dim}},)$ | float32 | Uniform | Second layer bias for phase MLP |
| $\mathbf{W}_3^{\text{imag}}$ | $(h_{\text{dim}}, 1)$ | float32 | Kaiming uniform | Output layer weights for phase MLP |
| $\mathbf{b}_3^{\text{imag}}$ | $(1,)$ | float32 | Uniform | Output layer bias for phase MLP |

### 2.2 Dual Variables

None (this algorithm uses direct gradient descent, not a primal-dual formulation)

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}_0, \mathbf{y}_0$ | $(M_{\text{HR}} \cdot N_{\text{HR}},)$ | int64 | Computed from coordinates | Lower interpolation indices for XY |
| $\mathbf{x}_1, \mathbf{y}_1$ | $(M_{\text{HR}} \cdot N_{\text{HR}},)$ | int64 | Computed from coordinates | Upper interpolation indices for XY |
| $\boldsymbol{\alpha}$ | $(M_{\text{HR}} \cdot N_{\text{HR}}, 2)$ | float32 | Computed from coordinates | Bilinear interpolation weights |
| $\mathbf{m}_1, \mathbf{m}_2$ | $(B,)$ | float32 | 0, 0 | Adam first and second moment estimates |
| $t$ | scalar | int | 0 | Adam iteration counter |
| $\mathbf{A}$ | $(1, M_{\text{HR}}, N_{\text{HR}})$ | float32 | - | Reconstructed amplitude image |
| $\boldsymbol{\Phi}$ | $(1, M_{\text{HR}}, N_{\text{HR}})$ | float32 | - | Reconstructed phase image |
| $\mathbf{f}$ | $(1, M_{\text{HR}}, N_{\text{HR}})$ | complex64 | - | Complex field |
| $\mathbf{O}$ | $(1, M_{\text{HR}}, N_{\text{HR}})$ | complex64 | - | Fourier spectrum of complex field |
| $\mathbf{O}_{\text{pad}}$ | $(1, M_{\text{HR}} \cdot \text{MAG}, N_{\text{HR}} \cdot \text{MAG})$ | complex64 | - | Zero-padded spectrum |
| $\mathbf{O}_l^{\text{sub}}$ | $(1, L_{\text{batch}}, M, N)$ | complex64 | - | Sub-aperture spectra for LED batch |
| $\mathbf{o}_l$ | $(1, L_{\text{batch}}, M, N)$ | complex64 | - | Sub-aperture fields |
| $\hat{\mathbf{I}}_l$ | $(1, L_{\text{batch}}, M, N)$ | float32 | - | Predicted intensity amplitudes |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $n_f$ | 32 | Number of features in feature tensors |
| $h_{\text{dim}}$ | 32 | Hidden dimension of MLP |
| $x_{\text{mode}}, y_{\text{mode}}$ | 512 | Spatial modes in feature tensor |
| $z_{\text{dim}}$ | 5 | Depth dimension for z-features |
| $z_{\text{min}}, z_{\text{max}}$ | 0.0, 1.0 | Z-coordinate bounds |
| MAG | 3 | Magnification factor (MAGimg) |
| $\lambda$ | 0.632 μm (red), 0.5126 μm (green), 0.471 μm (blue) | Wavelength |
| $k_0$ | $2\pi/\lambda$ | Wavenumber |
| $\eta$ | $10^{-3}$ | Initial learning rate |
| $\beta_1, \beta_2$ | 0.9, 0.999 | Adam momentum parameters |
| $\epsilon$ | $10^{-8}$ | Adam numerical stability |
| $\gamma$ | 0.1 | Learning rate decay factor |
| $T_{\text{decay}}$ | 6 | Learning rate decay step |
| $L_{\text{batch}}$ | 1 | LED batch size |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{I}^{\text{meas}}$ | $(M, N, L)$ | float32 | Measured low-resolution intensity images |
| $\mathbf{P}_0$ | $(1, 1, M, N)$ | float32 | Binary pupil support mask |
| $\mathbf{k}_z$ | $(1, M, N)$ | complex64 | Angular spectrum propagation kernel |
| $\mathbf{p}_{\text{LED}}$ | $(L, 2)$ | int64 | LED positions in Fourier space |

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs | assumptions |
|------------|-----------|---------|--------|---------|-------------|
| 1 | Load raw data | $\mathbf{I}_{\text{raw}} \leftarrow \text{loadmat}(\text{path})$ | File path | $\mathbf{I}_{\text{raw}}$: $(M_{\text{full}}, N_{\text{full}}, L_{\text{full}})$ | MATLAB .mat format |
| 2 | Crop intensity stack | $\mathbf{I} \leftarrow \mathbf{I}_{\text{raw}}[0:n_{\text{modes}}, 0:n_{\text{modes}}, :]$ | $\mathbf{I}_{\text{raw}}$, $n_{\text{modes}}=512$ | $\mathbf{I}$: $(M, N, L)$ | $M=N=512$ |
| 3 | Compute wavenumber | $k_0 = \frac{2\pi}{\lambda}$ | $\lambda$ | $k_0$: scalar | Wavelength in μm |
| 4 | Compute high-res dimensions | $M_{\text{HR}} = M \cdot \text{MAG}, \quad N_{\text{HR}} = N \cdot \text{MAG}$ | $M, N$, MAG=3 | $M_{\text{HR}}=N_{\text{HR}}=1536$ | Integer magnification |
| 5 | Compute Fourier coordinates (high-res) | $F_{xx}[j] = \frac{j - N_{\text{HR}}/2}{N \cdot D_{\text{pixel}}} \cdot 2\pi, \quad j=0,\ldots,N_{\text{HR}}-1$ | $N_{\text{HR}}, N, D_{\text{pixel}}$ | $\mathbf{F}_{xx}$: $(N_{\text{HR}},)$ | Centered coordinates |
| 6 | Compute Fourier coordinates (high-res, y) | $F_{yy}[i] = \frac{i - M_{\text{HR}}/2}{M \cdot D_{\text{pixel}}} \cdot 2\pi, \quad i=0,\ldots,M_{\text{HR}}-1$ | $M_{\text{HR}}, M, D_{\text{pixel}}$ | $\mathbf{F}_{yy}$: $(M_{\text{HR}},)$ | Centered coordinates |
| 7 | Sort LEDs by NA | $\text{NA}_{\text{illu}}[l] = \sqrt{u_l^2 + v_l^2}; \quad \text{order} = \text{argsort}(\text{NA}_{\text{illu}})$ | $\mathbf{u}, \mathbf{v}$ (LED NAs) | order: $(L,)$ | Ascending illumination NA |
| 8 | Reorder LED coordinates | $u_l \leftarrow -\text{NAx}[\text{order}[l]], \quad v_l \leftarrow -\text{NAy}[\text{order}[l]]$ | NAx, NAy, order | $\mathbf{u}, \mathbf{v}$: $(L,)$ | Sign flip for coordinate convention |
| 9 | Compute LED positions in spectrum | $p_x[l] = \arg\min_j |F_{xx}[j] - k_0 u_l|, \quad p_y[l] = \arg\min_i |F_{yy}[i] - k_0 v_l|$ | $\mathbf{F}_{xx}, \mathbf{F}_{yy}, \mathbf{u}, \mathbf{v}, k_0$ | $\mathbf{p}_{\text{LED}}$: $(L, 2)$ | Nearest bin assignment |
| 10 | Normalize intensity stack | $\mathbf{I}_{\text{sum}} = \frac{\mathbf{I}[:,:,\text{order}]}{\max(\mathbf{I})}$ | $\mathbf{I}$, order | $\mathbf{I}_{\text{sum}}$: $(M, N, L)$ | Values in $[0, 1]$ |
| 11 | Compute low-res Fourier grid | $k_{xx}[i,j], k_{yy}[i,j] = \text{meshgrid}\left(\text{linspace}(-\pi/D, \pi/D, M), \text{linspace}(-\pi/D, \pi/D, N)\right)$ | $M, N, D_{\text{pixel}}$ | $\mathbf{k}_{xx}, \mathbf{k}_{yy}$: $(M, N)$ | Centered |
| 12 | Compute radial frequency | $k_{rr}[i,j] = \sqrt{k_{xx}[i,j]^2 + k_{yy}[i,j]^2}$ | $\mathbf{k}_{xx}, \mathbf{k}_{yy}$ | $\mathbf{k}_{rr}$: $(M, N)$ | Non-negative |
| 13 | Compute propagation mask | $\text{mask}_k[i,j] = \mathbb{1}[k_0^2 - k_{rr}[i,j]^2 > 0]$ | $k_0, \mathbf{k}_{rr}$ | $\text{mask}_k$: $(M, N)$ | Binary |
| 14 | Compute kz amplitude | $k_z^{\text{amp}}[i,j] = \text{mask}_k[i,j] \cdot \left|\sqrt{k_0^2 - k_{rr}[i,j]^2}\right|$ | $\text{mask}_k, k_0, \mathbf{k}_{rr}$ | $\mathbf{k}_z^{\text{amp}}$: $(M, N)$ | Evanescent waves zeroed |
| 15 | Compute kz phase | $k_z^{\text{phase}}[i,j] = \angle\sqrt{k_0^2 - k_{rr}[i,j]^2}$ | $k_0, \mathbf{k}_{rr}$ | $\mathbf{k}_z^{\text{phase}}$: $(M, N)$ | Complex sqrt |
| 16 | Combine kz | $k_z[i,j] = k_z^{\text{amp}}[i,j] \cdot \exp(i \cdot k_z^{\text{phase}}[i,j])$ | $\mathbf{k}_z^{\text{amp}}, \mathbf{k}_z^{\text{phase}}$ | $\mathbf{k}_z$: $(1, M, N)$ complex64 | Angular spectrum kernel |
| 17 | Compute pupil Fourier grid | $F_x^2[i,j] = \left(\frac{j - N/2}{N \cdot D_{\text{pixel}}} \cdot 2\pi\right)^2$ | $N, D_{\text{pixel}}$ | $\mathbf{F}_x^2$: $(M, N)$ | Squared frequencies |
| 18 | Compute pupil Fourier grid (y) | $F_y^2[i,j] = \left(\frac{i - M/2}{M \cdot D_{\text{pixel}}} \cdot 2\pi\right)^2$ | $M, D_{\text{pixel}}$ | $\mathbf{F}_y^2$: $(M, N)$ | Squared frequencies |
| 19 | Compute pupil support | $P_0[i,j] = \mathbb{1}[F_x^2[i,j] + F_y^2[i,j] \leq k_{\max}^2]$ where $k_{\max} = \text{NA} \cdot k_0$ | $\mathbf{F}_x^2, \mathbf{F}_y^2, k_{\max}$ | $\mathbf{P}_0$: $(1, 1, M, N)$ | Binary pupil |
| 20 | Transfer to device | $\mathbf{P}_0, \mathbf{k}_z, \mathbf{I}_{\text{sum}} \leftarrow \text{to\_device}(\cdot)$ | CPU tensors | GPU tensors | CUDA available |

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | variable_name | initial_value | shape | dtype | source |
|------------|---------------|---------------|-------|-------|--------|
| 1 | $\mathbf{D}^{\text{real}}$ | $2 \times 10^{-4} \cdot \mathcal{N}(0,1) \sim \mathcal{U}(-10^{-4}, 10^{-4})$ | $(512, 512, 32)$ | float32 | Random |
| 2 | $\mathbf{D}^{\text{imag}}$ | $2 \times 10^{-4} \cdot \mathcal{N}(0,1)$ | $(512, 512, 32)$ | float32 | Random |
| 3 | $\mathbf{Z}^{\text{real}}$ | $\mathcal{N}(0,1)$ | $(5, 32)$ | float32 | Random |
| 4 | $\mathbf{Z}^{\text{imag}}$ | $\mathcal{N}(0,1)$ | $(5, 32)$ | float32 | Random |
| 5 | MLP weights (all 12 tensors) | PyTorch default (Kaiming uniform for weights, uniform for biases) | Various | float32 | Random |
| 6 | Compute half-pixel offsets | $\delta_x = \frac{0.5}{M_{\text{HR}}}, \quad \delta_y = \frac{0.5}{N_{\text{HR}}}$ | scalars | float32 | Constant |
| 7 | Generate normalized coordinates | $\mathbf{xs} = \text{linspace}(\delta_x, 1-\delta_x, M_{\text{HR}})$ | $(M_{\text{HR}},)$ | float32 | Computed |
| 8 | Generate normalized coordinates (y) | $\mathbf{ys} = \text{linspace}(\delta_y, 1-\delta_y, N_{\text{HR}})$ | $(N_{\text{HR}},)$ | float32 | Computed |
| 9 | Create coordinate meshgrid | $\mathbf{xv}, \mathbf{yv} = \text{meshgrid}(\mathbf{xs}, \mathbf{ys})$ | $(M_{\text{HR}}, N_{\text{HR}})$ each | float32 | Computed |
| 10 | Flatten and stack coordinates | $\mathbf{xy} = [\mathbf{yv}.\text{flat}, \mathbf{xv}.\text{flat}]^T$ | $(M_{\text{HR}} \cdot N_{\text{HR}}, 2)$ | float32 | Computed |
| 11 | Scale to mode indices | $\mathbf{xs}_{\text{scaled}} = \mathbf{xy} \cdot [x_{\text{mode}}, y_{\text{mode}}]$ | $(M_{\text{HR}} \cdot N_{\text{HR}}, 2)$ | float32 | Computed |
| 12 | Compute integer indices | $\mathbf{idx} = \lfloor \mathbf{xs}_{\text{scaled}} \rfloor$ | $(M_{\text{HR}} \cdot N_{\text{HR}}, 2)$ | int64 | Computed |
| 13 | Compute interpolation weights | $\boldsymbol{\alpha} = \mathbf{xs}_{\text{scaled}} - \mathbf{idx}$ | $(M_{\text{HR}} \cdot N_{\text{HR}}, 2)$ | float32 | Computed |
| 14 | Compute clamped indices | $\mathbf{x}_0 = \text{clamp}(\mathbf{idx}[:,0], 0, x_{\text{mode}}-1)$ | $(M_{\text{HR}} \cdot N_{\text{HR}},)$ | int64 | Computed |
| 15 | Compute clamped indices | $\mathbf{y}_0 = \text{clamp}(\mathbf{idx}[:,1], 0, y_{\text{mode}}-1)$ | $(M_{\text{HR}} \cdot N_{\text{HR}},)$ | int64 | Computed |
| 16 | Compute clamped indices | $\mathbf{x}_1 = \text{clamp}(\mathbf{x}_0 + 1, 0, x_{\text{mode}}-1)$ | $(M_{\text{HR}} \cdot N_{\text{HR}},)$ | int64 | Computed |
| 17 | Compute clamped indices | $\mathbf{y}_1 = \text{clamp}(\mathbf{y}_0 + 1, 0, y_{\text{mode}}-1)$ | $(M_{\text{HR}} \cdot N_{\text{HR}},)$ | int64 | Computed |
| 18 | Initialize Adam moments | $\mathbf{m}_1^{(0)} = \mathbf{0}, \quad \mathbf{m}_2^{(0)} = \mathbf{0}$ | Same as each parameter | float32 | Zeros |
| 19 | Initialize iteration counter | $t = 0$ | scalar | int | Zero |
| 20 | Initialize learning rate | $\eta^{(0)} = 10^{-3}$ | scalar | float32 | Constant |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (epoch $e = 0, 1, \ldots, E-1$ where $E = 5$)

**BEGIN_LOOP** (LED index $l = 0, 1, \ldots, L-1$)

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 21 | Set defocus value | $dz = 0.0$ | Constant | $dz$: $(1,)$ float32 | Single focal plane |
| 22 | Compute defocus mask | $\mathbf{H}_{\text{df}} = \exp\left(i \cdot \mathbf{k}_z \cdot dz\right)$ | $\mathbf{k}_z$: $(1, M, N)$, $dz$: $(1,)$ | $\mathbf{H}_{\text{df}}$: $(1, 1, M, N)$ complex64 | Element-wise exponential |
| 23 | Compute combined mask amplitude | $\mathbf{M}_{\text{amp}} = \mathbf{P}_0 \odot |\mathbf{H}_{\text{df}}|$ | $\mathbf{P}_0$: $(1, 1, M, N)$, $\mathbf{H}_{\text{df}}$ | $\mathbf{M}_{\text{amp}}$: $(1, 1, M, N)$ float32 | Element-wise product |
| 24 | Compute combined mask phase | $\mathbf{M}_{\text{phase}} = \mathbf{P}_0 \odot \angle\mathbf{H}_{\text{df}}$ | $\mathbf{P}_0$, $\mathbf{H}_{\text{df}}$ | $\mathbf{M}_{\text{phase}}$: $(1, 1, M, N)$ float32 | Element-wise product |
| 25 | Combine spectrum mask | $\mathbf{M}_{\text{spec}} = \mathbf{M}_{\text{amp}} \odot \exp(i \cdot \mathbf{M}_{\text{phase}})$ | $\mathbf{M}_{\text{amp}}, \mathbf{M}_{\text{phase}}$ | $\mathbf{M}_{\text{spec}}$: $(1, 1, M, N)$ complex64 | Complex mask |
| 26 | Normalize z-coordinate | $\tilde{z} = (z_{\text{dim}} - 1) \cdot \frac{dz - z_{\min}}{z_{\max} - z_{\min}}$ | $dz$, $z_{\min}=0$, $z_{\max}=1$, $z_{\text{dim}}=5$ | $\tilde{z}$: $(1,)$ float32 | Maps to [0, z_dim-1] |
| 27 | Compute z interpolation indices | $z_0 = \text{clamp}(\lfloor \tilde{z} \rfloor, 0, z_{\text{dim}}-1), \quad z_1 = \text{clamp}(z_0 + 1, 0, z_{\text{dim}}-1)$ | $\tilde{z}$ | $z_0, z_1$: $(1,)$ int64 | Clamped indices |
| 28 | Compute z interpolation weight | $\alpha_z = \tilde{z} - \lfloor \tilde{z} \rfloor$ | $\tilde{z}$ | $\alpha_z$: $(1, 1)$ float32 | Fractional part |
| 29 | [FORWARD] Bilinear interpolation (amplitude XY features) | $\mathbf{F}_{xy}^{\text{real}} = \mathbf{D}^{\text{real}}[\mathbf{y}_0, \mathbf{x}_0] \odot (1-\boldsymbol{\alpha}_{:,0}) \odot (1-\boldsymbol{\alpha}_{:,1}) + \mathbf{D}^{\text{real}}[\mathbf{y}_0, \mathbf{x}_1] \odot \boldsymbol{\alpha}_{:,0} \odot (1-\boldsymbol{\alpha}_{:,1}) + \mathbf{D}^{\text{real}}[\mathbf{y}_1, \mathbf{x}_0] \odot (1-\boldsymbol{\alpha}_{:,0}) \odot \boldsymbol{\alpha}_{:,1} + \mathbf{D}^{\text{real}}[\mathbf{y}_1, \mathbf{x}_1] \odot \boldsymbol{\alpha}_{:,0} \odot \boldsymbol{\alpha}_{:,1}$ | $\mathbf{D}^{\text{real}}$: $(512, 512, 32)$, indices, weights | $\mathbf{F}_{xy}^{\text{real}}$: $(M_{\text{