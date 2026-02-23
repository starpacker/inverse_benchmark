## 0. ALGORITHM IDENTIFICATION

- **algorithm_name**: ADMM for Lensless Image Reconstruction with Total Variation
- **algorithm_family**: Alternating Direction Method of Multipliers (ADMM) / Proximal Splitting
- **summary**: Reconstructs an image from lensless camera measurements by solving a convex optimization problem with L2 data fidelity, anisotropic Total Variation regularization, and non-negativity constraint via variable splitting in the Fourier domain.

## 1. OBJECTIVE FUNCTION

### full_expression
$$\min_{\mathbf{x}} \frac{1}{2}\|\mathbf{H}\mathbf{x} - \mathbf{y}\|_2^2 + \tau \|\mathbf{D}\mathbf{x}\|_1 + \mathcal{I}_{\mathbf{x} \geq 0}(\mathbf{x})$$

where $\mathcal{I}_{\mathbf{x} \geq 0}(\mathbf{x}) = \begin{cases} 0 & \text{if } x_{d,i,j,c} \geq 0 \; \forall d,i,j,c \\ +\infty & \text{otherwise} \end{cases}$

### data_fidelity_term
$$f(\mathbf{x}) = \frac{1}{2}\|\mathbf{H}\mathbf{x} - \mathbf{y}\|_2^2 = \frac{1}{2}\sum_{d,i,j,c}\left((\mathbf{H}\mathbf{x})_{d,i,j,c} - y_{d,i,j,c}\right)^2$$

### regularization_term
**Anisotropic Total Variation**:
$$g(\mathbf{x}) = \tau \|\mathbf{D}\mathbf{x}\|_1 = \tau \sum_{d,i,j,c} \left( |(\mathbf{D}_h \mathbf{x})_{d,i,j,c}| + |(\mathbf{D}_v \mathbf{x})_{d,i,j,c}| \right)$$

where:
- $(\mathbf{D}_h \mathbf{x})_{d,i,j,c} = x_{d,i-1,j,c} - x_{d,i,j,c}$ (horizontal/row difference with periodic boundary)
- $(\mathbf{D}_v \mathbf{x})_{d,i,j,c} = x_{d,i,j-1,c} - x_{d,i,j,c}$ (vertical/column difference with periodic boundary)

### constraints
**Non-negativity constraint**: $\mathbf{x} \geq 0$ (element-wise), enforced via indicator function $\mathcal{I}_{\mathbf{x} \geq 0}$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}$ | $(1, D_p, H_p, W_p, C)$ | float32 | $\mathbf{0}$ | Estimated image in padded domain |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\xi}$ | $(1, D_p, H_p, W_p, C)$ | float32 | $\mathbf{0}$ | Dual variable for convolution constraint $\mathbf{Hx} = \mathbf{X}$ |
| $\boldsymbol{\eta}$ | $(1, D_p, H_p, W_p, C, 2)$ | float32 | $\mathbf{0}$ | Dual variable for TV constraint $\mathbf{Dx} = \mathbf{U}$ |
| $\boldsymbol{\rho}$ | $(1, D_p, H_p, W_p, C)$ | float32 | $\mathbf{0}$ | Dual variable for non-negativity constraint $\mathbf{x} = \mathbf{W}$ |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{X}$ | $(1, D_p, H_p, W_p, C)$ | float32 | $\mathbf{0}$ | Splitting variable for data fidelity term |
| $\mathbf{U}$ | $(1, D_p, H_p, W_p, C, 2)$ | float32 | $\mathbf{0}$ | Splitting variable for TV regularization |
| $\mathbf{W}$ | $(1, D_p, H_p, W_p, C)$ | float32 | $\mathbf{0}$ | Splitting variable for non-negativity projection |
| $\mathbf{r}^{(k)}$ | $(1, D_p, H_p, W_p, C)$ | float32 | N/A | Aggregated gradient for image update |
| $\boldsymbol{\Psi}_{\text{out}}$ | $(1, D_p, H_p, W_p, C, 2)$ | float32 | N/A | Finite difference output $\mathbf{Dx}$ |
| $\mathbf{f}_{\text{out}}$ | $(1, D_p, H_p, W_p, C)$ | float32 | N/A | Forward convolution output $\mathbf{Hx}$ |

### 2.4 Constants

| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $\mu_1$ | scalar | float | $1 \times 10^{-6}$ | ADMM penalty for data fidelity constraint |
| $\mu_2$ | scalar | float | $1 \times 10^{-5}$ | ADMM penalty for TV constraint |
| $\mu_3$ | scalar | float | $4 \times 10^{-5}$ | ADMM penalty for non-negativity constraint |
| $\tau$ | scalar | float | $1 \times 10^{-4}$ | TV regularization weight |
| $N_{\text{iter}}$ | scalar | int | $50$ (default) | Maximum number of iterations |
| downsample | scalar | int | $4$ | Spatial downsampling factor |
| $\mathbf{H}$ | $(D_p, H_p, \lfloor W_p/2 \rfloor + 1, C)$ | complex64 | Precomputed | Fourier transform of zero-padded PSF |
| $\mathbf{H}^*$ | $(D_p, H_p, \lfloor W_p/2 \rfloor + 1, C)$ | complex64 | $\overline{\mathbf{H}}$ | Complex conjugate of $\mathbf{H}$ |
| $\mathbf{R}_{\text{div}}$ | $(D_p, H_p, \lfloor W_p/2 \rfloor + 1, C)$ | complex64 | Precomputed | Frequency domain division matrix for image update |
| $\mathbf{X}_{\text{div}}$ | $(D_p, H_p, W_p, C)$ | float32 | Precomputed | Spatial domain division matrix for X update |
| $\boldsymbol{\Psi}^T\boldsymbol{\Psi}$ | $(D_p, H_p, \lfloor W_p/2 \rfloor + 1, C)$ | complex64 | Precomputed | Fourier transform of TV Gram matrix |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(D, H, W, C)$ | float32 | Raw lensless measurement (normalized) |
| $\mathbf{h}$ | $(D, H, W, C)$ | float32 | Point spread function (normalized) |
| $\mathbf{y}_{\text{pad}}$ | $(D_p, H_p, W_p, C)$ | float32 | Zero-padded measurement |

---

## 3. PREPROCESSING PIPELINE

| step_order | step_name | formula | inputs | outputs |
|------------|-----------|---------|--------|---------|
| 1 | Load and normalize data | $\mathbf{y}, \mathbf{h} \leftarrow \texttt{load\_data}(\cdot, \text{normalize}=\text{True})$ | psf_path, data_path, downsample=4 | $\mathbf{y} \in \mathbb{R}^{D \times H \times W \times C}$, $\mathbf{h} \in \mathbb{R}^{D \times H \times W \times C}$ |
| 2 | Compute padded dimensions | $H_p = \texttt{next\_fast\_len}(2H - 1), \quad W_p = \texttt{next\_fast\_len}(2W - 1)$ | $H, W$ from psf_shape | $H_p, W_p$ (FFT-efficient dimensions) |
| 3 | Compute padding indices | $i_{\text{start}} = \lfloor (H_p - H)/2 \rfloor, \quad j_{\text{start}} = \lfloor (W_p - W)/2 \rfloor$ | $H_p, W_p, H, W$ | $i_{\text{start}}, j_{\text{start}}, i_{\text{end}} = i_{\text{start}} + H, j_{\text{end}} = j_{\text{start}} + W$ |
| 4 | Zero-pad PSF | $\mathbf{h}_{\text{pad}} \in \mathbb{R}^{D_p \times H_p \times W_p \times C}$ with $(\mathbf{h}_{\text{pad}})_{d, i_{\text{start}}:i_{\text{end}}, j_{\text{start}}:j_{\text{end}}, c} = h_{d,i,j,c}$ and zeros elsewhere | $\mathbf{h}$, padding indices | $\mathbf{h}_{\text{pad}}$ |
| 5 | Compute PSF frequency response | $\mathbf{H} = \mathcal{F}_{2D}^{\text{ortho}}(\mathbf{h}_{\text{pad}})$ where $\mathcal{F}_{2D}^{\text{ortho}}$ is 2D rFFT with orthonormal normalization over axes $(-3, -2)$ | $\mathbf{h}_{\text{pad}}$ | $\mathbf{H} \in \mathbb{C}^{D_p \times H_p \times (\lfloor W_p/2 \rfloor + 1) \times C}$ |
| 6 | Compute adjoint frequency response | $\mathbf{H}^* = \overline{\mathbf{H}}$ (element-wise complex conjugate) | $\mathbf{H}$ | $\mathbf{H}^*$ |
| 7 | Construct TV Gram kernel | For $D_p = 1$: $g_{0,0,0,\cdot} = 4$, $g_{0,0,1,\cdot} = g_{0,0,-1,\cdot} = g_{0,1,0,\cdot} = g_{0,-1,0,\cdot} = -1$, else $0$. For $D_p > 1$: $g_{0,0,0,\cdot} = 6$, with $-1$ at 6-connected neighbors. | padded_shape, dtype | $\mathbf{g} \in \mathbb{R}^{D_p \times H_p \times W_p \times C}$ |
| 8 | Compute TV Gram in frequency | $\boldsymbol{\Psi}^T\boldsymbol{\Psi} = \mathcal{F}_{2D}(\mathbf{g})$ | $\mathbf{g}$ | $\boldsymbol{\Psi}^T\boldsymbol{\Psi} \in \mathbb{C}^{D_p \times H_p \times (\lfloor W_p/2 \rfloor + 1) \times C}$ |
| 9 | Compute image update division matrix | $(\mathbf{R}_{\text{div}})_{d,i,j,c} = \frac{1}{\mu_1 |\mathbf{H}^*_{d,i,j,c} \cdot \mathbf{H}_{d,i,j,c}| + \mu_2 |(\boldsymbol{\Psi}^T\boldsymbol{\Psi})_{d,i,j,c}| + \mu_3}$ | $\mathbf{H}, \mathbf{H}^*, \boldsymbol{\Psi}^T\boldsymbol{\Psi}, \mu_1, \mu_2, \mu_3$ | $\mathbf{R}_{\text{div}} \in \mathbb{C}^{D_p \times H_p \times (\lfloor W_p/2 \rfloor + 1) \times C}$ |
| 10 | Compute X update division matrix | $(\mathbf{X}_{\text{div}})_{d,i,j,c} = \frac{1}{\mathbf{M}_{d,i,j,c} + \mu_1}$ where $\mathbf{M}$ is zero-padded array of ones matching PSF valid region | psf_shape, $\mu_1$, padding indices | $\mathbf{X}_{\text{div}} \in \mathbb{R}^{D_p \times H_p \times W_p \times C}$ |
| 11 | Zero-pad measurement | $\mathbf{y}_{\text{pad}} \in \mathbb{R}^{D_p \times H_p \times W_p \times C}$ with $(\mathbf{y}_{\text{pad}})_{d, i_{\text{start}}:i_{\text{end}}, j_{\text{start}}:j_{\text{end}}, c} = y_{d,i,j,c}$ | $\mathbf{y}$, padding indices | $\mathbf{y}_{\text{pad}}$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 1 | Initialize image estimate | $\mathbf{x}^{(0)}_{1,d,i,j,c} = 0 \quad \forall d,i,j,c$ | padded_shape | $\mathbf{x}^{(0)} \in \mathbb{R}^{1 \times D_p \times H_p \times W_p \times C}$ | Zero initialization |
| 2 | Initialize X splitting variable | $\mathbf{X}^{(0)}_{1,d,i,j,c} = 0 \quad \forall d,i,j,c$ | shape of $\mathbf{x}^{(0)}$ | $\mathbf{X}^{(0)}$ | Same shape as image estimate |
| 3 | Initialize U splitting variable | $\mathbf{U}^{(0)}_{1,d,i,j,c,\ell} = 0 \quad \forall d,i,j,c,\ell \in \{0,1\}$ | shape of $\mathbf{Dx}^{(0)}$ | $\mathbf{U}^{(0)} \in \mathbb{R}^{1 \times D_p \times H_p \times W_p \times C \times 2}$ | Last dimension indexes gradient direction |
| 4 | Initialize W splitting variable | $\mathbf{W}^{(0)}_{1,d,i,j,c} = 0 \quad \forall d,i,j,c$ | shape of $\mathbf{x}^{(0)}$ | $\mathbf{W}^{(0)}$ | Same shape as image estimate |
| 5 | Initialize dual variable xi | $\boldsymbol{\xi}^{(0)}_{1,d,i,j,c} = 0 \quad \forall d,i,j,c$ | shape of $\mathbf{x}^{(0)}$ | $\boldsymbol{\xi}^{(0)}$ | Dual for convolution constraint |
| 6 | Initialize dual variable eta | $\boldsymbol{\eta}^{(0)}_{1,d,i,j,c,\ell} = 0 \quad \forall d,i,j,c,\ell$ | shape of $\mathbf{U}^{(0)}$ | $\boldsymbol{\eta}^{(0)}$ | Dual for TV constraint |
| 7 | Initialize dual variable rho | $\boldsymbol{\rho}^{(0)}_{1,d,i,j,c} = 0 \quad \forall d,i,j,c$ | shape of $\mathbf{x}^{(0)}$ | $\boldsymbol{\rho}^{(0)}$ | Dual for non-negativity constraint |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** for $k = 0, 1, \ldots, N_{\text{iter}} - 1$

| step_order | step_name | formula | inputs | outputs | computational_note |
|------------|-----------|---------|--------|---------|-------------------|
| 8 | Compute finite difference of image | $(\boldsymbol{\Psi}_{\text{out}})_{1,d,i,j,c,0} = x^{(k)}_{1,d,i-1 \mod H_p,j,c} - x^{(k)}_{1,d,i,j,c}$; $(\boldsymbol{\Psi}_{\text{out}})_{1,d,i,j,c,1} = x^{(k)}_{1,d,i,j-1 \mod W_p,c} - x^{(k)}_{1,d,i,j,c}$ | $\mathbf{x}^{(k)}$ | $\boldsymbol{\Psi}_{\text{out}} \in \mathbb{R}^{1 \times D_p \times H_p \times W_p \times C \times 2}$ | Periodic boundary via np.roll with shift +1 |
| 9 | U update: TV soft-thresholding | $\mathbf{U}^{(k+1)}_{1,d,i,j,c,\ell} = \text{sign}\left((\boldsymbol{\Psi}_{\text{out}})_{1,d,i,j,c,\ell} + \frac{\eta^{(k)}_{1,d,i,j,c,\ell}}{\mu_2}\right) \cdot \max\left(0, \left|(\boldsymbol{\Psi}_{\text{out}})_{1,d,i,j,c,\ell} + \frac{\eta^{(k)}_{1,d,i,j,c,\ell}}{\mu_2}\right| - \frac{\tau}{\mu_2}\right)$ | $\boldsymbol{\Psi}_{\text{out}}, \boldsymbol{\eta}^{(k)}, \mu_2, \tau$ | $\mathbf{U}^{(k+1)}$ | Proximal operator for $\ell_1$ norm: $\text{prox}_{\tau/\mu_2 \|\cdot\|_1}(v) = \text{sign}(v) \max(0, |v| - \tau/\mu_2)$ |
| 10 | Compute forward convolution | $\mathbf{f}_{\text{out}} = \mathcal{F}^{-1}_{2D}\left[\text{ifftshift}\left(\mathcal{F}_{2D}(\mathbf{x}^{(k)}) \odot \mathbf{H}\right)\right]$ | $\mathbf{x}^{(k)}, \mathbf{H}$ | $\mathbf{f}_{\text{out}} \in \mathbb{R}^{1 \times D_p \times H_p \times W_p \times C}$ | [FORWARD] FFT-based convolution with ifftshift for centering |
| 11 | X update: data fidelity | $\mathbf{X}^{(k+1)}_{1,d,i,j,c} = (\mathbf{X}_{\text{div}})_{d,i,j,c} \cdot \left(\xi^{(k)}_{1,d,i,j,c} + \mu_1 \cdot (\mathbf{f}_{\text{out}})_{1,d,i,j,c} + (\mathbf{y}_{\text{pad}})_{d,i,j,c}\right)$ | $\boldsymbol{\xi}^{(k)}, \mathbf{f}_{\text{out}}, \mathbf{y}_{\text{pad}}, \mathbf{X}_{\text{div}}, \mu_1$ | $\mathbf{X}^{(k+1)}$ | Closed-form solution to $\arg\min_X \frac{1}{2}\|X - y\|^2 + \frac{\mu_1}{2}\|Hx - X\|^2 + \xi^T(Hx - X)$ |
| 12 | W update: non-negativity projection | $\mathbf{W}^{(k+1)}_{1,d,i,j,c} = \max\left(0, \frac{\rho^{(k)}_{1,d,i,j,c}}{\mu_3} + x^{(k)}_{1,d,i,j,c}\right)$ | $\boldsymbol{\rho}^{(k)}, \mathbf{x}^{(k)}, \mu_3$ | $\mathbf{W}^{(k+1)}$ | Proximal operator for indicator $\mathcal{I}_{\geq 0}$: $\text{prox}(v) = \max(0, v)$ |
| 13 | Compute adjoint convolution | $\mathbf{H}^T\mathbf{v} = \mathcal{F}^{-1}_{2D}\left[\text{ifftshift}\left(\mathcal{F}_{2D}(\mathbf{v}) \odot \mathbf{H}^*\right)\right]$ where $\mathbf{v} = \mu_1 \mathbf{X}^{(k+1)} - \boldsymbol{\xi}^{(k)}$ | $\mathbf{X}^{(k+1)}, \boldsymbol{\xi}^{(k)}, \mathbf{H}^*, \mu_1$ | $\mathbf{H}^T\mathbf{v} \in \mathbb{R}^{1 \times D_p \times H_p \times W_p \times C}$ | [ADJOINT] Correlation via conjugate multiply |
| 14 | Compute TV adjoint | $(\mathbf{D}^T\mathbf{u})_{1,d,i,j,c} = (u_{1,d,i+1 \mod H_p,j,c,0} - u_{1,d,i,j,c,0}) + (u_{1,d,i,j+1 \mod W_p,c,1} - u_{1,d,i,j,c,1})$ where $\mathbf{u} = \mu_2 \mathbf{U}^{(k+1)} - \boldsymbol{\eta}^{(k)}$ | $\mathbf{U}^{(k+1)}, \boldsymbol{\eta}^{(k)}, \mu_2$ | $\mathbf{D}^T\mathbf{u} \in \mathbb{R}^{1 \times D_p \times H_p \times W_p \times C}$ | [ADJOINT] Negative divergence operator via np.roll with shift -1 |
| 15 | Aggregate RHS for image update | $\mathbf{r}^{(k)} = (\mu_3 \mathbf{W}^{(k+1)} - \boldsymbol{\rho}^{(k)}) + \mathbf{D}^T\mathbf{u} + \mathbf{H}^T\mathbf{v}$ | $\mathbf{W}^{(k+1)}, \boldsymbol{\rho}^{(k)}, \mathbf{D}^T\mathbf{u}, \mathbf{H}^T\mathbf{v}, \mu_3$ | $\mathbf{r}^{(k)}$ | Sum of all constraint contributions |
| 16 | Image update: frequency domain solve | $\mathbf{x}^{(k+1)} = \mathcal{F}^{-1}_{2D}\left[\mathbf{R}_{\text{div}} \odot \mathcal{F}_{2D}(\mathbf{r}^{(k)})\right]$ | $\mathbf{r}^{(k)}, \mathbf{R}_{\text{div}}$ | $\mathbf{x}^{(k+1)}$ | Solves $(mu_1 H^TH + \mu_2 D^TD + \mu_3 I)x = r$ via FFT diagonalization |
| 17 | Recompute forward convolution | $\mathbf{f}_{\text{out}} = \mathcal{F}^{-1}_{2D}\left[\text{ifftshift}\left(\mathcal{F}_{2D}(\mathbf{x}^{(k+1)}) \odot \mathbf{H}\right)\right]$ | $\mathbf{x}^{(k+1)}, \mathbf{H}$ | $\mathbf{f}_{\text{out}}$ | [FORWARD] Needed for dual update |
| 18 | Recompute