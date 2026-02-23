# Algorithmic Specification Document: APGD (FISTA) Image Deconvolution

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{x \in \mathbb{R}^{D \times H \times W \times C}} \frac{1}{2} \| \mathbf{A}x - y \|_2^2 \quad \text{subject to} \quad x \geq 0$$

**data_fidelity_term**:
$$f(x) = \frac{1}{2} \| \mathbf{A}x - y \|_2^2 = \frac{1}{2} \sum_{d,i,j,c} \left( (\mathbf{A}x)_{d,i,j,c} - y_{d,i,j,c} \right)^2$$

Where $\mathbf{A}$ is the linear convolution operator defined as:
$$(\mathbf{A}x)_{d,i,j,c} = (h * x)_{d,i,j,c} = \sum_{m,n} h_{d,m,n,c} \cdot x_{d,i-m,j-n,c}$$

**regularization_term**: None (non-negativity enforced via constraint)

**constraints**:
| Constraint | Expression |
|------------|------------|
| Non-negativity | $x_{d,i,j,c} \geq 0 \quad \forall \ d,i,j,c$ |

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $x^{(k)}$ | primal | $(D, H, W, C)$ | float32 | $x^{(0)} = \mathbf{0}$ | Current image estimate at iteration $k$ |

### Auxiliary Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $z^{(k)}$ | dual | $(D, H, W, C)$ | float32 | $z^{(0)} = x^{(0)}$ | Extrapolated point for momentum |
| $t^{(k)}$ | dual | scalar | float64 | $t^{(0)} = 1$ | FISTA momentum coefficient |
| $\mathbf{H}$ | dual | $(D, H_p, \lfloor W_p/2 \rfloor + 1, C)$ | complex64 | Computed from $h$ | FFT of zero-padded PSF |
| $\mathbf{H}^*$ | dual | $(D, H_p, \lfloor W_p/2 \rfloor + 1, C)$ | complex64 | $\overline{\mathbf{H}}$ | Complex conjugate of $\mathbf{H}$ |
| $L$ | dual | scalar | float32 | Power method estimate | Lipschitz constant of $\nabla f$ |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $h$ | constants | $(D, H, W, C)$ | float32 | Loaded from file | Point spread function (PSF) |
| $\alpha$ | constants | scalar | float32 | $\alpha = 1/L$ | Step size |
| $N_{\text{iter}}$ | constants | scalar | int | User-specified (default: 50) | Maximum iterations |
| $\delta$ | constants | scalar | int | User-specified (default: 4) | Downsampling factor |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $y$ | observations | $(D, H, W, C)$ | float32 | Loaded from file | Observed blurred measurement |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Image Loading
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load Raw Images |
| formula | $I_{\text{raw}} = \texttt{imread}(\text{filepath}) \in \mathbb{R}^{H_0 \times W_0 \times C}$ |
| inputs | File path (string) |
| outputs | $I_{\text{raw}} \in \mathbb{R}^{H_0 \times W_0 \times C}$ (or $\mathbb{R}^{H_0 \times W_0}$ for grayscale) |

### Step 3.2: Downsampling
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Spatial Downsampling |
| formula | $I_{\text{ds}}[i,j,c] = I_{\text{raw}}[\delta \cdot i, \delta \cdot j, c]$ where $i \in \{0, ..., \lfloor H_0/\delta \rfloor - 1\}$, $j \in \{0, ..., \lfloor W_0/\delta \rfloor - 1\}$ |
| inputs | $I_{\text{raw}} \in \mathbb{R}^{H_0 \times W_0 \times C}$, $\delta \in \mathbb{Z}^+$ |
| outputs | $I_{\text{ds}} \in \mathbb{R}^{\lfloor H_0/\delta \rfloor \times \lfloor W_0/\delta \rfloor \times C}$ |

### Step 3.3: Intensity Normalization
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Min-Max Normalization |
| formula | $I_{\text{norm}} = \frac{I_{\text{ds}} - \min(I_{\text{ds}})}{\max(I_{\text{ds}}) - \min(I_{\text{ds}})}$ |
| inputs | $I_{\text{ds}} \in \mathbb{R}^{H \times W \times C}$ |
| outputs | $I_{\text{norm}} \in [0,1]^{H \times W \times C}$ |

### Step 3.4: Dimension Expansion to 4D
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Reshape to 4D Tensor |
| formula | For 2D input: $I_{\text{4D}} = I_{\text{norm}}[\texttt{newaxis}, :, :, \texttt{newaxis}]$. For 3D input: $I_{\text{4D}} = I_{\text{norm}}[\texttt{newaxis}, :, :, :]$ |
| inputs | $I_{\text{norm}} \in \mathbb{R}^{H \times W}$ or $\mathbb{R}^{H \times W \times C}$ |
| outputs | $I_{\text{4D}} \in \mathbb{R}^{1 \times H \times W \times C}$ |

### Step 3.5: FFT Padding Shape Computation
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Compute Padded Dimensions |
| formula | $H_p = \texttt{next\_fast\_len}(2H - 1)$, $W_p = \texttt{next\_fast\_len}(2W - 1)$, where $\texttt{next\_fast\_len}(n)$ returns smallest integer $\geq n$ with small prime factors |
| inputs | PSF shape $(D, H, W, C)$ |
| outputs | Padded shape $(D, H_p, W_p, C)$, start indices $s = \lfloor (H_p - H)/2 \rfloor$, $\lfloor (W_p - W)/2 \rfloor$, end indices $e = s + (H, W)$ |

### Step 3.6: PSF FFT Computation
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Compute PSF Frequency Response |
| formula | $\mathbf{H} = \mathcal{F}_{2D}[\texttt{pad}(h)]$ where $\mathcal{F}_{2D}$ is the orthonormalized 2D real FFT: $\mathcal{F}_{2D}[g]_{d,u,v,c} = \frac{1}{\sqrt{H_p W_p}} \sum_{m=0}^{H_p-1} \sum_{n=0}^{W_p-1} g_{d,m,n,c} \cdot e^{-2\pi i (um/H_p + vn/W_p)}$ |
| inputs | $h \in \mathbb{R}^{D \times H \times W \times C}$, padding parameters |
| outputs | $\mathbf{H} \in \mathbb{C}^{D \times H_p \times (\lfloor W_p/2 \rfloor + 1) \times C}$ |

### Step 3.7: Lipschitz Constant Estimation
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Power Method for Lipschitz Constant |
| formula | Initialize $v^{(0)} \sim \mathcal{N}(0,1)^{D \times H \times W \times C}$, $v^{(0)} \leftarrow v^{(0)}/\|v^{(0)}\|_2$. For $j = 1, ..., 20$: $v^{(j)} = \mathbf{A}^T \mathbf{A} v^{(j-1)}$, $L^{(j)} = \|v^{(j)}\|_2$, $v^{(j)} \leftarrow v^{(j)}/L^{(j)}$. Output $L = L^{(20)}$ |
| inputs | $\mathbf{H}$, $\mathbf{H}^*$, PSF shape |
| outputs | $L \in \mathbb{R}^+$ (estimate of $\|\mathbf{A}^T\mathbf{A}\|_2$) |

---

## 4. ALGORITHM STEPS

### Initialization
```
BEGIN INITIALIZATION
```

| Step | step_name | formula | inputs | outputs | computational_note |
|------|-----------|---------|--------|---------|-------------------|
| I.1 | Initialize primal variable | $x^{(0)} = \mathbf{0} \in \mathbb{R}^{D \times H \times W \times C}$ | PSF shape | $x^{(0)}$ | Zero array with same shape as PSF |
| I.2 | Initialize extrapolation variable | $z^{(0)} = x^{(0)}$ | $x^{(0)}$ | $z^{(0)}$ | Copy of initial estimate |
| I.3 | Initialize momentum parameter | $t^{(0)} = 1.0$ | — | $t^{(0)}$ | Scalar initialization |
| I.4 | Compute step size | $\alpha = 1/L$ | $L$ | $\alpha$ | If $L = 0$, set $\alpha = 1$ |

```
END INITIALIZATION
```

### Iteration Loop
```
BEGIN ITERATION LOOP (for k = 0, 1, ..., N_iter - 1)
```

| Step | step_name | formula | inputs | outputs | computational_note |
|------|-----------|---------|--------|---------|-------------------|
| L.1 | Pad extrapolation variable | $z^{(k)}_{\text{pad}}[d,i,j,c] = \begin{cases} z^{(k)}[d, i-s_0, j-s_1, c] & \text{if } s_0 \leq i < e_0, s_1 \leq j < e_1 \\ 0 & \text{otherwise} \end{cases}$ | $z^{(k)} \in \mathbb{R}^{D \times H \times W \times C}$, $(s_0, s_1)$, $(e_0, e_1)$ | $z^{(k)}_{\text{pad}} \in \mathbb{R}^{D \times H_p \times W_p \times C}$ | Zero-padding for FFT |
| L.2 | Forward FFT of extrapolation | $\hat{z}^{(k)} = \mathcal{F}_{2D}[z^{(k)}_{\text{pad}}]$ | $z^{(k)}_{\text{pad}}$ | $\hat{z}^{(k)} \in \mathbb{C}^{D \times H_p \times (\lfloor W_p/2\rfloor+1) \times C}$ | Orthonormalized rFFT2 |
| L.3 | Frequency-domain multiplication (forward) | $\hat{w}^{(k)} = \hat{z}^{(k)} \odot \mathbf{H}$ | $\hat{z}^{(k)}$, $\mathbf{H}$ | $\hat{w}^{(k)}$ | Element-wise complex multiplication |
| L.4 | Inverse FFT for forward model | $w^{(k)}_{\text{pad}} = \texttt{ifftshift}\left(\mathcal{F}^{-1}_{2D}[\hat{w}^{(k)}]\right)$ | $\hat{w}^{(k)}$ | $w^{(k)}_{\text{pad}} \in \mathbb{R}^{D \times H_p \times W_p \times C}$ | ifftshift along spatial axes |
| L.5 | Crop forward result | $(\mathbf{A}z^{(k)})[d,i,j,c] = w^{(k)}_{\text{pad}}[d, i+s_0, j+s_1, c]$ for $i \in [0,H)$, $j \in [0,W)$ | $w^{(k)}_{\text{pad}}$, $(s_0,s_1)$, $(e_0,e_1)$ | $\mathbf{A}z^{(k)} \in \mathbb{R}^{D \times H \times W \times C}$ | Extract real part and crop |
| L.6 | Compute data residual | $r^{(k)} = \mathbf{A}z^{(k)} - y$ | $\mathbf{A}z^{(k)}$, $y$ | $r^{(k)} \in \mathbb{R}^{D \times H \times W \times C}$ | Element-wise subtraction |
| L.7 | Pad residual | $r^{(k)}_{\text{pad}}[d,i,j,c] = \begin{cases} r^{(k)}[d, i-s_0, j-s_1, c] & \text{if } s_0 \leq i < e_0, s_1 \leq j < e_1 \\ 0 & \text{otherwise} \end{cases}$ | $r^{(k)}$ | $r^{(k)}_{\text{pad}} \in \mathbb{R}^{D \times H_p \times W_p \times C}$ | Zero-padding for adjoint FFT |
| L.8 | Forward FFT of residual | $\hat{r}^{(k)} = \mathcal{F}_{2D}[r^{(k)}_{\text{pad}}]$ | $r^{(k)}_{\text{pad}}$ | $\hat{r}^{(k)}$ | Orthonormalized rFFT2 |
| L.9 | Frequency-domain multiplication (adjoint) | $\hat{g}^{(k)} = \hat{r}^{(k)} \odot \mathbf{H}^*$ | $\hat{r}^{(k)}$, $\mathbf{H}^* = \overline{\mathbf{H}}$ | $\hat{g}^{(k)}$ | Complex conjugate for adjoint |
| L.10 | Inverse FFT for gradient | $g^{(k)}_{\text{pad}} = \texttt{ifftshift}\left(\mathcal{F}^{-1}_{2D}[\hat{g}^{(k)}]\right)$ | $\hat{g}^{(k)}$ | $g^{(k)}_{\text{pad}} \in \mathbb{R}^{D \times H_p \times W_p \times C}$ | ifftshift along spatial axes |
| L.11 | Crop gradient | $\nabla f(z^{(k)})[d,i,j,c] = g^{(k)}_{\text{pad}}[d, i+s_0, j+s_1, c]$ | $g^{(k)}_{\text{pad}}$ | $\nabla f(z^{(k)}) = \mathbf{A}^T(\mathbf{A}z^{(k)} - y)$ | Gradient of data fidelity |
| L.12 | Gradient descent step | $\tilde{x}^{(k+1)} = z^{(k)} - \alpha \cdot \nabla f(z^{(k)})$ | $z^{(k)}$, $\nabla f(z^{(k)})$, $\alpha$ | $\tilde{x}^{(k+1)} \in \mathbb{R}^{D \times H \times W \times C}$ | Unconstrained update |
| L.13 | Proximal operator (non-negativity) | $x^{(k+1)} = \text{prox}_{\mathbb{R}_+}(\tilde{x}^{(k+1)}) = \max(\tilde{x}^{(k+1)}, 0)$ | $\tilde{x}^{(k+1)}$ | $x^{(k+1)} \in \mathbb{R}_+^{D \times H \times W \times C}$ | Element-wise maximum with zero |
| L.14 | Update momentum parameter | $t^{(k+1)} = \frac{1 + \sqrt{1 + 4(t^{(k)})^2}}{2}$ | $t^{(k)}$ | $t^{(k+1)}$ | FISTA momentum schedule |
| L.15 | Compute extrapolation coefficient | $\beta^{(k)} = \frac{t^{(k)} - 1}{t^{(k+1)}}$ | $t^{(k)}$, $t^{(k+1)}$ | $\beta^{(k)} \in [0, 1)$ | Nesterov momentum weight |
| L.16 | Extrapolation step | $z^{(k+1)} = x^{(k+1)} + \beta^{(k)} \cdot (x^{(k+1)} - x^{(k)})$ | $x^{(k+1)}$, $x^{(k)}$, $\beta^{(k)}$ | $z^{(k+1)}$ | Momentum-based extrapolation |
| L.17 | Convergence check | Check if $k + 1 \geq N_{\text{iter}}$ | $k$, $N_{\text{iter}}$ | Boolean | Iteration limit only (no early stopping) |

```
END ITERATION LOOP
```

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $k \geq N_{\text{iter}}$ |
| parameters | $N_{\text{iter}} = 50$ (default), configurable via `n_iter` argument |

---

## 6. EVALUATION METRICS

### Metric 6.1: Reconstruction Value Range
| Field | Value |
|-------|-------|
| name | Value Statistics |
| formula | $x_{\min} = \min_{d,i,j,c} x^{(N_{\text{iter}})}_{d,i,j,c}$, $x_{\max} = \max_{d,i,j,c} x^{(N_{\text{iter}})}_{d,i,j,c}$, $\bar{x} = \frac{1}{D \cdot H \cdot W \cdot C} \sum_{d,i,j,c} x^{(N_{\text{iter}})}_{d,i,j,c}$ |
| description | Minimum, maximum, and mean intensity values of the reconstructed image |

### Metric 6.2: Display Normalization
| Field | Value |
|-------|-------|
| name | Display Normalization |
| formula | $x_{\text{display}} = \text{clip}\left(\frac{x^{(N_{\text{iter}})} - x_{\min}}{x_{\max} - x_{\min}}, 0, 1\right)$ where $\text{clip}(v, a, b) = \max(a, \min(v, b))$ |
| description | Normalized image for visualization, scaled to $[0,1]$ and clipped |

### Metric 6.3: Lipschitz Constant
| Field | Value |
|-------|-------|
| name | Lipschitz Constant |
| formula | $L \approx \|\mathbf{A}^T\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A}^T\mathbf{A})$ estimated via power iteration |
| description | Spectral norm of the Hessian, determines step size for guaranteed convergence |