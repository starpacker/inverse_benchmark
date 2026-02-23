# Algorithmic Specification Document: ADMM-Based Lensless Image Reconstruction

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{x \geq 0} \frac{1}{2}\|Hx - y\|_2^2 + \tau \|\Psi x\|_1$$

Subject to the ADMM reformulation with splitting variables:
$$\min_{x, X, U, W} \frac{1}{2}\|X - y\|_2^2 + \tau \|U\|_1 \quad \text{s.t.} \quad Hx = X, \; \Psi x = U, \; x = W, \; W \geq 0$$

**data_fidelity_term**:
$$\mathcal{L}_{\text{data}}(X, y) = \frac{1}{2}\|X - y\|_2^2 = \frac{1}{2}\sum_{d,i,j,c}(X_{d,i,j,c} - y_{d,i,j,c})^2$$

**regularization_term**:
$$\mathcal{R}(U) = \tau \|U\|_1 = \tau \sum_{d,i,j,c,k} |U_{d,i,j,c,k}|$$

where $k \in \{0,1\}$ indexes horizontal and vertical gradient components.

**constraints**:
| Constraint | Expression | Description |
|------------|------------|-------------|
| Forward model coupling | $Hx = X$ | Convolution constraint |
| TV coupling | $\Psi x = U$ | Finite difference constraint |
| Non-negativity coupling | $x = W$ | Auxiliary variable constraint |
| Non-negativity | $W \geq 0$ | Element-wise non-negativity |

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $x$ | primal | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Estimated image in padded domain |
| $X$ | primal (split) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Splitting variable for forward model |
| $U$ | primal (split) | $[1, D, H_p, W_p, C, 2]$ | float32 | $\mathbf{0}$ | Splitting variable for TV regularization |
| $W$ | primal (split) | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Splitting variable for non-negativity |

### Dual Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\xi$ | dual | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Lagrange multiplier for $Hx = X$ |
| $\eta$ | dual | $[1, D, H_p, W_p, C, 2]$ | float32 | $\mathbf{0}$ | Lagrange multiplier for $\Psi x = U$ |
| $\rho$ | dual | $[1, D, H_p, W_p, C]$ | float32 | $\mathbf{0}$ | Lagrange multiplier for $x = W$ |

### Constants
| Symbol | Category | Shape | Dtype | Value | Description |
|--------|----------|-------|-------|-------|-------------|
| $\mu_1$ | constant | scalar | float32 | $10^{-6}$ | Penalty parameter for forward model |
| $\mu_2$ | constant | scalar | float32 | $10^{-5}$ | Penalty parameter for TV |
| $\mu_3$ | constant | scalar | float32 | $4 \times 10^{-5}$ | Penalty parameter for non-negativity |
| $\tau$ | constant | scalar | float32 | $10^{-4}$ | TV regularization weight |
| $n_{\text{iter}}$ | constant | scalar | int | 50 | Maximum iterations |

### Observations and PSF
| Symbol | Category | Shape | Dtype | Description |
|--------|----------|-------|-------|-------------|
| $y$ | observation | $[D, H, W, C]$ | float32 | Measured lensless image |
| $h$ | constant | $[D, H, W, C]$ | float32 | Point spread function |
| $y_p$ | derived | $[1, D, H_p, W_p, C]$ | float32 | Zero-padded measurement |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Data Loading and Normalization
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load and Normalize Data |
| formula | $y \leftarrow \frac{y_{\text{raw}}}{\max(y_{\text{raw}})}, \quad h \leftarrow \frac{h_{\text{raw}}}{\max(h_{\text{raw}})}$ |
| inputs | $y_{\text{raw}}$: raw measurement $[H_0, W_0, C]$; $h_{\text{raw}}$: raw PSF $[H_0, W_0, C]$; downsample factor $s=4$ |
| outputs | $y$: normalized measurement $[D, H, W, C]$; $h$: normalized PSF $[D, H, W, C]$ where $H = \lfloor H_0/s \rfloor$, $W = \lfloor W_0/s \rfloor$ |

### Step 3.2: Compute Padded Dimensions
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Compute FFT-Optimal Padded Shape |
| formula | $H_p = \text{next\_fast\_len}(2H - 1), \quad W_p = \text{next\_fast\_len}(2W - 1)$ |
| inputs | PSF shape $[D, H, W, C]$ |
| outputs | Padded shape $[D, H_p, W_p, C]$ |

**Definition**: $\text{next\_fast\_len}(n)$ returns the smallest integer $\geq n$ that can be factored into small primes (2, 3, 5) for FFT efficiency.

### Step 3.3: Compute Padding Indices
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Compute Start/End Indices for Centered Padding |
| formula | $i_{\text{start}} = \lfloor(H_p - H)/2\rfloor, \quad j_{\text{start}} = \lfloor(W_p - W)/2\rfloor$ |
| | $i_{\text{end}} = i_{\text{start}} + H, \quad j_{\text{end}} = j_{\text{start}} + W$ |
| inputs | $H_p, W_p, H, W$ |
| outputs | $(i_{\text{start}}, i_{\text{end}}), (j_{\text{start}}, j_{\text{end}})$ |

### Step 3.4: Pad PSF and Compute Frequency Response
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Compute PSF Frequency Response |
| formula | $h_p = \text{Pad}(h)$, where $[\text{Pad}(v)]_{d,i,j,c} = \begin{cases} v_{d,i-i_{\text{start}},j-j_{\text{start}},c} & i_{\text{start}} \leq i < i_{\text{end}}, j_{\text{start}} \leq j < j_{\text{end}} \\ 0 & \text{otherwise} \end{cases}$ |
| | $\hat{H} = \mathcal{F}_{\text{2D}}(h_p)$, where $\mathcal{F}_{\text{2D}}$ is 2D rFFT with orthonormal normalization |
| | $\hat{H}^* = \text{conj}(\hat{H})$ |
| inputs | $h$: PSF $[D, H, W, C]$ |
| outputs | $\hat{H}$: frequency response $[D, H_p, W_p/2+1, C]$ (complex64); $\hat{H}^*$: adjoint frequency response |

**Definition (Orthonormal 2D rFFT)**:
$$[\mathcal{F}_{\text{2D}}(v)]_{d,k,l,c} = \frac{1}{\sqrt{H_p W_p}} \sum_{i=0}^{H_p-1}\sum_{j=0}^{W_p-1} v_{d,i,j,c} \exp\left(-2\pi \mathbf{i}\left(\frac{ki}{H_p} + \frac{lj}{W_p}\right)\right)$$

### Step 3.5: Compute TV Gram Matrix in Frequency Domain
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Precompute TV Gram Operator |
| formula | Define kernel $g \in \mathbb{R}^{D \times H_p \times W_p \times C}$: |
| | If $D=1$: $g_{0,0,0,c}=4$, $g_{0,0,1,c}=g_{0,0,-1,c}=g_{0,1,0,c}=g_{0,-1,0,c}=-1$, else $0$ |
| | If $D>1$: $g_{0,0,0,c}=6$, add $g_{1,0,0,c}=g_{-1,0,0,c}=-1$ |
| | $\hat{\Psi}^T\hat{\Psi} = \mathcal{F}_{\text{2D}}(g)$ |
| inputs | Padded shape $[D, H_p, W_p, C]$, dtype |
| outputs | $\hat{\Psi}^T\hat{\Psi}$: TV Gram in frequency domain $[D, H_p, W_p/2+1, C]$ (complex64) |

### Step 3.6: Precompute Division Matrices
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Precompute Inversion Denominators |
| formula | $\hat{R}^{-1} = \frac{1}{\mu_1 |\hat{H}^* \hat{H}| + \mu_2 |\hat{\Psi}^T\hat{\Psi}| + \mu_3}$ |
| | $M_X = \frac{1}{\mathbf{1}_{\text{pad}} + \mu_1}$, where $\mathbf{1}_{\text{pad}} = \text{Pad}(\mathbf{1}_{[D,H,W,C]})$ |
| inputs | $\hat{H}$, $\hat{\Psi}^T\hat{\Psi}$, $\mu_1, \mu_2, \mu_3$ |
| outputs | $\hat{R}^{-1}$: frequency-domain division matrix (complex64); $M_X$: spatial-domain division matrix (float32) |

### Step 3.7: Pad Measurement
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Pad Measurement to FFT Domain |
| formula | $y_p = \text{Pad}(y)$ |
| inputs | $y$: measurement $[D, H, W, C]$ |
| outputs | $y_p$: padded measurement $[1, D, H_p, W_p, C]$ |

---

## 4. ALGORITHM STEPS

### Initialization
| Field | Value |
|-------|-------|
| step_name | Initialize All Variables |
| formula | $x^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ |
| | $X^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ |
| | $U^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C,2]}$ |
| | $W^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ |
| | $\xi^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ |
| | $\eta^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C,2]}$ |
| | $\rho^{(0)} = \mathbf{0}_{[1,D,H_p,W_p,C]}$ |
| inputs | Padded shape |
| outputs | All initialized variables |

---

### BEGIN ITERATION LOOP (for $k = 0, 1, \ldots, n_{\text{iter}}-1$)

#### Step 4.1: Compute Finite Differences
| Field | Value |
|-------|-------|
| step_name | Compute TV Gradient of Current Estimate |
| formula | $[\Psi x^{(k)}]_{d,i,j,c,0} = x^{(k)}_{d,(i+1)\mod H_p,j,c} - x^{(k)}_{d,i,j,c}$ |
| | $[\Psi x^{(k)}]_{d,i,j,c,1} = x^{(k)}_{d,i,(j+1)\mod W_p,c} - x^{(k)}_{d,i,j,c}$ |
| inputs | $x^{(k)}$: current image estimate $[1,D,H_p,W_p,C]$ |
| outputs | $\Psi x^{(k)}$: finite differences $[1,D,H_p,W_p,C,2]$ |
| computational_note | Roll with periodic boundary conditions |

#### Step 4.2: U Update (TV Soft Thresholding)
| Field | Value |
|-------|-------|
| step_name | Soft Thresholding for TV Proximal Operator |
| formula | $v = \Psi x^{(k)} + \frac{\eta^{(k)}}{\mu_2}$ |
| | $U^{(k+1)} = \text{sign}(v) \odot \max\left(|v| - \frac{\tau}{\mu_2}, 0\right)$ |
| inputs | $\Psi x^{(k)}$, $\eta^{(k)}$, $\mu_2$, $\tau$ |
| outputs | $U^{(k+1)}$: updated TV auxiliary variable $[1,D,H_p,W_p,C,2]$ |
| computational_note | Element-wise operations; $\text{sign}(0) = 0$ |

**Definition (Soft Thresholding)**:
$$\mathcal{S}_\lambda(v) = \text{sign}(v) \cdot \max(|v| - \lambda, 0)$$

#### Step 4.3: Forward Convolution for X Update
| Field | Value |
|-------|-------|
| step_name | Apply Forward Model |
| formula | $\hat{x} = \mathcal{F}_{\text{2D}}(x^{(k)})$ |
| | $\hat{z} = \hat{x} \odot \hat{H}$ |
| | $z = \text{ifftshift}\left(\mathcal{F}_{\text{2D}}^{-1}(\hat{z})\right)$ |
| | $(Hx^{(k)}) = z$ |
| inputs | $x^{(k)}$, $\hat{H}$ |
| outputs | $Hx^{(k)}$: convolution result $[1,D,H_p,W_p,C]$ |
| computational_note | ifftshift applied along spatial axes $(-3, -2)$ |

**Definition (Inverse 2D rFFT)**:
$$[\mathcal{F}_{\text{2D}}^{-1}(\hat{v})]_{d,i,j,c} = \frac{1}{\sqrt{H_p W_p}} \sum_{k=0}^{H_p-1}\sum_{l=0}^{W_p/2} \hat{v}_{d,k,l,c} \exp\left(2\pi \mathbf{i}\left(\frac{ki}{H_p} + \frac{lj}{W_p}\right)\right)$$

**Definition (ifftshift)**:
$$[\text{ifftshift}(v)]_{d,i,j,c} = v_{d,(i+\lfloor H_p/2\rfloor)\mod H_p,(j+\lfloor W_p/2\rfloor)\mod W_p,c}$$

#### Step 4.4: X Update (Data Fidelity Variable)
| Field | Value |
|-------|-------|
| step_name | Update Data Fidelity Auxiliary Variable |
| formula | $X^{(k+1)} = M_X \odot \left(\xi^{(k)} + \mu_1 (Hx^{(k)}) + y_p\right)$ |
| inputs | $M_X$, $\xi^{(k)}$, $Hx^{(k)}$, $y_p$, $\mu_1$ |
| outputs | $X^{(k+1)}$: updated auxiliary variable $[1,D,H_p,W_p,C]$ |
| computational_note | Element-wise multiplication |

#### Step 4.5: W Update (Non-negativity Projection)
| Field | Value |
|-------|-------|
| step_name | Project onto Non-negative Orthant |
| formula | $W^{(k+1)} = \max\left(\frac{\rho^{(k)}}{\mu_3} + x^{(k)}, 0\right)$ |
| inputs | $\rho^{(k)}$, $x^{(k)}$, $\mu_3$ |
| outputs | $W^{(k+1)}$: non-negative auxiliary variable $[1,D,H_p,W_p,C]$ |
| computational_note | Element-wise maximum with zero |

#### Step 4.6: Compute Adjoint Finite Differences
| Field | Value |
|-------|-------|
| step_name | Compute Adjoint TV Operator |
| formula | $[\Psi^T U]_{d,i,j,c} = \left(U_{d,(i-1)\mod H_p,j,c,0} - U_{d,i,j,c,0}\right) + \left(U_{d,i,(j-1)\mod W_p,c,1} - U_{d,i,j,c,1}\right)$ |
| inputs | $U^{(k+1)}$: TV auxiliary $[1,D,H_p,W_p,C,2]$ |
| outputs | $\Psi^T U$: adjoint result $[1,D,H_p,W_p,C]$ |
| computational_note | Roll with negative shift |

#### Step 4.7: Compute Adjoint Convolution
| Field | Value |
|-------|-------|
| step_name | Apply Adjoint Forward Model |
| formula | $\hat{q} = \mathcal{F}_{\text{2D}}(\mu_1 X^{(k+1)} - \xi^{(k)})$ |
| | $\hat{r} = \hat{q} \odot \hat{H}^*$ |
| | $H^T(\mu_1 X^{(k+1)} - \xi^{(k)}) = \text{ifftshift}\left(\mathcal{F}_{\text{2D}}^{-1}(\hat{r})\right)$ |
| inputs | $X^{(k+1)}$, $\xi^{(k)}$, $\hat{H}^*$, $\mu_1$ |
| outputs | Adjoint convolution result $[1,D,H_p,W_p,C]$ |

#### Step 4.8: Assemble Right-Hand Side
| Field | Value |
|-------|-------|
| step_name | Compute RHS for Image Update |
| formula | $r^{(k)} = (\mu_3 W^{(k+1)} - \rho^{(k)}) + \Psi^T(\mu_2 U^{(k+1)} - \eta^{(k)}) + H^T(\mu_1 X^{(k+1)} - \xi^{(k)})$ |
| inputs | $W^{(k+1)}$, $U^{(k+1)}$, $X^{(k+1)}$, $\rho^{(k)}$, $\eta^{(k)}$, $\xi^{(k)}$, $\mu_1, \mu_2, \mu_3$ |
| outputs | $r^{(k)}$: RHS vector $[1,D,H_p,W_p,C]$ |

#### Step 4.9: Image Update (Frequency Domain Solve)
| Field | Value |
|-------|-------|
| step_name | Solve Linear System in Frequency Domain |
| formula | $\hat{r}^{(k)} = \mathcal{F}_{\text{2D}}(r^{(k)})$ |
| | $\hat{x}^{(k+1)} = \hat{R}^{-1} \odot \hat{r}^{(k)}$ |
| | $x^{(k+1)} = \mathcal{F}_{\text{2D}}^{-1}(\hat{x}^{(k+1)})$ |
| inputs | $r^{(k)}$, $\hat{R}^{-1}$ |
| outputs | $x^{(k+1)}$: updated image estimate $[1,D,H_p,W_p,C]$ |
| computational_note | Solves $(\mu_1 H^T H + \mu_2 \Psi^T\Psi + \mu_3 I)x = r$ via FFT diagonalization |

#### Step 4.10: Recompute Forward Operations for Dual Update
| Field | Value |
|-------|-------|
| step_name | Recompute Constraints for Dual Updates |
| formula | $Hx^{(k+1)} = \text{ifftshift}\left(\mathcal{F}_{\text{2D}}^{-1}\left(\mathcal{F}_{\text{2D}}(x^{(k+1)}) \odot \hat{H}\right)\right)$ |
| | $\Psi x^{(k+1)} = \text{finite\_diff}(x^{(k+1)})$ |
| inputs | $x^{(k+1)}$, $\hat{H}$ |
| outputs | $Hx^{(k+1)}$, $\Psi x^{(k+1)}$ |

#### Step 4.11: Dual Variable Updates
| Field | Value |
|-------|-------|
| step_name | Update Lagrange Multipliers |
| formula | $\xi^{(k+1)} = \xi^{(k)} + \mu_1 (Hx^{(k+1)} - X^{(k+1)})$ |
| | $\eta^{(k+1)} = \eta^{(k)} + \mu_2 (\Psi x^{(k+1)} - U^{(k+1)})$ |
| | $\rho^{(k+1)} = \rho^{(k)} + \mu_3 (x^{(k+1)} - W^{(k+1)})$ |
| inputs | $\xi^{(k)}, \eta^{(k)}, \rho^{(k)}$, constraint residuals, $\mu_1, \mu_2, \mu_3$ |
| outputs | $\xi^{(k+1)}, \eta^{(k+1)}, \rho^{(k+1)}$ |
| computational_note | Standard ADMM dual ascent step |

#### Step 4.12: Convergence Check
| Field | Value |
|-------|-------|
| step_name | Check Iteration Count |
| formula | Continue if $k < n_{\text{iter}} - 1$ |
| inputs | $k$, $n_{\text{iter}}$ |
| outputs | Loop continuation flag |
| computational_note | No residual-based convergence check implemented |

### END ITERATION LOOP

---

### Post-processing

#### Step 4.13: Crop Result
| Field | Value |
|-------|-------|
| step_name | Extract Valid Region from Padded Result |
| formula | $x_{\text{out}} = x^{(n_{\text{iter}})}_{:, :, i_{\text{start}}:i_{\text{end}}, j_{\text{start}}:j_{\text{end}}, :}$ |
| inputs | $x^{(n_{\text{iter}})}$, crop indices |
| outputs | $x_{\text{out}}$: final reconstruction $[D, H, W, C]$ |

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $k \geq n_{\text{iter}}$ |
| parameters | $n_{\text{iter}} = 50$ (default), configurable |

---

## 6. EVALUATION METRICS

| Name | Formula | Description |
|------|---------|-------------|
| Minimum Value | $x_{\min} = \min_{d,i,j,c} x_{\text{out},d,i,j,c}$ | Minimum pixel intensity in reconstruction |
| Maximum Value | $x_{\max} = \max_{d,i,j,c} x_{\text{out},d,i,j,c}$ | Maximum pixel intensity in reconstruction |
| Mean | $\bar{x} = \frac{1}{N}\sum_{d,i,j,c} x_{\text{out},d,i,j,c}$ | Average pixel intensity, where $N = D \cdot H \cdot W \cdot C$ |
| Standard Deviation | $\sigma_x = \sqrt{\frac{1}{N}\sum_{d,i,j,c}(x_{\text{out},d,i,j,c} - \bar{x})^2}$ | Spread of pixel intensities |