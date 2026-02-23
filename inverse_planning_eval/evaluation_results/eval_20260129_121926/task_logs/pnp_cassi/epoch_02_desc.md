# Algorithmic Specification Document: GAP-TV for CASSI Reconstruction

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\min_{\mathbf{X} \in \mathbb{R}^{r \times c \times n_C}} \frac{1}{2}\|\mathbf{y} - \mathcal{A}(\mathbf{X})\|_2^2 + \tau \sum_{i=1}^{n_C} \text{TV}(\mathbf{X}_{:,:,i})$$

where the Total Variation is the isotropic 2D TV:
$$\text{TV}(\mathbf{U}) = \sum_{i,j} \sqrt{(\nabla_h \mathbf{U})_{i,j}^2 + (\nabla_v \mathbf{U})_{i,j}^2}$$

**data_fidelity_term**:
$$\mathcal{D}(\mathbf{X}) = \frac{1}{2}\|\mathbf{y} - \mathcal{A}(\mathbf{X})\|_2^2 = \frac{1}{2}\sum_{i,j}\left(y_{i,j} - \sum_{k=1}^{n_C} \Phi_{i,j,k} \cdot \mathcal{S}_k(\mathbf{X}_{:,:,k})_{i,j}\right)^2$$

**regularization_term**:
$$\mathcal{R}(\mathbf{X}) = \tau \sum_{k=1}^{n_C} \sum_{i,j} \sqrt{(X_{i,j+1,k} - X_{i,j,k})^2 + (X_{i+1,j,k} - X_{i,j,k})^2}$$

**constraints**:
- $X_{i,j,k} \in [0, 1]$ (soft constraint via normalization)

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{X}$ | primal | $(r, c, n_C)$ | float64 | $\mathcal{S}^{-1}(\mathbf{\Phi}^T \mathbf{y})$ | Reconstructed spectral image cube |
| $\tilde{\mathbf{X}}$ | primal | $(r, c + (n_C-1)s, n_C)$ | float64 | $\mathbf{\Phi}^T \mathbf{y}$ | Shifted spectral cube in measurement domain |

### Dual Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}_1$ | dual | $(r, c + (n_C-1)s)$ | float64 | $\mathbf{0}$ | Accumulated residual (GAP dual variable) |
| $\mathbf{p}_1$ | dual | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Horizontal dual variable for TV denoiser |
| $\mathbf{p}_2$ | dual | $(r, c, n_C)$ | float64 | $\mathbf{0}$ | Vertical dual variable for TV denoiser |

### Constants
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\lambda$ | constants | scalar | float64 | 1.0 | GAP step size parameter |
| $\tau$ | constants | scalar | float64 | 6.0 | TV regularization weight |
| $s$ | constants | scalar | int | 1 | Spectral dispersion step size |
| $K$ | constants | scalar | int | 20 | Maximum outer iterations |
| $K_{TV}$ | constants | scalar | int | 5 | TV denoiser iterations |
| $\delta t$ | constants | scalar | float64 | 0.25 | TV denoiser time step |

### Observations and Sensing
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $\mathbf{y}$ | observations | $(r, c + (n_C-1)s)$ | float64 | from data | Compressed measurement |
| $\mathbf{\Phi}$ | constants | $(r, c + (n_C-1)s, n_C)$ | float64 | from mask | 3D coded aperture mask |
| $\mathbf{\Phi}_{\Sigma}$ | constants | $(r, c + (n_C-1)s)$ | float64 | $\sum_{k=1}^{n_C} \Phi_{:,:,k}$ | Sum of masks across spectral bands |
| $\mathbf{X}^*$ | observations | $(r, c, n_C)$ | float64 | from data | Ground truth (for evaluation only) |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Ground Truth Normalization
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | normalize_truth |
| formula | $\mathbf{X}^* \leftarrow \frac{\mathbf{X}^*_{raw}}{255}$ if $\max(\mathbf{X}^*_{raw}) > 1$, else $\mathbf{X}^* \leftarrow \mathbf{X}^*_{raw}$ |
| inputs | $\mathbf{X}^*_{raw} \in \mathbb{R}^{r \times c \times n_C}$ |
| outputs | $\mathbf{X}^* \in [0,1]^{r \times c \times n_C}$ |

### Step 3.2: 3D Mask Construction
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | construct_3d_mask |
| formula | $\Phi_{i,j,k} = \begin{cases} M^{2D}_{i, j-k} & \text{if } k \leq j < k + c \\ 0 & \text{otherwise} \end{cases}$ for $k \in \{0, \ldots, n_C-1\}$ |
| inputs | $\mathbf{M}^{2D} \in \{0,1\}^{r \times c}$ (2D base mask) |
| outputs | $\mathbf{\Phi} \in \{0,1\}^{r \times (c + (n_C-1)s) \times n_C}$ |

### Step 3.3: Spectral Shift (Dispersion Simulation)
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | shift_truth |
| formula | $\mathcal{S}_k(\mathbf{X}^*_{:,:,k})_{i,j} = \begin{cases} X^*_{i, j - ks, k} & \text{if } ks \leq j < ks + c \\ 0 & \text{otherwise} \end{cases}$ |
| inputs | $\mathbf{X}^* \in \mathbb{R}^{r \times c \times n_C}$ |
| outputs | $\tilde{\mathbf{X}}^* \in \mathbb{R}^{r \times (c + (n_C-1)s) \times n_C}$ |

### Step 3.4: Measurement Synthesis
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | generate_measurement |
| formula | $y_{i,j} = \sum_{k=1}^{n_C} \Phi_{i,j,k} \cdot \tilde{X}^*_{i,j,k}$ |
| inputs | $\mathbf{\Phi} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$, $\tilde{\mathbf{X}}^* \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| outputs | $\mathbf{y} \in \mathbb{R}^{r \times (c+(n_C-1)s)}$ |

### Step 3.5: Mask Sum Computation
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | compute_mask_sum |
| formula | $(\Phi_\Sigma)_{i,j} = \max\left(\sum_{k=1}^{n_C} \Phi_{i,j,k}, 1\right)$ |
| inputs | $\mathbf{\Phi} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| outputs | $\mathbf{\Phi}_\Sigma \in \mathbb{R}^{r \times (c+(n_C-1)s)}$ with all entries $\geq 1$ |

---

## 4. ALGORITHM STEPS

### Initialization

| Field | Value |
|-------|-------|
| step_name | initialize_primal |
| formula | $\tilde{\mathbf{X}}^{(0)} = \mathcal{A}^T(\mathbf{y})$ where $(\mathcal{A}^T(\mathbf{y}))_{i,j,k} = y_{i,j} \cdot \Phi_{i,j,k}$ |
| inputs | $\mathbf{y} \in \mathbb{R}^{r \times (c+(n_C-1)s)}$, $\mathbf{\Phi} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| outputs | $\tilde{\mathbf{X}}^{(0)} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| computational_note | Element-wise broadcast of 2D measurement to 3D cube |

| Field | Value |
|-------|-------|
| step_name | initialize_dual |
| formula | $\mathbf{y}_1^{(0)} = \mathbf{0}_{r \times (c+(n_C-1)s)}$ |
| inputs | Shape parameters $r$, $c$, $n_C$, $s$ |
| outputs | $\mathbf{y}_1^{(0)} \in \mathbb{R}^{r \times (c+(n_C-1)s)}$ |
| computational_note | Zero initialization |

---

### Iteration Loop

**BEGIN LOOP** for $k = 0, 1, \ldots, K-1$:

---

#### Step 4.1: Forward Projection
| Field | Value |
|-------|-------|
| step_name | forward_projection |
| formula | $\mathbf{y}_b^{(k)} = \mathcal{A}(\tilde{\mathbf{X}}^{(k)}) = \sum_{l=1}^{n_C} \tilde{X}^{(k)}_{:,:,l} \odot \Phi_{:,:,l}$ |
| inputs | $\tilde{\mathbf{X}}^{(k)} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$, $\mathbf{\Phi}$ |
| outputs | $\mathbf{y}_b^{(k)} \in \mathbb{R}^{r \times (c+(n_C-1)s)}$ |
| computational_note | $\odot$ denotes element-wise multiplication; sum over spectral axis |

#### Step 4.2: Dual Variable Update
| Field | Value |
|-------|-------|
| step_name | dual_update |
| formula | $\mathbf{y}_1^{(k+1)} = \mathbf{y}_1^{(k)} + (\mathbf{y} - \mathbf{y}_b^{(k)})$ |
| inputs | $\mathbf{y}_1^{(k)}$, $\mathbf{y}$, $\mathbf{y}_b^{(k)}$ |
| outputs | $\mathbf{y}_1^{(k+1)} \in \mathbb{R}^{r \times (c+(n_C-1)s)}$ |
| computational_note | Accumulates measurement residual |

#### Step 4.3: GAP Data Projection
| Field | Value |
|-------|-------|
| step_name | gap_data_projection |
| formula | $\tilde{\mathbf{X}}^{(k+\frac{1}{2})} = \tilde{\mathbf{X}}^{(k)} + \lambda \cdot \mathcal{A}^T\left(\frac{\mathbf{y}_1^{(k+1)} - \mathbf{y}_b^{(k)}}{\mathbf{\Phi}_\Sigma}\right)$ |
| inputs | $\tilde{\mathbf{X}}^{(k)}$, $\mathbf{y}_1^{(k+1)}$, $\mathbf{y}_b^{(k)}$, $\mathbf{\Phi}_\Sigma$, $\mathbf{\Phi}$, $\lambda$ |
| outputs | $\tilde{\mathbf{X}}^{(k+\frac{1}{2})} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| computational_note | Division is element-wise; $\mathcal{A}^T$ broadcasts result to 3D |

**Sub-step 4.3.1: Residual Normalization**
$$\mathbf{r}^{(k)} = \frac{\mathbf{y}_1^{(k+1)} - \mathbf{y}_b^{(k)}}{\mathbf{\Phi}_\Sigma}$$

**Sub-step 4.3.2: Backward Operator Application**
$$\mathcal{A}^T(\mathbf{r}^{(k)})_{i,j,l} = r^{(k)}_{i,j} \cdot \Phi_{i,j,l}$$

**Sub-step 4.3.3: Scaled Addition**
$$\tilde{X}^{(k+\frac{1}{2})}_{i,j,l} = \tilde{X}^{(k)}_{i,j,l} + \lambda \cdot r^{(k)}_{i,j} \cdot \Phi_{i,j,l}$$

#### Step 4.4: Inverse Shift (Measurement → Image Domain)
| Field | Value |
|-------|-------|
| step_name | shift_back |
| formula | $\mathbf{X}^{(k+\frac{1}{2})} = \mathcal{S}^{-1}(\tilde{\mathbf{X}}^{(k+\frac{1}{2})})$ where $X^{(k+\frac{1}{2})}_{i,j,l} = \tilde{X}^{(k+\frac{1}{2})}_{i, j + ls, l}$ |
| inputs | $\tilde{\mathbf{X}}^{(k+\frac{1}{2})} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| outputs | $\mathbf{X}^{(k+\frac{1}{2})} \in \mathbb{R}^{r \times c \times n_C}$ |
| computational_note | Circular shift by $-ls$ pixels along column axis, then crop to first $c$ columns |

#### Step 4.5: TV Denoising (Chambolle's Algorithm)

**Input**: $\mathbf{X}^{(k+\frac{1}{2})}$, regularization weight $\tau$, iterations $K_{TV}$

**Sub-step 4.5.0: TV Denoiser Initialization**
| Field | Value |
|-------|-------|
| step_name | tv_init |
| formula | $\mathbf{p}_1^{(0)} = \mathbf{0}$, $\mathbf{p}_2^{(0)} = \mathbf{0}$, $\text{div}(\mathbf{p})^{(0)} = \mathbf{0}$ |
| inputs | Shape $(r, c, n_C)$ |
| outputs | $\mathbf{p}_1^{(0)}, \mathbf{p}_2^{(0)}, \text{div}(\mathbf{p})^{(0)} \in \mathbb{R}^{r \times c \times n_C}$ |

**BEGIN INNER LOOP** for $t = 0, 1, \ldots, K_{TV}-1$:

**Sub-step 4.5.1: Compute Auxiliary Variable**
| Field | Value |
|-------|-------|
| step_name | compute_z |
| formula | $\mathbf{Z}^{(t)} = \text{div}(\mathbf{p})^{(t)} - \tau \cdot \mathbf{X}^{(k+\frac{1}{2})}$ |
| inputs | $\text{div}(\mathbf{p})^{(t)}$, $\mathbf{X}^{(k+\frac{1}{2})}$, $\tau$ |
| outputs | $\mathbf{Z}^{(t)} \in \mathbb{R}^{r \times c \times n_C}$ |

**Sub-step 4.5.2: Compute Gradients with Neumann Boundary**
| Field | Value |
|-------|-------|
| step_name | compute_gradients |
| formula | $(\nabla_h \mathbf{Z}^{(t)})_{i,j,l} = Z^{(t)}_{i, \min(j+1, c-1), l} - Z^{(t)}_{i,j,l}$; $(\nabla_v \mathbf{Z}^{(t)})_{i,j,l} = Z^{(t)}_{\min(i+1, r-1), j, l} - Z^{(t)}_{i,j,l}$ |
| inputs | $\mathbf{Z}^{(t)} \in \mathbb{R}^{r \times c \times n_C}$ |
| outputs | $\nabla_h \mathbf{Z}^{(t)}, \nabla_v \mathbf{Z}^{(t)} \in \mathbb{R}^{r \times c \times n_C}$ |

**Sub-step 4.5.3: Compute Normalization Denominator**
| Field | Value |
|-------|-------|
| step_name | compute_denominator |
| formula | $D^{(t)}_{i,j} = 1 + \delta t \cdot \sqrt{\sum_{l=1}^{n_C} \left[(\nabla_h Z^{(t)}_{i,j,l})^2 + (\nabla_v Z^{(t)}_{i,j,l})^2\right]}$ |
| inputs | $\nabla_h \mathbf{Z}^{(t)}$, $\nabla_v \mathbf{Z}^{(t)}$, $\delta t$ |
| outputs | $\mathbf{D}^{(t)} \in \mathbb{R}^{r \times c}$ |
| computational_note | Sum over spectral dimension inside square root; replicate to 3D for division |

**Sub-step 4.5.4: Update Dual Variables**
| Field | Value |
|-------|-------|
| step_name | update_dual_p |
| formula | $(p_1^{(t+1)})_{i,j,l} = \frac{(p_1^{(t)})_{i,j,l} + \delta t \cdot (\nabla_h Z^{(t)})_{i,j,l}}{D^{(t)}_{i,j}}$; $(p_2^{(t+1)})_{i,j,l} = \frac{(p_2^{(t)})_{i,j,l} + \delta t \cdot (\nabla_v Z^{(t)})_{i,j,l}}{D^{(t)}_{i,j}}$ |
| inputs | $\mathbf{p}_1^{(t)}$, $\mathbf{p}_2^{(t)}$, $\nabla_h \mathbf{Z}^{(t)}$, $\nabla_v \mathbf{Z}^{(t)}$, $\mathbf{D}^{(t)}$, $\delta t$ |
| outputs | $\mathbf{p}_1^{(t+1)}, \mathbf{p}_2^{(t+1)} \in \mathbb{R}^{r \times c \times n_C}$ |

**Sub-step 4.5.5: Compute Divergence with Neumann Boundary**
| Field | Value |
|-------|-------|
| step_name | compute_divergence |
| formula | $\text{div}(\mathbf{p})^{(t+1)}_{i,j,l} = (p_1^{(t+1)})_{i,j,l} - (p_1^{(t+1)})_{i, \max(j-1,0), l} + (p_2^{(t+1)})_{i,j,l} - (p_2^{(t+1)})_{\max(i-1,0), j, l}$ |
| inputs | $\mathbf{p}_1^{(t+1)}$, $\mathbf{p}_2^{(t+1)}$ |
| outputs | $\text{div}(\mathbf{p})^{(t+1)} \in \mathbb{R}^{r \times c \times n_C}$ |

**END INNER LOOP**

**Sub-step 4.5.6: Recover Denoised Image**
| Field | Value |
|-------|-------|
| step_name | recover_denoised |
| formula | $\mathbf{X}^{(k+1)}_{img} = \mathbf{X}^{(k+\frac{1}{2})} - \frac{\text{div}(\mathbf{p})^{(K_{TV})}}{\tau}$ |
| inputs | $\mathbf{X}^{(k+\frac{1}{2})}$, $\text{div}(\mathbf{p})^{(K_{TV})}$, $\tau$ |
| outputs | $\mathbf{X}^{(k+1)}_{img} \in \mathbb{R}^{r \times c \times n_C}$ |

#### Step 4.6: Forward Shift (Image → Measurement Domain)
| Field | Value |
|-------|-------|
| step_name | shift_forward |
| formula | $\tilde{\mathbf{X}}^{(k+1)} = \mathcal{S}(\mathbf{X}^{(k+1)}_{img})$ where $\tilde{X}^{(k+1)}_{i,j,l} = \begin{cases} X^{(k+1)}_{i, j-ls, l} & \text{if } ls \leq j < ls + c \\ 0 & \text{otherwise} \end{cases}$ |
| inputs | $\mathbf{X}^{(k+1)}_{img} \in \mathbb{R}^{r \times c \times n_C}$ |
| outputs | $\tilde{\mathbf{X}}^{(k+1)} \in \mathbb{R}^{r \times (c+(n_C-1)s) \times n_C}$ |
| computational_note | Zero-pads extended regions |

#### Step 4.7: Compute Evaluation Metric
| Field | Value |
|-------|-------|
| step_name | compute_psnr |
| formula | $\text{PSNR}^{(k+1)} = 20 \log_{10}\left(\frac{1}{\sqrt{\frac{1}{r \cdot c \cdot n_C}\sum_{i,j,l}(X^*_{i,j,l} - X^{(k+1)}_{i,j,l})^2}}\right)$ |
| inputs | $\mathbf{X}^*$, $\mathbf{X}^{(k+1)}_{img}$ |
| outputs | $\text{PSNR}^{(k+1)} \in \mathbb{R}$ (dB) |
| computational_note | Computed only if ground truth is available |

**END LOOP**

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| type | iteration_limit |
| expression | $k \geq K$ |
| parameters | $K = 20$ (default), configurable via `iter_max` |

**Note**: No convergence-based stopping criterion is implemented. The algorithm runs for exactly $K$ iterations.

---

## 6. EVALUATION METRICS

### Peak Signal-to-Noise Ratio (PSNR)
| Field | Value |
|-------|-------|
| name | PSNR |
| formula | $\text{PSNR}(\mathbf{X}^*, \mathbf{X}) = 20 \log_{10}\left(\frac{\text{MAX}}{\sqrt{\text{MSE}}}\right)$ where $\text{MSE} = \frac{1}{N}\|\mathbf{X}^* - \mathbf{X}\|_F^2$, $\text{MAX} = 1.0$, $N = r \cdot c \cdot n_C$ |
| description | Measures reconstruction quality in decibels; higher is better. Returns 100 dB if MSE = 0. |

### Per-Iteration PSNR Trajectory
| Field | Value |
|-------|-------|
| name | PSNR_trajectory |
| formula | $\{\text{PSNR}^{(k)}\}_{k=1}^{K}$ |
| description | Sequence of PSNR values at each iteration for monitoring convergence behavior |