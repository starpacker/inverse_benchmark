# Algorithmic Specification Document: Light Field Microscopy Volume Reconstruction via Richardson-Lucy Deconvolution

---

## 1. OBJECTIVE FUNCTION

**full_expression**:
$$\hat{V} = \arg\max_{V \geq 0} \mathcal{L}(V) = \arg\max_{V \geq 0} \sum_{i} \left[ I_i \log(HV)_i - (HV)_i \right]$$

Equivalently, minimizing the Kullback-Leibler divergence:
$$\hat{V} = \arg\min_{V \geq 0} D_{KL}(I \| HV) = \arg\min_{V \geq 0} \sum_{i} \left[ I_i \log\frac{I_i}{(HV)_i} - I_i + (HV)_i \right]$$

**data_fidelity_term**:
$$D_{KL}(I \| HV) = \sum_{i=1}^{N_{img}} \left[ I_i \log\frac{I_i}{(HV)_i} + (HV)_i - I_i \right]$$

Where:
- $I \in \mathbb{R}^{N_x \times N_y}$ is the observed corrected light field image
- $H: \mathbb{R}^{T_x \times T_y \times N_d} \rightarrow \mathbb{R}^{N_x \times N_y}$ is the forward projection operator
- $V \in \mathbb{R}^{T_x \times T_y \times N_d}$ is the 3D volume to reconstruct

**regularization_term**:
$$R(V) = \sum_{j=1}^{N_d} \left\| V_{:,:,j} - \mathcal{F}^{-1}\left[ K_j \cdot \mathcal{F}[V_{:,:,j}] \right] \right\|_2^2$$

Applied implicitly via Lanczos anti-aliasing filtering at each iteration (when `filter_flag=True`). This acts as a depth-adaptive low-pass regularizer.

**constraints**:
| Constraint | Expression |
|------------|------------|
| Non-negativity | $V_{i,j,k} \geq 0 \quad \forall (i,j,k)$ |
| Finite values | $V_{i,j,k} \in \mathbb{R} \setminus \{\pm\infty, \text{NaN}\}$ |

---

## 2. VARIABLES

### Primal Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $V^{(n)}$ | primal | $(T_x, T_y, N_d)$ | float32 | $V^{(0)}_{i,j,k} = 1 \; \forall (i,j,k)$ | Reconstructed volume at iteration $n$ |

### Dual Variables
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| N/A | — | — | — | — | Algorithm does not use explicit dual variables |

### Constants / Operators
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $H$ | constants | operator | — | Precomputed via `LFM_computeLFMatrixOperators` | Forward projection operator (volume → image) |
| $H^T$ | constants | operator | — | Precomputed via `LFM_computeLFMatrixOperators` | Backward projection operator (image → volume) |
| $\mathbf{1}_V$ | constants | $(T_x, T_y, N_d)$ | float32 | $(\mathbf{1}_V)_{i,j,k} = 1$ | Unit volume for normalization computation |
| $P_{\text{fwd}}$ | constants | $(N_x, N_y)$ | float32 | $P_{\text{fwd}} = H(\mathbf{1}_V)$ | Forward projection of unit volume |
| $P_{\text{bwd}}$ | constants | $(T_x, T_y, N_d)$ | float32 | $P_{\text{bwd}} = H^T(P_{\text{fwd}})$ | Backward projection of $P_{\text{fwd}}$ |
| $K_j$ | constants | $(T_x, T_y)$ | complex64 | Precomputed via `lanczosfft` | FFT of Lanczos kernel for depth $j$ |
| $w_j$ | constants | scalar | float32 | Computed via `LFM_computeDepthAdaptiveWidth` | Depth-adaptive filter width for depth $j$ |
| $a$ | constants | scalar | int | $a = 4$ (default) | Lanczos window size parameter |

### Observations
| Symbol | Category | Shape | Dtype | Initialization | Description |
|--------|----------|-------|-------|----------------|-------------|
| $I_{\text{raw}}$ | observations | $(M_x, M_y)$ | float32 | Loaded from file | Raw lenslet image |
| $I_{\text{calib}}$ | observations | $(M_x, M_y)$ | float32 | Loaded from file | White calibration image |
| $I$ | observations | $(N_x, N_y)$ | float32 | Derived from $I_{\text{raw}}$ | Corrected and normalized light field image |

### Dimension Parameters
| Symbol | Description | Derivation |
|--------|-------------|------------|
| $(M_x, M_y)$ | Raw image dimensions | From input file |
| $(N_x, N_y)$ | Transformed image dimensions (`imgSize`) | $\text{imgSize} = \text{get\_transformed\_shape}(M_x, M_y, \mathcal{T}) + (1 - \text{mod}(\cdot, 2))$ |
| $(T_x, T_y)$ | Texture space dimensions (`texSize`) | $\text{texSize} = \lceil \text{imgSize} \cdot s_{\text{tex}} \rceil + (1 - \text{mod}(\cdot, 2))$ |
| $N_d$ | Number of depth planes | $N_d = \text{len}(\text{depths}) = \lfloor (d_{\max} - d_{\min}) / \Delta d \rfloor + 1$ |

---

## 3. PREPROCESSING PIPELINE

### Step 3.1: Data Loading
| Field | Value |
|-------|-------|
| step_order | 1 |
| step_name | Load LFM Data |
| formula | $I_{\text{raw}} = \text{imread}(f_{\text{img}}), \quad I_{\text{calib}} = \text{imread}(f_{\text{calib}}), \quad \Theta = \text{yaml\_load}(f_{\text{config}})$ |
| inputs | File paths: $f_{\text{img}}, f_{\text{calib}}, f_{\text{config}}$ |
| outputs | $I_{\text{raw}} \in \mathbb{R}^{M_x \times M_y}$, $I_{\text{calib}} \in \mathbb{R}^{M_x \times M_y}$, $\Theta$ (config dict) |

### Step 3.2: Camera Parameter Setup
| Field | Value |
|-------|-------|
| step_order | 2 |
| step_name | Set Camera Parameters |
| formula | $\text{Camera} = \text{LFM\_setCameraParams\_v2}(\Theta, p_{\text{spacing}})$ |
| inputs | $\Theta$ (config), $p_{\text{spacing}} \in \mathbb{Z}^+$ (new lenslet spacing in pixels) |
| outputs | Camera struct containing optical parameters |

### Step 3.3: Geometry Computation
| Field | Value |
|-------|-------|
| step_order | 3 |
| step_name | Compute Geometry Parameters |
| formula | $(\mathcal{C}, \mathcal{R}, \mathcal{G}, \mathcal{G}') = \text{LFM\_computeGeometryParameters}(\text{Camera}, I_{\text{calib}}, [d_{\min}, d_{\max}], \Delta d, s_{\text{res}})$ |
| inputs | Camera struct, $I_{\text{calib}} \in \mathbb{R}^{M_x \times M_y}$, depth range $[d_{\min}, d_{\max}]$ (mm), depth step $\Delta d$ (mm), super-resolution factor $s_{\text{res}} \in \mathbb{Z}^+$ |
| outputs | $\mathcal{C}$ (LensletCenters), $\mathcal{R}$ (Resolution), $\mathcal{G}$ (LensletGridModel), $\mathcal{G}'$ (NewLensletGridModel) |

### Step 3.4: Operator Construction
| Field | Value |
|-------|-------|
| step_order | 4 |
| step_name | Compute Projection Operators |
| formula | $(H, H^T) = \text{LFM\_computeLFMatrixOperators}(\text{Camera}, \mathcal{R}, \mathcal{C})$ |
| inputs | Camera struct, $\mathcal{R}$ (Resolution), $\mathcal{C}$ (LensletCenters) |
| outputs | $H$ (forward operator), $H^T$ (backward/adjoint operator) |

### Step 3.5: Transformation Retrieval
| Field | Value |
|-------|-------|
| step_order | 5 |
| step_name | Retrieve Geometric Transformation |
| formula | $\mathcal{T}_{\text{raw}} = \text{LFM\_retrieveTransformation}(\mathcal{G}, \mathcal{G}'), \quad \mathcal{T} = \text{format\_transform}(\mathcal{T}_{\text{raw}})$ |
| inputs | $\mathcal{G} \in \mathbb{R}^{k \times 2}$, $\mathcal{G}' \in \mathbb{R}^{k \times 2}$ |
| outputs | $\mathcal{T}$ (formatted affine transformation parameters) |

### Step 3.6: Output Size Computation
| Field | Value |
|-------|-------|
| step_order | 6 |
| step_name | Compute Output Dimensions |
| formula | $\text{imgSize}_{\text{raw}} = \text{get\_transformed\_shape}((M_x, M_y), \mathcal{T})$ |
| | $\text{imgSize} = \text{imgSize}_{\text{raw}} + (1 - \text{mod}(\text{imgSize}_{\text{raw}}, 2))$ |
| | $\text{texSize}_{\text{raw}} = \lceil \text{imgSize} \cdot s_{\text{tex}} \rceil$ |
| | $\text{texSize} = \text{texSize}_{\text{raw}} + (1 - \text{mod}(\text{texSize}_{\text{raw}}, 2))$ |
| | $\text{volumeSize} = (\text{texSize}_x, \text{texSize}_y, N_d)$ |
| inputs | $(M_x, M_y)$, $\mathcal{T}$, $s_{\text{tex}} = \mathcal{R}[\text{texScaleFactor}]$, $N_d = \text{len}(\mathcal{R}[\text{depths}])$ |
| outputs | $(N_x, N_y) = \text{imgSize}$, $(T_x, T_y) = \text{texSize}$, $(T_x, T_y, N_d) = \text{volumeSize}$ |

### Step 3.7: Image Transformation
| Field | Value |
|-------|-------|
| step_order | 7 |
| step_name | Apply Geometric Transformation |
| formula | $I_{\text{trans}} = \text{transform\_img}(I_{\text{raw}}, \mathcal{T}, \mathbf{o})$ |
| inputs | $I_{\text{raw}} \in \mathbb{R}^{M_x \times M_y}$, $\mathcal{T}$ (transformation), $\mathbf{o} = \mathcal{C}[\text{offset}]$ (offset vector) |
| outputs | $I_{\text{trans}} \in \mathbb{R}^{N_x \times N_y}$ |

### Step 3.8: Intensity Normalization
| Field | Value |
|-------|-------|
| step_order | 8 |
| step_name | Min-Max Normalization |
| formula | $I = \frac{I_{\text{trans}} - \min(I_{\text{trans}})}{\max(I_{\text{trans}}) - \min(I_{\text{trans}})}$ |
| inputs | $I_{\text{trans}} \in \mathbb{R}^{N_x \times N_y}$ |
| outputs | $I \in [0, 1]^{N_x \times N_y}$ |

### Step 3.9: Normalization Volume Computation
| Field | Value |
|-------|-------|
| step_order | 9 |
| step_name | Compute Forward Normalization |
| formula | $P_{\text{fwd}} = H(\mathbf{1}_V)$ where $\mathbf{1}_V \in \mathbb{R}^{T_x \times T_y \times N_d}$, $(\mathbf{1}_V)_{i,j,k} = 1$ |
| inputs | $H$ (forward operator), $\mathbf{1}_V$ (unit volume) |
| outputs | $P_{\text{fwd}} \in \mathbb{R}^{N_x \times N_y}$ |

### Step 3.10: Backward Normalization Computation
| Field | Value |
|-------|-------|
| step_order | 10 |
| step_name | Compute Backward Normalization |
| formula | $P_{\text{bwd}} = H^T(P_{\text{fwd}})$ |
| inputs | $H^T$ (backward operator), $P_{\text{fwd}} \in \mathbb{R}^{N_x \times N_y}$ |
| outputs | $P_{\text{bwd}} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |

### Step 3.11: Anti-Aliasing Kernel Construction
| Field | Value |
|-------|-------|
| step_order | 11 |
| step_name | Build Lanczos Filter Kernels |
| formula | For each depth $j \in \{1, \ldots, N_d\}$: |
| | $w_j = \text{LFM\_computeDepthAdaptiveWidth}(\text{Camera}, \mathcal{R})_j$ |
| | $L_j(x, y) = \begin{cases} \text{sinc}(\frac{x}{w_j})\text{sinc}(\frac{x}{a \cdot w_j})\text{sinc}(\frac{y}{w_j})\text{sinc}(\frac{y}{a \cdot w_j}) & \text{if } |x| < a \cdot w_j \text{ and } |y| < a \cdot w_j \\ 0 & \text{otherwise} \end{cases}$ |
| | $K_j = \mathcal{F}[L_j]$ |
| inputs | Camera, $\mathcal{R}$, $(T_x, T_y, N_d)$, $a$ (Lanczos window size) |
| outputs | $K \in \mathbb{C}^{T_x \times T_y \times N_d}$ (FFT of Lanczos kernels per depth) |

**Definition of sinc function**:
$$\text{sinc}(x) = \begin{cases} 1 & \text{if } x = 0 \\ \frac{\sin(\pi x)}{\pi x} & \text{otherwise} \end{cases}$$

---

## 4. ALGORITHM STEPS

### Initialization

**BEGIN INITIALIZATION**

#### Step I.1: Volume Initialization
| Field | Value |
|-------|-------|
| step_name | Initialize Reconstruction Volume |
| formula | $V^{(0)}_{i,j,k} = 1 \quad \forall (i,j,k) \in \{1,\ldots,T_x\} \times \{1,\ldots,T_y\} \times \{1,\ldots,N_d\}$ |
| inputs | volumeSize $(T_x, T_y, N_d)$ |
| outputs | $V^{(0)} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |
| computational_note | Allocate as float32 array; uniform initialization is standard for Richardson-Lucy |

#### Step I.2: Iteration Counter Initialization
| Field | Value |
|-------|-------|
| step_name | Initialize Iteration Counter |
| formula | $n \leftarrow 0$ |
| inputs | None |
| outputs | $n \in \mathbb{Z}_{\geq 0}$ |
| computational_note | — |

#### Step I.3: Error Metrics List Initialization
| Field | Value |
|-------|-------|
| step_name | Initialize Error Storage |
| formula | $\mathcal{E} \leftarrow []$ (empty list) |
| inputs | None |
| outputs | $\mathcal{E}$ (list of scalars) |
| computational_note | Will store MAE values per iteration |

**END INITIALIZATION**

---

### Iteration Loop

**BEGIN ITERATION LOOP** (for $n = 0, 1, \ldots, N_{\text{iter}} - 1$)

#### Step L.1: Forward Projection
| Field | Value |
|-------|-------|
| step_name | Compute Forward Projection |
| formula | $\hat{I}^{(n)} = \begin{cases} P_{\text{fwd}} & \text{if } n = 0 \\ H(V^{(n)}) & \text{if } n > 0 \end{cases}$ |
| inputs | $V^{(n)} \in \mathbb{R}^{T_x \times T_y \times N_d}$ (or $\mathbf{1}_V$ for $n=0$), $H$ |
| outputs | $\hat{I}^{(n)} \in \mathbb{R}^{N_x \times N_y}$ |
| computational_note | Forward projection via `LFM_forwardProject` with step parameter for memory efficiency |

#### Step L.2: Compute Ratio Image
| Field | Value |
|-------|-------|
| step_name | Compute Observation-to-Estimate Ratio |
| formula | $R^{(n)}_{i,j} = \frac{I_{i,j}}{\hat{I}^{(n)}_{i,j}}$ |
| inputs | $I \in \mathbb{R}^{N_x \times N_y}$, $\hat{I}^{(n)} \in \mathbb{R}^{N_x \times N_y}$ |
| outputs | $R^{(n)} \in \mathbb{R}^{N_x \times N_y}$ |
| computational_note | Element-wise division |

#### Step L.3: Apply Forward Normalization Weighting
| Field | Value |
|-------|-------|
| step_name | Weight Ratio by Forward Normalization |
| formula | $E^{(n)}_{\text{img}} = R^{(n)} \odot P_{\text{fwd}}$ |
| | Equivalently: $(E^{(n)}_{\text{img}})_{i,j} = \frac{I_{i,j}}{\hat{I}^{(n)}_{i,j}} \cdot (P_{\text{fwd}})_{i,j}$ |
| inputs | $R^{(n)} \in \mathbb{R}^{N_x \times N_y}$, $P_{\text{fwd}} \in \mathbb{R}^{N_x \times N_y}$ |
| outputs | $E^{(n)}_{\text{img}} \in \mathbb{R}^{N_x \times N_y}$ |
| computational_note | $\odot$ denotes Hadamard (element-wise) product |

#### Step L.4: Handle Non-Finite Values in Image Error
| Field | Value |
|-------|-------|
| step_name | Sanitize Image Error |
| formula | $(E^{(n)}_{\text{img}})_{i,j} \leftarrow \begin{cases} 0 & \text{if } (E^{(n)}_{\text{img}})_{i,j} \notin \mathbb{R} \\ (E^{(n)}_{\text{img}})_{i,j} & \text{otherwise} \end{cases}$ |
| inputs | $E^{(n)}_{\text{img}} \in \mathbb{R}^{N_x \times N_y}$ |
| outputs | $E^{(n)}_{\text{img}} \in \mathbb{R}^{N_x \times N_y}$ (sanitized) |
| computational_note | Handles division by zero and overflow; checks for $\pm\infty$ and NaN |

#### Step L.5: Compute Error Metric
| Field | Value |
|-------|-------|
| step_name | Calculate Mean Absolute Error |
| formula | $\epsilon^{(n)} = \frac{1}{N_x \cdot N_y} \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} |E^{(n)}_{\text{img}, i,j} - P_{\text{fwd}, i,j}|$ |
| inputs | $E^{(n)}_{\text{img}} \in \mathbb{R}^{N_x \times N_y}$, $P_{\text{fwd}} \in \mathbb{R}^{N_x \times N_y}$ |
| outputs | $\epsilon^{(n)} \in \mathbb{R}_{\geq 0}$ |
| computational_note | Scalar metric; append to $\mathcal{E}$: $\mathcal{E} \leftarrow \mathcal{E} \cup \{\epsilon^{(n)}\}$ |

#### Step L.6: Backward Projection
| Field | Value |
|-------|-------|
| step_name | Compute Backward Projection of Error |
| formula | $E^{(n)}_{\text{vol}} = H^T(E^{(n)}_{\text{img}})$ |
| inputs | $E^{(n)}_{\text{img}} \in \mathbb{R}^{N_x \times N_y}$, $H^T$ |
| outputs | $E^{(n)}_{\text{vol}} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |
| computational_note | Backward projection via `LFM_backwardProject` |

#### Step L.7: Normalize Backward Projection
| Field | Value |
|-------|-------|
| step_name | Apply Backward Normalization |
| formula | $(\tilde{E}^{(n)}_{\text{vol}})_{i,j,k} = \frac{(E^{(n)}_{\text{vol}})_{i,j,k}}{(P_{\text{bwd}})_{i,j,k}}$ |
| inputs | $E^{(n)}_{\text{vol}} \in \mathbb{R}^{T_x \times T_y \times N_d}$, $P_{\text{bwd}} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |
| outputs | $\tilde{E}^{(n)}_{\text{vol}} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |
| computational_note | Element-wise division |

#### Step L.8: Handle Non-Finite Values in Volume Error
| Field | Value |
|-------|-------|
| step_name | Sanitize Volume Error |
| formula | $(\tilde{E}^{(n)}_{\text{vol}})_{i,j,k} \leftarrow \begin{cases} 0 & \text{if } (\tilde{E}^{(n)}_{\text{vol}})_{i,j,k} \notin \mathbb{R} \\ (\tilde{E}^{(n)}_{\text{vol}})_{i,j,k} & \text{otherwise} \end{cases}$ |
| inputs | $\tilde{E}^{(n)}_{\text{vol}} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |
| outputs | $\tilde{E}^{(n)}_{\text{vol}} \in \mathbb{R}^{T_x \times T_y \times N_d}$ (sanitized) |
| computational_note | Handles division by zero |

#### Step L.9: Multiplicative Update
| Field | Value |
|-------|-------|
| step_name | Richardson-Lucy Multiplicative Update |
| formula | $V^{(n+1)} = V^{(n)} \odot \tilde{E}^{(n)}_{\text{vol}}$ |
| | Equivalently: $V^{(n+1)}_{i,j,k} = V^{(n)}_{i,j,k} \cdot (\tilde{E}^{(n)}_{\text{vol}})_{i,j,k}$ |
| inputs | $V^{(n)} \in \mathbb{R}^{T_x \times T_y \times N_d}$, $\tilde{E}^{(n)}_{\text{vol}} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |
| outputs | $V^{(n+1)} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |
| computational_note | Core Richardson-Lucy update step |

#### Step L.10: Anti-Aliasing Filtering (Conditional)
| Field | Value |
|-------|-------|
| step_name | Apply Depth-Adaptive Lanczos Filter |
| formula | **If** `filter_flag = True`: |
| | For each depth slice $j \in \{1, \ldots, N_d\}$: |
| | $V^{(n+1)}_{:,:,j} \leftarrow \left| \mathcal{F}^{-1}_{\text{shift}}\left[ K_j \odot \mathcal{F}[V^{(n+1)}_{:,:,j}] \right] \right|$ |
| | **Expanded**: |
| | $\mathcal{F}[V^{(n+1)}_{:,:,j}]_{u,v} = \sum_{x=0}^{T_x-1} \sum_{y=0}^{T_y-1} V^{(n+1)}_{x,y,j} \exp\left(-2\pi i \left(\frac{ux}{T_x} + \frac{vy}{T_y}\right)\right)$ |
| | $V^{(n+1)}_{x,y,j} = \left| \frac{1}{T_x T_y} \sum_{u=0}^{T_x-1} \sum_{v=0}^{T_y-1} K_{j,u,v} \cdot \mathcal{F}[V^{(n+1)}_{:,:,j}]_{u,v} \exp\left(2\pi i \left(\frac{ux}{T_x} + \frac{vy}{T_y}\right)\right) \right|$ |
| inputs | $V^{(n+1)} \in \mathbb{R}^{T_x \times T_y \times N_d}$, $K \in \mathbb{C}^{T_x \times T_y \times N_d}$, `filter_flag` $\in \{\text{True}, \text{False}\}$ |
| outputs | $V^{(n+1)} \in \mathbb{R}^{T_x \times T_y \times N_d}$ (filtered) |
| computational_note | `fftshift` centers zero-frequency; absolute value ensures non-negativity after filtering |

#### Step L.11: Final Sanitization
| Field | Value |
|-------|-------|
| step_name | Final Non-Finite Value Handling |
| formula | $V^{(n+1)}_{i,j,k} \leftarrow \begin{cases} 0 & \text{if } V^{(n+1)}_{i,j,k} \notin \mathbb{R} \\ V^{(n+1)}_{i,j,k} & \text{otherwise} \end{cases}$ |
| inputs | $V^{(n+1)} \in \mathbb{R}^{T_x \times T_y \times N_d}$ |
| outputs | $V^{(n+1)} \in \mathbb{R}^{T_x \times T_y \times N_d}$ (sanitized) |
| computational_note | Final safety check after all operations |

#### Step L.12: Increment Counter
| Field | Value |
|-------|-------|
| step_name | Increment Iteration Counter |
| formula | $n \leftarrow n + 1$ |
| inputs | $n \in \mathbb{Z}_{\geq 0}$ |
| outputs | $n \in \mathbb{Z}_{> 0}$ |
| computational_note | — |

#### Step L.13: Convergence Check
| Field | Value |
|-------|-------|
| step_name | Check Stopping Criterion |
| formula | **If** $n \geq N_{\text{iter