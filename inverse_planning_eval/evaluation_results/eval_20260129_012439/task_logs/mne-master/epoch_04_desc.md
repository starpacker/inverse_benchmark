

## 0. ALGORITHM IDENTIFICATION

| Field | Value |
|-------|-------|
| **algorithm_name** | Minimum Norm Estimate with dSPM Normalization (MNE-dSPM) |
| **algorithm_family** | Linear Inverse Methods / Tikhonov Regularization |
| **summary** | Computes a regularized linear inverse operator via whitened SVD decomposition and applies dynamic Statistical Parametric Mapping normalization to produce noise-normalized source estimates from MEG sensor data. |

---

## 1. OBJECTIVE FUNCTION

### 1.1 full_expression

The algorithm solves the regularized least-squares problem:

$$\hat{x} = \arg\min_{x} \left\{ \|W P (G x - y)\|_2^2 + \lambda \|x\|_2^2 \right\}$$

where the closed-form solution is:

$$\hat{x} = K y$$

with inverse operator $K$ derived via SVD of the whitened, projected, scaled gain matrix.

### 1.2 data_fidelity_term

$$\mathcal{D}(x) = \|W P (G x - y)\|_2^2 = (Gx - y)^\top P^\top W^\top W P (Gx - y)$$

where:
- $W = V_C \Lambda_C^{-1/2} V_C^\top$ is the whitening matrix derived from eigendecomposition of projected noise covariance
- $P = I - Q Q^\top$ is the SSP projection matrix
- The norm is **L2-squared** (verified from matrix multiplication structure)

### 1.3 regularization_term

**Tikhonov (L2) regularization on source amplitudes:**

$$\mathcal{R}(x) = \lambda \|x\|_2^2 = \lambda \sum_{j=1}^{n_{\text{sources}}} x_j^2$$

where $\lambda = \frac{1}{\text{SNR}^2}$

### 1.4 constraints

$$\mathcal{C} = \emptyset \quad \text{(no explicit constraints; unconstrained optimization)}$$

---

## 2. VARIABLES

### 2.1 Primal Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $x$ | $(n_{\text{sources}}, n_{\text{times}})$ | float64 | Computed via $K y$ | Source current estimates |
| $\hat{x}_{\text{dSPM}}$ | $(n_{\text{sources}}, n_{\text{times}})$ | float64 | $\hat{x} \odot \sigma^{-1}$ | Noise-normalized source estimate (final output) |

### 2.2 Dual Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| — | — | — | — | No dual variables (closed-form solution) |

### 2.3 Auxiliary Variables

| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $C_{\text{scaled}}$ | $(n_{\text{ch}}, n_{\text{ch}})$ | float64 | $C / n_{\text{ave}}$ | Noise covariance scaled by number of averages |
| $C_{\text{proj}}$ | $(n_{\text{ch}}, n_{\text{ch}})$ | float64 | $P C_{\text{scaled}} P^\top$ | Projected noise covariance |
| $G_{\text{proj}}$ | $(n_{\text{ch}}, n_{\text{sources}})$ | float64 | $P G$ | Projected gain matrix |
| $\Lambda_C$ | $(n_{\text{rank}},)$ | float64 | From `eigh(C_proj)` | Eigenvalues of projected covariance (filtered) |
| $V_C$ | $(n_{\text{ch}}, n_{\text{rank}})$ | float64 | From `eigh(C_proj)` | Eigenvectors of projected covariance (filtered) |
| $W$ | $(n_{\text{ch}}, n_{\text{ch}})$ | float64 | $V_C \Lambda_C^{-1/2} V_C^\top$ | Whitening matrix |
| $\tilde{G}$ | $(n_{\text{ch}}, n_{\text{sources}})$ | float64 | $g_{\text{scale}} \cdot W G_{\text{proj}}$ | Whitened and scaled gain matrix |
| $g_{\text{scale}}$ | scalar | float64 | $\sqrt{n_{\text{nzero}} / \text{tr}(\tilde{G}_0 \tilde{G}_0^\top)}$ | Gain scaling factor |
| $U$ | $(n_{\text{ch}}, n_{\text{sv}})$ | float64 | From `svd(G̃)` | Left singular vectors |
| $S$ | $(n_{\text{sv}},)$ | float64 | From `svd(G̃)` | Singular values |
| $V$ | $(n_{\text{sources}}, n_{\text{sv}})$ | float64 | From `svd(G̃)` | Right singular vectors |
| $\Gamma$ | $(n_{\text{sv}},)$ | float64 | $S / (S^2 + \lambda)$ | Tikhonov filter factors |
| $K$ | $(n_{\text{sources}}, n_{\text{ch}})$ | float64 | $g_{\text{scale}} (V \text{diag}(\Gamma)) U^\top W$ | Inverse operator |
| $\sigma^2$ | $(n_{\text{sources}},)$ | float64 | $g_{\text{scale}}^2 \sum_k (V_{jk} \Gamma_k)^2$ | Noise variance per source |

### 2.4 Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\text{SNR}$ | 3.0 | Signal-to-noise ratio assumption |
| $\lambda$ | $1/9 \approx 0.1111$ | Regularization parameter ($1/\text{SNR}^2$) |
| $\text{tol}$ | $10^{-6} \cdot \max(\lambda_C)$ | Eigenvalue truncation tolerance |
| `method` | `'dSPM'` | Normalization method flag |

### 2.5 Observations

| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $y$ | $(n_{\text{ch}}, n_{\text{times}})$ | float64 | Sensor-space MEG measurements (evoked response) |
| $G$ | $(n_{\text{ch}}, n_{\text{sources}})$ | float64 | Lead field / gain matrix (forward model) |
| $C$ | $(n_{\text{ch}}, n_{\text{ch}})$ | float64 | Noise covariance matrix |
| $P$ | $(n_{\text{ch}}, n_{\text{ch}})$ | float64 | SSP projection matrix |
| $n_{\text{ave}}$ | scalar | int | Number of epochs averaged |

---

## 3. PREPROCESSING PIPELINE

| Step | step_name | Formula | Inputs | Outputs |
|------|-----------|---------|--------|---------|
| 0 | Parameter Derivation | $\lambda = \frac{1}{\text{SNR}^2} = \frac{1}{3.0^2} = \frac{1}{9}$ | SNR: scalar | $\lambda$: scalar |
| 1 | Noise Covariance Scaling | $C_{\text{scaled}} = \frac{C}{n_{\text{ave}}}$ | $C$: $(n_{\text{ch}}, n_{\text{ch}})$, $n_{\text{ave}}$: scalar | $C_{\text{scaled}}$: $(n_{\text{ch}}, n_{\text{ch}})$ |
| 2 | Covariance Projection | $C_{\text{proj}} = P \, C_{\text{scaled}} \, P^\top$ | $P$: $(n_{\text{ch}}, n_{\text{ch}})$, $C_{\text{scaled}}$: $(n_{\text{ch}}, n_{\text{ch}})$ | $C_{\text{proj}}$: $(n_{\text{ch}}, n_{\text{ch}})$ |
| 3 | Gain Matrix Projection | $G_{\text{proj}} = P \, G$ | $P$: $(n_{\text{ch}}, n_{\text{ch}})$, $G$: $(n_{\text{ch}}, n_{\text{sources}})$ | $G_{\text{proj}}$: $(n_{\text{ch}}, n_{\text{sources}})$ |
| 4 | Eigendecomposition | $C_{\text{proj}} = V_C \, \text{diag}(\lambda_C) \, V_C^\top$ via symmetric eigendecomposition | $C_{\text{proj}}$: $(n_{\text{ch}}, n_{\text{ch}})$ | $\lambda_C$: $(n_{\text{ch}},)$, $V_C$: $(n_{\text{ch}}, n_{\text{ch}})$ |
| 5 | Eigenvalue Filtering | $\text{mask}_i = \mathbb{1}[\lambda_{C,i} > 10^{-6} \cdot \max(\lambda_C)]$; retain only $\lambda_C[\text{mask}]$, $V_C[:, \text{mask}]$ | $\lambda_C$, $V_C$ | $\lambda_C$: $(n_{\text{rank}},)$, $V_C$: $(n_{\text{ch}}, n_{\text{rank}})$ |
| 6 | Whitening Matrix | $W = V_C \, \text{diag}(\lambda_C^{-1/2}) \, V_C^\top$ | $\lambda_C$: $(n_{\text{rank}},)$, $V_C$: $(n_{\text{ch}}, n_{\text{rank}})$ | $W$: $(n_{\text{ch}}, n_{\text{ch}})$ |
| 7 | Gain Whitening | $\tilde{G}_0 = W \, G_{\text{proj}}$ | $W$: $(n_{\text{ch}}, n_{\text{ch}})$, $G_{\text{proj}}$: $(n_{\text{ch}}, n_{\text{sources}})$ | $\tilde{G}_0$: $(n_{\text{ch}}, n_{\text{sources}})$ |
| 8 | Gain Scaling Factor | $g_{\text{scale}} = \sqrt{\frac{n_{\text{nzero}}}{\|\tilde{G}_0\|_F^2}} = \sqrt{\frac{n_{\text{rank}}}{\sum_{ij} \tilde{G}_{0,ij}^2}}$ | $\tilde{G}_0$: $(n_{\text{ch}}, n_{\text{sources}})$, $n_{\text{rank}}$: scalar | $g_{\text{scale}}$: scalar |
| 9 | Scaled Whitened Gain | $\tilde{G} = g_{\text{scale}} \cdot \tilde{G}_0$ | $g_{\text{scale}}$: scalar, $\tilde{G}_0$: $(n_{\text{ch}}, n_{\text{sources}})$ | $\tilde{G}$: $(n_{\text{ch}}, n_{\text{sources}})$ |

---

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION

| Step | step_name | Formula | Inputs | Outputs | computational_note |
|------|-----------|---------|--------|---------|-------------------|
| I.1 | SVD of Whitened Gain | $\tilde{G} = U \, \text{diag}(S) \, V^\top$ (economy SVD) | $\tilde{G}$: $(n_{\text{ch}}, n_{\text{sources}})$ | $U$: $(n_{\text{ch}}, n_{\text{sv}})$, $S$: $(n_{\text{sv}},)$, $V$: $(n_{\text{sources}}, n_{\text{sv}})$ | `full_matrices=False` |
| I.2 | Tikhonov Filter Factors | $\Gamma_k = \frac{S_k}{S_k^2 + \lambda}$ for $k = 1, \ldots, n_{\text{sv}}$ | $S$: $(n_{\text{sv}},)$, $\lambda$: scalar | $\Gamma$: $(n_{\text{sv}},)$ | Element-wise division |
| I.3 | Inverse Operator Assembly | $K = g_{\text{scale}} \cdot (V \odot \Gamma^\top) \cdot U^\top \cdot W$ where $(V \odot \Gamma^\top)_{jk} = V_{jk} \Gamma_k$ | $V$: $(n_{\text{sources}}, n_{\text{sv}})$, $\Gamma$: $(n_{\text{sv}},)$, $U$: $(n_{\text{ch}}, n_{\text{sv}})$, $W$: $(n_{\text{ch}}, n_{\text{ch}})$, $g_{\text{scale}}$: scalar | $K$: $(n_{\text{sources}}, n_{\text{ch}})$ | Broadcast multiply then matmul |

### 4.2 ITERATION LOOP

**BEGIN_LOOP** (Single-pass; no iterative loop)

| Step | step_name | Formula | Inputs | Outputs | computational_note |
|------|-----------|---------|--------|---------|-------------------|
| L.1 | Apply Inverse Operator [FORWARD] | $\hat{x} = K \, y$ | $K$: $(n_{\text{sources}}, n_{\text{ch}})$, $y$: $(n_{\text{ch}}, n_{\text{times}})$ | $\hat{x}$: $(n_{\text{sources}}, n_{\text{times}})$ | Matrix multiplication |

**END_LOOP**

### 4.3 FINALIZATION

| Step | step_name | Formula | Inputs | Outputs | computational_note |
|------|-----------|---------|--------|---------|-------------------|
| F.1 | Noise Variance Computation | $\sigma_j^2 = g_{\text{scale}}^2 \sum_{k=1}^{n_{\text{sv}}} (V_{jk} \Gamma_k)^2$ | $V$: $(n_{\text{sources}}, n_{\text{sv}})$, $\Gamma$: $(n_{\text{sv}},)$, $g_{\text{scale}}$: scalar | $\sigma^2$: $(n_{\text{sources}},)$ | Row-wise sum of squares |
| F.2 | dSPM Normalization | $\hat{x}_{\text{dSPM},jt} = \frac{\hat{x}_{jt}}{\sigma_j} = \hat{x}_{jt} \cdot (\sigma_j^2)^{-1/2}$ | $\hat{x}$: $(n_{\text{sources}}, n_{\text{times}})$, $\sigma^2$: $(n_{\text{sources}},)$ | $\hat{x}_{\text{dSPM}}$: $(n_{\text{sources}}, n_{\text{times}})$ | Column broadcast |
| F.3 | Output Formatting | Return $\hat{x}_{\text{dSPM}}$ as `source_estimate` | $\hat{x}_{\text{dSPM}}$ | `source_estimate`: $(n_{\text{sources}}, n_{\text{times}})$ | Final output array |

---

## 5. STOPPING CRITERION

| Field | Value |
|-------|-------|
| **type** | `none` (closed-form solution) |
| **expression** | N/A — algorithm computes exact solution in single pass |
| **parameters** | No iteration parameters |
| **check_frequency** | N/A |

---

## 6. EVALUATION METRICS

| Metric Name | Formula | Description |
|-------------|---------|-------------|
| **MSE** | $\text{MSE} = \frac{1}{n_{\text{sources}} \cdot n_{\text{times}}} \sum_{j,t} (\hat{x}_{jt} - x^{\text{ref}}_{jt})^2$ | Mean squared error between standalone and MNE reference |
| **PSNR** | $\text{PSNR} = 10 \log_{10}\left(\frac{(\max_{jt} x^{\text{ref}}_{jt})^2}{\text{MSE}}\right)$ dB; if MSE $= 0$, PSNR $= \infty$ | Peak signal-to-noise ratio |
| **Correlation** | $\rho = \frac{\sum_{i}(\hat{x}_i - \bar{\hat{x}})(x^{\text{ref}}_i - \bar{x}^{\text{ref}})}{\sqrt{\sum_i(\hat{x}_i - \bar{\hat{x}})^2}\sqrt{\sum_i(x^{\text{ref}}_i - \bar{x}^{\text{ref}})^2}}$ where $i$ indexes flattened arrays | Pearson correlation coefficient |
| **Success Criterion** | $\rho > 0.99$ | Boolean pass/fail for implementation validation |

---

## 7. REQUIRED OUTPUT FIELDS

The downstream JSON output MUST include the following fields:

```
algorithm_identification:
  - algorithm_name
  - algorithm_family
  - summary

objective_function:
  - full_expression
  - data_fidelity_term
  - regularization_term
  - constraints

variables:
  - primal_variables[]
  - dual_variables[]
  - auxiliary_variables[]
  - constants[]
  - observations[]

data_preprocessing:
  - steps[]:
      - step_order
      - step_name
      - formula
      - inputs[]
      - outputs[]

initialization:
  - steps[]:
      - step_name
      - formula
      - inputs[]
      - outputs[]
      - computational_note

iteration_loop:
  - loop_type: "single_pass"
  - steps[]:
      - step_name
      - formula
      - inputs[]
      - outputs[]
      - computational_note

finalization:
  - steps[]:
      - step_name
      - formula
      - inputs[]
      - outputs[]
      - computational_note

stopping_criterion:
  - type
  - expression
  - parameters
  - check_frequency

evaluation:
  - metrics[]:
      - name
      - formula
      - description
```