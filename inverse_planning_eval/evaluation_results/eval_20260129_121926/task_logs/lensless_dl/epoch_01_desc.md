# Algorithmic Specification Document: FISTA-based Image Deconvolution

---

## 1. Mathematical Formulation

### 1.1 Problem Statement

The algorithm solves a **non-negative least-squares image deconvolution** problem. Given a blurred measurement $\mathbf{y}$ and a known point spread function (PSF) $\mathbf{h}$, recover the latent image $\mathbf{x}$.

### 1.2 Objective Function

$$
\min_{\mathbf{x}} \quad \underbrace{\frac{1}{2} \|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2}_{f(\mathbf{x}): \text{ Data Fidelity}} + \underbrace{\iota_{\geq 0}(\mathbf{x})}_{g(\mathbf{x}): \text{ Constraint}}
$$

where:
- $\mathbf{A}: \mathbb{R}^{D \times H \times W \times C} \rightarrow \mathbb{R}^{D \times H \times W \times C}$ is the **linear convolution operator** with PSF $\mathbf{h}$
- $\|\cdot\|_2$ denotes the Frobenius norm (element-wise $\ell_2$)
- $\iota_{\geq 0}(\mathbf{x})$ is the **indicator function** for the non-negative orthant:

$$
\iota_{\geq 0}(\mathbf{x}) = 
\begin{cases}
0 & \text{if } x_{d,i,j,c} \geq 0 \;\; \forall d,i,j,c \\
+\infty & \text{otherwise}
\end{cases}
$$

### 1.3 Forward Model

$$
\mathbf{y} = \mathbf{A}\mathbf{x} = \mathbf{h} \circledast \mathbf{x}
$$

where $\circledast$ denotes 2D spatial convolution applied independently per depth slice and channel.

### 1.4 Gradient of Data Fidelity Term

$$
\nabla f(\mathbf{x}) = \mathbf{A}^\top (\mathbf{A}\mathbf{x} - \mathbf{y})
$$

where $\mathbf{A}^\top$ is the **adjoint operator** (correlation), implemented as convolution with the conjugate-transposed PSF in Fourier domain.

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | Dtype | Initialization |
|----------|--------|-------|-------|----------------|
| Current image estimate | $\mathbf{x}^{(k)}$ | $(D, H, W, C)$ | `float32` | $\mathbf{0}$ (zeros) |
| Momentum variable | $\mathbf{z}^{(k)}$ | $(D, H, W, C)$ | `float32` | $\mathbf{x}^{(0)} = \mathbf{0}$ |

### 2.2 Auxiliary/Algorithm Variables

| Variable | Symbol | Shape | Dtype | Initialization |
|----------|--------|-------|-------|----------------|
| FISTA momentum parameter | $t_k$ | scalar | `float64` | $t_0 = 1.0$ |
| Gradient | $\mathbf{g}^{(k)}$ | $(D, H, W, C)$ | `float32` | computed per iteration |
| Residual | $\mathbf{r}^{(k)}$ | $(D, H, W, C)$ | `float32` | computed per iteration |
| PSF Fourier transform | $\mathbf{H}$ | $(D, H', \lfloor W'/2 \rfloor + 1, C)$ | `complex64` | precomputed |
| Adjoint PSF transform | $\mathbf{H}^*$ | $(D, H', \lfloor W'/2 \rfloor + 1, C)$ | `complex64` | $\overline{\mathbf{H}}$ (conjugate) |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Number of iterations | $K$ | 50 | Maximum FISTA iterations |
| Downsampling factor | $s$ | 4 | Spatial downsampling |
| Power method iterations | $M$ | 20 | For Lipschitz estimation |
| Lipschitz constant | $L$ | estimated | $\|\mathbf{A}^\top \mathbf{A}\|_{\text{op}}$ |
| Step size | $\alpha$ | $1/L$ | Gradient descent step |

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | Source |
|----------|--------|-------|--------|
| Raw PSF image | $\mathbf{h}_{\text{raw}}$ | $(H_0, W_0)$ or $(H_0, W_0, C)$ | Image file |
| Raw measurement | $\mathbf{y}_{\text{raw}}$ | $(H_0, W_0)$ or $(H_0, W_0, C)$ | Image file |
| Preprocessed PSF | $\mathbf{h}$ | $(1, H, W, C)$ | After pipeline |
| Preprocessed measurement | $\mathbf{y}$ | $(1, H, W, C)$ | After pipeline |

---

## 3. Preprocessing Pipeline

### Step 1: Image Loading
$$
\mathbf{h}_{\text{raw}}, \mathbf{y}_{\text{raw}} \leftarrow \texttt{load\_image}(\text{filepath}) \in \mathbb{R}^{H_0 \times W_0 (\times C)}
$$
Convert to `float32` dtype.

### Step 2: Spatial Downsampling
$$
\mathbf{h}_{\text{ds}}[i, j, c] = \mathbf{h}_{\text{raw}}[s \cdot i, s \cdot j, c]
$$
$$
\mathbf{y}_{\text{ds}}[i, j, c] = \mathbf{y}_{\text{raw}}[s \cdot i, s \cdot j, c]
$$
Output shape: $(H, W, C) = (\lfloor H_0/s \rfloor, \lfloor W_0/s \rfloor, C)$

### Step 3: Intensity Normalization
$$
\mathbf{h}_{\text{norm}} = \frac{\mathbf{h}_{\text{ds}} - \min(\mathbf{h}_{\text{ds}})}{\max(\mathbf{h}_{\text{ds}}) - \min(\mathbf{h}_{\text{ds}})} \in [0, 1]
$$
$$
\mathbf{y}_{\text{norm}} = \frac{\mathbf{y}_{\text{ds}} - \min(\mathbf{y}_{\text{ds}})}{\max(\mathbf{y}_{\text{ds}}) - \min(\mathbf{y}_{\text{ds}})} \in [0, 1]
$$

### Step 4: Dimension Expansion to 4D
$$
\mathbf{h} = \texttt{reshape}(\mathbf{h}_{\text{norm}}) \in \mathbb{R}^{1 \times H \times W \times C}
$$
$$
\mathbf{y} = \texttt{reshape}(\mathbf{y}_{\text{norm}}) \in \mathbb{R}^{1 \times H \times W \times C}
$$

If input is 2D $(H, W)$: add axes to get $(1, H, W, 1)$.

---

## 4. Algorithm Steps

### 4.1 FFT Setup and Padding Configuration

**Step 0a: Compute Padded Dimensions**
$$
H' = \texttt{next\_fast\_len}(2H - 1), \quad W' = \texttt{next\_fast\_len}(2W - 1)
$$

**Step 0b: Compute Padding Indices**
$$
i_{\text{start}} = \left\lfloor \frac{H' - H}{2} \right\rfloor, \quad j_{\text{start}} = \left\lfloor \frac{W' - W}{2} \right\rfloor
$$
$$
i_{\text{end}} = i_{\text{start}} + H, \quad j_{\text{end}} = j_{\text{start}} + W
$$

**Step 0c: Compute PSF Fourier Transform**
$$
\mathbf{h}_{\text{pad}} \in \mathbb{R}^{D \times H' \times W' \times C}, \quad \mathbf{h}_{\text{pad}}[d, i_{\text{start}}:i_{\text{end}}, j_{\text{start}}:j_{\text{end}}, c] = \mathbf{h}[d, :, :, c]
$$
$$
\mathbf{H} = \texttt{rfft2}(\mathbf{h}_{\text{pad}}, \text{axes}=(-3, -2), \text{norm}=\text{"ortho"})
$$
$$
\mathbf{H}^* = \overline{\mathbf{H}} \quad \text{(complex conjugate)}
$$

### 4.2 Lipschitz Constant Estimation (Power Method)

**Input:** $\mathbf{H}, \mathbf{H}^*$, max iterations $M$

**Initialize:** $\mathbf{v}^{(0)} \sim \mathcal{N}(0, 1)^{D \times H \times W \times C}$, $\mathbf{v}^{(0)} \leftarrow \mathbf{v}^{(0)} / \|\mathbf{v}^{(0)}\|_F$

**For** $m = 0, 1, \ldots, M-1$:

| Step | Formula | Description |
|------|---------|-------------|
| (a) | $\mathbf{u} = \mathbf{A}\mathbf{v}^{(m)}$ | Forward convolution |
| (b) | $\mathbf{v}^{(m+1)} = \mathbf{A}^\top \mathbf{u}$ | Adjoint convolution |
| (c) | $\lambda_m = \|\mathbf{v}^{(m+1)}\|_F$ | Estimate eigenvalue |
| (d) | $\mathbf{v}^{(m+1)} \leftarrow \mathbf{v}^{(m+1)} / \lambda_m$ | Normalize |

**Output:** $L = \lambda_{M-1}$ (Lipschitz constant)

**FFT-based Operations:**

Forward convolution $\mathbf{A}\mathbf{v}$:
$$
\mathbf{A}\mathbf{v} = \texttt{crop}\left( \texttt{ifftshift}\left( \texttt{irfft2}\left( \texttt{rfft2}(\texttt{pad}(\mathbf{v})) \cdot \mathbf{H} \right) \right) \right)
$$

Adjoint convolution $\mathbf{A}^\top \mathbf{u}$:
$$
\mathbf{A}^\top \mathbf{u} = \texttt{crop}\left( \texttt{ifftshift}\left( \texttt{irfft2}\left( \texttt{rfft2}(\texttt{pad}(\mathbf{u})) \cdot \mathbf{H}^* \right) \right) \right)
$$

### 4.3 FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

**Initialize:**
$$
\mathbf{x}^{(0)} = \mathbf{0} \in \mathbb{R}^{D \times H \times W \times C}
$$
$$
\mathbf{z}^{(0)} = \mathbf{x}^{(0)}
$$
$$
t_0 = 1
$$
$$
\alpha = \frac{1}{L}
$$

**For** $k = 0, 1, \ldots, K-1$:

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 1 | Forward model | $\hat{\mathbf{y}}^{(k)} = \mathbf{A}\mathbf{z}^{(k)}$ | $\mathbf{z}^{(k)}, \mathbf{H}$ | $\hat{\mathbf{y}}^{(k)}$ |
| 2 | Residual | $\mathbf{r}^{(k)} = \hat{\mathbf{y}}^{(k)} - \mathbf{y}$ | $\hat{\mathbf{y}}^{(k)}, \mathbf{y}$ | $\mathbf{r}^{(k)}$ |
| 3 | Gradient | $\mathbf{g}^{(k)} = \mathbf{A}^\top \mathbf{r}^{(k)}$ | $\mathbf{r}^{(k)}, \mathbf{H}^*$ | $\mathbf{g}^{(k)}$ |
| 4 | Gradient descent | $\tilde{\mathbf{x}}^{(k+1)} = \mathbf{z}^{(k)} - \alpha \mathbf{g}^{(k)}$ | $\mathbf{z}^{(k)}, \mathbf{g}^{(k)}, \alpha$ | $\tilde{\mathbf{x}}^{(k+1)}$ |
| 5 | Proximal (non-negativity) | $\mathbf{x}^{(k+1)} = \max(\tilde{\mathbf{x}}^{(k+1)}, 0)$ | $\tilde{\mathbf{x}}^{(k+1)}$ | $\mathbf{x}^{(k+1)}$ |
| 6 | Momentum update | $t_{k+1} = \dfrac{1 + \sqrt{1 + 4t_k^2}}{2}$ | $t_k$ | $t_{k+1}$ |
| 7 | Extrapolation | $\mathbf{z}^{(k+1)} = \mathbf{x}^{(k+1)} + \dfrac{t_k - 1}{t_{k+1}}(\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)})$ | $\mathbf{x}^{(k+1)}, \mathbf{x}^{(k)}, t_k, t_{k+1}$ | $\mathbf{z}^{(k+1)}$ |

**Output:** $\mathbf{x}^{(K)}$ (reconstructed image)

---

## 5. Stopping Criterion

The algorithm terminates based on a **fixed iteration count**:

$$
\text{Terminate when } k = K
$$

| Criterion Type | Condition | Default |
|----------------|-----------|---------|
| Maximum iterations | $k \geq K$ | $K = 50$ |

**Note:** No convergence-based early stopping is implemented (e.g., no relative change threshold $\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|/\|\mathbf{x}^{(k)}\| < \epsilon$).

---

## 6. Evaluation Metrics

### 6.1 Output Normalization for Display

$$
\mathbf{x}_{\text{display}} = \texttt{clip}\left( \frac{\mathbf{x}^{(K)} - \min(\mathbf{x}^{(K)})}{\max(\mathbf{x}^{(K)}) - \min(\mathbf{x}^{(K)})}, 0, 1 \right)
$$

### 6.2 Summary Statistics (Reported)

| Metric | Formula |
|--------|---------|
| Minimum | $x_{\min} = \min_{d,i,j,c} x^{(K)}_{d,i,j,c}$ |
| Maximum | $x_{\max} = \max_{d,i,j,c} x^{(K)}_{d,i,j,c}$ |
| Mean | $\bar{x} = \dfrac{1}{DHWC} \sum_{d,i,j,c} x^{(K)}_{d,i,j,c}$ |

### 6.3 Implicit Quality Metric (Data Fidelity)

Though not explicitly computed in output, the algorithm minimizes:
$$
\mathcal{L}_{\text{data}} = \frac{1}{2}\|\mathbf{A}\mathbf{x}^{(K)} - \mathbf{y}\|_F^2 = \frac{1}{2}\sum_{d,i,j,c}\left([\mathbf{h} \circledast \mathbf{x}^{(K)}]_{d,i,j,c} - y_{d,i,j,c}\right)^2
$$

---

## 7. Summary: Complete Algorithm Pseudocode

```
ALGORITHM: FISTA for Non-negative Image Deconvolution

INPUT: PSF h ∈ ℝ^(D×H×W×C), Measurement y ∈ ℝ^(D×H×W×C), Iterations K, Power iterations M

OUTPUT: Reconstructed image x* ∈ ℝ^(D×H×W×C)

1. PRECOMPUTE:
   H ← rfft2(pad(h))           // PSF Fourier transform
   H* ← conj(H)                // Adjoint transform
   
2. ESTIMATE LIPSCHITZ CONSTANT:
   v ← randn(D,H,W,C); v ← v/‖v‖
   FOR m = 1 TO M:
       v ← A^T(A(v)); v ← v/‖v‖
   L ← ‖v‖

3. INITIALIZE:
   x⁰ ← 0; z⁰ ← 0; t₀ ← 1; α ← 1/L

4. MAIN LOOP:
   FOR k = 0 TO K-1:
       g^k ← A^T(A(z^k) - y)                    // Gradient
       x̃^(k+1) ← z^k - α·g^k                    // Gradient step  
       x^(k+1) ← max(x̃^(k+1), 0)                // Proximal step
       t_(k+1) ← (1 + √(1 + 4t_k²))/2           // Momentum
       z^(k+1) ← x^(k+1) + ((t_k-1)/t_(k+1))·(x^(k+1) - x^k)  // Extrapolation

5. RETURN x^K
```

---

## 8. Computational Complexity

| Operation | Complexity per Iteration |
|-----------|-------------------------|
| Forward FFT | $\mathcal{O}(DHW \log(HW))$ |
| Inverse FFT | $\mathcal{O}(DHW \log(HW))$ |
| Pointwise operations | $\mathcal{O}(DHW)$ |
| **Total per iteration** | $\mathcal{O}(DHW \log(HW))$ |
| **Total algorithm** | $\mathcal{O}(K \cdot DHW \log(HW))$ |