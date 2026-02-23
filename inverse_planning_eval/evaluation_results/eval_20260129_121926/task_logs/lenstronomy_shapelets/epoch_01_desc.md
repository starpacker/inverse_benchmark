# Algorithmic Specification Document: Gravitational Lensing Source Reconstruction via Shapelet Linear Inversion

---

## 1. Mathematical Formulation

### 1.1 Problem Statement

Reconstruct an unlensed source light distribution from a noisy gravitationally lensed observation using a shapelet basis decomposition.

### 1.2 Forward Model

The observed image $\mathbf{d} \in \mathbb{R}^{N_{\text{pix}} \times N_{\text{pix}}}$ is modeled as:

$$\mathbf{d} = \mathbf{H} \cdot \mathbf{L} \cdot \mathbf{s}(\boldsymbol{\theta}) + \mathbf{n}$$

Where:
- $\mathbf{s}(\boldsymbol{\theta})$: Source surface brightness distribution
- $\mathbf{L}$: Gravitational lensing operator (ray-tracing via lens equation)
- $\mathbf{H}$: PSF convolution operator
- $\mathbf{n}$: Combined noise (Poisson + Gaussian)

### 1.3 Lens Equation

The lensing operator maps image-plane coordinates $\boldsymbol{\theta}$ to source-plane coordinates $\boldsymbol{\beta}$:

$$\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$$

For SIE (Singular Isothermal Ellipsoid) + External Shear:

$$\boldsymbol{\alpha}_{\text{SIE}}(\boldsymbol{\theta}) = \theta_E \cdot \frac{\boldsymbol{\theta}'}{|\boldsymbol{\theta}'|} \cdot f(e_1, e_2)$$

$$\boldsymbol{\alpha}_{\text{shear}}(\boldsymbol{\theta}) = \begin{pmatrix} \gamma_1 & \gamma_2 \\ \gamma_2 & -\gamma_1 \end{pmatrix} \boldsymbol{\theta}$$

### 1.4 Source Parameterization (Shapelet Basis)

The source is decomposed using 2D Cartesian shapelets:

$$s(\mathbf{x}) = \sum_{n_1=0}^{n_{\max}} \sum_{n_2=0}^{n_{\max}-n_1} c_{n_1,n_2} \cdot B_{n_1,n_2}\left(\frac{\mathbf{x} - \mathbf{x}_0}{\beta}\right)$$

Where the shapelet basis functions are:

$$B_{n_1,n_2}(u, v) = \phi_{n_1}(u) \cdot \phi_{n_2}(v)$$

$$\phi_n(x) = \frac{1}{\sqrt{2^n n! \sqrt{\pi}}} H_n(x) \exp\left(-\frac{x^2}{2}\right)$$

$H_n(x)$ are Hermite polynomials. Number of basis functions:

$$N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$$

### 1.5 Linear System Formulation

Flattening the image to vector form $\mathbf{d} \in \mathbb{R}^{N_{\text{pix}}^2}$ and stacking shapelet coefficients $\mathbf{c} \in \mathbb{R}^{N_{\text{coeff}}}$:

$$\mathbf{d} = \mathbf{M} \mathbf{c} + \mathbf{n}$$

Where $\mathbf{M} \in \mathbb{R}^{N_{\text{pix}}^2 \times N_{\text{coeff}}}$ is the response matrix encoding lensing + PSF convolution for each shapelet mode.

### 1.6 Objective Function (Weighted Least Squares)

$$\hat{\mathbf{c}} = \arg\min_{\mathbf{c}} \left( \mathbf{d} - \mathbf{M}\mathbf{c} \right)^T \mathbf{C}_n^{-1} \left( \mathbf{d} - \mathbf{M}\mathbf{c} \right)$$

Where $\mathbf{C}_n$ is the noise covariance matrix (diagonal for independent pixels):

$$[\mathbf{C}_n]_{ii} = \sigma_{\text{bkg}}^2 + \frac{d_i}{t_{\text{exp}}}$$

### 1.7 Closed-Form Solution

The WLS solution is:

$$\hat{\mathbf{c}} = \left( \mathbf{M}^T \mathbf{C}_n^{-1} \mathbf{M} \right)^{-1} \mathbf{M}^T \mathbf{C}_n^{-1} \mathbf{d}$$

Parameter covariance:

$$\text{Cov}(\hat{\mathbf{c}}) = \left( \mathbf{M}^T \mathbf{C}_n^{-1} \mathbf{M} \right)^{-1}$$

---

## 2. Variables Taxonomy

### 2.1 Observed Data / Inputs

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| `image_data` | $\mathbf{d}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | `float64` | Noisy observed lensed image |
| `image_clean` | $\mathbf{d}_{\text{true}}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | `float64` | Ground truth (simulation only) |

### 2.2 Hyperparameters / Constants

| Variable | Symbol | Value | Description |
|----------|--------|-------|-------------|
| `numPix` | $N_{\text{pix}}$ | 100 | Image dimension (pixels per side) |
| `deltaPix` | $\Delta_{\text{pix}}$ | 0.05 arcsec | Pixel scale |
| `background_rms` | $\sigma_{\text{bkg}}$ | 0.05 | Gaussian background noise std |
| `exp_time` | $t_{\text{exp}}$ | 100 s | Exposure time |
| `fwhm` | FWHM | 0.1 arcsec | PSF full-width half-maximum |
| `n_max` | $n_{\max}$ | 8 | Maximum shapelet order |
| `beta` | $\beta$ | 0.2 arcsec | Shapelet scale parameter |
| `center_x` | $x_0$ | 0.0 | Shapelet center x-coordinate |
| `center_y` | $y_0$ | 0.0 | Shapelet center y-coordinate |
| `theta_E` | $\theta_E$ | 1.0 arcsec | Einstein radius (SIE) |
| `e1, e2` | $e_1, e_2$ | 0.1, -0.1 | SIE ellipticity components |
| `gamma1, gamma2` | $\gamma_1, \gamma_2$ | 0.05, 0.01 | External shear components |
| `random_seed` | — | 42 | RNG seed for reproducibility |

### 2.3 Primal / Optimization Variables

| Variable | Symbol | Shape | Dtype | Initialization |
|----------|--------|-------|-------|----------------|
| `shapelet_coeffs` | $\mathbf{c}$ | $(N_{\text{coeff}},)$ | `float64` | Solved via WLS |

Where: $N_{\text{coeff}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2} = 45$ for $n_{\max}=8$

### 2.4 Auxiliary / Derived Variables

| Variable | Symbol | Shape | Description |
|----------|--------|-------|-------------|
| `model_image` | $\hat{\mathbf{d}}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | Reconstructed model image |
| `error_map` | $\boldsymbol{\sigma}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | Per-pixel noise estimate |
| `cov_param` | $\text{Cov}(\hat{\mathbf{c}})$ | $(N_{\text{coeff}}, N_{\text{coeff}})$ | Parameter covariance matrix |
| `residuals` | $\mathbf{r}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | $\mathbf{d} - \hat{\mathbf{d}}$ |
| `transform_pix2angle` | $\mathbf{T}$ | $(2, 2)$ | Pixel-to-angular coordinate transform |

---

## 3. Preprocessing Pipeline

### Step 1: Coordinate System Setup
**Input:** $N_{\text{pix}}$, $\Delta_{\text{pix}}$  
**Output:** $\mathbf{T}$, $(\text{RA}_0, \text{Dec}_0)$

$$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$

$$c_x = c_y = \frac{N_{\text{pix}} - 1}{2}$$

$$\text{RA}_0 = -(\mathbf{T}_{00} \cdot c_x + \mathbf{T}_{01} \cdot c_y)$$

$$\text{Dec}_0 = -(\mathbf{T}_{10} \cdot c_x + \mathbf{T}_{11} \cdot c_y)$$

### Step 2: Data Class Initialization
**Input:** Coordinate parameters, $\sigma_{\text{bkg}}$, $t_{\text{exp}}$  
**Output:** `ImageData` object with zeros placeholder

### Step 3: PSF Configuration
**Input:** FWHM, $\Delta_{\text{pix}}$  
**Output:** Gaussian PSF kernel with truncation at $3\sigma$

$$\text{PSF}(r) \propto \exp\left(-\frac{r^2}{2\sigma_{\text{PSF}}^2}\right), \quad \sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$$

### Step 4: True Source Image Simulation
**Input:** Lens parameters, true source parameters  
**Output:** Clean lensed image $\mathbf{d}_{\text{clean}}$

$$\mathbf{d}_{\text{clean}} = \mathbf{H} \cdot \mathbf{L} \cdot \mathbf{s}_{\text{true}}$$

### Step 5: Noise Addition
**Input:** $\mathbf{d}_{\text{clean}}$, $\sigma_{\text{bkg}}$, $t_{\text{exp}}$  
**Output:** Noisy image $\mathbf{d}$

**5a. Poisson noise (signal-dependent):**
$$\mathbf{d}_{\text{counts}} = \max(\mathbf{d}_{\text{clean}} \cdot t_{\text{exp}}, 0)$$
$$\mathbf{d}_{\text{Poisson}} \sim \text{Poisson}(\mathbf{d}_{\text{counts}}) / t_{\text{exp}}$$
$$\mathbf{n}_{\text{Poisson}} = \mathbf{d}_{\text{Poisson}} - \mathbf{d}_{\text{clean}}$$

**5b. Gaussian background noise:**
$$\mathbf{n}_{\text{bkg}} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2 \mathbf{I})$$

**5c. Combined noisy image:**
$$\mathbf{d} = \mathbf{d}_{\text{clean}} + \mathbf{n}_{\text{Poisson}} + \mathbf{n}_{\text{bkg}}$$

---

## 4. Algorithm Steps

### Algorithm: Shapelet-Based Source Reconstruction via Linear Inversion

---

**INITIALIZATION**

**Step 0.1:** Define reconstruction source model
- Model type: `'SHAPELETS'`
- Parameters: $\{n_{\max}, \beta, x_0, y_0\}$

**Step 0.2:** Instantiate `ImageLinearFit` object
- Associates data, PSF, lens model, and source model

---

**MAIN COMPUTATION (Single-Step Closed-Form Solution)**

**Step 1: Build Response Matrix**
- **Name:** Construct Design Matrix
- **Mathematical Formula:**  
  For each shapelet basis function $k \in \{1, \ldots, N_{\text{coeff}}\}$:
  $$[\mathbf{M}]_{:,k} = \text{vec}\left( \mathbf{H} \cdot \mathbf{L} \cdot B_k \right)$$
  where $B_k$ is the $k$-th shapelet basis evaluated at source-plane positions.
- **Input:** PSF kernel, lens model, shapelet parameters $(n_{\max}, \beta, x_0, y_0)$
- **Output:** Response matrix $\mathbf{M} \in \mathbb{R}^{N_{\text{pix}}^2 \times N_{\text{coeff}}}$

**Step 2: Compute Noise Covariance**
- **Name:** Error Map Construction
- **Mathematical Formula:**  
  $$\sigma_i^2 = \sigma_{\text{bkg}}^2 + \frac{\max(d_i, 0)}{t_{\text{exp}}}$$
- **Input:** $\mathbf{d}$, $\sigma_{\text{bkg}}$, $t_{\text{exp}}$
- **Output:** Diagonal weight matrix $\mathbf{W} = \text{diag}(1/\sigma_i^2)$

**Step 3: Solve Weighted Least Squares**
- **Name:** Normal Equations Solution
- **Mathematical Formula:**  
  $$\mathbf{A} = \mathbf{M}^T \mathbf{W} \mathbf{M}$$
  $$\mathbf{b} = \mathbf{M}^T \mathbf{W} \mathbf{d}$$
  $$\hat{\mathbf{c}} = \mathbf{A}^{-1} \mathbf{b}$$
- **Input:** $\mathbf{M}$, $\mathbf{W}$, $\mathbf{d}$
- **Output:** Optimal shapelet coefficients $\hat{\mathbf{c}}$

**Step 4: Compute Model Image**
- **Name:** Forward Model Evaluation
- **Mathematical Formula:**  
  $$\hat{\mathbf{d}} = \mathbf{M} \hat{\mathbf{c}}$$
  (reshaped to $(N_{\text{pix}}, N_{\text{pix}})$)
- **Input:** $\mathbf{M}$, $\hat{\mathbf{c}}$
- **Output:** Model image $\hat{\mathbf{d}}$

**Step 5: Compute Parameter Covariance (Optional)**
- **Name:** Uncertainty Quantification
- **Mathematical Formula:**  
  $$\text{Cov}(\hat{\mathbf{c}}) = \mathbf{A}^{-1} = \left( \mathbf{M}^T \mathbf{W} \mathbf{M} \right)^{-1}$$
- **Input:** $\mathbf{A}$
- **Output:** $\text{Cov}(\hat{\mathbf{c}})$

---

## 5. Stopping Criterion

This algorithm is a **direct (non-iterative) solver**. No iterative stopping criterion is required.

**Termination Condition:**  
The algorithm terminates upon successful computation of the closed-form WLS solution:
$$\hat{\mathbf{c}} = \left( \mathbf{M}^T \mathbf{W} \mathbf{M} \right)^{-1} \mathbf{M}^T \mathbf{W} \mathbf{d}$$

**Implicit Numerical Conditions:**
- Matrix $\mathbf{M}^T \mathbf{W} \mathbf{M}$ must be invertible (well-conditioned)
- No explicit convergence threshold or iteration limit

---

## 6. Evaluation Metrics

### 6.1 Reduced Chi-Square

$$\chi^2_{\text{red}} = \frac{1}{N_{\text{dof}}} \sum_{i=1}^{N_{\text{pix}}^2} \frac{(d_i - \hat{d}_i)^2}{\sigma_i^2}$$

Where:
$$N_{\text{dof}} = N_{\text{pix}}^2 - N_{\text{coeff}}$$

**Interpretation:** $\chi^2_{\text{red}} \approx 1$ indicates good fit.

### 6.2 Residual Root Mean Square

$$\text{RMS}_{\text{res}} = \sqrt{\frac{1}{N_{\text{pix}}^2} \sum_{i=1}^{N_{\text{pix}}^2} (d_i - \hat{d}_i)^2}$$

### 6.3 Peak Signal-to-Noise Ratio

$$\text{PSNR} = 20 \cdot \log_{10}\left( \frac{\max_i |\hat{d}_i|}{\text{RMS}_{\text{res}}} \right) \quad \text{[dB]}$$

### 6.4 Coefficient Statistics

$$\bar{c} = \frac{1}{N_{\text{coeff}}} \sum_{k=1}^{N_{\text{coeff}}} c_k$$

$$\sigma_c = \sqrt{\frac{1}{N_{\text{coeff}}} \sum_{k=1}^{N_{\text{coeff}}} (c_k - \bar{c})^2}$$

$$c_{\max} = \max_k |c_k|$$

### 6.5 Forward Operator Consistency

$$\epsilon_{\text{fwd}} = \max_i \left| [\mathbf{M}\hat{\mathbf{c}}]_i - \hat{d}_i \right|$$

**Expected:** $\epsilon_{\text{fwd}} < 10^{-10}$ (machine precision)

---

## Summary Table

| Component | Specification |
|-----------|---------------|
| **Problem Type** | Linear Inverse Problem |
| **Solution Method** | Weighted Least Squares (closed-form) |
| **Basis Functions** | 2D Cartesian Shapelets (Hermite-Gaussian) |
| **Optimization Variables** | $N_{\text{coeff}} = 45$ shapelet coefficients |
| **Data Dimension** | $100 \times 100 = 10,000$ pixels |
| **Computational Complexity** | $\mathcal{O}(N_{\text{pix}}^2 \cdot N_{\text{coeff}}^2 + N_{\text{coeff}}^3)$ |
| **Primary Metric** | Reduced $\chi^2$ |