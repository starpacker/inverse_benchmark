# Algorithmic Specification Document: Quasar-Host Galaxy Image Decomposition via PSO Optimization

## 1. Mathematical Formulation

### 1.1 Problem Statement
Given a noisy observed image $\mathbf{Y} \in \mathbb{R}^{N \times N}$ containing a point source (quasar) superimposed on an extended host galaxy, estimate the model parameters $\boldsymbol{\theta}$ that best explain the observation.

### 1.2 Objective Function
The optimization minimizes the negative log-likelihood (equivalent to weighted least squares for Gaussian noise):

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i,j} \frac{\left( Y_{ij} - \hat{Y}_{ij}(\boldsymbol{\theta}) \right)^2}{\sigma_{ij}^2}$$

where:
- $Y_{ij}$: observed pixel value at position $(i,j)$
- $\hat{Y}_{ij}(\boldsymbol{\theta})$: model-predicted pixel value
- $\sigma_{ij}^2$: noise variance at pixel $(i,j)$

### 1.3 Forward Model
The predicted image is computed as:

$$\hat{\mathbf{Y}}(\boldsymbol{\theta}) = \mathbf{K} \ast \left[ I_{\text{disk}}(\boldsymbol{\theta}_{\text{disk}}) + I_{\text{bulge}}(\boldsymbol{\theta}_{\text{bulge}}) + I_{\text{ps}}(\boldsymbol{\theta}_{\text{ps}}) \right]$$

where $\mathbf{K}$ is the Point Spread Function (PSF) kernel and $\ast$ denotes 2D convolution.

### 1.4 Component Models

#### 1.4.1 Sérsic Profile (Disk and Bulge)
$$I_{\text{Sérsic}}(r; A, R_s, n, e_1, e_2, c_x, c_y) = A \cdot \exp\left( -b_n \left[ \left(\frac{r'}{R_s}\right)^{1/n} - 1 \right] \right)$$

where:
- $b_n \approx 1.9992n - 0.3271$ (approximation ensuring half-light radius property)
- $r' = \sqrt{q \cdot x'^2 + y'^2/q}$ (elliptical radius)
- $(x', y')$: rotated coordinates relative to center $(c_x, c_y)$
- Ellipticity transformation: $q = \frac{1-\sqrt{e_1^2+e_2^2}}{1+\sqrt{e_1^2+e_2^2}}$, $\phi = \frac{1}{2}\arctan\left(\frac{e_2}{e_1}\right)$

#### 1.4.2 Point Source Model
$$I_{\text{ps}}(\mathbf{x}; A_{\text{ps}}, \alpha_{\text{ps}}, \delta_{\text{ps}}) = A_{\text{ps}} \cdot \delta(\mathbf{x} - (\alpha_{\text{ps}}, \delta_{\text{ps}}))$$

where $\delta(\cdot)$ is the Dirac delta function, rendered as a PSF-convolved point at position $(\alpha_{\text{ps}}, \delta_{\text{ps}})$.

### 1.5 PSF Model (Gaussian)
$$K(x, y) = \frac{1}{2\pi\sigma_{\text{PSF}}^2} \exp\left( -\frac{x^2 + y^2}{2\sigma_{\text{PSF}}^2} \right)$$

where $\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$

### 1.6 Noise Model
The observed image is corrupted by both Poisson and Gaussian noise:

$$Y_{ij} = \hat{Y}_{ij}^{\text{true}} + \eta_{ij}^{\text{Poisson}} + \eta_{ij}^{\text{Gaussian}}$$

where:
- $\eta_{ij}^{\text{Poisson}} \sim \frac{1}{t_{\text{exp}}}\left(\text{Poisson}(\hat{Y}_{ij}^{\text{true}} \cdot t_{\text{exp}}) - \hat{Y}_{ij}^{\text{true}} \cdot t_{\text{exp}}\right)$
- $\eta_{ij}^{\text{Gaussian}} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$

### 1.7 Constraints
1. **Joint Center Constraint (Disk-Bulge)**:
$$c_x^{\text{disk}} = c_x^{\text{bulge}}, \quad c_y^{\text{disk}} = c_y^{\text{bulge}}$$

2. **Joint Center Constraint (Galaxy-Point Source)**:
$$c_x^{\text{disk}} = \alpha_{\text{ps}}, \quad c_y^{\text{disk}} = \delta_{\text{ps}}$$

3. **Fixed Sérsic Indices**:
$$n_{\text{disk}} = 1, \quad n_{\text{bulge}} = 4$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables (Non-linear)

| Variable | Symbol | Component | Shape | Dtype | Init Value | Bounds |
|----------|--------|-----------|-------|-------|------------|--------|
| Disk half-light radius | $R_s^{\text{disk}}$ | Disk | scalar | float64 | 1.0 | [0.001, 10] |
| Disk ellipticity 1 | $e_1^{\text{disk}}$ | Disk | scalar | float64 | 0.0 | [-0.5, 0.5] |
| Disk ellipticity 2 | $e_2^{\text{disk}}$ | Disk | scalar | float64 | 0.0 | [-0.5, 0.5] |
| Center x-coordinate | $c_x$ | Shared | scalar | float64 | 0.0 | [-10, 10] |
| Center y-coordinate | $c_y$ | Shared | scalar | float64 | 0.0 | [-10, 10] |
| Bulge half-light radius | $R_s^{\text{bulge}}$ | Bulge | scalar | float64 | 0.5 | [0.001, 10] |

### 2.2 Linear Variables (Solved Analytically)

| Variable | Symbol | Component | Shape | Dtype | Description |
|----------|--------|-----------|-------|-------|-------------|
| Disk amplitude | $A_{\text{disk}}$ | Disk | scalar | float64 | Surface brightness normalization |
| Bulge amplitude | $A_{\text{bulge}}$ | Bulge | scalar | float64 | Surface brightness normalization |
| Point source amplitude | $A_{\text{ps}}$ | Point Source | scalar | float64 | Total flux of point source |

### 2.3 Fixed Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Disk Sérsic index | $n_{\text{disk}}$ | 1 | Exponential profile |
| Bulge Sérsic index | $n_{\text{bulge}}$ | 4 | de Vaucouleurs profile |

### 2.4 Hyperparameters/Constants

| Parameter | Symbol | Value | Type | Description |
|-----------|--------|-------|------|-------------|
| Number of pixels | $N$ | 80 | int | Image dimension |
| Pixel scale | $\Delta_{\text{pix}}$ | 0.05 arcsec | float64 | Angular size per pixel |
| Background RMS | $\sigma_{\text{bkg}}$ | 0.1 | float64 | Gaussian noise std dev |
| Exposure time | $t_{\text{exp}}$ | 100.0 s | float64 | Integration time |
| PSF FWHM | FWHM | 0.1 arcsec | float64 | Point spread function width |
| PSF truncation | - | 3 | float64 | Truncation in units of $\sigma$ |
| PSO particles | $N_p$ | 50 | int | Swarm size |
| PSO iterations | $N_{\text{iter}}$ | 50 | int | Maximum iterations |
| Supersampling factor | - | 1 | int | No supersampling |

### 2.5 Ground Truth Parameters (Simulation)

| Parameter | Symbol | Value | Component |
|-----------|--------|-------|-----------|
| True center x | $c_x^{\text{true}}$ | 0.02 | All |
| True center y | $c_y^{\text{true}}$ | 0.01 | All |
| True point amp | $A_{\text{ps}}^{\text{true}}$ | 10000 | Point Source |
| True disk amplitude | $A_{\text{disk}}^{\text{true}}$ | 400 | Disk |
| True disk $R_s$ | $R_s^{\text{disk,true}}$ | 0.7 | Disk |
| True bulge amplitude | $A_{\text{bulge}}^{\text{true}}$ | 400 | Bulge |
| True bulge $R_s$ | $R_s^{\text{bulge,true}}$ | 0.3 | Bulge |
| Axis ratio | $q$ | 0.6 | Disk |
| Position angle | $\phi$ | 0.3 rad | Disk |

### 2.6 Observed Data

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| Observed image | $\mathbf{Y}$ | $(N, N) = (80, 80)$ | float64 | Noisy observation |
| PSF kernel | $\mathbf{K}$ | $(k, k)$ | float64 | Convolution kernel |

---

## 3. Preprocessing Pipeline

### Step 1: Coordinate System Setup
**Input**: $N$, $\Delta_{\text{pix}}$  
**Output**: Transformation matrix $\mathbf{T}$, reference coordinates $(\alpha_0, \delta_0)$

$$\mathbf{T} = \begin{pmatrix} -\Delta_{\text{pix}} & 0 \\ 0 & \Delta_{\text{pix}} \end{pmatrix}$$

$$c_{\text{pix}} = \frac{N-1}{2}$$

$$\alpha_0 = -(\mathbf{T}_{00} \cdot c_{\text{pix}} + \mathbf{T}_{01} \cdot c_{\text{pix}})$$

$$\delta_0 = -(\mathbf{T}_{10} \cdot c_{\text{pix}} + \mathbf{T}_{11} \cdot c_{\text{pix}})$$

### Step 2: PSF Generation
**Input**: FWHM, $\Delta_{\text{pix}}$, truncation factor  
**Output**: PSF kernel $\mathbf{K}$

$$\sigma_{\text{PSF}} = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$$

$$K_{ij} = \exp\left(-\frac{(i-c)^2 + (j-c)^2}{2(\sigma_{\text{PSF}}/\Delta_{\text{pix}})^2}\right)$$

Normalized: $\mathbf{K} \leftarrow \mathbf{K} / \sum_{i,j} K_{ij}$

### Step 3: Ellipticity Conversion
**Input**: Position angle $\phi$, axis ratio $q$  
**Output**: Ellipticity components $(e_1, e_2)$

$$\epsilon = \frac{1-q}{1+q}$$

$$e_1 = \epsilon \cos(2\phi)$$

$$e_2 = \epsilon \sin(2\phi)$$

### Step 4: Clean Image Simulation
**Input**: Model parameters $\boldsymbol{\theta}^{\text{true}}$, PSF $\mathbf{K}$  
**Output**: Clean model image $\hat{\mathbf{Y}}^{\text{clean}}$

$$\hat{\mathbf{Y}}^{\text{clean}} = \mathbf{K} \ast \left[ I_{\text{disk}}(\boldsymbol{\theta}_{\text{disk}}^{\text{true}}) + I_{\text{bulge}}(\boldsymbol{\theta}_{\text{bulge}}^{\text{true}}) + I_{\text{ps}}(\boldsymbol{\theta}_{\text{ps}}^{\text{true}}) \right]$$

### Step 5: Noise Injection
**Input**: $\hat{\mathbf{Y}}^{\text{clean}}$, $t_{\text{exp}}$, $\sigma_{\text{bkg}}$  
**Output**: Noisy observed image $\mathbf{Y}$

$$\mathbf{C} = \max(\hat{\mathbf{Y}}^{\text{clean}} \cdot t_{\text{exp}}, 0) \quad \text{(counts, clipped)}$$

$$\mathbf{C}' \sim \text{Poisson}(\mathbf{C})$$

$$\boldsymbol{\eta}^{\text{Poisson}} = \frac{\mathbf{C}' - \mathbf{C}}{t_{\text{exp}}}$$

$$\boldsymbol{\eta}^{\text{Gaussian}}_{ij} \sim \mathcal{N}(0, \sigma_{\text{bkg}}^2)$$

$$\mathbf{Y} = \hat{\mathbf{Y}}^{\text{clean}} + \boldsymbol{\eta}^{\text{Poisson}} + \boldsymbol{\eta}^{\text{Gaussian}}$$

---

## 4. Algorithm Steps: Particle Swarm Optimization with Linear Amplitude Solving

### 4.1 Initialization

**Step 0.1: Define Parameter Vector**
The non-linear optimization parameter vector:
$$\boldsymbol{\theta}_{\text{NL}} = \left( R_s^{\text{disk}}, e_1^{\text{disk}}, e_2^{\text{disk}}, c_x, c_y, R_s^{\text{bulge}} \right)^T \in \mathbb{R}^6$$

**Step 0.2: Initialize Particle Swarm**
For each particle $p \in \{1, \ldots, N_p\}$:

$$\boldsymbol{\theta}_p^{(0)} \sim \mathcal{U}(\boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}})$$

$$\mathbf{v}_p^{(0)} \sim \mathcal{U}(-|\boldsymbol{\theta}_{\text{upper}} - \boldsymbol{\theta}_{\text{lower}}|, |\boldsymbol{\theta}_{\text{upper}} - \boldsymbol{\theta}_{\text{lower}}|)$$

$$\mathbf{p}_p^{\text{best}} = \boldsymbol{\theta}_p^{(0)}$$

**Step 0.3: Initialize Global Best**
$$\mathbf{g}^{\text{best}} = \arg\min_{\boldsymbol{\theta}_p^{(0)}} \mathcal{L}(\boldsymbol{\theta}_p^{(0)})$$

### 4.2 Main PSO Iteration Loop

For iteration $t = 1, \ldots, N_{\text{iter}}$:

---

**Step 1: Velocity Update**  
**Formula**:
$$\mathbf{v}_p^{(t)} = \omega \mathbf{v}_p^{(t-1)} + c_1 r_1 \left( \mathbf{p}_p^{\text{best}} - \boldsymbol{\theta}_p^{(t-1)} \right) + c_2 r_2 \left( \mathbf{g}^{\text{best}} - \boldsymbol{\theta}_p^{(t-1)} \right)$$

**Input**: $\mathbf{v}_p^{(t-1)}$, $\boldsymbol{\theta}_p^{(t-1)}$, $\mathbf{p}_p^{\text{best}}$, $\mathbf{g}^{\text{best}}$, $\omega$, $c_1$, $c_2$, $r_1 \sim \mathcal{U}(0,1)$, $r_2 \sim \mathcal{U}(0,1)$  
**Output**: $\mathbf{v}_p^{(t)}$

---

**Step 2: Position Update**  
**Formula**:
$$\boldsymbol{\theta}_p^{(t)} = \boldsymbol{\theta}_p^{(t-1)} + \mathbf{v}_p^{(t)}$$

$$\boldsymbol{\theta}_p^{(t)} = \text{clip}\left(\boldsymbol{\theta}_p^{(t)}, \boldsymbol{\theta}_{\text{lower}}, \boldsymbol{\theta}_{\text{upper}}\right)$$

**Input**: $\boldsymbol{\theta}_p^{(t-1)}$, $\mathbf{v}_p^{(t)}$, bounds  
**Output**: $\boldsymbol{\theta}_p^{(t)}$

---

**Step 3: Apply Constraints**  
**Formula**:
$$c_x^{\text{bulge}} \leftarrow c_x^{\text{disk}}, \quad c_y^{\text{bulge}} \leftarrow c_y^{\text{disk}}$$

$$\alpha_{\text{ps}} \leftarrow c_x^{\text{disk}}, \quad \delta_{\text{ps}} \leftarrow c_y^{\text{disk}}$$

**Input**: $\boldsymbol{\theta}_p^{(t)}$  
**Output**: Constrained $\boldsymbol{\theta}_p^{(t)}$

---

**Step 4: Build Design Matrix for Linear Solve**  
**Formula**:
Construct matrix $\mathbf{A} \in \mathbb{R}^{N^2 \times 3}$ where columns represent normalized model components:

$$\mathbf{A}_{:,1} = \text{vec}\left( \mathbf{K} \ast I_{\text{disk}}^{\text{unit}}(\boldsymbol{\theta}_p^{(t)}) \right)$$

$$\mathbf{A}_{:,2} = \text{vec}\left( \mathbf{K} \ast I_{\text{bulge}}^{\text{unit}}(\boldsymbol{\theta}_p^{(t)}) \right)$$

$$\mathbf{A}_{:,3} = \text{vec}\left( \mathbf{K} \ast I_{\text{ps}}^{\text{unit}}(\boldsymbol{\theta}_p^{(t)}) \right)$$

**Input**: $\boldsymbol{\theta}_p^{(t)}$, PSF $\mathbf{K}$  
**Output**: Design matrix $\mathbf{A}$

---

**Step 5: Linear Amplitude Solve**  
**Formula**:
$$\mathbf{a}^* = \arg\min_{\mathbf{a} \geq 0} \left\| \mathbf{W}^{1/2}(\mathbf{y} - \mathbf{A}\mathbf{a}) \right\|_2^2$$

Solved via weighted least squares (or NNLS):
$$\mathbf{a}^* = (\mathbf{A}^T \mathbf{W} \mathbf{A})^{-1} \mathbf{A}^T \mathbf{W} \mathbf{y}$$

where:
- $\mathbf{y} = \text{vec}(\mathbf{Y})$
- $\mathbf{W} = \text{diag}(1/\sigma_{ij}^2)$
- $\mathbf{a}^* = (A_{\text{disk}}^*, A_{\text{bulge}}^*, A_{\text{ps}}^*)^T$

**Input**: $\mathbf{A}$, $\mathbf{y}$, $\mathbf{W}$  
**Output**: Optimal amplitudes $\mathbf{a}^*$

---

**Step 6: Compute Objective Function**  
**Formula**:
$$\hat{\mathbf{y}} = \mathbf{A} \mathbf{a}^*$$

$$\mathcal{L}_p = \sum_{i=1}^{N^2} W_{ii} (y_i - \hat{y}_i)^2 = (\mathbf{y} - \hat{\mathbf{y}})^T \mathbf{W} (\mathbf{y} - \hat{\mathbf{y}})$$

**Input**: $\mathbf{y}$, $\hat{\mathbf{y}}$, $\mathbf{W}$  
**Output**: $\mathcal{L}_p$

---

**Step 7: Update Personal Best**  
**Formula**:
$$\text{If } \mathcal{L}_p < \mathcal{L}(\mathbf{p}_p^{\text{best}}): \quad \mathbf{p}_p^{\text{best}} \leftarrow \boldsymbol{\theta}_p^{(t)}$$

**Input**: $\mathcal{L}_p$, $\mathbf{p}_p^{\text{best}}$, $\boldsymbol{\theta}_p^{(t)}$  
**Output**: Updated $\mathbf{p}_p^{\text{best}}$

---

**Step 8: Update Global Best**  
**Formula**:
$$\mathbf{g}^{\text{best}} = \arg\min_{\mathbf{p}_p^{\text{best}}, \forall p} \mathcal{L}(\mathbf{p}_p^{\text{best}})$$

**Input**: All $\mathbf{p}_p^{\text{best}}$  
**Output**: $\mathbf{g}^{\text{best}}$

---

### 4.3 Final Output
After convergence, extract best-fit parameters:
$$\boldsymbol{\theta}^* = \mathbf{g}^{\text{best}}$$

Compute final linear amplitudes with $\boldsymbol{\theta}^*$ to obtain complete solution.

---

## 5. Stopping Criterion

The algorithm terminates when ANY of the following conditions is met:

| Criterion | Mathematical Expression | Threshold |
|-----------|------------------------|-----------|
| Maximum iterations reached | $t \geq N_{\text{iter}}$ | $N_{\text{iter}} = 50$ |
| Convergence (optional) | $\|\mathbf{g}^{\text{best},(t)} - \mathbf{g}^{\text{best},(t-k)}\| < \epsilon$ | Implementation-dependent |

**Note**: The provided implementation uses iteration count only ($N_{\text{iter}} = 50$).

---

## 6. Evaluation Metrics

### 6.1 Residual Root Mean Square (RMS)
$$\text{RMS} = \sqrt{\frac{1}{N^2} \sum_{i,j} (Y_{ij} - \hat{Y}_{ij}^*)^2}$$

**Input**: Observed image $\mathbf{Y}$, reconstructed image $\hat{\mathbf{Y}}^*$  
**Output**: Scalar RMS value

### 6.2 Total Flux Comparison
$$S_{\text{recon}} = \sum_{i,j} \hat{Y}_{ij}^*, \quad S_{\text{obs}} = \sum_{i,j} Y_{ij}$$

$$\text{Flux Ratio} = \frac{S_{\text{recon}}}{S_{\text{obs}}}$$

### 6.3 Residual Image
$$\mathbf{R} = \mathbf{Y} - \hat{\mathbf{Y}}^*$$

### 6.4 Chi-Square (implicit in optimization)
$$\chi^2 = \sum_{i,j} \frac{(Y_{ij} - \hat{Y}_{ij}^*)^2}{\sigma_{ij}^2}$$

$$\chi^2_{\text{reduced}} = \frac{\chi^2}{N^2 - n_{\text{params}}}$$

where $n_{\text{params}}$ is the number of free parameters.

---

## 7. Summary Table: Algorithm Flow

| Phase | Step | Operation |
|-------|------|-----------|
| **Preprocessing** | 1 | Setup coordinate system |
| | 2 | Generate Gaussian PSF |
| | 3 | Compute ellipticity from $(q, \phi)$ |
| | 4 | Simulate clean image |
| | 5 | Add Poisson + Gaussian noise |
| **Initialization** | 6 | Initialize particle positions/velocities |
| | 7 | Evaluate initial fitness, set personal/global bests |
| **Main Loop** | 8 | Update velocities (PSO equation) |
| | 9 | Update positions with bounds enforcement |
| | 10 | Apply joint-center constraints |
| | 11 | Build linear design matrix |
| | 12 | Solve for optimal amplitudes (linear least squares) |
| | 13 | Compute $\chi^2$ objective |
| | 14 | Update personal best if improved |
| | 15 | Update global best |
| | 16 | Check stopping criterion → repeat or exit |
| **Post-processing** | 17 | Extract best-fit parameters |
| | 18 | Compute final reconstruction |
| | 19 | Calculate evaluation metrics |