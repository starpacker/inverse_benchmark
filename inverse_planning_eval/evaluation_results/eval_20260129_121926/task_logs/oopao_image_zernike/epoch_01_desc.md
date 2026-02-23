# Algorithmic Specification Document
## Adaptive Optics Simulation: Zernike-Based Phase Decomposition and PSF Computation

---

## 1. Mathematical Formulation

### 1.1 Problem Overview

This algorithm addresses two coupled problems in adaptive optics:

1. **Forward Problem**: Compute the Point Spread Function (PSF) from a given wavefront phase aberration
2. **Inverse Problem**: Decompose atmospheric optical path difference (OPD) into Zernike polynomial coefficients

---

### 1.2 Zernike Polynomial Basis Definition

**Zernike Radial Polynomial** $R_n^m(r)$ for radial order $n$ and azimuthal order $m$:

$$R_n^m(r) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2}-k\right)! \left(\frac{n-m}{2}-k\right)!} \cdot r^{n-2k}$$

**Constraints**: $R_n^m(r) = 0$ if $(n-m)$ is odd.

**Complete Zernike Mode** $Z_n^m(r, \theta)$ with orthonormal normalization:

$$Z_n^m(r, \theta) = \begin{cases}
\sqrt{n+1} \cdot R_n^0(r) & \text{if } m = 0 \\[6pt]
\sqrt{2(n+1)} \cdot R_n^m(r) \cdot \cos(m\theta) & \text{if } m > 0 \\[6pt]
\sqrt{2(n+1)} \cdot R_n^{|m|}(r) \cdot \sin(|m|\theta) & \text{if } m < 0
\end{cases}$$

**Domain constraint**: $Z_n^m(r, \theta) = 0$ for $r > 1$ (outside unit pupil).

---

### 1.3 Forward Model: Phase-to-PSF

**Objective**: Compute the intensity distribution in the focal plane given wavefront aberrations.

**Electric Field in Pupil Plane**:
$$E_{\text{pupil}}(x, y) = A(x, y) \cdot \exp\left(i \cdot \phi(x, y)\right)$$

where:
- $A(x,y) \in \{0, 1\}$ is the binary pupil amplitude mask
- $\phi(x,y)$ is the phase aberration map in radians

**Propagation to Focal Plane** (Fraunhofer diffraction):
$$E_{\text{focal}}(u, v) = \mathcal{F}\left\{ E_{\text{pupil}}(x, y) \right\}$$

**Point Spread Function**:
$$\text{PSF}(u, v) = \left| E_{\text{focal}}(u, v) \right|^2$$

**Normalized PSF**:
$$\text{PSF}_{\text{norm}}(u, v) = \frac{\text{PSF}(u, v)}{\max_{u,v}\left(\text{PSF}(u, v)\right)}$$

---

### 1.4 Inverse Model: Zernike Decomposition

**Objective**: Solve the least-squares problem to decompose OPD into Zernike coefficients.

**Optimization Problem**:
$$\min_{\mathbf{c}} \left\| \boldsymbol{\phi}_{\text{obs}} - \mathbf{Z} \mathbf{c} \right\|_2^2$$

where:
- $\boldsymbol{\phi}_{\text{obs}} \in \mathbb{R}^{N_p}$ is the observed OPD at $N_p$ valid pupil pixels
- $\mathbf{Z} \in \mathbb{R}^{N_p \times J}$ is the Zernike mode matrix
- $\mathbf{c} \in \mathbb{R}^{J}$ is the coefficient vector for $J$ modes

**Closed-Form Solution**:
$$\mathbf{c}^* = \mathbf{Z}^{\dagger} \boldsymbol{\phi}_{\text{obs}}$$

where $\mathbf{Z}^{\dagger} = (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T$ is the Moore-Penrose pseudoinverse.

**Reconstruction**:
$$\boldsymbol{\phi}_{\text{rec}} = \mathbf{Z}_{\text{full}} \mathbf{c}^*$$

where $\mathbf{Z}_{\text{full}} \in \mathbb{R}^{N^2 \times J}$ contains modes at full resolution.

---

### 1.5 Phase-OPD Relationship

$$\phi = \frac{2\pi}{\lambda} \cdot \text{OPD}$$

$$\text{OPD} = \frac{\phi \cdot \lambda}{2\pi}$$

---

## 2. Variables Taxonomy

### 2.1 Primal/Optimization Variables

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| Zernike coefficients | $\mathbf{c}$ | $(J,)$ | `float64` | Coefficients for $J$ Zernike modes |

### 2.2 Auxiliary/Computed Variables

| Variable | Symbol | Shape | Dtype | Description |
|----------|--------|-------|-------|-------------|
| Electric field (pupil) | $E_{\text{pupil}}$ | $(N, N)$ | `complex128` | Complex amplitude in pupil plane |
| Electric field (padded) | $E_{\text{padded}}$ | $(N_p, N_p)$ | `complex128` | Zero-padded field, $N_p = 4N$ |
| Focal plane field | $E_{\text{focal}}$ | $(N_p, N_p)$ | `complex128` | FFT of padded field |
| PSF | $\text{PSF}$ | $(N_p, N_p)$ | `float64` | Intensity at focal plane |
| Reconstructed OPD | $\boldsymbol{\phi}_{\text{rec}}$ | $(N, N)$ | `float64` | Zernike-reconstructed OPD |
| Residual | $\boldsymbol{\epsilon}$ | $(N, N)$ | `float64` | $(\boldsymbol{\phi}_{\text{obs}} - \boldsymbol{\phi}_{\text{rec}}) \cdot A$ |

### 2.3 Hyperparameters/Constants

| Parameter | Symbol | Value | Units | Description |
|-----------|--------|-------|-------|-------------|
| Resolution | $N$ | 120 | pixels | Pupil sampling resolution |
| Diameter | $D$ | 8 | m | Telescope primary mirror diameter |
| Sampling time | $\Delta t$ | 0.001 | s | Temporal sampling interval |
| Central obstruction | $\epsilon_c$ | 0.0 | ratio | Secondary mirror obstruction ratio |
| Number of Zernike modes | $J$ | 100 | — | Basis size for decomposition |
| Number of iterations | $N_{\text{iter}}$ | 10 | — | Temporal frames to process |
| Fried parameter | $r_0$ | 0.15 | m | Atmospheric coherence length |
| Outer scale | $L_0$ | 25 | m | Von Kármán outer scale |
| Wind speed | $v$ | 10 | m/s | Atmospheric wind velocity |
| Wind direction | $\theta_w$ | 0 | deg | Wind direction angle |
| Altitude | $h$ | 0 | m | Turbulent layer altitude |
| Zero-padding factor | $p$ | 4 | — | FFT padding multiplier |
| Wavelength | $\lambda$ | band-dependent | m | Source wavelength (I-band) |
| Magnitude | $m_v$ | 10 | mag | Source apparent magnitude |

### 2.4 Observed Data/Inputs

| Variable | Symbol | Shape | Source | Description |
|----------|--------|-------|--------|-------------|
| Pupil mask | $A$ | $(N, N)$ | Telescope | Binary aperture function |
| Atmospheric OPD | $\boldsymbol{\phi}_{\text{atm}}$ | $(N, N)$ | Atmosphere | Current turbulence realization |
| Zernike mode matrix | $\mathbf{Z}$ | $(N_p, J)$ | Precomputed | Zernike basis (pupil pixels) |
| Zernike modes (full) | $\mathbf{Z}_{\text{full}}$ | $(N^2, J)$ | Precomputed | Zernike basis (full resolution) |
| Pseudoinverse | $\mathbf{Z}^{\dagger}$ | $(J, N_p)$ | Precomputed | $\text{pinv}(\mathbf{Z})$ |

---

## 3. Preprocessing Pipeline

### Step 1: Telescope Initialization
**Input**: $(N, D, \Delta t, \epsilon_c)$  
**Output**: Telescope object `tel` with attributes:
- `tel.pupil` $\in \{0,1\}^{N \times N}$: Binary pupil mask
- `tel.pixelSize` $= D/N$ [m/pixel]
- `tel.resolution` $= N$
- `tel.D` $= D$

### Step 2: Source Initialization
**Input**: (optical band, magnitude)  
**Output**: Source object `ngs` with:
- `ngs.wavelength` $= \lambda$ [m]

**Operation**: Couple source to telescope: `ngs * tel`

### Step 3: Coordinate Grid Generation
**Input**: $(N, \text{pixelSize})$  
**Output**: $(X, Y)$ coordinate arrays

$$x_{ij} = \left(j - \frac{N}{2}\right) \cdot \text{pixelSize}, \quad y_{ij} = \left(i - \frac{N}{2}\right) \cdot \text{pixelSize}$$

for $i, j \in \{0, 1, \ldots, N-1\}$

### Step 4: Explicit Zernike Basis Generation (First 6 Modes)
**Input**: $(X, Y, D)$, Noll index mapping  
**Output**: `zernike_basis_2d` $\in \mathbb{R}^{6 \times N \times N}$

**Noll-to-$(n,m)$ mapping**:
| Noll $j$ | $(n, m)$ | Name |
|----------|----------|------|
| 1 | (0, 0) | Piston |
| 2 | (1, 1) | Tip |
| 3 | (1, -1) | Tilt |
| 4 | (2, 0) | Defocus |
| 5 | (2, -2) | Astigmatism |
| 6 | (2, 2) | Astigmatism |

**Normalized coordinates**:
$$r = \frac{\sqrt{x^2 + y^2}}{D/2}, \quad \theta = \arctan2(y, x)$$

### Step 5: OOPAO Zernike Basis Computation
**Input**: Telescope object, $J$  
**Output**: 
- `Z.modes` $\in \mathbb{R}^{N_p \times J}$: Zernike modes at pupil pixels
- `Z.modesFullRes` $\in \mathbb{R}^{N^2 \times J}$: Modes at full resolution

### Step 6: Pseudoinverse Computation
**Input**: `Z.modes`  
**Output**: $\mathbf{Z}^{\dagger} = \text{pinv}(\mathbf{Z})$

$$\mathbf{Z}^{\dagger} = (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T$$

### Step 7: Atmosphere Initialization
**Input**: $(r_0, L_0, v, \theta_w, h)$  
**Output**: Initialized `Atmosphere` object with turbulence phase screens

### Step 8: Test Phase Map Generation
**Input**: Zernike basis modes  
**Output**: Combined phase map

$$\phi_{\text{test}} = 0.5 \cdot Z_4 + 0.5 \cdot Z_6$$

(0.5 rad of Defocus + 0.5 rad of Astigmatism)

$$\text{OPD}_{\text{test}} = \frac{\phi_{\text{test}} \cdot \lambda}{2\pi}$$

---

## 4. Algorithm Steps

### 4.1 Forward Operator Algorithm

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 1 | Extract pupil | $A = \text{tel.pupil}$ | Telescope | $A \in \{0,1\}^{N \times N}$ |
| 2 | Complex field | $E = A \odot \exp(i \cdot \phi)$ | $A$, $\phi$ | $E \in \mathbb{C}^{N \times N}$ |
| 3 | Zero-pad | $E_p = \text{pad}(E, p_w)$, $p_w = \frac{N(p-1)}{2}$ | $E$, $p=4$ | $E_p \in \mathbb{C}^{4N \times 4N}$ |
| 4 | FFT shift (pre) | $E_s = \text{ifftshift}(E_p)$ | $E_p$ | $E_s$ |
| 5 | Fourier transform | $\hat{E} = \text{fft2}(E_s)$ | $E_s$ | $\hat{E}$ |
| 6 | FFT shift (post) | $E_f = \text{fftshift}(\hat{E})$ | $\hat{E}$ | $E_f$ |
| 7 | Intensity | $\text{PSF} = |E_f|^2$ | $E_f$ | $\text{PSF} \in \mathbb{R}^{4N \times 4N}$ |
| 8 | Normalize | $\text{PSF}_n = \frac{\text{PSF}}{\max(\text{PSF})}$ | PSF | $\text{PSF}_n \in [0,1]^{4N \times 4N}$ |

### 4.2 Inverse Operator Algorithm (Iterative Decomposition)

**Initialization**:
- `rmse_history` $\leftarrow []$
- `all_coeffs` $\leftarrow []$
- `all_opd_original` $\leftarrow []$
- `all_opd_reconstructed` $\leftarrow []$

**Per-Iteration Steps** (for $i = 1, \ldots, N_{\text{iter}}$):

| Step | Name | Formula | Input | Output |
|------|------|---------|-------|--------|
| 1 | Update atmosphere | $\text{atm.update()}$ | Atmosphere state | New OPD realization |
| 2 | Extract current OPD | $\boldsymbol{\phi}_{\text{obs}} = \text{atm.OPD}$ | Atmosphere | $\boldsymbol{\phi}_{\text{obs}} \in \mathbb{R}^{N \times N}$ |
| 3 | Mask extraction | $\boldsymbol{\phi}_m = \boldsymbol{\phi}_{\text{obs}}[A = 1]$ | $\boldsymbol{\phi}_{\text{obs}}$, $A$ | $\boldsymbol{\phi}_m \in \mathbb{R}^{N_p}$ |
| 4 | Coefficient estimation | $\mathbf{c} = \mathbf{Z}^{\dagger} \boldsymbol{\phi}_m$ | $\mathbf{Z}^{\dagger}$, $\boldsymbol{\phi}_m$ | $\mathbf{c} \in \mathbb{R}^{J}$ |
| 5 | OPD reconstruction | $\boldsymbol{\phi}_{\text{rec}} = \mathbf{Z}_{\text{full}} \mathbf{c}$ | $\mathbf{Z}_{\text{full}}$, $\mathbf{c}$ | $\boldsymbol{\phi}_{\text{rec}} \in \mathbb{R}^{N \times N}$ |
| 6 | Residual computation | $\boldsymbol{\epsilon} = (\boldsymbol{\phi}_{\text{obs}} - \boldsymbol{\phi}_{\text{rec}}) \odot A$ | $\boldsymbol{\phi}_{\text{obs}}$, $\boldsymbol{\phi}_{\text{rec}}$, $A$ | $\boldsymbol{\epsilon} \in \mathbb{R}^{N \times N}$ |
| 7 | RMSE calculation | $\text{RMSE}_i = \text{std}(\boldsymbol{\epsilon}[A=1])$ | $\boldsymbol{\epsilon}$, $A$ | $\text{RMSE}_i \in \mathbb{R}$ |
| 8 | Store results | Append to history lists | All computed values | Updated lists |

---

## 5. Stopping Criterion

**Primary**: Fixed iteration count
$$i = N_{\text{iter}} = 10$$

**No convergence-based termination** is implemented. The algorithm runs for exactly $N_{\text{iter}}$ iterations regardless of RMSE values.

---

## 6. Evaluation Metrics

### 6.1 Root Mean Square Error (RMSE)

$$\text{RMSE}_i = \sqrt{\frac{1}{N_p} \sum_{(x,y) \in \text{pupil}} \left( \phi_{\text{obs}}(x,y) - \phi_{\text{rec}}(x,y) \right)^2}$$

Equivalently computed as standard deviation within pupil:
$$\text{RMSE}_i = \text{std}\left( \boldsymbol{\epsilon}[A = 1] \right)$$

### 6.2 Summary Statistics

| Metric | Formula | Units |
|--------|---------|-------|
| Mean RMSE | $\bar{\text{RMSE}} = \frac{1}{N_{\text{iter}}} \sum_{i=1}^{N_{\text{iter}}} \text{RMSE}_i$ | m (reported in nm) |
| Final RMSE | $\text{RMSE}_{N_{\text{iter}}}$ | m (reported in nm) |
| Min RMSE | $\min_i \text{RMSE}_i$ | m (reported in nm) |
| Max RMSE | $\max_i \text{RMSE}_i$ | m (reported in nm) |
| Std RMSE | $\sqrt{\frac{1}{N_{\text{iter}}} \sum_{i=1}^{N_{\text{iter}}} (\text{RMSE}_i - \bar{\text{RMSE}})^2}$ | m (reported in nm) |

### 6.3 PSF Quality Metrics

| Metric | Formula |
|--------|---------|
| Peak value | $\max_{u,v} \text{PSF}(u,v)$ |
| Normalization | $\text{PSF}_{\text{norm}} = \text{PSF} / \max(\text{PSF})$ |

---

## 7. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  Parameters ──► Telescope ──► Source ──► Atmosphere             │
│       │              │            │           │                  │
│       │              ▼            ▼           ▼                  │
│       │         tel.pupil    ngs.λ      atm.OPD                 │
│       │              │                                           │
│       ▼              ▼                                           │
│  Coordinates ──► Zernike Basis ──► Pseudoinverse                │
│   (X, Y)           Z, Z_full         Z†                         │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│   FORWARD MODEL     │               │   INVERSE MODEL     │
├─────────────────────┤               ├─────────────────────┤
│ φ_test ──► E_pupil  │               │  Loop i=1..N_iter:  │
│    │                │               │    atm.update()     │
│    ▼                │               │    φ_obs ◄── atm    │
│ E_padded            │               │       │             │
│    │                │               │       ▼             │
│    ▼                │               │  c = Z† · φ_masked  │
│ FFT ──► E_focal     │               │       │             │
│    │                │               │       ▼             │
│    ▼                │               │ φ_rec = Z_full · c  │
│ PSF = |E|²          │               │       │             │
└─────────────────────┘               │       ▼             │
                                      │ RMSE = std(φ-φ_rec) │
                                      └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │     EVALUATION      │
                    ├─────────────────────┤
                    │ • RMSE statistics   │
                    │ • OPD comparison    │
                    │ • PSF visualization │
                    └─────────────────────┘
```

---

## 8. Notation Summary

| Symbol | Definition |
|--------|------------|
| $N$ | Resolution (pixels per side) |
| $D$ | Telescope diameter [m] |
| $J$ | Number of Zernike modes |
| $N_p$ | Number of valid pupil pixels |
| $\phi$ | Phase [radians] |
| OPD | Optical Path Difference [m] |
| $\lambda$ | Wavelength [m] |
| $A$ | Pupil amplitude mask |
| $\mathbf{Z}$ | Zernike mode matrix |
| $\mathbf{Z}^{\dagger}$ | Moore-Penrose pseudoinverse of $\mathbf{Z}$ |
| $\mathbf{c}$ | Zernike coefficient vector |
| $r_0$ | Fried parameter [m] |
| $L_0$ | Outer scale [m] |
| $\mathcal{F}\{\cdot\}$ | 2D Fourier transform |
| $\odot$ | Element-wise (Hadamard) product |