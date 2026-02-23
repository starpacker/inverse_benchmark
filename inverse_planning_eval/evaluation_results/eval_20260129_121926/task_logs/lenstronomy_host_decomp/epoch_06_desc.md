## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Particle Swarm Optimization for Quasar-Host Galaxy Image Decomposition",
  "algorithm_family": "Metaheuristic Optimization / Particle Swarm Optimization (PSO)",
  "summary": "PSO-based fitting algorithm that decomposes astronomical images into point source (quasar) and extended host galaxy (disk + bulge Sérsic profiles) components using a forward model with PSF convolution and linear amplitude solving."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\boldsymbol{\\theta}} \\chi^2(\\boldsymbol{\\theta}) = \\sum_{i,j} \\frac{(y_{ij} - \\hat{y}_{ij}(\\boldsymbol{\\theta}))^2}{\\sigma_{ij}^2} \\quad \\text{subject to} \\quad \\boldsymbol{\\theta}_{\\text{lower}} \\leq \\boldsymbol{\\theta} \\leq \\boldsymbol{\\theta}_{\\text{upper}}",
  "data_fidelity_term": "\\chi^2(\\boldsymbol{\\theta}) = \\sum_{i=0}^{N_{\\text{pix}}-1} \\sum_{j=0}^{N_{\\text{pix}}-1} \\frac{(y_{ij} - \\hat{y}_{ij}(\\boldsymbol{\\theta}))^2}{\\sigma_{ij}^2}, \\quad \\text{where } \\sigma_{ij}^2 = \\sigma_{\\text{bkg}}^2 + \\frac{\\max(y_{ij}, 0)}{t_{\\text{exp}}}",
  "regularization_term": "None",
  "constraints": "\\mathcal{I}_{\\mathcal{C}}(\\boldsymbol{\\theta}) = \\begin{cases} 0 & \\text{if } \\boldsymbol{\\theta} \\in \\mathcal{C} \\\\ +\\infty & \\text{otherwise} \\end{cases}, \\quad \\mathcal{C} = \\{\\boldsymbol{\\theta} : \\boldsymbol{\\theta}_{\\text{lower}} \\leq \\boldsymbol{\\theta} \\leq \\boldsymbol{\\theta}_{\\text{upper}}\\}"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\boldsymbol{\theta}_{\text{disk}}$ | $(5,)$ | float64 | $[R_s=1.0, e_1=0, e_2=0, c_x=0, c_y=0]$ | Disk component parameters (Sérsic ellipse): $R_{\text{sersic}}, e_1, e_2, \text{center}_x, \text{center}_y$ |
| $\boldsymbol{\theta}_{\text{bulge}}$ | $(3,)$ | float64 | $[R_s=0.5, c_x=0, c_y=0]$ | Bulge component parameters (Sérsic circular): $R_{\text{sersic}}, \text{center}_x, \text{center}_y$ |
| $\boldsymbol{\theta}_{\text{ps}}$ | $(2,)$ | float64 | $[\text{ra}=0, \text{dec}=0]$ | Point source position: $\text{ra}_{\text{image}}, \text{dec}_{\text{image}}$ |
| $A_{\text{disk}}, A_{\text{bulge}}, A_{\text{ps}}$ | $(1,), (1,), (1,)$ | float64 | Solved linearly | Amplitude coefficients for each component |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| N/A | — | — | — | PSO is primal-only; no dual variables |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{x}^{(p,k)}$ | $(N_{\text{params}},)$ | float64 | Random in bounds | Position of particle $p$ at iteration $k$ |
| $\mathbf{v}^{(p,k)}$ | $(N_{\text{params}},)$ | float64 | Random | Velocity of particle $p$ at iteration $k$ |
| $\mathbf{p}_{\text{best}}^{(p)}$ | $(N_{\text{params}},)$ | float64 | $\mathbf{x}^{(p,0)}$ | Personal best position for particle $p$ |
| $\mathbf{g}_{\text{best}}$ | $(N_{\text{params}},)$ | float64 | Best of initial | Global best position across all particles |
| $f_{\text{pbest}}^{(p)}$ | $(1,)$ | float64 | $\chi^2(\mathbf{x}^{(p,0)})$ | Personal best fitness for particle $p$ |
| $f_{\text{gbest}}$ | $(1,)$ | float64 | $\min_p f_{\text{pbest}}^{(p)}$ | Global best fitness |
| $\mathbf{M}_{\text{response}}$ | $(N_{\text{pix}}^2, N_{\text{comp}})$ | float64 | Computed per eval | Response matrix for linear amplitude solve |

### 2.4 Constants
| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $N_{\text{pix}}$ | scalar | int | 80 | Number of pixels per dimension |
| $\Delta_{\text{pix}}$ | scalar | float64 | 0.05 arcsec | Pixel scale |
| $\sigma_{\text{bkg}}$ | scalar | float64 | 0.1 | Background RMS noise |
| $t_{\text{exp}}$ | scalar | float64 | 100.0 s | Exposure time |
| $\text{FWHM}$ | scalar | float64 | 0.1 arcsec | PSF full-width half-maximum |
| $N_{\text{particles}}$ | scalar | int | 50 | Number of PSO particles |
| $N_{\text{iter}}$ | scalar | int | 50 | Number of PSO iterations |
| $\sigma_{\text{scale}}$ | scalar | float64 | 1.0 | PSO sigma scale parameter |
| $n_{\text{disk}}$ | scalar | float64 | 1.0 (fixed) | Disk Sérsic index |
| $n_{\text{bulge}}$ | scalar | float64 | 4.0 (fixed) | Bulge Sérsic index |
| $\omega$ | scalar | float64 | ~0.5 (library default) | PSO inertia weight |
| $c_1, c_2$ | scalar | float64 | ~1.5 (library default) | PSO cognitive/social coefficients |
| $\text{truncation}$ | scalar | float64 | 3.0 | PSF kernel truncation in FWHM units |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{Y}$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Observed image (simulated with noise) |
| $\mathbf{K}$ | $(N_{\text{psf}}, N_{\text{psf}})$ | float64 | PSF kernel (Gaussian) |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": 1,
      "step_name": "Compute coordinate transformation matrix",
      "formula": "\\mathbf{T} = \\begin{pmatrix} -\\Delta_{\\text{pix}} & 0 \\\\ 0 & \\Delta_{\\text{pix}} \\end{pmatrix}",
      "inputs": ["$\\Delta_{\\text{pix}}$: scalar"],
      "outputs": ["$\\mathbf{T}$: $(2, 2)$"],
      "assumptions": "Cartesian pixel grid with RA increasing leftward"
    },
    {
      "step_order": 2,
      "step_name": "Compute image center offset",
      "formula": "c_x = c_y = \\frac{N_{\\text{pix}} - 1}{2}, \\quad \\text{ra}_0 = -(T_{00} c_x + T_{01} c_y), \\quad \\text{dec}_0 = -(T_{10} c_x + T_{11} c_y)",
      "inputs": ["$N_{\\text{pix}}$: scalar", "$\\mathbf{T}$: $(2,2)$"],
      "outputs": ["$\\text{ra}_0$: scalar", "$\\text{dec}_0$: scalar"],
      "assumptions": "Image centered at angular origin (0, 0)"
    },
    {
      "step_order": 3,
      "step_name": "Generate pixel coordinate grids",
      "formula": "\\alpha_{ij} = \\text{ra}_0 + T_{00} i + T_{01} j, \\quad \\delta_{ij} = \\text{dec}_0 + T_{10} i + T_{11} j, \\quad i,j \\in [0, N_{\\text{pix}}-1]",
      "inputs": ["$\\text{ra}_0, \\text{dec}_0$: scalars", "$\\mathbf{T}$: $(2,2)$"],
      "outputs": ["$\\boldsymbol{\\alpha}$: $(N_{\\text{pix}}, N_{\\text{pix}})$", "$\\boldsymbol{\\delta}$: $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "assumptions": "Regular grid spacing"
    },
    {
      "step_order": 4,
      "step_name": "Generate Gaussian PSF kernel",
      "formula": "\\sigma_{\\text{psf}} = \\frac{\\text{FWHM}}{2\\sqrt{2\\ln 2}}, \\quad K_{ij} = \\frac{1}{2\\pi\\sigma_{\\text{psf}}^2} \\exp\\left(-\\frac{(i - c_K)^2 + (j - c_K)^2}{2\\sigma_{\\text{psf}}^2}\\right)",
      "inputs": ["$\\text{FWHM}$: scalar", "$\\Delta_{\\text{pix}}$: scalar", "truncation: scalar"],
      "outputs": ["$\\mathbf{K}$: $(N_{\\text{psf}}, N_{\\text{psf}})$ where $N_{\\text{psf}} = 2 \\lfloor \\text{truncation} \\cdot \\text{FWHM} / \\Delta_{\\text{pix}} \\rfloor + 1$"],
      "assumptions": "Normalized to unit sum, circular symmetric"
    },
    {
      "step_order": 5,
      "step_name": "Simulate clean image (ground truth)",
      "formula": "\\mathbf{Y}_{\\text{clean}} = \\mathbf{K} * (I_{\\text{disk}} + I_{\\text{bulge}}) + \\mathbf{K} * I_{\\text{ps}}",
      "inputs": ["$\\mathbf{K}$: PSF kernel", "Ground truth parameters"],
      "outputs": ["$\\mathbf{Y}_{\\text{clean}}$: $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "assumptions": "All components share same center in ground truth"
    },
    {
      "step_order": 6,
      "step_name": "Apply Poisson noise",
      "formula": "\\mathbf{C} = \\max(\\mathbf{Y}_{\\text{clean}} \\cdot t_{\\text{exp}}, 0), \\quad \\mathbf{C}_{\\text{noisy}} \\sim \\text{Poisson}(\\mathbf{C}), \\quad \\mathbf{N}_{\\text{poisson}} = \\frac{\\mathbf{C}_{\\text{noisy}}}{t_{\\text{exp}}} - \\mathbf{Y}_{\\text{clean}}",
      "inputs": ["$\\mathbf{Y}_{\\text{clean}}$: $(N_{\\text{pix}}, N_{\\text{pix}})$", "$t_{\\text{exp}}$: scalar"],
      "outputs": ["$\\mathbf{N}_{\\text{poisson}}$: $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "assumptions": "Photon counting statistics"
    },
    {
      "step_order": 7,
      "step_name": "Apply Gaussian background noise",
      "formula": "\\mathbf{N}_{\\text{bkg}} = \\sigma_{\\text{bkg}} \\cdot \\boldsymbol{\\epsilon}, \\quad \\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, 1)^{N_{\\text{pix}} \\times N_{\\text{pix}}}",
      "inputs": ["$\\sigma_{\\text{bkg}}$: scalar"],
      "outputs": ["$\\mathbf{N}_{\\text{bkg}}$: $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "assumptions": "Independent Gaussian noise per pixel"
    },
    {
      "step_order": 8,
      "step_name": "Generate observed image",
      "formula": "\\mathbf{Y} = \\mathbf{Y}_{\\text{clean}} + \\mathbf{N}_{\\text{bkg}} + \\mathbf{N}_{\\text{poisson}}",
      "inputs": ["$\\mathbf{Y}_{\\text{clean}}, \\mathbf{N}_{\\text{bkg}}, \\mathbf{N}_{\\text{poisson}}$: all $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "outputs": ["$\\mathbf{Y}$: $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "assumptions": "Additive noise model"
    }
  ]
}
```

## 4. ALGORITHM STEPS

### 4.1 INITIALIZATION
```json
{
  "steps": [
    {
      "step_order": 1,
      "variable_name": "$\\boldsymbol{\\theta}_{\\text{lower}}$",
      "initial_value": "[e_1^{\\text{disk}}=-0.5, e_2^{\\text{disk}}=-0.5, R_s^{\\text{disk}}=0.001, c_x=-10, c_y=-10, R_s^{\\text{bulge}}=0.001, \\text{ra}=-10, \\text{dec}=-10]",
      "shape": "$(N_{\\text{params}},)$ where $N_{\\text{params}}=8$",
      "dtype": "float64",
      "source": "kwargs_lower_source, kwargs_lower_ps"
    },
    {
      "step_order": 2,
      "variable_name": "$\\boldsymbol{\\theta}_{\\text{upper}}$",
      "initial_value": "[e_1^{\\text{disk}}=0.5, e_2^{\\text{disk}}=0.5, R_s^{\\text{disk}}=10, c_x=10, c_y=10, R_s^{\\text{bulge}}=10, \\text{ra}=10, \\text{dec}=10]",
      "shape": "$(N_{\\text{params}},)$",
      "dtype": "float64",
      "source": "kwargs_upper_source, kwargs_upper_ps"
    },
    {
      "step_order": 3,
      "variable_name": "$\\boldsymbol{\\theta}_{\\text{init}}$",
      "initial_value": "[R_s^{\\text{disk}}=1.0, e_1=0, e_2=0, c_x=0, c_y=0, R_s^{\\text{bulge}}=0.5, \\text{ra}=0, \\text{dec}=0]",
      "shape": "$(N_{\\text{params}},)$",
      "dtype": "float64",
      "source": "kwargs_source_init, kwargs_ps_init"
    },
    {
      "step_order": 4,
      "variable_name": "$\\boldsymbol{\\sigma}$",
      "initial_value": "[R_s^{\\text{disk}}=0.5, e_1=0.1, e_2=0.1, c_x=0.1, c_y=0.1, R_s^{\\text{bulge}}=0.3, \\text{ra}=0.01, \\text{dec}=0.01]",
      "shape": "$(N_{\\text{params}},)$",
      "dtype": "float64",
      "source": "kwargs_source_sigma, kwargs_ps_sigma (scaled by sigma_scale=1.0)"
    },
    {
      "step_order": 5,
      "variable_name": "$\\mathbf{x}^{(p,0)}$ for $p = 1, \\ldots, N_{\\text{particles}}$",
      "initial_value": "$x_i^{(p,0)} = \\theta_{\\text{init},i} + \\sigma_{\\text{scale}} \\cdot \\sigma_i \\cdot \\mathcal{U}(-1, 1)$, clamped to $[\\theta_{\\text{lower},i}, \\theta_{\\text{upper},i}]$",
      "shape": "$(N_{\\text{particles}}, N_{\\text{params}})$",
      "dtype": "float64",
      "source": "PSO initialization with uniform scatter"
    },
    {
      "step_order": 6,
      "variable_name": "$\\mathbf{v}^{(p,0)}$ for $p = 1, \\ldots, N_{\\text{particles}}$",
      "initial_value": "$v_i^{(p,0)} = 0.5 \\cdot \\sigma_i \\cdot \\mathcal{U}(-1, 1)$",
      "shape": "$(N_{\\text{particles}}, N_{\\text{params}})$",
      "dtype": "float64",
      "source": "Random initial velocities"
    },
    {
      "step_order": 7,
      "variable_name": "$f^{(p,0)}$ for $p = 1, \\ldots, N_{\\text{particles}}$",
      "initial_value": "$f^{(p,0)} = \\chi^2(\\mathbf{x}^{(p,0)})$ via forward model and linear solve",
      "shape": "$(N_{\\text{particles}},)$",
      "dtype": "float64",
      "source": "Initial fitness evaluation"
    },
    {
      "step_order": 8,
      "variable_name": "$\\mathbf{p}_{\\text{best}}^{(p)}$",
      "initial_value": "$\\mathbf{p}_{\\text{best}}^{(p)} = \\mathbf{x}^{(p,0)}$",
      "shape": "$(N_{\\text{particles}}, N_{\\text{params}})$",
      "dtype": "float64",
      "source": "Personal best initialized to starting position"
    },
    {
      "step_order": 9,
      "variable_name": "$f_{\\text{pbest}}^{(p)}$",
      "initial_value": "$f_{\\text{pbest}}^{(p)} = f^{(p,0)}$",
      "shape": "$(N_{\\text{particles}},)$",
      "dtype": "float64",
      "source": "Personal best fitness"
    },
    {
      "step_order": 10,
      "variable_name": "$\\mathbf{g}_{\\text{best}}, f_{\\text{gbest}}$",
      "initial_value": "$p^* = \\arg\\min_p f_{\\text{pbest}}^{(p)}, \\quad \\mathbf{g}_{\\text{best}} = \\mathbf{p}_{\\text{best}}^{(p^*)}, \\quad f_{\\text{gbest}} = f_{\\text{pbest}}^{(p^*)}$",
      "shape": "$(N_{\\text{params}},)$, scalar",
      "dtype": "float64",
      "source": "Global best from initial population"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "steps": [
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "$k = 0, 1, \\ldots, N_{\\text{iter}} - 1$",
      "description": "Main PSO iteration loop"
    },
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "$p = 1, \\ldots, N_{\\text{particles}}$",
      "description": "Particle update loop"
    },
    {
      "step_order": 11,
      "step_name": "Generate random coefficients",
      "formula": "$r_1, r_2 \\sim \\mathcal{U}(0, 1)$",
      "inputs": [],
      "outputs": ["$r_1$: scalar", "$r_2$: scalar"],
      "computational_note": "Independent uniform random numbers per particle per iteration"
    },
    {
      "step_order": 12,
      "step_name": "Update particle velocity",
      "formula": "$\\mathbf{v}^{(p,k+1)} = \\omega \\mathbf{v}^{(p,k)} + c_1 r_1 (\\mathbf{p}_{\\text{best}}^{(p)} - \\mathbf{x}^{(p,k)}) + c_2 r_2 (\\mathbf{g}_{\\text{best}} - \\mathbf{x}^{(p,k)})$",
      "inputs": ["$\\mathbf{v}^{(p,k)}$: $(N_{\\text{params}},)$", "$\\mathbf{x}^{(p,k)}$: $(N_{\\text{params}},)$", "$\\mathbf{p}_{\\text{best}}^{(p)}$: $(N_{\\text{params}},)$", "$\\mathbf{g}_{\\text{best}}$: $(N_{\\text{params}},)$"],
      "outputs": ["$\\mathbf{v}^{(p,k+1)}$: $(N_{\\text{params}},)$"],
      "computational_note": "Element-wise vector operations; $\\omega \\approx 0.5$, $c_1, c_2 \\approx 1.5$"
    },
    {
      "step_order": 13,
      "step_name": "Update particle position",
      "formula": "$\\tilde{\\mathbf{x}}^{(p,k+1)} = \\mathbf{x}^{(p,k)} + \\mathbf{v}^{(p,k+1)}$",
      "inputs": ["$\\mathbf{x}^{(p,k)}$: $(N_{\\text{params}},)$", "$\\mathbf{v}^{(p,k+1)}$: $(N_{\\text{params}},)$"],
      "outputs": ["$\\tilde{\\mathbf{x}}^{(p,k+1)}$: $(N_{\\text{params}},)$"],
      "computational_note": "Unconstrained position update"
    },
    {
      "step_order": 14,
      "step_name": "Apply box constraints (projection)",
      "formula": "$x_i^{(p,k+1)} = \\max(\\theta_{\\text{lower},i}, \\min(\\theta_{\\text{upper},i}, \\tilde{x}_i^{(p,k+1)}))$ for each $i$",
      "inputs": ["$\\tilde{\\mathbf{x}}^{(p,k+1)}$: $(N_{\\text{params}},)$", "$\\boldsymbol{\\theta}_{\\text{lower}}, \\boldsymbol{\\theta}_{\\text{upper}}$: $(N_{\\text{params}},)$"],
      "outputs": ["$\\mathbf{x}^{(p,k+1)}$: $(N_{\\text{params}},)$"],
      "computational_note": "Element-wise clamp enforcing check_bounds=True"
    },
    {
      "step_order": 15,
      "step_name": "[FORWARD] Compute disk Sérsic surface brightness",
      "formula": "$r_{\\text{ell},ij}^{\\text{disk}} = \\sqrt{q_{\\text{disk}} \\cdot [(\\alpha_{ij} - c_x)\\cos\\phi + (\\delta_{ij} - c_y)\\sin\\phi]^2 + \\frac{[-(\\alpha_{ij} - c_x)\\sin\\phi + (\\delta_{ij} - c_y)\\cos\\phi]^2}{q_{\\text{disk}}}}, \\quad I_{\\text{disk},ij}^{(\\text{norm})} = \\exp\\left\\{-b_{n=1}\\left[\\left(\\frac{r_{\\text{ell},ij}^{\\text{disk}}}{R_s^{\\text{disk}}}\\right)^{1/1} - 1\\right]\\right\\}$",
      "inputs": ["$\\boldsymbol{\\alpha}, \\boldsymbol{\\delta}$: $(N_{\\text{pix}}, N_{\\text{pix}})$", "$R_s^{\\text{disk}}, e_1, e_2, c_x, c_y$ from $\\mathbf{x}^{(p,k+1)}$"],
      "outputs": ["$\\mathbf{I}_{\\text{disk}}^{(\\text{norm})}$: $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "computational_note": "$b_1 \\approx 1.678$; ellipticity $(e_1, e_2) \\to (q, \\phi)$ via $q = \\frac{1-|e|}{1+|e|}$, $\\phi = \\frac{1}{2}\\arctan2(e_2, e_1)$, $|e| = \\sqrt{e_1^2 + e_2^2}$"
    },
    {
      "step_order": 16,
      "step_name": "[FORWARD] Compute bulge Sérsic surface brightness",
      "formula": "$r_{ij}^{\\text{bulge}} = \\sqrt{(\\alpha_{ij} - c_x)^2 + (\\delta_{ij} - c_y)^2}, \\quad I_{\\text{bulge},ij}^{(\\text{norm})} = \\exp\\left\\{-b_{n=4}\\left[\\left(\\frac{r_{ij}^{\\text{bulge}}}{R_s^{\\text{bulge}}}\\right)^{1/4} - 1\\right]\\right\\}$",
      "inputs": ["$\\boldsymbol{\\alpha}, \\boldsymbol{\\delta}$: $(N_{\\text{pix}}, N_{\\text{pix}})$", "$R_s^{\\text{bulge}}, c_x, c_y$ from $\\mathbf{x}^{(p,k+1)}$"],
      "outputs": ["$\\mathbf{I}_{\\text{bulge}}^{(\\text{norm})}$: $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "computational_note": "$b_4 \\approx 7.669$; circular profile (no ellipticity)"
    },
    {
      "step_order": 17,
      "step_name": "[FORWARD] Compute point source image",
      "formula": "$I_{\\text{ps},ij}^{(\\text{norm})} = \\delta(\\alpha_{ij} - \\text{ra}_{\\text{ps}}) \\cdot \\delta(\\delta_{ij} - \\text{dec}_{\\text{ps}})$",
      "inputs": ["$\\boldsymbol{\\alpha}, \\boldsymbol{\\delta}$: $(N_{\\text{pix}}, N_{\\text{pix}})$", "$\\text{ra}_{\\text{ps}}, \\text{dec}_{\\text{ps}}$ from $\\mathbf{x}^{(p,k+1)}$"],
      "outputs": ["$\\mathbf{I}_{\\text{ps}}^{(\\text{norm})}$: $(N_{\\text{pix}}, N_{\\text{pix}})$"],
      "computational_note": "Subpixel interpolation used; effectively places unit flux at (ra, dec)"
    },
    {
      "step_order": 18,
      "step_name": "[FORWARD]