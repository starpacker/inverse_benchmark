## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Gravitational Lens Modeling via PSO-MCMC",
  "algorithm_family": "Hybrid Global Optimization (Particle Swarm Optimization + Markov Chain Monte Carlo)",
  "summary": "A two-stage optimization pipeline that fits parametric gravitational lens models to observed imaging data by first performing global optimization via PSO, then refining and sampling the posterior via MCMC."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\boldsymbol{\\theta}} \\quad \\frac{1}{2} \\sum_{i,j} \\left( \\frac{y_{ij} - \\mathcal{F}(\\boldsymbol{\\theta})_{ij}}{\\sigma_{\\text{bkg}}} \\right)^2 + \\mathbb{I}_{\\mathcal{C}}(\\boldsymbol{\\theta})",
  "data_fidelity_term": "\\chi^2(\\boldsymbol{\\theta}) = \\sum_{i=1}^{N_{\\text{pix}}} \\sum_{j=1}^{N_{\\text{pix}}} \\left( \\frac{y_{ij} - \\mathcal{F}(\\boldsymbol{\\theta})_{ij}}{\\sigma_{\\text{bkg}}} \\right)^2 \\quad \\text{(L2 norm / Gaussian likelihood)}",
  "regularization_term": "None (implicit regularization through parameter bounds and MCMC prior)",
  "constraints": "\\mathbb{I}_{\\mathcal{C}}(\\boldsymbol{\\theta}) = \\begin{cases} 0 & \\text{if } \\boldsymbol{\\theta}_{\\text{lower}} \\leq \\boldsymbol{\\theta} \\leq \\boldsymbol{\\theta}_{\\text{upper}} \\\\ +\\infty & \\text{otherwise} \\end{cases}, \\quad \\text{Joint constraint: source center = point source center}"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 1.1 | Einstein radius of EPL lens |
| $\gamma_{\text{EPL}}$ | scalar | float64 | 2.0 | Power-law slope of EPL mass profile |
| $e_1^{\text{lens}}, e_2^{\text{lens}}$ | scalar each | float64 | 0.0, 0.0 | Ellipticity components of lens |
| $x_c^{\text{lens}}, y_c^{\text{lens}}$ | scalar each | float64 | 0.0, 0.0 | Center coordinates of lens |
| $\gamma_1, \gamma_2$ | scalar each | float64 | 0.0, 0.0 | External shear components |
| $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.03 | Source Sersic half-light radius |
| $n_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}, e_2^{\text{src}}$ | scalar each | float64 | 0.0, 0.0 | Source ellipticity |
| $x_c^{\text{src}}, y_c^{\text{src}}$ | scalar each | float64 | 0.0, 0.0 | Source center (joint with point source) |
| $R_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 0.1 | Lens light Sersic radius |
| $n_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 1.0 | Lens light Sersic index |
| $e_1^{\text{ll}}, e_2^{\text{ll}}$ | scalar each | float64 | 0.0, 0.0 | Lens light ellipticity |
| $x_c^{\text{ll}}, y_c^{\text{ll}}$ | scalar each | float64 | 0.0, 0.0 | Lens light center |
| $\mathbf{ra}_{\text{img}}, \mathbf{dec}_{\text{img}}$ | $(N_{\text{img}},)$ each | float64 | $\mathbf{x}_{\text{image}} + 0.01$, $\mathbf{y}_{\text{image}} - 0.01$ | Point source image positions |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| N/A | — | — | — | No explicit dual variables (handled internally by MCMC sampler) |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{p}_i$ | $(N_{\text{params}},)$ | float64 | Gaussian around init | PSO particle position for particle $i$ |
| $\mathbf{v}_i$ | $(N_{\text{params}},)$ | float64 | Random | PSO particle velocity |
| $\mathbf{p}_{\text{best},i}$ | $(N_{\text{params}},)$ | float64 | $\mathbf{p}_i$ | Personal best position for particle $i$ |
| $\mathbf{g}_{\text{best}}$ | $(N_{\text{params}},)$ | float64 | Best of initial | Global best position |
| $\chi^2_{\text{best},i}$ | scalar | float64 | $\chi^2(\mathbf{p}_i)$ | Personal best fitness |
| $\chi^2_{\text{global}}$ | scalar | float64 | $\min_i \chi^2(\mathbf{p}_i)$ | Global best fitness |
| $\mathbf{W}_k$ | $(N_{\text{walkers}}, N_{\text{params}})$ | float64 | Ball around PSO best | MCMC walker positions at step $k$ |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $\sigma_{\text{bkg}}$ | 0.5 | Background RMS noise level |
| $t_{\text{exp}}$ | 100 | Exposure time in seconds |
| $N_{\text{pix}}$ | 100 | Number of pixels per dimension |
| $\Delta_{\text{pix}}$ | 0.05 | Pixel scale in arcsec/pixel |
| FWHM | 0.2 | PSF full-width half-maximum in arcsec |
| $N_{\text{particles}}$ | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | 10 | Number of PSO iterations |
| $N_{\text{burn}}$ | 10 | MCMC burn-in steps |
| $N_{\text{run}}$ | 10 | MCMC sampling steps |
| walker_ratio | 4 | MCMC walkers per parameter |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | 1.0 | PSO initialization scale |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | 0.1 | MCMC proposal scale |
| $\sigma_{\text{pos}}$ | 0.004 | Image position uncertainty |
| $\text{tol}_{\text{src}}$ | 0.001 | Source position tolerance |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{y}$ | $(N_{\text{pix}}, N_{\text{pix}})$ = $(100, 100)$ | float64 | Observed noisy image |
| $\mathbf{x}_{\text{image}}, \mathbf{y}_{\text{image}}$ | $(N_{\text{img}},)$ | float64 | True lensed image positions (from lens equation solver) |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": 1,
      "step_name": "Initialize coordinate system",
      "formula": "\\mathbf{T} = \\begin{pmatrix} -\\Delta_{\\text{pix}} & 0 \\\\ 0 & \\Delta_{\\text{pix}} \\end{pmatrix}, \\quad c_x = c_y = \\frac{N_{\\text{pix}} - 1}{2}, \\quad \\text{RA}_0 = -(T_{00} c_x + T_{01} c_y), \\quad \\text{Dec}_0 = -(T_{10} c_x + T_{11} c_y)",
      "inputs": {"N_pix": "scalar (100)", "delta_pix": "scalar (0.05)"},
      "outputs": {"T": "(2,2)", "RA_0": "scalar (2.475)", "Dec_0": "scalar (-2.475)"},
      "assumptions": "Image centered at (RA, Dec) = (0, 0)"
    },
    {
      "step_order": 2,
      "step_name": "Configure PSF kernel",
      "formula": "\\text{PSF}(r) = \\exp\\left( -\\frac{r^2}{2\\sigma_{\\text{PSF}}^2} \\right), \\quad \\sigma_{\\text{PSF}} = \\frac{\\text{FWHM}}{2\\sqrt{2 \\ln 2}} \\approx 0.0849",
      "inputs": {"FWHM": "scalar (0.2)", "delta_pix": "scalar (0.05)"},
      "outputs": {"PSF_class": "PSF object with Gaussian kernel"},
      "assumptions": "Gaussian PSF, truncated at 3σ"
    },
    {
      "step_order": 3,
      "step_name": "Define true lens model parameters",
      "formula": "\\boldsymbol{\\theta}_{\\text{EPL}}^{\\text{true}} = (\\theta_E=1.0, \\gamma=1.96, e_1=0.07, e_2=-0.03, x_c=0, y_c=0), \\quad \\boldsymbol{\\theta}_{\\text{shear}}^{\\text{true}} = (\\gamma_1=0.01, \\gamma_2=0.01)",
      "inputs": {},
      "outputs": {"kwargs_lens_true": "list of 2 dicts"},
      "assumptions": "EPL + external shear model"
    },
    {
      "step_order": 4,
      "step_name": "Define true lens light parameters",
      "formula": "\\boldsymbol{\\theta}_{\\text{ll}}^{\\text{true}} = (A=400, R_{\\text{sersic}}=1.0, n_{\\text{sersic}}=2, x_c=0, y_c=0)",
      "inputs": {},
      "outputs": {"kwargs_lens_light_true": "list of 1 dict"},
      "assumptions": "Spherical Sersic profile"
    },
    {
      "step_order": 5,
      "step_name": "Define true source parameters",
      "formula": "\\boldsymbol{\\theta}_{\\text{src}}^{\\text{true}} = (A=160, R_{\\text{sersic}}=0.5, n_{\\text{sersic}}=7, x_c=0.1, y_c=0.3, e_1=0, e_2=0.1)",
      "inputs": {},
      "outputs": {"kwargs_source_true": "list of 1 dict"},
      "assumptions": "Elliptical Sersic profile"
    },
    {
      "step_order": 6,
      "step_name": "Solve lens equation for image positions",
      "formula": "\\boldsymbol{\\beta} = \\boldsymbol{\\theta} - \\boldsymbol{\\alpha}(\\boldsymbol{\\theta}), \\quad \\text{Find } \\{\\boldsymbol{\\theta}_k\\}_{k=1}^{N_{\\text{img}}} \\text{ such that } \\boldsymbol{\\beta} = (0.1, 0.3)",
      "inputs": {"ra_source": "0.1", "dec_source": "0.3", "kwargs_lens_true": "lens parameters"},
      "outputs": {"x_image": "(N_img,)", "y_image": "(N_img,)"},
      "assumptions": "Numerical root finding with numImages=4"
    },
    {
      "step_order": 7,
      "step_name": "Compute magnification at image positions",
      "formula": "\\mu_k = \\frac{1}{\\det(\\mathbf{J}_k)}, \\quad \\mathbf{J}_k = \\mathbf{I} - \\frac{\\partial \\boldsymbol{\\alpha}}{\\partial \\boldsymbol{\\theta}}\\bigg|_{\\boldsymbol{\\theta}_k}",
      "inputs": {"x_image": "(N_img,)", "y_image": "(N_img,)", "kwargs_lens_true": "lens parameters"},
      "outputs": {"mag": "(N_img,)"},
      "assumptions": "Magnification from Jacobian determinant"
    },
    {
      "step_order": 8,
      "step_name": "Set point source amplitudes",
      "formula": "A_k^{\\text{ps}} = |\\mu_k| \\times 100",
      "inputs": {"mag": "(N_img,)"},
      "outputs": {"kwargs_ps_true": "list with point_amp array"},
      "assumptions": "Point source flux proportional to magnification"
    },
    {
      "step_order": 9,
      "step_name": "Generate clean model image",
      "formula": "\\mathbf{I}_{\\text{clean}} = \\mathcal{F}(\\boldsymbol{\\theta}^{\\text{true}}) = \\text{PSF} * \\left[ I_{\\text{lens}}(\\boldsymbol{\\theta}) + I_{\\text{src}}(\\boldsymbol{\\beta}(\\boldsymbol{\\theta})) + \\sum_k A_k^{\\text{ps}} \\delta(\\boldsymbol{\\theta} - \\boldsymbol{\\theta}_k) \\right]",
      "inputs": {"all true kwargs": "model parameters", "PSF_class": "PSF object"},
      "outputs": {"image_sim": "(100, 100)"},
      "assumptions": "Forward model includes lens light, lensed source, and point sources"
    },
    {
      "step_order": 10,
      "step_name": "Add Poisson noise",
      "formula": "\\mathbf{C} = \\mathbf{I}_{\\text{clean}} \\times t_{\\text{exp}}, \\quad \\mathbf{C}[\\mathbf{C} < 0] = 0, \\quad \\tilde{\\mathbf{C}} \\sim \\text{Poisson}(\\lfloor \\mathbf{C} \\rfloor), \\quad \\mathbf{n}_{\\text{Poisson}} = \\frac{\\tilde{\\mathbf{C}}}{t_{\\text{exp}}} - \\mathbf{I}_{\\text{clean}}",
      "inputs": {"image_sim": "(100, 100)", "exp_time": "100"},
      "outputs": {"poisson_noise": "(100, 100)"},
      "assumptions": "Integer counts for Poisson sampling"
    },
    {
      "step_order": 11,
      "step_name": "Add Gaussian background noise",
      "formula": "\\mathbf{n}_{\\text{bkg}} \\sim \\mathcal{N}(0, \\sigma_{\\text{bkg}}^2 \\mathbf{I}), \\quad n_{\\text{bkg},ij} = \\sigma_{\\text{bkg}} \\times z_{ij}, \\quad z_{ij} \\sim \\mathcal{N}(0,1)",
      "inputs": {"image_shape": "(100, 100)", "background_rms": "0.5"},
      "outputs": {"bkg_noise": "(100, 100)"},
      "assumptions": "i.i.d. Gaussian noise"
    },
    {
      "step_order": 12,
      "step_name": "Combine to form observed image",
      "formula": "\\mathbf{y} = \\mathbf{I}_{\\text{clean}} + \\mathbf{n}_{\\text{bkg}} + \\mathbf{n}_{\\text{Poisson}}",
      "inputs": {"image_sim": "(100, 100)", "bkg_noise": "(100, 100)", "poisson_noise": "(100, 100)"},
      "outputs": {"image_noisy": "(100, 100)"},
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
      "variable_name": "kwargs_lens_init",
      "initial_value": "[{\\theta_E: 1.1, \\gamma: 2.0, e_1: 0, e_2: 0, x_c: 0, y_c: 0}, {\\gamma_1: 0, \\gamma_2: 0}]",
      "shape": "list of 2 dicts (8 scalars total)",
      "dtype": "float64",
      "source": "User-specified prior guess"
    },
    {
      "step_order": 2,
      "variable_name": "kwargs_source_init",
      "initial_value": "[{R_{\\text{sersic}}: 0.03, n_{\\text{sersic}}: 1, e_1: 0, e_2: 0, x_c: 0, y_c: 0}]",
      "shape": "list of 1 dict (6 scalars)",
      "dtype": "float64",
      "source": "User-specified prior guess"
    },
    {
      "step_order": 3,
      "variable_name": "kwargs_lens_light_init",
      "initial_value": "[{R_{\\text{sersic}}: 0.1, n_{\\text{sersic}}: 1, e_1: 0, e_2: 0, x_c: 0, y_c: 0}]",
      "shape": "list of 1 dict (6 scalars)",
      "dtype": "float64",
      "source": "User-specified prior guess"
    },
    {
      "step_order": 4,
      "variable_name": "kwargs_ps_init",
      "initial_value": "[{\\mathbf{ra}_{\\text{img}}: \\mathbf{x}_{\\text{image}} + 0.01, \\mathbf{dec}_{\\text{img}}: \\mathbf{y}_{\\text{image}} - 0.01}]",
      "shape": "list of 1 dict (2 arrays of size N_img)",
      "dtype": "float64",
      "source": "Perturbed from true image positions"
    },
    {
      "step_order": 5,
      "variable_name": "bounds_lower",
      "initial_value": "\\boldsymbol{\\theta}_{\\text{lower}} = [\\theta_E: 0, e_1: -0.5, ..., \\gamma_1: -0.2, R_{\\text{sersic}}^{\\text{src}}: 0.001, ..., \\mathbf{ra}_{\\text{img}}: -10\\mathbf{1}]",
      "shape": "(N_params,)",
      "dtype": "float64",
      "source": "User-specified parameter bounds"
    },
    {
      "step_order": 6,
      "variable_name": "bounds_upper",
      "initial_value": "\\boldsymbol{\\theta}_{\\text{upper}} = [\\theta_E: 10, e_1: 0.5, ..., \\gamma_1: 0.2, R_{\\text{sersic}}^{\\text{src}}: 10, ..., \\mathbf{ra}_{\\text{img}}: 10\\mathbf{1}]",
      "shape": "(N_params,)",
      "dtype": "float64",
      "source": "User-specified parameter bounds"
    },
    {
      "step_order": 7,
      "variable_name": "sigma_init",
      "initial_value": "\\boldsymbol{\\sigma} = [\\theta_E: 0.1, e_1: 0.2, e_2: 0.2, \\gamma: 0.1, x_c: 0.1, y_c: 0.1, \\gamma_1: 0.1, \\gamma_2: 0.1, ...]",
      "shape": "(N_params,)",
      "dtype": "float64",
      "source": "User-specified exploration widths"
    },
    {
      "step_order": 8,
      "variable_name": "PSO particles",
      "initial_value": "\\mathbf{p}_i^{(0)} \\sim \\mathcal{N}(\\boldsymbol{\\theta}_{\\text{init}}, (\\sigma_{\\text{scale}} \\cdot \\boldsymbol{\\sigma})^2), \\quad i = 1, ..., N_{\\text{particles}}",
      "shape": "(N_particles, N_params) = (50, N_params)",
      "dtype": "float64",
      "source": "Gaussian sampling around initial values"
    },
    {
      "step_order": 9,
      "variable_name": "PSO velocities",
      "initial_value": "\\mathbf{v}_i^{(0)} \\sim \\text{Uniform}(-|\\boldsymbol{\\theta}_{\\text{upper}} - \\boldsymbol{\\theta}_{\\text{lower}}|, |\\boldsymbol{\\theta}_{\\text{upper}} - \\boldsymbol{\\theta}_{\\text{lower}}|)",
      "shape": "(N_particles, N_params)",
      "dtype": "float64",
      "source": "Random initialization within bounds range"
    },
    {
      "step_order": 10,
      "variable_name": "Personal bests",
      "initial_value": "\\mathbf{p}_{\\text{best},i}^{(0)} = \\mathbf{p}_i^{(0)}, \\quad \\chi^2_{\\text{best},i}^{(0)} = \\chi^2(\\mathbf{p}_i^{(0)})",
      "shape": "(N_particles, N_params), (N_particles,)",
      "dtype": "float64",
      "source": "Copy of initial positions"
    },
    {
      "step_order": 11,
      "variable_name": "Global best",
      "initial_value": "\\mathbf{g}_{\\text{best}}^{(0)} = \\mathbf{p}_{\\text{best},i^*}^{(0)}, \\quad i^* = \\arg\\min_i \\chi^2_{\\text{best},i}^{(0)}",
      "shape": "(N_params,)",
      "dtype": "float64",
      "source": "Best among initial particles"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "steps": [
    {
      "step_order": 12,
      "step_name": "BEGIN_LOOP (PSO iterations)",
      "formula": "\\text{for } t = 1, ..., N_{\\text{iter}}^{\\text{PSO}} = 10:",
      "inputs": {},
      "outputs": {},
      "computational_note": "Outer loop over PSO iterations"
    },
    {
      "step_order": 13,
      "step_name": "BEGIN_LOOP (particles)",
      "formula": "\\text{for } i = 1, ..., N_{\\text{particles}} = 50:",
      "inputs": {},
      "outputs": {},
      "computational_note": "Inner loop over particles"
    },
    {
      "step_order": 14,
      "step_name": "Generate random coefficients",
      "formula": "r_1, r_2 \\sim \\text{Uniform}(0, 1)",
      "inputs": {},
      "outputs": {"r_1": "scalar", "r_2": "scalar"},
      "computational_note": "Independent random numbers for cognitive and social terms"
    },
    {
      "step_order": 15,
      "step_name": "Update velocity",
      "formula": "\\mathbf{v}_i^{(t)} = \\omega \\mathbf{v}_i^{(t-1)} + c_1 r_1 (\\mathbf{p}_{\\text{best},i} - \\mathbf{p}_i^{(t-1)}) + c_2 r_2 (\\mathbf{g}_{\\text{best}} - \\mathbf{p}_i^{(t-1)})",
      "inputs": {"v_i^{(t-1)}": "(N_params,)", "p_best_i": "(N_params,)", "g_best": "(N_params,)", "p_i^{(t-1)}": "(N_params,)"},
      "outputs": {"v_i^{(t)}": "(N_params,)"},
      "computational_note": "Standard PSO velocity update with inertia ω, cognitive c₁, social c₂ coefficients"
    },
    {
      "step_order": 16,
      "step_name": "Update position",
      "formula": "\\tilde{\\mathbf{p}}_i^{(t)} = \\mathbf{p}_i^{(t-1)} + \\mathbf{v}_i^{(t)}",
      "inputs": {"p_i^{(t-1)}": "(N_params,)", "v_i^{(t)}": "(N_params,)"},
      "outputs": {"p_i_tilde^{(t)}": "(N_params,)"},
      "computational_note": "Unconstrained position update"
    },
    {
      "step_order": 17,
      "step_name": "Project onto feasible region",
      "formula": "\\mathbf{p}_i^{(t)} = \\text{clip}(\\tilde{\\mathbf{p}}_i^{(t)}, \\boldsymbol{\\theta}_{\\text{lower}}, \\boldsymbol{\\theta}_{\\text{upper}})",
      "inputs": {"p_i_tilde^{(t)}": "(N_params,)", "theta_lower": "(N_params,)", "theta_upper": "(N_params,)"},
      "outputs": {"p_i^{(t)}": "(N_params,)"},
      "computational_note": "Element-wise clipping to enforce box constraints"
    },
    {
      "step_order": 18,
      "step_name": "[FORWARD] Compute lens deflection field",
      "formula": "\\boldsymbol{\\alpha}(\\boldsymbol{\\theta}) = \\boldsymbol{\\alpha}_{\\text{EPL}}(\\boldsymbol{\\theta}; \\theta_E, \\gamma, e_1, e_2, x_c, y_c) + \\boldsymbol{\\alpha}_{\\text{shear}}(\\boldsymbol{\\theta}; \\gamma_1, \\gamma_2)",
      "inputs": {"theta_grid": "(N_pix, N_pix, 2)", "lens_params": "from p_i^{(t)}"},
      "outputs": {"alpha": "(N_pix, N_pix, 2)"},
      "computational_note": "EPL deflection + linear shear"
    },
    {
      "step_order": 19,
      "step_name": "[FORWARD] Apply lens equation (ray trace)",
      "formula": "\\boldsymbol{\\beta}_{ij} = \\boldsymbol{\\theta}_{ij} - \\boldsymbol{\\alpha}(\\boldsymbol{\\theta}_{ij})",
      "inputs": {"theta_grid": "(N_pix, N_pix, 2)", "alpha": "(N_pix, N_pix, 2)"},
      "outputs": {"beta_grid": "(N_pix, N_pix, 2)"},
      "computational_note": "Map image plane to source plane"
    },
    {
      "step_order": 20,
      "step_name": "[FORWARD] Evaluate source light at traced positions",
      "formula": "I_{\\text{src},ij} = A_{\\text{src}} \\exp\\left( -b_n \\left[ \\left( \\frac{r_{\\text{ell}}(\\boldsymbol{\\beta}_{ij})}{R_{\\text{sersic}}^{\\text{src}}} \\right)^{1/n_{\\text{sersic}}^{\\text{src}}} - 1 \\right] \\right)",
      "inputs": {"beta_grid": "(N_pix, N_pix, 2)", "source_params": "from p_i^{(t)}"},
      "outputs": {"I_src": "(N_pix, N_pix)"},
      "computational_note": "Elliptical Sersic profile, r_ell includes ellipticity