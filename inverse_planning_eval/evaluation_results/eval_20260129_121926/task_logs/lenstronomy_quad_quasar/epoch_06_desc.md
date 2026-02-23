## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Gravitational Lens Modeling via PSO-MCMC",
  "algorithm_family": "Hybrid Metaheuristic-Bayesian Inference",
  "summary": "Sequential optimization using Particle Swarm Optimization for global parameter search followed by Markov Chain Monte Carlo sampling for posterior estimation in strong gravitational lens modeling."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\boldsymbol{\\theta}} \\left[ \\frac{1}{2} \\sum_{i=1}^{N_{\\text{pix}}} \\frac{(d_i - m_i(\\boldsymbol{\\theta}))^2}{\\sigma_i^2} - \\log p(\\boldsymbol{\\theta}) \\right] \\quad \\text{subject to} \\quad \\boldsymbol{\\theta}_{\\text{lower}} \\leq \\boldsymbol{\\theta} \\leq \\boldsymbol{\\theta}_{\\text{upper}}",
  "data_fidelity_term": "\\chi^2(\\boldsymbol{\\theta}) = \\sum_{i=1}^{N_{\\text{pix}}} \\frac{(d_i - m_i(\\boldsymbol{\\theta}))^2}{\\sigma_i^2}, \\quad \\text{where } \\sigma_i^2 = \\sigma_{\\text{bkg}}^2 + \\frac{m_i(\\boldsymbol{\\theta})}{t_{\\text{exp}}}",
  "regularization_term": "\\log p(\\boldsymbol{\\theta}) = -\\frac{(e_1^{\\text{lens}} - 0)^2}{2 \\cdot 0.2^2} - \\frac{(e_2^{\\text{lens}} - 0)^2}{2 \\cdot 0.2^2}",
  "constraints": "\\mathbb{I}_{[\\boldsymbol{\\theta}_{\\text{lower}}, \\boldsymbol{\\theta}_{\\text{upper}}]}(\\boldsymbol{\\theta}) = \\begin{cases} 0 & \\text{if } \\boldsymbol{\\theta}_{\\text{lower}} \\leq \\boldsymbol{\\theta} \\leq \\boldsymbol{\\theta}_{\\text{upper}} \\\\ +\\infty & \\text{otherwise} \\end{cases}"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta_E$ | scalar | float64 | 1.2 | Einstein radius of EPL lens (arcsec) |
| $\gamma^{\text{lens}}$ | scalar | float64 | 2.0 | Power-law slope of EPL mass profile |
| $e_1^{\text{lens}}$ | scalar | float64 | 0.0 | First ellipticity component of lens |
| $e_2^{\text{lens}}$ | scalar | float64 | 0.0 | Second ellipticity component of lens |
| $x_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center x-coordinate (arcsec) |
| $y_c^{\text{lens}}$ | scalar | float64 | 0.0 | Lens center y-coordinate (arcsec) |
| $\gamma_1$ | scalar | float64 | 0.0 | First external shear component |
| $\gamma_2$ | scalar | float64 | 0.0 | Second external shear component |
| $R_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 0.03 | Source Sersic half-light radius (arcsec) |
| $n_{\text{sersic}}^{\text{src}}$ | scalar | float64 | 1.0 | Source Sersic index |
| $e_1^{\text{src}}$ | scalar | float64 | 0.0 | Source first ellipticity component |
| $e_2^{\text{src}}$ | scalar | float64 | 0.0 | Source second ellipticity component |
| $x_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center x-coordinate (arcsec) |
| $y_c^{\text{src}}$ | scalar | float64 | 0.0 | Source center y-coordinate (arcsec) |
| $R_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 0.1 | Lens light Sersic half-light radius |
| $n_{\text{sersic}}^{\text{ll}}$ | scalar | float64 | 1.0 | Lens light Sersic index |
| $e_1^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light first ellipticity |
| $e_2^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light second ellipticity |
| $x_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center x-coordinate |
| $y_c^{\text{ll}}$ | scalar | float64 | 0.0 | Lens light center y-coordinate |
| $\mathbf{x}_{\text{img}}$ | (4,) | float64 | $x_{\text{image}}^{\text{true}} + 0.01$ | RA positions of lensed point source images |
| $\mathbf{y}_{\text{img}}$ | (4,) | float64 | $y_{\text{image}}^{\text{true}} - 0.01$ | Dec positions of lensed point source images |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| — | — | — | — | No explicit dual variables in this formulation |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{v}_p$ | $(N_{\text{part}}, N_{\text{param}})$ | float64 | Sampled from $\mathcal{U}(-\boldsymbol{\sigma}, \boldsymbol{\sigma})$ | PSO particle velocities |
| $\mathbf{X}_p$ | $(N_{\text{part}}, N_{\text{param}})$ | float64 | Sampled around init | PSO particle positions |
| $\mathbf{p}_{\text{best}}$ | $(N_{\text{part}}, N_{\text{param}})$ | float64 | $\mathbf{X}_p^{(0)}$ | Personal best positions for each particle |
| $\mathbf{g}_{\text{best}}$ | $(N_{\text{param}},)$ | float64 | Best of initial | Global best position across swarm |
| $\chi^2_{\text{best}}$ | scalar | float64 | $+\infty$ | Best chi-squared value found |
| $\mathbf{W}$ | $(N_{\text{walk}}, N_{\text{param}})$ | float64 | Ball around PSO result | MCMC walker positions |
| $\mathbf{m}(\boldsymbol{\theta})$ | $(N_{\text{pix}}, N_{\text{pix}})$ | float64 | Computed | Model predicted image |

### 2.4 Constants
| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $\sigma_{\text{bkg}}$ | scalar | float64 | 0.5 | Background RMS noise (counts/s) |
| $t_{\text{exp}}$ | scalar | float64 | 100.0 | Exposure time (seconds) |
| $N_{\text{pix}}$ | scalar | int | 100 | Number of pixels per side |
| $\Delta_{\text{pix}}$ | scalar | float64 | 0.05 | Pixel scale (arcsec/pixel) |
| $\text{FWHM}$ | scalar | float64 | 0.1 | PSF full-width half-maximum (arcsec) |
| $N_{\text{part}}$ | scalar | int | 50 | Number of PSO particles |
| $N_{\text{iter}}^{\text{PSO}}$ | scalar | int | 10 | Number of PSO iterations |
| $N_{\text{walk}}$ | scalar | int | 50 | Number of MCMC walkers |
| $N_{\text{burn}}$ | scalar | int | 10 | Number of MCMC burn-in steps |
| $N_{\text{run}}$ | scalar | int | 10 | Number of MCMC production steps |
| $\sigma_{\text{scale}}^{\text{PSO}}$ | scalar | float64 | 1.0 | PSO sigma scale factor |
| $\sigma_{\text{scale}}^{\text{MCMC}}$ | scalar | float64 | 0.1 | MCMC initialization scale |
| $\omega$ | scalar | float64 | 0.5 | PSO inertia weight (typical) |
| $c_1$ | scalar | float64 | 1.5 | PSO cognitive coefficient |
| $c_2$ | scalar | float64 | 1.5 | PSO social coefficient |
| $a_{\text{stretch}}$ | scalar | float64 | 2.0 | MCMC affine stretch parameter |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{d}$ | $(100, 100)$ | float64 | Observed (simulated) noisy lensed image |
| $\mathbf{T}$ | $(2, 2)$ | float64 | Pixel-to-angle transformation matrix |
| $\text{PSF}$ | kernel | float64 | Gaussian PSF with FWHM=0.1 arcsec |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": 1,
      "step_name": "Compute coordinate transformation",
      "formula": "\\mathbf{T} = \\begin{pmatrix} -\\Delta_{\\text{pix}} & 0 \\\\ 0 & \\Delta_{\\text{pix}} \\end{pmatrix}, \\quad \\text{RA}_0 = -(T_{00} \\cdot c_x + T_{01} \\cdot c_y), \\quad \\text{Dec}_0 = -(T_{10} \\cdot c_x + T_{11} \\cdot c_y)",
      "inputs": {"deltaPix": "scalar", "numPix": "int"},
      "outputs": {"T": "(2,2)", "ra_at_xy_0": "scalar", "dec_at_xy_0": "scalar"},
      "assumptions": "Image centered at (0,0) in angular coordinates; c_x = c_y = (numPix-1)/2 = 49.5"
    },
    {
      "step_order": 2,
      "step_name": "Generate true lens model parameters",
      "formula": "\\gamma_1 = \\gamma \\cos(2\\phi), \\quad \\gamma_2 = \\gamma \\sin(2\\phi) \\quad \\text{with } \\phi=0.1, \\gamma=0.02",
      "inputs": {"phi": "0.1 rad", "gamma_shear": "0.02"},
      "outputs": {"kwargs_lens": "list of dicts"},
      "assumptions": "EPL profile: θ_E=1.0, γ=1.96, e1=0.1, e2=0.2, center at origin"
    },
    {
      "step_order": 3,
      "step_name": "Solve lens equation for image positions",
      "formula": "\\boldsymbol{\\beta} = \\boldsymbol{\\theta} - \\boldsymbol{\\alpha}(\\boldsymbol{\\theta}), \\quad \\text{find } \\{\\boldsymbol{\\theta}_i\\}_{i=1}^{4} \\text{ such that } \\boldsymbol{\\beta} = (0, 0.1)",
      "inputs": {"ra_source": "0.0", "dec_source": "0.1", "kwargs_lens": "list"},
      "outputs": {"x_image": "(4,)", "y_image": "(4,)"},
      "assumptions": "Search window = numPix × deltaPix = 5 arcsec; minimum separation = deltaPix"
    },
    {
      "step_order": 4,
      "step_name": "Compute point source magnifications",
      "formula": "\\mu_i = \\frac{1}{\\det(\\mathbf{A}_i)}, \\quad \\mathbf{A} = \\mathbf{I} - \\nabla\\nabla\\psi, \\quad \\mu_i^{\\text{pert}} \\sim \\mathcal{N}(|\\mu_i|, 0.5)",
      "inputs": {"x_image": "(4,)", "y_image": "(4,)", "kwargs_lens": "list"},
      "outputs": {"point_amp": "(4,) = 100 × μ_pert"},
      "assumptions": "Absolute magnification used; Gaussian perturbation added"
    },
    {
      "step_order": 5,
      "step_name": "Generate noiseless model image",
      "formula": "\\mathbf{m} = (\\mathbf{S}_{\\text{lens}} + \\mathbf{S}_{\\text{src}}^{\\text{lensed}} + \\mathbf{S}_{\\text{PS}}) * \\text{PSF}",
      "inputs": {"kwargs_lens": "list", "kwargs_source": "list", "kwargs_lens_light": "list", "kwargs_ps": "list"},
      "outputs": {"image_sim": "(100,100)"},
      "assumptions": "Convolution performed in Fourier space; supersampling_factor=1"
    },
    {
      "step_order": 6,
      "step_name": "Add Poisson noise",
      "formula": "\\mathbf{c} = \\mathbf{m} \\cdot t_{\\text{exp}}, \\quad \\mathbf{c}[\\mathbf{c}<0] = 0, \\quad \\mathbf{c}_{\\text{noisy}} \\sim \\text{Poisson}(\\mathbf{c}), \\quad \\mathbf{n}_{\\text{Poisson}} = \\frac{\\mathbf{c}_{\\text{noisy}}}{t_{\\text{exp}}} - \\mathbf{m}",
      "inputs": {"image_sim": "(100,100)", "exp_time": "100"},
      "outputs": {"poisson_noise": "(100,100)"},
      "assumptions": "Negative counts clipped to zero before Poisson sampling"
    },
    {
      "step_order": 7,
      "step_name": "Add Gaussian background noise",
      "formula": "\\mathbf{n}_{\\text{bkg}} \\sim \\mathcal{N}(0, \\sigma_{\\text{bkg}}^2 \\mathbf{I}), \\quad \\sigma_{\\text{bkg}} = 0.5",
      "inputs": {"shape": "(100,100)", "background_rms": "0.5"},
      "outputs": {"bkg_noise": "(100,100)"},
      "assumptions": "i.i.d. Gaussian noise per pixel"
    },
    {
      "step_order": 8,
      "step_name": "Combine to create observed image",
      "formula": "\\mathbf{d} = \\mathbf{m} + \\mathbf{n}_{\\text{bkg}} + \\mathbf{n}_{\\text{Poisson}}",
      "inputs": {"image_sim": "(100,100)", "bkg_noise": "(100,100)", "poisson_noise": "(100,100)"},
      "outputs": {"d": "(100,100)"},
      "assumptions": "Final observed data used for fitting"
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
      "variable_name": "θ_init (parameter vector)",
      "initial_value": "[θ_E=1.2, γ=2.0, e1=0, e2=0, x_c=0, y_c=0, γ1=0, γ2=0, R_src=0.03, n_src=1, e1_src=0, e2_src=0, x_src=0, y_src=0, R_ll=0.1, n_ll=1, e1_ll=0, e2_ll=0, x_ll=0, y_ll=0, x_img, y_img]",
      "shape": "(N_param,)",
      "dtype": "float64",
      "source": "kwargs_*_init dictionaries"
    },
    {
      "step_order": 2,
      "variable_name": "σ (parameter uncertainties)",
      "initial_value": "[0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.02, 0.02]",
      "shape": "(N_param,)",
      "dtype": "float64",
      "source": "kwargs_*_sigma dictionaries"
    },
    {
      "step_order": 3,
      "variable_name": "θ_lower",
      "initial_value": "[0, 1.5, -0.5, -0.5, -10, -10, -0.5, -0.5, 0.001, 0.5, -0.5, -0.5, -10, -10, 0.001, 0.5, -0.5, -0.5, -10, -10, -10, -10]",
      "shape": "(N_param,)",
      "dtype": "float64",
      "source": "kwargs_lower_* dictionaries"
    },
    {
      "step_order": 4,
      "variable_name": "θ_upper",
      "initial_value": "[10, 2.5, 0.5, 0.5, 10, 10, 0.5, 0.5, 10, 5, 0.5, 0.5, 10, 10, 10, 5, 0.5, 0.5, 10, 10, 10, 10]",
      "shape": "(N_param,)",
      "dtype": "float64",
      "source": "kwargs_upper_* dictionaries"
    },
    {
      "step_order": 5,
      "variable_name": "X_p (PSO particle positions)",
      "initial_value": "X_p^{(i,j)} = \\theta_{\\text{init},j} + \\sigma_{\\text{scale}} \\cdot \\sigma_j \\cdot U_{ij}, \\quad U_{ij} \\sim \\mathcal{U}(-1,1)",
      "shape": "(50, N_param)",
      "dtype": "float64",
      "source": "Random initialization around θ_init"
    },
    {
      "step_order": 6,
      "variable_name": "v_p (PSO particle velocities)",
      "initial_value": "v_p^{(i,j)} = \\sigma_{\\text{scale}} \\cdot \\sigma_j \\cdot U_{ij}, \\quad U_{ij} \\sim \\mathcal{U}(-1,1)",
      "shape": "(50, N_param)",
      "dtype": "float64",
      "source": "Random initialization"
    },
    {
      "step_order": 7,
      "variable_name": "p_best (personal best positions)",
      "initial_value": "p_{\\text{best}}^{(i)} = X_p^{(i)}",
      "shape": "(50, N_param)",
      "dtype": "float64",
      "source": "Copy of initial positions"
    },
    {
      "step_order": 8,
      "variable_name": "χ²_p (personal best fitness)",
      "initial_value": "\\chi^2_{p}^{(i)} = \\chi^2(X_p^{(i)})",
      "shape": "(50,)",
      "dtype": "float64",
      "source": "Evaluated from initial positions"
    },
    {
      "step_order": 9,
      "variable_name": "g_best (global best position)",
      "initial_value": "g_{\\text{best}} = \\arg\\min_{i} \\chi^2_p^{(i)}",
      "shape": "(N_param,)",
      "dtype": "float64",
      "source": "Best of initial evaluations"
    },
    {
      "step_order": 10,
      "variable_name": "χ²_global (global best fitness)",
      "initial_value": "\\chi^2_{\\text{global}} = \\min_i \\chi^2_p^{(i)}",
      "shape": "scalar",
      "dtype": "float64",
      "source": "Minimum of initial evaluations"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "steps": [
    {
      "step_order": 11,
      "step_name": "BEGIN_LOOP (PSO outer iteration)",
      "formula": "\\text{for } k = 1, \\ldots, N_{\\text{iter}}^{\\text{PSO}} = 10",
      "inputs": {},
      "outputs": {},
      "computational_note": "Main PSO iteration loop"
    },
    {
      "step_order": 12,
      "step_name": "BEGIN_LOOP (particle update)",
      "formula": "\\text{for } i = 1, \\ldots, N_{\\text{part}} = 50",
      "inputs": {},
      "outputs": {},
      "computational_note": "Loop over all particles"
    },
    {
      "step_order": 13,
      "step_name": "Generate random coefficients",
      "formula": "r_1, r_2 \\sim \\mathcal{U}(0, 1)",
      "inputs": {},
      "outputs": {"r_1": "scalar", "r_2": "scalar"},
      "computational_note": "Independent uniform random numbers"
    },
    {
      "step_order": 14,
      "step_name": "Update particle velocity",
      "formula": "\\mathbf{v}_p^{(i)} \\leftarrow \\omega \\mathbf{v}_p^{(i)} + c_1 r_1 (\\mathbf{p}_{\\text{best}}^{(i)} - \\mathbf{X}_p^{(i)}) + c_2 r_2 (\\mathbf{g}_{\\text{best}} - \\mathbf{X}_p^{(i)})",
      "inputs": {"v_p^(i)": "(N_param,)", "X_p^(i)": "(N_param,)", "p_best^(i)": "(N_param,)", "g_best": "(N_param,)", "ω": "0.5", "c_1": "1.5", "c_2": "1.5"},
      "outputs": {"v_p^(i)": "(N_param,)"},
      "computational_note": "Inertia + cognitive + social components"
    },
    {
      "step_order": 15,
      "step_name": "Update particle position",
      "formula": "\\mathbf{X}_p^{(i)} \\leftarrow \\mathbf{X}_p^{(i)} + \\mathbf{v}_p^{(i)}",
      "inputs": {"X_p^(i)": "(N_param,)", "v_p^(i)": "(N_param,)"},
      "outputs": {"X_p^(i)": "(N_param,)"},
      "computational_note": "Position update step"
    },
    {
      "step_order": 16,
      "step_name": "Enforce box constraints",
      "formula": "X_{p,j}^{(i)} \\leftarrow \\max(\\theta_{\\text{lower},j}, \\min(\\theta_{\\text{upper},j}, X_{p,j}^{(i)})), \\quad \\forall j",
      "inputs": {"X_p^(i)": "(N_param,)", "θ_lower": "(N_param,)", "θ_upper": "(N_param,)"},
      "outputs": {"X_p^(i)": "(N_param,)"},
      "computational_note": "Clamp to feasible region"
    },
    {
      "step_order": 17,
      "step_name": "[FORWARD] Compute lens deflection angles",
      "formula": "\\boldsymbol{\\alpha}(\\boldsymbol{\\theta}) = \\boldsymbol{\\alpha}_{\\text{EPL}}(\\boldsymbol{\\theta}) + \\boldsymbol{\\alpha}_{\\text{shear}}(\\boldsymbol{\\theta})",
      "inputs": {"X_p^(i)": "(N_param,)", "pixel_coords": "(N_pix, N_pix, 2)"},
      "outputs": {"α": "(N_pix, N_pix, 2)"},
      "computational_note": "EPL + external shear deflection field"
    },
    {
      "step_order": 18,
      "step_name": "[FORWARD] Ray-trace to source plane",
      "formula": "\\boldsymbol{\\beta}_{\\text{pix}} = \\boldsymbol{\\theta}_{\\text{pix}} - \\boldsymbol{\\alpha}(\\boldsymbol{\\theta}_{\\text{pix}})",
      "inputs": {"θ_pix": "(N_pix, N_pix, 2)", "α": "(N_pix, N_pix, 2)"},
      "outputs": {"β_pix": "(N_pix, N_pix, 2)"},
      "computational_note": "Source plane coordinates for each image pixel"
    },
    {
      "step_order": 19,
      "step_name": "[FORWARD] Evaluate source surface brightness",
      "formula": "I_{\\text{src}}(\\boldsymbol{\\beta}) = A_{\\text{src}} \\exp\\left(-b_n \\left[\\left(\\frac{R(\\boldsymbol{\\beta})}{R_{\\text{sersic}}}\\right)^{1/n} - 1\\right]\\right)",
      "inputs": {"β_pix": "(N_pix, N_pix, 2)", "source_params": "dict"},
      "outputs": {"S_src_lensed": "(N_pix, N_pix)"},
      "computational_note": "Elliptical Sersic profile with R = √((x-x_c)²q + (y-y_c)²/q)"
    },
    {
      "step_order": 20,
      "step_name": "[FORWARD] Evaluate lens light surface brightness",
      "formula": "I_{\\text{lens}}(\\boldsymbol{\\theta}) = A_{\\text{ll}} \\exp\\left(-b_n \\left[\\left(\\frac{R(\\boldsymbol{\\theta})}{R_{\\text{sersic}}^{\\text{ll}}}\\right)^{1/n_{\\text{ll}}} - 1\\right]\\right)",
      "inputs": {"θ_pix": "(N_pix, N_pix, 2)", "lens_light_params": "dict"},
      "outputs": {"S_lens": "(N_pix, N_pix)"},
      "computational_note": "Circular Sersic profile for lens galaxy"
    },
    {
      "step_order": 21,
      "step_name": "[FORWARD] Render point sources",
      "formula": "S_{\\text{PS}} = \\sum_{j=1}^{4} A_j \\cdot \\delta(\\boldsymbol{\\theta} - \\boldsymbol{\\theta}_j^{\\text{img}})",
      "inputs": {"x_img": "(4,)", "y_img": "(4,)", "point_amp": "(4,)"},
      "outputs": {"S_PS": "(N_pix, N_pix)"},
      "computational_note": "Delta functions convolved later with PSF"
    },
    {
      "step_order": 22,
      "step_