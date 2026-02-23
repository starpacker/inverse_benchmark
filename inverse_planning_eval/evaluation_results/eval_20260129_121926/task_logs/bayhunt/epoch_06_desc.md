## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Transdimensional Bayesian MCMC Inversion",
  "algorithm_family": "Markov Chain Monte Carlo / Bayesian Inference",
  "summary": "A transdimensional Bayesian inversion algorithm using parallel-tempered MCMC to jointly invert surface wave dispersion and receiver function data for subsurface shear velocity structure with adaptive layer parameterization."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\arg\\max_{\\mathbf{m}} p(\\mathbf{m}|\\mathbf{d}) \\propto p(\\mathbf{d}|\\mathbf{m}) p(\\mathbf{m})",
  "data_fidelity_term": "p(\\mathbf{d}|\\mathbf{m}) = p(\\mathbf{d}_{\\text{SWD}}|\\mathbf{m}) \\cdot p(\\mathbf{d}_{\\text{RF}}|\\mathbf{m})",
  "swd_likelihood": "\\log p(\\mathbf{d}_{\\text{SWD}}|\\mathbf{m}) = -\\frac{1}{2}(\\mathbf{d}_{\\text{SWD}} - \\mathbf{G}_{\\text{SWD}}(\\mathbf{m}))^T \\mathbf{C}_{\\text{SWD}}^{-1} (\\mathbf{d}_{\\text{SWD}} - \\mathbf{G}_{\\text{SWD}}(\\mathbf{m})) - \\frac{1}{2}\\log|\\mathbf{C}_{\\text{SWD}}|",
  "rf_likelihood": "\\log p(\\mathbf{d}_{\\text{RF}}|\\mathbf{m}) = -\\frac{1}{2}(\\mathbf{d}_{\\text{RF}} - \\mathbf{G}_{\\text{RF}}(\\mathbf{m}))^T \\mathbf{C}_{\\text{RF}}^{-1} (\\mathbf{d}_{\\text{RF}} - \\mathbf{G}_{\\text{RF}}(\\mathbf{m})) - \\frac{1}{2}\\log|\\mathbf{C}_{\\text{RF}}|",
  "regularization_term": "p(\\mathbf{m}) = p(k) \\cdot p(v_{P}/v_{S}) \\cdot \\prod_{i=1}^{k} p(z_i) \\cdot \\prod_{i=1}^{k} p(v_{S,i})",
  "constraints": "\\mathbb{I}_{[1,20]}(k) \\cdot \\mathbb{I}_{[1.4,2.1]}(v_P/v_S) \\cdot \\mathbb{I}_{[2,5]}(v_S) \\cdot \\mathbb{I}_{[0,60]}(z) \\cdot \\mathbb{I}_{[0.1,\\infty)}(h_{\\min})"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $k$ | scalar | int | Sampled from $\mathcal{U}(1, 20)$ | Number of layers (transdimensional) |
| $\mathbf{h}$ | $(k,)$ | float64 | Random from prior | Layer thicknesses in km |
| $\mathbf{v}_S$ | $(k,)$ | float64 | Random from prior | Shear wave velocities in km/s per layer |
| $v_P/v_S$ | scalar | float64 | Sampled from $\mathcal{U}(1.4, 2.1)$ | Vp/Vs ratio |
| $\sigma_{\text{SWD}}$ | scalar | float64 | Sampled from $\mathcal{U}(10^{-5}, 0.05)$ | SWD noise standard deviation |
| $\sigma_{\text{RF}}$ | scalar | float64 | Sampled from $\mathcal{U}(10^{-5}, 0.05)$ | RF noise standard deviation |
| $\rho_{\text{RF}}$ | scalar | float64 | Fixed at 0.98 | RF noise correlation coefficient |
| $\rho_{\text{SWD}}$ | scalar | float64 | Fixed at 0.0 | SWD noise correlation coefficient |

### 2.2 Dual Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| N/A | — | — | — | No explicit dual variables in MCMC formulation |

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\mathbf{m}^{(t)}$ | variable | dict | — | Current model state at iteration $t$ |
| $\mathbf{m}^{*}$ | variable | dict | — | Proposed model state |
| $\mathcal{L}^{(t)}$ | scalar | float64 | — | Current log-likelihood |
| $\mathcal{L}^{*}$ | scalar | float64 | — | Proposed log-likelihood |
| $\alpha$ | scalar | float64 | — | Acceptance probability |
| $u$ | scalar | float64 | — | Uniform random draw for acceptance |
| $\mathbf{d}_{\text{SWD}}^{\text{pred}}$ | $(N_{\text{SWD}},)$ | float64 | — | Predicted SWD from forward model |
| $\mathbf{d}_{\text{RF}}^{\text{pred}}$ | $(N_{\text{RF}},)$ | float64 | — | Predicted RF from forward model |
| $\mathbf{C}_{\text{SWD}}$ | $(N_{\text{SWD}}, N_{\text{SWD}})$ | float64 | — | SWD data covariance matrix |
| $\mathbf{C}_{\text{RF}}$ | $(N_{\text{RF}}, N_{\text{RF}})$ | float64 | — | RF data covariance matrix |
| acceptance_rate | scalar | float64 | — | Running acceptance rate per chain |

### 2.4 Constants
| Symbol | Shape | Dtype | Value | Description |
|--------|-------|-------|-------|-------------|
| $N_{\text{chains}}$ | scalar | int | 5 | Number of parallel MCMC chains |
| $N_{\text{burnin}}$ | scalar | int | $2048 \times 32 = 65536$ | Burn-in iterations per chain |
| $N_{\text{main}}$ | scalar | int | $2048 \times 16 = 32768$ | Main sampling iterations per chain |
| $\boldsymbol{\sigma}_{\text{prop}}$ | $(5,)$ | float64 | $(0.025, 0.025, 0.015, 0.005, 0.005)$ | Proposal distribution std for $(v_S, z, v_P/v_S, \sigma_{\text{SWD}}, \sigma_{\text{RF}})$ |
| $[\text{acc}_{\min}, \text{acc}_{\max}]$ | $(2,)$ | int | $(40, 45)$ | Target acceptance rate range (%) |
| $h_{\min}$ | scalar | float64 | 0.1 | Minimum layer thickness (km) |
| rcond | scalar | float64 | $10^{-5}$ | Regularization for covariance inversion |
| $z_{\text{Moho}}$ | $(2,)$ | float64 | $(38, 4)$ | Moho depth prior: mean, std (km) |
| gauss | scalar | float64 | 1.0 | Gaussian filter width for RF |
| water | scalar | float64 | 0.01 | Water level for RF deconvolution |
| $p$ | scalar | float64 | 6.4 | Ray parameter (s/deg) |
| $N_{\text{threads}}$ | scalar | int | 6 | Number of parallel threads |
| $N_{\text{maxmodels}}$ | scalar | int | 50000 | Maximum models to save |
| dev | scalar | float64 | 0.05 | Outlier chain deviation threshold |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $\mathbf{x}_{\text{SWD}}$ | $(21,)$ | float64 | Period values for SWD (s), range $[1, 41]$ |
| $\mathbf{d}_{\text{SWD}}$ | $(21,)$ | float64 | Observed Rayleigh phase velocities (km/s) |
| $\mathbf{x}_{\text{RF}}$ | $(N_{\text{RF}},)$ | float64 | Time values for receiver function (s) |
| $\mathbf{d}_{\text{RF}}$ | $(N_{\text{RF}},)$ | float64 | Observed P-receiver function amplitudes |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": 1,
      "step_name": "Generate synthetic true model",
      "formula": "\\mathbf{h}_{\\text{true}} = [5, 23, 8, 0]^T, \\quad \\mathbf{v}_{S,\\text{true}} = [2.7, 3.6, 3.8, 4.4]^T, \\quad (v_P/v_S)_{\\text{true}} = 1.73",
      "inputs": [],
      "outputs": [{"name": "h_true", "shape": "(4,)"}, {"name": "vs_true", "shape": "(4,)"}, {"name": "vpvs_true", "shape": "scalar"}],
      "assumptions": "Four-layer velocity model with half-space at bottom"
    },
    {
      "step_order": 2,
      "step_name": "Generate noise-free SWD data",
      "formula": "\\mathbf{x}_{\\text{SWD}} = \\text{linspace}(1, 41, 21), \\quad \\mathbf{d}_{\\text{SWD}}^{\\text{clean}} = \\mathbf{G}_{\\text{SWD}}(\\mathbf{h}_{\\text{true}}, \\mathbf{v}_{S,\\text{true}}, (v_P/v_S)_{\\text{true}})",
      "inputs": [{"name": "h_true", "shape": "(4,)"}, {"name": "vs_true", "shape": "(4,)"}, {"name": "vpvs_true", "shape": "scalar"}],
      "outputs": [{"name": "x_swd", "shape": "(21,)"}, {"name": "d_swd_clean", "shape": "(21,)"}],
      "assumptions": "Forward model computes Rayleigh wave phase velocity dispersion"
    },
    {
      "step_order": 3,
      "step_name": "Generate noise-free RF data",
      "formula": "\\mathbf{d}_{\\text{RF}}^{\\text{clean}} = \\mathbf{G}_{\\text{RF}}(\\mathbf{h}_{\\text{true}}, \\mathbf{v}_{S,\\text{true}}, (v_P/v_S)_{\\text{true}}, p=6.4)",
      "inputs": [{"name": "h_true", "shape": "(4,)"}, {"name": "vs_true", "shape": "(4,)"}, {"name": "vpvs_true", "shape": "scalar"}],
      "outputs": [{"name": "x_rf", "shape": "(N_RF,)"}, {"name": "d_rf_clean", "shape": "(N_RF,)"}],
      "assumptions": "Forward model computes P-wave receiver function"
    },
    {
      "step_order": 4,
      "step_name": "Compute SWD exponential noise",
      "formula": "\\boldsymbol{\\epsilon}_{\\text{SWD}} = \\text{compute\\_expnoise}(\\mathbf{d}_{\\text{SWD}}^{\\text{clean}}, \\rho=0.0, \\sigma=0.012)",
      "inputs": [{"name": "d_swd_clean", "shape": "(21,)"}],
      "outputs": [{"name": "epsilon_swd", "shape": "(21,)"}],
      "assumptions": "Uncorrelated exponential noise model for SWD"
    },
    {
      "step_order": 5,
      "step_name": "Compute RF Gaussian correlated noise",
      "formula": "\\boldsymbol{\\epsilon}_{\\text{RF}} = \\text{compute\\_gaussnoise}(\\mathbf{d}_{\\text{RF}}^{\\text{clean}}, \\rho=0.98, \\sigma=0.005)",
      "inputs": [{"name": "d_rf_clean", "shape": "(N_RF,)"}],
      "outputs": [{"name": "epsilon_rf", "shape": "(N_RF,)"}],
      "assumptions": "Correlated Gaussian noise with correlation 0.98"
    },
    {
      "step_order": 6,
      "step_name": "Create noisy observations",
      "formula": "\\mathbf{d}_{\\text{SWD}} = \\mathbf{d}_{\\text{SWD}}^{\\text{clean}} + \\boldsymbol{\\epsilon}_{\\text{SWD}}, \\quad \\mathbf{d}_{\\text{RF}} = \\mathbf{d}_{\\text{RF}}^{\\text{clean}} + \\boldsymbol{\\epsilon}_{\\text{RF}}",
      "inputs": [{"name": "d_swd_clean", "shape": "(21,)"}, {"name": "d_rf_clean", "shape": "(N_RF,)"}, {"name": "epsilon_swd", "shape": "(21,)"}, {"name": "epsilon_rf", "shape": "(N_RF,)"}],
      "outputs": [{"name": "d_swd", "shape": "(21,)"}, {"name": "d_rf", "shape": "(N_RF,)"}],
      "assumptions": "Additive noise model"
    },
    {
      "step_order": 7,
      "step_name": "Compute true noise statistics",
      "formula": "\\boldsymbol{\\theta}_{\\text{noise}} = [\\rho_{\\text{SWD}}, \\text{std}(\\boldsymbol{\\epsilon}_{\\text{SWD}}), \\rho_{\\text{RF}}, \\text{std}(\\boldsymbol{\\epsilon}_{\\text{RF}})]^T",
      "inputs": [{"name": "epsilon_swd", "shape": "(21,)"}, {"name": "epsilon_rf", "shape": "(N_RF,)"}],
      "outputs": [{"name": "truenoise", "shape": "(4,)"}],
      "assumptions": "Reference noise for evaluation"
    },
    {
      "step_order": 8,
      "step_name": "Build velocity-depth profile for reference",
      "formula": "\\mathbf{z}_{\\text{ref}} = \\text{cumsum}(\\mathbf{h}_{\\text{true}}), \\quad \\mathbf{z}_{\\text{plot}} = [0, z_1, z_1, z_2, z_2, \\ldots, 150]^T, \\quad \\mathbf{v}_{S,\\text{plot}} = [v_1, v_1, v_2, v_2, \\ldots]^T",
      "inputs": [{"name": "h_true", "shape": "(4,)"}, {"name": "vs_true", "shape": "(4,)"}],
      "outputs": [{"name": "pdep", "shape": "(2k,)"}, {"name": "pvs", "shape": "(2k,)"}],
      "assumptions": "Staircase representation for plotting"
    },
    {
      "step_order": 9,
      "step_name": "Compute expected log-likelihood",
      "formula": "\\mathcal{L}_{\\text{exp}} = \\text{compute\\_explike}([\\mathbf{d}_{\\text{SWD}}, \\mathbf{d}_{\\text{RF}}], [\\mathbf{d}_{\\text{SWD}}^{\\text{clean}}, \\mathbf{d}_{\\text{RF}}^{\\text{clean}}], \\boldsymbol{\\theta}_{\\text{noise}}, \\text{rcond}=10^{-5})",
      "inputs": [{"name": "d_swd", "shape": "(21,)"}, {"name": "d_rf", "shape": "(N_RF,)"}, {"name": "d_swd_clean", "shape": "(21,)"}, {"name": "d_rf_clean", "shape": "(N_RF,)"}, {"name": "truenoise", "shape": "(4,)"}],
      "outputs": [{"name": "explike", "shape": "scalar"}],
      "assumptions": "Gaussian likelihood for RF, non-Gaussian for SWD"
    },
    {
      "step_order": 10,
      "step_name": "Configure joint target",
      "formula": "\\text{target}_1 = \\text{RayleighDispersionPhase}(\\mathbf{x}_{\\text{SWD}}, \\mathbf{d}_{\\text{SWD}}, \\boldsymbol{\\epsilon}_{\\text{SWD}}), \\quad \\text{target}_2 = \\text{PReceiverFunction}(\\mathbf{x}_{\\text{RF}}, \\mathbf{d}_{\\text{RF}})",
      "inputs": [{"name": "x_swd", "shape": "(21,)"}, {"name": "d_swd", "shape": "(21,)"}, {"name": "epsilon_swd", "shape": "(21,)"}, {"name": "x_rf", "shape": "(N_RF,)"}, {"name": "d_rf", "shape": "(N_RF,)"}],
      "outputs": [{"name": "targets", "shape": "JointTarget object"}],
      "assumptions": "Target2 uses gauss=1.0, water=0.01, p=6.4"
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
      "variable_name": "random_seed",
      "initial_value": "\\text{None (system entropy)}",
      "shape": "scalar",
      "dtype": "int or None",
      "source": "initparams['random_seed']"
    },
    {
      "step_order": 2,
      "variable_name": "chains",
      "initial_value": "\\text{List of } N_{\\text{chains}}=5 \\text{ chain objects}",
      "shape": "(5,)",
      "dtype": "list",
      "source": "MCMC_Optimizer constructor"
    },
    {
      "step_order": 3,
      "variable_name": "k^{(c,0)}",
      "initial_value": "k^{(c,0)} \\sim \\mathcal{U}\\{1, 20\\} \\quad \\forall c \\in [1, N_{\\text{chains}}]",
      "shape": "scalar per chain",
      "dtype": "int",
      "source": "priors['layers']"
    },
    {
      "step_order": 4,
      "variable_name": "\\mathbf{v}_S^{(c,0)}",
      "initial_value": "v_{S,i}^{(c,0)} \\sim \\mathcal{U}(2, 5) \\quad \\forall i \\in [1, k^{(c,0)}]",
      "shape": "(k^{(c,0)},) per chain",
      "dtype": "float64",
      "source": "priors['vs']"
    },
    {
      "step_order": 5,
      "variable_name": "\\mathbf{z}^{(c,0)}",
      "initial_value": "z_i^{(c,0)} \\sim \\mathcal{U}(0, 60) \\quad \\text{sorted ascending}, \\quad h_i = z_i - z_{i-1} \\geq h_{\\min}",
      "shape": "(k^{(c,0)},) per chain",
      "dtype": "float64",
      "source": "priors['z'], initparams['thickmin']"
    },
    {
      "step_order": 6,
      "variable_name": "(v_P/v_S)^{(c,0)}",
      "initial_value": "(v_P/v_S)^{(c,0)} \\sim \\mathcal{U}(1.4, 2.1)",
      "shape": "scalar per chain",
      "dtype": "float64",
      "source": "priors['vpvs']"
    },
    {
      "step_order": 7,
      "variable_name": "\\sigma_{\\text{SWD}}^{(c,0)}",
      "initial_value": "\\sigma_{\\text{SWD}}^{(c,0)} \\sim \\mathcal{U}(10^{-5}, 0.05)",
      "shape": "scalar per chain",
      "dtype": "float64",
      "source": "priors['swdnoise_sigma']"
    },
    {
      "step_order": 8,
      "variable_name": "\\sigma_{\\text{RF}}^{(c,0)}",
      "initial_value": "\\sigma_{\\text{RF}}^{(c,0)} \\sim \\mathcal{U}(10^{-5}, 0.05)",
      "shape": "scalar per chain",
      "dtype": "float64",
      "source": "priors['rfnoise_sigma']"
    },
    {
      "step_order": 9,
      "variable_name": "\\mathcal{L}^{(c,0)}",
      "initial_value": "\\mathcal{L}^{(c,0)} = \\log p(\\mathbf{d}|\\mathbf{m}^{(c,0)})",
      "shape": "scalar per chain",
      "dtype": "float64",
      "source": "Computed from initial model"
    },
    {
      "step_order": 10,
      "variable_name": "acceptance\\_count^{(c)}",
      "initial_value": "0",
      "shape": "scalar per chain",
      "dtype": "int",
      "source": "Counter initialization"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "loop_structure": "Two-phase: burn-in then main sampling",
  "steps": [
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "phase",
      "range": "[burnin, main]"
    },
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "t",
      "range": "[1, N_{phase}] where N_{burnin}=65536, N_{main}=32768"
    },
    {
      "marker": "BEGIN_LOOP",
      "loop_variable": "c",
      "range": "[1, N_{chains}=5] (parallel execution)"
    },
    {
      "step_order": 11,
      "step_name": "Select proposal type",
      "formula": "\\text{type} \\sim \\text{Categorical}(\\{\\text{birth}, \\text{death}, \\text{move}, \\text{update}\\})",
      "inputs": [{"name": "k^{(c,t-1)}", "shape": "scalar"}],
      "outputs": [{"name": "proposal_type", "shape": "categorical"}],
      "computational_note": "Birth/death modify dimension k; move changes interface depth; update changes velocity"
    },
    {
      "step_order": 12,
      "step_name": "Generate proposal - Birth move",
      "formula": "\\text{If birth}: k^* = k^{(c,t-1)} + 1, \\quad z_{\\text{new}} \\sim \\mathcal{U}(0, 60), \\quad v_{S,\\text{new}} \\sim \\mathcal{U}(2, 5)",
      "inputs": [{"name": "k^{(c,t-1)}", "shape": "scalar"}, {"name": "\\mathbf{z}^{(c,t-1)}", "shape": "(k,)"}, {"name": "\\mathbf{v}_S^{(c,t-1)}", "shape": "(k,)"}],
      "outputs": [{"name": "k^*", "shape": "scalar"}, {"name": "\\mathbf{z}^*", "shape": "(k+1,)"}, {"name": "\\mathbf{v}_S^*", "shape": "(k+1,)"}],
      "computational_note": "Insert new layer and sort by depth"
    },
    {
      "step_order": 13,
      "step_name": "Generate proposal - Death move",
      "formula": "\\text{If death}: k^* = k^{(c,t-1)} - 1, \\quad i_{\\text{del}} \\sim \\mathcal{U}\\{1, k^{(c,t-1)}\\}, \\quad \\text{remove layer } i_{\\text{del}}",
      "inputs": [{"name": "k^{(c,t-1)}", "shape": "scalar"}, {"name": "\\mathbf{z}^{(c,t-1)}", "shape": "(k,)"}, {"name": "\\mathbf{v}_S^{(c,t-1)}", "shape": "(k,)"}],
      "outputs": [{"name": "k^*", "shape": "scalar"}, {"name": "\\mathbf{z}^*", "shape": "(k-1,)"}, {"name": "\\mathbf{v}_S^*", "shape": "(k-1,)"}],
      "computational_note": "Only valid if k > 1"
    },
    {
      "step_order": 14,
      "step_name": "Generate proposal - Move interface",
      "formula": "\\text{If move}: i \\sim \\mathcal{U}\\{1, k\\}, \\quad z_i^* = z_i^{(c,t-1)} + \\mathcal{N}(0, \\sigma_{\\text{prop},2}^2)",
      "inputs": [{"name": "\\mathbf{z}^{(c,t-1)}", "shape": "(k,)"}, {"name": "\\sigma_{prop,2}", "shape": "scalar", "value": "0.015"}],
      "outputs": [{"name": "\\mathbf{z}^*", "shape": "(k,)"}],
      "computational_note": "Ensure h_min constraint satisfied after perturbation"
    },
    {
      "step_order": 15,
      "step_name": "Generate proposal - Update velocity",
      "formula": "\\text{If update}: i \\sim \\mathcal{U}\\{1, k\\}, \\quad v_{S,i}^* = v_{S,i}^{(c,t-1)} + \\mathcal{N}(0, \\sigma_{\\text{prop},1}^2)",
      "inputs": [{"name": "\\mathbf{v}_S^{(c,t-1)}", "shape": "(k,)"}, {"name": "\\sigma_{prop,1}", "shape": "scalar", "value": "0.025"}],
      "outputs": [{"name": "\\mathbf{v}_S^*", "shape": "(k,)"}],
      "computational_note": "Clamp to [2, 5] km/s"
    },
    {
      "step_order": 16,
      "step_name": "Generate proposal - Vp/Vs ratio",
      "formula": "(v_P/v_S)^* = (v_P/v_S)^{(c,t-1)} + \\mathcal{N}(0, \\sigma_{\\text{prop},3}^2)",
      "inputs": [{"name": "(v_P/v_S)^{(c,t-1)}", "shape": "scalar"}, {"name": "\\sigma_{prop,3}", "shape": "scalar", "value": "0.015"}],
      "outputs": [{"name": "(v_P/v_S)^*", "shape": "scalar"}],
      "computational_note": "Clamp to [1.4, 2.1]"
    },
    {
      "step_order": 17,
      "step_name": "Generate proposal - Noise parameters",
      "formula": "\\sigma_{\\text{SWD}}^* = \\sigma_{\\text{SWD}}^{(c,t-1)} + \\mathcal{N}(0, \\sigma_{\\text{prop},4}^2), \\quad \\sigma_{\\text{RF}}^* = \\sigma_{\\text{RF}}^{(c,t-1)} + \\mathcal{N}(0, \\sigma_{\\text{prop},5}^2)",
      "inputs": [{"name": "\\sigma_{\\text{SWD}}^{(c,t-1)}", "shape": "scalar"}, {"name": "\\sigma_{\\text{RF}}^{(c,t-1)}", "shape": "scalar"}, {"name": "\\sigma_{prop,4}", "shape": "scalar", "value": "0.005"}, {"name": "\\sigma_{prop,5}", "shape": "scalar", "value": "0.005"}],
      "outputs": [{"name": "\\sigma_{\\text{SWD}}^*", "shape": "scalar"}, {"name": "\\sigma_{\\text{RF}}^*", "shape": "scalar"}],
      "computational_note": "Clamp to [1e-5, 0.05]"
    },
    {
      "step_order": 18,
      "step_name": "