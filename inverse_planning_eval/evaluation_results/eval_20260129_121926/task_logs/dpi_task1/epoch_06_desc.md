## 0. ALGORITHM IDENTIFICATION
```json
{
  "algorithm_name": "Deep Probabilistic Imaging (DPI) with RealNVP",
  "algorithm_family": "Variational Inference / Normalizing Flows",
  "summary": "Amortized variational inference for radio interferometric image reconstruction using a RealNVP normalizing flow to approximate the posterior distribution over images, optimized via closure phase data fidelity, entropy regularization, and flux constraints."
}
```

## 1. OBJECTIVE FUNCTION
```json
{
  "full_expression": "\\min_{\\theta, \\alpha} \\mathbb{E}_{z \\sim \\mathcal{N}(0,I)} \\left[ \\lambda_{\\text{cp}} \\mathcal{L}_{\\text{cphase}}(x(z;\\theta,\\alpha)) + \\lambda_{\\text{ce}} \\mathcal{L}_{\\text{CE}}(x(z;\\theta,\\alpha)) + \\lambda_{\\text{flux}} \\mathcal{L}_{\\text{flux}}(x(z;\\theta,\\alpha)) - \\lambda_{\\text{det}} \\log \\left| \\det \\frac{\\partial x}{\\partial z} \\right| \\right]",
  "data_fidelity_term": "\\mathcal{L}_{\\text{cphase}} = \\frac{2}{N_{\\text{cp}}} \\sum_{i=1}^{N_{\\text{cp}}} \\frac{1 - \\cos(\\phi_i^{\\text{true}} - \\phi_i^{\\text{pred}})}{\\sigma_{\\phi,i}^2}",
  "regularization_term": {
    "cross_entropy": "\\mathcal{L}_{\\text{CE}} = \\frac{1}{N^2} \\sum_{i,j} x_{ij} \\left( \\log(x_{ij} + 10^{-12}) - \\log(p_{ij} + 10^{-12}) \\right)",
    "flux_constraint": "\\mathcal{L}_{\\text{flux}} = \\left( \\sum_{i,j} x_{ij} - F_0 \\right)^2",
    "log_determinant": "-\\lambda_{\\text{det}} \\log \\left| \\det \\frac{\\partial x}{\\partial z} \\right| = -\\lambda_{\\text{det}} \\left( \\log|\\det J_{\\text{flow}}| + \\sum_{i,j}(\\tilde{x}_{ij} - \\text{softplus}(\\tilde{x}_{ij})) + \\alpha \\cdot N^2 \\right)"
  },
  "constraints": "x_{ij} \\geq 0 \\text{ (enforced via softplus transformation)}"
}
```

## 2. VARIABLES

### 2.1 Primal Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $\theta$ | (model params) | float32 | Random (Xavier) | RealNVP flow parameters |
| $\alpha$ | $(1,)$ | float32 | $\log\left(\frac{F_0}{0.8 \cdot N^2}\right)$ | Log-scale factor for image intensity |

### 2.2 Dual Variables
None (unconstrained optimization with Adam)

### 2.3 Auxiliary Variables
| Symbol | Shape | Dtype | Initialization | Description |
|--------|-------|-------|----------------|-------------|
| $z$ | $(B, N^2)$ | float32 | $\mathcal{N}(0, I)$ each iteration | Latent space samples |
| $\tilde{x}$ | $(B, N, N)$ | float32 | Computed | Pre-activation image from flow |
| $\log\det J$ | $(B,)$ | float32 | Computed | Log-determinant from flow |
| $x$ | $(B, N, N)$ | float32 | Computed | Reconstructed images |
| $V$ | $(B, 2, M)$ | float32 | Computed | Complex visibilities (real, imag) |
| $|V|$ | $(B, M)$ | float32 | Computed | Visibility amplitudes |
| $\phi$ | $(B, N_{\text{cp}})$ | float32 | Computed | Predicted closure phases (degrees) |

### 2.4 Constants
| Symbol | Value | Description |
|--------|-------|-------------|
| $N$ | 32 | Image dimension (npix) |
| $B$ | 32 | Batch size |
| $n_{\text{flow}}$ | 16 | Number of RealNVP coupling layers |
| $\eta$ | $10^{-4}$ | Learning rate |
| $K$ | 100 | Number of epochs |
| $\epsilon$ | $10^{-16}$ | Numerical stability constant for amplitude |
| $\epsilon_{\log}$ | $10^{-12}$ | Numerical stability for logarithm |
| $\lambda_{\text{cp}}$ | $N_{\text{cp}} / N_{\text{camp}}$ | Closure phase weight |
| $\lambda_{\text{flux}}$ | 1000.0 | Flux constraint weight |
| $\lambda_{\text{ce}}$ | 1024.0 | Cross-entropy weight |
| $\lambda_{\text{det}}$ | $2.0 \cdot \text{logdet\_factor} / N_{\text{camp}}$ | Log-determinant weight |
| $\text{clip\_norm}$ | 0.1 | Gradient clipping threshold |

### 2.5 Observations
| Symbol | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| $(u, v)$ | $(M, 2)$ | float64 | Baseline coordinates |
| $\phi^{\text{true}}$ | $(N_{\text{cp}},)$ | float32 | Observed closure phases (degrees) |
| $\sigma_\phi$ | $(N_{\text{cp}},)$ | float32 | Closure phase uncertainties (degrees) |
| $F_0$ | scalar | float64 | Flux constant from APEX-ALMA baseline |
| $p$ | $(N, N)$ | float32 | Prior image (Gaussian blob) |

## 3. PREPROCESSING PIPELINE
```json
{
  "steps": [
    {
      "step_order": 1,
      "step_name": "Load observation data",
      "formula": "\\text{obs} = \\text{load\\_uvfits}(\\text{obspath})",
      "inputs": {"obspath": "string"},
      "outputs": {"obs": "ehtim Obsdata object"},
      "assumptions": "Valid UVFITS format"
    },
    {
      "step_order": 2,
      "step_name": "Extract flux constant",
      "formula": "F_0 = \\text{median}(|V_{\\text{APEX-ALMA}}|)",
      "inputs": {"obs": "Obsdata"},
      "outputs": {"F_0": "scalar"},
      "assumptions": "APEX-ALMA baseline exists in data"
    },
    {
      "step_order": 3,
      "step_name": "Create prior Gaussian image",
      "formula": "p_{ij} = F_0 \\cdot \\exp\\left( -\\frac{(x_i - x_c)^2 + (y_j - y_c)^2}{2\\sigma_{\\text{prior}}^2} \\right), \\quad \\sigma_{\\text{prior}} = \\frac{\\text{FWHM}_{\\text{prior}}}{2\\sqrt{2\\ln 2}}",
      "inputs": {"F_0": "scalar", "FWHM_prior": "50 μas", "fov": "160 μas", "npix": "32"},
      "outputs": {"p": "(N, N)"},
      "assumptions": "Centered Gaussian"
    },
    {
      "step_order": 4,
      "step_name": "Extract UV coordinates",
      "formula": "\\mathbf{uv} = \\begin{bmatrix} u_1 & v_1 \\\\ \\vdots & \\vdots \\\\ u_M & v_M \\end{bmatrix}",
      "inputs": {"obs": "Obsdata"},
      "outputs": {"uv": "(M, 2)"},
      "assumptions": "None"
    },
    {
      "step_order": 5,
      "step_name": "Compute pixel coordinates",
      "formula": "x_i = (i - N/2) \\cdot \\Delta, \\quad y_j = (j - N/2) \\cdot \\Delta, \\quad \\Delta = \\frac{\\text{fov}}{N}",
      "inputs": {"N": "32", "fov": "in radians"},
      "outputs": {"coords": "(2, N^2)"},
      "assumptions": "Zero-centered grid"
    },
    {
      "step_order": 6,
      "step_name": "Construct DFT matrix",
      "formula": "F_{mk} = \\exp\\left( -2\\pi i (u_m x_k + v_m y_k) \\right), \\quad [F]_{\\text{real}} = \\Re(F^T), \\quad [F]_{\\text{imag}} = \\Im(F^T)",
      "inputs": {"uv": "(M, 2)", "coords": "(2, N^2)"},
      "outputs": {"dft_mat": "(N^2, M, 2)"},
      "assumptions": "Phase convention: negative exponent"
    },
    {
      "step_order": 7,
      "step_name": "Add closure phase data products",
      "formula": "\\text{obs.add\\_cphase}(\\text{count}=\\text{'min-cut0bl'}, uv_{\\min}=10^8)",
      "inputs": {"obs": "Obsdata"},
      "outputs": {"cphase_data": "closure phase table"},
      "assumptions": "Minimum cut with no zero-baseline"
    },
    {
      "step_order": 8,
      "step_name": "Build closure phase index maps",
      "formula": "\\text{For each closure triangle } (t_1, t_2, t_3): \\text{find visibility indices } (i_1, i_2, i_3) \\text{ and signs } (s_1, s_2, s_3)",
      "inputs": {"obs": "Obsdata with cphase"},
      "outputs": {"cphase_ind_list": "3 tensors of (N_cp,)", "cphase_sign_list": "3 tensors of (N_cp,)"},
      "assumptions": "Consistent baseline ordering"
    },
    {
      "step_order": 9,
      "step_name": "Extract closure phase observations",
      "formula": "\\phi^{\\text{true}}_i = \\text{obs.cphase['cphase']}_i, \\quad \\sigma_{\\phi,i} = \\text{obs.cphase['sigmacp']}_i",
      "inputs": {"obs": "Obsdata"},
      "outputs": {"phi_true": "(N_cp,)", "sigma_phi": "(N_cp,)"},
      "assumptions": "Units in degrees"
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
      "variable_name": "img_generator (θ)",
      "initial_value": "RealNVP(N^2, n_{flow}, affine=True) with random initialization",
      "shape": "model parameters",
      "dtype": "float32",
      "source": "PyTorch default initialization"
    },
    {
      "step_order": 2,
      "variable_name": "α (log_scale)",
      "initial_value": "\\alpha^{(0)} = \\log\\left(\\frac{F_0}{0.8 \\cdot N^2}\\right)",
      "shape": "(1,)",
      "dtype": "float32",
      "source": "Heuristic based on flux and image size"
    },
    {
      "step_order": 3,
      "variable_name": "optimizer",
      "initial_value": "Adam(params=[θ, α], lr=η)",
      "shape": "N/A",
      "dtype": "N/A",
      "source": "Algorithm parameter"
    },
    {
      "step_order": 4,
      "variable_name": "loss_history",
      "initial_value": "[]",
      "shape": "list",
      "dtype": "float32",
      "source": "Empty list"
    }
  ]
}
```

### 4.2 ITERATION LOOP
```json
{
  "loop_structure": "for k = 0 to K-1",
  "steps": [
    {
      "step_order": 5,
      "step_name": "BEGIN_LOOP (k = 0 to K-1)",
      "formula": "k \\leftarrow 0",
      "inputs": {},
      "outputs": {"k": "iteration counter"},
      "computational_note": "Main optimization loop"
    },
    {
      "step_order": 6,
      "step_name": "Sample latent vectors",
      "formula": "z \\sim \\mathcal{N}(0, I_{N^2}), \\quad z \\in \\mathbb{R}^{B \\times N^2}",
      "inputs": {"B": "32", "N": "32"},
      "outputs": {"z": "(B, N^2)"},
      "computational_note": "torch.randn(32, npix*npix)"
    },
    {
      "step_order": 7,
      "step_name": "[FORWARD] RealNVP inverse pass",
      "formula": "(\\tilde{x}_{\\text{flat}}, \\log\\det J_{\\text{flow}}) = G^{-1}(z; \\theta)",
      "inputs": {"z": "(B, N^2)", "θ": "flow params"},
      "outputs": {"x_tilde_flat": "(B, N^2)", "logdet_flow": "(B,)"},
      "computational_note": "img_generator.reverse(z_sample) - generates images from latent"
    },
    {
      "step_order": 8,
      "step_name": "Reshape pre-activation image",
      "formula": "\\tilde{x} = \\text{reshape}(\\tilde{x}_{\\text{flat}}, (B, N, N))",
      "inputs": {"x_tilde_flat": "(B, N^2)"},
      "outputs": {"x_tilde": "(B, N, N)"},
      "computational_note": "img_samp.reshape((-1, npix, npix))"
    },
    {
      "step_order": 9,
      "step_name": "Compute scale factor",
      "formula": "s = \\exp(\\alpha)",
      "inputs": {"α": "(1,)"},
      "outputs": {"s": "(1,)"},
      "computational_note": "scale_factor = torch.exp(logscale_factor.forward())"
    },
    {
      "step_order": 10,
      "step_name": "Apply softplus and scaling",
      "formula": "x_{bij} = \\text{softplus}(\\tilde{x}_{bij}) \\cdot s = \\log(1 + \\exp(\\tilde{x}_{bij})) \\cdot s",
      "inputs": {"x_tilde": "(B, N, N)", "s": "(1,)"},
      "outputs": {"x": "(B, N, N)"},
      "computational_note": "img = torch.nn.Softplus()(img_samp) * scale_factor"
    },
    {
      "step_order": 11,
      "step_name": "Compute softplus log-determinant contribution",
      "formula": "\\Delta_{\\text{sp},b} = \\sum_{i,j} \\left( \\tilde{x}_{bij} - \\text{softplus}(\\tilde{x}_{bij}) \\right) = \\sum_{i,j} \\left( \\tilde{x}_{bij} - \\log(1 + \\exp(\\tilde{x}_{bij})) \\right)",
      "inputs": {"x_tilde": "(B, N, N)"},
      "outputs": {"det_softplus": "(B,)"},
      "computational_note": "det_softplus = torch.sum(img_samp - Softplus()(img_samp), (1,2))"
    },
    {
      "step_order": 12,
      "step_name": "Compute scale log-determinant contribution",
      "formula": "\\Delta_{\\text{scale}} = \\alpha \\cdot N^2",
      "inputs": {"α": "(1,)", "N": "32"},
      "outputs": {"det_scale": "(1,)"},
      "computational_note": "det_scale = logscale_factor_value * npix * npix"
    },
    {
      "step_order": 13,
      "step_name": "Accumulate total log-determinant",
      "formula": "\\log\\det J_b = \\log\\det J_{\\text{flow},b} + \\Delta_{\\text{sp},b} + \\Delta_{\\text{scale}}",
      "inputs": {"logdet_flow": "(B,)", "det_softplus": "(B,)", "det_scale": "(1,)"},
      "outputs": {"logdet": "(B,)"},
      "computational_note": "logdet = logdet + det_softplus + det_scale"
    },
    {
      "step_order": 14,
      "step_name": "Flatten image for DFT",
      "formula": "x_{\\text{flat}} = \\text{reshape}(x, (B, N^2))",
      "inputs": {"x": "(B, N, N)"},
      "outputs": {"x_flat": "(B, N^2)"},
      "computational_note": "x_flat = torch.reshape(x, (-1, npix*npix))"
    },
    {
      "step_order": 15,
      "step_name": "[FORWARD] Compute complex visibilities (real part)",
      "formula": "V_{\\text{real}} = x_{\\text{flat}} \\cdot F_{\\text{real}}, \\quad V_{\\text{real},bm} = \\sum_{k=1}^{N^2} x_{\\text{flat},bk} \\cdot F_{km,\\text{real}}",
      "inputs": {"x_flat": "(B, N^2)", "F_real": "(N^2, M)"},
      "outputs": {"V_real": "(B, M)"},
      "computational_note": "Fx_real = torch.matmul(x, F[:,:,0])"
    },
    {
      "step_order": 16,
      "step_name": "[FORWARD] Compute complex visibilities (imag part)",
      "formula": "V_{\\text{imag}} = x_{\\text{flat}} \\cdot F_{\\text{imag}}, \\quad V_{\\text{imag},bm} = \\sum_{k=1}^{N^2} x_{\\text{flat},bk} \\cdot F_{km,\\text{imag}}",
      "inputs": {"x_flat": "(B, N^2)", "F_imag": "(N^2, M)"},
      "outputs": {"V_imag": "(B, M)"},
      "computational_note": "Fx_imag = torch.matmul(x, F[:,:,1])"
    },
    {
      "step_order": 17,
      "step_name": "Stack complex visibilities",
      "formula": "V = \\text{stack}(V_{\\text{real}}, V_{\\text{imag}}, \\text{dim}=1)",
      "inputs": {"V_real": "(B, M)", "V_imag": "(B, M)"},
      "outputs": {"V": "(B, 2, M)"},
      "computational_note": "vis_torch = torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)"
    },
    {
      "step_order": 18,
      "step_name": "Compute visibility amplitudes",
      "formula": "|V|_{bm} = \\sqrt{V_{\\text{real},bm}^2 + V_{\\text{imag},bm}^2 + \\epsilon}",
      "inputs": {"V": "(B, 2, M)", "ε": "1e-16"},
      "outputs": {"vis_amp": "(B, M)"},
      "computational_note": "vis_amp = torch.sqrt(vis_torch[:,0,:]**2 + vis_torch[:,1,:]**2 + eps)"
    },
    {
      "step_order": 19,
      "step_name": "Index visibilities for closure triangle (leg 1)",
      "formula": "V^{(1)} = V[:, :, \\text{ind}_1], \\quad V^{(1)} \\in \\mathbb{R}^{B \\times 2 \\times N_{\\text{cp}}}",
      "inputs": {"V": "(B, 2, M)", "ind_1": "(N_cp,)"},
      "outputs": {"V1": "(B, 2, N_cp)"},
      "computational_note": "vis1_torch = torch.index_select(vis_torch, -1, cphase_ind1)"
    },
    {
      "step_order": 20,
      "step_name": "Index visibilities for closure triangle (leg 2)",
      "formula": "V^{(2)} = V[:, :, \\text{ind}_2]",
      "inputs": {"V": "(B, 2, M)", "ind_2": "(N_cp,)"},
      "outputs": {"V2": "(B, 2, N_cp)"},
      "computational_note": "vis2_torch = torch.index_select(vis_torch, -1, cphase_ind2)"
    },
    {
      "step_order": 21,
      "step_name": "Index visibilities for closure triangle (leg 3)",
      "formula": "V^{(3)} = V[:, :, \\text{ind}_3]",
      "inputs": {"V": "(B, 2, M)", "ind_3": "(N_cp,)"},
      "outputs": {"V3": "(B, 2, N_cp)"},
      "computational_note": "vis3_torch = torch.index_select(vis_torch, -1, cphase_ind3)"
    },
    {
      "step_order": 22,
      "step_name": "Compute phase angle (leg 1)",
      "formula": "\\psi^{(1)}_{bi} = \\text{atan2}(V^{(1)}_{b,1,i}, V^{(1)}_{b,0,i})",
      "inputs": {"V1": "(B, 2, N_cp)"},
      "outputs": {"ang1": "(B, N_cp)"},
      "computational_note": "ang1 = torch.atan2(vis1_torch[:,1,:], vis1_torch[:,0,:])"
    },
    {
      "step_order": 23,
      "step_name": "Compute phase angle (leg 2)",
      "formula": "\\psi^{(2)}_{bi} = \\text{atan2}(V^{(2)}_{b,1,i}, V^{(2)}_{b,0,i})",
      "inputs": {"V2": "(B, 2, N_cp)"},
      "outputs": {"ang2": "(B, N_cp)"},
      "computational_note": "ang2 = torch.atan2(vis2_torch[:,1,:], vis2_torch[:,0,:])"
    },
    {
      "step_order": 24,
      "step_name": "Compute phase angle (leg 3)",
      "formula": "\\psi^{(3)}_{bi} = \\text{atan2}(V^{(3)}_{b,1,i}, V^{(3)}_{b,0,i})",
      "inputs": {"V3": "(B, 2, N_cp)"},
      "outputs": {"ang3": "(B, N_cp)"},
      "computational_note": "ang3 = torch.atan2(vis3_torch[:,1,:], vis3_torch[:,0,:])"
    },
    {
      "step_order": 25,
      "step_name": "Compute closure phases",
      "formula": "\\phi_{bi} = (s_1^{(i)} \\psi^{(1)}_{bi} + s_2^{(i)} \\psi^{(2)}_{bi} + s_3^{(i)} \\psi^{(3)}_{bi}) \\cdot \\frac{180}{\\pi}",
      "inputs": {"ang1": "(B, N_cp)", "ang2": "(B, N_cp)", "ang3": "(B, N_cp)", "s1": "(N_cp,)", "s2": "(N_cp,)", "s3": "(N_cp,)"},
      "outputs": {"phi_pred": "(B, N_cp)"},
      "computational_note": "cphase = (sign1*ang1 + sign2*ang2 + sign3*ang3) * 180/pi"
    },
    {
      "step_order": 26,
      "step_name": "Compute cross-entropy loss",
      "formula": "\\mathcal{L}_{\\text{CE},b} = \\frac{1}{N^2} \\sum_{i,j} x_{bij} \\cdot \\left( \\log(x_{bij} + 10^{-12}) - \\log(p_{ij} + 10^{-12}) \\right)",
      "inputs": {"x": "(B, N, N)", "p": "(N, N)"},
      "outputs": {"loss_ce": "(B,)"},
      "computational_note": "loss_cross_entropy = torch.mean(img * (log(img+1e-12) - log(prior+1e-12)), (-1,-2))"
    },
    {
      "step_order": 27,
      "step_name": "Compute image flux",
      "formula": "F_b = \\sum_{i,j} x_{bij}",
      "inputs": {"x": "(B, N, N)"},
      "outputs": {"img_flux": "(B,)"},
      "computational_note": "img_flux = torch.sum(img, (-1,-2))"
    },
    {
      "step_order": 28,
      "step_name": "Compute flux loss",
      "formula": "\\mathcal{L}_{\\text{flux},b} = (F_b - F_0)^2",
      "inputs": {"img_flux": "(B,)", "F_0": "scalar"},
      "outputs": {"loss_flux": "(B,)"},
      "computational_note": "loss_flux = (img_flux - flux_const)**2"
    },
    {
      "step_order": 29,
      "step_name": "Convert angles to radians",
      "formula": "\\phi^{\\text{true}}_{\\text{rad}} = \\phi^{\\text{true}} \\cdot \\frac{\\pi}{180}, \\quad \\phi^{\\text{pred}}_{\\text{rad}} = \\phi_{\\text{pred}} \\cdot \\frac{\\pi}{180}",
      "inputs": {"phi_true": "(N_cp,)", "phi_pred": "(B, N_cp)"},
      "outputs": {"angle_true": "(N_cp,)", "angle_pred": "(B, N_cp)"},
      "computational_note": "angle_true = cphase_true * pi/180; angle_pred = cphase * pi/180"
    },
    {
      "step_order": 30,
      "step_name": "Compute closure phase loss",
      "formula": "\\mathcal{L}_{\\text{cp},b} = \\frac{2}{N_{\\text{cp}}} \\sum_{i=1}^{N_{\\text{cp}}} \\frac{1 - \\cos(\\phi^{\\text{true}}_{\\text{rad},i} - \\phi^{\\text{pred}}_{\\text{rad},bi})}{(\\sigma_{\\phi,i} \\cdot \\pi/180)^2}",
      "inputs": {"angle_true": "(N_cp,)", "angle_pred": "(B, N_cp)", "sigma_phi": "(N_cp,)"},
      "outputs": {"loss_cphase": "(B,)"},
      "computational_note": "loss_cphase = 2.0 * torch.mean((1 - cos(angle_true - angle_pred)) / (sigma_cphase * pi/180)**2, 1)"
    },
    {
      "step_order": 31,
      "step_name": "Compute data fidelity term",
      "formula": "\\mathcal{L}_{\\text{data},b} = \\lambda_{\\text{cp}} \\cdot \\mathcal{L}_{\\text{cp},b}",
      "inputs": {"loss_cphase": "(B,)", "λ_cp": "N_cp/N_camp"},
      "outputs": {"loss_data": "(B,)"},
      "computational_note": "loss_data = cphase_weight * loss_cphase"
    },
    {
      "step_order": 32,
      "step_name": "Compute prior/regularization term",
      "formula": "\\mathcal{L}_{\\text{prior},b} = \\lambda_{\\text{ce}} \\cdot \\mathcal{L}_{\\text{CE},b} + \\lambda_{\\text{flux}} \\cdot \\mathcal{L}_{\\text{flux},b}",
      "inputs": {"loss_ce": "(B,)", "loss_flux": "(B,)", "λ_ce": "1024.0", "λ_flux": "1000.0"},
      "outputs": {"loss_prior": "(B,)"},
      "computational_note": "loss_prior = imgcrossentropy_weight * loss_cross_entropy + imgflux_weight * loss_flux"
    },