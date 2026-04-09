import time

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")

import bilby

def forward_operator(
    params,
    waveform_generator,
    detector,
):
    """
    Forward operator: compute detector response for given source parameters.
    
    Maps source parameters theta -> predicted strain h(f).
    
    Args:
        params: dict of CBC source parameters
        waveform_generator: bilby WaveformGenerator
        detector: bilby Interferometer (e.g., H1)
    
    Returns:
        predicted_strain_fd: complex frequency-domain strain array
    """
    # Generate frequency-domain polarisations from source parameters
    polarisations = waveform_generator.frequency_domain_strain(params)
    
    # Apply detector antenna pattern and time/phase shifts
    predicted_strain_fd = detector.get_detector_response(polarisations, params)
    
    return predicted_strain_fd

def run_inversion(
    data_dict,
    injection_parameters,
    outdir,
):
    """
    Run the Bayesian parameter estimation (inversion) using nested sampling.
    
    Args:
        data_dict: output from load_and_preprocess_data
        injection_parameters: true injection parameters
        outdir: output directory for results
    
    Returns:
        dict containing:
            - result: bilby Result object
            - posterior: pandas DataFrame of posterior samples
            - map_params: dict of MAP parameter estimates
            - median_params: dict of median parameter estimates
            - relative_errors: dict of relative errors per parameter
            - recon_signal_fd: reconstructed frequency-domain signal (MAP)
            - runtime: elapsed time in seconds
    """
    ifos = data_dict["ifos"]
    waveform_generator = data_dict["waveform_generator"]
    h1 = data_dict["h1"]
    
    # Setup priors (3 free parameters)
    print("[3/7] Setting up priors (3 free parameters)...")
    priors = bilby.gw.prior.BBHPriorDict()
    
    # Fix everything except chirp_mass, mass_ratio, luminosity_distance
    for key in ["ra", "dec", "psi", "phase", "tilt_1", "tilt_2",
                "phi_12", "phi_jl", "a_1", "a_2", "theta_jn", "geocent_time"]:
        priors[key] = injection_parameters[key]
    
    priors["chirp_mass"] = bilby.core.prior.Uniform(
        name="chirp_mass", minimum=25.0, maximum=32.0,
        latex_label="$\\mathcal{M}$",
    )
    priors["mass_ratio"] = bilby.core.prior.Uniform(
        name="mass_ratio", minimum=0.5, maximum=1.0,
        latex_label="$q$",
    )
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        name="luminosity_distance", minimum=200.0, maximum=1000.0, unit="Mpc",
    )
    
    # Setup likelihood
    print("[4/7] Setting up likelihood...")
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
    )
    
    # Run sampler
    print("[5/7] Running nested sampling (dynesty, nlive=100, 3 free params)...")
    print("       Expected runtime: 2-8 minutes...")
    t_start = time.time()
    
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=100,
        nact=3,
        maxmcmc=2000,
        walks=5,
        dlogz=0.5,
        outdir=outdir,
        label="bilby_gw",
        injection_parameters=injection_parameters,
        save=True,
        resume=False,
        clean=True,
    )
    
    t_elapsed = time.time() - t_start
    print(f"       Sampling completed in {t_elapsed:.1f} s ({t_elapsed/60:.1f} min)")
    
    # Extract results
    print("[6/7] Extracting results and computing metrics...")
    posterior = result.posterior
    
    # Derive mass_1, mass_2 from chirp_mass, mass_ratio
    if "mass_1" not in posterior.columns:
        q = posterior["mass_ratio"]
        mc = posterior["chirp_mass"]
        eta = q / (1 + q) ** 2
        mtotal = mc / eta ** (3.0 / 5)
        posterior["mass_1"] = mtotal / (1 + q)
        posterior["mass_2"] = mtotal * q / (1 + q)
    
    estimated_params = ["chirp_mass", "mass_ratio", "luminosity_distance"]
    derived_params = ["mass_1", "mass_2"]
    all_report_params = estimated_params + derived_params
    
    map_idx = posterior["log_likelihood"].idxmax()
    median_params = {}
    map_params = {}
    for p in all_report_params:
        if p in posterior.columns:
            median_params[p] = float(np.median(posterior[p]))
            map_params[p] = float(posterior[p].iloc[map_idx])
    
    # Compute true values
    m1 = injection_parameters["mass_1"]
    m2 = injection_parameters["mass_2"]
    true_chirp_mass = (m1 * m2) ** (3.0 / 5) / (m1 + m2) ** (1.0 / 5)
    true_mass_ratio = m2 / m1
    
    true_values = dict(injection_parameters)
    true_values["chirp_mass"] = true_chirp_mass
    true_values["mass_ratio"] = true_mass_ratio
    
    print("\n=== Parameter Recovery ===")
    print(f"{'Parameter':<25} {'True':>12} {'MAP':>12} {'Median':>12} {'RelErr%':>10}")
    print("-" * 75)
    relative_errors = {}
    for p in all_report_params:
        true_val = true_values[p]
        map_val = map_params.get(p, np.nan)
        med_val = median_params.get(p, np.nan)
        rel_err = abs(med_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-10 else 0.0
        relative_errors[p] = rel_err
        print(f"{p:<25} {true_val:>12.4f} {map_val:>12.4f} {med_val:>12.4f} {rel_err:>9.2f}%")
    
    # Reconstruct waveform from MAP parameters
    recon_params = dict(injection_parameters)
    for p in estimated_params:
        recon_params[p] = map_params[p]
    q_map = map_params["mass_ratio"]
    mc_map = map_params["chirp_mass"]
    eta_map = q_map / (1 + q_map) ** 2
    mtotal_map = mc_map / eta_map ** (3.0 / 5)
    recon_params["mass_1"] = mtotal_map / (1 + q_map)
    recon_params["mass_2"] = mtotal_map * q_map / (1 + q_map)
    
    # Use forward operator to get reconstructed signal
    recon_signal_fd = forward_operator(recon_params, waveform_generator, h1)
    
    return {
        "result": result,
        "posterior": posterior,
        "map_params": map_params,
        "median_params": median_params,
        "relative_errors": relative_errors,
        "recon_signal_fd": recon_signal_fd,
        "runtime": t_elapsed,
        "true_values": true_values,
        "all_report_params": all_report_params,
        "estimated_params": estimated_params,
    }
