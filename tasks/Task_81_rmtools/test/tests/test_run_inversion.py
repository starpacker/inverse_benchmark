import sys
import os
import dill
import numpy as np
import traceback
from scipy.ndimage import gaussian_filter1d

# Import target function
from agent_run_inversion import run_inversion

# Inject the referee (evaluation function) from Reference B
def generate_faraday_depth_spectrum(phi_arr, components, d_phi):
    """
    Generate ground truth Faraday depth spectrum F(φ) on a given φ grid.
    Each component is a delta function convolved with a narrow Gaussian
    for numerical representation.
    """
    F_gt = np.zeros(len(phi_arr), dtype=complex)
    
    for comp in components:
        phi0 = comp['phi']
        amp = comp['amplitude']
        chi0 = comp['chi0']
        
        # Represent as narrow Gaussian (delta-like)
        sigma_phi = d_phi * 0.5
        gaussian = amp * np.exp(-(phi_arr - phi0)**2 / (2 * sigma_phi**2))
        phase = np.exp(2j * chi0)
        F_gt += gaussian * phase
    
    return F_gt

def evaluate_results(ground_truth, reconstruction, components, d_phi):
    """
    Compute Faraday depth spectrum recovery metrics.
    
    Following standard radio interferometry practice, the GT spectrum is
    convolved with a Gaussian 'CLEAN beam' whose FWHM matches the RMSF
    before comparison. This is analogous to comparing a CLEAN image with
    the model convolved with the restoring beam.
    
    Args:
        ground_truth: dict with Q_clean, U_clean, components
        reconstruction: dict with phi_arr, cleanFDF, dirtyFDF, fwhmRMSF
        components: list of dicts with 'phi', 'amplitude', 'chi0'
        d_phi: Faraday depth resolution
    
    Returns:
        dict with psnr, rmse, cc, avg_peak_position_error, peak_position_errors,
             avg_amplitude_recovery, fwhm_rmsf, gt_peaks, clean_peaks
    """
    phi_arr = reconstruction['phi_arr']
    cleanFDF = reconstruction['cleanFDF']
    fwhmRMSF = reconstruction['fwhmRMSF']
    
    # Generate GT Faraday depth spectrum on same grid
    gt_FDF = generate_faraday_depth_spectrum(phi_arr, components, d_phi)
    
    # Convolve GT amplitude with CLEAN beam (Gaussian with FWHM = RMSF FWHM)
    # sigma_beam in pixel units: FWHM / (2*sqrt(2*ln2)) / D_PHI
    sigma_beam_pix = fwhmRMSF / (2.0 * np.sqrt(2.0 * np.log(2.0))) / d_phi
    gt_amp_raw = np.abs(gt_FDF)
    gt_amp = gaussian_filter1d(gt_amp_raw, sigma_beam_pix)
    
    # Use absolute values (amplitude spectrum)
    clean_amp = np.abs(cleanFDF)
    dirty_amp = np.abs(reconstruction['dirtyFDF'])
    
    # Normalize for comparison
    gt_amp_norm = gt_amp / (gt_amp.max() + 1e-10)
    clean_amp_norm = clean_amp / (clean_amp.max() + 1e-10)
    
    # RMSE
    rmse = np.sqrt(np.mean((gt_amp_norm - clean_amp_norm)**2))
    
    # Correlation coefficient
    cc = np.corrcoef(gt_amp_norm, clean_amp_norm)[0, 1]
    
    # Peak detection accuracy
    gt_peaks = []
    clean_peaks = []
    for comp in components:
        phi0 = comp['phi']
        # Find GT peak
        idx_gt = np.argmin(np.abs(phi_arr - phi0))
        gt_peaks.append(phi_arr[idx_gt])
        
        # Find clean peak within ±fwhmRMSF of true position
        fwhm = reconstruction['fwhmRMSF']
        mask = np.abs(phi_arr - phi0) < 2 * fwhm
        if np.any(mask):
            idx_clean = np.argmax(clean_amp[mask])
            clean_peaks.append(phi_arr[mask][idx_clean])
        else:
            clean_peaks.append(phi0)  # fallback
    
    # Peak position errors
    peak_errors = [abs(g - c) for g, c in zip(gt_peaks, clean_peaks)]
    avg_peak_error = np.mean(peak_errors)
    
    # Amplitude recovery at peak positions
    amp_recoveries = []
    for comp in components:
        phi0 = comp['phi']
        fwhm = reconstruction['fwhmRMSF']
        mask = np.abs(phi_arr - phi0) < fwhm
        if np.any(mask):
            peak_amp = np.max(clean_amp[mask])
            amp_recoveries.append(peak_amp / comp['amplitude'])
    avg_amp_recovery = np.mean(amp_recoveries) if amp_recoveries else 0.0
    
    # PSNR
    data_range = gt_amp_norm.max() - gt_amp_norm.min()
    mse = np.mean((gt_amp_norm - clean_amp_norm)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    return {
        'psnr': float(psnr),
        'rmse': float(rmse),
        'cc': float(cc),
        'avg_peak_position_error': float(avg_peak_error),
        'peak_position_errors': [float(e) for e in peak_errors],
        'avg_amplitude_recovery': float(avg_amp_recovery),
        'fwhm_rmsf': float(reconstruction['fwhmRMSF']),
        'gt_peaks': [float(p) for p in gt_peaks],
        'clean_peaks': [float(p) for p in clean_peaks],
    }


def main():
    # Data paths provided
    data_paths = ['/data/yjh/rmtools_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # We should have at least one outer file
    if not outer_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load the primary (outer) data
    outer_path = outer_files[0]
    print(f"Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {outer_data.keys()}")
    
    # Extract args and kwargs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Args count: {len(args)}")
    print(f"Kwargs keys: {kwargs.keys()}")
    
    # Run the agent's function
    try:
        print("Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent run_inversion completed successfully.")
    except Exception as e:
        print(f"ERROR running agent's run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if we have inner data (chained execution)
    if inner_files:
        # Chained execution pattern
        print("Chained execution detected (inner data found)")
        inner_path = inner_files[0]
        print(f"Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the operator returned by run_inversion
        try:
            print("Running agent operator with inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
            print("Agent operator completed successfully.")
        except Exception as e:
            print(f"ERROR running agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        print("Direct execution detected")
        final_result = agent_output
        std_result = std_output
    
    # For evaluation, we need ground_truth and components
    # These should be available in the kwargs or we need to extract them
    # Looking at the function signature and evaluate_results signature:
    # evaluate_results(ground_truth, reconstruction, components, d_phi)
    
    # The observations dict contains the input data
    # We need to figure out what components and ground_truth are
    # Based on the gen_data_code, observations is the first arg
    
    observations = args[0] if args else kwargs.get('observations', {})
    d_phi = args[2] if len(args) > 2 else kwargs.get('d_phi', 1.0)
    
    # Check if components are stored in observations or elsewhere
    # Based on evaluate_results, we need components with 'phi', 'amplitude', 'chi0'
    # These might be in observations or we need to infer them
    
    components = observations.get('components', None)
    ground_truth = observations.get('ground_truth', observations)
    
    if components is None:
        # Try to extract from observations or use a default
        print("WARNING: 'components' not found in observations. Attempting to infer...")
        # Check if there's a 'components' key at any level
        if 'components' in kwargs:
            components = kwargs['components']
        else:
            # We might need to create dummy components based on the data
            # For now, let's check the std_result structure
            print("Components not found - checking std_output for reference...")
            if std_result and isinstance(std_result, dict):
                # If std_result has the expected structure, we can proceed
                # But we still need components for evaluate_results
                # Let's try to find peaks in the clean FDF as components
                if 'cleanFDF' in std_result:
                    phi_arr = std_result['phi_arr']
                    clean_amp = np.abs(std_result['cleanFDF'])
                    # Find prominent peaks
                    threshold = 0.5 * np.max(clean_amp)
                    peak_indices = np.where(clean_amp > threshold)[0]
                    if len(peak_indices) > 0:
                        # Group consecutive indices and find peak in each group
                        groups = np.split(peak_indices, np.where(np.diff(peak_indices) > 1)[0] + 1)
                        components = []
                        for group in groups:
                            if len(group) > 0:
                                peak_idx = group[np.argmax(clean_amp[group])]
                                components.append({
                                    'phi': phi_arr[peak_idx],
                                    'amplitude': clean_amp[peak_idx],
                                    'chi0': 0.0  # Default phase
                                })
                        print(f"Inferred {len(components)} components from std_result")
    
    if components is None:
        print("ERROR: Cannot determine components for evaluation")
        # Fall back to a simple comparison without the full evaluate_results
        print("Falling back to direct output comparison...")
        
        # Compare key metrics directly
        if isinstance(final_result, dict) and isinstance(std_result, dict):
            # Compare cleanFDF correlation
            if 'cleanFDF' in final_result and 'cleanFDF' in std_result:
                agent_clean = np.abs(final_result['cleanFDF'])
                std_clean = np.abs(std_result['cleanFDF'])
                
                # Normalize
                agent_norm = agent_clean / (np.max(agent_clean) + 1e-10)
                std_norm = std_clean / (np.max(std_clean) + 1e-10)
                
                # Correlation coefficient
                cc = np.corrcoef(agent_norm.flatten(), std_norm.flatten())[0, 1]
                rmse = np.sqrt(np.mean((agent_norm - std_norm)**2))
                
                print(f"Direct comparison - CC: {cc:.4f}, RMSE: {rmse:.4f}")
                
                # Success if correlation > 0.9
                if cc > 0.9:
                    print("SUCCESS: Agent output matches standard output (CC > 0.9)")
                    sys.exit(0)
                else:
                    print(f"FAILURE: Correlation too low ({cc:.4f} < 0.9)")
                    sys.exit(1)
        
        print("Could not perform comparison")
        sys.exit(1)
    
    # Now we have components, run evaluation
    print(f"Using {len(components)} components for evaluation")
    print(f"d_phi = {d_phi}")
    
    try:
        # Evaluate agent result
        print("Evaluating agent result...")
        score_agent = evaluate_results(ground_truth, final_result, components, d_phi)
        print(f"Agent scores: PSNR={score_agent['psnr']:.2f}, RMSE={score_agent['rmse']:.4f}, CC={score_agent['cc']:.4f}")
        
        # Evaluate standard result
        print("Evaluating standard result...")
        score_std = evaluate_results(ground_truth, std_result, components, d_phi)
        print(f"Standard scores: PSNR={score_std['psnr']:.2f}, RMSE={score_std['rmse']:.4f}, CC={score_std['cc']:.4f}")
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        traceback.print_exc()
        
        # Fall back to direct comparison
        print("Falling back to direct output comparison...")
        if isinstance(final_result, dict) and isinstance(std_result, dict):
            if 'cleanFDF' in final_result and 'cleanFDF' in std_result:
                agent_clean = np.abs(final_result['cleanFDF'])
                std_clean = np.abs(std_result['cleanFDF'])
                
                agent_norm = agent_clean / (np.max(agent_clean) + 1e-10)
                std_norm = std_clean / (np.max(std_clean) + 1e-10)
                
                cc = np.corrcoef(agent_norm.flatten(), std_norm.flatten())[0, 1]
                
                print(f"Direct comparison CC: {cc:.4f}")
                
                if cc > 0.9:
                    print("SUCCESS: Agent output matches standard output")
                    sys.exit(0)
                else:
                    print(f"FAILURE: Correlation too low")
                    sys.exit(1)
        
        sys.exit(1)
    
    # Compare scores
    # For PSNR and CC, higher is better
    # For RMSE, lower is better
    
    print("\n=== COMPARISON ===")
    print(f"Scores -> Agent PSNR: {score_agent['psnr']:.2f}, Standard PSNR: {score_std['psnr']:.2f}")
    print(f"Scores -> Agent CC: {score_agent['cc']:.4f}, Standard CC: {score_std['cc']:.4f}")
    print(f"Scores -> Agent RMSE: {score_agent['rmse']:.4f}, Standard RMSE: {score_std['rmse']:.4f}")
    
    # Use correlation coefficient as primary metric (higher is better)
    # Allow 10% margin
    margin = 0.10
    
    # Check if agent performance is acceptable
    psnr_ok = score_agent['psnr'] >= score_std['psnr'] * (1 - margin)
    cc_ok = score_agent['cc'] >= score_std['cc'] * (1 - margin)
    rmse_ok = score_agent['rmse'] <= score_std['rmse'] * (1 + margin)
    
    print(f"\nPSNR acceptable: {psnr_ok}")
    print(f"CC acceptable: {cc_ok}")
    print(f"RMSE acceptable: {rmse_ok}")
    
    # Primary criterion: CC should be good
    if cc_ok and (psnr_ok or rmse_ok):
        print("\nSUCCESS: Agent performance is acceptable")
        sys.exit(0)
    else:
        print("\nFAILURE: Agent performance degraded significantly")
        sys.exit(1)


if __name__ == '__main__':
    main()