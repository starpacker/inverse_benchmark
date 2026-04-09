import numpy as np

import matplotlib

matplotlib.use('Agg')

def run_inversion(
    observations,
    phi_max,
    d_phi,
    clean_cutoff_sigma,
    clean_gain,
    clean_max_iter,
    noise_sigma_jy
):
    """
    RM Synthesis + Hogbom CLEAN to recover Faraday depth spectrum.
    
    Steps:
    1. RM Synthesis: Discrete Fourier transform of P(λ²) to get dirty F(φ)
    2. Compute RMSF (Rotation Measure Spread Function)
    3. CLEAN deconvolution to remove sidelobes
    
    Args:
        observations: dict with lambda_sq, Q, U, dQ, dU
        phi_max: Maximum Faraday depth (rad/m²)
        d_phi: Faraday depth resolution (rad/m²)
        clean_cutoff_sigma: CLEAN cutoff in units of sigma
        clean_gain: CLEAN loop gain
        clean_max_iter: Maximum CLEAN iterations
        noise_sigma_jy: Noise level per channel
    
    Returns:
        dict with phi_arr, dirtyFDF, cleanFDF, ccArr, residFDF, RMSFArr, phi2_arr, fwhmRMSF, lam0Sq
    """
    from RMtools_1D.do_RMsynth_1D import do_rmsynth_planes, get_rmsf_planes
    from RMtools_1D.do_RMclean_1D import do_rmclean_hogbom
    
    lambda_sq = observations['lambda_sq']
    Q = observations['Q']
    U = observations['U']
    dQ = observations['dQ']
    dU = observations['dU']
    
    # Create Faraday depth array
    phi_arr = np.arange(-phi_max, phi_max + d_phi, d_phi)
    
    # Weight array (inverse variance)
    weight_arr = 1.0 / (dQ**2 + dU**2)
    
    # Step 1: Compute RMSF
    print("  [RM] Computing Rotation Measure Spread Function (RMSF)...")
    # phi2 array must be twice the length of phi_arr (for RMSF deconvolution)
    phi2_arr = np.arange(-2 * phi_max, 2 * phi_max + d_phi, d_phi)
    rmsf_results = get_rmsf_planes(
        lambdaSqArr_m2=lambda_sq,
        phiArr_radm2=phi2_arr,
        weightArr=weight_arr,
        lam0Sq_m2=None,
        nBits=64
    )
    RMSFArr = rmsf_results.RMSFcube
    phi2_arr = rmsf_results.phi2Arr  # Use the actual phi2 from RMSF result
    fwhmRMSF = float(rmsf_results.fwhmRMSFArr)
    print(f"  [RM] RMSF FWHM: {fwhmRMSF:.2f} rad/m²")
    print(f"  [RM] RMSF shape: {RMSFArr.shape}, phi2 shape: {phi2_arr.shape}")
    
    # Step 2: RM Synthesis (dirty FDF)
    print("  [RM] Computing dirty Faraday Dispersion Function (FDF)...")
    synth_results = do_rmsynth_planes(
        dataQ=Q,
        dataU=U,
        lambdaSqArr_m2=lambda_sq,
        phiArr_radm2=phi_arr,
        weightArr=weight_arr,
        lam0Sq_m2=None,
        nBits=64
    )
    dirtyFDF = synth_results.FDFcube
    lam0Sq = synth_results.lam0Sq_m2
    
    print(f"  [RM] Dirty FDF shape: {dirtyFDF.shape}")
    print(f"  [RM] Reference λ²: {lam0Sq:.6f} m²")
    
    # Step 3: CLEAN deconvolution
    print("  [RM] Running RM-CLEAN (Hogbom)...")
    noise_level = noise_sigma_jy / np.sqrt(len(lambda_sq))
    cutoff = clean_cutoff_sigma * noise_level
    
    clean_results = do_rmclean_hogbom(
        dirtyFDF=dirtyFDF,
        phiArr_radm2=phi_arr,
        RMSFArr=RMSFArr,
        phi2Arr_radm2=phi2_arr,
        fwhmRMSFArr=np.array([fwhmRMSF]),
        cutoff=cutoff,
        maxIter=clean_max_iter,
        gain=clean_gain,
        nBits=64,
        verbose=False,
        doPlots=False
    )
    
    # Handle both named tuple and regular tuple returns
    if hasattr(clean_results, 'cleanFDF'):
        cleanFDF = clean_results.cleanFDF
        ccArr = clean_results.ccArr
        iterCount = clean_results.iterCountArr
        residFDF = clean_results.residFDF
    else:
        # Fallback: tuple return (cleanFDF, ccArr, iterCount, residFDF)
        cleanFDF, ccArr, iterCount, residFDF = clean_results
    
    print(f"  [RM] CLEAN iterations: {iterCount}")
    print(f"  [RM] Clean FDF shape: {cleanFDF.shape}")
    
    return {
        'phi_arr': phi_arr,
        'dirtyFDF': dirtyFDF,
        'cleanFDF': cleanFDF,
        'ccArr': ccArr,
        'residFDF': residFDF,
        'RMSFArr': RMSFArr,
        'phi2_arr': phi2_arr,
        'fwhmRMSF': fwhmRMSF,
        'lam0Sq': lam0Sq,
    }
