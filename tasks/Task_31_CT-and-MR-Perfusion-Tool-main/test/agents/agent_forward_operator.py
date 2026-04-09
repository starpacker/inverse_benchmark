import numpy as np

import warnings

warnings.filterwarnings("ignore")

def forward_operator(volume_stack, mask, image_type='ctp', echo_time=0.03):
    """
    Transforms raw intensity (signal) space to concentration space (physical model).
    x: raw intensity volumes (4D array)
    y_pred: concentration time curves (4D array) and auxiliary metrics (TTP)
    """
    print("Executing Forward Operator (Intensity -> Concentration)...")
    
    nt, nz, ny, nx = volume_stack.shape
    
    # Handle MRP vs CTP physics
    # CTP: Linear relationship (Beer-Lambert approx for iodine: HU approx linear with conc)
    # MRP: Exponential relationship (Signal approx S0 * exp(-TE * R2*))
    
    # 1. Determine baseline (S0)
    # Simple heuristic: average of first few frames or bolus arrival logic
    # We calculate global means to find bolus arrival
    global_means = []
    valid_mask = mask.astype(bool)
    for t in range(nt):
        val = volume_stack[t][valid_mask].mean()
        global_means.append(val)
    
    # Normalize to find start
    gm = np.array(global_means)
    if len(gm) > 0:
        gm_norm = (gm - gm[0]) / (gm[0] + 1e-6)
    else:
        gm_norm = np.zeros(nt)

    # Simple derivative check for bolus arrival
    # Look for sudden change
    diffs = np.diff(gm_norm)
    # Threshold heuristic
    bolus_idx = 0
    for t in range(2, len(diffs)):
        # Look at window
        window_diff = diffs[t]
        if abs(window_diff) > 0.01: # 1% change
            bolus_idx = t
            break
            
    s0_idx = max(0, bolus_idx - 1)
    
    # Calculate S0 map
    s0_vol = volume_stack[:s0_idx+1].mean(axis=0)
    
    # 2. Calculate Concentration
    if image_type == 'ctp':
        # C(t) = S(t) - S0
        ctc = volume_stack - s0_vol[np.newaxis, :, :, :]
        # Clip negatives strictly? Usually yes for physics, but noise exists.
        # We allow noise but zero out very negative values later or keep for SVD
    elif image_type == 'mrp':
        # C(t) = - (1/TE) * ln(S(t) / S0)
        # Invert intensity for calculation stability if needed (handled in log)
        epsilon = 1e-8
        ratio = volume_stack / (s0_vol[np.newaxis, :, :, :] + epsilon)
        ratio = np.clip(ratio, epsilon, None)
        ctc = -(1.0 / echo_time) * np.log(ratio)
    else:
        raise ValueError(f"Unknown image type: {image_type}")

    # Apply mask
    ctc = ctc * mask[np.newaxis, :, :, :]
    
    # 3. Calculate Time-To-Peak (TTP) on the CTC
    # This is a derived feature often used to guide the inversion (AIF selection)
    # Shift time so TTP is relative to start
    # We return the index or time map. Let's return the map.
    # We need time_index passed in? No, we can just return indices or treat strictly as 
    # part of the operator output.
    # To keep this function pure regarding time, we return argmax indices.
    
    ttp_indices = np.argmax(ctc, axis=0)
    
    return ctc, s0_idx, ttp_indices
