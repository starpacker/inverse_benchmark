import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def align_gains(g_cal: np.ndarray, g_true: np.ndarray, ref_ant: int) -> np.ndarray:
    """Align calibrated gains to true gains by removing global phase offset."""
    ratio = g_true / np.where(np.abs(g_cal) > 1e-15, g_cal, 1e-15)
    mask = np.ones(g_cal.shape[0], dtype=bool)
    mask[ref_ant] = False
    ratio_excl = ratio[mask]
    phase_offset = np.angle(np.mean(ratio_excl))
    g_aligned = g_cal * np.exp(1j * phase_offset)
    return g_aligned

def run_inversion(
    v_obs: np.ndarray,
    v_model: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    n_ant: int,
    g_true: np.ndarray,
    max_iter: int,
    conv_tol: float,
    ref_ant: int,
) -> dict:
    """
    Run the Stefcal inverse solver to recover gains from corrupted visibilities.
    
    Stefcal (StefCal / Salvini & Wijnholds 2014) gain calibration.
    
    The measurement equation per baseline (p, q):
        V_obs_{pq} = g_p * V_model_{pq} * conj(g_q) + noise
    
    Parameters:
        v_obs: Observed visibilities (n_bl, n_freq, n_time)
        v_model: Model visibilities (n_bl, n_freq, n_time)
        ant1: First antenna indices
        ant2: Second antenna indices
        n_ant: Number of antennas
        g_true: True gains for alignment
        max_iter: Maximum iterations
        conv_tol: Convergence tolerance
        ref_ant: Reference antenna index
    
    Returns:
        Dictionary containing calibrated gains and convergence history
    """
    n_bl, n_freq, n_time = v_obs.shape
    
    # Initialize gains to unity
    g = np.ones((n_ant, n_freq, n_time), dtype=np.complex128)
    convergence = []
    
    for iteration in range(max_iter):
        g_old = g.copy()
        g_new = g.copy()
        
        for p in range(n_ant):
            if p == ref_ant:
                continue
            
            numerator = np.zeros((n_freq, n_time), dtype=np.complex128)
            denominator = np.zeros((n_freq, n_time), dtype=np.float64)
            
            for bl_idx in range(n_bl):
                i, j = ant1[bl_idx], ant2[bl_idx]
                
                if i == p:
                    q = j
                    z = v_model[bl_idx] * np.conj(g_old[q])
                    numerator += v_obs[bl_idx] * np.conj(z)
                    denominator += np.abs(z) ** 2
                
                elif j == p:
                    q = i
                    v_obs_pq = np.conj(v_obs[bl_idx])
                    v_mod_pq = np.conj(v_model[bl_idx])
                    z = v_mod_pq * np.conj(g_old[q])
                    numerator += v_obs_pq * np.conj(z)
                    denominator += np.abs(z) ** 2
            
            safe_denom = np.where(denominator > 1e-30, denominator, 1e-30)
            g_new[p] = numerator / safe_denom
        
        # Damped update for stability
        damping = 0.5
        g = damping * g_new + (1.0 - damping) * g_old
        
        # Fix reference antenna
        g[ref_ant] = 1.0 + 0j
        
        # Check convergence
        rel_change = np.linalg.norm(g - g_old) / max(np.linalg.norm(g_old), 1e-30)
        convergence.append(rel_change)
        
        if rel_change < conv_tol:
            print(f"  Stefcal converged at iteration {iteration + 1}, "
                  f"rel_change = {rel_change:.2e}")
            break
    else:
        print(f"  Stefcal did not converge after {max_iter} iterations, "
              f"final rel_change = {convergence[-1]:.2e}")
    
    # Align gains to resolve phase ambiguity
    g_cal = align_gains(g, g_true, ref_ant)
    
    result = {
        'g_cal': g_cal,
        'convergence': convergence,
        'n_iterations': len(convergence),
    }
    
    return result
