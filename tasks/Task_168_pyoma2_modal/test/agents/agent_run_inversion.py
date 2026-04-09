import os

import numpy as np

from scipy import linalg, signal

import matplotlib

matplotlib.use('Agg')

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inversion(data):
    """
    Frequency Domain Decomposition (FDD) with Enhanced FDD for damping estimation.
    
    Identifies modal parameters (frequencies, damping ratios, mode shapes) from
    ambient vibration acceleration data.
    
    Returns a dict with identified modal parameters.
    """
    accelerations = data["accelerations"]
    fs = data["fs"]
    n_dof = data["n_dof"]
    dt = data["dt"]
    
    nfft = 8192
    noverlap = nfft * 3 // 4
    
    # Compute cross-spectral density matrix
    freqs_psd = None
    Gxx = None
    
    for i in range(n_dof):
        for j in range(n_dof):
            f, Pij = signal.csd(accelerations[:, i], accelerations[:, j],
                                fs=fs, nperseg=nfft, noverlap=noverlap,
                                window='hann')
            if Gxx is None:
                freqs_psd = f
                n_freq = len(f)
                Gxx = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
            Gxx[:, i, j] = Pij
    
    # SVD of spectral matrix at each frequency
    sv1 = np.zeros(n_freq)
    sv2 = np.zeros(n_freq)
    U_all = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
    
    for k in range(n_freq):
        U, s, Vh = np.linalg.svd(Gxx[k])
        sv1[k] = s[0]
        sv2[k] = s[1] if len(s) > 1 else 0
        U_all[k] = U
    
    sv1_db = 10 * np.log10(sv1 + 1e-30)
    
    # Intelligent peak picking
    f_max_search = 6.0
    freq_mask = (freqs_psd >= 0.1) & (freqs_psd <= f_max_search)
    freq_idx_start = np.argmax(freq_mask)
    freq_idx_end = len(freq_mask) - np.argmax(freq_mask[::-1])
    
    sv1_db_restricted = sv1_db[freq_idx_start:freq_idx_end]
    freqs_restricted = freqs_psd[freq_idx_start:freq_idx_end]
    
    df = freqs_psd[1] - freqs_psd[0]
    min_distance = max(int(0.3 / df), 3)
    
    peak_indices_rel, peak_props = signal.find_peaks(
        sv1_db_restricted,
        distance=min_distance,
        prominence=2.0,
        height=np.max(sv1_db_restricted) - 40,
    )
    
    # Map back to absolute indices
    peak_indices = peak_indices_rel + freq_idx_start
    
    # Sort by prominence and take top n_dof
    if len(peak_indices) > n_dof:
        prominences = peak_props['prominences']
        top_idx = np.argsort(prominences)[-n_dof:]
        peak_indices = np.sort(peak_indices[top_idx])
    
    freq_identified = freqs_psd[peak_indices]
    
    # Mode shapes from first singular vector
    mode_shapes_identified = np.zeros((n_dof, len(peak_indices)))
    for m, pk in enumerate(peak_indices):
        u1 = U_all[pk, :, 0]
        phi = np.real(u1)
        phi /= np.max(np.abs(phi))
        mode_shapes_identified[:, m] = phi
    
    # Enhanced FDD damping estimation
    damping_identified = np.zeros(len(peak_indices))
    
    for m, pk in enumerate(peak_indices):
        fn = freqs_psd[pk]
        phi_peak = U_all[pk, :, 0]
        
        # Define SDOF bell: region where MAC > 0.80 around the peak
        left_idx = pk
        right_idx = pk
        for idx in range(pk - 1, max(freq_idx_start, pk - 200), -1):
            phi_test = U_all[idx, :, 0]
            mac_test = np.abs(np.dot(np.conj(phi_peak), phi_test))**2 / \
                       (np.dot(np.conj(phi_peak), phi_peak).real *
                        np.dot(np.conj(phi_test), phi_test).real)
            if mac_test < 0.80:
                break
            left_idx = idx
        
        for idx in range(pk + 1, min(freq_idx_end, pk + 200)):
            phi_test = U_all[idx, :, 0]
            mac_test = np.abs(np.dot(np.conj(phi_peak), phi_test))**2 / \
                       (np.dot(np.conj(phi_peak), phi_peak).real *
                        np.dot(np.conj(phi_test), phi_test).real)
            if mac_test < 0.80:
                break
            right_idx = idx
        
        # Extract SDOF bell from the first singular value
        bell = np.zeros(n_freq)
        bell[left_idx:right_idx + 1] = sv1[left_idx:right_idx + 1]
        
        # IFFT -> free-decay (autocorrelation of SDOF response)
        bell_sym = np.concatenate([bell, bell[-2:0:-1]])
        free_decay = np.fft.ifft(bell_sym).real
        free_decay = free_decay[:len(free_decay) // 2]
        
        # Normalize
        if np.abs(free_decay[0]) > 1e-30:
            free_decay_norm = free_decay / free_decay[0]
        else:
            damping_identified[m] = 0.03
            continue
        
        # Find zero crossings to estimate damped frequency
        t_decay = np.arange(len(free_decay)) / fs
        crossings = []
        for ci in range(1, min(len(free_decay_norm), 500)):
            if free_decay_norm[ci - 1] * free_decay_norm[ci] < 0:
                t_cross = t_decay[ci - 1] + (0 - free_decay_norm[ci - 1]) / \
                          (free_decay_norm[ci] - free_decay_norm[ci - 1]) * dt
                crossings.append(t_cross)
        
        if len(crossings) >= 4:
            # Period from consecutive positive-going crossings
            periods = []
            for ci in range(0, len(crossings) - 2, 2):
                periods.append(crossings[ci + 2] - crossings[ci])
            T_d = np.median(periods)
            f_d = 1.0 / T_d if T_d > 0 else fn
            
            # Logarithmic decrement from envelope peaks
            env_peaks_idx, _ = signal.find_peaks(np.abs(free_decay_norm[:500]))
            if len(env_peaks_idx) >= 2:
                env_vals = np.abs(free_decay_norm[env_peaks_idx])
                log_decs = []
                for ci in range(len(env_vals) - 1):
                    if env_vals[ci + 1] > 1e-6 and env_vals[ci] > env_vals[ci + 1]:
                        log_decs.append(np.log(env_vals[ci] / env_vals[ci + 1]))
                if log_decs:
                    delta = np.median(log_decs)
                    zeta_est = delta / np.sqrt(4 * np.pi**2 + delta**2)
                    damping_identified[m] = max(0.001, min(zeta_est, 0.2))
                else:
                    damping_identified[m] = 0.03
            else:
                damping_identified[m] = 0.03
        else:
            # Fallback: half-power bandwidth
            peak_val = sv1_db[pk]
            half_power = peak_val - 3.0
            left_hp = pk
            while left_hp > 0 and sv1_db[left_hp] > half_power:
                left_hp -= 1
            right_hp = pk
            while right_hp < n_freq - 1 and sv1_db[right_hp] > half_power:
                right_hp += 1
            f_left = freqs_psd[left_hp]
            f_right = freqs_psd[right_hp]
            bandwidth = f_right - f_left
            damping_identified[m] = bandwidth / (2.0 * fn) if fn > 0 else 0.03
    
    result = {
        "freq_identified": freq_identified,
        "damping_identified": damping_identified,
        "mode_shapes_identified": mode_shapes_identified,
        "peak_indices": peak_indices,
        "freqs_psd": freqs_psd,
        "sv1": sv1,
        "sv2": sv2,
        "sv1_db": sv1_db,
        "U_all": U_all,
        "n_freq": n_freq,
        "f_max_search": f_max_search,
    }
    
    return result
