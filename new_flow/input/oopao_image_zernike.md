# main.py
```python
import sys
import os
import time

# Ensure OOPAO is in path
sys.path.append('/home/yjh/OOPAO')

import numpy as np
import matplotlib.pyplot as plt
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Zernike import Zernike

# =============================================================================
# 1. HELPER FUNCTIONS (Explicit Implementation of Algorithms)
# =============================================================================

def zernike_radial_explicit(n, m, r):
    """
    Explicit calculation of Zernike Radial Polynomial R_n^m(r)
    Formula:
    R_n^m(r) = Sum_{k=0}^{(n-m)/2} [(-1)^k * (n-k)!] / [k! * ((n+m)/2 - k)! * ((n-m)/2 - k)!] * r^(n-2k)
    """
    R = np.zeros_like(r)
    
    # Check parity
    if (n - m) % 2 != 0:
        return R # R is zero if n-m is odd
        
    for k in range((n - m) // 2 + 1):
        # Coefficients
        num = ((-1)**k) * np.math.factorial(n - k)
        denom = (np.math.factorial(k) * 
                 np.math.factorial((n + m) // 2 - k) * 
                 np.math.factorial((n - m) // 2 - k))
        
        R += (num / denom) * (r**(n - 2 * k))
        
    return R

def zernike_mode_explicit(n, m, X, Y, D):
    """
    Generates a Zernike mode Z_n^m on the grid (X, Y)
    """
    # Normalized coordinates
    R = np.sqrt(X**2 + Y**2) / (D/2)
    Theta = np.arctan2(Y, X)
    
    # Mask outside pupil
    mask = R <= 1.0
    
    # Radial part
    # We use our explicit function, but need to handle scalar inputs if vectorized isn't working perfectly, 
    # though numpy handles array inputs for power operations.
    # To be safe and efficient, we implement a vectorized version of radial poly above implicitly.
    
    # Initialize Z
    Z = np.zeros_like(X)
    
    # Calculate R_nm only inside pupil
    R_vals = R[mask]
    Theta_vals = Theta[mask]
    
    # Radial function values
    # Re-implement small loop here for vectorized R_vals
    Rad = np.zeros_like(R_vals)
    if (n - m) % 2 == 0:
        for k in range((n - m) // 2 + 1):
            if (n - k) < 0 or ((n + m) // 2 - k) < 0 or ((n - m) // 2 - k) < 0:
                continue # Should not happen if n>=m and n-m is even, but safety first
            num = ((-1)**k) * np.math.factorial(n - k)
            denom = (np.math.factorial(k) * 
                     np.math.factorial((n + m) // 2 - k) * 
                     np.math.factorial((n - m) // 2 - k))
            Rad += (num / denom) * (R_vals**(n - 2 * k))
            
    # Azimuthal part
    if m == 0:
        Z[mask] = np.sqrt(n + 1) * Rad
    elif m > 0:
        Z[mask] = np.sqrt(2 * (n + 1)) * Rad * np.cos(m * Theta_vals)
    else: # m < 0
        Z[mask] = np.sqrt(2 * (n + 1)) * Rad * np.sin(-m * Theta_vals)
        
    return Z

def compute_psf_physical_explicit(tel, phase_map):
    """
    Manually computes the PSF from the phase map using physical optics principles.
    PSF = | FFT( Amplitude * exp(i * Phase) ) |^2
    """
    # 1. Get Pupil Amplitude (Binary mask * Reflectivity)
    amplitude = tel.pupil
    
    # 2. Create Complex Field (Electric Field)
    # E = A * e^(i * phi)
    electric_field = amplitude * np.exp(1j * phase_map)
    
    # 3. Apply Zero Padding (for sampling)
    zero_padding = 4
    N = tel.resolution
    N_padded = N * zero_padding
    
    # Pad the electric field
    pad_width = (N_padded - N) // 2
    electric_field_padded = np.pad(electric_field, pad_width)
    
    # 4. Fourier Transform (Propagation to Focal Plane)
    # Shift before FFT to center zero frequency
    complex_focal_plane = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(electric_field_padded)))
    
    # 5. Compute Intensity (PSF)
    psf = np.abs(complex_focal_plane)**2
    
    # Normalize
    psf = psf / psf.max()
    
    return psf

def inverse_model_explicit(opd_map, tel, Z_inv):
    """
    Inverse Model: Decomposes the OPD map into Zernike coefficients.
    c = Z^+ * phi
    """
    # 1. Mask OPD with pupil (Extract valid phase points)
    # We only care about phase inside the telescope pupil
    opd_masked = opd_map[np.where(tel.pupil==1)]
    
    # 2. Project onto Zernike Basis (Least Squares fitting)
    coeffs = Z_inv @ opd_masked
    
    return coeffs

# =============================================================================
# 2. MAIN SIMULATION
# =============================================================================

def main():
    print("=================================================================")
    print("   Explicit Image Formation & Zernike Decomposition (Deepened)   ")
    print("=================================================================")
    
    # --- 1. System Initialization ---
    print("\n[1] Initializing System...")
    tel = Telescope(resolution=120, diameter=8, samplingTime=1/1000, centralObstruction=0.0) # No obstruction for clean Zernikes
    ngs = Source(optBand='I', magnitude=10)
    ngs * tel
    
    # --- 2. Explicit Zernike Generation ---
    print("\n[2] Generating Zernike Basis Explicitly...")
    
    # Create coordinate grid
    y, x = np.indices((tel.resolution, tel.resolution))
    y = (y - tel.resolution/2) * tel.pixelSize
    x = (x - tel.resolution/2) * tel.pixelSize
    
    # Let's generate first 15 modes (Noll index 1 to 15)
    n_modes = 15
    zernike_basis = np.zeros((tel.resolution**2, n_modes))
    zernike_basis_2d = np.zeros((n_modes, tel.resolution, tel.resolution))
    
    # Mapping Noll Index (j) to (n, m)
    # j=1: n=0, m=0 (Piston)
    # j=2: n=1, m=1 (Tilt X)
    # j=3: n=1, m=-1 (Tilt Y)
    # j=4: n=2, m=0 (Defocus)
    # ...
    # We use OOPAO's indexer for convenience or hardcode a few
    # Hardcoding a few for demonstration of the explicit function
    # Note: Zernike radial index n must be >= |m|
    noll_indices = [
        (0,0), # Piston (j=1)
        (1,1), # Tilt X (j=2)
        (1,-1), # Tilt Y (j=3)
        (2,0), # Defocus (j=4)
        (2,-2), # Astigmatism (j=5)
        (2,2), # Astigmatism (j=6)
    ]
    
    print("    Generating modes using explicit radial polynomials...")
    for j, (n, m) in enumerate(noll_indices):
        mode = zernike_mode_explicit(n, m, x, y, tel.D)
        # Flatten and store
        zernike_basis[:, j] = mode.flatten()
        zernike_basis_2d[j] = mode
        
    # Use OOPAO for the full set to ensure coverage for decomposition
    # (The explicit function works but generating the (n,m) list for high orders is tedious)
    Z = Zernike(telObject=tel, J=100)
    Z.computeZernike(tel)
    Z_inv = np.linalg.pinv(Z.modes) # Pseudoinverse of the basis
    
    # --- 3. Forward Model: PSF from Explicit Zernike ---
    print("\n[3] Forward Model: Generating PSF from specific aberrations...")
    
    # Create a phase map: 0.5 rad of Defocus (n=2, m=0) + 0.5 rad of Astigmatism (n=2, m=2)
    # Defocus is index 3 in our manual list (0-based)
    # Astigmatism is index 5
    
    phase_map = (0.5 * zernike_basis_2d[3] + 0.5 * zernike_basis_2d[5])
    # Convert to OPD [m] for consistency with OOPAO
    # Phase = 2*pi*OPD / lambda  => OPD = Phase * lambda / 2*pi
    opd_map = phase_map * ngs.wavelength / (2*np.pi)
    
    # Compute PSF
    print("    Computing PSF via FFT...")
    psf = compute_psf_physical_explicit(tel, phase_map)
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(phase_map * tel.pupil)
    plt.title("Input Phase (Explicit Zernikes)")
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(np.log10(psf + 1e-10))
    plt.title("Resulting PSF (Log)")
    plt.colorbar()
    plt.savefig("zernike_forward.png")
    print("    Saved zernike_forward.png")
    
    # --- 4. Inverse Model: Decomposing Atmosphere ---
    print("\n[4] Inverse Model: Decomposing Atmospheric Turbulence...")
    
    atm = Atmosphere(telescope=tel, r0=0.15, L0=25, fractionalR0=[1], windSpeed=[10], windDirection=[0], altitude=[0])
    atm.initializeAtmosphere(tel)
    
    n_iter = 10
    rmse_history = []
    
    for i in range(n_iter):
        atm.update()
        current_opd = atm.OPD.copy()
        
        # Explicit Decomposition Step
        # c = Z_dagger * opd
        coeffs = inverse_model_explicit(current_opd, tel, Z_inv)
        
        # Reconstruction
        # opd_rec = Z * c
        reconstructed_opd = np.squeeze(Z.modesFullRes @ coeffs)
        
        # Error
        diff = (current_opd - reconstructed_opd) * tel.pupil
        rmse = np.std(diff[tel.pupil==1])
        rmse_history.append(rmse)
        
        print(f"    Iter {i+1}: Fitting RMSE = {rmse*1e9:.1f} nm")
        
    plt.figure()
    plt.plot(rmse_history, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE [m]')
    plt.title('Zernike Decomposition Residual')
    plt.savefig("zernike_inverse.png")
    print("    Saved zernike_inverse.png")

if __name__ == "__main__":
    main()
```
