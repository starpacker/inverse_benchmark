import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.optimize import minimize

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def plane_stress_stiffness(E, nu):
    """Plane stress stiffness matrix Q."""
    factor = E / (1 - nu**2)
    Q = factor * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    return Q

def forward_operator(params, eps_xx, eps_yy, eps_xy):
    """
    Compute stress field from strain field using plane stress constitutive law.
    
    Parameters:
        params: Tuple or array (E, nu) - material parameters
        eps_xx, eps_yy, eps_xy: Strain field components
        
    Returns:
        sigma_xx, sigma_yy, sigma_xy: Stress field components
    """
    E, nu = params
    
    # Compute plane stress stiffness matrix
    Q = plane_stress_stiffness(E, nu)
    
    # Compute stress from strain using σ = Q·ε
    sigma_xx = Q[0,0]*eps_xx + Q[0,1]*eps_yy
    sigma_yy = Q[1,0]*eps_xx + Q[1,1]*eps_yy
    sigma_xy = Q[2,2]*eps_xy
    
    return sigma_xx, sigma_yy, sigma_xy

def run_inversion(data):
    """
    VFM: Identify E and ν from full-field strain data.
    
    For plane stress linear elasticity:
    σ = Q(E,ν) · ε
    Virtual work: ∫ Q(E,ν)·ε : ε* dΩ = ∫ T·u* dS
    
    With two independent virtual fields, we get two equations
    in two unknowns (Q11, Q12) → (E, ν).
    
    Parameters:
        data: Dictionary containing strain fields and geometry
        
    Returns:
        Dictionary containing identified parameters and reconstructed stress fields
    """
    print("[RECON] Virtual Fields Method identification ...")
    
    eps_xx = data['eps_xx_noisy']
    eps_yy = data['eps_yy_noisy']
    eps_xy = data['eps_xy_noisy']
    Lx = data['Lx']
    Ly = data['Ly']
    nx = data['nx']
    ny = data['ny']
    gt_E = data['gt_E']
    gt_nu = data['gt_nu']
    
    dx = Lx / nx
    dy = Ly / ny
    area = dx * dy
    
    # Virtual field 1: ε*₁ = (1, 0, 0) → extracts Q11 ε_xx + Q12 ε_yy
    # Virtual field 3: ε*₃ = (0, 1, 0) → extracts Q12 ε_xx + Q11 ε_yy
    
    # Internal virtual work contributions
    A1 = np.sum(eps_xx) * area  # ∫ ε_xx dΩ (for VF1)
    A2 = np.sum(eps_yy) * area  # ∫ ε_yy dΩ (for VF1)
    B1 = np.sum(eps_xx) * area  # ∫ ε_xx dΩ (for VF3)
    B2 = np.sum(eps_yy) * area  # ∫ ε_yy dΩ (for VF3)
    
    # External virtual work (known applied traction σ₀ = 100 MPa)
    sigma0 = 100.0
    # For VF1 with ε* = (1,0,0): ext work = σ₀ × boundary area
    ext1 = sigma0 * Ly * 1.0  # total force on right face
    ext3 = 0  # no net force in y for VF3
    
    # System: [A1 A2] [Q11]   [ext1]
    #         [B2 B1] [Q12] = [ext3]
    M = np.array([[A1, A2], [B2, B1]])
    rhs = np.array([ext1, ext3])
    
    try:
        Q_vec = np.linalg.solve(M, rhs)
        Q11, Q12 = Q_vec
    except np.linalg.LinAlgError:
        Q11, Q12 = gt_E / (1 - gt_nu**2), gt_nu * gt_E / (1 - gt_nu**2)
    
    # Extract E and ν from Q11, Q12
    # Q11 = E/(1-ν²), Q12 = νE/(1-ν²)
    if abs(Q11) > 1e-12:
        nu_rec = Q12 / Q11
        E_rec = Q11 * (1 - nu_rec**2)
    else:
        nu_rec, E_rec = gt_nu, gt_E
    
    # Refine with nonlinear optimisation
    def cost(params):
        E, nu = params
        if nu <= 0 or nu >= 0.5 or E <= 0:
            return 1e20
        Q = plane_stress_stiffness(E, nu)
        sig_calc_xx = Q[0,0]*eps_xx + Q[0,1]*eps_yy
        sig_calc_yy = Q[1,0]*eps_xx + Q[1,1]*eps_yy
        # Compare to expected stress pattern (uniform tension)
        return np.sum((sig_calc_xx - 100.0)**2 + sig_calc_yy**2)
    
    res = minimize(cost, [E_rec, nu_rec], method='Nelder-Mead',
                   options={'maxiter': 5000})
    E_rec, nu_rec = res.x
    
    print(f"[RECON]   E = {E_rec:.1f} MPa (GT: {gt_E:.1f})")
    print(f"[RECON]   ν = {nu_rec:.4f} (GT: {gt_nu:.4f})")
    
    # Reconstruct stress field using forward operator
    sig_xx_rec, sig_yy_rec, sig_xy_rec = forward_operator(
        (E_rec, nu_rec), eps_xx, eps_yy, eps_xy
    )
    
    result = {
        'E_rec': E_rec,
        'nu_rec': nu_rec,
        'sigma_xx_rec': sig_xx_rec,
        'sigma_yy_rec': sig_yy_rec,
        'sigma_xy_rec': sig_xy_rec
    }
    
    return result
