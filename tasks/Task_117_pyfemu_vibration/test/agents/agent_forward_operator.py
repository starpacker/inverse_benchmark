import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.linalg import eigh

def forward_operator(damage_vec, n_elem, L_total, EI, rhoA, n_modes):
    """
    Forward operator: Given damage parameters, compute modal response.
    
    Physics:
        - N-element Euler-Bernoulli beam → global K, M matrices
        - Damage in element i → K_i × (1 − d_i), d_i ∈ [0, 1]
        - Eigenvalue problem: (K − ω² M) φ = 0
    
    Args:
        damage_vec: Damage parameters for each element (length n_elem)
        n_elem: Number of elements
        L_total: Total beam length
        EI: Flexural rigidity
        rhoA: Mass per unit length
        n_modes: Number of modes to compute
    
    Returns:
        freqs: Natural frequencies (Hz)
        modes: Mode shape matrix (n_dof_free x n_modes)
    """
    # Assemble global stiffness and mass matrices with damage
    n_dof = 2 * (n_elem + 1)
    L_e = L_total / n_elem
    K_global = np.zeros((n_dof, n_dof))
    M_global = np.zeros((n_dof, n_dof))
    
    for i in range(n_elem):
        # Apply damage: reduce stiffness by factor (1 - d_i)
        EI_eff = EI * (1.0 - damage_vec[i])
        
        # Element stiffness matrix
        k_e = (EI_eff / L_e ** 3) * np.array([
            [ 12,    6*L_e,  -12,    6*L_e],
            [  6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
            [-12,   -6*L_e,   12,   -6*L_e],
            [  6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
        ])
        
        # Element mass matrix (consistent mass formulation)
        m_e = (rhoA * L_e / 420.0) * np.array([
            [156,    22*L_e,   54,   -13*L_e],
            [ 22*L_e,  4*L_e**2,  13*L_e, -3*L_e**2],
            [ 54,    13*L_e,  156,   -22*L_e],
            [-13*L_e, -3*L_e**2, -22*L_e,  4*L_e**2]
        ])
        
        # Assembly
        idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for a in range(4):
            for b in range(4):
                K_global[idx[a], idx[b]] += k_e[a, b]
                M_global[idx[a], idx[b]] += m_e[a, b]
    
    # Apply simply-supported boundary conditions: w=0 at both ends
    bc_dofs = [0, 2 * n_elem]
    free = [d for d in range(n_dof) if d not in bc_dofs]
    K_free = K_global[np.ix_(free, free)]
    M_free = M_global[np.ix_(free, free)]
    
    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(K_free, M_free)
    
    # Extract first n_modes
    omega2 = eigvals[:n_modes]
    freqs = np.sqrt(np.abs(omega2)) / (2 * np.pi)
    modes = eigvecs[:, :n_modes]
    
    # Normalize mode shapes
    for j in range(n_modes):
        modes[:, j] /= np.max(np.abs(modes[:, j])) + 1e-30
    
    return freqs, modes
