import numpy as np

import matplotlib

matplotlib.use("Agg")

def euler_bernoulli_element(L_e, EI, rhoA):
    """
    4×4 element stiffness and consistent mass matrices for an
    Euler-Bernoulli beam element of length L_e.
    """
    k_e = (EI / L_e ** 3) * np.array([
        [ 12,    6*L_e,  -12,    6*L_e],
        [  6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12,   -6*L_e,   12,   -6*L_e],
        [  6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ])

    m_e = (rhoA * L_e / 420.0) * np.array([
        [156,    22*L_e,   54,   -13*L_e],
        [ 22*L_e,  4*L_e**2,  13*L_e, -3*L_e**2],
        [ 54,    13*L_e,  156,   -22*L_e],
        [-13*L_e, -3*L_e**2, -22*L_e,  4*L_e**2]
    ])
    return k_e, m_e

def assemble(n_elem, L_total, EI, rhoA, damage_vec):
    """
    Assemble global K and M for a simply-supported beam.
    damage_vec[i] ∈ [0, 1] reduces stiffness of element i.
    """
    n_dof = 2 * (n_elem + 1)   # 2 DOF per node (w, θ)
    L_e   = L_total / n_elem
    K_global = np.zeros((n_dof, n_dof))
    M_global = np.zeros((n_dof, n_dof))

    for i in range(n_elem):
        EI_eff = EI * (1.0 - damage_vec[i])
        k_e, m_e = euler_bernoulli_element(L_e, EI_eff, rhoA)
        idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for a in range(4):
            for b in range(4):
                K_global[idx[a], idx[b]] += k_e[a, b]
                M_global[idx[a], idx[b]] += m_e[a, b]

    # Simply-supported BC: w=0 at node 0 and node N
    bc_dofs = [0, 2 * n_elem]   # translational DOFs at ends
    free = [d for d in range(n_dof) if d not in bc_dofs]
    K_free = K_global[np.ix_(free, free)]
    M_free = M_global[np.ix_(free, free)]
    return K_free, M_free, free
