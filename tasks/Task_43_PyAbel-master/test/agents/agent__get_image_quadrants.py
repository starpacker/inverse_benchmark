import numpy as np

def _get_image_quadrants(IM, reorient=True, symmetry_axis=None):
    IM = np.atleast_2d(IM)
    n, m = IM.shape
    n_c = n // 2 + n % 2
    m_c = m // 2 + m % 2

    Q0 = IM[:n_c, -m_c:]
    Q1 = IM[:n_c, :m_c]
    Q2 = IM[-n_c:, :m_c]
    Q3 = IM[-n_c:, -m_c:]

    if reorient:
        Q1 = np.fliplr(Q1)
        Q3 = np.flipud(Q3)
        Q2 = np.fliplr(np.flipud(Q2))

    # Average symmetrization
    if symmetry_axis==(0, 1):
        Q = (Q0 + Q1 + Q2 + Q3)/4.0
        return Q, Q, Q, Q
    return Q0, Q1, Q2, Q3
