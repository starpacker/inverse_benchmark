import numpy as np

def forward_operator(x, dr=1.0):
    """
    Forward Abel transform using Hansen-Law method.
    Maps Object (x) -> Projection (y_pred)
    """
    # Hansen-Law constants
    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9, -47391.1])

    # State equation integral
    def I_func(n_arr, lam_arr, a_val):
        integral = np.empty((n_arr.size, lam_arr.size))
        ratio = n_arr / (n_arr - 1)
        if a_val == 0:
            integral[:, 0] = -np.log(ratio)
        ra = (n_arr - 1)**a_val
        k0 = int(not a_val)
        
        # Slicing loop for stability
        lam_plus_a = lam_arr + a_val
        if k0 < len(lam_plus_a):
            sub_lam = lam_plus_a[k0:]
            # Vectorized calc for k >= k0
            # k maps to index in integral starting at k0
            # sub_lam matches these columns
            term = ra[:, None] * (1 - ratio[:, None]**sub_lam) / sub_lam
            integral[:, k0:] = term
        return integral

    image = np.atleast_2d(x)
    aim = np.empty_like(image)
    rows, cols = image.shape
    
    # Forward specific setup
    drive = -2 * dr * np.pi * np.copy(image)
    a = 1
    
    n = np.arange(cols - 1, 1, -1)
    
    # Calculate phi
    # phi[i, k] = (n[i] / (n[i]-1)) ** lam[k]
    phi = (n[:, None] / (n[:, None] - 1)) ** lam[None, :]

    gamma0 = I_func(n, lam, a) * h
    
    # B matrices for forward
    B1 = gamma0
    B0 = gamma0 * 0

    # Recursive calculation
    state_x = np.zeros((h.size, rows)) # State variable x
    
    # Iterate from outside in
    for indx, col in enumerate(n - 1):
        # drive indices: col+1 is outer, col is inner
        d_outer = drive[:, col + 1]
        d_inner = drive[:, col]
        
        # Update state: x = phi*x + B0*u[k+1] + B1*u[k]
        # Dimensions: state_x is (H, Rows). phi[indx] is (H,). B is (H,). Drive is (Rows,)
        # Broadcast: (H, 1) * (H, Rows) + (H, 1)*(1, Rows) ...
        term1 = phi[indx][:, None] * state_x
        term2 = B0[indx][:, None] * d_outer[None, :]
        term3 = B1[indx][:, None] * d_inner[None, :]
        
        state_x = term1 + term2 + term3
        aim[:, col] = state_x.sum(axis=0)

    # Boundary handling
    aim[:, 0] = aim[:, 1]
    aim[:, -1] = aim[:, -2]
    
    if rows == 1: 
        aim = aim[0]
        
    return aim
