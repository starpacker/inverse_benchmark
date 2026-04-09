import numpy as np
from scipy.fft import dctn, idctn

def p_shrink(X, lmbda=1, p=0, epsilon=0):
    """
    p-shrinkage in 1-D, with mollification.
    
    This function applies a soft-thresholding-like operation. It reduces the 
    magnitude of the input vector X based on the regularization parameter lambda.
    
    Parameters
    ----------
    X : ndarray
        Input data (e.g., the difference between estimated and observed gradients).
        Shape is typically (2, rows, cols) representing x and y components.
    lmbda : float
        Regularization threshold. Higher values force more sparsity (more zeros).
    p : float
        The norm parameter. 
        p=1 results in Soft Thresholding.
        p=0 results in Hard Thresholding.
    epsilon : float
        Smoothing parameter for numerical stability (usually 0).
        
    Returns
    -------
    ndarray
        The "shrunk" version of X.
    """
    mag = np.sqrt(np.sum(X ** 2, axis=0))
    nonzero = mag.copy()
    nonzero[mag == 0.0] = 1.0
    mag = (
        np.maximum(
            mag
            - lmbda ** (2.0 - p) * (nonzero ** 2 + epsilon) ** (p / 2.0 - 0.5),
            0,
        )
        / nonzero
    )
    return mag * X

def forward_operator(F, Dx, Dy):
    """
    Compute gradients of the unwrapped phase using differentiation matrices.
    
    Parameters
    ----------
    F : ndarray
        2D array of unwrapped phase values (shape: rows, columns).
    Dx : sparse matrix
        Differentiation matrix for x-direction (shape: N*M x N*M).
    Dy : sparse matrix
        Differentiation matrix for y-direction (shape: N*M x N*M).
        
    Returns
    -------
    Fx : ndarray
        Gradient of F in x-direction (shape: rows, columns).
    Fy : ndarray
        Gradient of F in y-direction (shape: rows, columns).
    """
    rows, columns = F.shape
    Fx = (Dx @ F.ravel()).reshape(rows, columns)
    Fy = (Dy @ F.ravel()).reshape(rows, columns)
    return Fx, Fy

def run_inversion(preprocessed_data, max_iters=500, tol=1.6, lmbda=1.0, p=0, c=1.3, debug=True):
    """
    Perform phase unwrapping using ADMM.
    
    Parameters
    ----------
    preprocessed_data : dict
        Must contain keys: 'f_wrapped', 'phi_x', 'phi_y', 'Dx', 'Dy', 'K', 'rows', 'columns', 'dtype'.
        'K' is the inverse Laplacian kernel in the DCT domain.
    max_iters : int
        Limit on iterations.
    tol : float
        Stop if max(abs(F_new - F_old)) < tol.
    lmbda : float
        Regularization strength.
    p : float
        Norm type (0 or 1 recommended).
    c : float
        Penalty parameter (step size for dual update).
    debug : bool
        Print progress.
        
    Returns
    -------
    F : ndarray
        The recovered unwrapped phase.
    """
    f_wrapped = preprocessed_data['f_wrapped']
    phi_x = preprocessed_data['phi_x']
    phi_y = preprocessed_data['phi_y']
    Dx = preprocessed_data['Dx']
    Dy = preprocessed_data['Dy']
    K = preprocessed_data['K']
    rows = preprocessed_data['rows']
    columns = preprocessed_data['columns']
    dtype = preprocessed_data['dtype']

    Lambda_x = np.zeros_like(phi_x, dtype=dtype)
    Lambda_y = np.zeros_like(phi_y, dtype=dtype)
    w_x = np.zeros_like(phi_x, dtype=dtype)
    w_y = np.zeros_like(phi_y, dtype=dtype)
    F_old = np.zeros_like(f_wrapped)
    F = np.zeros_like(f_wrapped)

    print(f"Starting ADMM optimization (max_iters={max_iters})...")

    for iteration in range(max_iters):
        rx = w_x.ravel() + phi_x.ravel() - Lambda_x.ravel()
        ry = w_y.ravel() + phi_y.ravel() - Lambda_y.ravel()
        RHS = Dx.T @ rx + Dy.T @ ry
        rho_hat = dctn(RHS.reshape(rows, columns), type=2, norm='ortho', workers=-1)
        F = idctn(rho_hat * K, type=2, norm='ortho', workers=-1)

        Fx, Fy = forward_operator(F, Dx, Dy)
        input_x = Fx - phi_x + Lambda_x
        input_y = Fy - phi_y + Lambda_y
        shrink_result = p_shrink(
            np.stack((input_x, input_y), axis=0), lmbda=lmbda, p=p, epsilon=0
        )
        w_x = shrink_result[0]
        w_y = shrink_result[1]

        Lambda_x += c * (Fx - phi_x - w_x)
        Lambda_y += c * (Fy - phi_y - w_y)

        change = np.max(np.abs(F - F_old))
        
        if debug and iteration % 20 == 0:
            print(f"Iteration:{iteration} change={change}")

        if change < tol or np.isnan(change):
            print(f"Converged at iteration {iteration} with change={change}")
            break
        else:
            F_old = F.copy()

    if iteration == max_iters - 1:
        print(f"Finished max iterations ({max_iters}) with change={change}")

    return F