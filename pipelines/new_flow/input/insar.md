```python
import numpy as np
import os
import argparse
from scipy import sparse as sp
from scipy.fft import dctn, idctn

# ==============================================================================
# Helper Functions from core.py and loading.py
# ==============================================================================

def make_differentiation_matrices(rows, columns, boundary_conditions="neumann", dtype=np.float32):
    """Generate derivative operators as sparse matrices."""
    bc_opts = ["neumann", "periodic", "dirichlet"]
    bc = boundary_conditions.strip().lower()
    if bc not in bc_opts:
        raise ValueError(f"boundary_conditions must be in {bc_opts}")

    # construct derivative with respect to x (axis=1)
    D = sp.diags([-1.0, 1.0], [0, 1], shape=(columns, columns), dtype=dtype).tolil()

    if boundary_conditions.lower() == bc_opts[0]:  # neumann
        D[-1, -1] = 0.0
    elif boundary_conditions.lower() == bc_opts[1]:  # periodic
        D[-1, 0] = 1.0
    else:
        pass

    S = sp.eye(rows, dtype=dtype)
    Dx = sp.kron(S, D, "csr")

    # construct derivative with respect to y (axis=0)
    D = sp.diags([-1.0, 1.0], [0, 1], shape=(rows, rows), dtype=dtype).tolil()

    if boundary_conditions.lower() == bc_opts[0]:
        D[-1, -1] = 0.0
    elif boundary_conditions.lower() == bc_opts[1]:
        D[-1, 0] = 1.0
    else:
        pass

    S = sp.eye(columns, dtype=dtype)
    Dy = sp.kron(D, S, "csr")

    return Dx, Dy

def est_wrapped_gradient(arr, Dx=None, Dy=None, boundary_conditions="neumann", dtype=np.float32):
    """Estimate the wrapped gradient of `arr` using differential operators `Dx, Dy`"""
    rows, columns = arr.shape
    if Dx is None or Dy is None:
        Dx, Dy = make_differentiation_matrices(
            rows, columns, boundary_conditions=boundary_conditions, dtype=dtype
        )

    phi_x = (Dx @ arr.ravel()).reshape((rows, columns))
    phi_y = (Dy @ arr.ravel()).reshape((rows, columns))
    # Make wrapped adjustmend (eq. (2), (3))
    idxs = np.abs(phi_x) > np.pi
    phi_x[idxs] -= 2 * np.pi * np.sign(phi_x[idxs])
    idxs = np.abs(phi_y) > np.pi
    phi_y[idxs] -= 2 * np.pi * np.sign(phi_y[idxs])
    return phi_x, phi_y

def p_shrink(X, lmbda=1, p=0, epsilon=0):
    """p-shrinkage in 1-D, with mollification."""
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

def make_laplace_kernel(rows, columns, dtype='float32'):
    """Generate eigenvalues of diagonalized Laplacian operator"""
    xi_y = (2 - 2 * np.cos(np.pi * np.arange(rows) / rows)).reshape((-1, 1))
    xi_x = (2 - 2 * np.cos(np.pi * np.arange(columns) / columns)).reshape((1, -1))
    eigvals = xi_y + xi_x

    with np.errstate(divide="ignore"):
        K = np.nan_to_num(1 / eigvals, posinf=0, neginf=0)
    return K.astype(dtype)

def load_dem_rsc(filename, lower=False, **kwargs):
    """Loads and parses the .dem.rsc file"""
    RSC_KEY_TYPES = [
        ("width", int),
        ("file_length", int),
        ("x_first", float),
        ("y_first", float),
        ("x_step", float),
        ("y_step", float),
        ("x_unit", str),
        ("y_unit", str),
        ("z_offset", int),
        ("z_scale", int),
        ("projection", str),
    ]
    
    import collections
    output_data = collections.OrderedDict()
    
    rsc_filename = (
        "{}.rsc".format(filename) if not filename.endswith(".rsc") else filename
    )
    with open(rsc_filename, "r") as f:
        for line in f.readlines():
            for field, num_type in RSC_KEY_TYPES:
                if line.startswith(field.upper()):
                    output_data[field] = num_type(line.split()[1])

    if lower:
        output_data = {k.lower(): d for k, d in output_data.items()}
    return output_data

def load_interferogram(filename, dtype=np.complex64, columns=None, rsc_file=None, band=1):
    """Load binary complex interferogram file."""
    data = np.fromfile(filename, dtype)
    if columns is None:
        if rsc_file is None:
            try:
                rsc_file = filename + ".rsc"
                if os.path.exists(rsc_file):
                    rsc_data = load_dem_rsc(rsc_file, lower=True)
                    columns = rsc_data["width"]
            except Exception:
                pass
    
    if columns is None:
        raise ValueError("Could not determine number of columns for interferogram.")

    return data.reshape((-1, columns))


# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================

def load_and_preprocess_data(filename, dtype="float32"):
    """
    Load interferogram and prepare data for unwrapping.
    """
    print(f"Loading interferogram from {filename}...")
    igram = load_interferogram(filename)
    
    mag = np.abs(igram)
    f_wrapped = np.angle(igram)
    
    if dtype is not None:
        f_wrapped = f_wrapped.astype(dtype)
        
    rows, columns = f_wrapped.shape
    boundary_conditions = "neumann"
    
    print(f"Generating differentiation matrices (rows={rows}, cols={columns})...")
    Dx, Dy = make_differentiation_matrices(
        rows, columns, boundary_conditions=boundary_conditions
    )
    
    print("Estimating wrapped gradients...")
    phi_x, phi_y = est_wrapped_gradient(f_wrapped, Dx, Dy, dtype=dtype)
    
    # Pre-compute Laplacian kernel
    K = make_laplace_kernel(rows, columns, dtype=dtype)
    
    preprocessed_data = {
        'f_wrapped': f_wrapped,
        'mag': mag,
        'phi_x': phi_x,
        'phi_y': phi_y,
        'Dx': Dx,
        'Dy': Dy,
        'K': K,
        'rows': rows,
        'columns': columns,
        'dtype': dtype
    }
    
    return preprocessed_data

# ==============================================================================
# 2. Forward Operator (Not strictly applicable here, but placeholders for structure)
# ==============================================================================

def forward_operator(F, Dx, Dy):
    """
    Compute gradients of the unwrapped phase.
    """
    rows, columns = F.shape
    Fx = (Dx @ F.ravel()).reshape(rows, columns)
    Fy = (Dy @ F.ravel()).reshape(rows, columns)
    return Fx, Fy

# ==============================================================================
# 3. Run Inversion (ADMM Unwrapping)
# ==============================================================================

def run_inversion(preprocessed_data, max_iters=500, tol=1.6, lmbda=1.0, p=0, c=1.3, debug=True):
    """
    Perform phase unwrapping using ADMM.
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
    
    # Lagrange multiplier variables
    Lambda_x = np.zeros_like(phi_x, dtype=dtype)
    Lambda_y = np.zeros_like(phi_y, dtype=dtype)

    # aux. variables for ADMM
    w_x = np.zeros_like(phi_x, dtype=dtype)
    w_y = np.zeros_like(phi_y, dtype=dtype)

    F_old = np.zeros_like(f_wrapped)
    F = np.zeros_like(f_wrapped)

    print(f"Starting ADMM optimization (max_iters={max_iters})...")
    
    for iteration in range(max_iters):
        # 1. Update Unwrapped Phase F: solve linear eqn in fourier domain
        # rhs = dx.T @ phi[0].ravel() + dy.T @ phi[1].ravel()
        rx = w_x.ravel() + phi_x.ravel() - Lambda_x.ravel()
        ry = w_y.ravel() + phi_y.ravel() - Lambda_y.ravel()
        RHS = Dx.T * rx + Dy.T * ry
        
        # Use DCT for neumann:
        rho_hat = dctn(RHS.reshape(rows, columns), type=2, norm='ortho', workers=-1)
        F = idctn(rho_hat * K, type=2, norm='ortho', workers=-1)

        # 2. Calculate x, y gradients of new unwrapped phase estimate
        Fx, Fy = forward_operator(F, Dx, Dy)

        # 3. Update w (auxiliary variable) using shrinkage
        input_x = Fx - phi_x + Lambda_x
        input_y = Fy - phi_y + Lambda_y
        w_x, w_y = p_shrink(
            np.stack((input_x, input_y), axis=0), lmbda=lmbda, p=p, epsilon=0
        )

        # 4. Update lagrange multipliers
        Lambda_x += c * (Fx - phi_x - w_x)
        Lambda_y += c * (Fy - phi_y - w_y)

        # Check convergence
        change = np.max(np.abs(F - F_old))
        if debug and iteration % 20 == 0:
            print(f"Iteration:{iteration} change={change}")

        if change < tol or np.isnan(change):
            print(f"Converged at iteration {iteration} with change={change}")
            break
        else:
            F_old = F

    if iteration == max_iters - 1:
        print(f"Finished max iterations ({max_iters}) with change={change}")
        
    return F

# ==============================================================================
# 4. Evaluate Results
# ==============================================================================

def evaluate_results(F, preprocessed_data, outname):
    """
    Save results and print statistics.
    """
    mag = preprocessed_data['mag']
    
    print(f"Evaluation: Unwrapped phase range [{np.min(F)}, {np.max(F)}]")
    
    if outname.endswith(".tif"):
        try:
            import rasterio as rio
            height, width = F.shape
            with rio.open(
                outname,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                dtype=F.dtype,
                count=1,
            ) as dst:
                dst.write(F, 1)
            print(f"Saved result to {outname}")
        except ImportError:
            print("rasterio not found, saving as npy instead")
            np.save(outname + ".npy", F)
            print(f"Saved numpy result to {outname}.npy")
        
    elif outname.endswith(".unw"):
        unw_with_mag = np.hstack((mag, F))
        unw_with_mag.tofile(outname)
        print(f"Saved binary result to {outname}")
    else:
        # Default fallback, just save npy
        np.save(outname + ".npy", F)
        print(f"Saved numpy result to {outname}.npy")

    return np.mean(F)

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    # Hardcoded parameters for the task
    filename = "20150328_20150409.int"
    outname = "result.tif"
    
    # Reduced iterations for pipeline efficiency
    max_iters = 50 
    tol = 0.08
    lmbda = 2.0
    p = 0.01
    c = 1.6
    
    print("Running Insar Unwrapping Pipeline...")
    
    if not os.path.exists(filename):
        print(f"Error: Input file {filename} not found.")
        # Create dummy file for testing if real data missing in sandbox? 
        # No, we expect data to be copied.
    else:
        # Step 1
        preprocessed_data = load_and_preprocess_data(filename)
        
        # Step 2 & 3
        unwrapped_phase = run_inversion(
            preprocessed_data, 
            max_iters=max_iters, 
            tol=tol, 
            lmbda=lmbda, 
            p=p, 
            c=c
        )
        
        # Step 4
        evaluate_results(unwrapped_phase, preprocessed_data, outname)
        
        print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
```