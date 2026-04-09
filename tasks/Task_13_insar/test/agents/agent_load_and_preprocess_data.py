import numpy as np
import os
import collections
from scipy import sparse as sp

def make_differentiation_matrices(rows, columns, boundary_conditions="neumann", dtype=np.float32):
    bc_opts = ["neumann", "periodic", "dirichlet"]
    bc = boundary_conditions.strip().lower()
    if bc not in bc_opts:
        raise ValueError(f"boundary_conditions must be in {bc_opts}")

    D_1d_col = sp.diags([-1.0, 1.0], [0, 1], shape=(columns, columns), dtype=dtype).tolil()

    if bc == "neumann":
        D_1d_col[-1, -1] = 0.0
    elif bc == "periodic":
        D_1d_col[-1, 0] = 1.0
        D_1d_col[-1, -1] = -1.0

    S_rows = sp.eye(rows, dtype=dtype)
    Dx = sp.kron(S_rows, D_1d_col, format="csr")

    D_1d_row = sp.diags([-1.0, 1.0], [0, 1], shape=(rows, rows), dtype=dtype).tolil()

    if bc == "neumann":
        D_1d_row[-1, -1] = 0.0
    elif bc == "periodic":
        D_1d_row[-1, 0] = 1.0
        D_1d_row[-1, -1] = -1.0

    S_cols = sp.eye(columns, dtype=dtype)
    Dy = sp.kron(D_1d_row, S_cols, format="csr")

    return Dx, Dy

def est_wrapped_gradient(arr, Dx, Dy, dtype=np.float32):
    rows, columns = arr.shape

    phi_x = (Dx @ arr.ravel()).reshape((rows, columns))
    phi_y = (Dy @ arr.ravel()).reshape((rows, columns))

    idxs_x = np.abs(phi_x) > np.pi
    phi_x[idxs_x] -= 2 * np.pi * np.sign(phi_x[idxs_x])
    
    idxs_y = np.abs(phi_y) > np.pi
    phi_y[idxs_y] -= 2 * np.pi * np.sign(phi_y[idxs_y])

    return phi_x.astype(dtype), phi_y.astype(dtype)

def make_laplace_kernel(rows, columns, dtype='float32'):
    xi_y = (2 - 2 * np.cos(np.pi * np.arange(rows) / rows)).reshape((-1, 1))
    xi_x = (2 - 2 * np.cos(np.pi * np.arange(columns) / columns)).reshape((1, -1))
    
    eigvals = xi_y + xi_x

    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.nan_to_num(1.0 / eigvals, posinf=0.0, neginf=0.0)
        
    return K.astype(dtype)

def load_dem_rsc(filename, lower=False):
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

    output_data = collections.OrderedDict()

    rsc_filename = (
        "{}.rsc".format(filename) if not filename.endswith(".rsc") else filename
    )
    
    if not os.path.exists(rsc_filename):
        raise FileNotFoundError(f"RSC file not found: {rsc_filename}")

    with open(rsc_filename, "r") as f:
        for line in f.readlines():
            parts = line.split()
            if not parts: continue
            
            key = parts[0].upper()
            value = parts[1]
            
            for field, num_type in RSC_KEY_TYPES:
                if key == field.upper():
                    try:
                        output_data[field] = num_type(value)
                    except ValueError:
                        pass

    if lower:
        output_data = {k.lower(): d for k, d in output_data.items()}
    return output_data

def load_interferogram(filename, dtype=np.complex64, columns=None, rsc_file=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Interferogram file not found: {filename}")
        
    data = np.fromfile(filename, dtype)
    
    if columns is None:
        if rsc_file and os.path.exists(rsc_file):
             rsc_data = load_dem_rsc(rsc_file, lower=True)
             columns = rsc_data.get("width")
        else:
            try:
                rsc_path = filename + ".rsc"
                if os.path.exists(rsc_path):
                    rsc_data = load_dem_rsc(rsc_path, lower=True)
                    columns = rsc_data.get("width")
            except Exception:
                pass

    if columns is None:
        raise ValueError("Could not determine number of columns for interferogram. Please provide 'columns' or ensure .rsc file exists.")

    if data.size % columns != 0:
        raise ValueError(f"Data size {data.size} is not divisible by columns {columns}.")
        
    return data.reshape((-1, columns))

def load_and_preprocess_data(filename, dtype="float32"):
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
        rows, columns, boundary_conditions=boundary_conditions, dtype=np.float32
    )

    print("Estimating wrapped gradients...")
    phi_x, phi_y = est_wrapped_gradient(f_wrapped, Dx, Dy, dtype=dtype)

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