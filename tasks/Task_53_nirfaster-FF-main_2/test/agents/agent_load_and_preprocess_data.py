import numpy as np

import scipy.sparse as sp

import scipy.spatial as spatial

import os

import copy

def boundary_attenuation(n_incidence, n_transmission=1.0):
    """Calculate the boundary attenuation factor A using Fresnel's law (Robin BC)."""
    n = n_incidence / n_transmission
    R0 = ((n - 1.) ** 2) / ((n + 1.) ** 2)
    theta_c = np.arcsin(1.0 / n)
    cos_theta_c = np.cos(theta_c)
    A = (2.0 / (1.0 - R0) - 1.0 + np.abs(cos_theta_c) ** 3) / (1.0 - np.abs(cos_theta_c) ** 2)
    return A

class StndMesh:
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.bndvtx = None
        self.mua = None
        self.kappa = None
        self.ri = None
        self.mus = None
        self.ksi = None 
        self.c = None 
        self.source = {}
        self.meas = {}
        self.link = None
        self.dimension = 2
        self.vol = {}

    def copy_from(self, other):
        self.nodes = copy.deepcopy(other.nodes)
        self.elements = copy.deepcopy(other.elements)
        self.bndvtx = copy.deepcopy(other.bndvtx)
        self.mua = copy.deepcopy(other.mua)
        self.kappa = copy.deepcopy(other.kappa)
        self.ri = copy.deepcopy(other.ri)
        self.mus = copy.deepcopy(other.mus)
        self.ksi = copy.deepcopy(other.ksi)
        self.c = copy.deepcopy(other.c)
        self.source = copy.deepcopy(other.source)
        self.meas = copy.deepcopy(other.meas)
        self.link = copy.deepcopy(other.link)
        self.dimension = other.dimension
        self.vol = copy.deepcopy(other.vol)

def load_and_preprocess_data(mesh_path, anomaly_center=None, anomaly_radius=None, anomaly_factor=1.1, grid_step=2.0):
    """
    1. Loads the mesh.
    2. Creates a second mesh with an anomaly (if specified) to simulate data.
    3. Generates the interpolation matrix for the grid.
    Returns: (mesh_baseline, mesh_anomaly, grid_info)
    """
    if not os.path.exists(mesh_path + '.node'):
        raise FileNotFoundError(f"Mesh files not found at {mesh_path}")

    # Load Baseline Mesh
    mesh = StndMesh()
    base = os.path.splitext(mesh_path)[0]
    
    # Load Nodes
    node_data = np.genfromtxt(base + '.node', delimiter='\t')
    mesh.bndvtx = node_data[:, 0].astype(int)
    mesh.nodes = node_data[:, 1:]
    mesh.dimension = mesh.nodes.shape[1]
    
    # Load Elements
    elem_data = np.genfromtxt(base + '.elem', delimiter='\t', dtype=int)
    mesh.elements = elem_data - 1 
    
    # Load Params
    try:
        param_data = np.genfromtxt(base + '.param', skip_header=1)
    except:
        param_data = np.genfromtxt(base + '.param')
            
    mesh.mua = param_data[:, 0]
    mesh.kappa = param_data[:, 1]
    mesh.ri = param_data[:, 2]
    
    # Derived params
    mesh.mus = (1.0 / mesh.kappa) / 3.0 - mesh.mua
    c0 = 299792458000.0
    mesh.c = c0 / mesh.ri
    
    A_val = boundary_attenuation(mesh.ri, 1.0)
    mesh.ksi = 1.0 / (2 * A_val)
    
    # Load Sources
    if os.path.isfile(base + '.source'):
        with open(base + '.source', 'r') as f:
            header = f.readline()
            skip = 1 if 'fixed' in header else 0
        src_data = np.genfromtxt(base + '.source', skip_header=skip+1) 
        mesh.source['coord'] = src_data[:, 1:3]
        mesh.source['int_func'] = src_data[:, 4:] 
        mesh.source['int_func'][:, 0] -= 1
        
    # Load Detectors
    if os.path.isfile(base + '.meas'):
        meas_data = np.genfromtxt(base + '.meas', skip_header=2)
        mesh.meas['coord'] = meas_data[:, 1:3]
        mesh.meas['int_func'] = meas_data[:, 3:]
        mesh.meas['int_func'][:, 0] -= 1
        
    # Load Link
    if os.path.isfile(base + '.link'):
        link_data = np.genfromtxt(base + '.link', skip_header=1)
        mesh.link = link_data
        mesh.link[:, 0:2] -= 1

    # Create Anomaly Mesh
    mesh_anomaly = StndMesh()
    mesh_anomaly.copy_from(mesh)
    
    if anomaly_center is not None:
        dist = np.linalg.norm(mesh.nodes - np.array(anomaly_center), axis=1)
        idx = np.where(dist < anomaly_radius)[0]
        # Change MUA
        mesh_anomaly.mua[idx] *= anomaly_factor
        # Update derived
        mesh_anomaly.kappa[idx] = 1.0 / (3.0 * (mesh_anomaly.mua[idx] + mesh_anomaly.mus[idx]))
        # c and ksi depend on RI, assumed constant here for anomaly

    # Generate Grid Info
    x_min, x_max = np.min(mesh.nodes[:,0]), np.max(mesh.nodes[:,0])
    y_min, y_max = np.min(mesh.nodes[:,1]), np.max(mesh.nodes[:,1])
    # Slight buffer
    xgrid = np.arange(x_min, x_max, grid_step)
    ygrid = np.arange(y_min, y_max, grid_step)
    
    X, Y = np.meshgrid(xgrid, ygrid)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    tri = spatial.Delaunay(mesh.nodes)
    simplex_indices = tri.find_simplex(grid_points)
    valid = simplex_indices != -1
    valid_simplices = simplex_indices[valid]
    
    b = tri.transform[valid_simplices, :2]
    c = tri.transform[valid_simplices, 2]
    coords = grid_points[valid] - c
    bary = np.einsum('ijk,ik->ij', b, coords)
    weights = np.c_[bary, 1 - bary.sum(axis=1)]
    
    nodes_tri = tri.simplices[valid_simplices]
    rows = np.repeat(np.where(valid)[0], 3)
    cols = nodes_tri.flatten()
    vals = weights.flatten()
    
    mesh2grid = sp.coo_matrix((vals, (rows, cols)), shape=(len(grid_points), mesh.nodes.shape[0]))
    
    grid_info = {
        'xgrid': xgrid,
        'ygrid': ygrid,
        'shape': X.shape,
        'mesh2grid': mesh2grid,
        'valid_mask': valid,
        'pixel_area': grid_step * grid_step
    }
    
    # Store grid info in meshes just in case
    mesh.vol = grid_info
    mesh_anomaly.vol = grid_info
    
    return mesh, mesh_anomaly, grid_info
