import os

import time

import numpy as np

from scipy.interpolate import griddata

import matplotlib.image as mpimg

from skimage.transform import resize

import triangle

def load_and_preprocess_data(filename, target_size=64):
    """
    Loads an image, converts it to a mesh-based stiffness distribution (ground truth),
    and computes the necessary Finite Element matrices.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    start = time.time()
    Eimg = mpimg.imread(filename)
    if Eimg.ndim == 3:
        Eimg = Eimg[:, :, 0]  # Take first channel if RGB

    # --- Mesh Generation ---
    # Downsample significantly for speed if needed
    if Eimg.shape[0] > target_size:
        AE = resize(Eimg, (target_size, target_size), anti_aliasing=True)
    else:
        AE = Eimg
        
    A = resize(AE, (np.minimum(AE.shape[0], AE.shape[1]), np.minimum(AE.shape[0], AE.shape[1])))
    area = 'qa0.0005'
    height = 30e-3  # mm
    width = A.shape[1] / A.shape[0] * 30e-3  # mm
    verta = height
    vertb = width
    pts2 = np.array([[0, 0], [verta, 0], [0, verta / 2], [verta, vertb], [verta, vertb / 2], [0, vertb]])
    roi = dict(vertices=pts2)
    mesh_raw = triangle.triangulate(roi, area)

    vertices = mesh_raw['vertices']
    triangles = mesh_raw['triangles']
    N = len(vertices)
    
    # Grid data interpolation to map image pixels to mesh nodes
    x = np.array(vertices[:, 0])
    y = np.array(vertices[:, 1])
    x_new = np.linspace(x.min(), x.max(), N)
    y_new = np.linspace(y.min(), y.max(), N)
    X, Y = np.meshgrid(x_new, y_new, indexing='ij')
    
    step = 10
    x0 = np.arange(0, A.shape[0], step)
    y0 = np.arange(0, A.shape[1], step)
    xi, yi = np.meshgrid(x0, y0)
    
    xi = xi / np.max(xi) * 30e-3
    yi = yi / np.max(yi) * 30e-3
    x1 = np.squeeze(np.reshape(xi, (xi.size, 1)))
    y1 = np.squeeze(np.reshape(yi, (yi.size, 1)))
    A1 = np.squeeze(np.reshape(A[::step, ::step], (A[::step, ::step].size, 1)))
    z = griddata((x1, y1), A1, (X, Y), method='linear')
    
    # Custom GridData logic integrated here
    xmin, xmax = y.min(), y.max() # Note the swap in original code usage
    ymin, ymax = x.min(), x.max()
    binsize = (xmax - xmin) / (N - 1)
    xi_g = np.arange(xmin, xmax + binsize, binsize)
    yi_g = np.arange(ymin, ymax + binsize, binsize)
    xi_g, yi_g = np.meshgrid(xi_g, yi_g)
    
    zvect = np.zeros((N, 1))
    nrow, ncol = xi_g.shape
    for row in range(nrow):
        for col in range(ncol):
            xc = xi_g[row, col]
            yc = yi_g[row, col]
            posx = np.abs(y - xc) # Note swap
            posy = np.abs(x - yc) # Note swap
            ibin = np.logical_and(posx < binsize / 2., posy < binsize / 2.)
            ind = np.where(ibin == True)[0]
            if len(ind) > 0:
                zvect[ind] = z[row, col]

    E_ground_truth = np.ravel(zvect).astype('float32') * 1e+5
    
    # --- FEM Matrices Calculation ---
    n_element = len(triangles)
    Ae = np.zeros(n_element)
    for i in range(n_element):
        Ax = vertices[triangles[i, 0], 0]
        Ay = vertices[triangles[i, 0], 1]
        Bx = vertices[triangles[i, 1], 0]
        By = vertices[triangles[i, 1], 1]
        Cx = vertices[triangles[i, 2], 0]
        Cy = vertices[triangles[i, 2], 1]
        Ae[i] = (1 / 2) * np.abs(Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))

    Be = np.zeros((3, 6, len(Ae)))
    for i in range(len(Ae)):
        y23 = vertices[triangles[i, 1], 1] - vertices[triangles[i, 2], 1]
        y31 = vertices[triangles[i, 2], 1] - vertices[triangles[i, 0], 1]
        y12 = vertices[triangles[i, 0], 1] - vertices[triangles[i, 1], 1]
        x32 = vertices[triangles[i, 2], 0] - vertices[triangles[i, 1], 0]
        x13 = vertices[triangles[i, 0], 0] - vertices[triangles[i, 2], 0]
        x21 = vertices[triangles[i, 1], 0] - vertices[triangles[i, 0], 0]
        Be[:, :, i] = (1 / (2 * Ae[i])) * np.array([
            [y23, 0, y31, 0, y12, 0],
            [0, x32, 0, x13, 0, x21],
            [x32, y23, x13, y31, x21, y12]
        ])

    # Global Stiffness Setup
    v = 0.495 # Poisson
    n_node = len(vertices)
    Tens = np.zeros((2 * n_node, 2 * n_node, n_node)).astype('float32')
    KT = np.zeros((2 * n_node, 2 * n_node))
    
    c11 = (1 - v) / ((1 + v) * (1 - 2 * v))
    c12 = v / ((1 + v) * (1 - 2 * v))
    c66 = 1 / (2 * (1 + v))
    C = np.array([[c11, c12, 0], [c12, c11, 0], [0, 0, c66]])
    
    rho = 1000
    w_freq = 2 * np.pi * 90
    d = rho * w_freq**2
    kt_base = np.array([
        [2 * d, 0, d, 0, d, 0],
        [0, 2 * d, 0, d, 0, d],
        [d, 0, 2 * d, 0, d, 0],
        [0, d, 0, 2 * d, 0, d],
        [d, 0, d, 0, 2 * d, 0],
        [0, d, 0, d, 0, 2 * d]
    ])

    for i in range(n_element):
        kt = -1 * Ae[i] / 12 * kt_base
        Ke = Ae[i] / 3 * np.dot(np.dot((np.transpose(Be[:, :, i])), C), Be[:, :, i])
        nodes = triangles[i, :]
        a, b, c = nodes[0], nodes[1], nodes[2]
        idx = [2*a, 2*a+1, 2*b, 2*b+1, 2*c, 2*c+1]
        for r in range(6):
            for col in range(6):
                Tens[idx[r], idx[col], [a, b, c]] += Ke[r, col]
                KT[idx[r], idx[col]] += kt[r, col]

    # Boundaries for Forward Problem
    # Re-implement boundaries2 logic
    f = 10
    flag = 1
    inn = triangles
    
    # GK calculation for forward solve requires E
    GK = np.squeeze(Tens @ E_ground_truth) / 3 + KT

    height = 30e-3
    tb = height
    width = 30e-3
    bcy = -0.2
    fxy = np.zeros(2 * n_node)
    
    mesh_b = np.zeros((n_element, 3, 4))
    for i in range(n_element):
        for j in range(3):
            mesh_b[i, j, 0] = inn[i, j]
            mesh_b[i, j, 1:3] = vertices[inn[i, j], :]
            
    w_centre = width / 2
    
    for i in range(n_element):
        for j in range(3):
            if mesh_b[i, j, 2] == 0:  # Bottom
                fxy[(inn[i, j]) * 2 + 1] = 0
                GK[(inn[i, j]) * 2 + 1, :] = 0
                GK[(inn[i, j]) * 2 + 1, (inn[i, j]) * 2 + 1] = 1
                if mesh_b[i, j, 1] == w_centre:
                    fxy[(inn[i, j]) * 2] = 0
                    GK[(inn[i, j]) * 2, :] = 0
                    GK[(inn[i, j]) * 2, (inn[i, j]) * 2] = 1
            if mesh_b[i, j, 2] == tb and flag == 1:
                GK[(inn[i, j]) * 2 + 1, :] = 0
                GK[(inn[i, j]) * 2 + 1, (inn[i, j]) * 2 + 1] = 1
                fxy[(inn[i, j]) * 2 + 1] = tb * bcy
    
    # Boundary logic for Tensor (used in Inverse)
    Tens_bc = Tens.copy()
    for i in range(n_element):
        for j in range(3):
            if mesh_b[i, j, 2] == 0:
                Tens_bc[(inn[i, j]) * 2 + 1, :, :] = 0
                Tens_bc[(inn[i, j]) * 2 + 1, (inn[i, j]) * 2 + 1, :] = 1/10000
                if mesh_b[i, j, 1] == w_centre:
                    Tens_bc[(inn[i, j]) * 2, :, :] = 0
                    Tens_bc[(inn[i, j]) * 2, (inn[i, j]) * 2, :] = 1/10000
            if mesh_b[i, j, 2] == tb and flag == 1:
                Tens_bc[(inn[i, j]) * 2 + 1, :, :] = 0
                Tens_bc[(inn[i, j]) * 2 + 1, (inn[i, j]) * 2 + 1, :] = 1/10000
                
    matTens = np.transpose(Tens_bc, (2, 0, 1))

    print(f"Data Loaded. E shape: {E_ground_truth.shape}, Vertices: {vertices.shape}")
    return E_ground_truth * 1e-5, GK, Tens_bc, KT, matTens, triangles, vertices, fxy
