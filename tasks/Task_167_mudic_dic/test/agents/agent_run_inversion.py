import matplotlib

matplotlib.use('Agg')

import os

import sys

import logging

import numpy as np

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

sys.path.insert(0, REPO_DIR)

import muDIC as dic

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(x, displacement_function, omega, amp):
    """
    Forward operator: Given material coordinates, compute the displacement field
    using the known harmonic bilateral deformation model.
    
    The forward model is:
        u_x(x, y) = amp * sin(omega * x) * sin(omega * y)
        u_y(x, y) = amp * sin(omega * x) * sin(omega * y)
    
    Parameters
    ----------
    x : tuple of ndarray
        (x_coords, y_coords) - material point coordinates in image space
    displacement_function : callable
        The muDIC displacement function (harmonic_bilat)
    omega : float
        Angular frequency in the coordinate system of x
    amp : float
        Amplitude in the coordinate system of x
    
    Returns
    -------
    y_pred : tuple of ndarray
        (u_x, u_y) - predicted displacement components
    """
    x_coords, y_coords = x
    
    # Apply the harmonic bilateral displacement function
    u_x, u_y = displacement_function(x_coords, y_coords, omega=omega, amp=amp)
    
    return (u_x, u_y)

def run_inversion(data, mesh_margin=30, n_elx=8, n_ely=8, deg_n=3, deg_e=3, tol=1e-6):
    """
    Run muDIC Digital Image Correlation inversion to recover displacement and strain fields.
    
    This function:
    1. Sets up a B-spline mesh on the image region of interest
    2. Runs DIC correlation to find displacement field that maps reference to deformed image
    3. Computes strain fields from the recovered displacement
    4. Computes ground truth on the same grid for comparison
    
    Parameters
    ----------
    data : dict
        Output from load_and_preprocess_data containing image stack and deformation params.
    mesh_margin : int
        Border margin to exclude from analysis.
    n_elx, n_ely : int
        Number of elements in x and y directions.
    deg_n, deg_e : int
        B-spline degree for nodes and elements.
    tol : float
        Convergence tolerance for DIC.
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'dic_ux', 'dic_uy': DIC-recovered displacement components
        - 'gt_ux', 'gt_uy': Ground truth displacement components
        - 'eps_xx', 'eps_yy', 'eps_xy': DIC-recovered strain components
        - 'gt_exx', 'gt_eyy', 'gt_exy': Ground truth strain components
        - 'e_coords', 'n_coords': Element coordinates
        - 'fields': muDIC Fields object
        - 'mesh': muDIC Mesh object
        - 'image_stack': Reference to image stack
    """
    image_stack = data['image_stack']
    displacement_function = data['displacement_function']
    image_shape = data['image_shape']
    omega_out = data['omega_out']
    amp_out = data['amp_out']
    omega_super = data['omega_super']
    amp_super = data['amp_super']
    downsample_factor = data['downsample_factor']
    
    # Create mesh with B-spline elements
    mesher = dic.Mesher(deg_n=deg_n, deg_e=deg_e, type="spline")
    
    # Mesh the ROI (leave border margin)
    mesh = mesher.mesh(
        image_stack,
        Xc1=mesh_margin, Xc2=image_shape[1] - mesh_margin,
        Yc1=mesh_margin, Yc2=image_shape[0] - mesh_margin,
        n_elx=n_elx, n_ely=n_ely,
        GUI=False,
    )
    
    # DIC input
    dic_input = dic.DICInput(mesh, image_stack)
    dic_input.tol = tol
    
    # Run DIC analysis (the inversion)
    dic_job = dic.DICAnalysis(dic_input)
    results = dic_job.run()
    
    # Post-process to get displacement and strain fields
    fields = dic.Fields(results, seed=101)
    
    # Extract DIC displacement: shape [elm, component, e, n, frame]
    disp = fields.disp()  # displacement wrt frame-0
    coords = fields.coords()
    
    # Material-point image coordinates at frame 1
    e_coords = coords[0, 1, :, :, 1]  # x (column) in image pixels
    n_coords = coords[0, 0, :, :, 1]  # y (row) in image pixels
    
    # DIC displacement at frame 1
    dic_ux = disp[0, 0, :, :, 1]  # x-component
    dic_uy = disp[0, 1, :, :, 1]  # y-component
    
    # Compute ground truth displacement using forward operator
    xs, ys = dic.utils.image_coordinates(image_stack[0])
    
    # Get full-field ground truth displacement
    u_x_full, u_y_full = forward_operator(
        (xs, ys), 
        displacement_function, 
        omega=omega_out, 
        amp=amp_out
    )
    
    # Extract GT at DIC material points
    gt_ux = dic.utils.extract_points_from_image(u_x_full, np.array([e_coords, n_coords]))
    gt_uy = dic.utils.extract_points_from_image(u_y_full, np.array([e_coords, n_coords]))
    
    # Compute DIC strain fields
    strain = fields.eng_strain()
    # strain shape: [elm, i, j, e, n, frame]
    eps_xx = strain[0, 0, 0, :, :, 1]
    eps_yy = strain[0, 1, 1, :, :, 1]
    eps_xy = strain[0, 0, 1, :, :, 1]
    
    # Compute ground truth strain analytically
    # For harmonic_bilat: u_x = u_y = A*sin(ωx)*sin(ωy)
    # muDIC strain[0,0,0] = e-direction = y-derivative
    # muDIC strain[0,1,1] = n-direction = x-derivative
    x = e_coords
    y = n_coords
    w = omega_out
    A = amp_out
    
    gt_exx = A * w * np.sin(w * x) * np.cos(w * y)   # ∂u/∂y (matches strain[0,0,0])
    gt_eyy = A * w * np.cos(w * x) * np.sin(w * y)   # ∂u/∂x (matches strain[0,1,1])
    gt_exy = 0.5 * (A * w * np.cos(w * x) * np.sin(w * y) +
                   A * w * np.sin(w * x) * np.cos(w * y))  # ∂u_y/∂x + ∂u_x/∂y
    
    result = {
        'dic_ux': dic_ux,
        'dic_uy': dic_uy,
        'gt_ux': gt_ux,
        'gt_uy': gt_uy,
        'eps_xx': eps_xx,
        'eps_yy': eps_yy,
        'eps_xy': eps_xy,
        'gt_exx': gt_exx,
        'gt_eyy': gt_eyy,
        'gt_exy': gt_exy,
        'e_coords': e_coords,
        'n_coords': n_coords,
        'fields': fields,
        'mesh': mesh,
        'image_stack': image_stack,
    }
    
    return result
