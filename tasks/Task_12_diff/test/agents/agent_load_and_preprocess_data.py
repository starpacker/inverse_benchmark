import numpy as np
import torch
import diffmetrology as dm
from matplotlib.image import imread

def load_and_preprocess_data(data_path, lens_name):
    """
    Load and preprocess all data required for the inversion.
    
    Args:
        data_path (str): Path to the root data directory.
        lens_name (str): Name of the lens (e.g., 'LE1234-A').
        
    Returns:
        dict: A dictionary containing:
            - 'DM': The initialized DiffMetrology object.
            - 'ps_cap': Captured screen intersections (ground truth).
            - 'valid_cap': Validity mask for the lens area.
            - 'angle': Rotation angle (default 0.0).
            - 'I0': Reconstructed intensity image for visualization.
            - 'device': The compute device (CPU/GPU).
    """
    print("Initialize a DiffMetrology object.")
    device = dm.init()
    
    # Define origin shift (calibration refinement)
    origin_shift = np.array([0.0, 0.0, 0.0])
    
    # Initialize the main simulation engine
    DM = dm.DiffMetrology(
        calibration_path=data_path + '/calibration/',
        rotation_path=data_path + '/rotation_calibration/rotation.mat',
        lut_path=data_path + '/gamma_calibration/gammas.mat',
        origin_shift=origin_shift,
        scale=1.0,
        device=device
    )

    print("Crop the region of interest in the original images.")
    filmsize = np.array([768, 768])
    # Calculate offset to center the crop on a 2048x2048 sensor
    crop_offset = ((2048 - filmsize) / 2).astype(int)
    
    # Apply crop settings to the virtual camera
    for cam in DM.scene.cameras:
        cam.filmsize = filmsize
        cam.crop_offset = torch.Tensor(crop_offset).to(device)

    # Helper function to crop numpy arrays or tensors
    def crop(x):
        return x[..., crop_offset[0]:crop_offset[0] + filmsize[0], 
                     crop_offset[1]:crop_offset[1] + filmsize[1]]

    # Verify setup integrity
    DM.test_setup()

    # --- Read Lens Specifications ---
    print(f"Loading lens file: {lens_name}")
    # Note: Ensure the 'ThorLabs' folder exists relative to execution path
    DM.scene.lensgroup.load_file('ThorLabs/' + lens_name + '.txt')

    print("Ground Truth Lens Parameters:")
    for i in range(len(DM.scene.lensgroup.surfaces)):
        # Print curvature radius (1/c)
        c_val = DM.scene.lensgroup.surfaces[i].c.item()
        if c_val != 0:
            print(f"Lens radius of curvature at surface[{i}]: {1.0 / c_val}")
        else:
            print(f"Lens radius of curvature at surface[{i}]: Infinity")
    
    # Print distance to the second surface
    if len(DM.scene.lensgroup.surfaces) > 1:
        print(f"Distance to surface 1: {DM.scene.lensgroup.surfaces[1].d}")

    angle = 0.0
    Ts = np.array([70, 100, 110])  # Periods of the sinusoids used in PSP
    t = 0 # Index for the starting period

    # --- Load Measurement Data ---
    print("Loading measurement data...")
    data_file = data_path + '/measurement/' + lens_name + '/data_new.npz'
    data = np.load(data_file)
    imgs = data['imgs']
    refs = data['refs']
    
    # Apply cropping
    imgs = crop(imgs)
    refs = crop(refs)
    del data # Free memory

    # --- Solve for Phase/Intersections ---
    print("Solving for intersections...")
    # ps_cap: Screen coordinates (pixels)
    # valid_cap: Binary mask
    # C: Camera centers
    ps_cap, valid_cap, C = DM.solve_for_intersections(imgs, refs, Ts[t:])

    # --- Set Display Pattern (Texture) ---
    # This section loads a specific sinusoid image to use as the texture map
    xs = [0]
    # Note: This path is hardcoded in the provided snippet. 
    # Ensure './camera_acquisitions/...' exists or mock this image.
    sinusoid_path = './camera_acquisitions/images/sinusoids/T=' + str(Ts[t])
    
    try:
        # Attempt to load real texture images
        ims = [np.mean(imread(sinusoid_path + '/' + str(x) + '.png'), axis=-1) for x in xs]
        ims = np.array([im / im.max() for im in ims])
        ims = np.sum(ims, axis=0)
    except (FileNotFoundError, OSError):
        print("Warning: Texture images not found. Using synthetic white texture.")
        ims = np.ones((1080, 1920)) # Assuming standard screen size

    DM.set_texture(ims)
    del ims

    # Set texture shift (hardcoded calibration adjustment for LE1234-A)
    DM.scene.screen.texture_shift = torch.Tensor([0., 1.1106231]).to(device)

    # --- Geometry Alignment ---
    print("Shift `origin` by an estimated value")
    # Estimate the lens mount position based on camera centers C
    origin = DM._compute_mount_geometry(C, verbose=True)
    DM.scene.lensgroup.origin = torch.Tensor(origin).to(device)
    DM.scene.lensgroup.update()
    print(f"Computed Origin: {origin}")

    # --- Visualization Image (I0) ---
    print("Load real images for visualization")
    FR = dm.Fringe()
    # Solve basic fringe parameters (DC, Amplitude, Phase)
    a_cap, b_cap, psi_cap = FR.solve(imgs)
    
    # Reconstruct a clean intensity image
    imgs_sub = np.array([imgs[0, x, ...] for x in xs])
    # Subtract ambient light (a_cap)
    imgs_sub = imgs_sub - a_cap[:, 0, ...] 
    imgs_sub = np.sum(imgs_sub, axis=0)
    
    # Apply validity mask and move to device
    imgs_sub = valid_cap * torch.Tensor(imgs_sub).to(device)
    
    # Normalize for display
    min_val = imgs_sub.min().item()
    max_val = imgs_sub.max().item()
    if max_val - min_val > 1e-6:
        I0 = valid_cap * len(xs) * (imgs_sub - min_val) / (max_val - min_val)
    else:
        I0 = imgs_sub # Avoid division by zero

    preprocessed_data = {
        'DM': DM,
        'ps_cap': ps_cap,
        'valid_cap': valid_cap,
        'angle': angle,
        'I0': I0,
        'device': device
    }
    return preprocessed_data