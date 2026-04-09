import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def setup_cameras(n_cameras, image_w, image_h, focal_length_px):
    """
    Set up n_cameras cameras arranged around the measurement volume.
    """
    cameras = []
    cam_distance = 300.0
    azimuth_angles = [30.0, 120.0, 210.0, 300.0]
    elevation_angle = 20.0

    for i in range(n_cameras):
        az = np.radians(azimuth_angles[i])
        el = np.radians(elevation_angle)

        cam_pos = cam_distance * np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])

        look_at = np.array([0.0, 0.0, 0.0])
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        R = np.array([right, cam_up, forward])
        t = -R @ cam_pos

        K = np.array([
            [focal_length_px, 0.0, image_w / 2.0],
            [0.0, focal_length_px, image_h / 2.0],
            [0.0, 0.0, 1.0]
        ])

        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt

        cameras.append({
            'K': K, 'R': R, 't': t, 'P': P,
            'cam_pos': cam_pos, 'id': i
        })

    return cameras

def load_and_preprocess_data(n_particles, vol_min, vol_max, n_cameras,
                              image_w, image_h, focal_length_px, random_seed=42):
    """
    Load and preprocess data for 3D PTV reconstruction.
    
    This function:
    1. Sets up the multi-camera system
    2. Generates ground truth 3D particle positions
    
    Args:
        n_particles: Number of tracer particles
        vol_min: Minimum coordinates of measurement volume [mm]
        vol_max: Maximum coordinates of measurement volume [mm]
        n_cameras: Number of cameras
        image_w: Sensor width [pixels]
        image_h: Sensor height [pixels]
        focal_length_px: Focal length [pixels]
        random_seed: Random seed for reproducibility
    
    Returns:
        particles_gt: (N, 3) array of ground truth 3D positions
        cameras: List of camera dictionaries with calibration data
        config: Dictionary containing configuration parameters
    """
    np.random.seed(random_seed)
    
    # Setup cameras
    cameras = setup_cameras(n_cameras, image_w, image_h, focal_length_px)
    
    # Generate ground truth particles
    particles_gt = np.random.uniform(vol_min, vol_max, size=(n_particles, 3))
    
    config = {
        'n_particles': n_particles,
        'vol_min': vol_min,
        'vol_max': vol_max,
        'n_cameras': n_cameras,
        'image_w': image_w,
        'image_h': image_h,
        'focal_length_px': focal_length_px
    }
    
    return particles_gt, cameras, config
