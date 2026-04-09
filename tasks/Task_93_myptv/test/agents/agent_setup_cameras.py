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
