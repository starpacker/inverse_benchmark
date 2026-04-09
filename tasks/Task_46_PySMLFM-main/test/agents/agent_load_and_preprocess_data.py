from enum import Enum, unique

from typing import Callable, List, Optional, Tuple

import numpy as np

import numpy.typing as npt

from sklearn.neighbors import NearestNeighbors

class LatticeType(Enum):
    HEXAGONAL = 1
    SQUARE = 2

class MicroLensArray:
    def __init__(self, lattice_type: LatticeType, focal_length: float,
                 lens_pitch: float, optic_size: float, centre: npt.NDArray[float]):
        self.lattice_type = lattice_type
        self.focal_length = focal_length
        self.lens_pitch = lens_pitch
        self.optic_size = optic_size
        self.centre = centre
        self.lens_centres = self._generate_lattice()

    def _generate_lattice(self) -> npt.NDArray[float]:
        if self.lattice_type == LatticeType.SQUARE:
            width = self.optic_size / self.lens_pitch
            marks = np.arange(-np.floor(width / 2), np.ceil(width / 2) + 1)
            x, y = np.meshgrid(marks, marks)
            return np.column_stack((x.flatten('F'), y.flatten('F')))
        elif self.lattice_type == LatticeType.HEXAGONAL:
            width = self.optic_size / self.lens_pitch
            marks = np.arange(-np.floor(width / 2), np.ceil(width / 2) + 1)
            xx, yy = np.meshgrid(marks, marks)
            length = np.max(xx.shape)
            dx = np.tile([0.5, 0], [length, np.ceil(length / 2).astype(int)])
            dx = dx[0:length, 0:length]
            x = yy * np.sqrt(3) / 2
            y = xx + dx.T + 0.5
            return np.column_stack((x.flatten('F'), y.flatten('F')))
        else:
            raise ValueError(f'Unsupported lattice type {self.lattice_type}')

class FourierMicroscope:
    def __init__(self, num_aperture: float, mla_lens_pitch: float, focal_length_mla: float,
                 focal_length_obj_lens: float, focal_length_tube_lens: float,
                 focal_length_fourier_lens: float, pixel_size_camera: float,
                 ref_idx_immersion: float, ref_idx_medium: float):
        self.num_aperture = num_aperture
        self.mla_lens_pitch = mla_lens_pitch
        self.focal_length_mla = focal_length_mla
        self.focal_length_obj_lens = focal_length_obj_lens
        self.focal_length_tube_lens = focal_length_tube_lens
        self.focal_length_fourier_lens = focal_length_fourier_lens
        self.pixel_size_camera = pixel_size_camera
        self.ref_idx_immersion = ref_idx_immersion
        self.ref_idx_medium = ref_idx_medium

        self.bfp_radius = (1000 * num_aperture * focal_length_obj_lens
                           * (focal_length_fourier_lens / focal_length_tube_lens))
        self.bfp_lens_count = 2.0 * self.bfp_radius / mla_lens_pitch
        self.pixels_per_lens = mla_lens_pitch / pixel_size_camera
        self.magnification = ((focal_length_tube_lens / focal_length_obj_lens)
                              * (focal_length_mla / focal_length_fourier_lens))
        self.pixel_size_sample = pixel_size_camera / self.magnification
        self.rho_scaling = self.magnification / self.bfp_radius
        self.xy_to_uv_scale = self.rho_scaling
        self.mla_to_uv_scale = 2.0 / self.bfp_lens_count
        self.mla_to_xy_scale = self.mla_to_uv_scale / self.xy_to_uv_scale

def load_and_preprocess_data(lfm: FourierMicroscope, mla: MicroLensArray, num_points: int = 5) -> Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """
    Generates synthetic 2D localisations and preprocesses them by assigning to lenses.
    Returns: (ground_truth_3d, filtered_locs_2d, original_raw_locs)
    """
    print("Generating synthetic data...")
    # Ground Truth 3D points (x, y, z) in microns
    gt_points_list = []
    np.random.seed(42)
    for _ in range(num_points):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(-3, 3)
        gt_points_list.append([x, y, z])
    gt_points = np.array(gt_points_list)

    locs_list = []
    na = lfm.num_aperture
    n = lfm.ref_idx_medium
    
    # Identify valid lenses
    lens_centres_uv = (mla.lens_centres - mla.centre) * lfm.mla_to_uv_scale
    radius_sq_cutoff = (lfm.bfp_radius / mla.lens_pitch)**2
    dist_sq = np.sum((mla.lens_centres - mla.centre)**2, axis=1)
    valid_lens_mask = dist_sq < radius_sq_cutoff
    valid_lenses_uv = lens_centres_uv[valid_lens_mask]

    # Forward project ground truth points to create synthetic measurements
    for pt in gt_points:
        X, Y, Z = pt
        for uv_lens in valid_lenses_uv:
            u, v = uv_lens
            rho = np.sqrt(u**2 + v**2)
            if rho >= 1.0: continue
            
            dr_sq = 1 - rho * (na/n)**2
            if dr_sq < 0: continue
            phi = -(na / n) / np.sqrt(dr_sq)
            alpha_u = u * phi
            alpha_v = v * phi
            
            u_micron = u / lfm.rho_scaling
            v_micron = v / lfm.rho_scaling
            
            x_obs = X + Z * alpha_u + u_micron
            y_obs = Y + Z * alpha_v + v_micron
            
            noise_x = np.random.normal(0, 0.02)
            noise_y = np.random.normal(0, 0.02)
            
            # Format: frame, x, y, sx, sy, I, bg, prec
            locs_list.append([1, x_obs + noise_x, y_obs + noise_y, 0.5, 0.5, 1000, 10, 0.01])

    locs_2d_csv = np.array(locs_list)
    
    # -- PREPROCESSING --
    # Transform raw CSV into internal structure
    locs_2d = np.zeros((locs_2d_csv.shape[0], 13))
    locs_2d[:, 0] = locs_2d_csv[:, 0].copy() # Frame
    locs_2d[:, 3:10] = locs_2d_csv[:, 1:8].copy() # Data columns

    # Assign to lenses
    xy = locs_2d[:, 3:5].copy()
    xy *= lfm.xy_to_uv_scale
    
    lens_centres = mla.lens_centres - mla.centre
    lens_centres_scaled = lens_centres * lfm.mla_to_uv_scale
    
    knn = NearestNeighbors(n_neighbors=1).fit(lens_centres_scaled)
    lens_indices = knn.kneighbors(xy, return_distance=False)
    
    locs_2d[:, 1:3] = lens_centres_scaled[lens_indices, :][:, 0, :] # U, V
    locs_2d[:, 12] = lens_indices[:, 0]

    # Filter invalid lenses (outside BFP)
    radius = lfm.bfp_radius / mla.lens_pitch
    radius_sq = radius ** 2
    centres = mla.lens_centres - mla.centre
    distance_sq = np.sum(centres ** 2, axis=1)
    valid_lens_indices = np.nonzero(distance_sq < radius_sq)[0]
    
    index_col = locs_2d[:, 12]
    sel = np.any(index_col[:, None] == valid_lens_indices, axis=-1)
    filtered_locs_2d = locs_2d[sel, :]

    # Calculate Alpha (Parallax vector) - Sphere Model
    uv = filtered_locs_2d[:, 1:3]
    alpha_uv = filtered_locs_2d[:, 10:12] # Columns 10 and 11
    
    rho_arr = np.sqrt(np.sum(uv**2, axis=1))
    dr_sq_arr = 1 - rho_arr * (na / n)**2
    # Avoid sqrt of negative
    dr_sq_arr[dr_sq_arr < 0.0] = np.nan
    phi_arr = -(na / n) / np.sqrt(dr_sq_arr)
    alpha_uv[:] = uv * phi_arr[:, np.newaxis]

    return gt_points, filtered_locs_2d, locs_2d_csv
