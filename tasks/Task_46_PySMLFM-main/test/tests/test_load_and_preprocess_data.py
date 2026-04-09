import sys
import os
import dill
import torch
import numpy as np
import traceback
import numpy.typing as npt
from enum import Enum, unique
from sklearn.neighbors import NearestNeighbors

# ==================================================================================
# 1. CLASS DEFINITIONS (CRITICAL FOR DILL DESERIALIZATION)
# ==================================================================================
# These classes must be defined exactly as they were when the data was pickled
# so that dill can reconstruct the objects found in the .pkl files.

@unique
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

# ==================================================================================
# 2. IMPORTS FROM SYSTEM UNDER TEST
# ==================================================================================
# We import the actual function to test.
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# ==================================================================================
# 3. TEST SETUP
# ==================================================================================
def run_test():
    data_paths = ['/data/yjh/PySMLFM-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Analyze paths
    outer_path = None
    inner_path = None
    
    for p in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_load_and_preprocess_data_' in p:
            inner_path = p

    if not outer_path:
        print("Error: standard_data_load_and_preprocess_data.pkl not found.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        # Sometimes dill fails if classes aren't in namespace. We defined them above.
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Scenario A: Simple Function Execution
    # The provided data path list only shows the 'standard' file, suggesting
    # this is a direct function call, not a factory pattern in this specific instance.
    # ----------------------------------------------------------------------
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')

        print(f"Executing load_and_preprocess_data with captured arguments...")
        
        # We need to ensure randomness is controlled similarly to the recording time if possible,
        # but the function itself sets seed=42 internally, so results should be deterministic.
        
        actual_output = load_and_preprocess_data(*args, **kwargs)
        
        # ----------------------------------------------------------------------
        # Verification
        # ----------------------------------------------------------------------
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, actual_output)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print("Exception during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()