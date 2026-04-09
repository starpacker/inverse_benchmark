import numpy as np

import torch

import torch.nn.functional as F

from collections.abc import Sequence, Callable

from skimage.data import shepp_logan_phantom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = torch.float32

def compute_pad_size(width: int):
    return int(np.ceil((np.sqrt(2)*width - width)/2))

class ObjectMeta():
    def __init__(self, dr, shape) -> None:
        self.dr = dr
        self.dx = dr[0]
        self.dy = dr[1]
        self.dz = dr[2]
        self.shape = shape

class ProjMeta():
    def __init__(self, angles) -> None:
        self.angles = angles
        self.num_projections = len(angles)

class SPECTObjectMeta(ObjectMeta):
    def __init__(self, dr: list[float], shape: list[int]) -> None:
        super().__init__(dr, shape)
        self.compute_padded_shape()

    def compute_padded_shape(self) -> list:
        self.pad_size = compute_pad_size(self.shape[0])
        x_padded = self.shape[0] + 2*self.pad_size
        y_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        self.padded_shape = (int(x_padded), int(y_padded), int(z_padded))

class SPECTProjMeta(ProjMeta):
    def __init__(self, projection_shape: Sequence, dr: list[float], angles: Sequence, radii=None) -> None:
        super().__init__(angles)
        self.angles = torch.tensor(angles).to(dtype).to(device)
        self.dr = dr
        self.radii = radii
        self.shape = (self.num_projections, projection_shape[0], projection_shape[1])
        self.compute_padded_shape()
        
    def compute_padded_shape(self) -> list:
        self.pad_size = compute_pad_size(self.shape[1])
        theta_padded = self.shape[0]
        r_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        self.padded_shape =  (int(theta_padded), int(r_padded), int(z_padded))

def generate_phantom_3d(shape=(128, 128, 128)):
    p2d = shepp_logan_phantom()
    p2d = torch.tensor(p2d, dtype=dtype).unsqueeze(0).unsqueeze(0) 
    p2d = F.interpolate(p2d, size=(shape[0], shape[1]), mode='bilinear').squeeze()
    p3d = p2d.unsqueeze(-1).repeat(1, 1, shape[2])
    p3d = (p3d - p3d.min()) / (p3d.max() - p3d.min())
    return p3d

def load_and_preprocess_data(shape=(128, 128, 64), noise_scale=50.0):
    """
    Generates synthetic phantom data and returns necessary metadata and ground truth.
    """
    print("Generating Synthetic 3D Phantom...")
    gt_object = generate_phantom_3d(shape).to(device)
    
    object_meta = SPECTObjectMeta(dr=[1.0, 1.0, 1.0], shape=list(shape))
    
    angles = np.linspace(0, 360, 64, endpoint=False)
    proj_meta = SPECTProjMeta(projection_shape=(shape[1], shape[2]), dr=[1.0, 1.0], angles=angles)
    
    print(f"Object Shape: {gt_object.shape}")
    print(f"Projections: {len(angles)} angles")
    
    return gt_object, object_meta, proj_meta, noise_scale
