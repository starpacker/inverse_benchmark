import sys

import abc

import numpy as np

import torch

import torch.nn.functional as F

from collections.abc import Sequence, Callable

try:
    from kornia.geometry.transform import rotate
except ImportError:
    print("Kornia is required. Please install it via pip install kornia")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = torch.float32

def compute_pad_size(width: int):
    return int(np.ceil((np.sqrt(2)*width - width)/2))

def compute_pad_size_padded(width: int):
    a = (np.sqrt(2) - 1)/2
    if width%2==0:
        width_old = int(2*np.floor((width/2)/(1+2*a)))
    else:
        width_old = int(2*np.floor(((width-1)/2)/(1+2*a)))
    return int((width-width_old)/2)

def pad_object(object: torch.Tensor, mode='constant'):
    pad_size = compute_pad_size(object.shape[-2]) 
    if mode=='back_project':
        object = F.pad(object.unsqueeze(0), [0,0,0,0,pad_size,pad_size], mode='replicate').squeeze()
        object = F.pad(object, [0,0,pad_size,pad_size], mode='constant')
        return object
    else:
        return F.pad(object, [0,0,pad_size,pad_size,pad_size,pad_size], mode=mode)

def unpad_object(object: torch.Tensor):
    pad_size = compute_pad_size_padded(object.shape[-2])
    return object[pad_size:-pad_size,pad_size:-pad_size,:]

def pad_proj(proj: torch.Tensor, mode: str = 'constant', value: float = 0):
    pad_size = compute_pad_size(proj.shape[-2])  
    return F.pad(proj, [0,0,pad_size,pad_size], mode=mode, value=value)

def unpad_proj(proj: torch.Tensor):
    pad_size = compute_pad_size_padded(proj.shape[-2])
    return proj[:,pad_size:-pad_size,:]

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

class Transform(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.device = device

    def configure(self, object_meta: ObjectMeta, proj_meta: ProjMeta) -> None:
        self.object_meta = object_meta
        self.proj_meta = proj_meta

    @abc.abstractmethod
    def forward(self, x: torch.tensor): ...
    @abc.abstractmethod
    def backward(self, x: torch.tensor): ...

class RotationTransform(Transform):
    def __init__(self, mode: str = 'bilinear')-> None:
        super(RotationTransform, self).__init__()
        self.mode = mode
                
    @torch.no_grad()
    def forward(self, object: torch.Tensor, angles: torch.Tensor)-> torch.Tensor:
        return rotate(object.permute(2,0,1).unsqueeze(0), angles, mode=self.mode).squeeze().permute(1,2,0)

    @torch.no_grad()
    def backward(self, object: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return rotate(object.permute(2,0,1).unsqueeze(0), -angles, mode=self.mode).squeeze().permute(1,2,0)

class SystemMatrix():
    def __init__(self, object_meta: ObjectMeta, proj_meta: ProjMeta, obj2obj_transforms: list[Transform] = [], proj2proj_transforms: list[Transform] = []) -> None:
        self.obj2obj_transforms = obj2obj_transforms
        self.proj2proj_transforms = proj2proj_transforms
        self.object_meta = object_meta
        self.proj_meta = proj_meta
        self.initialize_transforms()
             
    def initialize_transforms(self):
        for transform in self.obj2obj_transforms:
            transform.configure(self.object_meta, self.proj_meta)
        for transform in self.proj2proj_transforms:
            transform.configure(self.object_meta, self.proj_meta)
            
    def _get_object_initial(self, device=None):
        if device is None: device = device
        return torch.ones(self.object_meta.shape).to(device)

class SPECTSystemMatrix(SystemMatrix):
    def __init__(self, obj2obj_transforms: list[Transform], proj2proj_transforms: list[Transform], object_meta: SPECTObjectMeta, proj_meta: SPECTProjMeta) -> None:
        super(SPECTSystemMatrix, self).__init__(object_meta, proj_meta, obj2obj_transforms, proj2proj_transforms)
        self.rotation_transform = RotationTransform()
        self.subset_indices_array = []
        
    def set_n_subsets(self, n_subsets: int) -> list:
        indices = torch.arange(self.proj_meta.shape[0]).to(torch.long).to(device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def get_projection_subset(self, projections: torch.tensor, subset_idx: int) -> torch.tensor: 
        return projections[...,self.subset_indices_array[subset_idx],:,:]
    
    def get_weighting_subset(self, subset_idx: int) -> float:
        if subset_idx is None: return 1
        else: return len(self.subset_indices_array[subset_idx]) / self.proj_meta.num_projections

    def compute_normalization_factor(self, subset_idx: int = None) -> torch.tensor:
        norm_proj = torch.ones(self.proj_meta.shape).to(device)
        if subset_idx is not None:
            norm_proj = self.get_projection_subset(norm_proj, subset_idx)
        return self.backward(norm_proj, subset_idx)
    
    def forward(self, object: torch.tensor, subset_idx: int = None) -> torch.tensor:
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles).to(device) if subset_idx is None else angle_subset
        
        object = object.to(device)
        proj = torch.zeros((N_angles,*self.proj_meta.padded_shape[1:])).to(device)
        
        for i in range(0, len(angle_indices)):
            angle_indices_i = angle_indices[i]
            object_i = pad_object(object)
            object_i = self.rotation_transform.backward(object_i, 270-self.proj_meta.angles[angle_indices_i])
            for transform in self.obj2obj_transforms:
                object_i = transform.forward(object_i, angle_indices_i)
            proj[i] = object_i.sum(axis=0)
            
        for transform in self.proj2proj_transforms:
            proj = transform.forward(proj)
        return unpad_proj(proj)
    
    def backward(self, proj: torch.tensor, subset_idx: int = None) -> torch.tensor:
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles).to(device) if subset_idx is None else angle_subset
        
        boundary_box_bp = pad_object(torch.ones(self.object_meta.shape).to(device), mode='back_project')
        proj = pad_proj(proj)
        
        for transform in self.proj2proj_transforms[::-1]:
            proj = transform.backward(proj)
            
        object = torch.zeros(self.object_meta.padded_shape).to(device)
        for i in range(0, len(angle_indices)):
            angle_indices_i = angle_indices[i]
            object_i = proj[i].unsqueeze(0) * boundary_box_bp
            for transform in self.obj2obj_transforms[::-1]:
                object_i  = transform.backward(object_i, angle_indices_i)
            object_i = self.rotation_transform.forward(object_i, 270-self.proj_meta.angles[angle_indices_i])
            object += object_i
            
        object = unpad_object(object)
        return object

def forward_operator(x, object_meta, proj_meta, noise_scale=50.0):
    """
    Simulates the forward projection (system matrix application) and adds Poisson noise.
    """
    print("Initializing System Matrix...")
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta
    )
    
    print("Forward Projecting...")
    # x is expected to be on device or moved to device
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype).to(device)
    else:
        x = x.to(device)

    projections = system_matrix.forward(x)
    
    # Add Noise (Poisson)
    projections_noisy = torch.poisson(projections * noise_scale) / noise_scale
    print(f"Projections Shape: {projections.shape}")
    
    return projections_noisy, system_matrix
