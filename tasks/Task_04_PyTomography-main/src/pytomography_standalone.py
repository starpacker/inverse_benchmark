
import os
import sys
import abc
import copy
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
import matplotlib.pyplot as plt
from collections.abc import Sequence, Callable
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Kornia is a dependency of PyTomography, used for rotation
try:
    from kornia.geometry.transform import rotate
except ImportError:
    print("Kornia is required. Please install it via pip install kornia")
    sys.exit(1)

# ==============================================================================
# 1. Configuration & Utils
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
delta = 1e-11

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
        # replicate along back projected dimension (x)
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

# ==============================================================================
# 2. Metadata Classes
# ==============================================================================

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

# ==============================================================================
# 3. Transforms
# ==============================================================================

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
        # Input: [Lx, Ly, Lz], Angle scalar
        # Kornia expects [B, C, H, W]
        # We rotate in XY plane. PyTomography convention seems to be [Lx, Ly, Lz]
        # rotate expects rotation around center.
        return rotate(object.permute(2,0,1).unsqueeze(0), angles, mode=self.mode).squeeze().permute(1,2,0)

    @torch.no_grad()
    def backward(self, object: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return rotate(object.permute(2,0,1).unsqueeze(0), -angles, mode=self.mode).squeeze().permute(1,2,0)

# ==============================================================================
# 4. Projectors (System Matrix)
# ==============================================================================

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

    @abc.abstractmethod
    def forward(self, object: torch.tensor, **kwargs): ...
    @abc.abstractmethod
    def backward(self, proj: torch.tensor, **kwargs): ...

class SPECTSystemMatrix(SystemMatrix):
    def __init__(self, obj2obj_transforms: list[Transform], proj2proj_transforms: list[Transform], object_meta: SPECTObjectMeta, proj_meta: SPECTProjMeta) -> None:
        super(SPECTSystemMatrix, self).__init__(object_meta, proj_meta, obj2obj_transforms, proj2proj_transforms)
        self.rotation_transform = RotationTransform()
        
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

# ==============================================================================
# 5. Likelihoods
# ==============================================================================

class Likelihood:
    def __init__(self, system_matrix: SystemMatrix, projections: torch.Tensor = None, additive_term: torch.Tensor = None) -> None:
        self.system_matrix = system_matrix
        self.projections = projections if projections is not None else torch.tensor([1.]).to(device)
        self.additive_term = additive_term.to(device) if additive_term is not None else torch.zeros(self.projections.shape).to(device)
        self.n_subsets_previous = -1
    
    def _set_n_subsets(self, n_subsets: int)-> None:
        self.n_subsets = n_subsets
        if n_subsets < 2:
            self.norm_BP = self.system_matrix.compute_normalization_factor()
        else:
            self.system_matrix.set_n_subsets(n_subsets)
            if self.n_subsets_previous!=self.n_subsets:
                self.norm_BPs = []
                for k in range(self.n_subsets):
                    self.norm_BPs.append(self.system_matrix.compute_normalization_factor(k))
        self.n_subsets_previous = n_subsets
        
    def _get_projection_subset(self, projections: torch.Tensor, subset_idx: int = None) -> torch.Tensor:
        if subset_idx is None: return projections
        else: return self.system_matrix.get_projection_subset(projections, subset_idx)
        
    def _get_normBP(self, subset_idx: int, return_sum: bool = False):
        if subset_idx is None: return self.norm_BP
        else:
            if return_sum: return torch.stack(self.norm_BPs).sum(axis=0)
            else: return self.norm_BPs[subset_idx].to(device)
    
    def compute_gradient(self, *args, **kwargs): raise NotImplementedError

class PoissonLogLikelihood(Likelihood):
    def compute_gradient(self, object: torch.Tensor, subset_idx: int = None) -> torch.Tensor:
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        additive_term_subset = self._get_projection_subset(self.additive_term, subset_idx)
        self.projections_predicted = self.system_matrix.forward(object, subset_idx) + additive_term_subset
        norm_BP = self._get_normBP(subset_idx)
        return self.system_matrix.backward(proj_subset / (self.projections_predicted + delta), subset_idx) - norm_BP

# ==============================================================================
# 6. Reconstruction Algorithms
# ==============================================================================

class PreconditionedGradientAscentAlgorithm:
    def __init__(self, likelihood: Likelihood, object_initial: torch.Tensor = None) -> None:
        self.likelihood = likelihood
        if object_initial is None:
            self.object_prediction = self.likelihood.system_matrix._get_object_initial(device)
        else:
            self.object_prediction = object_initial.to(device).to(dtype)
                
    def _set_n_subsets(self, n_subsets: int):
        self.n_subsets = n_subsets
        self.likelihood._set_n_subsets(n_subsets)
    
    def _compute_preconditioner(self, object: torch.Tensor, n_iter: int, n_subset: int) -> None:
        raise NotImplementedError

    def __call__(self, n_iters: int, n_subsets: int = 1):
        self.n_iters = n_iters
        self._set_n_subsets(n_subsets)
        print(f"Starting reconstruction: {n_iters} iterations, {n_subsets} subsets")
        for j in range(n_iters):
            for k in range(n_subsets):
                subset_idx = k if n_subsets > 1 else None
                likelihood_gradient = self.likelihood.compute_gradient(self.object_prediction, subset_idx)
                preconditioner = self._compute_preconditioner(self.object_prediction, j, subset_idx)
                self.object_prediction += preconditioner * likelihood_gradient
                self.object_prediction[self.object_prediction<0] = 0 # Non-negativity constraint
            print(f"  Iteration {j+1}/{n_iters} completed.")
        return self.object_prediction 

class LinearPreconditionedGradientAscentAlgorithm(PreconditionedGradientAscentAlgorithm):
    def _linear_preconditioner_factor(self, n_iter: int, n_subset: int):
        raise NotImplementedError
        
    def _compute_preconditioner(self, object: torch.Tensor, n_iter: int, n_subset: int) -> torch.Tensor:
        return object * self._linear_preconditioner_factor(n_iter, n_subset)

class OSEM(LinearPreconditionedGradientAscentAlgorithm):
    def _linear_preconditioner_factor(self, n_iter: int, n_subset: int) -> torch.Tensor:
        return 1/(self.likelihood._get_normBP(n_subset) + delta)

# ==============================================================================
# 7. Main Execution
# ==============================================================================

def generate_phantom_3d(shape=(128, 128, 128)):
    # Generate 2D phantom and stack
    p2d = shepp_logan_phantom()
    # Resize to shape[0]xshape[1]
    p2d = torch.tensor(p2d, dtype=dtype).unsqueeze(0).unsqueeze(0) # B,C,H,W
    p2d = F.interpolate(p2d, size=(shape[0], shape[1]), mode='bilinear').squeeze()
    
    # Replicate along Z
    p3d = p2d.unsqueeze(-1).repeat(1, 1, shape[2])
    # Normalize
    p3d = (p3d - p3d.min()) / (p3d.max() - p3d.min())
    return p3d

if __name__ == "__main__":
    print("=== PyTomography Standalone Demo (SPECT OSEM) ===")
    
    # 1. Setup Data & Geometry
    print("Generating Synthetic 3D Phantom...")
    shape = (128, 128, 64) # Reduced Z for speed
    gt_object = generate_phantom_3d(shape).to(device)
    
    object_meta = SPECTObjectMeta(dr=[1.0, 1.0, 1.0], shape=list(shape))
    
    angles = np.linspace(0, 360, 64, endpoint=False)
    proj_meta = SPECTProjMeta(projection_shape=(shape[1], shape[2]), dr=[1.0, 1.0], angles=angles)
    
    print(f"Object Shape: {gt_object.shape}")
    print(f"Projections: {len(angles)} angles")
    
    # 2. Forward Projection
    print("Initializing System Matrix...")
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta
    )
    
    print("Forward Projecting...")
    projections = system_matrix.forward(gt_object)
    
    # Add Noise (Poisson)
    projections_noisy = torch.poisson(projections * 50) / 50.0 # Scale for realistic counts
    print(f"Projections Shape: {projections.shape}")
    
    # 3. Reconstruction
    print("Initializing OSEM Reconstruction...")
    likelihood = PoissonLogLikelihood(system_matrix, projections_noisy)
    reconstruction_algorithm = OSEM(likelihood)
    
    recon = reconstruction_algorithm(n_iters=4, n_subsets=8)
    
    # 4. Evaluation
    print("\n=== Evaluation ===")
    gt_np = gt_object.cpu().numpy()
    recon_np = recon.cpu().numpy()
    
    # Normalize for metric calculation
    gt_norm = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min())
    recon_norm = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min())
    
    # Slice for metrics (middle slice)
    mid_z = shape[2] // 2
    gt_slice = gt_norm[:, :, mid_z]
    recon_slice = recon_norm[:, :, mid_z]
    
    p = psnr(gt_slice, recon_slice, data_range=1.0)
    s = ssim(gt_slice, recon_slice, data_range=1.0)
    
    print(f"PSNR: {p:.2f} dB")
    print(f"SSIM: {s:.4f}")
    
    # 5. Save Results
    print("\nSaving results to 'reconstruction_results_spect.png'...")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(gt_slice, cmap='gray')
    ax[0].set_title("Ground Truth (Z-slice)")
    ax[1].imshow(projections_noisy.cpu().numpy()[0, :, :].T, cmap='gray') # Show one projection (transposed to look upright)
    ax[1].set_title("Projection (Angle 0)")
    ax[2].imshow(recon_slice, cmap='gray')
    ax[2].set_title(f"OSEM Recon\nPSNR: {p:.2f}")
    
    plt.tight_layout()
    plt.savefig("reconstruction_results_spect.png")
    print("Done.")
