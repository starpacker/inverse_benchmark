import abc

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = torch.float32

delta = 1e-11

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

class PoissonLogLikelihood(Likelihood):
    def compute_gradient(self, object: torch.Tensor, subset_idx: int = None) -> torch.Tensor:
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        additive_term_subset = self._get_projection_subset(self.additive_term, subset_idx)
        self.projections_predicted = self.system_matrix.forward(object, subset_idx) + additive_term_subset
        norm_BP = self._get_normBP(subset_idx)
        return self.system_matrix.backward(proj_subset / (self.projections_predicted + delta), subset_idx) - norm_BP

class OSEM:
    def __init__(self, likelihood: PoissonLogLikelihood, object_initial: torch.Tensor = None) -> None:
        self.likelihood = likelihood
        if object_initial is None:
            self.object_prediction = self.likelihood.system_matrix._get_object_initial(device)
        else:
            self.object_prediction = object_initial.to(device).to(dtype)
                
    def _set_n_subsets(self, n_subsets: int):
        self.n_subsets = n_subsets
        self.likelihood._set_n_subsets(n_subsets)

    def _linear_preconditioner_factor(self, n_iter: int, n_subset: int) -> torch.Tensor:
        return 1/(self.likelihood._get_normBP(n_subset) + delta)

    def _compute_preconditioner(self, object: torch.Tensor, n_iter: int, n_subset: int) -> torch.Tensor:
        return object * self._linear_preconditioner_factor(n_iter, n_subset)

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
                self.object_prediction[self.object_prediction<0] = 0 
            print(f"  Iteration {j+1}/{n_iters} completed.")
        return self.object_prediction

def run_inversion(system_matrix, projections_noisy, n_iters=4, n_subsets=8):
    """
    Runs the OSEM reconstruction algorithm.
    """
    print("Initializing OSEM Reconstruction...")
    likelihood = PoissonLogLikelihood(system_matrix, projections_noisy)
    reconstruction_algorithm = OSEM(likelihood)
    
    recon = reconstruction_algorithm(n_iters=n_iters, n_subsets=n_subsets)
    return recon
