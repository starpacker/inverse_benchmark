from torch import Tensor

def gaussian_quadrature_integrator(F: Tensor, weight: Tensor):
    return (F * weight).sum(axis=-1)
