import torch

from typing import Optional, Union, Tuple, Literal, Annotated

from torch import Tensor

def translate_rotate(x, y, x0, y0, phi: Optional[Tensor] = None):
    xt = x - x0
    yt = y - y0
    if phi is not None:
        c_phi = phi.cos()
        s_phi = phi.sin()
        return xt * c_phi + yt * s_phi, yt * c_phi - xt * s_phi
    return xt, yt

def derotate(vx, vy, phi: Optional[Tensor] = None):
    if phi is None:
        return vx, vy
    c_phi = phi.cos()
    s_phi = phi.sin()
    return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi

def reduced_deflection_angle_sie(x0, y0, q, phi, Rein, x, y, s=0.0):
    q = torch.where(q == 1.0, q - 1e-6, q)
    x, y = translate_rotate(x, y, x0, y0, phi)
    q2_ = q**2
    f = (1 - q2_).sqrt()
    rein_q_sqrt_f_ = Rein * q.sqrt() / f
    psi = (q2_ * (x**2 + s**2) + y**2).sqrt()
    ax = rein_q_sqrt_f_ * (f * x / (psi + s)).atan()
    ay = rein_q_sqrt_f_ * (f * y / (psi + q2_ * s)).atanh()
    return derotate(ax, ay, phi)
