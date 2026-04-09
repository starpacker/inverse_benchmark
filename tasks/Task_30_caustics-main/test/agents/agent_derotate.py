from typing import Optional, Union, Tuple, Literal, Annotated

from torch import Tensor

def derotate(vx, vy, phi: Optional[Tensor] = None):
    if phi is None:
        return vx, vy
    c_phi = phi.cos()
    s_phi = phi.sin()
    return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi
