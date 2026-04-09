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
