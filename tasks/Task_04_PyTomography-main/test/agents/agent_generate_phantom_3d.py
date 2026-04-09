import torch

import torch.nn.functional as F

from skimage.data import shepp_logan_phantom

dtype = torch.float32

def generate_phantom_3d(shape=(128, 128, 128)):
    p2d = shepp_logan_phantom()
    p2d = torch.tensor(p2d, dtype=dtype).unsqueeze(0).unsqueeze(0) 
    p2d = F.interpolate(p2d, size=(shape[0], shape[1]), mode='bilinear').squeeze()
    p3d = p2d.unsqueeze(-1).repeat(1, 1, shape[2])
    p3d = (p3d - p3d.min()) / (p3d.max() - p3d.min())
    return p3d
