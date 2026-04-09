import matplotlib

matplotlib.use('Agg')

import torch

def augment_image(img_tensor: torch.Tensor, num_augments: int = 16):
    """Generate augmented versions of a single image."""
    augmented = [img_tensor]
    for _ in range(num_augments - 1):
        x = img_tensor.clone()
        if torch.rand(1).item() > 0.5:
            x = x.flip(-1)
        if torch.rand(1).item() > 0.5:
            x = x.flip(-2)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=(-2, -1))
        x = x + (torch.rand(1, device=x.device) - 0.5) * 0.1
        x = x.clamp(0, 1)
        augmented.append(x)
    return torch.cat(augmented, dim=0)
