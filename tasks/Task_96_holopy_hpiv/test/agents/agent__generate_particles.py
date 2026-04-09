import numpy as np

import matplotlib

matplotlib.use("Agg")

def _generate_particles(n, nx, ny, dx, zlo, zhi, rlo, rhi, margin, rng):
    """Generate n particles with random positions and radii."""
    pts = []
    sep_xy = 25 * dx
    sep_z = 40e-6
    for _ in range(n * 100):
        if len(pts) >= n:
            break
        x = rng.uniform(margin * dx, (nx - margin) * dx)
        y = rng.uniform(margin * dx, (ny - margin) * dx)
        z = rng.uniform(zlo, zhi)
        r = rng.uniform(rlo, rhi)
        if all(not (abs(x - p[0]) < sep_xy and abs(y - p[1]) < sep_xy and abs(z - p[2]) < sep_z) for p in pts):
            pts.append((x, y, z, r))
    return np.array(pts)
