import numpy as np

def make_si110_potential(nx, ny, pixel_size):
    """
    Create 2D projected potential for Si [110] zone axis.
    Returns V(x,y) in Volts.
    """
    a, b = 3.84, 5.43  # Si [110] unit cell (Å)
    positions = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.25), (0.75, 0.75)]

    Lx = nx * pixel_size
    Ly = ny * pixel_size
    x = np.arange(nx) * pixel_size
    y = np.arange(ny) * pixel_size
    X, Y = np.meshgrid(x, y, indexing='xy')

    V = np.zeros((ny, nx), dtype=np.float64)
    sigma_atom = 0.30  # Gaussian width (Å)
    V0 = 15.0          # peak potential (V)

    for ix in range(int(np.ceil(Lx / a)) + 1):
        for iy in range(int(np.ceil(Ly / b)) + 1):
            for fx, fy in positions:
                ax = ix * a + fx * a
                ay = iy * b + fy * b
                V += V0 * np.exp(-((X - ax)**2 + (Y - ay)**2) / (2 * sigma_atom**2))
    return V
