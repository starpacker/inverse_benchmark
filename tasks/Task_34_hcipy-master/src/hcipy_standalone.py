import numpy as np
import scipy.ndimage as ndimage
import scipy.special
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
from functools import reduce
import operator

# --- Minimal hcipy reimplementation ---

def field_dot(f1, f2):
    return np.sum(f1 * f2, axis=0)

def field_kron(f1, f2):
    return np.kron(f1, f2)

class Grid(object):
    def __init__(self, coords, weights=None):
        self.coords = coords
        self.weights = weights
        self._input_grid = None
        
    @property
    def x(self):
        return self.coords[0]
    
    @property
    def y(self):
        return self.coords[1]
        
    @property
    def size(self):
        return self.coords[0].size
        
    @property
    def ndim(self):
        return len(self.coords)
    
    def zeros(self, dtype=None):
        return Field(np.zeros(self.size, dtype=dtype), self)
        
    def ones(self, dtype=None):
        return Field(np.ones(self.size, dtype=dtype), self)
        
    def scaled(self, scale):
        new_coords = [c * scale for c in self.coords]
        # Simplified scaling of weights for 2D
        new_weights = self.weights * (scale**2) if self.weights is not None else None
        return CartesianGrid(new_coords, new_weights)
        
    def shifted(self, shift):
        new_coords = [c + s for c, s in zip(self.coords, shift)]
        return CartesianGrid(new_coords, self.weights)
        
    def subset(self, mask):
        new_coords = [c[mask] for c in self.coords]
        new_weights = self.weights[mask] if self.weights is not None else None
        return Grid(new_coords, new_weights)

class CartesianGrid(Grid):
    def __init__(self, coords, weights=None):
        super().__init__(coords, weights)
        self.is_regular = True # Simplified assumption
        # Attempt to detect delta/dims/zero from coords if regular
        try:
            # Assuming coords are reshaped or flat? 
            # In hcipy coords are flat arrays.
            # We assume 2D square grid for simplicity often
            self.dims = [int(np.sqrt(self.size))] * 2
            self.delta = [coords[0][1] - coords[0][0], coords[1][self.dims[0]] - coords[1][0]] # Approx
        except:
            self.dims = None
            self.delta = None

class Field(np.ndarray):
    def __new__(cls, arr, grid):
        obj = np.asarray(arr).view(cls)
        obj.grid = grid
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)
        
    @property
    def shaped(self):
        if self.grid.dims:
            return self.reshape(self.grid.dims)
        return self

def make_uniform_grid(dims, extent):
    if np.isscalar(dims): dims = [dims, dims]
    if np.isscalar(extent): extent = [extent, extent]
    
    x = np.linspace(-extent[0]/2, extent[0]/2, dims[0], endpoint=False) + extent[0]/(2*dims[0])
    y = np.linspace(-extent[1]/2, extent[1]/2, dims[1], endpoint=False) + extent[1]/(2*dims[1])
    
    xx, yy = np.meshgrid(x, y)
    coords = [xx.ravel(), yy.ravel()]
    weights = (extent[0]/dims[0]) * (extent[1]/dims[1])
    
    grid = CartesianGrid(coords, weights)
    grid.dims = dims
    grid.delta = [extent[0]/dims[0], extent[1]/dims[1]]
    return grid

def make_pupil_grid(dims, diameter):
    return make_uniform_grid(dims, diameter)

def make_focal_grid(q, num_airy, pupil_diameter=None, focal_length=None, wavelength=None):
    # Simplified focal grid creation
    # spatial_resolution = lambda f / D
    # q is pixels per resolution element
    if pupil_diameter is None: pupil_diameter = 1
    if focal_length is None: focal_length = 1
    if wavelength is None: wavelength = 1
    
    f_lambda = focal_length * wavelength
    spatial_resolution = f_lambda / pupil_diameter
    
    delta = spatial_resolution / q
    extent = 2 * num_airy * spatial_resolution
    dim = int(np.ceil(extent / delta))
    
    return make_uniform_grid(dim, extent)

def imshow_field(field, grid=None, **kwargs):
    if grid is None: grid = field.grid
    plt.imshow(field.shaped.real, origin='lower', extent=[grid.coords[0].min(), grid.coords[0].max(), grid.coords[1].min(), grid.coords[1].max()], **kwargs)


class Wavefront(object):
    def __init__(self, electric_field, wavelength=1):
        self.electric_field = electric_field.astype(complex)
        self.wavelength = wavelength
        
    @property
    def grid(self):
        return self.electric_field.grid
        
    @property
    def intensity(self):
        return np.abs(self.electric_field)**2
        
    @property
    def power(self):
        return self.intensity * self.grid.weights
        
    @property
    def total_power(self):
        return np.sum(self.power)
        
    @total_power.setter
    def total_power(self, p):
        self.electric_field *= np.sqrt(p / self.total_power)
        
    @property
    def wavenumber(self):
        return 2 * np.pi / self.wavelength
        
    def copy(self):
        return copy.deepcopy(self)

class OpticalElement(object):
    def forward(self, wavefront):
        return wavefront
    
    def backward(self, wavefront):
        return wavefront
    
    def __call__(self, wavefront):
        return self.forward(wavefront)

class Magnifier(OpticalElement):
    def __init__(self, magnification):
        self.magnification = magnification
        
    def forward(self, wavefront):
        wf = wavefront.copy()
        wf.electric_field.grid = wf.electric_field.grid.scaled(self.magnification)
        return wf
        
    def backward(self, wavefront):
        wf = wavefront.copy()
        wf.electric_field.grid = wf.electric_field.grid.scaled(1/self.magnification)
        return wf


def make_circular_aperture(diameter, center=None):
    if center is None: shift = np.zeros(2)
    else: shift = np.array(center)
    
    def func(grid):
        x = grid.x - shift[0]
        y = grid.y - shift[1]
        f = (x**2 + y**2) <= (diameter / 2)**2
        return Field(f.astype('float'), grid)
    return func

def make_spider(p1, p2, spider_width):
    delta = np.array(p2) - np.array(p1)
    shift = delta / 2 + np.array(p1)
    spider_angle = np.arctan2(delta[1], delta[0])
    spider_length = np.linalg.norm(delta)
    
    def func(grid):
        x = grid.x - shift[0]
        y = grid.y - shift[1]
        x_new = x * np.cos(spider_angle) + y * np.sin(spider_angle)
        y_new = y * np.cos(spider_angle) - x * np.sin(spider_angle)
        
        spider = (np.abs(x_new) <= (spider_length / 2)) * (np.abs(y_new) <= (spider_width / 2))
        return Field(1 - spider.astype('float'), grid)
    return func

def make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio, num_spiders=0, spider_width=0.01):
    central_obscuration_diameter = pupil_diameter * central_obscuration_ratio
    
    def func(grid):
        pupil_outer = make_circular_aperture(pupil_diameter)(grid)
        pupil_inner = make_circular_aperture(central_obscuration_diameter)(grid)
        spiders = 1
        
        spider_angles = np.linspace(0, 2 * np.pi, num_spiders, endpoint=False)
        for angle in spider_angles:
            x = pupil_diameter * np.cos(angle)
            y = pupil_diameter * np.sin(angle)
            spiders *= make_spider((0, 0), (x, y), spider_width)(grid)
            
        return (pupil_outer - pupil_inner) * spiders
    return func


class FraunhoferPropagator(OpticalElement):
    def __init__(self, input_grid, output_grid, focal_length=1):
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.focal_length = focal_length
        
    def forward(self, wavefront):
        # Simplified Fraunhofer propagation using FFT
        field = wavefront.electric_field
        n = int(np.sqrt(field.size))
        f = field.reshape((n, n))
        
        # FFT
        ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))
        
        # Calculate grid parameters for the FFT output
        n_out = ft.shape[0]
        wavelength = wavefront.wavelength
        dx_in = self.input_grid.delta[0]
        
        # dx_out = f * lambda / (N * dx_in)
        dx_out = self.focal_length * wavelength / (n_out * dx_in)
        extent = n_out * dx_out
        
        # Create new grid matching FFT output
        new_grid = make_uniform_grid(n_out, extent)
        
        # Normalize
        norm_factor = 1 / (1j * self.focal_length * wavelength)
        ft = ft * (dx_in**2) * norm_factor
        
        return Wavefront(Field(ft.ravel(), new_grid), wavelength)


class ModeBasis(list):
    def __init__(self, modes, grid=None):
        super().__init__(modes)
        self.grid = grid
        self._transformation_matrix = np.array(modes).T
        
    @property
    def transformation_matrix(self):
        return self._transformation_matrix
        
    def linear_combination(self, coefficients):
        return Field(self.transformation_matrix.dot(coefficients), self.grid)

def disk_harmonic(n, m, D=1, bc='dirichlet', grid=None):
    # Simplified disk harmonic using polar coordinates
    x = grid.x
    y = grid.y
    r = np.sqrt(x**2 + y**2) * 2 / D
    theta = np.arctan2(y, x)
    
    m_negative = m < 0
    m = abs(m)
    
    if bc == 'dirichlet':
        lambda_mn = scipy.special.jn_zeros(m, n)[-1]
    elif bc == 'neumann':
        lambda_mn = scipy.special.jnp_zeros(m, n)[-1]
    
    if m_negative:
        z = scipy.special.jv(m, lambda_mn * r) * np.sin(m * theta)
    else:
        z = scipy.special.jv(m, lambda_mn * r) * np.cos(m * theta)
        
    mask = r <= 1
    z = z * mask
    norm = np.sqrt(np.sum(z[mask]**2)) # Simplified normalization (ignoring dx)
    if norm > 0: z /= norm
    
    return Field(z, grid)

def disk_harmonic_energy(n, m, bc='dirichlet'):
    m = abs(m)
    if bc == 'dirichlet':
        lambda_mn = scipy.special.jn_zeros(m, n)[-1]
    elif bc == 'neumann':
        lambda_mn = scipy.special.jnp_zeros(m, n)[-1]
    return lambda_mn**2

def get_disk_harmonic_orders_sorted(num_modes, bc='dirichlet'):
    orders = [(1, 0)]
    energies = [disk_harmonic_energy(1, 0, bc)]
    results = []
    
    while len(results) < num_modes:
        k = np.argmin(energies)
        order = orders[k]
        
        if order[1] != 0:
            results.append((order[0], -order[1]))
        results.append(order)
        
        del orders[k]
        del energies[k]
        
        new_order = (order[0], order[1] + 1)
        if new_order not in results and new_order not in orders:
            orders.append(new_order)
            energies.append(disk_harmonic_energy(new_order[0], new_order[1], bc))
        new_order = (order[0] + 1, order[1])
        if new_order not in results and new_order not in orders:
            orders.append(new_order)
            energies.append(disk_harmonic_energy(new_order[0], new_order[1], bc))
            
    return results[:num_modes]

def make_disk_harmonic_basis(grid, num_modes, D=1, bc='dirichlet'):
    orders = get_disk_harmonic_orders_sorted(num_modes, bc)
    modes = [disk_harmonic(order[0], order[1], D, bc, grid) for order in orders]
    # Re-normalize properly with grid weights if possible, but simplified is OK
    return ModeBasis(modes, grid)

class DeformableMirror(OpticalElement):
    def __init__(self, influence_functions):
        self.influence_functions = influence_functions
        self.actuators = np.zeros(len(influence_functions))
        self.surface = self.influence_functions.grid.zeros()
        
    def forward(self, wavefront):
        wf = wavefront.copy()
        # Surface is computed on demand or cached. 
        # Here simplified: compute on fly
        self.surface = self.influence_functions.linear_combination(self.actuators)
        
        phase = 2 * wavefront.wavenumber * self.surface
        wf.electric_field *= np.exp(1j * phase)
        return wf
        
    def flatten(self):
        self.actuators[:] = 0


class FresnelPropagator(OpticalElement):
    def __init__(self, input_grid, distance):
        self.input_grid = input_grid
        self.distance = distance
        
    def forward(self, wavefront):
        # Angular Spectrum Method (simplified)
        field = wavefront.electric_field
        n = int(np.sqrt(field.size))
        f = field.reshape((n, n))
        
        ft = np.fft.fft2(f)
        ft = np.fft.fftshift(ft)
        
        wavelength = wavefront.wavelength
        k = 2 * np.pi / wavelength
        
        # Spatial frequencies
        dx = self.input_grid.delta[0]
        fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
        fx, fy = np.meshgrid(fx, fy)
        
        # Transfer function (Paraxial / Fresnel approximation)
        # H(u,v) = exp(1j*k*z) * exp(-1j*pi*lambda*z*(u^2+v^2))
        H = np.exp(1j * k * self.distance) * np.exp(-1j * np.pi * wavelength * self.distance * (fx**2 + fy**2))
        
        ft_prop = ft * H
        
        # Inverse FFT
        ft_prop = np.fft.ifftshift(ft_prop)
        f_prop = np.fft.ifft2(ft_prop)
        
        return Wavefront(Field(f_prop.ravel(), self.input_grid), wavelength)

class MicroLensArray(OpticalElement):
    def __init__(self, input_grid, mla_grid, focal_length):
        self.input_grid = input_grid
        self.mla_grid = mla_grid
        self.focal_length = focal_length
        self.pitch = mla_grid.delta[0] # Assuming square
        
        # Calculate which lenslet each pixel belongs to
        # Simplified: assume input_grid covers mla_grid
        # mla_index maps input pixel to lenslet index
        
        # For simplicity in this standalone, we just generate the phase screen directly
        x = input_grid.x
        y = input_grid.y
        
        # Local coordinates relative to lenslet center
        # We assume regular grid
        # x_local = (x - x0) % pitch - pitch/2
        
        # Better: find closest lenslet center
        # self.mla_index is not strictly needed for the phase, but needed for estimator
        
        # For the phase:
        # We can simulate the phase of an array of lenslets
        # phase = -k * r_local^2 / (2*f)
        
        # Vectorized "modulo" coordinate
        x_local = np.mod(x + self.pitch/2, self.pitch) - self.pitch/2
        y_local = np.mod(y + self.pitch/2, self.pitch) - self.pitch/2
        
        self.phase_map = -(x_local**2 + y_local**2) / (2 * focal_length)
        
        # Pre-calculate indices for estimator
        # For each pixel in input_grid, find the index of the lenslet in mla_grid
        # mla_grid is the centers of the lenslets
        
        # Map x,y to integer indices of lenslets
        # lenslet_i = round((x - center_x) / pitch)
        
        self.mla_index = np.zeros(input_grid.size, dtype=int)
        
        # This is slow, but done once
        # Assume mla_grid is centered at 0
        ix = np.round((x - mla_grid.coords[0].min()) / self.pitch).astype(int)
        iy = np.round((y - mla_grid.coords[1].min()) / self.pitch).astype(int)
        
        # Map 2D index to 1D index of mla_grid
        # mla_grid dims
        nx_mla = mla_grid.dims[0]
        self.mla_index = iy * nx_mla + ix
        
        # Handle out of bounds
        mask = (ix >= 0) & (ix < nx_mla) & (iy >= 0) & (iy < mla_grid.dims[1])
        self.mla_index[~mask] = -1 
        
    def forward(self, wavefront):
        wf = wavefront.copy()
        k = wavefront.wavenumber
        wf.electric_field *= np.exp(1j * k * self.phase_map)
        return wf

class SquareShackHartmannWavefrontSensorOptics(OpticalElement):
    def __init__(self, input_grid, f_number, num_lenslets, pupil_diameter):
        lenslet_diameter = float(pupil_diameter) / num_lenslets
        
        # Make MLA grid
        x = np.arange(num_lenslets) * lenslet_diameter
        x -= x.mean()
        
        xx, yy = np.meshgrid(x, x)
        coords = [xx.ravel(), yy.ravel()]
        mla_grid = CartesianGrid(coords)
        mla_grid.dims = [num_lenslets, num_lenslets]
        mla_grid.delta = [lenslet_diameter, lenslet_diameter]
        
        focal_length = f_number * lenslet_diameter
        self.micro_lens_array = MicroLensArray(input_grid, mla_grid, focal_length)
        self.propagator = FresnelPropagator(input_grid, focal_length)
        
        self.mla_grid = mla_grid
        self.output_grid = input_grid # The detector is at the same grid resolution
        
    def forward(self, wavefront):
        wf = self.micro_lens_array(wavefront)
        wf = self.propagator(wf)
        return wf

class NoiselessDetector(object):
    def __init__(self, grid):
        self.grid = grid
        self.image = grid.zeros()
        
    def integrate(self, wavefront, dt):
        self.image += wavefront.power * dt
        
    def read_out(self):
        res = self.image.copy()
        self.image[:] = 0
        return res

class ShackHartmannWavefrontSensorEstimator(object):
    def __init__(self, mla_grid, mla_index):
        self.mla_grid = mla_grid
        self.mla_index = mla_index
        self.estimation_subapertures = np.unique(mla_index[mla_index >= 0])
        
    def estimate(self, images):
        image = images[0]
        
        # Centroiding
        # sum_x = sum(I * x) / sum(I)
        
        centroids = np.zeros((2, self.mla_grid.size))
        
        # Loop over lenslets (slow but simple for standalone)
        # Or use ndimage.sum
        
        fluxes = ndimage.sum(image, self.mla_index, index=self.estimation_subapertures)
        sum_x = ndimage.sum(image * image.grid.x, self.mla_index, index=self.estimation_subapertures)
        sum_y = ndimage.sum(image * image.grid.y, self.mla_index, index=self.estimation_subapertures)
        
        # Avoid division by zero
        fluxes[fluxes == 0] = 1
        
        cx = sum_x / fluxes
        cy = sum_y / fluxes
        
        # Relative to lenslet centers
        centers_x = self.mla_grid.x[self.estimation_subapertures]
        centers_y = self.mla_grid.y[self.estimation_subapertures]
        
        slopes_x = cx - centers_x
        slopes_y = cy - centers_y
        
        res = np.zeros((2, self.mla_grid.size))
        res[0, self.estimation_subapertures] = slopes_x
        res[1, self.estimation_subapertures] = slopes_y
        
        # Return as Field?
        # The estimator returns a Field on mla_grid
        # Here we return array (2, N_lenslets)
        return Field(res, self.mla_grid)

def inverse_tikhonov(matrix, rcond=1e-3):
    # matrix is (M, N)
    # We want to solve Ax = y => x = A_inv * y
    # Tikhonov: (A^T A + rcond * I)^-1 A^T
    
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_inv = S / (S**2 + rcond**2 * S.max()**2)
    
    return Vt.T.dot(np.diag(S_inv)).dot(U.T)


if __name__ == "__main__":
    print("Initializing AO Simulation...")
    
    # --- 1. System Setup ---
    wavelength_wfs = 0.7e-6
    wavelength_sci = 2.2e-6
    telescope_diameter = 8.0
    
    # Grid
    num_pupil_pixels = 128
    pupil_grid_diameter = telescope_diameter * 1.05
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
    
    # Aperture
    aperture_func = make_circular_aperture(telescope_diameter)
    aperture = aperture_func(pupil_grid)
    
    # --- 2. Wavefront Sensor ---
    # 20x20 lenslets
    num_lenslets = 20
    sh_diameter = 5e-3 # 5mm beam size for WFS
    magnification = sh_diameter / telescope_diameter
    magnifier = Magnifier(magnification)
    
    shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), f_number=50, num_lenslets=num_lenslets, pupil_diameter=sh_diameter)
    camera = NoiselessDetector(pupil_grid.scaled(magnification)) # Detector grid matches input
    
    # Estimator
    # We need to run a reference to get mla_index properly set up if it wasn't perfect
    # But our simplified class does it in init.
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)
    
    # Reference (flat wavefront)
    wf_ref = Wavefront(aperture, wavelength_wfs)
    wf_ref.total_power = 1
    
    # Propagate ref
    wf_wfs_ref = shwfs(magnifier(wf_ref))
    camera.integrate(wf_wfs_ref, 1)
    image_ref = camera.read_out()
    
    # Slopes ref
    slopes_ref = shwfse.estimate([image_ref])
    
    # --- 3. Deformable Mirror ---
    num_modes = 20
    print(f"Generating {num_modes} DM modes...")
    dm_modes = make_disk_harmonic_basis(pupil_grid, num_modes, telescope_diameter)
    dm = DeformableMirror(dm_modes)
    
    # --- 4. Calibration (Interaction Matrix) ---
    print("Calibrating Interaction Matrix...")
    probe_amp = 0.01 * wavelength_wfs
    response_matrix = []
    
    for i in tqdm(range(num_modes)):
        # Push
        dm.flatten()
        dm.actuators[i] = probe_amp
        wf_push = shwfs(magnifier(dm(wf_ref)))
        camera.integrate(wf_push, 1)
        slopes_push = shwfse.estimate([camera.read_out()])
        
        # Pull
        dm.flatten()
        dm.actuators[i] = -probe_amp
        wf_pull = shwfs(magnifier(dm(wf_ref)))
        camera.integrate(wf_pull, 1)
        slopes_pull = shwfse.estimate([camera.read_out()])
        
        slope_response = (slopes_push - slopes_pull) / (2 * probe_amp)
        response_matrix.append(slope_response.ravel())
        
    response_matrix = np.array(response_matrix).T
    
    # Invert
    print("Inverting Matrix...")
    reconstruction_matrix = inverse_tikhonov(response_matrix, rcond=1e-2)
    
    # --- 5. Closed Loop Simulation ---
    print("Starting Closed Loop...")
    
    # Introduce aberration (random DM shape)
    # Using a stronger aberration to make the correction more visible
    atmosphere = DeformableMirror(dm_modes)
    atmosphere.actuators = np.random.randn(num_modes) * 0.2 * wavelength_sci
    
    # Science Propagator (to check PSF)
    # Focal grid
    focal_grid = make_focal_grid(q=4, num_airy=20, pupil_diameter=telescope_diameter, focal_length=100, wavelength=wavelength_sci)
    prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=100)
    
    # Define Science Scene: Binary Star
    # Star 1: On-axis (Guide Star), Flux 1.0
    # Star 2: Off-axis (Companion), Flux 0.1, Separation 6 lambda/D
    
    star_sep_lambda_D = 6
    star_sep_rad = star_sep_lambda_D * (wavelength_sci / telescope_diameter)
    
    # Positions in (x, y) radians
    # Star 1 is at (0, 0)
    # Star 2 is at (star_sep_rad, 0)
    
    def get_science_image(atmosphere_dm, correction_dm):
        # We simulate the image by incoherent sum of PSFs
        
        # 1. Star 1 (On-axis)
        wf1 = Wavefront(aperture, wavelength_sci)
        wf1.total_power = 1.0
        # Propagate through atmosphere and correction
        wf1 = correction_dm(atmosphere_dm(wf1))
        psf1 = prop(wf1).power
        
        # 2. Star 2 (Off-axis)
        # Tilt the input wavefront
        wf2 = Wavefront(aperture, wavelength_sci)
        wf2.total_power = 0.1 # Faint companion
        
        # Add tilt: exp(i * k * (theta_x * x + theta_y * y))
        # theta_x = star_sep_rad
        k = wf2.wavenumber
        tilt_phase = k * (star_sep_rad * wf2.grid.x)
        wf2.electric_field *= np.exp(1j * tilt_phase)
        
        wf2 = correction_dm(atmosphere_dm(wf2))
        psf2 = prop(wf2).power
        
        return psf1 + psf2

    # Initial Image (Aberrated)
    dm.flatten()
    image_aberrated = get_science_image(atmosphere, dm)
    
    # Reference Image (Perfect)
    perfect_dm = DeformableMirror(dm_modes) # Flat
    image_ref = get_science_image(perfect_dm, perfect_dm)
    max_ref = image_ref.max()
    
    # Loop
    gain = 0.6
    dm.flatten() # DM starts flat
    
    psnrs = []
    strehls = []
    
    for i in range(15):
        # 1. Sensing
        # WFS sees Atmosphere + DM ( DM should become -Atmosphere )
        # Guide star is on-axis
        wf_wfs_in = dm(atmosphere(Wavefront(aperture, wavelength_wfs)))
        wf_wfs_out = shwfs(magnifier(wf_wfs_in))
        
        camera.integrate(wf_wfs_out, 1)
        img = camera.read_out()
        
        # 2. Estimation
        slopes = shwfse.estimate([img])
        slopes -= slopes_ref
        
        # 3. Reconstruction
        cmd = reconstruction_matrix.dot(slopes.ravel())
        
        # 4. Control
        dm.actuators -= gain * cmd
        
        # 5. Evaluation
        # Check science image
        image_curr = get_science_image(atmosphere, dm)
        
        mse = np.mean((image_curr - image_ref)**2)
        psnr = 10 * np.log10(max_ref**2 / mse)
        psnrs.append(psnr)
        
        # Strehl Ratio (peak of Guide Star approx)
        strehl = image_curr.max() / max_ref
        strehls.append(strehl)
        
        print(f"Iter {i}: PSNR = {psnr:.2f} dB, Strehl = {strehl:.4f}")

    print("Done.")
    print(f"Final PSNR: {psnrs[-1]:.2f}")
    
    # Save results
    plt.figure(figsize=(12, 4))
    
    # Log scale for better visibility of faint companion
    def log_scale(img):
        return np.log10(img + 1e-10)
        
    vmin = -5
    vmax = 0
    
    plt.subplot(1, 3, 1)
    imshow_field(np.log10(image_aberrated / image_aberrated.max() + 1e-10), vmin=vmin, vmax=vmax, cmap='inferno')
    plt.title("Aberrated Image (Log)")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    imshow_field(np.log10(image_curr / image_curr.max() + 1e-10), vmin=vmin, vmax=vmax, cmap='inferno')
    plt.title("Corrected Image (Log)")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.plot(strehls)
    plt.title("Strehl Ratio Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Strehl Ratio")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("hcipy_standalone_results.png")
    print("Results saved to hcipy_standalone_results.png")

