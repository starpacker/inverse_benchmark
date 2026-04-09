import numpy as np

import scipy.ndimage as ndimage

import scipy.special

import copy

from tqdm import tqdm

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
        self.is_regular = True 
        try:
            self.dims = [int(np.sqrt(self.size))] * 2
            self.delta = [coords[0][1] - coords[0][0], coords[1][self.dims[0]] - coords[1][0]]
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
    if pupil_diameter is None: pupil_diameter = 1
    if focal_length is None: focal_length = 1
    if wavelength is None: wavelength = 1
    
    f_lambda = focal_length * wavelength
    spatial_resolution = f_lambda / pupil_diameter
    
    delta = spatial_resolution / q
    extent = 2 * num_airy * spatial_resolution
    dim = int(np.ceil(extent / delta))
    
    return make_uniform_grid(dim, extent)

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

class FraunhoferPropagator(OpticalElement):
    def __init__(self, input_grid, output_grid, focal_length=1):
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.focal_length = focal_length
        
    def forward(self, wavefront):
        field = wavefront.electric_field
        n = int(np.sqrt(field.size))
        f = field.reshape((n, n))
        
        ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))
        
        n_out = ft.shape[0]
        wavelength = wavefront.wavelength
        dx_in = self.input_grid.delta[0]
        dx_out = self.focal_length * wavelength / (n_out * dx_in)
        extent = n_out * dx_out
        
        new_grid = make_uniform_grid(n_out, extent)
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
    norm = np.sqrt(np.sum(z[mask]**2))
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
    return ModeBasis(modes, grid)

class DeformableMirror(OpticalElement):
    def __init__(self, influence_functions):
        self.influence_functions = influence_functions
        self.actuators = np.zeros(len(influence_functions))
        self.surface = self.influence_functions.grid.zeros()
        
    def forward(self, wavefront):
        wf = wavefront.copy()
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
        field = wavefront.electric_field
        n = int(np.sqrt(field.size))
        f = field.reshape((n, n))
        
        ft = np.fft.fft2(f)
        ft = np.fft.fftshift(ft)
        
        wavelength = wavefront.wavelength
        k = 2 * np.pi / wavelength
        
        dx = self.input_grid.delta[0]
        fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
        fx, fy = np.meshgrid(fx, fy)
        
        H = np.exp(1j * k * self.distance) * np.exp(-1j * np.pi * wavelength * self.distance * (fx**2 + fy**2))
        
        ft_prop = ft * H
        
        ft_prop = np.fft.ifftshift(ft_prop)
        f_prop = np.fft.ifft2(ft_prop)
        
        return Wavefront(Field(f_prop.ravel(), self.input_grid), wavelength)

class MicroLensArray(OpticalElement):
    def __init__(self, input_grid, mla_grid, focal_length):
        self.input_grid = input_grid
        self.mla_grid = mla_grid
        self.focal_length = focal_length
        self.pitch = mla_grid.delta[0]
        
        x = input_grid.x
        y = input_grid.y
        
        x_local = np.mod(x + self.pitch/2, self.pitch) - self.pitch/2
        y_local = np.mod(y + self.pitch/2, self.pitch) - self.pitch/2
        
        self.phase_map = -(x_local**2 + y_local**2) / (2 * focal_length)
        
        ix = np.round((x - mla_grid.coords[0].min()) / self.pitch).astype(int)
        iy = np.round((y - mla_grid.coords[1].min()) / self.pitch).astype(int)
        
        nx_mla = mla_grid.dims[0]
        self.mla_index = iy * nx_mla + ix
        
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
        self.output_grid = input_grid 
        
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
        
        centroids = np.zeros((2, self.mla_grid.size))
        
        fluxes = ndimage.sum(image, self.mla_index, index=self.estimation_subapertures)
        sum_x = ndimage.sum(image * image.grid.x, self.mla_index, index=self.estimation_subapertures)
        sum_y = ndimage.sum(image * image.grid.y, self.mla_index, index=self.estimation_subapertures)
        
        fluxes[fluxes == 0] = 1
        
        cx = sum_x / fluxes
        cy = sum_y / fluxes
        
        centers_x = self.mla_grid.x[self.estimation_subapertures]
        centers_y = self.mla_grid.y[self.estimation_subapertures]
        
        slopes_x = cx - centers_x
        slopes_y = cy - centers_y
        
        res = np.zeros((2, self.mla_grid.size))
        res[0, self.estimation_subapertures] = slopes_x
        res[1, self.estimation_subapertures] = slopes_y
        
        return Field(res, self.mla_grid)

def inverse_tikhonov(matrix, rcond=1e-3):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_inv = S / (S**2 + rcond**2 * S.max()**2)
    return Vt.T.dot(np.diag(S_inv)).dot(U.T)

def load_and_preprocess_data():
    """
    Setup optical system, including grids, WFS, DM, and calibration.
    Returns a dictionary with all components.
    """
    wavelength_wfs = 0.7e-6
    wavelength_sci = 2.2e-6
    telescope_diameter = 8.0
    
    # 1. Grid & Aperture
    num_pupil_pixels = 128
    pupil_grid_diameter = telescope_diameter * 1.05
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
    
    aperture_func = make_circular_aperture(telescope_diameter)
    aperture = aperture_func(pupil_grid)
    
    # 2. Wavefront Sensor
    num_lenslets = 20
    sh_diameter = 5e-3 # 5mm beam
    magnification = sh_diameter / telescope_diameter
    magnifier = Magnifier(magnification)
    
    shwfs = SquareShackHartmannWavefrontSensorOptics(
        pupil_grid.scaled(magnification), 
        f_number=50, 
        num_lenslets=num_lenslets, 
        pupil_diameter=sh_diameter
    )
    camera = NoiselessDetector(pupil_grid.scaled(magnification))
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)
    
    # 3. Deformable Mirror
    num_modes = 20
    dm_modes = make_disk_harmonic_basis(pupil_grid, num_modes, telescope_diameter)
    dm = DeformableMirror(dm_modes)
    atmosphere = DeformableMirror(dm_modes) # For aberrations
    
    # 4. Science Propagator
    focal_grid = make_focal_grid(q=4, num_airy=20, pupil_diameter=telescope_diameter, focal_length=100, wavelength=wavelength_sci)
    prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=100)
    
    # 5. Calibration (Reference Slopes & Interaction Matrix)
    # Reference Slopes
    wf_ref = Wavefront(aperture, wavelength_wfs)
    wf_ref.total_power = 1
    
    wf_wfs_ref = shwfs(magnifier(wf_ref))
    camera.integrate(wf_wfs_ref, 1)
    image_ref = camera.read_out()
    slopes_ref = shwfse.estimate([image_ref])
    
    # Interaction Matrix
    probe_amp = 0.01 * wavelength_wfs
    response_matrix = []
    
    # This loop is technically "preprocessing" or calibration
    for i in tqdm(range(num_modes), desc="Calibrating"):
        dm.flatten()
        dm.actuators[i] = probe_amp
        wf_push = shwfs(magnifier(dm(wf_ref)))
        camera.integrate(wf_push, 1)
        slopes_push = shwfse.estimate([camera.read_out()])
        
        dm.flatten()
        dm.actuators[i] = -probe_amp
        wf_pull = shwfs(magnifier(dm(wf_ref)))
        camera.integrate(wf_pull, 1)
        slopes_pull = shwfse.estimate([camera.read_out()])
        
        slope_response = (slopes_push - slopes_pull) / (2 * probe_amp)
        response_matrix.append(slope_response.ravel())
        
    response_matrix = np.array(response_matrix).T
    reconstruction_matrix = inverse_tikhonov(response_matrix, rcond=1e-2)
    
    dm.flatten()
    
    return {
        'wavelength_wfs': wavelength_wfs,
        'wavelength_sci': wavelength_sci,
        'telescope_diameter': telescope_diameter,
        'aperture': aperture,
        'pupil_grid': pupil_grid,
        'magnifier': magnifier,
        'shwfs': shwfs,
        'camera': camera,
        'shwfse': shwfse,
        'dm': dm,
        'atmosphere': atmosphere,
        'prop': prop,
        'reconstruction_matrix': reconstruction_matrix,
        'slopes_ref': slopes_ref,
        'num_modes': num_modes
    }
