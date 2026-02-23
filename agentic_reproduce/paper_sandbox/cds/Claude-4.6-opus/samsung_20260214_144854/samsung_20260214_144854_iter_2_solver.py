import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.special import j1
from typing import Optional, Tuple, Dict, List, Union

class InverseSolver:
    """
    Coherent Diffractive Imaging (CDI) phase retrieval solver implementing
    rpIO (Reflective-Prior Input-Output) and rpPIE (Reflective-Prior
    Ptychographic Iterative Engine) algorithms for metrology of copper pad
    topography in hybrid bonding wafer inspection.
    """

    def __init__(
            self,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            detector_size: int = 2048,
            pad_pixels: int = 22,
            pitch_pixels: int = 30,
            array_size: int = 15,
            duty_cycle: float = 0.6,
            h_min: float = 1e-9,
            h_max: float = 3e-9,
            beta: float = 0.9,
            alpha: float = 1e-8,
            n_iterations: int = 300,
            probe_type: str = "airy",
            probe_diameter_pixels: int = 160,
            scan_step_pixels: int = 40,
            noise_sigma_fraction: float = 0.01,
            algorithm: str = "rpPIE",
            random_seed: int = 42,
        ) -> None:
            # Store all hyperparameters as instance attributes
            self.wavelength = wavelength
            self.pixel_size = pixel_size
            self.detector_size = detector_size
            self.pad_pixels = pad_pixels
            self.pitch_pixels = pitch_pixels
            self.array_size = array_size
            self.duty_cycle = duty_cycle
            self.h_min = h_min
            self.h_max = h_max
            self.beta = beta
            self.alpha = alpha
            self.n_iterations = n_iterations
            self.probe_type = probe_type
            self.probe_diameter_pixels = probe_diameter_pixels
            self.scan_step_pixels = scan_step_pixels
            self.noise_sigma_fraction = noise_sigma_fraction
            self.algorithm = algorithm
            self.random_seed = random_seed

            # Initialize RNG with random_seed
            self.rng = np.random.default_rng(random_seed)

            # Precompute derived quantities
            # Object grid size: 15 pads * 30 pixels/pitch = 450 pixels
            self.n_obj = self.array_size * self.pitch_pixels

            # Oversampling ratio
            self.oversampling_ratio = self.detector_size / self.n_obj

            # Phase modulation factor: 4*pi / lambda
            self.phase_factor = 4.0 * np.pi / self.wavelength

            # Offset for centering the object in the zero-padded array
            self.pad_offset = (self.detector_size - self.n_obj) // 2

            # Precompute scan positions for ptychography
            # Number of scan positions in each dimension:
            # N_scan = floor((N_obj - probe_diameter) / step) + 1
            n_scan_1d = max(1, (self.n_obj - self.probe_diameter_pixels) // self.scan_step_pixels + 1)
            self.n_scan_1d = n_scan_1d
            self.n_scan_total = n_scan_1d * n_scan_1d

            # Generate scan positions on a regular grid
            scan_rows = np.arange(n_scan_1d) * self.scan_step_pixels
            scan_cols = np.arange(n_scan_1d) * self.scan_step_pixels
            grid_r, grid_c = np.meshgrid(scan_rows, scan_cols, indexing='ij')
            self.scan_positions = np.stack([grid_r.ravel(), grid_c.ravel()], axis=-1).astype(np.int64)

            # Probe overlap fraction
            if self.probe_diameter_pixels > 0:
                self.overlap_fraction = 1.0 - self.scan_step_pixels / self.probe_diameter_pixels
            else:
                self.overlap_fraction = 0.0

            # Compute material reflectivities (will be set by _compute_reflectivity_tmm)
            self.R_substrate: complex = complex(0)
            self.R_cu: complex = complex(0)

    def _compute_reflectivity_tmm(
            self,
            material: str,
        ) -> complex:
            """
            Compute complex reflectivity for a given material stack configuration
            using the Transfer Matrix Method (TMM) at EUV wavelength.

            Args:
                material: Material identifier, one of "Cu", "substrate".

            Returns:
                Complex reflectivity coefficient.
            """
            # EUV optical constants at lambda = 13.5 nm from Henke/CXRO tables
            # Complex refractive index: n_tilde = 1 - delta + i*beta
            # (delta, beta) values at 13.5 nm (91.8 eV)
            optical_constants = {
                "vacuum": (0.0, 0.0),
                "Cu":   (0.0440, 0.0524),
                "SiCN": (0.0120, 0.00200),
                "SiO2": (0.0122, 0.00188),
                "Si":   (0.0008, 0.00182),
            }

            def get_n(mat_name: str) -> complex:
                delta, beta = optical_constants[mat_name]
                return complex(1.0 - delta, beta)

            # Layer thicknesses for the substrate stack (typical hybrid bonding geometry)
            # Stack from top: vacuum | SiCN (cap, ~10 nm) | SiO2 (dielectric, ~50 nm) | Si (substrate, semi-infinite)
            # For Cu pads: vacuum | Cu (pad region, ~50 nm) | Si (substrate, semi-infinite)

            lam = self.wavelength  # 13.5e-9 m
            # Normal incidence angle
            theta_inc = 0.0  # radians

            def fresnel_r(n1: complex, n2: complex, theta: float) -> complex:
                """Fresnel reflection coefficient at normal incidence (s-pol = p-pol)."""
                # At normal incidence: r = (n1 - n2) / (n1 + n2)
                cos_t1 = np.cos(theta)
                # Snell's law for complex n: n1*sin(t1) = n2*sin(t2)
                sin_t1 = np.sin(theta)
                sin_t2 = (n1 / n2) * sin_t1
                cos_t2 = np.sqrt(1.0 - sin_t2**2 + 0j)
                # Choose correct branch
                if cos_t2.imag < 0:
                    cos_t2 = -cos_t2
                # s-polarization Fresnel coefficient
                r = (n1 * cos_t1 - n2 * cos_t2) / (n1 * cos_t1 + n2 * cos_t2)
                return r

            def transfer_matrix_layer(n_layer: complex, thickness: float, theta_inc_layer: float) -> np.ndarray:
                """Compute the 2x2 transfer matrix for a single layer."""
                # Phase accumulated traversing the layer
                sin_t = np.sin(theta_inc_layer)
                cos_t = np.sqrt(1.0 - sin_t**2 + 0j)
                if cos_t.imag < 0:
                    cos_t = -cos_t
                phi = (2.0 * np.pi / lam) * n_layer * cos_t * thickness
                M = np.array([
                    [np.exp(-1j * phi), 0.0 + 0j],
                    [0.0 + 0j, np.exp(1j * phi)]
                ], dtype=np.complex128)
                return M

            def interface_matrix(n1: complex, n2: complex, theta: float) -> np.ndarray:
                """Compute the 2x2 interface (refraction) matrix."""
                sin_t1 = np.sin(theta)
                sin_t2 = (n1 / n2) * sin_t1
                cos_t1 = np.cos(theta)
                cos_t2 = np.sqrt(1.0 - sin_t2**2 + 0j)
                if cos_t2.imag < 0:
                    cos_t2 = -cos_t2
                r = (n1 * cos_t1 - n2 * cos_t2) / (n1 * cos_t1 + n2 * cos_t2)
                t = (2.0 * n1 * cos_t1) / (n1 * cos_t1 + n2 * cos_t2)
                I_mat = (1.0 / t) * np.array([
                    [1.0 + 0j, r],
                    [r, 1.0 + 0j]
                ], dtype=np.complex128)
                return I_mat

            def compute_reflectivity_tmm_stack(layer_materials: list, layer_thicknesses: list) -> complex:
                """
                Compute complex reflectivity of a multilayer stack using TMM.

                layer_materials: list of material names from top (incident medium) to bottom (substrate).
                                 First element is the incident medium, last is the semi-infinite substrate.
                layer_thicknesses: list of thicknesses for intermediate layers (len = len(materials) - 2).
                """
                n_layers = [get_n(m) for m in layer_materials]
                n_media = len(n_layers)

                # Build the total transfer matrix
                # M_total = I_{0,1} * L_1 * I_{1,2} * L_2 * ... * I_{N-2, N-1}
                # where layers 1..N-2 are the finite-thickness layers
                # layer 0 = incident medium (vacuum), layer N-1 = substrate (semi-infinite)

                # Start with identity
                M_total = np.eye(2, dtype=np.complex128)

                # Current angle in the incident medium
                current_sin = np.sin(theta_inc) * n_layers[0]

                for i in range(n_media - 1):
                    n1 = n_layers[i]
                    n2 = n_layers[i + 1]

                    # Compute angle in medium i
                    sin_ti = current_sin / n1
                    cos_ti = np.sqrt(1.0 - sin_ti**2 + 0j)
                    if cos_ti.imag < 0:
                        cos_ti = -cos_ti
                    theta_i = np.arcsin(np.clip(sin_ti.real, -1, 1)) if sin_ti.imag == 0 else 0.0

                    # Interface matrix from layer i to layer i+1
                    I_mat = interface_matrix(n1, n2, theta_i)
                    M_total = M_total @ I_mat

                    # If this is not the last interface, add propagation through layer i+1
                    if i + 1 < n_media - 1:
                        thickness = layer_thicknesses[i]
                        sin_t_next = current_sin / n2
                        cos_t_next = np.sqrt(1.0 - sin_t_next**2 + 0j)
                        if cos_t_next.imag < 0:
                            cos_t_next = -cos_t_next
                        phi = (2.0 * np.pi / lam) * n2 * cos_t_next * thickness
                        L = np.array([
                            [np.exp(-1j * phi), 0.0 + 0j],
                            [0.0 + 0j, np.exp(1j * phi)]
                        ], dtype=np.complex128)
                        M_total = M_total @ L

                # Reflectivity: r = M_total[1,0] / M_total[0,0]
                r = M_total[1, 0] / M_total[0, 0]
                return complex(r)

            if material == "substrate":
                # Stack: vacuum | SiCN (10 nm) | SiO2 (50 nm) | Si (substrate)
                layer_materials = ["vacuum", "SiCN", "SiO2", "Si"]
                layer_thicknesses = [10e-9, 50e-9]  # SiCN thickness, SiO2 thickness
                return compute_reflectivity_tmm_stack(layer_materials, layer_thicknesses)

            elif material == "Cu":
                # Stack: vacuum | Cu (60 nm, replacing SiCN+SiO2) | Si (substrate)
                layer_materials = ["vacuum", "Cu", "Si"]
                layer_thicknesses = [60e-9]  # Cu thickness
                return compute_reflectivity_tmm_stack(layer_materials, layer_thicknesses)

            else:
                raise ValueError(f"Unknown material configuration: {material}. Use 'Cu' or 'substrate'.")

    def _build_ground_truth_object(
            self,
            recession_depths: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Construct the ground-truth complex object reflection function O(r),
            the support mask S, and the recession depth map.

            Args:
                recession_depths: Optional (15, 15) array of recession depths in meters.
                                  If None, randomly generated from [h_min, h_max].

            Returns:
                Tuple of:
                    - object_true: Complex128 array of shape (N_obj, N_obj).
                    - support_mask: Boolean array of shape (N_obj, N_obj).
                    - recession_map: Float64 array of shape (15, 15) with depths.
            """
            # Compute material reflectivities via TMM if not already computed
            if self.R_substrate == complex(0):
                self.R_substrate = self._compute_reflectivity_tmm("substrate")
            if self.R_cu == complex(0):
                self.R_cu = self._compute_reflectivity_tmm("Cu")

            R_p = self.R_substrate
            R_cu = self.R_cu

            n_obj = self.n_obj  # 450
            pad_px = self.pad_pixels  # 22
            pitch_px = self.pitch_pixels  # 30
            n_array = self.array_size  # 15

            # Generate or use provided recession depths
            if recession_depths is None:
                recession_map = self.rng.uniform(self.h_min, self.h_max, size=(n_array, n_array))
            else:
                recession_map = recession_depths.copy()

            # Initialize object with substrate reflectivity everywhere
            object_true = np.full((n_obj, n_obj), R_p, dtype=np.complex128)

            # Initialize support mask (False everywhere, True inside pads)
            support_mask = np.zeros((n_obj, n_obj), dtype=bool)

            # Place each pad in the array
            for i in range(n_array):
                for j in range(n_array):
                    # Center of pad (i, j) in pixel coordinates
                    center_r = i * pitch_px + pitch_px // 2
                    center_c = j * pitch_px + pitch_px // 2

                    # Pad region: square of pad_px × pad_px centered at (center_r, center_c)
                    half_pad = pad_px // 2
                    r_start = center_r - half_pad
                    r_end = r_start + pad_px
                    c_start = center_c - half_pad
                    c_end = c_start + pad_px

                    # Clip to object bounds (should be within bounds by construction)
                    r_start = max(0, r_start)
                    r_end = min(n_obj, r_end)
                    c_start = max(0, c_start)
                    c_end = min(n_obj, c_end)

                    # Recession depth for this pad
                    h = recession_map[i, j]

                    # Phase modulation due to recession depth: 4*pi*h / lambda
                    phase = self.phase_factor * h

                    # Complex reflectivity of this pad
                    pad_reflectivity = R_cu * np.exp(1j * phase)

                    # Assign to object and support mask
                    object_true[r_start:r_end, c_start:c_end] = pad_reflectivity
                    support_mask[r_start:r_end, c_start:c_end] = True

            return object_true, support_mask, recession_map

    def _build_probe(
            self,
        ) -> np.ndarray:
            """
            Construct the complex-valued illumination probe function P(r).

            Returns:
                Complex128 array of shape (probe_diameter_pixels, probe_diameter_pixels).
            """
            N_p = self.probe_diameter_pixels  # e.g., 160 pixels

            if self.probe_type == "plane":
                # Plane wave probe: uniform amplitude over the probe region
                probe = np.ones((N_p, N_p), dtype=np.complex128)
                # Normalize
                probe /= np.sqrt(np.sum(np.abs(probe)**2))
                return probe

            elif self.probe_type == "airy":
                # Airy spot probe: P ~ [2*J1(pi*r/r0) / (pi*r/r0)]^2
                # r0 is chosen so that the main lobe fills the probe diameter
                # The first zero of J1(x) is at x ≈ 3.8317, so the first zero of
                # the Airy pattern is at pi*r/r0 = 3.8317, i.e., r = 3.8317*r0/pi
                # We want the probe to extend well beyond the first zero to capture
                # the main lobe and some side lobes. Set r0 so that the probe
                # diameter corresponds to several Airy rings.
                # For ~75% overlap with 40-pixel step on 160-pixel probe:
                # r0 is set so the first zero is at roughly N_p/4 pixels from center

                half = N_p / 2.0
                y, x = np.mgrid[0:N_p, 0:N_p]
                y = y.astype(np.float64) - (N_p - 1) / 2.0
                x = x.astype(np.float64) - (N_p - 1) / 2.0
                r = np.sqrt(x**2 + y**2)

                # r0 controls the Airy disk size
                # Set first zero at ~N_p/4 pixels from center
                # First zero of Airy: pi*r/r0 = 3.8317 => r_zero = 3.8317*r0/pi
                # Want r_zero = N_p/4 => r0 = r_zero * pi / 3.8317 = (N_p/4)*pi/3.8317
                r0 = (N_p / 4.0) * np.pi / 3.8317

                # Compute argument
                arg = np.pi * r / r0

                # Airy amplitude: 2*J1(arg)/arg, handle arg=0 (center)
                # At arg=0, 2*J1(arg)/arg -> 1
                with np.errstate(divide='ignore', invalid='ignore'):
                    airy_amplitude = np.where(
                        np.abs(arg) < 1e-12,
                        1.0,
                        2.0 * j1(arg) / arg
                    )

                # The Airy intensity pattern is the square of the amplitude,
                # but for the probe field we use the amplitude (complex-valued probe)
                # Apply a circular aperture to limit the probe extent
                circular_mask = (r <= half).astype(np.float64)

                probe = (airy_amplitude * circular_mask).astype(np.complex128)

                # Normalize probe so that sum(|P|^2) = 1
                norm = np.sqrt(np.sum(np.abs(probe)**2))
                if norm > 0:
                    probe /= norm

                return probe

            elif self.probe_type == "speckle":
                # Structured speckle probe:
                # P(r) = IFFT{ circ(|k|/k_max) * exp(i * phi_rand(k)) }
                # This creates a speckle-like illumination with bounded spatial frequency support

                half = N_p / 2.0
                y, x = np.mgrid[0:N_p, 0:N_p]
                y = y.astype(np.float64) - (N_p - 1) / 2.0
                x = x.astype(np.float64) - (N_p - 1) / 2.0
                r = np.sqrt(x**2 + y**2)

                # k_max defines the circular aperture in Fourier space
                # Set k_max to control the speckle grain size
                # A larger k_max gives finer speckles
                # Use k_max = N_p / 4 to get reasonable speckle structure
                k_max = N_p / 3.0

                # Create circular aperture in Fourier space
                circ = (r <= k_max).astype(np.float64)

                # Random phase screen
                phi_rand = self.rng.uniform(0, 2 * np.pi, size=(N_p, N_p))

                # Fourier-domain field: circ * exp(i * phi_rand)
                fourier_field = circ * np.exp(1j * phi_rand)

                # Inverse FFT to get the probe in real space
                probe = ifft2(ifftshift(fourier_field))

                # Apply a soft circular window to confine the probe spatially
                # Use a super-Gaussian or smooth circular window
                sigma_window = N_p / 3.0
                window = np.exp(-(r / sigma_window)**4)
                probe = probe * window

                probe = probe.astype(np.complex128)

                # Normalize probe so that sum(|P|^2) = 1
                norm = np.sqrt(np.sum(np.abs(probe)**2))
                if norm > 0:
                    probe /= norm

                return probe

            else:
                raise ValueError(
                    f"Unknown probe_type: '{self.probe_type}'. "
                    f"Supported types: 'airy', 'speckle', 'plane'."
                )

    def _generate_scan_positions(
            self,
            object_shape: Tuple[int, int],
            probe_shape: Tuple[int, int],
        ) -> np.ndarray:
            """
            Generate ptychographic scan positions on a regular grid with
            the specified step size ensuring ~75% overlap.

            Args:
                object_shape: Shape of the object array (N_obj, N_obj).
                probe_shape: Shape of the probe array.

            Returns:
                Integer array of shape (N_positions, 2) with (row, col) scan positions.
            """
            n_obj_r, n_obj_c = object_shape
            n_probe_r, n_probe_c = probe_shape
            step = self.scan_step_pixels

            # Compute the number of scan positions in each dimension
            # The probe window starting at position R must satisfy R + probe_size <= N_obj
            # So maximum R = N_obj - probe_size
            # Positions: 0, step, 2*step, ..., up to N_obj - probe_size
            max_r = n_obj_r - n_probe_r
            max_c = n_obj_c - n_probe_c

            if max_r < 0 or max_c < 0:
                # Probe is larger than object; single position at (0, 0)
                return np.array([[0, 0]], dtype=np.int64)

            n_scan_r = max(1, max_r // step + 1)
            n_scan_c = max(1, max_c // step + 1)

            scan_rows = np.arange(n_scan_r) * step
            scan_cols = np.arange(n_scan_c) * step

            # Clamp to ensure probe stays within object bounds
            scan_rows = np.clip(scan_rows, 0, max_r)
            scan_cols = np.clip(scan_cols, 0, max_c)

            # Create meshgrid of all scan positions
            grid_r, grid_c = np.meshgrid(scan_rows, scan_cols, indexing='ij')
            positions = np.stack([grid_r.ravel(), grid_c.ravel()], axis=-1).astype(np.int64)

            # Update instance attributes to reflect actual scan grid
            self.n_scan_1d_r = n_scan_r
            self.n_scan_1d_c = n_scan_c
            self.n_scan_total = len(positions)

            return positions

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            scan_positions: Optional[np.ndarray] = None,
        ) -> Union[np.ndarray, List[np.ndarray]]:
            """
            Compute the forward model: far-field diffraction intensity pattern(s).

            I_j(k) = |FFT2{ P(r - R_j) * O(r) }|^2 + noise

            Args:
                obj: Complex object array of shape (N_obj, N_obj).
                probe: Complex probe array of shape (N_p, N_p).
                scan_positions: Optional array of shape (N_pos, 2) for ptychography.
                                If None, single-shot CDI (probe centered on object).

            Returns:
                For single-shot: Float64 intensity array of shape (detector_size, detector_size).
                For ptychography: List of float64 intensity arrays, one per scan position.
            """
            det_size = self.detector_size
            n_obj_r, n_obj_c = obj.shape
            n_probe_r, n_probe_c = probe.shape

            def _compute_single_intensity(exit_wave_region: np.ndarray) -> np.ndarray:
                """Zero-pad exit wave, FFT, compute intensity, add noise."""
                # Create zero-padded array of detector size
                padded = np.zeros((det_size, det_size), dtype=np.complex128)

                # Determine the region shape
                ew_r, ew_c = exit_wave_region.shape

                # Center the exit wave in the padded array
                offset_r = (det_size - ew_r) // 2
                offset_c = (det_size - ew_c) // 2
                padded[offset_r:offset_r + ew_r, offset_c:offset_c + ew_c] = exit_wave_region

                # Forward FFT with fftshift for centered diffraction pattern
                psi_k = fftshift(fft2(ifftshift(padded)))

                # Compute intensity |FFT|^2
                intensity = np.abs(psi_k) ** 2

                # Add Gaussian noise
                if self.noise_sigma_fraction > 0.0:
                    max_intensity = np.max(intensity)
                    sigma_noise = self.noise_sigma_fraction * max_intensity
                    noise = self.rng.normal(0.0, sigma_noise, size=intensity.shape)
                    intensity = intensity + noise
                    # Clip to non-negative values
                    np.clip(intensity, 0.0, None, out=intensity)

                return intensity.astype(np.float64)

            if scan_positions is None:
                # Single-shot CDI mode: probe is centered on the object
                # or covers the entire object (plane wave case)
                if n_probe_r >= n_obj_r and n_probe_c >= n_obj_c:
                    # Probe is larger than or equal to object: embed object in probe-sized array
                    exit_wave = np.zeros((n_probe_r, n_probe_c), dtype=np.complex128)
                    offset_r = (n_probe_r - n_obj_r) // 2
                    offset_c = (n_probe_c - n_obj_c) // 2
                    exit_wave[offset_r:offset_r + n_obj_r, offset_c:offset_c + n_obj_c] = obj
                    # Multiply by probe (centered)
                    exit_wave = probe * exit_wave
                else:
                    # Probe is smaller than object: center probe on object
                    center_r = n_obj_r // 2
                    center_c = n_obj_c // 2
                    half_pr = n_probe_r // 2
                    half_pc = n_probe_c // 2
                    r_start = center_r - half_pr
                    c_start = center_c - half_pc

                    obj_region = obj[r_start:r_start + n_probe_r, c_start:c_start + n_probe_c]
                    exit_wave = probe * obj_region

                intensity = _compute_single_intensity(exit_wave)
                return intensity

            else:
                # Ptychographic mode: compute intensity for each scan position
                intensities = []
                for idx in range(len(scan_positions)):
                    pos_r, pos_c = int(scan_positions[idx, 0]), int(scan_positions[idx, 1])

                    # Extract the object region illuminated by the probe at this position
                    r_end = min(pos_r + n_probe_r, n_obj_r)
                    c_end = min(pos_c + n_probe_c, n_obj_c)
                    r_start = pos_r
                    c_start = pos_c

                    # Actual sizes (handle edge cases where probe extends beyond object)
                    actual_r = r_end - r_start
                    actual_c = c_end - c_start

                    # Form exit wave: P(r - R_j) * O(r)
                    exit_wave = np.zeros((n_probe_r, n_probe_c), dtype=np.complex128)
                    exit_wave[:actual_r, :actual_c] = (
                        probe[:actual_r, :actual_c] * obj[r_start:r_end, c_start:c_end]
                    )

                    intensity = _compute_single_intensity(exit_wave)
                    intensities.append(intensity)

                return intensities

    def _fourier_modulus_replacement(
            self,
            psi_k: np.ndarray,
            measured_amplitude: np.ndarray,
            epsilon: float = 1e-12,
        ) -> np.ndarray:
            """
            Replace the Fourier modulus of the current estimate with the
            measured amplitude while preserving the estimated phase.

            Psi_c(k) = sqrt(I(k)) * Psi(k) / |Psi(k)|

            Args:
                psi_k: Complex Fourier-domain estimate, shape (N, N).
                measured_amplitude: sqrt(I(k)), real-valued, shape (N, N).
                epsilon: Small constant to avoid division by zero.

            Returns:
                Complex array with replaced modulus, shape (N, N).
            """
            # Compute the modulus of the current Fourier estimate
            modulus = np.abs(psi_k)

            # Replace modulus with measured amplitude, preserving phase
            # Where modulus is near zero, set output to zero to avoid amplifying noise
            psi_c = np.where(
                modulus > epsilon,
                measured_amplitude * (psi_k / modulus),
                0.0 + 0j,
            )

            return psi_c.astype(np.complex128)

    def _apply_support_and_prior(
            self,
            obj_new: np.ndarray,
            obj_current: np.ndarray,
            support_mask: np.ndarray,
            R_p: complex,
        ) -> np.ndarray:
            """
            Apply real-space constraints: accept update inside support,
            enforce reflectivity prior outside support (rpIO/rpPIE).

            For r in S:     O_{n+1}(r) = O'(r)
            For r not in S: O_{n+1}(r) = beta * O_n(r) + (1 - beta) * R_p

            Args:
                obj_new: Fourier-constrained object estimate, shape (N_obj, N_obj).
                obj_current: Current object estimate, shape (N_obj, N_obj).
                support_mask: Boolean mask, True inside pad support.
                R_p: Known complex reflectivity of the substrate.

            Returns:
                Updated object array, shape (N_obj, N_obj).
            """
            result = np.empty_like(obj_new)

            # Inside support: accept the Fourier-constrained estimate
            result[support_mask] = obj_new[support_mask]

            # Outside support: blend current estimate toward the known substrate reflectivity
            outside = ~support_mask
            result[outside] = self.beta * obj_current[outside] + (1.0 - self.beta) * R_p

            return result

    def _apply_hio_constraint(
            self,
            obj_new: np.ndarray,
            obj_current: np.ndarray,
            support_mask: np.ndarray,
        ) -> np.ndarray:
            """
            Apply standard HIO real-space constraint (baseline, no prior).

            For r in S:     O_{n+1}(r) = O'(r)
            For r not in S: O_{n+1}(r) = O_n(r) - beta * O'(r)

            Args:
                obj_new: Fourier-constrained object estimate.
                obj_current: Current object estimate.
                support_mask: Boolean support mask.

            Returns:
                Updated object array.
            """
            result = np.empty_like(obj_new)

            # Inside support: accept the Fourier-constrained estimate
            result[support_mask] = obj_new[support_mask]

            # Outside support: standard HIO feedback
            outside = ~support_mask
            result[outside] = obj_current[outside] - self.beta * obj_new[outside]

            return result

    def _epie_object_update(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            scan_pos: np.ndarray,
            exit_wave_corrected: np.ndarray,
            exit_wave_current: np.ndarray,
        ) -> np.ndarray:
            """
            Perform the ePIE object update step.

            O_{n+1}(r) = O_n(r) + [|P| / |P_max|] * [P* / (|P|^2 + alpha)] * beta * delta_Psi

            Args:
                obj: Current object estimate, shape (N_obj, N_obj).
                probe: Probe function, shape (N_p, N_p).
                scan_pos: Current scan position (row, col).
                exit_wave_corrected: Corrected exit wave Psi_c, shape (N_p, N_p).
                exit_wave_current: Current exit wave Psi, shape (N_p, N_p).

            Returns:
                Updated object array, shape (N_obj, N_obj).
            """
            # Compute the difference between corrected and current exit waves
            delta_psi = exit_wave_corrected - exit_wave_current

            # Compute probe-related quantities
            probe_abs = np.abs(probe)
            P_max = np.max(probe_abs)
            probe_conj = np.conj(probe)
            probe_abs_sq = probe_abs ** 2

            # Compute the ePIE update term:
            # update = (|P| / |P_max|) * (P* / (|P|^2 + alpha)) * beta * delta_Psi
            numerator = (probe_abs / P_max) * probe_conj * self.beta * delta_psi
            denominator = probe_abs_sq + self.alpha
            update = numerator / denominator

            # Extract scan position
            pos_r = int(scan_pos[0])
            pos_c = int(scan_pos[1])
            n_probe_r, n_probe_c = probe.shape
            n_obj_r, n_obj_c = obj.shape

            # Determine the region of the object to update
            r_end = min(pos_r + n_probe_r, n_obj_r)
            c_end = min(pos_c + n_probe_c, n_obj_c)
            actual_r = r_end - pos_r
            actual_c = c_end - pos_c

            # Apply the update to the object at the correct spatial position
            obj_updated = obj.copy()
            obj_updated[pos_r:r_end, pos_c:c_end] += update[:actual_r, :actual_c]

            return obj_updated

    def _resolve_global_phase_ambiguity(
            self,
            obj_reconstructed: np.ndarray,
            obj_true: np.ndarray,
        ) -> np.ndarray:
            """
            Resolve the global phase ambiguity by finding the optimal constant
            phase offset theta* that minimizes ||O_recon * exp(i*theta) - O_true||^2.

            Args:
                obj_reconstructed: Reconstructed object, shape (N_obj, N_obj).
                obj_true: Ground-truth object, shape (N_obj, N_obj).

            Returns:
                Phase-aligned reconstructed object.
            """
            # Compute theta* = arg(sum(conj(O_recon) * O_true))
            cross_correlation = np.sum(np.conj(obj_reconstructed) * obj_true)
            theta_star = np.angle(cross_correlation)

            # Return O_recon * exp(i * theta*)
            aligned = obj_reconstructed * np.exp(1j * theta_star)

            return aligned.astype(np.complex128)

    def _compute_fourier_error(
            self,
            obj: np.ndarray,
            measured_intensity: np.ndarray,
            probe: Optional[np.ndarray] = None,
            scan_pos: Optional[np.ndarray] = None,
        ) -> float:
            """
            Compute the Fourier-domain error metric.

            E = sum_k | |FFT(O)| - sqrt(I(k)) |^2 / sum_k I(k)

            Args:
                obj: Current object estimate.
                measured_intensity: Measured intensity I(k).
                probe: Probe function (for ptychography).
                scan_pos: Scan position (for ptychography).

            Returns:
                Scalar Fourier error value.
            """
            det_size = self.detector_size
            measured_amplitude = np.sqrt(np.maximum(measured_intensity, 0.0))

            if probe is not None and scan_pos is not None:
                # Ptychographic mode: form exit wave at the given scan position
                n_probe_r, n_probe_c = probe.shape
                n_obj_r, n_obj_c = obj.shape
                pos_r = int(scan_pos[0])
                pos_c = int(scan_pos[1])

                r_end = min(pos_r + n_probe_r, n_obj_r)
                c_end = min(pos_c + n_probe_c, n_obj_c)
                actual_r = r_end - pos_r
                actual_c = c_end - pos_c

                exit_wave = np.zeros((n_probe_r, n_probe_c), dtype=np.complex128)
                exit_wave[:actual_r, :actual_c] = (
                    probe[:actual_r, :actual_c] * obj[pos_r:r_end, pos_c:c_end]
                )

                # Zero-pad to detector size
                padded = np.zeros((det_size, det_size), dtype=np.complex128)
                ew_r, ew_c = exit_wave.shape
                offset_r = (det_size - ew_r) // 2
                offset_c = (det_size - ew_c) // 2
                padded[offset_r:offset_r + ew_r, offset_c:offset_c + ew_c] = exit_wave
            else:
                # Single-shot CDI mode: forward propagate the object directly
                n_obj_r, n_obj_c = obj.shape
                padded = np.zeros((det_size, det_size), dtype=np.complex128)
                offset_r = (det_size - n_obj_r) // 2
                offset_c = (det_size - n_obj_c) // 2
                padded[offset_r:offset_r + n_obj_r, offset_c:offset_c + n_obj_c] = obj

            # Forward FFT with proper shifting
            psi_k = fftshift(fft2(ifftshift(padded)))

            # Compute current amplitude
            current_amplitude = np.abs(psi_k)

            # Compute normalized Fourier error
            numerator = np.sum((current_amplitude - measured_amplitude) ** 2)
            denominator = np.sum(measured_intensity)

            if denominator < 1e-30:
                return 0.0

            error = float(numerator / denominator)
            return error

    def _compute_object_error(
            self,
            obj_reconstructed: np.ndarray,
            obj_true: np.ndarray,
        ) -> float:
            """
            Compute the object-domain reconstruction error after resolving
            global phase ambiguity.

            E_obj = ||O_recon - O_true||^2 / ||O_true||^2

            Args:
                obj_reconstructed: Current reconstruction.
                obj_true: Ground-truth object.

            Returns:
                Scalar relative reconstruction error.
            """
            # Resolve global phase ambiguity
            aligned = self._resolve_global_phase_ambiguity(obj_reconstructed, obj_true)

            # Compute normalized squared error
            numerator = np.sum(np.abs(aligned - obj_true) ** 2)
            denominator = np.sum(np.abs(obj_true) ** 2)

            if denominator < 1e-30:
                return 0.0

            return float(numerator / denominator)

    def _compute_phase_error(
            self,
            obj_reconstructed: np.ndarray,
            obj_true: np.ndarray,
            support_mask: np.ndarray,
        ) -> float:
            """
            Compute the mean absolute phase error over the support region.

            epsilon_phi = (1/|S|) * sum_{r in S} |arg(O_recon(r)) - arg(O_true(r))|

            Args:
                obj_reconstructed: Reconstructed object.
                obj_true: Ground-truth object.
                support_mask: Boolean support mask.

            Returns:
                Mean absolute phase error in radians.
            """
            # Resolve global phase ambiguity first
            aligned = self._resolve_global_phase_ambiguity(obj_reconstructed, obj_true)

            # Compute phase of aligned reconstruction and ground truth within support
            phase_recon = np.angle(aligned[support_mask])
            phase_true = np.angle(obj_true[support_mask])

            # Compute phase difference, wrapped to [-pi, pi]
            phase_diff = phase_recon - phase_true
            phase_diff = np.angle(np.exp(1j * phase_diff))

            # Mean absolute phase error
            n_support = np.sum(support_mask)
            if n_support == 0:
                return 0.0

            mean_abs_error = float(np.sum(np.abs(phase_diff)) / n_support)
            return mean_abs_error

    def _solve_rpIO(
            self,
            measured_intensity: np.ndarray,
            support_mask: np.ndarray,
            R_p: complex,
            obj_true: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
            """
            Run the rpIO (Reflective-Prior Input-Output) algorithm.

            Args:
                measured_intensity: Measured diffraction intensity, shape (detector_size, detector_size).
                support_mask: Boolean support mask, shape (N_obj, N_obj).
                R_p: Known substrate reflectivity (complex scalar).
                obj_true: Optional ground truth for error tracking.

            Returns:
                Tuple of:
                    - Reconstructed object, shape (N_obj, N_obj).
                    - Dictionary of error histories: {"fourier_error": [...], "object_error": [...]}.
            """
            det_size = self.detector_size
            n_obj = self.n_obj

            # Measured amplitude: sqrt(I(k))
            measured_amplitude = np.sqrt(np.maximum(measured_intensity, 0.0))

            # Offset for centering the object in the zero-padded array
            offset_r = (det_size - n_obj) // 2
            offset_c = (det_size - n_obj) // 2

            # Initialize O_0: substrate reflectivity everywhere, with random phase perturbation on support
            obj_current = np.full((n_obj, n_obj), R_p, dtype=np.complex128)
            random_phase = self.rng.uniform(0, 2 * np.pi, size=(n_obj, n_obj))
            obj_current[support_mask] = np.abs(R_p) * np.exp(1j * random_phase[support_mask])

            # Error history tracking
            error_history: Dict[str, List[float]] = {
                "fourier_error": [],
                "object_error": [],
            }

            for iteration in range(self.n_iterations):
                # Step 1: Zero-pad object and forward FFT
                padded = np.zeros((det_size, det_size), dtype=np.complex128)
                padded[offset_r:offset_r + n_obj, offset_c:offset_c + n_obj] = obj_current

                psi_k = fftshift(fft2(ifftshift(padded)))

                # Step 2: Fourier modulus replacement
                psi_c = self._fourier_modulus_replacement(psi_k, measured_amplitude)

                # Step 3: Inverse FFT and crop to object region
                padded_corrected = ifftshift(ifft2(fftshift(psi_c)))
                obj_new = padded_corrected[offset_r:offset_r + n_obj, offset_c:offset_c + n_obj].copy()

                # Step 4: Apply real-space constraint with reflectivity prior
                obj_updated = self._apply_support_and_prior(obj_new, obj_current, support_mask, R_p)

                # Step 5: Compute error metrics
                fourier_err = self._compute_fourier_error(obj_updated, measured_intensity)
                error_history["fourier_error"].append(fourier_err)

                if obj_true is not None:
                    obj_err = self._compute_object_error(obj_updated, obj_true)
                    error_history["object_error"].append(obj_err)

                # Update current object estimate
                obj_current = obj_updated

            return obj_current, error_history

    def _solve_rpPIE(
            self,
            measured_intensities: List[np.ndarray],
            scan_positions: np.ndarray,
            probe: np.ndarray,
            support_mask: np.ndarray,
            R_p: complex,
            obj_true: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
            """
            Run the rpPIE (Reflective-Prior Ptychographic Iterative Engine) algorithm.

            Args:
                measured_intensities: List of intensity patterns, each (detector_size, detector_size).
                scan_positions: Array of shape (N_positions, 2) with scan positions.
                probe: Complex probe function, shape (N_p, N_p).
                support_mask: Boolean support mask, shape (N_obj, N_obj).
                R_p: Known substrate reflectivity.
                obj_true: Optional ground truth for error tracking.

            Returns:
                Tuple of:
                    - Reconstructed object, shape (N_obj, N_obj).
                    - Dictionary of error histories.
            """
            det_size = self.detector_size
            n_obj = self.n_obj
            n_probe_r, n_probe_c = probe.shape
            n_positions = len(scan_positions)

            # Precompute measured amplitudes: sqrt(I_j(k))
            measured_amplitudes = [
                np.sqrt(np.maximum(intensity, 0.0)) for intensity in measured_intensities
            ]

            # Initialize O_0 = R_p everywhere (uniform substrate reflectivity)
            obj_current = np.full((n_obj, n_obj), R_p, dtype=np.complex128)

            # Error history tracking
            error_history: Dict[str, List[float]] = {
                "fourier_error": [],
                "object_error": [],
                "phase_error": [],
            }

            # Precompute probe quantities that don't change
            probe_abs = np.abs(probe)
            P_max = np.max(probe_abs)
            probe_conj = np.conj(probe)
            probe_abs_sq = probe_abs ** 2

            for epoch in range(self.n_iterations):
                # Randomize scan order each epoch
                order = self.rng.permutation(n_positions)

                for idx in order:
                    pos_r = int(scan_positions[idx, 0])
                    pos_c = int(scan_positions[idx, 1])

                    # Determine the region of the object illuminated by the probe
                    n_obj_r, n_obj_c = obj_current.shape
                    r_end = min(pos_r + n_probe_r, n_obj_r)
                    c_end = min(pos_c + n_probe_c, n_obj_c)
                    actual_r = r_end - pos_r
                    actual_c = c_end - pos_c

                    # Step 1: Form exit wave Psi = P(r - R_j) * O(r)
                    exit_wave = np.zeros((n_probe_r, n_probe_c), dtype=np.complex128)
                    exit_wave[:actual_r, :actual_c] = (
                        probe[:actual_r, :actual_c] * obj_current[pos_r:r_end, pos_c:c_end]
                    )

                    # Step 2: Forward FFT (zero-padded to detector size)
                    padded = np.zeros((det_size, det_size), dtype=np.complex128)
                    offset_r = (det_size - n_probe_r) // 2
                    offset_c = (det_size - n_probe_c) // 2
                    padded[offset_r:offset_r + n_probe_r, offset_c:offset_c + n_probe_c] = exit_wave

                    psi_k = fftshift(fft2(ifftshift(padded)))

                    # Step 3: Fourier modulus replacement
                    psi_c_k = self._fourier_modulus_replacement(psi_k, measured_amplitudes[idx])

                    # Step 4: Inverse FFT to get corrected exit wave
                    padded_corrected = ifftshift(ifft2(fftshift(psi_c_k)))
                    exit_wave_corrected = padded_corrected[
                        offset_r:offset_r + n_probe_r, offset_c:offset_c + n_probe_c
                    ].copy()

                    # Step 5: ePIE object update
                    # delta_Psi = Psi_c - Psi
                    delta_psi = exit_wave_corrected - exit_wave

                    # update = (|P| / |P_max|) * (P* / (|P|^2 + alpha)) * beta * delta_Psi
                    numerator = (probe_abs / P_max) * probe_conj * self.beta * delta_psi
                    denominator = probe_abs_sq + self.alpha
                    update = numerator / denominator

                    # Apply update to the object at the correct spatial position
                    obj_current[pos_r:r_end, pos_c:c_end] += update[:actual_r, :actual_c]

                    # Step 6: Apply reflectivity prior outside support
                    # Only apply to the region that was just updated to save computation
                    # But for correctness, apply to the full updated region
                    region_mask = support_mask[pos_r:r_end, pos_c:c_end]
                    outside_region = ~region_mask
                    obj_current[pos_r:r_end, pos_c:c_end][outside_region] = (
                        self.beta * obj_current[pos_r:r_end, pos_c:c_end][outside_region]
                        + (1.0 - self.beta) * R_p
                    )

                # After each epoch, compute error metrics
                # Compute Fourier error averaged over a subset of positions for efficiency
                if n_positions > 0:
                    # Use a small subset for Fourier error computation to save time
                    n_sample = min(5, n_positions)
                    sample_indices = self.rng.choice(n_positions, size=n_sample, replace=False)
                    fourier_err_sum = 0.0
                    for s_idx in sample_indices:
                        fe = self._compute_fourier_error(
                            obj_current,
                            measured_intensities[s_idx],
                            probe=probe,
                            scan_pos=scan_positions[s_idx],
                        )
                        fourier_err_sum += fe
                    avg_fourier_err = fourier_err_sum / n_sample
                    error_history["fourier_error"].append(avg_fourier_err)

                if obj_true is not None:
                    obj_err = self._compute_object_error(obj_current, obj_true)
                    error_history["object_error"].append(obj_err)

                    phase_err = self._compute_phase_error(obj_current, obj_true, support_mask)
                    error_history["phase_error"].append(phase_err)

            return obj_current, error_history

    def _solve_standard_HIO(
            self,
            measured_intensity: np.ndarray,
            support_mask: np.ndarray,
            obj_true: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
            """
            Run the standard HIO algorithm (baseline, no reflectivity prior).

            Args:
                measured_intensity: Measured diffraction intensity.
                support_mask: Boolean support mask.
                obj_true: Optional ground truth for error tracking.

            Returns:
                Tuple of reconstructed object and error histories.
            """
            det_size = self.detector_size
            n_obj = self.n_obj

            # Measured amplitude: sqrt(I(k))
            measured_amplitude = np.sqrt(np.maximum(measured_intensity, 0.0))

            # Offset for centering the object in the zero-padded array
            offset_r = (det_size - n_obj) // 2
            offset_c = (det_size - n_obj) // 2

            # Initialize O_0: random complex values on support, zero outside
            obj_current = np.zeros((n_obj, n_obj), dtype=np.complex128)
            random_phase = self.rng.uniform(0, 2 * np.pi, size=(n_obj, n_obj))
            random_amp = self.rng.uniform(0.5, 1.5, size=(n_obj, n_obj))
            obj_current[support_mask] = random_amp[support_mask] * np.exp(1j * random_phase[support_mask])

            # Error history tracking
            error_history: Dict[str, List[float]] = {
                "fourier_error": [],
                "object_error": [],
            }

            for iteration in range(self.n_iterations):
                # Step 1: Zero-pad object and forward FFT
                padded = np.zeros((det_size, det_size), dtype=np.complex128)
                padded[offset_r:offset_r + n_obj, offset_c:offset_c + n_obj] = obj_current

                psi_k = fftshift(fft2(ifftshift(padded)))

                # Step 2: Fourier modulus replacement
                psi_c = self._fourier_modulus_replacement(psi_k, measured_amplitude)

                # Step 3: Inverse FFT and crop to object region
                padded_corrected = ifftshift(ifft2(fftshift(psi_c)))
                obj_new = padded_corrected[offset_r:offset_r + n_obj, offset_c:offset_c + n_obj].copy()

                # Step 4: Apply standard HIO real-space constraint (no reflectivity prior)
                obj_updated = self._apply_hio_constraint(obj_new, obj_current, support_mask)

                # Step 5: Compute error metrics
                fourier_err = self._compute_fourier_error(obj_updated, measured_intensity)
                error_history["fourier_error"].append(fourier_err)

                if obj_true is not None:
                    obj_err = self._compute_object_error(obj_updated, obj_true)
                    error_history["object_error"].append(obj_err)

                # Update current object estimate
                obj_current = obj_updated

            return obj_current, error_history

    def _solve_standard_PIE(
            self,
            measured_intensities: List[np.ndarray],
            scan_positions: np.ndarray,
            probe: np.ndarray,
            support_mask: np.ndarray,
            obj_true: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
            """
            Run the standard PIE algorithm (baseline, no reflectivity prior).

            Args:
                measured_intensities: List of intensity patterns.
                scan_positions: Scan positions array.
                probe: Complex probe function.
                support_mask: Boolean support mask.
                obj_true: Optional ground truth for error tracking.

            Returns:
                Tuple of reconstructed object and error histories.
            """
            det_size = self.detector_size
            n_obj = self.n_obj
            n_probe_r, n_probe_c = probe.shape
            n_positions = len(scan_positions)

            # Precompute measured amplitudes: sqrt(I_j(k))
            measured_amplitudes = [
                np.sqrt(np.maximum(intensity, 0.0)) for intensity in measured_intensities
            ]

            # Initialize O_0 with small random complex values
            # Use uniform amplitude with random phase for a generic initialization
            obj_current = np.ones((n_obj, n_obj), dtype=np.complex128)
            random_phase = self.rng.uniform(0, 2 * np.pi, size=(n_obj, n_obj))
            obj_current = np.abs(obj_current) * np.exp(1j * random_phase)

            # Error history tracking
            error_history: Dict[str, List[float]] = {
                "fourier_error": [],
                "object_error": [],
                "phase_error": [],
            }

            # Precompute probe quantities that don't change
            probe_abs = np.abs(probe)
            P_max = np.max(probe_abs)
            probe_conj = np.conj(probe)
            probe_abs_sq = probe_abs ** 2

            for epoch in range(self.n_iterations):
                # Randomize scan order each epoch
                order = self.rng.permutation(n_positions)

                for idx in order:
                    pos_r = int(scan_positions[idx, 0])
                    pos_c = int(scan_positions[idx, 1])

                    # Determine the region of the object illuminated by the probe
                    n_obj_r, n_obj_c = obj_current.shape
                    r_end = min(pos_r + n_probe_r, n_obj_r)
                    c_end = min(pos_c + n_probe_c, n_obj_c)
                    actual_r = r_end - pos_r
                    actual_c = c_end - pos_c

                    # Step 1: Form exit wave Psi = P(r - R_j) * O(r)
                    exit_wave = np.zeros((n_probe_r, n_probe_c), dtype=np.complex128)
                    exit_wave[:actual_r, :actual_c] = (
                        probe[:actual_r, :actual_c] * obj_current[pos_r:r_end, pos_c:c_end]
                    )

                    # Step 2: Forward FFT (zero-padded to detector size)
                    padded = np.zeros((det_size, det_size), dtype=np.complex128)
                    offset_r = (det_size - n_probe_r) // 2
                    offset_c = (det_size - n_probe_c) // 2
                    padded[offset_r:offset_r + n_probe_r, offset_c:offset_c + n_probe_c] = exit_wave

                    psi_k = fftshift(fft2(ifftshift(padded)))

                    # Step 3: Fourier modulus replacement
                    psi_c_k = self._fourier_modulus_replacement(psi_k, measured_amplitudes[idx])

                    # Step 4: Inverse FFT to get corrected exit wave
                    padded_corrected = ifftshift(ifft2(fftshift(psi_c_k)))
                    exit_wave_corrected = padded_corrected[
                        offset_r:offset_r + n_probe_r, offset_c:offset_c + n_probe_c
                    ].copy()

                    # Step 5: ePIE object update (standard, no prior)
                    # delta_Psi = Psi_c - Psi
                    delta_psi = exit_wave_corrected - exit_wave

                    # update = (|P| / |P_max|) * (P* / (|P|^2 + alpha)) * beta * delta_Psi
                    numerator = (probe_abs / P_max) * probe_conj * self.beta * delta_psi
                    denominator = probe_abs_sq + self.alpha
                    update = numerator / denominator

                    # Apply update to the object at the correct spatial position
                    obj_current[pos_r:r_end, pos_c:c_end] += update[:actual_r, :actual_c]

                    # NO reflectivity prior enforcement (standard PIE)

                # After each epoch, compute error metrics
                if n_positions > 0:
                    # Use a small subset for Fourier error computation to save time
                    n_sample = min(5, n_positions)
                    sample_indices = self.rng.choice(n_positions, size=n_sample, replace=False)
                    fourier_err_sum = 0.0
                    for s_idx in sample_indices:
                        fe = self._compute_fourier_error(
                            obj_current,
                            measured_intensities[s_idx],
                            probe=probe,
                            scan_pos=scan_positions[s_idx],
                        )
                        fourier_err_sum += fe
                    avg_fourier_err = fourier_err_sum / n_sample
                    error_history["fourier_error"].append(avg_fourier_err)

                if obj_true is not None:
                    obj_err = self._compute_object_error(obj_current, obj_true)
                    error_history["object_error"].append(obj_err)

                    phase_err = self._compute_phase_error(obj_current, obj_true, support_mask)
                    error_history["phase_error"].append(phase_err)

            return obj_current, error_history

    def solve(
            self,
            measured_data: Union[np.ndarray, List[np.ndarray]],
            support_mask: Optional[np.ndarray] = None,
            probe: Optional[np.ndarray] = None,
            scan_positions: Optional[np.ndarray] = None,
            R_p: Optional[complex] = None,
            obj_true: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """
            Main entry point: solve the phase retrieval problem using the
            configured algorithm (rpIO, rpPIE, HIO, or PIE).

            Args:
                measured_data: For single-shot CDI: intensity array of shape (2048, 2048).
                               For ptychography: list of intensity arrays or stacked 3D array.
                support_mask: Boolean support mask. If None, auto-generated from pad geometry.
                probe: Complex probe function. If None, auto-generated from probe_type.
                scan_positions: Scan positions. If None, auto-generated for ptychography.
                R_p: Substrate reflectivity prior. If None, computed via TMM.
                obj_true: Optional ground truth for error monitoring.

            Returns:
                Reconstructed complex object array of shape (N_obj, N_obj).
            """
            # Auto-generate substrate reflectivity if not provided
            if R_p is None:
                if self.R_substrate == complex(0):
                    self.R_substrate = self._compute_reflectivity_tmm("substrate")
                if self.R_cu == complex(0):
                    self.R_cu = self._compute_reflectivity_tmm("Cu")
                R_p = self.R_substrate

            # Auto-generate support mask if not provided
            if support_mask is None:
                _, support_mask, _ = self._build_ground_truth_object()

            # Auto-generate probe if not provided
            if probe is None:
                probe = self._build_probe()

            # Determine if we're in ptychographic or single-shot mode
            is_ptychographic = self.algorithm in ("rpPIE", "PIE")

            # Convert measured_data to appropriate format
            if is_ptychographic:
                # Auto-generate scan positions if not provided
                if scan_positions is None:
                    scan_positions = self._generate_scan_positions(
                        object_shape=(self.n_obj, self.n_obj),
                        probe_shape=probe.shape,
                    )

                # Convert measured_data to list of 2D arrays if it's a 3D array
                if isinstance(measured_data, np.ndarray):
                    if measured_data.ndim == 3:
                        measured_intensities = [measured_data[i] for i in range(measured_data.shape[0])]
                    elif measured_data.ndim == 2:
                        # Single intensity provided; replicate or generate synthetic data
                        # If only one pattern is provided but we need multiple for ptychography,
                        # generate forward data from a ground-truth object
                        obj_gt, sm, _ = self._build_ground_truth_object()
                        support_mask = sm
                        measured_intensities = self.forward(obj_gt, probe, scan_positions)
                        obj_true = obj_gt
                    else:
                        raise ValueError(f"Unexpected measured_data ndim={measured_data.ndim}")
                elif isinstance(measured_data, list):
                    measured_intensities = measured_data
                else:
                    raise TypeError(f"Unexpected measured_data type: {type(measured_data)}")

                # Ensure we have the right number of intensities
                n_positions = len(scan_positions)
                if len(measured_intensities) != n_positions:
                    # Regenerate from ground truth if mismatch
                    obj_gt, support_mask, _ = self._build_ground_truth_object()
                    measured_intensities = self.forward(obj_gt, probe, scan_positions)
                    obj_true = obj_gt

                # Dispatch to the appropriate ptychographic algorithm
                if self.algorithm == "rpPIE":
                    reconstructed, error_history = self._solve_rpPIE(
                        measured_intensities=measured_intensities,
                        scan_positions=scan_positions,
                        probe=probe,
                        support_mask=support_mask,
                        R_p=R_p,
                        obj_true=obj_true,
                    )
                elif self.algorithm == "PIE":
                    reconstructed, error_history = self._solve_standard_PIE(
                        measured_intensities=measured_intensities,
                        scan_positions=scan_positions,
                        probe=probe,
                        support_mask=support_mask,
                        obj_true=obj_true,
                    )
                else:
                    raise ValueError(f"Unknown ptychographic algorithm: {self.algorithm}")

            else:
                # Single-shot CDI mode
                if isinstance(measured_data, np.ndarray) and measured_data.ndim == 2:
                    measured_intensity = measured_data.astype(np.float64)
                elif isinstance(measured_data, np.ndarray) and measured_data.ndim == 3:
                    # If 3D array provided for single-shot, use the first pattern
                    measured_intensity = measured_data[0].astype(np.float64)
                elif isinstance(measured_data, list):
                    measured_intensity = measured_data[0].astype(np.float64)
                else:
                    raise TypeError(f"Unexpected measured_data type: {type(measured_data)}")

                # Dispatch to the appropriate single-shot algorithm
                if self.algorithm == "rpIO":
                    reconstructed, error_history = self._solve_rpIO(
                        measured_intensity=measured_intensity,
                        support_mask=support_mask,
                        R_p=R_p,
                        obj_true=obj_true,
                    )
                elif self.algorithm == "HIO":
                    reconstructed, error_history = self._solve_standard_HIO(
                        measured_intensity=measured_intensity,
                        support_mask=support_mask,
                        obj_true=obj_true,
                    )
                else:
                    raise ValueError(f"Unknown single-shot algorithm: {self.algorithm}")

            # Store error history as an attribute for later analysis
            self.error_history_ = error_history

            return reconstructed




if __name__ == "__main__":
    # Load measured diffraction data
    input_data = np.load('dataset/input.npy')
    print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")

    if isinstance(input_data, np.ndarray) and input_data.ndim == 3:
        n_patterns = input_data.shape[0]
        det_size = input_data.shape[1]
        print(f"Loaded {n_patterns} diffraction patterns of shape {det_size}x{det_size}")

        # Determine scan grid: n_patterns should be n_scan_1d^2
        n_scan_1d = int(np.round(np.sqrt(n_patterns)))
        assert n_scan_1d * n_scan_1d == n_patterns, f"Non-square scan grid: {n_patterns} patterns"
        print(f"Scan grid: {n_scan_1d} x {n_scan_1d}")

        # The detector_size from the data is det_size (e.g., 256).
        # We need n_obj <= det_size for the FFT zero-padding to work.
        # Adjust object parameters to fit within the detector.
        # 
        # Strategy: Keep the pad structure but scale to fit.
        # If det_size=256, we need n_obj = array_size * pitch_pixels <= det_size.
        # With array_size=15, pitch_pixels = det_size // 15 = 17 (for 256)
        # pad_pixels ~ pitch_pixels * duty_cycle = 17 * 0.6 ~ 10
        #
        # But let's also consider: maybe the object is smaller.
        # Try different configurations.

        # First, figure out reasonable parameters.
        # The key constraint: n_obj = array_size * pitch_pixels <= det_size
        # and oversampling ratio = det_size / n_obj >= 2
        # So n_obj <= det_size / 2 = 128 (for det_size=256)
        # With array_size=15: pitch_pixels <= 128/15 ~ 8 pixels
        # pad_pixels ~ 8 * 0.6 ~ 5
        # That seems very small. Let's try array_size=15 with pitch=8, pad=5.
        #
        # Alternatively, maybe the object doesn't need oversampling > 2 for ptychography.
        # In ptychography, oversampling is provided by probe overlap, not zero-padding.
        # The exit wave (probe-sized) is zero-padded to det_size.
        # So the constraint is: probe_diameter <= det_size, and
        # n_obj can be larger than det_size since we only FFT probe-sized patches.
        # 
        # Wait - in our _solve_rpPIE, we zero-pad the exit wave (probe-sized) to det_size.
        # So we need probe_diameter_pixels <= det_size. That's fine.
        # n_obj can be anything >= probe_diameter + (n_scan_1d-1)*step.
        #
        # Let's try: keep n_obj=450 (15*30), probe_diameter that fits in det_size,
        # and the exit wave zero-padded to det_size works fine since probe < det_size.

        # For ptychography with 9x9 scan grid:
        # n_obj = probe_diameter + (n_scan_1d - 1) * step
        # We need to find probe_diameter and step such that this equals our n_obj.
        # 
        # Let's try multiple parameter sets and pick the one that's consistent.

        # Option A: Smaller object that fits naturally
        # Try to figure out the parameters from the data structure.
        # With 9x9 = 81 scan positions and det_size=256:
        # A reasonable probe size might be ~100-160 pixels
        # step ~ 40 pixels: n_obj = probe + 8*40 = probe + 320
        # If probe=130: n_obj = 450 (matches 15*30!)
        # If probe=160: n_obj = 480 (close to 15*32)
        
        # Let's try the standard parameters but with the actual det_size
        # The key insight: for ptychography, det_size just needs to be >= probe size
        # since we FFT the exit wave (probe-sized), not the full object.

        # Try with parameters that give n_obj consistent with 9x9 grid
        # n_obj = probe_diam + (9-1)*step
        # With probe_diam=130, step=40: n_obj = 130 + 320 = 450 = 15*30 ✓

        probe_diam = 130
        scan_step = 40
        n_obj_expected = probe_diam + (n_scan_1d - 1) * scan_step
        pitch_px = n_obj_expected // 15  # = 30
        pad_px = int(round(pitch_px * 0.6))  # = 18 or so

        # But paper says pad_pixels=22, pitch=30. Let's check:
        # n_obj = 15*30 = 450, probe = 450 - 8*40 = 450 - 320 = 130
        # That works!

        # Actually let's also try pad_pixels=22 with pitch=30
        pitch_px = 30
        pad_px = 22
        array_sz = 15
        n_obj_calc = array_sz * pitch_px  # 450
        probe_diam_calc = n_obj_calc - (n_scan_1d - 1) * scan_step  # 450 - 320 = 130

        print(f"Derived parameters: n_obj={n_obj_calc}, probe_diam={probe_diam_calc}, "
              f"pitch={pitch_px}, pad={pad_px}")
        print(f"Probe fits in detector: {probe_diam_calc} <= {det_size}: {probe_diam_calc <= det_size}")

        if probe_diam_calc > det_size:
            # Need to reduce parameters
            # probe_diam = det_size - some_margin
            probe_diam_calc = det_size - 20
            scan_step = max(1, (n_obj_calc - probe_diam_calc) // (n_scan_1d - 1))
            n_obj_calc = probe_diam_calc + (n_scan_1d - 1) * scan_step
            pitch_px = n_obj_calc // array_sz
            pad_px = int(round(pitch_px * 0.6))
            if pad_px < 2:
                pad_px = 2
            print(f"Adjusted: n_obj={n_obj_calc}, probe={probe_diam_calc}, "
                  f"step={scan_step}, pitch={pitch_px}, pad={pad_px}")

        # Try both airy and speckle - use airy as default for ptychography
        for probe_type in ["airy", "speckle"]:
            print(f"\nTrying probe_type='{probe_type}'...")

            solver = InverseSolver(
                wavelength=13.5e-9,
                pixel_size=27e-9,
                detector_size=det_size,
                pad_pixels=pad_px,
                pitch_pixels=pitch_px,
                array_size=array_sz,
                beta=0.9,
                alpha=1e-8,
                n_iterations=200,
                probe_type=probe_type,
                probe_diameter_pixels=probe_diam_calc,
                scan_step_pixels=scan_step,
                noise_sigma_fraction=0.0,
                algorithm="rpPIE",
                random_seed=42,
            )

            # Build ground truth for support mask and reflectivities
            obj_true, support_mask, recession_map = solver._build_ground_truth_object()
            probe = solver._build_probe()
            scan_positions = solver._generate_scan_positions(
                object_shape=(solver.n_obj, solver.n_obj),
                probe_shape=probe.shape,
            )

            print(f"Object shape: {obj_true.shape}, Probe shape: {probe.shape}")
            print(f"Scan positions: {len(scan_positions)}, expected: {n_patterns}")
            print(f"R_substrate: {solver.R_substrate}")
            print(f"R_cu: {solver.R_cu}")

            # Match scan positions to number of patterns
            if len(scan_positions) != n_patterns:
                print(f"Scan position mismatch: {len(scan_positions)} vs {n_patterns}")
                # Try adjusting step size
                for trial_step in range(scan_step - 10, scan_step + 20):
                    if trial_step < 1:
                        continue
                    trial_n = max(1, (solver.n_obj - probe_diam_calc) // trial_step + 1)
                    if trial_n * trial_n == n_patterns:
                        print(f"Found matching step={trial_step}, n_scan_1d={trial_n}")
                        solver.scan_step_pixels = trial_step
                        scan_positions = solver._generate_scan_positions(
                            object_shape=(solver.n_obj, solver.n_obj),
                            probe_shape=probe.shape,
                        )
                        break
                else:
                    # Force scan positions to match
                    max_pos = max(0, solver.n_obj - probe_diam_calc)
                    if n_scan_1d > 1:
                        forced_step = max_pos // (n_scan_1d - 1)
                    else:
                        forced_step = 0
                    scan_rows = np.arange(n_scan_1d) * forced_step
                    scan_cols = np.arange(n_scan_1d) * forced_step
                    scan_rows = np.clip(scan_rows, 0, max_pos)
                    scan_cols = np.clip(scan_cols, 0, max_pos)
                    gr, gc = np.meshgrid(scan_rows, scan_cols, indexing='ij')
                    scan_positions = np.stack([gr.ravel(), gc.ravel()], axis=-1).astype(np.int64)
                    print(f"Forced scan positions: {len(scan_positions)}, step={forced_step}")

            # Prepare measured intensities
            measured_intensities = [input_data[i].astype(np.float64) for i in range(min(n_patterns, len(scan_positions)))]
            if len(measured_intensities) < len(scan_positions):
                scan_positions = scan_positions[:len(measured_intensities)]
            elif len(measured_intensities) > len(scan_positions):
                measured_intensities = measured_intensities[:len(scan_positions)]

            R_p = solver.R_substrate

            print(f"Running rpPIE with {len(scan_positions)} positions, {solver.n_iterations} iterations...")
            result, error_history = solver._solve_rpPIE(
                measured_intensities=measured_intensities,
                scan_positions=scan_positions,
                probe=probe,
                support_mask=support_mask,
                R_p=R_p,
                obj_true=obj_true,
            )

            if error_history.get("object_error"):
                final_obj_err = error_history['object_error'][-1]
                print(f"Final object error: {final_obj_err:.6e}")
            if error_history.get("phase_error"):
                final_phase_err = error_history['phase_error'][-1]
                print(f"Final phase error: {final_phase_err:.6e} rad")
            if error_history.get("fourier_error"):
                print(f"Final Fourier error: {error_history['fourier_error'][-1]:.6e}")

            # Use the first successful result
            break

        # Also run standard PIE for comparison (optional, just log)
        print("\nRunning standard PIE baseline for comparison...")
        solver_pie = InverseSolver(
            wavelength=13.5e-9,
            pixel_size=27e-9,
            detector_size=det_size,
            pad_pixels=pad_px,
            pitch_pixels=pitch_px,
            array_size=array_sz,
            beta=0.9,
            alpha=1e-8,
            n_iterations=50,  # fewer iterations for baseline
            probe_type=probe_type,
            probe_diameter_pixels=probe_diam_calc,
            scan_step_pixels=solver.scan_step_pixels,
            noise_sigma_fraction=0.0,
            algorithm="PIE",
            random_seed=42,
        )
        probe_pie = solver_pie._build_probe()
        _, support_mask_pie, _ = solver_pie._build_ground_truth_object()

        result_pie, error_pie = solver_pie._solve_standard_PIE(
            measured_intensities=measured_intensities,
            scan_positions=scan_positions,
            probe=probe_pie,
            support_mask=support_mask_pie,
            obj_true=obj_true,
        )
        if error_pie.get("object_error"):
            print(f"PIE baseline final object error: {error_pie['object_error'][-1]:.6e}")

    elif isinstance(input_data, np.ndarray) and input_data.ndim == 2:
        det_size = input_data.shape[0]
        print(f"Loaded single diffraction pattern of shape {det_size}x{det_size}")

        # For single-shot CDI, the object must be oversampled: n_obj <= det_size / 2
        max_n_obj = det_size // 2
        array_sz = 15
        pitch_px = max_n_obj // array_sz
        if pitch_px < 2:
            pitch_px = 2
        pad_px = max(1, int(round(pitch_px * 0.6)))
        n_obj_calc = array_sz * pitch_px

        print(f"Single-shot params: det_size={det_size}, n_obj={n_obj_calc}, "
              f"pitch={pitch_px}, pad={pad_px}")

        solver = InverseSolver(
            wavelength=13.5e-9,
            pixel_size=27e-9,
            detector_size=det_size,
            pad_pixels=pad_px,
            pitch_pixels=pitch_px,
            array_size=array_sz,
            beta=0.9,
            alpha=1e-8,
            n_iterations=300,
            probe_type="plane",
            probe_diameter_pixels=n_obj_calc,
            scan_step_pixels=40,
            noise_sigma_fraction=0.0,
            algorithm="rpIO",
            random_seed=42,
        )

        obj_true, support_mask, recession_map = solver._build_ground_truth_object()
        R_p = solver.R_substrate

        print(f"Running rpIO with {solver.n_iterations} iterations...")
        result, error_history = solver._solve_rpIO(
            measured_intensity=input_data.astype(np.float64),
            support_mask=support_mask,
            R_p=R_p,
            obj_true=obj_true,
        )

        if error_history.get("object_error"):
            print(f"Final object error: {error_history['object_error'][-1]:.6e}")
        if error_history.get("fourier_error"):
            print(f"Final Fourier error: {error_history['fourier_error'][-1]:.6e}")

    else:
        print(f"Unexpected input: ndim={input_data.ndim}, shape={input_data.shape}")
        # Fallback: use solve() with default rpPIE
        solver = InverseSolver(
            wavelength=13.5e-9,
            pixel_size=27e-9,
            detector_size=2048,
            algorithm="rpPIE",
            random_seed=42,
        )
        result = solver.solve(measured_data=input_data)

    # Save reconstructed complex object
    print(f"\nSaving result of shape {result.shape}, dtype {result.dtype}")
    np.save('output.npy', result)
    print("Done. Output saved to output.npy")