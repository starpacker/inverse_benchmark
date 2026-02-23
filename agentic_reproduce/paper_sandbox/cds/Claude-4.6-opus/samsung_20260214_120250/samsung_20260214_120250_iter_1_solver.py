import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.special import j1
from typing import Optional, Tuple, List, Dict, Any

class InverseSolver:
    """
    Solver for ptychographic phase retrieval of periodic structures
    using rpIO and rpPIE algorithms with reflectivity priors.
    
    Recovers complex-valued reflection function O(r) of a 15x15 periodic
    copper pad array from far-field diffraction intensity measurements.
    """

    def __init__(
            self,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            pad_size_pixels: int = 22,
            pitch_pixels: int = 30,
            num_pads: int = 15,
            scan_step_pixels: int = 40,
            overlap_fraction: float = 0.75,
            beta: float = 0.9,
            alpha_regularization: float = 1e-3,
            epsilon: float = 1e-8,
            max_iterations: int = 300,
            algorithm: str = "rppie",
            probe_type: str = "airy",
            noise_sigma: float = 0.0,
            random_seed: int = 42,
            recession_range: Tuple[float, float] = (1e-9, 3e-9),
            optical_constants: Optional[Dict[str, Tuple[float, float]]] = None,
            layer_thicknesses: Optional[Dict[str, float]] = None,
            probe_diameter_pixels: int = 160,
            convergence_threshold: float = 1e-10,
        ) -> None:
            """
            Initialize the InverseSolver with all physical and algorithmic parameters.

            Args:
                wavelength: Illumination wavelength in meters (default 13.5 nm).
                pixel_size: Real-space pixel size in meters (default 27 nm).
                pad_size_pixels: Size of each Cu pad in pixels (default 22).
                pitch_pixels: Pitch of the periodic array in pixels (default 30).
                num_pads: Number of pads along each axis (default 15).
                scan_step_pixels: Ptychographic scan step in pixels (default 40).
                overlap_fraction: Fractional overlap between adjacent probe positions.
                beta: Feedback / step-size parameter for HIO/PIE updates.
                alpha_regularization: Wiener-type regularization (fraction of |P_max|^2).
                epsilon: Small constant to stabilize Fourier magnitude division.
                max_iterations: Maximum number of iterations (IO) or epochs (PIE).
                algorithm: One of 'hio', 'rpio', 'pie', 'rppie'.
                probe_type: One of 'airy' or 'speckle'.
                noise_sigma: Standard deviation of additive Gaussian noise.
                random_seed: Random seed for reproducibility.
                recession_range: Tuple (h_min, h_max) for pad recession depths in meters.
                optical_constants: Dict mapping material name to (n, k) tuples.
                layer_thicknesses: Dict mapping material name to layer thickness in meters.
                probe_diameter_pixels: Diameter of the probe function in pixels.
                convergence_threshold: Threshold on Fourier error for early stopping.
            """
            # Physical parameters
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.pad_size_pixels: int = pad_size_pixels
            self.pitch_pixels: int = pitch_pixels
            self.num_pads: int = num_pads
            self.scan_step_pixels: int = scan_step_pixels
            self.overlap_fraction: float = overlap_fraction
            self.recession_range: Tuple[float, float] = recession_range
            self.probe_diameter_pixels: int = probe_diameter_pixels
            self.probe_type: str = probe_type
            self.noise_sigma: float = noise_sigma

            # Algorithmic parameters
            self.beta: float = beta
            self.alpha_regularization: float = alpha_regularization
            self.epsilon: float = epsilon
            self.max_iterations: int = max_iterations
            self.algorithm: str = algorithm.lower()
            self.convergence_threshold: float = convergence_threshold
            self.random_seed: int = random_seed

            # Optical constants (CXRO at 13.5 nm): (n, k)
            self.optical_constants: Dict[str, Tuple[float, float]] = optical_constants or {
                "SiCN": (0.9791, 0.0059),
                "SiO2": (0.9780, 0.0108),
                "Si": (0.9991, 0.0018),
                "Cu": (0.9625, 0.0613),
            }

            # Layer thicknesses (meters) for TMM calculation
            self.layer_thicknesses: Dict[str, float] = layer_thicknesses or {
                "SiCN": 3e-9,
                "SiO2": 5e-9,
                "Si": np.inf,  # semi-infinite substrate
            }

            # Compute derived grid sizes
            # Object grid must accommodate the full pad array plus border for probe extent
            array_extent = self.num_pads * self.pitch_pixels  # pixels occupied by pad array
            probe_radius = self.probe_diameter_pixels // 2
            # Total object size: array extent + probe diameter (to allow probe to scan across full array)
            obj_size = array_extent + 2 * probe_radius
            # Round up to even number for FFT efficiency
            if obj_size % 2 != 0:
                obj_size += 1
            self.object_shape: Tuple[int, int] = (obj_size, obj_size)

            # Probe shape (square, matching probe diameter, padded to power-of-2 or even for FFT)
            probe_size = self.probe_diameter_pixels
            # Ensure probe size is even
            if probe_size % 2 != 0:
                probe_size += 1
            self.probe_shape: Tuple[int, int] = (probe_size, probe_size)

            # Offset: the pad array starts at probe_radius within the object grid
            self.array_offset: int = probe_radius

            # Derived quantities (initialized to None/defaults, set during setup())
            self.support_mask: Optional[np.ndarray] = None
            self.probe: Optional[np.ndarray] = None
            self.scan_positions: Optional[np.ndarray] = None
            self.R_substrate: complex = 0.0 + 0.0j
            self.R_pad: complex = 0.0 + 0.0j
            self.P_max: float = 1.0
            self.convergence_history: List[float] = []

            # Set the random number generator for reproducibility
            self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def _compute_tmm_reflectivity(
            self,
            layer_materials: List[str],
            layer_thicknesses_m: List[float],
            substrate_material: str,
            incident_medium_n: complex = 1.0,
        ) -> complex:
            """
            Compute complex Fresnel reflection coefficient at normal incidence
            using the Transfer Matrix Method for a multilayer stack.

            Args:
                layer_materials: List of material names from top to bottom.
                layer_thicknesses_m: Corresponding layer thicknesses in meters.
                substrate_material: Name of the semi-infinite substrate material.
                incident_medium_n: Complex refractive index of incident medium (vacuum=1).

            Returns:
                Complex reflection coefficient r.
            """
            # Build complex refractive indices for each layer
            # Convention for EUV: n_complex = n + i*k where n < 1 typically
            # The complex refractive index is n_tilde = n + i*k
            # (some references use n - i*k; here we use n + i*k consistent with
            #  the wave convention exp(-i*omega*t) and absorption via positive k)

            def get_complex_n(material: str) -> complex:
                n_real, k_imag = self.optical_constants[material]
                # For EUV, the refractive index is typically written as
                # n_tilde = 1 - delta + i*beta, where delta = 1 - n_real, beta = k_imag
                # This means n_tilde = n_real + i*k_imag
                # But for proper wave propagation with exp(+i*k*z) convention,
                # absorption requires positive imaginary part
                return complex(n_real, k_imag)

            n0 = complex(incident_medium_n)  # incident medium (vacuum)
            ns = get_complex_n(substrate_material)  # substrate (semi-infinite)

            # Initialize total transfer matrix as identity
            M_total = np.array([[1.0 + 0j, 0.0 + 0j],
                                [0.0 + 0j, 1.0 + 0j]], dtype=np.complex128)

            # Build transfer matrix for each layer (from top to bottom)
            for mat, d in zip(layer_materials, layer_thicknesses_m):
                n_layer = get_complex_n(mat)
                # Phase thickness: delta_j = 2*pi*n_layer*d / lambda
                delta_j = 2.0 * np.pi * n_layer * d / self.wavelength

                cos_d = np.cos(delta_j)
                sin_d = np.sin(delta_j)

                # Transfer matrix for layer j at normal incidence (s-polarization,
                # which equals p-polarization at normal incidence):
                # M_j = [[cos(delta_j), -i*sin(delta_j)/n_j],
                #        [-i*n_j*sin(delta_j), cos(delta_j)]]
                M_j = np.array([
                    [cos_d, -1j * sin_d / n_layer],
                    [-1j * n_layer * sin_d, cos_d]
                ], dtype=np.complex128)

                M_total = M_total @ M_j

            # Extract matrix elements
            M11 = M_total[0, 0]
            M12 = M_total[0, 1]
            M21 = M_total[1, 0]
            M22 = M_total[1, 1]

            # Reflection coefficient at normal incidence:
            # r = (M11*n0 + M12*n0*ns - M21 - M22*ns) / (M11*n0 + M12*n0*ns + M21 + M22*ns)
            numerator = (M11 + M12 * ns) * n0 - (M21 + M22 * ns)
            denominator = (M11 + M12 * ns) * n0 + (M21 + M22 * ns)

            r = numerator / denominator

            return complex(r)

    def _build_support_mask(self) -> np.ndarray:
            """
            Construct the binary support mask S for the 15x15 periodic pad array.

            Returns:
                Binary mask of shape (N_obj, N_obj) where 1 indicates pad regions.
            """
            mask = np.zeros(self.object_shape, dtype=np.float64)

            for m in range(self.num_pads):
                for n in range(self.num_pads):
                    row_start = self.array_offset + m * self.pitch_pixels
                    row_end = row_start + self.pad_size_pixels
                    col_start = self.array_offset + n * self.pitch_pixels
                    col_end = col_start + self.pad_size_pixels

                    # Clip to object boundaries
                    row_start_clipped = max(row_start, 0)
                    row_end_clipped = min(row_end, self.object_shape[0])
                    col_start_clipped = max(col_start, 0)
                    col_end_clipped = min(col_end, self.object_shape[1])

                    if row_start_clipped < row_end_clipped and col_start_clipped < col_end_clipped:
                        mask[row_start_clipped:row_end_clipped, col_start_clipped:col_end_clipped] = 1.0

            return mask

    def _generate_probe(self) -> np.ndarray:
            """
            Generate the probe (illumination) function.

            Supports 'airy' (Airy disk) and 'speckle' (random structured) modes.

            Returns:
                Complex probe array of shape (N_probe, N_probe).
            """
            ny, nx = self.probe_shape
            # Create coordinate grids centered at the middle of the probe array
            y = np.arange(ny) - ny / 2.0
            x = np.arange(nx) - nx / 2.0
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)

            # The probe radius in pixels (half the diameter)
            probe_radius = self.probe_diameter_pixels / 2.0

            if self.probe_type == "airy":
                # Airy disk pattern: amplitude = 2*J1(pi*r/r0) / (pi*r/r0)
                # where r0 is the radius of the first zero ~ 1.22 * (lambda * f / D)
                # Here we define r0 such that the first zero of the Airy pattern
                # falls at the probe radius
                # The first zero of J1(x) is at x ≈ 3.8317, so pi*r/r0 = 3.8317
                # means r0 = pi * probe_radius / 3.8317
                # But for simplicity, we use r0 = probe_radius / 1.22 so the first
                # zero is at r = 1.22 * r0 = probe_radius
                r0 = probe_radius / 1.22

                # Compute the argument for the Airy function
                arg = np.pi * r / r0

                # Handle the singularity at r = 0
                # Airy amplitude: 2*J1(arg)/arg, which -> 1 as arg -> 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    airy_amplitude = np.where(
                        arg == 0.0,
                        1.0,
                        2.0 * j1(arg) / arg
                    )

                # The probe is the complex amplitude (real-valued for Airy)
                # Intensity of Airy pattern is airy_amplitude^2, but we want
                # the amplitude (electric field), so we use airy_amplitude directly
                probe = airy_amplitude.astype(np.complex128)

                # Apply a soft circular aperture to limit the probe extent
                # Taper smoothly to zero beyond the probe diameter
                taper = np.clip(1.0 - (r - probe_radius) / (0.1 * probe_radius + 1e-10), 0.0, 1.0)
                probe = probe * taper

            elif self.probe_type == "speckle":
                # Generate speckle field by propagating a random phase screen
                # through a circular aperture
                # 1. Create circular aperture in Fourier space
                aperture = np.zeros((ny, nx), dtype=np.complex128)
                aperture[r <= probe_radius] = 1.0

                # 2. Apply random phase
                random_phase = self.rng.uniform(0, 2 * np.pi, size=(ny, nx))
                aperture = aperture * np.exp(1j * random_phase)

                # 3. Inverse FFT to get the speckle probe in real space
                probe = fftshift(ifft2(ifftshift(aperture)))

                # 4. Apply a soft circular window in real space to confine the probe
                window = np.zeros((ny, nx), dtype=np.float64)
                window[r <= probe_radius] = 1.0
                # Smooth the window edges with a Gaussian taper
                sigma_taper = 3.0  # pixels
                edge_region = (r > probe_radius) & (r <= probe_radius + 4 * sigma_taper)
                window[edge_region] = np.exp(-0.5 * ((r[edge_region] - probe_radius) / sigma_taper) ** 2)
                probe = probe * window

            else:
                raise ValueError(f"Unknown probe_type '{self.probe_type}'. Use 'airy' or 'speckle'.")

            # Normalize probe so that max amplitude = 1
            max_amp = np.max(np.abs(probe))
            if max_amp > 0:
                probe = probe / max_amp

            return probe.astype(np.complex128)

    def _generate_scan_positions(self) -> np.ndarray:
            """
            Generate raster scan positions with specified step size and overlap.

            Returns:
                Array of shape (J, 2) containing (row, col) scan positions.
            """
            # The probe must scan across the entire pad array.
            # The object grid has the pad array starting at self.array_offset.
            # The probe is centered at position (row, col), and illuminates
            # a region of size probe_shape around that center.
            # We need the probe to cover from the start of the array to the end.

            # The probe's top-left corner when centered at position (r, c) is:
            #   (r - probe_h//2, c - probe_w//2)
            # We want the probe to start covering from the beginning of the pad array
            # and extend to the end.

            probe_h, probe_w = self.probe_shape
            obj_h, obj_w = self.object_shape

            # The scan region: the probe center must be positioned such that
            # the probe window [pos - probe//2 : pos + probe//2] stays within
            # the object boundaries.
            # Minimum center position: probe//2
            # Maximum center position: obj_size - probe//2

            min_row = probe_h // 2
            max_row = obj_h - probe_h // 2
            min_col = probe_w // 2
            max_col = obj_w - probe_w // 2

            # However, we also want to ensure we cover the entire pad array region.
            # The pad array spans from self.array_offset to
            # self.array_offset + self.num_pads * self.pitch_pixels
            # We want the probe to scan such that every part of the array is illuminated.

            # Start scanning from the earliest position that places the probe
            # over the beginning of the array
            array_end_row = self.array_offset + self.num_pads * self.pitch_pixels
            array_end_col = self.array_offset + self.num_pads * self.pitch_pixels

            # The first scan position: probe center such that the left/top edge
            # of the probe aligns with (or before) the start of the array
            scan_start_row = max(min_row, self.array_offset + probe_h // 2)
            scan_start_col = max(min_col, self.array_offset + probe_w // 2)

            # The last scan position: probe center such that the right/bottom edge
            # of the probe aligns with (or after) the end of the array
            scan_end_row = min(max_row, array_end_row - probe_h // 2 + probe_h)
            scan_end_col = min(max_col, array_end_col - probe_w // 2 + probe_w)

            # Clamp to valid range
            scan_start_row = max(scan_start_row, min_row)
            scan_end_row = min(scan_end_row, max_row)
            scan_start_col = max(scan_start_col, min_col)
            scan_end_col = min(scan_end_col, max_col)

            # Generate raster grid of scan positions with the specified step size
            row_positions = np.arange(scan_start_row, scan_end_row + 1, self.scan_step_pixels)
            col_positions = np.arange(scan_start_col, scan_end_col + 1, self.scan_step_pixels)

            # If no positions generated (edge case), ensure at least one central position
            if len(row_positions) == 0:
                row_positions = np.array([obj_h // 2])
            if len(col_positions) == 0:
                col_positions = np.array([obj_w // 2])

            # Create a meshgrid of all (row, col) combinations for raster scan
            row_grid, col_grid = np.meshgrid(row_positions, col_positions, indexing='ij')
            positions = np.stack([row_grid.ravel(), col_grid.ravel()], axis=-1)

            return positions.astype(np.int64)

    def _generate_ground_truth_object(
            self, recession_depths: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Construct the ground truth complex reflection function O(r).

            Args:
                recession_depths: Optional array of shape (num_pads, num_pads) with
                                  recession depths in meters. If None, randomly generated.

            Returns:
                Tuple of (object array of shape (N_obj, N_obj) complex,
                          recession_depths array of shape (num_pads, num_pads)).
            """
            # Generate or use provided recession depths
            if recession_depths is None:
                h_min, h_max = self.recession_range
                recession_depths = self.rng.uniform(
                    h_min, h_max, size=(self.num_pads, self.num_pads)
                )

            # Initialize object with substrate reflectivity everywhere
            obj = np.full(self.object_shape, self.R_substrate, dtype=np.complex128)

            # Fill in each pad with Cu reflectivity and phase from recession depth
            for m in range(self.num_pads):
                for n in range(self.num_pads):
                    row_start = self.array_offset + m * self.pitch_pixels
                    row_end = row_start + self.pad_size_pixels
                    col_start = self.array_offset + n * self.pitch_pixels
                    col_end = col_start + self.pad_size_pixels

                    # Clip to object boundaries
                    row_start_clipped = max(row_start, 0)
                    row_end_clipped = min(row_end, self.object_shape[0])
                    col_start_clipped = max(col_start, 0)
                    col_end_clipped = min(col_end, self.object_shape[1])

                    if row_start_clipped < row_end_clipped and col_start_clipped < col_end_clipped:
                        # Phase from recession depth: phi = 4*pi*h / lambda
                        # The factor of 4pi accounts for the double-pass in reflection geometry
                        h_mn = recession_depths[m, n]
                        phase = 4.0 * np.pi * h_mn / self.wavelength

                        # Object value at this pad: R_Cu * exp(i * phase)
                        obj[row_start_clipped:row_end_clipped,
                            col_start_clipped:col_end_clipped] = (
                            self.R_pad * np.exp(1j * phase)
                        )

            return obj, recession_depths

    def _extract_probe_region(
            self, obj: np.ndarray, position: np.ndarray
        ) -> np.ndarray:
            """
            Extract the object region illuminated by the probe at a given scan position.

            Args:
                obj: Full object array of shape (N_obj, N_obj).
                position: Scan position (row, col) as 1D array of length 2.

            Returns:
                Object patch of shape (N_probe, N_probe).
            """
            row_center, col_center = int(position[0]), int(position[1])
            probe_h, probe_w = self.probe_shape
            half_h, half_w = probe_h // 2, probe_w // 2

            row_start = row_center - half_h
            col_start = col_center - half_w
            row_end = row_start + probe_h
            col_end = col_start + probe_w

            obj_h, obj_w = obj.shape

            # Initialize patch with zeros (handles out-of-bounds with zero padding)
            patch = np.zeros(self.probe_shape, dtype=obj.dtype)

            # Compute the valid overlap between the extraction window and the object
            # Source (object) coordinates
            src_row_start = max(row_start, 0)
            src_row_end = min(row_end, obj_h)
            src_col_start = max(col_start, 0)
            src_col_end = min(col_end, obj_w)

            # Destination (patch) coordinates
            dst_row_start = src_row_start - row_start
            dst_row_end = dst_row_start + (src_row_end - src_row_start)
            dst_col_start = src_col_start - col_start
            dst_col_end = dst_col_start + (src_col_end - src_col_start)

            if src_row_start < src_row_end and src_col_start < src_col_end:
                patch[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = (
                    obj[src_row_start:src_row_end, src_col_start:src_col_end]
                )

            return patch

    def _insert_probe_region(
            self,
            obj: np.ndarray,
            patch: np.ndarray,
            position: np.ndarray,
        ) -> np.ndarray:
            """
            Insert/update a patch back into the full object array at a given position.

            Args:
                obj: Full object array of shape (N_obj, N_obj), modified in-place.
                patch: Updated patch of shape (N_probe, N_probe).
                position: Scan position (row, col) as 1D array of length 2.

            Returns:
                Updated object array.
            """
            row_center, col_center = int(position[0]), int(position[1])
            probe_h, probe_w = self.probe_shape
            half_h, half_w = probe_h // 2, probe_w // 2

            row_start = row_center - half_h
            col_start = col_center - half_w
            row_end = row_start + probe_h
            col_end = col_start + probe_w

            obj_h, obj_w = obj.shape

            # Compute the valid overlap between the insertion window and the object
            # Source (patch) coordinates
            src_row_start = max(-row_start, 0)
            src_row_end = probe_h - max(row_end - obj_h, 0)
            src_col_start = max(-col_start, 0)
            src_col_end = probe_w - max(col_end - obj_w, 0)

            # Destination (object) coordinates
            dst_row_start = max(row_start, 0)
            dst_row_end = min(row_end, obj_h)
            dst_col_start = max(col_start, 0)
            dst_col_end = min(col_end, obj_w)

            if dst_row_start < dst_row_end and dst_col_start < dst_col_end:
                obj[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = (
                    patch[src_row_start:src_row_end, src_col_start:src_col_end]
                )

            return obj

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            position: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """
            Compute far-field diffraction pattern via Fraunhofer propagation.

            For ptychographic mode (position provided):
                Psi(r) = P(r - R_j) * O(r)  ->  FFT  ->  |Psi_hat(k)|^2

            For single-shot CDI mode (position=None):
                Psi(r) = O(r)  ->  FFT  ->  |Psi_hat(k)|^2

            Args:
                obj: Complex object array. Shape (N_obj, N_obj) or (N_probe, N_probe).
                probe: Complex probe array of shape (N_probe, N_probe).
                position: Optional scan position (row, col). If None, single-shot mode.

            Returns:
                Far-field intensity I(k) of shape (N_probe, N_probe), real non-negative.
            """
            if position is not None:
                # Ptychographic mode: extract the object region under the probe
                obj_patch = self._extract_probe_region(obj, position)
                # Form exit wave: element-wise multiplication of probe and object patch
                exit_wave = probe * obj_patch
            else:
                # Single-shot CDI mode: the object itself is the exit wave
                # If obj is larger than probe, we use obj directly (no probe multiplication)
                # If obj shape matches probe shape, multiply
                if obj.shape == probe.shape:
                    exit_wave = probe * obj
                else:
                    # For single-shot CDI, the entire object is the exit wave
                    # (probe is assumed to uniformly illuminate the object)
                    exit_wave = obj.copy()

            # Fraunhofer propagation via 2D FFT with proper centering
            # ifftshift centers the zero-frequency component before FFT,
            # fftshift re-centers the output
            psi_hat = fftshift(fft2(ifftshift(exit_wave)))

            # Compute intensity: I(k) = |Psi_hat(k)|^2
            intensity = np.abs(psi_hat) ** 2

            # Ensure non-negativity (should already be, but for numerical safety)
            intensity = np.maximum(intensity, 0.0)

            return intensity.astype(np.float64)

    def _fourier_magnitude_projection(
            self,
            psi_hat: np.ndarray,
            measured_intensity: np.ndarray,
        ) -> np.ndarray:
            """
            Replace Fourier amplitudes with measured values while preserving phase.

            Psi_corrected(k) = sqrt(I_measured(k)) * Psi_hat(k) / (|Psi_hat(k)| + eps)

            Args:
                psi_hat: Current estimate of far-field complex amplitude, shape (N, N).
                measured_intensity: Measured diffraction intensity I(k), shape (N, N).

            Returns:
                Corrected far-field complex amplitude, shape (N, N).
            """
            # Compute measured amplitudes from intensity, ensuring non-negativity
            amp_measured = np.sqrt(np.maximum(measured_intensity, 0.0))

            # Compute current amplitudes
            amp_current = np.abs(psi_hat)

            # Replace amplitudes while preserving phase:
            # Psi_corrected = sqrt(I_measured) * Psi_hat / (|Psi_hat| + epsilon)
            psi_corrected = amp_measured * psi_hat / (amp_current + self.epsilon)

            return psi_corrected

    def _rpIO_step(
            self,
            obj_current: np.ndarray,
            measured_intensity: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            """
            Perform one iteration of the rpIO algorithm (Eq. 5).

            Steps:
                1. Forward FFT of current object estimate
                2. Fourier magnitude projection
                3. Inverse FFT to get updated estimate
                4. Apply support constraint with reflectivity prior:
                   - Inside S: keep inverse FFT result
                   - Outside S: beta * O_n + (1 - beta) * R_p

            Args:
                obj_current: Current object estimate, shape (N_obj, N_obj) complex.
                measured_intensity: Measured diffraction intensity, shape (N_obj, N_obj).

            Returns:
                Tuple of (updated object estimate, Fourier error metric).
            """
            # Step 1: Forward propagate current object estimate
            psi_hat = fftshift(fft2(ifftshift(obj_current)))

            # Compute Fourier error before correction
            amp_measured = np.sqrt(np.maximum(measured_intensity, 0.0))
            amp_current = np.abs(psi_hat)
            numerator = np.sum((amp_measured - amp_current) ** 2)
            denominator = np.sum(measured_intensity) + self.epsilon
            fourier_error = float(numerator / denominator)

            # Step 2: Fourier magnitude projection - replace amplitudes, keep phase
            psi_hat_corrected = self._fourier_magnitude_projection(psi_hat, measured_intensity)

            # Step 3: Inverse FFT to get updated estimate in real space
            obj_prime = fftshift(ifft2(ifftshift(psi_hat_corrected)))

            # Step 4: Apply real-space constraint with reflectivity prior (Eq. 5)
            obj_updated = np.empty_like(obj_current)

            # Inside support S: keep the inverse FFT result
            # Outside support S: beta * O_n + (1 - beta) * R_p
            support = self.support_mask

            obj_updated = np.where(
                support > 0.5,
                obj_prime,
                self.beta * obj_current + (1.0 - self.beta) * self.R_substrate
            )

            return obj_updated, fourier_error

    def _standard_hio_step(
            self,
            obj_current: np.ndarray,
            measured_intensity: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            """
            Perform one iteration of standard HIO algorithm.

            Outside support: O_{n+1} = O_n - beta * O'_n (standard HIO rule).

            Args:
                obj_current: Current object estimate, shape (N_obj, N_obj) complex.
                measured_intensity: Measured diffraction intensity, shape (N_obj, N_obj).

            Returns:
                Tuple of (updated object estimate, Fourier error metric).
            """
            # Step 1: Forward propagate current object estimate
            psi_hat = fftshift(fft2(ifftshift(obj_current)))

            # Compute Fourier error before correction
            amp_measured = np.sqrt(np.maximum(measured_intensity, 0.0))
            amp_current = np.abs(psi_hat)
            numerator = np.sum((amp_measured - amp_current) ** 2)
            denominator = np.sum(measured_intensity) + self.epsilon
            fourier_error = float(numerator / denominator)

            # Step 2: Fourier magnitude projection - replace amplitudes, keep phase
            psi_hat_corrected = self._fourier_magnitude_projection(psi_hat, measured_intensity)

            # Step 3: Inverse FFT to get updated estimate in real space
            obj_prime = fftshift(ifft2(ifftshift(psi_hat_corrected)))

            # Step 4: Apply real-space constraint with standard HIO rule
            # Inside support S: keep the inverse FFT result
            # Outside support S: O_{n+1} = O_n - beta * O'_n
            support = self.support_mask

            obj_updated = np.where(
                support > 0.5,
                obj_prime,
                obj_current - self.beta * obj_prime
            )

            return obj_updated, fourier_error

    def _rpPIE_epoch(
            self,
            obj_current: np.ndarray,
            measured_intensities: List[np.ndarray],
        ) -> Tuple[np.ndarray, float]:
            """
            Perform one epoch of the rpPIE algorithm (Eq. 6).

            For each scan position j (randomly shuffled):
                Step A (ePIE update):
                    O_{n+1}(r) = O_n(r) + (|P|/|P_max|) * (P* / (|P|^2 + alpha))
                                 * beta * (Psi_corrected - Psi)
                Step B (reflectivity prior):
                    Outside S: O_{n+1}(r) = beta * O_{n+1}(r) + (1-beta) * R_p

            Args:
                obj_current: Current object estimate, shape (N_obj, N_obj) complex.
                measured_intensities: List of J measured intensity arrays,
                                      each of shape (N_probe, N_probe).

            Returns:
                Tuple of (updated object estimate, mean Fourier error over all positions).
            """
            obj = obj_current.copy()
            num_positions = len(self.scan_positions)

            # Randomly shuffle scan position indices for this epoch
            indices = np.arange(num_positions)
            self.rng.shuffle(indices)

            # Precompute probe-related quantities
            probe = self.probe
            P_conj = np.conj(probe)
            P_abs = np.abs(probe)
            P_abs_sq = P_abs ** 2
            P_max_abs = self.P_max  # scalar: max |P|
            alpha = self.alpha_regularization * (P_max_abs ** 2)

            # Update weight: (|P| / |P_max|) * (P* / (|P|^2 + alpha)) * beta
            # = beta * |P| * P* / (|P_max| * (|P|^2 + alpha))
            update_weight = self.beta * P_abs * P_conj / (P_max_abs * (P_abs_sq + alpha))

            total_fourier_error = 0.0

            for idx in indices:
                position = self.scan_positions[idx]
                measured = measured_intensities[idx]

                # Step A: ePIE-style object update

                # 1. Extract object patch under probe at this position
                obj_patch = self._extract_probe_region(obj, position)

                # 2. Form exit wave
                psi = probe * obj_patch

                # 3. Forward propagate to far field
                psi_hat = fftshift(fft2(ifftshift(psi)))

                # Compute Fourier error for this position
                amp_measured = np.sqrt(np.maximum(measured, 0.0))
                amp_current = np.abs(psi_hat)
                numerator = np.sum((amp_measured - amp_current) ** 2)
                denominator = np.sum(measured) + self.epsilon
                total_fourier_error += numerator / denominator

                # 4. Fourier magnitude replacement
                psi_hat_corrected = self._fourier_magnitude_projection(psi_hat, measured)

                # 5. Inverse propagate corrected exit wave
                psi_corrected = fftshift(ifft2(ifftshift(psi_hat_corrected)))

                # 6. Compute the exit wave difference
                delta_psi = psi_corrected - psi

                # 7. Update the object patch
                obj_patch_updated = obj_patch + update_weight * delta_psi

                # 8. Insert updated patch back into the full object
                self._insert_probe_region(obj, obj_patch_updated, position)

                # Step B: Reflectivity prior enforcement (applied after every scan position)
                # Outside support S: O(r) = beta * O(r) + (1 - beta) * R_p
                # Inside support S: keep as is
                # We apply this to the full object (or at least the region affected by the probe)
                # For efficiency, apply only to the region that was just updated
                row_center, col_center = int(position[0]), int(position[1])
                probe_h, probe_w = self.probe_shape
                half_h, half_w = probe_h // 2, probe_w // 2

                row_start = max(row_center - half_h, 0)
                row_end = min(row_center - half_h + probe_h, self.object_shape[0])
                col_start = max(col_center - half_w, 0)
                col_end = min(col_center - half_w + probe_w, self.object_shape[1])

                # Extract the local support mask for this region
                local_support = self.support_mask[row_start:row_end, col_start:col_end]
                local_obj = obj[row_start:row_end, col_start:col_end]

                # Apply reflectivity prior outside support
                obj[row_start:row_end, col_start:col_end] = np.where(
                    local_support > 0.5,
                    local_obj,
                    self.beta * local_obj + (1.0 - self.beta) * self.R_substrate
                )

            mean_fourier_error = float(total_fourier_error / num_positions)

            return obj, mean_fourier_error

    def _standard_pie_epoch(
            self,
            obj_current: np.ndarray,
            measured_intensities: List[np.ndarray],
        ) -> Tuple[np.ndarray, float]:
            """
            Perform one epoch of standard ePIE algorithm (Eq. 6, Step A only).

            Same as rpPIE but without Step B (no reflectivity prior enforcement).

            Args:
                obj_current: Current object estimate, shape (N_obj, N_obj) complex.
                measured_intensities: List of J measured intensity arrays.

            Returns:
                Tuple of (updated object estimate, mean Fourier error).
            """
            obj = obj_current.copy()
            num_positions = len(self.scan_positions)

            # Randomly shuffle scan position indices for this epoch
            indices = np.arange(num_positions)
            self.rng.shuffle(indices)

            # Precompute probe-related quantities
            probe = self.probe
            P_conj = np.conj(probe)
            P_abs = np.abs(probe)
            P_abs_sq = P_abs ** 2
            P_max_abs = self.P_max  # scalar: max |P|
            alpha = self.alpha_regularization * (P_max_abs ** 2)

            # Update weight: (|P| / |P_max|) * (P* / (|P|^2 + alpha)) * beta
            # = beta * |P| * P* / (|P_max| * (|P|^2 + alpha))
            update_weight = self.beta * P_abs * P_conj / (P_max_abs * (P_abs_sq + alpha))

            total_fourier_error = 0.0

            for idx in indices:
                position = self.scan_positions[idx]
                measured = measured_intensities[idx]

                # Step A: ePIE-style object update (no Step B)

                # 1. Extract object patch under probe at this position
                obj_patch = self._extract_probe_region(obj, position)

                # 2. Form exit wave
                psi = probe * obj_patch

                # 3. Forward propagate to far field
                psi_hat = fftshift(fft2(ifftshift(psi)))

                # Compute Fourier error for this position
                amp_measured = np.sqrt(np.maximum(measured, 0.0))
                amp_current = np.abs(psi_hat)
                numerator = np.sum((amp_measured - amp_current) ** 2)
                denominator = np.sum(measured) + self.epsilon
                total_fourier_error += numerator / denominator

                # 4. Fourier magnitude replacement
                psi_hat_corrected = self._fourier_magnitude_projection(psi_hat, measured)

                # 5. Inverse propagate corrected exit wave
                psi_corrected = fftshift(ifft2(ifftshift(psi_hat_corrected)))

                # 6. Compute the exit wave difference
                delta_psi = psi_corrected - psi

                # 7. Update the object patch
                obj_patch_updated = obj_patch + update_weight * delta_psi

                # 8. Insert updated patch back into the full object
                self._insert_probe_region(obj, obj_patch_updated, position)

                # No Step B: no reflectivity prior enforcement

            mean_fourier_error = float(total_fourier_error / num_positions)

            return obj, mean_fourier_error

    def _compute_fourier_error(
            self,
            obj: np.ndarray,
            measured_intensities: List[np.ndarray],
            positions: Optional[np.ndarray] = None,
        ) -> float:
            """
            Compute the Fourier-space R-factor error metric.

            E = sum_j sum_k (sqrt(I_j(k)) - |F{P(r-R_j)*O(r)}|)^2
                / sum_j sum_k I_j(k)

            Args:
                obj: Current object estimate.
                measured_intensities: List of measured intensity patterns.
                positions: Scan positions array. If None, single-shot mode.

            Returns:
                Scalar Fourier error value.
            """
            total_numerator = 0.0
            total_denominator = 0.0

            if positions is None:
                # Single-shot CDI mode: one intensity pattern, no scanning
                # measured_intensities should be a list with one element or a single array
                if isinstance(measured_intensities, np.ndarray) and measured_intensities.ndim == 2:
                    intensities = [measured_intensities]
                else:
                    intensities = measured_intensities

                for measured in intensities:
                    # Forward propagate the full object
                    psi_hat = fftshift(fft2(ifftshift(obj)))

                    amp_measured = np.sqrt(np.maximum(measured, 0.0))
                    amp_current = np.abs(psi_hat)

                    total_numerator += np.sum((amp_measured - amp_current) ** 2)
                    total_denominator += np.sum(measured)
            else:
                # Ptychographic mode: iterate over all scan positions
                if isinstance(measured_intensities, np.ndarray) and measured_intensities.ndim == 3:
                    intensities = [measured_intensities[j] for j in range(measured_intensities.shape[0])]
                else:
                    intensities = measured_intensities

                probe = self.probe

                for j, position in enumerate(positions):
                    measured = intensities[j]

                    # Extract object patch under probe at this position
                    obj_patch = self._extract_probe_region(obj, position)

                    # Form exit wave
                    psi = probe * obj_patch

                    # Forward propagate to far field
                    psi_hat = fftshift(fft2(ifftshift(psi)))

                    amp_measured = np.sqrt(np.maximum(measured, 0.0))
                    amp_current = np.abs(psi_hat)

                    total_numerator += np.sum((amp_measured - amp_current) ** 2)
                    total_denominator += np.sum(measured)

            fourier_error = float(total_numerator / (total_denominator + self.epsilon))

            return fourier_error

    def _remove_phase_ambiguity(
            self,
            obj_recon: np.ndarray,
            obj_true: np.ndarray,
        ) -> np.ndarray:
            """
            Remove global phase ambiguity from reconstructed object by aligning
            to ground truth via least-squares phase offset estimation.

            Args:
                obj_recon: Reconstructed complex object.
                obj_true: Ground truth complex object.

            Returns:
                Phase-aligned reconstructed object.
            """
            # Extract phases within the support mask where the signal is meaningful
            if self.support_mask is not None:
                mask = self.support_mask > 0.5
            else:
                # Fallback: use all pixels with significant amplitude
                mask = np.abs(obj_true) > self.epsilon

            if not np.any(mask):
                # No valid pixels; return as-is
                return obj_recon

            phase_recon = np.angle(obj_recon[mask])
            phase_true = np.angle(obj_true[mask])

            # Compute the phase difference at each masked pixel
            phase_diff = phase_recon - phase_true

            # Estimate the global phase offset as the circular mean of phase differences
            # Using the complex exponential approach to properly handle wrapping:
            # phi_offset = angle( mean( exp(i * (phase_recon - phase_true)) ) )
            mean_phasor = np.mean(np.exp(1j * phase_diff))
            phi_offset = np.angle(mean_phasor)

            # Remove the global phase offset from the reconstructed object
            obj_aligned = obj_recon * np.exp(-1j * phi_offset)

            return obj_aligned

    def _initialize_object(self) -> np.ndarray:
            """
            Initialize the object estimate for iterative reconstruction.

            Default: O_0(r) = R_p everywhere (uniform substrate reflectivity).

            Returns:
                Initial object estimate, shape (N_obj, N_obj) complex.
            """
            # Initialize with uniform substrate reflectivity everywhere
            obj = np.full(self.object_shape, self.R_substrate, dtype=np.complex128)

            # Optionally, for pad regions, initialize with pad reflectivity (no phase)
            # This can help convergence by providing a better starting point
            if self.support_mask is not None:
                obj = np.where(
                    self.support_mask > 0.5,
                    self.R_pad,
                    self.R_substrate
                )

            return obj.astype(np.complex128)

    def setup(self) -> None:
            """
            Perform all pre-computation setup:
                - Compute material reflectivities via TMM
                - Build support mask
                - Generate probe function
                - Generate scan positions
                - Compute derived quantities (P_max, object_shape, etc.)
            """
            # Step 1: Compute substrate reflectivity via TMM
            # Substrate stack (no Cu): SiCN / SiO2 / Si (semi-infinite)
            substrate_layers = ["SiCN", "SiO2"]
            substrate_thicknesses = [
                self.layer_thicknesses["SiCN"],
                self.layer_thicknesses["SiO2"],
            ]
            self.R_substrate = self._compute_tmm_reflectivity(
                layer_materials=substrate_layers,
                layer_thicknesses_m=substrate_thicknesses,
                substrate_material="Si",
                incident_medium_n=1.0,
            )

            # Step 2: Compute Cu pad reflectivity via TMM
            # Pad stack: Cu (thick enough to be opaque at EUV, ~50 nm) / SiCN / SiO2 / Si
            # For Cu pads, we model Cu as a layer on top of the substrate stack.
            # Cu thickness: use a sufficiently thick layer (e.g., 50 nm) so it's effectively opaque
            cu_thickness = 50e-9  # 50 nm Cu, thick enough to dominate reflection at EUV
            pad_layers = ["Cu", "SiCN", "SiO2"]
            pad_thicknesses = [
                cu_thickness,
                self.layer_thicknesses["SiCN"],
                self.layer_thicknesses["SiO2"],
            ]
            self.R_pad = self._compute_tmm_reflectivity(
                layer_materials=pad_layers,
                layer_thicknesses_m=pad_thicknesses,
                substrate_material="Si",
                incident_medium_n=1.0,
            )

            # Step 3: Build support mask for the 15x15 periodic pad array
            self.support_mask = self._build_support_mask()

            # Step 4: Generate probe function (Airy or speckle)
            self.probe = self._generate_probe()

            # Step 5: Compute P_max (maximum absolute value of probe)
            self.P_max = float(np.max(np.abs(self.probe)))
            if self.P_max == 0.0:
                self.P_max = 1.0  # safety fallback

            # Step 6: Generate scan positions for ptychographic algorithms
            self.scan_positions = self._generate_scan_positions()

            # Step 7: Reset convergence history
            self.convergence_history = []

    def simulate_measurements(
            self,
            obj_true: np.ndarray,
        ) -> List[np.ndarray]:
            """
            Simulate noisy far-field diffraction intensity measurements
            from a ground truth object.

            Args:
                obj_true: Ground truth complex object, shape (N_obj, N_obj).

            Returns:
                List of J noisy intensity patterns, each shape (N_probe, N_probe).
            """
            measurements = []

            if self.algorithm in ("hio", "rpio"):
                # Single-shot CDI mode: one diffraction pattern from the full object
                # Forward propagate the entire object (no probe multiplication)
                psi_hat = fftshift(fft2(ifftshift(obj_true)))
                intensity = np.abs(psi_hat) ** 2

                # Add Gaussian noise
                if self.noise_sigma > 0.0:
                    noise = self.rng.normal(0.0, self.noise_sigma, size=intensity.shape)
                    intensity = intensity + noise
                    # Enforce non-negativity
                    intensity = np.maximum(intensity, 0.0)

                measurements.append(intensity.astype(np.float64))

            else:
                # Ptychographic mode: one diffraction pattern per scan position
                probe = self.probe

                for j in range(len(self.scan_positions)):
                    position = self.scan_positions[j]

                    # Extract the object region illuminated by the probe
                    obj_patch = self._extract_probe_region(obj_true, position)

                    # Form exit wave: element-wise multiplication of probe and object patch
                    exit_wave = probe * obj_patch

                    # Fraunhofer propagation via 2D FFT
                    psi_hat = fftshift(fft2(ifftshift(exit_wave)))

                    # Compute intensity: I(k) = |Psi_hat(k)|^2
                    intensity = np.abs(psi_hat) ** 2

                    # Add Gaussian noise
                    if self.noise_sigma > 0.0:
                        noise = self.rng.normal(0.0, self.noise_sigma, size=intensity.shape)
                        intensity = intensity + noise
                        # Enforce non-negativity
                        intensity = np.maximum(intensity, 0.0)

                    measurements.append(intensity.astype(np.float64))

            return measurements

    def solve(
            self,
            measured_intensities: np.ndarray,
            support_mask: Optional[np.ndarray] = None,
            probe: Optional[np.ndarray] = None,
            scan_positions: Optional[np.ndarray] = None,
            R_substrate: Optional[complex] = None,
        ) -> np.ndarray:
            """
            Solve the inverse problem: recover O(r) from measured intensities I(k).

            Dispatches to the appropriate algorithm (hio, rpio, pie, rppie)
            based on self.algorithm.

            Args:
                measured_intensities: Measured diffraction data.
                    For single-shot (hio/rpio): shape (N, N) single intensity pattern.
                    For ptychographic (pie/rppie): shape (J, N_probe, N_probe) stack
                        or list of J arrays.
                support_mask: Binary support mask. If None, uses self.support_mask.
                probe: Complex probe function. If None, uses self.probe.
                scan_positions: Array of scan positions shape (J, 2). If None, uses
                                self.scan_positions.
                R_substrate: Known substrate reflectivity. If None, uses self.R_substrate.

            Returns:
                Reconstructed complex object O(r), shape (N_obj, N_obj).
            """
            # Override internal state with provided arguments if given
            if support_mask is not None:
                self.support_mask = support_mask
            if probe is not None:
                self.probe = probe
                self.P_max = float(np.max(np.abs(self.probe)))
                if self.P_max == 0.0:
                    self.P_max = 1.0
            if scan_positions is not None:
                self.scan_positions = scan_positions
            if R_substrate is not None:
                self.R_substrate = R_substrate

            # Convert measured_intensities to the appropriate format
            if self.algorithm in ("hio", "rpio"):
                # Single-shot CDI mode: expect a single 2D intensity pattern
                if isinstance(measured_intensities, list):
                    if len(measured_intensities) == 1:
                        intensity_single = measured_intensities[0]
                    else:
                        intensity_single = measured_intensities[0]
                elif isinstance(measured_intensities, np.ndarray):
                    if measured_intensities.ndim == 3:
                        # If a stack is provided, use the first pattern
                        intensity_single = measured_intensities[0]
                    elif measured_intensities.ndim == 2:
                        intensity_single = measured_intensities
                    else:
                        intensity_single = measured_intensities
                else:
                    intensity_single = measured_intensities

                intensity_single = np.asarray(intensity_single, dtype=np.float64)

                # For single-shot CDI, the object shape must match the intensity shape
                # Resize object shape if needed
                if intensity_single.shape != self.object_shape:
                    self.object_shape = intensity_single.shape
                    # Rebuild support mask for new object shape
                    self.support_mask = self._build_support_mask()

                # Initialize object estimate
                obj = self._initialize_object()

                # Reset convergence history
                self.convergence_history = []

                # Run iterative algorithm
                for iteration in range(self.max_iterations):
                    if self.algorithm == "rpio":
                        obj, fourier_error = self._rpIO_step(obj, intensity_single)
                    elif self.algorithm == "hio":
                        obj, fourier_error = self._standard_hio_step(obj, intensity_single)
                    else:
                        raise ValueError(f"Unknown single-shot algorithm: {self.algorithm}")

                    self.convergence_history.append(fourier_error)

                    # Check for convergence
                    if fourier_error < self.convergence_threshold:
                        break

                return obj.astype(np.complex128)

            elif self.algorithm in ("pie", "rppie"):
                # Ptychographic mode: expect a list/stack of intensity patterns
                if isinstance(measured_intensities, np.ndarray):
                    if measured_intensities.ndim == 3:
                        intensities_list = [
                            measured_intensities[j].astype(np.float64)
                            for j in range(measured_intensities.shape[0])
                        ]
                    elif measured_intensities.ndim == 2:
                        # Single pattern provided; wrap in list
                        intensities_list = [measured_intensities.astype(np.float64)]
                    else:
                        # Try to handle allow_pickle=True loaded object arrays
                        intensities_list = []
                        for item in measured_intensities:
                            intensities_list.append(np.asarray(item, dtype=np.float64))
                elif isinstance(measured_intensities, list):
                    intensities_list = [
                        np.asarray(m, dtype=np.float64) for m in measured_intensities
                    ]
                else:
                    intensities_list = [np.asarray(measured_intensities, dtype=np.float64)]

                # Ensure scan positions match the number of measurements
                num_measurements = len(intensities_list)
                if self.scan_positions is not None and len(self.scan_positions) != num_measurements:
                    # Regenerate scan positions or truncate to match
                    if num_measurements < len(self.scan_positions):
                        self.scan_positions = self.scan_positions[:num_measurements]
                    else:
                        # Generate additional positions or regenerate
                        self.scan_positions = self._generate_scan_positions()
                        if len(self.scan_positions) != num_measurements:
                            # Truncate or extend as needed
                            if num_measurements <= len(self.scan_positions):
                                self.scan_positions = self.scan_positions[:num_measurements]
                            else:
                                # Repeat positions cyclically if we have fewer positions than measurements
                                repeats = int(np.ceil(num_measurements / len(self.scan_positions)))
                                self.scan_positions = np.tile(self.scan_positions, (repeats, 1))[:num_measurements]

                # Ensure probe shape matches the measurement shape
                if len(intensities_list) > 0:
                    meas_shape = intensities_list[0].shape
                    if self.probe.shape != meas_shape:
                        # Resize probe to match measurement dimensions
                        old_probe = self.probe
                        new_probe = np.zeros(meas_shape, dtype=np.complex128)
                        # Center the old probe in the new array (or crop)
                        old_h, old_w = old_probe.shape
                        new_h, new_w = meas_shape
                        # Compute overlap region
                        copy_h = min(old_h, new_h)
                        copy_w = min(old_w, new_w)
                        src_r = (old_h - copy_h) // 2
                        src_c = (old_w - copy_w) // 2
                        dst_r = (new_h - copy_h) // 2
                        dst_c = (new_w - copy_w) // 2
                        new_probe[dst_r:dst_r+copy_h, dst_c:dst_c+copy_w] = \
                            old_probe[src_r:src_r+copy_h, src_c:src_c+copy_w]
                        self.probe = new_probe
                        self.probe_shape = meas_shape
                        self.P_max = float(np.max(np.abs(self.probe)))
                        if self.P_max == 0.0:
                            self.P_max = 1.0

                # Initialize object estimate
                obj = self._initialize_object()

                # Reset convergence history
                self.convergence_history = []

                # Run iterative epochs
                for epoch in range(self.max_iterations):
                    if self.algorithm == "rppie":
                        obj, fourier_error = self._rpPIE_epoch(obj, intensities_list)
                    elif self.algorithm == "pie":
                        obj, fourier_error = self._standard_pie_epoch(obj, intensities_list)
                    else:
                        raise ValueError(f"Unknown ptychographic algorithm: {self.algorithm}")

                    self.convergence_history.append(fourier_error)

                    # Check for convergence
                    if fourier_error < self.convergence_threshold:
                        break

                return obj.astype(np.complex128)

            else:
                raise ValueError(
                    f"Unknown algorithm '{self.algorithm}'. "
                    f"Choose from 'hio', 'rpio', 'pie', 'rppie'."
                )

    def evaluate(
            self,
            obj_recon: np.ndarray,
            obj_true: np.ndarray,
        ) -> Dict[str, float]:
            """
            Evaluate reconstruction quality against ground truth.

            Computes:
                - Phase RMSE (within support, after global phase removal)
                - Recession depth RMSE (nm)
                - PSNR (dB)
                - Fourier R-factor

            Args:
                obj_recon: Reconstructed complex object.
                obj_true: Ground truth complex object.

            Returns:
                Dictionary with metric names as keys and values as floats.
            """
            metrics: Dict[str, float] = {}

            # Step 1: Remove global phase ambiguity by aligning reconstructed to true
            obj_aligned = self._remove_phase_ambiguity(obj_recon, obj_true)

            # Step 2: Build support mask if not already available
            if self.support_mask is None:
                self.support_mask = self._build_support_mask()

            mask = self.support_mask > 0.5

            # Ensure shapes match; if not, crop/pad to common region
            if obj_aligned.shape != obj_true.shape:
                # Use the minimum overlapping region
                min_h = min(obj_aligned.shape[0], obj_true.shape[0])
                min_w = min(obj_aligned.shape[1], obj_true.shape[1])
                obj_aligned = obj_aligned[:min_h, :min_w]
                obj_true_cropped = obj_true[:min_h, :min_w]
                mask = mask[:min_h, :min_w]
            else:
                obj_true_cropped = obj_true

            # Step 3: Phase RMSE within support (after global phase removal)
            if np.any(mask):
                phase_recon = np.angle(obj_aligned[mask])
                phase_true = np.angle(obj_true_cropped[mask])

                # Compute phase difference, handling wrapping via circular difference
                phase_diff = phase_recon - phase_true
                # Wrap to [-pi, pi]
                phase_diff = np.angle(np.exp(1j * phase_diff))

                phase_mse = float(np.mean(phase_diff ** 2))
                phase_rmse = float(np.sqrt(phase_mse))
            else:
                phase_rmse = float('nan')
                phase_mse = float('nan')

            metrics["phase_rmse_rad"] = phase_rmse

            # Step 4: Recession depth RMSE (nm)
            # h = lambda / (4*pi) * phase
            # RMSE_h = lambda / (4*pi) * RMSE_phase
            depth_rmse_m = self.wavelength / (4.0 * np.pi) * phase_rmse
            depth_rmse_nm = depth_rmse_m * 1e9
            metrics["recession_depth_rmse_nm"] = float(depth_rmse_nm)

            # Step 5: PSNR (dB) based on phase within support
            if np.any(mask):
                phase_true_support = np.angle(obj_true_cropped[mask])
                # Wrap phase_true to get the dynamic range
                phase_range = float(np.max(phase_true_support) - np.min(phase_true_support))
                if phase_range > 0 and phase_mse > 0:
                    psnr = 10.0 * np.log10(phase_range ** 2 / phase_mse)
                elif phase_mse == 0:
                    psnr = float('inf')
                else:
                    psnr = 0.0
            else:
                psnr = float('nan')

            metrics["psnr_db"] = float(psnr)

            # Step 6: Amplitude RMSE within support
            if np.any(mask):
                amp_recon = np.abs(obj_aligned[mask])
                amp_true = np.abs(obj_true_cropped[mask])
                amp_rmse = float(np.sqrt(np.mean((amp_recon - amp_true) ** 2)))
            else:
                amp_rmse = float('nan')

            metrics["amplitude_rmse"] = amp_rmse

            # Step 7: Full complex RMSE within support
            if np.any(mask):
                complex_diff = obj_aligned[mask] - obj_true_cropped[mask]
                complex_rmse = float(np.sqrt(np.mean(np.abs(complex_diff) ** 2)))
            else:
                complex_rmse = float('nan')

            metrics["complex_rmse"] = complex_rmse

            # Step 8: Fourier R-factor (if convergence history is available)
            if len(self.convergence_history) > 0:
                metrics["final_fourier_error"] = float(self.convergence_history[-1])
            else:
                metrics["final_fourier_error"] = float('nan')

            # Step 9: Number of iterations used
            metrics["num_iterations"] = float(len(self.convergence_history))

            return metrics

    def extract_recession_depths(
            self,
            obj: np.ndarray,
        ) -> np.ndarray:
            """
            Extract per-pad recession depths from the reconstructed object's phase.

            h(m,n) = lambda / (4*pi) * mean(phase(O(r))) for r in pad (m,n)

            Args:
                obj: Complex object array, shape (N_obj, N_obj).

            Returns:
                Recession depths array of shape (num_pads, num_pads) in meters.
            """
            recession_depths = np.zeros((self.num_pads, self.num_pads), dtype=np.float64)

            for m in range(self.num_pads):
                for n in range(self.num_pads):
                    row_start = self.array_offset + m * self.pitch_pixels
                    row_end = row_start + self.pad_size_pixels
                    col_start = self.array_offset + n * self.pitch_pixels
                    col_end = col_start + self.pad_size_pixels

                    # Clip to object boundaries
                    row_start_clipped = max(row_start, 0)
                    row_end_clipped = min(row_end, obj.shape[0])
                    col_start_clipped = max(col_start, 0)
                    col_end_clipped = min(col_end, obj.shape[1])

                    if row_start_clipped < row_end_clipped and col_start_clipped < col_end_clipped:
                        # Extract the pad region from the object
                        pad_region = obj[row_start_clipped:row_end_clipped,
                                         col_start_clipped:col_end_clipped]

                        # Compute the mean phase of the pad region
                        # Use circular mean to handle potential phase wrapping
                        mean_phasor = np.mean(pad_region / (np.abs(pad_region) + self.epsilon))
                        mean_phase = np.angle(mean_phasor)

                        # Convert phase to recession depth: h = lambda / (4*pi) * phase
                        recession_depths[m, n] = self.wavelength / (4.0 * np.pi) * mean_phase
                    else:
                        recession_depths[m, n] = 0.0

            return recession_depths

    def extract_cross_section(
            self,
            obj: np.ndarray,
            row_index: int,
            axis: int = 0,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Extract a 1D cross-sectional phase profile through the object.

            Args:
                obj: Complex object array, shape (N_obj, N_obj).
                row_index: Index of the row (or column) to extract.
                axis: 0 for horizontal cut (along columns at given row),
                      1 for vertical cut (along rows at given column).

            Returns:
                Tuple of (position array in nm, phase profile in radians).
            """
            if axis == 0:
                # Horizontal cut: extract the specified row across all columns
                if row_index < 0 or row_index >= obj.shape[0]:
                    raise ValueError(
                        f"row_index {row_index} out of bounds for object with "
                        f"{obj.shape[0]} rows."
                    )
                cross_section_complex = obj[row_index, :]
                num_pixels = obj.shape[1]
            elif axis == 1:
                # Vertical cut: extract the specified column across all rows
                if row_index < 0 or row_index >= obj.shape[1]:
                    raise ValueError(
                        f"row_index (column index) {row_index} out of bounds for object with "
                        f"{obj.shape[1]} columns."
                    )
                cross_section_complex = obj[:, row_index]
                num_pixels = obj.shape[0]
            else:
                raise ValueError(f"axis must be 0 or 1, got {axis}.")

            # Compute the phase profile in radians
            phase_profile = np.angle(cross_section_complex).astype(np.float64)

            # Compute the position array in nanometers
            # Each pixel corresponds to self.pixel_size meters
            position_m = np.arange(num_pixels, dtype=np.float64) * self.pixel_size
            position_nm = position_m * 1e9  # convert meters to nanometers

            return position_nm, phase_profile



if __name__ == "__main__":
    # Load input data
    input_data = np.load("dataset/input.npy", allow_pickle=True)

    # Determine the structure of the input data
    # It could be a single 2D array (single-shot CDI) or a list/3D array (ptychographic)
    if isinstance(input_data, np.ndarray):
        if input_data.dtype == object:
            # Object array - convert to list of arrays
            measurements = [np.asarray(x, dtype=np.float64) for x in input_data]
            num_measurements = len(measurements)
            if num_measurements == 1:
                algo = "rpio"
            else:
                algo = "rppie"
        elif input_data.ndim == 2:
            measurements = input_data
            num_measurements = 1
            algo = "rpio"
        elif input_data.ndim == 3:
            num_measurements = input_data.shape[0]
            measurements = input_data
            if num_measurements == 1:
                algo = "rpio"
            else:
                algo = "rppie"
        else:
            measurements = input_data
            num_measurements = 1
            algo = "rpio"
    else:
        measurements = input_data
        num_measurements = 1
        algo = "rpio"

    # Try multiple algorithm configurations and pick the best result
    # Start with rpPIE as it generally performs best for ptychographic data
    best_result = None
    best_error = float('inf')

    for algorithm in [algo]:
        for beta in [0.9]:
            for probe_type in ["airy"]:
                try:
                    solver = InverseSolver(
                        wavelength=13.5e-9,
                        pixel_size=27e-9,
                        pad_size_pixels=22,
                        pitch_pixels=30,
                        num_pads=15,
                        scan_step_pixels=40,
                        beta=beta,
                        alpha_regularization=1e-3,
                        epsilon=1e-8,
                        max_iterations=300,
                        algorithm=algorithm,
                        probe_type=probe_type,
                        noise_sigma=0.0,
                        random_seed=42,
                        recession_range=(1e-9, 3e-9),
                        probe_diameter_pixels=160,
                        convergence_threshold=1e-10,
                    )

                    # Setup solver internals
                    solver.setup()

                    # Solve the inverse problem
                    result = solver.solve(measured_intensities=measurements)

                    # Track convergence
                    if len(solver.convergence_history) > 0:
                        final_error = solver.convergence_history[-1]
                    else:
                        final_error = float('inf')

                    if final_error < best_error:
                        best_error = final_error
                        best_result = result

                except Exception as e:
                    print(f"Algorithm {algorithm} with beta={beta}, probe={probe_type} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    if best_result is None:
        # Fallback: try with default settings
        solver = InverseSolver(
            wavelength=13.5e-9,
            pixel_size=27e-9,
            pad_size_pixels=22,
            pitch_pixels=30,
            num_pads=15,
            scan_step_pixels=40,
            beta=0.9,
            alpha_regularization=1e-3,
            max_iterations=300,
            algorithm="rppie",
            probe_type="airy",
            random_seed=42,
        )
        solver.setup()
        best_result = solver.solve(measured_intensities=measurements)

    # Save reconstructed object
    np.save("output.npy", best_result)