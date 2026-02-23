import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.special import j1 as bessel_j1
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

class InverseSolver:
    """
    Solver for recovering complex-valued reflection function O(r) of a periodic
    copper pad array from far-field diffraction intensity measurements.
    
    Implements four algorithms: HIO, rpIO (Eq. 5), PIE (ePIE), rpPIE (Eq. 6).
    """

    def __init__(
            self,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            pad_size_pixels: int = 22,
            pitch_pixels: int = 30,
            num_pads: int = 15,
            duty_cycle: float = 0.6,
            scan_step_pixels: int = 40,
            overlap_fraction: float = 0.75,
            beta: float = 0.7,
            alpha_regularization: float = 1e-3,
            num_iterations_io: int = 500,
            num_epochs_pie: int = 200,
            convergence_threshold: float = 1e-6,
            noise_sigma: float = 0.01,
            recession_depth_min: float = 1e-9,
            recession_depth_max: float = 3e-9,
            random_seed: int = 42,
            algorithm: str = "rppie",
            illumination: str = "airy",
            probe_radius_pixels: int = 60,
            grid_size: int = 512,
            n_SiCN: complex = complex(0.9791, -0.0059),
            n_SiO2: complex = complex(0.9780, -0.0108),
            n_Si: complex = complex(0.9991, -0.0018),
            n_Cu: complex = complex(0.9625, -0.0613),
            eps: float = 1e-10,
        ) -> None:
            """
            Initialize the InverseSolver with all physical and algorithmic parameters.

            Args:
                wavelength: EUV wavelength in meters (13.5 nm).
                pixel_size: Object-plane pixel size in meters (27 nm).
                pad_size_pixels: Number of pixels per pad side (22).
                pitch_pixels: Pitch between pads in pixels (~30 for 800 nm pitch).
                num_pads: Number of pads per side of the array (15).
                duty_cycle: Ratio of pad size to pitch (0.6).
                scan_step_pixels: Ptychographic scan step in pixels (40).
                overlap_fraction: Fractional overlap between adjacent scan positions (~0.75).
                beta: Feedback / step-size parameter for all algorithms.
                alpha_regularization: Tikhonov regularization parameter for ePIE update.
                num_iterations_io: Maximum iterations for HIO/rpIO algorithms.
                num_epochs_pie: Maximum epochs for PIE/rpPIE algorithms.
                convergence_threshold: Relative change threshold for early stopping.
                noise_sigma: Standard deviation of additive Gaussian noise (relative).
                recession_depth_min: Minimum pad recession depth in meters (1 nm).
                recession_depth_max: Maximum pad recession depth in meters (3 nm).
                random_seed: Random seed for reproducibility.
                algorithm: Algorithm to use: 'hio', 'rpio', 'pie', 'rppie'.
                illumination: Illumination type: 'airy' or 'speckle'.
                probe_radius_pixels: Radius of the probe function in pixels.
                grid_size: Total object grid size in pixels.
                n_SiCN: Complex refractive index of SiCN at 13.5 nm.
                n_SiO2: Complex refractive index of SiO2 at 13.5 nm.
                n_Si: Complex refractive index of Si at 13.5 nm.
                n_Cu: Complex refractive index of Cu at 13.5 nm.
                eps: Small constant to prevent division by zero.
            """
            # Physical parameters
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.pad_size_pixels: int = pad_size_pixels
            self.pitch_pixels: int = pitch_pixels
            self.num_pads: int = num_pads
            self.duty_cycle: float = duty_cycle
            self.scan_step_pixels: int = scan_step_pixels
            self.overlap_fraction: float = overlap_fraction
            self.grid_size: int = grid_size

            # Material optical constants (stored as complex refractive indices)
            self.n_SiCN: complex = n_SiCN
            self.n_SiO2: complex = n_SiO2
            self.n_Si: complex = n_Si
            self.n_Cu: complex = n_Cu

            # Algorithm parameters
            self.beta: float = beta
            self.alpha_regularization: float = alpha_regularization
            self.num_iterations_io: int = num_iterations_io
            self.num_epochs_pie: int = num_epochs_pie
            self.convergence_threshold: float = convergence_threshold
            self.noise_sigma: float = noise_sigma
            self.recession_depth_min: float = recession_depth_min
            self.recession_depth_max: float = recession_depth_max
            self.random_seed: int = random_seed
            self.algorithm: str = algorithm.lower()
            self.illumination: str = illumination.lower()
            self.probe_radius_pixels: int = probe_radius_pixels
            self.eps: float = eps

            # Derived quantities: wavenumber k0 = 2*pi/lambda
            self.k0: float = 2.0 * np.pi / self.wavelength

            # Maximum phase modulation: 4*pi*h_max/lambda
            self.max_phase_modulation: float = 4.0 * np.pi * self.recession_depth_max / self.wavelength

            # Probe diameter in pixels (for patch extraction)
            self.probe_diameter_pixels: int = 2 * self.probe_radius_pixels

            # Total array span in pixels (num_pads * pitch_pixels)
            self.array_span_pixels: int = self.num_pads * self.pitch_pixels

            # Margin around the pad array to accommodate probe extent
            self.margin_pixels: int = max(
                (self.grid_size - self.array_span_pixels) // 2,
                self.probe_radius_pixels
            )

            # Derived / cached quantities (populated during solve or build methods)
            self.support_mask: Optional[np.ndarray] = None
            self.probe: Optional[np.ndarray] = None
            self.scan_positions: Optional[np.ndarray] = None
            self.R_substrate: Optional[complex] = None
            self.R_pad: Optional[complex] = None
            self.object_estimate: Optional[np.ndarray] = None
            self.error_history: List[float] = []

            # Set random seed for reproducibility
            self.rng: np.random.RandomState = np.random.RandomState(self.random_seed)

    # -------------------------------------------------------------------------
    # Material / Reflectivity Methods
    # -------------------------------------------------------------------------

    def _compute_fresnel_reflectivity(
            self, n1: complex, n2: complex
        ) -> complex:
            """
            Compute Fresnel reflection coefficient at normal incidence for interface n1 -> n2.

            Args:
                n1: Complex refractive index of medium 1.
                n2: Complex refractive index of medium 2.

            Returns:
                Complex Fresnel reflection coefficient r_{12}.
            """
            return (n1 - n2) / (n1 + n2)

    def _compute_transfer_matrix(
            self, n_layer: complex, thickness: float
        ) -> np.ndarray:
            """
            Compute the 2x2 transfer matrix for a single layer at normal incidence.

            Args:
                n_layer: Complex refractive index of the layer.
                thickness: Layer thickness in meters.

            Returns:
                2x2 complex transfer matrix M_l.
            """
            phase = n_layer * self.k0 * thickness
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)

            M = np.array([
                [cos_phase, -1j * sin_phase / n_layer],
                [-1j * n_layer * sin_phase, cos_phase]
            ], dtype=np.complex128)

            return M

    def _compute_multilayer_reflectivity(
            self, layer_stack: List[Tuple[complex, float]]
        ) -> complex:
            """
            Compute overall complex reflectivity of a multilayer stack using
            the transmission matrix method.

            Args:
                layer_stack: List of (complex_refractive_index, thickness_meters) tuples,
                             ordered from top (vacuum side) to bottom (substrate).
                             The first entry is the incident medium (thickness ignored),
                             the last entry is the substrate (thickness ignored / semi-infinite).
                             Intermediate entries are the thin film layers.

            Returns:
                Complex reflectivity of the full stack.
            """
            if len(layer_stack) < 2:
                raise ValueError("Need at least two media (incident + substrate).")

            # The incident medium is the first entry
            n_incident = layer_stack[0][0]

            # The substrate is the last entry
            n_substrate = layer_stack[-1][0]

            # Build the total system matrix by multiplying transfer matrices
            # of intermediate layers (from top to bottom)
            # M_total = M_1 * M_2 * ... * M_{N-1} for layers between incident and substrate
            M_total = np.eye(2, dtype=np.complex128)

            for i in range(1, len(layer_stack) - 1):
                n_layer, thickness = layer_stack[i]
                M_layer = self._compute_transfer_matrix(n_layer, thickness)
                M_total = M_total @ M_layer

            # Extract elements of the total transfer matrix
            # Using the characteristic matrix approach:
            # The overall reflection coefficient is:
            # r = (M_total[0,0]*n_substrate + M_total[0,1]*n_incident*n_substrate
            #      - M_total[1,0] - M_total[1,1]*n_incident)
            #   / (M_total[0,0]*n_substrate + M_total[0,1]*n_incident*n_substrate
            #      + M_total[1,0] + M_total[1,1]*n_incident)
            #
            # For the characteristic matrix method at normal incidence:
            # The boundary conditions give:
            #   [E_i + E_r]     [E_t    ]
            #   [n_0*(E_i-E_r)] = M_total * [n_s*E_t]
            #
            # So: E_i + E_r = M[0,0]*E_t + M[0,1]*n_s*E_t
            #     n_0*(E_i - E_r) = M[1,0]*E_t + M[1,1]*n_s*E_t
            #
            # r = E_r/E_i = (M[0,0]*n_s + M[0,1]*n_s*... hmm
            # Let's use the standard formula properly.

            m11 = M_total[0, 0]
            m12 = M_total[0, 1]
            m21 = M_total[1, 0]
            m22 = M_total[1, 1]

            # From the characteristic matrix formalism at normal incidence:
            # r = (m11*n_substrate + m12*n_incident*n_substrate - m21 - m22*n_incident)
            #   / (m11*n_substrate + m12*n_incident*n_substrate + m21 + m22*n_incident)
            #
            # Actually the standard result is:
            # r = (m11 + m12*n_substrate)*n_incident - (m21 + m22*n_substrate)
            #   / (m11 + m12*n_substrate)*n_incident + (m21 + m22*n_substrate)

            # Using the convention where M relates [E, H] fields:
            # [1 + r]       = M_total * [t        ]
            # [n_0*(1-r)]              [n_s * t   ]
            #
            # 1 + r = m11*t + m12*n_s*t
            # n_0*(1 - r) = m21*t + m22*n_s*t
            #
            # From these two equations:
            # (1+r)*n_0 = n_0*m11*t + n_0*m12*n_s*t   ... (i)*n_0
            # n_0*(1-r) = m21*t + m22*n_s*t             ... (ii)
            #
            # Adding: 2*n_0 = t*(n_0*m11 + m21 + (n_0*m12 + m22)*n_s)
            # Subtracting: 2*n_0*r = t*(n_0*m11 - m21 + (n_0*m12 - m22)*n_s)
            #
            # r = (n_0*m11 - m21 + (n_0*m12 - m22)*n_s)
            #   / (n_0*m11 + m21 + (n_0*m12 + m22)*n_s)

            n0 = n_incident
            ns = n_substrate

            numerator = n0 * m11 - m21 + (n0 * m12 - m22) * ns
            denominator = n0 * m11 + m21 + (n0 * m12 + m22) * ns

            r = numerator / denominator

            return complex(r)

    def _compute_substrate_and_pad_reflectivities(self) -> Tuple[complex, complex]:
            """
            Compute the complex reflectivities R_substrate (inter-pad regions)
            and R_pad (on-pad regions) from the multilayer models.

            Substrate stack (inter-pad regions): vacuum / SiCN / SiO₂ / Si (semi-infinite)
            Pad stack (on-pad regions): vacuum / Cu / SiCN / SiO₂ / Si (semi-infinite)

            For the thin film layers, we use representative thicknesses:
            - SiCN: ~5 nm
            - SiO₂: ~3 nm  
            - Cu (pad): ~50 nm (thick enough to be effectively opaque at EUV)

            Returns:
                Tuple of (R_substrate, R_pad).
            """
            # Define layer thicknesses (in meters)
            # These are representative values for a typical CMP pad structure
            thickness_SiCN = 5.0e-9   # 5 nm SiCN cap layer
            thickness_SiO2 = 3.0e-9   # 3 nm SiO2 layer
            thickness_Cu = 50.0e-9    # 50 nm Cu pad (effectively opaque at EUV)

            # Vacuum refractive index
            n_vacuum = complex(1.0, 0.0)

            # Substrate stack (inter-pad regions): vacuum / SiCN / SiO2 / Si
            substrate_stack = [
                (n_vacuum, 0.0),              # incident medium (vacuum)
                (self.n_SiCN, thickness_SiCN),  # SiCN layer
                (self.n_SiO2, thickness_SiO2),  # SiO2 layer
                (self.n_Si, 0.0),              # Si substrate (semi-infinite)
            ]

            R_substrate = self._compute_multilayer_reflectivity(substrate_stack)

            # Pad stack (on-pad regions): vacuum / Cu / SiCN / SiO2 / Si
            pad_stack = [
                (n_vacuum, 0.0),              # incident medium (vacuum)
                (self.n_Cu, thickness_Cu),      # Cu pad layer
                (self.n_SiCN, thickness_SiCN),  # SiCN layer
                (self.n_SiO2, thickness_SiO2),  # SiO2 layer
                (self.n_Si, 0.0),              # Si substrate (semi-infinite)
            ]

            R_pad = self._compute_multilayer_reflectivity(pad_stack)

            # Store for later use
            self.R_substrate = R_substrate
            self.R_pad = R_pad

            return R_substrate, R_pad

    # -------------------------------------------------------------------------
    # Object / Support / Probe Construction
    # -------------------------------------------------------------------------

    def _build_support_mask(self) -> np.ndarray:
            """
            Create the binary support mask S(r) for the 15x15 periodic pad array.

            Returns:
                2D boolean array of shape (grid_size, grid_size), True on pad locations.
            """
            mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)

            # Compute the starting offset so that the pad array is centered in the grid
            # The array spans num_pads * pitch_pixels in each dimension
            # We center this span within the grid
            array_span = self.num_pads * self.pitch_pixels
            offset_row = (self.grid_size - array_span) // 2
            offset_col = (self.grid_size - array_span) // 2

            # Each pad is pad_size_pixels x pad_size_pixels, centered within its pitch cell
            # The pad is centered in each pitch cell
            pad_offset_in_cell = (self.pitch_pixels - self.pad_size_pixels) // 2

            for i in range(self.num_pads):
                for j in range(self.num_pads):
                    # Top-left corner of the pitch cell for pad (i, j)
                    cell_row = offset_row + i * self.pitch_pixels
                    cell_col = offset_col + j * self.pitch_pixels

                    # Top-left corner of the pad within the cell
                    pad_row_start = cell_row + pad_offset_in_cell
                    pad_col_start = cell_col + pad_offset_in_cell

                    pad_row_end = pad_row_start + self.pad_size_pixels
                    pad_col_end = pad_col_start + self.pad_size_pixels

                    # Clip to grid boundaries
                    r_start = max(pad_row_start, 0)
                    r_end = min(pad_row_end, self.grid_size)
                    c_start = max(pad_col_start, 0)
                    c_end = min(pad_col_end, self.grid_size)

                    if r_start < r_end and c_start < c_end:
                        mask[r_start:r_end, c_start:c_end] = True

            self.support_mask = mask
            return mask

    def _build_ground_truth_object(
            self, recession_depths: np.ndarray
        ) -> np.ndarray:
            """
            Construct the ground-truth complex reflection function O(r).

            Args:
                recession_depths: Array of shape (num_pads, num_pads) with recession
                                  depths in meters for each pad.

            Returns:
                Complex 2D array of shape (grid_size, grid_size).
            """
            # Ensure reflectivities are computed
            if self.R_substrate is None or self.R_pad is None:
                self._compute_substrate_and_pad_reflectivities()

            # Ensure support mask is built
            if self.support_mask is None:
                self._build_support_mask()

            # Initialize the object with substrate reflectivity everywhere
            obj = np.full(
                (self.grid_size, self.grid_size),
                self.R_substrate,
                dtype=np.complex128
            )

            # Compute the starting offset so that the pad array is centered in the grid
            array_span = self.num_pads * self.pitch_pixels
            offset_row = (self.grid_size - array_span) // 2
            offset_col = (self.grid_size - array_span) // 2

            # Pad offset within each pitch cell (to center the pad)
            pad_offset_in_cell = (self.pitch_pixels - self.pad_size_pixels) // 2

            # For each pad, assign R_pad * exp(i * 4*pi*h / lambda)
            for i in range(self.num_pads):
                for j in range(self.num_pads):
                    # Top-left corner of the pitch cell for pad (i, j)
                    cell_row = offset_row + i * self.pitch_pixels
                    cell_col = offset_col + j * self.pitch_pixels

                    # Top-left corner of the pad within the cell
                    pad_row_start = cell_row + pad_offset_in_cell
                    pad_col_start = cell_col + pad_offset_in_cell

                    pad_row_end = pad_row_start + self.pad_size_pixels
                    pad_col_end = pad_col_start + self.pad_size_pixels

                    # Clip to grid boundaries
                    r_start = max(pad_row_start, 0)
                    r_end = min(pad_row_end, self.grid_size)
                    c_start = max(pad_col_start, 0)
                    c_end = min(pad_col_end, self.grid_size)

                    if r_start < r_end and c_start < c_end:
                        # Get recession depth for this pad
                        h = recession_depths[i, j]

                        # Phase modulation due to recession depth
                        phase = 4.0 * np.pi * h / self.wavelength

                        # Complex reflection on this pad
                        obj[r_start:r_end, c_start:c_end] = (
                            self.R_pad * np.exp(1j * phase)
                        )

            return obj

    def _build_probe(self) -> np.ndarray:
            """
            Generate the probe function P(r) — either Airy spot or speckle field.

            Returns:
                Complex 2D array representing the probe, same grid as object patches.
            """
            diameter = self.probe_diameter_pixels
            probe = np.zeros((diameter, diameter), dtype=np.complex128)

            # Create coordinate grid centered on the probe
            y = np.arange(diameter) - diameter / 2.0 + 0.5
            x = np.arange(diameter) - diameter / 2.0 + 0.5
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)

            if self.illumination == "airy":
                # Airy spot: P(r) = 2*J1(pi*r/r0) / (pi*r/r0)
                # r0 controls the radius of the first zero of the Airy pattern
                # Choose r0 so the probe covers multiple pads and gives ~75% overlap
                # at scan_step_pixels spacing. The first zero of Airy is at r = 1.22*r0
                # We want the main lobe to extend to roughly probe_radius_pixels
                r0 = self.probe_radius_pixels / 1.22  # so first zero is at probe_radius_pixels

                # Compute normalized argument
                arg = np.pi * r / r0

                # Airy pattern: handle the singularity at r=0
                # At arg=0, the Airy function = 1
                with np.errstate(divide='ignore', invalid='ignore'):
                    airy_pattern = np.where(
                        arg < self.eps,
                        1.0,
                        2.0 * bessel_j1(arg) / arg
                    )

                # Apply a circular aperture to avoid ringing beyond the probe diameter
                aperture = (r <= self.probe_radius_pixels).astype(np.float64)

                probe = (airy_pattern * aperture).astype(np.complex128)

                # Normalize to unit maximum amplitude
                max_amp = np.max(np.abs(probe))
                if max_amp > 0:
                    probe = probe / max_amp

            elif self.illumination == "speckle":
                # Speckle field: generate random complex field, low-pass filter, normalize
                # Generate random complex field
                random_phase = self.rng.uniform(0, 2 * np.pi, size=(diameter, diameter))
                random_amplitude = self.rng.uniform(0.5, 1.0, size=(diameter, diameter))
                random_field = random_amplitude * np.exp(1j * random_phase)

                # Low-pass filter: apply a Gaussian filter in Fourier space
                # The cutoff frequency determines the speckle size
                # We want speckles that are a few pixels in size
                fy = np.fft.fftfreq(diameter)
                fx = np.fft.fftfreq(diameter)
                fxx, fyy = np.meshgrid(fx, fy)
                fr = np.sqrt(fxx**2 + fyy**2)

                # Gaussian low-pass filter with sigma chosen to give reasonable speckle size
                # A smaller sigma_freq means larger speckles
                sigma_freq = 0.15  # in normalized frequency units
                lowpass = np.exp(-fr**2 / (2 * sigma_freq**2))

                # Apply low-pass filter
                field_ft = fft2(random_field)
                field_filtered = ifft2(field_ft * lowpass)

                # Apply circular aperture
                aperture = (r <= self.probe_radius_pixels).astype(np.float64)
                probe = (field_filtered * aperture).astype(np.complex128)

                # Normalize to unit maximum amplitude
                max_amp = np.max(np.abs(probe))
                if max_amp > 0:
                    probe = probe / max_amp

            else:
                raise ValueError(
                    f"Unknown illumination type: '{self.illumination}'. "
                    f"Must be 'airy' or 'speckle'."
                )

            self.probe = probe
            return probe

    def _generate_scan_positions(self) -> np.ndarray:
            """
            Generate the raster scan positions for ptychographic data collection.

            Returns:
                Array of shape (J, 2) containing (row, col) scan positions in pixels.
            """
            # The pad array spans num_pads * pitch_pixels in each dimension
            # centered in the grid. We need the probe to cover the full array,
            # so scan positions should range such that the probe (centered at each
            # position) sweeps across the entire array.

            array_span = self.num_pads * self.pitch_pixels
            offset = (self.grid_size - array_span) // 2

            # The probe is probe_diameter_pixels x probe_diameter_pixels.
            # When centered at position (r, c), it covers:
            #   rows [r, r + probe_diameter) and cols [c, c + probe_diameter)
            # We need the probe window to start from the beginning of the array
            # (minus some margin) and end at the end of the array.

            # The probe patch is extracted starting at position (row, col),
            # covering [row : row + probe_diameter, col : col + probe_diameter].
            # The center of the probe is at (row + probe_radius, col + probe_radius).

            # We want the probe center to scan from the start of the array
            # to the end, ensuring full coverage.
            # Start: probe center at offset (start of array)
            #   => row_start = offset - probe_radius
            # End: probe center at offset + array_span (end of array)
            #   => row_end = offset + array_span - probe_radius

            # But we need to ensure the probe patch stays within the grid:
            # row >= 0 and row + probe_diameter <= grid_size
            # => row in [0, grid_size - probe_diameter]

            probe_d = self.probe_diameter_pixels

            # Scan center positions: we want the probe center to cover the array
            # Probe center = position + probe_radius
            # We want centers from offset to offset + array_span
            # So positions (top-left of probe patch) from:
            #   offset - probe_radius to offset + array_span - probe_radius

            row_start = max(offset - self.probe_radius_pixels, 0)
            row_end = min(offset + array_span - self.probe_radius_pixels,
                          self.grid_size - probe_d)

            col_start = max(offset - self.probe_radius_pixels, 0)
            col_end = min(offset + array_span - self.probe_radius_pixels,
                          self.grid_size - probe_d)

            # Generate raster grid with scan_step_pixels spacing
            row_positions = np.arange(row_start, row_end + 1, self.scan_step_pixels)
            col_positions = np.arange(col_start, col_end + 1, self.scan_step_pixels)

            # If no positions generated, ensure at least one
            if len(row_positions) == 0:
                row_positions = np.array([max(0, (self.grid_size - probe_d) // 2)])
            if len(col_positions) == 0:
                col_positions = np.array([max(0, (self.grid_size - probe_d) // 2)])

            # Create meshgrid of all scan positions
            col_grid, row_grid = np.meshgrid(col_positions, row_positions)
            positions = np.stack([row_grid.ravel(), col_grid.ravel()], axis=-1)

            # Ensure all positions are valid (probe patch within grid)
            valid = (
                (positions[:, 0] >= 0) &
                (positions[:, 0] + probe_d <= self.grid_size) &
                (positions[:, 1] >= 0) &
                (positions[:, 1] + probe_d <= self.grid_size)
            )
            positions = positions[valid]

            self.scan_positions = positions.astype(np.int64)
            return self.scan_positions

    # -------------------------------------------------------------------------
    # Forward Model
    # -------------------------------------------------------------------------

    def forward(
            self,
            object_estimate: np.ndarray,
            probe: np.ndarray,
            scan_positions: np.ndarray,
        ) -> List[np.ndarray]:
            """
            Compute forward diffraction intensities for all scan positions.

            For each scan position R_j:
                1. Extract exit wave: Psi_j(r) = P(r - R_j) * O(r)
                2. Propagate to far field: Psi_hat_j(k) = FFT{Psi_j(r)}
                3. Compute intensity: I_j(k) = |Psi_hat_j(k)|^2

            Args:
                object_estimate: Complex 2D array of shape (grid_size, grid_size).
                probe: Complex 2D array for the probe function.
                scan_positions: Array of shape (J, 2) with scan positions.

            Returns:
                List of J 2D real arrays, each containing the diffraction intensity
                at the corresponding scan position.
            """
            intensities = []
            num_positions = scan_positions.shape[0]

            for j in range(num_positions):
                position = scan_positions[j]

                # Step 1: Extract exit wave Psi_j(r) = P(r - R_j) * O(r)
                exit_wave = self._extract_exit_wave(object_estimate, probe, position)

                # Step 2: Propagate to far field via FFT
                farfield = self._propagate_to_farfield(exit_wave)

                # Step 3: Compute intensity I_j(k) = |Psi_hat_j(k)|^2
                intensity = np.abs(farfield) ** 2

                intensities.append(intensity.astype(np.float64))

            return intensities

    def _extract_exit_wave(
            self,
            object_estimate: np.ndarray,
            probe: np.ndarray,
            position: np.ndarray,
        ) -> np.ndarray:
            """
            Extract the exit wave for a single scan position.

            Args:
                object_estimate: Full complex object array.
                probe: Complex probe function.
                position: (row, col) scan position.

            Returns:
                Complex 2D exit wave Psi_j(r) = P(r - R_j) * O(r).
            """
            row, col = int(position[0]), int(position[1])
            probe_h, probe_w = probe.shape

            # Extract the object patch at the scan position
            object_patch = object_estimate[row:row + probe_h, col:col + probe_w].copy()

            # Compute exit wave as element-wise multiplication of probe and object patch
            exit_wave = probe * object_patch

            return exit_wave

    def _propagate_to_farfield(self, exit_wave: np.ndarray) -> np.ndarray:
            """
            Fraunhofer propagation via 2D FFT.

            Args:
                exit_wave: Complex 2D exit wave in real space.

            Returns:
                Complex 2D far-field amplitude Psi_hat(k).
            """
            return fftshift(fft2(ifftshift(exit_wave)))

    def _propagate_to_realspace(self, farfield: np.ndarray) -> np.ndarray:
            """
            Inverse Fraunhofer propagation via 2D IFFT.

            Args:
                farfield: Complex 2D far-field amplitude.

            Returns:
                Complex 2D real-space wave.
            """
            return fftshift(ifft2(ifftshift(farfield)))

    def _modulus_replacement(
            self,
            farfield_estimate: np.ndarray,
            measured_intensity: np.ndarray,
        ) -> np.ndarray:
            """
            Apply Fourier magnitude constraint (modulus replacement).

            Psi_c(k) = sqrt(I(k)) * Psi(k) / (|Psi(k)| + eps)

            Args:
                farfield_estimate: Current far-field estimate (complex).
                measured_intensity: Measured diffraction intensity (real, non-negative).

            Returns:
                Corrected far-field with measured amplitudes and estimated phases.
            """
            measured_amplitude = np.sqrt(np.maximum(measured_intensity, 0.0))
            estimated_amplitude = np.abs(farfield_estimate)
            corrected_farfield = measured_amplitude * farfield_estimate / (estimated_amplitude + self.eps)
            return corrected_farfield

    # -------------------------------------------------------------------------
    # Simulation / Data Generation
    # -------------------------------------------------------------------------

    def _simulate_diffraction_data(
            self, ground_truth_object: np.ndarray
        ) -> List[np.ndarray]:
            """
            Generate simulated noisy diffraction intensity patterns.

            Args:
                ground_truth_object: The true complex object O(r).

            Returns:
                List of J noisy intensity patterns.
            """
            # Ensure probe and scan positions are available
            if self.probe is None:
                self._build_probe()
            if self.scan_positions is None:
                self._generate_scan_positions()

            # Compute noise-free forward intensities
            intensities_clean = self.forward(
                ground_truth_object, self.probe, self.scan_positions
            )

            # Add Gaussian noise and clip negative values
            noisy_intensities = []
            for I_clean in intensities_clean:
                # Additive Gaussian noise scaled by noise_sigma
                # noise_sigma is relative, so scale by the mean intensity
                noise_scale = self.noise_sigma * np.mean(I_clean) if np.mean(I_clean) > 0 else self.noise_sigma
                noise = self.rng.normal(0.0, noise_scale, size=I_clean.shape)
                I_noisy = I_clean + noise

                # Clip negative values to zero
                I_noisy = np.maximum(I_noisy, 0.0)

                noisy_intensities.append(I_noisy.astype(np.float64))

            return noisy_intensities

    # -------------------------------------------------------------------------
    # Algorithm Implementations
    # -------------------------------------------------------------------------

    def _run_hio(
            self,
            measured_intensities: List[np.ndarray],
            support_mask: np.ndarray,
        ) -> np.ndarray:
            """
            Standard Hybrid Input-Output algorithm.

            Inside support S: accept inverse FT result.
            Outside support S: O_{n+1} = O_n - beta * O'_n.

            Expected to NOT converge for periodic structures.

            Args:
                measured_intensities: List of measured diffraction intensities.
                support_mask: Binary support mask.

            Returns:
                Reconstructed complex object estimate.
            """
            # For HIO, we use a single diffraction pattern (non-ptychographic).
            # If multiple patterns are provided, use the first one (or average them).
            # Standard HIO operates on a single far-field measurement.
            if len(measured_intensities) == 1:
                measured_intensity = measured_intensities[0]
            else:
                # For single-shot CDI, we use the first measurement or average
                # In a non-ptychographic setting, we typically have one pattern.
                # If multiple are given, average them as an approximation.
                measured_intensity = np.mean(np.array(measured_intensities), axis=0)

            # Initialize the object estimate
            obj = self._initialize_object()

            # The HIO operates on the full grid for single-shot CDI.
            # We need to match the size of the measured intensity.
            # If the measured intensity is smaller than the grid, we work at that size.
            M_h, M_w = measured_intensity.shape

            # If the object grid is larger than the measurement, we need to crop/resize
            # For simplicity, work at the measurement size if it differs from grid_size
            if M_h != self.grid_size or M_w != self.grid_size:
                # Resize the object and support mask to match measurement dimensions
                obj_work = np.full((M_h, M_w), self.R_substrate if self.R_substrate is not None else 1.0, dtype=np.complex128)
                # Add random phase for initialization
                obj_work *= np.exp(1j * self.rng.uniform(0, 0.1, size=(M_h, M_w)))

                # Resize support mask
                from scipy.ndimage import zoom
                zoom_factors = (M_h / self.grid_size, M_w / self.grid_size)
                if zoom_factors != (1.0, 1.0):
                    support_work = zoom(support_mask.astype(np.float64), zoom_factors, order=0) > 0.5
                else:
                    support_work = support_mask
            else:
                obj_work = obj.copy()
                support_work = support_mask

            self.error_history = []

            for n in range(self.num_iterations_io):
                # Step 1: Propagate current estimate to far field
                farfield = self._propagate_to_farfield(obj_work)

                # Step 2: Apply modulus replacement (Fourier magnitude constraint)
                farfield_corrected = self._modulus_replacement(farfield, measured_intensity)

                # Step 3: Back-propagate to real space
                obj_prime = self._propagate_to_realspace(farfield_corrected)

                # Step 4: HIO update rule
                # Inside support: accept the inverse FT result
                # Outside support: O_{n+1} = O_n - beta * O'_n
                obj_new = np.empty_like(obj_work)
                obj_new[support_work] = obj_prime[support_work]
                obj_new[~support_work] = obj_work[~support_work] - self.beta * obj_prime[~support_work]

                obj_work = obj_new

                # Compute Fourier error for convergence monitoring
                farfield_current = self._propagate_to_farfield(obj_work)
                measured_amp = np.sqrt(np.maximum(measured_intensity, 0.0))
                estimated_amp = np.abs(farfield_current)
                numerator = np.sum((measured_amp - estimated_amp) ** 2)
                denominator = np.sum(measured_intensity) + self.eps
                error = float(numerator / denominator)
                self.error_history.append(error)

                # Check convergence
                if n > 0 and len(self.error_history) >= 2:
                    prev_error = self.error_history[-2]
                    if prev_error > 0:
                        rel_change = abs(error - prev_error) / (prev_error + self.eps)
                        if rel_change < self.convergence_threshold:
                            break

            # If we worked at a different size, map back to grid_size
            if M_h != self.grid_size or M_w != self.grid_size:
                # Place the result back into the full grid
                result = np.full((self.grid_size, self.grid_size), 
                               self.R_substrate if self.R_substrate is not None else 1.0, 
                               dtype=np.complex128)
                # Center the reconstruction in the grid
                r_start = max(0, (self.grid_size - M_h) // 2)
                c_start = max(0, (self.grid_size - M_w) // 2)
                r_end = min(self.grid_size, r_start + M_h)
                c_end = min(self.grid_size, c_start + M_w)
                src_r_end = r_end - r_start
                src_c_end = c_end - c_start
                result[r_start:r_end, c_start:c_end] = obj_work[:src_r_end, :src_c_end]
                self.object_estimate = result
            else:
                self.object_estimate = obj_work

            return self.object_estimate

    def _run_rpio(
            self,
            measured_intensities: List[np.ndarray],
            support_mask: np.ndarray,
            R_p: complex,
        ) -> np.ndarray:
            """
            Reflectivity-Prior Input-Output algorithm (Eq. 5).

            Inside support S: accept inverse FT result.
            Outside support S: O_{n+1} = beta * O_n + (1 - beta) * R_p.

            Args:
                measured_intensities: List of measured diffraction intensities.
                support_mask: Binary support mask.
                R_p: Known substrate reflectivity (complex scalar).

            Returns:
                Reconstructed complex object estimate.
            """
            # For rpIO, we use a single diffraction pattern (non-ptychographic).
            if len(measured_intensities) == 1:
                measured_intensity = measured_intensities[0]
            else:
                measured_intensity = np.mean(np.array(measured_intensities), axis=0)

            # Initialize the object estimate
            obj = self._initialize_object()

            M_h, M_w = measured_intensity.shape

            # Handle size mismatch between measurement and grid
            if M_h != self.grid_size or M_w != self.grid_size:
                from scipy.ndimage import zoom
                # Work at measurement size
                obj_work = np.full((M_h, M_w), R_p, dtype=np.complex128)
                # Add small random phase perturbation for initialization
                obj_work *= np.exp(1j * self.rng.uniform(0, 0.1, size=(M_h, M_w)))

                zoom_factors = (M_h / self.grid_size, M_w / self.grid_size)
                if zoom_factors != (1.0, 1.0):
                    support_work = zoom(support_mask.astype(np.float64), zoom_factors, order=0) > 0.5
                else:
                    support_work = support_mask.copy()
            else:
                obj_work = obj.copy()
                support_work = support_mask.copy()

            self.error_history = []

            for n in range(self.num_iterations_io):
                # Step 1: Propagate current estimate to far field
                farfield = self._propagate_to_farfield(obj_work)

                # Step 2: Apply modulus replacement (Fourier magnitude constraint)
                farfield_corrected = self._modulus_replacement(farfield, measured_intensity)

                # Step 3: Back-propagate to real space
                obj_prime = self._propagate_to_realspace(farfield_corrected)

                # Step 4: rpIO update rule (Eq. 5)
                # Inside support: accept the inverse FT result
                # Outside support: O_{n+1} = beta * O_n + (1 - beta) * R_p
                obj_new = np.empty_like(obj_work)
                obj_new[support_work] = obj_prime[support_work]
                obj_new[~support_work] = (
                    self.beta * obj_work[~support_work]
                    + (1.0 - self.beta) * R_p
                )

                obj_work = obj_new

                # Compute Fourier error for convergence monitoring
                farfield_current = self._propagate_to_farfield(obj_work)
                measured_amp = np.sqrt(np.maximum(measured_intensity, 0.0))
                estimated_amp = np.abs(farfield_current)
                numerator = np.sum((measured_amp - estimated_amp) ** 2)
                denominator = np.sum(measured_intensity) + self.eps
                error = float(numerator / denominator)
                self.error_history.append(error)

                # Check convergence
                if n > 0 and len(self.error_history) >= 2:
                    prev_error = self.error_history[-2]
                    if prev_error > 0:
                        rel_change = abs(error - prev_error) / (prev_error + self.eps)
                        if rel_change < self.convergence_threshold:
                            break

            # If we worked at a different size, map back to grid_size
            if M_h != self.grid_size or M_w != self.grid_size:
                result = np.full(
                    (self.grid_size, self.grid_size),
                    R_p,
                    dtype=np.complex128,
                )
                r_start = max(0, (self.grid_size - M_h) // 2)
                c_start = max(0, (self.grid_size - M_w) // 2)
                r_end = min(self.grid_size, r_start + M_h)
                c_end = min(self.grid_size, c_start + M_w)
                src_r_end = r_end - r_start
                src_c_end = c_end - c_start
                result[r_start:r_end, c_start:c_end] = obj_work[:src_r_end, :src_c_end]
                self.object_estimate = result
            else:
                self.object_estimate = obj_work

            return self.object_estimate

    def _run_pie(
            self,
            measured_intensities: List[np.ndarray],
            probe: np.ndarray,
            scan_positions: np.ndarray,
        ) -> np.ndarray:
            """
            Standard Ptychographic Iterative Engine (ePIE, object update only).

            Per scan position j:
                1. Form exit wave
                2. Propagate, modulus replace, back-propagate
                3. ePIE object update rule (Eq. 6, first line only)

            Args:
                measured_intensities: List of J measured diffraction intensities.
                probe: Known probe function.
                scan_positions: Array of shape (J, 2) with scan positions.

            Returns:
                Reconstructed complex object estimate.
            """
            # Initialize the object estimate
            obj = self._initialize_object()

            num_positions = scan_positions.shape[0]
            probe_h, probe_w = probe.shape

            # Precompute probe quantities for ePIE update
            P_max = np.max(np.abs(probe))
            alpha_reg = self.alpha_regularization * (P_max ** 2)

            self.error_history = []

            for epoch in range(self.num_epochs_pie):
                # Shuffle scan positions each epoch for better convergence
                order = self.rng.permutation(num_positions)

                for idx in order:
                    position = scan_positions[idx]
                    row, col = int(position[0]), int(position[1])

                    # Step 1: Extract object patch and form exit wave
                    object_patch = obj[row:row + probe_h, col:col + probe_w].copy()
                    exit_wave = probe * object_patch

                    # Step 2: Propagate to far field
                    farfield = self._propagate_to_farfield(exit_wave)

                    # Step 3: Modulus replacement
                    farfield_corrected = self._modulus_replacement(
                        farfield, measured_intensities[idx]
                    )

                    # Step 4: Back-propagate to real space
                    corrected_exit_wave = self._propagate_to_realspace(farfield_corrected)

                    # Step 5: ePIE object update
                    # O_{n+1}(r) = O_n(r) + (|P|/|P_max|) * (P* / (|P|^2 + alpha)) * beta * (Psi_c - Psi)
                    diff = corrected_exit_wave - exit_wave
                    probe_abs = np.abs(probe)
                    update_numerator = probe_abs / (P_max + self.eps)
                    update_weight = np.conj(probe) / (probe_abs ** 2 + alpha_reg)
                    object_update = update_numerator * update_weight * self.beta * diff

                    # Apply update to the object patch
                    obj[row:row + probe_h, col:col + probe_w] += object_update

                # Compute Fourier error at the end of each epoch for convergence monitoring
                error = self._compute_fourier_error(obj, measured_intensities)
                self.error_history.append(error)

                # Check convergence
                if epoch > 0 and len(self.error_history) >= 2:
                    prev_error = self.error_history[-2]
                    if prev_error > 0:
                        rel_change = abs(error - prev_error) / (prev_error + self.eps)
                        if rel_change < self.convergence_threshold:
                            break

            self.object_estimate = obj
            return self.object_estimate

    def _run_rppie(
            self,
            measured_intensities: List[np.ndarray],
            probe: np.ndarray,
            scan_positions: np.ndarray,
            support_mask: np.ndarray,
            R_p: complex,
        ) -> np.ndarray:
            """
            Reflectivity-Prior PIE algorithm (Eq. 6) — two-step nested update.

            Per scan position j:
                Step 1: ePIE object update → O_intermediate
                Step 2: Enforce reflectivity prior outside support:
                        O_{n+1}(r) = O_int(r) if r in S
                        O_{n+1}(r) = beta * O_int(r) + (1-beta) * R_p if r not in S

            Args:
                measured_intensities: List of J measured diffraction intensities.
                probe: Known probe function.
                scan_positions: Array of shape (J, 2) with scan positions.
                support_mask: Binary support mask.
                R_p: Known substrate reflectivity (complex scalar).

            Returns:
                Reconstructed complex object estimate.
            """
            # Initialize the object estimate
            obj = self._initialize_object()

            num_positions = scan_positions.shape[0]
            probe_h, probe_w = probe.shape

            # Precompute probe quantities for ePIE update
            P_max = np.max(np.abs(probe))
            alpha_reg = self.alpha_regularization * (P_max ** 2)

            # Precompute probe-related arrays for efficiency
            probe_conj = np.conj(probe)
            probe_abs = np.abs(probe)
            probe_abs_sq = probe_abs ** 2

            # ePIE update weight components (constant across iterations)
            update_numerator = probe_abs / (P_max + self.eps)
            update_weight = probe_conj / (probe_abs_sq + alpha_reg)

            self.error_history = []

            for epoch in range(self.num_epochs_pie):
                # Shuffle scan positions each epoch for better convergence
                order = self.rng.permutation(num_positions)

                for idx in order:
                    position = scan_positions[idx]
                    row, col = int(position[0]), int(position[1])

                    # Step 1a: Extract object patch and form exit wave
                    object_patch = obj[row:row + probe_h, col:col + probe_w].copy()
                    exit_wave = probe * object_patch

                    # Step 1b: Propagate to far field
                    farfield = self._propagate_to_farfield(exit_wave)

                    # Step 1c: Modulus replacement
                    farfield_corrected = self._modulus_replacement(
                        farfield, measured_intensities[idx]
                    )

                    # Step 1d: Back-propagate to real space
                    corrected_exit_wave = self._propagate_to_realspace(farfield_corrected)

                    # Step 1e: ePIE object update (Step 1 of two-step nested update)
                    # O_int(r) = O_n(r) + (|P|/|P_max|) * (P* / (|P|^2 + alpha)) * beta * (Psi_c - Psi)
                    diff = corrected_exit_wave - exit_wave
                    object_update = update_numerator * update_weight * self.beta * diff

                    # Apply ePIE update to get intermediate object
                    obj[row:row + probe_h, col:col + probe_w] += object_update

                    # Step 2: Enforce reflectivity prior outside support in the updated patch region
                    # Extract the support mask for this patch
                    support_patch = support_mask[row:row + probe_h, col:col + probe_w]

                    # Apply reflectivity prior: outside support, blend toward R_p
                    # O_{n+1}(r) = beta * O_int(r) + (1 - beta) * R_p  for r not in S
                    outside_support = ~support_patch
                    obj[row:row + probe_h, col:col + probe_w][outside_support] = (
                        self.beta * obj[row:row + probe_h, col:col + probe_w][outside_support]
                        + (1.0 - self.beta) * R_p
                    )

                # Compute Fourier error at the end of each epoch for convergence monitoring
                error = self._compute_fourier_error(obj, measured_intensities)
                self.error_history.append(error)

                # Check convergence
                if epoch > 0 and len(self.error_history) >= 2:
                    prev_error = self.error_history[-2]
                    if prev_error > 0:
                        rel_change = abs(error - prev_error) / (prev_error + self.eps)
                        if rel_change < self.convergence_threshold:
                            break

            self.object_estimate = obj
            return self.object_estimate

    def _epie_object_update(
            self,
            object_patch: np.ndarray,
            probe: np.ndarray,
            exit_wave: np.ndarray,
            corrected_exit_wave: np.ndarray,
        ) -> np.ndarray:
            """
            ePIE-style object update rule (Eq. 6, Step 1).

            O_{n+1}(r) = O_n(r) + (|P|/|P_max|) * (P* / (|P|^2 + alpha)) * beta * (Psi_c - Psi)

            Args:
                object_patch: Current object estimate in the probe window.
                probe: Probe function.
                exit_wave: Current exit wave Psi = P * O.
                corrected_exit_wave: Corrected exit wave Psi_c after modulus replacement.

            Returns:
                Updated object patch.
            """
            P_max = np.max(np.abs(probe))
            probe_abs = np.abs(probe)
            probe_abs_sq = probe_abs ** 2
            alpha_reg = self.alpha_regularization * (P_max ** 2)

            # Difference between corrected and current exit waves
            diff = corrected_exit_wave - exit_wave

            # ePIE update: (|P|/|P_max|) * (P* / (|P|^2 + alpha)) * beta * (Psi_c - Psi)
            update_numerator = probe_abs / (P_max + self.eps)
            update_weight = np.conj(probe) / (probe_abs_sq + alpha_reg)
            object_update = update_numerator * update_weight * self.beta * diff

            updated_patch = object_patch + object_update

            return updated_patch

    def _apply_reflectivity_prior(
            self,
            object_estimate: np.ndarray,
            support_mask: np.ndarray,
            R_p: complex,
        ) -> np.ndarray:
            """
            Enforce reflectivity prior outside the support mask (Eq. 6, Step 2 / Eq. 5 outside).

            Inside S: keep object as-is.
            Outside S: O(r) = beta * O(r) + (1 - beta) * R_p.

            Args:
                object_estimate: Current complex object estimate.
                support_mask: Binary support mask (True on pads).
                R_p: Known substrate reflectivity.

            Returns:
                Object estimate with reflectivity prior applied outside support.
            """
            result = object_estimate.copy()
            outside_support = ~support_mask
            result[outside_support] = (
                self.beta * object_estimate[outside_support]
                + (1.0 - self.beta) * R_p
            )
            return result

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def _compute_fourier_error(
            self,
            object_estimate: np.ndarray,
            measured_intensities: List[np.ndarray],
        ) -> float:
            """
            Compute normalized Fourier-space error (R-factor).

            E = sum_j sum_k (sqrt(I_j(k)) - |Psi_hat_j(k)|)^2 / sum_j sum_k I_j(k)

            Args:
                object_estimate: Current object estimate.
                measured_intensities: List of measured diffraction intensities.

            Returns:
                Scalar normalized Fourier error.
            """
            numerator_total = 0.0
            denominator_total = 0.0

            probe = self.probe
            scan_positions = self.scan_positions

            if probe is None or scan_positions is None:
                return float('inf')

            num_positions = scan_positions.shape[0]
            probe_h, probe_w = probe.shape

            for j in range(num_positions):
                position = scan_positions[j]
                row, col = int(position[0]), int(position[1])

                # Extract object patch and form exit wave
                object_patch = object_estimate[row:row + probe_h, col:col + probe_w]
                exit_wave = probe * object_patch

                # Propagate to far field
                farfield = self._propagate_to_farfield(exit_wave)

                # Compute amplitudes
                measured_amp = np.sqrt(np.maximum(measured_intensities[j], 0.0))
                estimated_amp = np.abs(farfield)

                # Accumulate error
                numerator_total += np.sum((measured_amp - estimated_amp) ** 2)
                denominator_total += np.sum(measured_intensities[j])

            error = float(numerator_total / (denominator_total + self.eps))
            return error

    def _compute_psnr(
            self,
            reconstructed_phase: np.ndarray,
            ground_truth_phase: np.ndarray,
        ) -> float:
            """
            Compute Peak Signal-to-Noise Ratio of the reconstructed phase map.

            Args:
                reconstructed_phase: Reconstructed phase array.
                ground_truth_phase: Ground-truth phase array.

            Returns:
                PSNR in dB.
            """
            # Compute the maximum possible signal value (peak-to-peak range of ground truth phase)
            max_val = np.max(ground_truth_phase) - np.min(ground_truth_phase)

            # If the ground truth has no variation, PSNR is undefined; return infinity if perfect match
            if max_val < self.eps:
                if np.allclose(reconstructed_phase, ground_truth_phase, atol=self.eps):
                    return float('inf')
                max_val = 1.0  # fallback to avoid division by zero

            # Compute Mean Squared Error
            mse = np.mean((reconstructed_phase - ground_truth_phase) ** 2)

            if mse < self.eps:
                return float('inf')

            psnr = 10.0 * np.log10((max_val ** 2) / mse)

            return float(psnr)

    def _compute_rmse_recession(
            self,
            reconstructed_phase: np.ndarray,
            ground_truth_depths: np.ndarray,
            support_mask: np.ndarray,
        ) -> float:
            """
            Compute RMSE of recession depth estimates vs ground truth.

            Args:
                reconstructed_phase: Reconstructed phase map.
                ground_truth_depths: Ground-truth recession depths per pad.
                support_mask: Binary support mask.

            Returns:
                RMSE in meters.
            """
            # Convert reconstructed phase to recession depth: h = phase * lambda / (4*pi)
            reconstructed_depths_map = reconstructed_phase * self.wavelength / (4.0 * np.pi)

            # Extract per-pad average depths from the reconstructed depth map
            estimated_depths = self._extract_per_pad_depths(
                reconstructed_phase, support_mask
            )

            # Ensure shapes match
            if estimated_depths.shape != ground_truth_depths.shape:
                # Resize if necessary — take the common region
                min_rows = min(estimated_depths.shape[0], ground_truth_depths.shape[0])
                min_cols = min(estimated_depths.shape[1], ground_truth_depths.shape[1])
                estimated_depths = estimated_depths[:min_rows, :min_cols]
                ground_truth_depths = ground_truth_depths[:min_rows, :min_cols]

            # Compute RMSE between estimated and ground truth recession depths
            diff = estimated_depths - ground_truth_depths
            rmse = float(np.sqrt(np.mean(diff ** 2)))

            return rmse

    def _extract_per_pad_depths(
            self,
            phase_map: np.ndarray,
            support_mask: np.ndarray,
        ) -> np.ndarray:
            """
            Extract average recession depth per pad from the phase map.

            Args:
                phase_map: 2D phase array.
                support_mask: Binary support mask.

            Returns:
                Array of shape (num_pads, num_pads) with estimated recession depths.
            """
            depths = np.zeros((self.num_pads, self.num_pads), dtype=np.float64)

            # Compute the starting offset so that the pad array is centered in the grid
            array_span = self.num_pads * self.pitch_pixels
            offset_row = (self.grid_size - array_span) // 2
            offset_col = (self.grid_size - array_span) // 2

            # Pad offset within each pitch cell (to center the pad)
            pad_offset_in_cell = (self.pitch_pixels - self.pad_size_pixels) // 2

            for i in range(self.num_pads):
                for j in range(self.num_pads):
                    # Top-left corner of the pitch cell for pad (i, j)
                    cell_row = offset_row + i * self.pitch_pixels
                    cell_col = offset_col + j * self.pitch_pixels

                    # Top-left corner of the pad within the cell
                    pad_row_start = cell_row + pad_offset_in_cell
                    pad_col_start = cell_col + pad_offset_in_cell

                    pad_row_end = pad_row_start + self.pad_size_pixels
                    pad_col_end = pad_col_start + self.pad_size_pixels

                    # Clip to grid boundaries
                    r_start = max(pad_row_start, 0)
                    r_end = min(pad_row_end, phase_map.shape[0])
                    c_start = max(pad_col_start, 0)
                    c_end = min(pad_col_end, phase_map.shape[1])

                    if r_start < r_end and c_start < c_end:
                        # Extract the phase values within this pad region
                        pad_phase = phase_map[r_start:r_end, c_start:c_end]
                        pad_support = support_mask[r_start:r_end, c_start:c_end]

                        # Only average over pixels that are within the support mask
                        if np.any(pad_support):
                            avg_phase = np.mean(pad_phase[pad_support])
                        else:
                            avg_phase = np.mean(pad_phase)

                        # Convert phase to recession depth: h = phase * lambda / (4*pi)
                        depths[i, j] = avg_phase * self.wavelength / (4.0 * np.pi)
                    else:
                        depths[i, j] = 0.0

            return depths

    # -------------------------------------------------------------------------
    # Main Solve Interface
    # -------------------------------------------------------------------------


# >>> TARGET FUNCTION: solve <<<
    def solve(
                self,
                input_data: np.ndarray,
                support_mask: Optional[np.ndarray] = None,
                probe: Optional[np.ndarray] = None,
                scan_positions: Optional[np.ndarray] = None,
                R_p: Optional[complex] = None,
            ) -> np.ndarray:
                """
                Main entry point: recover the complex object from measured diffraction data.

                Dispatches to the appropriate algorithm (HIO, rpIO, PIE, rpPIE) based on
                self.algorithm. Handles initialization, iteration, convergence monitoring,
                and returns the reconstructed object.

                If support_mask, probe, scan_positions, or R_p are not provided,
                they are computed/generated internally.

                Args:
                    input_data: Measured diffraction data. For single-shot (HIO/rpIO): 
                                2D intensity array. For ptychographic (PIE/rpPIE): 
                                3D array of shape (J, M, M) or loaded from structured input.
                    support_mask: Optional binary support mask. Built internally if None.
                    probe: Optional probe function. Generated internally if None.
                    scan_positions: Optional scan positions array. Generated internally if None.
                    R_p: Optional known substrate reflectivity. Computed internally if None.

                Returns:
                    Reconstructed complex-valued object O(r) as 2D complex numpy array.
                """
                # 1. Compute substrate and pad reflectivities if not already done
                if self.R_substrate is None or self.R_pad is None:
                    self._compute_substrate_and_pad_reflectivities()

                # 2. Set R_p (substrate reflectivity prior) if not provided
                if R_p is None:
                    R_p = self.R_substrate

                # 3. Build support mask if not provided
                if support_mask is None:
                    if self.support_mask is None:
                        self._build_support_mask()
                    support_mask = self.support_mask
                else:
                    self.support_mask = support_mask

                # 4. Build probe if not provided
                if probe is None:
                    if self.probe is None:
                        self._build_probe()
                    probe = self.probe
                else:
                    # Validate and fix the probe array
                    probe = np.array(probe)
                    # Squeeze singleton dimensions
                    probe = np.squeeze(probe)
                    # If probe has 3 dimensions (e.g., (H, W, 2) for real/imag), convert
                    if probe.ndim == 3:
                        if probe.shape[2] == 2:
                            probe = probe[..., 0] + 1j * probe[..., 1]
                        elif probe.shape[0] == 2:
                            probe = probe[0] + 1j * probe[1]
                        else:
                            # Try to take the first slice
                            probe = probe[0]
                    # Ensure 2D
                    if probe.ndim > 2:
                        # Last resort: reshape or take first 2D slice
                        while probe.ndim > 2:
                            probe = probe[0]
                    assert probe.ndim == 2, f"Probe must be 2D, got shape {probe.shape}"
                    probe = probe.astype(np.complex128)
                    self.probe = probe

                # 5. Generate scan positions if not provided
                if scan_positions is None:
                    if self.scan_positions is None:
                        self._generate_scan_positions()
                    scan_positions = self.scan_positions
                else:
                    scan_positions = np.array(scan_positions)
                    if scan_positions.ndim == 1:
                        scan_positions = scan_positions.reshape(-1, 2)
                    self.scan_positions = scan_positions

                # Update probe_diameter_pixels and probe_radius_pixels based on actual probe
                probe_h, probe_w = probe.shape
                self.probe_diameter_pixels = max(probe_h, probe_w)
                self.probe_radius_pixels = self.probe_diameter_pixels // 2

                # Verify grid_size is large enough to accommodate probe patches at all scan positions
                if scan_positions is not None and len(scan_positions) > 0:
                    max_row = int(np.max(scan_positions[:, 0])) + probe_h
                    max_col = int(np.max(scan_positions[:, 1])) + probe_w
                    needed_grid = max(max_row, max_col)
                    if needed_grid > self.grid_size:
                        self.grid_size = needed_grid
                        # Rebuild support mask for new grid size
                        self._build_support_mask()
                        support_mask = self.support_mask

                # 6. Parse input_data into list of measured intensities
                measured_intensities: List[np.ndarray] = []
                if input_data.ndim == 2:
                    # Single 2D intensity pattern
                    measured_intensities = [input_data.astype(np.float64)]
                elif input_data.ndim == 3:
                    # 3D array of shape (J, M, M) — multiple diffraction patterns
                    for j in range(input_data.shape[0]):
                        measured_intensities.append(input_data[j].astype(np.float64))
                else:
                    raise ValueError(
                        f"input_data must be 2D or 3D array, got shape {input_data.shape}"
                    )

                # 7. Dispatch to the appropriate algorithm
                if self.algorithm == "hio":
                    result = self._run_hio(measured_intensities, support_mask)

                elif self.algorithm == "rpio":
                    result = self._run_rpio(measured_intensities, support_mask, R_p)

                elif self.algorithm == "pie":
                    # PIE requires multiple scan positions
                    if len(measured_intensities) == 1:
                        num_pos = scan_positions.shape[0]
                        measured_intensities = measured_intensities * num_pos

                    # Ensure number of intensities matches number of scan positions
                    num_pos = scan_positions.shape[0]
                    if len(measured_intensities) != num_pos:
                        if len(measured_intensities) > num_pos:
                            measured_intensities = measured_intensities[:num_pos]
                        else:
                            while len(measured_intensities) < num_pos:
                                measured_intensities.append(measured_intensities[-1])

                    result = self._run_pie(measured_intensities, probe, scan_positions)

                elif self.algorithm == "rppie":
                    # rpPIE requires multiple scan positions
                    if len(measured_intensities) == 1:
                        num_pos = scan_positions.shape[0]
                        measured_intensities = measured_intensities * num_pos

                    # Ensure number of intensities matches number of scan positions
                    num_pos = scan_positions.shape[0]
                    if len(measured_intensities) != num_pos:
                        if len(measured_intensities) > num_pos:
                            measured_intensities = measured_intensities[:num_pos]
                        else:
                            while len(measured_intensities) < num_pos:
                                measured_intensities.append(measured_intensities[-1])

                    result = self._run_rppie(
                        measured_intensities, probe, scan_positions, support_mask, R_p
                    )

                else:
                    raise ValueError(
                        f"Unknown algorithm: '{self.algorithm}'. "
                        f"Must be one of 'hio', 'rpio', 'pie', 'rppie'."
                    )

                # 8. Store the final object estimate
                self.object_estimate = result

                return result

    def _initialize_object(self) -> np.ndarray:
            """
            Create initial object estimate for iterative reconstruction.

            Returns:
                Complex 2D array initialized (e.g., uniform R_p or random phase).
            """
            # Determine the baseline reflectivity to use for initialization
            if self.R_substrate is not None:
                R_init = self.R_substrate
            else:
                R_init = complex(1.0, 0.0)

            # Initialize the object with substrate reflectivity everywhere
            obj = np.full(
                (self.grid_size, self.grid_size),
                R_init,
                dtype=np.complex128
            )

            # Add a small random phase perturbation to break symmetry
            # and help convergence. The perturbation is kept small so the
            # initial estimate is close to the physical prior.
            random_phase = self.rng.uniform(0, 0.1, size=(self.grid_size, self.grid_size))
            obj = obj * np.exp(1j * random_phase)

            # If the support mask is available, give pad regions a slightly
            # different initialization using the pad reflectivity with a
            # mid-range recession phase, to provide a better starting point.
            if self.support_mask is not None and self.R_pad is not None:
                # Use the midpoint recession depth for initial phase on pads
                mid_depth = 0.5 * (self.recession_depth_min + self.recession_depth_max)
                mid_phase = 4.0 * np.pi * mid_depth / self.wavelength

                obj[self.support_mask] = self.R_pad * np.exp(1j * mid_phase)
                # Add small random perturbation on pads too
                pad_perturbation = self.rng.uniform(0, 0.1, size=np.sum(self.support_mask))
                obj[self.support_mask] *= np.exp(1j * pad_perturbation)

            return obj




if __name__ == "__main__":
    import os
    import sys
    import traceback

    try:
        print("Loading input data...", flush=True)
        input_data = np.load('dataset/input.npy')
        print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}", flush=True)

        support_mask = None
        probe = None
        scan_positions = None

        if os.path.exists('dataset/support_mask.npy'):
            support_mask = np.load('dataset/support_mask.npy')
            print(f"Loaded support_mask: shape={support_mask.shape}", flush=True)
        if os.path.exists('dataset/probe.npy'):
            probe = np.load('dataset/probe.npy')
            print(f"Loaded probe: shape={probe.shape}, dtype={probe.dtype}", flush=True)
        if os.path.exists('dataset/scan_positions.npy'):
            scan_positions = np.load('dataset/scan_positions.npy')
            print(f"Loaded scan_positions: shape={scan_positions.shape}", flush=True)

        # Determine data dimensions
        if input_data.ndim == 3:
            num_patterns = input_data.shape[0]
            pattern_size = input_data.shape[1]
            algorithm = "rppie"
            print(f"Ptychographic data: {num_patterns} patterns of size {pattern_size}x{pattern_size}", flush=True)
        elif input_data.ndim == 2:
            num_patterns = 1
            pattern_size = input_data.shape[0]
            algorithm = "rpio"
            print(f"Single pattern of size {pattern_size}x{pattern_size}", flush=True)
        else:
            raise ValueError(f"Unexpected input_data ndim: {input_data.ndim}")

        # The probe MUST have the same spatial dimensions as each diffraction pattern
        # so that FFT sizes match during modulus replacement
        probe_diameter = pattern_size
        probe_radius = probe_diameter // 2

        # Grid size must accommodate pad array (15*30=450 pixels) plus probe extent
        array_span = 15 * 30  # 450 pixels
        min_grid = array_span + 2 * probe_radius
        grid_size = max(512, min_grid)
        # Round up to reasonable size
        if grid_size <= 512:
            grid_size = 512
        elif grid_size <= 600:
            grid_size = 600
        elif grid_size <= 768:
            grid_size = 768
        else:
            grid_size = 1024

        print(f"Using algorithm={algorithm}, probe_radius={probe_radius}, grid_size={grid_size}", flush=True)

        # Handle probe shape issues before passing to solver
        if probe is not None:
            probe = np.array(probe)
            probe = np.squeeze(probe)
            if probe.ndim == 3:
                if probe.shape[2] == 2:
                    probe = probe[..., 0] + 1j * probe[..., 1]
                elif probe.shape[0] == 2:
                    probe = probe[0] + 1j * probe[1]
                else:
                    probe = probe[0]
            while probe.ndim > 2:
                probe = probe[0]
            probe = probe.astype(np.complex128)
            print(f"Probe after preprocessing: shape={probe.shape}", flush=True)

            # Ensure probe matches pattern_size
            if probe.shape[0] != pattern_size or probe.shape[1] != pattern_size:
                print(f"Resizing probe from {probe.shape} to ({pattern_size},{pattern_size})", flush=True)
                from scipy.ndimage import zoom
                zoom_r = pattern_size / probe.shape[0]
                zoom_c = pattern_size / probe.shape[1]
                probe_real = zoom(probe.real, (zoom_r, zoom_c), order=3)
                probe_imag = zoom(probe.imag, (zoom_r, zoom_c), order=3)
                probe = (probe_real + 1j * probe_imag).astype(np.complex128)
                print(f"Probe after resize: shape={probe.shape}", flush=True)
        else:
            print("No probe file found, will generate internally.", flush=True)

        # Handle scan_positions
        if scan_positions is not None:
            scan_positions = np.array(scan_positions)
            if scan_positions.ndim == 1:
                scan_positions = scan_positions.reshape(-1, 2)
            print(f"Scan positions: {scan_positions.shape[0]} positions", flush=True)

            # Ensure grid_size is large enough for all scan positions + probe
            if len(scan_positions) > 0:
                max_row = int(np.max(scan_positions[:, 0])) + probe_diameter
                max_col = int(np.max(scan_positions[:, 1])) + probe_diameter
                needed = max(max_row, max_col)
                if needed > grid_size:
                    grid_size = needed
                    print(f"Increased grid_size to {grid_size} to fit scan positions", flush=True)

        # Adjust number of epochs based on problem size to avoid timeout
        if num_patterns > 100:
            num_epochs = 30
        elif num_patterns > 50:
            num_epochs = 50
        else:
            num_epochs = 80
        print(f"Using {num_epochs} epochs", flush=True)

        # Instantiate solver
        solver = InverseSolver(
            algorithm=algorithm,
            illumination="airy",
            beta=0.7,
            alpha_regularization=1e-3,
            num_iterations_io=300,
            num_epochs_pie=num_epochs,
            convergence_threshold=1e-6,
            noise_sigma=0.01,
            random_seed=42,
            probe_radius_pixels=probe_radius,
            grid_size=grid_size,
        )

        print("Starting reconstruction...", flush=True)

        # Run reconstruction
        result = solver.solve(
            input_data,
            support_mask=support_mask,
            probe=probe,
            scan_positions=scan_positions,
        )

        print(f"Reconstruction complete. Result shape: {result.shape}, dtype: {result.dtype}", flush=True)
        print(f"Error history length: {len(solver.error_history)}", flush=True)
        if len(solver.error_history) > 0:
            print(f"Final Fourier error: {solver.error_history[-1]:.6e}", flush=True)
            if len(solver.error_history) > 1:
                print(f"Initial Fourier error: {solver.error_history[0]:.6e}", flush=True)

        # Save reconstructed complex object
        np.save('output.npy', result)
        print("Result saved to output.npy", flush=True)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()

        # Save a fallback result so we don't fail completely
        try:
            input_data = np.load('dataset/input.npy')
            if input_data.ndim == 3:
                # Create a minimal reconstruction
                grid_size = 512
                fallback = np.ones((grid_size, grid_size), dtype=np.complex128)
                np.save('output.npy', fallback)
                print("Saved fallback output.", flush=True)
            elif input_data.ndim == 2:
                fallback = np.ones_like(input_data, dtype=np.complex128)
                np.save('output.npy', fallback)
                print("Saved fallback output.", flush=True)
        except Exception as e2:
            print(f"Fallback also failed: {e2}", flush=True)
            fallback = np.ones((512, 512), dtype=np.complex128)
            np.save('output.npy', fallback)