import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
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

        # Material optical constants
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
        self.algorithm: str = algorithm
        self.illumination: str = illumination
        self.probe_radius_pixels: int = probe_radius_pixels
        self.eps: float = eps

        # Derived / cached quantities (populated during solve)
        self.support_mask: Optional[np.ndarray] = None
        self.probe: Optional[np.ndarray] = None
        self.scan_positions: Optional[np.ndarray] = None
        self.R_substrate: Optional[complex] = None
        self.R_pad: Optional[complex] = None
        self.object_estimate: Optional[np.ndarray] = None
        self.error_history: List[float] = []

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
        # TODO: Implement r = (n1 - n2) / (n1 + n2)
        pass

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
        # TODO: Implement transfer matrix M_l per Eq. in plan
        pass

    def _compute_multilayer_reflectivity(
        self, layer_stack: List[Tuple[complex, float]]
    ) -> complex:
        """
        Compute overall complex reflectivity of a multilayer stack using
        the transmission matrix method.

        Args:
            layer_stack: List of (complex_refractive_index, thickness_meters) tuples,
                         ordered from top (vacuum side) to bottom (substrate).

        Returns:
            Complex reflectivity of the full stack.
        """
        # TODO: Implement TMM to get overall reflectivity
        pass

    def _compute_substrate_and_pad_reflectivities(self) -> Tuple[complex, complex]:
        """
        Compute the complex reflectivities R_substrate (inter-pad regions)
        and R_pad (on-pad regions) from the multilayer models.

        Returns:
            Tuple of (R_substrate, R_pad).
        """
        # TODO: Implement using _compute_multilayer_reflectivity for both stacks
        pass

    # -------------------------------------------------------------------------
    # Object / Support / Probe Construction
    # -------------------------------------------------------------------------

    def _build_support_mask(self) -> np.ndarray:
        """
        Create the binary support mask S(r) for the 15x15 periodic pad array.

        Returns:
            2D boolean array of shape (grid_size, grid_size), True on pad locations.
        """
        # TODO: Implement 15x15 array of 22x22 pixel pads at pitch_pixels spacing
        pass

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
        # TODO: Implement O(r) = R_pad * exp(i * 4*pi*h/lambda) on pads, R_p elsewhere
        pass

    def _build_probe(self) -> np.ndarray:
        """
        Generate the probe function P(r) — either Airy spot or speckle field.

        Returns:
            Complex 2D array representing the probe, same grid as object patches.
        """
        # TODO: Implement Airy: 2*J1(pi*r/r0)/(pi*r/r0), or speckle: random + lowpass
        pass

    def _generate_scan_positions(self) -> np.ndarray:
        """
        Generate the raster scan positions for ptychographic data collection.

        Returns:
            Array of shape (J, 2) containing (row, col) scan positions in pixels.
        """
        # TODO: Implement raster grid with scan_step_pixels spacing, covering full array
        pass

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
        # TODO: Implement Fraunhofer propagation for each scan position
        pass

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
        # TODO: Implement windowed multiplication
        pass

    def _propagate_to_farfield(self, exit_wave: np.ndarray) -> np.ndarray:
        """
        Fraunhofer propagation via 2D FFT.

        Args:
            exit_wave: Complex 2D exit wave in real space.

        Returns:
            Complex 2D far-field amplitude Psi_hat(k).
        """
        # TODO: Implement fftshift(fft2(ifftshift(exit_wave)))
        pass

    def _propagate_to_realspace(self, farfield: np.ndarray) -> np.ndarray:
        """
        Inverse Fraunhofer propagation via 2D IFFT.

        Args:
            farfield: Complex 2D far-field amplitude.

        Returns:
            Complex 2D real-space wave.
        """
        # TODO: Implement fftshift(ifft2(ifftshift(farfield)))
        pass

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
        # TODO: Implement modulus replacement
        pass

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
        # TODO: Implement forward model + Gaussian noise + clipping
        pass

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
        # TODO: Implement standard HIO iteration loop
        pass

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
        # TODO: Implement rpIO with reflectivity prior enforcement outside support
        pass

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
        # TODO: Implement standard ePIE loop over epochs and scan positions
        pass

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
        # TODO: Implement rpPIE with two-step nested update per scan position
        pass

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
        # TODO: Implement ePIE update formula
        pass

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
        # TODO: Implement convex combination outside support
        pass

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
        # TODO: Implement R-factor computation
        pass

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
        # TODO: Implement PSNR = 10 * log10(max_val^2 / MSE)
        pass

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
        # TODO: Implement phase-to-depth conversion and RMSE computation
        pass

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
        # TODO: Implement per-pad averaging of h = phase * lambda / (4*pi)
        pass

    # -------------------------------------------------------------------------
    # Main Solve Interface
    # -------------------------------------------------------------------------

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
        # TODO: Implement dispatch logic:
        #   1. Initialize support, probe, scan positions, R_p if not provided
        #   2. Initialize object estimate (e.g., uniform R_p)
        #   3. Parse input_data into list of measured intensities
        #   4. Dispatch to _run_hio, _run_rpio, _run_pie, or _run_rppie
        #   5. Store error_history
        #   6. Return reconstructed object
        pass

    def _initialize_object(self) -> np.ndarray:
        """
        Create initial object estimate for iterative reconstruction.

        Returns:
            Complex 2D array initialized (e.g., uniform R_p or random phase).
        """
        # TODO: Implement initialization strategy
        pass


if __name__ == "__main__":
    # Load measured diffraction data
    input_data = np.load('dataset/input.npy')

    # Load auxiliary data if available
    # support_mask = np.load('dataset/support_mask.npy') if exists
    # probe = np.load('dataset/probe.npy') if exists
    # scan_positions = np.load('dataset/scan_positions.npy') if exists

    # Instantiate solver with rpPIE algorithm (best performing)
    solver = InverseSolver(
        algorithm="rppie",
        illumination="airy",
        beta=0.7,
        alpha_regularization=1e-3,
        num_epochs_pie=200,
        random_seed=42,
    )

    # Run reconstruction
    result = solver.solve(input_data)

    # Save reconstructed complex object
    np.save('output.npy', result)