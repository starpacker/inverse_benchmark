import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
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
        """
        Initialize the InverseSolver with all hyperparameters.

        Args:
            wavelength: EUV wavelength in meters (13.5 nm).
            pixel_size: Pixel size in meters (27 nm).
            detector_size: Detector array dimension (2048 x 2048).
            pad_pixels: Number of pixels per pad side (22).
            pitch_pixels: Number of pixels per pitch period (~30).
            array_size: Number of pads per side in the array (15).
            duty_cycle: Ratio of pad CD to pitch (0.6).
            h_min: Minimum pad recession depth in meters (1 nm).
            h_max: Maximum pad recession depth in meters (3 nm).
            beta: Feedback parameter for HIO/PIE update (0 < beta <= 1).
            alpha: Regularization constant for PIE denominator.
            n_iterations: Number of iterations (or epochs for ptychography).
            probe_type: Illumination type, one of "airy", "speckle", "plane".
            probe_diameter_pixels: Diameter of the probe in pixels (~160).
            scan_step_pixels: Ptychographic scan step size in pixels (40).
            noise_sigma_fraction: Noise std as fraction of max intensity.
            algorithm: Algorithm to use: "rpIO", "rpPIE", "HIO", "PIE".
            random_seed: Random seed for reproducibility.
        """
        # TODO: Store all hyperparameters as instance attributes
        # TODO: Initialize RNG with random_seed
        # TODO: Precompute derived quantities (object grid size, scan positions, etc.)
        pass

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
        # TODO: Implement TMM using Henke/CXRO optical constants for
        #       SiCN, SiO2, Si, Cu at lambda = 13.5 nm
        pass

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
        # TODO: Build 15x15 pad array with per-pad recession depths
        # TODO: Assign R_Cu * exp(i * 4*pi*h/lambda) inside pads
        # TODO: Assign R_p outside pads (substrate reflectivity prior)
        # TODO: Generate boolean support mask
        pass

    def _build_probe(
        self,
    ) -> np.ndarray:
        """
        Construct the complex-valued illumination probe function P(r).

        Returns:
            Complex128 array of shape (probe_diameter_pixels, probe_diameter_pixels).
        """
        # TODO: Implement Airy spot probe: P ~ [2*J1(pi*r/r0) / (pi*r/r0)]^2
        # TODO: Implement speckle probe: IFFT of circ * exp(i*phi_rand)
        # TODO: Implement plane wave probe (uniform amplitude)
        # TODO: Normalize probe
        pass

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
        # TODO: Compute grid of scan positions with scan_step_pixels spacing
        # TODO: Ensure probe stays within object bounds
        pass

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
        # TODO: For each scan position, form exit wave Psi = P(r - R_j) * O(r)
        # TODO: Zero-pad exit wave to (detector_size, detector_size)
        # TODO: Compute FFT2 with proper fftshift
        # TODO: Compute |FFT|^2
        # TODO: Add Gaussian noise with specified sigma
        # TODO: Clip to non-negative
        pass

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
        # TODO: Compute |psi_k|, handle near-zero values
        # TODO: Return measured_amplitude * psi_k / |psi_k|
        pass

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
        # TODO: Inside support: keep obj_new
        # TODO: Outside support: blend obj_current toward R_p with beta
        pass

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
        # TODO: Standard HIO feedback outside support
        pass

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
        # TODO: Compute delta_Psi = exit_wave_corrected - exit_wave_current
        # TODO: Compute P_max = max(|P|)
        # TODO: Compute numerator: |P| / |P_max| * conj(P) * beta * delta_Psi
        # TODO: Compute denominator: |P|^2 + alpha
        # TODO: Add update to object at correct spatial position
        pass

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
        # TODO: Compute theta* = arg(sum(conj(O_recon) * O_true))
        # TODO: Return O_recon * exp(i * theta*)
        pass

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
        # TODO: Forward propagate current estimate
        # TODO: Compute normalized squared difference of amplitudes
        pass

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
        # TODO: Resolve phase ambiguity
        # TODO: Compute normalized squared error
        pass

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
        # TODO: Resolve global phase ambiguity first
        # TODO: Compute arg difference only within support
        # TODO: Return mean absolute phase error
        pass

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
        # TODO: Initialize O_0 with random complex values or R_p + random phase on S
        # TODO: Main iteration loop:
        #   1. Zero-pad and FFT2
        #   2. Fourier modulus replacement
        #   3. IFFT2 and crop
        #   4. Apply support + reflectivity prior constraint
        #   5. Record error metrics
        pass

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
        # TODO: Initialize O_0 = R_p everywhere
        # TODO: Main epoch loop:
        #   For each scan position (randomized order):
        #     1. Form exit wave Psi = P(r - R_j) * O(r)
        #     2. FFT2 exit wave (zero-padded)
        #     3. Fourier modulus replacement
        #     4. IFFT2 to get corrected exit wave
        #     5. ePIE object update
        #     6. Apply reflectivity prior outside support
        #   Record error metrics after each epoch
        pass

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
        # TODO: Standard HIO without reflectivity prior (expected to not converge well)
        pass

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
        # TODO: Standard ePIE without reflectivity prior enforcement
        pass

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
        # TODO: Auto-generate missing inputs (support, probe, scan positions, R_p)
        # TODO: Dispatch to appropriate algorithm (_solve_rpIO, _solve_rpPIE, etc.)
        # TODO: Return reconstructed object
        pass


if __name__ == "__main__":
    # Load measured diffraction data
    input_data = np.load('dataset/input.npy')

    # Instantiate solver with default hyperparameters
    solver = InverseSolver(
        wavelength=13.5e-9,
        pixel_size=27e-9,
        detector_size=2048,
        pad_pixels=22,
        pitch_pixels=30,
        array_size=15,
        beta=0.9,
        alpha=1e-8,
        n_iterations=300,
        probe_type="speckle",
        probe_diameter_pixels=160,
        scan_step_pixels=40,
        noise_sigma_fraction=0.01,
        algorithm="rpPIE",
        random_seed=42,
    )

    # Run phase retrieval
    result = solver.solve(measured_data=input_data)

    # Save reconstructed complex object
    np.save('output.npy', result)