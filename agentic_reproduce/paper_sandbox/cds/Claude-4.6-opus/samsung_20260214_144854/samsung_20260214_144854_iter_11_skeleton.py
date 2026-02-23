import numpy as np
import time
from typing import Tuple, List, Optional


class InverseSolver:
    def __init__(
        self,
        obj_size: int = 600,
        probe_size: int = 256,
        n_scan: int = 9,
        n_patterns: int = 81,
        wavelength: float = 13.5e-9,
        pixel_size: float = 27e-9,
        pad_cd_pixels: int = 22,
        pad_pitch_nm: float = 800.0,
        pad_cd_nm: float = 600.0,
        duty_cycle: float = 0.6,
        pad_array_size: int = 15,
        recession_min: float = 1.0e-9,
        recession_max: float = 3.0e-9,
        sigma_candidates: Optional[List[float]] = None,
        n_iter_quick: int = 30,
        n_iter_main: int = 300,
        n_iter_second: int = 200,
        beta_obj: float = 1.0,
        beta_probe: float = 1.0,
        probe_update_start: int = 5,
        patience: int = 80,
        patience_second: int = 60,
        probe_power_constraint_interval: int = 10,
        probe_power_soft_factor: float = 0.9,
        illum_threshold_fraction: float = 0.1,
        magnitude_coeff_var_threshold: float = 5.0,
        eps: float = 1e-12,
        random_seed: int = 42,
        second_run_time_budget: float = 120.0,
    ) -> None:
        # Grid / geometry
        self.obj_size: int = obj_size
        self.probe_size: int = probe_size
        self.n_scan: int = n_scan
        self.n_patterns: int = n_patterns

        # Physics
        self.wavelength: float = wavelength
        self.pixel_size: float = pixel_size
        self.pad_cd_pixels: int = pad_cd_pixels
        self.pad_pitch_nm: float = pad_pitch_nm
        self.pad_cd_nm: float = pad_cd_nm
        self.duty_cycle: float = duty_cycle
        self.pad_array_size: int = pad_array_size
        self.recession_min: float = recession_min
        self.recession_max: float = recession_max

        # Probe search
        self.sigma_candidates: List[float] = sigma_candidates if sigma_candidates is not None else [10, 20, 30, 40, 55, 75, 100]
        self.n_iter_quick: int = n_iter_quick

        # Main reconstruction
        self.n_iter_main: int = n_iter_main
        self.n_iter_second: int = n_iter_second
        self.beta_obj: float = beta_obj
        self.beta_probe: float = beta_probe
        self.probe_update_start: int = probe_update_start
        self.patience: int = patience
        self.patience_second: int = patience_second

        # Probe power constraint
        self.probe_power_constraint_interval: int = probe_power_constraint_interval
        self.probe_power_soft_factor: float = probe_power_soft_factor

        # Quality / masking
        self.illum_threshold_fraction: float = illum_threshold_fraction
        self.magnitude_coeff_var_threshold: float = magnitude_coeff_var_threshold

        # Numerics
        self.eps: float = eps
        self.random_seed: int = random_seed
        self.second_run_time_budget: float = second_run_time_budget

        # Derived state (populated during solve)
        self.use_fftshift: bool = False
        self.positions: List[Tuple[int, int]] = []
        self.mean_total_intensity: float = 0.0
        self.probe_target_energy: float = 0.0
        self.amplitudes: Optional[np.ndarray] = None
        self.illum_weight: Optional[np.ndarray] = None
        self.well_lit: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Forward / inverse propagation
    # ------------------------------------------------------------------

    def forward_prop(self, x: np.ndarray) -> np.ndarray:
        """Propagate a 2-D complex field to the far-field (Fourier) domain."""
        # TODO: Implement using np.fft.fft2 with optional fftshift based on self.use_fftshift
        pass

    def backward_prop(self, X: np.ndarray) -> np.ndarray:
        """Propagate a 2-D complex field back to real space (inverse Fourier)."""
        # TODO: Implement using np.fft.ifft2 with optional fftshift based on self.use_fftshift
        pass

    def forward(
        self,
        obj: np.ndarray,
        probe: np.ndarray,
        positions: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Simulate the full forward model for all scan positions.

        Parameters
        ----------
        obj : np.ndarray, shape (obj_size, obj_size), complex128
            Complex-valued object reflection function O(r).
        probe : np.ndarray, shape (probe_size, probe_size), complex128
            Illumination probe P(r).
        positions : list of (int, int)
            Top-left (row, col) for each scan position.

        Returns
        -------
        intensities : np.ndarray, shape (n_patterns, probe_size, probe_size), float64
            Simulated diffraction intensities |F{P * O_patch}|^2.
        """
        # TODO: Implement forward model I_j = |F{P * O[ry:ry+N, rx:rx+N]}|^2
        pass

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _detect_dc_position(self, data: np.ndarray) -> None:
        """Determine whether DC is centred or at corners and set self.use_fftshift."""
        # TODO: Implement centre-vs-corner energy comparison
        pass

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
        """Compute the 9×9 raster-scan positions centred on the object grid."""
        # TODO: Implement uniform grid with step = (obj_size - probe_size) / (n_scan - 1)
        pass

    def _make_probe(self, sigma: float) -> np.ndarray:
        """
        Construct a Gaussian probe with given sigma and normalise its energy
        so that Parseval's theorem is consistent with the measured intensity.

        Parameters
        ----------
        sigma : float
            Standard deviation (in pixels) of the Gaussian envelope.

        Returns
        -------
        P : np.ndarray, shape (probe_size, probe_size), complex128
        """
        # TODO: Implement Gaussian probe with energy normalisation
        pass

    def _constrain_probe_power(self, P: np.ndarray, target_energy: float) -> np.ndarray:
        """Soft probe-power constraint to prevent the probe absorbing object features."""
        # TODO: Implement soft rescaling toward target_energy
        pass

    def _compute_backprojection_init(self) -> np.ndarray:
        """
        Spectral (back-projection) initialisation of the object.

        Returns
        -------
        O_init : np.ndarray, shape (obj_size, obj_size), complex128
        """
        # TODO: Implement averaged back-projection from all diffraction patterns
        pass

    # ------------------------------------------------------------------
    # Probe selection
    # ------------------------------------------------------------------

    def _quick_ePIE(
        self,
        P_init: np.ndarray,
        n_iter: int,
    ) -> float:
        """
        Run a short ePIE reconstruction to evaluate a candidate probe.

        Returns
        -------
        rel_err : float
            Final relative Fourier-domain error.
        """
        # TODO: Implement short ePIE loop returning final relative error
        pass

    def _select_best_probe(self) -> Tuple[float, float]:
        """
        Search over sigma_candidates and return the best sigma and its error.

        Returns
        -------
        best_sigma : float
        best_err : float
        """
        # TODO: Iterate over self.sigma_candidates, call _quick_ePIE, pick best
        pass

    # ------------------------------------------------------------------
    # Main reconstruction
    # ------------------------------------------------------------------

    def _run_ePIE(
        self,
        O_init: np.ndarray,
        P_init: np.ndarray,
        n_iter: int,
        patience: int,
        time_budget: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Full ePIE reconstruction loop with early stopping.

        Parameters
        ----------
        O_init : np.ndarray, shape (obj_size, obj_size), complex128
        P_init : np.ndarray, shape (probe_size, probe_size), complex128
        n_iter : int
        patience : int
        time_budget : float or None
            Wall-clock seconds after which to stop (None = unlimited).

        Returns
        -------
        best_O : np.ndarray
        best_P : np.ndarray
        best_err : float
        best_epoch : int
        """
        # TODO: Implement ePIE with modulus replacement, probe update, probe power
        #       constraint, early stopping, and best-state tracking
        pass

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _compute_illumination_mask(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build illumination weight map and well-lit boolean mask.

        Returns
        -------
        illum_weight : np.ndarray, shape (obj_size, obj_size), float64
        well_lit : np.ndarray, shape (obj_size, obj_size), bool
        """
        # TODO: Implement illumination coverage from probe and positions
        pass

    def _quality_check(
        self,
        mag: np.ndarray,
        well_lit: np.ndarray,
    ) -> bool:
        """
        Return True if the reconstruction looks reasonable, False if diverged.
        """
        # TODO: Implement coefficient-of-variation check
        pass

    def _phase_reference(
        self,
        phase: np.ndarray,
        well_lit: np.ndarray,
    ) -> np.ndarray:
        """
        Median-based phase referencing: subtract substrate median, wrap, fix sign.

        Parameters
        ----------
        phase : np.ndarray, shape (obj_size, obj_size), float64
        well_lit : np.ndarray, shape (obj_size, obj_size), bool

        Returns
        -------
        phase_corrected : np.ndarray, shape (obj_size, obj_size), float64
        """
        # TODO: Implement median subtraction, wrapping, sign convention check
        pass

    def _normalise_magnitude(
        self,
        mag: np.ndarray,
        well_lit: np.ndarray,
    ) -> np.ndarray:
        """Normalise magnitude to a physically reasonable range [0, ~1]."""
        # TODO: Implement 99th-percentile-based rescaling
        pass

    # ------------------------------------------------------------------
    # Top-level solver
    # ------------------------------------------------------------------

    def solve(self, input_data: np.ndarray) -> np.ndarray:
        """
        End-to-end phase retrieval from measured diffraction intensities.

        Parameters
        ----------
        input_data : np.ndarray, shape (n_patterns, probe_size, probe_size), float64
            Measured far-field diffraction intensity patterns (non-negative).

        Returns
        -------
        output : np.ndarray, shape (2, obj_size, obj_size), float64
            Channel 0 — object magnitude.
            Channel 1 — object phase (radians).
        """
        # TODO: Orchestrate full pipeline:
        #   1. Seed RNG
        #   2. Compute amplitudes = sqrt(max(data, 0))
        #   3. Detect DC position (_detect_dc_position)
        #   4. Compute scan positions
        #   5. Compute mean total intensity & probe target energy
        #   6. Select best probe sigma (_select_best_probe)
        #   7. Build initial probe & back-projection object
        #   8. Run main ePIE (_run_ePIE with back-projection init)
        #   9. Run second ePIE (_run_ePIE with all-ones init, time-budgeted)
        #  10. Pick reconstruction with lower Fourier error
        #  11. Compute illumination mask
        #  12. Quality check & fallback
        #  13. Phase referencing
        #  14. Magnitude normalisation
        #  15. Assemble output array (2, obj_size, obj_size)
        pass


if __name__ == "__main__":
    # Load data
    input_data: np.ndarray = np.load("dataset/input.npy")

    # Instantiate solver with default hyperparameters
    solver: InverseSolver = InverseSolver()

    # Run phase retrieval
    result: np.ndarray = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)