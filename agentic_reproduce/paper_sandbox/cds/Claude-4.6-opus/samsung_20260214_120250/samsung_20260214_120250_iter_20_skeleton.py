import numpy as np
import time
from typing import Tuple, Optional, Dict, Any


class InverseSolver:
    def __init__(
        self,
        obj_size: int = 600,
        probe_size: int = 256,
        n_side: int = 9,
        step: int = 40,
        wavelength: float = 13.5,
        pixel_size: float = 27.0,
        pad_pixels: int = 22,
        duty_cycle: float = 0.6,
        beta_O: float = 0.8,
        beta_P: float = 0.5,
        alpha: float = 0.1,
        probe_warmup_epochs: int = 5,
        probe_sigma: float = 50.0,
        object_init_amp: float = 0.85,
        phase_ramp_interval: int = 5,
        best_track_interval: int = 10,
        residual_subsample: int = 20,
        amp_percentile: float = 99.0,
        valid_amp_threshold: float = 0.1,
        well_lit_fraction: float = 0.3,
        support_threshold: float = 0.5,
        convention_test_epochs: int = 2,
        total_time_budget: float = 55.0,
        postprocess_reserve: float = 5.0,
        eps: float = 1e-12,
        n_sicn: complex = complex(0.9791, 0.0059),
        n_sio2: complex = complex(0.9780, 0.0108),
        n_si: complex = complex(0.9991, 0.0018),
        n_cu: complex = complex(0.9625, 0.0613),
    ) -> None:
        # Grid / geometry
        self.obj_size: int = obj_size
        self.probe_size: int = probe_size
        self.n_side: int = n_side
        self.step: int = step
        self.n_pos: int = n_side * n_side
        self.wavelength: float = wavelength
        self.pixel_size: float = pixel_size
        self.pad_pixels: int = pad_pixels
        self.duty_cycle: float = duty_cycle

        # Reconstruction hyperparameters
        self.beta_O: float = beta_O
        self.beta_P: float = beta_P
        self.alpha: float = alpha
        self.probe_warmup_epochs: int = probe_warmup_epochs
        self.probe_sigma: float = probe_sigma
        self.object_init_amp: float = object_init_amp

        # Maintenance intervals
        self.phase_ramp_interval: int = phase_ramp_interval
        self.best_track_interval: int = best_track_interval
        self.residual_subsample: int = residual_subsample

        # Post-processing
        self.amp_percentile: float = amp_percentile
        self.valid_amp_threshold: float = valid_amp_threshold
        self.well_lit_fraction: float = well_lit_fraction
        self.support_threshold: float = support_threshold

        # Timing
        self.convention_test_epochs: int = convention_test_epochs
        self.total_time_budget: float = total_time_budget
        self.postprocess_reserve: float = postprocess_reserve

        # Numerical
        self.eps: float = eps

        # Material optical constants (CXRO @ 13.5 nm)
        self.n_sicn: complex = n_sicn
        self.n_sio2: complex = n_sio2
        self.n_si: complex = n_si
        self.n_cu: complex = n_cu

        # Derived: scan positions
        self.positions: np.ndarray = self._build_positions()

        # State (populated during solve)
        self.O: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.scan_count: Optional[np.ndarray] = None
        self.illuminated: Optional[np.ndarray] = None
        self.convention: Optional[str] = None

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _build_positions(self) -> np.ndarray:
        # TODO: Implement — compute (n_pos, 2) int array of (row, col) scan offsets
        pass

    # ------------------------------------------------------------------
    # Probe initialisation
    # ------------------------------------------------------------------

    def _init_probe(self, diff_amp: np.ndarray) -> np.ndarray:
        # TODO: Implement — Gaussian probe, energy-matched to mean diffraction intensity
        pass

    # ------------------------------------------------------------------
    # Object initialisation
    # ------------------------------------------------------------------

    def _init_object(self) -> np.ndarray:
        # TODO: Implement — uniform complex array of amplitude object_init_amp
        pass

    # ------------------------------------------------------------------
    # Forward model (single position)
    # ------------------------------------------------------------------

    def forward(
        self,
        O: np.ndarray,
        P: np.ndarray,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """
        Compute far-field diffraction amplitude for one scan position.

        Parameters
        ----------
        O : np.ndarray, shape (obj_size, obj_size), complex128
            Current object estimate.
        P : np.ndarray, shape (probe_size, probe_size), complex128
            Current probe estimate.
        position : Tuple[int, int]
            (row, col) top-left corner of the probe window in the object.

        Returns
        -------
        np.ndarray, shape (probe_size, probe_size), complex128
            Far-field (Fourier-domain) exit wave Ψ(k).
        """
        # TODO: Implement — extract patch, multiply by probe, FFT2
        pass

    # ------------------------------------------------------------------
    # Fourier magnitude replacement
    # ------------------------------------------------------------------

    def _replace_magnitude(
        self,
        Psi: np.ndarray,
        measured_amp: np.ndarray,
    ) -> np.ndarray:
        # TODO: Implement — replace |Ψ| with measured amplitude, keep phase
        pass

    # ------------------------------------------------------------------
    # Single-position rPIE update
    # ------------------------------------------------------------------

    def _update_position(
        self,
        O: np.ndarray,
        P: np.ndarray,
        measured_amp: np.ndarray,
        position: Tuple[int, int],
        epoch: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Implement — rPIE object (and optionally probe) update for one position
        pass

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    def _compute_residual(
        self,
        O: np.ndarray,
        P: np.ndarray,
        diff_amp: np.ndarray,
        n_sample: int,
    ) -> float:
        # TODO: Implement — normalised Fourier residual over n_sample positions
        pass

    # ------------------------------------------------------------------
    # Support / illumination mask
    # ------------------------------------------------------------------

    def _build_illumination_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Implement — scan_count array and boolean illuminated mask
        pass

    def _build_support_mask(self, O: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # TODO: Implement — detect pad regions by amplitude thresholding
        pass

    # ------------------------------------------------------------------
    # Phase ramp removal
    # ------------------------------------------------------------------

    def _remove_phase_ramp(
        self,
        O: np.ndarray,
        illuminated: np.ndarray,
    ) -> np.ndarray:
        # TODO: Implement — weighted least-squares linear phase ramp subtraction
        pass

    # ------------------------------------------------------------------
    # Global phase removal
    # ------------------------------------------------------------------

    def _remove_global_phase(
        self,
        O: np.ndarray,
        well_lit: np.ndarray,
    ) -> np.ndarray:
        # TODO: Implement — rotate O so mean phase in well-lit region ≈ 0
        pass

    # ------------------------------------------------------------------
    # Amplitude normalisation
    # ------------------------------------------------------------------

    def _normalise_amplitude(
        self,
        O: np.ndarray,
        illuminated: np.ndarray,
    ) -> np.ndarray:
        # TODO: Implement — scale amplitude to [0, 1] using percentile clipping
        pass

    # ------------------------------------------------------------------
    # FFT convention detection
    # ------------------------------------------------------------------

    def _detect_convention(
        self,
        diff_amp_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Run a short rPIE trial with both raw and ifftshifted data,
        return the diffraction amplitude array that yields lower residual.

        Parameters
        ----------
        diff_amp_raw : np.ndarray, shape (n_pos, probe_size, probe_size)
            Square-root of measured intensities (raw ordering).

        Returns
        -------
        np.ndarray
            The diffraction amplitude array with the winning convention.
        """
        # TODO: Implement — 2-epoch test for each convention, pick winner
        pass

    # ------------------------------------------------------------------
    # Core rPIE loop
    # ------------------------------------------------------------------

    def _run_rpie(
        self,
        diff_amp: np.ndarray,
        time_budget: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Full rPIE reconstruction with rpPIE support prior.

        Parameters
        ----------
        diff_amp : np.ndarray, shape (n_pos, probe_size, probe_size)
            Diffraction amplitudes (correct FFT convention).
        time_budget : float
            Wall-clock seconds available.

        Returns
        -------
        best_O : np.ndarray, shape (obj_size, obj_size), complex128
        best_P : np.ndarray, shape (probe_size, probe_size), complex128
        best_residual : float
        n_epochs : int
        """
        # TODO: Implement — epoch loop with object/probe updates, phase ramp
        #       removal, support enforcement, best-tracking by residual
        pass

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(self, O: np.ndarray) -> np.ndarray:
        """
        Final post-processing: phase ramp removal, global phase removal,
        amplitude normalisation, masking.

        Parameters
        ----------
        O : np.ndarray, shape (obj_size, obj_size), complex128

        Returns
        -------
        np.ndarray, shape (2, obj_size, obj_size), float32
            Channel 0: amplitude |O|, Channel 1: phase angle(O).
        """
        # TODO: Implement
        pass

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, input_data: np.ndarray) -> np.ndarray:
        """
        End-to-end solver: convention detection → rPIE → post-processing.

        Parameters
        ----------
        input_data : np.ndarray, shape (n_pos, probe_size, probe_size)
            Measured far-field diffraction intensities I(k).

        Returns
        -------
        np.ndarray, shape (2, obj_size, obj_size), float32
            Reconstructed object: [amplitude, phase].
        """
        # TODO: Implement
        #   1. Preprocess: clamp negatives, sqrt → diff_amp
        #   2. Detect FFT convention
        #   3. Run rPIE with remaining time budget
        #   4. Post-process and return (2, H, W) float32 array
        pass


if __name__ == "__main__":
    t_start = time.time()

    # Load data
    input_data = np.load("dataset/input.npy")

    # Instantiate solver
    solver = InverseSolver(total_time_budget=55.0)

    # Solve
    result = solver.solve(input_data)

    # Save
    np.save("output.npy", result)

    print(f"Total wall time: {time.time() - t_start:.1f}s, output shape: {result.shape}")