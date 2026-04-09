import numpy as np
import time
from typing import Tuple, List, Optional


class InverseSolver:
    def __init__(
        self,
        probe_size: int = 256,
        obj_size: int = 600,
        n_side: int = 9,
        step: int = 40,
        n_pos: int = 81,
        eps: float = 1e-8,
        total_time: float = 55.0,
        detect_time: float = 4.0,
        safety_margin: float = 2.0,
        beta_object: float = 1.0,
        beta_probe: float = 1.0,
        beta_momentum: float = 0.3,
        momentum_start_epoch: int = 2,
        object_init_amplitude: float = 0.7,
        probe_window_sigma: float = 80.0,
        amplitude_clip: float = 1.0,
        phase_histogram_bins: int = 720,
        phase_histogram_smooth_kernel: int = 21,
        residual_n_sample: int = 20,
        well_illuminated_threshold: int = 4,
        valid_amplitude_threshold: float = 0.1,
        stabilize_fraction: float = 0.3,
        wavelength_nm: float = 13.5,
        pixel_size_nm: float = 27.0,
    ) -> None:
        # Grid / geometry
        self.probe_size: int = probe_size
        self.obj_size: int = obj_size
        self.n_side: int = n_side
        self.step: int = step
        self.n_pos: int = n_pos

        # Numerical stability
        self.eps: float = eps

        # Timing budget
        self.total_time: float = total_time
        self.detect_time: float = detect_time
        self.safety_margin: float = safety_margin

        # ePIE step sizes (alpha=1 convention)
        self.beta_object: float = beta_object
        self.beta_probe: float = beta_probe

        # Momentum
        self.beta_momentum: float = beta_momentum
        self.momentum_start_epoch: int = momentum_start_epoch

        # Initialization
        self.object_init_amplitude: float = object_init_amplitude
        self.probe_window_sigma: float = probe_window_sigma

        # Constraints
        self.amplitude_clip: float = amplitude_clip

        # Phase alignment post-processing
        self.phase_histogram_bins: int = phase_histogram_bins
        self.phase_histogram_smooth_kernel: int = phase_histogram_smooth_kernel

        # Convention detection
        self.residual_n_sample: int = residual_n_sample

        # Masking thresholds
        self.well_illuminated_threshold: int = well_illuminated_threshold
        self.valid_amplitude_threshold: float = valid_amplitude_threshold

        # Two-phase reconstruction split
        self.stabilize_fraction: float = stabilize_fraction

        # Physical parameters
        self.wavelength_nm: float = wavelength_nm
        self.pixel_size_nm: float = pixel_size_nm

        # Derived: scan positions (computed once)
        self.positions: Optional[np.ndarray] = None

    def _build_positions(self) -> np.ndarray:
        """Compute (n_pos, 2) array of (row, col) scan positions on the object grid."""
        # TODO: Implement — 9x9 grid, step=40, centered in obj_size
        pass

    def _init_probe(self, diff_amp: np.ndarray) -> np.ndarray:
        """
        Initialize probe from average diffraction amplitude.
        Apply Gaussian spatial window. Normalize ONCE to unit max amplitude.

        Args:
            diff_amp: (n_pos, probe_size, probe_size) diffraction amplitudes

        Returns:
            P: (probe_size, probe_size) complex128 probe array
        """
        # TODO: Implement
        pass

    def _init_object(self) -> np.ndarray:
        """
        Initialize object as uniform complex reflector with amplitude = object_init_amplitude.

        Returns:
            O: (obj_size, obj_size) complex128 object array
        """
        # TODO: Implement
        pass

    def forward(
        self,
        O: np.ndarray,
        P: np.ndarray,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """
        Forward model: compute far-field diffraction amplitude for one scan position.

        Ψ(r) = P(r - R_j) · O(r)
        A(k) = |FFT2{Ψ(r)}|

        Args:
            O: (obj_size, obj_size) complex object
            P: (probe_size, probe_size) complex probe
            position: (row, col) top-left corner of the probe on the object grid

        Returns:
            amplitude: (probe_size, probe_size) real-valued diffraction amplitude
        """
        # TODO: Implement Fraunhofer propagation
        pass

    def _fourier_magnitude_replace(
        self,
        psi: np.ndarray,
        measured_amp: np.ndarray,
    ) -> np.ndarray:
        """
        Replace Fourier magnitude of exit wave with measured amplitude.

        Args:
            psi: (probe_size, probe_size) complex exit wave in real space
            measured_amp: (probe_size, probe_size) measured diffraction amplitude

        Returns:
            psi_corrected: (probe_size, probe_size) complex corrected exit wave in real space
        """
        # TODO: Implement FFT → magnitude replace → IFFT
        pass

    def _update_object_patch(
        self,
        O_patch: np.ndarray,
        P: np.ndarray,
        dpsi: np.ndarray,
        P_max_sq: float,
    ) -> np.ndarray:
        """
        ePIE object update (alpha=1):
        O_patch += conj(P) * dpsi / (P_max_sq + eps)

        Args:
            O_patch: (probe_size, probe_size) complex object patch
            P: (probe_size, probe_size) complex probe
            dpsi: (probe_size, probe_size) complex exit wave difference
            P_max_sq: scalar, max(|P|^2) over probe

        Returns:
            O_patch_updated: (probe_size, probe_size) complex updated object patch
        """
        # TODO: Implement
        pass

    def _update_probe(
        self,
        P: np.ndarray,
        O_patch_updated: np.ndarray,
        dpsi: np.ndarray,
        O_max_sq: float,
    ) -> np.ndarray:
        """
        ePIE probe update (alpha=1) using UPDATED object patch:
        P += conj(O_updated) * dpsi / (O_max_sq + eps)

        Args:
            P: (probe_size, probe_size) complex probe
            O_patch_updated: (probe_size, probe_size) complex updated object patch
            dpsi: (probe_size, probe_size) complex exit wave difference
            O_max_sq: scalar, max(|O_patch_updated|^2)

        Returns:
            P_updated: (probe_size, probe_size) complex updated probe
        """
        # TODO: Implement
        pass

    def _clip_object_amplitude(self, O: np.ndarray) -> np.ndarray:
        """
        Enforce physical reflectance constraint: |O(r)| <= amplitude_clip.

        Args:
            O: (obj_size, obj_size) or (probe_size, probe_size) complex array

        Returns:
            O_clipped: same shape, amplitude-clipped complex array
        """
        # TODO: Implement
        pass

    def _apply_momentum(
        self,
        O_current: np.ndarray,
        O_before_epoch: np.ndarray,
    ) -> np.ndarray:
        """
        Apply gentle momentum: O = O + beta_momentum * (O - O_before_epoch), then clip.

        Args:
            O_current: (obj_size, obj_size) complex object after epoch
            O_before_epoch: (obj_size, obj_size) complex object before epoch

        Returns:
            O_accelerated: (obj_size, obj_size) complex object with momentum applied
        """
        # TODO: Implement
        pass

    def _run_ePIE(
        self,
        O: np.ndarray,
        P: np.ndarray,
        diff_amp: np.ndarray,
        positions: np.ndarray,
        time_budget: float,
        use_momentum: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Run ePIE iterations (alpha=1) with optional momentum.

        Iterates epochs until time_budget is exhausted. Each epoch:
        1. Random permutation of scan positions
        2. For each position: forward → magnitude replace → object update → probe update
        3. Optionally apply momentum after epoch >= momentum_start_epoch

        Args:
            O: (obj_size, obj_size) complex object (modified in-place and returned)
            P: (probe_size, probe_size) complex probe (modified in-place and returned)
            diff_amp: (n_pos, probe_size, probe_size) measured diffraction amplitudes
            positions: (n_pos, 2) scan positions
            time_budget: float, seconds available
            use_momentum: bool, whether to apply momentum acceleration

        Returns:
            O: updated complex object
            P: updated complex probe
            n_epochs: number of completed epochs
        """
        # TODO: Implement main ePIE loop
        pass

    def _compute_residual(
        self,
        O: np.ndarray,
        P: np.ndarray,
        diff_amp: np.ndarray,
        positions: np.ndarray,
        n_sample: int = 20,
    ) -> float:
        """
        Compute normalized Fourier residual on a random subset of positions.

        Args:
            O: (obj_size, obj_size) complex object
            P: (probe_size, probe_size) complex probe
            diff_amp: (n_pos, probe_size, probe_size) measured amplitudes
            positions: (n_pos, 2) scan positions
            n_sample: number of positions to evaluate

        Returns:
            residual: float, normalized residual metric
        """
        # TODO: Implement
        pass

    def _detect_fft_convention(
        self,
        diff_amp: np.ndarray,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Test two FFT conventions (raw vs ifftshift) and pick the one with lower residual.
        Uses short ePIE runs (~2s each). Returns warm-started O, P, and chosen diff_amp.

        Args:
            diff_amp: (n_pos, probe_size, probe_size) raw diffraction amplitudes
            positions: (n_pos, 2) scan positions

        Returns:
            best_O: (obj_size, obj_size) complex object from best convention
            best_P: (probe_size, probe_size) complex probe from best convention
            best_damp: (n_pos, probe_size, probe_size) amplitudes for best convention
        """
        # TODO: Implement convention A (raw) and B (ifftshift), compare residuals
        pass

    def _compute_illumination_mask(
        self,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute illuminated and well-illuminated boolean masks from scan positions.

        Args:
            positions: (n_pos, 2) scan positions

        Returns:
            illuminated: (obj_size, obj_size) bool, any coverage
            well_illuminated: (obj_size, obj_size) bool, coverage >= threshold
        """
        # TODO: Implement
        pass

    def _align_phase(
        self,
        O: np.ndarray,
        well_illuminated: np.ndarray,
    ) -> np.ndarray:
        """
        Remove global phase offset by finding histogram mode in well-illuminated region.
        Substrate phase should map to ~0.

        Args:
            O: (obj_size, obj_size) complex object
            well_illuminated: (obj_size, obj_size) bool mask

        Returns:
            O_aligned: (obj_size, obj_size) complex object with global phase removed
        """
        # TODO: Implement histogram-based phase reference estimation
        pass

    def _postprocess(
        self,
        O: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Post-process reconstructed object: phase alignment, masking, output formatting.

        Args:
            O: (obj_size, obj_size) complex object
            positions: (n_pos, 2) scan positions

        Returns:
            output: (2, obj_size, obj_size) float32 — channel 0: amplitude, channel 1: phase
        """
        # TODO: Implement
        pass

    def solve(
        self,
        input_data: np.ndarray,
    ) -> np.ndarray:
        """
        Full inverse solver pipeline:
        1. Preprocess input intensities → diffraction amplitudes
        2. Build scan positions
        3. Detect FFT convention (warm-started)
        4. Main ePIE reconstruction in two phases (stabilize, then accelerate with momentum)
        5. Post-process and return (2, obj_size, obj_size) output

        Args:
            input_data: (81, 256, 256) real-valued diffraction intensity measurements

        Returns:
            output: (2, 600, 600) float32 array
                    channel 0: |O(r)| amplitude
                    channel 1: angle(O(r)) phase in radians
        """
        # TODO: Implement full pipeline
        pass


if __name__ == "__main__":
    # Load data
    input_data = np.load('dataset/input.npy')

    # Create solver and run
    solver = InverseSolver()
    result = solver.solve(input_data)

    # Save output
    np.save('output.npy', result)