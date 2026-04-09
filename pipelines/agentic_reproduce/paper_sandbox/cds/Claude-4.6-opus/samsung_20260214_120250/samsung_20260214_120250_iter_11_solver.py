import numpy as np
import time
from typing import Tuple, List, Optional

class InverseSolver:
    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_side: int = 9,
            step: int = 40,
            n_pos: int = 81,
            beta_obj: float = 0.5,
            beta_probe: float = 0.5,
            alpha_obj: float = 1e-2,
            alpha_probe: float = 1e-2,
            sigma_probe: float = 35.0,
            eps: float = 1e-12,
            max_time: float = 55.0,
            safety_margin: float = 1.0,
            alt_safety_margin: float = 0.5,
            alt_min_time: float = 8.0,
            alt_min_epochs: int = 30,
            max_epochs: int = 400,
            min_epochs: int = 100,
            epoch_time_estimate: float = 0.17,
            alt_epoch_time_estimate: float = 0.20,
            probe_update_start_epoch: int = 1,
            probe_energy_upper_ratio: float = 3.0,
            probe_energy_lower_ratio: float = 0.3,
            phase_histogram_bins: int = 360,
        ) -> None:
            # Grid / geometry
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_pos: int = n_pos

            # ePIE update hyperparameters
            self.beta_obj: float = beta_obj
            self.beta_probe: float = beta_probe
            self.alpha_obj: float = alpha_obj
            self.alpha_probe: float = alpha_probe

            # Probe initialization
            self.sigma_probe: float = sigma_probe

            # Numerical stability
            self.eps: float = eps

            # Timing budget
            self.max_time: float = max_time
            self.safety_margin: float = safety_margin
            self.alt_safety_margin: float = alt_safety_margin
            self.alt_min_time: float = alt_min_time
            self.alt_min_epochs: int = alt_min_epochs

            # Iteration control
            self.max_epochs: int = max_epochs
            self.min_epochs: int = min_epochs
            self.epoch_time_estimate: float = epoch_time_estimate
            self.alt_epoch_time_estimate: float = alt_epoch_time_estimate

            # Probe update schedule
            self.probe_update_start_epoch: int = probe_update_start_epoch

            # Probe energy stabilization
            self.probe_energy_upper_ratio: float = probe_energy_upper_ratio
            self.probe_energy_lower_ratio: float = probe_energy_lower_ratio

            # Post-processing
            self.phase_histogram_bins: int = phase_histogram_bins

            # Internal state (populated during solve)
            self.positions: Optional[np.ndarray] = None
            self.O: Optional[np.ndarray] = None
            self.P: Optional[np.ndarray] = None
            self.t_start: Optional[float] = None

    def _compute_scan_positions(self) -> np.ndarray:
            """Compute the (n_pos, 2) array of scan positions (row, col offsets)."""
            total_scan = self.probe_size + (self.n_side - 1) * self.step  # 256 + 320 = 576
            offset = (self.obj_size - total_scan) // 2  # 12

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions, dtype=np.int64)

    def _detect_dc_convention(self, intensities: np.ndarray) -> bool:
            """Detect whether DC component is at center or corner of diffraction patterns.

            Args:
                intensities: Raw measured intensities of shape (n_pos, probe_size, probe_size).

            Returns:
                True if DC is at center, False if at corner.
            """
            ps = self.probe_size
            center_region = slice(ps // 2 - 32, ps // 2 + 32)
            center_energy = np.mean(intensities[:, center_region, center_region])
            corner_energy = (
                np.mean(intensities[:, :32, :32])
                + np.mean(intensities[:, :32, -32:])
                + np.mean(intensities[:, -32:, :32])
                + np.mean(intensities[:, -32:, -32:])
            ) / 4.0

            return center_energy > corner_energy

    def _prepare_diff_amplitudes(
            self, intensities: np.ndarray, dc_at_center: bool
        ) -> np.ndarray:
            """Compute sqrt of intensities and optionally ifftshift to match FFT convention.

            Args:
                intensities: Raw measured intensities of shape (n_pos, probe_size, probe_size).
                dc_at_center: Whether DC is at center of the patterns.

            Returns:
                Diffraction amplitudes ready for use in Fourier constraint, shape (n_pos, probe_size, probe_size).
            """
            diff_amp = np.sqrt(np.maximum(intensities, 0.0))
            if dc_at_center:
                diff_amp = np.fft.ifftshift(diff_amp, axes=(-2, -1))
            return diff_amp

    def _initialize_object(self) -> np.ndarray:
            """Initialize the complex object as uniform ones.

            Returns:
                Complex object array of shape (obj_size, obj_size).
            """
            return np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

    def _initialize_probe(self, diff_amp: np.ndarray) -> np.ndarray:
            """Initialize Gaussian probe and scale to match measured amplitudes.

            Args:
                diff_amp: Diffraction amplitudes of shape (n_pos, probe_size, probe_size).

            Returns:
                Complex probe array of shape (probe_size, probe_size).
            """
            ps = self.probe_size
            yy, xx = np.mgrid[:ps, :ps]
            yy = yy - ps / 2.0
            xx = xx - ps / 2.0
            P = np.exp(-(xx**2 + yy**2) / (2 * self.sigma_probe**2)).astype(np.complex128)

            # Scale probe so that |FFT(P)| roughly matches measured amplitudes
            P_fft_amp = np.abs(np.fft.fft2(P))
            scale = np.mean(diff_amp) / (np.mean(P_fft_amp) + self.eps)
            P *= scale

            return P

    def forward(
            self, O: np.ndarray, P: np.ndarray, position: Tuple[int, int]
        ) -> np.ndarray:
            """Compute forward model: FFT of exit wave P * O_patch.

            Args:
                O: Complex object of shape (obj_size, obj_size).
                P: Complex probe of shape (probe_size, probe_size).
                position: (row, col) offset into the object for this scan position.

            Returns:
                Complex far-field diffraction pattern of shape (probe_size, probe_size).
            """
            ry, rx = position
            ps = self.probe_size
            O_patch = O[ry:ry + ps, rx:rx + ps]
            psi = P * O_patch
            Psi = np.fft.fft2(psi)
            return Psi

    def _fourier_magnitude_constraint(
            self, Psi: np.ndarray, measured_amp: np.ndarray
        ) -> np.ndarray:
            """Replace Fourier magnitudes with measured amplitudes, keep phases.

            Args:
                Psi: Complex Fourier-space wave of shape (probe_size, probe_size).
                measured_amp: Measured diffraction amplitude of shape (probe_size, probe_size).

            Returns:
                Corrected complex Fourier-space wave of shape (probe_size, probe_size).
            """
            Psi_amp = np.abs(Psi) + self.eps
            Psi_c = measured_amp * Psi / Psi_amp
            return Psi_c

    def _update_object(
            self,
            O: np.ndarray,
            P: np.ndarray,
            dpsi: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """Apply ePIE object update rule.

            Args:
                O: Current complex object of shape (obj_size, obj_size).
                P: Current complex probe of shape (probe_size, probe_size).
                dpsi: Exit wave difference (corrected - current) of shape (probe_size, probe_size).
                position: (row, col) offset into object.

            Returns:
                Updated complex object of shape (obj_size, obj_size).
            """
            ry, rx = position
            ps = self.probe_size

            P_conj = np.conj(P)
            P_abs_sq = np.abs(P) ** 2
            P_max_sq = np.max(P_abs_sq) + self.eps

            O[ry:ry + ps, rx:rx + ps] += self.beta_obj * P_conj / (P_max_sq + self.alpha_obj) * dpsi

            return O

    def _update_probe(
            self,
            P: np.ndarray,
            O_patch: np.ndarray,
            dpsi: np.ndarray,
        ) -> np.ndarray:
            """Apply ePIE probe update rule.

            Args:
                P: Current complex probe of shape (probe_size, probe_size).
                O_patch: Object patch at current position of shape (probe_size, probe_size).
                dpsi: Exit wave difference of shape (probe_size, probe_size).

            Returns:
                Updated complex probe of shape (probe_size, probe_size).
            """
            O_conj = np.conj(O_patch)
            O_abs_sq = np.abs(O_patch) ** 2
            O_max_sq = np.max(O_abs_sq) + self.eps

            P = P + self.beta_probe * O_conj / (O_max_sq + self.alpha_probe) * dpsi

            return P

    def _stabilize_probe_energy(
            self, P: np.ndarray, P_energy_init: float
        ) -> np.ndarray:
            """Rescale probe if its energy drifts too far from initial value.

            Args:
                P: Current complex probe of shape (probe_size, probe_size).
                P_energy_init: Initial probe energy (sum of |P|^2).

            Returns:
                Possibly rescaled complex probe of shape (probe_size, probe_size).
            """
            P_energy = np.sum(np.abs(P) ** 2)
            if (P_energy > self.probe_energy_upper_ratio * P_energy_init or
                    P_energy < self.probe_energy_lower_ratio * P_energy_init):
                P = P * np.sqrt(P_energy_init / (P_energy + self.eps))
            return P

    def _run_epie(
            self,
            diff_amp: np.ndarray,
            O: np.ndarray,
            P: np.ndarray,
            n_epochs: int,
            time_limit: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Run the core ePIE loop for a given number of epochs or until time limit.

            Args:
                diff_amp: Diffraction amplitudes of shape (n_pos, probe_size, probe_size).
                O: Initial complex object of shape (obj_size, obj_size).
                P: Initial complex probe of shape (probe_size, probe_size).
                n_epochs: Maximum number of epochs to run.
                time_limit: Absolute wall-clock time after which to stop.

            Returns:
                Tuple of (reconstructed object, reconstructed probe).
            """
            P_energy_init = np.sum(np.abs(P) ** 2)
            ps = self.probe_size

            for epoch in range(n_epochs):
                if time.time() > time_limit:
                    break

                # Random scan order each epoch
                order = np.random.permutation(self.n_pos)

                for idx in order:
                    ry, rx = self.positions[idx]

                    # Extract object patch
                    O_patch = O[ry:ry + ps, rx:rx + ps]

                    # Exit wave
                    psi = P * O_patch

                    # Forward propagation
                    Psi = np.fft.fft2(psi)

                    # Fourier magnitude constraint
                    Psi_c = self._fourier_magnitude_constraint(Psi, diff_amp[idx])

                    # Corrected exit wave
                    psi_c = np.fft.ifft2(Psi_c)

                    # Exit wave difference
                    dpsi = psi_c - psi

                    # Object update
                    O = self._update_object(O, P, dpsi, (ry, rx))

                    # Probe update (start after first epoch)
                    if epoch >= self.probe_update_start_epoch:
                        O_patch_new = O[ry:ry + ps, rx:rx + ps]
                        P = self._update_probe(P, O_patch_new, dpsi)

                # Stabilize probe energy after each epoch
                P = self._stabilize_probe_energy(P, P_energy_init)

            return O, P

    def _compute_residual(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
        ) -> float:
            """Compute total Fourier residual across all scan positions.

            Args:
                O: Complex object of shape (obj_size, obj_size).
                P: Complex probe of shape (probe_size, probe_size).
                diff_amp: Measured diffraction amplitudes of shape (n_pos, probe_size, probe_size).

            Returns:
                Scalar residual (sum of squared amplitude differences).
            """
            ps = self.probe_size
            res = 0.0
            for idx in range(self.n_pos):
                ry, rx = self.positions[idx]
                O_patch = O[ry:ry + ps, rx:rx + ps]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                res += np.sum((np.abs(Psi) - diff_amp[idx]) ** 2)
            return float(res)

    def _try_alternative_convention(
            self,
            intensities: np.ndarray,
            dc_at_center: bool,
            primary_residual: float,
            primary_O: np.ndarray,
            primary_P: np.ndarray,
            primary_n_epochs: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Run a shorter reconstruction with the opposite FFT convention and compare.

            Args:
                intensities: Raw measured intensities of shape (n_pos, probe_size, probe_size).
                dc_at_center: DC convention used in primary run.
                primary_residual: Residual from primary run.
                primary_O: Object from primary run.
                primary_P: Probe from primary run.
                primary_n_epochs: Number of epochs used in primary run.

            Returns:
                Best (object, probe) tuple chosen by residual comparison.
            """
            time_left = (self.t_start + self.max_time - self.safety_margin) - time.time()
            if time_left < self.alt_min_time:
                return primary_O, primary_P

            # Prepare alternative diffraction amplitudes with opposite convention
            diff_amp_alt = self._prepare_diff_amplitudes(intensities, not dc_at_center)

            # Initialize fresh object and probe for alternative run
            O_alt = self._initialize_object()
            P_alt = self._initialize_probe(diff_amp_alt)

            # Determine number of epochs for alternative run
            alt_epochs = int(time_left / self.alt_epoch_time_estimate) - 5
            alt_epochs = min(alt_epochs, primary_n_epochs)
            alt_epochs = max(alt_epochs, self.alt_min_epochs)

            # Time limit for alternative run
            alt_time_limit = self.t_start + self.max_time - self.safety_margin - self.alt_safety_margin

            # Run ePIE with alternative convention
            O_alt, P_alt = self._run_epie(diff_amp_alt, O_alt, P_alt, alt_epochs, alt_time_limit)

            # Compute residual for alternative run
            res_alt = self._compute_residual(O_alt, P_alt, diff_amp_alt)

            # Return whichever has lower residual
            if res_alt < primary_residual:
                return O_alt, P_alt
            else:
                return primary_O, primary_P

    def _remove_global_phase(self, O: np.ndarray) -> np.ndarray:
            """Remove global phase ambiguity by setting background phase to zero.

            Args:
                O: Complex object of shape (obj_size, obj_size).

            Returns:
                Phase-corrected complex object.
            """
            phase = np.angle(O)
            hist, bin_edges = np.histogram(phase.ravel(), bins=self.phase_histogram_bins)
            mode_idx = np.argmax(hist)
            phase_ref = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
            O = O * np.exp(-1j * phase_ref)
            return O

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """Convert complex object to (2, obj_size, obj_size) output array.

            Args:
                O: Complex object of shape (obj_size, obj_size).

            Returns:
                Float32 array of shape (2, obj_size, obj_size) — channel 0: amplitude, channel 1: phase.
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)
            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """Full reconstruction pipeline: load data, run ePIE, post-process, return result.

            Args:
                input_data: Measured diffraction intensities of shape (n_pos, probe_size, probe_size).

            Returns:
                Reconstructed object as float32 array of shape (2, obj_size, obj_size).
            """
            # 1. Start timer
            self.t_start = time.time()

            # 2. Preprocess intensities
            intensities = np.maximum(input_data, 0).astype(np.float64)

            # 3. Detect FFT convention
            dc_at_center = self._detect_dc_convention(intensities)

            # 4. Prepare diffraction amplitudes
            diff_amp = self._prepare_diff_amplitudes(intensities, dc_at_center)

            # 5. Compute scan positions
            self.positions = self._compute_scan_positions()

            # 6. Initialize object and probe
            O = self._initialize_object()
            P = self._initialize_probe(diff_amp)

            # 7. Determine number of epochs from time budget
            time_elapsed = time.time() - self.t_start
            time_remaining = self.max_time - self.safety_margin - time_elapsed
            max_epochs_from_time = int(time_remaining / self.epoch_time_estimate)
            n_epochs = min(max(max_epochs_from_time, self.min_epochs), self.max_epochs)

            # 8. Run primary ePIE
            time_limit = self.t_start + self.max_time - self.safety_margin
            O, P = self._run_epie(diff_amp, O, P, n_epochs, time_limit)

            # 9. Compute residual
            res_main = self._compute_residual(O, P, diff_amp)

            # 10. Try alternative FFT convention if time permits
            O, P = self._try_alternative_convention(
                intensities, dc_at_center, res_main, O, P, n_epochs
            )

            # 11. Remove global phase
            O = self._remove_global_phase(O)

            # 12. Format and return output
            output = self._format_output(O)
            return output



if __name__ == "__main__":
    # Load data
    input_data = np.load("dataset/input.npy")

    # Create solver and run
    solver = InverseSolver()
    result = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)