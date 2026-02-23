import numpy as np
import time
from typing import Tuple, Optional, List

class InverseSolver:
    def __init__(
            self,
            probe_size: int = 256,
            obj_size: int = 600,
            n_side: int = 9,
            step: int = 40,
            n_pos: int = 81,
            beta_obj: float = 1.0,
            beta_probe: float = 1.0,
            alpha_obj_ratio: float = 1e-3,
            alpha_probe_ratio: float = 1e-3,
            alpha_probe_initial: float = 1e-1,
            sigma_probe: float = 50.0,
            eps: float = 1e-12,
            probe_update_start_epoch: int = 2,
            quick_test_epochs: int = 5,
            max_epochs: int = 500,
            min_epochs: int = 50,
            time_limit: float = 54.0,
            quick_test_time_limit: float = 50.0,
            phase_hist_bins: int = 360,
            valid_amp_threshold: float = 0.1,
        ) -> None:
            self.probe_size: int = probe_size
            self.obj_size: int = obj_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_pos: int = n_pos
            self.beta_obj: float = beta_obj
            self.beta_probe: float = beta_probe
            self.alpha_obj_ratio: float = alpha_obj_ratio
            self.alpha_probe_ratio: float = alpha_probe_ratio
            self.alpha_probe_initial: float = alpha_probe_initial
            self.sigma_probe: float = sigma_probe
            self.eps: float = eps
            self.probe_update_start_epoch: int = probe_update_start_epoch
            self.quick_test_epochs: int = quick_test_epochs
            self.max_epochs: int = max_epochs
            self.min_epochs: int = min_epochs
            self.time_limit: float = time_limit
            self.quick_test_time_limit: float = quick_test_time_limit
            self.phase_hist_bins: int = phase_hist_bins
            self.valid_amp_threshold: float = valid_amp_threshold
            self.t_start: float = 0.0
            self.positions: np.ndarray = self._build_scan_positions()

    def _build_scan_positions(self) -> np.ndarray:
            """Compute the (81, 2) array of (row, col) scan positions on a 9x9 grid."""
            total_scan = self.probe_size + (self.n_side - 1) * self.step  # 576
            offset = (self.obj_size - total_scan) // 2  # 12

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions, dtype=np.int32)

    def _detect_dc_location(self, intensities: np.ndarray) -> bool:
            """Determine whether DC component is at center or corner of diffraction patterns.

            Args:
                intensities: Raw measured intensities, shape (n_pos, probe_size, probe_size).

            Returns:
                True if DC is at center, False if at corners.
            """
            ps = self.probe_size
            half_w = 32

            center_sl_r = slice(ps // 2 - half_w, ps // 2 + half_w)
            center_sl_c = slice(ps // 2 - half_w, ps // 2 + half_w)
            center_energy = np.mean(intensities[:, center_sl_r, center_sl_c])

            corner_energy = (
                np.mean(intensities[:, :half_w, :half_w])
                + np.mean(intensities[:, :half_w, -half_w:])
                + np.mean(intensities[:, -half_w:, :half_w])
                + np.mean(intensities[:, -half_w:, -half_w:])
            ) / 4.0

            return center_energy > corner_energy

    def _prepare_diffraction_amplitudes(
            self, intensities: np.ndarray, dc_at_center: bool
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Compute sqrt(I) and prepare both shifted and unshifted versions.

            Args:
                intensities: Non-negative measured intensities, shape (n_pos, ps, ps).
                dc_at_center: Whether DC is at center.

            Returns:
                Tuple of (primary_diff_amp, alternate_diff_amp), each shape (n_pos, ps, ps).
            """
            diff_amp = np.sqrt(intensities)

            if dc_at_center:
                # Primary: shift DC from center to corner (to match fft2 convention)
                primary = np.fft.ifftshift(diff_amp, axes=(-2, -1))
                # Alternate: leave as-is (DC at center, non-standard)
                alternate = diff_amp.copy()
            else:
                # Primary: already DC at corner, matches fft2 convention
                primary = diff_amp.copy()
                # Alternate: shift from corner to center then back (i.e., try ifftshift)
                alternate = np.fft.ifftshift(diff_amp, axes=(-2, -1))

            return primary, alternate

    def _init_probe(self, diff_amp: np.ndarray) -> np.ndarray:
            """Initialize Gaussian probe function scaled to match diffraction amplitudes.

            Args:
                diff_amp: Diffraction amplitudes for scaling, shape (n_pos, ps, ps).

            Returns:
                Complex probe array, shape (ps, ps).
            """
            ps = self.probe_size

            yy, xx = np.mgrid[:ps, :ps]
            yy = yy - ps / 2.0
            xx = xx - ps / 2.0

            sigma = self.sigma_probe
            P = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)).astype(np.complex128)

            # Scale probe so that |FFT(P)| matches mean measured diffraction amplitudes
            P_fft_amp = np.abs(np.fft.fft2(P))
            scale = np.mean(diff_amp) / (np.mean(P_fft_amp) + self.eps)
            P *= scale

            return P

    def _init_object(self) -> np.ndarray:
            """Initialize object as uniform complex array.

            Returns:
                Complex object array, shape (obj_size, obj_size).
            """
            return np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

    def forward(
            self, O: np.ndarray, P: np.ndarray, position: Tuple[int, int]
        ) -> np.ndarray:
            """Compute forward model: FFT of exit wave P * O_patch.

            Args:
                O: Complex object array, shape (obj_size, obj_size).
                P: Complex probe array, shape (ps, ps).
                position: (row, col) top-left corner of the patch in the object.

            Returns:
                Complex far-field diffraction pattern, shape (ps, ps).
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
            """Replace Fourier magnitudes with measured amplitudes, preserving phase.

            Args:
                Psi: Complex Fourier-space wave, shape (ps, ps).
                measured_amp: Measured diffraction amplitudes, shape (ps, ps).

            Returns:
                Corrected complex Fourier-space wave, shape (ps, ps).
            """
            Psi_amp = np.abs(Psi) + self.eps
            Psi_c = measured_amp * Psi / Psi_amp
            return Psi_c

    def _update_object_epie(
            self,
            O: np.ndarray,
            P: np.ndarray,
            dpsi: np.ndarray,
            position: Tuple[int, int],
            alpha_obj: float,
        ) -> np.ndarray:
            """Standard ePIE object update with per-pixel normalization.

            Args:
                O: Current object estimate, shape (obj_size, obj_size). Modified in-place.
                P: Current probe estimate, shape (ps, ps).
                dpsi: Exit wave difference (psi_corrected - psi), shape (ps, ps).
                position: (row, col) top-left corner of patch.
                alpha_obj: Regularization parameter for object update.

            Returns:
                Updated object array (same reference, modified in-place).
            """
            ry, rx = position
            ps = self.probe_size
            P_conj = np.conj(P)
            P_abs_sq = np.abs(P) ** 2  # per-pixel normalization
            O[ry:ry + ps, rx:rx + ps] += self.beta_obj * P_conj / (P_abs_sq + alpha_obj) * dpsi
            return O

    def _update_probe_epie(
            self,
            P: np.ndarray,
            O_patch: np.ndarray,
            dpsi: np.ndarray,
            alpha_probe: float,
        ) -> np.ndarray:
            """Standard ePIE probe update with per-pixel normalization.

            Args:
                P: Current probe estimate, shape (ps, ps). Modified in-place.
                O_patch: Object patch at current scan position, shape (ps, ps).
                dpsi: Exit wave difference, shape (ps, ps).
                alpha_probe: Regularization parameter for probe update.

            Returns:
                Updated probe array (same reference, modified in-place).
            """
            O_conj = np.conj(O_patch)
            O_abs_sq = np.abs(O_patch) ** 2  # per-pixel normalization
            P += self.beta_probe * O_conj / (O_abs_sq + alpha_probe) * dpsi
            return P

    def _run_epie(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
            n_epochs: int,
            time_limit: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Run ePIE reconstruction loop for a given number of epochs or until time limit.

            Args:
                O: Initial object estimate, shape (obj_size, obj_size).
                P: Initial probe estimate, shape (ps, ps).
                diff_amp: Measured diffraction amplitudes, shape (n_pos, ps, ps).
                positions: Scan positions array, shape (n_pos, 2).
                n_epochs: Maximum number of epochs to run.
                time_limit: Wall-clock time limit in seconds from self.t_start.

            Returns:
                Tuple of (reconstructed_object, reconstructed_probe).
            """
            ps = self.probe_size
            n_pos = len(positions)

            for epoch in range(n_epochs):
                if time.time() - self.t_start > time_limit:
                    break

                # Adaptive regularization based on current probe/object
                alpha_obj = self.alpha_obj_ratio * np.max(np.abs(P) ** 2)
                if epoch >= self.probe_update_start_epoch:
                    alpha_probe = self.alpha_probe_ratio * np.max(np.abs(O) ** 2)
                else:
                    alpha_probe = self.alpha_probe_initial

                # Shuffle scan positions each epoch
                order = np.random.permutation(n_pos)

                for idx in order:
                    ry, rx = positions[idx]
                    O_patch = O[ry:ry + ps, rx:rx + ps].copy()

                    # Exit wave
                    psi = P * O_patch

                    # Forward propagation
                    Psi = np.fft.fft2(psi)

                    # Fourier magnitude constraint
                    Psi_c = self._fourier_magnitude_constraint(Psi, diff_amp[idx])

                    # Corrected exit wave
                    psi_c = np.fft.ifft2(Psi_c)

                    # Difference
                    dpsi = psi_c - psi

                    # Object update (per-pixel normalization)
                    O = self._update_object_epie(O, P, dpsi, (ry, rx), alpha_obj)

                    # Probe update (per-pixel normalization), enabled after initial epochs
                    if epoch >= self.probe_update_start_epoch:
                        O_patch_new = O[ry:ry + ps, rx:rx + ps]
                        P = self._update_probe_epie(P, O_patch_new, dpsi, alpha_probe)

            return O, P

    def _compute_residual(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
        ) -> float:
            """Compute mean squared residual between model and measured amplitudes.

            Args:
                O: Object estimate, shape (obj_size, obj_size).
                P: Probe estimate, shape (ps, ps).
                diff_amp: Measured diffraction amplitudes, shape (n_pos, ps, ps).
                positions: Scan positions, shape (n_pos, 2).

            Returns:
                Mean squared amplitude residual (scalar).
            """
            ps = self.probe_size
            n_pos = len(positions)
            res = 0.0
            for idx in range(n_pos):
                ry, rx = positions[idx]
                psi = P * O[ry:ry + ps, rx:rx + ps]
                Psi = np.fft.fft2(psi)
                res += np.sum((np.abs(Psi) - diff_amp[idx]) ** 2)
            return res / n_pos

    def _select_convention(
            self,
            diff_amp_primary: np.ndarray,
            diff_amp_alternate: np.ndarray,
            positions: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Run quick test with both FFT conventions, return best (diff_amp, O, P).

            Args:
                diff_amp_primary: Primary convention amplitudes, shape (n_pos, ps, ps).
                diff_amp_alternate: Alternate convention amplitudes, shape (n_pos, ps, ps).
                positions: Scan positions, shape (n_pos, 2).

            Returns:
                Tuple of (best_diff_amp, best_O, best_P).
            """
            # Test primary convention
            O1 = self._init_object()
            P1 = self._init_probe(diff_amp_primary)
            O1, P1 = self._run_epie(
                O1, P1, diff_amp_primary, positions,
                self.quick_test_epochs, self.quick_test_time_limit
            )
            res1 = self._compute_residual(O1, P1, diff_amp_primary, positions)

            # Test alternate convention
            O2 = self._init_object()
            P2 = self._init_probe(diff_amp_alternate)
            O2, P2 = self._run_epie(
                O2, P2, diff_amp_alternate, positions,
                self.quick_test_epochs, self.quick_test_time_limit
            )
            res2 = self._compute_residual(O2, P2, diff_amp_alternate, positions)

            # Pick the convention with lower residual
            if res1 <= res2:
                return diff_amp_primary, O1, P1
            else:
                return diff_amp_alternate, O2, P2

    def _remove_global_phase(self, O: np.ndarray) -> np.ndarray:
            """Remove global phase ambiguity using histogram mode of phase values.

            Args:
                O: Reconstructed complex object, shape (obj_size, obj_size).

            Returns:
                Phase-corrected complex object.
            """
            phase = np.angle(O)
            amp = np.abs(O)

            # Focus on well-reconstructed regions with significant amplitude
            threshold = self.valid_amp_threshold * np.max(amp)
            mask_valid = amp > threshold

            if np.sum(mask_valid) > 100:
                phase_valid = phase[mask_valid]
                hist, bin_edges = np.histogram(
                    phase_valid, bins=self.phase_hist_bins, range=(-np.pi, np.pi)
                )
                mode_idx = np.argmax(hist)
                phase_ref = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
            else:
                phase_ref = np.median(phase)

            O = O * np.exp(-1j * phase_ref)
            return O

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """Convert complex object to (2, obj_size, obj_size) output: [amplitude, phase].

            Args:
                O: Complex object array, shape (obj_size, obj_size).

            Returns:
                Float32 array of shape (2, obj_size, obj_size).
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)
            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """Full reconstruction pipeline: load data, detect convention, run ePIE, post-process.

            Args:
                input_data: Measured diffraction intensities, shape (n_pos, ps, ps).

            Returns:
                Reconstructed object as float32 array, shape (2, obj_size, obj_size).
                    Channel 0: amplitude |O|
                    Channel 1: phase angle(O)
            """
            # 1. Start timer
            self.t_start = time.time()

            # 2. Preprocess intensities (clip negatives, convert to float64, sqrt)
            intensities = np.maximum(input_data, 0).astype(np.float64)

            # 3. Scan positions already built in __init__ (self.positions)
            positions = self.positions

            # 4. Detect DC location / FFT convention
            dc_at_center = self._detect_dc_location(intensities)

            # 5. Prepare both amplitude conventions
            diff_amp_primary, diff_amp_alternate = self._prepare_diffraction_amplitudes(
                intensities, dc_at_center
            )

            # 6. Quick convention test (run a few epochs with each, pick lower residual)
            diff_amp_use, O, P = self._select_convention(
                diff_amp_primary, diff_amp_alternate, positions
            )

            # 7. Full ePIE reconstruction with remaining time
            time_elapsed = time.time() - self.t_start
            time_remaining = self.time_limit - time_elapsed
            # Estimate time per epoch based on elapsed time for quick tests
            # quick_test ran 2 * quick_test_epochs epochs total
            total_quick_epochs = 2 * self.quick_test_epochs
            if total_quick_epochs > 0 and time_elapsed > 0.1:
                est_per_epoch = time_elapsed / total_quick_epochs
            else:
                est_per_epoch = 0.15  # fallback estimate

            if est_per_epoch > 0:
                max_epochs = int(time_remaining / est_per_epoch)
            else:
                max_epochs = self.max_epochs

            n_epochs = min(max(max_epochs, self.min_epochs), self.max_epochs)

            O, P = self._run_epie(O, P, diff_amp_use, positions, n_epochs, self.time_limit)

            # 8. Remove global phase ambiguity
            O = self._remove_global_phase(O)

            # 9. Format and return output
            output = self._format_output(O)
            return output



if __name__ == "__main__":
    # Load data
    input_data = np.load('dataset/input.npy')

    # Create solver and run reconstruction
    solver = InverseSolver()
    result = solver.solve(input_data)

    # Save output
    np.save('output.npy', result)