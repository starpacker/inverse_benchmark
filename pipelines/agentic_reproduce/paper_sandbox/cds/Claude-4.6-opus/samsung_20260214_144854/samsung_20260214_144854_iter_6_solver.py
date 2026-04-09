import numpy as np
from typing import Tuple, List, Optional, Callable

class InverseSolver:
    """
    Coherent Diffractive Imaging (CDI) phase retrieval solver using rPIE/ePIE
    for ptychographic reconstruction of copper pad topography.
    """

    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_scan: int = 9,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            alpha: float = 0.3,
            n_iterations: int = 800,
            probe_update_start: int = 20,
            probe_sigma_candidates: List[int] = None,
            probe_sigma_search_iters: int = 10,
            early_stop_patience: int = 200,
            fft_convention_switch_epoch: int = 20,
            fft_convergence_threshold: float = 0.95,
            poor_convergence_threshold: float = 0.5,
            modulus_eps: float = 1e-12,
            update_eps: float = 1e-12,
        ) -> None:
            """
            Initialize the inverse solver with all hyperparameters.

            Args:
                obj_size: Size of the reconstructed object grid (pixels per side).
                probe_size: Size of the illumination probe (pixels per side).
                n_scan: Number of scan positions per axis (n_scan x n_scan grid).
                wavelength: EUV wavelength in meters.
                pixel_size: Real-space pixel size in meters.
                alpha: rPIE regularization parameter (constant, no annealing).
                n_iterations: Maximum number of reconstruction iterations.
                probe_update_start: Epoch at which probe updates begin.
                probe_sigma_candidates: List of Gaussian sigma values to search for probe init.
                probe_sigma_search_iters: Number of quick ePIE iterations for probe sigma selection.
                early_stop_patience: Stop if no improvement for this many epochs.
                fft_convention_switch_epoch: Epoch at which to check FFT convention.
                fft_convergence_threshold: Threshold ratio to trigger FFT convention switch.
                poor_convergence_threshold: Forward consistency threshold for fallback.
                modulus_eps: Small constant for modulus replacement denominator.
                update_eps: Small constant for rPIE update denominator.
            """
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_scan: int = n_scan
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.alpha: float = alpha
            self.n_iterations: int = n_iterations
            self.probe_update_start: int = probe_update_start
            self.probe_sigma_candidates: List[int] = probe_sigma_candidates or [20, 30, 40, 55]
            self.probe_sigma_search_iters: int = probe_sigma_search_iters
            self.early_stop_patience: int = early_stop_patience
            self.fft_convention_switch_epoch: int = fft_convention_switch_epoch
            self.fft_convergence_threshold: float = fft_convergence_threshold
            self.poor_convergence_threshold: float = poor_convergence_threshold
            self.modulus_eps: float = modulus_eps
            self.update_eps: float = update_eps

            # Derived constants
            self.n_positions: int = n_scan * n_scan  # Total number of scan positions (81)
            self.n_pixels: int = probe_size * probe_size  # Total pixels per diffraction pattern (65536)
            self.scan_step: float = float(obj_size - probe_size) / float(n_scan - 1)  # Step between scan positions (43.0)

            # State variables set during solve
            self.use_shift: bool = False
            self.positions: List[Tuple[int, int]] = []
            self.probe: Optional[np.ndarray] = None
            self.object_estimate: Optional[np.ndarray] = None
            self.best_object: Optional[np.ndarray] = None
            self.best_probe: Optional[np.ndarray] = None
            self.best_fourier_error: float = np.inf
            self.best_epoch: int = 0
            self.errors: List[float] = []

    def _detect_fft_convention(self, data: np.ndarray) -> bool:
            """
            Detect whether the input diffraction data has DC at center (fftshifted)
            or at corner (unshifted) by comparing average intensity values.

            Args:
                data: Input diffraction intensity patterns, shape (N, H, W).

            Returns:
                True if DC is at center (use fftshift), False otherwise.
            """
            avg_pattern = np.mean(data, axis=0)
            h, w = avg_pattern.shape
            center_r, center_c = h // 2, w // 2

            # Compute mean intensity in a small region around the center
            r = 5  # radius of the region to average
            center_val = np.mean(
                avg_pattern[
                    max(0, center_r - r) : min(h, center_r + r + 1),
                    max(0, center_c - r) : min(w, center_c + r + 1),
                ]
            )

            # Compute mean intensity in a small region around the corners
            corner_val = np.mean(
                [
                    np.mean(avg_pattern[:r, :r]),
                    np.mean(avg_pattern[:r, -r:]),
                    np.mean(avg_pattern[-r:, :r]),
                    np.mean(avg_pattern[-r:, -r:]),
                ]
            )

            # DC component (zero frequency) has the highest intensity
            # If center_val >> corner_val, DC is at center (data is fftshifted)
            if center_val > corner_val * 5:
                return True
            else:
                return False

    def forward_prop(self, x: np.ndarray) -> np.ndarray:
            """
            Forward propagation (real space -> Fourier space) with correct FFT convention.

            Args:
                x: Complex-valued 2D array in real space, shape (probe_size, probe_size).

            Returns:
                Complex-valued 2D array in Fourier space, shape (probe_size, probe_size).
            """
            if self.use_shift:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            else:
                return np.fft.fft2(x)

    def backward_prop(self, X: np.ndarray) -> np.ndarray:
            """
            Backward propagation (Fourier space -> real space) with correct FFT convention.

            Args:
                X: Complex-valued 2D array in Fourier space, shape (probe_size, probe_size).

            Returns:
                Complex-valued 2D array in real space, shape (probe_size, probe_size).
            """
            if self.use_shift:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))
            else:
                return np.fft.ifft2(X)

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
            """
            Compute the 2D grid of scan positions for ptychographic scanning.

            Returns:
                List of (row, col) tuples for each scan position, length n_scan^2.
            """
            positions = []
            for iy in range(self.n_scan):
                for ix in range(self.n_scan):
                    ry = int(round(iy * self.scan_step))
                    rx = int(round(ix * self.scan_step))
                    positions.append((ry, rx))
            return positions

    def _compute_parseval_probe_energy(
            self, amplitudes: np.ndarray
        ) -> float:
            """
            Compute the target probe energy using Parseval's theorem.
            target_energy = mean_total_intensity / N_pixels

            Args:
                amplitudes: Square root of measured intensities, shape (N, H, W).

            Returns:
                Target sum of |P|^2 for correct probe scaling.
            """
            # Intensities = amplitudes^2
            # For each pattern j: sum(I_j) = sum(|FFT(P * O_j)|^2) = N * sum(|P * O_j|^2)
            # If |O| ~ 1 on average: sum(I_j) ~ N * sum(|P|^2)
            # So: sum(|P|^2) ~ mean(sum(I_j)) / N
            # where N = probe_size * probe_size = N_pixels

            intensities = amplitudes ** 2  # shape (N, H, W)
            mean_total_intensity = np.mean(np.sum(intensities, axis=(1, 2)))
            target_energy = mean_total_intensity / float(self.n_pixels)
            return target_energy

    def _init_gaussian_probe(
            self, sigma: int, target_energy: float
        ) -> np.ndarray:
            """
            Initialize a Gaussian probe with given sigma and scaled to target energy.

            Args:
                sigma: Standard deviation of the Gaussian in pixels.
                target_energy: Target value for sum(|P|^2).

            Returns:
                Complex-valued probe array, shape (probe_size, probe_size).
            """
            half = self.probe_size // 2  # 128
            Y, X = np.mgrid[-half:half, -half:half]
            R2 = (X.astype(np.float64) ** 2 + Y.astype(np.float64) ** 2)

            P = np.exp(-R2 / (2.0 * float(sigma) ** 2)).astype(np.complex128)

            # Scale probe so that sum(|P|^2) = target_energy (Parseval-correct scaling)
            current_energy = np.sum(np.abs(P) ** 2)
            scale = np.sqrt(target_energy / (current_energy + 1e-30))
            P *= scale

            return P

    def _run_quick_epie(
            self,
            object_init: np.ndarray,
            probe: np.ndarray,
            amplitudes: np.ndarray,
            n_iters: int,
        ) -> float:
            """
            Run a short ePIE reconstruction to evaluate a probe candidate.

            Args:
                object_init: Initial object estimate, shape (obj_size, obj_size).
                probe: Probe candidate, shape (probe_size, probe_size).
                amplitudes: Measured amplitudes, shape (N, H, W).
                n_iters: Number of quick iterations.

            Returns:
                Final relative Fourier error after n_iters iterations.
            """
            O = object_init.copy()
            P = probe.copy()
            total_amp_energy = np.sum(amplitudes ** 2)

            rel_err = np.inf

            for epoch in range(n_iters):
                order = np.random.permutation(self.n_positions)
                epoch_err = 0.0

                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]

                    # Extract object patch
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]

                    # Exit wave
                    psi = P * O_patch

                    # Forward propagate
                    Psi = self.forward_prop(psi)

                    # Compute amplitude
                    mag = np.abs(Psi)

                    # Fourier error accumulation
                    epoch_err += np.sum((mag - amplitudes[j]) ** 2)

                    # Modulus constraint
                    Psi_c = amplitudes[j] * Psi / (mag + self.modulus_eps)

                    # Back-propagate corrected exit wave
                    psi_c = self.backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # ePIE object update (alpha=1 simplification: denom = |P|^2_max)
                    P_abs2 = np.abs(P) ** 2
                    P_max = np.max(P_abs2)
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] += (
                        np.conj(P) * dpsi / (P_max + self.update_eps)
                    )

                rel_err = epoch_err / (total_amp_energy + 1e-30)

            return rel_err

    def _select_best_probe(self, amplitudes: np.ndarray) -> np.ndarray:
            """
            Grid search over probe sigma candidates, running quick ePIE for each,
            and return the probe with lowest Fourier error.

            Args:
                amplitudes: Measured amplitudes, shape (N, H, W).

            Returns:
                Best probe array, shape (probe_size, probe_size).
            """
            target_energy = self._compute_parseval_probe_energy(amplitudes)

            best_probe = None
            best_err = np.inf

            for sigma in self.probe_sigma_candidates:
                P_test = self._init_gaussian_probe(sigma, target_energy)

                O_test = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

                err = self._run_quick_epie(O_test, P_test, amplitudes, self.probe_sigma_search_iters)

                if err < best_err:
                    best_err = err
                    best_probe = P_test.copy()

            # Fallback: if no probe was selected (shouldn't happen), use middle sigma
            if best_probe is None:
                fallback_sigma = self.probe_sigma_candidates[len(self.probe_sigma_candidates) // 2]
                best_probe = self._init_gaussian_probe(fallback_sigma, target_energy)

            return best_probe

    def _rpie_object_update(
            self,
            O_patch: np.ndarray,
            P: np.ndarray,
            dpsi: np.ndarray,
        ) -> np.ndarray:
            """
            Compute the rPIE object update for a single scan position.

            O_patch += conj(P) * dpsi / ((1-alpha)*|P|^2 + alpha*|P|^2_max + eps)

            Args:
                O_patch: Object patch at current position, shape (probe_size, probe_size).
                P: Current probe estimate, shape (probe_size, probe_size).
                dpsi: Difference between corrected and estimated exit wave, shape (probe_size, probe_size).

            Returns:
                Updated object patch, shape (probe_size, probe_size).
            """
            P_abs2 = np.abs(P) ** 2
            P_max = np.max(P_abs2)
            denom = (1.0 - self.alpha) * P_abs2 + self.alpha * P_max + self.update_eps
            O_patch_updated = O_patch + np.conj(P) * dpsi / denom
            return O_patch_updated

    def _rpie_probe_update(
            self,
            P: np.ndarray,
            O_patch: np.ndarray,
            dpsi: np.ndarray,
        ) -> np.ndarray:
            """
            Compute the rPIE probe update.

            P += conj(O_patch) * dpsi / ((1-alpha)*|O_patch|^2 + alpha*|O_patch|^2_max + eps)

            Args:
                P: Current probe estimate, shape (probe_size, probe_size).
                O_patch: Object patch (before object update), shape (probe_size, probe_size).
                dpsi: Difference between corrected and estimated exit wave, shape (probe_size, probe_size).

            Returns:
                Updated probe, shape (probe_size, probe_size).
            """
            O_abs2 = np.abs(O_patch) ** 2
            O_max = np.max(O_abs2)
            denom = (1.0 - self.alpha) * O_abs2 + self.alpha * O_max + self.update_eps
            P_updated = P + np.conj(O_patch) * dpsi / denom
            return P_updated

    def _modulus_constraint(
            self, Psi: np.ndarray, measured_amplitude: np.ndarray
        ) -> np.ndarray:
            """
            Apply the Fourier modulus constraint: replace amplitude with measured,
            preserve phase.

            Psi_c = measured_amplitude * Psi / (|Psi| + eps)

            Args:
                Psi: Forward-propagated exit wave in Fourier space, shape (H, W).
                measured_amplitude: Measured amplitude for this pattern, shape (H, W).

            Returns:
                Corrected Fourier-space wave, shape (H, W).
            """
            mag = np.abs(Psi)
            Psi_c = measured_amplitude * Psi / (mag + self.modulus_eps)
            return Psi_c

    def _compute_fourier_error(
            self, Psi: np.ndarray, measured_amplitude: np.ndarray
        ) -> float:
            """
            Compute the squared Fourier error between predicted and measured amplitudes.

            Args:
                Psi: Forward-propagated exit wave, shape (H, W).
                measured_amplitude: Measured amplitude, shape (H, W).

            Returns:
                Sum of squared amplitude differences.
            """
            mag = np.abs(Psi)
            return float(np.sum((mag - measured_amplitude) ** 2))

    def _remove_global_phase(self, O: np.ndarray) -> np.ndarray:
            """
            Remove the global phase ambiguity from the reconstruction by setting
            the mean phase in high-reflectivity regions to zero.

            Args:
                O: Reconstructed complex object, shape (obj_size, obj_size).

            Returns:
                Phase-corrected complex object, shape (obj_size, obj_size).
            """
            mag = np.abs(O)
            threshold = np.median(mag)
            mask = mag > threshold

            if np.sum(mask) == 0:
                # Fallback: use all pixels if no pixels exceed threshold
                mean_complex = np.mean(O)
            else:
                mean_complex = np.mean(O[mask])

            global_phase = np.angle(mean_complex)
            O_corrected = O * np.exp(-1j * global_phase)
            return O_corrected

    def _compute_forward_consistency(
            self,
            O: np.ndarray,
            P: np.ndarray,
            amplitudes: np.ndarray,
        ) -> float:
            """
            Compute the overall forward consistency metric between the reconstruction
            and the measured data.

            Args:
                O: Reconstructed complex object, shape (obj_size, obj_size).
                P: Reconstructed probe, shape (probe_size, probe_size).
                amplitudes: Measured amplitudes, shape (N, H, W).

            Returns:
                Relative forward consistency error (total_err / total_energy).
            """
            total_err = 0.0
            total_energy = 0.0
            for j in range(self.n_positions):
                ry, rx = self.positions[j]
                O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
                psi = P * O_patch
                Psi = self.forward_prop(psi)
                pred_amp = np.abs(Psi)
                total_err += np.sum((pred_amp - amplitudes[j]) ** 2)
                total_energy += np.sum(amplitudes[j] ** 2)
            return total_err / (total_energy + 1e-30)

    def _format_output_real_imag(self, O: np.ndarray) -> np.ndarray:
            """
            Format the complex object as (2, H, W) with real and imaginary channels.

            Args:
                O: Complex object, shape (obj_size, obj_size).

            Returns:
                Output array, shape (2, obj_size, obj_size), dtype float64.
            """
            out = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            out[0] = np.real(O).astype(np.float64)
            out[1] = np.imag(O).astype(np.float64)
            return out

    def _format_output_mag_phase(self, O: np.ndarray) -> np.ndarray:
            """
            Format the complex object as (2, H, W) with magnitude and phase channels.

            Args:
                O: Complex object, shape (obj_size, obj_size).

            Returns:
                Output array, shape (2, obj_size, obj_size), dtype float64.
            """
            out = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            out[0] = np.abs(O).astype(np.float64)
            out[1] = np.angle(O).astype(np.float64)
            return out

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Compute the forward model for a single scan position:
            I(k) = |F{P(r - R) * O(r)}|^2

            Args:
                O: Complex object, shape (obj_size, obj_size).
                P: Complex probe, shape (probe_size, probe_size).
                position: (row, col) top-left corner of the probe on the object grid.

            Returns:
                Predicted diffraction intensity, shape (probe_size, probe_size).
            """
            ry, rx = position
            # Extract the object patch at the probe position
            O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
            # Form the exit wave: element-wise product of probe and object patch
            psi = P * O_patch
            # Forward propagate to Fourier space
            Psi = self.forward_prop(psi)
            # Compute diffraction intensity (squared modulus)
            intensity = np.abs(Psi) ** 2
            return intensity

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Main reconstruction pipeline: takes measured diffraction intensities
            and returns the reconstructed complex object.

            Steps:
                1. Compute amplitudes from intensities.
                2. Detect FFT convention (DC at center vs corner).
                3. Compute scan positions.
                4. Select best probe via sigma grid search.
                5. Run rPIE reconstruction with best-state tracking.
                6. Handle FFT convention fallback if non-convergence detected.
                7. Restore best state.
                8. Remove global phase ambiguity.
                9. Check forward consistency; fallback to pure ePIE if needed.
                10. Format output as (2, obj_size, obj_size).

            Args:
                input_data: Measured diffraction intensities, shape (81, 256, 256), float64, non-negative.

            Returns:
                Reconstructed object, shape (2, obj_size, obj_size), float64.
            """
            # Step 1: Compute amplitudes from intensities
            data = input_data.astype(np.float64)
            amplitudes = np.sqrt(np.maximum(data, 0.0))

            # Step 2: Detect FFT convention
            self.use_shift = self._detect_fft_convention(data)

            # Step 3: Compute scan positions
            self.positions = self._compute_scan_positions()

            # Step 4: Select best probe via sigma grid search
            best_probe = self._select_best_probe(amplitudes)

            # Step 5: Run rPIE reconstruction with best-state tracking
            O_init = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            P_init = best_probe.copy()

            best_O, best_P, errors = self._run_rpie_main_loop(O_init, P_init, amplitudes)

            # Step 6: Check if FFT convention switch is needed
            # (This is handled inside _run_rpie_main_loop)

            # Step 7: Best state already restored by _run_rpie_main_loop

            # Step 8: Check forward consistency; fallback to pure ePIE if needed
            consistency = self._compute_forward_consistency(best_O, best_P, amplitudes)

            if consistency > self.poor_convergence_threshold:
                # Try the other FFT convention
                self.use_shift = not self.use_shift
                best_probe_alt = self._select_best_probe(amplitudes)
                O_init_alt = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
                best_O_alt, best_P_alt, errors_alt = self._run_rpie_main_loop(
                    O_init_alt, best_probe_alt.copy(), amplitudes
                )
                consistency_alt = self._compute_forward_consistency(best_O_alt, best_P_alt, amplitudes)

                if consistency_alt < consistency:
                    best_O = best_O_alt
                    best_P = best_P_alt
                    consistency = consistency_alt
                else:
                    # Revert FFT convention
                    self.use_shift = not self.use_shift

                # If still poor, try fallback ePIE
                if consistency > self.poor_convergence_threshold:
                    fallback_O, fallback_P = self._run_fallback_epie(amplitudes)
                    fallback_consistency = self._compute_forward_consistency(
                        fallback_O, fallback_P, amplitudes
                    )
                    if fallback_consistency < consistency:
                        best_O = fallback_O
                        best_P = fallback_P

            # Step 9: Remove global phase ambiguity
            best_O = self._remove_global_phase(best_O)

            # Step 10: Format output
            # Use real/imaginary format as default
            output = self._format_output_real_imag(best_O)

            return output

    def _run_rpie_main_loop(
            self,
            O: np.ndarray,
            P: np.ndarray,
            amplitudes: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
            """
            Execute the main rPIE iterative loop with best-state tracking,
            early stopping, and optional FFT convention switching.

            Args:
                O: Initial complex object estimate, shape (obj_size, obj_size).
                P: Initial complex probe estimate, shape (probe_size, probe_size).
                amplitudes: Measured amplitudes, shape (N, H, W).

            Returns:
                Tuple of (best_object, best_probe, error_history).
            """
            # Initialize best-state tracking
            best_O = O.copy()
            best_P = P.copy()
            best_fourier_error = np.inf
            best_epoch = 0
            errors: List[float] = []

            # Precompute total amplitude energy for relative error
            total_amp_energy = float(np.sum(amplitudes ** 2))

            # Track initial error for FFT convention switch check
            initial_error: Optional[float] = None
            fft_switched = False

            for epoch in range(self.n_iterations):
                # Random scan order per epoch
                order = np.random.permutation(self.n_positions)
                epoch_err = 0.0

                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]

                    # Extract object patch
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size].copy()

                    # Exit wave
                    psi = P * O_patch

                    # Forward propagate
                    Psi = self.forward_prop(psi)

                    # Compute Fourier error for this pattern
                    epoch_err += self._compute_fourier_error(Psi, amplitudes[j])

                    # Apply modulus constraint
                    Psi_c = self._modulus_constraint(Psi, amplitudes[j])

                    # Back-propagate corrected exit wave
                    psi_c = self.backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # rPIE Object update
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] = self._rpie_object_update(
                        O_patch, P, dpsi
                    )

                    # rPIE Probe update (delayed start)
                    if epoch >= self.probe_update_start:
                        P = self._rpie_probe_update(P, O_patch, dpsi)

                # Compute relative Fourier error
                rel_err = epoch_err / (total_amp_energy + 1e-30)
                errors.append(rel_err)

                # Store initial error for convergence check
                if epoch == 0:
                    initial_error = rel_err

                # Best-state tracking (CRITICAL)
                if rel_err < best_fourier_error:
                    best_fourier_error = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                # FFT convention switch check
                if (
                    not fft_switched
                    and epoch == self.fft_convention_switch_epoch
                    and initial_error is not None
                    and rel_err > initial_error * self.fft_convergence_threshold
                ):
                    # Non-convergence detected — switch FFT convention
                    self.use_shift = not self.use_shift
                    fft_switched = True

                    # Reset reconstruction
                    O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
                    # Re-select probe with new FFT convention
                    P = self._select_best_probe(amplitudes)

                    # Reset tracking
                    best_fourier_error = np.inf
                    best_epoch = epoch + 1
                    errors = []
                    initial_error = None
                    continue

                # Early stopping: if no improvement for patience epochs, stop
                if epoch - best_epoch > self.early_stop_patience:
                    break

            # Restore best state
            return best_O, best_P, errors

    def _run_fallback_epie(
            self,
            amplitudes: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Fallback pure ePIE reconstruction (alpha=1.0) if rPIE fails to converge.

            Args:
                amplitudes: Measured amplitudes, shape (N, H, W).

            Returns:
                Tuple of (reconstructed_object, reconstructed_probe).
            """
            target_energy = self._compute_parseval_probe_energy(amplitudes)

            # Initialize with middle sigma candidate
            mid_sigma = self.probe_sigma_candidates[len(self.probe_sigma_candidates) // 2]
            P = self._init_gaussian_probe(mid_sigma, target_energy)
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

            # Best-state tracking
            best_O = O.copy()
            best_P = P.copy()
            best_fourier_error = np.inf
            best_epoch = 0

            total_amp_energy = float(np.sum(amplitudes ** 2))

            # Use fewer iterations for fallback but still enough to converge
            n_fallback_iters = 500
            fallback_probe_update_start = 15

            for epoch in range(n_fallback_iters):
                order = np.random.permutation(self.n_positions)
                epoch_err = 0.0

                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]

                    # Extract object patch
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size].copy()

                    # Exit wave
                    psi = P * O_patch

                    # Forward propagate
                    Psi = self.forward_prop(psi)

                    # Compute Fourier error
                    epoch_err += self._compute_fourier_error(Psi, amplitudes[j])

                    # Apply modulus constraint
                    Psi_c = self._modulus_constraint(Psi, amplitudes[j])

                    # Back-propagate corrected exit wave
                    psi_c = self.backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # ePIE Object update: O += conj(P) * dpsi / max(|P|^2)
                    P_abs2 = np.abs(P) ** 2
                    P_max = np.max(P_abs2)
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] = (
                        O_patch + np.conj(P) * dpsi / (P_max + self.update_eps)
                    )

                    # ePIE Probe update (delayed start)
                    if epoch >= fallback_probe_update_start:
                        O_abs2 = np.abs(O_patch) ** 2
                        O_max = np.max(O_abs2)
                        P = P + np.conj(O_patch) * dpsi / (O_max + self.update_eps)

                # Compute relative Fourier error
                rel_err = epoch_err / (total_amp_energy + 1e-30)

                # Best-state tracking
                if rel_err < best_fourier_error:
                    best_fourier_error = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                # Early stopping
                if epoch - best_epoch > self.early_stop_patience:
                    break

            return best_O, best_P



if __name__ == "__main__":
    # Load measured diffraction intensity data
    input_data = np.load("dataset/input.npy")  # Expected shape: (81, 256, 256)

    # Instantiate solver with default hyperparameters
    solver = InverseSolver(
        obj_size=600,
        probe_size=256,
        n_scan=9,
        alpha=0.3,
        n_iterations=800,
        probe_update_start=20,
        probe_sigma_candidates=[20, 30, 40, 55],
        probe_sigma_search_iters=10,
        early_stop_patience=200,
        fft_convention_switch_epoch=20,
        fft_convergence_threshold=0.95,
        poor_convergence_threshold=0.5,
    )

    # Run reconstruction
    result = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)