import numpy as np
from typing import Tuple, List, Optional
from scipy import ndimage

class InverseSolver:
    """
    Coherent Diffractive Imaging (CDI) phase retrieval solver using rPIE
    (regularized Ptychographic Iterative Engine) with reflective prior.

    Reconstructs a complex-valued 2D object reflection function from
    ptychographic far-field diffraction intensity patterns. Outputs
    magnitude (channel 0) and phase (channel 1).
    """

    def __init__(
            self,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            probe_size: int = 256,
            obj_size: int = 600,
            n_scan: int = 9,
            n_iter: int = 400,
            alpha: float = 0.3,
            probe_update_start: int = 15,
            early_stop_patience: int = 150,
            probe_sigma_candidates: List[float] = None,
            probe_quick_test_iters: int = 10,
            fft_fallback_epoch: int = 30,
            fft_fallback_threshold: float = 0.8,
            eps: float = 1e-12,
        ) -> None:
            """
            Initialize the inverse solver with all hyperparameters.

            Args:
                wavelength: EUV wavelength in meters (13.5 nm).
                pixel_size: Detector/object pixel size in meters (27 nm).
                probe_size: Size of the illumination probe in pixels (256).
                obj_size: Size of the object grid in pixels (600).
                n_scan: Number of scan positions along each axis (9 for 9x9=81).
                n_iter: Maximum number of rPIE iterations.
                alpha: rPIE regularization parameter (0=ePIE, 1=PIE-like).
                probe_update_start: Epoch at which to begin updating the probe.
                early_stop_patience: Stop if no improvement for this many epochs.
                probe_sigma_candidates: List of Gaussian sigma values to test for probe init.
                probe_quick_test_iters: Number of quick iterations for probe sigma selection.
                fft_fallback_epoch: Epoch at which to check FFT convention and possibly switch.
                fft_fallback_threshold: Relative error reduction threshold for FFT fallback.
                eps: Small constant to prevent division by zero.
            """
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.probe_size: int = probe_size
            self.obj_size: int = obj_size
            self.n_scan: int = n_scan
            self.n_iter: int = n_iter
            self.alpha: float = alpha
            self.probe_update_start: int = probe_update_start
            self.early_stop_patience: int = early_stop_patience
            self.probe_sigma_candidates: List[float] = probe_sigma_candidates if probe_sigma_candidates is not None else [20, 30, 40, 55]
            self.probe_quick_test_iters: int = probe_quick_test_iters
            self.fft_fallback_epoch: int = fft_fallback_epoch
            self.fft_fallback_threshold: float = fft_fallback_threshold
            self.eps: float = eps

            # State variables set during solve
            self.use_shift: bool = False
            self.positions: List[Tuple[int, int]] = []
            self.probe: Optional[np.ndarray] = None
            self.object_estimate: Optional[np.ndarray] = None
            self.best_object: Optional[np.ndarray] = None
            self.best_probe: Optional[np.ndarray] = None
            self.best_fourier_error: float = np.inf
            self.best_epoch: int = 0
            self.initial_error: float = np.inf

    def _detect_fft_convention(self, data: np.ndarray) -> bool:
            """
            Detect whether the DC component is centered or at the corner.

            Args:
                data: Input diffraction intensity patterns, shape (N, H, W).

            Returns:
                True if DC is at center (use fftshift), False if at corner.
            """
            avg_pattern = np.mean(data, axis=0)
            h, w = avg_pattern.shape
            center_r, center_c = h // 2, w // 2

            # Compare a small region around the center vs corners
            r = 5  # radius of region to average
            center_val = np.mean(avg_pattern[center_r - r:center_r + r, center_c - r:center_c + r])

            # Average of all four corners
            corner_val = np.mean([
                np.mean(avg_pattern[:2 * r, :2 * r]),
                np.mean(avg_pattern[:2 * r, -2 * r:]),
                np.mean(avg_pattern[-2 * r:, :2 * r]),
                np.mean(avg_pattern[-2 * r:, -2 * r:]),
            ])

            if center_val > corner_val * 5:
                return True  # DC at center
            else:
                return False  # DC at corner

    def forward_prop(self, x: np.ndarray) -> np.ndarray:
            """
            Forward propagation (real space -> Fourier space) with correct shift convention.

            Args:
                x: Complex-valued 2D array in real space, shape (H, W).

            Returns:
                Complex-valued 2D array in Fourier space, shape (H, W).
            """
            if self.use_shift:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            else:
                return np.fft.fft2(x)

    def backward_prop(self, X: np.ndarray) -> np.ndarray:
            """
            Backward propagation (Fourier space -> real space) with correct shift convention.

            Args:
                X: Complex-valued 2D array in Fourier space, shape (H, W).

            Returns:
                Complex-valued 2D array in real space, shape (H, W).
            """
            if self.use_shift:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))
            else:
                return np.fft.ifft2(X)

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
            """
            Compute the (row, col) scan positions for the ptychographic grid.

            Returns:
                List of (ry, rx) tuples for each scan position.
            """
            scan_step = (self.obj_size - self.probe_size) / (self.n_scan - 1)  # = 43.0

            positions = []
            for iy in range(self.n_scan):
                for ix in range(self.n_scan):
                    ry = int(round(iy * scan_step))
                    rx = int(round(ix * scan_step))
                    positions.append((ry, rx))
            return positions

    def _initialize_probe(
            self,
            amplitudes: np.ndarray,
            positions: List[Tuple[int, int]],
        ) -> np.ndarray:
            """
            Initialize the illumination probe by testing multiple Gaussian sigma values
            and selecting the one with lowest quick-test error. Applies Parseval scaling.

            Args:
                amplitudes: Square root of measured intensities, shape (N, H, W).
                positions: List of scan positions.

            Returns:
                Best initial probe estimate, shape (probe_size, probe_size), complex128.
            """
            N_pix = self.probe_size * self.probe_size
            mean_total_intensity = np.mean(np.sum(amplitudes ** 2, axis=(1, 2)))

            half = self.probe_size // 2
            Y, X = np.mgrid[-half:half, -half:half]
            R2 = (X ** 2 + Y ** 2).astype(np.float64)

            best_probe = None
            best_err = np.inf

            for sigma in self.probe_sigma_candidates:
                P_test = np.exp(-R2 / (2.0 * sigma ** 2)).astype(np.complex128)

                # Parseval scaling: sum(|FFT(P*O)|^2) = N_pix * sum(|P*O|^2)
                # If |O| ~ 1: sum(I) ~ N_pix * sum(|P|^2)
                # So: sum(|P|^2) ~ mean_total_intensity / N_pix
                current_energy = np.sum(np.abs(P_test) ** 2)
                target_energy = mean_total_intensity / N_pix
                if current_energy > 0:
                    scale = np.sqrt(target_energy / current_energy)
                    P_test *= scale

                # Quick ePIE test
                O_test = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
                err = self._quick_epie(O_test, P_test, positions, amplitudes, self.probe_quick_test_iters)

                if err < best_err:
                    best_err = err
                    best_probe = P_test.copy()

            return best_probe

    def _quick_epie(
            self,
            O_init: np.ndarray,
            P_init: np.ndarray,
            positions: List[Tuple[int, int]],
            amplitudes: np.ndarray,
            n_iters: int,
        ) -> float:
            """
            Run a quick ePIE reconstruction for probe selection testing.

            Args:
                O_init: Initial object estimate, shape (obj_size, obj_size), complex128.
                P_init: Initial probe estimate, shape (probe_size, probe_size), complex128.
                positions: List of scan positions.
                amplitudes: Measured amplitudes, shape (N, H, W).
                n_iters: Number of iterations to run.

            Returns:
                Final relative Fourier error (scalar).
            """
            O = O_init.copy()
            P = P_init.copy()
            n_pos = len(positions)
            total_amp_energy = np.sum(amplitudes ** 2)

            final_err = np.inf

            for iteration in range(n_iters):
                order = np.random.permutation(n_pos)
                epoch_err = 0.0

                for idx in range(n_pos):
                    j = order[idx]
                    ry, rx = positions[j]
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]

                    # Exit wave
                    psi = P * O_patch

                    # Forward propagate
                    Psi = self.forward_prop(psi)

                    # Fourier modulus replacement
                    mag = np.abs(Psi)
                    Psi_corrected = amplitudes[j] * Psi / (mag + self.eps)

                    # Accumulate error
                    epoch_err += np.sum((mag - amplitudes[j]) ** 2)

                    # Backward propagate
                    psi_corrected = self.backward_prop(Psi_corrected)
                    dpsi = psi_corrected - psi

                    # Simple ePIE object update (alpha=1 equivalent, no probe update)
                    P_abs2 = np.abs(P) ** 2
                    P_max = np.max(P_abs2)
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] += (
                        np.conj(P) * dpsi / (P_max + self.eps)
                    )

                final_err = epoch_err / (total_amp_energy + self.eps)

            return float(final_err)

    def _rpie_update_object(
            self,
            O_patch: np.ndarray,
            P: np.ndarray,
            dpsi: np.ndarray,
            alpha: float,
        ) -> np.ndarray:
            """
            Compute the rPIE object update for a single scan position.

            Args:
                O_patch: Current object patch, shape (probe_size, probe_size), complex128.
                P: Current probe, shape (probe_size, probe_size), complex128.
                dpsi: Difference between corrected and current exit wave, same shape.
                alpha: rPIE regularization parameter.

            Returns:
                Updated object patch, same shape as O_patch.
            """
            P_abs2 = np.abs(P) ** 2
            P_max = np.max(P_abs2)
            denom = (1.0 - alpha) * P_abs2 + alpha * P_max
            updated_patch = O_patch + np.conj(P) * dpsi / (denom + self.eps)
            return updated_patch

    def _rpie_update_probe(
            self,
            O_patch: np.ndarray,
            P: np.ndarray,
            dpsi: np.ndarray,
            alpha: float,
        ) -> np.ndarray:
            """
            Compute the rPIE probe update for a single scan position.

            Args:
                O_patch: Current object patch, shape (probe_size, probe_size), complex128.
                P: Current probe, shape (probe_size, probe_size), complex128.
                dpsi: Difference between corrected and current exit wave, same shape.
                alpha: rPIE regularization parameter.

            Returns:
                Updated probe, same shape as P.
            """
            O_abs2 = np.abs(O_patch) ** 2
            O_max = np.max(O_abs2)
            denom = (1.0 - alpha) * O_abs2 + alpha * O_max
            updated_probe = P + np.conj(O_patch) * dpsi / (denom + self.eps)
            return updated_probe

    def _fourier_modulus_replacement(
            self,
            psi: np.ndarray,
            measured_amplitude: np.ndarray,
        ) -> np.ndarray:
            """
            Apply Fourier modulus constraint: replace magnitude with measured amplitude,
            preserve phase.

            Args:
                psi: Exit wave in real space, shape (H, W), complex128.
                measured_amplitude: Measured amplitude for this position, shape (H, W).

            Returns:
                Corrected exit wave in real space, shape (H, W), complex128.
            """
            # Forward propagate to Fourier space
            Psi = self.forward_prop(psi)

            # Replace magnitude with measured amplitude, preserve phase
            mag = np.abs(Psi)
            Psi_corrected = measured_amplitude * Psi / (mag + self.eps)

            # Backward propagate to real space
            psi_corrected = self.backward_prop(Psi_corrected)

            return psi_corrected

    def _compute_fourier_error(
            self,
            O: np.ndarray,
            P: np.ndarray,
            positions: List[Tuple[int, int]],
            amplitudes: np.ndarray,
        ) -> float:
            """
            Compute the relative Fourier-domain error for the current reconstruction.

            Args:
                O: Current object estimate, shape (obj_size, obj_size), complex128.
                P: Current probe estimate, shape (probe_size, probe_size), complex128.
                positions: List of scan positions.
                amplitudes: Measured amplitudes, shape (N, H, W).

            Returns:
                Relative Fourier error (scalar).
            """
            total_err = 0.0
            total_energy = 0.0
            for j in range(len(positions)):
                ry, rx = positions[j]
                O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
                psi = P * O_patch
                Psi = self.forward_prop(psi)
                pred_amp = np.abs(Psi)
                total_err += np.sum((pred_amp - amplitudes[j]) ** 2)
                total_energy += np.sum(amplitudes[j] ** 2)
            return float(total_err / (total_energy + self.eps))

    def _compute_forward_consistency(
            self,
            O: np.ndarray,
            P: np.ndarray,
            positions: List[Tuple[int, int]],
            amplitudes: np.ndarray,
        ) -> float:
            """
            Compute forward consistency metric between reconstruction and measured data.

            Args:
                O: Reconstructed object, shape (obj_size, obj_size), complex128.
                P: Reconstructed probe, shape (probe_size, probe_size), complex128.
                positions: List of scan positions.
                amplitudes: Measured amplitudes, shape (N, H, W).

            Returns:
                Relative forward consistency error (scalar).
            """
            total_err = 0.0
            total_energy = 0.0
            for j in range(len(positions)):
                ry, rx = positions[j]
                O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
                psi = P * O_patch
                Psi = self.forward_prop(psi)
                pred_amp = np.abs(Psi)
                total_err += np.sum((pred_amp - amplitudes[j]) ** 2)
                total_energy += np.sum(amplitudes[j] ** 2)
            return float(total_err / (total_energy + self.eps))

    def _resolve_phase_ambiguity(self, O: np.ndarray) -> np.ndarray:
            """
            Remove the global phase ambiguity by referencing phase to the substrate
            (lower-magnitude) region.

            Args:
                O: Complex object estimate, shape (obj_size, obj_size).

            Returns:
                Phase-corrected complex object, same shape.
            """
            mag = np.abs(O)
            phase = np.angle(O)

            # Identify substrate vs pad regions using magnitude
            # Use a simple threshold at the midpoint between min and max of magnitude
            mag_min = np.percentile(mag, 5)
            mag_max = np.percentile(mag, 95)
            threshold = 0.5 * (mag_min + mag_max)

            # Substrate is typically the lower-magnitude region
            substrate_mask = mag < threshold

            if np.sum(substrate_mask) > 100:
                # Use median phase of substrate as reference (robust to outliers)
                phase_ref = np.median(phase[substrate_mask])
            else:
                # Fallback: use median of entire object
                phase_ref = np.median(phase)

            # Apply global phase correction by multiplying by exp(-j * phase_ref)
            O_corrected = O * np.exp(-1j * phase_ref)

            # After correction, check if pad regions have positive phase (physical expectation:
            # recession h > 0 gives positive phase = 4*pi*h/lambda)
            corrected_phase = np.angle(O_corrected)
            pad_mask = ~substrate_mask

            if np.sum(pad_mask) > 100:
                pad_mean_phase = np.mean(corrected_phase[pad_mask])
                # If pad phase is predominantly negative, flip by pi
                # (this handles the sign ambiguity)
                if pad_mean_phase < -0.3:
                    # Try adding pi to see if that puts pad phase in expected range [0.93, 2.79]
                    O_flipped = O_corrected * np.exp(1j * np.pi)
                    flipped_phase = np.angle(O_flipped)
                    flipped_pad_mean = np.mean(flipped_phase[pad_mask])
                    # Check which is more physically plausible
                    # Expected pad phase ~ 1.86 rad (midpoint of [0.93, 2.79])
                    if abs(flipped_pad_mean - 1.86) < abs(pad_mean_phase - 1.86):
                        O_corrected = O_flipped

            return O_corrected

    def _format_output_magnitude_phase(self, O: np.ndarray) -> np.ndarray:
            """
            Convert complex object to (magnitude, phase) output format.
            Magnitude is normalized to [0, 1]. Phase is in radians, substrate-referenced.

            Args:
                O: Complex object estimate, shape (obj_size, obj_size).

            Returns:
                Output array of shape (2, obj_size, obj_size), float64.
                Channel 0: normalized magnitude [0, 1].
                Channel 1: phase in radians [-pi, pi].
            """
            # Resolve global phase ambiguity first
            O_corrected = self._resolve_phase_ambiguity(O)

            # Compute magnitude
            mag = np.abs(O_corrected).astype(np.float64)

            # Normalize magnitude to [0, 1]
            mag_min = mag.min()
            mag_max = mag.max()
            mag_range = mag_max - mag_min
            if mag_range > self.eps:
                mag_normalized = (mag - mag_min) / mag_range
            else:
                mag_normalized = np.zeros_like(mag)

            # Compute phase in radians [-pi, pi]
            phase = np.angle(O_corrected).astype(np.float64)

            # Ensure phase is wrapped to [-pi, pi] (np.angle already does this,
            # but be explicit for safety)
            phase = np.arctan2(np.sin(phase), np.cos(phase))

            # Assemble output array
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            output[0] = mag_normalized
            output[1] = phase

            return output

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            positions: List[Tuple[int, int]],
        ) -> np.ndarray:
            """
            Simulate the forward model: generate diffraction intensity patterns
            from object and probe.

            I_j(k) = |F{P(r - R_j) * O(r)}|^2

            Args:
                O: Complex object, shape (obj_size, obj_size).
                P: Complex probe, shape (probe_size, probe_size).
                positions: List of (ry, rx) scan positions.

            Returns:
                Simulated intensity patterns, shape (N, probe_size, probe_size), float64.
            """
            n_pos = len(positions)
            intensities = np.zeros((n_pos, self.probe_size, self.probe_size), dtype=np.float64)

            for j in range(n_pos):
                ry, rx = positions[j]
                # Extract the object patch at this scan position
                O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
                # Form the exit wave: probe * object patch
                psi = P * O_patch
                # Forward propagate to Fourier space
                Psi = self.forward_prop(psi)
                # Compute intensity (squared magnitude)
                intensities[j] = np.abs(Psi) ** 2

            return intensities

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Main solver: reconstruct the complex object from ptychographic diffraction data
            using rPIE with best-state tracking and output as magnitude/phase.

            Args:
                input_data: Measured diffraction intensity patterns,
                            shape (81, 256, 256), float64, non-negative.

            Returns:
                Reconstruction result, shape (2, obj_size, obj_size), float64.
                Channel 0: normalized magnitude in [0, 1].
                Channel 1: phase in radians, substrate-referenced.
            """
            # Step 1: Compute amplitudes from intensity data
            amplitudes = np.sqrt(np.maximum(input_data, 0.0))

            # Step 2: Detect FFT convention
            self.use_shift = self._detect_fft_convention(input_data)

            # Step 3: Compute scan positions
            self.positions = self._compute_scan_positions()

            # Step 4: Initialize probe with sigma search
            self.probe = self._initialize_probe(amplitudes, self.positions)

            # Step 5: Initialize object as ones
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            P = self.probe.copy()

            # Best state tracking
            best_O = O.copy()
            best_P = P.copy()
            best_fourier_error = np.inf
            best_epoch = 0
            initial_error = None

            n_pos = len(self.positions)
            total_amp_energy = np.sum(amplitudes ** 2)

            # Store initial use_shift for fallback
            initial_use_shift = self.use_shift

            # Step 6: Main rPIE loop
            for epoch in range(self.n_iter):
                order = np.random.permutation(n_pos)
                epoch_err = 0.0

                for idx in range(n_pos):
                    j = order[idx]
                    ry, rx = self.positions[j]
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]

                    # Exit wave
                    psi = P * O_patch

                    # Forward propagate
                    Psi = self.forward_prop(psi)

                    # Fourier modulus replacement
                    mag = np.abs(Psi)
                    Psi_corrected = amplitudes[j] * Psi / (mag + self.eps)
                    epoch_err += np.sum((mag - amplitudes[j]) ** 2)

                    # Backward propagate
                    psi_corrected = self.backward_prop(Psi_corrected)
                    dpsi = psi_corrected - psi

                    # rPIE object update
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] = \
                        self._rpie_update_object(O_patch, P, dpsi, self.alpha)

                    # rPIE probe update (after warmup)
                    if epoch >= self.probe_update_start:
                        # Re-read the updated object patch for probe update
                        O_patch_updated = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
                        P = self._rpie_update_probe(O_patch_updated, P, dpsi, self.alpha)

                rel_err = epoch_err / (total_amp_energy + self.eps)

                # Store initial error for fallback check
                if epoch == 0:
                    initial_error = rel_err

                # FFT convention fallback check
                if epoch == self.fft_fallback_epoch and initial_error is not None:
                    if rel_err > initial_error * self.fft_fallback_threshold:
                        # Error hasn't decreased enough, try switching FFT convention
                        self.use_shift = not self.use_shift

                        # Reset everything
                        O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
                        P = self.probe.copy()
                        best_O = O.copy()
                        best_P = P.copy()
                        best_fourier_error = np.inf
                        best_epoch = epoch
                        initial_error = None
                        continue

                # Best state tracking
                if rel_err < best_fourier_error:
                    best_fourier_error = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                # Early stopping
                if epoch - best_epoch > self.early_stop_patience:
                    break

            # Step 7: Restore best state
            O = best_O
            P = best_P

            # Compute forward consistency for diagnostics
            consistency = self._compute_forward_consistency(O, P, self.positions, amplitudes)

            # Step 8: Format output as magnitude/phase
            output = self._format_output_magnitude_phase(O)

            return output



if __name__ == "__main__":
    # Load measured ptychographic diffraction intensity data
    input_data: np.ndarray = np.load("dataset/input.npy")  # (81, 256, 256)

    # Instantiate solver with default hyperparameters
    solver: InverseSolver = InverseSolver(
        wavelength=13.5e-9,
        pixel_size=27e-9,
        probe_size=256,
        obj_size=600,
        n_scan=9,
        n_iter=400,
        alpha=0.3,
        probe_update_start=15,
        early_stop_patience=150,
        probe_sigma_candidates=[20, 30, 40, 55],
        probe_quick_test_iters=10,
        fft_fallback_epoch=30,
        fft_fallback_threshold=0.8,
        eps=1e-12,
    )

    # Run phase retrieval reconstruction
    result: np.ndarray = solver.solve(input_data)  # (2, 600, 600)

    # Save output: channel 0 = magnitude, channel 1 = phase
    np.save("output.npy", result)