import numpy as np
import time
from typing import Tuple, Optional, List

class InverseSolver:
    """
    Standard sequential ePIE (extended Ptychographic Iterative Engine) solver
    for coherent diffractive imaging of periodic Cu pad arrays.

    Reconstructs complex-valued reflection function O(r) from far-field
    diffraction intensity measurements I(k) collected at multiple scan positions.
    """

    def __init__(
            self,
            probe_size: int = 256,
            obj_size: int = 600,
            n_side: int = 9,
            step: int = 40,
            alpha_O: float = 1.0,
            alpha_P: float = 1.0,
            reg_fraction: float = 1e-3,
            eps: float = 1e-12,
            probe_window_sigma: float = 50.0,
            time_limit: float = 53.0,
            phase_hist_bins: int = 720,
            phase_hist_smooth_kernel_size: int = 21,
            amp_valid_threshold: float = 0.1,
            residual_subset_size: int = 20,
            fallback_time_budget: float = 15.0,
        ) -> None:
            """
            Initialize the ePIE solver with all hyperparameters.

            Args:
                probe_size: Size of the probe/diffraction pattern in pixels (ps x ps).
                obj_size: Size of the object array in pixels (obj_size x obj_size).
                n_side: Number of scan positions along each axis (n_side x n_side grid).
                step: Step size in pixels between scan positions.
                alpha_O: Object update step size.
                alpha_P: Probe update step size.
                reg_fraction: Regularization added to denominator to prevent division by zero.
                eps: Small constant for numerical stability.
                probe_window_sigma: Gaussian window sigma for probe localization in real space.
                time_limit: Maximum wall-clock time in seconds for the solve routine.
                phase_hist_bins: Number of bins for phase histogram in post-processing.
                phase_hist_smooth_kernel_size: Kernel size for histogram smoothing.
                amp_valid_threshold: Fraction of max amplitude to define valid region.
                residual_subset_size: Number of positions to use for residual computation.
                fallback_time_budget: Minimum remaining time to attempt alternate FFT convention.
            """
            self.probe_size: int = probe_size
            self.obj_size: int = obj_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_pos: int = n_side * n_side
            self.alpha_O: float = alpha_O
            self.alpha_P: float = alpha_P
            self.reg_fraction: float = reg_fraction
            self.eps: float = eps
            self.probe_window_sigma: float = probe_window_sigma
            self.time_limit: float = time_limit
            self.phase_hist_bins: int = phase_hist_bins
            self.phase_hist_smooth_kernel_size: int = phase_hist_smooth_kernel_size
            self.amp_valid_threshold: float = amp_valid_threshold
            self.residual_subset_size: int = residual_subset_size
            self.fallback_time_budget: float = fallback_time_budget

    def _compute_scan_positions(self) -> np.ndarray:
            """
            Compute the (row, col) scan positions for the ptychographic grid.

            Returns:
                positions: Integer array of shape (n_pos, 2) with (row, col) offsets.
            """
            total_scan = self.probe_size + (self.n_side - 1) * self.step  # 256 + 320 = 576
            offset = (self.obj_size - total_scan) // 2  # (600 - 576) // 2 = 12

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions, dtype=np.int32)

    def _detect_fft_convention(self, intensities: np.ndarray) -> bool:
            """
            Detect whether DC component is at center or corner of diffraction patterns.

            Args:
                intensities: Raw measured diffraction intensities, shape (n_pos, ps, ps).

            Returns:
                dc_at_center: True if DC is at center, False if at corners.
            """
            ps = self.probe_size
            half_width = 40

            center_slice = slice(ps // 2 - half_width, ps // 2 + half_width)
            center_energy = np.mean(intensities[:, center_slice, center_slice])

            corner_energy = (
                np.mean(intensities[:, :half_width, :half_width])
                + np.mean(intensities[:, :half_width, -half_width:])
                + np.mean(intensities[:, -half_width:, :half_width])
                + np.mean(intensities[:, -half_width:, -half_width:])
            ) / 4.0

            return bool(center_energy > corner_energy)

    def _prepare_diffraction_amplitudes(
            self, intensities: np.ndarray, dc_at_center: bool
        ) -> np.ndarray:
            """
            Compute sqrt of intensities and shift to FFT-native layout if needed.

            Args:
                intensities: Non-negative diffraction intensities, shape (n_pos, ps, ps).
                dc_at_center: Whether DC is at center (requires ifftshift).

            Returns:
                diff_amp_fft: Diffraction amplitudes in FFT-native layout, shape (n_pos, ps, ps).
            """
            diff_amp = np.sqrt(intensities)
            if dc_at_center:
                diff_amp_fft = np.fft.ifftshift(diff_amp, axes=(-2, -1))
            else:
                diff_amp_fft = diff_amp.copy()
            return diff_amp_fft

    def _initialize_probe(self, diff_amp_fft: np.ndarray) -> np.ndarray:
            """
            Initialize probe from average measured diffraction amplitude via IFFT with
            Gaussian windowing in real space.

            Args:
                diff_amp_fft: Diffraction amplitudes in FFT-native layout, shape (n_pos, ps, ps).

            Returns:
                P: Initial complex probe array, shape (ps, ps), normalized to max|P|=1.
            """
            ps = self.probe_size

            # Average measured amplitude across all positions (FFT-native layout)
            avg_amp = np.mean(diff_amp_fft, axis=0)  # (ps, ps)

            # Create probe as ifft2 of average amplitude (zero phase in Fourier domain)
            P_raw = np.fft.ifft2(avg_amp)  # (ps, ps), complex

            # Shift to center for windowing in real space
            P_shifted = np.fft.fftshift(P_raw)

            # Apply localizing Gaussian window in real space
            window = self._create_localizing_window()  # (ps, ps)
            P_shifted = P_shifted * window

            # Shift back to FFT-native layout
            P = np.fft.ifftshift(P_shifted)

            # Normalize probe so max|P| = 1
            max_abs = np.max(np.abs(P))
            P = P / (max_abs + self.eps)

            return P

    def _initialize_object(self) -> np.ndarray:
            """
            Initialize the object as a uniform reflective surface (amplitude=1, phase=0).

            Returns:
                O: Initial complex object array, shape (obj_size, obj_size).
            """
            return np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

    def _create_localizing_window(self) -> np.ndarray:
            """
            Create a Gaussian window for probe localization in real space.

            Returns:
                window: Real-valued window array, shape (ps, ps).
            """
            ps = self.probe_size
            yy, xx = np.mgrid[:ps, :ps]
            cy, cx = ps / 2.0, ps / 2.0
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            window = np.exp(-r2 / (2.0 * self.probe_window_sigma ** 2))
            return window

    def forward(
            self, O: np.ndarray, P: np.ndarray, position: Tuple[int, int]
        ) -> np.ndarray:
            """
            Compute the forward model: far-field diffraction amplitude for a single
            scan position.

            ψ_j = P(r - R_j) · O(r)
            Ψ_j = FFT2(ψ_j)
            I_j = |Ψ_j|²

            Args:
                O: Complex object array, shape (obj_size, obj_size).
                P: Complex probe array, shape (ps, ps).
                position: (row, col) offset for this scan position.

            Returns:
                diff_amplitude: |FFT2(P · O_patch)|, shape (ps, ps).
            """
            ry, rx = position
            ps = self.probe_size
            O_patch = O[ry:ry + ps, rx:rx + ps]
            psi = P * O_patch
            Psi = np.fft.fft2(psi)
            return np.abs(Psi)

    def _fourier_magnitude_constraint(
            self, psi: np.ndarray, measured_amp: np.ndarray
        ) -> np.ndarray:
            """
            Apply Fourier magnitude constraint: replace amplitude with measured,
            keep current phase.

            Ψ' = sqrt(I) · Ψ / |Ψ|

            Args:
                psi: Exit wave in real space, shape (ps, ps).
                measured_amp: Measured diffraction amplitude (FFT-native layout), shape (ps, ps).

            Returns:
                psi_corrected: Corrected exit wave in real space, shape (ps, ps).
            """
            # Forward propagate to Fourier space
            Psi = np.fft.fft2(psi)

            # Compute current amplitude
            Psi_amp = np.abs(Psi) + self.eps

            # Replace amplitude with measured, keep phase
            Psi_corrected = measured_amp * Psi / Psi_amp

            # Inverse propagate back to real space
            psi_corrected = np.fft.ifft2(Psi_corrected)

            return psi_corrected

    def _update_object(
            self,
            O: np.ndarray,
            P: np.ndarray,
            dpsi: np.ndarray,
            position: Tuple[int, int],
        ) -> None:
            """
            Apply standard ePIE object update rule (in-place).

            O(r) += α_O · conj(P) / (|P_max|² + reg) · Δψ

            Args:
                O: Complex object array, shape (obj_size, obj_size). Modified in-place.
                P: Complex probe array, shape (ps, ps).
                dpsi: Exit wave difference (ψ_corrected - ψ), shape (ps, ps).
                position: (row, col) offset for this scan position.
            """
            ry, rx = position
            ps = self.probe_size
            P_max_sq = np.max(np.abs(P) ** 2)
            O[ry:ry + ps, rx:rx + ps] += self.alpha_O * np.conj(P) * dpsi / (P_max_sq + self.reg_fraction)

    def _update_probe(
            self,
            P: np.ndarray,
            O_patch: np.ndarray,
            dpsi: np.ndarray,
        ) -> None:
            """
            Apply standard ePIE probe update rule (in-place).

            P(r) += α_P · conj(O_patch) / (|O_patch_max|² + reg) · Δψ

            Args:
                P: Complex probe array, shape (ps, ps). Modified in-place.
                O_patch: Object patch at current position, shape (ps, ps).
                dpsi: Exit wave difference, shape (ps, ps).
            """
            O_patch_max_sq = np.max(np.abs(O_patch) ** 2)
            P += self.alpha_P * np.conj(O_patch) * dpsi / (O_patch_max_sq + self.reg_fraction)

    def _run_epie(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp_fft: np.ndarray,
            positions: np.ndarray,
            t_start: float,
            time_limit: float,
        ) -> Tuple[np.ndarray, np.ndarray, int]:
            """
            Run the standard sequential ePIE algorithm with randomized position ordering.

            Iterates epochs until time budget is exhausted. Each epoch visits all
            scan positions in random order, applying object and probe updates.

            Args:
                O: Initial complex object, shape (obj_size, obj_size).
                P: Initial complex probe, shape (ps, ps).
                diff_amp_fft: Measured diffraction amplitudes in FFT layout, shape (n_pos, ps, ps).
                positions: Scan positions, shape (n_pos, 2).
                t_start: Wall-clock start time.
                time_limit: Maximum allowed elapsed time in seconds.

            Returns:
                O: Reconstructed complex object, shape (obj_size, obj_size).
                P: Reconstructed complex probe, shape (ps, ps).
                n_epochs: Number of completed epochs.
            """
            n_pos = len(positions)
            ps = self.probe_size
            epoch = 0

            while True:
                elapsed = time.time() - t_start
                if elapsed > time_limit:
                    break

                # Randomize position ordering each epoch to prevent periodic artifacts
                order = np.random.permutation(n_pos)

                for idx in range(n_pos):
                    j = order[idx]
                    ry, rx = int(positions[j, 0]), int(positions[j, 1])

                    # Extract object patch (copy for probe update, reference for object update)
                    O_patch = O[ry:ry + ps, rx:rx + ps].copy()

                    # Compute exit wave
                    psi = P * O_patch  # (ps, ps)

                    # Apply Fourier magnitude constraint
                    psi_corrected = self._fourier_magnitude_constraint(psi, diff_amp_fft[j])

                    # Compute exit wave difference
                    dpsi = psi_corrected - psi

                    # Update object (in-place)
                    self._update_object(O, P, dpsi, (ry, rx))

                    # Update probe (in-place)
                    self._update_probe(P, O_patch, dpsi)

                epoch += 1

                # Check time after each epoch
                elapsed = time.time() - t_start
                if elapsed > time_limit:
                    break

            return O, P, epoch

    def _compute_residual(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
        ) -> float:
            """
            Compute the Fourier magnitude residual on a subset of positions.

            Args:
                O: Complex object, shape (obj_size, obj_size).
                P: Complex probe, shape (ps, ps).
                diff_amp: Diffraction amplitudes, shape (n_subset, ps, ps).
                positions: Scan positions for subset, shape (n_subset, 2).

            Returns:
                residual: Sum of squared amplitude differences.
            """
            ps = self.probe_size
            residual = 0.0
            for j in range(len(positions)):
                ry, rx = int(positions[j, 0]), int(positions[j, 1])
                O_patch = O[ry:ry + ps, rx:rx + ps]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                residual += np.sum((np.abs(Psi) - diff_amp[j]) ** 2)
            return residual

    def _try_alternate_convention(
            self,
            O_result: np.ndarray,
            P_result: np.ndarray,
            diff_amp: np.ndarray,
            diff_amp_fft: np.ndarray,
            positions: np.ndarray,
            dc_at_center: bool,
            window: np.ndarray,
            t_start: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Try the opposite FFT convention and keep the result with lower residual.

            Args:
                O_result: Current best object reconstruction.
                P_result: Current best probe reconstruction.
                diff_amp: Unshifted diffraction amplitudes, shape (n_pos, ps, ps).
                diff_amp_fft: Current convention amplitudes, shape (n_pos, ps, ps).
                positions: Scan positions, shape (n_pos, 2).
                dc_at_center: Current convention flag.
                window: Localizing window for probe init, shape (ps, ps).
                t_start: Wall-clock start time.

            Returns:
                O_best: Best object reconstruction.
                P_best: Best probe reconstruction.
            """
            time_left = self.time_limit + 2.0 - (time.time() - t_start)

            if time_left < self.fallback_time_budget:
                return O_result, P_result

            # Compute residual for current result on a subset of positions
            n_pos = len(positions)
            subset_size = min(self.residual_subset_size, n_pos)
            subset = np.random.choice(n_pos, subset_size, replace=False)
            sub_pos = positions[subset]
            sub_damp = diff_amp_fft[subset]
            res1 = self._compute_residual(O_result, P_result, sub_damp, sub_pos)

            # Prepare alternate convention diffraction amplitudes
            if dc_at_center:
                damp2 = diff_amp.copy()  # no shift (treat as corners)
            else:
                damp2 = np.fft.ifftshift(diff_amp, axes=(-2, -1))  # shift to corners

            # Initialize probe for alternate convention
            avg2 = np.mean(damp2, axis=0)  # (ps, ps)
            P2_raw = np.fft.ifft2(avg2)
            P2_shifted = np.fft.fftshift(P2_raw)
            P2_shifted = P2_shifted * window
            P2 = np.fft.ifftshift(P2_shifted)
            P2 = P2 / (np.max(np.abs(P2)) + self.eps)

            # Initialize object for alternate convention
            O2 = self._initialize_object()

            # Run ePIE with alternate convention
            alt_time_limit = self.time_limit + 2.0
            O2_res, P2_res, _ = self._run_epie(
                O2, P2, damp2, positions, t_start, alt_time_limit
            )

            # Compute residual for alternate result
            sub_damp2 = damp2[subset]
            res2 = self._compute_residual(O2_res, P2_res, sub_damp2, sub_pos)

            if res2 < res1:
                return O2_res, P2_res
            else:
                return O_result, P_result

    def _build_scan_coverage_mask(self, positions: np.ndarray) -> np.ndarray:
            """
            Build a boolean mask indicating which object pixels are covered by at
            least one scan position.

            Args:
                positions: Scan positions, shape (n_pos, 2).

            Returns:
                mask: Boolean array, shape (obj_size, obj_size).
            """
            mask = np.zeros((self.obj_size, self.obj_size), dtype=bool)
            ps = self.probe_size
            for j in range(len(positions)):
                ry, rx = int(positions[j, 0]), int(positions[j, 1])
                mask[ry:ry + ps, rx:rx + ps] = True
            return mask

    def _align_phase(self, O: np.ndarray, scan_mask: np.ndarray) -> np.ndarray:
            """
            Remove global phase offset by aligning substrate phase to zero using
            histogram-based mode estimation.

            Args:
                O: Complex object, shape (obj_size, obj_size).
                scan_mask: Boolean coverage mask, shape (obj_size, obj_size).

            Returns:
                O_aligned: Phase-aligned complex object.
            """
            amp = np.abs(O)
            phase = np.angle(O)

            # Identify valid reconstructed region: within scan coverage and above amplitude threshold
            max_amp_in_scan = np.max(amp[scan_mask]) if np.any(scan_mask) else 1.0
            valid = scan_mask & (amp > self.amp_valid_threshold * max_amp_in_scan)

            if np.sum(valid) > 100:
                # Find dominant phase via histogram (substrate = most common phase)
                phase_valid = phase[valid]
                n_bins = self.phase_hist_bins
                hist, bin_edges = np.histogram(phase_valid, bins=n_bins, range=(-np.pi, np.pi))

                # Smooth histogram to find robust mode
                kernel_size = self.phase_hist_smooth_kernel_size
                kernel = np.ones(kernel_size) / float(kernel_size)
                hist_smooth = np.convolve(hist.astype(np.float64), kernel, mode='same')

                # Find the bin with maximum count (mode of phase distribution)
                mode_idx = np.argmax(hist_smooth)
                phase_ref = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
            else:
                phase_ref = 0.0

            # Remove global phase offset so substrate phase ≈ 0
            O_aligned = O * np.exp(-1j * phase_ref)

            return O_aligned

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """
            Convert complex object to output format: channel 0 = amplitude, channel 1 = phase.

            Args:
                O: Complex object, shape (obj_size, obj_size).

            Returns:
                output: Float32 array, shape (2, obj_size, obj_size).
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)
            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Full reconstruction pipeline: preprocess, initialize, run ePIE,
            post-process, and format output.

            Args:
                input_data: Measured diffraction intensities, shape (81, 256, 256).

            Returns:
                result: Reconstructed object as float32 array, shape (2, 600, 600),
                        where result[0] = |O(r)| and result[1] = angle(O(r)).
            """
            # 1. Record start time
            t_start = time.time()

            # 2. Preprocess intensities (clip negatives, cast to float64)
            intensities = np.maximum(input_data, 0).astype(np.float64)  # (81, 256, 256)

            # 3. Detect FFT convention
            dc_at_center = self._detect_fft_convention(intensities)

            # 4. Prepare diffraction amplitudes in FFT-native layout
            diff_amp = np.sqrt(intensities)  # unshifted amplitudes for potential alternate convention
            diff_amp_fft = self._prepare_diffraction_amplitudes(intensities, dc_at_center)

            # 5. Compute scan positions
            positions = self._compute_scan_positions()

            # 6. Initialize probe from data
            P = self._initialize_probe(diff_amp_fft)

            # 7. Initialize object
            O = self._initialize_object()

            # 8. Run ePIE
            O_result, P_result, n_epochs = self._run_epie(
                O, P, diff_amp_fft, positions, t_start, self.time_limit
            )

            # 9. Optionally try alternate FFT convention if enough time remains
            window = self._create_localizing_window()
            O_result, P_result = self._try_alternate_convention(
                O_result, P_result, diff_amp, diff_amp_fft, positions,
                dc_at_center, window, t_start
            )

            # 10. Phase alignment
            scan_mask = self._build_scan_coverage_mask(positions)
            O_aligned = self._align_phase(O_result, scan_mask)

            # 11. Format output
            result = self._format_output(O_aligned)

            return result



if __name__ == "__main__":
    # Load data
    input_data = np.load("dataset/input.npy")

    # Create solver and run reconstruction
    solver = InverseSolver()
    result = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)