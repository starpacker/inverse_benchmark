import numpy as np
import time
from typing import Tuple, Optional, List

class InverseSolver:
    def __init__(
            self,
            obj_size: int = 600,
            patch_size: int = 256,
            n_side: int = 9,
            step_pixels: int = 40,
            n_positions: int = 81,
            wavelength_nm: float = 13.5,
            pixel_size_nm: float = 27.0,
            n_pads: int = 15,
            pixels_per_pad: int = 22,
            duty_cycle: float = 0.6,
            step_obj: float = 1.0,
            step_probe: float = 0.8,
            gamma_obj: float = 0.6,
            gamma_probe: float = 0.4,
            alpha_obj_factor: float = 1e-3,
            alpha_probe_factor: float = 1e-3,
            probe_update_start_epoch: int = 3,
            n_epochs: int = 300,
            time_limit: float = 55.0,
            eps: float = 1e-12,
            probe_window_sigma: float = 60.0,
            phase_histogram_bins: int = 720,
            phase_histogram_smooth_kernel_size: int = 15,
            amp_valid_threshold_fraction: float = 0.05,
        ) -> None:
            # Object / grid parameters
            self.obj_size: int = obj_size
            self.patch_size: int = patch_size
            self.n_side: int = n_side
            self.step_pixels: int = step_pixels
            self.n_positions: int = n_positions

            # Physical parameters
            self.wavelength_nm: float = wavelength_nm
            self.pixel_size_nm: float = pixel_size_nm
            self.n_pads: int = n_pads
            self.pixels_per_pad: int = pixels_per_pad
            self.duty_cycle: float = duty_cycle

            # Optimizer hyperparameters
            self.step_obj: float = step_obj
            self.step_probe: float = step_probe
            self.gamma_obj: float = gamma_obj
            self.gamma_probe: float = gamma_probe
            self.alpha_obj_factor: float = alpha_obj_factor
            self.alpha_probe_factor: float = alpha_probe_factor
            self.probe_update_start_epoch: int = probe_update_start_epoch
            self.n_epochs: int = n_epochs

            # Runtime constraints
            self.time_limit: float = time_limit
            self.eps: float = eps

            # Probe initialization
            self.probe_window_sigma: float = probe_window_sigma

            # Phase post-processing
            self.phase_histogram_bins: int = phase_histogram_bins
            self.phase_histogram_smooth_kernel_size: int = phase_histogram_smooth_kernel_size
            self.amp_valid_threshold_fraction: float = amp_valid_threshold_fraction

            # Internal state (populated during solve)
            self.positions: Optional[np.ndarray] = None
            self.dc_at_center: Optional[bool] = None
            self.O: Optional[np.ndarray] = None
            self.P: Optional[np.ndarray] = None
            self.t_start: Optional[float] = None

    def _compute_scan_positions(self) -> np.ndarray:
            """Compute (n_positions, 2) array of (row, col) scan positions."""
            total_scan = self.patch_size + (self.n_side - 1) * self.step_pixels
            offset = (self.obj_size - total_scan) // 2

            positions = np.zeros((self.n_positions, 2), dtype=np.int32)
            idx = 0
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions[idx, 0] = offset + i * self.step_pixels
                    positions[idx, 1] = offset + j * self.step_pixels
                    idx += 1

            return positions

    def _detect_fft_convention(self, intensities: np.ndarray) -> bool:
            """Detect whether DC component is at center or corner of diffraction patterns.

            Args:
                intensities: Measured diffraction intensities, shape (n_positions, patch_size, patch_size).

            Returns:
                True if DC is at center, False if at corners.
            """
            ps = self.patch_size
            half_width = 40

            center_region_r = slice(ps // 2 - half_width, ps // 2 + half_width)
            center_region_c = slice(ps // 2 - half_width, ps // 2 + half_width)
            center_energy = np.mean(intensities[:, center_region_r, center_region_c])

            corner_energy = (
                np.mean(intensities[:, :half_width, :half_width])
                + np.mean(intensities[:, :half_width, -half_width:])
                + np.mean(intensities[:, -half_width:, :half_width])
                + np.mean(intensities[:, -half_width:, -half_width:])
            ) / 4.0

            return bool(center_energy > corner_energy)

    def _init_probe_from_data(
            self, diff_amp: np.ndarray, dc_at_center: bool
        ) -> np.ndarray:
            """Initialize probe estimate from average diffraction amplitude.

            Args:
                diff_amp: Square root of measured intensities, shape (n_positions, patch_size, patch_size).
                dc_at_center: Whether DC frequency is at center of patterns.

            Returns:
                Complex probe array, shape (patch_size, patch_size).
            """
            ps = self.patch_size
            avg_amp = np.mean(diff_amp, axis=0)  # (patch_size, patch_size)

            # Shift to standard FFT layout (DC at corner) if DC is at center
            if dc_at_center:
                avg_amp_shifted = np.fft.ifftshift(avg_amp)
            else:
                avg_amp_shifted = avg_amp.copy()

            # Inverse FFT of amplitude gives an estimate related to the probe
            P = np.fft.ifft2(avg_amp_shifted)

            # The probe should be spatially localized — shift so peak is at center,
            # apply Gaussian window, then shift back
            P_shifted = np.fft.fftshift(P)

            yy, xx = np.mgrid[:ps, :ps]
            yy = yy - ps / 2.0
            xx = xx - ps / 2.0
            r2 = xx**2 + yy**2

            window = np.exp(-r2 / (2.0 * self.probe_window_sigma**2))
            P_centered = P_shifted * window

            P_final = np.fft.ifftshift(P_centered)

            return P_final

    def _extract_patches(self, O: np.ndarray, positions: np.ndarray) -> np.ndarray:
            """Extract all object patches at scan positions.

            Args:
                O: Complex object array, shape (obj_size, obj_size).
                positions: Scan positions, shape (n_positions, 2).

            Returns:
                Patches array, shape (n_positions, patch_size, patch_size).
            """
            ps = self.patch_size
            n = len(positions)
            patches = np.empty((n, ps, ps), dtype=O.dtype)
            for j in range(n):
                ry, rx = positions[j]
                patches[j] = O[ry:ry + ps, rx:rx + ps]
            return patches

    def _scatter_add_complex(
            self,
            target: np.ndarray,
            patches: np.ndarray,
            positions: np.ndarray,
        ) -> None:
            """Accumulate complex patches back into target array (in-place).

            Args:
                target: Target array, shape (obj_size, obj_size), complex.
                patches: Patches to scatter, shape (n_positions, patch_size, patch_size).
                positions: Scan positions, shape (n_positions, 2).
            """
            ps = self.patch_size
            for j in range(len(positions)):
                ry, rx = positions[j]
                target[ry:ry + ps, rx:rx + ps] += patches[j]

    def _scatter_add_real(
            self,
            target: np.ndarray,
            values: np.ndarray,
            positions: np.ndarray,
        ) -> None:
            """Accumulate real-valued patches back into target array (in-place).

            Args:
                target: Target array, shape (obj_size, obj_size), float.
                values: Real patches to scatter, shape (n_positions, patch_size, patch_size).
                positions: Scan positions, shape (n_positions, 2).
            """
            ps = self.patch_size
            for j in range(len(positions)):
                ry, rx = positions[j]
                target[ry:ry + ps, rx:rx + ps] += values[j]

    def forward(
            self, O: np.ndarray, P: np.ndarray, positions: np.ndarray
        ) -> np.ndarray:
            """Compute forward model: batch FFT of exit waves at all scan positions.

            Args:
                O: Complex object, shape (obj_size, obj_size).
                P: Complex probe, shape (patch_size, patch_size).
                positions: Scan positions, shape (n_positions, 2).

            Returns:
                Far-field complex amplitudes, shape (n_positions, patch_size, patch_size).
            """
            O_patches = self._extract_patches(O, positions)  # (n_positions, patch_size, patch_size)
            psi_batch = P[np.newaxis, :, :] * O_patches  # (n_positions, patch_size, patch_size)
            Psi_batch = np.fft.fft2(psi_batch)  # (n_positions, patch_size, patch_size)
            return Psi_batch

    def _compute_residual(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp_w: np.ndarray,
            positions: np.ndarray,
        ) -> float:
            """Compute mean squared amplitude residual.

            Args:
                O: Complex object, shape (obj_size, obj_size).
                P: Complex probe, shape (patch_size, patch_size).
                diff_amp_w: Working-frame diffraction amplitudes, shape (n_positions, patch_size, patch_size).
                positions: Scan positions, shape (n_positions, 2).

            Returns:
                Scalar residual value.
            """
            Psi_batch = self.forward(O, P, positions)  # (n_positions, patch_size, patch_size)
            model_amp = np.abs(Psi_batch)  # (n_positions, patch_size, patch_size)
            residual = np.mean((model_amp - diff_amp_w) ** 2)
            return float(residual)

    def _reconstruct_batch(
            self,
            diff_amp: np.ndarray,
            positions: np.ndarray,
            P_init: np.ndarray,
            dc_at_center: bool,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Core batch gradient descent reconstruction with Nesterov momentum.

            Iterates:
              1. Lookahead (Nesterov) for O and P
              2. Batch extract patches, compute exit waves
              3. Batch FFT, Fourier magnitude replacement
              4. Batch IFFT, compute difference
              5. Accumulate object gradient via scatter-add, normalize, momentum update
              6. Accumulate probe gradient, normalize, momentum update (after warmup)

            Args:
                diff_amp: Square root of measured intensities, shape (n_positions, patch_size, patch_size).
                positions: Scan positions, shape (n_positions, 2).
                P_init: Initial probe estimate, shape (patch_size, patch_size).
                dc_at_center: Whether DC is at center of diffraction patterns.

            Returns:
                Tuple of (O_final, P_final, diff_amp_working_frame).
            """
            ps = self.patch_size
            obj_size = self.obj_size
            n_pos = self.n_positions
            eps = self.eps

            # Prepare measured amplitudes in working FFT frame (DC at corner)
            if dc_at_center:
                diff_amp_w = np.fft.ifftshift(diff_amp, axes=(-2, -1))
            else:
                diff_amp_w = diff_amp.copy()

            # Initialize object and probe
            O = np.ones((obj_size, obj_size), dtype=np.complex128)
            P = P_init.copy()

            # Momentum buffers
            O_momentum = np.zeros_like(O)
            P_momentum = np.zeros_like(P)

            gamma_obj = self.gamma_obj
            gamma_probe = self.gamma_probe
            step_obj = self.step_obj
            step_probe = self.step_probe
            probe_update_start = self.probe_update_start_epoch
            n_epochs = self.n_epochs

            for epoch in range(n_epochs):
                if time.time() - self.t_start > self.time_limit - 1.0:
                    break

                # --- Lookahead (Nesterov) ---
                O_look = O + gamma_obj * O_momentum
                if epoch >= probe_update_start:
                    P_look = P + gamma_probe * P_momentum
                else:
                    P_look = P.copy()

                # --- Extract all patches from lookahead object ---
                O_patches = self._extract_patches(O_look, positions)  # (n_pos, ps, ps)

                # --- Batch forward: exit waves ---
                psi_batch = P_look[np.newaxis, :, :] * O_patches  # (n_pos, ps, ps)

                # --- Batch FFT ---
                Psi_batch = np.fft.fft2(psi_batch)  # (n_pos, ps, ps)

                # --- Fourier magnitude constraint ---
                Psi_amp = np.abs(Psi_batch) + eps
                Psi_corrected = diff_amp_w * Psi_batch / Psi_amp  # (n_pos, ps, ps)

                # --- Batch IFFT ---
                psi_c_batch = np.fft.ifft2(Psi_corrected)  # (n_pos, ps, ps)

                # --- Difference ---
                dpsi_batch = psi_c_batch - psi_batch  # (n_pos, ps, ps)

                # --- Object gradient: sum_j conj(P) * dpsi_j ---
                obj_grad_patches = np.conj(P_look)[np.newaxis, :, :] * dpsi_batch  # (n_pos, ps, ps)

                obj_numerator = np.zeros((obj_size, obj_size), dtype=np.complex128)
                self._scatter_add_complex(obj_numerator, obj_grad_patches, positions)

                # Denominator: sum_j |P(r-Rj)|^2
                P_abs_sq = np.abs(P_look) ** 2
                obj_denom = np.zeros((obj_size, obj_size), dtype=np.float64)
                P_sq_tiles = np.broadcast_to(P_abs_sq[np.newaxis, :, :], (n_pos, ps, ps)).copy()
                self._scatter_add_real(obj_denom, P_sq_tiles, positions)

                alpha_obj = self.alpha_obj_factor * np.max(P_abs_sq)
                obj_update = step_obj * obj_numerator / (obj_denom + alpha_obj)

                # Nesterov update for object
                O_momentum = gamma_obj * O_momentum + obj_update
                O = O + O_momentum

                # --- Probe gradient ---
                if epoch >= probe_update_start:
                    # sum_j conj(O_j) * dpsi_j (all in probe frame)
                    probe_grad = np.sum(np.conj(O_patches) * dpsi_batch, axis=0)  # (ps, ps)

                    # Denominator: sum_j |O_j|^2
                    probe_denom = np.sum(np.abs(O_patches) ** 2, axis=0)  # (ps, ps)
                    alpha_probe = self.alpha_probe_factor * np.max(probe_denom)

                    probe_update = step_probe * probe_grad / (probe_denom + alpha_probe)

                    P_momentum = gamma_probe * P_momentum + probe_update
                    P = P + P_momentum

            return O, P, diff_amp_w

    def _try_alternate_convention(
            self,
            diff_amp: np.ndarray,
            positions: np.ndarray,
            O_current: np.ndarray,
            P_current: np.ndarray,
            diff_amp_w_current: np.ndarray,
            dc_at_center_current: bool,
        ) -> np.ndarray:
            """If time permits, try the opposite FFT convention and keep the better result.

            Args:
                diff_amp: Original diffraction amplitudes, shape (n_positions, patch_size, patch_size).
                positions: Scan positions, shape (n_positions, 2).
                O_current: Current best object estimate.
                P_current: Current best probe estimate.
                diff_amp_w_current: Working-frame amplitudes for current convention.
                dc_at_center_current: Current convention flag.

            Returns:
                Best object estimate, shape (obj_size, obj_size).
            """
            time_used = time.time() - self.t_start
            time_left = self.time_limit - time_used

            if time_left < 15.0:
                return O_current

            # Compute residual for current result
            res_current = self._compute_residual(O_current, P_current, diff_amp_w_current, positions)

            # Try opposite convention
            dc_at_center_alt = not dc_at_center_current
            P_init_alt = self._init_probe_from_data(diff_amp, dc_at_center_alt)
            O_alt, P_alt, diff_amp_w_alt = self._reconstruct_batch(
                diff_amp, positions, P_init_alt, dc_at_center_alt
            )

            # Compute residual for alternate result
            res_alt = self._compute_residual(O_alt, P_alt, diff_amp_w_alt, positions)

            if res_alt < res_current:
                return O_alt
            else:
                return O_current

    def _compute_scan_coverage_mask(self, positions: np.ndarray) -> np.ndarray:
            """Compute boolean mask of pixels covered by at least one scan position.

            Args:
                positions: Scan positions, shape (n_positions, 2).

            Returns:
                Boolean mask, shape (obj_size, obj_size).
            """
            mask = np.zeros((self.obj_size, self.obj_size), dtype=bool)
            ps = self.patch_size
            for j in range(len(positions)):
                ry, rx = positions[j]
                mask[ry:ry + ps, rx:rx + ps] = True
            return mask

    def _align_global_phase(self, O: np.ndarray, positions: np.ndarray) -> np.ndarray:
            """Remove global phase ambiguity by aligning dominant phase to zero.

            Uses histogram-based mode detection on phase values within scan coverage.

            Args:
                O: Complex object, shape (obj_size, obj_size).
                positions: Scan positions, shape (n_positions, 2).

            Returns:
                Phase-aligned complex object, shape (obj_size, obj_size).
            """
            phase = np.angle(O)
            amp = np.abs(O)

            # Build scan coverage mask
            scan_mask = self._compute_scan_coverage_mask(positions)

            # Only consider pixels with significant amplitude within scan coverage
            amp_threshold = self.amp_valid_threshold_fraction * np.max(amp)
            valid = scan_mask & (amp > amp_threshold)

            if np.sum(valid) > 100:
                phase_valid = phase[valid]

                # Build histogram of phase values
                hist, bin_edges = np.histogram(
                    phase_valid,
                    bins=self.phase_histogram_bins,
                    range=(-np.pi, np.pi),
                )

                # Smooth histogram to find robust mode
                k_size = self.phase_histogram_smooth_kernel_size
                kernel = np.ones(k_size) / float(k_size)
                hist_smooth = np.convolve(hist.astype(np.float64), kernel, mode='same')

                mode_idx = np.argmax(hist_smooth)
                phase_ref = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
            else:
                phase_ref = 0.0

            # Subtract reference phase to align dominant phase to zero
            O_aligned = O * np.exp(-1j * phase_ref)

            return O_aligned

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """Convert complex object to (2, obj_size, obj_size) output array.

            Channel 0: amplitude |O|
            Channel 1: phase angle(O)

            Args:
                O: Complex object, shape (obj_size, obj_size).

            Returns:
                Output array, shape (2, obj_size, obj_size), float32.
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)
            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """Full reconstruction pipeline.

            Steps:
              1. Preprocess input (clip negatives, compute sqrt)
              2. Compute scan positions
              3. Detect FFT convention
              4. Initialize probe from data
              5. Run batch gradient descent reconstruction
              6. Optionally try alternate FFT convention
              7. Align global phase
              8. Format and return output

            Args:
                input_data: Measured diffraction intensities, shape (n_positions, patch_size, patch_size).

            Returns:
                Reconstructed object as (2, obj_size, obj_size) float32 array.
                  Channel 0: amplitude, Channel 1: phase.
            """
            self.t_start = time.time()

            # Step 1: Preprocess input
            I = np.maximum(input_data, 0).astype(np.float64)
            diff_amp = np.sqrt(I)

            # Step 2: Compute scan positions
            positions = self._compute_scan_positions()
            self.positions = positions

            # Step 3: Detect FFT convention
            dc_at_center = self._detect_fft_convention(I)
            self.dc_at_center = dc_at_center

            # Step 4: Initialize probe from data
            P_init = self._init_probe_from_data(diff_amp, dc_at_center)

            # Step 5: Run batch gradient descent reconstruction
            O_final, P_final, diff_amp_w = self._reconstruct_batch(
                diff_amp, positions, P_init, dc_at_center
            )
            self.O = O_final
            self.P = P_final

            # Step 6: Optionally try alternate FFT convention
            O_final = self._try_alternate_convention(
                diff_amp, positions, O_final, P_final, diff_amp_w, dc_at_center
            )

            # Step 7: Align global phase
            O_final = self._align_global_phase(O_final, positions)

            # Step 8: Format and return output
            output = self._format_output(O_final)

            return output



if __name__ == "__main__":
    # Load data
    input_data = np.load("dataset/input.npy")

    # Create solver and run reconstruction
    solver = InverseSolver()
    result = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)