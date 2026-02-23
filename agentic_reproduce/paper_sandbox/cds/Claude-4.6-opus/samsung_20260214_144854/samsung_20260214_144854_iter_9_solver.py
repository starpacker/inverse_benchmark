import numpy as np
from typing import Tuple, List, Dict, Optional, Any

class InverseSolver:
    """
    Ptychographic phase retrieval solver using rPIE (regularized Ptychographic Iterative Engine)
    for coherent diffractive imaging of copper pad topography in hybrid bonding wafer inspection.
    """

    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_scan: int = 9,
            n_positions: int = 81,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            obj_init_value: float = 1.0,
            alpha: float = 0.5,
            n_iter_main: int = 800,
            n_iter_quick: int = 50,
            probe_update_start: int = 5,
            stagnation_patience: int = 250,
            sigma_candidates: Optional[List[int]] = None,
            tv_weight: float = 0.005,
            tv_iterations: int = 30,
            phase_hist_bins: int = 100,
            mag_hist_bins: int = 50,
            illum_percentile: float = 50.0,
        ) -> None:
            # Grid / geometry
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_scan: int = n_scan
            self.n_positions: int = n_positions

            # Physics
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size

            # Initialization
            self.obj_init_value: float = obj_init_value

            # rPIE hyperparameters
            self.alpha: float = alpha
            self.n_iter_main: int = n_iter_main
            self.n_iter_quick: int = n_iter_quick
            self.probe_update_start: int = probe_update_start
            self.stagnation_patience: int = stagnation_patience

            # Probe selection
            self.sigma_candidates: List[int] = sigma_candidates if sigma_candidates is not None else [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]

            # TV denoising
            self.tv_weight: float = tv_weight
            self.tv_iterations: int = tv_iterations

            # Phase referencing
            self.phase_hist_bins: int = phase_hist_bins
            self.mag_hist_bins: int = mag_hist_bins
            self.illum_percentile: float = illum_percentile

            # Derived constants
            self.n_pix: int = probe_size * probe_size  # 65536

            # Precompute coordinate grids for probe creation
            half: int = probe_size // 2  # 128
            Y, X = np.mgrid[-half:half, -half:half]
            self.R2: np.ndarray = (X**2 + Y**2).astype(np.float64)

            # Internal state (set during solve)
            self.use_fftshift: bool = False
            self.positions: List[Tuple[int, int]] = []
            self.amplitudes: Optional[np.ndarray] = None
            self.mean_total_intensity: float = 0.0
            self.probe: Optional[np.ndarray] = None
            self.obj: Optional[np.ndarray] = None
            self.best_obj: Optional[np.ndarray] = None
            self.best_probe: Optional[np.ndarray] = None
            self.best_fourier_error: float = np.inf
            self.best_epoch: int = 0
            self.error_history: List[float] = []

    def _detect_dc_location(self, data: np.ndarray) -> bool:
            """Detect whether DC component is at center (needs fftshift) or corner."""
            avg_pattern = np.mean(data, axis=0)
            center_region = avg_pattern[120:136, 120:136].mean()
            corner_region = np.mean([
                avg_pattern[:8, :8].mean(),
                avg_pattern[:8, -8:].mean(),
                avg_pattern[-8:, :8].mean(),
                avg_pattern[-8:, -8:].mean()
            ])
            return center_region > corner_region * 1.5

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
            """Compute 9x9 grid scan positions with appropriate step size."""
            scan_step = (self.obj_size - self.probe_size) / (self.n_scan - 1)  # = 43.0

            positions = []
            for iy in range(self.n_scan):
                for ix in range(self.n_scan):
                    ry = int(round(iy * scan_step))
                    rx = int(round(ix * scan_step))
                    # Clamp to valid range
                    ry = min(ry, self.obj_size - self.probe_size)
                    rx = min(rx, self.obj_size - self.probe_size)
                    positions.append((ry, rx))

            assert len(positions) == self.n_positions
            return positions

    def forward_prop(self, x: np.ndarray) -> np.ndarray:
            """Real-space to Fourier-space far-field propagation (FFT with appropriate shifting)."""
            if self.use_fftshift:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            else:
                return np.fft.fft2(x)

    def backward_prop(self, X: np.ndarray) -> np.ndarray:
            """Fourier-space to real-space inverse propagation (IFFT with appropriate shifting)."""
            if self.use_fftshift:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))
            else:
                return np.fft.ifft2(X)

    def _make_probe(self, sigma: float) -> np.ndarray:
            """Create Gaussian probe with correct energy scaling from Parseval's theorem."""
            P = np.exp(-self.R2 / (2.0 * sigma**2)).astype(np.complex128)
            current_energy = np.sum(np.abs(P)**2)
            target_energy = self.mean_total_intensity / (self.n_pix * self.obj_init_value**2)
            P *= np.sqrt(target_energy / current_energy)
            return P

    def _quick_reconstruction(
            self, P_init: np.ndarray, n_iter: int = 50
        ) -> Tuple[float, List[float]]:
            """Run short rPIE reconstruction to evaluate probe quality."""
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128) * self.obj_init_value
            P = P_init.copy()

            errors: List[float] = []
            total_energy = np.sum(self.amplitudes ** 2)

            for ep in range(n_iter):
                order = np.random.permutation(self.n_positions)
                epoch_err = 0.0

                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size].copy()

                    psi = P * O_patch
                    Psi = self.forward_prop(psi)
                    mag = np.abs(Psi)

                    # Modulus replacement
                    Psi_c = self.amplitudes[j] * Psi / (mag + 1e-12)
                    epoch_err += np.sum((mag - self.amplitudes[j]) ** 2)

                    psi_c = self.backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # rPIE object update
                    P_conj = np.conj(P)
                    P_abs2 = np.abs(P) ** 2
                    P_max2 = np.max(P_abs2)
                    denom_O = (1.0 - self.alpha) * P_abs2 + self.alpha * P_max2
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] += P_conj * dpsi / (denom_O + 1e-12)

                    # Probe update after initial epochs
                    if ep >= self.probe_update_start:
                        O_conj = np.conj(O_patch)
                        O_abs2 = np.abs(O_patch) ** 2
                        O_max2 = np.max(O_abs2)
                        denom_P = (1.0 - self.alpha) * O_abs2 + self.alpha * O_max2
                        P += O_conj * dpsi / (denom_P + 1e-12)

                rel_err = epoch_err / (total_energy + 1e-30)
                errors.append(rel_err)

            return errors[-1], errors

    def _select_best_probe(self) -> np.ndarray:
            """Select optimal probe width by running quick reconstructions for all sigma candidates."""
            best_sigma = 30
            best_err = np.inf

            for sigma in self.sigma_candidates:
                P_test = self._make_probe(sigma)
                final_err, _ = self._quick_reconstruction(P_test, n_iter=self.n_iter_quick)
                if final_err < best_err:
                    best_err = final_err
                    best_sigma = sigma

            return self._make_probe(best_sigma)

    def _run_rpie(
            self,
            P_init: np.ndarray,
            O_init: np.ndarray,
            n_iter: int = 800,
        ) -> Tuple[np.ndarray, np.ndarray, float, int, List[float]]:
            """
            Run full rPIE reconstruction with early stopping.

            Returns:
                best_O: Best object estimate (complex, obj_size x obj_size)
                best_P: Best probe estimate (complex, probe_size x probe_size)
                best_error: Best Fourier error achieved
                best_epoch: Epoch at which best error was achieved
                error_history: List of per-epoch relative Fourier errors
            """
            O = O_init.copy()
            P = P_init.copy()

            best_O = O.copy()
            best_P = P.copy()
            best_fourier_error = np.inf
            best_epoch = 0
            error_history: List[float] = []

            total_energy = np.sum(self.amplitudes ** 2)

            for epoch in range(n_iter):
                order = np.random.permutation(self.n_positions)
                epoch_err = 0.0

                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size].copy()

                    # Forward propagation
                    psi = P * O_patch
                    Psi = self.forward_prop(psi)
                    mag = np.abs(Psi)

                    # Modulus replacement (Fourier constraint)
                    Psi_c = self.amplitudes[j] * Psi / (mag + 1e-12)
                    epoch_err += np.sum((mag - self.amplitudes[j]) ** 2)

                    # Backward propagation
                    psi_c = self.backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # rPIE object update
                    P_abs2 = np.abs(P) ** 2
                    P_max2 = np.max(P_abs2)
                    denom_O = (1.0 - self.alpha) * P_abs2 + self.alpha * P_max2
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] += np.conj(P) * dpsi / (denom_O + 1e-12)

                    # Probe update after initial epochs
                    if epoch >= self.probe_update_start:
                        O_abs2 = np.abs(O_patch) ** 2
                        O_max2 = np.max(O_abs2)
                        denom_P = (1.0 - self.alpha) * O_abs2 + self.alpha * O_max2
                        P += np.conj(O_patch) * dpsi / (denom_P + 1e-12)

                rel_err = epoch_err / (total_energy + 1e-30)
                error_history.append(rel_err)

                # Track best state
                if rel_err < best_fourier_error:
                    best_fourier_error = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                # Early stopping: if no improvement for stagnation_patience epochs
                if epoch - best_epoch > self.stagnation_patience:
                    break

            return best_O, best_P, best_fourier_error, best_epoch, error_history

    def _try_alternate_fft_convention(self) -> None:
            """If initial reconstruction error is high, try flipping FFT shift convention and re-run."""
            # Check if error after 50 epochs is still high (less than 10% improvement)
            if len(self.error_history) >= 50 and self.error_history[49] > self.error_history[0] * 0.9:
                # Save current state
                original_use_fftshift = self.use_fftshift
                original_best_error = self.best_fourier_error
                original_best_O = self.best_obj.copy() if self.best_obj is not None else None
                original_best_P = self.best_probe.copy() if self.best_probe is not None else None

                # Flip the convention
                self.use_fftshift = not self.use_fftshift

                # Re-run probe selection with reduced candidate set for speed
                best_sigma2 = 30
                best_err2 = np.inf
                for sigma in [15, 25, 30, 40, 50, 60]:
                    P_test = self._make_probe(sigma)
                    final_err, _ = self._quick_reconstruction(P_test, n_iter=self.n_iter_quick)
                    if final_err < best_err2:
                        best_err2 = final_err
                        best_sigma2 = sigma

                if best_err2 < original_best_error * 0.8:
                    # The alternate convention is better — rerun full reconstruction
                    P_new = self._make_probe(best_sigma2)
                    O_new = np.ones((self.obj_size, self.obj_size), dtype=np.complex128) * self.obj_init_value

                    best_O, best_P, best_error, best_epoch, error_history = self._run_rpie(
                        P_new, O_new, n_iter=self.n_iter_main
                    )

                    self.best_obj = best_O
                    self.best_probe = best_P
                    self.best_fourier_error = best_error
                    self.best_epoch = best_epoch
                    self.error_history = error_history
                else:
                    # Revert convention — original was better or comparable
                    self.use_fftshift = original_use_fftshift
                    # Restore original best state
                    if original_best_O is not None:
                        self.best_obj = original_best_O
                    if original_best_P is not None:
                        self.best_probe = original_best_P
                    self.best_fourier_error = original_best_error

    def _tv_denoise_2d(
            self, img: np.ndarray, weight: float, n_iter: int = 50
        ) -> np.ndarray:
            """Chambolle primal-dual TV denoising for 2D real-valued image."""
            px = np.zeros_like(img)
            py = np.zeros_like(img)
            tau = 0.125

            for _ in range(n_iter):
                # Divergence of (px, py)
                div_p = np.zeros_like(img)
                # d/dx: div_p += d(px)/dx
                div_p[1:, :] += px[:-1, :]
                div_p[0, :] = 0.0  # boundary
                div_p[:-1, :] -= px[:-1, :]
                # Note: div_p[-1, :] has no subtraction from px (boundary)
                # d/dy: div_p += d(py)/dy
                div_p[:, 1:] += py[:, :-1]
                div_p[:, :-1] -= py[:, :-1]

                u = img + weight * div_p

                # Gradient of u
                gx = np.zeros_like(img)
                gy = np.zeros_like(img)
                gx[:-1, :] = u[1:, :] - u[:-1, :]
                gy[:, :-1] = u[:, 1:] - u[:, :-1]

                norm_g = np.sqrt(gx**2 + gy**2)
                denom = 1.0 + tau * norm_g
                px = (px + tau * gx) / denom
                py = (py + tau * gy) / denom

            # Final reconstruction
            div_p = np.zeros_like(img)
            div_p[1:, :] += px[:-1, :]
            div_p[:-1, :] -= px[:-1, :]
            div_p[:, 1:] += py[:, :-1]
            div_p[:, :-1] -= py[:, :-1]

            return img + weight * div_p

    def _apply_tv_denoising(
            self, O: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Apply conservative TV denoising to magnitude and phase separately.

            Returns:
                mag_tv: TV-denoised magnitude (obj_size x obj_size)
                phase_tv: TV-denoised phase via cos/sin decomposition (obj_size x obj_size)
            """
            mag_raw = np.abs(O)
            phase_raw = np.angle(O)

            # TV denoise magnitude directly
            mag_tv = self._tv_denoise_2d(mag_raw, weight=self.tv_weight, n_iter=self.tv_iterations)

            # For phase: denoise cos and sin separately to handle wrapping
            cos_tv = self._tv_denoise_2d(np.cos(phase_raw), weight=self.tv_weight, n_iter=self.tv_iterations)
            sin_tv = self._tv_denoise_2d(np.sin(phase_raw), weight=self.tv_weight, n_iter=self.tv_iterations)
            phase_tv = np.arctan2(sin_tv, cos_tv)

            return mag_tv, phase_tv

    def _compute_illumination_map(self) -> np.ndarray:
            """Compute per-pixel illumination count from all scan positions."""
            illum_count = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            for j in range(self.n_positions):
                ry, rx = self.positions[j]
                illum_count[ry:ry + self.probe_size, rx:rx + self.probe_size] += 1.0
            return illum_count

    def _find_substrate_phase(
            self, phase: np.ndarray, well_lit_mask: np.ndarray
        ) -> float:
            """Find dominant substrate phase via histogram peak detection with weighted average refinement."""
            phase_well = phase[well_lit_mask]

            # Build histogram over [-pi, pi]
            hist_phase, bin_edges_phase = np.histogram(
                phase_well, bins=self.phase_hist_bins, range=(-np.pi, np.pi)
            )
            bin_centers_phase = 0.5 * (bin_edges_phase[:-1] + bin_edges_phase[1:])

            # The dominant peak corresponds to the substrate (largest area fraction)
            dominant_idx = np.argmax(hist_phase)
            substrate_phase = bin_centers_phase[dominant_idx]

            # Refine: weighted average around the peak (±3 bins)
            lo = max(0, dominant_idx - 3)
            hi = min(len(hist_phase), dominant_idx + 4)
            weights = hist_phase[lo:hi].astype(np.float64)
            if np.sum(weights) > 0:
                substrate_phase = np.average(bin_centers_phase[lo:hi], weights=weights)

            return float(substrate_phase)

    def _correct_phase_sign(
            self,
            phase: np.ndarray,
            mag: np.ndarray,
            well_lit_mask: np.ndarray,
        ) -> np.ndarray:
            """
            Ensure pad-substrate phase difference is positive (recessed pads in reflection geometry).
            Uses Otsu thresholding on magnitude to separate pad/substrate regions.
            """
            mag_well = mag[well_lit_mask]

            # Otsu threshold on magnitude
            hist_mag, bin_edges_mag = np.histogram(mag_well, bins=self.mag_hist_bins)
            bin_centers_mag = 0.5 * (bin_edges_mag[:-1] + bin_edges_mag[1:])

            total = np.sum(hist_mag)
            sum_total = np.sum(hist_mag * bin_centers_mag)
            sum_bg = 0.0
            w_bg = 0
            max_var = 0.0
            threshold_mag = float(np.median(mag_well))

            for i in range(len(hist_mag)):
                w_bg += hist_mag[i]
                if w_bg == 0:
                    continue
                w_fg = total - w_bg
                if w_fg == 0:
                    break
                sum_bg += hist_mag[i] * bin_centers_mag[i]
                mean_bg = sum_bg / w_bg
                mean_fg = (sum_total - sum_bg) / w_fg
                var_between = float(w_bg) * float(w_fg) * (mean_bg - mean_fg) ** 2
                if var_between > max_var:
                    max_var = var_between
                    threshold_mag = float(bin_centers_mag[i])

            # Two classes based on magnitude threshold
            class1 = well_lit_mask & (mag <= threshold_mag)
            class2 = well_lit_mask & (mag > threshold_mag)

            n1, n2 = np.sum(class1), np.sum(class2)

            # Substrate is the larger class (occupies ~64% of area for duty cycle 0.6)
            if n1 >= n2:
                substrate_mask = class1
                pad_mask = class2
            else:
                substrate_mask = class2
                pad_mask = class1

            # Check if pads have positive or negative mean phase relative to substrate
            phase_out = phase.copy()
            if np.sum(pad_mask) > 50:
                mean_pad_phase = np.mean(phase_out[pad_mask])
                mean_sub_phase = np.mean(phase_out[substrate_mask])
                pad_phase_diff = mean_pad_phase - mean_sub_phase

                # Expected: pad_phase_diff > 0 (recessed pads in reflection geometry)
                # If negative, negate phase
                if pad_phase_diff < -0.1:
                    phase_out = -phase_out

            return phase_out

    def _reference_phase(
            self, mag: np.ndarray, phase: np.ndarray
        ) -> np.ndarray:
            """
            Full phase referencing pipeline: illumination map → substrate phase → sign correction.
            """
            # Compute illumination map
            illum_count = self._compute_illumination_map()

            # Determine well-illuminated region
            nonzero_illum = illum_count[illum_count > 0]
            if len(nonzero_illum) > 0:
                illum_thresh = np.percentile(nonzero_illum, self.illum_percentile)
            else:
                illum_thresh = 1.0
            well_lit_mask = illum_count >= illum_thresh

            # Find substrate phase via histogram peak detection
            substrate_phase = self._find_substrate_phase(phase, well_lit_mask)

            # Reference phase to substrate (subtract substrate phase)
            phase_ref = phase - substrate_phase
            # Wrap to [-pi, pi]
            phase_ref = np.arctan2(np.sin(phase_ref), np.cos(phase_ref))

            # Correct phase sign so pads have positive phase relative to substrate
            phase_ref = self._correct_phase_sign(phase_ref, mag, well_lit_mask)

            return phase_ref

    def _format_output(
            self, mag: np.ndarray, phase: np.ndarray
        ) -> np.ndarray:
            """
            Assemble final output array of shape (2, obj_size, obj_size).
            Channel 0: magnitude in natural units. Channel 1: referenced phase in radians.
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            output[0] = mag.astype(np.float64)
            output[1] = phase.astype(np.float64)
            return output

    def solve(self, data: np.ndarray) -> np.ndarray:
            """
            Full ptychographic phase retrieval pipeline.

            Args:
                data: Input diffraction intensity patterns, shape (81, 256, 256), real non-negative.

            Returns:
                output: Reconstructed object, shape (2, 600, 600).
                        Channel 0 = magnitude, Channel 1 = phase (radians).
            """
            # 1. Compute amplitudes = sqrt(max(data, 0))
            self.amplitudes = np.sqrt(np.maximum(data, 0.0))

            # Compute mean total intensity per pattern (for probe scaling)
            self.mean_total_intensity = np.mean(np.sum(data, axis=(1, 2)))

            # 2. Detect DC location
            self.use_fftshift = self._detect_dc_location(data)

            # 3. Compute scan positions
            self.positions = self._compute_scan_positions()

            # 4. Select best probe via quick reconstructions
            self.probe = self._select_best_probe()

            # 5. Run main rPIE reconstruction
            O_init = np.ones((self.obj_size, self.obj_size), dtype=np.complex128) * self.obj_init_value

            self.best_obj, self.best_probe, self.best_fourier_error, self.best_epoch, self.error_history = \
                self._run_rpie(self.probe, O_init, n_iter=self.n_iter_main)

            # 6. Try alternate FFT convention if error is high
            self._try_alternate_fft_convention()

            # 7. Apply TV denoising
            mag_tv, phase_tv = self._apply_tv_denoising(self.best_obj)

            # 8. Reference phase
            phase_ref = self._reference_phase(mag_tv, phase_tv)

            # 9. Format and return output
            output = self._format_output(mag_tv, phase_ref)

            return output



if __name__ == "__main__":
    # Load data
    input_data: np.ndarray = np.load("dataset/input.npy")

    # Create solver and run
    solver: InverseSolver = InverseSolver()
    result: np.ndarray = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)