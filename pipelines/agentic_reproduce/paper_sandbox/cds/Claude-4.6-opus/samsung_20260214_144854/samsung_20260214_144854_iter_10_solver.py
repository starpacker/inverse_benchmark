import numpy as np
from typing import Tuple, List, Optional

class InverseSolver:
    """
    Coherent Diffractive Imaging (CDI) phase retrieval solver using rPIE
    (reflective-prior Ptychographic Iterative Engine) for copper pad topography
    metrology in hybrid bonding wafer inspection.
    """

    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_scan: int = 9,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            sigma_candidates: List[float] = [20.0, 40.0, 70.0],
            n_iter_quick: int = 15,
            n_iter_main: int = 150,
            patience: int = 40,
            alpha: float = 0.5,
            probe_update_start: int = 3,
            tv_weight: float = 0.005,
            tv_iterations: int = 15,
            tv_tau: float = 0.125,
            eps: float = 1e-12,
            phase_hist_bins: int = 100,
            mag_hist_bins: int = 50,
            illum_percentile: float = 50.0,
        ) -> None:
            # Grid / geometry
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_scan: int = n_scan
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size

            # Probe search
            self.sigma_candidates: List[float] = sigma_candidates
            self.n_iter_quick: int = n_iter_quick

            # Main reconstruction
            self.n_iter_main: int = n_iter_main
            self.patience: int = patience
            self.alpha: float = alpha
            self.probe_update_start: int = probe_update_start

            # TV denoising
            self.tv_weight: float = tv_weight
            self.tv_iterations: int = tv_iterations
            self.tv_tau: float = tv_tau

            # Numerical stability
            self.eps: float = eps

            # Phase / magnitude referencing
            self.phase_hist_bins: int = phase_hist_bins
            self.mag_hist_bins: int = mag_hist_bins
            self.illum_percentile: float = illum_percentile

            # Runtime state (populated during solve)
            self.use_fftshift: bool = False
            self.positions: List[Tuple[int, int]] = []
            self.amplitudes: Optional[np.ndarray] = None
            self.mean_total_intensity: float = 0.0
            self.best_sigma: float = 40.0
            self.probe: Optional[np.ndarray] = None
            self.object_estimate: Optional[np.ndarray] = None
            self.error_history: List[float] = []

    def _detect_dc_location(self, data: np.ndarray) -> bool:
            """
            Analytically detect whether DC component is at center or corner
            of the diffraction patterns.

            Args:
                data: Input diffraction intensity patterns, shape (N, H, W).

            Returns:
                True if DC is at center (use fftshift), False otherwise.
            """
            avg_pattern = np.mean(data, axis=0)  # (256, 256)

            # Compare center region intensity vs corner region intensity
            center_val = avg_pattern[124:132, 124:132].mean()
            corner_val = np.mean([
                avg_pattern[:4, :4].mean(),
                avg_pattern[:4, -4:].mean(),
                avg_pattern[-4:, :4].mean(),
                avg_pattern[-4:, -4:].mean(),
            ])

            use_fftshift = center_val > corner_val * 2.0

            # Also check: find actual peak location for robustness
            peak_idx = np.unravel_index(np.argmax(avg_pattern), avg_pattern.shape)
            # If peak is near center, definitely use fftshift
            if 100 < peak_idx[0] < 156 and 100 < peak_idx[1] < 156:
                use_fftshift = True
            elif peak_idx[0] < 30 or peak_idx[0] > 226:
                if peak_idx[1] < 30 or peak_idx[1] > 226:
                    use_fftshift = False

            return use_fftshift

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
            """
            Compute the 9x9 grid of scan positions for ptychographic overlap.

            Returns:
                List of (row, col) tuples for each scan position.
            """
            scan_step = (self.obj_size - self.probe_size) / (self.n_scan - 1)

            positions: List[Tuple[int, int]] = []
            for iy in range(self.n_scan):
                for ix in range(self.n_scan):
                    ry = int(round(iy * scan_step))
                    rx = int(round(ix * scan_step))
                    ry = min(ry, self.obj_size - self.probe_size)
                    rx = min(rx, self.obj_size - self.probe_size)
                    positions.append((ry, rx))

            return positions

    def forward_prop(self, x: np.ndarray) -> np.ndarray:
            """
            Forward propagation (real space -> Fourier space) with correct
            FFT convention based on detected DC location.

            Args:
                x: Complex-valued 2D array in real space, shape (H, W).

            Returns:
                Complex-valued 2D array in Fourier space, shape (H, W).
            """
            if self.use_fftshift:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            else:
                return np.fft.fft2(x)

    def backward_prop(self, X: np.ndarray) -> np.ndarray:
            """
            Backward propagation (Fourier space -> real space) with correct
            FFT convention based on detected DC location.

            Args:
                X: Complex-valued 2D array in Fourier space, shape (H, W).

            Returns:
                Complex-valued 2D array in real space, shape (H, W).
            """
            if self.use_fftshift:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))
            else:
                return np.fft.ifft2(X)

    def _make_probe(self, sigma: float) -> np.ndarray:
            """
            Create a Gaussian probe with given sigma, scaled to match
            expected energy from diffraction data.

            Args:
                sigma: Standard deviation of the Gaussian probe in pixels.

            Returns:
                Complex-valued 2D probe array, shape (probe_size, probe_size).
            """
            half = self.probe_size // 2  # 128
            Y, X = np.mgrid[-half:half, -half:half]
            R2 = (X.astype(np.float64))**2 + (Y.astype(np.float64))**2

            P = np.exp(-R2 / (2.0 * sigma**2)).astype(np.complex128)

            # Scale probe energy so that |P|^2 sums to match expected intensity
            # Since O_init = 1, exit wave energy = sum(|P|^2) * 1
            # This should equal mean_total_intensity / N_pix * N_pix = mean_total_intensity
            # Actually: sum of diffraction pattern = sum(|F{P*O}|^2) = sum(|P*O|^2) (Parseval)
            # With |O|=1: sum(|P|^2) should equal mean_total_intensity
            N_pix = self.probe_size ** 2
            current_energy = np.sum(np.abs(P)**2)
            target_energy = self.mean_total_intensity / N_pix  # per-pixel energy
            # We want sum(|P|^2) = mean_total_intensity
            # So scale factor = sqrt(mean_total_intensity / current_energy)
            P *= np.sqrt(self.mean_total_intensity / current_energy)

            return P

    def _rpie_update_step(
            self,
            O: np.ndarray,
            P: np.ndarray,
            j: int,
            update_probe: bool,
        ) -> Tuple[np.ndarray, np.ndarray, float]:
            """
            Perform a single rPIE update step for one scan position.

            Args:
                O: Current object estimate, shape (obj_size, obj_size), complex128.
                P: Current probe estimate, shape (probe_size, probe_size), complex128.
                j: Index of the current scan position (0..80).
                update_probe: Whether to update the probe in this step.

            Returns:
                Tuple of (updated_O, updated_P, position_error).
            """
            ry, rx = self.positions[j]
            ps = self.probe_size

            # 1. Extract object patch at position j
            O_patch = O[ry:ry + ps, rx:rx + ps].copy()

            # 2. Compute exit wave psi = P * O_patch
            psi = P * O_patch

            # 3. Forward propagate to get Psi
            Psi = self.forward_prop(psi)

            # 4. Modulus replacement: Psi_c = amplitudes[j] * Psi / |Psi|
            mag = np.abs(Psi)
            Psi_c = self.amplitudes[j] * Psi / (mag + self.eps)

            # 9. Compute error = sum((|Psi| - amplitudes[j])^2)
            position_error = float(np.sum((mag - self.amplitudes[j]) ** 2))

            # 5. Back propagate to get corrected exit wave psi_c
            psi_c = self.backward_prop(Psi_c)

            # 6. Compute dpsi = psi_c - psi
            dpsi = psi_c - psi

            # 7. Object update with rPIE denominator
            P_abs2 = np.abs(P) ** 2
            P_max2 = np.max(P_abs2)
            denom_O = (1.0 - self.alpha) * P_abs2 + self.alpha * P_max2
            O[ry:ry + ps, rx:rx + ps] += np.conj(P) * dpsi / (denom_O + self.eps)

            # 8. Optionally probe update with rPIE denominator
            if update_probe:
                O_abs2 = np.abs(O_patch) ** 2
                O_max2 = np.max(O_abs2)
                denom_P = (1.0 - self.alpha) * O_abs2 + self.alpha * O_max2
                P = P + np.conj(O_patch) * dpsi / (denom_P + self.eps)

            return O, P, position_error

    def _quick_reconstruction(
            self, P_init: np.ndarray, n_iter: int
        ) -> float:
            """
            Run a short rPIE reconstruction to evaluate probe quality.

            Args:
                P_init: Initial probe estimate, shape (probe_size, probe_size).
                n_iter: Number of iterations to run.

            Returns:
                Final relative Fourier error (float).
            """
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            P = P_init.copy()
            n_positions = len(self.positions)

            total_energy = np.sum(self.amplitudes ** 2)
            final_err = np.inf

            for ep in range(n_iter):
                order = np.random.permutation(n_positions)
                epoch_err = 0.0

                for idx in range(n_positions):
                    j = order[idx]
                    update_probe = (ep >= 3)
                    O, P, pos_err = self._rpie_update_step(O, P, j, update_probe)
                    epoch_err += pos_err

                final_err = epoch_err / total_energy

            return final_err

    def _select_best_probe(self) -> np.ndarray:
            """
            Select the best probe sigma from candidates using quick reconstruction.

            Returns:
                Best probe array, shape (probe_size, probe_size), complex128.
            """
            best_sigma = self.sigma_candidates[len(self.sigma_candidates) // 2]  # default middle
            best_err = np.inf

            for sigma in self.sigma_candidates:
                P_test = self._make_probe(sigma)
                err = self._quick_reconstruction(P_test, n_iter=self.n_iter_quick)
                if err < best_err:
                    best_err = err
                    best_sigma = sigma

            self.best_sigma = best_sigma
            return self._make_probe(best_sigma)

    def _run_main_reconstruction(
            self, P_init: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Run the main rPIE reconstruction with early stopping.

            Args:
                P_init: Initial probe estimate, shape (probe_size, probe_size).

            Returns:
                Tuple of (best_object, best_probe) arrays.
            """
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            P = P_init.copy()
            n_positions = len(self.positions)

            total_energy = float(np.sum(self.amplitudes ** 2))

            best_O = O.copy()
            best_P = P.copy()
            best_fourier_error = np.inf
            best_epoch = 0

            self.error_history = []

            for epoch in range(self.n_iter_main):
                order = np.random.permutation(n_positions)
                epoch_err = 0.0

                for idx in range(n_positions):
                    j = int(order[idx])
                    update_probe = (epoch >= self.probe_update_start)
                    O, P, pos_err = self._rpie_update_step(O, P, j, update_probe)
                    epoch_err += pos_err

                rel_err = epoch_err / total_energy
                self.error_history.append(rel_err)

                if rel_err < best_fourier_error:
                    best_fourier_error = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                if epoch - best_epoch > self.patience:
                    break

            return best_O, best_P

    def _tv_denoise_2d(
            self, img: np.ndarray, weight: float, n_iter: int
        ) -> np.ndarray:
            """
            Chambolle total variation denoising for a 2D real-valued image.

            Args:
                img: Input 2D real-valued image, shape (H, W).
                weight: TV regularization weight.
                n_iter: Number of Chambolle iterations.

            Returns:
                Denoised image, shape (H, W).
            """
            px = np.zeros_like(img)
            py = np.zeros_like(img)
            tau = self.tv_tau

            for _ in range(n_iter):
                # Compute divergence of (px, py)
                div_p = np.zeros_like(img)
                # d/dx: div_p += d(px)/dx  (backward difference)
                div_p[1:, :] += px[:-1, :]
                div_p[0, :] += 0.0  # boundary: px[-1] = 0
                div_p[:-1, :] -= px[:-1, :]
                # d/dy: div_p += d(py)/dy  (backward difference)
                div_p[:, 1:] += py[:, :-1]
                div_p[:, 0] += 0.0  # boundary: py[:,-1] = 0
                div_p[:, :-1] -= py[:, :-1]

                # Compute u = img + weight * div_p
                u = img + weight * div_p

                # Compute gradient of u (forward differences)
                gx = np.zeros_like(img)
                gy = np.zeros_like(img)
                gx[:-1, :] = u[1:, :] - u[:-1, :]
                gy[:, :-1] = u[:, 1:] - u[:, :-1]

                # Update dual variables
                norm_g = np.sqrt(gx ** 2 + gy ** 2)
                denom = 1.0 + tau * norm_g
                px = (px + tau * gx) / denom
                py = (py + tau * gy) / denom

            # Final primal solution
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
            Apply TV denoising separately to magnitude and phase (via cos/sin).

            Args:
                O: Complex-valued object, shape (obj_size, obj_size).

            Returns:
                Tuple of (denoised_magnitude, denoised_phase), each (obj_size, obj_size).
            """
            mag_raw = np.abs(O)
            phase_raw = np.angle(O)

            # Denoise magnitude directly
            mag_out = self._tv_denoise_2d(mag_raw, weight=self.tv_weight, n_iter=self.tv_iterations)

            # Denoise cos(phase) and sin(phase) separately, recombine with arctan2
            cos_tv = self._tv_denoise_2d(np.cos(phase_raw), weight=self.tv_weight, n_iter=self.tv_iterations)
            sin_tv = self._tv_denoise_2d(np.sin(phase_raw), weight=self.tv_weight, n_iter=self.tv_iterations)
            phase_out = np.arctan2(sin_tv, cos_tv)

            return mag_out, phase_out

    def _compute_illumination_mask(self) -> np.ndarray:
            """
            Compute a boolean mask of well-illuminated pixels based on
            scan position overlap.

            Returns:
                Boolean mask, shape (obj_size, obj_size).
            """
            illum_count = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            for j in range(len(self.positions)):
                ry, rx = self.positions[j]
                illum_count[ry:ry + self.probe_size, rx:rx + self.probe_size] += 1.0

            nonzero_illum = illum_count[illum_count > 0]
            if len(nonzero_illum) > 0:
                illum_thresh = np.percentile(nonzero_illum, self.illum_percentile)
            else:
                illum_thresh = 1.0

            well_lit = illum_count >= illum_thresh
            return well_lit

    def _reference_phase(
            self, mag: np.ndarray, phase: np.ndarray, well_lit: np.ndarray
        ) -> np.ndarray:
            """
            Reference the phase by subtracting the dominant (substrate) phase
            and ensuring pad phase is positive (ΔΦ = 4πh/λ > 0).

            Args:
                mag: Object magnitude, shape (obj_size, obj_size).
                phase: Object phase in radians, shape (obj_size, obj_size).
                well_lit: Boolean mask of well-illuminated pixels.

            Returns:
                Referenced phase, shape (obj_size, obj_size).
            """
            phase_out = phase.copy()

            # 1. Histogram of phase in well-lit region to find substrate phase
            phase_well = phase_out[well_lit]
            hist_phase, bin_edges = np.histogram(
                phase_well, bins=self.phase_hist_bins, range=(-np.pi, np.pi)
            )
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            dominant_idx = np.argmax(hist_phase)
            lo = max(0, dominant_idx - 3)
            hi = min(len(hist_phase), dominant_idx + 4)
            weights = hist_phase[lo:hi].astype(np.float64)
            if np.sum(weights) > 0:
                substrate_phase = np.average(bin_centers[lo:hi], weights=weights)
            else:
                substrate_phase = bin_centers[dominant_idx]

            # 2. Subtract substrate phase, wrap to [-pi, pi]
            phase_out = phase_out - substrate_phase
            phase_out = np.arctan2(np.sin(phase_out), np.cos(phase_out))

            # 3. Otsu threshold on magnitude to separate pads from substrate
            mag_well = mag[well_lit]
            threshold_mag = self._otsu_threshold(mag_well, self.mag_hist_bins)

            class1 = well_lit & (mag <= threshold_mag)
            class2 = well_lit & (mag > threshold_mag)

            n1, n2 = np.sum(class1), np.sum(class2)
            if n1 >= n2:
                pad_mask = class2
                sub_mask = class1
            else:
                pad_mask = class1
                sub_mask = class2

            # 4. Check sign of pad phase relative to substrate; flip if negative
            if np.sum(pad_mask) > 50:
                mean_pad_phase = np.mean(phase_out[pad_mask])
                mean_sub_phase = np.mean(phase_out[sub_mask])
                pad_phase_diff = mean_pad_phase - mean_sub_phase
                if pad_phase_diff < -0.1:
                    phase_out = -phase_out

            return phase_out

    def _otsu_threshold(
            self, values: np.ndarray, n_bins: int
        ) -> float:
            """
            Compute Otsu's threshold for a 1D array of values.

            Args:
                values: 1D array of values to threshold.
                n_bins: Number of histogram bins.

            Returns:
                Optimal threshold value (float).
            """
            hist, bin_edges = np.histogram(values, bins=n_bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            total = np.sum(hist)
            if total == 0:
                return float(np.median(values))

            sum_total = np.sum(hist * bin_centers)
            sum_bg = 0.0
            w_bg = 0
            max_var = 0.0
            threshold = float(np.median(values))

            for i in range(len(hist)):
                w_bg += hist[i]
                if w_bg == 0:
                    continue
                w_fg = total - w_bg
                if w_fg == 0:
                    break
                sum_bg += hist[i] * bin_centers[i]
                mean_bg = sum_bg / w_bg
                mean_fg = (sum_total - sum_bg) / w_fg
                var_between = float(w_bg) * float(w_fg) * (mean_bg - mean_fg) ** 2
                if var_between > max_var:
                    max_var = var_between
                    threshold = float(bin_centers[i])

            return threshold

    def solve(self, data: np.ndarray) -> np.ndarray:
            """
            Full CDI phase retrieval pipeline.

            Args:
                data: Input diffraction intensity patterns, shape (N, 256, 256),
                      real-valued, non-negative.

            Returns:
                Reconstructed object as array of shape (2, obj_size, obj_size),
                where channel 0 is magnitude and channel 1 is phase (radians).
            """
            # 1. Compute amplitudes = sqrt(max(data, 0))
            self.amplitudes = np.sqrt(np.maximum(data, 0.0))

            # 2. Detect DC location (fftshift or not)
            self.use_fftshift = self._detect_dc_location(data)

            # 3. Compute scan positions
            self.positions = self._compute_scan_positions()

            # 4. Compute mean total intensity for probe scaling
            self.mean_total_intensity = float(np.mean(np.sum(data, axis=(1, 2))))

            # 5. Select best probe via quick reconstruction
            P_init = self._select_best_probe()

            # 6. Run main rPIE reconstruction
            best_O, best_P = self._run_main_reconstruction(P_init)

            # 7. Apply TV denoising
            mag_out, phase_out = self._apply_tv_denoising(best_O)

            # 8. Reference phase
            well_lit = self._compute_illumination_mask()
            phase_out = self._reference_phase(mag_out, phase_out, well_lit)

            # 9. Assemble output array (2, obj_size, obj_size)
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            output[0] = mag_out
            output[1] = phase_out

            return output



if __name__ == "__main__":
    # Load data
    input_data: np.ndarray = np.load("dataset/input.npy")

    # Create solver and run
    solver: InverseSolver = InverseSolver()
    result: np.ndarray = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)