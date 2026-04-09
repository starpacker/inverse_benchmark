import numpy as np
from typing import Tuple, List, Optional

class InverseSolver:
    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_scan: int = 9,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            pad_cd_pixels: int = 22,
            n_pads: int = 15,
            duty_cycle: float = 0.6,
            pitch_nm: float = 800.0,
            cd_nm: float = 600.0,
            recession_range: Tuple[float, float] = (1.0, 3.0),
            max_phase_modulation: float = 2.8,
            alpha_rpie: float = 0.3,
            momentum: float = 0.6,
            n_iter_main: int = 500,
            n_iter_quick_test: int = 15,
            n_iter_epie_fallback: int = 300,
            probe_update_start: int = 10,
            early_stop_patience: int = 200,
            sigma_candidates: Optional[List[int]] = None,
            tv_weight: float = 0.01,
            tv_n_iter: int = 50,
            tv_heavy_weight: float = 0.05,
            tv_heavy_n_iter: int = 100,
            consistency_threshold: float = 0.5,
            fallback_error_threshold: float = 0.3,
            blend_recon_weight: float = 0.7,
            blend_mean_weight: float = 0.3,
            reflectivity_clamp: float = 1.0,
            obj_init_value: float = 0.5,
            target_phase_midpoint: float = 1.86,
            substrate_phase_min_pixels: int = 100,
            well_lit_percentile: float = 75.0,
            histogram_bins: int = 50,
        ) -> None:
            # Grid and geometry
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_scan: int = n_scan
            self.n_positions: int = n_scan * n_scan

            # Physical parameters
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.pad_cd_pixels: int = pad_cd_pixels
            self.n_pads: int = n_pads
            self.duty_cycle: float = duty_cycle
            self.pitch_nm: float = pitch_nm
            self.cd_nm: float = cd_nm
            self.recession_range: Tuple[float, float] = recession_range
            self.max_phase_modulation: float = max_phase_modulation

            # rPIE hyperparameters
            self.alpha_rpie: float = alpha_rpie
            self.momentum: float = momentum
            self.n_iter_main: int = n_iter_main
            self.n_iter_quick_test: int = n_iter_quick_test
            self.n_iter_epie_fallback: int = n_iter_epie_fallback
            self.probe_update_start: int = probe_update_start
            self.early_stop_patience: int = early_stop_patience

            # Probe search
            self.sigma_candidates: List[int] = sigma_candidates if sigma_candidates is not None else [15, 20, 25, 30, 40, 50, 60, 80]

            # TV denoising
            self.tv_weight: float = tv_weight
            self.tv_n_iter: int = tv_n_iter
            self.tv_heavy_weight: float = tv_heavy_weight
            self.tv_heavy_n_iter: int = tv_heavy_n_iter

            # Consistency and fallback
            self.consistency_threshold: float = consistency_threshold
            self.fallback_error_threshold: float = fallback_error_threshold
            self.blend_recon_weight: float = blend_recon_weight
            self.blend_mean_weight: float = blend_mean_weight

            # Physical constraints
            self.reflectivity_clamp: float = reflectivity_clamp
            self.obj_init_value: float = obj_init_value

            # Phase referencing
            self.target_phase_midpoint: float = target_phase_midpoint
            self.substrate_phase_min_pixels: int = substrate_phase_min_pixels
            self.well_lit_percentile: float = well_lit_percentile
            self.histogram_bins: int = histogram_bins

            # Runtime state (set during solve)
            self.use_shift: bool = True
            self.scan_step: float = (obj_size - probe_size) / (n_scan - 1)
            self.positions: List[Tuple[int, int]] = []
            self.mean_total_intensity: float = 0.0
            self.amplitudes: Optional[np.ndarray] = None
            self.probe: Optional[np.ndarray] = None
            self.object_estimate: Optional[np.ndarray] = None
            self.best_object: Optional[np.ndarray] = None
            self.best_probe: Optional[np.ndarray] = None
            self.best_fourier_error: float = np.inf
            self.errors: List[float] = []

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            positions: List[Tuple[int, int]],
        ) -> np.ndarray:
            """Compute forward diffraction patterns for all scan positions.

            Parameters
            ----------
            obj : np.ndarray
                Complex-valued object, shape (obj_size, obj_size).
            probe : np.ndarray
                Complex-valued probe, shape (probe_size, probe_size).
            positions : List[Tuple[int, int]]
                List of (row, col) top-left positions for each scan point.

            Returns
            -------
            np.ndarray
                Diffraction intensities, shape (n_positions, probe_size, probe_size).
            """
            n_pos = len(positions)
            intensities = np.zeros((n_pos, self.probe_size, self.probe_size), dtype=np.float64)
            for j, (ry, rx) in enumerate(positions):
                obj_patch = obj[ry:ry + self.probe_size, rx:rx + self.probe_size]
                exit_wave = probe * obj_patch
                far_field = self._forward_prop(exit_wave)
                intensities[j] = np.abs(far_field) ** 2
            return intensities

    def _forward_prop(self, x: np.ndarray) -> np.ndarray:
            """Propagate from real space to Fourier space (far-field).

            Parameters
            ----------
            x : np.ndarray
                Complex-valued 2D field, shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Complex-valued Fourier-space field, shape (probe_size, probe_size).
            """
            if self.use_shift:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            else:
                return np.fft.fft2(x)

    def _backward_prop(self, X: np.ndarray) -> np.ndarray:
            """Propagate from Fourier space back to real space.

            Parameters
            ----------
            X : np.ndarray
                Complex-valued Fourier-space field, shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Complex-valued real-space field, shape (probe_size, probe_size).
            """
            if self.use_shift:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))
            else:
                return np.fft.ifft2(X)

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
            """Compute the (row, col) scan positions for the ptychographic grid.

            Returns
            -------
            List[Tuple[int, int]]
                List of 81 (row, col) positions.
            """
            positions = []
            for iy in range(self.n_scan):
                for ix in range(self.n_scan):
                    ry = int(round(iy * self.scan_step))
                    rx = int(round(ix * self.scan_step))
                    positions.append((ry, rx))
            return positions

    def _detect_dc_location(self, data: np.ndarray) -> bool:
            """Determine whether DC component is centered or at corner.

            Parameters
            ----------
            data : np.ndarray
                Input diffraction intensities, shape (81, 256, 256).

            Returns
            -------
            bool
                True if DC is at center (use fftshift), False otherwise.
            """
            avg_pattern = np.mean(data, axis=0)
            h, w = avg_pattern.shape
            center_y, center_x = h // 2, w // 2

            # Compute mean intensity in a small region around the center
            r = 5
            center_val = np.mean(avg_pattern[center_y - r:center_y + r, center_x - r:center_x + r])

            # Compute mean intensity in corner regions
            corner_val = np.mean([
                np.mean(avg_pattern[:2 * r, :2 * r]),
                np.mean(avg_pattern[:2 * r, -2 * r:]),
                np.mean(avg_pattern[-2 * r:, :2 * r]),
                np.mean(avg_pattern[-2 * r:, -2 * r:]),
            ])

            return center_val > corner_val * 2

    def _init_probe(self, sigma: float) -> np.ndarray:
            """Initialize a Gaussian probe with Parseval-consistent scaling.

            Parameters
            ----------
            sigma : float
                Standard deviation of the Gaussian probe in pixels.

            Returns
            -------
            np.ndarray
                Complex-valued probe, shape (probe_size, probe_size).
            """
            half = self.probe_size // 2
            Y, X = np.mgrid[-half:half, -half:half]
            R2 = (X.astype(np.float64) ** 2 + Y.astype(np.float64) ** 2)
            P = np.exp(-R2 / (2.0 * sigma ** 2)).astype(np.complex128)

            # Parseval scaling: match probe energy to measured intensity
            # If O ~ 1, then sum(|F{P * O_patch}|^2) = N_pix * sum(|P|^2)
            # So sum(|P|^2) ≈ mean_total_intensity / N_pix
            N_pix = self.probe_size ** 2
            current_energy = np.sum(np.abs(P) ** 2)
            target_energy = self.mean_total_intensity / N_pix
            if current_energy > 0 and target_energy > 0:
                P *= np.sqrt(target_energy / current_energy)

            return P

    def _quick_test(self, probe: np.ndarray, n_iter: int) -> float:
            """Run a short rPIE reconstruction to evaluate probe quality.

            Parameters
            ----------
            probe : np.ndarray
                Complex-valued probe candidate, shape (probe_size, probe_size).
            n_iter : int
                Number of quick-test iterations.

            Returns
            -------
            float
                Relative Fourier error after n_iter iterations.
            """
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128) * self.obj_init_value
            P = probe.copy()
            alpha = self.alpha_rpie

            err = 0.0
            for ep in range(n_iter):
                order = np.random.permutation(self.n_positions)
                err = 0.0
                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]

                    psi = P * O_patch
                    Psi = self._forward_prop(psi)

                    mag = np.abs(Psi)
                    Psi_c = self.amplitudes[j] * Psi / (mag + 1e-12)
                    err += np.sum((mag - self.amplitudes[j]) ** 2)

                    psi_c = self._backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # rPIE object update
                    P_abs2 = np.abs(P) ** 2
                    P_max = np.max(P_abs2)
                    denom = (1.0 - alpha) * P_abs2 + alpha * P_max
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] += (
                        np.conj(P) * dpsi / (denom + 1e-12)
                    )

                    # Reflectivity constraint
                    patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
                    mag_patch = np.abs(patch)
                    over = mag_patch > self.reflectivity_clamp
                    if np.any(over):
                        patch[over] *= self.reflectivity_clamp / mag_patch[over]
                        O[ry:ry + self.probe_size, rx:rx + self.probe_size] = patch

            total_energy = np.sum(self.amplitudes ** 2)
            return err / total_energy if total_energy > 0 else np.inf

    def _select_best_probe(self) -> np.ndarray:
            """Search over sigma_candidates and return the best-performing probe.

            Returns
            -------
            np.ndarray
                Complex-valued probe with lowest quick-test error.
            """
            best_sigma = self.sigma_candidates[0]
            best_err = np.inf

            for sigma in self.sigma_candidates:
                P_test = self._init_probe(sigma)
                err = self._quick_test(P_test, self.n_iter_quick_test)
                if err < best_err:
                    best_err = err
                    best_sigma = sigma

            return self._init_probe(best_sigma)

    def _clamp_reflectivity(self, obj_patch: np.ndarray) -> np.ndarray:
            """Enforce physical reflectivity constraint |O(r)| <= reflectivity_clamp.

            Parameters
            ----------
            obj_patch : np.ndarray
                Complex-valued object patch, shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Clamped object patch.
            """
            mag = np.abs(obj_patch)
            over = mag > self.reflectivity_clamp
            if np.any(over):
                obj_patch[over] *= self.reflectivity_clamp / mag[over]
            return obj_patch

    def _rpie_epoch(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            epoch: int,
        ) -> Tuple[np.ndarray, np.ndarray, float]:
            """Run one epoch of the rPIE algorithm over all scan positions.

            Parameters
            ----------
            obj : np.ndarray
                Current object estimate, shape (obj_size, obj_size), complex128.
            probe : np.ndarray
                Current probe estimate, shape (probe_size, probe_size), complex128.
            epoch : int
                Current epoch index (controls probe update activation).

            Returns
            -------
            Tuple[np.ndarray, np.ndarray, float]
                Updated object, updated probe, epoch relative Fourier error.
            """
            order = np.random.permutation(self.n_positions)
            epoch_err = 0.0
            alpha = self.alpha_rpie

            for idx in range(self.n_positions):
                j = order[idx]
                ry, rx = self.positions[j]

                # Extract current object patch
                O_patch = obj[ry:ry + self.probe_size, rx:rx + self.probe_size].copy()

                # Form exit wave and propagate to far field
                psi = probe * O_patch
                Psi = self._forward_prop(psi)

                # Modulus replacement
                mag = np.abs(Psi)
                Psi_c = self.amplitudes[j] * Psi / (mag + 1e-12)
                epoch_err += np.sum((mag - self.amplitudes[j]) ** 2)

                # Back-propagate corrected wave
                psi_c = self._backward_prop(Psi_c)
                dpsi = psi_c - psi

                # rPIE object update
                P_abs2 = np.abs(probe) ** 2
                P_max = np.max(P_abs2)
                denom_O = (1.0 - alpha) * P_abs2 + alpha * P_max
                obj[ry:ry + self.probe_size, rx:rx + self.probe_size] += (
                    np.conj(probe) * dpsi / (denom_O + 1e-12)
                )

                # Reflectivity constraint: |O| <= reflectivity_clamp
                obj[ry:ry + self.probe_size, rx:rx + self.probe_size] = (
                    self._clamp_reflectivity(
                        obj[ry:ry + self.probe_size, rx:rx + self.probe_size]
                    )
                )

                # Probe update (after warmup epochs)
                if epoch >= self.probe_update_start:
                    O_abs2 = np.abs(O_patch) ** 2
                    O_max = np.max(O_abs2)
                    denom_P = (1.0 - alpha) * O_abs2 + alpha * O_max
                    probe += np.conj(O_patch) * dpsi / (denom_P + 1e-12)

            # Compute relative Fourier error
            total_energy = np.sum(self.amplitudes ** 2)
            rel_err = epoch_err / total_energy if total_energy > 0 else np.inf

            return obj, probe, rel_err

    def _apply_nesterov_momentum(
            self,
            obj_current: np.ndarray,
            obj_prev: np.ndarray,
            momentum: float,
        ) -> np.ndarray:
            """Apply Nesterov momentum extrapolation to the object estimate.

            Parameters
            ----------
            obj_current : np.ndarray
                Current object, shape (obj_size, obj_size), complex128.
            obj_prev : np.ndarray
                Previous object, shape (obj_size, obj_size), complex128.
            momentum : float
                Momentum coefficient.

            Returns
            -------
            np.ndarray
                Extrapolated object estimate.
            """
            O_extrap = obj_current + momentum * (obj_current - obj_prev)
            # Clamp extrapolated object to enforce reflectivity constraint |O| <= reflectivity_clamp
            mag_ext = np.abs(O_extrap)
            over = mag_ext > self.reflectivity_clamp
            if np.any(over):
                O_extrap[over] *= self.reflectivity_clamp / mag_ext[over]
            return O_extrap

    def _run_main_reconstruction(self) -> Tuple[np.ndarray, np.ndarray, float]:
            """Run the full momentum-accelerated rPIE reconstruction loop.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray, float]
                Best object, best probe, best Fourier error.
            """
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128) * self.obj_init_value
            O_prev = O.copy()
            P = self.probe.copy()

            best_O = O.copy()
            best_P = P.copy()
            best_fourier_error = np.inf
            best_epoch = 0

            errors = []
            current_momentum = self.momentum
            original_use_shift = self.use_shift
            fft_fallback_done = False

            for epoch in range(self.n_iter_main):
                # Nesterov momentum: evaluate at extrapolated point
                if epoch > 1 and current_momentum > 0:
                    O_extrap = self._apply_nesterov_momentum(O, O_prev, current_momentum)
                    O_prev = O.copy()
                    O = O_extrap
                else:
                    O_prev = O.copy()

                # Run one rPIE epoch
                O, P, rel_err = self._rpie_epoch(O, P, epoch)
                errors.append(rel_err)

                # Best state tracking
                if rel_err < best_fourier_error:
                    best_fourier_error = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                # FFT convention fallback at epoch 25
                if epoch == 25 and not fft_fallback_done:
                    if len(errors) < 2 or errors[-1] > errors[0] * 0.85:
                        # Flip FFT convention and restart
                        self.use_shift = not self.use_shift
                        fft_fallback_done = True
                        O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128) * self.obj_init_value
                        O_prev = O.copy()
                        P = self.probe.copy()
                        best_fourier_error = np.inf
                        best_O = O.copy()
                        best_P = P.copy()
                        best_epoch = epoch
                        errors = []
                        current_momentum = self.momentum
                        continue

                # Reduce momentum if oscillating (error increasing for 3 consecutive epochs)
                if epoch > 5 and len(errors) >= 3:
                    if errors[-1] > errors[-2] > errors[-3]:
                        current_momentum = max(0.0, current_momentum - 0.1)

                # Early stopping if no improvement for patience epochs
                if epoch - best_epoch > self.early_stop_patience:
                    break

            self.best_fourier_error = best_fourier_error
            self.errors = errors
            return best_O, best_P, best_fourier_error

    def _run_epie_fallback(self) -> Tuple[np.ndarray, np.ndarray, float]:
            """Run ePIE reconstruction as fallback if rPIE fails.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray, float]
                ePIE object, probe, and consistency error.
            """
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128) * self.obj_init_value
            P = self.probe.copy()

            best_O = O.copy()
            best_P = P.copy()
            best_err = np.inf
            best_epoch = 0

            for epoch in range(self.n_iter_epie_fallback):
                order = np.random.permutation(self.n_positions)
                epoch_err = 0.0

                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]

                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size].copy()

                    # Form exit wave and propagate
                    psi = P * O_patch
                    Psi = self._forward_prop(psi)

                    # Modulus replacement
                    mag = np.abs(Psi)
                    Psi_c = self.amplitudes[j] * Psi / (mag + 1e-12)
                    epoch_err += np.sum((mag - self.amplitudes[j]) ** 2)

                    # Back-propagate corrected wave
                    psi_c = self._backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # ePIE object update: standard form with P_max denominator
                    P_max_sq = np.max(np.abs(P) ** 2)
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] += (
                        np.conj(P) * dpsi / (P_max_sq + 1e-12)
                    )

                    # Reflectivity constraint: |O| <= reflectivity_clamp
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] = (
                        self._clamp_reflectivity(
                            O[ry:ry + self.probe_size, rx:rx + self.probe_size]
                        )
                    )

                    # Probe update after warmup
                    if epoch >= self.probe_update_start:
                        O_max_sq = np.max(np.abs(O_patch) ** 2)
                        P += np.conj(O_patch) * dpsi / (O_max_sq + 1e-12)

                # Compute relative Fourier error
                total_energy = np.sum(self.amplitudes ** 2)
                rel_err = epoch_err / total_energy if total_energy > 0 else np.inf

                # Track best state
                if rel_err < best_err:
                    best_err = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                # Early stopping
                if epoch - best_epoch > self.early_stop_patience:
                    break

            return best_O, best_P, best_err

    def _tv_denoise_2d(
            self,
            img: np.ndarray,
            weight: float,
            n_iter: int,
        ) -> np.ndarray:
            """Chambolle total-variation denoising on a real-valued 2D image.

            Parameters
            ----------
            img : np.ndarray
                Real-valued 2D image, shape (H, W).
            weight : float
                TV regularization weight.
            n_iter : int
                Number of Chambolle iterations.

            Returns
            -------
            np.ndarray
                Denoised image, shape (H, W).
            """
            img = img.astype(np.float64)
            H, W = img.shape
            px = np.zeros((H, W), dtype=np.float64)
            py = np.zeros((H, W), dtype=np.float64)
            tau = 0.125  # Step size for Chambolle's algorithm (1/4 for 2D)

            for _ in range(n_iter):
                # Compute divergence of (px, py)
                # div_p[i,j] = px[i,j] - px[i-1,j] + py[i,j] - py[i,j-1]
                div_p = np.zeros((H, W), dtype=np.float64)
                # x-component divergence
                div_p[0, :] = px[0, :]
                div_p[1:, :] = px[1:, :] - px[:-1, :]
                # y-component divergence
                div_p[:, 0] += py[:, 0]
                div_p[:, 1:] += py[:, 1:] - py[:, :-1]

                # Compute gradient of (img + weight * div_p)
                u = img + weight * div_p
                # Forward differences with Neumann boundary (zero at boundary)
                gx = np.zeros((H, W), dtype=np.float64)
                gy = np.zeros((H, W), dtype=np.float64)
                gx[:-1, :] = u[1:, :] - u[:-1, :]  # last row stays zero
                gy[:, :-1] = u[:, 1:] - u[:, :-1]  # last col stays zero

                # Update dual variables
                norm_g = np.sqrt(gx ** 2 + gy ** 2)
                denom = 1.0 + tau * norm_g
                px = (px + tau * gx) / denom
                py = (py + tau * gy) / denom

            # Final primal reconstruction
            div_p = np.zeros((H, W), dtype=np.float64)
            div_p[0, :] = px[0, :]
            div_p[1:, :] = px[1:, :] - px[:-1, :]
            div_p[:, 0] += py[:, 0]
            div_p[:, 1:] += py[:, 1:] - py[:, :-1]

            return img + weight * div_p

    def _tv_denoise_complex(
            self,
            obj: np.ndarray,
            weight: float,
            n_iter: int,
        ) -> np.ndarray:
            """Apply TV denoising to a complex-valued object (mag and phase separately).

            Parameters
            ----------
            obj : np.ndarray
                Complex-valued 2D object, shape (obj_size, obj_size).
            weight : float
                TV regularization weight.
            n_iter : int
                Number of Chambolle iterations per channel.

            Returns
            -------
            np.ndarray
                Denoised complex object, shape (obj_size, obj_size).
            """
            mag = np.abs(obj)
            phase = np.angle(obj)

            # Denoise magnitude directly
            mag_denoised = self._tv_denoise_2d(mag, weight, n_iter)

            # Denoise phase via cos/sin decomposition to handle wrapping
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            cos_denoised = self._tv_denoise_2d(cos_phase, weight, n_iter)
            sin_denoised = self._tv_denoise_2d(sin_phase, weight, n_iter)
            phase_denoised = np.arctan2(sin_denoised, cos_denoised)

            # Reconstruct complex object from denoised magnitude and phase
            return mag_denoised * np.exp(1j * phase_denoised)

    def _compute_illumination_maps(
            self,
            probe: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Compute illumination count and weight maps over the object grid.

            Parameters
            ----------
            probe : np.ndarray
                Complex-valued probe, shape (probe_size, probe_size).

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                Illumination count map and intensity weight map, each (obj_size, obj_size).
            """
            illum_count = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            illum_weight = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            probe_intensity = np.abs(probe) ** 2

            for j in range(self.n_positions):
                ry, rx = self.positions[j]
                illum_count[ry:ry + self.probe_size, rx:rx + self.probe_size] += 1.0
                illum_weight[ry:ry + self.probe_size, rx:rx + self.probe_size] += probe_intensity

            return illum_count, illum_weight

    def _otsu_threshold(
            self,
            values: np.ndarray,
            n_bins: int,
        ) -> float:
            """Compute Otsu threshold for a 1D array of values.

            Parameters
            ----------
            values : np.ndarray
                1D array of values to threshold.
            n_bins : int
                Number of histogram bins.

            Returns
            -------
            float
                Otsu threshold value.
            """
            hist, bin_edges = np.histogram(values, bins=n_bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            total = np.sum(hist)
            if total == 0:
                return np.median(values)

            sum_total = np.sum(hist * bin_centers)
            sum_bg = 0.0
            w_bg = 0.0
            max_var = 0.0
            threshold = np.median(values)

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
                var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
                if var_between > max_var:
                    max_var = var_between
                    threshold = bin_centers[i]

            return float(threshold)

    def _identify_substrate_and_pad_masks(
            self,
            mag: np.ndarray,
            well_lit_mask: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Segment the object into substrate and pad regions using Otsu thresholding.

            Parameters
            ----------
            mag : np.ndarray
                Object magnitude, shape (obj_size, obj_size).
            well_lit_mask : np.ndarray
                Boolean mask of well-illuminated pixels.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                substrate_mask, pad_mask — both boolean, shape (obj_size, obj_size).
            """
            # Extract magnitude values in well-illuminated region
            mag_well = mag[well_lit_mask]

            # Compute Otsu threshold on the well-lit magnitude values
            threshold = self._otsu_threshold(mag_well, self.histogram_bins)

            # Create masks: pixels below threshold vs above threshold
            substrate_mask = well_lit_mask & (mag < threshold)
            pad_mask = well_lit_mask & (mag >= threshold)

            n_sub = np.sum(substrate_mask)
            n_pad = np.sum(pad_mask)

            # Duty cycle 0.6 means pads occupy ~36% of area, substrate ~64%
            # The larger class should be substrate
            if n_sub < n_pad:
                substrate_mask, pad_mask = pad_mask, substrate_mask

            return substrate_mask, pad_mask

    def _reference_phase(
            self,
            phase: np.ndarray,
            mag: np.ndarray,
            probe: np.ndarray,
        ) -> np.ndarray:
            """Remove global phase ambiguity using substrate as reference.

            Parameters
            ----------
            phase : np.ndarray
                Raw phase of reconstructed object, shape (obj_size, obj_size).
            mag : np.ndarray
                Magnitude of reconstructed object, shape (obj_size, obj_size).
            probe : np.ndarray
                Reconstructed probe, shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Referenced phase, shape (obj_size, obj_size), in [-pi, pi].
            """
            # Compute illumination maps
            illum_count, illum_weight = self._compute_illumination_maps(probe)

            # Determine well-illuminated region
            nonzero_counts = illum_count[illum_count > 0]
            if len(nonzero_counts) == 0:
                # No illumination info; just return phase as-is centered on median
                phase_ref = np.median(phase)
                phase_out = phase - phase_ref
                return np.arctan2(np.sin(phase_out), np.cos(phase_out))

            threshold_count = np.percentile(nonzero_counts, self.well_lit_percentile)
            well_lit_mask = illum_count >= threshold_count

            # Segment into substrate and pad using Otsu on magnitude
            substrate_mask, pad_mask = self._identify_substrate_and_pad_masks(mag, well_lit_mask)

            # Compute phase reference from substrate median
            if np.sum(substrate_mask) >= self.substrate_phase_min_pixels:
                phase_ref = np.median(phase[substrate_mask])
            else:
                phase_ref = np.median(phase[well_lit_mask])

            # Subtract reference phase
            phase_out = phase - phase_ref
            # Wrap to [-pi, pi]
            phase_out = np.arctan2(np.sin(phase_out), np.cos(phase_out))

            # Validate: pad phase should be positive and near target_phase_midpoint
            if np.sum(pad_mask) >= self.substrate_phase_min_pixels:
                mean_pad_phase = np.mean(phase_out[pad_mask])

                # Option 1: as-is
                opt1 = phase_out.copy()
                err1 = abs(mean_pad_phase - self.target_phase_midpoint)

                # Option 2: add pi
                phase_out2 = phase_out + np.pi
                phase_out2 = np.arctan2(np.sin(phase_out2), np.cos(phase_out2))
                mean_pad_phase2 = np.mean(phase_out2[pad_mask])
                err2 = abs(mean_pad_phase2 - self.target_phase_midpoint)

                # Option 3: negate
                phase_out3 = -phase_out
                mean_pad_phase3 = np.mean(phase_out3[pad_mask])
                err3 = abs(mean_pad_phase3 - self.target_phase_midpoint)

                # Pick the option closest to expected physical range
                options = [
                    (opt1, err1),
                    (phase_out2, err2),
                    (phase_out3, err3),
                ]
                phase_out = min(options, key=lambda x: x[1])[0]

            return phase_out

    def _format_output(
            self,
            obj_denoised: np.ndarray,
            probe: np.ndarray,
        ) -> np.ndarray:
            """Format the reconstructed object into the required output array.

            Parameters
            ----------
            obj_denoised : np.ndarray
                Denoised complex object, shape (obj_size, obj_size).
            probe : np.ndarray
                Reconstructed probe, shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Output array, shape (2, obj_size, obj_size), channel 0 = magnitude,
                channel 1 = phase.
            """
            # Extract magnitude and clip to [0, 1] (already enforced by reflectivity clamping)
            mag = np.abs(obj_denoised)
            mag_out = np.clip(mag, 0.0, 1.0)

            # Extract raw phase
            phase_raw = np.angle(obj_denoised)

            # Reference the phase using substrate identification
            phase_out = self._reference_phase(phase_raw, mag, probe)

            # Assemble 2-channel output
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            output[0] = mag_out.astype(np.float64)
            output[1] = phase_out.astype(np.float64)

            return output

    def _compute_consistency(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
        ) -> float:
            """Compute forward-model consistency (relative Fourier error).

            Parameters
            ----------
            obj : np.ndarray
                Complex-valued object, shape (obj_size, obj_size).
            probe : np.ndarray
                Complex-valued probe, shape (probe_size, probe_size).

            Returns
            -------
            float
                Relative error: sum(|predicted_amp - measured_amp|^2) / sum(measured_amp^2).
            """
            total_err = 0.0
            total_energy = 0.0
            for j in range(self.n_positions):
                ry, rx = self.positions[j]
                obj_patch = obj[ry:ry + self.probe_size, rx:rx + self.probe_size]
                psi = probe * obj_patch
                Psi = self._forward_prop(psi)
                pred_amp = np.abs(Psi)
                total_err += np.sum((pred_amp - self.amplitudes[j]) ** 2)
                total_energy += np.sum(self.amplitudes[j] ** 2)
            return total_err / total_energy if total_energy > 0 else np.inf

    def _apply_safety_net(
            self,
            output: np.ndarray,
            obj: np.ndarray,
            probe: np.ndarray,
            phase_ref: float,
            consistency: float,
        ) -> np.ndarray:
            """Apply safety-net blending if reconstruction consistency is poor.

            Parameters
            ----------
            output : np.ndarray
                Current output array, shape (2, obj_size, obj_size).
            obj : np.ndarray
                Non-denoised complex object, shape (obj_size, obj_size).
            probe : np.ndarray
                Reconstructed probe, shape (probe_size, probe_size).
            phase_ref : float
                Phase reference value used for the main output.
            consistency : float
                Forward-model consistency metric.

            Returns
            -------
            np.ndarray
                Possibly blended output, shape (2, obj_size, obj_size).
            """
            if consistency > self.consistency_threshold:
                # Apply heavier TV denoising to the raw (non-denoised) object
                O_heavy = self._tv_denoise_complex(obj, self.tv_heavy_weight, self.tv_heavy_n_iter)

                # Extract magnitude and phase from heavily denoised object
                mag_heavy = np.clip(np.abs(O_heavy), 0.0, 1.0)
                phase_raw_heavy = np.angle(O_heavy)

                # Reference the phase using the same substrate-based approach
                phase_heavy = self._reference_phase(phase_raw_heavy, np.abs(O_heavy), probe)

                # Compute spatial means of the current output channels
                mean_mag = np.mean(output[0])
                mean_phase = np.mean(output[1])

                # Blend: weighted combination of heavy-TV reconstruction and spatial mean
                output[0] = self.blend_recon_weight * mag_heavy + self.blend_mean_weight * mean_mag
                output[1] = self.blend_recon_weight * phase_heavy + self.blend_mean_weight * mean_phase

            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """Full phase retrieval pipeline: preprocess, reconstruct, postprocess.

            Parameters
            ----------
            input_data : np.ndarray
                Measured diffraction intensities, shape (81, 256, 256), float64, non-negative.

            Returns
            -------
            np.ndarray
                Reconstructed object, shape (2, 600, 600), float64.
                Channel 0: magnitude (reflectivity), clipped to [0, 1].
                Channel 1: phase (radians), referenced to substrate.
            """
            # Step 1: Compute amplitudes
            data = input_data.astype(np.float64)
            self.amplitudes = np.sqrt(np.maximum(data, 0.0))

            # Compute mean total intensity for probe scaling
            self.mean_total_intensity = np.mean(np.sum(data, axis=(1, 2)))

            # Step 2: Detect DC location
            self.use_shift = self._detect_dc_location(data)

            # Step 3: Compute scan positions
            self.positions = self._compute_scan_positions()

            # Step 4: Select best probe via sigma search
            self.probe = self._select_best_probe()

            # Step 5: Run main momentum rPIE reconstruction
            best_O, best_P, best_err = self._run_main_reconstruction()

            # Step 6: If error too high, run ePIE fallback
            if best_err > self.fallback_error_threshold:
                O_epie, P_epie, err_epie = self._run_epie_fallback()
                # Use ePIE result if it has better consistency
                consistency_rpie = self._compute_consistency(best_O, best_P)
                consistency_epie = self._compute_consistency(O_epie, P_epie)
                if consistency_epie < consistency_rpie:
                    best_O = O_epie
                    best_P = P_epie
                    best_err = err_epie

            # Step 7: TV denoise the best reconstruction
            O_denoised = self._tv_denoise_complex(best_O, self.tv_weight, self.tv_n_iter)

            # Step 8: Format output (magnitude + referenced phase)
            output = self._format_output(O_denoised, best_P)

            # Step 9: Compute consistency, apply safety net if needed
            consistency = self._compute_consistency(O_denoised, best_P)

            # For the safety net, we need a phase_ref value. We'll compute it from
            # the substrate identification used in _reference_phase.
            # The _apply_safety_net method will re-reference phase internally.
            output = self._apply_safety_net(
                output,
                best_O,  # non-denoised object for heavy TV
                best_P,
                0.0,  # phase_ref not used directly in _apply_safety_net
                consistency,
            )

            # Step 10: Return shape (2, 600, 600) float64
            return output.astype(np.float64)



if __name__ == "__main__":
    # Load data
    input_data: np.ndarray = np.load("dataset/input.npy")

    # Initialize solver
    solver: InverseSolver = InverseSolver()

    # Run phase retrieval
    result: np.ndarray = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)