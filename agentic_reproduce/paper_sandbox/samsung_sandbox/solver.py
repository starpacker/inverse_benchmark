import numpy as np
import time
from typing import Tuple, List, Optional

class InverseSolver:
    """Coherent Diffractive Imaging phase retrieval via mPIE (momentum/mixed PIE)
    with TV regularization for ptychographic reconstruction of copper pad topography."""

    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_scan: int = 9,
            n_positions: int = 81,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            pad_cd_pixels: int = 22,
            n_iter_main: int = 400,
            n_iter_probe_search: int = 25,
            probe_update_start: int = 3,
            patience: int = 100,
            tv_interval: int = 20,
            tv_weight: float = 0.015,
            tv_n_iter: int = 8,
            tv_final_weight: float = 0.01,
            tv_final_n_iter: int = 10,
            alpha_fraction: float = 0.1,
            beta_start: float = 0.5,
            beta_ramp: float = 0.005,
            gamma_start: float = 0.05,
            gamma_decay: float = 0.0002,
            probe_energy_drift_max: float = 2.0,
            sigma_candidates: Optional[List[float]] = None,
            illum_threshold_fraction: float = 0.05,
            mag_clip_max: float = 3.0,
            phase_sign_ratio: float = 1.5,
            cv_heavy_blend_threshold: float = 0.5,
            cv_light_blend_threshold: float = 0.3,
            phase_std_max: float = 3.0,
            time_limit_seconds: float = 270.0,
            random_seed: int = 42,
        ) -> None:
            # Grid / geometry
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_scan: int = n_scan
            self.n_positions: int = n_positions

            # Physics
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.pad_cd_pixels: int = pad_cd_pixels

            # Main reconstruction iterations
            self.n_iter_main: int = n_iter_main
            self.n_iter_probe_search: int = n_iter_probe_search
            self.probe_update_start: int = probe_update_start
            self.patience: int = patience

            # TV regularization
            self.tv_interval: int = tv_interval
            self.tv_weight: float = tv_weight
            self.tv_n_iter: int = tv_n_iter
            self.tv_final_weight: float = tv_final_weight
            self.tv_final_n_iter: int = tv_final_n_iter

            # mPIE update parameters
            self.alpha_fraction: float = alpha_fraction
            self.beta_start: float = beta_start
            self.beta_ramp: float = beta_ramp
            self.gamma_start: float = gamma_start
            self.gamma_decay: float = gamma_decay

            # Probe constraints
            self.probe_energy_drift_max: float = probe_energy_drift_max
            self.sigma_candidates: List[float] = sigma_candidates if sigma_candidates is not None else [15.0, 30.0, 50.0, 75.0, 110.0]

            # Illumination / normalization
            self.illum_threshold_fraction: float = illum_threshold_fraction
            self.mag_clip_max: float = mag_clip_max

            # Phase referencing
            self.phase_sign_ratio: float = phase_sign_ratio

            # Quality safeguard blending thresholds
            self.cv_heavy_blend_threshold: float = cv_heavy_blend_threshold
            self.cv_light_blend_threshold: float = cv_light_blend_threshold
            self.phase_std_max: float = phase_std_max

            # Runtime
            self.time_limit_seconds: float = time_limit_seconds
            self.random_seed: int = random_seed

            # Internal state (populated during solve)
            self.use_fftshift: bool = False
            self.positions: List[Tuple[int, int]] = []
            self.amplitudes: Optional[np.ndarray] = None
            self.mean_total_intensity: float = 0.0
            self.probe_target_energy: float = 0.0
            self.best_probe: Optional[np.ndarray] = None
            self.best_object: Optional[np.ndarray] = None
            self.illumination_weight: Optional[np.ndarray] = None
            self.well_lit_mask: Optional[np.ndarray] = None
            self.error_history: List[float] = []
            self.t_start: float = 0.0

    def forward(self, obj_patch: np.ndarray) -> np.ndarray:
            """Propagate a 2D complex field to the far-field (Fraunhofer) detector plane.

            Parameters
            ----------
            obj_patch : np.ndarray
                Complex-valued 2D array (probe_size x probe_size) representing
                the exit wave P(r) * O(r).

            Returns
            -------
            np.ndarray
                Complex-valued 2D far-field diffraction pattern.
            """
            if self.use_fftshift:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj_patch)))
            else:
                return np.fft.fft2(obj_patch)

    def backward(self, field: np.ndarray) -> np.ndarray:
            """Inverse propagation from detector plane back to object plane.

            Parameters
            ----------
            field : np.ndarray
                Complex-valued 2D array in Fourier / detector space.

            Returns
            -------
            np.ndarray
                Complex-valued 2D array in real / object space.
            """
            if self.use_fftshift:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))
            else:
                return np.fft.ifft2(field)

    def detect_dc_center(self, data: np.ndarray) -> bool:
            """Determine whether diffraction patterns have DC at center or corner.

            Parameters
            ----------
            data : np.ndarray
                Input diffraction intensities, shape (n_positions, H, W).

            Returns
            -------
            bool
                True if DC is at center (use fftshift), False otherwise.
            """
            avg_pattern = np.mean(data, axis=0)
            H, W = avg_pattern.shape
            center_region = avg_pattern[H//2-16:H//2+16, W//2-16:W//2+16].sum()
            corner_region = (avg_pattern[:16, :16].sum() + avg_pattern[:16, -16:].sum() +
                             avg_pattern[-16:, :16].sum() + avg_pattern[-16:, -16:].sum())
            return center_region > corner_region

    def compute_scan_positions(self) -> List[Tuple[int, int]]:
            """Compute the (row, col) scan positions for the 9x9 ptychographic grid.

            Returns
            -------
            List[Tuple[int, int]]
                List of 81 (ry, rx) top-left corner positions for each scan point.
            """
            scan_step = (self.obj_size - self.probe_size) / (self.n_scan - 1)

            positions = []
            for iy in range(self.n_scan):
                for ix in range(self.n_scan):
                    ry = int(round(iy * scan_step))
                    rx = int(round(ix * scan_step))
                    ry = min(ry, self.obj_size - self.probe_size)
                    rx = min(rx, self.obj_size - self.probe_size)
                    positions.append((ry, rx))

            return positions

    def make_probe(self, sigma: float) -> np.ndarray:
            """Construct a Gaussian probe with energy normalized to match measured intensities.

            Parameters
            ----------
            sigma : float
                Standard deviation of the Gaussian probe in pixels.

            Returns
            -------
            np.ndarray
                Complex-valued 2D probe array, shape (probe_size, probe_size).
            """
            half = self.probe_size // 2
            Y, X = np.mgrid[-half:half, -half:half]
            R2 = (X**2 + Y**2).astype(np.float64)

            P = np.exp(-R2 / (2.0 * sigma**2)).astype(np.complex128)

            # Parseval's theorem: sum(|F{P*O}|^2) = N_pix * sum(|P*O|^2)
            # With |O| ≈ 1: sum(I) ≈ N_pix * sum(|P|^2)
            # So target: sum(|P|^2) = mean_total_intensity / N_pix
            N_pix = self.probe_size ** 2
            current_energy = np.sum(np.abs(P)**2)
            target_energy = self.mean_total_intensity / N_pix

            if current_energy > 0:
                P *= np.sqrt(target_energy / current_energy)

            return P

    def tv_denoise_2d(self, img: np.ndarray, weight: float, n_iter: int) -> np.ndarray:
            """Chambolle's projection algorithm for 2D total variation denoising.

            Parameters
            ----------
            img : np.ndarray
                Real-valued 2D image to denoise.
            weight : float
                TV regularization weight.
            n_iter : int
                Number of Chambolle projection iterations.

            Returns
            -------
            np.ndarray
                Denoised real-valued 2D image.
            """
            px = np.zeros_like(img)
            py = np.zeros_like(img)
            tau = 0.25

            for _ in range(n_iter):
                # Compute divergence of p
                div_p = np.zeros_like(img)
                div_p[:-1, :] += px[:-1, :]
                div_p[1:, :] -= px[:-1, :]
                div_p[:, :-1] += py[:, :-1]
                div_p[:, 1:] -= py[:, :-1]

                # Gradient of (img + weight * div_p)
                u = img + weight * div_p
                gx = np.zeros_like(img)
                gy = np.zeros_like(img)
                gx[:-1, :] = u[1:, :] - u[:-1, :]
                gy[:, :-1] = u[:, 1:] - u[:, :-1]

                # Update dual variables with projection
                norm_g = np.sqrt(gx**2 + gy**2)
                denom = 1.0 + tau * norm_g / weight
                px = (px + tau * gx) / denom
                py = (py + tau * gy) / denom

            # Compute final denoised image using final divergence
            div_p = np.zeros_like(img)
            div_p[:-1, :] += px[:-1, :]
            div_p[1:, :] -= px[:-1, :]
            div_p[:, :-1] += py[:, :-1]
            div_p[:, 1:] -= py[:, :-1]

            return img + weight * div_p

    def tv_denoise_complex(self, obj: np.ndarray, weight: float, n_iter: int) -> np.ndarray:
            """Apply TV denoising to real and imaginary parts of a complex object separately.

            Parameters
            ----------
            obj : np.ndarray
                Complex-valued 2D object array.
            weight : float
                TV regularization weight.
            n_iter : int
                Number of Chambolle projection iterations per channel.

            Returns
            -------
            np.ndarray
                TV-denoised complex-valued 2D object array.
            """
            real_part = self.tv_denoise_2d(obj.real.copy(), weight, n_iter)
            imag_part = self.tv_denoise_2d(obj.imag.copy(), weight, n_iter)
            return real_part + 1j * imag_part

    def run_quick_mpie(self, probe_init: np.ndarray, n_iter: int) -> float:
            """Run a short mPIE reconstruction for probe candidate evaluation.

            Parameters
            ----------
            probe_init : np.ndarray
                Initial probe guess, shape (probe_size, probe_size), complex128.
            n_iter : int
                Number of mPIE epochs to run.

            Returns
            -------
            float
                Final relative Fourier error (sum of squared amplitude discrepancy / total energy).
            """
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            P = probe_init.copy()
            alpha = self.alpha_fraction * np.max(np.abs(P) ** 2)

            final_err = np.inf
            total_energy = np.sum(self.amplitudes ** 2)

            for ep in range(n_iter):
                order = np.random.permutation(self.n_positions)
                epoch_err = 0.0

                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]

                    psi = P * O_patch
                    Psi = self.forward(psi)
                    mag = np.abs(Psi)

                    Psi_c = self.amplitudes[j] * Psi / (mag + 1e-12)
                    epoch_err += np.sum((mag - self.amplitudes[j]) ** 2)

                    psi_c = self.backward(Psi_c)
                    dpsi = psi_c - psi

                    # mPIE object update
                    P_abs2 = np.abs(P) ** 2
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] += (
                        np.conj(P) * dpsi / (P_abs2 + alpha)
                    )

                    # Probe update after probe_update_start epochs
                    if ep >= self.probe_update_start:
                        O_abs2 = np.abs(O_patch) ** 2
                        alpha_o = self.alpha_fraction * np.max(O_abs2) if np.max(O_abs2) > 0 else 1e-12
                        P += np.conj(O_patch) * dpsi / (O_abs2 + alpha_o)

                final_err = epoch_err / total_energy

            return final_err

    def select_best_probe(self) -> np.ndarray:
            """Evaluate multiple Gaussian probe candidates and return the best one.

            Returns
            -------
            np.ndarray
                Best probe array, shape (probe_size, probe_size), complex128.
            """
            best_sigma = self.sigma_candidates[len(self.sigma_candidates) // 2]  # default middle
            best_err = np.inf

            for sigma in self.sigma_candidates:
                np.random.seed(self.random_seed)
                P_test = self.make_probe(sigma)
                err = self.run_quick_mpie(P_test, n_iter=self.n_iter_probe_search)
                if err < best_err:
                    best_err = err
                    best_sigma = sigma

            np.random.seed(self.random_seed)
            return self.make_probe(best_sigma)

    def reconstruct_mpie(self, probe_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Main mPIE reconstruction loop with TV regularization and early stopping.

            Parameters
            ----------
            probe_init : np.ndarray
                Initial probe, shape (probe_size, probe_size), complex128.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                (best_object, best_probe) — the object and probe at the lowest Fourier error epoch.
            """
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            P = probe_init.copy()
            alpha_p = self.alpha_fraction * np.max(np.abs(P) ** 2)

            best_O = O.copy()
            best_P = P.copy()
            best_fourier_error = np.inf
            best_epoch = 0

            total_energy = np.sum(self.amplitudes ** 2)
            self.error_history = []

            for epoch in range(self.n_iter_main):
                # Time cap check
                if time.time() - self.t_start > self.time_limit_seconds:
                    break

                order = np.random.permutation(self.n_positions)
                epoch_err = 0.0

                # Adaptive step size: ramp from beta_start toward 1.0
                beta = min(1.0, self.beta_start + epoch * self.beta_ramp)

                # Relaxed modulus replacement: gamma decays from gamma_start to 0
                gamma = max(0.0, self.gamma_start - epoch * self.gamma_decay)

                for idx in range(self.n_positions):
                    j = order[idx]
                    ry, rx = self.positions[j]
                    O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size].copy()

                    psi = P * O_patch
                    Psi = self.forward(psi)
                    mag = np.abs(Psi)

                    # Relaxed modulus replacement (Wiener-like blending)
                    blended_amp = (1.0 - gamma) * self.amplitudes[j] + gamma * mag
                    Psi_c = blended_amp * Psi / (mag + 1e-12)

                    epoch_err += np.sum((mag - self.amplitudes[j]) ** 2)

                    psi_c = self.backward(Psi_c)
                    dpsi = psi_c - psi

                    # mPIE object update: O += beta * conj(P) * dpsi / (|P|^2 + alpha_p)
                    P_abs2 = np.abs(P) ** 2
                    O[ry:ry + self.probe_size, rx:rx + self.probe_size] += (
                        beta * np.conj(P) * dpsi / (P_abs2 + alpha_p)
                    )

                    # Probe update after probe_update_start epochs
                    if epoch >= self.probe_update_start:
                        O_abs2 = np.abs(O_patch) ** 2
                        max_O_abs2 = np.max(O_abs2)
                        alpha_o = self.alpha_fraction * max_O_abs2 if max_O_abs2 > 0 else 1e-12
                        P += beta * np.conj(O_patch) * dpsi / (O_abs2 + alpha_o)

                # Probe power constraint every 10 epochs
                if epoch % 10 == 0 and epoch >= self.probe_update_start:
                    current_energy = np.sum(np.abs(P) ** 2)
                    if current_energy > 0:
                        ratio = self.probe_target_energy / current_energy
                        if ratio < (1.0 / self.probe_energy_drift_max) or ratio > self.probe_energy_drift_max:
                            P *= np.sqrt(self.probe_target_energy / current_energy)
                    # Update alpha_p based on current probe
                    alpha_p = self.alpha_fraction * np.max(np.abs(P) ** 2)

                # TV regularization on object periodically
                if epoch > 0 and epoch % self.tv_interval == 0:
                    O = self.tv_denoise_complex(O, weight=self.tv_weight, n_iter=self.tv_n_iter)

                rel_err = epoch_err / total_energy
                self.error_history.append(rel_err)

                if rel_err < best_fourier_error:
                    best_fourier_error = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                # Early stopping with generous patience
                if epoch - best_epoch > self.patience and epoch > 80:
                    break

            return best_O, best_P

    def compute_illumination_map(self, probe: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Compute the illumination weight map and well-lit boolean mask.

            Parameters
            ----------
            probe : np.ndarray
                Final reconstructed probe, shape (probe_size, probe_size), complex128.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                (illum_weight, well_lit_mask) — both shape (obj_size, obj_size).
            """
            illum_weight = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            probe_intensity = np.abs(probe) ** 2

            for j in range(self.n_positions):
                ry, rx = self.positions[j]
                illum_weight[ry:ry + self.probe_size, rx:rx + self.probe_size] += probe_intensity

            illum_max = np.max(illum_weight)
            well_lit_mask = illum_weight > self.illum_threshold_fraction * illum_max

            return illum_weight, well_lit_mask

    def normalize_magnitude(
            self, mag: np.ndarray, well_lit: np.ndarray
        ) -> np.ndarray:
            """Normalize object magnitude so that the median in well-lit region equals 1.0.

            Parameters
            ----------
            mag : np.ndarray
                Raw magnitude array, shape (obj_size, obj_size).
            well_lit : np.ndarray
                Boolean mask of well-illuminated pixels, shape (obj_size, obj_size).

            Returns
            -------
            np.ndarray
                Normalized and clipped magnitude array.
            """
            mag_out = mag.copy()
            n_well_lit = np.sum(well_lit)

            if n_well_lit > 100:
                median_mag = np.median(mag_out[well_lit])
                if median_mag > 1e-15:
                    mag_out = mag_out / median_mag
                else:
                    # Reconstruction completely failed; use uniform magnitude
                    mag_out = np.ones((self.obj_size, self.obj_size), dtype=np.float64)
            else:
                # No well-lit region detected; use uniform fallback
                mag_out = np.ones((self.obj_size, self.obj_size), dtype=np.float64)

            # Clip extreme outliers (>mag_clip_max times median, which would be unphysical)
            mag_out = np.clip(mag_out, 0.0, self.mag_clip_max)

            # Set poorly illuminated edges to the median of well-lit region (substrate-like)
            if n_well_lit > 0:
                fill_value = np.median(mag_out[well_lit])
            else:
                fill_value = 1.0
            mag_out[~well_lit] = fill_value

            return mag_out

    def reference_phase(
            self, phase: np.ndarray, well_lit: np.ndarray
        ) -> np.ndarray:
            """Reference phase so that substrate (majority population) is at zero.

            Parameters
            ----------
            phase : np.ndarray
                Raw phase array in radians, shape (obj_size, obj_size).
            well_lit : np.ndarray
                Boolean mask of well-illuminated pixels, shape (obj_size, obj_size).

            Returns
            -------
            np.ndarray
                Phase array with substrate at ~0 and pads at positive values.
            """
            phase_out = phase.copy()
            n_well_lit = np.sum(well_lit)

            if n_well_lit > 100:
                phase_well = phase_out[well_lit]

                # Compute histogram to find the mode (substrate phase)
                n_bins = 100
                hist, bin_edges = np.histogram(phase_well, bins=n_bins)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                # The mode (peak of histogram) corresponds to substrate phase
                mode_idx = np.argmax(hist)
                substrate_phase = bin_centers[mode_idx]

                # Subtract substrate phase so substrate → 0
                phase_out = phase_out - substrate_phase

                # Wrap to [-pi, pi]
                phase_out = np.arctan2(np.sin(phase_out), np.cos(phase_out))

                # Determine sign convention using mass-based approach
                # Physical constraint: pads should have POSITIVE phase (ΔΦ = 4πh/λ > 0)
                phase_well_corrected = phase_out[well_lit]

                # Compute signal mass away from zero (beyond ±0.3 rad threshold)
                positive_mask = phase_well_corrected > 0.3
                negative_mask = phase_well_corrected < -0.3
                positive_mass = np.sum(phase_well_corrected[positive_mask]) if np.any(positive_mask) else 0.0
                negative_mass = np.sum(np.abs(phase_well_corrected[negative_mask])) if np.any(negative_mask) else 0.0

                # If more signal mass is negative, flip the sign
                if negative_mass > positive_mass * self.phase_sign_ratio:
                    phase_out = -phase_out
            else:
                # No well-lit region; return zeros
                phase_out = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)

            # Set poorly-illuminated regions to 0 phase (substrate-like)
            phase_out[~well_lit] = 0.0

            return phase_out

    def validate_and_blend(
            self, mag: np.ndarray, phase: np.ndarray, well_lit: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Quality check: blend noisy reconstructions toward uniform to protect SSIM.

            Parameters
            ----------
            mag : np.ndarray
                Normalized magnitude, shape (obj_size, obj_size).
            phase : np.ndarray
                Referenced phase, shape (obj_size, obj_size).
            well_lit : np.ndarray
                Boolean mask, shape (obj_size, obj_size).

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                (blended_mag, blended_phase) after quality-aware blending.
            """
            mag_out = mag.copy()
            phase_out = phase.copy()
            n_well_lit = np.sum(well_lit)

            if n_well_lit > 100:
                mag_well = mag_out[well_lit]
                phase_well = phase_out[well_lit]

                # Coefficient of variation of magnitude (should be small for this object)
                mean_mag = np.mean(mag_well)
                cv_mag = np.std(mag_well) / (mean_mag + 1e-12)

                # If CV is very high, the reconstruction is noisy — blend with uniform
                if cv_mag > self.cv_heavy_blend_threshold:
                    blend_factor = min(0.7, cv_mag - 0.3)
                    mag_out = (1.0 - blend_factor) * mag_out + blend_factor * 1.0
                elif cv_mag > self.cv_light_blend_threshold:
                    blend_factor = 0.2
                    mag_out = (1.0 - blend_factor) * mag_out + blend_factor * 1.0

                # Phase: if too noisy, attenuate
                phase_std = np.std(phase_well)
                # Expected phase std: bimodal with values 0 and ~1-3 rad, so std ~0.5-1.5
                if phase_std > self.phase_std_max:
                    attenuation = 1.5 / phase_std
                    phase_out *= attenuation

            return mag_out, phase_out

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """Full inverse solve pipeline: from diffraction intensities to (magnitude, phase).

            Parameters
            ----------
            input_data : np.ndarray
                Diffraction intensity patterns, shape (n_positions, 256, 256), float64, non-negative.

            Returns
            -------
            np.ndarray
                Reconstructed object, shape (2, obj_size, obj_size), where
                channel 0 = magnitude (median-normalized ~1.0) and
                channel 1 = phase (radians, substrate ~0, pads > 0).
            """
            self.t_start = time.time()

            # Step 1: Compute amplitudes
            self.amplitudes = np.sqrt(np.maximum(input_data, 0.0))

            # Step 2: Detect DC center convention
            self.use_fftshift = self.detect_dc_center(input_data)

            # Step 3: Compute scan positions
            self.positions = self.compute_scan_positions()

            # Step 4: Compute mean total intensity and probe target energy
            self.mean_total_intensity = np.mean(np.sum(input_data, axis=(1, 2)))
            N_pix = self.probe_size ** 2
            self.probe_target_energy = self.mean_total_intensity / N_pix

            # Step 5: Select best probe via quick mPIE trials
            best_probe = self.select_best_probe()

            # Step 6: Run main mPIE reconstruction
            np.random.seed(self.random_seed)
            best_O, best_P = self.reconstruct_mpie(best_probe)

            # Step 7: Final TV smoothing pass
            O_final = self.tv_denoise_complex(best_O, weight=self.tv_final_weight, n_iter=self.tv_final_n_iter)

            # Step 8: Compute illumination map
            illum_weight, well_lit = self.compute_illumination_map(best_P)
            self.illumination_weight = illum_weight
            self.well_lit_mask = well_lit

            # Step 9: Extract magnitude and phase
            mag_final = np.abs(O_final)
            phase_final = np.angle(O_final)

            # Step 10: Normalize magnitude
            mag_final = self.normalize_magnitude(mag_final, well_lit)

            # Step 11: Reference phase
            phase_final = self.reference_phase(phase_final, well_lit)

            # Step 12: Validate and blend
            mag_final, phase_final = self.validate_and_blend(mag_final, phase_final, well_lit)

            # Step 13: Assemble output
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            output[0] = mag_final
            output[1] = phase_final

            return output



if __name__ == "__main__":
    # Load data
    input_data = np.load("dataset/input.npy")

    # Create solver and run
    solver = InverseSolver()
    result = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)
    print(f"Output shape: {result.shape}")
    print(f"Magnitude: mean={result[0].mean():.4f}, std={result[0].std():.4f}, range=[{result[0].min():.4f}, {result[0].max():.4f}]")
    print(f"Phase: mean={result[1].mean():.4f}, std={result[1].std():.4f}, range=[{result[1].min():.4f}, {result[1].max():.4f}]")