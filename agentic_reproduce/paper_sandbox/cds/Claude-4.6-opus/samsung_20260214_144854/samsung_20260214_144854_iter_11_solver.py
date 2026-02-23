import numpy as np
import time
from typing import Tuple, List, Optional

class InverseSolver:
    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_scan: int = 9,
            n_patterns: int = 81,
            wavelength: float = 13.5e-9,
            pixel_size: float = 27e-9,
            pad_cd_pixels: int = 22,
            pad_pitch_nm: float = 800.0,
            pad_cd_nm: float = 600.0,
            duty_cycle: float = 0.6,
            pad_array_size: int = 15,
            recession_min: float = 1.0e-9,
            recession_max: float = 3.0e-9,
            sigma_candidates: Optional[List[float]] = None,
            n_iter_quick: int = 30,
            n_iter_main: int = 300,
            n_iter_second: int = 200,
            beta_obj: float = 1.0,
            beta_probe: float = 1.0,
            probe_update_start: int = 5,
            patience: int = 80,
            patience_second: int = 60,
            probe_power_constraint_interval: int = 10,
            probe_power_soft_factor: float = 0.9,
            illum_threshold_fraction: float = 0.1,
            magnitude_coeff_var_threshold: float = 5.0,
            eps: float = 1e-12,
            random_seed: int = 42,
            second_run_time_budget: float = 120.0,
        ) -> None:
            # Grid / geometry
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_scan: int = n_scan
            self.n_patterns: int = n_patterns

            # Physics
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.pad_cd_pixels: int = pad_cd_pixels
            self.pad_pitch_nm: float = pad_pitch_nm
            self.pad_cd_nm: float = pad_cd_nm
            self.duty_cycle: float = duty_cycle
            self.pad_array_size: int = pad_array_size
            self.recession_min: float = recession_min
            self.recession_max: float = recession_max

            # Probe search
            self.sigma_candidates: List[float] = sigma_candidates if sigma_candidates is not None else [10, 20, 30, 40, 55, 75, 100]
            self.n_iter_quick: int = n_iter_quick

            # Main reconstruction
            self.n_iter_main: int = n_iter_main
            self.n_iter_second: int = n_iter_second
            self.beta_obj: float = beta_obj
            self.beta_probe: float = beta_probe
            self.probe_update_start: int = probe_update_start
            self.patience: int = patience
            self.patience_second: int = patience_second

            # Probe power constraint
            self.probe_power_constraint_interval: int = probe_power_constraint_interval
            self.probe_power_soft_factor: float = probe_power_soft_factor

            # Quality / masking
            self.illum_threshold_fraction: float = illum_threshold_fraction
            self.magnitude_coeff_var_threshold: float = magnitude_coeff_var_threshold

            # Numerics
            self.eps: float = eps
            self.random_seed: int = random_seed
            self.second_run_time_budget: float = second_run_time_budget

            # Derived state (populated during solve)
            self.use_fftshift: bool = False
            self.positions: List[Tuple[int, int]] = []
            self.mean_total_intensity: float = 0.0
            self.probe_target_energy: float = 0.0
            self.amplitudes: Optional[np.ndarray] = None
            self.illum_weight: Optional[np.ndarray] = None
            self.well_lit: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Forward / inverse propagation
    # ------------------------------------------------------------------

    def forward_prop(self, x: np.ndarray) -> np.ndarray:
            """Propagate a 2-D complex field to the far-field (Fourier) domain."""
            if self.use_fftshift:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            else:
                return np.fft.fft2(x)

    def backward_prop(self, X: np.ndarray) -> np.ndarray:
            """Propagate a 2-D complex field back to real space (inverse Fourier)."""
            if self.use_fftshift:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))
            else:
                return np.fft.ifft2(X)

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            positions: List[Tuple[int, int]],
        ) -> np.ndarray:
            """
            Simulate the full forward model for all scan positions.

            Parameters
            ----------
            obj : np.ndarray, shape (obj_size, obj_size), complex128
                Complex-valued object reflection function O(r).
            probe : np.ndarray, shape (probe_size, probe_size), complex128
                Illumination probe P(r).
            positions : list of (int, int)
                Top-left (row, col) for each scan position.

            Returns
            -------
            intensities : np.ndarray, shape (n_patterns, probe_size, probe_size), float64
                Simulated diffraction intensities |F{P * O_patch}|^2.
            """
            n = len(positions)
            ps = self.probe_size
            intensities = np.zeros((n, ps, ps), dtype=np.float64)

            for j, (ry, rx) in enumerate(positions):
                O_patch = obj[ry:ry + ps, rx:rx + ps]
                psi = probe * O_patch
                Psi = self.forward_prop(psi)
                intensities[j] = np.abs(Psi) ** 2

            return intensities

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _detect_dc_position(self, data: np.ndarray) -> None:
            """Determine whether DC is centred or at corners and set self.use_fftshift."""
            avg_pattern = np.mean(data, axis=0)

            # Compare energy in center region vs corner regions
            h, w = avg_pattern.shape
            ch, cw = h // 2, w // 2
            half_win = 16

            center_region = avg_pattern[ch - half_win:ch + half_win, cw - half_win:cw + half_win].sum()

            corner_region = (
                avg_pattern[:half_win, :half_win].sum()
                + avg_pattern[:half_win, -half_win:].sum()
                + avg_pattern[-half_win:, :half_win].sum()
                + avg_pattern[-half_win:, -half_win:].sum()
            )

            self.use_fftshift = center_region > corner_region

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
            """Compute the 9×9 raster-scan positions centred on the object grid."""
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

    def _make_probe(self, sigma: float) -> np.ndarray:
            """
            Construct a Gaussian probe with given sigma and normalise its energy
            so that Parseval's theorem is consistent with the measured intensity.

            Parameters
            ----------
            sigma : float
                Standard deviation (in pixels) of the Gaussian envelope.

            Returns
            -------
            P : np.ndarray, shape (probe_size, probe_size), complex128
            """
            half = self.probe_size // 2
            Y, X = np.mgrid[-half:half, -half:half]
            R2 = (X ** 2 + Y ** 2).astype(np.float64)

            P = np.exp(-R2 / (2.0 * sigma ** 2)).astype(np.complex128)

            # Normalise probe energy using Parseval's theorem:
            # For unnormalised FFT: sum(|F{P*O}|^2) = N_pix * sum(|P*O|^2)
            # With |O| ≈ 1: sum(I_j) ≈ N_pix * sum(|P|^2)
            # => target sum(|P|^2) = mean_total_intensity / N_pix
            N_pix = self.probe_size ** 2
            current_energy = np.sum(np.abs(P) ** 2)
            target_energy = self.probe_target_energy if self.probe_target_energy > 0 else self.mean_total_intensity / N_pix
            if current_energy > 0 and target_energy > 0:
                P *= np.sqrt(target_energy / current_energy)

            return P

    def _constrain_probe_power(self, P: np.ndarray, target_energy: float) -> np.ndarray:
            """Soft probe-power constraint to prevent the probe absorbing object features."""
            current_energy = np.sum(np.abs(P) ** 2)
            if current_energy > 0 and target_energy > 0:
                ratio = np.sqrt(target_energy / current_energy)
                # Soft constraint: blend toward target rather than hard rescaling
                P = P * (self.probe_power_soft_factor + (1.0 - self.probe_power_soft_factor) * ratio)
            return P

    def _compute_backprojection_init(self) -> np.ndarray:
            """
            Spectral (back-projection) initialisation of the object.

            Returns
            -------
            O_init : np.ndarray, shape (obj_size, obj_size), complex128
            """
            O_num = np.zeros((self.obj_size, self.obj_size), dtype=np.complex128)
            O_den = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)

            # Use a moderate-width probe for initialization
            P_init = self._make_probe(40.0)

            ps = self.probe_size

            for j in range(self.n_patterns):
                ry, rx = self.positions[j]

                # Back-project: inverse FT of measured amplitude (with zero phase)
                rough_exit = self.backward_prop(self.amplitudes[j].astype(np.complex128))

                # Accumulate weighted back-projection
                O_num[ry:ry + ps, rx:rx + ps] += np.conj(P_init) * rough_exit
                O_den[ry:ry + ps, rx:rx + ps] += np.abs(P_init) ** 2

            # Avoid division by zero
            O_den_safe = np.maximum(O_den, self.eps * np.max(O_den) if np.max(O_den) > 0 else self.eps)
            O_init = O_num / O_den_safe

            # Normalize: set mean magnitude to ~1 in well-covered regions
            well_covered = O_den > 0.1 * np.max(O_den) if np.max(O_den) > 0 else np.ones_like(O_den, dtype=bool)
            if np.sum(well_covered) > 0:
                mean_mag = np.mean(np.abs(O_init[well_covered]))
                if mean_mag > self.eps:
                    O_init /= mean_mag

            # Safety check
            if np.any(np.isnan(O_init)) or np.any(np.isinf(O_init)):
                O_init = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

            return O_init

    # ------------------------------------------------------------------
    # Probe selection
    # ------------------------------------------------------------------

    def _quick_ePIE(
            self,
            P_init: np.ndarray,
            n_iter: int,
        ) -> float:
            """
            Run a short ePIE reconstruction to evaluate a candidate probe.

            Returns
            -------
            rel_err : float
                Final relative Fourier-domain error.
            """
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            P = P_init.copy()
            ps = self.probe_size
            final_err = np.inf

            for ep in range(n_iter):
                order = np.random.permutation(self.n_patterns)
                epoch_err = 0.0

                for idx in range(self.n_patterns):
                    j = order[idx]
                    ry, rx = self.positions[j]
                    O_patch = O[ry:ry + ps, rx:rx + ps]

                    psi = P * O_patch
                    Psi = self.forward_prop(psi)
                    mag = np.abs(Psi)

                    # Modulus replacement
                    Psi_c = self.amplitudes[j] * Psi / (mag + self.eps)
                    epoch_err += np.sum((mag - self.amplitudes[j]) ** 2)

                    psi_c = self.backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # ePIE object update
                    P_conj = np.conj(P)
                    P_abs2_max = np.max(np.abs(P) ** 2)
                    O[ry:ry + ps, rx:rx + ps] += self.beta_obj * P_conj * dpsi / (P_abs2_max + self.eps)

                    # ePIE probe update (start after a few epochs)
                    if ep >= 3:
                        O_conj = np.conj(O_patch)
                        O_abs2_max = np.max(np.abs(O_patch) ** 2)
                        P += self.beta_probe * O_conj * dpsi / (O_abs2_max + self.eps)

                total_energy = np.sum(self.amplitudes ** 2)
                final_err = epoch_err / total_energy if total_energy > 0 else epoch_err

            return final_err

    def _select_best_probe(self) -> Tuple[float, float]:
            """
            Search over sigma_candidates and return the best sigma and its error.

            Returns
            -------
            best_sigma : float
            best_err : float
            """
            best_sigma = 40.0
            best_err = np.inf

            for sigma in self.sigma_candidates:
                P_test = self._make_probe(sigma)
                err = self._quick_ePIE(P_test, n_iter=self.n_iter_quick)
                if err < best_err:
                    best_err = err
                    best_sigma = sigma

            return best_sigma, best_err

    # ------------------------------------------------------------------
    # Main reconstruction
    # ------------------------------------------------------------------

    def _run_ePIE(
            self,
            O_init: np.ndarray,
            P_init: np.ndarray,
            n_iter: int,
            patience: int,
            time_budget: Optional[float] = None,
        ) -> Tuple[np.ndarray, np.ndarray, float, int]:
            """
            Full ePIE reconstruction loop with early stopping.

            Parameters
            ----------
            O_init : np.ndarray, shape (obj_size, obj_size), complex128
            P_init : np.ndarray, shape (probe_size, probe_size), complex128
            n_iter : int
            patience : int
            time_budget : float or None
                Wall-clock seconds after which to stop (None = unlimited).

            Returns
            -------
            best_O : np.ndarray
            best_P : np.ndarray
            best_err : float
            best_epoch : int
            """
            t_start = time.time()

            O = O_init.copy()
            P = P_init.copy()
            ps = self.probe_size

            best_O = O.copy()
            best_P = P.copy()
            best_err = np.inf
            best_epoch = 0

            total_energy = np.sum(self.amplitudes ** 2)
            if total_energy <= 0:
                total_energy = 1.0

            for epoch in range(n_iter):
                # Check time budget
                if time_budget is not None and (time.time() - t_start) > time_budget:
                    break

                order = np.random.permutation(self.n_patterns)
                epoch_err = 0.0

                for idx in range(self.n_patterns):
                    j = order[idx]
                    ry, rx = self.positions[j]
                    O_patch = O[ry:ry + ps, rx:rx + ps].copy()

                    psi = P * O_patch
                    Psi = self.forward_prop(psi)
                    mag = np.abs(Psi)

                    # Modulus replacement
                    Psi_c = self.amplitudes[j] * Psi / (mag + self.eps)
                    epoch_err += np.sum((mag - self.amplitudes[j]) ** 2)

                    psi_c = self.backward_prop(Psi_c)
                    dpsi = psi_c - psi

                    # ePIE object update: O += beta * P* * dpsi / max(|P|^2)
                    P_conj = np.conj(P)
                    P_abs2_max = np.max(np.abs(P) ** 2)
                    O[ry:ry + ps, rx:rx + ps] += self.beta_obj * P_conj * dpsi / (P_abs2_max + self.eps)

                    # ePIE probe update (start after probe_update_start epochs)
                    if epoch >= self.probe_update_start:
                        O_conj = np.conj(O_patch)
                        O_abs2_max = np.max(np.abs(O_patch) ** 2)
                        P += self.beta_probe * O_conj * dpsi / (O_abs2_max + self.eps)

                # Apply probe power constraint periodically
                if (epoch % self.probe_power_constraint_interval == 0
                        and epoch >= self.probe_update_start
                        and self.probe_target_energy > 0):
                    P = self._constrain_probe_power(P, self.probe_target_energy)

                rel_err = epoch_err / total_energy

                if rel_err < best_err:
                    best_err = rel_err
                    best_O = O.copy()
                    best_P = P.copy()
                    best_epoch = epoch

                # Early stopping with patience
                if epoch - best_epoch > patience and epoch > 50:
                    break

            return best_O, best_P, best_err, best_epoch

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _compute_illumination_mask(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build illumination weight map and well-lit boolean mask.

            Returns
            -------
            illum_weight : np.ndarray, shape (obj_size, obj_size), float64
            well_lit : np.ndarray, shape (obj_size, obj_size), bool
            """
            illum_weight = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            ps = self.probe_size
            P_abs2 = np.abs(P) ** 2

            for j in range(self.n_patterns):
                ry, rx = self.positions[j]
                illum_weight[ry:ry + ps, rx:rx + ps] += P_abs2

            max_illum = np.max(illum_weight)
            illum_thresh = self.illum_threshold_fraction * max_illum if max_illum > 0 else 0.0
            well_lit = illum_weight > illum_thresh

            self.illum_weight = illum_weight
            self.well_lit = well_lit

            return illum_weight, well_lit

    def _quality_check(
            self,
            mag: np.ndarray,
            well_lit: np.ndarray,
        ) -> bool:
            """
            Return True if the reconstruction looks reasonable, False if diverged.
            """
            if np.sum(well_lit) == 0:
                return False

            mag_well = mag[well_lit]
            mag_mean = np.mean(mag_well)
            mag_std = np.std(mag_well)
            coeff_var = mag_std / (mag_mean + self.eps)

            if coeff_var > self.magnitude_coeff_var_threshold:
                return False

            if mag_mean < self.eps:
                return False

            if np.any(np.isnan(mag_well)) or np.any(np.isinf(mag_well)):
                return False

            return True

    def _phase_reference(
            self,
            phase: np.ndarray,
            well_lit: np.ndarray,
        ) -> np.ndarray:
            """
            Median-based phase referencing: subtract substrate median, wrap, fix sign.

            Parameters
            ----------
            phase : np.ndarray, shape (obj_size, obj_size), float64
            well_lit : np.ndarray, shape (obj_size, obj_size), bool

            Returns
            -------
            phase_corrected : np.ndarray, shape (obj_size, obj_size), float64
            """
            phase_corrected = phase.copy()

            if np.sum(well_lit) == 0:
                return phase_corrected

            phase_well = phase_corrected[well_lit]

            # The substrate occupies ~64% of the area (duty cycle 0.6 for pads)
            # The median phase in the well-lit region should correspond to substrate
            substrate_phase = np.median(phase_well)
            phase_corrected = phase_corrected - substrate_phase

            # Wrap to [-pi, pi]
            phase_corrected = np.arctan2(np.sin(phase_corrected), np.cos(phase_corrected))

            # Phase sign convention: ΔΦ = 4πh/λ > 0 for recessed pads
            # Pads have DIFFERENT reflectivity than substrate
            # We check: the minority population (pads, ~36% area) should have positive phase
            # Use the 75th percentile vs 25th percentile of the re-centered phase
            phase_well_centered = phase_corrected[well_lit]
            p25 = np.percentile(phase_well_centered, 25)
            p75 = np.percentile(phase_well_centered, 75)

            # The distribution should be bimodal: substrate near 0, pads at positive values
            # If the tail is more negative than positive, flip
            if abs(p25) > abs(p75) and p25 < -0.1:
                phase_corrected = -phase_corrected

            # Zero out poorly illuminated regions
            phase_corrected[~well_lit] = 0.0

            return phase_corrected

    def _normalise_magnitude(
            self,
            mag: np.ndarray,
            well_lit: np.ndarray,
        ) -> np.ndarray:
            """Normalise magnitude to a physically reasonable range [0, ~1]."""
            mag_out = mag.copy()

            if np.sum(well_lit) == 0:
                return mag_out

            mag_well = mag_out[well_lit]
            mag_p99 = np.percentile(mag_well, 99)

            if mag_p99 > 0:
                # If magnitude scale is far from 1, rescale to bring it into [0, ~1]
                if mag_p99 > 2.0 or mag_p99 < 0.01:
                    mag_out = mag_out / mag_p99

            # Zero out poorly illuminated regions
            mag_out[~well_lit] = 0.0

            return mag_out

    # ------------------------------------------------------------------
    # Top-level solver
    # ------------------------------------------------------------------

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            End-to-end phase retrieval from measured diffraction intensities.

            Parameters
            ----------
            input_data : np.ndarray, shape (n_patterns, probe_size, probe_size), float64
                Measured far-field diffraction intensity patterns (non-negative).

            Returns
            -------
            output : np.ndarray, shape (2, obj_size, obj_size), float64
                Channel 0 — object magnitude.
                Channel 1 — object phase (radians).
            """
            # 1. Seed RNG for reproducibility
            np.random.seed(self.random_seed)

            # 2. Compute amplitudes = sqrt(max(data, 0))
            self.amplitudes = np.sqrt(np.maximum(input_data, 0.0))

            # 3. Detect DC position
            self._detect_dc_position(input_data)

            # 4. Compute scan positions
            self.positions = self._compute_scan_positions()

            # 5. Compute mean total intensity & probe target energy
            self.mean_total_intensity = float(np.mean(np.sum(input_data, axis=(1, 2))))
            N_pix = self.probe_size ** 2
            self.probe_target_energy = self.mean_total_intensity / N_pix

            # 6. Select best probe sigma
            np.random.seed(self.random_seed)
            best_sigma, best_probe_err = self._select_best_probe()
            print(f"Best sigma: {best_sigma}, error: {best_probe_err:.6f}")

            # 7. Build initial probe & back-projection object
            np.random.seed(self.random_seed)
            P_main = self._make_probe(best_sigma)
            O_backproj = self._compute_backprojection_init()

            # 8. Run main ePIE with back-projection init
            np.random.seed(self.random_seed)
            best_O1, best_P1, best_err1, best_epoch1 = self._run_ePIE(
                O_init=O_backproj,
                P_init=P_main.copy(),
                n_iter=self.n_iter_main,
                patience=self.patience,
                time_budget=None,
            )
            print(f"Back-projection init: best error={best_err1:.6f} at epoch {best_epoch1}")

            # 9. Run second ePIE with all-ones init, time-budgeted
            np.random.seed(self.random_seed)
            O_ones = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            P_second = self._make_probe(best_sigma)
            best_O2, best_P2, best_err2, best_epoch2 = self._run_ePIE(
                O_init=O_ones,
                P_init=P_second,
                n_iter=self.n_iter_second,
                patience=self.patience_second,
                time_budget=self.second_run_time_budget,
            )
            print(f"All-ones init: best error={best_err2:.6f} at epoch {best_epoch2}")

            # 10. Pick reconstruction with lower Fourier error
            if best_err2 < best_err1:
                print(f"All-ones init was better: {best_err2:.6f} vs {best_err1:.6f}")
                O_final = best_O2
                P_final = best_P2
            else:
                print(f"Back-projection init was better: {best_err1:.6f} vs {best_err2:.6f}")
                O_final = best_O1
                P_final = best_P1

            # 11. Compute illumination mask
            illum_weight, well_lit = self._compute_illumination_mask(P_final)

            # 12. Extract magnitude and phase
            mag_final = np.abs(O_final)
            phase_final = np.angle(O_final)

            # 13. Quality check & fallback
            is_good = self._quality_check(mag_final, well_lit)
            if not is_good:
                print("WARNING: Reconstruction may have diverged. Applying fallback normalization.")
                max_mag = np.max(mag_final)
                if max_mag > 0:
                    mag_final = mag_final / max_mag

            # 14. Phase referencing
            phase_final = self._phase_reference(phase_final, well_lit)

            # 15. Magnitude normalisation
            mag_final = self._normalise_magnitude(mag_final, well_lit)

            # Zero out poorly illuminated regions for magnitude too
            mag_final[~well_lit] = 0.0

            # 16. Assemble output array (2, obj_size, obj_size)
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            output[0] = mag_final
            output[1] = phase_final

            print(f"Output shape: {output.shape}")
            print(f"Magnitude range: [{output[0].min():.4f}, {output[0].max():.4f}]")
            print(f"Phase range: [{output[1].min():.4f}, {output[1].max():.4f}]")

            return output



if __name__ == "__main__":
    # Load data
    input_data: np.ndarray = np.load("dataset/input.npy")

    # Instantiate solver with default hyperparameters
    solver: InverseSolver = InverseSolver()

    # Run phase retrieval
    result: np.ndarray = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)