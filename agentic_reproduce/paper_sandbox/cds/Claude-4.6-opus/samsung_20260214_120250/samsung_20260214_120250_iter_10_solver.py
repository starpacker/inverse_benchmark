import numpy as np
import time
from typing import Tuple, Optional, Dict, List, Any

class InverseSolver:
    """
    rpPIE (regularized ptychographic iterative engine) solver for recovering
    complex-valued reflection functions of periodic copper pad arrays from
    far-field diffraction intensity measurements.
    """

    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_side: int = 9,
            step: int = 40,
            n_pads: int = 15,
            pad_sz: int = 22,
            pixel_size_nm: float = 27.0,
            wavelength_nm: float = 13.5,
            beta_obj: float = 1.0,
            alpha: float = 1e-3,
            beta_probe: float = 0.5,
            alpha_probe: float = 1e-3,
            beta_prior_start: float = 0.8,
            beta_prior_end: float = 0.3,
            sigma_probe: float = 35.0,
            probe_update_start_epoch: int = 3,
            probe_weight_update_interval: int = 10,
            pitch_candidates: Optional[List[int]] = None,
            calibration_epochs: int = 3,
            max_epochs: int = 350,
            time_budget_seconds: float = 55.0,
            eps: float = 1e-12,
            n_SiCN: complex = 0.9791 - 0.0059j,
            n_Cu: complex = 0.9625 - 0.0613j,
            n_SiO2: complex = 0.9780 - 0.0108j,
            n_Si: complex = 0.9991 - 0.0018j,
        ) -> None:
            # Grid / geometry
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_pads: int = n_pads
            self.pad_sz: int = pad_sz
            self.pixel_size_nm: float = pixel_size_nm
            self.wavelength_nm: float = wavelength_nm
            self.n_pos: int = n_side * n_side

            # ePIE / rpPIE hyperparameters
            self.beta_obj: float = beta_obj
            self.alpha: float = alpha
            self.beta_probe: float = beta_probe
            self.alpha_probe: float = alpha_probe
            self.beta_prior_start: float = beta_prior_start
            self.beta_prior_end: float = beta_prior_end

            # Probe initialisation
            self.sigma_probe: float = sigma_probe
            self.probe_update_start_epoch: int = probe_update_start_epoch
            self.probe_weight_update_interval: int = probe_weight_update_interval

            # Support / pitch search
            self.pitch_candidates: List[int] = pitch_candidates if pitch_candidates is not None else [29, 30, 31, 37]

            # Iteration control
            self.calibration_epochs: int = calibration_epochs
            self.max_epochs: int = max_epochs
            self.time_budget_seconds: float = time_budget_seconds
            self.eps: float = eps

            # Material optical constants (complex refractive indices)
            self.n_SiCN: complex = n_SiCN
            self.n_Cu: complex = n_Cu
            self.n_SiO2: complex = n_SiO2
            self.n_Si: complex = n_Si

            # Derived reflectivities (Fresnel at normal incidence from vacuum)
            # r = (1 - n_complex) / (1 + n_complex)
            self.R_sub: complex = (1.0 - self.n_SiCN) / (1.0 + self.n_SiCN)
            self.R_Cu: complex = (1.0 - self.n_Cu) / (1.0 + self.n_Cu)
            self.R_p: complex = self.R_sub

            # Runtime state (populated during solve)
            self.positions: Optional[np.ndarray] = None
            self.support: Optional[np.ndarray] = None
            self.outside_support: Optional[np.ndarray] = None
            self.probe: Optional[np.ndarray] = None
            self.object: Optional[np.ndarray] = None
            self.diff_amp_work: Optional[np.ndarray] = None
            self.t_start: float = 0.0

    # ------------------------------------------------------------------
    # Scan positions
    # ------------------------------------------------------------------

    def _build_positions(self) -> np.ndarray:
            """Compute (n_pos, 2) array of (row, col) scan positions."""
            total_scan = self.probe_size + (self.n_side - 1) * self.step  # 256 + 320 = 576
            offset = (self.obj_size - total_scan) // 2  # 12

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions, dtype=int)

    # ------------------------------------------------------------------
    # Support mask construction
    # ------------------------------------------------------------------

    def _make_support(self, pitch_px: int) -> Tuple[Optional[np.ndarray], Optional[int]]:
            """
            Build a binary support mask for a 15×15 pad array with given pitch.

            Returns (mask, array_offset) or (None, None) if array doesn't fit.
            """
            arr_ext = (self.n_pads - 1) * pitch_px + self.pad_sz
            arr_off = (self.obj_size - arr_ext) // 2
            if arr_off < 0:
                return None, None
            S = np.zeros((self.obj_size, self.obj_size), dtype=bool)
            for m in range(self.n_pads):
                for n_idx in range(self.n_pads):
                    r0 = arr_off + m * pitch_px
                    c0 = arr_off + n_idx * pitch_px
                    r1 = r0 + self.pad_sz
                    c1 = c0 + self.pad_sz
                    if r1 <= self.obj_size and c1 <= self.obj_size:
                        S[r0:r1, c0:c1] = True
            return S, arr_off

    def _build_candidate_supports(self) -> Dict[int, Tuple[np.ndarray, int]]:
            """Return dict mapping pitch → (support_mask, offset) for valid candidates."""
            supports = {}
            for p in self.pitch_candidates:
                S, arr_off = self._make_support(p)
                if S is not None:
                    supports[p] = (S, arr_off)
            return supports

    # ------------------------------------------------------------------
    # Probe initialisation helpers
    # ------------------------------------------------------------------

    def _init_probe_gaussian(self) -> np.ndarray:
            """Create a Gaussian probe of shape (probe_size, probe_size)."""
            ps = self.probe_size
            yy, xx = np.mgrid[:ps, :ps]
            yy = yy - ps / 2.0
            xx = xx - ps / 2.0
            P = np.exp(-(xx**2 + yy**2) / (2.0 * self.sigma_probe**2)).astype(np.complex128)
            P /= np.max(np.abs(P))
            return P

    def _init_probe_data_driven(self, diff_amp: np.ndarray) -> np.ndarray:
            """Create a data-driven probe via ifft2 of mean diffraction amplitude."""
            mean_amp = np.mean(diff_amp, axis=0)
            P = np.fft.ifft2(mean_amp)
            P = np.fft.fftshift(P)
            P = P.astype(np.complex128)
            max_abs = np.max(np.abs(P))
            if max_abs > 0:
                P /= max_abs
            return P

    # ------------------------------------------------------------------
    # Object initialisation
    # ------------------------------------------------------------------

    def _init_object(self, support: np.ndarray) -> np.ndarray:
            """Initialise complex object: R_Cu inside support, R_p outside."""
            O = np.full((self.obj_size, self.obj_size), self.R_p, dtype=np.complex128)
            O[support] = self.R_Cu
            return O

    # ------------------------------------------------------------------
    # Diffraction amplitude preprocessing
    # ------------------------------------------------------------------

    def _prepare_diff_amp(
            self, diff_amp_raw: np.ndarray, apply_ifftshift: bool
        ) -> np.ndarray:
            """Optionally apply ifftshift to move DC to corner for fft2 convention."""
            if apply_ifftshift:
                return np.fft.ifftshift(diff_amp_raw, axes=(-2, -1))
            else:
                return diff_amp_raw.copy()

    def _detect_dc_centered(self, intensities: np.ndarray) -> bool:
            """Heuristic: compare centre vs corner energy to decide FFT convention."""
            ps = intensities.shape[-1]
            half = ps // 2
            hw = 32  # half-width of region to check

            center_region_r = slice(half - hw, half + hw)
            center_region_c = slice(half - hw, half + hw)
            center_energy = np.mean(intensities[:, center_region_r, center_region_c])

            corner_energy = (
                np.mean(intensities[:, :hw, :hw]) +
                np.mean(intensities[:, :hw, -hw:]) +
                np.mean(intensities[:, -hw:, :hw]) +
                np.mean(intensities[:, -hw:, -hw:])
            ) / 4.0

            return bool(center_energy > corner_energy)

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Compute predicted diffraction amplitudes for all scan positions.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Current object estimate.
            P : np.ndarray, shape (probe_size, probe_size), complex128
                Current probe estimate.
            positions : np.ndarray, shape (n_pos, 2), int
                Scan positions (row, col).

            Returns
            -------
            np.ndarray, shape (n_pos, probe_size, probe_size), float64
                Predicted diffraction amplitudes |F{P · O_patch}|.
            """
            n_pos = positions.shape[0]
            ps = self.probe_size
            pred_amp = np.empty((n_pos, ps, ps), dtype=np.float64)

            for idx in range(n_pos):
                ry, rx = positions[idx]
                O_patch = O[ry:ry + ps, rx:rx + ps]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                pred_amp[idx] = np.abs(Psi)

            return pred_amp

    # ------------------------------------------------------------------
    # Fourier residual
    # ------------------------------------------------------------------

    def _compute_residual(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
        ) -> float:
            """Sum of squared differences between predicted and measured amplitudes."""
            ps = self.probe_size
            residual = 0.0
            for idx in range(positions.shape[0]):
                ry, rx = positions[idx]
                O_patch = O[ry:ry + ps, rx:rx + ps]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                residual += np.sum((np.abs(Psi) - diff_amp[idx]) ** 2)
            return float(residual)

    # ------------------------------------------------------------------
    # rpPIE epoch (core inner loop)
    # ------------------------------------------------------------------

    def _run_epoch(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
            epoch: int,
            n_epochs: int,
            update_probe: bool,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Execute one full rpPIE epoch (all scan positions + prior enforcement).

            Parameters
            ----------
            O : np.ndarray — current object (modified in-place and returned).
            P : np.ndarray — current probe (modified in-place and returned).
            diff_amp : np.ndarray — measured diffraction amplitudes.
            positions : np.ndarray — scan positions.
            epoch : int — current epoch index.
            n_epochs : int — total planned epochs.
            update_probe : bool — whether to update the probe this epoch.

            Returns
            -------
            (O, P) after one epoch of rpPIE updates.
            """
            ps = self.probe_size
            eps = self.eps
            beta_obj = self.beta_obj
            alpha = self.alpha
            beta_probe = self.beta_probe
            alpha_probe = self.alpha_probe
            n_pos = positions.shape[0]

            # Random scan order for better convergence
            order = np.random.permutation(n_pos)

            # Precompute probe-derived weights for object update
            P_abs = np.abs(P)
            P_max = np.max(P_abs) + eps
            P_conj = np.conj(P)
            P_abs_sq = P_abs ** 2
            obj_weight = (P_abs / P_max) * P_conj / (P_abs_sq + alpha)

            for pos_count, idx in enumerate(order):
                ry, rx = positions[idx]

                O_patch = O[ry:ry + ps, rx:rx + ps]

                # Exit wave
                psi = P * O_patch

                # Forward propagation
                Psi = np.fft.fft2(psi)

                # Standard magnitude projection (Fourier constraint)
                Psi_amp = np.abs(Psi) + eps
                Psi_c = diff_amp[idx] * Psi / Psi_amp

                # Inverse propagation
                psi_c = np.fft.ifft2(Psi_c)

                # Difference
                dpsi = psi_c - psi

                # Object update (ePIE / rpPIE Eq. 6 Step 1)
                O[ry:ry + ps, rx:rx + ps] = O_patch + beta_obj * obj_weight * dpsi

                # Probe update
                if update_probe:
                    O_abs = np.abs(O_patch)
                    O_max = np.max(O_abs) + eps
                    O_conj = np.conj(O_patch)
                    O_abs_sq = O_abs ** 2
                    probe_weight = (O_abs / O_max) * O_conj / (O_abs_sq + alpha_probe)
                    P = P + beta_probe * probe_weight * dpsi

                    # Recompute probe-derived weights periodically
                    if (pos_count + 1) % self.probe_weight_update_interval == 0:
                        P_abs = np.abs(P)
                        P_max = np.max(P_abs) + eps
                        P_conj = np.conj(P)
                        P_abs_sq = P_abs ** 2
                        obj_weight = (P_abs / P_max) * P_conj / (P_abs_sq + alpha)

            # --- End-of-epoch constraints ---

            # rpPIE reflectivity prior (Eq. 6 Step 2)
            beta_prior = self._get_beta_prior(epoch, n_epochs)
            if self.outside_support is not None:
                O[self.outside_support] = (
                    beta_prior * O[self.outside_support]
                    + (1.0 - beta_prior) * self.R_p
                )

            # Amplitude constraint: |O| should not exceed 1.0 (physical bound)
            amp = np.abs(O)
            mask_large = amp > 1.0
            if np.any(mask_large):
                O[mask_large] *= 1.0 / amp[mask_large]

            # Probe energy normalization (prevent divergence)
            if not hasattr(self, '_P_energy_ref') or self._P_energy_ref is None:
                self._P_energy_ref = np.sum(np.abs(P) ** 2)
            else:
                P_energy = np.sum(np.abs(P) ** 2)
                if P_energy > 2.0 * self._P_energy_ref:
                    P *= np.sqrt(self._P_energy_ref / P_energy)

            return O, P

    # ------------------------------------------------------------------
    # Prior schedule
    # ------------------------------------------------------------------

    def _get_beta_prior(self, epoch: int, n_epochs: int) -> float:
            """Linear ramp from beta_prior_start → beta_prior_end over first half."""
            half = max(n_epochs // 2, 1)
            if epoch < half:
                return self.beta_prior_start - (self.beta_prior_start - self.beta_prior_end) * (epoch / half)
            return self.beta_prior_end

    # ------------------------------------------------------------------
    # Calibration: select best FFT convention, probe, and pitch
    # ------------------------------------------------------------------

    def _calibrate(
            self,
            diff_amp_raw: np.ndarray,
            supports: Dict[int, Tuple[np.ndarray, int]],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            Run short calibration sweeps to choose:
              - FFT shift convention
              - Probe initialisation (Gaussian vs data-driven)
              - Support mask / pitch

            Returns
            -------
            (diff_amp_work, P_best, S_best, outside_S_best)
            """
            # Detect DC location
            intensities = diff_amp_raw ** 2  # approximate intensities for DC detection
            dc_centered = self._detect_dc_centered(intensities)

            # Prepare both FFT conventions
            diff_amp_A = self._prepare_diff_amp(diff_amp_raw, dc_centered)       # shift if DC centered
            diff_amp_B = self._prepare_diff_amp(diff_amp_raw, not dc_centered)   # opposite

            # Probe candidates
            P_gauss = self._init_probe_gaussian()
            P_data = self._init_probe_data_driven(diff_amp_raw)
            probe_candidates = [P_gauss, P_data]

            # Pick a primary pitch for initial convention/probe selection
            primary_pitch = 30 if 30 in supports else self.pitch_candidates[0]
            if primary_pitch not in supports:
                # fallback to first available
                primary_pitch = list(supports.keys())[0]

            S_primary, _ = supports[primary_pitch]

            best_residual = np.inf
            best_conv_idx = 0
            best_probe_idx = 0
            best_pitch = primary_pitch

            # Phase 1: Test all conventions × probes with primary pitch
            for conv_idx, da in enumerate([diff_amp_A, diff_amp_B]):
                for p_idx, P_init in enumerate(probe_candidates):
                    # Check time budget
                    if time.time() - self.t_start > self.time_budget_seconds - 10.0:
                        break

                    O_cal = self._init_object(S_primary)
                    P_cal = P_init.copy()

                    # Store support state for _run_epoch
                    self.outside_support = ~S_primary
                    self._P_energy_ref = None

                    for ep in range(self.calibration_epochs):
                        update_probe = (ep >= 1)  # allow minimal probe update in calibration
                        O_cal, P_cal = self._run_epoch(
                            O_cal, P_cal, da, self.positions, ep,
                            self.calibration_epochs, update_probe
                        )

                    res = self._compute_residual(O_cal, P_cal, da, self.positions)

                    if res < best_residual:
                        best_residual = res
                        best_conv_idx = conv_idx
                        best_probe_idx = p_idx
                        best_pitch = primary_pitch

            # Select best convention and probe
            diff_amp_work = [diff_amp_A, diff_amp_B][best_conv_idx]
            P_best = probe_candidates[best_probe_idx].copy()

            # Phase 2: Test other pitches with the best convention/probe
            for p_cand in self.pitch_candidates:
                if p_cand == best_pitch or p_cand not in supports:
                    continue

                # Check time budget
                if time.time() - self.t_start > self.time_budget_seconds - 8.0:
                    break

                S_cand, _ = supports[p_cand]
                O_cal = self._init_object(S_cand)
                P_cal = probe_candidates[best_probe_idx].copy()

                self.outside_support = ~S_cand
                self._P_energy_ref = None

                for ep in range(self.calibration_epochs):
                    update_probe = (ep >= 1)
                    O_cal, P_cal = self._run_epoch(
                        O_cal, P_cal, diff_amp_work, self.positions, ep,
                        self.calibration_epochs, update_probe
                    )

                res = self._compute_residual(O_cal, P_cal, diff_amp_work, self.positions)

                if res < best_residual:
                    best_residual = res
                    best_pitch = p_cand

            # Final selections
            S_best, _ = supports[best_pitch]
            outside_S_best = ~S_best

            # Reset probe energy reference for main loop
            self._P_energy_ref = None

            return diff_amp_work, P_best, S_best, outside_S_best

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _remove_global_phase_ambiguity(self, O: np.ndarray) -> np.ndarray:
            """Align substrate region phase to expected R_p phase."""
            if self.outside_support is None:
                return O

            substrate_vals = O[self.outside_support]
            expected_phase = np.angle(self.R_p)

            # Use median of substrate phase as reference
            measured_sub_phase = np.angle(substrate_vals)
            phase_offset = np.median(measured_sub_phase) - expected_phase
            O = O * np.exp(-1j * phase_offset)

            # Check if a π flip is needed (sign ambiguity)
            # After correction, substrate phase should be close to expected_phase
            # If it's ~π off, flip
            substrate_vals_corrected = O[self.outside_support]
            residual_phase = np.median(np.angle(substrate_vals_corrected)) - expected_phase
            if np.abs(residual_phase) > np.pi / 2:
                O = O * np.exp(-1j * np.pi)

            return O

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """
            Convert complex object to (2, obj_size, obj_size) float32 output.
            Channel 0: amplitude |O|, Channel 1: phase angle(O).
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)
            return output

    # ------------------------------------------------------------------
    # Main solve entry point
    # ------------------------------------------------------------------

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Full rpPIE reconstruction pipeline.

            Parameters
            ----------
            input_data : np.ndarray, shape (n_pos, probe_size, probe_size), float64
                Measured far-field diffraction intensities I(k).

            Returns
            -------
            np.ndarray, shape (2, obj_size, obj_size), float32
                Reconstructed object: [amplitude, phase].
            """
            self.t_start = time.time()

            # 1. Preprocessing: sqrt intensities, clip negatives
            I = np.maximum(input_data, 0).astype(np.float64)
            diff_amp = np.sqrt(I)

            # 2. Build scan positions
            self.positions = self._build_positions()

            # 3. Build candidate supports
            supports = self._build_candidate_supports()

            # 4. Calibrate (FFT convention, probe, pitch)
            diff_amp_work, P_best, S_best, outside_S_best = self._calibrate(diff_amp, supports)

            # Store best support
            self.support = S_best
            self.outside_support = outside_S_best
            self.diff_amp_work = diff_amp_work

            # 5. Initialise object and probe
            O = self._init_object(S_best)
            P = P_best.copy()
            self._P_energy_ref = None

            # 6. Main rpPIE loop (max_epochs, time-budgeted)
            time_elapsed = time.time() - self.t_start
            time_remaining = self.time_budget_seconds - time_elapsed
            # Estimate time per epoch: ~0.18s
            estimated_epoch_time = 0.18
            max_epochs_budget = min(int(time_remaining / estimated_epoch_time), self.max_epochs)
            n_epochs = max(max_epochs_budget, 50)

            for epoch in range(n_epochs):
                # Check time budget with safety margin
                if time.time() - self.t_start > self.time_budget_seconds - 1.0:
                    break

                update_probe = (epoch >= self.probe_update_start_epoch)
                O, P = self._run_epoch(
                    O, P, diff_amp_work, self.positions,
                    epoch, n_epochs, update_probe
                )

            # 7. Post-process (phase alignment)
            O = self._remove_global_phase_ambiguity(O)

            # 8. Format output
            output = self._format_output(O)
            return output



if __name__ == "__main__":
    # Load measured diffraction intensities
    input_data = np.load("dataset/input.npy")

    # Instantiate solver with default hyperparameters
    solver = InverseSolver()

    # Run full reconstruction
    result = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)