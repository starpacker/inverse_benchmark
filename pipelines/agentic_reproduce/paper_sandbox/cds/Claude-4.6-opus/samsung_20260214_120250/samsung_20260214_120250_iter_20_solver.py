import numpy as np
import time
from typing import Tuple, Optional, Dict, Any

class InverseSolver:
    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_side: int = 9,
            step: int = 40,
            wavelength: float = 13.5,
            pixel_size: float = 27.0,
            pad_pixels: int = 22,
            duty_cycle: float = 0.6,
            beta_O: float = 0.8,
            beta_P: float = 0.5,
            alpha: float = 0.1,
            probe_warmup_epochs: int = 5,
            probe_sigma: float = 50.0,
            object_init_amp: float = 0.85,
            phase_ramp_interval: int = 5,
            best_track_interval: int = 10,
            residual_subsample: int = 20,
            amp_percentile: float = 99.0,
            valid_amp_threshold: float = 0.1,
            well_lit_fraction: float = 0.3,
            support_threshold: float = 0.5,
            convention_test_epochs: int = 2,
            total_time_budget: float = 55.0,
            postprocess_reserve: float = 5.0,
            eps: float = 1e-12,
            n_sicn: complex = complex(0.9791, 0.0059),
            n_sio2: complex = complex(0.9780, 0.0108),
            n_si: complex = complex(0.9991, 0.0018),
            n_cu: complex = complex(0.9625, 0.0613),
        ) -> None:
            # Grid / geometry
            self.obj_size: int = obj_size
            self.probe_size: int = probe_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_pos: int = n_side * n_side
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.pad_pixels: int = pad_pixels
            self.duty_cycle: float = duty_cycle

            # Reconstruction hyperparameters
            self.beta_O: float = beta_O
            self.beta_P: float = beta_P
            self.alpha: float = alpha
            self.probe_warmup_epochs: int = probe_warmup_epochs
            self.probe_sigma: float = probe_sigma
            self.object_init_amp: float = object_init_amp

            # Maintenance intervals
            self.phase_ramp_interval: int = phase_ramp_interval
            self.best_track_interval: int = best_track_interval
            self.residual_subsample: int = residual_subsample

            # Post-processing
            self.amp_percentile: float = amp_percentile
            self.valid_amp_threshold: float = valid_amp_threshold
            self.well_lit_fraction: float = well_lit_fraction
            self.support_threshold: float = support_threshold

            # Timing
            self.convention_test_epochs: int = convention_test_epochs
            self.total_time_budget: float = total_time_budget
            self.postprocess_reserve: float = postprocess_reserve

            # Numerical
            self.eps: float = eps

            # Material optical constants (CXRO @ 13.5 nm)
            self.n_sicn: complex = n_sicn
            self.n_sio2: complex = n_sio2
            self.n_si: complex = n_si
            self.n_cu: complex = n_cu

            # Derived: scan positions
            self.positions: np.ndarray = self._build_positions()

            # State (populated during solve)
            self.O: Optional[np.ndarray] = None
            self.P: Optional[np.ndarray] = None
            self.scan_count: Optional[np.ndarray] = None
            self.illuminated: Optional[np.ndarray] = None
            self.convention: Optional[str] = None

            # Record start time for global budget tracking
            self.t_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _build_positions(self) -> np.ndarray:
            total_scan = self.probe_size + (self.n_side - 1) * self.step  # 576
            offset = (self.obj_size - total_scan) // 2  # 12

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions, dtype=np.int32)

    # ------------------------------------------------------------------
    # Probe initialisation
    # ------------------------------------------------------------------

    def _init_probe(self, diff_amp: np.ndarray) -> np.ndarray:
            ps = self.probe_size
            eps = self.eps

            # Gaussian probe
            yy, xx = np.mgrid[:ps, :ps]
            cy, cx = ps / 2.0, ps / 2.0
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            gauss = np.exp(-r2 / (2.0 * self.probe_sigma ** 2))

            P = gauss.astype(np.complex128)

            # Energy matching: scale probe so that |FFT(P * O_avg)|^2 ≈ mean measured intensity
            # Assume average object amplitude ~ object_init_amp
            mean_data_energy = np.mean(np.sum(diff_amp ** 2, axis=(-2, -1)))
            obj_amp_est = self.object_init_amp
            current_energy = ps * ps * np.sum(np.abs(P) ** 2) * obj_amp_est ** 2
            scale = np.sqrt(mean_data_energy / (current_energy + eps))
            P *= scale

            return P

    # ------------------------------------------------------------------
    # Object initialisation
    # ------------------------------------------------------------------

    def _init_object(self) -> np.ndarray:
            return self.object_init_amp * np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

    # ------------------------------------------------------------------
    # Forward model (single position)
    # ------------------------------------------------------------------

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Compute far-field diffraction amplitude for one scan position.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Current object estimate.
            P : np.ndarray, shape (probe_size, probe_size), complex128
                Current probe estimate.
            position : Tuple[int, int]
                (row, col) top-left corner of the probe window in the object.

            Returns
            -------
            np.ndarray, shape (probe_size, probe_size), complex128
                Far-field (Fourier-domain) exit wave Ψ(k).
            """
            ry, rx = position
            ps = self.probe_size
            O_patch = O[ry:ry + ps, rx:rx + ps]
            psi = P * O_patch
            Psi = np.fft.fft2(psi)
            return Psi

    # ------------------------------------------------------------------
    # Fourier magnitude replacement
    # ------------------------------------------------------------------

    def _replace_magnitude(
            self,
            Psi: np.ndarray,
            measured_amp: np.ndarray,
        ) -> np.ndarray:
            """Replace |Ψ| with measured amplitude, keep phase."""
            amp_Psi = np.abs(Psi)
            Psi_corrected = measured_amp * Psi / (amp_Psi + self.eps)
            return Psi_corrected

    # ------------------------------------------------------------------
    # Single-position rPIE update
    # ------------------------------------------------------------------

    def _update_position(
            self,
            O: np.ndarray,
            P: np.ndarray,
            measured_amp: np.ndarray,
            position: Tuple[int, int],
            epoch: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
            ry, rx = position
            ps = self.probe_size
            eps = self.eps
            alpha = self.alpha

            # Extract object patch
            O_patch = O[ry:ry + ps, rx:rx + ps]

            # Forward: exit wave and its Fourier transform
            psi = P * O_patch
            Psi = np.fft.fft2(psi)

            # Fourier magnitude replacement
            Psi_c = self._replace_magnitude(Psi, measured_amp)

            # Inverse FFT to get corrected exit wave
            psi_c = np.fft.ifft2(Psi_c)

            # Difference
            dpsi = psi_c - psi

            # rPIE object update: O += beta_O * conj(P) * dpsi / (|P|^2 + alpha * max|P|^2)
            P_abs2 = np.abs(P) ** 2
            P_max2 = np.max(P_abs2)
            denom_O = P_abs2 + alpha * P_max2
            O[ry:ry + ps, rx:rx + ps] += self.beta_O * np.conj(P) * dpsi / (denom_O + eps)

            # rPIE probe update after warmup epochs
            if epoch >= self.probe_warmup_epochs:
                O_patch_new = O[ry:ry + ps, rx:rx + ps]
                O_abs2 = np.abs(O_patch_new) ** 2
                O_max2 = np.max(O_abs2)
                denom_P = O_abs2 + alpha * O_max2
                P += self.beta_P * np.conj(O_patch_new) * dpsi / (denom_P + eps)

            return O, P

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    def _compute_residual(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            n_sample: int,
        ) -> float:
            """
            Compute normalised Fourier residual over n_sample positions.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Current object estimate.
            P : np.ndarray, shape (probe_size, probe_size), complex128
                Current probe estimate.
            diff_amp : np.ndarray, shape (n_pos, probe_size, probe_size)
                Measured diffraction amplitudes.
            n_sample : int
                Number of positions to subsample for residual computation.

            Returns
            -------
            float
                Normalised residual: sum(|.|Psi| - measured|^2) / sum(measured^2).
            """
            n_sample = min(n_sample, self.n_pos)
            residual = 0.0
            norm = 0.0
            ps = self.probe_size

            for j in range(n_sample):
                ry, rx = self.positions[j]
                O_patch = O[ry:ry + ps, rx:rx + ps]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                residual += np.sum((np.abs(Psi) - diff_amp[j]) ** 2)
                norm += np.sum(diff_amp[j] ** 2)

            return residual / (norm + self.eps)

    # ------------------------------------------------------------------
    # Support / illumination mask
    # ------------------------------------------------------------------

    def _build_illumination_mask(self) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute scan count array and boolean illuminated mask.

            Returns
            -------
            scan_count : np.ndarray, shape (obj_size, obj_size), float64
                Number of times each pixel is illuminated across all scan positions.
            illuminated : np.ndarray, shape (obj_size, obj_size), bool
                True where scan_count > 0.
            """
            scan_count = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            ps = self.probe_size
            for j in range(self.n_pos):
                ry, rx = self.positions[j]
                scan_count[ry:ry + ps, rx:rx + ps] += 1.0
            illuminated = scan_count > 0
            return scan_count, illuminated

    def _build_support_mask(self, O: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """
            Detect support from amplitude pattern — periodic pad array.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Current object estimate.

            Returns
            -------
            illuminated : np.ndarray, shape (obj_size, obj_size), bool
                True where the object is illuminated by at least one scan position.
            pad_mask : Optional[np.ndarray], shape (obj_size, obj_size), bool or None
                True where amplitude exceeds threshold (pad regions), or None if
                amplitude is too low to determine.
            """
            scan_count, illuminated = self._build_illumination_mask()

            amp = np.abs(O) * illuminated
            amp_max = np.max(amp)
            if amp_max < self.eps:
                return illuminated, None

            # Threshold to find pads (high reflectivity regions)
            amp_norm = amp / (amp_max + self.eps)
            pad_mask = amp_norm > self.support_threshold

            return illuminated, pad_mask

    # ------------------------------------------------------------------
    # Phase ramp removal
    # ------------------------------------------------------------------

    def _remove_phase_ramp(
            self,
            O: np.ndarray,
            illuminated: np.ndarray,
        ) -> np.ndarray:
            """Remove linear phase ramp from reconstruction via weighted least-squares.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Current object estimate.
            illuminated : np.ndarray, shape (obj_size, obj_size), bool
                Mask of illuminated pixels.

            Returns
            -------
            np.ndarray, shape (obj_size, obj_size), complex128
                Object with linear phase ramp subtracted.
            """
            amp = np.abs(O)
            phase = np.angle(O)
            eps = self.eps
            obj_size = self.obj_size

            # Select valid pixels: illuminated and with sufficient amplitude
            amp_max = np.max(amp)
            if amp_max < eps:
                return O

            valid = illuminated & (amp > amp_max * 0.2)
            if np.sum(valid) < 100:
                return O

            yy, xx = np.mgrid[:obj_size, :obj_size]
            y_valid = yy[valid].astype(np.float64)
            x_valid = xx[valid].astype(np.float64)
            ph_valid = phase[valid].astype(np.float64)
            w_valid = amp[valid].astype(np.float64)

            # Weighted least squares: phase = a*y + b*x + c
            # Weights = amplitude^2
            W = w_valid ** 2
            Sw = np.sum(W)
            Sy = np.sum(W * y_valid)
            Sx = np.sum(W * x_valid)
            Syy = np.sum(W * y_valid ** 2)
            Sxx = np.sum(W * x_valid ** 2)
            Sxy = np.sum(W * y_valid * x_valid)
            Sp = np.sum(W * ph_valid)
            Spy = np.sum(W * ph_valid * y_valid)
            Spx = np.sum(W * ph_valid * x_valid)

            A = np.array([
                [Syy, Sxy, Sy],
                [Sxy, Sxx, Sx],
                [Sy,  Sx,  Sw],
            ], dtype=np.float64)
            b = np.array([Spy, Spx, Sp], dtype=np.float64)

            try:
                coeffs = np.linalg.solve(A, b)
                ramp = coeffs[0] * yy.astype(np.float64) + coeffs[1] * xx.astype(np.float64) + coeffs[2]
                O_corrected = amp * np.exp(1j * (phase - ramp))
                return O_corrected
            except np.linalg.LinAlgError:
                return O

    # ------------------------------------------------------------------
    # Global phase removal
    # ------------------------------------------------------------------

    def _remove_global_phase(
            self,
            O: np.ndarray,
            well_lit: np.ndarray,
        ) -> np.ndarray:
            """Rotate O so mean phase in well-lit region ≈ 0.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Current object estimate.
            well_lit : np.ndarray, shape (obj_size, obj_size), bool
                Mask of well-illuminated pixels.

            Returns
            -------
            np.ndarray, shape (obj_size, obj_size), complex128
                Object with global phase removed.
            """
            amp = np.abs(O)
            eps = self.eps

            # Select valid pixels: well-lit and with sufficient amplitude
            amp_max = np.max(amp[well_lit]) if np.any(well_lit) else np.max(amp)
            if amp_max < eps:
                return O

            valid = well_lit & (amp > amp_max * self.valid_amp_threshold)
            if np.sum(valid) < 100:
                return O

            # Compute amplitude-weighted mean phase
            # mean_phase = angle( sum( |O|^2 * O ) ) over valid pixels
            mean_phase = np.angle(np.sum(amp[valid] ** 2 * O[valid]))
            O_corrected = O * np.exp(-1j * mean_phase)

            return O_corrected

    # ------------------------------------------------------------------
    # Amplitude normalisation
    # ------------------------------------------------------------------

    def _normalise_amplitude(
            self,
            O: np.ndarray,
            illuminated: np.ndarray,
        ) -> np.ndarray:
            """Scale amplitude to [0, 1] using percentile clipping within illuminated region.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Current object estimate.
            illuminated : np.ndarray, shape (obj_size, obj_size), bool
                Mask of illuminated pixels.

            Returns
            -------
            np.ndarray, shape (obj_size, obj_size), complex128
                Object with amplitude normalised to [0, 1].
            """
            eps = self.eps
            amp = np.abs(O)

            if not np.any(illuminated) or np.max(amp[illuminated]) < eps:
                return O

            amp_max = np.percentile(amp[illuminated], self.amp_percentile)
            if amp_max < eps:
                return O

            # Scale so that the percentile value maps to 1.0
            scale = 1.0 / (amp_max + eps)
            O_scaled = O * scale

            # Clip amplitude to [0, 1] while preserving phase
            amp_scaled = np.abs(O_scaled)
            phase = np.angle(O_scaled)
            amp_clipped = np.clip(amp_scaled, 0.0, 1.0)
            O_normalised = amp_clipped * np.exp(1j * phase)

            return O_normalised

    # ------------------------------------------------------------------
    # FFT convention detection
    # ------------------------------------------------------------------

    def _detect_convention(
            self,
            diff_amp_raw: np.ndarray,
        ) -> np.ndarray:
            """
            Run a short rPIE trial with both raw and ifftshifted data,
            return the diffraction amplitude array that yields lower residual.

            Parameters
            ----------
            diff_amp_raw : np.ndarray, shape (n_pos, probe_size, probe_size)
                Square-root of measured intensities (raw ordering).

            Returns
            -------
            np.ndarray
                The diffraction amplitude array with the winning convention.
            """
            n_test_epochs = self.convention_test_epochs
            eps = self.eps

            # Prepare two conventions
            damp_A = diff_amp_raw.copy()  # raw (DC at corner if data is FFT-native)
            damp_B = np.fft.ifftshift(diff_amp_raw, axes=(-2, -1))  # shift DC from center to corner

            def _quick_trial(damp, n_epochs):
                """Run n_epochs of ePIE-like updates, return residual."""
                P = self._init_probe(damp)
                O = self._init_object()
                beta_O = self.beta_O
                alpha = self.alpha
                ps = self.probe_size

                for ep in range(n_epochs):
                    order = np.random.permutation(self.n_pos)
                    P_abs2 = np.abs(P) ** 2
                    P_max2 = np.max(P_abs2)
                    denom_O = P_abs2 + alpha * P_max2

                    for idx in range(self.n_pos):
                        j = order[idx]
                        ry, rx = self.positions[j]
                        O_patch = O[ry:ry + ps, rx:rx + ps]
                        psi = P * O_patch
                        Psi = np.fft.fft2(psi)
                        Psi_c = damp[j] * Psi / (np.abs(Psi) + eps)
                        psi_c = np.fft.ifft2(Psi_c)
                        dpsi = psi_c - psi
                        O[ry:ry + ps, rx:rx + ps] += beta_O * np.conj(P) * dpsi / (denom_O + eps)

                # Compute residual over all positions
                return self._compute_residual(O, P, damp, self.n_pos)

            res_A = _quick_trial(damp_A, n_test_epochs)
            res_B = _quick_trial(damp_B, n_test_epochs)

            if res_B < res_A:
                self.convention = "B_shifted"
                return damp_B
            else:
                self.convention = "A_raw"
                return damp_A

    # ------------------------------------------------------------------
    # Core rPIE loop
    # ------------------------------------------------------------------

    def _run_rpie(
            self,
            diff_amp: np.ndarray,
            time_budget: float,
        ) -> Tuple[np.ndarray, np.ndarray, float, int]:
            """
            Full rPIE reconstruction with rpPIE support prior.

            Parameters
            ----------
            diff_amp : np.ndarray, shape (n_pos, probe_size, probe_size)
                Diffraction amplitudes (correct FFT convention).
            time_budget : float
                Wall-clock seconds available.

            Returns
            -------
            best_O : np.ndarray, shape (obj_size, obj_size), complex128
            best_P : np.ndarray, shape (probe_size, probe_size), complex128
            best_residual : float
            n_epochs : int
            """
            P = self._init_probe(diff_amp)
            O = self._init_object()

            ps = self.probe_size
            eps = self.eps
            alpha = self.alpha
            beta_O = self.beta_O
            beta_P = self.beta_P

            # Build illumination mask once
            scan_count, illuminated = self._build_illumination_mask()

            best_O = O.copy()
            best_P = P.copy()
            best_res = 1e10
            epoch = 0

            t0 = time.time()

            while True:
                elapsed = time.time() - t0
                if elapsed > time_budget:
                    break

                order = np.random.permutation(self.n_pos)
                time_exceeded = False

                for idx in range(self.n_pos):
                    # Check time every 27 positions
                    if idx % 27 == 0 and (time.time() - t0) > time_budget:
                        time_exceeded = True
                        break

                    j = order[idx]
                    ry, rx = self.positions[j]

                    # Extract object patch
                    O_patch = O[ry:ry + ps, rx:rx + ps]

                    # Forward: exit wave and its Fourier transform
                    psi = P * O_patch
                    Psi = np.fft.fft2(psi)

                    # Fourier magnitude replacement
                    amp_Psi = np.abs(Psi)
                    Psi_c = diff_amp[j] * Psi / (amp_Psi + eps)

                    # Inverse FFT to get corrected exit wave
                    psi_c = np.fft.ifft2(Psi_c)

                    # Difference
                    dpsi = psi_c - psi

                    # rPIE object update
                    P_abs2 = np.abs(P) ** 2
                    P_max2 = np.max(P_abs2)
                    denom_O = P_abs2 + alpha * P_max2
                    O[ry:ry + ps, rx:rx + ps] += beta_O * np.conj(P) * dpsi / (denom_O + eps)

                    # rPIE probe update after warmup epochs
                    if epoch >= self.probe_warmup_epochs:
                        O_patch_new = O[ry:ry + ps, rx:rx + ps]
                        O_abs2 = np.abs(O_patch_new) ** 2
                        O_max2 = np.max(O_abs2)
                        denom_P = O_abs2 + alpha * O_max2
                        P += beta_P * np.conj(O_patch_new) * dpsi / (denom_P + eps)

                epoch += 1

                if time_exceeded:
                    break

                # Phase ramp removal and support constraint at regular intervals
                if epoch % self.phase_ramp_interval == 0:
                    O = self._remove_phase_ramp(O, illuminated)
                    # Zero outside illuminated region
                    O[~illuminated] = 0.0

                # Track best reconstruction by residual
                if epoch % self.best_track_interval == 0 or (time.time() - t0) > time_budget * 0.95:
                    res = self._compute_residual(O, P, diff_amp, self.residual_subsample)
                    if res < best_res:
                        best_res = res
                        best_O = O.copy()
                        best_P = P.copy()

            # Final residual check
            final_res = self._compute_residual(O, P, diff_amp, self.residual_subsample)
            if final_res < best_res:
                best_res = final_res
                best_O = O.copy()
                best_P = P.copy()

            return best_O, best_P, best_res, epoch

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(self, O: np.ndarray) -> np.ndarray:
            """
            Final post-processing: phase ramp removal, global phase removal,
            amplitude normalisation, masking.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128

            Returns
            -------
            np.ndarray, shape (2, obj_size, obj_size), float32
                Channel 0: amplitude |O|, Channel 1: phase angle(O).
            """
            # Build illumination mask
            scan_count, illuminated = self._build_illumination_mask()

            # Phase ramp removal
            O = self._remove_phase_ramp(O, illuminated)

            # Global phase removal using well-lit region
            well_lit = scan_count >= np.max(scan_count) * self.well_lit_fraction
            O = self._remove_global_phase(O, well_lit)

            # Amplitude normalisation to [0, 1]
            O = self._normalise_amplitude(O, illuminated)

            # Zero outside illuminated region
            O[~illuminated] = 0.0

            # Build output array
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)

            return output

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            End-to-end solver: convention detection → rPIE → post-processing.

            Parameters
            ----------
            input_data : np.ndarray, shape (n_pos, probe_size, probe_size)
                Measured far-field diffraction intensities I(k).

            Returns
            -------
            np.ndarray, shape (2, obj_size, obj_size), float32
                Reconstructed object: [amplitude, phase].
            """
            self.t_start = time.time()

            # 1. Preprocess: clamp negatives, sqrt → diff_amp
            I = np.maximum(input_data, 0).astype(np.float64)
            diff_amp_raw = np.sqrt(I)

            # 2. Detect FFT convention (quick test with both conventions)
            diff_amp = self._detect_convention(diff_amp_raw)

            # 3. Run rPIE with remaining time budget
            time_elapsed = time.time() - self.t_start
            time_remaining = self.total_time_budget - time_elapsed - self.postprocess_reserve
            time_remaining = max(time_remaining, 1.0)  # at least 1 second

            best_O, best_P, best_res, n_epochs = self._run_rpie(diff_amp, time_remaining)

            # Store state
            self.O = best_O
            self.P = best_P

            # 4. Post-process and return (2, H, W) float32 array
            output = self._postprocess(best_O)

            return output



if __name__ == "__main__":
    t_start = time.time()

    # Load data
    input_data = np.load("dataset/input.npy")

    # Instantiate solver
    solver = InverseSolver(total_time_budget=55.0)

    # Solve
    result = solver.solve(input_data)

    # Save
    np.save("output.npy", result)

    print(f"Total wall time: {time.time() - t_start:.1f}s, output shape: {result.shape}")
    print(f"Convention: {solver.convention}")
    print(f"Amplitude range: [{result[0].min():.4f}, {result[0].max():.4f}]")
    print(f"Phase range: [{result[1].min():.4f}, {result[1].max():.4f}]")