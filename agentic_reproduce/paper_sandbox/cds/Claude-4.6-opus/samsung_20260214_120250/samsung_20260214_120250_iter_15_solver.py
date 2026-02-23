import numpy as np
import time
from typing import Tuple, Optional, List

class InverseSolver:
    def __init__(
            self,
            total_time: float = 55.0,
            detect_time: float = 12.0,
            safety_margin: float = 2.0,
            patch_size: int = 256,
            obj_size: int = 600,
            n_side: int = 9,
            step: int = 40,
            n_pos: int = 81,
            eps: float = 1e-12,
            alpha_rPIE: float = 0.1,
            probe_sigma: float = 60.0,
            obj_init_amp: float = 0.5,
            amp_clip_max: float = 1.0,
            residual_n_sample: int = 15,
            phase_hist_bins: int = 360,
            phase_hist_kernel_size: int = 15,
            amp_valid_threshold: float = 0.05,
            time_check_interval: int = 20,
        ) -> None:
            self.total_time = total_time
            self.detect_time = detect_time
            self.safety_margin = safety_margin
            self.patch_size = patch_size
            self.obj_size = obj_size
            self.n_side = n_side
            self.step = step
            self.n_pos = n_pos
            self.eps = eps
            self.alpha_rPIE = alpha_rPIE
            self.probe_sigma = probe_sigma
            self.obj_init_amp = obj_init_amp
            self.amp_clip_max = amp_clip_max
            self.residual_n_sample = residual_n_sample
            self.phase_hist_bins = phase_hist_bins
            self.phase_hist_kernel_size = phase_hist_kernel_size
            self.amp_valid_threshold = amp_valid_threshold
            self.time_check_interval = time_check_interval

    def _build_scan_positions(self) -> np.ndarray:
            """
            Compute the (n_pos, 2) array of (row, col) scan positions
            on a regular n_side x n_side grid with given step and centering offset.

            Returns:
                positions: np.ndarray of shape (n_pos, 2), dtype int32
            """
            total_scan = self.patch_size + (self.n_side - 1) * self.step
            offset = (self.obj_size - total_scan) // 2

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions, dtype=np.int32)

    def _build_probe_window(self) -> np.ndarray:
            """
            Build a Gaussian support window for the probe in FFT-native layout.

            Returns:
                probe_window_fft: np.ndarray of shape (patch_size, patch_size), real-valued
            """
            ps = self.patch_size
            yy, xx = np.mgrid[:ps, :ps]
            cy, cx = ps / 2.0, ps / 2.0
            r2 = (yy - cy)**2 + (xx - cx)**2
            probe_window = np.exp(-r2 / (2.0 * self.probe_sigma**2))
            probe_window_fft = np.fft.ifftshift(probe_window)
            return probe_window_fft

    def _init_probe(self, diff_amp: np.ndarray, probe_window_fft: np.ndarray) -> np.ndarray:
            """
            Initialize the probe function from average diffraction amplitudes.

            Args:
                diff_amp: np.ndarray of shape (n_pos, patch_size, patch_size), real
                probe_window_fft: np.ndarray of shape (patch_size, patch_size), real

            Returns:
                P: np.ndarray of shape (patch_size, patch_size), complex128
            """
            avg_amp = np.mean(diff_amp, axis=0)  # (patch_size, patch_size)
            P = np.fft.ifft2(avg_amp)  # complex (patch_size, patch_size)
            P *= probe_window_fft
            P_max = np.max(np.abs(P))
            if P_max > self.eps:
                P = P / P_max
            return P.astype(np.complex128)

    def _init_object(self) -> np.ndarray:
            """
            Initialize the object as a uniform complex reflector.

            Returns:
                O: np.ndarray of shape (obj_size, obj_size), complex128
            """
            return np.full((self.obj_size, self.obj_size), self.obj_init_amp + 0j, dtype=np.complex128)

    def _run_rPIE(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
            probe_window_fft: np.ndarray,
            time_budget: float,
        ) -> Tuple[np.ndarray, np.ndarray, int]:
            t0 = time.time()
            n_pos = len(positions)
            ps = self.patch_size
            eps = self.eps
            alpha = self.alpha_rPIE
            epoch = 0

            while True:
                elapsed = time.time() - t0
                if elapsed > time_budget:
                    break

                # Randomize position ordering
                order = np.random.permutation(n_pos)

                for idx in range(n_pos):
                    # Time check periodically
                    if idx % self.time_check_interval == 0 and (time.time() - t0) > time_budget:
                        break

                    j = order[idx]
                    ry, rx = positions[j]

                    # Extract object patch
                    O_patch = O[ry:ry + ps, rx:rx + ps].copy()

                    # Exit wave
                    psi = P * O_patch

                    # Forward propagate
                    Psi = np.fft.fft2(psi)

                    # Fourier magnitude replacement using angle to avoid division by zero
                    Psi_corrected = diff_amp[j] * np.exp(1j * np.angle(Psi))

                    # Inverse propagate
                    psi_c = np.fft.ifft2(Psi_corrected)

                    # Difference
                    dpsi = psi_c - psi

                    # rPIE Object update (pixel-wise normalization)
                    P_abs2 = np.abs(P) ** 2
                    P_max2 = np.max(P_abs2)
                    denom_O = (1.0 - alpha) * P_abs2 + alpha * P_max2 + eps
                    O[ry:ry + ps, rx:rx + ps] += np.conj(P) * dpsi / denom_O

                    # Object amplitude clipping (physical constraint: reflectance <= 1)
                    O_patch_new = O[ry:ry + ps, rx:rx + ps]
                    amp_patch = np.abs(O_patch_new)
                    mask_clip = amp_patch > self.amp_clip_max
                    if np.any(mask_clip):
                        O_patch_new[mask_clip] *= self.amp_clip_max / amp_patch[mask_clip]
                        O[ry:ry + ps, rx:rx + ps] = O_patch_new

                    # rPIE Probe update (pixel-wise normalization)
                    O_abs2 = np.abs(O_patch) ** 2
                    O_max2 = np.max(O_abs2)
                    denom_P = (1.0 - alpha) * O_abs2 + alpha * O_max2 + eps
                    P += np.conj(O_patch) * dpsi / denom_P

                # Probe support constraint: re-apply window every epoch
                P *= probe_window_fft
                # Re-normalize probe
                P_maxval = np.max(np.abs(P))
                if P_maxval > eps:
                    P = P / P_maxval

                epoch += 1

            return O, P, epoch

    def _compute_residual(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
        ) -> float:
            """
            Compute average Fourier amplitude residual on a random subset of positions.

            Args:
                O: np.ndarray of shape (obj_size, obj_size), complex128
                P: np.ndarray of shape (patch_size, patch_size), complex128
                diff_amp: np.ndarray of shape (n_pos, patch_size, patch_size)
                positions: np.ndarray of shape (n_pos, 2)

            Returns:
                residual: float — average squared amplitude error over sampled positions
            """
            n_pos = len(positions)
            n_sample = min(self.residual_n_sample, n_pos)
            subset = np.random.RandomState(42).choice(n_pos, n_sample, replace=False)
            ps = self.patch_size
            res = 0.0
            for j in subset:
                ry, rx = positions[j]
                psi = P * O[ry:ry + ps, rx:rx + ps]
                Psi = np.fft.fft2(psi)
                res += np.sum((np.abs(Psi) - diff_amp[j]) ** 2)
            return float(res / n_sample)

    def _detect_fft_convention(
            self,
            diff_amp_raw: np.ndarray,
            positions: np.ndarray,
            probe_window_fft: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
            """
            Test both FFT conventions (raw and ifftshifted diffraction amplitudes)
            with short rPIE runs and select the one with lower residual.

            Args:
                diff_amp_raw: np.ndarray of shape (n_pos, patch_size, patch_size) — sqrt(I)
                positions: np.ndarray of shape (n_pos, 2)
                probe_window_fft: np.ndarray of shape (patch_size, patch_size)

            Returns:
                best_O: np.ndarray of shape (obj_size, obj_size), complex128
                best_P: np.ndarray of shape (patch_size, patch_size), complex128
                best_damp: np.ndarray of shape (n_pos, patch_size, patch_size)
                convention: str — "raw" or "shifted"
            """
            half_budget = self.detect_time / 2.0

            # Convention A: raw (as-is)
            damp_a = diff_amp_raw.copy()
            P_a = self._init_probe(damp_a, probe_window_fft)
            O_a = self._init_object()
            O_a, P_a, ep_a = self._run_rPIE(O_a, P_a, damp_a, positions, probe_window_fft, time_budget=half_budget)
            res_a = self._compute_residual(O_a, P_a, damp_a, positions)

            # Convention B: ifftshift applied to diffraction amplitudes
            damp_b = np.fft.ifftshift(diff_amp_raw, axes=(-2, -1))
            P_b = self._init_probe(damp_b, probe_window_fft)
            O_b = self._init_object()
            O_b, P_b, ep_b = self._run_rPIE(O_b, P_b, damp_b, positions, probe_window_fft, time_budget=half_budget)
            res_b = self._compute_residual(O_b, P_b, damp_b, positions)

            # Pick the convention with lower residual
            if res_a <= res_b:
                return O_a, P_a, damp_a, "raw"
            else:
                return O_b, P_b, damp_b, "shifted"

    def _build_scan_coverage_mask(self, positions: np.ndarray) -> np.ndarray:
            """
            Build a boolean mask indicating which object pixels are covered by at least
            one scan position.

            Args:
                positions: np.ndarray of shape (n_pos, 2)

            Returns:
                scan_mask: np.ndarray of shape (obj_size, obj_size), bool
            """
            scan_mask = np.zeros((self.obj_size, self.obj_size), dtype=bool)
            ps = self.patch_size
            for j in range(len(positions)):
                ry, rx = positions[j]
                scan_mask[ry:ry + ps, rx:rx + ps] = True
            return scan_mask

    def _align_global_phase(self, O: np.ndarray, scan_mask: np.ndarray) -> np.ndarray:
            """
            Remove global phase offset by finding the dominant (mode) phase in the
            valid region and rotating the object accordingly.

            Args:
                O: np.ndarray of shape (obj_size, obj_size), complex128
                scan_mask: np.ndarray of shape (obj_size, obj_size), bool

            Returns:
                O_aligned: np.ndarray of shape (obj_size, obj_size), complex128
            """
            amp_result = np.abs(O)
            phase_result = np.angle(O)

            # Identify valid region: within scan coverage and with sufficient amplitude
            valid = scan_mask & (amp_result > self.amp_valid_threshold)

            if np.sum(valid) > 100:
                # Find substrate phase (dominant/mode phase) via smoothed histogram
                phase_valid = phase_result[valid]
                n_bins = self.phase_hist_bins
                hist, bin_edges = np.histogram(phase_valid, bins=n_bins, range=(-np.pi, np.pi))

                # Smooth histogram to find robust mode
                kernel_size = self.phase_hist_kernel_size
                kernel = np.ones(kernel_size) / kernel_size
                hist_smooth = np.convolve(hist, kernel, mode='same')
                mode_idx = np.argmax(hist_smooth)
                phase_ref = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
            else:
                phase_ref = 0.0

            # Remove global phase offset
            O_aligned = O * np.exp(-1j * phase_ref)

            return O_aligned

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Simulate the forward ptychographic measurement model.

            For each scan position j:
                I_j(k) = |FFT2{ P(r - R_j) * O(r) }|^2

            Args:
                O: np.ndarray of shape (obj_size, obj_size), complex128 — object function
                P: np.ndarray of shape (patch_size, patch_size), complex128 — probe function
                positions: np.ndarray of shape (n_pos, 2) — scan positions (row, col)

            Returns:
                intensities: np.ndarray of shape (n_pos, patch_size, patch_size) — diffraction intensities
            """
            n_pos = len(positions)
            ps = self.patch_size
            intensities = np.zeros((n_pos, ps, ps), dtype=np.float64)

            for j in range(n_pos):
                ry, rx = positions[j]
                # Extract object patch at scan position j
                O_patch = O[ry:ry + ps, rx:rx + ps]
                # Compute exit wave: probe * object patch
                psi = P * O_patch
                # Forward propagate via FFT
                Psi = np.fft.fft2(psi)
                # Diffraction intensity is squared magnitude
                intensities[j] = np.abs(Psi) ** 2

            return intensities

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Full inverse reconstruction pipeline:
              1. Preprocess input intensities (clip negatives, compute sqrt)
              2. Build scan positions and probe window
              3. Detect FFT convention via short trial runs
              4. Run main rPIE reconstruction with remaining time budget
              5. Post-process: phase alignment, build output array

            Args:
                input_data: np.ndarray of shape (n_pos, patch_size, patch_size) — measured
                            diffraction intensities I(k), potentially noisy

            Returns:
                output: np.ndarray of shape (2, obj_size, obj_size), float32
                        channel 0: amplitude |O(r)|
                        channel 1: phase angle(O(r))
            """
            t_start = time.time()

            # Step 1: Preprocess input intensities
            I = np.maximum(input_data, 0).astype(np.float64)
            diff_amp = np.sqrt(I)  # (n_pos, patch_size, patch_size)

            # Step 2: Build scan positions and probe window
            positions = self._build_scan_positions()
            probe_window_fft = self._build_probe_window()

            # Step 3: Detect FFT convention via short trial runs
            best_O, best_P, best_damp, convention = self._detect_fft_convention(
                diff_amp, positions, probe_window_fft
            )

            # Step 4: Run main rPIE reconstruction with remaining time budget
            remaining_time = self.total_time - (time.time() - t_start) - self.safety_margin

            if remaining_time > 5.0:
                best_O, best_P, final_epochs = self._run_rPIE(
                    best_O, best_P, best_damp, positions, probe_window_fft,
                    time_budget=remaining_time
                )

            # Step 5: Post-process — phase alignment
            scan_mask = self._build_scan_coverage_mask(positions)
            best_O = self._align_global_phase(best_O, scan_mask)

            # Build output array
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(best_O).astype(np.float32)
            output[1] = np.angle(best_O).astype(np.float32)

            return output



if __name__ == "__main__":
    # Load measured diffraction intensity data
    input_data = np.load('dataset/input.npy')

    # Instantiate solver with default hyperparameters
    solver = InverseSolver(
        total_time=55.0,
        detect_time=12.0,
        safety_margin=2.0,
        patch_size=256,
        obj_size=600,
        n_side=9,
        step=40,
        n_pos=81,
        eps=1e-12,
        alpha_rPIE=0.1,
        probe_sigma=60.0,
        obj_init_amp=0.5,
        amp_clip_max=1.0,
        residual_n_sample=15,
        phase_hist_bins=360,
        phase_hist_kernel_size=15,
        amp_valid_threshold=0.05,
        time_check_interval=20,
    )

    # Run reconstruction
    result = solver.solve(input_data)

    # Save output: shape (2, 600, 600), channel 0 = amplitude, channel 1 = phase
    np.save('output.npy', result)