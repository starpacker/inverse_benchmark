import numpy as np
import time
from typing import Tuple, Optional, List

class InverseSolver:
    def __init__(
            self,
            patch_size: int = 256,
            obj_size: int = 600,
            n_side: int = 9,
            step: int = 40,
            n_pos: int = 81,
            eps: float = 1e-8,
            beta_obj: float = 0.9,
            beta_probe: float = 0.5,
            beta_mom: float = 0.2,
            use_momentum: bool = True,
            probe_sigma: float = 50.0,
            obj_init_amp: float = 1.0,
            amp_clip: float = 1.0,
            total_time_budget: float = 55.0,
            safety_margin: float = 2.0,
            well_illuminated_threshold: int = 4,
            phase_hist_bins: int = 720,
            phase_hist_kernel_size: int = 31,
            min_valid_pixels: int = 100,
            valid_amp_threshold: float = 0.3,
            time_check_interval: int = 27,
        ) -> None:
            # Geometry
            self.patch_size: int = patch_size
            self.obj_size: int = obj_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_pos: int = n_pos

            # Numerical stability
            self.eps: float = eps

            # ePIE step sizes
            self.beta_obj: float = beta_obj
            self.beta_probe: float = beta_probe
            self.beta_mom: float = beta_mom
            self.use_momentum: bool = use_momentum

            # Probe initialization
            self.probe_sigma: float = probe_sigma

            # Object initialization
            self.obj_init_amp: float = obj_init_amp
            self.amp_clip: float = amp_clip

            # Time management
            self.total_time_budget: float = total_time_budget
            self.safety_margin: float = safety_margin
            self.time_check_interval: int = time_check_interval

            # Post-processing
            self.well_illuminated_threshold: int = well_illuminated_threshold
            self.phase_hist_bins: int = phase_hist_bins
            self.phase_hist_kernel_size: int = phase_hist_kernel_size
            self.min_valid_pixels: int = min_valid_pixels
            self.valid_amp_threshold: float = valid_amp_threshold

            # Internal state (populated during solve)
            self.positions: Optional[np.ndarray] = None
            self.probe: Optional[np.ndarray] = None
            self.obj: Optional[np.ndarray] = None
            self.diff_amp_use: Optional[np.ndarray] = None
            self.t_start: Optional[float] = None

    def _compute_scan_positions(self) -> np.ndarray:
            """Compute the (n_pos, 2) array of (row, col) scan positions."""
            total_scan = self.patch_size + (self.n_side - 1) * self.step  # 256 + 320 = 576
            offset = (self.obj_size - total_scan) // 2  # (600 - 576) // 2 = 12

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions, dtype=np.int32)

    def _detect_fft_convention(self, diff_amp: np.ndarray) -> bool:
            """
            Detect whether diffraction data has DC at center (needs ifftshift)
            or at corner (standard FFT order).

            Args:
                diff_amp: Square root of measured intensities, shape (n_pos, ps, ps).

            Returns:
                True if data needs ifftshift (DC at center), False otherwise.
            """
            # Average intensity over first few patterns for robust estimation
            avg_pattern = np.mean(diff_amp[:10] ** 2, axis=0)
            h, w = avg_pattern.shape

            # Compare energy in center region vs corner regions
            c = 16  # half-size of comparison window

            # Center energy: 32x32 block around (h//2, w//2)
            center_energy = np.sum(avg_pattern[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c])

            # Corner energy: sum of 32x32 blocks at all four corners
            corner_energy = (
                np.sum(avg_pattern[:2 * c, :2 * c])
                + np.sum(avg_pattern[:2 * c, -2 * c:])
                + np.sum(avg_pattern[-2 * c:, :2 * c])
                + np.sum(avg_pattern[-2 * c:, -2 * c:])
            )

            # If DC (brightest region) is at center, we need ifftshift to move it to corner
            if center_energy > corner_energy:
                return True  # needs ifftshift
            else:
                return False  # already in FFT order

    def _validate_convention(
            self,
            diff_amp: np.ndarray,
            O: np.ndarray,
            P: np.ndarray,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Run quick mini-reconstruction with both conventions to validate heuristic.

            Args:
                diff_amp: Original diffraction amplitudes, shape (n_pos, ps, ps).
                O: Initial object estimate, shape (obj_size, obj_size).
                P: Initial probe estimate, shape (ps, ps).
                positions: Scan positions, shape (n_pos, 2).

            Returns:
                Diffraction amplitudes in correct FFT convention, shape (n_pos, ps, ps).
            """
            # Test with raw data (no shift)
            res_raw = self._quick_test(diff_amp, O.copy(), P.copy(), positions, max_iters=20)

            # Test with ifftshifted data (DC moved from center to corner)
            diff_amp_shifted = np.fft.ifftshift(diff_amp, axes=(-2, -1))
            res_shifted = self._quick_test(diff_amp_shifted, O.copy(), P.copy(), positions, max_iters=20)

            # Use heuristic as default
            needs_shift = self._detect_fft_convention(diff_amp)

            # Override heuristic if quick test disagrees strongly
            if res_raw < res_shifted * 0.5:
                return diff_amp.copy()
            elif res_shifted < res_raw * 0.5:
                return diff_amp_shifted
            else:
                # Keep heuristic decision
                if needs_shift:
                    return diff_amp_shifted
                else:
                    return diff_amp.copy()

    def _quick_test(
            self,
            damp: np.ndarray,
            O: np.ndarray,
            P: np.ndarray,
            positions: np.ndarray,
            max_iters: int = 20,
        ) -> float:
            """
            Run a small number of ePIE updates and return normalized residual.

            Args:
                damp: Diffraction amplitudes, shape (n_pos, ps, ps).
                O: Object estimate, shape (obj_size, obj_size).
                P: Probe estimate, shape (ps, ps).
                positions: Scan positions, shape (n_pos, 2).
                max_iters: Number of position updates to run.

            Returns:
                Normalized residual (float).
            """
            ps = self.patch_size
            eps = self.eps

            rng = np.random.RandomState(0)
            subset = rng.choice(len(positions), min(max_iters, len(positions)), replace=False)

            O_t = O.copy()
            P_abs2 = np.abs(P) ** 2

            for j in subset:
                ry, rx = positions[j]
                O_patch = O_t[ry:ry + ps, rx:rx + ps]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                amp_Psi = np.abs(Psi)
                Psi_c = damp[j] * Psi / (amp_Psi + eps)
                psi_c = np.fft.ifft2(Psi_c)
                dpsi = psi_c - psi
                O_t[ry:ry + ps, rx:rx + ps] = O_patch + 0.9 * np.conj(P) * dpsi / (P_abs2 + eps)

            # Compute residual on same subset
            res = 0.0
            norm = 0.0
            for j in subset:
                ry, rx = positions[j]
                psi = P * O_t[ry:ry + ps, rx:rx + ps]
                Psi = np.fft.fft2(psi)
                res += np.sum((np.abs(Psi) - damp[j]) ** 2)
                norm += np.sum(damp[j] ** 2)

            return res / (norm + eps)

    def _init_probe(self, diff_amp_use: np.ndarray) -> np.ndarray:
            """
            Initialize probe as Gaussian disk in real space, energy-matched to data.

            Args:
                diff_amp_use: Diffraction amplitudes in correct FFT order, shape (n_pos, ps, ps).

            Returns:
                Complex probe array, shape (ps, ps).
            """
            ps = self.patch_size
            eps = self.eps

            # Create coordinate grid centered on the patch
            yy, xx = np.mgrid[:ps, :ps]
            cy, cx = ps / 2.0, ps / 2.0
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2

            # Gaussian probe with specified sigma
            P = np.exp(-r2 / (2 * self.probe_sigma ** 2)).astype(np.complex128)

            # Scale probe to match average diffraction energy
            # By Parseval's theorem: sum|F{P*O}|^2 = N * sum|P*O|^2
            # With O ~ 1: sum|F{P}|^2 ~ sum(I_j) = sum(diff_amp^2)
            # And sum|F{P}|^2 = N * sum|P|^2 (Parseval)
            # So: N * sum|P|^2 ~ mean(sum(diff_amp^2))
            mean_energy = np.mean(np.sum(diff_amp_use ** 2, axis=(-2, -1)))
            current_energy = np.sum(np.abs(P) ** 2)
            P *= np.sqrt(mean_energy / (current_energy * ps * ps + eps))

            return P

    def _init_object(self) -> np.ndarray:
            """
            Initialize object as uniform complex array with given amplitude.

            Returns:
                Complex object array, shape (obj_size, obj_size).
            """
            return np.full((self.obj_size, self.obj_size), self.obj_init_amp, dtype=np.complex128)

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Compute forward model: far-field diffraction amplitude for one scan position.

            ψ = P · O_patch
            Ψ = FFT2(ψ)
            returns |Ψ|

            Args:
                O: Complex object, shape (obj_size, obj_size).
                P: Complex probe, shape (ps, ps).
                position: (row, col) top-left corner of patch in object.

            Returns:
                Diffraction amplitude |Ψ|, shape (ps, ps).
            """
            ry, rx = position
            ps = self.patch_size
            O_patch = O[ry:ry + ps, rx:rx + ps]
            psi = P * O_patch
            Psi = np.fft.fft2(psi)
            return np.abs(Psi)

    def _epie_single_position(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp_j: np.ndarray,
            position: Tuple[int, int],
            P_abs2: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Perform one ePIE update for a single scan position with PIXEL-WISE normalization.

            Object update:
                O(r) += beta_obj * conj(P) / (|P|^2 + eps) * dpsi

            Probe update (using updated O):
                P(r) += beta_probe * conj(O_new) / (|O_new|^2 + eps) * dpsi

            Args:
                O: Complex object (modified in-place), shape (obj_size, obj_size).
                P: Complex probe (modified in-place), shape (ps, ps).
                diff_amp_j: Measured diffraction amplitude for position j, shape (ps, ps).
                position: (row, col) top-left corner.
                P_abs2: Cached |P|^2, shape (ps, ps).

            Returns:
                Tuple of (updated O, updated P, updated P_abs2).
            """
            ry, rx = position
            ps = self.patch_size
            eps = self.eps

            # Extract object patch
            O_patch = O[ry:ry + ps, rx:rx + ps].copy()

            # Compute exit wave
            psi = P * O_patch

            # Forward propagate to far field
            Psi = np.fft.fft2(psi)

            # Fourier magnitude replacement
            amp_Psi = np.abs(Psi)
            phase_Psi = Psi / (amp_Psi + eps)
            Psi_c = diff_amp_j * phase_Psi

            # Inverse propagate back to real space
            psi_c = np.fft.ifft2(Psi_c)

            # Difference signal
            dpsi = psi_c - psi

            # === OBJECT UPDATE (pixel-wise normalization) ===
            O[ry:ry + ps, rx:rx + ps] += self.beta_obj * np.conj(P) * dpsi / (P_abs2 + eps)

            # Amplitude clipping: |O| <= amp_clip (physical reflectance constraint)
            O_patch_new = O[ry:ry + ps, rx:rx + ps]
            amp = np.abs(O_patch_new)
            mask = amp > self.amp_clip
            if np.any(mask):
                O_patch_new[mask] *= self.amp_clip / amp[mask]
                O[ry:ry + ps, rx:rx + ps] = O_patch_new

            # === PROBE UPDATE (pixel-wise, using UPDATED object) ===
            O_patch_updated = O[ry:ry + ps, rx:rx + ps]
            O_abs2 = np.abs(O_patch_updated) ** 2
            P += self.beta_probe * np.conj(O_patch_updated) * dpsi / (O_abs2 + eps)

            # Update cached P_abs2
            P_abs2 = np.abs(P) ** 2

            return O, P, P_abs2

    def _run_epie(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp_use: np.ndarray,
            positions: np.ndarray,
            time_budget: float,
        ) -> Tuple[np.ndarray, np.ndarray, int]:
            """
            Run full ePIE reconstruction loop with momentum.

            Args:
                O: Initial complex object, shape (obj_size, obj_size).
                P: Initial complex probe, shape (ps, ps).
                diff_amp_use: Diffraction amplitudes, shape (n_pos, ps, ps).
                positions: Scan positions, shape (n_pos, 2).
                time_budget: Maximum time in seconds.

            Returns:
                Tuple of (reconstructed O, reconstructed P, number of epochs completed).
            """
            t0 = time.time()
            n_pos = len(positions)
            epoch = 0

            # Pre-compute |P|^2
            P_abs2 = np.abs(P) ** 2

            while True:
                elapsed = time.time() - t0
                if elapsed > time_budget:
                    break

                # Save object state before epoch for momentum
                O_before = O.copy() if self.use_momentum else None

                # Randomize scan order each epoch
                order = np.random.permutation(n_pos)

                time_exceeded = False
                for idx in range(n_pos):
                    # Periodic time check
                    if idx % self.time_check_interval == 0 and (time.time() - t0) > time_budget:
                        time_exceeded = True
                        break

                    j = order[idx]
                    ry, rx = positions[j]

                    O, P, P_abs2 = self._epie_single_position(
                        O, P, diff_amp_use[j], (ry, rx), P_abs2
                    )

                if time_exceeded:
                    epoch += 1
                    break

                # Apply momentum on object after each complete epoch (from epoch >= 1)
                if self.use_momentum and epoch >= 1 and O_before is not None:
                    delta = O - O_before
                    O += self.beta_mom * delta

                    # Re-clip amplitudes after momentum
                    amp_O = np.abs(O)
                    mask_g = amp_O > self.amp_clip
                    if np.any(mask_g):
                        O[mask_g] *= self.amp_clip / amp_O[mask_g]

                epoch += 1

            return O, P, epoch

    def _compute_coverage_map(self, positions: np.ndarray) -> np.ndarray:
            """
            Compute how many times each pixel is illuminated across all scan positions.

            Args:
                positions: Scan positions, shape (n_pos, 2).

            Returns:
                Coverage count array, shape (obj_size, obj_size).
            """
            scan_count = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            ps = self.patch_size
            for j in range(len(positions)):
                ry, rx = positions[j]
                scan_count[ry:ry + ps, rx:rx + ps] += 1.0
            return scan_count

    def _align_global_phase(
            self,
            O: np.ndarray,
            scan_count: np.ndarray,
        ) -> np.ndarray:
            """
            Remove global phase offset using histogram-based mode finding
            on well-illuminated region.

            Args:
                O: Reconstructed complex object, shape (obj_size, obj_size).
                scan_count: Coverage map, shape (obj_size, obj_size).

            Returns:
                Phase-aligned complex object, shape (obj_size, obj_size).
            """
            illuminated = scan_count > 0
            well_illuminated = scan_count >= self.well_illuminated_threshold

            amp_result = np.abs(O)
            phase_result = np.angle(O)

            # Find dominant phase in well-illuminated, sufficiently bright region
            valid = well_illuminated & (amp_result > self.valid_amp_threshold)

            if np.sum(valid) > self.min_valid_pixels:
                phase_valid = phase_result[valid]

                # Histogram-based mode finding
                n_bins = self.phase_hist_bins
                hist, bin_edges = np.histogram(phase_valid, bins=n_bins, range=(-np.pi, np.pi))

                # Smooth histogram with uniform kernel for robust mode detection
                kernel_size = self.phase_hist_kernel_size
                kernel = np.ones(kernel_size) / kernel_size
                hist_smooth = np.convolve(hist, kernel, mode='same')

                # Find the bin with highest smoothed count
                mode_idx = np.argmax(hist_smooth)
                phase_ref = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
            else:
                # Fallback: use mean phase in illuminated region
                if np.any(illuminated):
                    phase_ref = np.mean(phase_result[illuminated])
                else:
                    phase_ref = 0.0

            # Remove global phase offset
            O = O * np.exp(-1j * phase_ref)

            # Zero out non-illuminated regions
            O[~illuminated] = 0.0

            return O

    def _format_output(
            self,
            O: np.ndarray,
            scan_count: np.ndarray,
        ) -> np.ndarray:
            """
            Format reconstructed object into (2, obj_size, obj_size) output array.
            Channel 0: amplitude |O|, Channel 1: phase angle(O).
            Non-illuminated regions set to zero.

            Args:
                O: Phase-aligned complex object, shape (obj_size, obj_size).
                scan_count: Coverage map, shape (obj_size, obj_size).

            Returns:
                Output array, shape (2, obj_size, obj_size), dtype float32.
            """
            illuminated = scan_count > 0

            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)

            # Zero out non-illuminated regions
            output[0][~illuminated] = 0.0
            output[1][~illuminated] = 0.0

            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Full reconstruction pipeline: preprocess → detect convention →
            initialize → run ePIE → post-process → output.

            Args:
                input_data: Measured diffraction intensities, shape (81, 256, 256).

            Returns:
                Reconstructed object as (2, 600, 600) float32 array.
                Channel 0: amplitude, Channel 1: phase.
            """
            self.t_start = time.time()

            # 1. Preprocess input (clip negatives, sqrt for amplitudes)
            I = np.maximum(input_data, 0).astype(np.float64)
            diff_amp = np.sqrt(I)

            # 2. Compute scan positions
            self.positions = self._compute_scan_positions()

            # 3. Detect and validate FFT convention
            # First, create initial probe and object for validation
            P_test = self._init_probe(diff_amp)
            O_test = self._init_object()

            # Validate convention using both heuristic and quick test
            self.diff_amp_use = self._validate_convention(
                diff_amp, O_test, P_test, self.positions
            )

            # 4. Initialize probe and object (fresh, using validated convention)
            self.probe = self._init_probe(self.diff_amp_use)
            self.obj = self._init_object()

            # 5. Run ePIE reconstruction within time budget
            elapsed_so_far = time.time() - self.t_start
            remaining_time = self.total_time_budget - elapsed_so_far - self.safety_margin
            remaining_time = max(remaining_time, 1.0)  # at least 1 second

            self.obj, self.probe, total_epochs = self._run_epie(
                self.obj, self.probe, self.diff_amp_use, self.positions, remaining_time
            )

            # 6. Compute coverage map
            scan_count = self._compute_coverage_map(self.positions)

            # 7. Align global phase
            self.obj = self._align_global_phase(self.obj, scan_count)

            # 8. Format and return output
            output = self._format_output(self.obj, scan_count)

            return output



if __name__ == "__main__":
    # Load data
    input_data = np.load('dataset/input.npy')

    # Create solver and run reconstruction
    solver = InverseSolver(
        patch_size=256,
        obj_size=600,
        n_side=9,
        step=40,
        n_pos=81,
        eps=1e-8,
        beta_obj=0.9,
        beta_probe=0.5,
        beta_mom=0.2,
        use_momentum=True,
        probe_sigma=50.0,
        obj_init_amp=1.0,
        amp_clip=1.0,
        total_time_budget=55.0,
        safety_margin=2.0,
    )

    result = solver.solve(input_data)

    # Save output
    np.save('output.npy', result)