import numpy as np
import time
from typing import Tuple, List, Optional

class InverseSolver:
    def __init__(
            self,
            probe_size: int = 256,
            obj_size: int = 600,
            n_side: int = 9,
            step: int = 40,
            n_pos: int = 81,
            eps: float = 1e-8,
            total_time: float = 55.0,
            detect_time: float = 4.0,
            safety_margin: float = 2.0,
            beta_object: float = 1.0,
            beta_probe: float = 1.0,
            beta_momentum: float = 0.3,
            momentum_start_epoch: int = 2,
            object_init_amplitude: float = 0.7,
            probe_window_sigma: float = 80.0,
            amplitude_clip: float = 1.0,
            phase_histogram_bins: int = 720,
            phase_histogram_smooth_kernel: int = 21,
            residual_n_sample: int = 20,
            well_illuminated_threshold: int = 4,
            valid_amplitude_threshold: float = 0.1,
            stabilize_fraction: float = 0.3,
            wavelength_nm: float = 13.5,
            pixel_size_nm: float = 27.0,
        ) -> None:
            # Grid / geometry
            self.probe_size: int = probe_size
            self.obj_size: int = obj_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_pos: int = n_pos

            # Numerical stability
            self.eps: float = eps

            # Timing budget
            self.total_time: float = total_time
            self.detect_time: float = detect_time
            self.safety_margin: float = safety_margin

            # ePIE step sizes (alpha=1 convention)
            self.beta_object: float = beta_object
            self.beta_probe: float = beta_probe

            # Momentum
            self.beta_momentum: float = beta_momentum
            self.momentum_start_epoch: int = momentum_start_epoch

            # Initialization
            self.object_init_amplitude: float = object_init_amplitude
            self.probe_window_sigma: float = probe_window_sigma

            # Constraints
            self.amplitude_clip: float = amplitude_clip

            # Phase alignment post-processing
            self.phase_histogram_bins: int = phase_histogram_bins
            self.phase_histogram_smooth_kernel: int = phase_histogram_smooth_kernel

            # Convention detection
            self.residual_n_sample: int = residual_n_sample

            # Masking thresholds
            self.well_illuminated_threshold: int = well_illuminated_threshold
            self.valid_amplitude_threshold: float = valid_amplitude_threshold

            # Two-phase reconstruction split
            self.stabilize_fraction: float = stabilize_fraction

            # Physical parameters
            self.wavelength_nm: float = wavelength_nm
            self.pixel_size_nm: float = pixel_size_nm

            # Derived: scan positions (computed once, lazily in solve or _build_positions)
            self.positions: Optional[np.ndarray] = None

    def _build_positions(self) -> np.ndarray:
            """Compute (n_pos, 2) array of (row, col) scan positions on the object grid."""
            total_scan = self.probe_size + (self.n_side - 1) * self.step  # 256 + 320 = 576
            offset = (self.obj_size - total_scan) // 2  # 12

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            self.positions = np.array(positions, dtype=np.int32)  # (81, 2)
            return self.positions

    def _init_probe(self, diff_amp: np.ndarray) -> np.ndarray:
            """
            Initialize probe from average diffraction amplitude.
            Apply Gaussian spatial window. Normalize ONCE to unit max amplitude.

            Args:
                diff_amp: (n_pos, probe_size, probe_size) diffraction amplitudes

            Returns:
                P: (probe_size, probe_size) complex128 probe array
            """
            ps = self.probe_size
            # Average diffraction amplitude across all positions
            avg_amp = np.mean(diff_amp, axis=0)  # (probe_size, probe_size)

            # Inverse FFT to get real-space probe estimate
            P = np.fft.ifft2(avg_amp)  # complex (probe_size, probe_size)

            # Apply gentle Gaussian window for spatial compactness
            yy, xx = np.mgrid[:ps, :ps]
            cy, cx = ps / 2.0, ps / 2.0
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            window = np.exp(-r2 / (2.0 * self.probe_window_sigma ** 2))
            # ifftshift so the window center aligns with the FFT convention (DC at corner)
            window_fft = np.fft.ifftshift(window)
            P *= window_fft

            # Normalize ONCE to unit max amplitude — then NEVER again
            max_abs = np.max(np.abs(P))
            P = P / (max_abs + self.eps) * 1.0

            return P.astype(np.complex128)

    def _init_object(self) -> np.ndarray:
            """
            Initialize object as uniform complex reflector with amplitude = object_init_amplitude.

            Returns:
                O: (obj_size, obj_size) complex128 object array
            """
            return np.full((self.obj_size, self.obj_size), self.object_init_amplitude + 0j, dtype=np.complex128)

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Forward model: compute far-field diffraction amplitude for one scan position.

            Ψ(r) = P(r - R_j) · O(r)
            A(k) = |FFT2{Ψ(r)}|

            Args:
                O: (obj_size, obj_size) complex object
                P: (probe_size, probe_size) complex probe
                position: (row, col) top-left corner of the probe on the object grid

            Returns:
                amplitude: (probe_size, probe_size) real-valued diffraction amplitude
            """
            ry, rx = position
            ps = self.probe_size
            # Extract object patch at scan position
            O_patch = O[ry:ry + ps, rx:rx + ps]
            # Form exit wave: element-wise product of probe and object patch
            psi = P * O_patch
            # Propagate to far field via 2D FFT
            Psi = np.fft.fft2(psi)
            # Return diffraction amplitude (magnitude)
            amplitude = np.abs(Psi)
            return amplitude

    def _fourier_magnitude_replace(
            self,
            psi: np.ndarray,
            measured_amp: np.ndarray,
        ) -> np.ndarray:
            """
            Replace Fourier magnitude of exit wave with measured amplitude.

            Args:
                psi: (probe_size, probe_size) complex exit wave in real space
                measured_amp: (probe_size, probe_size) measured diffraction amplitude

            Returns:
                psi_corrected: (probe_size, probe_size) complex corrected exit wave in real space
            """
            # Forward propagate to Fourier space
            Psi = np.fft.fft2(psi)
            # Replace magnitude with measured amplitude, keeping phase
            Psi_corrected = measured_amp * np.exp(1j * np.angle(Psi))
            # Inverse propagate back to real space
            psi_corrected = np.fft.ifft2(Psi_corrected)
            return psi_corrected

    def _update_object_patch(
            self,
            O_patch: np.ndarray,
            P: np.ndarray,
            dpsi: np.ndarray,
            P_max_sq: float,
        ) -> np.ndarray:
            """
            ePIE object update (alpha=1):
            O_patch += conj(P) * dpsi / (P_max_sq + eps)

            Args:
                O_patch: (probe_size, probe_size) complex object patch
                P: (probe_size, probe_size) complex probe
                dpsi: (probe_size, probe_size) complex exit wave difference
                P_max_sq: scalar, max(|P|^2) over probe

            Returns:
                O_patch_updated: (probe_size, probe_size) complex updated object patch
            """
            O_patch_updated = O_patch + np.conj(P) * dpsi / (P_max_sq + self.eps)
            return O_patch_updated

    def _update_probe(
            self,
            P: np.ndarray,
            O_patch_updated: np.ndarray,
            dpsi: np.ndarray,
            O_max_sq: float,
        ) -> np.ndarray:
            """
            ePIE probe update (alpha=1) using UPDATED object patch:
            P += conj(O_updated) * dpsi / (O_max_sq + eps)

            Args:
                P: (probe_size, probe_size) complex probe
                O_patch_updated: (probe_size, probe_size) complex updated object patch
                dpsi: (probe_size, probe_size) complex exit wave difference
                O_max_sq: scalar, max(|O_patch_updated|^2)

            Returns:
                P_updated: (probe_size, probe_size) complex updated probe
            """
            P_updated = P + np.conj(O_patch_updated) * dpsi / (O_max_sq + self.eps)
            return P_updated

    def _clip_object_amplitude(self, O: np.ndarray) -> np.ndarray:
            """
            Enforce physical reflectance constraint: |O(r)| <= amplitude_clip.

            Args:
                O: (obj_size, obj_size) or (probe_size, probe_size) complex array

            Returns:
                O_clipped: same shape, amplitude-clipped complex array
            """
            amp = np.abs(O)
            clip_mask = amp > self.amplitude_clip
            if np.any(clip_mask):
                O = O.copy()
                O[clip_mask] *= self.amplitude_clip / amp[clip_mask]
            return O

    def _apply_momentum(
            self,
            O_current: np.ndarray,
            O_before_epoch: np.ndarray,
        ) -> np.ndarray:
            """
            Apply gentle momentum: O = O + beta_momentum * (O - O_before_epoch), then clip.

            Args:
                O_current: (obj_size, obj_size) complex object after epoch
                O_before_epoch: (obj_size, obj_size) complex object before epoch

            Returns:
                O_accelerated: (obj_size, obj_size) complex object with momentum applied
            """
            O_accelerated = O_current + self.beta_momentum * (O_current - O_before_epoch)
            O_accelerated = self._clip_object_amplitude(O_accelerated)
            return O_accelerated

    def _run_ePIE(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
            time_budget: float,
            use_momentum: bool = True,
        ) -> Tuple[np.ndarray, np.ndarray, int]:
            """
            Run ePIE iterations (alpha=1) with optional momentum.

            Iterates epochs until time_budget is exhausted. Each epoch:
            1. Random permutation of scan positions
            2. For each position: forward → magnitude replace → object update → probe update
            3. Optionally apply momentum after epoch >= momentum_start_epoch

            Args:
                O: (obj_size, obj_size) complex object (modified in-place and returned)
                P: (probe_size, probe_size) complex probe (modified in-place and returned)
                diff_amp: (n_pos, probe_size, probe_size) measured diffraction amplitudes
                positions: (n_pos, 2) scan positions
                time_budget: float, seconds available
                use_momentum: bool, whether to apply momentum acceleration

            Returns:
                O: updated complex object
                P: updated complex probe
                n_epochs: number of completed epochs
            """
            t0 = time.time()
            n_pos = len(positions)
            ps = self.probe_size
            epoch = 0
            O_before_epoch = None

            while True:
                elapsed = time.time() - t0
                if elapsed > time_budget:
                    break

                # Save object state before epoch for momentum
                if use_momentum and epoch >= (self.momentum_start_epoch - 1):
                    O_before_epoch = O.copy()

                # Randomize position ordering each epoch
                order = np.random.permutation(n_pos)

                # Precompute max |P|^2 for object update
                P_max_sq = np.max(np.abs(P) ** 2)

                time_exceeded = False
                for idx in range(n_pos):
                    # Check time periodically (every 27 positions)
                    if idx % 27 == 0 and (time.time() - t0) > time_budget:
                        time_exceeded = True
                        break

                    j = order[idx]
                    ry, rx = positions[j]

                    # Extract object patch
                    O_patch = O[ry:ry + ps, rx:rx + ps]

                    # Exit wave
                    psi = P * O_patch

                    # Fourier magnitude replacement
                    psi_c = self._fourier_magnitude_replace(psi, diff_amp[j])

                    # Difference
                    dpsi = psi_c - psi

                    # ePIE Object update (alpha=1)
                    O_patch_updated = self._update_object_patch(O_patch, P, dpsi, P_max_sq)

                    # Object amplitude clipping (reflectance <= 1)
                    amp = np.abs(O_patch_updated)
                    clip_mask = amp > self.amplitude_clip
                    if np.any(clip_mask):
                        O_patch_updated[clip_mask] *= self.amplitude_clip / amp[clip_mask]

                    # Write back updated object patch
                    O[ry:ry + ps, rx:rx + ps] = O_patch_updated

                    # ePIE Probe update using UPDATED object patch
                    O_max_sq = np.max(np.abs(O_patch_updated) ** 2)
                    P = self._update_probe(P, O_patch_updated, dpsi, O_max_sq)

                if time_exceeded:
                    epoch += 1
                    break

                # Apply gentle momentum to object after sufficient epochs
                if use_momentum and epoch >= self.momentum_start_epoch and O_before_epoch is not None:
                    O = self._apply_momentum(O, O_before_epoch)

                epoch += 1

            return O, P, epoch

    def _compute_residual(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
            n_sample: int = 20,
        ) -> float:
            """
            Compute normalized Fourier residual on a random subset of positions.

            Args:
                O: (obj_size, obj_size) complex object
                P: (probe_size, probe_size) complex probe
                diff_amp: (n_pos, probe_size, probe_size) measured amplitudes
                positions: (n_pos, 2) scan positions
                n_sample: number of positions to evaluate

            Returns:
                residual: float, normalized residual metric
            """
            rng = np.random.RandomState(42)
            n_total = len(positions)
            subset = rng.choice(n_total, min(n_sample, n_total), replace=False)
            ps = self.probe_size
            res = 0.0
            norm = 0.0
            for j in subset:
                ry, rx = positions[j]
                O_patch = O[ry:ry + ps, rx:rx + ps]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                res += np.sum((np.abs(Psi) - diff_amp[j]) ** 2)
                norm += np.sum(diff_amp[j] ** 2)
            return res / (norm + self.eps)

    def _detect_fft_convention(
            self,
            diff_amp: np.ndarray,
            positions: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Test two FFT conventions (raw vs ifftshift) and pick the one with lower residual.
            Uses short ePIE runs (~2s each). Returns warm-started O, P, and chosen diff_amp.

            Args:
                diff_amp: (n_pos, probe_size, probe_size) raw diffraction amplitudes
                positions: (n_pos, 2) scan positions

            Returns:
                best_O: (obj_size, obj_size) complex object from best convention
                best_P: (probe_size, probe_size) complex probe from best convention
                best_damp: (n_pos, probe_size, probe_size) amplitudes for best convention
            """
            half_budget = self.detect_time / 2.0

            # Convention A: raw (no shift)
            damp_a = diff_amp.copy()
            P_a = self._init_probe(damp_a)
            O_a = self._init_object()
            O_a, P_a, ep_a = self._run_ePIE(
                O_a, P_a, damp_a, positions,
                time_budget=half_budget, use_momentum=False
            )
            res_a = self._compute_residual(O_a, P_a, damp_a, positions, n_sample=self.residual_n_sample)

            # Convention B: ifftshift applied to diffraction amplitudes
            damp_b = np.fft.ifftshift(diff_amp, axes=(-2, -1))
            P_b = self._init_probe(damp_b)
            O_b = self._init_object()
            O_b, P_b, ep_b = self._run_ePIE(
                O_b, P_b, damp_b, positions,
                time_budget=half_budget, use_momentum=False
            )
            res_b = self._compute_residual(O_b, P_b, damp_b, positions, n_sample=self.residual_n_sample)

            # Pick the convention with lower residual — warm start from its state
            if res_a <= res_b:
                return O_a, P_a, damp_a
            else:
                return O_b, P_b, damp_b

    def _compute_illumination_mask(
            self,
            positions: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute illuminated and well-illuminated boolean masks from scan positions.

            Args:
                positions: (n_pos, 2) scan positions

            Returns:
                illuminated: (obj_size, obj_size) bool, any coverage
                well_illuminated: (obj_size, obj_size) bool, coverage >= threshold
            """
            scan_count = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            ps = self.probe_size
            for j in range(len(positions)):
                ry, rx = positions[j]
                scan_count[ry:ry + ps, rx:rx + ps] += 1.0

            illuminated = scan_count > 0
            well_illuminated = scan_count >= self.well_illuminated_threshold

            return illuminated, well_illuminated

    def _align_phase(
            self,
            O: np.ndarray,
            well_illuminated: np.ndarray,
        ) -> np.ndarray:
            """
            Remove global phase offset by finding histogram mode in well-illuminated region.
            Substrate phase should map to ~0.

            Args:
                O: (obj_size, obj_size) complex object
                well_illuminated: (obj_size, obj_size) bool mask

            Returns:
                O_aligned: (obj_size, obj_size) complex object with global phase removed
            """
            amp_result = np.abs(O)
            phase_result = np.angle(O)

            # Find valid pixels: well-illuminated and with sufficient amplitude
            valid = well_illuminated & (amp_result > self.valid_amplitude_threshold)

            if np.sum(valid) > 100:
                phase_valid = phase_result[valid]

                # Histogram-based mode finding
                n_bins = self.phase_histogram_bins
                hist, bin_edges = np.histogram(phase_valid, bins=n_bins, range=(-np.pi, np.pi))

                # Smooth histogram with uniform kernel
                kernel_size = self.phase_histogram_smooth_kernel
                kernel = np.ones(kernel_size) / float(kernel_size)
                hist_smooth = np.convolve(hist, kernel, mode='same')

                # Find mode (peak of smoothed histogram)
                mode_idx = np.argmax(hist_smooth)
                phase_ref = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
            else:
                phase_ref = 0.0

            # Remove global phase offset (substrate → 0 phase)
            O_aligned = O * np.exp(-1j * phase_ref)

            return O_aligned

    def _postprocess(
            self,
            O: np.ndarray,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Post-process reconstructed object: phase alignment, masking, output formatting.

            Args:
                O: (obj_size, obj_size) complex object
                positions: (n_pos, 2) scan positions

            Returns:
                output: (2, obj_size, obj_size) float32 — channel 0: amplitude, channel 1: phase
            """
            # Compute illumination masks
            illuminated, well_illuminated = self._compute_illumination_mask(positions)

            # Align phase (remove global phase offset so substrate maps to ~0)
            O = self._align_phase(O, well_illuminated)

            # Zero out non-illuminated regions
            O[~illuminated] = 0.0

            # Build final output: channel 0 = amplitude, channel 1 = phase
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)

            return output

    def solve(
            self,
            input_data: np.ndarray,
        ) -> np.ndarray:
            """
            Full inverse solver pipeline:
            1. Preprocess input intensities → diffraction amplitudes
            2. Build scan positions
            3. Detect FFT convention (warm-started)
            4. Main ePIE reconstruction in two phases (stabilize, then accelerate with momentum)
            5. Post-process and return (2, obj_size, obj_size) output

            Args:
                input_data: (81, 256, 256) real-valued diffraction intensity measurements

            Returns:
                output: (2, 600, 600) float32 array
                        channel 0: |O(r)| amplitude
                        channel 1: angle(O(r)) phase in radians
            """
            t_start = time.time()

            # Step 1: Preprocess input intensities → diffraction amplitudes
            I = np.maximum(input_data, 0).astype(np.float64)  # (81, 256, 256)
            diff_amp = np.sqrt(I)  # (81, 256, 256)

            # Step 2: Build scan positions
            positions = self._build_positions()  # (81, 2)

            # Step 3: Detect FFT convention (warm-started, ~4s total)
            best_O, best_P, best_damp = self._detect_fft_convention(diff_amp, positions)

            # Step 4: Main ePIE reconstruction in two phases
            remaining_time = self.total_time - (time.time() - t_start) - self.safety_margin

            if remaining_time > 3.0:
                # Phase 1: Stabilize without momentum
                stabilize_time = remaining_time * self.stabilize_fraction
                best_O, best_P, ep1 = self._run_ePIE(
                    best_O, best_P, best_damp, positions,
                    time_budget=stabilize_time, use_momentum=False
                )

                # Phase 2: Accelerate with gentle momentum
                remaining2 = self.total_time - (time.time() - t_start) - 1.5
                if remaining2 > 2.0:
                    best_O, best_P, ep2 = self._run_ePIE(
                        best_O, best_P, best_damp, positions,
                        time_budget=remaining2, use_momentum=True
                    )

            # Step 5: Post-process and return
            output = self._postprocess(best_O, positions)

            return output



if __name__ == "__main__":
    # Load data
    input_data = np.load('dataset/input.npy')

    # Create solver and run
    solver = InverseSolver()
    result = solver.solve(input_data)

    # Save output
    np.save('output.npy', result)