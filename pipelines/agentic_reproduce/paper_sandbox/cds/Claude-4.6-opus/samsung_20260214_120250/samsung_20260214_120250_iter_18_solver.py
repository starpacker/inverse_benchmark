import numpy as np
import time
from typing import Tuple, Optional, List

class InverseSolver:
    def __init__(
            self,
            probe_size: int = 256,
            obj_size: int = 600,
            n_side: int = 9,
            step: int = 40,
            n_pos: int = 81,
            beta_O: float = 0.9,
            beta_P: float = 0.5,
            alpha_rPIE: float = 0.5,
            eps: float = 1e-12,
            probe_radius: float = 80.0,
            probe_edge_width: float = 20.0,
            object_init_amplitude: float = 0.9,
            total_time_budget: float = 55.0,
            safety_margin: float = 2.0,
            fft_test_epochs: int = 1,
            phase_hist_bins: int = 720,
            phase_hist_kernel_size: int = 31,
            min_scan_count_for_valid: int = 4,
            min_amp_for_valid: float = 0.1,
            min_valid_pixels: int = 100,
        ) -> None:
            self.probe_size: int = probe_size
            self.obj_size: int = obj_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_pos: int = n_pos
            self.beta_O: float = beta_O
            self.beta_P: float = beta_P
            self.alpha_rPIE: float = alpha_rPIE
            self.eps: float = eps
            self.probe_radius: float = probe_radius
            self.probe_edge_width: float = probe_edge_width
            self.object_init_amplitude: float = object_init_amplitude
            self.total_time_budget: float = total_time_budget
            self.safety_margin: float = safety_margin
            self.fft_test_epochs: int = fft_test_epochs
            self.phase_hist_bins: int = phase_hist_bins
            self.phase_hist_kernel_size: int = phase_hist_kernel_size
            self.min_scan_count_for_valid: int = min_scan_count_for_valid
            self.min_amp_for_valid: float = min_amp_for_valid
            self.min_valid_pixels: int = min_valid_pixels

            self.positions: Optional[np.ndarray] = None
            self.probe: Optional[np.ndarray] = None
            self.obj: Optional[np.ndarray] = None
            self.diff_amp: Optional[np.ndarray] = None
            self.t_start: float = 0.0

    def _compute_scan_positions(self) -> np.ndarray:
            """Compute the (n_pos, 2) array of (row, col) scan positions on the object grid."""
            total_scan = self.probe_size + (self.n_side - 1) * self.step  # 256 + 320 = 576
            offset = (self.obj_size - total_scan) // 2  # 12

            positions = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions, dtype=np.int32)

    def _make_probe(self, diff_amp: np.ndarray) -> np.ndarray:
            """
            Create initial probe function: flat-top disk with soft Gaussian edge,
            energy-scaled to match mean diffraction pattern energy.

            Args:
                diff_amp: Diffraction amplitudes (n_pos, probe_size, probe_size) for energy calibration.

            Returns:
                Complex probe array of shape (probe_size, probe_size).
            """
            ps = self.probe_size
            yy, xx = np.mgrid[:ps, :ps]
            cy, cx = ps / 2.0, ps / 2.0
            r = np.sqrt((yy - cy)**2 + (xx - cx)**2)

            # Flat-top disk with soft cosine edge roll-off
            radius = self.probe_radius
            edge_width = self.probe_edge_width

            P = np.zeros((ps, ps), dtype=np.complex128)
            # Inside the radius: amplitude = 1
            inside = r <= radius
            # In the edge region: cosine taper from 1 to 0
            edge = (r > radius) & (r <= radius + edge_width)
            # Outside: 0
            P[inside] = 1.0
            P[edge] = 0.5 * (1.0 + np.cos(np.pi * (r[edge] - radius) / edge_width))

            # Energy scaling: match mean diffraction pattern energy
            # By Parseval's theorem, sum(|FFT(P*O)|^2) = N * sum(|P*O|^2)
            # For initial uniform object with amplitude ~ object_init_amplitude:
            # sum(|Psi|^2) ~ N * obj_amp^2 * sum(|P|^2)
            # Measured energy: sum(diff_amp^2) = sum(I)
            mean_energy = np.mean(np.sum(diff_amp**2, axis=(-2, -1)))
            current_energy = np.sum(np.abs(P)**2)

            if current_energy > self.eps:
                # Scale factor: sqrt(mean_energy / (N * obj_amp^2 * current_energy))
                # where N = ps*ps (from FFT normalization)
                scale = np.sqrt(mean_energy / (ps * ps * self.object_init_amplitude**2 * current_energy + self.eps))
                P *= scale

            return P

    def _init_object(self) -> np.ndarray:
            """
            Initialize object as uniform complex array with amplitude = object_init_amplitude, phase = 0.

            Returns:
                Complex object array of shape (obj_size, obj_size).
            """
            return self.object_init_amplitude * np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Forward model: compute far-field diffraction amplitude for a single scan position.

            Ψ(k) = FFT2{ P(r - R_j) · O(r) }

            Args:
                obj: Complex object array (obj_size, obj_size).
                probe: Complex probe array (probe_size, probe_size).
                position: (row, col) top-left corner of the patch in the object.

            Returns:
                Complex diffraction pattern (probe_size, probe_size).
            """
            ry, rx = position
            obj_patch = obj[ry:ry + self.probe_size, rx:rx + self.probe_size]
            psi = probe * obj_patch
            Psi = np.fft.fft2(psi)
            return Psi

    def _fourier_magnitude_replacement(
            self,
            psi_k: np.ndarray,
            measured_amp: np.ndarray,
        ) -> np.ndarray:
            """
            Replace Fourier magnitudes with measured amplitudes while preserving phase.

            Ψ_c(k) = √I(k) · Ψ(k) / |Ψ(k)|

            Args:
                psi_k: Complex Fourier-space wave (probe_size, probe_size).
                measured_amp: Measured diffraction amplitudes (probe_size, probe_size).

            Returns:
                Corrected complex Fourier-space wave (probe_size, probe_size).
            """
            amp_psi = np.abs(psi_k)
            psi_c = measured_amp * psi_k / (amp_psi + self.eps)
            return psi_c

    def _compute_residual_one_epoch(
            self,
            diff_amp: np.ndarray,
            obj: np.ndarray,
            probe: np.ndarray,
            positions: np.ndarray,
        ) -> float:
            """
            Run one epoch of rPIE and return normalized Fourier residual for FFT convention detection.

            Args:
                diff_amp: Diffraction amplitudes (n_pos, probe_size, probe_size).
                obj: Complex object array (obj_size, obj_size).
                probe: Complex probe array (probe_size, probe_size).
                positions: Scan positions (n_pos, 2).

            Returns:
                Normalized residual scalar.
            """
            O = obj.copy()
            P = probe.copy()
            n_pos = len(positions)
            ps = self.probe_size

            # Run one epoch of rPIE updates
            for j in range(n_pos):
                ry, rx = positions[j]
                O_patch = O[ry:ry + ps, rx:rx + ps]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                amp_Psi = np.abs(Psi)
                Psi_c = diff_amp[j] * Psi / (amp_Psi + self.eps)
                psi_c = np.fft.ifft2(Psi_c)
                dpsi = psi_c - psi

                # rPIE object update
                P_abs2 = np.abs(P) ** 2
                P_max2 = np.max(P_abs2)
                denom_O = (1 - self.alpha_rPIE) * P_abs2 + self.alpha_rPIE * P_max2 + self.eps
                O[ry:ry + ps, rx:rx + ps] += self.beta_O * np.conj(P) * dpsi / denom_O

            # Compute residual over all positions after one epoch
            res = 0.0
            norm = 0.0
            for j in range(n_pos):
                ry, rx = positions[j]
                psi = P * O[ry:ry + ps, rx:rx + ps]
                Psi = np.fft.fft2(psi)
                res += np.sum((np.abs(Psi) - diff_amp[j]) ** 2)
                norm += np.sum(diff_amp[j] ** 2)

            return res / (norm + self.eps)

    def _detect_fft_convention(
            self,
            diff_amp_raw: np.ndarray,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Detect correct FFT convention by running one epoch each with raw and ifftshifted amplitudes.

            Args:
                diff_amp_raw: Raw diffraction amplitudes (n_pos, probe_size, probe_size).
                positions: Scan positions (n_pos, 2).

            Returns:
                Diffraction amplitudes with correct convention applied (n_pos, probe_size, probe_size).
            """
            # Prepare both conventions
            damp_raw = diff_amp_raw.copy()
            damp_shifted = np.fft.ifftshift(diff_amp_raw, axes=(-2, -1))

            # Initialize object for testing
            O_init = self._init_object()

            # Create probes scaled to each convention's energy
            P_raw = self._make_probe(damp_raw)
            P_shifted = self._make_probe(damp_shifted)

            # Run one epoch with raw convention
            res_raw = self._compute_residual_one_epoch(damp_raw, O_init.copy(), P_raw.copy(), positions)

            # Run one epoch with shifted convention
            res_shifted = self._compute_residual_one_epoch(damp_shifted, O_init.copy(), P_shifted.copy(), positions)

            if res_shifted < res_raw:
                return damp_shifted
            else:
                return damp_raw

    def _rPIE_object_update(
            self,
            obj_patch: np.ndarray,
            probe: np.ndarray,
            dpsi: np.ndarray,
            P_abs2: np.ndarray,
            P_max2: float,
        ) -> np.ndarray:
            """
            rPIE object update rule with regularized denominator.

            O += β_O · P* · Δψ / ((1-α)|P|² + α·max|P|²)

            Args:
                obj_patch: Current object patch (probe_size, probe_size).
                probe: Current probe (probe_size, probe_size).
                dpsi: Exit wave difference (probe_size, probe_size).
                P_abs2: |P|² precomputed (probe_size, probe_size).
                P_max2: max(|P|²) scalar.

            Returns:
                Updated object patch (probe_size, probe_size).
            """
            denom = (1.0 - self.alpha_rPIE) * P_abs2 + self.alpha_rPIE * P_max2 + self.eps
            updated_patch = obj_patch + self.beta_O * np.conj(probe) * dpsi / denom
            return updated_patch

    def _rPIE_probe_update(
            self,
            probe: np.ndarray,
            obj_patch_new: np.ndarray,
            dpsi: np.ndarray,
        ) -> np.ndarray:
            """
            rPIE probe update rule with regularized denominator.

            P += β_P · O* · Δψ / ((1-α)|O|² + α·max|O_patch|²)

            Args:
                probe: Current probe (probe_size, probe_size).
                obj_patch_new: Updated object patch (probe_size, probe_size).
                dpsi: Exit wave difference (probe_size, probe_size).

            Returns:
                Updated probe (probe_size, probe_size).
            """
            O_abs2 = np.abs(obj_patch_new) ** 2
            O_max2 = np.max(O_abs2)
            denom = (1.0 - self.alpha_rPIE) * O_abs2 + self.alpha_rPIE * O_max2 + self.eps
            updated_probe = probe + self.beta_P * np.conj(obj_patch_new) * dpsi / denom
            return updated_probe

    def _run_rPIE(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            diff_amp: np.ndarray,
            positions: np.ndarray,
            time_budget: float,
        ) -> Tuple[np.ndarray, np.ndarray, int]:
            """
            Main rPIE reconstruction loop. Iterates epochs until time budget exhausted.

            Args:
                obj: Initial complex object (obj_size, obj_size).
                probe: Initial complex probe (probe_size, probe_size).
                diff_amp: Diffraction amplitudes (n_pos, probe_size, probe_size).
                positions: Scan positions (n_pos, 2).
                time_budget: Maximum allowed time in seconds.

            Returns:
                Tuple of (reconstructed object, reconstructed probe, number of epochs completed).
            """
            t0 = time.time()
            n_pos = len(positions)
            ps = self.probe_size
            epoch = 0

            while True:
                elapsed = time.time() - t0
                if elapsed > time_budget:
                    break

                # Randomize scan order each epoch
                order = np.random.permutation(n_pos)

                # Pre-compute probe |P|^2 and max|P|^2 for this epoch
                P_abs2 = np.abs(probe) ** 2
                P_max2 = np.max(P_abs2)

                for idx in range(n_pos):
                    # Check time budget periodically
                    if idx % 27 == 0 and (time.time() - t0) > time_budget:
                        break

                    j = order[idx]
                    ry, rx = positions[j]

                    # Extract object patch
                    obj_patch = obj[ry:ry + ps, rx:rx + ps].copy()

                    # Compute exit wave
                    psi = probe * obj_patch

                    # Forward propagate to Fourier space
                    Psi = np.fft.fft2(psi)

                    # Fourier magnitude replacement
                    Psi_c = self._fourier_magnitude_replacement(Psi, diff_amp[j])

                    # Inverse propagate back to real space
                    psi_c = np.fft.ifft2(Psi_c)

                    # Difference signal
                    dpsi = psi_c - psi

                    # Object update (rPIE regularized denominator)
                    obj[ry:ry + ps, rx:rx + ps] = self._rPIE_object_update(
                        obj_patch, probe, dpsi, P_abs2, P_max2
                    )

                    # Probe update (rPIE regularized denominator, using updated object patch)
                    probe = self._rPIE_probe_update(
                        probe, obj[ry:ry + ps, rx:rx + ps], dpsi
                    )

                # Update cached probe values after full epoch
                P_abs2 = np.abs(probe) ** 2
                P_max2 = np.max(P_abs2)

                epoch += 1

            return obj, probe, epoch

    def _compute_scan_coverage(
            self,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Compute per-pixel scan coverage count.

            Args:
                positions: Scan positions (n_pos, 2).

            Returns:
                Coverage map (obj_size, obj_size) of float64.
            """
            scan_count = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            for j in range(len(positions)):
                ry, rx = positions[j]
                scan_count[ry:ry + self.probe_size, rx:rx + self.probe_size] += 1.0
            return scan_count

    def _align_global_phase(
            self,
            obj: np.ndarray,
            scan_count: np.ndarray,
        ) -> np.ndarray:
            """
            Remove global phase offset using histogram mode of well-illuminated region.

            Args:
                obj: Complex reconstructed object (obj_size, obj_size).
                scan_count: Per-pixel scan coverage (obj_size, obj_size).

            Returns:
                Phase-aligned complex object (obj_size, obj_size).
            """
            amp_result = np.abs(obj)
            phase_result = np.angle(obj)

            # Identify well-illuminated region with sufficient amplitude
            well_illuminated = scan_count >= self.min_scan_count_for_valid
            valid = well_illuminated & (amp_result > self.min_amp_for_valid)

            if np.sum(valid) > self.min_valid_pixels:
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
            obj_aligned = obj * np.exp(-1j * phase_ref)

            return obj_aligned

    def _postprocess(
            self,
            obj: np.ndarray,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Post-process reconstructed object: phase alignment, masking, output formatting.

            Args:
                obj: Complex reconstructed object (obj_size, obj_size).
                positions: Scan positions (n_pos, 2).

            Returns:
                Output array of shape (2, obj_size, obj_size) with channel 0 = amplitude, channel 1 = phase.
            """
            # Compute scan coverage map
            scan_count = self._compute_scan_coverage(positions)

            # Identify illuminated region
            illuminated = scan_count > 0

            # Align global phase using histogram mode
            obj = self._align_global_phase(obj, scan_count)

            # Zero out non-illuminated regions
            obj[~illuminated] = 0.0

            # Build output: channel 0 = amplitude, channel 1 = phase
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(obj).astype(np.float32)
            output[1] = np.angle(obj).astype(np.float32)

            return output

    def solve(
            self,
            input_data: np.ndarray,
        ) -> np.ndarray:
            """
            Full reconstruction pipeline: preprocess, detect FFT convention, initialize, run rPIE, postprocess.

            Args:
                input_data: Measured diffraction intensities of shape (n_pos, probe_size, probe_size).

            Returns:
                Reconstructed output of shape (2, obj_size, obj_size), float32.
                    Channel 0: amplitude |O(r)|
                    Channel 1: phase angle(O(r))
            """
            self.t_start = time.time()

            # Step 1: Data Loading & Preprocessing
            I = np.maximum(input_data, 0).astype(np.float64)  # (81, 256, 256)
            diff_amp_raw = np.sqrt(I)  # (81, 256, 256)

            # Step 2: Compute scan positions
            self.positions = self._compute_scan_positions()

            # Step 3: Detect FFT convention (run 1 full epoch each way)
            self.diff_amp = self._detect_fft_convention(diff_amp_raw, self.positions)

            # Step 4: Initialize probe and object
            self.probe = self._make_probe(self.diff_amp)
            self.obj = self._init_object()

            # Step 5: Run rPIE reconstruction
            remaining_time = self.total_time_budget - (time.time() - self.t_start) - self.safety_margin
            remaining_time = max(remaining_time, 1.0)  # at least 1 second

            self.obj, self.probe, total_epochs = self._run_rPIE(
                self.obj, self.probe, self.diff_amp, self.positions,
                time_budget=remaining_time,
            )

            # Step 6: Post-process and format output
            output = self._postprocess(self.obj, self.positions)

            return output



if __name__ == "__main__":
    # Load data
    input_data = np.load('dataset/input.npy')

    # Create solver and run reconstruction
    solver = InverseSolver()
    result = solver.solve(input_data)

    # Save output
    np.save('output.npy', result)