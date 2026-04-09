import numpy as np
import time
from typing import Tuple, Optional, List

class InverseSolver:
    """
    ePIE-based ptychographic phase retrieval solver for periodic copper pad arrays.
    
    Runs both FFT conventions in parallel (DC-at-corner vs DC-centered),
    selects the winner by final Fourier residual, and applies minimal
    post-processing (weighted mean phase removal only).
    """

    def __init__(
            self,
            probe_size: int = 256,
            object_size: int = 600,
            n_side: int = 9,
            step: int = 40,
            n_positions: int = 81,
            beta_object: float = 0.9,
            beta_probe: float = 0.3,
            probe_update_start_epoch: int = 3,
            probe_sigma: float = 55.0,
            initial_object_amplitude: float = 0.9,
            eps: float = 1e-12,
            total_time_budget: float = 55.0,
            safety_margin: float = 3.0,
            well_lit_threshold: float = 0.5,
            valid_amp_threshold: float = 0.1,
            min_valid_pixels: int = 100,
        ) -> None:
            """
            Initialize all hyperparameters for the ePIE solver.

            Args:
                probe_size: Size of the probe/diffraction pattern (pixels).
                object_size: Size of the object array (pixels).
                n_side: Number of scan positions per side (9x9 grid).
                step: Scan step size in pixels.
                n_positions: Total number of scan positions.
                beta_object: Object update feedback parameter.
                beta_probe: Probe update feedback parameter.
                probe_update_start_epoch: Epoch at which probe updates begin.
                probe_sigma: Gaussian probe initialization width (pixels).
                initial_object_amplitude: Initial uniform object amplitude.
                eps: Numerical stability constant.
                total_time_budget: Total wall-clock time budget (seconds).
                safety_margin: Time reserved for post-processing (seconds).
                well_lit_threshold: Fraction of max scan coverage for well-lit mask.
                valid_amp_threshold: Fraction of max amplitude for valid pixel mask.
                min_valid_pixels: Minimum valid pixels required for phase correction.
            """
            self.probe_size: int = probe_size
            self.object_size: int = object_size
            self.n_side: int = n_side
            self.step: int = step
            self.n_positions: int = n_positions
            self.beta_object: float = beta_object
            self.beta_probe: float = beta_probe
            self.probe_update_start_epoch: int = probe_update_start_epoch
            self.probe_sigma: float = probe_sigma
            self.initial_object_amplitude: float = initial_object_amplitude
            self.eps: float = eps
            self.total_time_budget: float = total_time_budget
            self.safety_margin: float = safety_margin
            self.well_lit_threshold: float = well_lit_threshold
            self.valid_amp_threshold: float = valid_amp_threshold
            self.min_valid_pixels: int = min_valid_pixels

            # Derived: compute scan positions immediately
            total_scan: int = self.probe_size + (self.n_side - 1) * self.step  # 256 + 8*40 = 576
            offset: int = (self.object_size - total_scan) // 2  # (600 - 576) // 2 = 12

            positions_list: List[Tuple[int, int]] = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions_list.append((offset + i * self.step, offset + j * self.step))
            self.positions: np.ndarray = np.array(positions_list, dtype=np.int32)  # (81, 2)

            # Precompute scan coverage map
            self.scan_count: np.ndarray = np.zeros((self.object_size, self.object_size), dtype=np.float64)
            for j in range(self.n_positions):
                ry: int = int(self.positions[j, 0])
                rx: int = int(self.positions[j, 1])
                self.scan_count[ry:ry + self.probe_size, rx:rx + self.probe_size] += 1.0

            # Track start time (will be set in solve())
            self.t_start: float = 0.0
        # TODO: Compute in solve()

    def _compute_scan_positions(self) -> np.ndarray:
            """
            Compute the (row, col) scan positions for the 9x9 grid.

            Returns:
                Array of shape (n_positions, 2) with (row, col) offsets.
            """
            total_scan: int = self.probe_size + (self.n_side - 1) * self.step
            offset: int = (self.object_size - total_scan) // 2

            positions_list: List[Tuple[int, int]] = []
            for i in range(self.n_side):
                for j in range(self.n_side):
                    positions_list.append((offset + i * self.step, offset + j * self.step))
            return np.array(positions_list, dtype=np.int32)

    def _make_initial_probe(self, diff_amp: np.ndarray) -> np.ndarray:
            """
            Create energy-matched Gaussian probe.

            Args:
                diff_amp: Diffraction amplitude array of shape (n_positions, probe_size, probe_size).

            Returns:
                Complex probe array of shape (probe_size, probe_size).
            """
            ps: int = self.probe_size
            yy, xx = np.mgrid[:ps, :ps]
            cy: float = ps / 2.0
            cx: float = ps / 2.0
            r2: np.ndarray = (yy - cy) ** 2 + (xx - cx) ** 2
            sigma: float = self.probe_sigma
            P: np.ndarray = np.exp(-r2 / (2.0 * sigma ** 2)).astype(np.complex128)

            # Energy matching via Parseval's theorem:
            # sum|FFT(P * O)|^2 = ps^2 * sum|P * O|^2
            # Assuming object amplitude ~ initial_object_amplitude:
            # predicted_data_energy = ps^2 * sum|P|^2 * obj_amp^2
            # We want this to match mean measured data energy = mean(sum(diff_amp^2))
            mean_data_energy: float = float(np.mean(np.sum(diff_amp ** 2, axis=(-2, -1))))
            obj_amp_est: float = self.initial_object_amplitude
            current_exit_energy: float = float(np.sum(np.abs(P) ** 2)) * obj_amp_est ** 2
            predicted_data_energy: float = float(ps * ps) * current_exit_energy
            scale: float = np.sqrt(mean_data_energy / (predicted_data_energy + self.eps))
            P *= scale

            return P

    def _make_initial_object(self) -> np.ndarray:
            """
            Create uniform initial object estimate.

            Returns:
                Complex object array of shape (object_size, object_size).
            """
            return self.initial_object_amplitude * np.ones(
                (self.object_size, self.object_size), dtype=np.complex128
            )

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Compute forward model: far-field diffraction amplitude for one scan position.

            Computes |FFT2(P * O_patch)| for the given scan position.

            Args:
                obj: Complex object array of shape (object_size, object_size).
                probe: Complex probe array of shape (probe_size, probe_size).
                position: (row, col) top-left corner of the patch in the object.

            Returns:
                Predicted diffraction amplitude of shape (probe_size, probe_size).
            """
            ry, rx = position
            ps: int = self.probe_size
            O_patch: np.ndarray = obj[ry:ry + ps, rx:rx + ps]
            psi: np.ndarray = probe * O_patch
            Psi: np.ndarray = np.fft.fft2(psi)
            return np.abs(Psi)

    def _epie_update_object(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            delta_psi: np.ndarray,
            position: Tuple[int, int],
            p_max_sq: float,
        ) -> None:
            ry, rx = position
            ps: int = self.probe_size
            obj[ry:ry + ps, rx:rx + ps] += self.beta_object * np.conj(probe) * delta_psi / (p_max_sq + self.eps)

    def _epie_update_probe(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            delta_psi: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Apply ePIE probe update with scalar max-norm denominator.

            P += beta_P * conj(O_patch) * delta_psi / (max|O_patch|^2 + eps)

            Args:
                obj: Complex object array.
                probe: Current complex probe array.
                delta_psi: Exit wave difference.
                position: (row, col) scan position.

            Returns:
                Updated complex probe array.
            """
            ry, rx = position
            ps: int = self.probe_size
            O_patch: np.ndarray = obj[ry:ry + ps, rx:rx + ps]
            O_max_sq: float = float(np.max(np.abs(O_patch) ** 2))
            probe = probe + self.beta_probe * np.conj(O_patch) * delta_psi / (O_max_sq + self.eps)
            return probe

    def _fourier_magnitude_replacement(
            self,
            psi: np.ndarray,
            measured_amp: np.ndarray,
        ) -> np.ndarray:
            """
            Replace Fourier magnitude while preserving phase.

            Psi_c = measured_amp * Psi / (|Psi| + eps)
            psi_c = ifft2(Psi_c)

            Args:
                psi: Exit wave in real space, shape (probe_size, probe_size).
                measured_amp: Measured diffraction amplitude, shape (probe_size, probe_size).

            Returns:
                Corrected exit wave in real space.
            """
            Psi: np.ndarray = np.fft.fft2(psi)
            amp_Psi: np.ndarray = np.abs(Psi)
            Psi_c: np.ndarray = measured_amp * Psi / (amp_Psi + self.eps)
            psi_c: np.ndarray = np.fft.ifft2(Psi_c)
            return psi_c

    def _run_epie(
            self,
            diff_amp: np.ndarray,
            time_budget: float,
        ) -> Tuple[np.ndarray, np.ndarray, float, int]:
            """
            Run ePIE reconstruction for one FFT convention.

            Args:
                diff_amp: Diffraction amplitudes, shape (n_positions, probe_size, probe_size).
                          Already in the correct FFT convention (DC at corners).
                time_budget: Wall-clock time budget in seconds.

            Returns:
                Tuple of (object, probe, relative_residual, num_epochs).
            """
            P: np.ndarray = self._make_initial_probe(diff_amp)
            O: np.ndarray = self._make_initial_object()

            t0: float = time.time()
            epoch: int = 0
            ps: int = self.probe_size

            while True:
                elapsed: float = time.time() - t0
                if elapsed > time_budget:
                    break

                order: np.ndarray = np.random.permutation(self.n_positions)

                P_max_sq: float = float(np.max(np.abs(P) ** 2))

                for idx in range(self.n_positions):
                    # Check time every 20 positions
                    if idx % 20 == 0 and (time.time() - t0) > time_budget:
                        break

                    j: int = int(order[idx])
                    ry: int = int(self.positions[j, 0])
                    rx: int = int(self.positions[j, 1])

                    # Extract object patch and compute exit wave
                    O_patch: np.ndarray = O[ry:ry + ps, rx:rx + ps].copy()
                    psi: np.ndarray = P * O_patch

                    # Forward propagate
                    Psi: np.ndarray = np.fft.fft2(psi)
                    amp_Psi: np.ndarray = np.abs(Psi)

                    # Fourier magnitude replacement
                    Psi_c: np.ndarray = diff_amp[j] * Psi / (amp_Psi + self.eps)
                    psi_c: np.ndarray = np.fft.ifft2(Psi_c)

                    # Exit wave difference
                    dpsi: np.ndarray = psi_c - psi

                    # Object update with scalar max-norm denominator
                    self._epie_update_object(O, P, dpsi, (ry, rx), P_max_sq)

                    # Probe update after warmup epochs
                    if epoch >= self.probe_update_start_epoch:
                        P = self._epie_update_probe(O, P, dpsi, (ry, rx))

                # Update P_max_sq for next epoch
                P_max_sq = float(np.max(np.abs(P) ** 2))
                epoch += 1

            # Compute final residual
            residual: float = 0.0
            norm: float = 0.0
            for j in range(self.n_positions):
                ry = int(self.positions[j, 0])
                rx = int(self.positions[j, 1])
                psi_final: np.ndarray = P * O[ry:ry + ps, rx:rx + ps]
                Psi_final: np.ndarray = np.fft.fft2(psi_final)
                residual += float(np.sum((np.abs(Psi_final) - diff_amp[j]) ** 2))
                norm += float(np.sum(diff_amp[j] ** 2))

            rel_res: float = residual / (norm + self.eps)
            return O, P, rel_res, epoch

    def _compute_residual(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            diff_amp: np.ndarray,
        ) -> float:
            """
            Compute normalized Fourier magnitude residual over all scan positions.

            residual = sum_j ||sqrt(I_j) - |FFT(P * O_j)||^2 / sum_j ||sqrt(I_j)||^2

            Args:
                obj: Reconstructed complex object.
                probe: Reconstructed complex probe.
                diff_amp: Measured diffraction amplitudes.

            Returns:
                Relative residual (scalar).
            """
            ps: int = self.probe_size
            residual: float = 0.0
            norm: float = 0.0
            for j in range(self.n_positions):
                ry: int = int(self.positions[j, 0])
                rx: int = int(self.positions[j, 1])
                O_patch: np.ndarray = obj[ry:ry + ps, rx:rx + ps]
                psi: np.ndarray = probe * O_patch
                Psi: np.ndarray = np.fft.fft2(psi)
                residual += float(np.sum((np.abs(Psi) - diff_amp[j]) ** 2))
                norm += float(np.sum(diff_amp[j] ** 2))
            rel_res: float = residual / (norm + self.eps)
            return rel_res

    def _compute_scan_coverage(self) -> np.ndarray:
            """
            Compute per-pixel scan overlap count.

            Returns:
                Array of shape (object_size, object_size) with scan counts.
            """
            coverage: np.ndarray = np.zeros((self.object_size, self.object_size), dtype=np.float64)
            for j in range(self.n_positions):
                ry: int = int(self.positions[j, 0])
                rx: int = int(self.positions[j, 1])
                coverage[ry:ry + self.probe_size, rx:rx + self.probe_size] += 1.0
            return coverage

    def _postprocess(
            self,
            obj: np.ndarray,
        ) -> np.ndarray:
            """
            Minimal post-processing: weighted mean phase removal, zero non-illuminated regions.

            No histogram-based correction. Only removes global phase ambiguity
            via amplitude-weighted mean phase in well-illuminated region.

            Args:
                obj: Reconstructed complex object of shape (object_size, object_size).

            Returns:
                Output array of shape (2, object_size, object_size) with
                channel 0 = amplitude, channel 1 = phase (float32).
            """
            O_final: np.ndarray = obj.copy()

            # Identify illuminated vs non-illuminated regions from precomputed scan_count
            illuminated: np.ndarray = self.scan_count > 0

            # Well-lit mask: pixels with coverage >= well_lit_threshold * max_coverage
            max_coverage: float = float(np.max(self.scan_count))
            well_lit: np.ndarray = self.scan_count >= self.well_lit_threshold * max_coverage

            # Amplitude of the object
            amp: np.ndarray = np.abs(O_final)
            max_amp: float = float(np.max(amp))

            # Valid pixels: well-lit AND amplitude above threshold
            valid: np.ndarray = well_lit & (amp > self.valid_amp_threshold * max_amp)

            if int(np.sum(valid)) > self.min_valid_pixels:
                # Amplitude-weighted mean phase removal
                weights: np.ndarray = amp[valid]
                # Compute weighted complex sum to find mean phase
                weighted_sum: complex = complex(np.sum(weights * O_final[valid]))
                mean_phase: float = float(np.angle(weighted_sum))
                O_final *= np.exp(-1j * mean_phase)

            # Zero out non-illuminated regions
            O_final[~illuminated] = 0.0

            # Build output: channel 0 = amplitude, channel 1 = phase
            output: np.ndarray = np.zeros((2, self.object_size, self.object_size), dtype=np.float32)
            output[0] = np.abs(O_final).astype(np.float32)
            output[1] = np.angle(O_final).astype(np.float32)

            return output

    def solve(
            self,
            input_data: np.ndarray,
        ) -> np.ndarray:
            """
            Main entry point: run ePIE with both FFT conventions, pick winner.

            Runs Convention A (raw diff_amp, DC at corners) and Convention B
            (ifftshift applied, DC shifted from center to corners) each for
            half the available time budget. Selects the reconstruction with
            lower Fourier residual.

            Args:
                input_data: Measured diffraction intensities, shape (n_positions, probe_size, probe_size).

            Returns:
                Reconstructed object as array of shape (2, object_size, object_size),
                channel 0 = amplitude, channel 1 = phase (float32).
            """
            self.t_start = time.time()

            # 1. Preprocess input: clip negatives, compute diffraction amplitudes
            I: np.ndarray = np.maximum(input_data, 0).astype(np.float64)  # (81, 256, 256)
            diff_amp: np.ndarray = np.sqrt(I)  # (81, 256, 256)

            # 3. Prepare both FFT conventions
            # Convention A: data is already in FFT-native order (DC at corners)
            diff_amp_A: np.ndarray = diff_amp.copy()
            # Convention B: data is DC-centered, shift DC from center to corners
            diff_amp_B: np.ndarray = np.fft.ifftshift(diff_amp, axes=(-2, -1))

            # 4 & 5. Run ePIE for both conventions with half the time budget each
            time_remaining: float = self.total_time_budget - (time.time() - self.t_start) - self.safety_margin
            time_per_convention: float = time_remaining / 2.0

            O_A, P_A, res_A, ep_A = self._run_epie(diff_amp_A, time_per_convention)
            O_B, P_B, res_B, ep_B = self._run_epie(diff_amp_B, time_per_convention)

            # 6. Compare residuals, select winner
            if res_B < res_A:
                O_final: np.ndarray = O_B
            else:
                O_final = O_A

            # 7. Post-process and return
            output: np.ndarray = self._postprocess(O_final)

            return output



if __name__ == "__main__":
    # Load data
    input_data: np.ndarray = np.load("dataset/input.npy")

    # Create solver and run
    solver: InverseSolver = InverseSolver()
    result: np.ndarray = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)