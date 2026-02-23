import numpy as np
from typing import Tuple, List, Optional
from numba import njit, prange

class InverseSolver:
    """
    ePIE (extended Ptychographic Iterative Engine) solver for coherent
    diffractive imaging phase retrieval.

    Recovers a complex-valued 2D object reflection function O(r) from
    a set of far-field diffraction intensity patterns collected on a
    ptychographic scan grid.
    """

    def __init__(
            self,
            object_shape: Tuple[int, int] = (600, 600),
            probe_shape: Tuple[int, int] = (256, 256),
            grid_size: Tuple[int, int] = (9, 9),
            scan_step: int = 43,
            n_iterations: int = 300,
            alpha_obj: float = 1.0,
            alpha_probe: float = 1.0,
            regularization: float = 1e-3,
            probe_radius: float = 100.0,
            probe_edge_sigma: float = 10.0,
            magnitude_clip_max: float = 5.0,
            convergence_threshold: float = 1e-6,
            wavelength_nm: float = 13.5,
            pixel_size_nm: float = 27.0,
            eps_division: float = 1e-12,
            update_probe: bool = True,
            probe_update_start_epoch: int = 5,
            seed: Optional[int] = 42,
        ) -> None:
            self.object_shape = object_shape
            self.probe_shape = probe_shape
            self.grid_size = grid_size
            self.scan_step = scan_step
            self.n_iterations = n_iterations
            self.alpha_obj = alpha_obj
            self.alpha_probe = alpha_probe
            self.regularization = regularization
            self.probe_radius = probe_radius
            self.probe_edge_sigma = probe_edge_sigma
            self.magnitude_clip_max = magnitude_clip_max
            self.convergence_threshold = convergence_threshold
            self.wavelength_nm = wavelength_nm
            self.pixel_size_nm = pixel_size_nm
            self.eps_division = eps_division
            self.update_probe = update_probe
            self.probe_update_start_epoch = probe_update_start_epoch
            self.seed = seed

            # State variables populated during solve()
            self.object: Optional[np.ndarray] = None
            self.probe: Optional[np.ndarray] = None
            self.scan_positions: Optional[np.ndarray] = None
            self.centered_data: bool = True

            # Set random seed for reproducibility
            if self.seed is not None:
                np.random.seed(self.seed)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _build_scan_positions(self) -> np.ndarray:
            """
            Compute the (row, col) top-left corner of each probe window on
            the object grid for a raster scan.

            Returns
            -------
            np.ndarray
                Array of shape (N_positions, 2) with integer pixel offsets.
            """
            ny, nx = self.grid_size
            positions = np.zeros((ny * nx, 2), dtype=np.int64)
            idx = 0
            for iy in range(ny):
                for ix in range(nx):
                    positions[idx, 0] = iy * self.scan_step
                    positions[idx, 1] = ix * self.scan_step
                    idx += 1
            return positions

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_object(self) -> np.ndarray:
            """
            Create the initial complex-valued object estimate.

            Returns
            -------
            np.ndarray
                Complex128 array of shape self.object_shape.
            """
            return np.ones(self.object_shape, dtype=np.complex128)

    def _init_probe(self, measured_amplitudes: np.ndarray) -> np.ndarray:
            """
            Create the initial probe estimate (Gaussian-tapered circular
            aperture) scaled to match total measured intensity.

            Parameters
            ----------
            measured_amplitudes : np.ndarray
                Square-root of measured intensities, shape (N, H, W).

            Returns
            -------
            np.ndarray
                Complex128 array of shape self.probe_shape.
            """
            H, W = self.probe_shape
            # Create coordinate grids centered at (H//2, W//2)
            Y, X = np.mgrid[0:H, 0:W]
            Y = Y - H / 2.0
            X = X - W / 2.0
            R = np.sqrt(X**2 + Y**2)

            # Gaussian-tapered circular aperture
            # Inside probe_radius: amplitude = 1
            # Outside: Gaussian falloff with sigma = probe_edge_sigma
            P = np.exp(-0.5 * np.maximum(R - self.probe_radius, 0.0)**2 / self.probe_edge_sigma**2)
            P = P.astype(np.complex128)

            # Scale probe so that sum(|P|^2) ≈ mean total measured intensity
            # measured_amplitudes has shape (N, H, W), so measured intensity per pattern
            # is sum(amplitudes^2) = sum(data) for that pattern
            mean_total_intensity = np.mean(np.sum(measured_amplitudes**2, axis=(1, 2)))
            current_probe_power = np.sum(np.abs(P)**2)
            if current_probe_power > 0:
                scale = np.sqrt(mean_total_intensity / current_probe_power)
                P *= scale

            return P

    # ------------------------------------------------------------------
    # FFT convention detection
    # ------------------------------------------------------------------

    def _detect_fft_convention(self, data: np.ndarray) -> bool:
            """
            Auto-detect whether the diffraction patterns have DC at the
            centre (shifted) or at the corner (unshifted).

            Parameters
            ----------
            data : np.ndarray
                Raw intensity data, shape (N, H, W).

            Returns
            -------
            bool
                True if data is centred (DC at centre), False otherwise.
            """
            avg_pattern = np.mean(data, axis=0)
            H, W = avg_pattern.shape
            center_val = avg_pattern[H // 2, W // 2]
            corner_val = avg_pattern[0, 0]
            # Also check a small region around center and corner for robustness
            cr = 5
            center_region = avg_pattern[H//2 - cr:H//2 + cr, W//2 - cr:W//2 + cr].mean()
            corner_region = avg_pattern[:2*cr, :2*cr].mean()
            return bool(center_region > corner_region)

    # ------------------------------------------------------------------
    # Forward / inverse propagation
    # ------------------------------------------------------------------

    def _forward_propagate(self, exit_wave: np.ndarray) -> np.ndarray:
            """
            Propagate an exit wave to the far-field detector plane using
            the 2D FFT (Fraunhofer approximation), respecting the detected
            shift convention.

            Parameters
            ----------
            exit_wave : np.ndarray
                Complex array of shape self.probe_shape.

            Returns
            -------
            np.ndarray
                Complex far-field wavefield, same shape.
            """
            if self.centered_data:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(exit_wave)))
            else:
                return np.fft.fft2(exit_wave)

    def _backward_propagate(self, far_field: np.ndarray) -> np.ndarray:
            """
            Inverse-propagate a far-field wavefield back to the sample
            plane.

            Parameters
            ----------
            far_field : np.ndarray
                Complex array of shape self.probe_shape.

            Returns
            -------
            np.ndarray
                Complex exit wave, same shape.
            """
            if self.centered_data:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field)))
            else:
                return np.fft.ifft2(far_field)

    # ------------------------------------------------------------------
    # Fourier constraint
    # ------------------------------------------------------------------

    def _modulus_replace(
            self,
            far_field: np.ndarray,
            measured_amplitude: np.ndarray,
        ) -> np.ndarray:
            """
            Replace the modulus of the far-field estimate with the measured
            amplitude while preserving the phase.

            Parameters
            ----------
            far_field : np.ndarray
                Current far-field estimate, complex, shape (H, W).
            measured_amplitude : np.ndarray
                Square-root of measured intensity, real, shape (H, W).

            Returns
            -------
            np.ndarray
                Corrected far-field with measured modulus, complex, shape (H, W).
            """
            mag = np.abs(far_field)
            eps = self.eps_division * np.max(mag) if np.max(mag) > 0 else self.eps_division
            corrected = measured_amplitude * far_field / (mag + eps)
            return corrected

    # ------------------------------------------------------------------
    # ePIE update steps
    # ------------------------------------------------------------------

    def _update_object(
            self,
            obj_patch: np.ndarray,
            probe: np.ndarray,
            delta_psi: np.ndarray,
        ) -> np.ndarray:
            """
            ePIE object patch update.

            Parameters
            ----------
            obj_patch : np.ndarray
                Current object patch, complex, shape (H, W).
            probe : np.ndarray
                Current probe estimate, complex, shape (H, W).
            delta_psi : np.ndarray
                Exit-wave difference (corrected − current), complex, shape (H, W).

            Returns
            -------
            np.ndarray
                Updated object patch, complex, shape (H, W).
            """
            probe_conj = np.conj(probe)
            max_probe_intensity = np.max(np.abs(probe) ** 2)
            updated_patch = obj_patch + self.alpha_obj * probe_conj / (max_probe_intensity + self.regularization) * delta_psi
            return updated_patch

    def _update_probe(
            self,
            probe: np.ndarray,
            obj_patch: np.ndarray,
            delta_psi: np.ndarray,
        ) -> np.ndarray:
            """
            ePIE probe update (blind ptychography).

            Parameters
            ----------
            probe : np.ndarray
                Current probe estimate, complex, shape (H, W).
            obj_patch : np.ndarray
                Object patch at current scan position, complex, shape (H, W).
            delta_psi : np.ndarray
                Exit-wave difference, complex, shape (H, W).

            Returns
            -------
            np.ndarray
                Updated probe, complex, shape (H, W).
            """
            obj_conj = np.conj(obj_patch)
            max_obj_intensity = np.max(np.abs(obj_patch) ** 2)
            updated_probe = probe + self.alpha_probe * obj_conj / (max_obj_intensity + self.regularization) * delta_psi
            return updated_probe

    # ------------------------------------------------------------------
    # Error metrics
    # ------------------------------------------------------------------

    def _fourier_error(
            self,
            measured_amplitudes: np.ndarray,
        ) -> float:
            """
            Compute the mean normalised Fourier-domain error across all
            scan positions.

            Parameters
            ----------
            measured_amplitudes : np.ndarray
                Shape (N, H, W), square-root of measured intensities.

            Returns
            -------
            float
                Scalar Fourier error.
            """
            N = measured_amplitudes.shape[0]
            total_error = 0.0
            for j in range(N):
                ry, rx = self.scan_positions[j]
                obj_patch = self.object[ry:ry + self.probe_shape[0], rx:rx + self.probe_shape[1]]
                exit_wave = self.probe * obj_patch
                far_field = self._forward_propagate(exit_wave)
                estimated_amp = np.abs(far_field)
                meas_amp = measured_amplitudes[j]
                diff = estimated_amp - meas_amp
                numerator = np.sum(diff ** 2)
                denominator = np.sum(meas_amp ** 2)
                if denominator > 0:
                    total_error += numerator / denominator
                else:
                    total_error += numerator
            return total_error / N

    def _object_change(self, O_prev: np.ndarray) -> float:
            """
            Compute relative change in the object between successive
            epochs.

            Parameters
            ----------
            O_prev : np.ndarray
                Object estimate from previous epoch, complex, shape object_shape.

            Returns
            -------
            float
                Relative L2 change.
            """
            norm_prev = np.linalg.norm(O_prev)
            if norm_prev < 1e-30:
                return np.linalg.norm(self.object - O_prev)
            return np.linalg.norm(self.object - O_prev) / norm_prev

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _remove_global_phase(self) -> None:
            """
            Remove the arbitrary global phase offset from the
            reconstructed object by setting the mean phase of bright
            regions to zero.
            """
            mag = np.abs(self.object)
            threshold = 0.5 * np.max(mag)
            bright_mask = mag > threshold
            if np.any(bright_mask):
                global_phase = np.angle(np.mean(self.object[bright_mask]))
            else:
                global_phase = np.angle(np.mean(self.object))
            self.object = self.object * np.exp(-1j * global_phase)

    def _clip_object_magnitude(self) -> None:
            """
            Clip the object magnitude to a physically plausible range
            [0, magnitude_clip_max].
            """
            mag = np.abs(self.object)
            phase = np.angle(self.object)
            mag_clipped = np.clip(mag, 0.0, self.magnitude_clip_max)
            self.object = mag_clipped * np.exp(1j * phase)

    # ------------------------------------------------------------------
    # Core public interface
    # ------------------------------------------------------------------

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Simulate the full set of diffraction intensity patterns from a
            known object and probe.

            Parameters
            ----------
            obj : np.ndarray
                Complex object, shape object_shape.
            probe : np.ndarray
                Complex probe, shape probe_shape.
            positions : np.ndarray
                Scan positions, shape (N, 2).

            Returns
            -------
            np.ndarray
                Simulated intensities, shape (N, H, W), real non-negative.
            """
            N = positions.shape[0]
            H, W = self.probe_shape
            intensities = np.zeros((N, H, W), dtype=np.float64)
            for j in range(N):
                ry, rx = positions[j]
                obj_patch = obj[ry:ry + H, rx:rx + W]
                exit_wave = probe * obj_patch
                far_field = self._forward_propagate(exit_wave)
                intensities[j] = np.abs(far_field) ** 2
            return intensities

    def solve(self, data: np.ndarray) -> np.ndarray:
            """
            Run the ePIE reconstruction to recover the complex object from
            measured diffraction intensities.

            Parameters
            ----------
            data : np.ndarray
                Measured intensity patterns, shape (N_patterns, H, W),
                real-valued, non-negative.

            Returns
            -------
            np.ndarray
                Reconstructed object as a (2, obj_H, obj_W) array where
                channel 0 is magnitude and channel 1 is phase (radians).
            """
            # 1. Compute amplitudes = sqrt(max(data, 0))
            amplitudes = np.sqrt(np.maximum(data, 0.0))
            N = data.shape[0]
            H, W = self.probe_shape

            # 2. Detect FFT convention
            self.centered_data = self._detect_fft_convention(data)

            # 3. Build scan positions
            self.scan_positions = self._build_scan_positions()

            # 4. Initialise object and probe
            self.object = self._init_object()
            self.probe = self._init_probe(amplitudes)

            # 5. ePIE main loop
            O_prev = self.object.copy()

            for epoch in range(self.n_iterations):
                # 5a. Randomise scan order
                order = np.random.permutation(N)

                for j_idx in range(N):
                    j = order[j_idx]
                    ry, rx = self.scan_positions[j]

                    # 5b-i. Extract object patch
                    obj_patch = self.object[ry:ry + H, rx:rx + W].copy()

                    # 5b-ii. Form exit wave
                    psi = self.probe * obj_patch

                    # 5b-iii. Forward propagate
                    Psi_hat = self._forward_propagate(psi)

                    # 5b-iv. Modulus replace
                    Psi_hat_c = self._modulus_replace(Psi_hat, amplitudes[j])

                    # 5b-v. Back propagate
                    psi_c = self._backward_propagate(Psi_hat_c)

                    # 5b-vi. Compute delta
                    delta_psi = psi_c - psi

                    # 5b-vii. Update object patch
                    updated_patch = self._update_object(obj_patch, self.probe, delta_psi)
                    self.object[ry:ry + H, rx:rx + W] = updated_patch

                    # 5b-viii. Update probe (if enabled and epoch > start)
                    if self.update_probe and epoch >= self.probe_update_start_epoch:
                        self.probe = self._update_probe(self.probe, obj_patch, delta_psi)

                # 5c. Optionally clip object magnitude every 10 epochs after epoch 50
                if epoch > 50 and epoch % 10 == 0:
                    self._clip_object_magnitude()

                # 5d. Check convergence (object change)
                delta_obj = self._object_change(O_prev)
                O_prev = self.object.copy()

                if delta_obj < self.convergence_threshold and epoch > 20:
                    break

            # 6. Remove global phase
            self._remove_global_phase()

            # 7. Return np.stack([|O|, arg(O)], axis=0)
            obj_mag = np.abs(self.object)
            obj_phase = np.angle(self.object)
            output = np.zeros((2, self.object_shape[0], self.object_shape[1]), dtype=np.float64)
            output[0, :, :] = obj_mag
            output[1, :, :] = obj_phase
            return output



if __name__ == "__main__":
    # Load measured diffraction intensity data
    input_data: np.ndarray = np.load("dataset/input.npy")

    # Validate input
    assert input_data.shape[0] == 81, f"Expected 81 patterns, got {input_data.shape[0]}"
    assert input_data.shape[1] == input_data.shape[2] == 256, f"Expected 256x256 patterns"

    # Output must be (2, 600, 600)
    obj_H, obj_W = 600, 600
    probe_H, probe_W = 256, 256
    grid_ny, grid_nx = 9, 9
    scan_step = (obj_H - probe_H) // (grid_ny - 1)  # = 43

    # Instantiate solver with tuned hyperparameters
    solver = InverseSolver(
        object_shape=(obj_H, obj_W),
        probe_shape=(probe_H, probe_W),
        grid_size=(grid_ny, grid_nx),
        scan_step=scan_step,
        n_iterations=300,
        alpha_obj=1.0,
        alpha_probe=1.0,
        regularization=1e-3,
        probe_radius=100.0,
        probe_edge_sigma=10.0,
        magnitude_clip_max=5.0,
        convergence_threshold=1e-6,
        update_probe=True,
        probe_update_start_epoch=5,
        seed=42,
    )

    # Run ePIE reconstruction
    result: np.ndarray = solver.solve(input_data)

    # Verify output shape
    assert result.shape == (2, obj_H, obj_W), f"Output shape {result.shape} != (2, {obj_H}, {obj_W})"

    # Persist result — shape (2, 600, 600): [magnitude, phase]
    np.save("output.npy", result)