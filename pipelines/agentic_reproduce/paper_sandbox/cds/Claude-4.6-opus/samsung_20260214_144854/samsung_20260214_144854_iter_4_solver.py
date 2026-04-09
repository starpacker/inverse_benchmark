import numpy as np
from typing import Tuple, List, Optional, Callable, Dict

class InverseSolver:
    """
    Coherent Diffractive Imaging (CDI) phase retrieval solver using rPIE
    (regularized Ptychographic Iterative Engine) for recovering complex-valued
    object reflection functions from far-field diffraction intensity patterns.
    """

    def __init__(
            self,
            obj_size: int = 600,
            probe_size: int = 256,
            n_scan: int = 9,
            n_iter: int = 800,
            alpha_O: float = 0.1,
            alpha_P: float = 0.1,
            alpha_O_stage2: float = 0.05,
            alpha_P_stage2: float = 0.05,
            alpha_O_stage3: float = 0.02,
            alpha_P_stage3: float = 0.02,
            stage2_epoch: int = 200,
            stage3_epoch: int = 500,
            probe_update_start: int = 50,
            probe_sigma: float = 60.0,
            modulus_eps: float = 1e-8,
            wavelength_nm: float = 13.5,
            pixel_size_nm: float = 27.0,
            n_conventions_to_test: int = 4,
            convention_test_iters: int = 10,
            monitor_interval: int = 50,
            seed: Optional[int] = 42,
        ) -> None:
            self.obj_size = obj_size
            self.probe_size = probe_size
            self.n_scan = n_scan
            self.n_iter = n_iter
            self.alpha_O = alpha_O
            self.alpha_P = alpha_P
            self.alpha_O_stage2 = alpha_O_stage2
            self.alpha_P_stage2 = alpha_P_stage2
            self.alpha_O_stage3 = alpha_O_stage3
            self.alpha_P_stage3 = alpha_P_stage3
            self.stage2_epoch = stage2_epoch
            self.stage3_epoch = stage3_epoch
            self.probe_update_start = probe_update_start
            self.probe_sigma = probe_sigma
            self.modulus_eps = modulus_eps
            self.wavelength_nm = wavelength_nm
            self.pixel_size_nm = pixel_size_nm
            self.n_conventions_to_test = n_conventions_to_test
            self.convention_test_iters = convention_test_iters
            self.monitor_interval = monitor_interval
            self.seed = seed

            # Derived: scan step size
            self.scan_step: int = (obj_size - probe_size) // (n_scan - 1)

            # Scan positions (will be computed in _compute_scan_positions)
            self.positions: List[Tuple[int, int]] = []

            # FFT convention functions (will be set in _select_fft_convention)
            self.fft_forward: Optional[Callable] = None
            self.fft_backward: Optional[Callable] = None

            # Reconstruction state (will be set during solve)
            self.object_estimate: Optional[np.ndarray] = None
            self.probe_estimate: Optional[np.ndarray] = None

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
            """
            Compute the raster scan positions for the 9x9 ptychographic grid.

            Returns:
                List of (row, col) tuples for each scan position.
            """
            positions = []
            for iy in range(self.n_scan):
                for ix in range(self.n_scan):
                    positions.append((iy * self.scan_step, ix * self.scan_step))
            self.positions = positions
            return positions

    def _define_fft_conventions(self) -> List[Tuple[Callable, Callable]]:
            """
            Define all candidate FFT forward/backward convention pairs.

            Returns:
                List of (forward_fn, backward_fn) tuples to test.
            """
            # Convention 0: bare FFT (DC at corner)
            def fwd_0(x: np.ndarray) -> np.ndarray:
                return np.fft.fft2(x)
            def bwd_0(x: np.ndarray) -> np.ndarray:
                return np.fft.ifft2(x)

            # Convention 1: fftshift output (DC at center in Fourier domain)
            def fwd_1(x: np.ndarray) -> np.ndarray:
                return np.fft.fftshift(np.fft.fft2(x))
            def bwd_1(x: np.ndarray) -> np.ndarray:
                return np.fft.ifft2(np.fft.ifftshift(x))

            # Convention 2: ifftshift input, fftshift output (centered both domains)
            def fwd_2(x: np.ndarray) -> np.ndarray:
                return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            def bwd_2(x: np.ndarray) -> np.ndarray:
                return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

            # Convention 3: fftshift input (object centered, DC at corner)
            def fwd_3(x: np.ndarray) -> np.ndarray:
                return np.fft.fft2(np.fft.fftshift(x))
            def bwd_3(x: np.ndarray) -> np.ndarray:
                return np.fft.ifftshift(np.fft.ifft2(x))

            conventions: List[Tuple[Callable, Callable]] = [
                (fwd_0, bwd_0),
                (fwd_1, bwd_1),
                (fwd_2, bwd_2),
                (fwd_3, bwd_3),
            ]

            return conventions

    def _select_fft_convention(
            self,
            amplitudes: np.ndarray,
            conventions: List[Tuple[Callable, Callable]],
        ) -> Tuple[Callable, Callable]:
            """
            Test each FFT convention with a few rPIE iterations and select the
            one producing the lowest Fourier error.

            Args:
                amplitudes: Square root of measured intensities, shape (n_positions, probe_size, probe_size).
                conventions: List of (forward_fn, backward_fn) pairs.

            Returns:
                The best (forward_fn, backward_fn) pair.
            """
            if self.seed is not None:
                np.random.seed(self.seed)

            n_positions = amplitudes.shape[0]
            ps = self.probe_size
            best_error = np.inf
            best_convention = conventions[0]

            # Create Gaussian probe for testing
            Y, X = np.mgrid[-ps // 2:ps // 2, -ps // 2:ps // 2]
            R2 = X.astype(np.float64)**2 + Y.astype(np.float64)**2
            sigma = self.probe_sigma
            P_init = np.exp(-R2 / (2.0 * sigma * sigma)).astype(np.complex128)

            # Calibrate probe power using Parseval's theorem
            mean_total_intensity = np.mean(np.sum(amplitudes**2, axis=(1, 2)))
            probe_power = mean_total_intensity / float(ps * ps)
            current_power = np.sum(np.abs(P_init)**2)
            if current_power > 0:
                P_init *= np.sqrt(probe_power / current_power)

            for conv_idx, (fwd_fn, bwd_fn) in enumerate(conventions):
                # Fresh initialization for each convention test
                O_test = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
                P_test = P_init.copy()

                alpha = 0.1

                for iteration in range(self.convention_test_iters):
                    order = np.random.permutation(n_positions)
                    for j in order:
                        ry, rx = self.positions[j]

                        # Extract object patch
                        O_patch = O_test[ry:ry + ps, rx:rx + ps].copy()

                        # Exit wave
                        psi = P_test * O_patch

                        # Forward propagate
                        Psi_hat = fwd_fn(psi)

                        # Modulus replacement
                        mag = np.abs(Psi_hat)
                        Psi_hat_c = amplitudes[j] * Psi_hat / (mag + self.modulus_eps)

                        # Back propagate
                        psi_c = bwd_fn(Psi_hat_c)

                        # Delta
                        dpsi = psi_c - psi

                        # Object update (rPIE)
                        P_abs_sq = np.abs(P_test)**2
                        P_max_sq = np.max(P_abs_sq)
                        denom_O = (1.0 - alpha) * P_abs_sq + alpha * P_max_sq
                        O_test[ry:ry + ps, rx:rx + ps] += np.conj(P_test) / denom_O * dpsi

                # Compute Fourier error for this convention
                total_error = 0.0
                total_norm = 0.0
                for j in range(n_positions):
                    ry, rx = self.positions[j]
                    O_patch = O_test[ry:ry + ps, rx:rx + ps]
                    psi = P_test * O_patch
                    Psi_hat = fwd_fn(psi)
                    mag = np.abs(Psi_hat)
                    diff = mag - amplitudes[j]
                    total_error += np.sum(diff**2)
                    total_norm += np.sum(amplitudes[j]**2)

                if total_norm > 0:
                    norm_error = total_error / total_norm
                else:
                    norm_error = total_error

                if norm_error < best_error:
                    best_error = norm_error
                    best_convention = (fwd_fn, bwd_fn)

            self.fft_forward, self.fft_backward = best_convention
            return best_convention

    def _initialize_probe(
            self,
            amplitudes: np.ndarray,
        ) -> np.ndarray:
            """
            Initialize the probe as a power-calibrated Gaussian.

            Args:
                amplitudes: Square root of measured intensities, shape (n_positions, probe_size, probe_size).

            Returns:
                Initial probe estimate, shape (probe_size, probe_size), complex128.
            """
            ps = self.probe_size
            half = ps // 2

            # Create coordinate grids centered at (0,0)
            Y, X = np.mgrid[-half:half, -half:half]
            R2 = X.astype(np.float64)**2 + Y.astype(np.float64)**2
            sigma = self.probe_sigma

            # Gaussian probe
            P = np.exp(-R2 / (2.0 * sigma * sigma)).astype(np.complex128)

            # Calibrate probe power using Parseval's theorem
            # sum of |FFT(P*O)|^2 = N^2 * sum(|P*O|^2)
            # With O ~ 1, sum(|P|^2) ~ mean_total_intensity / N^2
            mean_total_intensity = np.mean(np.sum(amplitudes**2, axis=(1, 2)))
            probe_power = mean_total_intensity / float(ps * ps)
            current_power = np.sum(np.abs(P)**2)
            if current_power > 0:
                P *= np.sqrt(probe_power / current_power)

            return P

    def _initialize_object(self) -> np.ndarray:
            """
            Initialize the object as uniform magnitude with zero phase.

            Returns:
                Initial object estimate, shape (obj_size, obj_size), complex128.
            """
            return np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

    def _modulus_replacement(
            self,
            psi_hat: np.ndarray,
            measured_amplitude: np.ndarray,
        ) -> np.ndarray:
            """
            Apply the Fourier modulus constraint: replace magnitude while preserving phase.

            Args:
                psi_hat: Fourier transform of exit wave, shape (probe_size, probe_size), complex.
                measured_amplitude: Measured amplitude (sqrt of intensity), shape (probe_size, probe_size), float.

            Returns:
                Corrected Fourier-domain exit wave, shape (probe_size, probe_size), complex.
            """
            mag = np.abs(psi_hat)
            psi_hat_corrected = measured_amplitude * psi_hat / (mag + self.modulus_eps)
            return psi_hat_corrected

    def _update_object_rpie(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            delta_psi: np.ndarray,
            ry: int,
            rx: int,
            alpha: float,
        ) -> np.ndarray:
            """
            Apply the rPIE object update for a single scan position.

            Args:
                obj: Current object estimate, shape (obj_size, obj_size), complex128. Modified in-place.
                probe: Current probe estimate, shape (probe_size, probe_size), complex128.
                delta_psi: Exit wave difference (corrected - current), shape (probe_size, probe_size), complex128.
                ry: Row offset for this scan position.
                rx: Column offset for this scan position.
                alpha: rPIE regularization parameter for object.

            Returns:
                Updated object array (same reference, modified in-place).
            """
            ps = self.probe_size
            P_abs_sq = np.abs(probe) ** 2
            P_max_sq = np.max(P_abs_sq)
            denom_O = (1.0 - alpha) * P_abs_sq + alpha * P_max_sq
            obj[ry:ry + ps, rx:rx + ps] += np.conj(probe) / denom_O * delta_psi
            return obj

    def _update_probe_rpie(
            self,
            probe: np.ndarray,
            obj_patch: np.ndarray,
            delta_psi: np.ndarray,
            alpha: float,
        ) -> np.ndarray:
            """
            Apply the rPIE probe update for a single scan position.

            Args:
                probe: Current probe estimate, shape (probe_size, probe_size), complex128. Modified in-place.
                obj_patch: Object patch at current position, shape (probe_size, probe_size), complex128.
                delta_psi: Exit wave difference, shape (probe_size, probe_size), complex128.
                alpha: rPIE regularization parameter for probe.

            Returns:
                Updated probe array (same reference, modified in-place).
            """
            O_abs_sq = np.abs(obj_patch) ** 2
            O_max_sq = np.max(O_abs_sq)
            denom_P = (1.0 - alpha) * O_abs_sq + alpha * O_max_sq
            probe += np.conj(obj_patch) / denom_P * delta_psi
            return probe

    def _compute_fourier_error(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            amplitudes: np.ndarray,
        ) -> float:
            """
            Compute the mean normalized Fourier error across all scan positions.

            Args:
                obj: Current object estimate, shape (obj_size, obj_size), complex128.
                probe: Current probe estimate, shape (probe_size, probe_size), complex128.
                amplitudes: Measured amplitudes, shape (n_positions, probe_size, probe_size), float.

            Returns:
                Mean normalized Fourier error (scalar).
            """
            ps = self.probe_size
            n_positions = amplitudes.shape[0]
            total_error = 0.0
            total_norm = 0.0

            for j in range(n_positions):
                ry, rx = self.positions[j]
                O_patch = obj[ry:ry + ps, rx:rx + ps]
                psi = probe * O_patch
                Psi_hat = self.fft_forward(psi)
                mag = np.abs(Psi_hat)
                diff = mag - amplitudes[j]
                total_error += np.sum(diff ** 2)
                total_norm += np.sum(amplitudes[j] ** 2)

            if total_norm > 0:
                return float(total_error / total_norm)
            else:
                return float(total_error)

    def _get_current_alpha(
            self,
            epoch: int,
            is_object: bool,
        ) -> float:
            """
            Get the current regularization parameter based on the epoch and schedule.

            Args:
                epoch: Current epoch number.
                is_object: If True, return object alpha; otherwise probe alpha.

            Returns:
                Current alpha value.
            """
            if epoch >= self.stage3_epoch:
                return self.alpha_O_stage3 if is_object else self.alpha_P_stage3
            elif epoch >= self.stage2_epoch:
                return self.alpha_O_stage2 if is_object else self.alpha_P_stage2
            else:
                return self.alpha_O if is_object else self.alpha_P

    def _remove_global_phase_ambiguity(
            self,
            obj: np.ndarray,
        ) -> np.ndarray:
            """
            Remove the global phase ambiguity by referencing the phase of
            high-magnitude regions.

            Args:
                obj: Reconstructed object, shape (obj_size, obj_size), complex128.

            Returns:
                Phase-corrected object, shape (obj_size, obj_size), complex128.
            """
            mag = np.abs(obj)
            max_mag = np.max(mag)
            if max_mag == 0:
                return obj

            # Find high-magnitude mask (top 50% of maximum magnitude)
            high_mag_mask = mag > 0.5 * max_mag
            if np.any(high_mag_mask):
                # Compute the mean complex value in high-magnitude regions
                # and extract its phase as the reference
                mean_complex = np.mean(obj[high_mag_mask])
                ref_phase = np.angle(mean_complex)
                # Rotate the entire object to remove this global phase offset
                obj = obj * np.exp(-1j * ref_phase)

            return obj

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Compute the forward model: far-field diffraction intensity for one scan position.

            I_j(k) = |F{P(r - R_j) * O(r)}|^2

            Args:
                obj: Complex object function, shape (obj_size, obj_size), complex128.
                probe: Complex probe function, shape (probe_size, probe_size), complex128.
                position: (row, col) scan position offset.

            Returns:
                Simulated diffraction intensity, shape (probe_size, probe_size), float64.
            """
            ry, rx = position
            ps = self.probe_size

            # Extract the object patch at the scan position
            O_patch = obj[ry:ry + ps, rx:rx + ps]

            # Form the exit wave: probe * object patch
            psi = probe * O_patch

            # Forward propagate to far-field using the selected FFT convention
            if self.fft_forward is not None:
                Psi_hat = self.fft_forward(psi)
            else:
                # Fallback: use bare fft2 if convention not yet selected
                Psi_hat = np.fft.fft2(psi)

            # Compute intensity (modulus squared)
            intensity = np.abs(Psi_hat) ** 2

            return intensity.astype(np.float64)

    def solve(
            self,
            diffraction_intensities: np.ndarray,
        ) -> np.ndarray:
            """
            Run the full rPIE phase retrieval pipeline.
            """
            if self.seed is not None:
                np.random.seed(self.seed)

            # Step 1: Compute amplitudes from intensities
            amplitudes = np.sqrt(np.maximum(diffraction_intensities, 0.0))

            # Step 2: Compute scan positions
            self._compute_scan_positions()

            # Step 3: Define and select best FFT convention
            conventions = self._define_fft_conventions()
            self._select_fft_convention(amplitudes, conventions)

            # Step 4: Initialize object and probe
            O = self._initialize_object()
            P = self._initialize_probe(amplitudes)

            ps = self.probe_size
            n_positions = amplitudes.shape[0]

            # Step 5: Run rPIE iterations
            for epoch in range(self.n_iter):
                # Shuffle scan order each epoch
                order = np.random.permutation(n_positions)

                # Get current alpha values based on schedule
                alpha_o = self._get_current_alpha(epoch, True)
                alpha_p = self._get_current_alpha(epoch, False)

                for j in order:
                    ry, rx = self.positions[j]

                    # Extract object patch
                    O_patch = O[ry:ry + ps, rx:rx + ps].copy()

                    # Compute exit wave
                    psi = P * O_patch

                    # Forward propagate
                    Psi_hat = self.fft_forward(psi)

                    # Modulus replacement
                    Psi_hat_c = self._modulus_replacement(Psi_hat, amplitudes[j])

                    # Back propagate
                    psi_c = self.fft_backward(Psi_hat_c)

                    # Compute delta
                    dpsi = psi_c - psi

                    # Object update (rPIE)
                    self._update_object_rpie(O, P, dpsi, ry, rx, alpha_o)

                    # Probe update (rPIE, delayed start)
                    if epoch >= self.probe_update_start:
                        self._update_probe_rpie(P, O_patch, dpsi, alpha_p)

            # Step 6: Post-process - remove global phase ambiguity
            O = self._remove_global_phase_ambiguity(O)

            # Step 7: Form output
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float64)
            output[0] = np.abs(O)
            output[1] = np.angle(O)

            return output



if __name__ == "__main__":
    # Load data
    input_data = np.load("dataset/input.npy")  # (81, 256, 256)

    # Create solver with tuned parameters
    solver = InverseSolver(
        obj_size=600,
        probe_size=256,
        n_scan=9,
        n_iter=800,
        alpha_O=0.1,
        alpha_P=0.1,
        alpha_O_stage2=0.05,
        alpha_P_stage2=0.05,
        alpha_O_stage3=0.02,
        alpha_P_stage3=0.02,
        stage2_epoch=200,
        stage3_epoch=500,
        probe_update_start=50,
        probe_sigma=60.0,
        modulus_eps=1e-8,
        n_conventions_to_test=4,
        convention_test_iters=10,
        monitor_interval=50,
        seed=42,
    )

    # Run phase retrieval
    result = solver.solve(input_data)  # (2, 600, 600)

    # Save output
    np.save("output.npy", result)