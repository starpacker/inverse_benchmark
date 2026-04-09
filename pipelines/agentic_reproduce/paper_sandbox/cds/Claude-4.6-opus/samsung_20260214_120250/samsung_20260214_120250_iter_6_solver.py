import numpy as np
from typing import Tuple, List, Optional

class InverseSolver:
    """
    Ptychographic phase retrieval solver implementing rpPIE (reflectivity-prior PIE)
    for periodic copper pad arrays in EUV coherent diffractive imaging.
    
    Recovers complex-valued reflection function O(r) from far-field diffraction
    intensity measurements I(k) using ePIE with reflectivity prior enforcement.
    """

    def __init__(
            self,
            wavelength: float = 13.5,
            pixel_size: float = 27.0,
            probe_size: int = 256,
            obj_size: int = 600,
            n_scan_side: int = 9,
            scan_step: int = 40,
            n_pads: int = 15,
            pad_size_pixels: int = 22,
            pitch_pixels: int = 30,
            n_epochs: int = 800,
            alpha_O: float = 1.0,
            alpha_P: float = 1.0,
            beta_prior: float = 0.7,
            prior_start_epoch: int = 3,
            prior_ramp_epochs: int = 20,
            probe_update_start_epoch: int = 3,
            eps: float = 1e-8,
            n_SiCN_real: float = 0.9791,
            n_SiCN_imag: float = 0.0059,
            n_SiO2_real: float = 0.9780,
            n_SiO2_imag: float = 0.0108,
            n_Si_real: float = 0.9991,
            n_Si_imag: float = 0.0018,
            n_Cu_real: float = 0.9625,
            n_Cu_imag: float = 0.0613,
            amplitude_clamp_max: float = 1.0,
            convergence_check_interval: int = 50,
            fft_convention_flip_epoch: int = 100,
            fft_convention_flip_threshold: float = 0.5,
        ) -> None:
            # Store all hyperparameters as instance attributes
            self.wavelength = wavelength
            self.pixel_size = pixel_size
            self.probe_size = probe_size
            self.obj_size = obj_size
            self.n_scan_side = n_scan_side
            self.scan_step = scan_step
            self.n_pads = n_pads
            self.pad_size_pixels = pad_size_pixels
            self.pitch_pixels = pitch_pixels
            self.n_epochs = n_epochs
            self.alpha_O = alpha_O
            self.alpha_P = alpha_P
            self.beta_prior = beta_prior
            self.prior_start_epoch = prior_start_epoch
            self.prior_ramp_epochs = prior_ramp_epochs
            self.probe_update_start_epoch = probe_update_start_epoch
            self.eps = eps
            self.n_SiCN = complex(n_SiCN_real, n_SiCN_imag)
            self.n_SiO2 = complex(n_SiO2_real, n_SiO2_imag)
            self.n_Si = complex(n_Si_real, n_Si_imag)
            self.n_Cu = complex(n_Cu_real, n_Cu_imag)
            self.amplitude_clamp_max = amplitude_clamp_max
            self.convergence_check_interval = convergence_check_interval
            self.fft_convention_flip_epoch = fft_convention_flip_epoch
            self.fft_convention_flip_threshold = fft_convention_flip_threshold

            # Derived attributes initialized later during solve()
            self.positions: List[Tuple[int, int]] = []
            self.support_mask: Optional[np.ndarray] = None
            self.R_substrate: complex = 0 + 0j
            self.R_copper: complex = 0 + 0j
            self.O: Optional[np.ndarray] = None
            self.P: Optional[np.ndarray] = None
            self.use_fftshift: bool = True

    def _compute_scan_positions(self) -> List[Tuple[int, int]]:
            """
            Compute the (row, col) top-left corner positions for each scan point
            on a regular grid with the configured step size, centered in the object array.

            Returns:
                List of (row, col) tuples for each of the n_scan_side^2 positions.
            """
            offset = (self.obj_size - (self.probe_size + (self.n_scan_side - 1) * self.scan_step)) // 2
            positions = []
            for i in range(self.n_scan_side):
                for j in range(self.n_scan_side):
                    positions.append((offset + i * self.scan_step, offset + j * self.scan_step))
            return positions

    def _build_support_mask(self) -> np.ndarray:
            """
            Build the binary support mask S defining the periodic pad array geometry.
            Each pad is pad_size_pixels x pad_size_pixels, arranged on a
            n_pads x n_pads grid with pitch_pixels spacing, centered in the object array.

            Returns:
                Boolean array of shape (obj_size, obj_size).
            """
            array_extent = (self.n_pads - 1) * self.pitch_pixels + self.pad_size_pixels
            array_offset = (self.obj_size - array_extent) // 2

            S = np.zeros((self.obj_size, self.obj_size), dtype=bool)
            for m in range(self.n_pads):
                for n in range(self.n_pads):
                    y0 = array_offset + m * self.pitch_pixels
                    x0 = array_offset + n * self.pitch_pixels
                    S[y0:y0 + self.pad_size_pixels, x0:x0 + self.pad_size_pixels] = True
            return S

    def _compute_reflectivities(self) -> Tuple[complex, complex]:
            """
            Compute complex Fresnel reflectivities at normal incidence for the
            substrate (vacuum/SiCN interface) and copper pads (vacuum/Cu interface)
            using the stored optical constants.

            Returns:
                Tuple of (R_substrate, R_copper) complex reflectivities.
            """
            n_vacuum = complex(1.0, 0.0)
            R_substrate = (n_vacuum - self.n_SiCN) / (n_vacuum + self.n_SiCN)
            R_copper = (n_vacuum - self.n_Cu) / (n_vacuum + self.n_Cu)
            return R_substrate, R_copper

    def _initialize_object(self) -> np.ndarray:
            """
            Initialize the complex object array O(r) uniformly.

            Returns:
                Complex array of shape (obj_size, obj_size).
            """
            return np.ones((self.obj_size, self.obj_size), dtype=np.complex128)

    def _initialize_probe(self, diff_intensities: np.ndarray) -> np.ndarray:
            """
            Initialize the probe function from the mean diffraction pattern
            by inverse Fourier transforming the mean amplitude with zero phase.

            Args:
                diff_intensities: Measured diffraction intensities, shape (n_positions, probe_size, probe_size).

            Returns:
                Complex array of shape (probe_size, probe_size).
            """
            mean_I = np.mean(diff_intensities, axis=0)  # (probe_size, probe_size)
            mean_amp = np.sqrt(np.maximum(mean_I, 0.0))
            # Assume input diffraction patterns have DC at center; use ifftshift to move
            # DC to corner before ifft2, then fftshift to center the probe in real space
            P = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mean_amp)))
            P = P.astype(np.complex128)
            return P

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Compute the forward model: far-field diffraction amplitude from the
            exit wave at a given scan position.

            Computes: |FFT2(P * O_patch)| where O_patch is the object region
            at the given scan position.

            Args:
                O: Complex object array, shape (obj_size, obj_size).
                P: Complex probe array, shape (probe_size, probe_size).
                position: (row, col) top-left corner of the probe window in the object.

            Returns:
                Predicted diffraction amplitude, shape (probe_size, probe_size).
            """
            ry, rx = position
            O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
            psi = P * O_patch
            Psi = np.fft.fft2(psi)
            return np.abs(Psi)

    def _fourier_magnitude_replacement(
            self,
            psi: np.ndarray,
            measured_amplitude: np.ndarray,
        ) -> np.ndarray:
            """
            Apply Fourier magnitude constraint: replace the amplitude of the
            far-field exit wave with the measured amplitude, preserving phase.

            Args:
                psi: Exit wave in real space, shape (probe_size, probe_size).
                measured_amplitude: Measured sqrt(I) in appropriate FFT order,
                                    shape (probe_size, probe_size).

            Returns:
                Corrected exit wave in real space after magnitude replacement,
                shape (probe_size, probe_size).
            """
            Psi = np.fft.fft2(psi)
            Psi_amp = np.abs(Psi)
            Psi_corrected = measured_amplitude * Psi / (Psi_amp + self.eps)
            psi_corrected = np.fft.ifft2(Psi_corrected)
            return psi_corrected

    def _update_object(
            self,
            O: np.ndarray,
            P: np.ndarray,
            position: Tuple[int, int],
            dpsi: np.ndarray,
        ) -> np.ndarray:
            """
            Apply the ePIE object update rule with scalar max|P|^2 denominator.

            O_patch += alpha_O * conj(P) / max(|P|^2) * dpsi

            Args:
                O: Complex object array, shape (obj_size, obj_size). Modified in-place.
                P: Complex probe array, shape (probe_size, probe_size).
                position: (row, col) top-left corner of the probe window.
                dpsi: Exit wave difference (psi_corrected - psi), shape (probe_size, probe_size).

            Returns:
                Updated object array (same reference, modified in-place).
            """
            ry, rx = position
            P_abs_sq = np.abs(P) ** 2
            P_max_sq = np.max(P_abs_sq)
            O[ry:ry + self.probe_size, rx:rx + self.probe_size] += (
                self.alpha_O * np.conj(P) * dpsi / (P_max_sq + self.eps)
            )
            return O

    def _update_probe(
            self,
            O: np.ndarray,
            P: np.ndarray,
            position: Tuple[int, int],
            dpsi: np.ndarray,
        ) -> np.ndarray:
            """
            Apply the ePIE probe update rule with scalar max|O_patch|^2 denominator.

            P += alpha_P * conj(O_patch) / max(|O_patch|^2) * dpsi

            Args:
                O: Complex object array, shape (obj_size, obj_size).
                P: Complex probe array, shape (probe_size, probe_size). Modified in-place.
                position: (row, col) top-left corner of the probe window.
                dpsi: Exit wave difference, shape (probe_size, probe_size).

            Returns:
                Updated probe array (same reference, modified in-place).
            """
            ry, rx = position
            O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
            O_abs_sq = np.abs(O_patch) ** 2
            O_max_sq = np.max(O_abs_sq)
            P += self.alpha_P * np.conj(O_patch) * dpsi / (O_max_sq + self.eps)
            return P

    def _apply_amplitude_constraint(self, O: np.ndarray) -> np.ndarray:
            """
            Clamp object amplitude to [0, amplitude_clamp_max] while preserving phase.
            Enforces physical constraint that EUV reflectivity cannot exceed 1.

            Args:
                O: Complex object array, shape (obj_size, obj_size).

            Returns:
                Amplitude-clamped object array.
            """
            amp = np.abs(O)
            phase = np.angle(O)
            amp = np.clip(amp, 0, self.amplitude_clamp_max)
            O = amp * np.exp(1j * phase)
            return O

    def _apply_reflectivity_prior(
            self,
            O: np.ndarray,
            epoch: int,
        ) -> np.ndarray:
            """
            Enforce the rpPIE reflectivity prior outside the support mask.
            Outside support S, blend object toward known substrate reflectivity R_p:
                O[~S] = (1 - blend) * O[~S] + blend * R_p
            where blend ramps from 0 to (1 - beta_prior) over prior_ramp_epochs
            starting at prior_start_epoch.

            Args:
                O: Complex object array, shape (obj_size, obj_size). Modified in-place.
                epoch: Current epoch index.

            Returns:
                Object array with prior enforced outside support.
            """
            if epoch >= self.prior_start_epoch and self.support_mask is not None:
                gamma = min(1.0, (epoch - self.prior_start_epoch) / self.prior_ramp_epochs)
                blend = gamma * (1.0 - self.beta_prior)
                outside_mask = ~self.support_mask
                O[outside_mask] = (1.0 - blend) * O[outside_mask] + blend * self.R_substrate
            return O

    def _compute_convergence_error(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amplitudes: np.ndarray,
        ) -> float:
            """
            Compute the normalized Fourier-domain error metric averaged over all positions.

            E = (1/J) * sum_j || |F{P * O_j}| - sqrt(I_j) ||^2 / || sqrt(I_j) ||^2

            Args:
                O: Current object estimate, shape (obj_size, obj_size).
                P: Current probe estimate, shape (probe_size, probe_size).
                diff_amplitudes: Measured diffraction amplitudes, shape (n_positions, probe_size, probe_size).

            Returns:
                Scalar normalized error.
            """
            total_err = 0.0
            n_positions = len(self.positions)
            for idx in range(n_positions):
                ry, rx = self.positions[idx]
                O_patch = O[ry:ry + self.probe_size, rx:rx + self.probe_size]
                psi = P * O_patch
                Psi = np.fft.fft2(psi)
                pred_amp = np.abs(Psi)

                if self.use_fftshift:
                    meas_amp = np.fft.ifftshift(diff_amplitudes[idx])
                else:
                    meas_amp = diff_amplitudes[idx]

                numerator = np.sum((pred_amp - meas_amp) ** 2)
                denominator = np.sum(meas_amp ** 2) + self.eps
                total_err += numerator / denominator

            return total_err / n_positions

    def _remove_global_phase_ambiguity(self, O: np.ndarray) -> np.ndarray:
            """
            Remove the global phase ambiguity by setting the mean substrate
            (outside support) phase to zero.

            Args:
                O: Complex object array, shape (obj_size, obj_size).

            Returns:
                Phase-referenced object array.
            """
            if self.support_mask is not None:
                outside_mask = ~self.support_mask
                substrate_values = O[outside_mask]
                # Use mean of substrate region to determine global phase offset
                substrate_mean = np.mean(substrate_values)
                if np.abs(substrate_mean) > self.eps:
                    phase_ref = np.angle(substrate_mean)
                    O = O * np.exp(-1j * phase_ref)
            return O

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """
            Format the reconstructed object into the required output format:
            channel 0 = amplitude |O|, channel 1 = phase angle(O).

            Args:
                O: Complex object array, shape (obj_size, obj_size).

            Returns:
                Float32 array of shape (2, obj_size, obj_size).
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)
            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Main reconstruction method implementing the full rpPIE pipeline.
            """
            # Step 1: Preprocess input
            I = np.maximum(input_data, 0.0)
            diff_amp = np.sqrt(I)  # (81, 256, 256)

            # Step 2: Setup geometry
            self.positions = self._compute_scan_positions()
            self.support_mask = self._build_support_mask()
            self.R_substrate, self.R_copper = self._compute_reflectivities()

            # Step 3: Initialize object and probe
            self.O = self._initialize_object()
            self.P = self._initialize_probe(I)

            # Normalize probe to have reasonable energy to avoid overflow
            probe_energy = np.sum(np.abs(self.P) ** 2)
            if probe_energy > 0:
                # Scale probe so max amplitude is ~1
                p_max = np.max(np.abs(self.P))
                if p_max > 0:
                    self.P = self.P / p_max

            ps = self.probe_size
            n_positions = len(self.positions)
            self.use_fftshift = True  # assume DC at center in input

            # Pre-compute measured amplitudes in FFT order for both conventions
            diff_amp_shifted = np.zeros_like(diff_amp)
            for idx in range(n_positions):
                diff_amp_shifted[idx] = np.fft.ifftshift(diff_amp[idx])

            best_error = np.inf
            best_O = self.O.copy()
            best_P = self.P.copy()

            for epoch in range(self.n_epochs):
                order = np.random.permutation(n_positions)

                # Adaptive step sizes: reduce over time for stability
                if epoch < 100:
                    cur_alpha_O = self.alpha_O
                    cur_alpha_P = self.alpha_P
                elif epoch < 400:
                    cur_alpha_O = self.alpha_O * 0.8
                    cur_alpha_P = self.alpha_P * 0.8
                else:
                    cur_alpha_O = self.alpha_O * 0.5
                    cur_alpha_P = self.alpha_P * 0.5

                for idx in order:
                    ry, rx = self.positions[idx]
                    O_patch = self.O[ry:ry + ps, rx:rx + ps].copy()

                    # Exit wave
                    psi = self.P * O_patch

                    # Forward propagate
                    Psi = np.fft.fft2(psi)
                    Psi_amp = np.abs(Psi)

                    # Measured amplitude in FFT order
                    if self.use_fftshift:
                        meas = diff_amp_shifted[idx]
                    else:
                        meas = diff_amp[idx]

                    # Fourier magnitude replacement
                    Psi_corrected = meas * Psi / (Psi_amp + self.eps)

                    # Back-propagate
                    psi_corrected = np.fft.ifft2(Psi_corrected)

                    # Exit wave difference
                    dpsi = psi_corrected - psi

                    # Object update (standard ePIE: scalar max denominator)
                    P_abs_sq = np.abs(self.P) ** 2
                    P_max_sq = np.max(P_abs_sq)
                    if P_max_sq > self.eps:
                        self.O[ry:ry + ps, rx:rx + ps] += (
                            cur_alpha_O * np.conj(self.P) * dpsi / (P_max_sq + self.eps)
                        )

                    # Probe update (after probe_start_epoch)
                    if epoch >= self.probe_update_start_epoch:
                        O_patch_new = self.O[ry:ry + ps, rx:rx + ps]
                        O_abs_sq = np.abs(O_patch_new) ** 2
                        O_max_sq = np.max(O_abs_sq)
                        if O_max_sq > self.eps:
                            self.P += (
                                cur_alpha_P * np.conj(O_patch_new) * dpsi / (O_max_sq + self.eps)
                            )

                # --- End of epoch constraints ---

                # Check for NaN/Inf and recover
                if not np.all(np.isfinite(self.O)) or not np.all(np.isfinite(self.P)):
                    self.O = best_O.copy()
                    self.P = best_P.copy()
                    continue

                # Amplitude clamping
                self.O = self._apply_amplitude_constraint(self.O)

                # Reflectivity prior outside support
                self.O = self._apply_reflectivity_prior(self.O, epoch)

                # Probe energy stabilization: prevent probe from growing unbounded
                probe_max = np.max(np.abs(self.P))
                if probe_max > 1e6:
                    self.P = self.P / probe_max

                # Convergence monitoring
                if epoch % self.convergence_check_interval == 0:
                    err = self._compute_convergence_error(self.O, self.P, diff_amp)

                    if err < best_error:
                        best_error = err
                        best_O = self.O.copy()
                        best_P = self.P.copy()

                    # FFT convention auto-detection
                    if epoch == self.fft_convention_flip_epoch and err > self.fft_convention_flip_threshold:
                        self.use_fftshift = not self.use_fftshift
                        # Reset to initial state
                        self.O = self._initialize_object()
                        p_max_init = np.max(np.abs(self._initialize_probe(I)))
                        self.P = self._initialize_probe(I)
                        if p_max_init > 0:
                            self.P = self.P / p_max_init
                        best_error = np.inf

            # Use best reconstruction
            self.O = best_O

            # Remove global phase ambiguity
            self.O = self._remove_global_phase_ambiguity(self.O)

            # Format output
            return self._format_output(self.O)



if __name__ == "__main__":
    # Load measured diffraction intensity data
    input_data = np.load('dataset/input.npy')

    # Instantiate the solver with carefully tuned hyperparameters
    solver = InverseSolver(
        n_epochs=800,
        alpha_O=1.0,
        alpha_P=1.0,
        beta_prior=0.7,
        prior_start_epoch=3,
        prior_ramp_epochs=20,
        probe_update_start_epoch=3,
        amplitude_clamp_max=1.0,
        convergence_check_interval=50,
        fft_convention_flip_epoch=100,
        fft_convention_flip_threshold=0.5,
    )

    # Override solve to fix numerical stability issues from previous iterations
    # The main problems were:
    # 1. Probe normalization causing scale issues
    # 2. Division by near-zero in updates causing NaN/overflow
    # 3. Need more careful probe initialization scaling

    I = np.maximum(input_data, 0.0)
    diff_amp = np.sqrt(I)  # (81, 256, 256)

    # Setup geometry
    solver.positions = solver._compute_scan_positions()
    solver.support_mask = solver._build_support_mask()
    solver.R_substrate, solver.R_copper = solver._compute_reflectivities()

    ps = solver.probe_size
    obj_size = solver.obj_size
    n_positions = len(solver.positions)
    eps = solver.eps

    # Initialize object
    O = np.ones((obj_size, obj_size), dtype=np.complex128)

    # Initialize probe from mean diffraction pattern
    mean_I = np.mean(I, axis=0)
    mean_amp = np.sqrt(np.maximum(mean_I, 0.0))
    # DC at center assumption: ifftshift moves DC to corner, ifft2, then fftshift centers probe
    P = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mean_amp)))
    P = P.astype(np.complex128)

    # Scale probe so that max|P| = 1 to keep numerics stable
    p_max = np.max(np.abs(P))
    if p_max > 0:
        P = P / p_max

    use_fftshift = True

    # Pre-compute shifted amplitudes
    diff_amp_shifted = np.zeros_like(diff_amp)
    for idx in range(n_positions):
        diff_amp_shifted[idx] = np.fft.ifftshift(diff_amp[idx])

    best_error = np.inf
    best_O = O.copy()
    best_P = P.copy()
    best_epoch = 0

    n_epochs = solver.n_epochs
    alpha_O = solver.alpha_O
    alpha_P = solver.alpha_P
    beta_prior = solver.beta_prior
    prior_start = solver.prior_start_epoch
    prior_ramp = solver.prior_ramp_epochs
    probe_start = solver.probe_update_start_epoch
    S = solver.support_mask
    R_sub = solver.R_substrate
    check_interval = solver.convergence_check_interval

    for epoch in range(n_epochs):
        order = np.random.permutation(n_positions)

        # Gentle step size decay
        if epoch < 100:
            cur_aO = alpha_O
            cur_aP = alpha_P
        elif epoch < 400:
            cur_aO = alpha_O * 0.8
            cur_aP = alpha_P * 0.8
        else:
            cur_aO = alpha_O * 0.5
            cur_aP = alpha_P * 0.5

        for idx in order:
            ry, rx = solver.positions[idx]
            O_patch = O[ry:ry + ps, rx:rx + ps].copy()

            # Exit wave
            psi = P * O_patch

            # Forward propagate
            Psi = np.fft.fft2(psi)
            Psi_amp = np.abs(Psi)

            # Measured amplitude in FFT order
            if use_fftshift:
                meas = diff_amp_shifted[idx]
            else:
                meas = diff_amp[idx]

            # Fourier magnitude replacement
            denom_fft = Psi_amp + eps
            Psi_corrected = meas * Psi / denom_fft

            # Back-propagate
            psi_corrected = np.fft.ifft2(Psi_corrected)

            # Exit wave difference
            dpsi = psi_corrected - psi

            # Object update (standard ePIE with scalar max denominator)
            P_abs_sq = np.abs(P) ** 2
            P_max_sq = np.max(P_abs_sq)
            if P_max_sq > eps:
                update_O = cur_aO * np.conj(P) * dpsi / (P_max_sq + eps)
                # Clip update to prevent explosions
                update_mag = np.abs(update_O)
                max_update = np.max(update_mag)
                if max_update > 0.5:
                    update_O = update_O * (0.5 / max_update)
                O[ry:ry + ps, rx:rx + ps] += update_O

            # Probe update
            if epoch >= probe_start:
                O_patch_new = O[ry:ry + ps, rx:rx + ps]
                O_abs_sq = np.abs(O_patch_new) ** 2
                O_max_sq = np.max(O_abs_sq)
                if O_max_sq > eps:
                    update_P = cur_aP * np.conj(O_patch_new) * dpsi / (O_max_sq + eps)
                    # Clip probe update
                    update_P_mag = np.abs(update_P)
                    max_up_P = np.max(update_P_mag)
                    if max_up_P > 0.5 * np.max(np.abs(P)):
                        update_P = update_P * (0.5 * np.max(np.abs(P)) / (max_up_P + eps))
                    P += update_P

        # --- End of epoch constraints ---

        # Check for NaN/Inf and recover from best
        if not np.all(np.isfinite(O)) or not np.all(np.isfinite(P)):
            O = best_O.copy()
            P = best_P.copy()
            continue

        # Amplitude clamping to [0, 1]
        amp = np.abs(O)
        phase = np.angle(O)
        amp = np.clip(amp, 0.0, 1.0)
        O = amp * np.exp(1j * phase)

        # Reflectivity prior outside support
        if epoch >= prior_start and S is not None:
            gamma = min(1.0, (epoch - prior_start) / max(prior_ramp, 1))
            blend = gamma * (1.0 - beta_prior)
            outside = ~S
            O[outside] = (1.0 - blend) * O[outside] + blend * R_sub

        # Probe energy stabilization
        probe_max = np.max(np.abs(P))
        if probe_max > 1e4:
            scale_factor = 1e4 / probe_max
            P *= scale_factor
            # Compensate object to maintain exit wave
            O /= scale_factor

        # Convergence monitoring
        if epoch % check_interval == 0:
            total_err = 0.0
            for cidx in range(n_positions):
                cry, crx = solver.positions[cidx]
                cpatch = O[cry:cry + ps, crx:crx + ps]
                cpsi = P * cpatch
                cPsi = np.fft.fft2(cpsi)
                cpred = np.abs(cPsi)
                if use_fftshift:
                    cmeas = diff_amp_shifted[cidx]
                else:
                    cmeas = diff_amp[cidx]
                num = np.sum((cpred - cmeas) ** 2)
                den = np.sum(cmeas ** 2) + eps
                total_err += num / den
            err = total_err / n_positions

            if err < best_error:
                best_error = err
                best_O = O.copy()
                best_P = P.copy()
                best_epoch = epoch

            # FFT convention auto-detection at epoch 100
            if epoch == solver.fft_convention_flip_epoch and err > solver.fft_convention_flip_threshold:
                use_fftshift = not use_fftshift
                # Reset
                O = np.ones((obj_size, obj_size), dtype=np.complex128)
                P = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mean_amp)))
                P = P.astype(np.complex128)
                pm = np.max(np.abs(P))
                if pm > 0:
                    P = P / pm
                # Recompute shifted amplitudes for new convention
                if not use_fftshift:
                    diff_amp_shifted = diff_amp.copy()
                else:
                    for sidx in range(n_positions):
                        diff_amp_shifted[sidx] = np.fft.ifftshift(diff_amp[sidx])
                best_error = np.inf
                best_O = O.copy()
                best_P = P.copy()

    # Use best reconstruction
    O = best_O

    # Remove global phase ambiguity
    if S is not None:
        outside = ~S
        substrate_vals = O[outside]
        substrate_mean = np.mean(substrate_vals)
        if np.abs(substrate_mean) > eps:
            phase_ref = np.angle(substrate_mean)
            O = O * np.exp(-1j * phase_ref)

    # Format output
    result = np.zeros((2, obj_size, obj_size), dtype=np.float32)
    result[0] = np.abs(O).astype(np.float32)
    result[1] = np.angle(O).astype(np.float32)

    # Save the output
    np.save('output.npy', result)