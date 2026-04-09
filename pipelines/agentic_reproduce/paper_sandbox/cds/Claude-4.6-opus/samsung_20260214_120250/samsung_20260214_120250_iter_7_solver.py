import numpy as np
from typing import Optional, Tuple, List

class InverseSolver:
    def __init__(
            self,
            wavelength: float = 13.5,
            pixel_size: float = 27.0,
            pad_size_px: int = 22,
            pitch_px: int = 30,
            n_pads: int = 15,
            probe_size: int = 256,
            obj_size: int = 600,
            n_scan_side: int = 9,
            scan_step: int = 40,
            n_epochs: int = 60,
            alpha_O: float = 1.0,
            alpha_P: float = 0.5,
            beta_prior: float = 0.7,
            prior_ramp_epochs: int = 5,
            batch_size: int = 9,
            eps: float = 1e-10,
            probe_update_start_epoch: int = 1,
            prior_start_epoch: int = 1,
            max_amplitude: float = 1.0,
            probe_energy_growth_factor: float = 5.0,
            n_SiCN: complex = complex(0.9791, 0.0059),
            n_SiO2: complex = complex(0.9780, 0.0108),
            n_Si: complex = complex(0.9991, 0.0018),
            n_Cu: complex = complex(0.9625, 0.0613),
        ) -> None:
            # Physical parameters
            self.wavelength: float = wavelength
            self.pixel_size: float = pixel_size
            self.pad_size_px: int = pad_size_px
            self.pitch_px: int = pitch_px
            self.n_pads: int = n_pads

            # Grid and scan geometry
            self.probe_size: int = probe_size
            self.obj_size: int = obj_size
            self.n_scan_side: int = n_scan_side
            self.scan_step: int = scan_step
            self.n_positions: int = n_scan_side * n_scan_side

            # Solver hyperparameters
            self.n_epochs: int = n_epochs
            self.alpha_O: float = alpha_O
            self.alpha_P: float = alpha_P
            self.beta_prior: float = beta_prior
            self.prior_ramp_epochs: int = prior_ramp_epochs
            self.batch_size: int = batch_size
            self.eps: float = eps
            self.probe_update_start_epoch: int = probe_update_start_epoch
            self.prior_start_epoch: int = prior_start_epoch
            self.max_amplitude: float = max_amplitude
            self.probe_energy_growth_factor: float = probe_energy_growth_factor

            # Material optical constants
            self.n_SiCN: complex = n_SiCN
            self.n_SiO2: complex = n_SiO2
            self.n_Si: complex = n_Si
            self.n_Cu: complex = n_Cu

            # Derived quantities (to be computed during solve)
            self.R_p: complex = complex(0.0, 0.0)
            self.positions: Optional[np.ndarray] = None
            self.support_mask: Optional[np.ndarray] = None
            self.use_shift: bool = False
            self.O: Optional[np.ndarray] = None
            self.P: Optional[np.ndarray] = None
            self.initial_probe_energy: float = 0.0

    def _compute_substrate_reflectivity(self) -> complex:
            """Compute Fresnel reflectivity for substrate (SiCN) at normal incidence."""
            R_p = (1 - self.n_SiCN) / (1 + self.n_SiCN)
            return R_p

    def _build_scan_positions(self) -> np.ndarray:
            """Build (n_positions, 2) array of (row, col) scan positions."""
            ps = self.probe_size
            offset = (self.obj_size - (ps + (self.n_scan_side - 1) * self.scan_step)) // 2
            positions = np.array(
                [(offset + i * self.scan_step, offset + j * self.scan_step)
                 for i in range(self.n_scan_side) for j in range(self.n_scan_side)],
                dtype=np.int64
            )
            return positions

    def _build_support_mask(self) -> np.ndarray:
            """Build binary support mask S for the 15x15 periodic pad array."""
            S = np.zeros((self.obj_size, self.obj_size), dtype=bool)
            arr_ext = (self.n_pads - 1) * self.pitch_px + self.pad_size_px  # 442
            arr_off = (self.obj_size - arr_ext) // 2  # 79

            for m in range(self.n_pads):
                for n in range(self.n_pads):
                    r_start = arr_off + m * self.pitch_px
                    c_start = arr_off + n * self.pitch_px
                    S[r_start:r_start + self.pad_size_px,
                      c_start:c_start + self.pad_size_px] = True
            return S

    def _detect_fft_convention(self, intensities: np.ndarray) -> bool:
            """Determine if DC component is at center (True) or corner (False)."""
            I0 = intensities[0]
            h, w = I0.shape
            ch, cw = h // 2, w // 2
            margin = h // 8  # 32 pixels

            center_energy = np.sum(I0[ch - margin:ch + margin, cw - margin:cw + margin])
            corner_energy = (np.sum(I0[:margin, :margin]) +
                             np.sum(I0[:margin, -margin:]) +
                             np.sum(I0[-margin:, :margin]) +
                             np.sum(I0[-margin:, -margin:]))

            return bool(center_energy > corner_energy)

    def _initialize_object(self) -> np.ndarray:
            """Initialize complex object O as uniform array of shape (obj_size, obj_size)."""
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            return O

    def _initialize_probe(
            self, diff_amp: np.ndarray, use_shift: bool
        ) -> np.ndarray:
            """Initialize probe P from mean diffraction amplitude via inverse FFT."""
            mean_amp = np.mean(diff_amp, axis=0)  # (256, 256) - average diffraction amplitude

            if use_shift:
                # DC is at center: ifftshift to move DC to corner, then ifft2, then fftshift
                P = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mean_amp)))
            else:
                # DC already at corner
                P = np.fft.ifft2(mean_amp)

            P = P.astype(np.complex128)
            return P

    def _extract_patches(
            self, O: np.ndarray, positions: np.ndarray
        ) -> np.ndarray:
            """Extract patches from O at given positions. Returns (n, ps, ps) complex array."""
            ps = self.probe_size
            n = len(positions)
            patches = np.empty((n, ps, ps), dtype=O.dtype)
            for k in range(n):
                ry, rx = positions[k]
                patches[k] = O[ry:ry + ps, rx:rx + ps]
            return patches

    def _scatter_add_updates(
            self,
            O: np.ndarray,
            positions: np.ndarray,
            updates: np.ndarray,
        ) -> np.ndarray:
            """Accumulate updates back into O with overlap normalization."""
            ps = self.probe_size
            num = np.zeros_like(O)
            den = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
            for k in range(len(positions)):
                ry, rx = positions[k]
                num[ry:ry + ps, rx:rx + ps] += updates[k]
                den[ry:ry + ps, rx:rx + ps] += 1.0
            den = np.maximum(den, 1.0)
            O = O + num / den
            return O

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            positions: np.ndarray,
        ) -> np.ndarray:
            """
            Compute forward model: far-field diffraction intensities.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Complex reflection function of the sample.
            P : np.ndarray, shape (probe_size, probe_size), complex128
                Probe/illumination function.
            positions : np.ndarray, shape (n_positions, 2), int
                Scan positions (row, col).

            Returns
            -------
            intensities : np.ndarray, shape (n_positions, probe_size, probe_size), float64
                Simulated diffraction intensities |F{P * O_patch}|^2.
            """
            ps = self.probe_size
            n = len(positions)
            intensities = np.empty((n, ps, ps), dtype=np.float64)

            # Extract all patches at once
            patches = self._extract_patches(O, positions)  # (n, ps, ps) complex128

            # Compute exit waves: element-wise multiply probe with each patch
            exit_waves = P[np.newaxis, :, :] * patches  # (n, ps, ps) complex128

            # Batch FFT over all positions
            exit_waves_freq = np.fft.fft2(exit_waves)  # (n, ps, ps) complex128

            # Compute intensities as squared magnitude
            intensities = np.abs(exit_waves_freq) ** 2  # (n, ps, ps) float64

            return intensities

    def _fourier_magnitude_replacement(
            self,
            Psi_freq: np.ndarray,
            measured_amp: np.ndarray,
        ) -> np.ndarray:
            """
            Replace Fourier magnitudes with measured amplitudes, keeping phase.

            Parameters
            ----------
            Psi_freq : np.ndarray, shape (batch, ps, ps), complex128
                FFT of exit waves.
            measured_amp : np.ndarray, shape (batch, ps, ps), float64
                Measured diffraction amplitudes sqrt(I).

            Returns
            -------
            Psi_corrected : np.ndarray, shape (batch, ps, ps), complex128
                Corrected Fourier-domain exit waves.
            """
            Psi_amp = np.abs(Psi_freq)
            Psi_corrected = measured_amp * Psi_freq / (Psi_amp + self.eps)
            return Psi_corrected

    def _update_object_batch(
            self,
            O: np.ndarray,
            P: np.ndarray,
            positions: np.ndarray,
            dpsi: np.ndarray,
        ) -> np.ndarray:
            """
            Apply ePIE-style object update from a batch of exit wave differences.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
            P : np.ndarray, shape (ps, ps), complex128
            positions : np.ndarray, shape (batch, 2), int
            dpsi : np.ndarray, shape (batch, ps, ps), complex128
                Exit wave differences (psi_corrected - psi).

            Returns
            -------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Updated object.
            """
            P_max_sq = np.max(np.abs(P) ** 2)
            obj_updates = self.alpha_O * np.conj(P)[np.newaxis, :, :] * dpsi / (P_max_sq + self.eps)
            O = self._scatter_add_updates(O, positions, obj_updates)
            return O

    def _update_probe(
            self,
            P: np.ndarray,
            patches: np.ndarray,
            dpsi: np.ndarray,
        ) -> np.ndarray:
            """
            Apply ePIE-style probe update averaged over batch.

            Parameters
            ----------
            P : np.ndarray, shape (ps, ps), complex128
            patches : np.ndarray, shape (batch, ps, ps), complex128
                Object patches at scan positions.
            dpsi : np.ndarray, shape (batch, ps, ps), complex128

            Returns
            -------
            P : np.ndarray, shape (ps, ps), complex128
                Updated probe.
            """
            O_abs_sq = np.abs(patches) ** 2
            O_max_sq = np.max(O_abs_sq)
            if O_max_sq < self.eps:
                return P
            probe_updates = np.conj(patches) * dpsi / (O_max_sq + self.eps)
            P = P + self.alpha_P * np.mean(probe_updates, axis=0)
            return P

    def _apply_amplitude_constraint(self, O: np.ndarray) -> np.ndarray:
            """Clip object amplitude to max_amplitude, preserving phase."""
            amp = np.abs(O)
            phase = np.angle(O)
            amp = np.minimum(amp, self.max_amplitude)
            O = amp * np.exp(1j * phase)
            return O

    def _apply_reflectivity_prior(
            self,
            O: np.ndarray,
            support: np.ndarray,
            R_p: complex,
            epoch: int,
        ) -> np.ndarray:
            """
            Enforce known substrate reflectivity outside support (rpPIE Eq. 6 Step 2).

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128
            support : np.ndarray, shape (obj_size, obj_size), bool
            R_p : complex
                Known substrate reflectivity.
            epoch : int
                Current epoch for ramping blend factor.

            Returns
            -------
            O : np.ndarray, shape (obj_size, obj_size), complex128
                Object with prior enforced outside support.
            """
            # gamma ramps from 0 to 1 over prior_ramp_epochs starting from prior_start_epoch
            gamma = min(1.0, (epoch - self.prior_start_epoch + 1) / self.prior_ramp_epochs)
            gamma = max(0.0, gamma)
            # blend = gamma * (1 - beta_prior)
            blend = gamma * (1.0 - self.beta_prior)
            # Soft blending: O[~S] = (1 - blend) * O[~S] + blend * R_p
            mask = ~support
            O[mask] = (1.0 - blend) * O[mask] + blend * R_p
            return O

    def _normalize_probe_energy(
            self, P: np.ndarray, initial_energy: float
        ) -> np.ndarray:
            """Renormalize probe if energy exceeds growth factor threshold."""
            current_energy = np.sum(np.abs(P) ** 2)
            max_energy = initial_energy * self.probe_energy_growth_factor ** 2
            if current_energy > max_energy and current_energy > 0:
                scale = np.sqrt(max_energy / current_energy)
                P = P * scale
            return P

    def _remove_global_phase_ambiguity(self, O: np.ndarray, support: np.ndarray) -> np.ndarray:
            """Remove global phase offset using substrate region as reference."""
            substrate_vals = O[~support]
            if len(substrate_vals) == 0:
                return O
            phase_ref = np.angle(np.mean(substrate_vals))
            O = O * np.exp(-1j * phase_ref)
            return O

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """
            Format reconstructed object into output array.

            Parameters
            ----------
            O : np.ndarray, shape (obj_size, obj_size), complex128

            Returns
            -------
            output : np.ndarray, shape (2, obj_size, obj_size), float32
                Channel 0: amplitude |O|, Channel 1: phase angle(O).
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)
            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Run rpPIE reconstruction from measured diffraction intensities.

            Parameters
            ----------
            input_data : np.ndarray, shape (81, 256, 256), float64
                Measured far-field diffraction intensities I(k).

            Returns
            -------
            result : np.ndarray, shape (2, 600, 600), float32
                Reconstructed object: [amplitude, phase].
            """
            # 1. Preprocess intensities → diff_amp
            I = np.maximum(input_data, 0.0).astype(np.float64)
            diff_amp = np.sqrt(I)  # (81, 256, 256)

            # 2. Compute substrate reflectivity
            self.R_p = self._compute_substrate_reflectivity()

            # 3. Build scan positions and support mask
            self.positions = self._build_scan_positions()
            self.support_mask = self._build_support_mask()

            # 4. Detect FFT convention
            self.use_shift = self._detect_fft_convention(I)

            # Pre-convert measured amplitudes to FFT-order once
            if self.use_shift:
                diff_amp_fft = np.fft.ifftshift(diff_amp, axes=(-2, -1))
            else:
                diff_amp_fft = diff_amp.copy()

            # 5. Initialize object and probe
            self.O = self._initialize_object()
            self.P = self._initialize_probe(diff_amp, self.use_shift)
            self.initial_probe_energy = np.sum(np.abs(self.P) ** 2)

            # Local references for speed
            O = self.O
            P = self.P
            positions = self.positions
            support = self.support_mask
            R_p = self.R_p
            ps = self.probe_size
            n_positions = self.n_positions
            batch_size = self.batch_size
            n_batches = n_positions // batch_size
            remainder = n_positions % batch_size
            eps = self.eps

            # 6. Main loop
            for epoch in range(self.n_epochs):
                # a. Random shuffle positions
                order = np.random.permutation(n_positions)

                # b. Mini-batch loop
                idx_start = 0
                for b in range(n_batches + (1 if remainder > 0 else 0)):
                    if b < n_batches:
                        batch_idx = order[b * batch_size:(b + 1) * batch_size]
                    else:
                        batch_idx = order[n_batches * batch_size:]
                        if len(batch_idx) == 0:
                            break

                    batch_pos = positions[batch_idx]
                    batch_amp = diff_amp_fft[batch_idx]  # (bs, 256, 256)

                    # Extract patches
                    bs = len(batch_idx)
                    patches = np.empty((bs, ps, ps), dtype=O.dtype)
                    for k in range(bs):
                        ry, rx = batch_pos[k]
                        patches[k] = O[ry:ry + ps, rx:rx + ps]

                    # Compute exit waves
                    psi = P[np.newaxis, :, :] * patches  # (bs, ps, ps)

                    # Batch FFT forward
                    Psi = np.fft.fft2(psi)  # (bs, ps, ps)

                    # Magnitude replacement
                    Psi_amp = np.abs(Psi)
                    Psi_c = batch_amp * Psi / (Psi_amp + eps)

                    # Batch IFFT
                    psi_c = np.fft.ifft2(Psi_c)

                    # Exit wave differences
                    dpsi = psi_c - psi

                    # Object update
                    P_abs_sq = np.abs(P) ** 2
                    P_max_sq = np.max(P_abs_sq)
                    if P_max_sq > eps:
                        obj_upd = self.alpha_O * np.conj(P)[np.newaxis, :, :] * dpsi / (P_max_sq + eps)

                        num = np.zeros_like(O)
                        den = np.zeros((self.obj_size, self.obj_size), dtype=np.float64)
                        for k in range(bs):
                            ry, rx = batch_pos[k]
                            num[ry:ry + ps, rx:rx + ps] += obj_upd[k]
                            den[ry:ry + ps, rx:rx + ps] += 1.0
                        den = np.maximum(den, 1.0)
                        O = O + num / den

                    # Probe update
                    if epoch >= self.probe_update_start_epoch:
                        O_abs_sq = np.abs(patches) ** 2
                        O_max_sq = np.max(O_abs_sq)
                        if O_max_sq > eps:
                            probe_upd = np.conj(patches) * dpsi / (O_max_sq + eps)
                            P = P + self.alpha_P * np.mean(probe_upd, axis=0)

                # c. Apply amplitude constraint
                O = self._apply_amplitude_constraint(O)

                # d. Apply reflectivity prior
                if epoch >= self.prior_start_epoch:
                    O = self._apply_reflectivity_prior(O, support, R_p, epoch)

                # e. Normalize probe energy
                P = self._normalize_probe_energy(P, self.initial_probe_energy)

            # Store final results
            self.O = O
            self.P = P

            # 7. Remove global phase ambiguity
            O = self._remove_global_phase_ambiguity(O, support)

            # 8. Format and return output
            result = self._format_output(O)
            return result



if __name__ == "__main__":
    # Load measured diffraction intensities
    input_data = np.load("dataset/input.npy")

    # Instantiate solver with default hyperparameters
    solver = InverseSolver()

    # Run reconstruction
    result = solver.solve(input_data)

    # Save output
    np.save("output.npy", result)