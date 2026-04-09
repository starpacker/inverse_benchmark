import numpy as np
import time
from typing import Tuple, List, Optional, Union

class InverseSolver:
    """
    Sequential rpPIE (reflectivity-prior Ptychographic Iterative Engine) solver
    for recovering complex-valued reflection functions from far-field diffraction
    intensity measurements of periodic copper pad arrays.

    Implements Eq. 5 (rpIO) and Eq. 6 (rpPIE) from the paper, with Fraunhofer
    propagation forward model and reflectivity prior enforcement outside support.
    """

    def __init__(
            self,
            probe_size: int = 256,
            obj_size: int = 600,
            n_scan_side: int = 9,
            scan_step: int = 40,
            pad_pixels: int = 22,
            pad_pitch: int = 30,
            n_pads: int = 15,
            wavelength_nm: float = 13.5,
            n_epochs: int = 200,
            beta: float = 1.0,
            beta_probe: float = 0.5,
            alpha: float = 1e-3,
            beta_prior: float = 0.3,
            prior_ramp_epochs: int = 10,
            probe_update_start_epoch: int = 2,
            fft_check_epoch: int = 5,
            time_limit_seconds: float = 50.0,
            eps: float = 1e-12,
            n_SiCN_real: float = 0.9791,
            n_SiCN_imag: float = 0.0059,
            n_SiO2_real: float = 0.9780,
            n_SiO2_imag: float = 0.0108,
            n_Si_real: float = 0.9991,
            n_Si_imag: float = 0.0018,
            n_Cu_real: float = 0.9625,
            n_Cu_imag: float = 0.0613,
        ) -> None:
            # Geometry parameters
            self.probe_size: int = probe_size
            self.obj_size: int = obj_size
            self.n_scan_side: int = n_scan_side
            self.scan_step: int = scan_step
            self.pad_pixels: int = pad_pixels
            self.pad_pitch: int = pad_pitch
            self.n_pads: int = n_pads
            self.wavelength_nm: float = wavelength_nm

            # Algorithm hyperparameters
            self.n_epochs: int = n_epochs
            self.beta: float = beta
            self.beta_probe: float = beta_probe
            self.alpha: float = alpha
            self.beta_prior: float = beta_prior
            self.prior_ramp_epochs: int = prior_ramp_epochs
            self.probe_update_start_epoch: int = probe_update_start_epoch
            self.fft_check_epoch: int = fft_check_epoch
            self.time_limit_seconds: float = time_limit_seconds
            self.eps: float = eps

            # Material optical constants (n - ik convention)
            self.n_SiCN: complex = complex(n_SiCN_real, -n_SiCN_imag)
            self.n_SiO2: complex = complex(n_SiO2_real, -n_SiO2_imag)
            self.n_Si: complex = complex(n_Si_real, -n_Si_imag)
            self.n_Cu: complex = complex(n_Cu_real, -n_Cu_imag)

            # Compute Fresnel reflectivities from optical constants
            # r = (1 - n) / (1 + n) for normal incidence
            self.R_p: complex = (1.0 - self.n_SiCN) / (1.0 + self.n_SiCN)
            self.R_Cu: complex = (1.0 - self.n_Cu) / (1.0 + self.n_Cu)

            # Derived quantities computed during solve
            self.positions: Optional[np.ndarray] = None
            self.support: Optional[np.ndarray] = None
            self.O: Optional[np.ndarray] = None
            self.P: Optional[np.ndarray] = None

    def _compute_fresnel_reflectivity(
            self, n_material: complex
        ) -> complex:
            """
            Compute normal-incidence Fresnel reflection coefficient for a material.

            Parameters
            ----------
            n_material : complex
                Complex refractive index of the material (n - ik convention).

            Returns
            -------
            complex
                Complex Fresnel reflection coefficient r = (1 - n) / (1 + n).
            """
            return (1.0 - n_material) / (1.0 + n_material)

    def _build_scan_positions(self) -> np.ndarray:
            """
            Construct the scan position grid centered on the object.

            Returns
            -------
            np.ndarray
                Array of shape (n_scan_side^2, 2) with (row, col) scan positions.
            """
            ps = self.probe_size
            n_side = self.n_scan_side
            step = self.scan_step

            # Total extent of the scan grid including probe footprint
            total_scan = ps + (n_side - 1) * step  # 256 + 320 = 576

            # Center the scan grid on the object
            offset = (self.obj_size - total_scan) // 2  # (600 - 576) // 2 = 12

            positions = []
            for i in range(n_side):
                for j in range(n_side):
                    positions.append((offset + i * step, offset + j * step))

            return np.array(positions, dtype=np.int64)

    def _build_support_mask(self) -> np.ndarray:
            """
            Build the binary support mask S defining pad geometry and periodicity.

            Returns
            -------
            np.ndarray
                Boolean array of shape (obj_size, obj_size), True on pad locations.
            """
            pad_sz = self.pad_pixels
            pitch = self.pad_pitch
            n_pads = self.n_pads

            # Total extent of the pad array
            arr_ext = (n_pads - 1) * pitch + pad_sz  # 14*30 + 22 = 442
            # Center the array on the object grid
            arr_off = (self.obj_size - arr_ext) // 2  # (600 - 442) // 2 = 79

            S = np.zeros((self.obj_size, self.obj_size), dtype=bool)
            for m in range(n_pads):
                for n_idx in range(n_pads):
                    r0 = arr_off + m * pitch
                    c0 = arr_off + n_idx * pitch
                    S[r0:r0 + pad_sz, c0:c0 + pad_sz] = True

            return S

    def _detect_fft_convention(
            self, diffraction_intensities: np.ndarray
        ) -> bool:
            """
            Determine whether the DC component is at the center or corner of
            the diffraction patterns.

            Parameters
            ----------
            diffraction_intensities : np.ndarray
                Raw measured intensities of shape (n_positions, probe_size, probe_size).

            Returns
            -------
            bool
                True if DC is at center (requires ifftshift before processing).
            """
            I = diffraction_intensities
            ps = self.probe_size
            # Define center region (middle ~1/8 of each dimension)
            c_lo = ps // 2 - ps // 8
            c_hi = ps // 2 + ps // 8
            center_energy = np.sum(I[:, c_lo:c_hi, c_lo:c_hi])

            # Define corner regions (each corner ~1/8 of each dimension)
            edge = ps // 8
            corner_energy = (
                np.sum(I[:, :edge, :edge]) +
                np.sum(I[:, :edge, -edge:]) +
                np.sum(I[:, -edge:, :edge]) +
                np.sum(I[:, -edge:, -edge:])
            )

            return center_energy > corner_energy

    def _prepare_diffraction_amplitudes(
            self,
            diffraction_intensities: np.ndarray,
            dc_at_center: bool,
        ) -> np.ndarray:
            """
            Compute diffraction amplitudes with appropriate FFT shift convention.

            Parameters
            ----------
            diffraction_intensities : np.ndarray
                Raw measured intensities of shape (n_positions, probe_size, probe_size).
            dc_at_center : bool
                Whether DC is centered in the input data.

            Returns
            -------
            np.ndarray
                Diffraction amplitudes of shape (n_positions, probe_size, probe_size),
                shifted to match numpy fft2 convention (DC at corner).
            """
            # Clip negative values (noise can cause small negatives) and take sqrt
            diff_amp = np.sqrt(np.maximum(diffraction_intensities, 0.0))

            # numpy fft2 expects DC at corner (index [0,0]).
            # If DC is at center in the measured data, apply ifftshift to move it to corner.
            if dc_at_center:
                diff_amp_fft = np.fft.ifftshift(diff_amp, axes=(-2, -1))
            else:
                diff_amp_fft = diff_amp.copy()

            return diff_amp_fft

    def _initialize_object(self) -> np.ndarray:
            """
            Initialize the complex object with physically meaningful reflectivities:
            substrate reflectivity everywhere, Cu reflectivity on pad support.

            Returns
            -------
            np.ndarray
                Complex array of shape (obj_size, obj_size).
            """
            O = np.full((self.obj_size, self.obj_size), self.R_p, dtype=np.complex128)
            O[self.support] = self.R_Cu
            return O

    def _initialize_probe(
            self, diff_amp_fft: np.ndarray
        ) -> np.ndarray:
            """
            Initialize the probe function from the mean diffraction amplitude
            via inverse FFT, normalized so max|P| = 1.

            Parameters
            ----------
            diff_amp_fft : np.ndarray
                FFT-convention-corrected diffraction amplitudes, shape (n_pos, ps, ps).

            Returns
            -------
            np.ndarray
                Complex probe array of shape (probe_size, probe_size).
            """
            # Compute mean diffraction amplitude across all scan positions
            mean_amp = np.mean(diff_amp_fft, axis=0)  # (probe_size, probe_size)

            # Inverse FFT to get real-space probe estimate
            P = np.fft.ifft2(mean_amp).astype(np.complex128)

            # Normalize so that max|P| = 1
            P_max = np.max(np.abs(P))
            if P_max > self.eps:
                P /= P_max

            return P

    def _compute_object_update_weight(
            self, probe: np.ndarray
        ) -> np.ndarray:
            """
            Precompute the per-pixel object update weight from Eq. 6, Step 1:
            w(r) = |P(r)| / |P_max| * P*(r) / (|P(r)|^2 + alpha)

            Parameters
            ----------
            probe : np.ndarray
                Current probe estimate, shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Complex weight array of shape (probe_size, probe_size).
            """
            P_abs = np.abs(probe)                        # |P(r)|
            P_max = np.max(P_abs)                        # |P_max|
            P_conj = np.conj(probe)                      # P*(r)
            P_abs_sq = P_abs ** 2                        # |P(r)|^2

            # Eq. 6 weight: |P(r)| / |P_max| * P*(r) / (|P(r)|^2 + alpha)
            weight = (P_abs / (P_max + self.eps)) * P_conj / (P_abs_sq + self.alpha)

            return weight

    def _compute_probe_update_weight(
            self, object_patch: np.ndarray
        ) -> np.ndarray:
            """
            Compute the per-pixel probe update weight (ePIE-style):
            w(r) = |O(r)| / |O_max| * O*(r) / (|O(r)|^2 + alpha)

            Parameters
            ----------
            object_patch : np.ndarray
                Object patch at current scan position, shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Complex weight array of shape (probe_size, probe_size).
            """
            O_abs = np.abs(object_patch)                    # |O(r)|
            O_max = np.max(O_abs)                           # |O_max|
            O_conj = np.conj(object_patch)                  # O*(r)
            O_abs_sq = O_abs ** 2                           # |O(r)|^2

            # ePIE-style weight: |O(r)| / |O_max| * O*(r) / (|O(r)|^2 + alpha)
            weight = (O_abs / (O_max + self.eps)) * O_conj / (O_abs_sq + self.alpha)

            return weight

    def forward(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Compute the forward model: far-field diffraction amplitude from
            the exit wave at a single scan position via Fraunhofer propagation.

            Psi(k) = FFT{ P(r - R) * O(r) }

            Parameters
            ----------
            obj : np.ndarray
                Complex object array of shape (obj_size, obj_size).
            probe : np.ndarray
                Complex probe array of shape (probe_size, probe_size).
            position : Tuple[int, int]
                (row, col) top-left corner of the probe on the object grid.

            Returns
            -------
            np.ndarray
                Complex far-field diffraction pattern of shape (probe_size, probe_size).
            """
            ry, rx = position
            ps = self.probe_size

            # Extract the object patch at the scan position
            obj_patch = obj[ry:ry + ps, rx:rx + ps]

            # Form the exit wave: element-wise multiplication of probe and object patch
            exit_wave = probe * obj_patch

            # Fraunhofer propagation via 2D FFT
            psi_k = np.fft.fft2(exit_wave)

            return psi_k

    def _fourier_magnitude_replacement(
            self,
            psi_k: np.ndarray,
            measured_amplitude: np.ndarray,
        ) -> np.ndarray:
            """
            Replace the Fourier magnitude with measured amplitude while preserving phase.

            Psi_c(k) = sqrt(I(k)) * Psi(k) / |Psi(k)|

            Parameters
            ----------
            psi_k : np.ndarray
                Current far-field estimate, shape (probe_size, probe_size).
            measured_amplitude : np.ndarray
                Measured diffraction amplitude sqrt(I), shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Corrected far-field with measured amplitude, shape (probe_size, probe_size).
            """
            psi_amp = np.abs(psi_k)
            psi_corrected = measured_amplitude * psi_k / (psi_amp + self.eps)
            return psi_corrected

    def _apply_reflectivity_prior(
            self,
            obj: np.ndarray,
            epoch: int,
        ) -> np.ndarray:
            """
            Enforce the rpPIE reflectivity prior outside the support mask (Eq. 6, Step 2):
            O(r) = beta_prior * O(r) + (1 - beta_prior) * R_p, for r not in S.
            Includes a ramp-in schedule over the first prior_ramp_epochs.

            Parameters
            ----------
            obj : np.ndarray
                Current complex object estimate, shape (obj_size, obj_size).
            epoch : int
                Current epoch index (for ramp-in scheduling).

            Returns
            -------
            np.ndarray
                Object with reflectivity prior applied outside support.
            """
            if epoch >= 1:
                # Ramp-in factor: linearly increase from 0 to 1 over prior_ramp_epochs
                gamma = min(1.0, epoch / float(self.prior_ramp_epochs))

                bp = self.beta_prior  # 0.3

                # Outside support: O = (1 - gamma*(1-bp)) * O + gamma*(1-bp) * R_p
                # At full ramp (gamma=1): O = bp * O + (1-bp) * R_p
                outside_mask = ~self.support
                blend_factor = gamma * (1.0 - bp)
                obj[outside_mask] = (1.0 - blend_factor) * obj[outside_mask] + blend_factor * self.R_p

            return obj

    def _enforce_amplitude_constraint(
            self, obj: np.ndarray
        ) -> np.ndarray:
            """
            Enforce physical constraint that reflection coefficient amplitude <= 1.

            Parameters
            ----------
            obj : np.ndarray
                Current complex object estimate, shape (obj_size, obj_size).

            Returns
            -------
            np.ndarray
                Object with amplitudes clipped to [0, 1].
            """
            amp = np.abs(obj)
            mask_too_large = amp > 1.0
            if np.any(mask_too_large):
                obj[mask_too_large] = obj[mask_too_large] / amp[mask_too_large]
            return obj

    def _normalize_probe_energy(
            self,
            probe: np.ndarray,
            reference_energy: float,
            max_ratio: float = 4.0,
        ) -> np.ndarray:
            """
            Prevent probe energy from blowing up by rescaling if it exceeds
            max_ratio times the reference energy.

            Parameters
            ----------
            probe : np.ndarray
                Current probe estimate, shape (probe_size, probe_size).
            reference_energy : float
                Initial probe energy for comparison.
            max_ratio : float
                Maximum allowed energy ratio before rescaling.

            Returns
            -------
            np.ndarray
                Energy-normalized probe.
            """
            current_energy = np.sum(np.abs(probe) ** 2)
            if current_energy > max_ratio * reference_energy:
                scale = np.sqrt(reference_energy / current_energy)
                probe = probe * scale
            return probe

    def _verify_fft_convention(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            diff_amp_fft: np.ndarray,
            diff_amp_alt: np.ndarray,
            positions: np.ndarray,
            sample_indices: Optional[np.ndarray] = None,
        ) -> Tuple[bool, np.ndarray]:
            """
            At fft_check_epoch, compare residuals for current and alternative FFT
            conventions. Switch if alternative yields significantly lower residual.

            Parameters
            ----------
            obj : np.ndarray
                Current object estimate, shape (obj_size, obj_size).
            probe : np.ndarray
                Current probe estimate, shape (probe_size, probe_size).
            diff_amp_fft : np.ndarray
                Current diffraction amplitudes, shape (n_pos, ps, ps).
            diff_amp_alt : np.ndarray
                Alternative-convention diffraction amplitudes, shape (n_pos, ps, ps).
            positions : np.ndarray
                Scan positions, shape (n_pos, 2).
            sample_indices : Optional[np.ndarray]
                Subset of position indices to test (for speed).

            Returns
            -------
            Tuple[bool, np.ndarray]
                (should_switch, selected_diff_amp) — whether to switch and the
                diffraction amplitudes to use going forward.
            """
            if sample_indices is None:
                # Sample every 20th position for speed
                sample_indices = np.arange(0, len(positions), 20)

            ps = self.probe_size

            # Compute residual for current convention
            residual_current = 0.0
            for idx in sample_indices:
                ry, rx = positions[idx]
                psi_test = probe * obj[ry:ry + ps, rx:rx + ps]
                Psi_test = np.fft.fft2(psi_test)
                residual_current += np.sum((np.abs(Psi_test) - diff_amp_fft[idx]) ** 2)

            # Compute residual for alternative convention
            residual_alt = 0.0
            for idx in sample_indices:
                ry, rx = positions[idx]
                psi_test = probe * obj[ry:ry + ps, rx:rx + ps]
                Psi_test = np.fft.fft2(psi_test)
                residual_alt += np.sum((np.abs(Psi_test) - diff_amp_alt[idx]) ** 2)

            # Switch if alternative convention has significantly lower residual
            should_switch = residual_alt < 0.5 * residual_current

            if should_switch:
                return True, diff_amp_alt
            else:
                return False, diff_amp_fft

    def _remove_global_phase_ambiguity(
            self, obj: np.ndarray
        ) -> np.ndarray:
            """
            Remove the global phase ambiguity by referencing to the substrate
            (outside support) phase.

            Parameters
            ----------
            obj : np.ndarray
                Final complex object estimate, shape (obj_size, obj_size).

            Returns
            -------
            np.ndarray
                Phase-referenced object.
            """
            # Use substrate pixels (outside support) as phase reference
            substrate_vals = obj[~self.support]

            # Compute reference phase using the median of real and imaginary parts
            # for robustness against outliers
            median_real = np.median(np.real(substrate_vals))
            median_imag = np.median(np.imag(substrate_vals))
            phase_ref = np.angle(complex(median_real, median_imag))

            # Remove global phase by multiplying by exp(-j * phase_ref)
            obj = obj * np.exp(-1j * phase_ref)

            return obj

    def _format_output(
            self, obj: np.ndarray
        ) -> np.ndarray:
            """
            Format the reconstructed object into the output array:
            channel 0 = amplitude |O|, channel 1 = phase angle(O).

            Parameters
            ----------
            obj : np.ndarray
                Final complex object estimate, shape (obj_size, obj_size).

            Returns
            -------
            np.ndarray
                Output array of shape (2, obj_size, obj_size), float32.
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(obj).astype(np.float32)
            output[1] = np.angle(obj).astype(np.float32)
            return output

    def solve(
            self, diffraction_intensities: np.ndarray
        ) -> np.ndarray:
            """
            Main solver: run the sequential rpPIE algorithm to reconstruct the
            complex reflection function from measured diffraction intensities.
            """
            t_start = time.time()

            # Step 1: Data preprocessing
            I = np.maximum(diffraction_intensities, 0.0).astype(np.float64)

            # Step 2: Detect FFT convention
            dc_at_center = self._detect_fft_convention(I)

            # Step 3: Prepare diffraction amplitudes (DC moved to corner for numpy fft2)
            diff_amp_fft = self._prepare_diffraction_amplitudes(I, dc_at_center)

            # Also prepare the alternative convention for verification
            diff_amp_raw = np.sqrt(np.maximum(I, 0.0))
            if dc_at_center:
                # Current: ifftshift applied. Alternative: no shift
                diff_amp_alt = diff_amp_raw.copy()
            else:
                # Current: no shift. Alternative: ifftshift applied
                diff_amp_alt = np.fft.ifftshift(diff_amp_raw, axes=(-2, -1))

            # Step 4: Build geometry
            self.positions = self._build_scan_positions()
            self.support = self._build_support_mask()

            # Step 5: Initialize object and probe
            self.O = self._initialize_object()
            self.P = self._initialize_probe(diff_amp_fft)

            # Record initial probe energy for normalization
            P_energy_init = np.sum(np.abs(self.P) ** 2)

            ps = self.probe_size
            positions = self.positions
            n_pos = len(positions)

            # Step 6: Sequential rpPIE iterations
            for epoch in range(self.n_epochs):
                # Time check
                if time.time() - t_start > self.time_limit_seconds:
                    break

                # Random scan order for convergence
                order = np.random.permutation(n_pos)

                # Precompute object update weight from current probe (Eq. 6)
                obj_weight = self._compute_object_update_weight(self.P)

                for idx in order:
                    ry, rx = positions[idx]

                    # Extract object patch (copy for probe update later)
                    O_patch = self.O[ry:ry + ps, rx:rx + ps].copy()

                    # Form exit wave
                    psi = self.P * O_patch

                    # Forward propagate
                    Psi = np.fft.fft2(psi)

                    # Fourier magnitude replacement
                    Psi_c = self._fourier_magnitude_replacement(Psi, diff_amp_fft[idx])

                    # Inverse propagate
                    psi_c = np.fft.ifft2(Psi_c)

                    # Exit wave difference
                    dpsi = psi_c - psi

                    # Object update (Eq. 6, Step 1)
                    self.O[ry:ry + ps, rx:rx + ps] += self.beta * obj_weight * dpsi

                    # Probe update (ePIE-style, after warmup)
                    if epoch >= self.probe_update_start_epoch:
                        probe_weight = self._compute_probe_update_weight(O_patch)
                        self.P += self.beta_probe * probe_weight * dpsi

                # End-of-epoch constraints

                # 1. Amplitude constraint: |O| <= 1
                self.O = self._enforce_amplitude_constraint(self.O)

                # 2. Reflectivity prior outside support (Eq. 6, Step 2)
                self.O = self._apply_reflectivity_prior(self.O, epoch)

                # 3. Probe energy normalization
                self.P = self._normalize_probe_energy(self.P, P_energy_init)

                # Step 7: FFT convention verification at fft_check_epoch
                if epoch == self.fft_check_epoch:
                    should_switch, new_diff_amp = self._verify_fft_convention(
                        self.O, self.P, diff_amp_fft, diff_amp_alt, positions
                    )
                    if should_switch:
                        diff_amp_fft = new_diff_amp
                        # Reset object and probe with new convention
                        self.O = self._initialize_object()
                        self.P = self._initialize_probe(diff_amp_fft)
                        P_energy_init = np.sum(np.abs(self.P) ** 2)

            # Step 8: Remove global phase ambiguity
            self.O = self._remove_global_phase_ambiguity(self.O)

            # Step 9: Format output
            output = self._format_output(self.O)

            return output



if __name__ == "__main__":
    # Load measured diffraction intensity data
    input_data = np.load("dataset/input.npy")

    # Instantiate solver with default hyperparameters
    solver = InverseSolver(
        probe_size=256,
        obj_size=600,
        n_scan_side=9,
        scan_step=40,
        pad_pixels=22,
        pad_pitch=30,
        n_pads=15,
        wavelength_nm=13.5,
        n_epochs=200,
        beta=1.0,
        beta_probe=0.5,
        alpha=1e-3,
        beta_prior=0.3,
        prior_ramp_epochs=10,
        probe_update_start_epoch=2,
        fft_check_epoch=5,
        time_limit_seconds=50.0,
    )

    # Run rpPIE reconstruction
    result = solver.solve(input_data)

    # Save reconstructed object (amplitude + phase)
    np.save("output.npy", result)