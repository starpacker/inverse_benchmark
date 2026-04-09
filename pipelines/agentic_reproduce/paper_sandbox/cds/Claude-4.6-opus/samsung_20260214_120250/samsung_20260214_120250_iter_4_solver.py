import numpy as np
from typing import Tuple, Optional, List
from numpy.fft import fft2, ifft2, fftshift, ifftshift

class InverseSolver:
    """
    Joint rpPIE (reflectivity-prior Ptychographic Iterative Engine) solver
    for recovering complex-valued reflection functions from far-field
    diffraction intensity measurements of periodic Cu pad arrays.
    """

    def __init__(
            self,
            n_epochs: int = 500,
            probe_size: int = 256,
            obj_size: int = 600,
            scan_step: int = 40,
            n_scan_side: int = 9,
            pad_size: int = 22,
            pitch: int = 30,
            n_pads: int = 15,
            pixel_size_nm: float = 27.0,
            wavelength_nm: float = 13.5,
            alpha_O: float = 1.0,
            alpha_P: float = 1.0,
            beta_start: float = 0.9,
            beta_end: float = 0.5,
            reg_O_factor: float = 0.1,
            reg_P_factor: float = 0.1,
            eps: float = 1e-8,
            probe_start_epoch: int = 5,
            prior_start_epoch: int = 20,
            ramp_epochs: int = 50,
            rp_re_estimate_epoch: int = 60,
            convergence_check_interval: int = 10,
            lr_decay_patience: int = 20,
            lr_decay_factor: float = 0.5,
            n_SiCN: complex = 0.9791 + 0.0059j,
            n_SiO2: complex = 0.9780 + 0.0108j,
            n_Si: complex = 0.9991 + 0.0018j,
            n_Cu: complex = 0.9625 + 0.0613j,
        ) -> None:
            self.n_epochs = n_epochs
            self.probe_size = probe_size
            self.obj_size = obj_size
            self.scan_step = scan_step
            self.n_scan_side = n_scan_side
            self.pad_size = pad_size
            self.pitch = pitch
            self.n_pads = n_pads
            self.pixel_size_nm = pixel_size_nm
            self.wavelength_nm = wavelength_nm
            self.alpha_O = alpha_O
            self.alpha_P = alpha_P
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.reg_O_factor = reg_O_factor
            self.reg_P_factor = reg_P_factor
            self.eps = eps
            self.probe_start_epoch = probe_start_epoch
            self.prior_start_epoch = prior_start_epoch
            self.ramp_epochs = ramp_epochs
            self.rp_re_estimate_epoch = rp_re_estimate_epoch
            self.convergence_check_interval = convergence_check_interval
            self.lr_decay_patience = lr_decay_patience
            self.lr_decay_factor = lr_decay_factor
            self.n_SiCN = n_SiCN
            self.n_SiO2 = n_SiO2
            self.n_Si = n_Si
            self.n_Cu = n_Cu

            # Derived attributes set during solve
            self.scan_positions: List[Tuple[int, int]] = []
            self.support_mask: Optional[np.ndarray] = None
            self.R_p: complex = 1.0 + 0j
            self.object_estimate: Optional[np.ndarray] = None
            self.probe_estimate: Optional[np.ndarray] = None
            self.error_history: List[float] = []

    def _build_scan_positions(self) -> List[Tuple[int, int]]:
            """
            Compute the list of (row, col) scan positions on the object grid.

            Returns
            -------
            List[Tuple[int, int]]
                List of (row_offset, col_offset) positions for each scan point.
            """
            # Compute the total extent covered by the scan
            extent = self.probe_size + (self.n_scan_side - 1) * self.scan_step
            # Center the scan within the object grid
            offset = (self.obj_size - extent) // 2

            positions = []
            for i in range(self.n_scan_side):
                for j in range(self.n_scan_side):
                    ry = offset + i * self.scan_step
                    rx = offset + j * self.scan_step
                    positions.append((ry, rx))
            return positions

    def _build_support_mask(self) -> np.ndarray:
            """
            Construct the binary support mask S for the periodic pad array.

            Returns
            -------
            np.ndarray
                Boolean array of shape (obj_size, obj_size), True inside pads.
            """
            S = np.zeros((self.obj_size, self.obj_size), dtype=bool)
            # Total extent of the pad array
            array_extent = (self.n_pads - 1) * self.pitch + self.pad_size
            # Center the array in the object grid
            offset = (self.obj_size - array_extent) // 2

            for m in range(self.n_pads):
                for n in range(self.n_pads):
                    y0 = offset + m * self.pitch
                    x0 = offset + n * self.pitch
                    y1 = min(y0 + self.pad_size, self.obj_size)
                    x1 = min(x0 + self.pad_size, self.obj_size)
                    if y0 >= 0 and x0 >= 0:
                        S[y0:y1, x0:x1] = True
            return S

    def _compute_substrate_reflectivity(self) -> complex:
            """
            Compute substrate reflectivity R_p from material optical constants
            using Fresnel reflection at normal incidence.

            Returns
            -------
            complex
                Complex substrate reflectivity value.
            """
            # Fresnel reflection at normal incidence for vacuum/SiCN interface
            # r = (n_vacuum - n_SiCN) / (n_vacuum + n_SiCN)
            n_vacuum = 1.0 + 0j
            r = (n_vacuum - self.n_SiCN) / (n_vacuum + self.n_SiCN)
            return r

    def _initialize_object(self, diff_amplitudes: np.ndarray) -> np.ndarray:
            """
            Initialize the complex object estimate.

            Parameters
            ----------
            diff_amplitudes : np.ndarray
                Diffraction amplitudes, shape (n_patterns, probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Complex array of shape (obj_size, obj_size).
            """
            # Uniform complex initialization with unit amplitude and zero phase
            O = np.ones((self.obj_size, self.obj_size), dtype=np.complex128)
            return O


# >>> TARGET FUNCTION: _initialize_probe <<<
    def _initialize_probe(self, diff_amplitudes: np.ndarray) -> np.ndarray:
                """
                Initialize the probe function from the mean diffraction pattern.

                Parameters
                ----------
                diff_amplitudes : np.ndarray
                    Diffraction amplitudes, shape (n_patterns, probe_size, probe_size).

                Returns
                -------
                np.ndarray
                    Complex array of shape (probe_size, probe_size).
                """
                ps = self.probe_size

                # Create a Gaussian probe in real space
                y = np.arange(ps) - ps / 2.0
                x = np.arange(ps) - ps / 2.0
                yy, xx = np.meshgrid(y, x, indexing='ij')

                sigma = ps / 4.0  # Gaussian width
                gaussian = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

                # Scale probe amplitude so that sum(|P|^2) matches mean(sum(diff_amplitudes^2))
                # By Parseval's theorem: sum(|FFT(P*O)|^2) = N^2 * sum(|P*O|^2)
                # For flat object (O=1): sum(|FFT(P)|^2) = N^2 * sum(|P|^2)
                # And sum(|FFT(P)|^2) ≈ mean(sum(I_j)) = mean(sum(diff_amplitudes^2))
                mean_total_intensity = np.mean(np.sum(diff_amplitudes**2, axis=(1, 2)))

                # sum(|P|^2) should equal mean_total_intensity / N^2
                target_probe_energy = mean_total_intensity / (ps * ps)
                current_energy = np.sum(gaussian**2)

                if current_energy > 0 and target_probe_energy > 0:
                    scale = np.sqrt(target_probe_energy / current_energy)
                else:
                    scale = 1.0

                P = (gaussian * scale).astype(np.complex128)

                return P

    def _estimate_rp_from_reconstruction(
            self, O: np.ndarray, S: np.ndarray
        ) -> complex:
            """
            Adaptively re-estimate substrate reflectivity from current reconstruction.

            Parameters
            ----------
            O : np.ndarray
                Current complex object estimate, shape (obj_size, obj_size).
            S : np.ndarray
                Boolean support mask, shape (obj_size, obj_size).

            Returns
            -------
            complex
                Updated substrate reflectivity estimate.
            """
            outside_mask = ~S
            if np.any(outside_mask):
                vals = O[outside_mask]
                rp_real = np.median(np.real(vals))
                rp_imag = np.median(np.imag(vals))
                return complex(rp_real, rp_imag)
            else:
                return self.R_p

    def _get_beta(self, epoch: int) -> float:
            """
            Compute the annealed feedback parameter beta for the current epoch.

            Parameters
            ----------
            epoch : int
                Current epoch index.

            Returns
            -------
            float
                Annealed beta value.
            """
            # Linear anneal from beta_start to beta_end over all epochs
            beta = max(self.beta_end, self.beta_start - (self.beta_start - self.beta_end) * epoch / self.n_epochs)
            return beta

    def _get_prior_gamma(self, epoch: int) -> float:
            """
            Compute the ramp-up factor gamma for reflectivity prior enforcement.

            Parameters
            ----------
            epoch : int
                Current epoch index.

            Returns
            -------
            float
                Ramp factor in [0, 1], 0 before prior_start_epoch.
            """
            if epoch < self.prior_start_epoch:
                return 0.0
            return min(1.0, (epoch - self.prior_start_epoch) / max(self.ramp_epochs, 1))

    def forward(
            self, O_patch: np.ndarray, P: np.ndarray
        ) -> np.ndarray:
            """
            Forward model: compute far-field diffraction amplitude from
            exit wave via Fraunhofer propagation (FFT).

            Parameters
            ----------
            O_patch : np.ndarray
                Object patch at current scan position, shape (probe_size, probe_size), complex.
            P : np.ndarray
                Probe function, shape (probe_size, probe_size), complex.

            Returns
            -------
            np.ndarray
                Complex far-field diffraction pattern, shape (probe_size, probe_size).
            """
            psi = P * O_patch
            Psi = fftshift(fft2(ifftshift(psi)))
            return Psi

    def _inverse_propagate(self, Psi: np.ndarray) -> np.ndarray:
            """
            Inverse propagation: compute exit wave from far-field pattern via IFFT.

            Parameters
            ----------
            Psi : np.ndarray
                Complex far-field pattern, shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Complex exit wave, shape (probe_size, probe_size).
            """
            return fftshift(ifft2(ifftshift(Psi)))

    def _fourier_magnitude_constraint(
            self, Psi: np.ndarray, measured_amplitude: np.ndarray
        ) -> np.ndarray:
            """
            Replace Fourier magnitudes with measured amplitudes, preserving phase.

            Parameters
            ----------
            Psi : np.ndarray
                Current far-field estimate, shape (probe_size, probe_size), complex.
            measured_amplitude : np.ndarray
                Measured diffraction amplitude sqrt(I), shape (probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Corrected far-field pattern with measured magnitudes and estimated phases.
            """
            Psi_amp = np.abs(Psi)
            Psi_c = measured_amplitude * Psi / (Psi_amp + self.eps)
            return Psi_c

    def _update_object(
            self,
            O: np.ndarray,
            P: np.ndarray,
            delta_psi: np.ndarray,
            ry: int,
            rx: int,
        ) -> np.ndarray:
            """
            ePIE-style object update for the patch at position (ry, rx).

            Parameters
            ----------
            O : np.ndarray
                Full object array, shape (obj_size, obj_size), complex. Modified in-place.
            P : np.ndarray
                Probe function, shape (probe_size, probe_size), complex.
            delta_psi : np.ndarray
                Exit wave difference (psi_c - psi), shape (probe_size, probe_size), complex.
            ry : int
                Row offset of the current scan position.
            rx : int
                Column offset of the current scan position.

            Returns
            -------
            np.ndarray
                Updated object array (same reference, modified in-place).
            """
            ps = self.probe_size
            P_conj = np.conj(P)
            P_abs2 = np.abs(P) ** 2
            P_max = np.max(np.abs(P))
            reg_O = self.reg_O_factor * P_max ** 2
            O[ry:ry + ps, rx:rx + ps] += self.alpha_O * P_conj / (P_max * (P_abs2 + reg_O)) * delta_psi
            return O

    def _update_probe(
            self,
            P: np.ndarray,
            O_patch: np.ndarray,
            delta_psi: np.ndarray,
        ) -> np.ndarray:
            """
            ePIE-style probe update.

            Parameters
            ----------
            P : np.ndarray
                Probe function, shape (probe_size, probe_size), complex. Modified in-place.
            O_patch : np.ndarray
                Object patch at current position, shape (probe_size, probe_size), complex.
            delta_psi : np.ndarray
                Exit wave difference, shape (probe_size, probe_size), complex.

            Returns
            -------
            np.ndarray
                Updated probe array.
            """
            O_conj = np.conj(O_patch)
            O_abs2 = np.abs(O_patch) ** 2
            O_max = np.max(np.abs(O_patch))
            reg_P = self.reg_P_factor * O_max ** 2
            P += self.alpha_P * O_conj / (O_max * (O_abs2 + reg_P)) * delta_psi
            return P

    def _apply_reflectivity_prior(
            self,
            O: np.ndarray,
            S: np.ndarray,
            R_p: complex,
            beta: float,
            gamma: float,
            ry: int,
            rx: int,
        ) -> np.ndarray:
            """
            Enforce reflectivity prior outside support mask (rpPIE Step 2).

            Parameters
            ----------
            O : np.ndarray
                Full object array, shape (obj_size, obj_size), complex. Modified in-place.
            S : np.ndarray
                Boolean support mask, shape (obj_size, obj_size).
            R_p : complex
                Known substrate reflectivity.
            beta : float
                Current feedback parameter.
            gamma : float
                Ramp-up factor for prior enforcement.
            ry : int
                Row offset of the current scan position.
            rx : int
                Column offset of the current scan position.

            Returns
            -------
            np.ndarray
                Updated object array with prior enforced outside support.
            """
            ps = self.probe_size
            S_patch = S[ry:ry + ps, rx:rx + ps]
            outside = ~S_patch
            if np.any(outside):
                blend = gamma * (1.0 - beta)
                O[ry:ry + ps, rx:rx + ps][outside] = \
                    (1.0 - blend) * O[ry:ry + ps, rx:rx + ps][outside] + blend * R_p
            return O

    def _compute_error(
            self,
            O: np.ndarray,
            P: np.ndarray,
            diff_amplitudes: np.ndarray,
        ) -> float:
            """
            Compute normalized reconstruction error metric across all scan positions.

            Parameters
            ----------
            O : np.ndarray
                Current object estimate, shape (obj_size, obj_size), complex.
            P : np.ndarray
                Current probe estimate, shape (probe_size, probe_size), complex.
            diff_amplitudes : np.ndarray
                Measured diffraction amplitudes, shape (n_patterns, probe_size, probe_size).

            Returns
            -------
            float
                Mean normalized error across all diffraction patterns.
            """
            total_error = 0.0
            ps = self.probe_size
            n_patterns = len(self.scan_positions)
            for idx in range(n_patterns):
                ry, rx = self.scan_positions[idx]
                O_patch = O[ry:ry + ps, rx:rx + ps]
                Psi = self.forward(O_patch, P)
                Psi_amp = np.abs(Psi)
                measured = diff_amplitudes[idx]
                numerator = np.sum((measured - Psi_amp) ** 2)
                denominator = np.sum(measured ** 2) + self.eps
                total_error += numerator / denominator
            return total_error / max(n_patterns, 1)

    def _remove_global_phase_ambiguity(
            self, O: np.ndarray, S: np.ndarray
        ) -> np.ndarray:
            """
            Remove global phase ambiguity by referencing phase to substrate region.

            Parameters
            ----------
            O : np.ndarray
                Complex object estimate, shape (obj_size, obj_size).
            S : np.ndarray
                Boolean support mask, shape (obj_size, obj_size).

            Returns
            -------
            np.ndarray
                Phase-corrected complex object.
            """
            outside_mask = ~S
            if np.any(outside_mask):
                mean_substrate = np.mean(O[outside_mask])
                phase_ref = np.angle(mean_substrate)
                O = O * np.exp(-1j * phase_ref)
            return O

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """
            Format the reconstructed object into the required output tensor.

            Parameters
            ----------
            O : np.ndarray
                Complex object estimate, shape (obj_size, obj_size).

            Returns
            -------
            np.ndarray
                Output tensor of shape (2, obj_size, obj_size), float32.
                Channel 0: amplitude |O|, Channel 1: phase angle(O).
            """
            output = np.zeros((2, self.obj_size, self.obj_size), dtype=np.float32)
            output[0] = np.abs(O).astype(np.float32)
            output[1] = np.angle(O).astype(np.float32)
            return output

    def solve(self, input_data: np.ndarray) -> np.ndarray:
            """
            Main reconstruction loop: joint rpPIE with probe update.

            Parameters
            ----------
            input_data : np.ndarray
                Measured diffraction intensities, shape (n_patterns, probe_size, probe_size).

            Returns
            -------
            np.ndarray
                Reconstructed object, shape (2, obj_size, obj_size), float32.
                Channel 0: amplitude, Channel 1: phase.
            """
            # --- Step 1: Preprocess diffraction data ---
            diff_intensities = np.maximum(input_data.astype(np.float64), 0.0)
            diff_amplitudes = np.sqrt(diff_intensities)
            n_patterns = diff_amplitudes.shape[0]

            # --- Step 2: Build scan positions ---
            self.scan_positions = self._build_scan_positions()
            assert len(self.scan_positions) == n_patterns, \
                f"Expected {n_patterns} positions, got {len(self.scan_positions)}"

            # --- Step 3: Build support mask ---
            self.support_mask = self._build_support_mask()

            # --- Step 4: Compute initial substrate reflectivity ---
            self.R_p = self._compute_substrate_reflectivity()

            # --- Step 5: Initialize object and probe ---
            O = self._initialize_object(diff_amplitudes)
            P = self._initialize_probe(diff_amplitudes)

            # Store estimates
            self.object_estimate = O
            self.probe_estimate = P
            self.error_history = []

            ps = self.probe_size
            current_alpha_O = self.alpha_O
            current_alpha_P = self.alpha_P
            increasing_error_count = 0
            prev_error = None

            # --- Step 6: Main iteration loop ---
            for epoch in range(self.n_epochs):
                beta = self._get_beta(epoch)
                gamma = self._get_prior_gamma(epoch)

                # Shuffle scan order
                order = np.random.permutation(n_patterns)

                for idx in order:
                    ry, rx = self.scan_positions[idx]

                    # Extract object patch
                    O_patch = O[ry:ry + ps, rx:rx + ps].copy()

                    # Forward propagation: exit wave -> far field
                    psi = P * O_patch
                    Psi = self.forward(O_patch, P)

                    # Fourier magnitude constraint
                    Psi_c = self._fourier_magnitude_constraint(Psi, diff_amplitudes[idx])

                    # Inverse propagation
                    psi_c = self._inverse_propagate(Psi_c)

                    # Exit wave difference
                    delta_psi = psi_c - psi

                    # Object update
                    P_conj = np.conj(P)
                    P_abs2 = np.abs(P) ** 2
                    P_max = np.max(np.abs(P))
                    reg_O = self.reg_O_factor * P_max ** 2
                    O[ry:ry + ps, rx:rx + ps] += current_alpha_O * P_conj / (P_max * (P_abs2 + reg_O) + self.eps) * delta_psi

                    # Probe update (after probe_start_epoch)
                    if epoch >= self.probe_start_epoch:
                        O_patch_new = O[ry:ry + ps, rx:rx + ps]
                        O_conj = np.conj(O_patch_new)
                        O_abs2 = np.abs(O_patch_new) ** 2
                        O_max = np.max(np.abs(O_patch_new))
                        reg_P = self.reg_P_factor * O_max ** 2
                        P += current_alpha_P * O_conj / (O_max * (O_abs2 + reg_P) + self.eps) * delta_psi

                    # Reflectivity prior enforcement
                    if gamma > 0:
                        self._apply_reflectivity_prior(O, self.support_mask, self.R_p, beta, gamma, ry, rx)

                # Adaptive R_p re-estimation
                if epoch == self.rp_re_estimate_epoch:
                    self.R_p = self._estimate_rp_from_reconstruction(O, self.support_mask)

                # Convergence monitoring
                if epoch % self.convergence_check_interval == 0 and epoch > 0:
                    error = self._compute_error(O, P, diff_amplitudes)
                    self.error_history.append(error)

                    if prev_error is not None:
                        if error > prev_error:
                            increasing_error_count += 1
                        else:
                            increasing_error_count = 0

                        if increasing_error_count >= self.lr_decay_patience:
                            current_alpha_O *= self.lr_decay_factor
                            current_alpha_P *= self.lr_decay_factor
                            increasing_error_count = 0

                    prev_error = error

            # --- Step 7: Post-processing ---
            # Remove global phase ambiguity
            O = self._remove_global_phase_ambiguity(O, self.support_mask)

            # Store final estimates
            self.object_estimate = O
            self.probe_estimate = P

            # Format and return output
            output = self._format_output(O)
            return output



if __name__ == "__main__":
    import sys
    import traceback
    
    try:
        # Load measured diffraction intensity data
        print("Loading input data...", flush=True)
        input_data = np.load("dataset/input.npy")
        print(f"Input shape: {input_data.shape}, dtype: {input_data.dtype}", flush=True)
        print(f"Input min: {input_data.min()}, max: {input_data.max()}, mean: {input_data.mean()}", flush=True)

        # Instantiate solver with default hyperparameters
        solver = InverseSolver(n_epochs=300)

        # Run reconstruction
        print("Starting reconstruction...", flush=True)
        result = solver.solve(input_data)
        print(f"Output shape: {result.shape}, dtype: {result.dtype}", flush=True)
        print(f"Amplitude - min: {result[0].min():.6f}, max: {result[0].max():.6f}, mean: {result[0].mean():.6f}", flush=True)
        print(f"Phase - min: {result[1].min():.6f}, max: {result[1].max():.6f}, mean: {result[1].mean():.6f}", flush=True)

        if len(solver.error_history) > 0:
            print(f"Error history (first 5): {solver.error_history[:5]}", flush=True)
            print(f"Error history (last 5): {solver.error_history[-5:]}", flush=True)

        # Save output
        np.save("output.npy", result)
        print("Result saved to output.npy", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)