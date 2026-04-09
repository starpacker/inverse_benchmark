import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional, Union
import sys

class InverseSolver:
    def __init__(
            self,
            object_shape: Tuple[int, int] = (600, 600),
            probe_shape: Tuple[int, int] = (256, 256),
            n_epochs_phase1: int = 50,
            n_epochs_phase2: int = 150,
            beta: float = 0.9,
            alpha_fraction: float = 1e-2,
            eps: float = 1e-10,
            step_size: int = 40,
            pad_size: int = 22,
            pitch: int = 30,
            n_pads: int = 15,
            wavelength_nm: float = 13.5,
            n_SiCN: complex = complex(0.9791, -0.0059),
            n_SiO2: complex = complex(0.9780, -0.0108),
            n_Si: complex = complex(0.9991, -0.0018),
            n_Cu: complex = complex(0.9625, -0.0613),
            support_threshold_quantile: float = 0.5,
            output_mode: str = "real_imag",
        ) -> None:
            self.object_shape = object_shape
            self.probe_shape = probe_shape
            self.n_epochs_phase1 = n_epochs_phase1
            self.n_epochs_phase2 = n_epochs_phase2
            self.beta = beta
            self.alpha_fraction = alpha_fraction
            self.eps = eps
            self.step_size = step_size
            self.pad_size = pad_size
            self.pitch = pitch
            self.n_pads = n_pads
            self.wavelength_nm = wavelength_nm
            self.n_SiCN = n_SiCN
            self.n_SiO2 = n_SiO2
            self.n_Si = n_Si
            self.n_Cu = n_Cu
            self.support_threshold_quantile = support_threshold_quantile
            self.output_mode = output_mode

            # Derived attributes set during solve
            self.probe: Optional[np.ndarray] = None
            self.scan_positions: Optional[np.ndarray] = None
            self.sqrt_intensities: Optional[np.ndarray] = None
            self.obj: Optional[np.ndarray] = None
            self.support: Optional[np.ndarray] = None
            self.R_p: Optional[complex] = None
            self.alpha: Optional[float] = None
            self.P_max_abs: Optional[float] = None

    def _parse_probe(self, probe_data: np.ndarray) -> np.ndarray:
            """
            Convert 2-channel probe data (2, H, W) to complex array (H, W).

            Parameters
            ----------
            probe_data : np.ndarray
                Probe array of shape (2, H, W) with real and imaginary parts.

            Returns
            -------
            np.ndarray
                Complex-valued probe of shape (H, W).
            """
            return probe_data[0].astype(np.complex128) + 1j * probe_data[1].astype(np.complex128)

    def _parse_scan_positions(self, positions: np.ndarray) -> np.ndarray:
            """
            Normalize and center scan positions to fit within the object grid.

            Parameters
            ----------
            positions : np.ndarray
                Raw scan positions of shape (N, 2) as (row, col).

            Returns
            -------
            np.ndarray
                Adjusted integer scan positions of shape (N, 2).
            """
            pos = positions.copy().astype(np.float64)

            # Shift so minimum position is at 0
            pos[:, 0] -= pos[:, 0].min()
            pos[:, 1] -= pos[:, 1].min()

            # Round to nearest integer
            pos = np.round(pos).astype(np.int64)

            # Compute the extent needed: max_pos + probe_shape
            probe_h, probe_w = self.probe_shape
            obj_h, obj_w = self.object_shape

            max_row = pos[:, 0].max() + probe_h
            max_col = pos[:, 1].max() + probe_w

            # Center the reconstruction within the object grid
            offset_row = max((obj_h - max_row) // 2, 0)
            offset_col = max((obj_w - max_col) // 2, 0)

            pos[:, 0] += offset_row
            pos[:, 1] += offset_col

            # Safety check: if any position would go out of bounds, clip
            # This shouldn't happen if object_shape is large enough
            pos[:, 0] = np.clip(pos[:, 0], 0, obj_h - probe_h)
            pos[:, 1] = np.clip(pos[:, 1], 0, obj_w - probe_w)

            return pos

    def _compute_substrate_reflectivity(self) -> complex:
            """
            Compute Fresnel reflection coefficient at vacuum/substrate interface.

            Returns
            -------
            complex
                Complex reflectivity R_p for the substrate.
            """
            # Fresnel reflection at normal incidence: R = (1 - n_tilde) / (1 + n_tilde)
            n_tilde = self.n_SiCN
            R_p = (1.0 - n_tilde) / (1.0 + n_tilde)
            return R_p

    def _build_physics_support_mask(self) -> np.ndarray:
            """
            Build a binary support mask for the 15x15 periodic pad array on the object grid.

            Returns
            -------
            np.ndarray
                Binary mask of shape object_shape, 1 inside pads, 0 outside.
            """
            obj_h, obj_w = self.object_shape
            mask = np.zeros((obj_h, obj_w), dtype=np.float64)

            # Array of n_pads x n_pads pads with given pitch and pad_size
            # Total array extent: n_pads * pitch
            array_extent = self.n_pads * self.pitch

            # Center the array within the object grid
            offset_row = (obj_h - array_extent) // 2
            offset_col = (obj_w - array_extent) // 2

            # Each pad is centered within its pitch cell
            pad_offset_within_cell = (self.pitch - self.pad_size) // 2

            for i in range(self.n_pads):
                for j in range(self.n_pads):
                    r_start = offset_row + i * self.pitch + pad_offset_within_cell
                    c_start = offset_col + j * self.pitch + pad_offset_within_cell
                    r_end = r_start + self.pad_size
                    c_end = c_start + self.pad_size

                    # Clip to object boundaries
                    r_start_c = max(r_start, 0)
                    c_start_c = max(c_start, 0)
                    r_end_c = min(r_end, obj_h)
                    c_end_c = min(c_end, obj_w)

                    if r_start_c < r_end_c and c_start_c < c_end_c:
                        mask[r_start_c:r_end_c, c_start_c:c_end_c] = 1.0

            return mask

    def _estimate_adaptive_support(self, obj: np.ndarray) -> np.ndarray:
            """
            Estimate support mask from the amplitude of a preliminary reconstruction.

            Parameters
            ----------
            obj : np.ndarray
                Complex object array of shape object_shape.

            Returns
            -------
            np.ndarray
                Binary support mask of shape object_shape.
            """
            amplitude = np.abs(obj)

            # Compute the threshold based on the given quantile
            # Regions with amplitude significantly different from the median (substrate)
            # are likely the pads (support regions)
            median_amp = np.median(amplitude)

            # Compute deviation from median
            deviation = np.abs(amplitude - median_amp)

            # Threshold: pixels whose amplitude deviates more than the quantile threshold
            # of the deviation distribution are considered support
            threshold = np.quantile(deviation, self.support_threshold_quantile)

            # Support is where deviation exceeds the threshold
            mask = (deviation > threshold).astype(np.float64)

            # Apply morphological operations to clean up the mask
            # Use a simple approach: smooth and re-threshold
            from scipy.ndimage import binary_opening, binary_closing, uniform_filter

            # Clean up with morphological operations
            struct = np.ones((3, 3), dtype=bool)
            mask_bool = mask.astype(bool)
            mask_bool = binary_opening(mask_bool, structure=struct, iterations=2)
            mask_bool = binary_closing(mask_bool, structure=struct, iterations=2)

            return mask_bool.astype(np.float64)

    def _estimate_Rp_from_reconstruction(self, obj: np.ndarray, support: np.ndarray) -> complex:
            """
            Estimate substrate reflectivity from median of object values outside support.

            Parameters
            ----------
            obj : np.ndarray
                Complex object of shape object_shape.
            support : np.ndarray
                Binary support mask.

            Returns
            -------
            complex
                Estimated R_p.
            """
            outside_mask = (support < 0.5)
            if np.any(outside_mask):
                outside_values = obj[outside_mask]
                # Use median of real and imaginary parts separately for robustness
                R_p = complex(np.median(outside_values.real), np.median(outside_values.imag))
            else:
                # Fallback to physics-based estimate
                R_p = self._compute_substrate_reflectivity()
            return R_p

    def forward(self, obj_patch: np.ndarray, probe: np.ndarray) -> np.ndarray:
            """
            Compute far-field diffraction amplitude from an object patch and probe.

            Forward model: FFT of (probe * object_patch).

            Parameters
            ----------
            obj_patch : np.ndarray
                Complex object patch of shape probe_shape.
            probe : np.ndarray
                Complex probe of shape probe_shape.

            Returns
            -------
            np.ndarray
                Complex far-field wavefield of shape probe_shape.
            """
            exit_wave = probe * obj_patch
            psi_hat = fft2(exit_wave)
            return psi_hat

    def _modulus_replacement(
            self, psi_hat: np.ndarray, sqrt_I: np.ndarray
        ) -> np.ndarray:
            """
            Replace the amplitude of the Fourier-domain exit wave with measured sqrt intensity.

            Parameters
            ----------
            psi_hat : np.ndarray
                Complex Fourier-domain exit wave of shape probe_shape.
            sqrt_I : np.ndarray
                Square root of measured intensity of shape probe_shape.

            Returns
            -------
            np.ndarray
                Corrected Fourier-domain exit wave with measured amplitude.
            """
            amp = np.abs(psi_hat) + self.eps
            psi_hat_c = sqrt_I * psi_hat / amp
            return psi_hat_c

    def _backward(self, psi_hat_c: np.ndarray) -> np.ndarray:
            """
            Inverse Fourier transform to recover corrected exit wave in real space.

            Parameters
            ----------
            psi_hat_c : np.ndarray
                Corrected Fourier-domain exit wave of shape probe_shape.

            Returns
            -------
            np.ndarray
                Corrected real-space exit wave of shape probe_shape.
            """
            return ifft2(psi_hat_c)

    def _epie_object_update(
            self,
            obj_patch: np.ndarray,
            probe: np.ndarray,
            psi: np.ndarray,
            psi_c: np.ndarray,
        ) -> np.ndarray:
            P_abs = np.abs(probe)
            P_max = self.P_max_abs if self.P_max_abs is not None else P_abs.max()

            weight = P_abs / (P_max + self.eps)
            numerator = np.conj(probe) * (psi_c - psi)
            denominator = P_abs**2 + self.alpha

            obj_patch_new = obj_patch + self.beta * weight * numerator / denominator
            return obj_patch_new

    def _apply_reflectivity_prior(
            self,
            obj_patch: np.ndarray,
            support_patch: np.ndarray,
            R_p: complex,
            beta: float,
        ) -> np.ndarray:
            """
            Enforce reflectivity prior outside support (Eq. 6, Step 2).

            Outside support: O = beta * O + (1 - beta) * R_p

            Parameters
            ----------
            obj_patch : np.ndarray
                Updated object patch of shape probe_shape.
            support_patch : np.ndarray
                Support mask patch of shape probe_shape.
            R_p : complex
                Known substrate reflectivity.
            beta : float
                Feedback constant.

            Returns
            -------
            np.ndarray
                Object patch with reflectivity prior enforced outside support.
            """
            outside = (support_patch < 0.5)
            result = obj_patch.copy()
            result[outside] = beta * obj_patch[outside] + (1.0 - beta) * R_p
            return result

    def _compute_fourier_error(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            scan_positions: np.ndarray,
            sqrt_intensities: np.ndarray,
        ) -> float:
            """
            Compute normalized Fourier error metric across all scan positions.

            E = sum_j sum_k (sqrt(I_j) - |F{P * O_patch_j}|)^2 / sum_j sum_k I_j

            Parameters
            ----------
            obj : np.ndarray
                Current complex object of shape object_shape.
            probe : np.ndarray
                Complex probe of shape probe_shape.
            scan_positions : np.ndarray
                Scan positions of shape (N, 2).
            sqrt_intensities : np.ndarray
                Square root of measured intensities, shape (N, H, W).

            Returns
            -------
            float
                Normalized Fourier error.
            """
            n_positions = scan_positions.shape[0]
            probe_h, probe_w = self.probe_shape
            numerator_sum = 0.0
            denominator_sum = 0.0

            for j in range(n_positions):
                r, c = scan_positions[j]
                obj_patch = obj[r:r + probe_h, c:c + probe_w]
                psi_hat = fft2(probe * obj_patch)
                sim_amp = np.abs(psi_hat)
                meas_amp = sqrt_intensities[j]

                numerator_sum += np.sum((meas_amp - sim_amp) ** 2)
                denominator_sum += np.sum(meas_amp ** 2)

            if denominator_sum < self.eps:
                return 0.0

            return float(numerator_sum / denominator_sum)

    def _run_epie_phase(
            self,
            obj: np.ndarray,
            probe: np.ndarray,
            scan_positions: np.ndarray,
            sqrt_intensities: np.ndarray,
            n_epochs: int,
            beta: float,
            apply_prior: bool,
            support: Optional[np.ndarray],
            R_p: Optional[complex],
        ) -> np.ndarray:
            """
            Run ePIE or rpPIE iterations for a given number of epochs.

            Parameters
            ----------
            obj : np.ndarray
                Initial complex object of shape object_shape.
            probe : np.ndarray
                Complex probe of shape probe_shape.
            scan_positions : np.ndarray
                Scan positions of shape (N, 2).
            sqrt_intensities : np.ndarray
                Square root of measured intensities, shape (N, H, W).
            n_epochs : int
                Number of epochs to run.
            beta : float
                Feedback constant.
            apply_prior : bool
                Whether to apply reflectivity prior (rpPIE vs standard ePIE).
            support : Optional[np.ndarray]
                Binary support mask, required if apply_prior is True.
            R_p : Optional[complex]
                Substrate reflectivity, required if apply_prior is True.

            Returns
            -------
            np.ndarray
                Updated complex object of shape object_shape.
            """
            n_positions = scan_positions.shape[0]
            probe_h, probe_w = self.probe_shape

            for epoch in range(n_epochs):
                # Shuffle scan positions each epoch
                order = np.random.permutation(n_positions)

                for idx in order:
                    r, c = int(scan_positions[idx, 0]), int(scan_positions[idx, 1])

                    # Extract object patch
                    obj_patch = obj[r:r + probe_h, c:c + probe_w].copy()

                    # Forward: exit wave
                    psi = probe * obj_patch

                    # Forward propagation to Fourier domain
                    psi_hat = self.forward(obj_patch, probe)

                    # Modulus replacement
                    psi_hat_c = self._modulus_replacement(psi_hat, sqrt_intensities[idx])

                    # Backward propagation
                    psi_c = self._backward(psi_hat_c)

                    # ePIE object update
                    obj_patch_new = self._epie_object_update(obj_patch, probe, psi, psi_c)

                    # Optionally apply reflectivity prior
                    if apply_prior and support is not None and R_p is not None:
                        support_patch = support[r:r + probe_h, c:c + probe_w]
                        obj_patch_new = self._apply_reflectivity_prior(
                            obj_patch_new, support_patch, R_p, beta
                        )

                    # Write back to object
                    obj[r:r + probe_h, c:c + probe_w] = obj_patch_new

            return obj

    def _format_output(self, obj: np.ndarray) -> np.ndarray:
            """
            Format the complex object into a 2-channel output array.

            Parameters
            ----------
            obj : np.ndarray
                Complex object of shape object_shape.

            Returns
            -------
            np.ndarray
                Output array of shape (2, H, W) — either (real, imag) or (amp, phase).
            """
            h, w = obj.shape
            result = np.zeros((2, h, w), dtype=np.float64)

            if self.output_mode == "amp_phase":
                result[0] = np.abs(obj)
                result[1] = np.angle(obj)
            else:
                # Default: real_imag
                result[0] = np.real(obj)
                result[1] = np.imag(obj)

            return result

    def solve(
            self,
            intensities: np.ndarray,
            probe_data: np.ndarray,
            scan_positions: np.ndarray,
        ) -> np.ndarray:
            """
            Full rpPIE reconstruction pipeline.

            Phase 1: Standard ePIE (no prior) for initial estimate.
            Phase 2: rpPIE with adaptive support and reflectivity prior.

            Parameters
            ----------
            intensities : np.ndarray
                Measured diffraction intensities of shape (N, H, W).
            probe_data : np.ndarray
                Probe function of shape (2, H, W) — real and imaginary channels.
            scan_positions : np.ndarray
                Scan positions of shape (N, 2) as (row, col).

            Returns
            -------
            np.ndarray
                Reconstructed object of shape (2, obj_H, obj_W).
            """
            # Step 1: Parse inputs
            # Parse probe: (2, H, W) -> complex (H, W)
            self.probe = self._parse_probe(probe_data)
            probe_h, probe_w = self.probe.shape

            # Parse scan positions: normalize and center within object grid
            self.scan_positions = self._parse_scan_positions(scan_positions)

            # Precompute sqrt of intensities (ensure non-negative)
            safe_intensities = np.maximum(intensities, 0.0)
            self.sqrt_intensities = np.sqrt(safe_intensities)

            # Compute probe-related constants
            P_abs = np.abs(self.probe)
            self.P_max_abs = float(P_abs.max())
            self.alpha = self.alpha_fraction * (self.P_max_abs ** 2)

            # Step 2: Initialize object as uniform complex ones
            obj_h, obj_w = self.object_shape
            self.obj = np.ones((obj_h, obj_w), dtype=np.complex128)

            # Step 3: Phase 1 — Standard ePIE (no prior)
            print(f"Phase 1: Running ePIE for {self.n_epochs_phase1} epochs...", flush=True)
            self.obj = self._run_epie_phase(
                obj=self.obj,
                probe=self.probe,
                scan_positions=self.scan_positions,
                sqrt_intensities=self.sqrt_intensities,
                n_epochs=self.n_epochs_phase1,
                beta=self.beta,
                apply_prior=False,
                support=None,
                R_p=None,
            )

            # Compute Fourier error after Phase 1
            err1 = self._compute_fourier_error(
                self.obj, self.probe, self.scan_positions, self.sqrt_intensities
            )
            print(f"Phase 1 complete. Fourier error: {err1:.6f}", flush=True)

            # Step 4: Estimate support and R_p from Phase 1 result
            print("Estimating adaptive support mask...", flush=True)
            self.support = self._estimate_adaptive_support(self.obj)
            self.R_p = self._estimate_Rp_from_reconstruction(self.obj, self.support)
            print(f"Estimated R_p: {self.R_p:.6f}", flush=True)

            # Step 5: Phase 2 — rpPIE with reflectivity prior
            print(f"Phase 2: Running rpPIE for {self.n_epochs_phase2} epochs...", flush=True)
            self.obj = self._run_epie_phase(
                obj=self.obj,
                probe=self.probe,
                scan_positions=self.scan_positions,
                sqrt_intensities=self.sqrt_intensities,
                n_epochs=self.n_epochs_phase2,
                beta=self.beta,
                apply_prior=True,
                support=self.support,
                R_p=self.R_p,
            )

            # Compute Fourier error after Phase 2
            err2 = self._compute_fourier_error(
                self.obj, self.probe, self.scan_positions, self.sqrt_intensities
            )
            print(f"Phase 2 complete. Fourier error: {err2:.6f}", flush=True)

            # Step 6: Format and return output
            result = self._format_output(self.obj)

            # Verify shape
            assert result.shape == (2, obj_h, obj_w), \
                f"Output shape mismatch: expected (2, {obj_h}, {obj_w}), got {result.shape}"

            return result



if __name__ == "__main__":
    import sys
    import traceback

    try:
        # Load measured diffraction intensity data
        input_data = np.load("dataset/input.npy")  # shape: (81, 256, 256)

        # Load probe function (2-channel: real, imaginary)
        probe_data = np.load("dataset/probe.npy")  # shape: (2, 256, 256)

        # Load scan positions
        scan_positions = np.load("dataset/scan_positions.npy")  # shape: (81, 2)

        print(f"Input data shape: {input_data.shape}", flush=True)
        print(f"Probe data shape: {probe_data.shape}", flush=True)
        print(f"Scan positions shape: {scan_positions.shape}", flush=True)
        print(f"Scan positions range: rows [{scan_positions[:, 0].min()}, {scan_positions[:, 0].max()}], cols [{scan_positions[:, 1].min()}, {scan_positions[:, 1].max()}]", flush=True)

        # Handle probe shape - it might be (2, 256, 256) or something else
        if probe_data.ndim == 3 and probe_data.shape[0] == 2:
            probe_shape = (probe_data.shape[1], probe_data.shape[2])
        elif probe_data.ndim == 2:
            # Single real-valued probe
            probe_shape = probe_data.shape
        else:
            probe_shape = (256, 256)

        print(f"Detected probe shape: {probe_shape}", flush=True)

        # Instantiate solver with default hyperparameters
        solver = InverseSolver(
            object_shape=(600, 600),
            probe_shape=probe_shape,
            n_epochs_phase1=50,
            n_epochs_phase2=150,
            beta=0.9,
            alpha_fraction=1e-2,
            eps=1e-10,
            output_mode="real_imag",
        )

        # Handle probe parsing manually if needed
        if probe_data.ndim == 2:
            # Real-valued probe, wrap it
            probe_wrapped = np.zeros((2, probe_data.shape[0], probe_data.shape[1]), dtype=probe_data.dtype)
            probe_wrapped[0] = probe_data
            probe_data = probe_wrapped
        elif probe_data.ndim == 3 and probe_data.shape[0] != 2:
            # Unexpected shape, try to handle
            print(f"Warning: unexpected probe shape {probe_data.shape}, attempting to use first two channels", flush=True)
            if probe_data.shape[0] >= 2:
                probe_data = probe_data[:2]
            else:
                probe_wrapped = np.zeros((2, probe_data.shape[1], probe_data.shape[2]), dtype=probe_data.dtype)
                probe_wrapped[0] = probe_data[0]
                probe_data = probe_wrapped

        # Run reconstruction
        result = solver.solve(
            intensities=input_data,
            probe_data=probe_data,
            scan_positions=scan_positions,
        )

        # Verify output shape
        assert result.shape == (2, 600, 600), f"Unexpected output shape: {result.shape}"

        print(f"Result shape: {result.shape}", flush=True)
        print(f"Result channel 0 range: [{result[0].min():.6f}, {result[0].max():.6f}]", flush=True)
        print(f"Result channel 1 range: [{result[1].min():.6f}, {result[1].max():.6f}]", flush=True)

        # Save result
        np.save("output.npy", result)
        print("Output saved to output.npy", flush=True)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

        # Save a fallback output so we don't fail silently
        fallback = np.zeros((2, 600, 600), dtype=np.float64)
        np.save("output.npy", fallback)
        print("Saved fallback zero output.", flush=True)
        sys.exit(1)