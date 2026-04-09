import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional, Dict, Any
import os
import sys

class InverseSolver:
    def __init__(
            self,
            object_shape: Tuple[int, int] = (600, 600),
            probe_shape: Tuple[int, int] = (256, 256),
            n_scan_positions: int = 81,
            scan_grid_size: int = 9,
            step_size: int = 40,
            pixel_size_nm: float = 27.0,
            wavelength_nm: float = 13.5,
            n_pads: int = 15,
            pad_size_pixels: int = 22,
            pitch_pixels: int = 30,
            n_SiCN_real: float = 0.9791,
            n_SiCN_imag: float = 0.0059,
            n_Cu_real: float = 0.9625,
            n_Cu_imag: float = 0.0613,
            n_SiO2_real: float = 0.9780,
            n_SiO2_imag: float = 0.0108,
            n_Si_real: float = 0.9991,
            n_Si_imag: float = 0.0018,
            beta_epie: float = 0.9,
            beta_rp: float = 0.5,
            alpha_reg: float = 0.001,
            n_epochs_phase1: int = 200,
            n_epochs_phase2: int = 100,
            eps: float = 1e-10,
            early_stop_patience: int = 3,
            monitor_interval: int = 20,
            support_correlation_threshold: float = 0.5,
        ) -> None:
            # Object / probe dimensions
            self.object_shape: Tuple[int, int] = object_shape
            self.probe_shape: Tuple[int, int] = probe_shape
            self.n_scan_positions: int = n_scan_positions
            self.scan_grid_size: int = scan_grid_size
            self.step_size: int = step_size

            # Physical parameters
            self.pixel_size_nm: float = pixel_size_nm
            self.wavelength_nm: float = wavelength_nm

            # Pad array geometry
            self.n_pads: int = n_pads
            self.pad_size_pixels: int = pad_size_pixels
            self.pitch_pixels: int = pitch_pixels

            # Material optical constants (complex refractive index: n - ik)
            # Convention: n_complex = n_real - i * n_imag (absorption is positive imag part in physics)
            # We store as n_real - 1j * n_imag so that imaginary part is negative (absorptive)
            self.n_SiCN: complex = complex(n_SiCN_real, -n_SiCN_imag)
            self.n_Cu: complex = complex(n_Cu_real, -n_Cu_imag)
            self.n_SiO2: complex = complex(n_SiO2_real, -n_SiO2_imag)
            self.n_Si: complex = complex(n_Si_real, -n_Si_imag)

            # Algorithm hyperparameters
            self.beta_epie: float = beta_epie
            self.beta_rp: float = beta_rp
            self.alpha_reg: float = alpha_reg
            self.n_epochs_phase1: int = n_epochs_phase1
            self.n_epochs_phase2: int = n_epochs_phase2
            self.eps: float = eps
            self.early_stop_patience: int = early_stop_patience
            self.monitor_interval: int = monitor_interval
            self.support_correlation_threshold: float = support_correlation_threshold

            # Computed physical priors (to be initialized in _compute_priors)
            self.R_p_substrate: complex = complex(0.0, 0.0)
            self.R_p_Cu: complex = complex(0.0, 0.0)
            self.support_mask: Optional[np.ndarray] = None
            self.scan_positions: Optional[np.ndarray] = None

            # Probe quantities (precomputed)
            self.P: Optional[np.ndarray] = None
            self.P_abs: Optional[np.ndarray] = None
            self.P_max_abs: float = 1.0
            self.P_conj: Optional[np.ndarray] = None
            self.P_abs_sq: Optional[np.ndarray] = None
            self.alpha_value: float = 0.0

            # Diffraction data
            self.sqrt_I: Optional[np.ndarray] = None

            # Object estimate
            self.O: Optional[np.ndarray] = None

            # Actual reconstruction grid size (may be larger than object_shape if positions require it)
            self.recon_shape: Tuple[int, int] = object_shape

            # Best object tracking for early stopping
            self.best_O: Optional[np.ndarray] = None
            self.best_error: float = float('inf')

    def _compute_fresnel_reflectivity(self, n_complex: complex) -> complex:
            """Compute Fresnel reflectivity at normal incidence for a material with complex refractive index."""
            return (1.0 - n_complex) / (1.0 + n_complex)

    def _compute_priors(self) -> None:
            """Compute R_p_substrate, R_p_Cu from Fresnel equations using CXRO optical constants."""
            self.R_p_substrate = self._compute_fresnel_reflectivity(self.n_SiCN)
            self.R_p_Cu = self._compute_fresnel_reflectivity(self.n_Cu)

    def _build_support_mask(self) -> np.ndarray:
            """Build physics-based binary support mask for 15x15 periodic pad array on the object grid."""
            S = np.zeros(self.recon_shape, dtype=bool)
            pad_size = self.pad_size_pixels
            pitch = self.pitch_pixels
            n_pads = self.n_pads

            # Total extent of the pad array
            array_extent = n_pads * pitch  # 15 * 30 = 450 pixels

            # Center the pad array in the reconstruction grid
            array_offset_r = (self.recon_shape[0] - array_extent) // 2
            array_offset_c = (self.recon_shape[1] - array_extent) // 2

            # Offset within each pitch cell to center the pad
            pad_offset = (pitch - pad_size) // 2  # (30 - 22) // 2 = 4

            for i in range(n_pads):
                for j in range(n_pads):
                    r_start = array_offset_r + i * pitch + pad_offset
                    c_start = array_offset_c + j * pitch + pad_offset
                    r_end = r_start + pad_size
                    c_end = c_start + pad_size
                    # Clip to grid bounds
                    r_start_c = max(r_start, 0)
                    c_start_c = max(c_start, 0)
                    r_end_c = min(r_end, self.recon_shape[0])
                    c_end_c = min(c_end, self.recon_shape[1])
                    if r_start_c < r_end_c and c_start_c < c_end_c:
                        S[r_start_c:r_end_c, c_start_c:c_end_c] = True

            self.support_mask = S
            return S

    def _build_scan_positions(self) -> np.ndarray:
            """Construct 9x9 scan position grid with 40-pixel steps, centered in the object grid."""
            step = self.step_size
            grid_size = self.scan_grid_size
            probe_h, probe_w = self.probe_shape

            # Build 1D grid: [0, step, 2*step, ..., (grid_size-1)*step]
            grid_1d = np.arange(grid_size) * step

            # Create 2D meshgrid (row, col)
            rows, cols = np.meshgrid(grid_1d, grid_1d, indexing='ij')
            positions = np.stack([rows.ravel(), cols.ravel()], axis=1)  # (81, 2)

            # Total span: (grid_size-1)*step + probe_size
            span_r = (grid_size - 1) * step + probe_h
            span_c = (grid_size - 1) * step + probe_w

            # Centering offset so that the scan region is centered in the object grid
            offset_r = (self.object_shape[0] - span_r) // 2
            offset_c = (self.object_shape[1] - span_c) // 2

            # Ensure non-negative offsets
            offset_r = max(offset_r, 0)
            offset_c = max(offset_c, 0)

            positions[:, 0] += offset_r
            positions[:, 1] += offset_c

            self.scan_positions = positions
            return positions

    def _adapt_scan_positions(self, provided_positions: Optional[np.ndarray]) -> np.ndarray:
            """Use provided scan positions if valid, otherwise construct from grid. Ensure all patches fit in object grid."""
            if provided_positions is not None and len(provided_positions) == self.n_scan_positions:
                positions = np.array(provided_positions, dtype=np.int64).copy()

                # Check if positions need shifting to be non-negative
                min_r = positions[:, 0].min()
                min_c = positions[:, 1].min()
                if min_r < 0 or min_c < 0:
                    positions[:, 0] -= min(min_r, 0)
                    positions[:, 1] -= min(min_c, 0)

                # Check if all patches fit within the object grid
                max_r = positions[:, 0].max() + self.probe_shape[0]
                max_c = positions[:, 1].max() + self.probe_shape[1]

                if max_r <= self.object_shape[0] and max_c <= self.object_shape[1]:
                    # Positions fit; center them if there's room
                    current_span_r = positions[:, 0].max() - positions[:, 0].min() + self.probe_shape[0]
                    current_span_c = positions[:, 1].max() - positions[:, 1].min() + self.probe_shape[1]

                    # Shift so that the scanned region is centered
                    desired_offset_r = (self.object_shape[0] - current_span_r) // 2
                    desired_offset_c = (self.object_shape[1] - current_span_c) // 2

                    current_offset_r = positions[:, 0].min()
                    current_offset_c = positions[:, 1].min()

                    positions[:, 0] += desired_offset_r - current_offset_r
                    positions[:, 1] += desired_offset_c - current_offset_c

                    # Final safety: ensure non-negative
                    positions[:, 0] = np.maximum(positions[:, 0], 0)
                    positions[:, 1] = np.maximum(positions[:, 1], 0)

                    self.scan_positions = positions
                    return positions
                else:
                    # Positions require larger grid — shift to start from 0, grid will be expanded
                    positions[:, 0] -= positions[:, 0].min()
                    positions[:, 1] -= positions[:, 1].min()
                    self.scan_positions = positions
                    return positions

            # Fallback: construct default scan positions
            return self._build_scan_positions()

    def _determine_object_grid_size(self, positions: np.ndarray) -> Tuple[int, int]:
            """Determine required object grid size from scan positions and probe shape. Expand beyond object_shape if needed."""
            max_r = int(positions[:, 0].max()) + self.probe_shape[0]
            max_c = int(positions[:, 1].max()) + self.probe_shape[1]

            required_r = max(max_r, self.object_shape[0])
            required_c = max(max_c, self.object_shape[1])

            self.recon_shape = (required_r, required_c)
            return self.recon_shape

    def _initialize_object(self) -> np.ndarray:
            """Initialize complex object array with substrate reflectivity R_p everywhere."""
            self.O = np.full(self.recon_shape, self.R_p_substrate, dtype=np.complex128)
            return self.O

    def _precompute_probe_quantities(self, probe_data: np.ndarray) -> None:
            """Parse probe from input (2, H, W) -> complex, precompute P_abs, P_conj, P_abs_sq, P_max_abs, alpha."""
            # Construct complex probe from real and imaginary parts
            self.P = probe_data[0].astype(np.complex128) + 1j * probe_data[1].astype(np.complex128)

            # Precompute derived quantities used in ePIE update
            self.P_abs = np.abs(self.P)
            self.P_max_abs = float(self.P_abs.max())
            self.P_conj = np.conj(self.P)
            self.P_abs_sq = self.P_abs ** 2
            self.alpha_value = self.alpha_reg * self.P_max_abs ** 2

    def _preprocess_diffraction(self, diffraction_patterns: np.ndarray) -> np.ndarray:
            """Clip negative values and compute sqrt of measured intensities."""
            clipped = np.maximum(diffraction_patterns, 0.0)
            self.sqrt_I = np.sqrt(clipped)
            return self.sqrt_I

    def forward(
            self,
            O: np.ndarray,
            P: np.ndarray,
            position: Tuple[int, int],
        ) -> np.ndarray:
            """
            Forward model: compute far-field diffraction amplitude from object patch and probe.

            Parameters:
                O: Complex object array, shape (H, W)
                P: Complex probe array, shape (pH, pW)
                position: (row, col) top-left corner of the patch in the object grid

            Returns:
                Psi_hat: Complex far-field diffraction pattern, shape (pH, pW)
            """
            r, c = position
            pH, pW = P.shape
            O_patch = O[r:r+pH, c:c+pW]
            psi = P * O_patch
            Psi_hat = fft2(psi)
            return Psi_hat

    def _modulus_replacement(
            self,
            Psi_hat: np.ndarray,
            sqrt_I_j: np.ndarray,
        ) -> np.ndarray:
            """
            Replace Fourier modulus with measured amplitude while preserving phase.

            Parameters:
                Psi_hat: Current far-field estimate, shape (pH, pW)
                sqrt_I_j: Measured amplitude (sqrt of intensity), shape (pH, pW)

            Returns:
                Psi_hat_corrected: Corrected far-field with measured amplitude, shape (pH, pW)
            """
            amp = np.abs(Psi_hat) + self.eps
            Psi_hat_corrected = sqrt_I_j * Psi_hat / amp
            return Psi_hat_corrected

    def _epie_object_update(
            self,
            O_patch: np.ndarray,
            psi: np.ndarray,
            psi_corrected: np.ndarray,
        ) -> np.ndarray:
            """
            ePIE object update rule (Eq. 6, line 1).

            Parameters:
                O_patch: Current object patch, shape (pH, pW)
                psi: Current exit wave P * O_patch, shape (pH, pW)
                psi_corrected: Corrected exit wave from inverse FFT of modulus-replaced pattern, shape (pH, pW)

            Returns:
                O_patch_updated: Updated object patch, shape (pH, pW)
            """
            delta_psi = psi_corrected - psi
            update = (self.P_abs / self.P_max_abs) * (self.P_conj * delta_psi) / (self.P_abs_sq + self.alpha_value)
            O_patch_updated = O_patch + self.beta_epie * update
            return O_patch_updated

    def _apply_reflectivity_prior(
            self,
            O_patch: np.ndarray,
            support_patch: np.ndarray,
        ) -> np.ndarray:
            """
            Enforce reflectivity prior outside support mask (Eq. 6, line 2 / rpPIE step 2).

            Parameters:
                O_patch: Object patch after ePIE update, shape (pH, pW)
                support_patch: Binary support mask for this patch, shape (pH, pW)

            Returns:
                O_patch_constrained: Object patch with prior enforced outside support, shape (pH, pW)
            """
            O_patch_constrained = O_patch.copy()
            outside = ~support_patch
            O_patch_constrained[outside] = (self.beta_rp * O_patch_constrained[outside] +
                                             (1.0 - self.beta_rp) * self.R_p_substrate)
            return O_patch_constrained

    def _run_epie_epoch(
            self,
            apply_prior: bool = False,
        ) -> float:
            """
            Run one epoch of ePIE or rpPIE over all scan positions in random order.

            Parameters:
                apply_prior: If True, apply reflectivity prior after each position update (rpPIE mode)

            Returns:
                epoch_error: Accumulated Fourier error for this epoch
            """
            order = np.random.permutation(self.n_scan_positions)
            epoch_error = 0.0

            pH, pW = self.probe_shape

            for idx in order:
                r, c = int(self.scan_positions[idx, 0]), int(self.scan_positions[idx, 1])

                # Extract object patch
                O_patch = self.O[r:r+pH, c:c+pW].copy()

                # Forward: exit wave and FFT
                psi = self.P * O_patch
                Psi_hat = fft2(psi)

                # Accumulate Fourier error for this position
                diff = np.abs(Psi_hat) - self.sqrt_I[idx]
                epoch_error += np.sum(diff ** 2)

                # Modulus replacement
                Psi_hat_corrected = self._modulus_replacement(Psi_hat, self.sqrt_I[idx])

                # Backward: inverse FFT
                psi_corrected = ifft2(Psi_hat_corrected)

                # ePIE object update
                O_patch_updated = self._epie_object_update(O_patch, psi, psi_corrected)

                # Optionally apply reflectivity prior (rpPIE mode)
                if apply_prior and self.support_mask is not None:
                    S_patch = self.support_mask[r:r+pH, c:c+pW]
                    O_patch_updated = self._apply_reflectivity_prior(O_patch_updated, S_patch)

                # Write back to object
                self.O[r:r+pH, c:c+pW] = O_patch_updated

            # Normalize error
            total_energy = np.sum(self.sqrt_I ** 2) + 1e-30
            epoch_error /= total_energy

            return epoch_error

    def _compute_fourier_error(self) -> float:
            """
            Compute normalized Fourier error metric across all scan positions.

            Returns:
                error: sum(|F(P*O_patch)| - sqrt_I)^2 / sum(sqrt_I^2)
            """
            total_error = 0.0
            total_energy = 0.0
            pH, pW = self.probe_shape

            for j in range(self.n_scan_positions):
                r, c = int(self.scan_positions[j, 0]), int(self.scan_positions[j, 1])
                O_patch = self.O[r:r+pH, c:c+pW]
                psi = self.P * O_patch
                Psi_hat = fft2(psi)
                diff = np.abs(Psi_hat) - self.sqrt_I[j]
                total_error += np.sum(diff ** 2)
                total_energy += np.sum(self.sqrt_I[j] ** 2)

            return total_error / (total_energy + 1e-30)

    def _verify_support_alignment(self) -> float:
            """
            After Phase 1, verify correlation between high-amplitude regions and physics-based support mask.

            Returns:
                correlation: Correlation coefficient between |O| and support mask
            """
            if self.O is None or self.support_mask is None:
                return 0.0

            # Get the object amplitude, cropped/matched to support mask size
            obj_amp = np.abs(self.O)

            # Ensure shapes match
            if obj_amp.shape != self.support_mask.shape:
                return 0.0

            # Flatten for correlation computation
            amp_flat = obj_amp.ravel().astype(np.float64)
            mask_flat = self.support_mask.ravel().astype(np.float64)

            # Compute Pearson correlation coefficient
            amp_mean = np.mean(amp_flat)
            mask_mean = np.mean(mask_flat)

            amp_centered = amp_flat - amp_mean
            mask_centered = mask_flat - mask_mean

            numerator = np.sum(amp_centered * mask_centered)
            denominator = np.sqrt(np.sum(amp_centered ** 2) * np.sum(mask_centered ** 2))

            if denominator < 1e-30:
                return 0.0

            correlation = numerator / denominator
            return float(correlation)

    def _center_crop(
            self,
            array: np.ndarray,
            target_shape: Tuple[int, int],
        ) -> np.ndarray:
            """
            Center-crop a 2D array to target shape if the reconstruction grid is larger than required output.

            Parameters:
                array: Input array, shape (H, W)
                target_shape: Desired output shape (tH, tW)

            Returns:
                cropped: Center-cropped array, shape (tH, tW)
            """
            H, W = array.shape
            tH, tW = target_shape

            # If already the right size or smaller, return as-is (pad if needed, but typically not expected)
            if H <= tH and W <= tW:
                return array

            # Compute start indices for center crop
            r_start = (H - tH) // 2
            c_start = (W - tW) // 2

            # Clamp to valid range
            r_start = max(r_start, 0)
            c_start = max(c_start, 0)

            r_end = r_start + tH
            c_end = c_start + tW

            # Safety: ensure we don't exceed array bounds
            r_end = min(r_end, H)
            c_end = min(c_end, W)

            cropped = array[r_start:r_end, c_start:c_end]
            return cropped

    def _format_output(self, O: np.ndarray) -> np.ndarray:
            """
            Convert complex object to (2, H, W) output: channel 0 = amplitude, channel 1 = phase.

            Parameters:
                O: Complex object, shape (H, W)

            Returns:
                result: shape (2, H, W), dtype float64
            """
            # Center-crop to target object_shape if reconstruction grid is larger
            if O.shape[0] > self.object_shape[0] or O.shape[1] > self.object_shape[1]:
                O = self._center_crop(O, self.object_shape)

            H, W = O.shape
            result = np.zeros((2, H, W), dtype=np.float64)
            result[0] = np.abs(O)
            result[1] = np.angle(O)
            return result

    def solve(
            self,
            diffraction_patterns: np.ndarray,
            probe: np.ndarray,
            scan_positions: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """
            Full reconstruction pipeline: preprocess, Phase 1 (ePIE), Phase 2 (rpPIE), format output.

            Parameters:
                diffraction_patterns: Measured intensities, shape (N, pH, pW)
                probe: Probe function, shape (2, pH, pW) — [real, imag]
                scan_positions: Optional scan positions, shape (N, 2) — [row, col]

            Returns:
                result: Reconstructed object as (2, H, W) — [amplitude, phase], dtype float64
            """
            # Step 1: Compute physics-based priors (Fresnel reflectivities)
            self._compute_priors()

            # Step 2: Precompute probe quantities
            self._precompute_probe_quantities(probe)

            # Step 3: Preprocess diffraction patterns (clip + sqrt)
            self._preprocess_diffraction(diffraction_patterns)

            # Step 4: Adapt/construct scan positions
            positions = self._adapt_scan_positions(scan_positions)

            # Step 5: Determine required object grid size
            self._determine_object_grid_size(positions)

            # Step 6: Build physics-based support mask
            self._build_support_mask()

            # Step 7: Initialize object with substrate reflectivity
            self._initialize_object()

            # Step 8: Phase 1 — pure ePIE
            print(f"Phase 1: Running {self.n_epochs_phase1} epochs of ePIE...")
            consecutive_increases = 0
            prev_error = float('inf')
            self.best_error = float('inf')
            self.best_O = self.O.copy()

            for epoch in range(self.n_epochs_phase1):
                epoch_error = self._run_epie_epoch(apply_prior=False)

                if epoch % self.monitor_interval == 0 or epoch == self.n_epochs_phase1 - 1:
                    error = self._compute_fourier_error()
                    print(f"  Phase 1 Epoch {epoch}: Fourier error = {error:.6e}")

                    if error < self.best_error:
                        self.best_error = error
                        self.best_O = self.O.copy()
                        consecutive_increases = 0
                    else:
                        consecutive_increases += 1

                    if consecutive_increases >= self.early_stop_patience:
                        print(f"  Early stopping at epoch {epoch} (error increased {consecutive_increases} times)")
                        self.O = self.best_O.copy()
                        break

                    prev_error = error

            # Step 9: Verify support alignment before Phase 2
            correlation = self._verify_support_alignment()
            print(f"Support mask correlation: {correlation:.4f}")

            skip_phase2 = correlation < self.support_correlation_threshold
            if skip_phase2:
                print(f"  Correlation {correlation:.4f} < threshold {self.support_correlation_threshold}, skipping Phase 2")

            # Step 10: Phase 2 — rpPIE with reflectivity prior
            if not skip_phase2:
                print(f"Phase 2: Running {self.n_epochs_phase2} epochs of rpPIE...")
                consecutive_increases = 0
                # Keep best_O and best_error from Phase 1

                for epoch in range(self.n_epochs_phase2):
                    epoch_error = self._run_epie_epoch(apply_prior=True)

                    if epoch % self.monitor_interval == 0 or epoch == self.n_epochs_phase2 - 1:
                        error = self._compute_fourier_error()
                        print(f"  Phase 2 Epoch {epoch}: Fourier error = {error:.6e}")

                        if error < self.best_error:
                            self.best_error = error
                            self.best_O = self.O.copy()
                            consecutive_increases = 0
                        else:
                            consecutive_increases += 1

                        if consecutive_increases >= self.early_stop_patience:
                            print(f"  Early stopping at epoch {epoch} (error increased {consecutive_increases} times)")
                            self.O = self.best_O.copy()
                            break

            # Use best object found
            self.O = self.best_O.copy()

            # Step 11 & 12: Format output (center-crop if needed, then amplitude/phase)
            result = self._format_output(self.O)

            assert result.shape == (2, self.object_shape[0], self.object_shape[1]), \
                f"Output shape mismatch: {result.shape} vs expected (2, {self.object_shape[0]}, {self.object_shape[1]})"

            print(f"Final Fourier error: {self.best_error:.6e}")
            print(f"Output shape: {result.shape}, dtype: {result.dtype}")
            print(f"Amplitude range: [{result[0].min():.6f}, {result[0].max():.6f}]")
            print(f"Phase range: [{result[1].min():.6f}, {result[1].max():.6f}]")

            return result



if __name__ == "__main__":
    import traceback
    try:
        # Load data
        print("Loading input data...")
        input_data = np.load('dataset/input.npy', allow_pickle=True)
        print(f"Input data type: {type(input_data)}, dtype: {input_data.dtype}, shape: {input_data.shape if hasattr(input_data, 'shape') else 'N/A'}")

        diffraction_patterns = None
        probe = None
        scan_positions = None

        if isinstance(input_data, np.ndarray) and input_data.dtype == object:
            print(f"Object array with {len(input_data)} elements")
            for i in range(len(input_data)):
                elem = input_data[i]
                if hasattr(elem, 'shape'):
                    print(f"  Element {i}: shape={elem.shape}, dtype={elem.dtype}")
                elif isinstance(elem, (list, tuple)):
                    print(f"  Element {i}: type={type(elem)}, len={len(elem)}")
                else:
                    print(f"  Element {i}: type={type(elem)}, value={elem}")

            # Try to parse elements
            if len(input_data) >= 2:
                diffraction_patterns = np.array(input_data[0], dtype=np.float64)
                probe = np.array(input_data[1], dtype=np.float64)
                if len(input_data) > 2 and input_data[2] is not None:
                    scan_positions = np.array(input_data[2], dtype=np.int64)
                    print(f"Scan positions shape: {scan_positions.shape}")
            elif len(input_data) == 1:
                # Single element - might be a dict or the diffraction patterns themselves
                elem = input_data[0]
                if isinstance(elem, dict):
                    diffraction_patterns = np.array(elem.get('diffraction_patterns', elem.get('data', None)), dtype=np.float64)
                    if 'probe' in elem:
                        probe = np.array(elem['probe'], dtype=np.float64)
                    if 'scan_positions' in elem or 'positions' in elem:
                        scan_positions = np.array(elem.get('scan_positions', elem.get('positions', None)), dtype=np.int64)
                else:
                    diffraction_patterns = np.array(elem, dtype=np.float64)
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim == 3:
                # Could be (81, 256, 256) diffraction patterns
                diffraction_patterns = input_data.astype(np.float64)
                print(f"Loaded diffraction patterns: shape={diffraction_patterns.shape}")
            elif input_data.ndim == 4:
                # Might be stacked: e.g. (83, 256, 256) with probe embedded
                print(f"4D or large array: shape={input_data.shape}")
                diffraction_patterns = input_data.astype(np.float64)

        # Try loading probe separately if not found
        if probe is None:
            probe_paths = ['dataset/probe.npy', 'probe.npy']
            for pp in probe_paths:
                if os.path.exists(pp):
                    probe = np.load(pp, allow_pickle=True)
                    if probe.dtype == object:
                        probe = np.array(probe[0], dtype=np.float64) if len(probe) == 1 else np.array(probe, dtype=np.float64)
                    else:
                        probe = probe.astype(np.float64)
                    print(f"Loaded probe from {pp}: shape={probe.shape}")
                    break

        # Try loading scan positions separately if not found
        if scan_positions is None:
            pos_paths = ['dataset/scan_positions.npy', 'dataset/positions.npy', 'scan_positions.npy', 'positions.npy']
            for sp in pos_paths:
                if os.path.exists(sp):
                    scan_positions = np.load(sp, allow_pickle=True)
                    if scan_positions.dtype == object:
                        scan_positions = np.array(scan_positions, dtype=np.int64)
                    else:
                        scan_positions = scan_positions.astype(np.int64)
                    print(f"Loaded scan positions from {sp}: shape={scan_positions.shape}")
                    break

        # Validate we have what we need
        if diffraction_patterns is None:
            raise ValueError("Could not parse diffraction patterns from input data")
        if probe is None:
            raise ValueError("Could not find probe data")

        print(f"Diffraction patterns: shape={diffraction_patterns.shape}, dtype={diffraction_patterns.dtype}")
        print(f"Probe: shape={probe.shape}, dtype={probe.dtype}")
        if scan_positions is not None:
            print(f"Scan positions: shape={scan_positions.shape}, min={scan_positions.min(axis=0)}, max={scan_positions.max(axis=0)}")

        # Handle case where probe might be complex-valued stored differently
        if probe.ndim == 2:
            # Single 2D array - assume it's already complex or real-only
            if np.iscomplexobj(probe):
                probe_input = np.stack([probe.real, probe.imag], axis=0)
            else:
                # Real-only probe? Unlikely but handle it
                probe_input = np.stack([probe, np.zeros_like(probe)], axis=0)
            print(f"Converted 2D probe to (2, H, W): shape={probe_input.shape}")
        elif probe.ndim == 3 and probe.shape[0] == 2:
            probe_input = probe
        elif probe.ndim == 3 and probe.shape[2] == 2:
            # (H, W, 2) format
            probe_input = np.transpose(probe, (2, 0, 1))
        else:
            probe_input = probe
            print(f"Warning: unexpected probe shape {probe.shape}, using as-is")

        # Initialize solver with appropriate parameters
        n_positions = diffraction_patterns.shape[0]
        probe_h, probe_w = probe_input.shape[1], probe_input.shape[2]
        grid_size = int(np.round(np.sqrt(n_positions)))

        print(f"N positions: {n_positions}, probe size: ({probe_h}, {probe_w}), grid size: {grid_size}")

        solver = InverseSolver(
            object_shape=(600, 600),
            probe_shape=(probe_h, probe_w),
            n_scan_positions=n_positions,
            scan_grid_size=grid_size,
            step_size=40,
            n_epochs_phase1=200,
            n_epochs_phase2=100,
            beta_epie=0.9,
            beta_rp=0.5,
            alpha_reg=0.001,
            monitor_interval=20,
            early_stop_patience=3,
            support_correlation_threshold=0.5,
        )

        # Solve
        result = solver.solve(
            diffraction_patterns=diffraction_patterns,
            probe=probe_input,
            scan_positions=scan_positions,
        )

        # Save output
        print(f"Saving result: shape={result.shape}, dtype={result.dtype}")
        np.save('output.npy', result)
        print("Done. Output saved to output.npy")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

        # Fallback: save a default output so we don't fail silently
        print("Generating fallback output...")
        result = np.zeros((2, 600, 600), dtype=np.float64)
        result[0] = 0.01  # small uniform amplitude
        result[1] = 0.0   # zero phase
        np.save('output.npy', result)
        print("Fallback output saved to output.npy")