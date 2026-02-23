import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional, Union


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
        """
        Initialize the rpPIE inverse solver.

        Parameters
        ----------
        object_shape : Tuple[int, int]
            Shape of the reconstruction grid (rows, cols).
        probe_shape : Tuple[int, int]
            Shape of the probe function (rows, cols), must match diffraction pattern size.
        n_epochs_phase1 : int
            Number of standard ePIE epochs (no reflectivity prior).
        n_epochs_phase2 : int
            Number of rpPIE epochs (with reflectivity prior).
        beta : float
            Feedback constant for ePIE/rpPIE update rule.
        alpha_fraction : float
            Regularization fraction relative to |P_max|^2.
        eps : float
            Small constant for numerical stability in division.
        step_size : int
            Ptychographic scan step in pixels.
        pad_size : int
            Size of each Cu pad in pixels.
        pitch : int
            Pitch of the periodic pad array in pixels.
        n_pads : int
            Number of pads per row/column (square array).
        wavelength_nm : float
            Illumination wavelength in nm.
        n_SiCN : complex
            Complex refractive index of SiCN at wavelength.
        n_SiO2 : complex
            Complex refractive index of SiO2 at wavelength.
        n_Si : complex
            Complex refractive index of Si at wavelength.
        n_Cu : complex
            Complex refractive index of Cu at wavelength.
        support_threshold_quantile : float
            Quantile for adaptive support estimation from amplitude.
        output_mode : str
            Output encoding: 'real_imag' or 'amp_phase'.
        """
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
        # TODO: Implement conversion from 2-channel to complex
        pass

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
        # TODO: Implement shift/centering so positions + probe_shape fits within object_shape
        pass

    def _compute_substrate_reflectivity(self) -> complex:
        """
        Compute Fresnel reflection coefficient at vacuum/substrate interface.

        Returns
        -------
        complex
            Complex reflectivity R_p for the substrate.
        """
        # TODO: Implement Fresnel reflection R_p = (1 - n_tilde) / (1 + n_tilde)
        pass

    def _build_physics_support_mask(self) -> np.ndarray:
        """
        Build a binary support mask for the 15x15 periodic pad array on the object grid.

        Returns
        -------
        np.ndarray
            Binary mask of shape object_shape, 1 inside pads, 0 outside.
        """
        # TODO: Implement periodic pad geometry on the 600x600 grid
        pass

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
        # TODO: Implement amplitude thresholding at given quantile
        pass

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
        # TODO: Implement median-based R_p estimation from non-support region
        pass

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
        # TODO: Implement exit_wave = probe * obj_patch, then fft2
        pass

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
        # TODO: Implement psi_hat_c = sqrt_I * psi_hat / (|psi_hat| + eps)
        pass

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
        # TODO: Implement ifft2
        pass

    def _epie_object_update(
        self,
        obj_patch: np.ndarray,
        probe: np.ndarray,
        psi: np.ndarray,
        psi_c: np.ndarray,
    ) -> np.ndarray:
        """
        ePIE-style object update rule (Eq. 6, Step 1).

        O_new = O + (|P|/|P_max|) * (P* / (|P|^2 + alpha)) * beta * (psi_c - psi)

        Parameters
        ----------
        obj_patch : np.ndarray
            Current object patch of shape probe_shape.
        probe : np.ndarray
            Complex probe of shape probe_shape.
        psi : np.ndarray
            Current exit wave of shape probe_shape.
        psi_c : np.ndarray
            Corrected exit wave of shape probe_shape.

        Returns
        -------
        np.ndarray
            Updated object patch of shape probe_shape.
        """
        # TODO: Implement ePIE update with weight, conjugate, regularization
        pass

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
        # TODO: Implement convex combination outside support
        pass

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
        # TODO: Implement error computation over all diffraction patterns
        pass

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
        # TODO: Implement epoch loop with shuffled scan order,
        #       calling forward, modulus_replacement, backward,
        #       epie_object_update, and optionally apply_reflectivity_prior
        pass

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
        # TODO: Implement based on self.output_mode
        pass

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
        # TODO: Implement full two-pass pipeline:
        #   1. Parse inputs (probe, positions, sqrt_intensities)
        #   2. Initialize object
        #   3. Phase 1: standard ePIE
        #   4. Estimate support and R_p from Phase 1 result
        #   5. Phase 2: rpPIE with prior
        #   6. Format and return output
        pass


if __name__ == "__main__":
    # Load measured diffraction intensity data
    input_data = np.load("dataset/input.npy")  # shape: (81, 256, 256)

    # Load probe function (2-channel: real, imaginary)
    probe_data = np.load("dataset/probe.npy")  # shape: (2, 256, 256)

    # Load scan positions
    scan_positions = np.load("dataset/scan_positions.npy")  # shape: (81, 2)

    # Instantiate solver with default hyperparameters
    solver = InverseSolver(
        object_shape=(600, 600),
        probe_shape=(256, 256),
        n_epochs_phase1=50,
        n_epochs_phase2=150,
        beta=0.9,
        alpha_fraction=1e-2,
        eps=1e-10,
        output_mode="real_imag",
    )

    # Run reconstruction
    result = solver.solve(
        intensities=input_data,
        probe_data=probe_data,
        scan_positions=scan_positions,
    )

    # Verify output shape
    assert result.shape == (2, 600, 600), f"Unexpected output shape: {result.shape}"

    # Save result
    np.save("output.npy", result)