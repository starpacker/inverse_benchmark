import matplotlib

matplotlib.use('Agg')

import os

import pymaster as nmt

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inversion(data):
    """
    NaMaster deconvolution: compute the mode-coupling matrix from the
    mask and apply its inverse to obtain unbiased Cl estimates.

    Parameters
    ----------
    data : dict from load_and_preprocess_data

    Returns
    -------
    result : dict with keys:
        'cl_recon' : array – decoupled power spectrum in each bin
        'ell_eff'  : array – effective multipole of each bin
    """
    full_map = data['full_map']
    mask = data['mask']
    nside = data['nside']
    lmax = data['lmax']

    # NaMaster field: pass the unmasked map; NaMaster applies the mask internally
    f = nmt.NmtField(mask, [full_map])

    # Binning scheme – 4 multipoles per bin
    bin_size = 4
    b = nmt.NmtBin.from_nside_linear(nside, bin_size)

    # Workspace: compute the mode-coupling matrix
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, b)

    # Decouple: pseudo-Cl -> true Cl
    cl_decoupled = nmt.compute_full_master(f, f, b)

    ell_eff = b.get_effective_ells()

    result = {
        'cl_recon': cl_decoupled[0],
        'ell_eff': ell_eff,
    }
    return result
