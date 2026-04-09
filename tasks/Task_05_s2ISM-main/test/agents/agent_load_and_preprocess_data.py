import numpy as np

from scipy.special import jv, kl_div

from scipy.stats import pearsonr

from scipy.signal import argrelmin, argrelmax

import brighteyes_ism.simulation.PSF_sim as psf_sim

import brighteyes_ism.simulation.Tubulin_sim as st

import brighteyes_ism.analysis.Tools_lib as tools

def psf_width(pxsizex: float, pxsizez: float, Nz: int, simPar: psf_sim.simSettings, spad_size, stack='positive') -> int:
    if stack == 'positive' or stack == 'negative':
        z = pxsizez * Nz
    else:
        z = pxsizez * (Nz//2)

    M2 = 3
    w0 = simPar.airy_unit/2
    z_r = (np.pi * w0**2 * simPar.n) / simPar.wl
    w_z = w0 * np.sqrt( 1 + (M2 * z / z_r )**2)
    Nx = int(np.round((2 * (w_z + spad_size) / pxsizex)))
    if Nx % 2 == 0:
        Nx += 1
    return Nx

def conditioning(gridPar: psf_sim.GridParameters, exPar: psf_sim.simSettings = None, emPar: psf_sim.simSettings = None,
                 stedPar: psf_sim.simSettings = None, mode='Pearson', stack='positive', input_psf=None):
    if input_psf is None:
        if exPar is None or emPar is None:
            raise Exception("PSF is not an input. PSF parameters are needed.")

        gridPar.Nx = psf_width(gridPar.pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
        PSF, detPSF, exPSF = psf_sim.SPAD_PSF_3D(gridPar, exPar, emPar, stedPar=stedPar, spad=None, stack=stack)

        npx = int(np.round(((gridPar.N // 2) * gridPar.pxpitch + gridPar.pxdim / 2) / gridPar.M / gridPar.pxsizex))

        PSF = tools.CropEdge(PSF, npx, edges='l,r,u,d', order='zxyc')
        detPSF = tools.CropEdge(detPSF, npx, edges='l,r,u,d', order='zxyc')
        exPSF = tools.CropEdge(exPSF, npx, edges='l,r,u,d', order='zxy')
    else:
        PSF, detPSF, exPSF = input_psf

    for i in range(gridPar.Nz):
        PSF[i] /= np.sum(PSF[i])

    corr = np.empty(gridPar.Nz)
    if mode == 'KL':
        for i in range(gridPar.Nz):
            corr[i] = kl_div(PSF[0, ...].flatten(), PSF[i, ...].flatten()).sum()

    elif mode == 'Pearson':
        for i in range(gridPar.Nz):
            corr[i] = pearsonr(PSF[0, ...].flatten(), PSF[i, ...].flatten())[0]

    return corr, [PSF, detPSF, exPSF]

def find_max_discrepancy(correlation: np.ndarray, gridpar: psf_sim.GridParameters, mode: str, graph: bool):
    if mode == 'KL':
        idx = np.asarray(argrelmax(correlation)).ravel()[0]
    elif mode == 'Pearson':
        idx = np.asarray(argrelmin(correlation)).ravel()[0]
    else:
        raise Exception("Discrepancy method unknown.")

    optimal_depth = idx * gridpar.pxsizez
    return optimal_depth

def find_out_of_focus_from_param(pxsizex: float = None, exPar: psf_sim.simSettings = None, emPar: psf_sim.simSettings = None,
                                 grid: psf_sim.GridParameters = None, stedPar: psf_sim.simSettings=None, mode: str = 'Pearson',
                                 stack: str = 'symmetrical', graph: bool = False):
    if exPar is None:
        raise Exception("PSF parameters are needed.")
    if emPar is None:
        raise Exception("PSF parameters are needed.")
    if pxsizex is None and grid is None:
        raise Exception("Pixel size is needed as input.")

    if grid is None:
        range_z = 1.5*exPar.depth_of_field
        nz = 60
        gridPar = psf_sim.GridParameters()
        gridPar.Nz = nz
        gridPar.pxsizez = np.round(range_z / nz)
        gridPar.pxsizex = pxsizex
    else:
        gridPar = grid.copy()

    Nx_temp = psf_width(pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
    gridPar.Nx = Nx_temp

    correlation, PSF = conditioning(gridPar=gridPar, emPar=emPar,
                                    exPar=exPar, stedPar=stedPar, mode=mode,
                                    stack=stack)

    optimal_depth = find_max_discrepancy(correlation=correlation, gridpar=gridPar, mode=mode, graph=graph)

    return optimal_depth, PSF

def load_and_preprocess_data(Nx=201, Nz=2, pxsizex=40):
    """
    Simulates ground truth phantom and PSF, calculates optimal background plane.
    """
    print("Generating Phantom...")
    tubulin_planar = st.tubSettings()
    tubulin_planar.xy_pixel_size = pxsizex
    tubulin_planar.xy_dimension = Nx
    tubulin_planar.xz_dimension = 1
    tubulin_planar.z_pixel = 1
    tubulin_planar.n_filament = 10
    tubulin_planar.radius_filament = 80
    tubulin_planar.intensity_filament = [0.6, 1]

    phTub = np.zeros([Nz, Nx, Nx])
    for i in range(Nz):
        phTub_planar = st.functionPhTub(tubulin_planar)
        phTub_planar = np.swapaxes(phTub_planar, 2, 0)
        phTub[i, :, :] = phTub_planar * (np.power(3, np.abs(i)))
    
    print("Simulating PSFs...")
    exPar = psf_sim.simSettings()
    exPar.na = 1.4
    exPar.wl = 640
    exPar.gamma = 45
    exPar.beta = 90
    exPar.n = 1.5
    exPar.mask_sampl = 50

    emPar = exPar.copy()
    emPar.wl = 660

    optimal_bkg_plane, _ = find_out_of_focus_from_param(pxsizex, exPar, emPar, mode='Pearson', stack='positive', graph=False)
    print(f'Optimal out-of-focus position = {optimal_bkg_plane} nm')

    gridPar = psf_sim.GridParameters()
    gridPar.Nz = Nz
    gridPar.pxsizex = pxsizex
    gridPar.pxsizez = optimal_bkg_plane
    gridPar.Nx = psf_width(gridPar.pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
    print(f'Number of pixels in simulation = {gridPar.Nx}')

    Psf, detPsf, exPsf = psf_sim.SPAD_PSF_3D(gridPar, exPar, emPar, stack='positive')
    
    # Normalize PSFs
    for i in range(Nz):
        Psf[i] /= Psf[i].sum()

    return phTub, Psf, optimal_bkg_plane
