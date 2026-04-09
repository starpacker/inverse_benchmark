import numpy as np

from scipy.special import jv, kl_div

from scipy.stats import pearsonr

import brighteyes_ism.simulation.PSF_sim as psf_sim

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
