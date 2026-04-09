import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'repo'))
from pyTFM.TFM_functions import ffttc_traction
import numpy as np

def forward_op(tx, ty, pixelsize, young, sigma=0.49):
    ax1_length, ax2_length = tx.shape
    max_ind = max(ax1_length, ax2_length)
    if max_ind % 2 != 0:
        max_ind += 1
    tx_e = np.zeros((max_ind, max_ind))
    ty_e = np.zeros((max_ind, max_ind))
    tx_e[:ax1_length, :ax2_length] = tx
    ty_e[:ax1_length, :ax2_length] = ty
    kx1 = np.array([list(range(0, max_ind//2)), ] * max_ind)
    kx2 = np.array([list(range(-max_ind//2, 0)), ] * max_ind)
    kx = np.append(kx1, kx2, axis=1) * 2 * np.pi / (pixelsize * max_ind)
    ky = kx.T
    k = np.sqrt(kx**2 + ky**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        A = (2 + 2*sigma) / (young * k**3)
    f11 = (1-sigma)*k**2 + sigma*ky**2
    f12 = -sigma*kx*ky
    f22 = (1-sigma)*k**2 + sigma*kx**2
    f12[:, max_ind//2] = 0
    f12[max_ind//2, :] = 0
    tx_ft = np.fft.fft2(tx_e)
    ty_ft = np.fft.fft2(ty_e)
    u_ft = A * (f11*tx_ft + f12*ty_ft)
    v_ft = A * (f12*tx_ft + f22*ty_ft)
    u_ft[0,0] = 0
    v_ft[0,0] = 0
    u = np.fft.ifft2(u_ft.astype(np.complex128)).real
    v = np.fft.ifft2(v_ft.astype(np.complex128)).real
    return u[:ax1_length, :ax2_length] / pixelsize, v[:ax1_length, :ax2_length] / pixelsize

N = 64
ps = 6.44e-6
E = 49000.0
sig = 0.49

y, x = np.mgrid[:N,:N]
cx, cy = N//2, N//2
g = np.exp(-((x-cx)**2 + (y-cy)**2)/(2*(N/8)**2))
tx_gt = 500.0 * g
ty_gt = np.zeros_like(tx_gt)

u_px, v_px = forward_op(tx_gt, ty_gt, ps, E, sig)
print(f'Forward: max |u|={np.max(np.abs(u_px)):.4f} px, max |v|={np.max(np.abs(v_px)):.4f} px')

# Inverse (no noise)
tx_rec, ty_rec = ffttc_traction(u_px, v_px, pixelsize1=ps, pixelsize2=ps, young=E, sigma=sig, spatial_filter='gaussian')
print(f'Inverse: max |tx|={np.max(np.abs(tx_rec)):.2f} Pa, gt max={tx_gt.max():.2f} Pa')
err = np.sqrt(np.mean((tx_gt - tx_rec)**2))
print(f'RMSE: {err:.2f} Pa')
cc = np.corrcoef(tx_gt.ravel(), tx_rec.ravel())[0,1]
print(f'Correlation: {cc:.4f}')

# Also test without filter
tx_rec2, ty_rec2 = ffttc_traction(u_px, v_px, pixelsize1=ps, pixelsize2=ps, young=E, sigma=sig, spatial_filter=None)
err2 = np.sqrt(np.mean((tx_gt - tx_rec2)**2))
cc2 = np.corrcoef(tx_gt.ravel(), tx_rec2.ravel())[0,1]
print(f'No filter: RMSE={err2:.2f}, CC={cc2:.4f}')
