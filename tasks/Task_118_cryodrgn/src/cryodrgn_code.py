"""
Task 118: CryoDRGN - Cryo-EM 3D Reconstruction Benchmark
=========================================================
Inverse Problem: Recover a 3D protein-like density volume from 2D cryo-EM
projection images generated via the Fourier Slice Theorem with CTF modulation
and additive noise.

Forward Model:
  3D Volume -> 3D FFT -> Extract 2D central Fourier slice at given rotation ->
  Apply CTF -> iFFT2 -> Add Gaussian noise -> 2D projection

Inverse Solver:
  Direct Fourier Inversion with CTF-weighted gridding:
  For each projection, compute its 2D FFT, apply CTF^2-weighted insertion
  into 3D Fourier space (normal equations for least-squares),
  then divide by accumulated CTF^2 weights and iFFT3 to get the volume.

Metrics: PSNR, SSIM, RMSE
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# ============================================================
# 1. Configuration & Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

VOL_SIZE = 64
N_PROJECTIONS = 5000
NOISE_STD = 0.01
CTF_DEFOCUS = 8000.0
PIXEL_SIZE = 2.5
WIENER_EPS = 0.001


# ============================================================
# 2. Data Generation: Synthetic 3D Phantom
# ============================================================
def create_phantom(N):
    vol = np.zeros((N, N, N), dtype=np.float64)
    center = N / 2.0
    z, y, x = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)

    r2 = (x - center)**2 + (y - center)**2 + (z - center)**2
    vol += 1.0 * np.exp(-r2 / (2 * (N * 0.15)**2))

    cx1, cy1, cz1 = center + N*0.18, center + N*0.12, center
    r2_1 = (x - cx1)**2 + (y - cy1)**2 + (z - cz1)**2
    vol += 0.8 * np.exp(-r2_1 / (2 * (N * 0.10)**2))

    cx2, cy2, cz2 = center - N*0.15, center - N*0.10, center + N*0.12
    r2_2 = (x - cx2)**2 + (y - cy2)**2 + (z - cz2)**2
    vol += 0.7 * np.exp(-r2_2 / (2 * (N * 0.08)**2))

    dist_cyl = np.sqrt((x - center - N*0.05)**2 + (y - center + N*0.15)**2)
    mask_cyl = (dist_cyl < N * 0.04) & (z > center - N*0.2) & (z < center + N*0.2)
    vol[mask_cyl] += 0.6

    for (dx, dy, dz) in [(0.1, 0.1, 0.15), (-0.12, 0.08, -0.1), (0.05, -0.15, 0.05)]:
        cx_s, cy_s, cz_s = center + N*dx, center + N*dy, center + N*dz
        r2_s = (x - cx_s)**2 + (y - cy_s)**2 + (z - cz_s)**2
        vol += 1.2 * np.exp(-r2_s / (2 * (N * 0.03)**2))

    vol = gaussian_filter(vol, sigma=1.0)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-10)
    return vol.astype(np.float32)


# ============================================================
# 3. Forward Operator: Fourier Slice + CTF
# ============================================================
def generate_rotations(n_proj):
    return Rotation.random(n_proj, random_state=42).as_matrix().astype(np.float64)


def compute_ctf_2d(N, defocus, pixel_size=2.5, voltage=300.0, cs=2.7, w=0.07):
    freq = np.fft.fftfreq(N, d=pixel_size)
    freq_s = np.fft.fftshift(freq)
    fy, fx = np.meshgrid(freq_s, freq_s, indexing='ij')
    s2 = fx**2 + fy**2
    voltage_V = voltage * 1e3
    lam = 12.2639 / np.sqrt(voltage_V + 0.97845e-6 * voltage_V**2)
    cs_A = cs * 1e7
    gamma = 2 * np.pi * (-0.5 * defocus * lam * s2 + 0.25 * cs_A * lam**3 * s2**2)
    ctf = np.sqrt(1 - w**2) * np.sin(gamma) - w * np.cos(gamma)
    return ctf.astype(np.float64)


def trilinear_interp(vol3d, coords, N):
    coords = np.clip(coords, 0, N - 1.001)
    x0 = np.floor(coords[:, 0]).astype(int)
    y0 = np.floor(coords[:, 1]).astype(int)
    z0 = np.floor(coords[:, 2]).astype(int)
    x1 = np.minimum(x0+1, N-1); y1 = np.minimum(y0+1, N-1); z1 = np.minimum(z0+1, N-1)
    xd = coords[:,0]-x0; yd = coords[:,1]-y0; zd = coords[:,2]-z0

    c000=vol3d[z0,y0,x0]; c001=vol3d[z0,y0,x1]
    c010=vol3d[z0,y1,x0]; c011=vol3d[z0,y1,x1]
    c100=vol3d[z1,y0,x0]; c101=vol3d[z1,y0,x1]
    c110=vol3d[z1,y1,x0]; c111=vol3d[z1,y1,x1]

    c00=c000*(1-xd)+c001*xd; c01=c010*(1-xd)+c011*xd
    c10=c100*(1-xd)+c101*xd; c11=c110*(1-xd)+c111*xd
    c0=c00*(1-yd)+c01*yd; c1=c10*(1-yd)+c11*yd
    return c0*(1-zd)+c1*zd


def forward_project(volume, rot_mats, noise_std=0.01, apply_ctf=True):
    N = volume.shape[0]
    vol_fft = np.fft.fftshift(np.fft.fftn(volume))
    freq_1d = np.fft.fftshift(np.fft.fftfreq(N))
    gy, gx = np.meshgrid(freq_1d, freq_1d, indexing='ij')
    ctf = compute_ctf_2d(N, CTF_DEFOCUS, PIXEL_SIZE) if apply_ctf else None
    coords_2d = np.stack([gx.ravel(), gy.ravel(), np.zeros(N*N)], axis=-1)

    projections = []
    for i in range(len(rot_mats)):
        R = rot_mats[i]
        coords_3d = coords_2d @ R.T
        coords_vox = coords_3d * N + N/2.0
        sl = trilinear_interp(vol_fft, coords_vox, N).reshape(N, N)
        if apply_ctf:
            sl = sl * ctf
        proj = np.real(np.fft.ifft2(np.fft.ifftshift(sl)))
        sig = np.std(proj) + 1e-10
        proj += np.random.RandomState(i).normal(0, noise_std*sig, proj.shape)
        projections.append(proj.astype(np.float32))
    return np.array(projections)


# ============================================================
# 4. Inverse Solver: CTF-Weighted DFI
# ============================================================
def reconstruct_dfi(projections, rot_mats, apply_ctf=True):
    N = projections.shape[1]
    n_proj = len(projections)
    vol_num = np.zeros((N,N,N), dtype=np.complex128)
    vol_den = np.zeros((N,N,N), dtype=np.float64)
    freq_1d = np.fft.fftshift(np.fft.fftfreq(N))
    gy, gx = np.meshgrid(freq_1d, freq_1d, indexing='ij')
    ctf = compute_ctf_2d(N, CTF_DEFOCUS, PIXEL_SIZE) if apply_ctf else np.ones((N,N))
    coords_2d = np.stack([gx.ravel(), gy.ravel(), np.zeros(N*N)], axis=-1)
    ctf_flat = ctf.ravel()
    ctf2_flat = ctf_flat**2

    print(f"Reconstructing from {n_proj} projections...")
    for i in range(n_proj):
        if (i+1)%500==0: print(f"  {i+1}/{n_proj}")
        R = rot_mats[i]
        coords_3d = coords_2d @ R.T
        cv = coords_3d * N + N/2.0
        cv = np.clip(cv, 0, N-1.001)

        proj_fft = np.fft.fftshift(np.fft.fft2(projections[i]))
        wvals = ctf_flat * proj_fft.ravel()

        ix = np.round(cv[:,0]).astype(int)
        iy = np.round(cv[:,1]).astype(int)
        iz = np.round(cv[:,2]).astype(int)
        mask = (ix>=0)&(ix<N)&(iy>=0)&(iy<N)&(iz>=0)&(iz<N)
        ix,iy,iz = ix[mask],iy[mask],iz[mask]

        np.add.at(vol_num, (iz,iy,ix), wvals[mask])
        np.add.at(vol_den, (iz,iy,ix), ctf2_flat[mask])

    vol_fft_r = vol_num / (vol_den + WIENER_EPS)
    vol_r = np.real(np.fft.ifftn(np.fft.ifftshift(vol_fft_r)))
    vol_r = gaussian_filter(vol_r, sigma=0.4)
    vol_r = vol_r.astype(np.float32)
    vol_r = (vol_r - vol_r.min()) / (vol_r.max() - vol_r.min() + 1e-10)
    return vol_r


# ============================================================
# 5. Metrics
# ============================================================
def compute_metrics(gt, recon):
    gt_n = (gt-gt.min())/(gt.max()-gt.min()+1e-10)
    rec_n = (recon-recon.min())/(recon.max()-recon.min()+1e-10)
    rmse = float(np.sqrt(np.mean((gt_n-rec_n)**2)))
    N = gt.shape[0]; mid = N//2
    slices = [(gt_n[mid,:,:], rec_n[mid,:,:]),
              (gt_n[:,mid,:], rec_n[:,mid,:]),
              (gt_n[:,:,mid], rec_n[:,:,mid])]
    ss = [float(ssim_metric(g,r,data_range=1.0)) for g,r in slices]
    ps = [float(psnr_metric(g,r,data_range=1.0)) for g,r in slices]
    psnr_3d = float(psnr_metric(gt_n, rec_n, data_range=1.0))
    return {
        'PSNR_dB': round(psnr_3d,4), 'SSIM': round(float(np.mean(ss)),4),
        'RMSE': round(rmse,6),
        'PSNR_per_axis': [round(v,4) for v in ps],
        'SSIM_per_axis': [round(v,4) for v in ss],
        'n_projections': N_PROJECTIONS, 'volume_size': VOL_SIZE, 'noise_std': NOISE_STD,
    }


# ============================================================
# 6. Visualization
# ============================================================
def plot_results(gt, projs, recon, metrics):
    fig, axes = plt.subplots(4, 4, figsize=(16,16))
    mid = gt.shape[0]//2
    gn = (gt-gt.min())/(gt.max()-gt.min()+1e-10)
    rn = (recon-recon.min())/(recon.max()-recon.min()+1e-10)

    axes[0,0].imshow(gn[mid,:,:],cmap='gray'); axes[0,0].set_title('GT: Axial'); axes[0,0].axis('off')
    axes[0,1].imshow(gn[:,mid,:],cmap='gray'); axes[0,1].set_title('GT: Coronal'); axes[0,1].axis('off')
    axes[0,2].imshow(gn[:,:,mid],cmap='gray'); axes[0,2].set_title('GT: Sagittal'); axes[0,2].axis('off')
    axes[0,3].imshow(np.max(gn,axis=0),cmap='hot'); axes[0,3].set_title('GT: MIP'); axes[0,3].axis('off')

    for j in range(4):
        idx = j*(len(projs)//4)
        axes[1,j].imshow(projs[idx],cmap='gray'); axes[1,j].set_title(f'Proj #{idx}'); axes[1,j].axis('off')

    axes[2,0].imshow(rn[mid,:,:],cmap='gray'); axes[2,0].set_title('Recon: Axial'); axes[2,0].axis('off')
    axes[2,1].imshow(rn[:,mid,:],cmap='gray'); axes[2,1].set_title('Recon: Coronal'); axes[2,1].axis('off')
    axes[2,2].imshow(rn[:,:,mid],cmap='gray'); axes[2,2].set_title('Recon: Sagittal'); axes[2,2].axis('off')
    axes[2,3].imshow(np.max(rn,axis=0),cmap='hot'); axes[2,3].set_title('Recon: MIP'); axes[2,3].axis('off')

    for j,(title,sl) in enumerate([('Axial',mid), ('Coronal',mid), ('Sagittal',mid)]):
        if j==0: err=np.abs(gn[sl,:,:]-rn[sl,:,:])
        elif j==1: err=np.abs(gn[:,sl,:]-rn[:,sl,:])
        else: err=np.abs(gn[:,:,sl]-rn[:,:,sl])
        im=axes[3,j].imshow(err,cmap='hot',vmin=0,vmax=0.5)
        axes[3,j].set_title(f'Error: {title}'); axes[3,j].axis('off')
        plt.colorbar(im,ax=axes[3,j],fraction=0.046)

    axes[3,3].axis('off')
    t = (f"PSNR: {metrics['PSNR_dB']:.2f} dB\n"
         f"SSIM: {metrics['SSIM']:.4f}\n"
         f"RMSE: {metrics['RMSE']:.6f}\n"
         f"N_proj: {metrics['n_projections']}\n"
         f"Vol: {metrics['volume_size']}^3\n"
         f"Noise: {metrics['noise_std']}")
    axes[3,3].text(0.1,0.5,t,transform=axes[3,3].transAxes,fontsize=14,
                   va='center',fontfamily='monospace',
                   bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.8))
    axes[3,3].set_title('Metrics')

    fig.suptitle('Task 118: CryoDRGN - Cryo-EM 3D Reconstruction\n'
                 'Forward: Fourier Slice + CTF + Noise | Inverse: CTF-Weighted DFI',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(RESULTS_DIR,'reconstruction_result.png'),dpi=150,bbox_inches='tight')
    plt.close()


# ============================================================
# 7. Main
# ============================================================
def main():
    print("="*60)
    print("Task 118: CryoDRGN - Cryo-EM 3D Reconstruction Benchmark")
    print("="*60)

    print("\n[1/5] Creating 3D phantom...")
    gt = create_phantom(VOL_SIZE)
    print(f"  Shape: {gt.shape}")

    print(f"\n[2/5] Generating {N_PROJECTIONS} orientations...")
    rots = generate_rotations(N_PROJECTIONS)

    print(f"\n[3/5] Forward projecting...")
    projs = forward_project(gt, rots, NOISE_STD, True)
    print(f"  Projections: {projs.shape}")

    print(f"\n[4/5] Reconstructing (CTF-weighted DFI)...")
    recon = reconstruct_dfi(projs, rots, True)
    print(f"  Reconstruction: {recon.shape}")

    print(f"\n[5/5] Metrics...")
    metrics = compute_metrics(gt, recon)
    print(f"  PSNR: {metrics['PSNR_dB']:.2f} dB")
    print(f"  SSIM: {metrics['SSIM']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")

    with open(os.path.join(RESULTS_DIR,'metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR,'ground_truth.npy'), gt)
    np.save(os.path.join(RESULTS_DIR,'reconstruction.npy'), recon)

    # Save normalized [0,1] versions as gt_output.npy and recon_output.npy
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-10)
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-10)
    np.save(os.path.join(SCRIPT_DIR, 'gt_output.npy'), gt_norm)
    np.save(os.path.join(SCRIPT_DIR, 'recon_output.npy'), recon_norm)
    print(f"  Saved gt_output.npy: range [{gt_norm.min():.4f}, {gt_norm.max():.4f}]")
    print(f"  Saved recon_output.npy: range [{recon_norm.min():.4f}, {recon_norm.max():.4f}]")

    print("\nGenerating visualization...")
    plot_results(gt, projs, recon, metrics)

    print("\n"+"="*60)
    print("Task 118 COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print("="*60)
    return metrics

if __name__ == '__main__':
    main()
