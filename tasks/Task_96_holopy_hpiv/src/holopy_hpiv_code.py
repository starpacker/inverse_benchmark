#!/usr/bin/env python3
"""
Holographic PIV — 3D Particle Position Recovery from Inline Holograms.

Forward model (Angular Spectrum Method):
  H(x,y) = |E_ref + Σ_i E_scat_i|²
  E_scat_i = ASM_propagate(shadow_i, z_i)

Inverse model:
  1. Back-propagate hologram to z-planes via ASM
  2. Local focus metric (gradient-based Tamura coefficient)
  3. Peak detection in 3-D focus volume → (x, y, z)

Metrics: RMSE, Pearson CC, PSNR of 3-D positions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.spatial.distance import cdist
from pathlib import Path
import json, time

RNG = np.random.RandomState(42)

# ── Parameters ───────────────────────────────────────────────────
WAVELENGTH  = 0.633e-6       # He-Ne 633 nm
PIXEL_SIZE  = 2.2e-6         # detector pitch
NX, NY      = 256, 256
N_PARTICLES = 12
Z_MIN, Z_MAX = 200e-6, 1200e-6
R_MIN, R_MAX = 3e-6, 6e-6
MARGIN      = 30

Z_SCAN_N    = 120
Z_SCAN_MIN  = 100e-6
Z_SCAN_MAX  = 1400e-6
MATCH_DIST  = 200e-6

WORKING_DIR = Path("/data/yjh/holopy_hpiv_sandbox")
ASSET_DIR   = Path("/data/yjh/website_assets/Task_96_holopy_hpiv")

# ═══════════════════  FORWARD  ═══════════════════════════════════

def _asm_kernel(nx, ny, dx, z, wl):
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    kz2 = (1.0/wl)**2 - FX**2 - FY**2
    prop = kz2 > 0
    return np.exp(1j*2*np.pi*np.sqrt(np.maximum(kz2, 0))*z) * prop

def _shadow(nx, ny, dx, x0, y0, r):
    xx = np.arange(nx)*dx;  yy = np.arange(ny)*dx
    XX, YY = np.meshgrid(xx, yy, indexing="ij")
    s = np.zeros((nx, ny), dtype=complex)
    s[(XX-x0)**2 + (YY-y0)**2 <= r**2] = -1.0
    return s

def generate_particles(n, nx, ny, dx, zlo, zhi, rlo, rhi, margin, rng):
    pts = []; sep_xy = 25*dx; sep_z = 40e-6
    for _ in range(n*100):
        if len(pts) >= n: break
        x = rng.uniform(margin*dx, (nx-margin)*dx)
        y = rng.uniform(margin*dx, (ny-margin)*dx)
        z = rng.uniform(zlo, zhi)
        r = rng.uniform(rlo, rhi)
        if all(not (abs(x-p[0])<sep_xy and abs(y-p[1])<sep_xy and abs(z-p[2])<sep_z) for p in pts):
            pts.append((x, y, z, r))
    return np.array(pts)

def simulate_hologram(parts, nx, ny, dx, wl):
    E = np.ones((nx, ny), dtype=complex)
    for x0, y0, z0, rad in parts:
        sh = _shadow(nx, ny, dx, x0, y0, rad)
        E += np.fft.ifft2(np.fft.fft2(sh) * _asm_kernel(nx, ny, dx, z0, wl))
    return np.abs(E)**2

# ═══════════════════  INVERSE  ═══════════════════════════════════

def detect_particles_3d(holo, dx, z_planes, wl, n_exp):
    nx, ny = holo.shape;  nz = len(z_planes)
    H_fft = np.fft.fft2(np.sqrt(holo.astype(complex)))
    grad_vol = np.zeros((nz, nx-1, ny-1), dtype=np.float32)
    for i, z in enumerate(z_planes):
        E = np.fft.ifft2(H_fft * _asm_kernel(nx, ny, dx, -z, wl))
        I = np.abs(E)**2
        gx = np.diff(I, axis=0)[:,:-1]
        gy = np.diff(I, axis=1)[:-1,:]
        grad_vol[i] = gaussian_filter(np.sqrt(gx**2+gy**2).astype(np.float32), sigma=3.0)

    mip = np.max(grad_vol, axis=0)
    thr = np.percentile(mip, 93)
    lab, nf = label(mip > thr)
    centroids = center_of_mass(mip, lab, range(1, nf+1))

    det = []
    for cy, cx in centroids:
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < grad_vol.shape[1] and 0 <= ix < grad_vol.shape[2]:
            zidx = int(np.argmax(grad_vol[:, iy, ix]))
            det.append([(ix+0.5)*dx, (iy+0.5)*dx, z_planes[zidx]])
    return (np.array(det) if det else np.zeros((0,3))), grad_vol

# ═══════════════════  METRICS  ═══════════════════════════════════

def match_particles(gt, det, md):
    if len(det)==0 or len(gt)==0: return np.zeros((0,3)), np.zeros((0,3))
    D = cdist(gt, det); mg=[]; md_=[]; ug=set(); ud=set()
    for idx in np.argsort(D, axis=None):
        gi, di = np.unravel_index(idx, D.shape)
        if gi in ug or di in ud: continue
        if D[gi,di] > md: break
        mg.append(gt[gi]); md_.append(det[di]); ug.add(gi); ud.add(di)
    return np.array(mg), np.array(md_)

def _rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
def _cc(a, b):
    af, bf = a.ravel(), b.ravel()
    return float(np.corrcoef(af, bf)[0,1]) if np.std(af)>1e-15 and np.std(bf)>1e-15 else 0.0
def _psnr(a, b):
    mse = np.mean((a-b)**2)
    mx = np.max(np.abs(a))
    return 100.0 if mse<1e-30 else (0.0 if mx<1e-30 else float(10*np.log10(mx**2/mse)))

# ═══════════════════  VISUALISATION  ═════════════════════════════

def make_fig(holo, gt, det, mg, md, gv, zp, path):
    um=1e6; dx=PIXEL_SIZE
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax=axes[0,0]; ext=[0,holo.shape[0]*dx*um,0,holo.shape[1]*dx*um]
    im=ax.imshow(holo.T, cmap="gray", origin="lower", extent=ext)
    ax.set_title("Simulated Inline Hologram"); ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax=axes[0,1]; mip=np.max(gv, axis=0); ext2=[0,mip.shape[0]*dx*um,0,mip.shape[1]*dx*um]
    im=ax.imshow(mip.T, cmap="hot", origin="lower", extent=ext2)
    ax.set_title("Focus MIP (x-y)"); ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax=axes[1,0]
    ax.scatter(gt[:,0]*um, gt[:,1]*um, facecolors="none", edgecolors="blue", s=120, lw=2, label="GT", zorder=2)
    if len(det)>0: ax.scatter(det[:,0]*um, det[:,1]*um, c="red", marker="x", s=80, lw=2, label="Det", zorder=3)
    if len(mg)>0:
        for g,d in zip(mg*um, md*um): ax.plot([g[0],d[0]],[g[1],d[1]],"g--",alpha=.5,lw=.8)
    ax.set_title("Top (x-y)"); ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)"); ax.legend(); ax.set_aspect("equal")

    ax=axes[1,1]
    ax.scatter(gt[:,0]*um, gt[:,2]*um, facecolors="none", edgecolors="blue", s=120, lw=2, label="GT", zorder=2)
    if len(det)>0: ax.scatter(det[:,0]*um, det[:,2]*um, c="red", marker="x", s=80, lw=2, label="Det", zorder=3)
    if len(mg)>0:
        for g,d in zip(mg*um, md*um): ax.plot([g[0],d[0]],[g[2],d[2]],"g--",alpha=.5,lw=.8)
    ax.set_title("Side (x-z)"); ax.set_xlabel("x (μm)"); ax.set_ylabel("z (μm)"); ax.legend()

    plt.tight_layout(); plt.savefig(str(path), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved → {path}")

# ═══════════════════  MAIN  ═════════════════════════════════════

def main():
    t0 = time.time()
    print("="*70)
    print("Holographic PIV — 3D Particle Recovery")
    print("="*70)

    print("\n[1/6] Particles …")
    gt_parts = generate_particles(N_PARTICLES, NX, NY, PIXEL_SIZE, Z_MIN, Z_MAX, R_MIN, R_MAX, MARGIN, RNG)
    gt_pos = gt_parts[:,:3]; n_gt = len(gt_parts)
    print(f"  {n_gt} particles  z∈[{gt_pos[:,2].min()*1e6:.0f},{gt_pos[:,2].max()*1e6:.0f}] μm")

    print("\n[2/6] Hologram …")
    holo = simulate_hologram(gt_parts, NX, NY, PIXEL_SIZE, WAVELENGTH)
    print(f"  {holo.shape}  I∈[{holo.min():.4f},{holo.max():.4f}]")

    print("\n[3/6] Backprop ({} z-planes) …".format(Z_SCAN_N))
    zp = np.linspace(Z_SCAN_MIN, Z_SCAN_MAX, Z_SCAN_N)
    det, gv = detect_particles_3d(holo, PIXEL_SIZE, zp, WAVELENGTH, n_gt)
    print(f"  Detected {len(det)}")

    print("\n[4/6] Metrics …")
    mg, md = match_particles(gt_pos, det, MATCH_DIST)
    nm = len(mg)
    if nm > 0:
        r3=_rmse(mg,md); rxy=_rmse(mg[:,:2],md[:,:2]); rz=_rmse(mg[:,2:],md[:,2:])
        cc=_cc(mg,md); p=_psnr(mg,md)
    else:
        r3=rxy=rz=float("inf"); cc=p=0.0
    print(f"  Matched {nm}/{n_gt}  RMSE={r3*1e6:.2f}μm  CC={cc:.4f}  PSNR={p:.2f}dB")

    print("\n[5/6] Save …")
    for d in [WORKING_DIR, ASSET_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        np.save(str(d/"gt_output.npy"), gt_pos)
        np.save(str(d/"recon_output.npy"), det)
    m = dict(n_gt=int(n_gt), n_detected=int(len(det)), n_matched=int(nm),
             detection_rate=round(nm/n_gt,4), rmse_3d_um=round(r3*1e6,2),
             rmse_xy_um=round(rxy*1e6,2), rmse_z_um=round(rz*1e6,2),
             cc=round(cc,4), psnr_db=round(p,2))
    with open(str(WORKING_DIR/"metrics.json"),"w") as f: json.dump(m, f, indent=2)

    print("\n[6/6] Plot …")
    for d in [ASSET_DIR, WORKING_DIR]:
        make_fig(holo, gt_parts, det, mg, md, gv, zp, d/"vis_result.png")

    el = time.time()-t0
    print(f"\n{'='*70}")
    print(f"DONE ({el:.1f}s)  PSNR={p:.2f}dB  CC={cc:.4f}  RMSE={r3*1e6:.2f}μm")
    print("="*70)
    return m

if __name__ == "__main__":
    main()
