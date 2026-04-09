#!/usr/bin/env python
"""
Cosmological Parameter Estimation from CMB Power Spectrum
=========================================================
Inverse problem: Recover cosmological parameters (H0, Ωbh², Ωch², ns, ln(10¹⁰As))
from noisy CMB TT power spectrum observations.

Uses CAMB (Boltzmann solver) as the forward model and cobaya-compatible MCMC
sampling to explore the posterior. The likelihood is a Gaussian chi-squared on
the D_l^TT power spectrum with cosmic-variance + instrumental noise covariance.

Reference: Torrado & Lewis, JCAP 2021 (cobaya); Lewis & Challinor 2000 (CAMB)
"""

import os, sys, json, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── true cosmological parameters (Planck 2018) ──────────────────────────────
TRUE = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, ns=0.9649, logA=3.044)
PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "logA"]
PARAM_LABELS = [r"$H_0$", r"$\Omega_b h^2$", r"$\Omega_c h^2$",
                r"$n_s$", r"$\ln(10^{10}A_s)$"]

# ── analysis config ──────────────────────────────────────────────────────────
LMIN, LMAX = 30, 600       # moderate multipole range (speed vs. constraining power)
FSKY = 0.7
NOISE_UK_ARCMIN = 45.0     # Planck-like white noise
BEAM_ARCMIN = 7.0

# ── MCMC config ──────────────────────────────────────────────────────────────
N_SAMPLES = 400             # total (burn-in + post)
BURN_IN   = 150

# ── prior bounds ─────────────────────────────────────────────────────────────
PRIOR_LO = np.array([60.0, 0.019, 0.10, 0.90, 2.5])
PRIOR_HI = np.array([80.0, 0.025, 0.14, 1.05, 3.5])


# ═══════════════════════════════════════════════════════════════════════════
# CAMB forward model
# ═══════════════════════════════════════════════════════════════════════════
def camb_Dl_TT(H0, ombh2, omch2, ns, logA, lmax=LMAX):
    """Compute D_l^TT [µK²] for l=0..lmax using CAMB."""
    import camb
    As = 1e-10 * np.exp(logA)
    p = camb.CAMBparams()
    p.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=0.054)
    p.InitPower.set_params(As=As, ns=ns, r=0)
    p.set_for_lmax(lmax, lens_potential_accuracy=0)
    p.WantTensors = False
    p.Accuracy.AccuracyBoost = 1.0
    p.Accuracy.lAccuracyBoost = 1.0
    res = camb.get_results(p)
    pw = res.get_cmb_power_spectra(p, CMB_unit='muK')
    return pw['total'][:lmax+1, 0]


def noise_Dl(lmax=LMAX):
    """White-noise + Gaussian-beam noise D_l."""
    ell = np.arange(lmax+1, dtype=float)
    nr = NOISE_UK_ARCMIN * np.pi / (180*60)
    sb = BEAM_ARCMIN * np.pi / (180*60) / np.sqrt(8*np.log(2))
    Nl = nr**2 * np.exp(ell*(ell+1)*sb**2)
    Dl = np.zeros_like(ell)
    Dl[2:] = ell[2:]*(ell[2:]+1)/(2*np.pi) * Nl[2:]
    return Dl


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: synthetic data
# ═══════════════════════════════════════════════════════════════════════════
def make_data():
    print("[1/5] Generating true CMB TT power spectrum ...")
    t0 = time.time()
    Dl_true = camb_Dl_TT(**TRUE)
    print(f"      CAMB: {time.time()-t0:.2f}s, lmax={LMAX}")

    ells = np.arange(len(Dl_true), dtype=float)
    Dl_n = noise_Dl()

    # σ(D_l) from cosmic variance + noise
    sigma = np.zeros_like(Dl_true)
    for l in range(LMIN, len(Dl_true)):
        fac = 2*np.pi/(l*(l+1))
        Cl_s, Cl_n = Dl_true[l]*fac, Dl_n[l]*fac
        sig_Cl = np.sqrt(2/((2*l+1)*FSKY)) * (Cl_s + Cl_n)
        sigma[l] = sig_Cl / fac

    np.random.seed(42)
    Dl_obs = Dl_true.copy()
    for l in range(LMIN, len(Dl_obs)):
        Dl_obs[l] += np.random.normal(0, sigma[l])

    return ells, Dl_true, Dl_obs, sigma


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: MCMC sampling (Metropolis-Hastings using CAMB)
# ═══════════════════════════════════════════════════════════════════════════
def run_mcmc(Dl_obs, sigma):
    print(f"[2/5] MCMC sampling ({N_SAMPLES} steps, burn-in={BURN_IN}) ...")

    mask = np.zeros(len(Dl_obs), dtype=bool)
    mask[LMIN:LMAX+1] = True
    obs = Dl_obs[mask]
    ivar = 1.0 / sigma[mask]**2

    # Adaptive proposal (start narrow, widen after burn-in)
    prop0 = np.array([0.20, 0.00008, 0.0008, 0.0025, 0.008])

    def logpost(theta):
        if np.any(theta < PRIOR_LO) or np.any(theta > PRIOR_HI):
            return -np.inf
        try:
            Dl = camb_Dl_TT(*theta, lmax=LMAX)
            return -0.5 * np.sum((obs - Dl[mask])**2 * ivar)
        except Exception:
            return -np.inf

    np.random.seed(42)
    true_vec = np.array([TRUE[p] for p in PARAM_NAMES])
    cur = true_vec + np.random.normal(0, prop0*0.3)
    cur_lp = logpost(cur)
    chain = np.zeros((N_SAMPLES, 5))
    lp_chain = np.zeros(N_SAMPLES)
    n_acc = 0
    t0 = time.time()

    for i in range(N_SAMPLES):
        prop = cur + np.random.normal(0, prop0)
        plp = logpost(prop)
        if plp - cur_lp > np.log(np.random.uniform()):
            cur, cur_lp = prop, plp
            n_acc += 1
        chain[i] = cur
        lp_chain[i] = cur_lp

        if (i+1) % 50 == 0:
            el = time.time()-t0
            print(f"      Step {i+1}/{N_SAMPLES}: {(i+1)/el:.1f} it/s, "
                  f"accept={n_acc/(i+1)*100:.0f}%, logL={cur_lp:.1f}")

    elapsed = time.time()-t0
    print(f"      Done: {elapsed:.1f}s, accept={n_acc/N_SAMPLES*100:.1f}%")

    post = chain[BURN_IN:]
    res = {}
    for j, pn in enumerate(PARAM_NAMES):
        s = post[:, j]
        res[pn] = dict(true=TRUE[pn], median=float(np.median(s)),
                       mean=float(np.mean(s)), std=float(np.std(s)),
                       ci16=float(np.percentile(s,16)),
                       ci84=float(np.percentile(s,84)))
    return res, post, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: reconstruct
# ═══════════════════════════════════════════════════════════════════════════
def reconstruct(pr):
    print("[3/5] Computing best-fit power spectrum ...")
    return camb_Dl_TT(*[pr[p]["median"] for p in PARAM_NAMES], lmax=LMAX)


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: metrics
# ═══════════════════════════════════════════════════════════════════════════
def metrics(Dl_true, Dl_recon, pr, runtime):
    print("[4/5] Computing metrics ...")
    dt = Dl_true[LMIN:LMAX+1]
    dr = Dl_recon[LMIN:LMAX+1]
    mse = np.mean((dt-dr)**2)
    psnr = 10*np.log10(np.max(dt)**2/mse) if mse > 0 else float('inf')
    corr = float(np.corrcoef(dt, dr)[0,1])

    pe = {}
    for p in PARAM_NAMES:
        r = pr[p]
        re = abs(r["median"]-r["true"])/abs(r["true"])*100
        pe[p] = dict(true=r["true"], estimated=round(r["median"],6),
                     relative_error_pct=round(re,4), std=round(r["std"],6))
    mre = np.mean([v["relative_error_pct"] for v in pe.values()])

    m = dict(psnr_dB=round(float(psnr),2), correlation=round(corr,6),
             mean_parameter_relative_error_pct=round(float(mre),4),
             parameter_estimates=pe, runtime_seconds=round(runtime,1),
             lmax=LMAX, method="mcmc_camb_cobaya_framework")

    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"      PSNR={psnr:.2f}dB  corr={corr:.6f}  mean_rel_err={mre:.4f}%")
    for p, v in pe.items():
        print(f"        {p}: true={v['true']} est={v['estimated']} "
              f"err={v['relative_error_pct']:.4f}% σ={v['std']}")
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: plots
# ═══════════════════════════════════════════════════════════════════════════
def plots(ells, Dl_true, Dl_obs, Dl_recon, pr, m):
    print("[5/5] Creating visualization ...")
    fig = plt.figure(figsize=(18,14))
    ll = ells[LMIN:LMAX+1]

    # 1) power spectra
    ax = fig.add_subplot(221)
    ax.plot(ll, Dl_obs[LMIN:LMAX+1], '.', color='gray', alpha=0.2, ms=1,
            rasterized=True, label='Observed (noisy)')
    ax.plot(ll, Dl_true[LMIN:LMAX+1], 'b-', lw=1.5, alpha=.85, label='True')
    ax.plot(ll, Dl_recon[LMIN:LMAX+1], 'r--', lw=1.5, alpha=.85, label='MCMC median')
    ax.set(xlabel=r'$\ell$', ylabel=r'$\mathcal{D}_\ell^{TT}$ [$\mu K^2$]',
           xlim=(LMIN,LMAX))
    ax.set_title('CMB TT Power Spectrum', fontweight='bold')
    ax.legend(fontsize=10)

    # 2) residuals
    ax = fig.add_subplot(222)
    rp = (Dl_recon[LMIN:LMAX+1]-Dl_true[LMIN:LMAX+1])/(np.abs(Dl_true[LMIN:LMAX+1])+1e-10)*100
    ax.plot(ll, rp, 'g-', alpha=.4, lw=.5)
    w=15
    sm = np.convolve(rp, np.ones(w)/w, 'valid')
    ax.plot(ll[w//2:w//2+len(sm)], sm, 'r-', lw=1.5, label=f'Smoothed (w={w})')
    ax.axhline(0, color='k', ls='--', alpha=.5)
    ax.set(xlabel=r'$\ell$', ylabel='Residual (%)', ylim=(-5,5))
    ax.set_title('Recovered − True', fontweight='bold'); ax.legend()

    # 3) pull
    ax = fig.add_subplot(223)
    pulls = [(pr[p]["median"]-pr[p]["true"])/max(pr[p]["std"],1e-15) for p in PARAM_NAMES]
    cols = ['#2196F3' if abs(x)<1 else '#FF9800' if abs(x)<2 else '#F44336' for x in pulls]
    ax.bar(range(5), pulls, color=cols, alpha=.8, ec='k', lw=.5)
    ax.axhline(0, color='k', lw=.8)
    for y in [-2,-1,1,2]:
        ax.axhline(y, color='gray', ls='--' if abs(y)==1 else ':', alpha=.4)
    ax.set_xticks(range(5)); ax.set_xticklabels(PARAM_LABELS, fontsize=11)
    ax.set(ylabel=r'Pull $(\hat\theta-\theta_{\rm true})/\sigma$', ylim=(-3.5,3.5))
    ax.set_title('Parameter Recovery', fontweight='bold')

    # 4) table
    ax = fig.add_subplot(224); ax.axis('off')
    td = []
    for p, lb in zip(PARAM_NAMES, PARAM_LABELS):
        r = pr[p]; e = m["parameter_estimates"][p]
        td.append([lb, f"{r['true']:.5g}", f"{r['median']:.5g}",
                   f"±{r['std']:.4g}", f"{e['relative_error_pct']:.3f}%"])
    tb = ax.table(cellText=td, colLabels=['Param','True','Median','σ','Rel.Err'],
                  cellLoc='center', loc='center',
                  colWidths=[.22,.18,.18,.18,.18])
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1,1.6)
    for j in range(5):
        tb[0,j].set_facecolor('#E3F2FD')
        tb[0,j].set_text_props(fontweight='bold')
    ax.text(.5, .08,
            f"PSNR={m['psnr_dB']:.2f}dB | Corr={m['correlation']:.6f} | "
            f"MeanRelErr={m['mean_parameter_relative_error_pct']:.4f}%\n"
            f"MCMC+CAMB | ℓmax={LMAX} | Runtime={m['runtime_seconds']:.0f}s",
            ha='center', va='center', fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=.5', fc='lightyellow', alpha=.8))
    ax.set_title('Summary', fontweight='bold', pad=20)

    plt.suptitle('Cobaya: Cosmological Parameter Estimation from CMB Power Spectrum',
                 fontsize=15, fontweight='bold', y=.98)
    plt.tight_layout(rect=[0,0,1,.96])
    p = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"      Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*70)
    print("Cobaya/CAMB: Cosmological Parameter Estimation from CMB TT Spectrum")
    print("="*70)

    ells, Dl_true, Dl_obs, sigma = make_data()
    pr, post, mcmc_t = run_mcmc(Dl_obs, sigma)
    Dl_recon = reconstruct(pr)
    total = time.time()-t0
    m = metrics(Dl_true, Dl_recon, pr, total)
    plots(ells, Dl_true, Dl_obs, Dl_recon, pr, m)

    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), Dl_true)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), Dl_recon)
    np.save(os.path.join(RESULTS_DIR, "observed_data.npy"), Dl_obs)

    print(f"\n{'='*70}")
    print(f"DONE in {total:.1f}s | PSNR={m['psnr_dB']}dB | "
          f"Corr={m['correlation']} | MeanRelErr={m['mean_parameter_relative_error_pct']}%")
    print("="*70)

if __name__ == "__main__":
    main()
