import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def _make_fig(holo, gt, det, mg, md, gv, pixel_size, path):
    """Create visualization figure."""
    um = 1e6
    dx = pixel_size
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ext = [0, holo.shape[0] * dx * um, 0, holo.shape[1] * dx * um]
    im = ax.imshow(holo.T, cmap="gray", origin="lower", extent=ext)
    ax.set_title("Simulated Inline Hologram")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    mip = np.max(gv, axis=0)
    ext2 = [0, mip.shape[0] * dx * um, 0, mip.shape[1] * dx * um]
    im = ax.imshow(mip.T, cmap="hot", origin="lower", extent=ext2)
    ax.set_title("Focus MIP (x-y)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    ax.scatter(gt[:, 0] * um, gt[:, 1] * um, facecolors="none", edgecolors="blue", s=120, lw=2, label="GT", zorder=2)
    if len(det) > 0:
        ax.scatter(det[:, 0] * um, det[:, 1] * um, c="red", marker="x", s=80, lw=2, label="Det", zorder=3)
    if len(mg) > 0:
        for g, d in zip(mg * um, md * um):
            ax.plot([g[0], d[0]], [g[1], d[1]], "g--", alpha=0.5, lw=0.8)
    ax.set_title("Top (x-y)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1, 1]
    ax.scatter(gt[:, 0] * um, gt[:, 2] * um, facecolors="none", edgecolors="blue", s=120, lw=2, label="GT", zorder=2)
    if len(det) > 0:
        ax.scatter(det[:, 0] * um, det[:, 2] * um, c="red", marker="x", s=80, lw=2, label="Det", zorder=3)
    if len(mg) > 0:
        for g, d in zip(mg * um, md * um):
            ax.plot([g[0], d[0]], [g[2], d[2]], "g--", alpha=0.5, lw=0.8)
    ax.set_title("Side (x-z)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("z (μm)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
