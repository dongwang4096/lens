#!/usr/bin/env python3
"""Visualize the initial surface error map (seed=42)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from env.lens_surface import LensSurface

surface = LensSurface(128, 50.0, 200.0)
surface.reset(error_scale=1.0, seed=42)
err = surface.get_error_map()
valid = err[surface.mask]

print("=" * 60)
print("Initial Error Map (seed=42)")
print("=" * 60)
print(f"  Grid: {err.shape}")
print(f"  Mask pixels: {surface.mask.sum()}")
print(f"  RMS:  {np.sqrt(np.mean(valid**2)):.4f} µm")
print(f"  Mean: {np.mean(valid):.4f} µm")
print(f"  Std:  {np.std(valid):.4f} µm")
print(f"  Min:  {np.min(valid):.4f} µm")
print(f"  Max:  {np.max(valid):.4f} µm")
print(f"  PV:   {np.max(valid) - np.min(valid):.4f} µm")
print()

for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p:2d}: {np.percentile(valid, p):.4f} µm")

print()
print("=" * 60)
print("Spatial non-uniformity")
print("=" * 60)

# Quadrant analysis
h, w = 64, 64
for r0, r1, c0, c1, name in [
    (0, h, 0, w, "Top-Left"),
    (0, h, w, 128, "Top-Right"),
    (h, 128, 0, w, "Bot-Left"),
    (h, 128, w, 128, "Bot-Right"),
]:
    sub_mask = surface.mask[r0:r1, c0:c1]
    if sub_mask.sum() > 0:
        sub_err = err[r0:r1, c0:c1][sub_mask]
        print(f"  {name:10s}: mean={np.mean(sub_err):.4f}, max={np.max(sub_err):.4f}, "
              f"std={np.std(sub_err):.4f} µm")

# Ring analysis (radial)
print()
print("Radial profile (ring averages):")
r2 = surface.r2
half = surface.lens_diameter_mm / 2
for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
    r_inner = frac * half
    r_outer = (frac + 0.15) * half
    ring = (r2 >= r_inner**2) & (r2 < r_outer**2) & surface.mask
    if ring.sum() > 0:
        ring_err = err[ring]
        print(f"  r=[{frac:.2f}, {frac+0.15:.2f}]R: mean={np.mean(ring_err):.4f}, "
              f"max={np.max(ring_err):.4f} µm  ({ring.sum()} px)")

# ASCII heatmap
print()
print("=" * 60)
print("ASCII heatmap (8x8 blocks, mean error in µm)")
print("=" * 60)
block = 16
for bi in range(8):
    row_str = "  "
    for bj in range(8):
        sub = err[bi*block:(bi+1)*block, bj*block:(bj+1)*block]
        sub_m = surface.mask[bi*block:(bi+1)*block, bj*block:(bj+1)*block]
        if sub_m.sum() > 0:
            val = np.mean(sub[sub_m])
            row_str += f"{val:5.2f} "
        else:
            row_str += "  .   "
    print(row_str)

# Save
out_dir = Path(__file__).resolve().parents[1] / "results"
out_dir.mkdir(exist_ok=True)
np.save(out_dir / "initial_error_map.npy", err)
np.save(out_dir / "mask.npy", surface.mask)
print(f"\nSaved to {out_dir}")

# Also create a matplotlib figure if possible
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    err_display = err.copy()
    err_display[~surface.mask] = np.nan

    im = axes[0].imshow(err_display, cmap="hot", origin="lower")
    axes[0].set_title("Initial Form Error (µm)")
    plt.colorbar(im, ax=axes[0], shrink=0.8)
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("y pixel")

    axes[1].hist(valid, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    axes[1].axvline(np.mean(valid), color="red", ls="--", label=f"mean={np.mean(valid):.2f}")
    axes[1].axvline(np.median(valid), color="orange", ls="--", label=f"median={np.median(valid):.2f}")
    axes[1].set_xlabel("Error (µm)")
    axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Error Distribution")
    axes[1].legend()

    fig.tight_layout()
    fig_path = out_dir / "initial_surface.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")
except ImportError:
    print("matplotlib not available, skipping figure")
