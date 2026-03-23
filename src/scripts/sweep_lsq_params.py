#!/usr/bin/env python3
"""Sweep key parameters to find what limits the least-squares solution.

Investigates:
  1. Dwell lower bound (forced minimum removal)
  2. Dwell upper bound (maximum removal per waypoint)
  3. Number of passes (degrees of freedom)
  4. Unconstrained solution (theoretical best with this path)
  5. Spiral pitch (coverage density)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
import yaml
from scipy.optimize import lsq_linear

from env.lens_surface import LensSurface
from env.spiral_path import SpiralPath
from models.tif_model import TIFModel


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_system(config, tif_ckpt, num_passes=None, pitch_mm=None):
    """Build A matrix and b vector. Optionally override passes/pitch."""
    sc = config["surface"]
    sp = config["spiral"]

    if num_passes is None:
        num_passes = sp["num_passes"]
    if pitch_mm is None:
        pitch_mm = sp["pitch_mm"]

    surface = LensSurface(sc["grid_size"], sc["lens_diameter_mm"], sc["curvature_radius_mm"])
    surface.reset(error_scale=1.0, seed=42)

    mask = surface.mask
    mask_indices = np.where(mask.ravel())[0]
    n_pixels = len(mask_indices)

    error_0 = surface.get_error_map()
    initial_rms = surface.get_rms_error()
    b = error_0.ravel()[mask_indices].astype(np.float64)

    spiral = SpiralPath(
        lens_radius_mm=sc["lens_diameter_mm"] / 2.0,
        pitch_mm=pitch_mm,
        min_radius_mm=3.0,
        center_sparse_power=0.7,
    )
    single_pass_len = spiral.get_single_pass_length()

    pass_pressures = sp["pass_pressures"]
    if num_passes > len(pass_pressures):
        extra = num_passes - len(pass_pressures)
        pass_pressures = pass_pressures + [pass_pressures[-1]] * extra

    all_waypoints = spiral.build_multi_pass(num_passes)
    n_waypoints = len(all_waypoints)

    waypoint_pressures = np.array([
        pass_pressures[min(i // single_pass_len, num_passes - 1)]
        for i in range(n_waypoints)
    ])

    tif_model = TIFModel(tif_ckpt)
    speed = sp["fixed_speed"]
    conc = config["tif"]["default_concentration"]

    unique_pressures = np.unique(waypoint_pressures)
    tif_params = {}
    for p in unique_pressures:
        width, depth = tif_model.predict(float(p), speed, conc)
        sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        tif_params[float(p)] = (sigma, depth)

    xx_mask = surface.xx.ravel()[mask_indices]
    yy_mask = surface.yy.ravel()[mask_indices]

    A = np.zeros((n_pixels, n_waypoints), dtype=np.float64)
    for i in range(n_waypoints):
        cx = float(all_waypoints[i, 0])
        cy = float(all_waypoints[i, 1])
        p = float(waypoint_pressures[i])
        sigma, depth = tif_params[p]
        r2 = (xx_mask - cx) ** 2 + (yy_mask - cy) ** 2
        A[:, i] = depth * np.exp(-r2 / (2.0 * sigma ** 2))

    return A, b, initial_rms, n_waypoints, single_pass_len


def solve_and_report(A, b, initial_rms, lb, ub, label=""):
    """Solve bounded least squares and return metrics."""
    t0 = time.time()
    result = lsq_linear(A, b, bounds=(lb, ub), method="bvls", verbose=0)
    elapsed = time.time() - t0

    dwells = result.x
    residual = b - A @ dwells
    rms_final = np.sqrt(np.mean(residual ** 2))
    neg_frac = np.mean(residual < 0) * 100
    overpolish_rms = np.sqrt(np.mean(np.minimum(residual, 0) ** 2))

    at_lb = np.sum(np.isclose(dwells, lb, atol=1e-4))
    at_ub = np.sum(np.isclose(dwells, ub, atol=1e-4))
    n = len(dwells)

    return {
        "label": label,
        "rms": rms_final,
        "reduction_pct": (1 - rms_final / initial_rms) * 100,
        "overpolish_pct": neg_frac,
        "overpolish_rms": overpolish_rms,
        "at_lb": at_lb,
        "at_ub": at_ub,
        "n_waypoints": n,
        "dwell_mean": dwells.mean(),
        "dwell_std": dwells.std(),
        "elapsed": elapsed,
        "dwells": dwells,
    }


def print_result(r):
    print(f"  {r['label']:40s}  RMS={r['rms']:.4f}µm ({r['reduction_pct']:.1f}%)  "
          f"overpol={r['overpolish_pct']:.1f}%  "
          f"at_lb={r['at_lb']:4d}/{r['n_waypoints']}  "
          f"at_ub={r['at_ub']:4d}/{r['n_waypoints']}  "
          f"dwell={r['dwell_mean']:.3f}±{r['dwell_std']:.3f}  "
          f"t={r['elapsed']:.1f}s")


def main():
    config_path = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    tif_ckpt_path = str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt")
    tif_ckpt = tif_ckpt_path if Path(tif_ckpt_path).exists() else None

    config = load_config(config_path)
    target = config["surface"]["target_rms_um"]

    print("=" * 120)
    print(f"PARAMETER SWEEP — Target RMS: {target} µm")
    print("=" * 120)

    # ── Build default system ─────────────────────────────────────────
    A, b, initial_rms, n_wp, spl = build_system(config, tif_ckpt)
    print(f"\nDefault: {A.shape[0]} pixels × {n_wp} waypoints, initial RMS = {initial_rms:.4f} µm")

    # ── 0. Unconstrained least squares (theoretical best) ────────────
    print("\n" + "-" * 120)
    print("0. THEORETICAL LIMITS (what's possible with this path geometry?)")
    print("-" * 120)

    from numpy.linalg import lstsq
    x_unc, residuals, rank, sv = lstsq(A, b, rcond=None)
    res_unc = b - A @ x_unc
    rms_unc = np.sqrt(np.mean(res_unc ** 2))
    print(f"  Unconstrained LS:  RMS = {rms_unc:.6f} µm  (rank={rank}/{n_wp}, "
          f"dwell range=[{x_unc.min():.2f}, {x_unc.max():.2f}])")
    print(f"  → With this spiral path, the absolute minimum achievable RMS is {rms_unc:.6f} µm")
    if x_unc.min() < 0:
        print(f"  → BUT requires negative dwell ({x_unc.min():.2f}s) = adding material, physically impossible")
        x_clip0 = np.maximum(x_unc, 0)
        res_clip0 = b - A @ x_clip0
        rms_clip0 = np.sqrt(np.mean(res_clip0 ** 2))
        print(f"  → Clipping negatives to 0:  RMS = {rms_clip0:.4f} µm")

    # ── 1. Sweep dwell lower bound ───────────────────────────────────
    print("\n" + "-" * 120)
    print("1. DWELL LOWER BOUND SWEEP (upper bound fixed at 5.0s)")
    print("   Why: 933/1130 waypoints hit the lower bound → forced minimum removal causes overpolishing")
    print("-" * 120)

    ub_fixed = 5.0
    for lb in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        r = solve_and_report(A, b, initial_rms, lb, ub_fixed, f"lb={lb:.3f}s, ub={ub_fixed}")
        print_result(r)

    # ── 2. Sweep dwell upper bound ───────────────────────────────────
    print("\n" + "-" * 120)
    print("2. DWELL UPPER BOUND SWEEP (lower bound fixed at 0.1s)")
    print("   Why: 44/1130 waypoints hit the upper bound → might need more removal time at bumps")
    print("-" * 120)

    lb_fixed = 0.1
    for ub in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        r = solve_and_report(A, b, initial_rms, lb_fixed, ub, f"lb={lb_fixed}, ub={ub:.0f}s")
        print_result(r)

    # ── 3. Combined: best bounds ─────────────────────────────────────
    print("\n" + "-" * 120)
    print("3. COMBINED: BEST LOWER + UPPER BOUND")
    print("-" * 120)

    for lb, ub in [(0.001, 5.0), (0.001, 20.0), (0.001, 50.0), (0.001, 100.0),
                    (0.0, 5.0), (0.0, 20.0), (0.0, 50.0), (0.0, 100.0)]:
        r = solve_and_report(A, b, initial_rms, lb, ub, f"lb={lb:.3f}s, ub={ub:.0f}s")
        print_result(r)

    # ── 4. Sweep number of passes ────────────────────────────────────
    print("\n" + "-" * 120)
    print("4. NUMBER OF PASSES SWEEP (more passes = more waypoints = more DOFs)")
    print("   Fixed lb=0.001, ub=5.0")
    print("-" * 120)

    for np_ in [5, 10, 15, 20, 30, 50]:
        A2, b2, rms2, n2, spl2 = build_system(config, tif_ckpt, num_passes=np_)
        r = solve_and_report(A2, b2, rms2, 0.001, 5.0, f"passes={np_:2d} ({n2} waypoints)")
        print_result(r)

    # ── 5. Sweep spiral pitch ────────────────────────────────────────
    print("\n" + "-" * 120)
    print("5. SPIRAL PITCH SWEEP (finer pitch = denser coverage)")
    print("   Fixed 10 passes, lb=0.001, ub=5.0")
    print("-" * 120)

    for pitch in [10.0, 7.0, 5.0, 3.5, 2.5, 2.0]:
        A3, b3, rms3, n3, spl3 = build_system(config, tif_ckpt, pitch_mm=pitch)
        r = solve_and_report(A3, b3, rms3, 0.001, 5.0, f"pitch={pitch:.1f}mm ({n3} waypoints)")
        print_result(r)

    # ── 6. Best combo ────────────────────────────────────────────────
    print("\n" + "-" * 120)
    print("6. BEST COMBINATIONS (push toward target)")
    print("-" * 120)

    combos = [
        (10, 7.0, 0.001, 5.0,   "default+relaxed_lb"),
        (10, 7.0, 0.0, 50.0,    "default+wide_bounds"),
        (20, 7.0, 0.001, 5.0,   "20passes"),
        (20, 5.0, 0.001, 5.0,   "20passes+pitch5"),
        (20, 3.5, 0.001, 5.0,   "20passes+pitch3.5"),
        (30, 3.5, 0.0, 10.0,    "30passes+pitch3.5+wide"),
        (30, 2.5, 0.0, 10.0,    "30passes+pitch2.5+wide"),
        (50, 2.5, 0.0, 10.0,    "50passes+pitch2.5+wide"),
    ]
    for np_, pitch, lb, ub, tag in combos:
        A4, b4, rms4, n4, spl4 = build_system(config, tif_ckpt, num_passes=np_, pitch_mm=pitch)
        r = solve_and_report(A4, b4, rms4, lb, ub, f"{tag} ({n4}wp)")
        print_result(r)

    print("\n" + "=" * 120)
    print(f"TARGET: {target} µm")
    print("=" * 120)


if __name__ == "__main__":
    main()
