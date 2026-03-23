#!/usr/bin/env python3
"""Joint optimization of waypoint positions + dwell times via PyTorch.

Key insight: the TIF removal pattern is a Gaussian centered at the tool
position (cx, cy). This is differentiable w.r.t. cx, cy:

    TIF = depth * exp(-((xx-cx)² + (yy-cy)²) / (2σ²))
    ∂TIF/∂cx = depth * (xx-cx)/σ² * exp(...)

So we can jointly optimize:
  - dwell_i:     how long to stay at each waypoint
  - (Δx_i, Δy_i): position perturbation from the nominal spiral

With regularization to keep the path smooth and physically feasible.

Three optimization levels:
  Level 1: Dwell only (baseline, same as optimize_dwell.py)
  Level 2: Dwell + global spiral params (pitch, alpha, num_passes)
  Level 3: Dwell + per-waypoint position perturbations (this script)

Usage:
    python src/scripts/optimize_joint.py
    python src/scripts/optimize_joint.py --max-perturb 3.0 --n-iters 5000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import numpy as np
import yaml
import torch

from env.lens_surface import LensSurface
from env.spiral_path import SpiralPath
from models.tif_model import TIFModel


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_problem(config, tif_ckpt):
    """Build the optimization inputs (surface, spiral, TIF params)."""
    sc = config["surface"]
    sp = config["spiral"]

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
        pitch_mm=sp["pitch_mm"],
        min_radius_mm=3.0,
        center_sparse_power=0.7,
    )
    all_waypoints = spiral.build_multi_pass(sp["num_passes"])
    single_pass_len = spiral.get_single_pass_length()
    n_waypoints = len(all_waypoints)

    pass_pressures = sp["pass_pressures"]
    waypoint_pressures = np.array([
        pass_pressures[min(i // single_pass_len, sp["num_passes"] - 1)]
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

    sigma_per_wp = np.array([tif_params[float(p)][0] for p in waypoint_pressures])
    depth_per_wp = np.array([tif_params[float(p)][1] for p in waypoint_pressures])

    xx_mask = surface.xx.ravel()[mask_indices]
    yy_mask = surface.yy.ravel()[mask_indices]

    print(f"Problem: {n_pixels} pixels × {n_waypoints} waypoints")
    print(f"  Initial RMS: {initial_rms:.4f} µm")
    print(f"  Lens radius: {sc['lens_diameter_mm'] / 2:.1f} mm")

    return dict(
        b=b,
        xx_mask=xx_mask,
        yy_mask=yy_mask,
        mask_indices=mask_indices,
        mask=mask,
        n_pixels=n_pixels,
        n_waypoints=n_waypoints,
        initial_rms=initial_rms,
        all_waypoints=all_waypoints,
        sigma_per_wp=sigma_per_wp,
        depth_per_wp=depth_per_wp,
        waypoint_pressures=waypoint_pressures,
        single_pass_len=single_pass_len,
        surface=surface,
        config=config,
        lens_radius=sc["lens_diameter_mm"] / 2.0,
    )


def compute_removal(wx, wy, dwells, depth_t, sigma_t, xx_t, yy_t):
    """Differentiable removal computation.

    wx, wy:   (N,) waypoint positions
    dwells:   (N,) dwell times
    depth_t:  (N,) TIF peak removal rates
    sigma_t:  (N,) TIF Gaussian sigmas
    xx_t, yy_t: (M,) mask pixel coordinates

    Returns: (M,) total removal at each mask pixel
    """
    dx = xx_t.unsqueeze(1) - wx.unsqueeze(0)    # (M, N)
    dy = yy_t.unsqueeze(1) - wy.unsqueeze(0)    # (M, N)
    r2 = dx ** 2 + dy ** 2                        # (M, N)
    inv_2sig2 = 1.0 / (2.0 * sigma_t ** 2)       # (N,)
    tif = depth_t * torch.exp(-r2 * inv_2sig2)    # (M, N)
    removal = tif @ dwells                         # (M,) = sum over N
    return removal


def optimize_joint(problem, dwell_range=(0.1, 5.0), max_perturb_mm=2.0,
                   n_iters=3000, lr=0.01, w_smooth=0.5, w_coverage=0.1,
                   init_dwells=None):
    """Joint optimization of dwell times + waypoint positions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")
    print("JOINT OPTIMIZATION: dwell + path perturbation")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    print(f"  Max perturbation: ±{max_perturb_mm:.1f} mm")
    print(f"  Dwell range: [{dwell_range[0]}, {dwell_range[1]}] s")
    print(f"  Smoothness weight: {w_smooth}")
    print(f"  Coverage weight: {w_coverage}")

    b_np = problem["b"]
    xx_np = problem["xx_mask"]
    yy_np = problem["yy_mask"]
    wp_np = problem["all_waypoints"]
    sigma_np = problem["sigma_per_wp"]
    depth_np = problem["depth_per_wp"]
    initial_rms = problem["initial_rms"]
    lens_R = problem["lens_radius"]
    N = problem["n_waypoints"]

    b_t = torch.tensor(b_np, dtype=torch.float64, device=device)
    xx_t = torch.tensor(xx_np, dtype=torch.float64, device=device)
    yy_t = torch.tensor(yy_np, dtype=torch.float64, device=device)
    sigma_t = torch.tensor(sigma_np, dtype=torch.float64, device=device)
    depth_t = torch.tensor(depth_np, dtype=torch.float64, device=device)
    wp_nominal = torch.tensor(wp_np, dtype=torch.float64, device=device)  # (N, 2)

    lb, ub = dwell_range

    if init_dwells is not None:
        d_init = np.clip(init_dwells, lb + 1e-6, ub - 1e-6)
    else:
        d_init = np.full(N, (lb + ub) / 2)
    dwell_raw = torch.nn.Parameter(
        torch.tensor(np.log((d_init - lb) / (ub - d_init)), dtype=torch.float64, device=device)
    )

    delta_raw = torch.nn.Parameter(
        torch.zeros(N, 2, dtype=torch.float64, device=device)
    )

    optimizer = torch.optim.Adam([dwell_raw, delta_raw], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    best_loss = float("inf")
    best_dwells = None
    best_delta = None
    best_rms = float("inf")

    t0 = time.time()
    for it in range(n_iters):
        optimizer.zero_grad()

        dwells = lb + (ub - lb) * torch.sigmoid(dwell_raw)

        delta = max_perturb_mm * torch.tanh(delta_raw)
        wx = wp_nominal[:, 0] + delta[:, 0]
        wy = wp_nominal[:, 1] + delta[:, 1]

        r_wp = torch.sqrt(wx ** 2 + wy ** 2)
        radius_penalty = torch.mean(torch.relu(r_wp - lens_R) ** 2)

        removal = compute_removal(wx, wy, dwells, depth_t, sigma_t, xx_t, yy_t)
        error_final = b_t - removal
        rms2 = torch.mean(error_final ** 2)

        neg_err = torch.clamp(error_final, max=0.0)
        overpolish = torch.mean(neg_err ** 2)

        # Path smoothness: penalize large acceleration (second derivative)
        dx = delta[1:] - delta[:-1]                  # first derivative
        ddx = dx[1:] - dx[:-1]                       # second derivative
        smoothness = torch.mean(ddx ** 2)

        # Coverage: penalize uneven spacing (deviation from nominal spacing)
        step_vec = torch.stack([wx[1:] - wx[:-1], wy[1:] - wy[:-1]], dim=1)
        step_len = torch.sqrt((step_vec ** 2).sum(dim=1) + 1e-12)
        nom_step_vec = wp_nominal[1:] - wp_nominal[:-1]
        nom_step_len = torch.sqrt((nom_step_vec ** 2).sum(dim=1) + 1e-12)
        spacing_var = torch.mean((step_len / (nom_step_len + 1e-6) - 1.0) ** 2)

        loss = (
            5.0 * rms2 / (initial_rms ** 2 + 1e-8)
            + 10.0 * overpolish / (initial_rms ** 2 + 1e-8)
            + w_smooth * smoothness
            + w_coverage * spacing_var
            + 100.0 * radius_penalty
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        cur_rms = torch.sqrt(rms2).item()
        if cur_rms < best_rms:
            best_rms = cur_rms
            best_loss = loss.item()
            best_dwells = dwells.detach().cpu().numpy().copy()
            best_delta = delta.detach().cpu().numpy().copy()

        if (it + 1) % 500 == 0 or it == 0:
            d_mag = torch.sqrt((delta ** 2).sum(dim=1)).mean().item()
            print(f"  Iter {it + 1:5d}: loss={loss.item():.6f}, "
                  f"RMS={cur_rms:.4f}µm, "
                  f"|Δ|={d_mag:.3f}mm, "
                  f"dwell=[{dwells.min().item():.3f}, {dwells.max().item():.3f}]")

    elapsed = time.time() - t0

    final_wp = wp_np + best_delta
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Best RMS: {best_rms:.4f} µm  (reduction: {(1 - best_rms / initial_rms) * 100:.1f}%)")
    print(f"  Mean perturbation: {np.sqrt((best_delta ** 2).sum(axis=1)).mean():.3f} mm")
    print(f"  Max perturbation:  {np.sqrt((best_delta ** 2).sum(axis=1)).max():.3f} mm")
    print(f"  Dwell mean={best_dwells.mean():.3f}, range=[{best_dwells.min():.3f}, {best_dwells.max():.3f}]")

    return best_dwells, best_delta, final_wp


def optimize_dwell_only(problem, dwell_range=(0.1, 5.0)):
    """Baseline: optimize dwell only (no path perturbation) via least squares."""
    from scipy.optimize import lsq_linear

    print(f"\n{'=' * 70}")
    print("BASELINE: Dwell-only optimization (bounded least squares)")
    print(f"{'=' * 70}")

    wp_np = problem["all_waypoints"]
    b_np = problem["b"]
    xx_np = problem["xx_mask"]
    yy_np = problem["yy_mask"]
    sigma_np = problem["sigma_per_wp"]
    depth_np = problem["depth_per_wp"]
    initial_rms = problem["initial_rms"]
    N = problem["n_waypoints"]
    M = problem["n_pixels"]

    A = np.zeros((M, N), dtype=np.float64)
    for i in range(N):
        r2 = (xx_np - wp_np[i, 0]) ** 2 + (yy_np - wp_np[i, 1]) ** 2
        A[:, i] = depth_np[i] * np.exp(-r2 / (2.0 * sigma_np[i] ** 2))

    t0 = time.time()
    result = lsq_linear(A, b_np, bounds=dwell_range, method="bvls", verbose=0)
    elapsed = time.time() - t0

    dwells = result.x
    residual = b_np - A @ dwells
    rms_final = np.sqrt(np.mean(residual ** 2))

    print(f"  Time: {elapsed:.1f}s")
    print(f"  RMS: {initial_rms:.4f} → {rms_final:.4f} µm  "
          f"(reduction: {(1 - rms_final / initial_rms) * 100:.1f}%)")
    print(f"  Dwell mean={dwells.mean():.3f}, range=[{dwells.min():.3f}, {dwells.max():.3f}]")

    return dwells, rms_final


def sweep_spiral_params(config, tif_ckpt):
    """Level 2: sweep global spiral parameters."""
    from scipy.optimize import lsq_linear

    print(f"\n{'=' * 70}")
    print("LEVEL 2: Global spiral parameter sweep")
    print(f"{'=' * 70}")

    sc = config["surface"]
    sp = config["spiral"]
    surface = LensSurface(sc["grid_size"], sc["lens_diameter_mm"], sc["curvature_radius_mm"])
    surface.reset(error_scale=1.0, seed=42)

    mask = surface.mask
    mask_indices = np.where(mask.ravel())[0]
    b = surface.get_error_map().ravel()[mask_indices].astype(np.float64)
    initial_rms = surface.get_rms_error()
    xx_m = surface.xx.ravel()[mask_indices]
    yy_m = surface.yy.ravel()[mask_indices]
    M = len(mask_indices)

    tif_model = TIFModel(tif_ckpt)
    speed = sp["fixed_speed"]
    conc = config["tif"]["default_concentration"]

    results = []
    for pitch in [7.0, 5.0, 3.5]:
        for alpha in [0.5, 0.7, 1.0]:
            for n_passes in [10, 20, 30]:
                spiral = SpiralPath(
                    lens_radius_mm=sc["lens_diameter_mm"] / 2.0,
                    pitch_mm=pitch, min_radius_mm=3.0,
                    center_sparse_power=alpha,
                )
                spl = spiral.get_single_pass_length()
                wp = spiral.build_multi_pass(n_passes)
                N = len(wp)

                pp = sp["pass_pressures"]
                if n_passes > len(pp):
                    pp = pp + [pp[-1]] * (n_passes - len(pp))
                wp_press = np.array([pp[min(i // spl, n_passes - 1)] for i in range(N)])

                A = np.zeros((M, N), dtype=np.float64)
                for i in range(N):
                    p = float(wp_press[i])
                    w, d = tif_model.predict(p, speed, conc)
                    s = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                    r2 = (xx_m - wp[i, 0]) ** 2 + (yy_m - wp[i, 1]) ** 2
                    A[:, i] = d * np.exp(-r2 / (2.0 * s ** 2))

                res = lsq_linear(A, b, bounds=(0.001, 5.0), method="bvls", verbose=0)
                rms = np.sqrt(np.mean((b - A @ res.x) ** 2))
                results.append((pitch, alpha, n_passes, N, rms))
                pct = (1 - rms / initial_rms) * 100
                print(f"  pitch={pitch:4.1f}  α={alpha:.1f}  passes={n_passes:2d}  "
                      f"wp={N:5d}  RMS={rms:.4f}µm  ({pct:.1f}%)")

    results.sort(key=lambda x: x[4])
    print(f"\n  Best: pitch={results[0][0]}, α={results[0][1]}, "
          f"passes={results[0][2]} → RMS={results[0][4]:.4f}µm")
    return results


def validate_in_env(dwells, waypoints, config, tif_ckpt):
    """Replay through the actual env (uses nominal path, reports metrics)."""
    from env.polishing_env import LensPolishingEnv

    sc = config["surface"]
    sp = config["spiral"]
    env = LensPolishingEnv(
        grid_size=sc["grid_size"],
        lens_diameter_mm=sc["lens_diameter_mm"],
        curvature_radius_mm=sc["curvature_radius_mm"],
        initial_roughness_nm=sc["initial_roughness_nm"],
        target_rms_um=sc["target_rms_um"],
        reward_form=config["env"]["reward_weights"]["form"],
        reward_roughness=config["env"]["reward_weights"]["roughness"],
        reward_bonus=config["env"]["reward_weights"]["completion_bonus"],
        num_passes=sp["num_passes"],
        pass_pressures=sp["pass_pressures"],
        fixed_speed=sp["fixed_speed"],
        concentration=config["tif"]["default_concentration"],
        dwell_range=tuple(config["env"]["action_ranges"]["dwell_time"]),
        pitch_mm=sp["pitch_mm"],
        tif_checkpoint=tif_ckpt,
    )
    obs, _ = env.reset()
    lb, ub = env.dt_range
    total_reward = 0.0
    for i, dwell in enumerate(dwells[:env.max_steps]):
        a = np.clip(2.0 * (dwell - lb) / (ub - lb) - 1.0, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(np.array([a], dtype=np.float32))
        total_reward += reward
        if terminated or truncated:
            break
    return info, total_reward


def main():
    parser = argparse.ArgumentParser(description="Joint path + dwell optimization")
    parser.add_argument("--config",
                        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--tif-checkpoint",
                        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"))
    parser.add_argument("--output",
                        default=str(Path(__file__).resolve().parents[1] / "results" / "joint_optimized.npz"))
    parser.add_argument("--max-perturb", type=float, default=3.0,
                        help="Max position perturbation in mm")
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--w-smooth", type=float, default=0.5)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    tif_ckpt = args.tif_checkpoint if Path(args.tif_checkpoint).exists() else None
    dwell_range = tuple(config["env"]["action_ranges"]["dwell_time"])

    problem = build_problem(config, tif_ckpt)

    # ── Level 1: Dwell only ──────────────────────────────────────────
    dwells_baseline, rms_baseline = optimize_dwell_only(problem, dwell_range)

    # ── Level 2: Global spiral param sweep ───────────────────────────
    if not args.skip_sweep:
        sweep_spiral_params(config, tif_ckpt)

    # ── Level 3: Joint dwell + path perturbation ─────────────────────
    dwells_joint, delta_joint, waypoints_joint = optimize_joint(
        problem,
        dwell_range=dwell_range,
        max_perturb_mm=args.max_perturb,
        n_iters=args.n_iters,
        lr=args.lr,
        w_smooth=args.w_smooth,
        init_dwells=dwells_baseline,
    )

    # ── Summary ──────────────────────────────────────────────────────
    rms_joint = np.sqrt(np.mean(
        (problem["b"] - compute_removal_np(
            waypoints_joint, dwells_joint, problem)) ** 2
    ))

    target = config["surface"]["target_rms_um"]
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Target RMS:         {target:.4f} µm")
    print(f"  Initial RMS:        {problem['initial_rms']:.4f} µm")
    print(f"  Dwell-only:         {rms_baseline:.4f} µm")
    print(f"  Joint (dwell+path): {rms_joint:.4f} µm")
    print(f"  Improvement:        {(1 - rms_joint / rms_baseline) * 100:.1f}% over dwell-only")

    # ── Save ─────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        dwells_baseline=dwells_baseline,
        dwells_joint=dwells_joint,
        delta_joint=delta_joint,
        waypoints_nominal=problem["all_waypoints"],
        waypoints_joint=waypoints_joint,
        initial_rms=problem["initial_rms"],
        rms_baseline=rms_baseline,
        rms_joint=rms_joint,
    )
    print(f"  Saved to {out}")

    # ── Visualize path perturbation ──────────────────────────────────
    try:
        visualize(problem, dwells_baseline, dwells_joint, delta_joint, out.parent)
    except Exception as e:
        print(f"  Visualization skipped: {e}")


def compute_removal_np(waypoints, dwells, problem):
    """NumPy version for validation."""
    xx = problem["xx_mask"]
    yy = problem["yy_mask"]
    sigma = problem["sigma_per_wp"]
    depth = problem["depth_per_wp"]
    removal = np.zeros_like(xx)
    for i in range(len(dwells)):
        r2 = (xx - waypoints[i, 0]) ** 2 + (yy - waypoints[i, 1]) ** 2
        removal += depth[i] * np.exp(-r2 / (2.0 * sigma[i] ** 2)) * dwells[i]
    return removal


def visualize(problem, dwells_bl, dwells_jt, delta, out_dir):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    wp_nom = problem["all_waypoints"]
    wp_jt = wp_nom + delta
    lens_R = problem["lens_radius"]
    mask = problem["mask"]
    surface = problem["surface"]
    b = problem["b"]
    mi = problem["mask_indices"]
    gs = surface.grid_size

    rem_bl = compute_removal_np(wp_nom, dwells_bl, problem)
    rem_jt = compute_removal_np(wp_jt, dwells_jt, problem)

    err_bl = np.zeros(gs * gs)
    err_bl[mi] = b - rem_bl
    err_bl = err_bl.reshape(gs, gs)

    err_jt = np.zeros(gs * gs)
    err_jt[mi] = b - rem_jt
    err_jt = err_jt.reshape(gs, gs)

    err_range = max(np.abs(err_bl[mask]).max(), np.abs(err_jt[mask]).max())
    err_range = min(err_range, 3.0)
    norm = TwoSlopeNorm(vmin=-err_range, vcenter=0, vmax=err_range)

    half = lens_R
    extent = [-half, half, -half, half]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13), facecolor="#0a0a1a")
    for ax in axes.flat:
        ax.set_facecolor("#0f0f2a")
        for s in ax.spines.values():
            s.set_color("#333366")

    # Row 0: paths + error maps
    ax = axes[0, 0]
    ax.plot(wp_nom[:, 0], wp_nom[:, 1], ".", color="#ff5533", ms=1, alpha=0.5, label="Nominal")
    ax.plot(wp_jt[:, 0], wp_jt[:, 1], ".", color="#33ff88", ms=1, alpha=0.5, label="Optimized")
    ax.add_patch(plt.Circle((0, 0), lens_R, fill=False, color="#555588", lw=1, ls="--"))
    ax.set_xlim(-lens_R * 1.1, lens_R * 1.1)
    ax.set_ylim(-lens_R * 1.1, lens_R * 1.1)
    ax.set_aspect("equal")
    ax.set_title("Path: Nominal vs Optimized", color="white", fontsize=14)
    ax.legend(fontsize=10, facecolor="#1a1a3a", edgecolor="#333366", labelcolor="white")
    ax.tick_params(colors="#888888")

    ax = axes[0, 1]
    im = ax.imshow(np.ma.array(err_bl, mask=~mask), cmap="RdBu_r", norm=norm,
                    extent=extent, origin="lower")
    ax.set_title("Residual: Dwell-only", color="white", fontsize=14)
    ax.tick_params(colors="#888888")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 2]
    im = ax.imshow(np.ma.array(err_jt, mask=~mask), cmap="RdBu_r", norm=norm,
                    extent=extent, origin="lower")
    ax.set_title("Residual: Joint (dwell+path)", color="white", fontsize=14)
    ax.tick_params(colors="#888888")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 1: perturbation magnitude + dwell comparison + quiver
    perturb_mag = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
    ax = axes[1, 0]
    sc_plot = ax.scatter(wp_nom[:, 0], wp_nom[:, 1], c=perturb_mag, s=3,
                          cmap="plasma", vmin=0, vmax=perturb_mag.max())
    ax.add_patch(plt.Circle((0, 0), lens_R, fill=False, color="#555588", lw=1, ls="--"))
    ax.set_xlim(-lens_R * 1.1, lens_R * 1.1)
    ax.set_ylim(-lens_R * 1.1, lens_R * 1.1)
    ax.set_aspect("equal")
    ax.set_title("Perturbation magnitude (mm)", color="white", fontsize=14)
    ax.tick_params(colors="#888888")
    plt.colorbar(sc_plot, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    steps = np.arange(len(dwells_bl))
    ax.plot(steps, dwells_bl, "-", color="#ff5533", lw=1, alpha=0.7, label="Dwell-only")
    ax.plot(steps, dwells_jt, "-", color="#33ff88", lw=1, alpha=0.7, label="Joint")
    ax.set_title("Dwell time comparison", color="white", fontsize=14)
    ax.set_xlabel("Waypoint", color="#888888")
    ax.set_ylabel("seconds", color="#888888")
    ax.tick_params(colors="#888888")
    ax.grid(True, alpha=0.1)
    ax.legend(fontsize=10, facecolor="#1a1a3a", edgecolor="#333366", labelcolor="white")

    ax = axes[1, 2]
    skip = max(1, len(delta) // 200)
    ax.quiver(wp_nom[::skip, 0], wp_nom[::skip, 1],
              delta[::skip, 0], delta[::skip, 1],
              perturb_mag[::skip], cmap="plasma", scale=30, width=0.004)
    ax.add_patch(plt.Circle((0, 0), lens_R, fill=False, color="#555588", lw=1, ls="--"))
    ax.set_xlim(-lens_R * 1.1, lens_R * 1.1)
    ax.set_ylim(-lens_R * 1.1, lens_R * 1.1)
    ax.set_aspect("equal")
    ax.set_title("Perturbation vectors", color="white", fontsize=14)
    ax.tick_params(colors="#888888")

    fig.suptitle("Joint Path + Dwell Optimization", fontsize=18, color="white", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = out_dir / "joint_optimization.png"
    fig.savefig(fig_path, dpi=120, facecolor="#0a0a1a")
    plt.close()
    print(f"  Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
