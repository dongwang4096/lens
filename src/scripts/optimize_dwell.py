#!/usr/bin/env python3
"""Optimal dwell-time trajectory for fixed-surface lens polishing.

Key insight: the TIF removal does NOT depend on the current surface state,
so the final form error is LINEAR in the dwell-time vector:

    error_final = error_0 - A @ dwells

where A[:,i] is the TIF removal pattern for waypoint i (on mask pixels).
This is a bounded least-squares problem (convex, globally optimal):

    minimize ||b - A x||²   s.t.  0.1 <= x_i <= 5.0

Roughness has multiplicative coupling (Ra *= 1 - decay) but is secondary.
An optional PyTorch refinement step jointly optimizes form + roughness.

Usage:
    python src/scripts/optimize_dwell.py
    python src/scripts/optimize_dwell.py --skip-pytorch
    python src/scripts/optimize_dwell.py --pytorch-iters 5000 --pytorch-lr 0.005
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import numpy as np
import yaml
from scipy.optimize import lsq_linear

import torch

from env.lens_surface import LensSurface
from env.roughness_model import RoughnessModel
from env.spiral_path import SpiralPath
from models.tif_model import TIFModel


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Build the linear system ──────────────────────────────────────────────────

def build_problem(config: dict, tif_checkpoint: str | None):
    """Construct A matrix and b vector for the least-squares formulation.

    Returns a dict with everything needed by both solvers.
    """
    sc = config["surface"]
    sp = config["spiral"]

    surface = LensSurface(sc["grid_size"], sc["lens_diameter_mm"], sc["curvature_radius_mm"])
    surface.reset(error_scale=1.0, seed=42)

    mask = surface.mask
    mask_flat = mask.ravel()
    mask_indices = np.where(mask_flat)[0]
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
    single_pass_len = spiral.get_single_pass_length()
    all_waypoints = spiral.build_multi_pass(sp["num_passes"])
    n_waypoints = len(all_waypoints)

    pass_pressures = sp["pass_pressures"]
    waypoint_pressures = np.array([
        pass_pressures[min(i // single_pass_len, sp["num_passes"] - 1)]
        for i in range(n_waypoints)
    ])

    tif_model = TIFModel(tif_checkpoint)
    speed = sp["fixed_speed"]
    conc = config["tif"]["default_concentration"]

    unique_pressures = np.unique(waypoint_pressures)
    tif_params = {}
    for p in unique_pressures:
        width, depth = tif_model.predict(float(p), speed, conc)
        sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        tif_params[float(p)] = (sigma, depth)

    print(f"Problem size: {n_pixels} mask pixels × {n_waypoints} waypoints")
    print(f"  Surface grid: {sc['grid_size']}×{sc['grid_size']}")
    print(f"  Initial RMS: {initial_rms:.4f} µm")
    print(f"  Passes: {sp['num_passes']}, waypoints/pass: {single_pass_len}")
    print(f"  Unique pressures: {sorted(unique_pressures)}")
    print(f"  TIF params per pressure:")
    for p in sorted(tif_params.keys()):
        s, d = tif_params[p]
        print(f"    P={p:5.1f}N → σ={s:.3f}mm, depth={d:.4f}µm/s")

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

    mem_mb = A.nbytes / 1e6
    print(f"  A matrix memory: {mem_mb:.1f} MB")

    roughness_model = RoughnessModel(sc["grid_size"], sc["initial_roughness_nm"])
    roughness_model.reset(seed=43)
    rough_init = roughness_model.roughness.ravel()[mask_indices].copy()

    qf_map = {}
    for p in unique_pressures:
        qf_map[float(p)] = roughness_model.quality_factor(float(p), speed, conc)

    qf_per_wp = np.array([qf_map[float(p)] for p in waypoint_pressures])

    A_col_max = A.max(axis=0, keepdims=True)
    A_col_max = np.maximum(A_col_max, 1e-12)
    footprint = A / A_col_max

    return dict(
        A=A,
        b=b,
        mask_indices=mask_indices,
        n_pixels=n_pixels,
        n_waypoints=n_waypoints,
        initial_rms=initial_rms,
        surface=surface,
        spiral=spiral,
        tif_model=tif_model,
        all_waypoints=all_waypoints,
        waypoint_pressures=waypoint_pressures,
        single_pass_len=single_pass_len,
        rough_init=rough_init,
        qf_per_wp=qf_per_wp,
        footprint=footprint,
        config=config,
    )


# ── Solver 1: Bounded Least Squares (scipy) ─────────────────────────────────

def solve_least_squares(problem: dict, dwell_range: tuple[float, float]) -> np.ndarray:
    """Globally optimal dwell sequence for form-error minimization."""
    A, b = problem["A"], problem["b"]
    initial_rms = problem["initial_rms"]

    print("\n" + "=" * 70)
    print("STAGE 1: Bounded Least Squares (form error only)")
    print("=" * 70)
    print(f"  minimize ||b - A x||²  s.t.  {dwell_range[0]} ≤ x_i ≤ {dwell_range[1]}")

    t0 = time.time()
    result = lsq_linear(
        A, b,
        bounds=(dwell_range[0], dwell_range[1]),
        method="bvls",
        verbose=2,
    )
    elapsed = time.time() - t0

    dwells = result.x
    residual = b - A @ dwells
    rms_final = np.sqrt(np.mean(residual ** 2))

    neg_mask = residual < 0
    neg_frac = np.mean(neg_mask) * 100
    overpolish_rms = np.sqrt(np.mean(np.minimum(residual, 0) ** 2))

    print(f"\n  Solver status: {result.message}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Initial RMS:      {initial_rms:.4f} µm")
    print(f"  Optimized RMS:    {rms_final:.4f} µm")
    print(f"  Reduction:        {(1 - rms_final / initial_rms) * 100:.1f}%")
    print(f"  Overpolished:     {neg_frac:.1f}% of pixels")
    print(f"  Overpolish RMS:   {overpolish_rms:.4f} µm")
    print(f"  Dwell mean={dwells.mean():.3f}, std={dwells.std():.3f}, "
          f"min={dwells.min():.3f}, max={dwells.max():.3f}")

    at_lb = np.sum(np.isclose(dwells, dwell_range[0], atol=1e-4))
    at_ub = np.sum(np.isclose(dwells, dwell_range[1], atol=1e-4))
    print(f"  At lower bound: {at_lb}/{len(dwells)}, at upper bound: {at_ub}/{len(dwells)}")

    return dwells


# ── Solver 2: PyTorch gradient descent (form + roughness) ───────────────────

def solve_pytorch(
    problem: dict,
    initial_dwells: np.ndarray,
    dwell_range: tuple[float, float],
    n_iters: int = 2000,
    lr: float = 0.01,
    w_form: float = 5.0,
    w_rough: float = 1.0,
    w_overpolish: float = 10.0,
) -> np.ndarray:
    """Refine dwells by jointly optimizing form error + roughness via autograd."""
    print("\n" + "=" * 70)
    print("STAGE 2: PyTorch Gradient Descent (form + roughness)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    A_np = problem["A"]
    b_np = problem["b"]
    initial_rms = problem["initial_rms"]
    rough_init_np = problem["rough_init"]
    qf_np = problem["qf_per_wp"]
    footprint_np = problem["footprint"]
    config = problem["config"]
    initial_ra = config["surface"]["initial_roughness_nm"]

    lb, ub = dwell_range

    init_clamped = np.clip(initial_dwells, lb + 1e-6, ub - 1e-6)
    raw_init = np.log((init_clamped - lb) / (ub - init_clamped))
    raw = torch.nn.Parameter(torch.tensor(raw_init, dtype=torch.float64, device=device))

    A_t = torch.tensor(A_np, dtype=torch.float64, device=device)
    b_t = torch.tensor(b_np, dtype=torch.float64, device=device)
    rough_init_t = torch.tensor(rough_init_np, dtype=torch.float64, device=device)
    qf_t = torch.tensor(qf_np, dtype=torch.float64, device=device)
    fp_t = torch.tensor(footprint_np, dtype=torch.float64, device=device)

    optimizer = torch.optim.Adam([raw], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    best_loss = float("inf")
    best_dwells = initial_dwells.copy()

    print(f"  Iterations: {n_iters}, lr={lr}")
    print(f"  Weights: form={w_form}, roughness={w_rough}, overpolish={w_overpolish}")
    print(f"  Initial Ra: {initial_ra:.1f} nm")

    t0 = time.time()
    for it in range(n_iters):
        optimizer.zero_grad()

        dwells = lb + (ub - lb) * torch.sigmoid(raw)

        error_final = b_t - A_t @ dwells
        rms2 = torch.mean(error_final ** 2)

        neg_err = torch.clamp(error_final, max=0.0)
        overpolish = torch.mean(neg_err ** 2)

        decay_raw = 0.05 * qf_t.unsqueeze(0) * dwells.unsqueeze(0) * fp_t
        decay = torch.clamp(decay_raw, 0.0, 0.9)
        log_survival = torch.sum(torch.log1p(-decay), dim=1)
        rough_final = rough_init_t * torch.exp(log_survival)
        rough_final = torch.clamp(rough_final, 5.0, 500.0)
        mean_rough = torch.mean(rough_final)

        loss = (
            w_form * rms2 / (initial_rms ** 2 + 1e-8)
            + w_rough * (mean_rough / initial_ra)
            + w_overpolish * overpolish / (initial_rms ** 2 + 1e-8)
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_dwells = dwells.detach().cpu().numpy().copy()

        if (it + 1) % 500 == 0 or it == 0:
            rms_val = torch.sqrt(rms2).item()
            r_val = mean_rough.item()
            print(f"  Iter {it + 1:5d}: loss={cur_loss:.6f}, "
                  f"RMS={rms_val:.4f}µm, Ra={r_val:.1f}nm, "
                  f"dwell=[{dwells.min().item():.3f}, {dwells.max().item():.3f}]")

    elapsed = time.time() - t0

    residual = b_np - A_np @ best_dwells
    rms_final = np.sqrt(np.mean(residual ** 2))
    neg_frac = np.mean(residual < 0) * 100

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Final RMS: {rms_final:.4f} µm  (reduction: {(1 - rms_final / initial_rms) * 100:.1f}%)")
    print(f"  Overpolished: {neg_frac:.1f}% of pixels")
    print(f"  Dwell mean={best_dwells.mean():.3f}, std={best_dwells.std():.3f}")

    return best_dwells


# ── Validation ───────────────────────────────────────────────────────────────

def validate_in_env(dwells: np.ndarray, config: dict, tif_checkpoint: str | None) -> dict:
    """Replay the dwell sequence through the actual Gymnasium environment."""
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
        tif_checkpoint=tif_checkpoint,
    )

    obs, _ = env.reset()
    total_reward = 0.0
    lb, ub = env.dt_range

    for i, dwell in enumerate(dwells):
        a = 2.0 * (dwell - lb) / (ub - lb) - 1.0
        a = np.clip(a, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(np.array([a], dtype=np.float32))
        total_reward += reward
        if terminated or truncated:
            break

    info["total_reward"] = total_reward
    info["steps_used"] = i + 1
    info["terminated_early"] = terminated
    return info


def run_constant_baseline(config: dict, tif_checkpoint: str | None, dwell_val: float) -> dict:
    """Baseline: constant dwell time for all waypoints."""
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
        tif_checkpoint=tif_checkpoint,
    )

    obs, _ = env.reset()
    total_reward = 0.0
    lb, ub = env.dt_range
    a = 2.0 * (dwell_val - lb) / (ub - lb) - 1.0
    a = np.clip(a, -1.0, 1.0)

    for i in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(np.array([a], dtype=np.float32))
        total_reward += reward
        if terminated or truncated:
            break

    info["total_reward"] = total_reward
    info["steps_used"] = i + 1
    return info


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimal dwell-time trajectory via convex optimization + gradient descent"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--tif-checkpoint",
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "results" / "optimal_dwells.npz"),
    )
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--pytorch-iters", type=int, default=3000)
    parser.add_argument("--pytorch-lr", type=float, default=0.01)
    args = parser.parse_args()

    config = load_config(args.config)
    dwell_range = tuple(config["env"]["action_ranges"]["dwell_time"])
    tif_ckpt = args.tif_checkpoint if Path(args.tif_checkpoint).exists() else None

    if tif_ckpt:
        print(f"Using TIF checkpoint: {tif_ckpt}")
    else:
        print("No TIF checkpoint found — using default-initialized TIF network")

    # ── Build problem ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING OPTIMIZATION PROBLEM")
    print("=" * 70)
    problem = build_problem(config, tif_ckpt)

    # ── Stage 1: Bounded Least Squares ───────────────────────────────
    dwells_lsq = solve_least_squares(problem, dwell_range)

    # ── Stage 2: PyTorch refinement ──────────────────────────────────
    dwells_final = dwells_lsq
    if not args.skip_pytorch:
        dwells_final = solve_pytorch(
            problem,
            initial_dwells=dwells_lsq,
            dwell_range=dwell_range,
            n_iters=args.pytorch_iters,
            lr=args.pytorch_lr,
            w_form=config["env"]["reward_weights"]["form"],
            w_rough=config["env"]["reward_weights"]["roughness"],
            w_overpolish=10.0,
        )

    # ── Validate in environment ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION: Replay through Gymnasium environment")
    print("=" * 70)

    print("\n  [Baseline] constant dwell = 1.0s ...")
    bl_1 = run_constant_baseline(config, tif_ckpt, 1.0)
    print(f"    RMS={bl_1['rms_error']:.4f}µm, Ra={bl_1['mean_roughness']:.1f}nm, "
          f"reward={bl_1['total_reward']:.2f}")

    print("\n  [Baseline] constant dwell = 2.5s ...")
    bl_mid = run_constant_baseline(config, tif_ckpt, 2.5)
    print(f"    RMS={bl_mid['rms_error']:.4f}µm, Ra={bl_mid['mean_roughness']:.1f}nm, "
          f"reward={bl_mid['total_reward']:.2f}")

    print("\n  [Baseline] constant dwell = 5.0s (max) ...")
    bl_max = run_constant_baseline(config, tif_ckpt, 5.0)
    print(f"    RMS={bl_max['rms_error']:.4f}µm, Ra={bl_max['mean_roughness']:.1f}nm, "
          f"reward={bl_max['total_reward']:.2f}")

    print("\n  [Least Squares] optimized (form only) ...")
    val_lsq = validate_in_env(dwells_lsq, config, tif_ckpt)
    lsq_status = "TERMINATED early!" if val_lsq.get("terminated_early") else f"steps={val_lsq['steps_used']}"
    print(f"    RMS={val_lsq['rms_error']:.4f}µm, Ra={val_lsq['mean_roughness']:.1f}nm, "
          f"reward={val_lsq['total_reward']:.2f}, {lsq_status}")

    if not args.skip_pytorch:
        print("\n  [PyTorch] optimized (form + roughness) ...")
        val_pt = validate_in_env(dwells_final, config, tif_ckpt)
        pt_status = "TERMINATED early!" if val_pt.get("terminated_early") else f"steps={val_pt['steps_used']}"
        print(f"    RMS={val_pt['rms_error']:.4f}µm, Ra={val_pt['mean_roughness']:.1f}nm, "
              f"reward={val_pt['total_reward']:.2f}, {pt_status}")

    # ── Summary ──────────────────────────────────────────────────────
    target = config["surface"]["target_rms_um"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Target RMS:          {target:.4f} µm")
    print(f"  Initial RMS:         {problem['initial_rms']:.4f} µm")
    print(f"  Const dwell=1.0:     RMS={bl_1['rms_error']:.4f}µm  Ra={bl_1['mean_roughness']:.1f}nm")
    print(f"  Const dwell=2.5:     RMS={bl_mid['rms_error']:.4f}µm  Ra={bl_mid['mean_roughness']:.1f}nm")
    print(f"  Const dwell=5.0:     RMS={bl_max['rms_error']:.4f}µm  Ra={bl_max['mean_roughness']:.1f}nm")
    print(f"  Least Squares:       RMS={val_lsq['rms_error']:.4f}µm  Ra={val_lsq['mean_roughness']:.1f}nm")
    if not args.skip_pytorch:
        print(f"  PyTorch (form+Ra):   RMS={val_pt['rms_error']:.4f}µm  Ra={val_pt['mean_roughness']:.1f}nm")

    # ── Save results ─────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        dwells_lsq=dwells_lsq,
        dwells_final=dwells_final,
        waypoints=problem["all_waypoints"],
        waypoint_pressures=problem["waypoint_pressures"],
        initial_rms=problem["initial_rms"],
        b=problem["b"],
    )
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
