#!/usr/bin/env python3
"""Free continuous path optimization with bounded least-squares dwell solves.

This script generalizes the fixed-spiral dwell optimization:

  - A continuous tool path is represented by a small set of control points.
  - Catmull-Rom interpolation samples those controls into polishing waypoints.
  - For each current path, dwell time is solved by bounded least squares.
  - Between LS solves, PyTorch updates the control points with smooth-turn
    penalties so the macroscopic path avoids sharp bends.

The output is compatible with animate_compare.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from scipy.optimize import lsq_linear

from env.lens_surface import LensSurface
from env.spiral_path import SpiralPath
from models.tif_model import TIFModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a polyline to approximately equal arc-length spacing."""
    if len(points) == n_points:
        return points.astype(np.float64, copy=True)

    seg = np.diff(points, axis=0)
    seg_len = np.sqrt((seg**2).sum(axis=1))
    dist = np.concatenate([[0.0], np.cumsum(seg_len)])
    if dist[-1] <= 1e-12:
        return np.repeat(points[:1], n_points, axis=0).astype(np.float64)

    target = np.linspace(0.0, dist[-1], n_points)
    out = np.empty((n_points, 2), dtype=np.float64)
    out[:, 0] = np.interp(target, dist, points[:, 0])
    out[:, 1] = np.interp(target, dist, points[:, 1])
    return out


def make_smooth_spiral_path(
    n_points: int,
    lens_radius_mm: float,
    n_turns: float,
    min_radius_mm: float,
    max_radius_frac: float,
) -> np.ndarray:
    """Generate a single-direction low-curvature spiral.

    Unlike the historical multi-pass spiral, this path has no radial direction
    reversals, so it is suitable as a hard-constraint-feasible initializer.
    """
    r0 = max(0.0, min_radius_mm)
    r1 = lens_radius_mm * max_radius_frac
    if r0 >= r1:
        r0 = 0.5 * r1

    t = np.linspace(0.0, 1.0, n_points)
    theta = 2.0 * np.pi * n_turns * t
    radius = r0 + (r1 - r0) * t
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y]).astype(np.float64)


def make_initial_waypoints(args, problem: dict, n_waypoints: int, lens_radius: float) -> np.ndarray:
    if args.init_path == "nominal_spiral":
        args.effective_init_min_radius_mm = float("nan")
        return problem["nominal_wp"].copy()
    if args.init_path == "smooth_spiral":
        min_radius = args.init_min_radius_mm
        if min_radius is None:
            min_radius = 0.0
        args.effective_init_min_radius_mm = min_radius
        return make_smooth_spiral_path(
            n_waypoints,
            lens_radius_mm=lens_radius,
            n_turns=args.init_turns,
            min_radius_mm=min_radius,
            max_radius_frac=args.init_max_radius_frac,
        )
    raise ValueError(f"Unknown init path: {args.init_path}")


def sample_catmull_rom(control: torch.Tensor, n_points: int) -> torch.Tensor:
    """Sample an open Catmull-Rom spline from control points."""
    k = control.shape[0]
    if k < 4:
        raise ValueError("At least 4 control points are required")

    device = control.device
    dtype = control.dtype
    u = torch.linspace(0.0, float(k - 1), n_points, device=device, dtype=dtype)
    seg = torch.floor(u).long().clamp(0, k - 2)
    t = (u - seg.to(dtype)).unsqueeze(1)

    i0 = (seg - 1).clamp(0, k - 1)
    i1 = seg
    i2 = (seg + 1).clamp(0, k - 1)
    i3 = (seg + 2).clamp(0, k - 1)

    p0 = control[i0]
    p1 = control[i1]
    p2 = control[i2]
    p3 = control[i3]

    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def build_problem(config: dict, tif_checkpoint: str | None, n_waypoints: int | None):
    """Build static surface, TIF arrays, and a stable initial path."""
    sc = config["surface"]
    sp = config["spiral"]

    surface = LensSurface(sc["grid_size"], sc["lens_diameter_mm"], sc["curvature_radius_mm"])
    surface.reset(error_scale=1.0, seed=42)
    mask = surface.mask
    mask_indices = np.where(mask.ravel())[0]

    b = surface.get_error_map().ravel()[mask_indices].astype(np.float64)
    initial_rms = surface.get_rms_error()
    xx_mask = surface.xx.ravel()[mask_indices].astype(np.float64)
    yy_mask = surface.yy.ravel()[mask_indices].astype(np.float64)

    spiral = SpiralPath(
        lens_radius_mm=sc["lens_diameter_mm"] / 2.0,
        pitch_mm=sp["pitch_mm"],
        min_radius_mm=3.0,
        center_sparse_power=0.7,
    )
    nominal_raw = spiral.build_multi_pass(sp["num_passes"]).astype(np.float64)
    if n_waypoints is None:
        n_waypoints = len(nominal_raw)
    nominal_wp = resample_polyline(nominal_raw, n_waypoints)

    pass_pressures = list(sp["pass_pressures"])
    if sp["num_passes"] > len(pass_pressures):
        pass_pressures.extend([pass_pressures[-1]] * (sp["num_passes"] - len(pass_pressures)))

    progress = np.linspace(0.0, sp["num_passes"], n_waypoints, endpoint=False)
    pass_idx = np.floor(progress).astype(int).clip(0, sp["num_passes"] - 1)
    waypoint_pressures = np.array([pass_pressures[i] for i in pass_idx], dtype=np.float64)

    tif_model = TIFModel(tif_checkpoint)
    speed = sp["fixed_speed"]
    conc = config["tif"]["default_concentration"]

    sigma_per_wp = np.zeros(n_waypoints, dtype=np.float64)
    depth_per_wp = np.zeros(n_waypoints, dtype=np.float64)
    tif_cache = {}
    for i, pressure in enumerate(waypoint_pressures):
        key = float(pressure)
        if key not in tif_cache:
            width, depth = tif_model.predict(key, speed, conc)
            sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            tif_cache[key] = (sigma, depth)
        sigma_per_wp[i], depth_per_wp[i] = tif_cache[key]

    return {
        "b": b,
        "xx_mask": xx_mask,
        "yy_mask": yy_mask,
        "mask_indices": mask_indices,
        "mask": mask,
        "surface": surface,
        "initial_rms": initial_rms,
        "nominal_wp": nominal_wp,
        "waypoint_pressures": waypoint_pressures,
        "sigma_per_wp": sigma_per_wp,
        "depth_per_wp": depth_per_wp,
        "lens_radius": sc["lens_diameter_mm"] / 2.0,
        "n_waypoints": n_waypoints,
        "config": config,
    }


def build_a_matrix(waypoints: np.ndarray, problem: dict) -> np.ndarray:
    """Build the linear removal matrix for fixed waypoint locations."""
    xx = problem["xx_mask"]
    yy = problem["yy_mask"]
    sigma = problem["sigma_per_wp"]
    depth = problem["depth_per_wp"]
    a = np.empty((len(xx), len(waypoints)), dtype=np.float64)
    for i, (cx, cy) in enumerate(waypoints):
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        a[:, i] = depth[i] * np.exp(-r2 / (2.0 * sigma[i] ** 2))
    return a


def waypoint_step_lengths_np(waypoints: np.ndarray) -> np.ndarray:
    """Return one path-distance value per waypoint/dwell variable."""
    step = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    if len(step) == 0:
        return np.ones(len(waypoints), dtype=np.float64)
    return np.concatenate([[step[0]], step]).astype(np.float64)


def dwell_bounds_for_speed(
    waypoints: np.ndarray,
    dwell_range: tuple[float, float],
    speed_min_mm_s: float | None,
    speed_max_mm_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine global dwell bounds with feed-speed bounds.

    Uses effective feed speed v_i = ds_i / dwell_i, where ds_i is the
    local path distance associated with waypoint i.
    """
    ds = waypoint_step_lengths_np(waypoints)
    lb = np.full(len(waypoints), dwell_range[0], dtype=np.float64)
    ub = np.full(len(waypoints), dwell_range[1], dtype=np.float64)

    if speed_max_mm_s is not None and speed_max_mm_s > 0:
        lb = np.maximum(lb, ds / speed_max_mm_s)
    if speed_min_mm_s is not None and speed_min_mm_s > 0:
        ub = np.minimum(ub, ds / speed_min_mm_s)

    infeasible = lb >= ub
    if np.any(infeasible):
        # Keep the LS problem feasible while staying as close as possible to
        # the requested feed-speed envelope.
        mid = 0.5 * (lb[infeasible] + ub[infeasible])
        lb[infeasible] = np.maximum(dwell_range[0], mid * 0.999)
        ub[infeasible] = np.minimum(dwell_range[1], mid * 1.001)
        still_bad = lb >= ub
        if np.any(still_bad):
            lb[still_bad] = dwell_range[0]
            ub[still_bad] = dwell_range[1]

    return lb, ub


def motion_metrics_np(waypoints: np.ndarray, dwells: np.ndarray) -> dict[str, np.ndarray | float]:
    ds = waypoint_step_lengths_np(waypoints)
    speed = ds / np.maximum(dwells, 1e-8)
    if len(speed) > 1:
        dt_mid = 0.5 * (dwells[1:] + dwells[:-1])
        accel = np.diff(speed) / np.maximum(dt_mid, 1e-8)
    else:
        accel = np.zeros(0, dtype=np.float64)
    return {
        "step_length": ds,
        "speed": speed,
        "acceleration": accel,
        "speed_min": float(speed.min()) if len(speed) else 0.0,
        "speed_max": float(speed.max()) if len(speed) else 0.0,
        "speed_mean": float(speed.mean()) if len(speed) else 0.0,
        "accel_abs_max": float(np.max(np.abs(accel))) if len(accel) else 0.0,
        "accel_abs_mean": float(np.mean(np.abs(accel))) if len(accel) else 0.0,
    }


def path_geometry_metrics_np(
    waypoints: np.ndarray,
    lens_radius: float,
    max_turn_angle_deg: float | None,
    min_turn_radius_mm: float | None,
    tol: float = 1e-6,
) -> dict[str, float | bool]:
    step = np.diff(waypoints, axis=0)
    step_len = np.linalg.norm(step, axis=1)
    if len(step_len) >= 2:
        v1 = step[:-1]
        v2 = step[1:]
        denom = np.maximum(step_len[:-1] * step_len[1:], 1e-12)
        cos_turn = np.sum(v1 * v2, axis=1) / denom
        cos_turn = np.clip(cos_turn, -1.0, 1.0)
        angles = np.arccos(cos_turn)
        local_ds = 0.5 * (step_len[:-1] + step_len[1:])
        curvature = angles / np.maximum(local_ds, 1e-12)
    else:
        angles = np.zeros(0, dtype=np.float64)
        curvature = np.zeros(0, dtype=np.float64)

    radius = np.linalg.norm(waypoints, axis=1)
    max_turn = float(np.rad2deg(angles.max())) if len(angles) else 0.0
    turn_radius = 1.0 / np.maximum(curvature, 1e-12)
    min_turn_radius = float(turn_radius.min()) if len(turn_radius) else float("inf")
    max_radius = float(radius.max()) if len(radius) else 0.0

    turn_violation = 0.0
    if max_turn_angle_deg is not None and max_turn_angle_deg > 0 and len(angles):
        turn_violation = float(np.maximum(np.rad2deg(angles) - max_turn_angle_deg, 0.0).max())
    turn_radius_violation = 0.0
    if min_turn_radius_mm is not None and min_turn_radius_mm > 0 and len(turn_radius):
        turn_radius_violation = float(np.maximum(min_turn_radius_mm - turn_radius, 0.0).max())
    aperture_violation = float(np.maximum(radius - lens_radius, 0.0).max()) if len(radius) else 0.0

    feasible = (
        turn_violation <= tol
        and turn_radius_violation <= tol
        and aperture_violation <= tol
    )
    violation_score = (
        turn_violation / max(max_turn_angle_deg or 1.0, 1e-12)
        + turn_radius_violation / max(min_turn_radius_mm or 1.0, 1e-12)
        + aperture_violation / max(lens_radius, 1e-12)
    )
    return {
        "feasible": bool(feasible),
        "violation_score": float(violation_score),
        "turn_violation": turn_violation,
        "turn_radius_violation": turn_radius_violation,
        "aperture_violation": aperture_violation,
        "max_turn_deg": max_turn,
        "min_turn_radius_mm": min_turn_radius,
        "max_radius_mm": max_radius,
    }


def path_constraint_metrics_np(
    waypoints: np.ndarray,
    dwells: np.ndarray,
    lens_radius: float,
    speed_min_mm_s: float | None,
    speed_max_mm_s: float | None,
    max_accel_mm_s2: float | None,
    max_turn_angle_deg: float | None,
    min_turn_radius_mm: float | None,
    tol: float = 1e-6,
) -> dict[str, float | bool]:
    motion = motion_metrics_np(waypoints, dwells)
    speed = motion["speed"]
    accel = motion["acceleration"]
    geometry = path_geometry_metrics_np(
        waypoints,
        lens_radius,
        max_turn_angle_deg,
        min_turn_radius_mm,
        tol,
    )

    speed_low_violation = 0.0
    if speed_min_mm_s is not None and speed_min_mm_s > 0:
        speed_low_violation = float(np.maximum(speed_min_mm_s - speed, 0.0).max())
    speed_high_violation = 0.0
    if speed_max_mm_s is not None and speed_max_mm_s > 0:
        speed_high_violation = float(np.maximum(speed - speed_max_mm_s, 0.0).max())
    accel_violation = 0.0
    if max_accel_mm_s2 is not None and max_accel_mm_s2 > 0 and len(accel):
        accel_violation = float(np.maximum(np.abs(accel) - max_accel_mm_s2, 0.0).max())
    feasible = (
        speed_low_violation <= tol
        and speed_high_violation <= tol
        and accel_violation <= tol
        and geometry["feasible"]
    )

    return {
        "feasible": bool(feasible),
        "speed_low_violation": speed_low_violation,
        "speed_high_violation": speed_high_violation,
        "accel_violation": accel_violation,
        "turn_violation": geometry["turn_violation"],
        "turn_radius_violation": geometry["turn_radius_violation"],
        "aperture_violation": geometry["aperture_violation"],
        "speed_min": float(motion["speed_min"]),
        "speed_max": float(motion["speed_max"]),
        "accel_abs_max": float(motion["accel_abs_max"]),
        "max_turn_deg": geometry["max_turn_deg"],
        "min_turn_radius_mm": geometry["min_turn_radius_mm"],
        "max_radius_mm": geometry["max_radius_mm"],
    }


def enforce_accel_limit_np(
    waypoints: np.ndarray,
    dwells: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_accel_mm_s2: float | None,
    max_passes: int = 20,
) -> np.ndarray:
    """Greedily increase dwell on faster adjacent segments to cap acceleration.

    This is a post-LS feasibility projection. It never decreases dwell, because
    increasing dwell lowers effective feed speed and increases the time base.
    If the upper dwell bound prevents full compliance, it gets as close as the
    configured bounds allow.
    """
    if max_accel_mm_s2 is None or max_accel_mm_s2 <= 0 or len(dwells) < 2:
        return dwells

    ds = waypoint_step_lengths_np(waypoints)
    adjusted = np.clip(dwells.astype(np.float64, copy=True), lb, ub)

    def pair_accel_abs(i: int, candidate: np.ndarray) -> float:
        v0 = ds[i] / max(candidate[i], 1e-8)
        v1 = ds[i + 1] / max(candidate[i + 1], 1e-8)
        dt = 0.5 * (candidate[i] + candidate[i + 1])
        return abs((v1 - v0) / max(dt, 1e-8))

    for _ in range(max_passes):
        changed = False
        speed = ds / np.maximum(adjusted, 1e-8)
        dt_mid = 0.5 * (adjusted[1:] + adjusted[:-1])
        accel = np.diff(speed) / np.maximum(dt_mid, 1e-8)
        bad_pairs = np.where(np.abs(accel) > max_accel_mm_s2)[0]
        if len(bad_pairs) == 0:
            break

        for i in bad_pairs:
            speed = ds / np.maximum(adjusted, 1e-8)
            j = i if speed[i] >= speed[i + 1] else i + 1
            if adjusted[j] >= ub[j] - 1e-10:
                continue

            lo = adjusted[j]
            hi = ub[j]
            candidate = adjusted.copy()
            candidate[j] = hi
            if pair_accel_abs(i, candidate) > max_accel_mm_s2:
                new_val = hi
            else:
                for _ in range(32):
                    mid = 0.5 * (lo + hi)
                    candidate[j] = mid
                    if pair_accel_abs(i, candidate) > max_accel_mm_s2:
                        lo = mid
                    else:
                        hi = mid
                new_val = hi

            if new_val > adjusted[j] + 1e-10:
                adjusted[j] = new_val
                changed = True

        if not changed:
            break

    return adjusted


def solve_dwell(
    waypoints: np.ndarray,
    problem: dict,
    dwell_range: tuple[float, float],
    speed_min_mm_s: float | None = None,
    speed_max_mm_s: float | None = None,
    max_accel_mm_s2: float | None = None,
):
    """Solve bounded LS dwell for a fixed path."""
    a = build_a_matrix(waypoints, problem)
    lb, ub = dwell_bounds_for_speed(waypoints, dwell_range, speed_min_mm_s, speed_max_mm_s)
    result = lsq_linear(a, problem["b"], bounds=(lb, ub), method="bvls", verbose=0)
    dwells = enforce_accel_limit_np(waypoints, result.x, lb, ub, max_accel_mm_s2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        residual = problem["b"] - a @ dwells
    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("Non-finite residual from bounded least-squares solve")
    rms = float(np.sqrt(np.mean(residual**2)))
    overpolish = float(np.sqrt(np.mean(np.minimum(residual, 0.0) ** 2)))
    return dwells, rms, overpolish


def compute_removal_torch(
    waypoints: torch.Tensor,
    dwells: torch.Tensor,
    sigma: torch.Tensor,
    depth: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
) -> torch.Tensor:
    dx = xx.unsqueeze(1) - waypoints[:, 0].unsqueeze(0)
    dy = yy.unsqueeze(1) - waypoints[:, 1].unsqueeze(0)
    r2 = dx**2 + dy**2
    tif = depth * torch.exp(-r2 / (2.0 * sigma**2))
    return tif @ dwells


def path_losses(
    control: torch.Tensor,
    waypoints: torch.Tensor,
    dwells: torch.Tensor,
    lens_radius: float,
    target_step_mm: float,
    max_turn_rad: float,
    min_turn_radius_mm: float,
    speed_min_mm_s: float | None,
    speed_max_mm_s: float | None,
    max_accel_mm_s2: float | None,
) -> dict[str, torch.Tensor]:
    """Geometry penalties for continuous, gentle-turn paths."""
    eps = torch.tensor(1e-8, dtype=waypoints.dtype, device=waypoints.device)

    c2 = control[2:] - 2.0 * control[1:-1] + control[:-2]
    control_smooth = torch.mean(torch.sum(c2**2, dim=1)) / (lens_radius**2 + 1e-8)

    step = waypoints[1:] - waypoints[:-1]
    step_len = torch.sqrt(torch.sum(step**2, dim=1) + eps)
    spacing = torch.mean((step_len / target_step_mm - 1.0) ** 2)

    ds_per_dwell = torch.cat([step_len[:1], step_len])
    speed = ds_per_dwell / torch.clamp(dwells, min=1e-6)
    speed_penalty = torch.zeros((), dtype=waypoints.dtype, device=waypoints.device)
    if speed_min_mm_s is not None and speed_min_mm_s > 0:
        speed_penalty = speed_penalty + torch.mean(torch.relu(speed_min_mm_s - speed) ** 2)
    if speed_max_mm_s is not None and speed_max_mm_s > 0:
        speed_penalty = speed_penalty + torch.mean(torch.relu(speed - speed_max_mm_s) ** 2)

    accel_penalty = torch.zeros((), dtype=waypoints.dtype, device=waypoints.device)
    if max_accel_mm_s2 is not None and max_accel_mm_s2 > 0 and len(speed) > 1:
        dt_mid = 0.5 * (dwells[1:] + dwells[:-1])
        accel = (speed[1:] - speed[:-1]) / torch.clamp(dt_mid, min=1e-6)
        accel_penalty = torch.mean(torch.relu(torch.abs(accel) - max_accel_mm_s2) ** 2)
    elif len(speed) > 1:
        dt_mid = 0.5 * (dwells[1:] + dwells[:-1])
        accel = (speed[1:] - speed[:-1]) / torch.clamp(dt_mid, min=1e-6)
    else:
        accel = torch.zeros(0, dtype=waypoints.dtype, device=waypoints.device)

    v1 = step[:-1]
    v2 = step[1:]
    l1 = step_len[:-1]
    l2 = step_len[1:]
    cos_turn = torch.sum(v1 * v2, dim=1) / (l1 * l2 + eps)
    cos_turn = torch.clamp(cos_turn, -1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos_turn)
    turn = torch.mean(torch.relu(angle - max_turn_rad) ** 2)

    local_ds = 0.5 * (l1 + l2)
    curvature = angle / (local_ds + eps)
    max_curvature = 1.0 / max(min_turn_radius_mm, 1e-6)
    turn_radius = torch.mean(torch.relu(curvature - max_curvature) ** 2)

    r_wp = torch.sqrt(torch.sum(waypoints**2, dim=1) + eps)
    r_cp = torch.sqrt(torch.sum(control**2, dim=1) + eps)
    radius = (
        torch.mean(torch.relu(r_wp - lens_radius) ** 2)
        + torch.mean(torch.relu(r_cp - lens_radius) ** 2)
    ) / (lens_radius**2 + 1e-8)

    return {
        "control_smooth": control_smooth,
        "spacing": spacing,
        "turn": turn,
        "turn_radius": turn_radius,
        "radius": radius,
        "speed": speed_penalty,
        "accel": accel_penalty,
        "mean_turn_deg": torch.mean(angle) * 180.0 / np.pi,
        "max_turn_deg": torch.max(angle) * 180.0 / np.pi,
        "min_step": torch.min(step_len),
        "max_step": torch.max(step_len),
        "min_speed": torch.min(speed),
        "max_speed": torch.max(speed),
        "mean_speed": torch.mean(speed),
        "max_abs_accel": torch.max(torch.abs(accel)) if len(accel) else torch.zeros((), dtype=waypoints.dtype, device=waypoints.device),
    }


def optimize_free_path(problem: dict, args, dwell_range: tuple[float, float]):
    cfg = problem["config"].get("free_path", {})
    weights = cfg.get("weights", {})

    n_waypoints = problem["n_waypoints"]
    n_control = args.n_control_points
    lens_radius = problem["lens_radius"]
    initial_reference_wp = make_initial_waypoints(args, problem, n_waypoints, lens_radius)
    control_init = resample_polyline(initial_reference_wp, n_control)

    init_control_t = torch.tensor(control_init, dtype=torch.float32)
    initial_wp = sample_catmull_rom(init_control_t, n_waypoints).cpu().numpy()
    target_step_mm = float(np.mean(np.linalg.norm(np.diff(initial_wp, axis=0), axis=1)))
    max_turn_rad = np.deg2rad(args.max_turn_angle_deg)

    print("=" * 78)
    print("FREE CONTINUOUS PATH OPTIMIZATION")
    print("=" * 78)
    print(f"  Waypoints:       {n_waypoints}")
    print(f"  Control points:  {n_control}")
    print(f"  Init path:       {args.init_path}")
    if args.init_path == "smooth_spiral":
        print(f"  Init turns:      {args.init_turns:.2f}")
        print(f"  Init inner r:    {args.effective_init_min_radius_mm:.2f} mm")
    print(f"  Initial RMS:     {problem['initial_rms']:.4f} um")
    print(f"  Target step:     {target_step_mm:.3f} mm")
    print(f"  Max turn angle:  {args.max_turn_angle_deg:.1f} deg")
    print(f"  Min turn radius: {args.min_turn_radius_mm:.1f} mm")
    print(f"  Feed speed:      [{args.speed_min_mm_s:.3f}, {args.speed_max_mm_s:.3f}] mm/s")
    print(f"  Max accel:       {args.max_accel_mm_s2:.3f} mm/s^2")

    print("\nSolving baseline dwell on initial free path ...")
    baseline_dwells, baseline_rms, baseline_over = solve_dwell(
        initial_wp,
        problem,
        dwell_range,
        args.speed_min_mm_s,
        args.speed_max_mm_s,
        args.max_accel_mm_s2,
    )
    baseline_motion = motion_metrics_np(initial_wp, baseline_dwells)
    print(f"  Baseline RMS: {baseline_rms:.4f} um, overpolish RMS: {baseline_over:.4f} um")
    print(
        f"  Baseline speed: {baseline_motion['speed_min']:.3f}-"
        f"{baseline_motion['speed_max']:.3f} mm/s, "
        f"|accel|max={baseline_motion['accel_abs_max']:.3f} mm/s^2"
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    control = torch.nn.Parameter(torch.tensor(control_init, dtype=dtype, device=device))

    b_t = torch.tensor(problem["b"], dtype=dtype, device=device)
    xx_t = torch.tensor(problem["xx_mask"], dtype=dtype, device=device)
    yy_t = torch.tensor(problem["yy_mask"], dtype=dtype, device=device)
    sigma_t = torch.tensor(problem["sigma_per_wp"], dtype=dtype, device=device)
    depth_t = torch.tensor(problem["depth_per_wp"], dtype=dtype, device=device)

    optimizer = torch.optim.Adam([control], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.n_iters, 1))

    current_dwells = baseline_dwells.copy()
    baseline_constraints = path_constraint_metrics_np(
        initial_wp,
        baseline_dwells,
        lens_radius,
        args.speed_min_mm_s,
        args.speed_max_mm_s,
        args.max_accel_mm_s2,
        args.max_turn_angle_deg,
        args.min_turn_radius_mm,
        args.constraint_tol,
    )
    feas = "OK" if baseline_constraints["feasible"] else "VIOL"
    print(
        f"  Baseline constraints: {feas}, "
        f"turn={baseline_constraints['max_turn_deg']:.2f} deg, "
        f"Rturn_min={baseline_constraints['min_turn_radius_mm']:.2f} mm, "
        f"|a|max={baseline_constraints['accel_abs_max']:.2f} mm/s^2"
    )

    best_rms = float("inf")
    best_control = control.detach().cpu().numpy().copy()
    best_waypoints = initial_wp.copy()
    best_dwells = baseline_dwells.copy()
    best_iter = 0
    best_constraints = baseline_constraints
    if baseline_constraints["feasible"]:
        best_rms = baseline_rms

    accepted_path_steps = 0
    backtracked_path_steps = 0
    rejected_path_steps = 0

    t0 = time.time()
    for it in range(args.n_iters):
        if it > 0 and it % args.lsq_interval == 0:
            with torch.no_grad():
                wp_np = sample_catmull_rom(control, n_waypoints).detach().cpu().numpy()
            current_dwells, lsq_rms, lsq_over = solve_dwell(
                wp_np,
                problem,
                dwell_range,
                args.speed_min_mm_s,
                args.speed_max_mm_s,
                args.max_accel_mm_s2,
            )
            constraint_metrics = path_constraint_metrics_np(
                wp_np,
                current_dwells,
                lens_radius,
                args.speed_min_mm_s,
                args.speed_max_mm_s,
                args.max_accel_mm_s2,
                args.max_turn_angle_deg,
                args.min_turn_radius_mm,
                args.constraint_tol,
            )
            if constraint_metrics["feasible"] and lsq_rms < best_rms:
                best_rms = lsq_rms
                best_control = control.detach().cpu().numpy().copy()
                best_waypoints = wp_np.copy()
                best_dwells = current_dwells.copy()
                best_iter = it
                best_constraints = constraint_metrics
            feas = "OK" if constraint_metrics["feasible"] else "VIOL"
            path_delta = np.linalg.norm(wp_np - initial_wp, axis=1)
            print(
                f"  LS refresh @ iter {it:5d}: RMS={lsq_rms:.4f} um, "
                f"over={lsq_over:.4f} um, constraints={feas}, "
                f"pathΔmean={path_delta.mean():.3f}mm, "
                f"v=[{constraint_metrics['speed_min']:.2f},{constraint_metrics['speed_max']:.2f}], "
                f"|a|max={constraint_metrics['accel_abs_max']:.2f}, "
                f"turn={constraint_metrics['max_turn_deg']:.1f}"
            )

        optimizer.zero_grad()

        waypoints = sample_catmull_rom(control, n_waypoints)
        dwell_t = torch.tensor(current_dwells, dtype=dtype, device=device)
        removal = compute_removal_torch(waypoints, dwell_t, sigma_t, depth_t, xx_t, yy_t)
        err = b_t - removal

        form = torch.mean(err**2) / (problem["initial_rms"] ** 2 + 1e-8)
        overpolish = torch.mean(torch.clamp(-err, min=0.0) ** 2) / (problem["initial_rms"] ** 2 + 1e-8)

        geom = path_losses(
            control,
            waypoints,
            dwell_t,
            lens_radius=lens_radius,
            target_step_mm=target_step_mm,
            max_turn_rad=max_turn_rad,
            min_turn_radius_mm=args.min_turn_radius_mm,
            speed_min_mm_s=args.speed_min_mm_s,
            speed_max_mm_s=args.speed_max_mm_s,
            max_accel_mm_s2=args.max_accel_mm_s2,
        )

        loss = (
            args.w_form * form
            + args.w_overpolish * overpolish
            + args.w_smooth * geom["control_smooth"]
            + args.w_turn * geom["turn"]
            + args.w_turn_radius * geom["turn_radius"]
            + args.w_spacing * geom["spacing"]
            + args.w_radius * geom["radius"]
            + args.w_speed * geom["speed"]
            + args.w_accel * geom["accel"]
        )
        loss.backward()
        previous_control = control.detach().clone()
        optimizer.step()
        scheduler.step()
        if not args.allow_infeasible_steps:
            candidate_control = control.detach().clone()
            with torch.no_grad():
                previous_wp = sample_catmull_rom(previous_control, n_waypoints).detach().cpu().numpy()
                candidate_wp = sample_catmull_rom(control, n_waypoints).detach().cpu().numpy()
            previous_geometry = path_geometry_metrics_np(
                previous_wp,
                lens_radius,
                args.max_turn_angle_deg,
                args.min_turn_radius_mm,
                args.constraint_tol,
            )
            candidate_geometry = path_geometry_metrics_np(
                candidate_wp,
                lens_radius,
                args.max_turn_angle_deg,
                args.min_turn_radius_mm,
                args.constraint_tol,
            )
            improves_infeasible_geometry = (
                not previous_geometry["feasible"]
                and candidate_geometry["violation_score"] <= previous_geometry["violation_score"]
            )
            if candidate_geometry["feasible"] or improves_infeasible_geometry:
                accepted_path_steps += 1
            else:
                accepted = False
                for bt in range(args.projection_backtrack_steps):
                    alpha = 0.5 ** (bt + 1)
                    trial_control = previous_control + alpha * (candidate_control - previous_control)
                    with torch.no_grad():
                        control.copy_(trial_control)
                        trial_wp = sample_catmull_rom(control, n_waypoints).detach().cpu().numpy()
                    trial_geometry = path_geometry_metrics_np(
                        trial_wp,
                        lens_radius,
                        args.max_turn_angle_deg,
                        args.min_turn_radius_mm,
                        args.constraint_tol,
                    )
                    improves_infeasible_geometry = (
                        not previous_geometry["feasible"]
                        and trial_geometry["violation_score"] <= previous_geometry["violation_score"]
                    )
                    if trial_geometry["feasible"] or improves_infeasible_geometry:
                        accepted = True
                        accepted_path_steps += 1
                        backtracked_path_steps += 1
                        break
                if not accepted:
                    with torch.no_grad():
                        control.copy_(previous_control)
                    rejected_path_steps += 1

        if (it + 1) % args.report_interval == 0 or it == 0:
            cur_rms = torch.sqrt(torch.mean(err**2)).item()
            print(
                f"  iter {it + 1:5d}: loss={loss.item():.5f}, "
                f"RMS={cur_rms:.4f} um, "
                f"mean_turn={geom['mean_turn_deg'].item():.2f} deg, "
                f"max_turn={geom['max_turn_deg'].item():.2f} deg, "
                f"step=[{geom['min_step'].item():.2f},{geom['max_step'].item():.2f}], "
                f"speed=[{geom['min_speed'].item():.2f},{geom['max_speed'].item():.2f}], "
                f"|accel|max={geom['max_abs_accel'].item():.2f}"
            )

    with torch.no_grad():
        best_control_t = torch.tensor(best_control, dtype=dtype, device=device)
        best_wp = sample_catmull_rom(best_control_t, n_waypoints).cpu().numpy()
        endpoint_wp = sample_catmull_rom(control, n_waypoints).detach().cpu().numpy()

    best_dwells_solved, best_rms_solved, best_over = solve_dwell(
        best_wp,
        problem,
        dwell_range,
        args.speed_min_mm_s,
        args.speed_max_mm_s,
        args.max_accel_mm_s2,
    )
    endpoint_dwells, endpoint_rms, endpoint_over = solve_dwell(
        endpoint_wp,
        problem,
        dwell_range,
        args.speed_min_mm_s,
        args.speed_max_mm_s,
        args.max_accel_mm_s2,
    )

    best_solved_constraints = path_constraint_metrics_np(
        best_wp,
        best_dwells_solved,
        lens_radius,
        args.speed_min_mm_s,
        args.speed_max_mm_s,
        args.max_accel_mm_s2,
        args.max_turn_angle_deg,
        args.min_turn_radius_mm,
        args.constraint_tol,
    )
    endpoint_constraints = path_constraint_metrics_np(
        endpoint_wp,
        endpoint_dwells,
        lens_radius,
        args.speed_min_mm_s,
        args.speed_max_mm_s,
        args.max_accel_mm_s2,
        args.max_turn_angle_deg,
        args.min_turn_radius_mm,
        args.constraint_tol,
    )

    if best_solved_constraints["feasible"] and best_rms_solved <= best_rms:
        best_dwells = best_dwells_solved
        best_rms = best_rms_solved
        best_waypoints = best_wp
        best_overpolish = best_over
        best_constraints = best_solved_constraints
    else:
        best_overpolish = baseline_over if best_iter == 0 else float("nan")

    if args.save_mode == "final" and endpoint_constraints["feasible"]:
        selected_wp = endpoint_wp
        selected_dwells = endpoint_dwells
        selected_rms = endpoint_rms
        selected_over = endpoint_over
        selected_control = control.detach().cpu().numpy().copy()
        selected_label = "final-smoothed"
        selected_constraints = endpoint_constraints
    elif np.isfinite(best_rms):
        selected_wp = best_waypoints
        selected_dwells = best_dwells
        selected_rms = best_rms
        selected_over = best_overpolish
        selected_control = best_control
        selected_label = "best-rms"
        selected_constraints = best_constraints
    else:
        selected_wp = endpoint_wp
        selected_dwells = endpoint_dwells
        selected_rms = endpoint_rms
        selected_over = endpoint_over
        selected_control = control.detach().cpu().numpy().copy()
        selected_label = "infeasible-final"
        selected_constraints = endpoint_constraints

    selected_motion = motion_metrics_np(selected_wp, selected_dwells)
    selected_path_delta = np.linalg.norm(selected_wp - initial_wp, axis=1)

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  Best iter:     {best_iter}")
    print(f"  Initial RMS:   {problem['initial_rms']:.4f} um")
    print(f"  Baseline RMS:  {baseline_rms:.4f} um")
    best_rms_text = f"{best_rms:.4f}" if np.isfinite(best_rms) else "none feasible"
    print(f"  Best RMS:      {best_rms_text} um")
    print(f"  Endpoint RMS:  {endpoint_rms:.4f} um")
    print(f"  Saved mode:    {selected_label}")
    print(f"  Saved RMS:     {selected_rms:.4f} um")
    print(f"  Improvement:   {(1.0 - selected_rms / baseline_rms) * 100.0:.1f}% over initial path")
    print(
        f"  Path delta:    mean {selected_path_delta.mean():.3f} mm, "
        f"max {selected_path_delta.max():.3f} mm"
    )
    print(
        f"  Path steps:    accepted={accepted_path_steps}, "
        f"backtracked={backtracked_path_steps}, rejected={rejected_path_steps}"
    )
    print(f"  Overpolish:    {selected_over:.4f} um")
    print(
        f"  Speed:        {selected_motion['speed_min']:.3f}-"
        f"{selected_motion['speed_max']:.3f} mm/s "
        f"(mean {selected_motion['speed_mean']:.3f})"
    )
    print(
        f"  Accel |max|:  {selected_motion['accel_abs_max']:.3f} mm/s^2 "
        f"(mean {selected_motion['accel_abs_mean']:.3f})"
    )
    print(f"  Feasible:     {selected_constraints['feasible']}")
    print(
        "  Violations:   "
        f"speed_low={selected_constraints['speed_low_violation']:.3g}, "
        f"speed_high={selected_constraints['speed_high_violation']:.3g}, "
        f"accel={selected_constraints['accel_violation']:.3g}, "
        f"turn={selected_constraints['turn_violation']:.3g}, "
        f"turn_radius={selected_constraints['turn_radius_violation']:.3g}, "
        f"aperture={selected_constraints['aperture_violation']:.3g}"
    )

    return {
        "baseline_dwells": baseline_dwells,
        "baseline_rms": baseline_rms,
        "initial_waypoints": initial_wp,
        "initial_reference_waypoints": initial_reference_wp,
        "dwells": selected_dwells,
        "waypoints": selected_wp,
        "control_points": selected_control,
        "rms": selected_rms,
        "overpolish": selected_over,
        "endpoint_rms": endpoint_rms,
        "best_rms": best_rms,
        "save_mode": selected_label,
        "motion": selected_motion,
        "constraints": selected_constraints,
        "path_delta_mean_mm": float(selected_path_delta.mean()),
        "path_delta_max_mm": float(selected_path_delta.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Free continuous path optimization")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--tif-checkpoint", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "results" / "free_path_optimized.npz"))
    parser.add_argument("--n-waypoints", type=int, default=None)
    parser.add_argument("--n-control-points", type=int, default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--lsq-interval", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--save-mode", choices=["best-rms", "final"], default="best-rms")
    parser.add_argument("--init-path", choices=["smooth_spiral", "nominal_spiral"], default=None)
    parser.add_argument("--init-turns", type=float, default=None)
    parser.add_argument("--init-min-radius-mm", type=float, default=None)
    parser.add_argument("--init-max-radius-frac", type=float, default=None)
    parser.add_argument(
        "--allow-infeasible-steps",
        action="store_true",
        help="Allow gradient steps to leave the hard-feasible region between LS refreshes.",
    )
    parser.add_argument("--projection-backtrack-steps", type=int, default=None)
    parser.add_argument("--max-turn-angle-deg", type=float, default=None)
    parser.add_argument("--min-turn-radius-mm", type=float, default=None)
    parser.add_argument("--w-form", type=float, default=None)
    parser.add_argument("--w-overpolish", type=float, default=None)
    parser.add_argument("--w-smooth", type=float, default=None)
    parser.add_argument("--w-turn", type=float, default=None)
    parser.add_argument("--w-turn-radius", type=float, default=None)
    parser.add_argument("--w-spacing", type=float, default=None)
    parser.add_argument("--w-radius", type=float, default=None)
    parser.add_argument("--speed-min-mm-s", type=float, default=None)
    parser.add_argument("--speed-max-mm-s", type=float, default=None)
    parser.add_argument("--max-accel-mm-s2", type=float, default=None)
    parser.add_argument("--w-speed", type=float, default=None)
    parser.add_argument("--w-accel", type=float, default=None)
    parser.add_argument("--constraint-tol", type=float, default=1e-6)
    args = parser.parse_args()

    config = load_config(args.config)
    free_cfg = config.get("free_path", {})
    weight_cfg = free_cfg.get("weights", {})

    args.n_waypoints = args.n_waypoints or free_cfg.get("n_waypoints")
    args.n_control_points = args.n_control_points or free_cfg.get("n_control_points", 48)
    args.n_iters = args.n_iters if args.n_iters is not None else free_cfg.get("n_iters", 3000)
    args.lsq_interval = args.lsq_interval if args.lsq_interval is not None else free_cfg.get("lsq_interval", 100)
    args.lr = args.lr if args.lr is not None else free_cfg.get("lr", 0.01)
    init_cfg = free_cfg.get("initializer", {})
    args.init_path = args.init_path or init_cfg.get("path", "smooth_spiral")
    args.init_turns = args.init_turns if args.init_turns is not None else init_cfg.get("turns", 5.0)
    args.init_min_radius_mm = (
        args.init_min_radius_mm
        if args.init_min_radius_mm is not None
        else init_cfg.get("min_radius_mm")
    )
    args.init_max_radius_frac = (
        args.init_max_radius_frac
        if args.init_max_radius_frac is not None
        else init_cfg.get("max_radius_frac", 0.96)
    )
    args.allow_infeasible_steps = bool(
        args.allow_infeasible_steps or free_cfg.get("allow_infeasible_steps", False)
    )
    args.projection_backtrack_steps = (
        args.projection_backtrack_steps
        if args.projection_backtrack_steps is not None
        else free_cfg.get("projection_backtrack_steps", 8)
    )
    args.max_turn_angle_deg = (
        args.max_turn_angle_deg
        if args.max_turn_angle_deg is not None
        else free_cfg.get("max_turn_angle_deg", 10.0)
    )
    args.min_turn_radius_mm = (
        args.min_turn_radius_mm
        if args.min_turn_radius_mm is not None
        else free_cfg.get("min_turn_radius_mm", 8.0)
    )

    args.w_form = args.w_form if args.w_form is not None else weight_cfg.get("form", 5.0)
    args.w_overpolish = (
        args.w_overpolish if args.w_overpolish is not None else weight_cfg.get("overpolish", 10.0)
    )
    args.w_smooth = args.w_smooth if args.w_smooth is not None else weight_cfg.get("smoothness", 0.5)
    args.w_turn = args.w_turn if args.w_turn is not None else weight_cfg.get("turn", 2.0)
    args.w_turn_radius = (
        args.w_turn_radius if args.w_turn_radius is not None else weight_cfg.get("turn_radius", 1.0)
    )
    args.w_spacing = args.w_spacing if args.w_spacing is not None else weight_cfg.get("spacing", 0.2)
    args.w_radius = args.w_radius if args.w_radius is not None else weight_cfg.get("radius", 100.0)
    motion_cfg = free_cfg.get("motion", {})
    args.speed_min_mm_s = (
        args.speed_min_mm_s if args.speed_min_mm_s is not None else motion_cfg.get("speed_min_mm_s", 0.05)
    )
    args.speed_max_mm_s = (
        args.speed_max_mm_s if args.speed_max_mm_s is not None else motion_cfg.get("speed_max_mm_s", 30.0)
    )
    args.max_accel_mm_s2 = (
        args.max_accel_mm_s2 if args.max_accel_mm_s2 is not None else motion_cfg.get("max_accel_mm_s2", 20.0)
    )
    args.w_speed = args.w_speed if args.w_speed is not None else weight_cfg.get("speed", 1.0)
    args.w_accel = args.w_accel if args.w_accel is not None else weight_cfg.get("accel", 0.1)

    tif_ckpt = args.tif_checkpoint if Path(args.tif_checkpoint).exists() else None
    if tif_ckpt:
        print(f"Using TIF checkpoint: {tif_ckpt}")
    else:
        print("No TIF checkpoint found - using default-initialized TIF network")

    problem = build_problem(config, tif_ckpt, args.n_waypoints)
    dwell_range = tuple(config["env"]["action_ranges"]["dwell_time"])
    result = optimize_free_path(problem, args, dwell_range)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        dwells_baseline=result["baseline_dwells"],
        dwells_joint=result["dwells"],
        delta_joint=result["waypoints"] - result["initial_waypoints"],
        waypoints_nominal=result["initial_waypoints"],
        waypoints_reference=result["initial_reference_waypoints"],
        waypoints_joint=result["waypoints"],
        control_points=result["control_points"],
        waypoint_pressures=problem["waypoint_pressures"],
        init_path=args.init_path,
        init_turns=args.init_turns,
        init_min_radius_mm=args.effective_init_min_radius_mm,
        init_max_radius_frac=args.init_max_radius_frac,
        allow_infeasible_steps=args.allow_infeasible_steps,
        initial_rms=problem["initial_rms"],
        rms_baseline=result["baseline_rms"],
        rms_joint=result["rms"],
        overpolish_rms=result["overpolish"],
        endpoint_rms=result["endpoint_rms"],
        best_rms=result["best_rms"],
        save_mode=result["save_mode"],
        step_length=result["motion"]["step_length"],
        feed_speed=result["motion"]["speed"],
        feed_acceleration=result["motion"]["acceleration"],
        speed_min_mm_s=args.speed_min_mm_s,
        speed_max_mm_s=args.speed_max_mm_s,
        max_accel_mm_s2=args.max_accel_mm_s2,
        feasible=result["constraints"]["feasible"],
        speed_low_violation=result["constraints"]["speed_low_violation"],
        speed_high_violation=result["constraints"]["speed_high_violation"],
        accel_violation=result["constraints"]["accel_violation"],
        turn_violation=result["constraints"]["turn_violation"],
        turn_radius_violation=result["constraints"]["turn_radius_violation"],
        aperture_violation=result["constraints"]["aperture_violation"],
        path_delta_mean_mm=result["path_delta_mean_mm"],
        path_delta_max_mm=result["path_delta_max_mm"],
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
