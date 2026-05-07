#!/usr/bin/env python3
"""Time-synchronized comparison animation: Baseline vs Optimized.

Each frame represents a fixed increment of real time. Since dwell times
differ, the tool reaches different waypoints at the same real time:
  - Baseline (constant dwell): uniform speed along the spiral
  - Optimized (variable dwell + perturbed path): fast through low-dwell
    waypoints, slow at high-dwell waypoints

Output: MP4 video.

Usage:
    python src/scripts/animate_compare.py
    python src/scripts/animate_compare.py --fps 15 --time-step 2.0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from env.lens_surface import LensSurface
from env.spiral_path import SpiralPath
from models.tif_model import TIFModel


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def simulate_episode(surface_template, waypoints, dwells, sigma, depth, mask):
    """Simulate polishing step-by-step, returning per-waypoint snapshots.

    Returns list of dicts with error_map, rms, tool_xy, cumulative_time.
    """
    surface = surface_template.copy()
    target = surface_template - surface_template  # will be set below
    # Actually we need to reconstruct: current = target + error
    # surface_template IS the initial current surface
    # We subtract removal from it directly.

    grid_size = surface.shape[0]
    xx = np.linspace(-25, 25, grid_size)  # will be overridden
    yy = np.linspace(-25, 25, grid_size)

    n = len(dwells)
    cum_time = np.zeros(n + 1)

    snapshots = [{
        "surface": surface.copy(),
        "tool_x": 0.0,
        "tool_y": 0.0,
        "cum_time": 0.0,
        "dwell": 0.0,
        "waypoint_idx": 0,
    }]

    for i in range(n):
        cx, cy = float(waypoints[i, 0]), float(waypoints[i, 1])
        d = dwells[i]
        cum_time[i + 1] = cum_time[i] + d

        # Apply removal
        # (we recompute TIF on the fly — waypoints may be perturbed)
        # This method is called with pre-extracted xx_grid, yy_grid
        pass  # actual removal done below

    return snapshots  # placeholder


def build_snapshots(surface_obj, waypoints, dwells, sigma_arr, depth_arr,
                    waypoint_pressures, speed, conc):
    """Build per-step surface snapshots for one method, including roughness."""
    from env.roughness_model import RoughnessModel

    current = surface_obj.current.copy()
    mask = surface_obj.mask
    target = surface_obj.target
    xx, yy = surface_obj.xx, surface_obj.yy
    grid_size = current.shape[0]

    rough_model = RoughnessModel(grid_size, 200.0)
    rough_model.reset(seed=43)

    cum_time = 0.0
    err = current - target
    err[~mask] = 0.0
    rms = float(np.sqrt(np.mean(err[mask] ** 2)))
    ra = rough_model.get_mean_roughness(mask)

    snapshots = [{
        "error": err.copy(),
        "tool_x": float(waypoints[0, 0]) if len(waypoints) > 0 else 0.0,
        "tool_y": float(waypoints[0, 1]) if len(waypoints) > 0 else 0.0,
        "rms": rms, "ra": ra,
        "cum_time": 0.0, "dwell": 0.0, "wp_idx": 0,
    }]

    for i in range(len(dwells)):
        cx = float(waypoints[i, 0])
        cy = float(waypoints[i, 1])
        sig = sigma_arr[i]
        dep = depth_arr[i]
        dwell = dwells[i]
        pressure = float(waypoint_pressures[i])

        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        tif = dep * np.exp(-r2 / (2.0 * sig ** 2))
        current -= tif * dwell * mask

        tif_max = tif.max()
        footprint = tif / (tif_max + 1e-12)
        rough_model.update(footprint, pressure, speed, conc, dwell, mask)

        cum_time += dwell
        err = current - target
        err[~mask] = 0.0
        rms = float(np.sqrt(np.mean(err[mask] ** 2)))
        ra = rough_model.get_mean_roughness(mask)

        snapshots.append({
            "error": err.copy(),
            "tool_x": cx, "tool_y": cy,
            "rms": rms, "ra": ra,
            "cum_time": cum_time, "dwell": dwell, "wp_idx": i,
        })

    return snapshots


def find_snapshot_at_time(snapshots, t):
    """Binary search for the last snapshot with cum_time <= t."""
    lo, hi = 0, len(snapshots) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if snapshots[mid]["cum_time"] <= t:
            lo = mid
        else:
            hi = mid - 1
    return snapshots[lo]


def interpolate_tool_position(waypoints, cum_time, t):
    """Interpolate tool position so dwell appears as slower feed, not pauses."""
    n = len(waypoints)
    if n == 0:
        return 0.0, 0.0, 0, 0.0
    if t <= cum_time[0]:
        return float(waypoints[0, 0]), float(waypoints[0, 1]), 0, 0.0
    if t >= cum_time[-1]:
        return float(waypoints[-1, 0]), float(waypoints[-1, 1]), n - 1, 1.0

    seg_idx = int(np.searchsorted(cum_time, t, side="right")) - 1
    seg_idx = max(0, min(seg_idx, n - 1))
    next_idx = min(seg_idx + 1, n - 1)
    t0 = cum_time[seg_idx]
    t1 = cum_time[min(seg_idx + 1, len(cum_time) - 1)]
    frac = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
    x = (1.0 - frac) * waypoints[seg_idx, 0] + frac * waypoints[next_idx, 0]
    y = (1.0 - frac) * waypoints[seg_idx, 1] + frac * waypoints[next_idx, 1]
    return float(x), float(y), seg_idx, frac


def trail_with_current_position(waypoints, idx, x, y):
    if len(waypoints) == 0:
        return [], []
    idx = max(0, min(idx, len(waypoints) - 1))
    xs = np.concatenate([waypoints[:idx + 1, 0], np.array([x])])
    ys = np.concatenate([waypoints[:idx + 1, 1], np.array([y])])
    return xs, ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--tif-checkpoint",
                        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"))
    parser.add_argument("--joint-results",
                        default=str(Path(__file__).resolve().parents[1] / "results" / "joint_optimized.npz"))
    parser.add_argument("--output",
                        default=str(Path(__file__).resolve().parents[1] / "runs" / "baseline_vs_optimized.mp4"))
    parser.add_argument("--baseline-dwell", type=float, default=1.0)
    parser.add_argument("--time-step", type=float, default=1.0,
                        help="Real-time seconds per animation frame")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    sc = config["surface"]
    sp = config["spiral"]
    tif_ckpt = args.tif_checkpoint if Path(args.tif_checkpoint).exists() else None

    # ── Load joint optimization results ──────────────────────────────
    data = np.load(args.joint_results)
    dwells_opt = data["dwells_joint"]
    wp_opt = data["waypoints_joint"]
    wp_nominal = data["waypoints_nominal"]
    N = len(dwells_opt)
    print(f"Loaded {N} waypoints from {args.joint_results}")

    dwells_bl = np.full(N, args.baseline_dwell)

    # ── Build TIF parameters per waypoint ────────────────────────────
    spiral = SpiralPath(
        lens_radius_mm=sc["lens_diameter_mm"] / 2.0,
        pitch_mm=sp["pitch_mm"],
        min_radius_mm=3.0,
        center_sparse_power=0.7,
    )
    single_pass_len = spiral.get_single_pass_length()
    pass_pressures = sp["pass_pressures"]
    if "waypoint_pressures" in data:
        waypoint_pressures = data["waypoint_pressures"].astype(float)
    else:
        waypoint_pressures = np.array([
            pass_pressures[min(i // single_pass_len, sp["num_passes"] - 1)]
            for i in range(N)
        ], dtype=float)

    tif_model = TIFModel(tif_ckpt)
    speed = sp["fixed_speed"]
    conc = config["tif"]["default_concentration"]

    sigma_arr = np.zeros(N)
    depth_arr = np.zeros(N)
    for i in range(N):
        p = waypoint_pressures[i]
        w, d = tif_model.predict(float(p), speed, conc)
        sigma_arr[i] = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        depth_arr[i] = d

    # ── Build surface and simulate both methods ──────────────────────
    surface = LensSurface(sc["grid_size"], sc["lens_diameter_mm"], sc["curvature_radius_mm"])
    surface.reset(error_scale=1.0, seed=42)

    speed = sp["fixed_speed"]
    conc = config["tif"]["default_concentration"]
    num_passes = sp["num_passes"]

    print("Simulating baseline ...")
    surface_bl = LensSurface(sc["grid_size"], sc["lens_diameter_mm"], sc["curvature_radius_mm"])
    surface_bl.reset(error_scale=1.0, seed=42)
    snaps_bl = build_snapshots(surface_bl, wp_nominal, dwells_bl, sigma_arr, depth_arr,
                               waypoint_pressures, speed, conc)

    print("Simulating optimized ...")
    surface_opt = LensSurface(sc["grid_size"], sc["lens_diameter_mm"], sc["curvature_radius_mm"])
    surface_opt.reset(error_scale=1.0, seed=42)
    snaps_opt = build_snapshots(surface_opt, wp_opt, dwells_opt, sigma_arr, depth_arr,
                                waypoint_pressures, speed, conc)

    T_bl = snaps_bl[-1]["cum_time"]
    T_opt = snaps_opt[-1]["cum_time"]
    T_max = max(T_bl, T_opt)
    print(f"  Baseline total time:  {T_bl:.1f}s ({len(snaps_bl)-1} waypoints)")
    print(f"  Optimized total time: {T_opt:.1f}s ({len(snaps_opt)-1} waypoints)")

    # ── Time-based frame schedule ────────────────────────────────────
    dt = args.time_step
    time_points = np.arange(0, T_max + dt, dt)
    n_frames = len(time_points)
    print(f"  Animation: {n_frames} frames at Δt={dt}s, {args.fps}fps "
          f"→ ~{n_frames / args.fps:.0f}s video")

    # ── Plot setup ───────────────────────────────────────────────────
    half = sc["lens_diameter_mm"] / 2.0
    extent = [-half, half, -half, half]
    mask = surface.mask

    err_range = max(
        max(abs(s["error"][mask]).max() for s in snaps_bl[::max(1, len(snaps_bl) // 20)]),
        max(abs(s["error"][mask]).max() for s in snaps_opt[::max(1, len(snaps_opt) // 20)]),
    )
    err_range = min(err_range, 10.0)

    rms_all_bl = [s["rms"] for s in snaps_bl]
    rms_all_opt = [s["rms"] for s in snaps_opt]
    time_all_bl = [s["cum_time"] for s in snaps_bl]
    time_all_opt = [s["cum_time"] for s in snaps_opt]
    rms_max = max(max(rms_all_bl), max(rms_all_opt)) * 1.05

    BG = "#08081a"
    PANEL = "#0e0e28"
    SPINE = "#2a2a55"
    TICK_C = "#999999"
    COL_BL = "#ff5533"
    COL_OPT = "#33ff88"

    ra_all_bl = [s.get("ra", 0) for s in snaps_bl]
    ra_all_opt = [s.get("ra", 0) for s in snaps_opt]
    ra_max = max(max(ra_all_bl), max(ra_all_opt)) * 1.1

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22,
                          left=0.04, right=0.96, top=0.88, bottom=0.06,
                          height_ratios=[1.4, 0.8])

    ax_bl = fig.add_subplot(gs[0, 0])
    ax_opt = fig.add_subplot(gs[0, 1])
    ax_rms = fig.add_subplot(gs[1, 0])
    ax_ra = fig.add_subplot(gs[1, 1])

    for ax in [ax_bl, ax_opt, ax_rms, ax_ra]:
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(SPINE)

    err_norm = TwoSlopeNorm(vmin=-err_range, vcenter=0, vmax=err_range)

    init_err = np.ma.array(snaps_bl[0]["error"], mask=~mask)
    im_bl = ax_bl.imshow(init_err, cmap="RdBu_r", norm=err_norm, extent=extent,
                          origin="lower", interpolation="bilinear")
    im_opt = ax_opt.imshow(init_err, cmap="RdBu_r", norm=err_norm, extent=extent,
                            origin="lower", interpolation="bilinear")

    for ax, col in [(ax_bl, COL_BL), (ax_opt, COL_OPT)]:
        ax.add_patch(plt.Circle((0, 0), half, fill=False, color="#555588", lw=1))
        ax.set_xlim(-half * 1.05, half * 1.05)
        ax.set_ylim(-half * 1.05, half * 1.05)
        ax.set_aspect("equal")
        ax.tick_params(colors=TICK_C, labelsize=11)
        for s in ax.spines.values():
            s.set_color(col)
            s.set_linewidth(2.5)

    ax_bl.set_title("Baseline", color=COL_BL, fontsize=24, fontweight="bold", pad=12)
    ax_opt.set_title("Optimized", color=COL_OPT, fontsize=24, fontweight="bold", pad=12)

    ax_bl.plot(wp_nominal[:, 0], wp_nominal[:, 1], "-", color="#ffffff", lw=0.5, alpha=0.18)
    ax_opt.plot(wp_opt[:, 0], wp_opt[:, 1], "-", color="#ffffff", lw=0.5, alpha=0.18)
    trail_bl, = ax_bl.plot([], [], "-", color=COL_BL, lw=3.0, alpha=0.95, zorder=9)
    trail_opt, = ax_opt.plot([], [], "-", color=COL_OPT, lw=3.0, alpha=0.95, zorder=9)

    dot_bl, = ax_bl.plot([], [], "o", color=COL_BL, ms=22,
                          markeredgecolor="white", markeredgewidth=2.5, zorder=10)
    dot_opt, = ax_opt.plot([], [], "o", color=COL_OPT, ms=22,
                            markeredgecolor="white", markeredgewidth=2.5, zorder=10)

    stat_bl = ax_bl.text(0.5, -0.06, "", transform=ax_bl.transAxes,
                          fontsize=20, color=COL_BL, fontfamily="monospace",
                          ha="center", va="top", fontweight="bold")
    stat_opt = ax_opt.text(0.5, -0.06, "", transform=ax_opt.transAxes,
                            fontsize=20, color=COL_OPT, fontfamily="monospace",
                            ha="center", va="top", fontweight="bold")

    # RMS curves
    target_rms = sc["target_rms_um"]
    ax_rms.set_xlim(0, T_max)
    ax_rms.set_ylim(0, rms_max)
    ax_rms.axhline(target_rms, color=COL_OPT, ls=":", lw=1.5, alpha=0.4)
    rms_line_bl, = ax_rms.plot([], [], "-", color=COL_BL, lw=3, label="Baseline")
    rms_line_opt, = ax_rms.plot([], [], "-", color=COL_OPT, lw=3, label="Optimized")
    rms_dot_bl, = ax_rms.plot([], [], "o", color=COL_BL, ms=8)
    rms_dot_opt, = ax_rms.plot([], [], "o", color=COL_OPT, ms=8)
    ax_rms.set_title("RMS Error (µm)", color="white", fontsize=20)
    ax_rms.set_xlabel("Time (s)", color=TICK_C, fontsize=14)
    ax_rms.tick_params(colors=TICK_C, labelsize=11)
    ax_rms.grid(True, alpha=0.12, color="#444466")
    ax_rms.legend(fontsize=13, loc="upper right",
                  facecolor="#1a1a3a", edgecolor=SPINE, labelcolor="white")

    # Ra curves
    ra_t_bl = np.array(time_all_bl)
    ra_v_bl = np.array(ra_all_bl)
    ra_t_opt = np.array(time_all_opt)
    ra_v_opt = np.array(ra_all_opt)

    ax_ra.set_xlim(0, T_max)
    ax_ra.set_ylim(0, ra_max)
    ra_line_bl, = ax_ra.plot([], [], "-", color=COL_BL, lw=3, label="Baseline")
    ra_line_opt, = ax_ra.plot([], [], "-", color=COL_OPT, lw=3, label="Optimized")
    ra_dot_bl, = ax_ra.plot([], [], "o", color=COL_BL, ms=8)
    ra_dot_opt, = ax_ra.plot([], [], "o", color=COL_OPT, ms=8)
    ax_ra.set_title("Roughness Ra (nm)", color="white", fontsize=20)
    ax_ra.set_xlabel("Time (s)", color=TICK_C, fontsize=14)
    ax_ra.tick_params(colors=TICK_C, labelsize=11)
    ax_ra.grid(True, alpha=0.12, color="#444466")
    ax_ra.legend(fontsize=13, loc="upper right",
                 facecolor="#1a1a3a", edgecolor=SPINE, labelcolor="white")

    suptitle = fig.suptitle("", fontsize=26, color="white", fontweight="bold", y=0.96)

    # Precompute time→snapshot index mapping for speed
    cum_bl = np.array([s["cum_time"] for s in snaps_bl])
    cum_opt = np.array([s["cum_time"] for s in snaps_opt])

    # Precompute RMS-vs-time arrays for curve plotting
    rms_t_bl = np.array(time_all_bl)
    rms_v_bl = np.array(rms_all_bl)
    rms_t_opt = np.array(time_all_opt)
    rms_v_opt = np.array(rms_all_opt)

    def update(frame_idx):
        t = time_points[frame_idx]

        idx_bl = int(np.searchsorted(cum_bl, t, side="right")) - 1
        idx_bl = max(0, min(idx_bl, len(snaps_bl) - 1))
        idx_opt = int(np.searchsorted(cum_opt, t, side="right")) - 1
        idx_opt = max(0, min(idx_opt, len(snaps_opt) - 1))

        s_bl = snaps_bl[idx_bl]
        s_opt = snaps_opt[idx_opt]
        x_bl, y_bl, move_idx_bl, move_frac_bl = interpolate_tool_position(wp_nominal, cum_bl, t)
        x_opt, y_opt, move_idx_opt, move_frac_opt = interpolate_tool_position(wp_opt, cum_opt, t)

        im_bl.set_data(np.ma.array(s_bl["error"], mask=~mask))
        im_opt.set_data(np.ma.array(s_opt["error"], mask=~mask))

        dot_bl.set_data([x_bl], [y_bl])
        dot_opt.set_data([x_opt], [y_opt])
        trail_bl.set_data(*trail_with_current_position(wp_nominal, move_idx_bl, x_bl, y_bl))
        trail_opt.set_data(*trail_with_current_position(wp_opt, move_idx_opt, x_opt, y_opt))

        stat_bl.set_text(f"RMS {s_bl['rms']:.3f} µm    Ra {s_bl.get('ra', 0):.1f} nm")
        stat_opt.set_text(f"RMS {s_opt['rms']:.3f} µm    Ra {s_opt.get('ra', 0):.1f} nm")

        sel_bl = rms_t_bl <= t + 0.01
        sel_opt = rms_t_opt <= t + 0.01
        rms_line_bl.set_data(rms_t_bl[sel_bl], rms_v_bl[sel_bl])
        rms_line_opt.set_data(rms_t_opt[sel_opt], rms_v_opt[sel_opt])
        rms_dot_bl.set_data([s_bl["cum_time"]], [s_bl["rms"]])
        rms_dot_opt.set_data([s_opt["cum_time"]], [s_opt["rms"]])

        ra_line_bl.set_data(ra_t_bl[sel_bl], ra_v_bl[sel_bl])
        ra_line_opt.set_data(ra_t_opt[sel_opt], ra_v_opt[sel_opt])
        ra_dot_bl.set_data([s_bl["cum_time"]], [s_bl.get("ra", 0)])
        ra_dot_opt.set_data([s_opt["cum_time"]], [s_opt.get("ra", 0)])

        bl_done = "  DONE" if t >= T_bl else ""
        opt_done = "  DONE" if t >= T_opt else ""
        pct_bl = min((move_idx_bl + move_frac_bl) / N * 100, 100)
        pct_opt = min((move_idx_opt + move_frac_opt) / N * 100, 100)
        suptitle.set_text(
            f"t = {t:.0f}s   |   "
            f"Baseline {pct_bl:.0f}%{bl_done}   ·   "
            f"Optimized {pct_opt:.0f}%{opt_done}"
        )

        return []

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import io

    dpi = 80
    w_px = int(fig.get_figwidth() * dpi)
    h_px = int(fig.get_figheight() * dpi)

    # Try multiple codecs in order of preference
    codecs = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]
    video = None
    final_path = output_path
    for fourcc_str, ext in codecs:
        try_path = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(str(try_path), fourcc, args.fps, (w_px, h_px))
        if vw.isOpened():
            video = vw
            final_path = try_path
            print(f"  Using codec: {fourcc_str} → {final_path.name}")
            break
        vw.release()

    if video is None:
        # Fallback: write GIF via PIL
        print("  No video codec available, falling back to GIF")
        from PIL import Image as PILImage
        final_path = output_path.with_suffix(".gif")
        pil_frames = []
        print(f"\nRendering {n_frames} frames ({w_px}x{h_px}) ...")
        for fi in range(n_frames):
            update(fi)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor=BG)
            buf.seek(0)
            pil_frames.append(PILImage.open(buf).convert("RGB").copy())
            if (fi + 1) % 50 == 0 or fi == 0:
                print(f"  frame {fi + 1}/{n_frames}")

        pil_frames[0].save(
            str(final_path), save_all=True, append_images=pil_frames[1:],
            duration=1000 // args.fps, loop=0,
        )
        plt.close()
    else:
        print(f"\nRendering {n_frames} frames ({w_px}x{h_px}) ...")
        for fi in range(n_frames):
            update(fi)
            buf = io.BytesIO()
            fig.savefig(buf, format="raw", dpi=dpi, facecolor=BG)
            buf.seek(0)
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h_px, w_px, 4)
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            video.write(frame_bgr)
            if (fi + 1) % 50 == 0 or fi == 0:
                print(f"  frame {fi + 1}/{n_frames}")

        video.release()
        plt.close()

    import os
    size_mb = os.path.getsize(final_path) / 1e6
    duration = n_frames / args.fps
    print(f"\nSaved: {final_path}")
    print(f"  {n_frames} frames, {duration:.0f}s at {args.fps}fps, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
