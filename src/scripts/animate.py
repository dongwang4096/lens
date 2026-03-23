"""Generate polishing animation: RL vs baseline comparison.

Shows surface error AND roughness side by side with convergence curves.
Supports --baseline mode (uniform dwell time) for comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from stable_baselines3 import SAC

from env.polishing_env import LensPolishingEnv


def make_env(config, tif_ckpt):
    return LensPolishingEnv(
        grid_size=config["surface"]["grid_size"],
        lens_diameter_mm=config["surface"]["lens_diameter_mm"],
        curvature_radius_mm=config["surface"]["curvature_radius_mm"],
        initial_roughness_nm=config["surface"]["initial_roughness_nm"],
        target_rms_um=config["surface"]["target_rms_um"],
        reward_form=config["env"]["reward_weights"]["form"],
        reward_roughness=config["env"]["reward_weights"]["roughness"],
        reward_bonus=config["env"]["reward_weights"]["completion_bonus"],
        num_passes=config["spiral"]["num_passes"],
        pass_pressures=config["spiral"]["pass_pressures"],
        fixed_speed=config["spiral"]["fixed_speed"],
        concentration=config["tif"]["default_concentration"],
        dwell_range=tuple(config["env"]["action_ranges"]["dwell_time"]),
        pitch_mm=config["spiral"]["pitch_mm"],
        patch_size=config["env"]["patch_size"],
        tif_checkpoint=tif_ckpt,
    )


def collect_frames(env, model, seed, baseline_dwell=None):
    """Run one episode. If baseline_dwell is set, use constant dwell instead of model."""
    obs, _ = env.reset(seed=seed)
    frames = [{
        "error": env.surface.get_error_map().copy(),
        "roughness": env.roughness_model.get_roughness_map().copy(),
        "tool_x": 0.0, "tool_y": 0.0,
        "rms": env.initial_rms,
        "ra": env.prev_roughness,
        "dwell": 0.0, "step": 0,
        "pass": 0, "pressure": env.pass_pressures[0],
    }]

    for step in range(env.max_steps):
        if baseline_dwell is not None:
            action = np.array([baseline_dwell], dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append({
            "error": env.surface.get_error_map().copy(),
            "roughness": env.roughness_model.get_roughness_map().copy(),
            "tool_x": info["tool_x"], "tool_y": info["tool_y"],
            "rms": info["rms_error"], "ra": info["mean_roughness"],
            "dwell": info["dwell_time"], "step": step + 1,
            "pass": info.get("current_pass", 0),
            "pressure": info.get("pressure", 25.0),
        })
        if terminated or truncated:
            break
    return frames


def render_animation(frames, env, output_path, config, label="RL Agent",
                     fixed_err_range=3.0, fixed_ra_range=220.0, fixed_rms_ylim=None):
    half = env.lens_diameter_mm / 2.0
    extent = [-half, half, -half, half]
    n = len(frames)
    FS_TITLE = 18
    FS_AX_TITLE = 14
    FS_TICK = 10
    FS_LABEL = 12
    FS_INFO = 16

    fig = plt.figure(figsize=(22, 12), facecolor="#0a0a1a")
    gs = fig.add_gridspec(2, 5, hspace=0.32, wspace=0.35,
                          left=0.04, right=0.97, top=0.90, bottom=0.07)

    ax_err = fig.add_subplot(gs[0, 0])
    ax_rough = fig.add_subplot(gs[0, 1])
    ax_path = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[0, 3:])
    ax_rms = fig.add_subplot(gs[1, 0:2])
    ax_ra = fig.add_subplot(gs[1, 2:4])
    ax_dwell = fig.add_subplot(gs[1, 4])

    all_axes = [ax_err, ax_rough, ax_path, ax_info, ax_rms, ax_ra, ax_dwell]
    for ax in all_axes:
        ax.set_facecolor("#0f0f2a")
        for s in ax.spines.values():
            s.set_color("#333366")

    err_norm = TwoSlopeNorm(vmin=-fixed_err_range, vcenter=0, vmax=fixed_err_range)

    title = fig.suptitle("", fontsize=FS_TITLE, color="white", fontweight="bold", y=0.96)

    err_disp = np.ma.array(frames[0]["error"], mask=~env.surface.mask)
    im_err = ax_err.imshow(err_disp, cmap="RdBu_r", norm=err_norm, extent=extent,
                            origin="lower", interpolation="bilinear")
    tool_dot1, = ax_err.plot([], [], "o", color="#00ff88", ms=12,
                              markeredgecolor="white", markeredgewidth=2, zorder=10)
    ax_err.add_patch(plt.Circle((0, 0), half, fill=False, color="#555588", lw=1))
    ax_err.set_xlim(-half*1.05, half*1.05)
    ax_err.set_ylim(-half*1.05, half*1.05)
    ax_err.set_aspect("equal")
    ax_err.set_title("Surface Error (um)", color="white", fontsize=FS_AX_TITLE)
    ax_err.tick_params(colors="#888888", labelsize=FS_TICK)

    rough_disp = np.ma.array(frames[0]["roughness"], mask=~env.surface.mask)
    im_rough = ax_rough.imshow(rough_disp, cmap="inferno", vmin=0, vmax=fixed_ra_range,
                                extent=extent, origin="lower", interpolation="bilinear")
    tool_dot2, = ax_rough.plot([], [], "o", color="#00ff88", ms=12,
                                markeredgecolor="white", markeredgewidth=2, zorder=10)
    ax_rough.add_patch(plt.Circle((0, 0), half, fill=False, color="#555588", lw=1))
    ax_rough.set_xlim(-half*1.05, half*1.05)
    ax_rough.set_ylim(-half*1.05, half*1.05)
    ax_rough.set_aspect("equal")
    ax_rough.set_title("Roughness Ra (nm)", color="white", fontsize=FS_AX_TITLE)
    ax_rough.tick_params(colors="#888888", labelsize=FS_TICK)

    spiral_xy = env.all_waypoints
    ax_path.plot(spiral_xy[:, 0], spiral_xy[:, 1], "-", color="#222255", lw=0.3, alpha=0.4)
    path_done, = ax_path.plot([], [], "-", color="#4488ff", lw=1.5, alpha=0.7)
    path_head, = ax_path.plot([], [], "o", color="#00ff88", ms=10,
                               markeredgecolor="white", markeredgewidth=2, zorder=10)
    ax_path.add_patch(plt.Circle((0, 0), half, fill=False, color="#555588", lw=1, ls="--"))
    ax_path.set_xlim(-half*1.15, half*1.15)
    ax_path.set_ylim(-half*1.15, half*1.15)
    ax_path.set_aspect("equal")
    ax_path.set_title("Tool Path", color="white", fontsize=FS_AX_TITLE)
    ax_path.tick_params(colors="#888888", labelsize=FS_TICK)

    ax_info.axis("off")
    info_text = ax_info.text(0.5, 0.5, "", transform=ax_info.transAxes,
                              fontsize=FS_INFO, color="white", fontfamily="monospace",
                              ha="center", va="center",
                              bbox=dict(boxstyle="round,pad=0.6", fc="#1a1a3a",
                                        ec="#333366", alpha=0.9))

    rms_vals = [f["rms"] for f in frames]
    ra_vals = [f["ra"] for f in frames]
    dwell_vals = [f["dwell"] for f in frames]
    trail_x = [f["tool_x"] for f in frames]
    trail_y = [f["tool_y"] for f in frames]

    rms_ymax = fixed_rms_ylim if fixed_rms_ylim else max(rms_vals) * 1.15
    ax_rms.set_xlim(0, n)
    ax_rms.set_ylim(0, rms_ymax)
    rms_line, = ax_rms.plot([], [], "-", color="#ff6644", lw=2.5)
    rms_dot, = ax_rms.plot([], [], "o", color="#ff6644", ms=7)
    ax_rms.set_title("RMS Error (um)", color="white", fontsize=FS_AX_TITLE)
    ax_rms.set_xlabel("Waypoint", color="#888888", fontsize=FS_LABEL)
    ax_rms.tick_params(colors="#888888", labelsize=FS_TICK)
    ax_rms.grid(True, alpha=0.15, color="#444466")

    ax_ra.set_xlim(0, n)
    ax_ra.set_ylim(0, fixed_ra_range)
    ra_line, = ax_ra.plot([], [], "-", color="#44bbff", lw=2.5)
    ra_dot, = ax_ra.plot([], [], "o", color="#44bbff", ms=7)
    ax_ra.set_title("Roughness Ra (nm)", color="white", fontsize=FS_AX_TITLE)
    ax_ra.set_xlabel("Waypoint", color="#888888", fontsize=FS_LABEL)
    ax_ra.tick_params(colors="#888888", labelsize=FS_TICK)
    ax_ra.grid(True, alpha=0.15, color="#444466")

    ax_dwell.set_xlim(0, n)
    ax_dwell.set_ylim(0, max(dwell_vals) * 1.2 + 0.1)
    dwell_line, = ax_dwell.plot([], [], "-", color="#ffaa00", lw=2)
    dwell_dot, = ax_dwell.plot([], [], "o", color="#ffaa00", ms=7)
    ax_dwell.set_title("Dwell (s)", color="white", fontsize=FS_AX_TITLE)
    ax_dwell.set_xlabel("Waypoint", color="#888888", fontsize=FS_LABEL)
    ax_dwell.tick_params(colors="#888888", labelsize=FS_TICK)
    ax_dwell.grid(True, alpha=0.15, color="#444466")

    def update(i):
        f = frames[i]
        im_err.set_data(np.ma.array(f["error"], mask=~env.surface.mask))
        im_rough.set_data(np.ma.array(f["roughness"], mask=~env.surface.mask))
        tool_dot1.set_data([f["tool_x"]], [f["tool_y"]])
        tool_dot2.set_data([f["tool_x"]], [f["tool_y"]])

        path_done.set_data(trail_x[:i+1], trail_y[:i+1])
        path_head.set_data([f["tool_x"]], [f["tool_y"]])

        rms_line.set_data(range(i+1), rms_vals[:i+1])
        rms_dot.set_data([i], [rms_vals[i]])
        ra_line.set_data(range(i+1), ra_vals[:i+1])
        ra_dot.set_data([i], [ra_vals[i]])
        dwell_line.set_data(range(i+1), dwell_vals[:i+1])
        dwell_dot.set_data([i], [dwell_vals[i]])

        pct = f["step"] / env.max_steps * 100
        title.set_text(
            f"{label}  |  "
            f"Pass {f['pass']+1}/{config['spiral']['num_passes']}  "
            f"({pct:.0f}%)"
        )
        info_text.set_text(
            f"  RMS:   {f['rms']:.3f} um\n"
            f"  Ra:    {f['ra']:.1f} nm\n"
            f"  Dwell: {f['dwell']:.2f} s\n"
            f"  P={f['pressure']:.0f}N  "
            f"V={config['spiral']['fixed_speed']:.0f}rpm"
        )
        return [im_err, im_rough, tool_dot1, tool_dot2,
                path_done, path_head, rms_line, rms_dot,
                ra_line, ra_dot, dwell_line, dwell_dot, title, info_text]

    frame_step = max(1, n // 250)
    frame_indices = list(range(0, n, frame_step))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=120, blit=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="pillow", fps=8, dpi=80)
    plt.close()
    print(f"Animation saved: {output_path} ({len(frame_indices)} frames, {len(frame_indices)/6:.0f}s at 6fps)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="RL model .zip (omit for baseline)")
    parser.add_argument("--baseline-dwell", type=float, default=None,
                        help="Constant dwell time for baseline (e.g. 0.5)")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--tif-checkpoint", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "runs" / "animation.gif"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = make_env(config, args.tif_checkpoint)

    if args.baseline_dwell is not None:
        dt_lo, dt_hi = config["env"]["action_ranges"]["dwell_time"]
        action_val = 2.0 * (args.baseline_dwell - dt_lo) / (dt_hi - dt_lo) - 1.0
        frames = collect_frames(env, None, args.seed, baseline_dwell=action_val)
        label = f"Baseline (uniform dwell={args.baseline_dwell:.1f}s)"
    else:
        model = SAC.load(args.model, device="cpu")
        frames = collect_frames(env, model, args.seed)
        label = "RL Agent (adaptive dwell)"

    print(f"Collected {len(frames)} frames. Rendering...")
    render_animation(frames, env, args.output, config, label=label,
                     fixed_err_range=3.0, fixed_ra_range=220.0, fixed_rms_ylim=3.0)


if __name__ == "__main__":
    main()
