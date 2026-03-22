"""Generate a polishing process animation for presentation.

Creates an MP4 video showing:
  - Left: 2D error map evolving in real-time with tool position marker
  - Right-top: tool trajectory trail
  - Right-mid: RMS error convergence curve
  - Right-bot: roughness convergence curve
  - Bottom bar: process parameters readout
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import TwoSlopeNorm
from stable_baselines3 import SAC

from env.polishing_env import LensPolishingEnv


def run_and_animate(model_path: str, config: dict, tif_ckpt: str, output_path: str, seed: int = 0):
    env = LensPolishingEnv(
        grid_size=config["surface"]["grid_size"],
        lens_diameter_mm=config["surface"]["lens_diameter_mm"],
        curvature_radius_mm=config["surface"]["curvature_radius_mm"],
        initial_roughness_nm=config["surface"]["initial_roughness_nm"],
        target_rms_um=config["surface"]["target_rms_um"],
        max_steps=config["env"]["max_steps"],
        time_budget_s=config["env"]["time_budget_s"],
        reward_form=config["env"]["reward_weights"]["form"],
        reward_roughness=config["env"]["reward_weights"]["roughness"],
        reward_time=config["env"]["reward_weights"]["time_penalty"],
        reward_bonus=config["env"]["reward_weights"]["completion_bonus"],
        pressure_range=tuple(config["env"]["action_ranges"]["pressure"]),
        speed_range=tuple(config["env"]["action_ranges"]["speed"]),
        dwell_range=tuple(config["env"]["action_ranges"]["dwell_time"]),
        concentration=config["tif"]["default_concentration"],
        tif_checkpoint=tif_ckpt,
    )

    model = SAC.load(model_path, device="cpu")

    obs, _ = env.reset(seed=seed)

    frames = []
    half = env.lens_diameter_mm / 2.0
    frame0 = {
        "error": env.surface.get_error_map().copy(),
        "roughness": env.roughness_model.get_roughness_map().copy(),
        "tool_x": 0.0,
        "tool_y": 0.0,
        "rms": env.surface.get_rms_error(),
        "ra": env.roughness_model.get_mean_roughness(env.surface.mask),
        "pressure": 0.0,
        "speed": 0.0,
        "dwell": 0.0,
        "time": 0.0,
        "step": 0,
    }
    frames.append(frame0)

    for step in range(config["env"]["max_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        frames.append({
            "error": env.surface.get_error_map().copy(),
            "roughness": env.roughness_model.get_roughness_map().copy(),
            "tool_x": info["tool_x"],
            "tool_y": info["tool_y"],
            "rms": info["rms_error"],
            "ra": info["mean_roughness"],
            "pressure": info["pressure"],
            "speed": info["speed"],
            "dwell": info["dwell_time"],
            "time": info["elapsed_time"],
            "step": step + 1,
        })
        if terminated or truncated:
            break

    print(f"Recorded {len(frames)} frames. Generating animation...")

    _render_animation(frames, env, output_path, config)


def _render_animation(frames, env, output_path, config):
    half = env.lens_diameter_mm / 2.0
    extent = [-half, half, -half, half]
    n_frames = len(frames)

    fig = plt.figure(figsize=(16, 10), facecolor="#0a0a1a")

    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.96, top=0.92, bottom=0.08)

    ax_error = fig.add_subplot(gs[:2, 0])
    ax_rough = fig.add_subplot(gs[:2, 1])
    ax_path = fig.add_subplot(gs[0, 2])
    ax_rms = fig.add_subplot(gs[1, 2])
    ax_info = fig.add_subplot(gs[2, :])

    for ax in [ax_error, ax_rough, ax_path, ax_rms, ax_info]:
        ax.set_facecolor("#0f0f2a")
        for spine in ax.spines.values():
            spine.set_color("#333366")

    all_errors = np.concatenate([f["error"][env.surface.mask] for f in frames])
    vmax_err = max(abs(all_errors.min()), abs(all_errors.max()))
    err_norm = TwoSlopeNorm(vmin=-vmax_err, vcenter=0, vmax=vmax_err)

    title = fig.suptitle("", fontsize=16, color="white", fontweight="bold", y=0.97)

    err_display = np.ma.array(frames[0]["error"], mask=~env.surface.mask)
    im_err = ax_error.imshow(err_display, cmap="RdBu_r", norm=err_norm,
                              extent=extent, origin="lower", interpolation="bilinear")
    tool_marker, = ax_error.plot([], [], "o", color="#00ff88", markersize=12,
                                  markeredgecolor="white", markeredgewidth=2, zorder=10)
    tool_ring = plt.Circle((0, 0), 0, fill=False, color="#00ff88", linewidth=1.5,
                            linestyle="--", alpha=0.6)
    ax_error.add_patch(tool_ring)
    circle_border = plt.Circle((0, 0), half, fill=False, color="#555588", linewidth=1.5)
    ax_error.add_patch(circle_border)
    ax_error.set_xlim(-half * 1.05, half * 1.05)
    ax_error.set_ylim(-half * 1.05, half * 1.05)
    ax_error.set_aspect("equal")
    ax_error.set_title("Surface Error (μm)", color="white", fontsize=12)
    ax_error.tick_params(colors="#888888")
    cb_err = plt.colorbar(im_err, ax=ax_error, shrink=0.8, pad=0.02)
    cb_err.ax.tick_params(colors="#888888")
    cb_err.set_label("μm", color="#888888")

    rough_display = np.ma.array(frames[0]["roughness"], mask=~env.surface.mask)
    im_rough = ax_rough.imshow(rough_display, cmap="inferno",
                                vmin=0, vmax=env.roughness_model.initial_ra_nm * 1.1,
                                extent=extent, origin="lower", interpolation="bilinear")
    tool_marker2, = ax_rough.plot([], [], "o", color="#00ff88", markersize=12,
                                   markeredgecolor="white", markeredgewidth=2, zorder=10)
    circle_border2 = plt.Circle((0, 0), half, fill=False, color="#555588", linewidth=1.5)
    ax_rough.add_patch(circle_border2)
    ax_rough.set_xlim(-half * 1.05, half * 1.05)
    ax_rough.set_ylim(-half * 1.05, half * 1.05)
    ax_rough.set_aspect("equal")
    ax_rough.set_title("Roughness Ra (nm)", color="white", fontsize=12)
    ax_rough.tick_params(colors="#888888")
    cb_rough = plt.colorbar(im_rough, ax=ax_rough, shrink=0.8, pad=0.02)
    cb_rough.ax.tick_params(colors="#888888")
    cb_rough.set_label("nm", color="#888888")

    trail_x = [f["tool_x"] for f in frames]
    trail_y = [f["tool_y"] for f in frames]
    ax_path.plot(trail_x, trail_y, "-", color="#334488", linewidth=0.8, alpha=0.4)
    path_line, = ax_path.plot([], [], "-", color="#4488ff", linewidth=1.5, alpha=0.8)
    path_head, = ax_path.plot([], [], "o", color="#00ff88", markersize=10,
                               markeredgecolor="white", markeredgewidth=2, zorder=10)
    path_start, = ax_path.plot(trail_x[0], trail_y[0], "s", color="#ffaa00",
                                markersize=8, zorder=9)
    circle_path = plt.Circle((0, 0), half, fill=False, color="#555588",
                              linewidth=1.5, linestyle="--")
    ax_path.add_patch(circle_path)
    ax_path.set_xlim(-half * 1.15, half * 1.15)
    ax_path.set_ylim(-half * 1.15, half * 1.15)
    ax_path.set_aspect("equal")
    ax_path.set_title("Tool Path", color="white", fontsize=11)
    ax_path.tick_params(colors="#888888")

    rms_vals = [f["rms"] for f in frames]
    ax_rms.set_xlim(0, n_frames)
    ax_rms.set_ylim(0, max(rms_vals) * 1.15)
    rms_line, = ax_rms.plot([], [], "-", color="#ff6644", linewidth=2)
    rms_dot, = ax_rms.plot([], [], "o", color="#ff6644", markersize=8)
    ax_rms.set_title("RMS Error (μm)", color="white", fontsize=11)
    ax_rms.set_xlabel("Step", color="#888888", fontsize=9)
    ax_rms.tick_params(colors="#888888")
    ax_rms.grid(True, alpha=0.15, color="#444466")

    ax_info.axis("off")
    info_text = ax_info.text(0.5, 0.5, "", transform=ax_info.transAxes,
                              fontsize=13, color="white", fontfamily="monospace",
                              ha="center", va="center",
                              bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a3a",
                                        edgecolor="#333366", alpha=0.9))

    def update(frame_idx):
        f = frames[frame_idx]

        err_display = np.ma.array(f["error"], mask=~env.surface.mask)
        im_err.set_data(err_display)

        rough_display = np.ma.array(f["roughness"], mask=~env.surface.mask)
        im_rough.set_data(rough_display)

        tx, ty = f["tool_x"], f["tool_y"]
        tool_marker.set_data([tx], [ty])
        tool_marker2.set_data([tx], [ty])

        w_mm = 5.0
        tool_ring.set_center((tx, ty))
        tool_ring.set_radius(w_mm)

        path_line.set_data(trail_x[:frame_idx + 1], trail_y[:frame_idx + 1])
        path_head.set_data([tx], [ty])

        rms_line.set_data(range(frame_idx + 1), rms_vals[:frame_idx + 1])
        rms_dot.set_data([frame_idx], [rms_vals[frame_idx]])

        step = f["step"]
        total = config["env"]["max_steps"]
        pct = step / total * 100 if total > 0 else 0

        title.set_text(
            f"Convex Lens Polishing — RL Agent  |  "
            f"Step {step}/{total} ({pct:.0f}%)"
        )

        info_str = (
            f"  RMS: {f['rms']:.3f} μm  |  "
            f"Ra: {f['ra']:.1f} nm  |  "
            f"Pressure: {f['pressure']:.1f} N  |  "
            f"Speed: {f['speed']:.0f} rpm  |  "
            f"Dwell: {f['dwell']:.2f} s  |  "
            f"Time: {f['time']:.1f} s  "
        )
        info_text.set_text(info_str)

        return [im_err, im_rough, tool_marker, tool_marker2, tool_ring,
                path_line, path_head, rms_line, rms_dot, title, info_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".mp4":
        writer = FFMpegWriter(fps=15, bitrate=2000)
        anim.save(str(output_path), writer=writer, dpi=120)
    else:
        anim.save(str(output_path), writer="pillow", fps=12, dpi=100)

    plt.close(fig)
    print(f"Animation saved to {output_path}")
    print(f"  Frames: {n_frames}, Duration: {n_frames / 15:.1f}s (at 15fps)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model .zip")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--tif-checkpoint", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "runs" / "polishing_animation.gif"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_and_animate(args.model, config, args.tif_checkpoint, args.output, args.seed)


if __name__ == "__main__":
    main()
