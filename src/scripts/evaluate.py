"""Evaluate a trained spiral-path polishing agent and generate plots."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def run_evaluation(model_path, config, tif_ckpt, output_dir, seed=0):
    env = make_env(config, tif_ckpt)
    model = SAC.load(model_path, device="cpu")

    obs, _ = env.reset(seed=seed)
    initial_error = env.surface.get_error_map().copy()
    initial_rms = env.initial_rms

    traj_x, traj_y = [], []
    rms_hist, ra_hist, dwell_hist, reward_hist = [], [], [], []

    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        traj_x.append(info["tool_x"])
        traj_y.append(info["tool_y"])
        rms_hist.append(info["rms_error"])
        ra_hist.append(info["mean_roughness"])
        dwell_hist.append(info["dwell_time"])
        reward_hist.append(reward)
        if terminated or truncated:
            break

    final_error = env.surface.get_error_map()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    half = env.lens_diameter_mm / 2.0
    extent = [-half, half, -half, half]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmax = max(abs(initial_error[env.surface.mask]).max(), abs(final_error[env.surface.mask]).max())
    axes[0, 0].imshow(initial_error, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent, origin="lower")
    axes[0, 0].set_title("Initial Error (um)")
    axes[0, 1].imshow(final_error, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent, origin="lower")
    axes[0, 1].set_title("Final Error (um)")

    spiral = env.all_waypoints
    axes[0, 2].plot(spiral[:, 0], spiral[:, 1], "-", color="lightgray", lw=0.5)
    sc = axes[0, 2].scatter(traj_x, traj_y, c=dwell_hist, cmap="hot", s=15, zorder=5)
    axes[0, 2].plot(traj_x[0], traj_y[0], "go", ms=8, label="Start")
    axes[0, 2].plot(traj_x[-1], traj_y[-1], "rs", ms=8, label="End")
    circ = plt.Circle((0, 0), half, fill=False, color="gray", ls="--")
    axes[0, 2].add_patch(circ)
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set_title("Path (color=dwell time)")
    axes[0, 2].legend(fontsize=8)
    plt.colorbar(sc, ax=axes[0, 2], label="dwell (s)")

    axes[1, 0].plot(rms_hist, "b-")
    axes[1, 0].set_title("RMS Error (um)")
    axes[1, 0].set_xlabel("Waypoint")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(range(len(dwell_hist)), dwell_hist, color="orange", width=1.0)
    axes[1, 1].set_title("Dwell Time per Waypoint (s)")
    axes[1, 1].set_xlabel("Waypoint")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(ra_hist, "r-")
    axes[1, 2].set_title("Mean Roughness Ra (nm)")
    axes[1, 2].set_xlabel("Waypoint")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_results.png", dpi=150)
    plt.close()

    print(f"Initial RMS: {initial_rms:.4f} um")
    print(f"Final RMS:   {rms_hist[-1]:.4f} um")
    print(f"Final Ra:    {ra_hist[-1]:.1f} nm")
    print(f"Mean dwell:  {np.mean(dwell_hist):.2f} s")
    print(f"Steps used:  {len(dwell_hist)}")
    print(f"Total reward: {sum(reward_hist):.2f}")
    print(f"Plots saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--tif-checkpoint", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "runs" / "eval"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_evaluation(args.model, config, args.tif_checkpoint, Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()
