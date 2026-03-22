"""Evaluate a trained agent and generate visualizations."""

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


def run_evaluation(model_path: str, config: dict, tif_ckpt: str, output_dir: Path, seed: int = 0):
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
    initial_error = env.surface.get_error_map().copy()
    initial_roughness = env.roughness_model.get_roughness_map().copy()
    initial_rms = env.surface.get_rms_error()

    trajectory_x, trajectory_y = [0.0], [0.0]
    rms_history, roughness_history = [initial_rms], [env.roughness_model.get_mean_roughness(env.surface.mask)]
    pressure_history, speed_history, dwell_history = [], [], []
    reward_history = []

    for step in range(config["env"]["max_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        trajectory_x.append(info["tool_x"])
        trajectory_y.append(info["tool_y"])
        rms_history.append(info["rms_error"])
        roughness_history.append(info["mean_roughness"])
        pressure_history.append(info["pressure"])
        speed_history.append(info["speed"])
        dwell_history.append(info["dwell_time"])
        reward_history.append(reward)

        if terminated or truncated:
            break

    final_error = env.surface.get_error_map()
    final_roughness = env.roughness_model.get_roughness_map()

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_results(
        output_dir, env, initial_error, final_error,
        initial_roughness, final_roughness,
        trajectory_x, trajectory_y,
        rms_history, roughness_history,
        pressure_history, speed_history, dwell_history,
        reward_history,
    )

    print(f"Initial RMS: {initial_rms:.4f} um")
    print(f"Final RMS:   {rms_history[-1]:.4f} um")
    print(f"Final Ra:    {roughness_history[-1]:.1f} nm")
    print(f"Steps used:  {len(reward_history)}")
    print(f"Total reward: {sum(reward_history):.2f}")
    print(f"Plots saved to {output_dir}")


def _plot_results(
    output_dir, env, initial_error, final_error,
    initial_roughness, final_roughness,
    traj_x, traj_y, rms_hist, rough_hist,
    p_hist, v_hist, dt_hist, rew_hist,
):
    half = env.lens_diameter_mm / 2.0
    extent = [-half, half, -half, half]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmax = max(abs(initial_error[env.surface.mask]).max(), abs(final_error[env.surface.mask]).max())
    im0 = axes[0, 0].imshow(initial_error, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent)
    axes[0, 0].set_title("Initial Error (um)")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(final_error, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent)
    axes[0, 1].set_title("Final Error (um)")
    plt.colorbar(im1, ax=axes[0, 1])

    axes[0, 2].plot(traj_x, traj_y, "b-", alpha=0.5, linewidth=0.5)
    axes[0, 2].plot(traj_x[0], traj_y[0], "go", markersize=8, label="Start")
    axes[0, 2].plot(traj_x[-1], traj_y[-1], "rs", markersize=8, label="End")
    circle = plt.Circle((0, 0), half, fill=False, color="gray", linestyle="--")
    axes[0, 2].add_patch(circle)
    axes[0, 2].set_xlim(-half * 1.1, half * 1.1)
    axes[0, 2].set_ylim(-half * 1.1, half * 1.1)
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set_title("Tool Path")
    axes[0, 2].legend()

    axes[1, 0].plot(rms_hist, "b-")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("RMS Error (um)")
    axes[1, 0].set_title("RMS Error Convergence")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(rough_hist, "r-")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Mean Ra (nm)")
    axes[1, 1].set_title("Roughness Convergence")
    axes[1, 1].grid(True, alpha=0.3)

    ax_p = axes[1, 2]
    ax_p.plot(p_hist, "g-", alpha=0.7, label="Pressure (N)")
    ax_p.set_xlabel("Step")
    ax_p.set_ylabel("Pressure (N)")
    ax_v = ax_p.twinx()
    ax_v.plot(v_hist, "m-", alpha=0.7, label="Speed (rpm)")
    ax_v.set_ylabel("Speed (rpm)")
    ax_p.set_title("Process Parameters")
    lines1, labels1 = ax_p.get_legend_handles_labels()
    lines2, labels2 = ax_v.get_legend_handles_labels()
    ax_p.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax_p.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_results.png", dpi=150)
    plt.close()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    im_r0 = axes2[0].imshow(initial_roughness, cmap="hot", extent=extent)
    axes2[0].set_title("Initial Roughness (nm)")
    plt.colorbar(im_r0, ax=axes2[0])
    im_r1 = axes2[1].imshow(final_roughness, cmap="hot", extent=extent)
    axes2[1].set_title("Final Roughness (nm)")
    plt.colorbar(im_r1, ax=axes2[1])
    plt.tight_layout()
    plt.savefig(output_dir / "roughness_comparison.png", dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model .zip")
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
