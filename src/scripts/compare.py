"""Compare RL champion vs baseline at various uniform dwell times."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
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


def run_episode(env, action_fn):
    obs, _ = env.reset(seed=42)
    init_rms = env.initial_rms
    for step in range(env.max_steps):
        a = action_fn(obs)
        obs, r, te, tr, info = env.step(a)
        if te or tr:
            break
    return init_rms, info["rms_error"], info["mean_roughness"]


def main():
    cfg_path = Path(__file__).resolve().parents[1] / "runs" / "sweep" / "s16_arch3x256" / "config.yaml"
    tif_ckpt = str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt")
    model_path = str(Path(__file__).resolve().parents[1] / "runs" / "sweep" / "s16_arch3x256" / "checkpoints" / "best_model.zip")

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    dt_lo, dt_hi = config["env"]["action_ranges"]["dwell_time"]

    print("=" * 60)
    print("  Baseline vs RL Comparison (same surface seed=42)")
    print("=" * 60)

    print(f"\n{'Method':<25} {'RMS (um)':>10} {'Ra (nm)':>10} {'RMS drop':>10}")
    print("-" * 60)

    for dwell in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        a_val = np.array([2.0 * (dwell - dt_lo) / (dt_hi - dt_lo) - 1.0])
        env = make_env(config, tif_ckpt)
        init, rms, ra = run_episode(env, lambda obs: a_val)
        drop = (1 - rms / init) * 100
        print(f"Baseline dwell={dwell:<5.2f}s  {rms:>10.4f} {ra:>10.1f} {drop:>9.1f}%")

    print("-" * 60)

    model = SAC.load(model_path, device="cpu")
    env = make_env(config, tif_ckpt)
    init, rms, ra = run_episode(env, lambda obs: model.predict(obs, deterministic=True)[0])
    drop = (1 - rms / init) * 100
    print(f"{'RL champion':<25} {rms:>10.4f} {ra:>10.1f} {drop:>9.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
