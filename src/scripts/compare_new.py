"""Compare NEW RL model vs baseline on the non-uniform surface."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
from stable_baselines3 import SAC
from env.polishing_env import LensPolishingEnv

BASE_DIR = Path(__file__).resolve().parents[1]


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
        obs, r, te, tr, info = env.step(action_fn(obs))
        if te or tr:
            break
    return init_rms, info["rms_error"], info["mean_roughness"]


def main():
    cfg_path = BASE_DIR / "configs" / "default.yaml"
    tif_ckpt = str(BASE_DIR / "checkpoints" / "tif_model.pt")

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    dt_lo, dt_hi = config["env"]["action_ranges"]["dwell_time"]

    print("=" * 65)
    print("  Non-Uniform Surface: RL vs Baseline (seed=42)")
    print("=" * 65)

    env_test = make_env(config, tif_ckpt)
    env_test.reset(seed=42)
    print(f"  Initial RMS: {env_test.initial_rms:.4f} um")
    print()

    print(f"{'Method':<28} {'RMS (um)':>10} {'Ra (nm)':>10} {'RMS drop':>10}")
    print("-" * 65)

    best_base_rms = 999
    best_base_dwell = 0
    for dwell in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.8, 1.0]:
        a_val = np.array([2.0 * (dwell - dt_lo) / (dt_hi - dt_lo) - 1.0])
        env = make_env(config, tif_ckpt)
        init, rms, ra = run_episode(env, lambda obs: a_val)
        drop = (1 - rms / init) * 100
        tag = ""
        if rms < best_base_rms:
            best_base_rms = rms
            best_base_dwell = dwell
            tag = " <-- best"
        print(f"Baseline dwell={dwell:<5.2f}s    {rms:>10.4f} {ra:>10.1f} {drop:>9.1f}%{tag}")

    print("-" * 65)

    for exp_name in ["exp0_baseline", "exp1_more_form", "exp2_more_coverage", "exp3_long_episode"]:
        model_path = BASE_DIR / "runs" / exp_name / "checkpoints" / "best_model.zip"
        if not model_path.exists():
            continue
        model = SAC.load(str(model_path), device="cpu")
        env = make_env(config, tif_ckpt)
        init, rms, ra = run_episode(env, lambda obs: model.predict(obs, deterministic=True)[0])
        drop = (1 - rms / init) * 100
        adv = (best_base_rms - rms) / best_base_rms * 100
        print(f"RL {exp_name:<23} {rms:>10.4f} {ra:>10.1f} {drop:>9.1f}%  (vs best base: {adv:+.1f}%)")

    print("=" * 65)
    print(f"  Best baseline: dwell={best_base_dwell}s -> RMS={best_base_rms:.4f}")


if __name__ == "__main__":
    main()
