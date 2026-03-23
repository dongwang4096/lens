"""Training script for spiral-path lens polishing RL (SAC, MlpPolicy).

The environment uses a fixed spiral path; the agent only controls dwell time.
Supports parallel environments via SubprocVecEnv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env.polishing_env import LensPolishingEnv


class PolishingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rms = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "rms_error" in info:
                self.episode_rms.append(info["rms_error"])
        if any(self.locals.get("dones", [])):
            if self.episode_rms:
                self.logger.record("polishing/final_rms", self.episode_rms[-1])
                self.logger.record("polishing/min_rms", min(self.episode_rms))
                self.episode_rms = []
        return True


def _make_env_fn(config: dict, tif_checkpoint: str, seed: int):
    def _init():
        env = LensPolishingEnv(
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
            tif_checkpoint=tif_checkpoint,
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def make_vec_env(config, tif_ckpt, n_envs, base_seed):
    if n_envs == 1:
        return DummyVecEnv([_make_env_fn(config, tif_ckpt, base_seed)])
    return SubprocVecEnv(
        [_make_env_fn(config, tif_ckpt, base_seed + i) for i in range(n_envs)],
        start_method="fork",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
    parser.add_argument("--tif-checkpoint", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"))
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    total_timesteps = args.total_timesteps or config["training"]["total_timesteps"]
    n_envs = args.n_envs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tb_dir = output_dir / "tb_logs"
    tb_dir.mkdir(exist_ok=True)

    print(f"Creating {n_envs} parallel environments...")
    train_env = make_vec_env(config, args.tif_checkpoint, n_envs, config["training"]["seed"])
    eval_env = make_vec_env(config, args.tif_checkpoint, 1, config["training"]["seed"] + 9999)

    print(f"Initializing SAC (MlpPolicy, device={args.device})...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        buffer_size=config["training"]["buffer_size"],
        learning_starts=config["training"]["learning_starts"],
        gamma=config["training"]["gamma"],
        tau=config["training"]["tau"],
        train_freq=config["training"]["train_freq"],
        gradient_steps=config["training"]["gradient_steps"],
        policy_kwargs=dict(net_arch=config["policy"]["net_arch"]),
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=config["training"]["seed"],
        device=args.device,
    )

    early_stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config["training"].get("patience", 15),
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(config["training"]["eval_freq"] // n_envs, 1),
        n_eval_episodes=config["training"]["n_eval_episodes"],
        deterministic=True,
        callback_after_eval=early_stop_cb,
    )

    print(f"\nTraining for {total_timesteps} timesteps...")
    print(f"  Grid: {config['surface']['grid_size']}x{config['surface']['grid_size']}")
    print(f"  Parallel envs: {n_envs}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, PolishingMetricsCallback()],
        log_interval=config["training"]["log_interval"],
        progress_bar=True,
    )

    model.save(str(ckpt_dir / "final_model"))
    print(f"\nDone. Model saved to {ckpt_dir}")

    print("\nFinal evaluation...")
    single = DummyVecEnv([_make_env_fn(config, args.tif_checkpoint, seed=0)])
    obs = single.reset()
    total_r = 0
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = single.step(action)
        total_r += reward[0]
        if done[0]:
            info = infos[0]
            print(f"  Episode done at step {step+1}: RMS={info.get('rms_error',0):.4f}um, Ra={info.get('mean_roughness',0):.1f}nm")
            break
    print(f"  Total reward: {total_r:.2f}")

    train_env.close()
    eval_env.close()
    single.close()


if __name__ == "__main__":
    main()
