"""Training script for the lens polishing RL agent using SAC.

Supports parallel environment collection via SubprocVecEnv for faster training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env.polishing_env import LensPolishingEnv
from agent.policy import LensPolishingExtractor


class PolishingMetricsCallback(BaseCallback):
    """Log polishing-specific metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rms = []
        self.episode_roughness = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "rms_error" in info:
                self.episode_rms.append(info["rms_error"])
                self.episode_roughness.append(info["mean_roughness"])

        dones = self.locals.get("dones", [])
        if any(dones):
            if self.episode_rms:
                self.logger.record("polishing/final_rms", self.episode_rms[-1])
                self.logger.record("polishing/final_roughness", self.episode_roughness[-1])
                self.logger.record("polishing/min_rms", min(self.episode_rms))
                self.episode_rms = []
                self.episode_roughness = []
        return True


def _make_env_fn(config: dict, tif_checkpoint: str, seed: int):
    """Return a callable that creates a single monitored env instance."""
    def _init():
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
            tif_checkpoint=tif_checkpoint,
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def make_vec_env(config: dict, tif_checkpoint: str, n_envs: int, base_seed: int):
    """Create a vectorized environment with n_envs parallel workers."""
    if n_envs == 1:
        return DummyVecEnv([_make_env_fn(config, tif_checkpoint, base_seed)])
    return SubprocVecEnv(
        [_make_env_fn(config, tif_checkpoint, base_seed + i) for i in range(n_envs)],
        start_method="fork",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
    parser.add_argument("--tif-checkpoint", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"))
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    total_timesteps = args.total_timesteps or config["training"]["total_timesteps"]
    n_envs = args.n_envs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = output_dir / "tb_logs"
    tb_log_dir.mkdir(exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Creating {n_envs} parallel training environments...")
    train_env = make_vec_env(config, args.tif_checkpoint, n_envs, base_seed=config["training"]["seed"])

    print("Creating eval environment...")
    eval_env = make_vec_env(config, args.tif_checkpoint, 1, base_seed=config["training"]["seed"] + 1000)

    policy_kwargs = dict(
        features_extractor_class=LensPolishingExtractor,
        features_extractor_kwargs=dict(
            cnn_channels=config["policy"]["cnn_channels"],
            features_dim=config["policy"]["mlp_hidden"][0],
        ),
        net_arch=config["policy"]["mlp_hidden"],
    )

    print(f"Initializing SAC agent (device={args.device}, n_envs={n_envs})...")
    model = SAC(
        "MultiInputPolicy",
        train_env,
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        buffer_size=config["training"]["buffer_size"],
        gamma=config["training"]["gamma"],
        tau=config["training"]["tau"],
        train_freq=config["training"]["train_freq"],
        gradient_steps=config["training"]["gradient_steps"],
        learning_starts=config["training"].get("learning_starts", 2000),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_log_dir),
        verbose=1,
        seed=config["training"]["seed"],
        device=args.device,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(config["training"]["eval_freq"] // n_envs, 1),
        n_eval_episodes=config["training"]["n_eval_episodes"],
        deterministic=True,
    )

    metrics_callback = PolishingMetricsCallback()

    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"  Grid size: {config['surface']['grid_size']}")
    print(f"  Max steps/episode: {config['env']['max_steps']}")
    print(f"  Target RMS: {config['surface']['target_rms_um']} um")
    print(f"  Parallel envs: {n_envs}")
    print(f"  TensorBoard logs: {tb_log_dir}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, metrics_callback],
        log_interval=config["training"]["log_interval"],
        progress_bar=True,
    )

    final_path = ckpt_dir / "final_model"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")

    print("\nRunning final evaluation...")
    single_eval = DummyVecEnv([_make_env_fn(config, args.tif_checkpoint, seed=0)])
    obs = single_eval.reset()
    total_reward = 0.0
    info = {}
    for step in range(config["env"]["max_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = single_eval.step(action)
        total_reward += reward[0]
        info = infos[0]
        if (step + 1) % 50 == 0 or done[0]:
            print(
                f"  Step {step+1}: RMS={info.get('rms_error', 0):.4f}um, "
                f"Ra={info.get('mean_roughness', 0):.1f}nm, "
                f"time={info.get('elapsed_time', 0):.1f}s"
            )
        if done[0]:
            break

    print(f"\nFinal evaluation reward: {total_reward:.2f}")
    if info:
        print(f"Final RMS: {info.get('rms_error', 0):.4f} um")
        print(f"Final Ra:  {info.get('mean_roughness', 0):.1f} nm")

    train_env.close()
    eval_env.close()
    single_eval.close()


if __name__ == "__main__":
    main()
