"""Quick smoke test for the spiral polishing environment."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.polishing_env import LensPolishingEnv


def main():
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"
    env = LensPolishingEnv(grid_size=128, tif_checkpoint=str(ckpt))

    obs, info = env.reset(seed=42)
    print("Spiral polishing environment created!")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Spiral waypoints: {env.max_steps}")
    print(f"  Initial RMS: {env.initial_rms:.4f} um")
    print(f"  Passes: {env.num_passes}, Pressures: {env.pass_pressures}")
    print(f"  Fixed V={env.fixed_speed}rpm")

    print(f"\nRunning {min(20, env.max_steps)} steps along spiral...")
    total_reward = 0.0
    for i in range(min(20, env.max_steps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i + 1) % 5 == 0:
            print(
                f"  Step {i+1}: RMS={info['rms_error']:.4f}um, "
                f"dwell={info['dwell_time']:.2f}s, "
                f"tool=({info['tool_x']:.1f},{info['tool_y']:.1f}), "
                f"radial={info['radial_progress']:.1%}"
            )
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.4f}")
    print("Test passed!")


if __name__ == "__main__":
    main()
