"""Quick smoke test for the polishing environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.polishing_env import LensPolishingEnv


def main():
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "tif_model.pt"

    env = LensPolishingEnv(
        grid_size=64,
        tif_checkpoint=str(ckpt),
        max_steps=20,
    )

    obs, info = env.reset(seed=42)
    print("Environment created successfully!")
    print(f"  Obs maps shape: {obs['maps'].shape}")
    print(f"  Obs scalar shape: {obs['scalar'].shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Initial RMS error: {env.initial_rms:.4f} um")
    print(f"  Initial roughness: {env.prev_roughness:.1f} nm")

    print("\nRunning 10 random steps...")
    total_reward = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  Step {i+1}: RMS={info['rms_error']:.4f}um, "
            f"Ra={info['mean_roughness']:.1f}nm, "
            f"reward={reward:.4f}, "
            f"tool=({info['tool_x']:.1f},{info['tool_y']:.1f}), "
            f"P={info['pressure']:.0f}N, V={info['speed']:.0f}rpm"
        )
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.4f}")
    print("Environment test passed!")


if __name__ == "__main__":
    main()
