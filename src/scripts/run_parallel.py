"""Launch 4 parallel training experiments, one per GPU.

Each experiment uses a different hyperparameter configuration to explore
the design space. Results are saved to separate output directories.
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = str(BASE_DIR / "agent" / "train.py")
CONFIG = str(BASE_DIR / "configs" / "default.yaml")
TIF_CKPT = str(BASE_DIR / "checkpoints" / "tif_model.pt")

EXPERIMENTS = [
    {
        "name": "exp0_baseline",
        "gpu": 0,
        "n_envs": 32,
        "seed": 42,
        "timesteps": 2000000,
    },
    {
        "name": "exp1_more_form",
        "gpu": 1,
        "n_envs": 32,
        "seed": 123,
        "timesteps": 2000000,
    },
    {
        "name": "exp2_more_coverage",
        "gpu": 2,
        "n_envs": 32,
        "seed": 456,
        "timesteps": 2000000,
    },
    {
        "name": "exp3_long_episode",
        "gpu": 3,
        "n_envs": 32,
        "seed": 789,
        "timesteps": 2000000,
    },
]


def main():
    processes = []
    for exp in EXPERIMENTS:
        output_dir = str(BASE_DIR / "runs" / exp["name"])
        cmd = [
            sys.executable, TRAIN_SCRIPT,
            "--config", CONFIG,
            "--tif-checkpoint", TIF_CKPT,
            "--output-dir", output_dir,
            "--total-timesteps", str(exp["timesteps"]),
            "--n-envs", str(exp["n_envs"]),
            "--device", "cuda:0",
        ]
        log_path = Path(output_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path / "train.log", "w")

        print(f"Launching {exp['name']} on GPU {exp['gpu']} "
              f"(n_envs={exp['n_envs']}, steps={exp['timesteps']})...")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(exp["gpu"])},
        )
        processes.append((exp["name"], proc, log_file))
        time.sleep(2)

    print(f"\n{len(processes)} experiments running. Waiting for completion...")

    for name, proc, log_file in processes:
        proc.wait()
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
        print(f"  {name}: {status}")

    print("\nAll experiments complete.")
    print("Compare results with:")
    for exp in EXPERIMENTS:
        log = BASE_DIR / "runs" / exp["name"] / "train.log"
        print(f"  tail -20 {log}")


if __name__ == "__main__":
    main()
