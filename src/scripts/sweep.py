"""Overnight hyperparameter sweep: runs multiple experiments sequentially
across 4 GPUs, evaluates each, and produces a summary comparison.

Usage (on compute node):
    python3 src/scripts/sweep.py

Results saved to src/runs/sweep_results.md
"""

import subprocess
import sys
import os
import time
import yaml
import json
from pathlib import Path
from copy import deepcopy

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = str(BASE_DIR / "agent" / "train.py")
EVALUATE_SCRIPT = str(BASE_DIR / "scripts" / "evaluate.py")
ANIMATE_SCRIPT = str(BASE_DIR / "scripts" / "animate.py")
BASE_CONFIG = str(BASE_DIR / "configs" / "default.yaml")
TIF_CKPT = str(BASE_DIR / "checkpoints" / "tif_model.pt")
SWEEP_DIR = BASE_DIR / "runs" / "sweep"

def _exp(name, gpu, **overrides):
    """Helper to define an experiment."""
    o = {}
    for k, v in overrides.items():
        section, key = k.split(".")
        o.setdefault(section, {})[key] = v
    return {"name": name, "overrides": o, "gpu": gpu, "n_envs": 32}

EXPERIMENTS = [
    # === Batch 1: learning rate sweep ===
    _exp("s01_lr1e5",   0, **{"training.learning_rate": 1e-5, "training.buffer_size": 500000, "training.patience": 30}),
    _exp("s02_lr3e5",   1, **{"training.learning_rate": 3e-5, "training.buffer_size": 500000, "training.patience": 30}),
    _exp("s03_lr5e5",   2, **{"training.learning_rate": 5e-5, "training.buffer_size": 500000, "training.patience": 30}),
    _exp("s04_lr1e4",   3, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30}),
    # === Batch 2: learning rate sweep high ===
    _exp("s05_lr2e4",   0, **{"training.learning_rate": 2e-4, "training.buffer_size": 500000, "training.patience": 30}),
    _exp("s06_lr3e4",   1, **{"training.learning_rate": 3e-4, "training.buffer_size": 300000, "training.patience": 30}),
    _exp("s07_lr5e4",   2, **{"training.learning_rate": 5e-4, "training.buffer_size": 300000, "training.patience": 30}),
    _exp("s08_lr1e3",   3, **{"training.learning_rate": 1e-3, "training.buffer_size": 300000, "training.patience": 30}),
    # === Batch 3: buffer size sweep (best lr from intuition: 1e-4) ===
    _exp("s09_buf100k",  0, **{"training.learning_rate": 1e-4, "training.buffer_size": 100000, "training.patience": 30}),
    _exp("s10_buf200k",  1, **{"training.learning_rate": 1e-4, "training.buffer_size": 200000, "training.patience": 30}),
    _exp("s11_buf500k",  2, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30}),
    _exp("s12_buf1m",    3, **{"training.learning_rate": 1e-4, "training.buffer_size": 1000000, "training.patience": 30}),
    # === Batch 4: network architecture ===
    _exp("s13_arch128",  0, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "policy.net_arch": [128, 128]}),
    _exp("s14_arch256",  1, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "policy.net_arch": [256, 256]}),
    _exp("s15_arch512",  2, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "policy.net_arch": [512, 512]}),
    _exp("s16_arch3x256",3, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "policy.net_arch": [256, 256, 256]}),
    # === Batch 5: gamma (discount factor) ===
    _exp("s17_g990",  0, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "training.gamma": 0.990}),
    _exp("s18_g995",  1, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "training.gamma": 0.995}),
    _exp("s19_g998",  2, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "training.gamma": 0.998}),
    _exp("s20_g999",  3, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "training.gamma": 0.999}),
    # === Batch 6: gradient steps & train freq ===
    _exp("s21_gstep1",  0, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "training.gradient_steps": 1, "training.train_freq": 1}),
    _exp("s22_gstep4",  1, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30, "training.gradient_steps": 4, "training.train_freq": 4}),
    _exp("s23_gstep8",  2, **{"training.learning_rate": 5e-5, "training.buffer_size": 500000, "training.patience": 30, "training.gradient_steps": 8, "training.train_freq": 8}),
    _exp("s24_gstep16", 3, **{"training.learning_rate": 3e-5, "training.buffer_size": 500000, "training.patience": 30, "training.gradient_steps": 16, "training.train_freq": 16}),
    # === Batch 7: pressure schedule variants ===
    _exp("s25_pres_gentle",  0, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30,
                                    "spiral.pass_pressures": [30, 20, 14, 10, 7, 5, 4, 3, 2, 1.5]}),
    _exp("s26_pres_steep",   1, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30,
                                    "spiral.pass_pressures": [50, 30, 15, 8, 5, 3, 2, 1.5, 1, 0.5]}),
    _exp("s27_pres_5pass",   2, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30,
                                    "spiral.num_passes": 5, "spiral.pass_pressures": [40, 20, 10, 5, 2]}),
    _exp("s28_pres_15pass",  3, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 30,
                                    "spiral.num_passes": 15,
                                    "spiral.pass_pressures": [40, 30, 22, 16, 12, 9, 7, 5, 4, 3, 2.5, 2, 1.5, 1, 0.5]}),
    # === Batch 8: best combos (educated guesses) ===
    _exp("s29_combo_a",  0, **{"training.learning_rate": 5e-5, "training.buffer_size": 500000, "training.patience": 40,
                                "training.gamma": 0.995, "policy.net_arch": [512, 512]}),
    _exp("s30_combo_b",  1, **{"training.learning_rate": 1e-4, "training.buffer_size": 500000, "training.patience": 40,
                                "training.gradient_steps": 4, "training.train_freq": 4, "policy.net_arch": [256, 256, 256]}),
    _exp("s31_combo_c",  2, **{"training.learning_rate": 3e-5, "training.buffer_size": 1000000, "training.patience": 50,
                                "training.gamma": 0.998, "training.total_timesteps": 5000000}),
    _exp("s32_combo_d",  3, **{"training.learning_rate": 5e-5, "training.buffer_size": 500000, "training.patience": 40,
                                "training.gradient_steps": 8, "training.train_freq": 8,
                                "spiral.pass_pressures": [30, 20, 14, 10, 7, 5, 4, 3, 2, 1.5]}),
]

BATCH_SIZE = 4


def make_config(base_config: dict, overrides: dict) -> dict:
    config = deepcopy(base_config)
    for section, vals in overrides.items():
        if section not in config:
            config[section] = {}
        config[section].update(vals)
    return config


def run_batch(batch, base_config):
    """Run a batch of up to 4 experiments in parallel (one per GPU)."""
    processes = []
    for exp in batch:
        exp_dir = SWEEP_DIR / exp["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        config = make_config(base_config, exp["overrides"])
        config_path = exp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        cmd = [
            sys.executable, TRAIN_SCRIPT,
            "--config", str(config_path),
            "--tif-checkpoint", TIF_CKPT,
            "--output-dir", str(exp_dir),
            "--total-timesteps", str(config["training"]["total_timesteps"]),
            "--n-envs", str(exp["n_envs"]),
            "--device", "cuda:0",
        ]

        log_file = open(exp_dir / "train.log", "w")
        env_vars = {**os.environ, "CUDA_VISIBLE_DEVICES": str(exp["gpu"])}

        print(f"  Starting {exp['name']} on GPU {exp['gpu']}...")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env_vars)
        processes.append((exp, proc, log_file))
        time.sleep(3)

    timeout_s = 1200
    start = time.time()
    for exp, proc, log_file in processes:
        remaining = max(10, timeout_s - (time.time() - start))
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"  {exp['name']}: TIMEOUT ({timeout_s}s)")
            log_file.close()
            continue
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        print(f"  {exp['name']}: {status}")


def evaluate_experiment(exp_name):
    """Evaluate best model and return metrics."""
    exp_dir = SWEEP_DIR / exp_name
    model_path = exp_dir / "checkpoints" / "best_model.zip"
    eval_dir = exp_dir / "eval"
    config_path = exp_dir / "config.yaml"

    if not model_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cmd = [
        sys.executable, EVALUATE_SCRIPT,
        "--model", str(model_path),
        "--config", str(config_path),
        "--tif-checkpoint", TIF_CKPT,
        "--output-dir", str(eval_dir),
        "--seed", "42",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    metrics = {}
    for line in result.stdout.split("\n"):
        if "Initial RMS" in line:
            metrics["initial_rms"] = float(line.split(":")[-1].strip().split()[0])
        elif "Final RMS" in line:
            metrics["final_rms"] = float(line.split(":")[-1].strip().split()[0])
        elif "Final Ra" in line:
            metrics["final_ra"] = float(line.split(":")[-1].strip().split()[0])
        elif "Mean dwell" in line:
            metrics["mean_dwell"] = float(line.split(":")[-1].strip().split()[0])
        elif "Total reward" in line:
            metrics["total_reward"] = float(line.split(":")[-1].strip())

    return metrics


def generate_animation(exp_name):
    """Generate animation for best model."""
    exp_dir = SWEEP_DIR / exp_name
    model_path = exp_dir / "checkpoints" / "best_model.zip"
    config_path = exp_dir / "config.yaml"
    gif_path = exp_dir / "animation.gif"

    if not model_path.exists():
        return

    cmd = [
        sys.executable, ANIMATE_SCRIPT,
        "--model", str(model_path),
        "--config", str(config_path),
        "--tif-checkpoint", TIF_CKPT,
        "--output", str(gif_path),
        "--seed", "42",
    ]
    subprocess.run(cmd, capture_output=True)


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    print(f"=== Overnight Sweep: {len(EXPERIMENTS)} experiments ===\n")

    for batch_idx in range(0, len(EXPERIMENTS), BATCH_SIZE):
        batch = EXPERIMENTS[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(EXPERIMENTS) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"--- Batch {batch_num}/{total_batches} ({len(batch)} experiments) ---")
        run_batch(batch, base_config)
        print()

    print("=== All training complete. Evaluating... ===\n")

    results = []
    for exp in EXPERIMENTS:
        print(f"Evaluating {exp['name']}...")
        metrics = evaluate_experiment(exp["name"])
        if metrics:
            results.append({"name": exp["name"], "overrides": str(exp["overrides"]), **metrics})
            print(f"  RMS: {metrics.get('initial_rms', '?'):.4f} -> {metrics.get('final_rms', '?'):.4f} um, "
                  f"Ra: {metrics.get('final_ra', '?'):.1f} nm, "
                  f"reward: {metrics.get('total_reward', '?'):.2f}")
        else:
            print(f"  SKIPPED (no model)")
        print()

    print("Generating animations for top 3...")
    sorted_results = sorted(results, key=lambda x: x.get("final_rms", 999))
    for r in sorted_results[:3]:
        print(f"  Animating {r['name']} (RMS={r['final_rms']:.4f})...")
        generate_animation(r["name"])

    summary_path = SWEEP_DIR / "sweep_results.md"
    with open(summary_path, "w") as f:
        f.write("# Sweep Results\n\n")
        f.write("| Rank | Experiment | Final RMS (um) | RMS Reduction | Final Ra (nm) | Mean Dwell (s) | Reward | Config |\n")
        f.write("|------|-----------|---------------|---------------|--------------|---------------|--------|--------|\n")
        for rank, r in enumerate(sorted_results, 1):
            init = r.get("initial_rms", 2.64)
            final = r.get("final_rms", 999)
            reduction = (1 - final / init) * 100 if init > 0 else 0
            f.write(f"| {rank} | {r['name']} | {final:.4f} | {reduction:.1f}% | "
                    f"{r.get('final_ra', 0):.1f} | {r.get('mean_dwell', 0):.2f} | "
                    f"{r.get('total_reward', 0):.2f} | {r['overrides']} |\n")
        f.write(f"\nBest: **{sorted_results[0]['name']}** with RMS={sorted_results[0]['final_rms']:.4f} um\n")

    print(f"\n=== SWEEP COMPLETE ===")
    print(f"Results: {summary_path}")
    print(f"Best: {sorted_results[0]['name']} (RMS={sorted_results[0]['final_rms']:.4f} um)")
    print(f"Animations generated for top 3 experiments.")

    with open(SWEEP_DIR / "results.json", "w") as f:
        json.dump(sorted_results, f, indent=2)


if __name__ == "__main__":
    main()
