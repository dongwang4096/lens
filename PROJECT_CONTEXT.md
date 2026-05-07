# Convex Lens Polishing - Traditional Optimization Context

## Goal

This project simulates convex lens polishing and focuses on deterministic optimization of the polishing trajectory. It supports fixed-path dwell optimization, local path perturbation, and free continuous trajectory optimization with smooth-turn constraints.

## Compute Environment

- Local editing and light CPU runs are supported with a Python virtual environment.
- Larger sweeps can run in the SLURM/enroot environment described by `1.salloc.sh` and `2.start_enroot.sh`.
- The required Python packages are listed in `src/requirements.txt`.

## Data And TIF Model

- Raw experiment data: `副本数据汇总正式版.xlsx`
- TIF fitting script: `src/scripts/fit_tif.py`
- Checkpoint output: `src/checkpoints/tif_model.pt`

The TIF model maps `(pressure, speed, concentration)` to `(removal_width, removal_depth)` and generates a 2D Gaussian removal function for each tool waypoint.

## Code Structure

```text
src/
├── models/
│   └── tif_model.py
├── env/
│   ├── lens_surface.py
│   ├── roughness_model.py
│   ├── spiral_path.py
│   └── polishing_env.py
├── utils/
│   └── data_loader.py
├── configs/
│   └── default.yaml
└── scripts/
    ├── fit_tif.py
    ├── test_env.py
    ├── optimize_dwell.py
    ├── optimize_joint.py
    ├── optimize_free_path.py
    ├── sweep_lsq_params.py
    ├── show_initial_surface.py
    └── animate_compare.py
```

## Optimization Routes

### Fixed Path, Dwell-Only

Entry point: `src/scripts/optimize_dwell.py`

For a fixed spiral and fixed process parameters per pass, final form error is linear in dwell time:

```text
error_final = error_0 - A @ dwells
```

The first stage solves bounded least squares:

```text
minimize ||b - A x||^2
subject to dwell_min <= x_i <= dwell_max
```

The optional second stage uses PyTorch autograd to refine dwell time against form error, roughness, and over-polishing terms.

### Joint Dwell And Path Perturbation

Entry point: `src/scripts/optimize_joint.py`

This route starts from the dwell-only solution and jointly optimizes:

- dwell time at each waypoint
- bounded `(dx, dy)` perturbations from the nominal spiral

It uses differentiable Gaussian TIF removal and regularizes smoothness, coverage, and aperture constraints.

### Free Continuous Path

Entry point: `src/scripts/optimize_free_path.py`

This route removes the fixed-spiral assumption at the optimization layer. A small set of control points defines a continuous Catmull-Rom path, which is sampled into polishing waypoints. The optimizer alternates between:

- rebuilding the removal matrix `A` for the current path and solving bounded least squares for dwell time
- updating control points with PyTorch while keeping dwell fixed

The hard-constraint route now starts from a smooth single-direction spiral initializer instead of the historical multi-pass spiral, because the pass reversals create infeasible sharp turns. By default the initializer has no artificial inner-radius floor (`min_radius_mm: 0.0`), so it can cover the lens center. The final accepted result is still required to satisfy the configured geometry and motion constraints.

The script computes local feed speed as `v_i = ds_i / dwell_i`, where `ds_i` is the local path distance around waypoint `i`. `speed_min_mm_s` and `speed_max_mm_s` are converted into per-waypoint dwell bounds during the least-squares solve, while `max_accel_mm_s2` is enforced by post-solve dwell projection. Gradient path updates are projected with backtracking against geometric hard bounds (aperture, turn angle, turn radius). Motion hard bounds are checked after each LS dwell refresh with the updated dwell schedule. RMS is only compared among fully feasible path/dwell candidates.

### Parameter Sweep

Entry point: `src/scripts/sweep_lsq_params.py`

This script explores practical limits of the least-squares formulation, including dwell bounds, pass count, pitch, and theoretical unconstrained solutions.

## Typical Commands

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r src/requirements.txt

.venv/bin/python src/scripts/fit_tif.py
.venv/bin/python src/scripts/test_env.py
.venv/bin/python src/scripts/optimize_dwell.py --skip-pytorch
.venv/bin/python src/scripts/optimize_dwell.py --pytorch-iters 3000
.venv/bin/python src/scripts/optimize_joint.py --skip-sweep
.venv/bin/python src/scripts/optimize_free_path.py --output src/results/free_path_optimized.npz
.venv/bin/python src/scripts/animate_compare.py --joint-results src/results/free_path_optimized.npz --output src/results/free_path.mp4
.venv/bin/python src/scripts/sweep_lsq_params.py
```

Generated checkpoints and optimization results are ignored by git.
