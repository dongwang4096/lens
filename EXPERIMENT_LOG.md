# Experiment Log - Traditional Optimization

## Local Smoke Run

- Environment: local Python virtual environment at `.venv`
- TIF fitting: `src/scripts/fit_tif.py`
- Samples: 218
- Width MAE: 0.5064 mm, relative error 4.9%
- Depth MAE: 0.0295 um/s, relative error 21.5%
- Checkpoint: `src/checkpoints/tif_model.pt`

## Dwell-Only Least Squares

Command:

```bash
.venv/bin/python src/scripts/optimize_dwell.py --skip-pytorch
```

Result:

- Initial RMS: 3.0368 um
- Least-squares RMS: 0.1543 um
- Reduction: 94.9%
- Mean Ra after replay: 130.2 nm
- Dwell range: 0.100 to 5.000 s
- Lower-bound waypoints: 918 / 1130
- Upper-bound waypoints: 52 / 1130
- Output: `src/results/optimal_dwells.npz`

## Short PyTorch Refinement Smoke Run

Command:

```bash
.venv/bin/python src/scripts/optimize_dwell.py --pytorch-iters 200 --pytorch-lr 0.01 --output src/results/optimal_dwells_pytorch200.npz
```

Result:

- Least-squares RMS: 0.1543 um, Ra 130.2 nm
- PyTorch refinement RMS: 0.1594 um, Ra 128.6 nm
- Output: `src/results/optimal_dwells_pytorch200.npz`

The short refinement confirms the code path runs on CPU, but it did not improve RMS in this small run.

## Free Continuous Path Optimization

Entry point:

```bash
.venv/bin/python src/scripts/optimize_free_path.py --output src/results/free_path_optimized.npz
```

This route represents the trajectory with Catmull-Rom control points and alternates between:

- bounded least-squares dwell solves for the current path
- PyTorch control-point updates with smoothness, turn-angle, turn-radius, spacing, and aperture penalties

Short smoke-test results should be appended here after each run.

### Smoke Test

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-waypoints 240 --n-control-points 24 --n-iters 20 \
  --lsq-interval 10 --report-interval 5 \
  --output src/results/free_path_optimized.npz
```

Result:

- Initial RMS: 3.0368 um
- Initial free-path LS RMS: 1.0005 um
- Free-path RMS after short run: 0.9942 um
- Improvement over initial free path: 0.6%
- Output: `src/results/free_path_optimized.npz`
- Preview animation: `src/results/free_path_preview.mp4`

This is only a short CPU smoke test. Use more waypoints, more control points, and more iterations for meaningful optimization.

### Long CPU Run - 96 Control Points

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-waypoints 1130 --n-control-points 96 --n-iters 600 \
  --lsq-interval 100 --report-interval 100 --save-mode final \
  --output src/results/free_path_long_96cp_600_final.npz
```

Result:

- Initial RMS: 3.0368 um
- Initial free-path LS RMS: 0.1577 um
- Best RMS during run: 0.1577 um
- Final smoothed RMS: 0.1728 um
- Max turn angle: 174.82 deg -> 83.28 deg
- Mean turn angle: about 9.07 deg -> 8.97 deg
- Optimized total dwell time in animation: 563.7 s
- Output: `src/results/free_path_long_96cp_600_final.npz`
- Preview animation: `src/results/free_path_long_96cp_600_final.mp4`

This run improves macroscopic smoothness but trades off form accuracy. The current objective still needs tuning if the target is to improve RMS and smoothness simultaneously.

### Balanced Rerun - 96 Control Points, 1000 Iterations

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-waypoints 1130 --n-control-points 96 --n-iters 1000 \
  --lsq-interval 100 --report-interval 100 --save-mode final \
  --w-form 20 --w-overpolish 12 \
  --w-smooth 0.25 --w-turn 0.5 --w-turn-radius 0.25 \
  --w-spacing 0.1 --w-radius 100 \
  --output src/results/free_path_rerun_balanced_96cp_1000_final.npz
```

Result:

- Initial RMS: 3.0368 um
- Initial free-path LS RMS: 0.1577 um
- Best RMS during run: 0.1045 um
- Final smoothed RMS: 0.1044 um
- Improvement over initial free path: 33.8%
- Overpolish RMS: 0.0995 um
- Max turn angle: 174.82 deg -> 77.35 deg
- Mean turn angle: about 9.07 deg -> 8.89 deg
- Output: `src/results/free_path_rerun_balanced_96cp_1000_final.npz`
- GIF: `src/results/free_path_rerun_balanced_96cp_1000_trace.gif`
- Storyboard: `src/results/free_path_rerun_balanced_96cp_1000_storyboard.png`

## Motion-Constrained Smoke Test

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-waypoints 240 --n-control-points 24 --n-iters 20 \
  --lsq-interval 10 --report-interval 5 --save-mode final \
  --speed-min-mm-s 0.05 --speed-max-mm-s 15 \
  --max-accel-mm-s2 8 --w-speed 2 --w-accel 0.5 \
  --output src/results/free_path_speed_constraints_smoke.npz
```

Result:

- Final RMS: 1.3459 um
- Effective feed speed: 0.054 to 15.000 mm/s
- Max absolute acceleration: 8.000 mm/s^2
- Output: `src/results/free_path_speed_constraints_smoke.npz`

The speed bounds are enforced through per-waypoint dwell bounds in the LS solve. The acceleration cap is enforced by a post-LS dwell projection that increases dwell on faster adjacent segments when needed, then reported in the saved result arrays.

### Motion-Constrained Long Run - 96 Control Points

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-waypoints 1130 --n-control-points 96 --n-iters 1000 \
  --lsq-interval 100 --report-interval 100 --save-mode final \
  --w-form 20 --w-overpolish 12 \
  --w-smooth 0.25 --w-turn 0.5 --w-turn-radius 0.25 \
  --w-spacing 0.1 --w-radius 100 \
  --speed-min-mm-s 0.05 --speed-max-mm-s 15 \
  --max-accel-mm-s2 8 --w-speed 2 --w-accel 0.5 \
  --output src/results/free_path_motion_constrained_96cp_1000_final.npz
```

Result:

- Initial RMS: 3.0368 um
- Saved RMS: 2.4151 um
- Effective feed speed: 0.120 to 15.000 mm/s
- Max absolute acceleration: 22.055 mm/s^2
- Acceleration violations over 8 mm/s^2: 10 / 1129
- Dwell range: 0.1 to 5.0 s
- Dwell points at 5.0 s upper bound: 39 / 1130
- Output: `src/results/free_path_motion_constrained_96cp_1000_final.npz`
- GIF: `src/results/free_path_motion_constrained_96cp_1000_trace.gif`
- Storyboard: `src/results/free_path_motion_constrained_96cp_1000_storyboard.png`

The speed upper bound was met, but the 8 mm/s^2 acceleration cap was not fully feasible with the current 5 s dwell upper bound. Several segments need longer dwell time or a smoother path parameterization to satisfy the acceleration limit.

### Hard-Constraint Smoke Test

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-waypoints 240 --n-control-points 24 --n-iters 20 \
  --lsq-interval 10 --report-interval 5 --save-mode final \
  --speed-min-mm-s 0.05 --speed-max-mm-s 15 \
  --max-accel-mm-s2 8 --w-speed 2 --w-accel 0.5 \
  --output src/results/free_path_hard_constraints_smoke.npz
```

Result:

- Saved mode: `infeasible-final`
- Speed: 0.054 to 15.000 mm/s
- Max absolute acceleration: 8.000 mm/s^2
- Max turn violation: 79.9 deg over the configured 10 deg limit
- Feasible: false

This verifies the hard-constraint acceptance behavior: a candidate is not accepted as the best solution unless all configured motion/geometry constraints are satisfied.

### Hard-Constraint Feasible Initializer

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-iters 5 --lsq-interval 2 --report-interval 1 \
  --speed-min-mm-s 0.05 --speed-max-mm-s 15 \
  --max-accel-mm-s2 8 --max-turn-angle-deg 10 \
  --min-turn-radius-mm 8 \
  --output src/results/free_path_smoke_feasible_default.npz
```

Result:

- Init path: `smooth_spiral`
- Baseline RMS: 1.1378 um
- Saved mode: `best-rms`
- Saved RMS: 1.1378 um
- Speed: 0.089 to 5.389 mm/s
- Max absolute acceleration: 8.000 mm/s^2
- Max turn angle: 3.25 deg
- Minimum turn radius: 8.04 mm
- Feasible: true

This confirms the corrected architecture: the starting path is feasible before RMS optimization begins.

### Hard-Constraint 80-Iteration Run

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-iters 80 --lsq-interval 20 --report-interval 20 --lr 0.001 \
  --speed-min-mm-s 0.05 --speed-max-mm-s 15 \
  --max-accel-mm-s2 8 --max-turn-angle-deg 10 \
  --min-turn-radius-mm 8 \
  --output src/results/free_path_hard_constraints_80.npz
```

Result:

- Baseline RMS: 1.1378 um
- Best feasible RMS: 1.1373 um
- Saved mode: `best-rms`
- Speed: 0.089 to 5.389 mm/s
- Max absolute acceleration: 8.000 mm/s^2
- Feasible: true

The run stays inside the hard-feasible region and only records feasible RMS improvements.

### Corrected Path-Movement Run

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-iters 300 --lsq-interval 50 --report-interval 50 --lr 0.01 \
  --speed-min-mm-s 0.05 --speed-max-mm-s 15 \
  --max-accel-mm-s2 8 --max-turn-angle-deg 10 \
  --min-turn-radius-mm 8 \
  --output src/results/free_path_hard_constraints_pathmove_300.npz
```

Result:

- Baseline RMS: 1.1378 um
- Best feasible RMS: 0.9290 um
- Improvement over initial feasible path: 18.3%
- Path displacement: mean 1.275 mm, max 1.788 mm
- Speed: 0.050 to 6.055 mm/s
- Max absolute acceleration: 8.000 mm/s^2
- Feasible: true
- MP4: `src/results/free_path_hard_constraints_pathmove_300_slow.mp4`

This fixes the earlier issue where per-step motion checking used stale dwell times and effectively rolled back almost every path update. The optimizer now projects gradient steps against geometry only, then enforces motion constraints after the LS dwell refresh.

### No Inner-Radius Initializer Test

Command:

```bash
.venv/bin/python src/scripts/optimize_free_path.py \
  --n-iters 300 --lsq-interval 50 --report-interval 50 --lr 0.01 \
  --n-control-points 128 \
  --speed-min-mm-s 0.05 --speed-max-mm-s 15 \
  --max-accel-mm-s2 8 --max-turn-angle-deg 10 \
  --min-turn-radius-mm 8 --w-turn 20 --w-turn-radius 20 \
  --output src/results/free_path_no_inner_radius_128cp_300.npz
```

Result:

- Initializer inner radius: 0.0 mm
- Saved RMS: 0.2224 um
- Center coverage: 160 optimized waypoints with `r < 10mm`
- Feasible: false
- Remaining violations: turn-radius violation 6.49 mm, aperture violation 0.088 mm

Removing the artificial inner-radius floor lets the path polish the center and improves RMS substantially, but the resulting center spiral is not yet compatible with the 8 mm minimum-turn-radius hard constraint.
