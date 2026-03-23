# Experiment Log — Lens Polishing RL

## Run 1: lr=1e-4, buffer=300K, patience=30
- **Config**: lr=1e-4, buffer=300K, pressures=[40,28,20,14,10,7,5,4,3,2], patience=30, fixed surface seed=42
- **Steps**: 479K (early stopped at 24%)
- **Best eval reward**: +3.34
- **Final RMS**: 0.80 um (from 2.64, -70%)
- **Final Ra**: 126 nm (from 200, -37%)
- **Mean dwell**: 0.41 s
- **Issue**: Training unstable after plateau, reward degraded to -632

## Run 2: lr=3e-5, buffer=500K, patience=30  *(current)*
- **Config**: lr=3e-5, buffer=500K, pressures=[40,28,20,14,10,7,5,4,3,2], patience=30, fixed surface seed=42
- **Goal**: More stable convergence, potentially better final RMS
- **Steps**: pending
- **Best eval reward**: pending
- **Final RMS**: pending
- **Final Ra**: pending
