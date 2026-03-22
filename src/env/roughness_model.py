"""Empirical surface roughness model for polishing simulation.

Tracks per-pixel roughness (Ra in nm) and updates based on polishing
parameters. Moderate conditions improve roughness; extreme conditions
degrade it.
"""

import numpy as np


class RoughnessModel:
    """Manages a 2D roughness map and applies empirical updates."""

    OPTIMAL_PRESSURE = 25.0
    OPTIMAL_SPEED = 600.0
    OPTIMAL_CONCENTRATION = 5.0

    PRESSURE_SIGMA = 20.0
    SPEED_SIGMA = 400.0
    CONC_SIGMA = 4.0

    MAX_IMPROVEMENT_RATE = 0.05
    MIN_ROUGHNESS_NM = 5.0
    MAX_ROUGHNESS_NM = 500.0

    def __init__(self, grid_size: int = 128, initial_ra_nm: float = 200.0):
        self.grid_size = grid_size
        self.initial_ra_nm = initial_ra_nm
        self.roughness = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.RandomState(seed)
        variation = rng.uniform(0.9, 1.1, (self.grid_size, self.grid_size))
        self.roughness = np.full(
            (self.grid_size, self.grid_size), self.initial_ra_nm, dtype=np.float64
        ) * variation
        return self.roughness.copy()

    def quality_factor(self, pressure: float, speed: float, concentration: float) -> float:
        """Compute polishing quality factor in [0, 1].
        Peaks at optimal parameters, decays as Gaussian for deviations.
        """
        dp = ((pressure - self.OPTIMAL_PRESSURE) / self.PRESSURE_SIGMA) ** 2
        dv = ((speed - self.OPTIMAL_SPEED) / self.SPEED_SIGMA) ** 2
        dc = ((concentration - self.OPTIMAL_CONCENTRATION) / self.CONC_SIGMA) ** 2
        return float(np.exp(-0.5 * (dp + dv + dc)))

    def update(
        self,
        tif_footprint: np.ndarray,
        pressure: float,
        speed: float,
        concentration: float,
        dwell_time: float,
        mask: np.ndarray,
    ):
        """Update roughness map based on a polishing action.

        Uses multiplicative decay: Ra_new = Ra * (1 - rate * qf * footprint * dt)
        Clamped to [MIN, MAX] to prevent numerical issues.
        """
        qf = self.quality_factor(pressure, speed, concentration)
        decay = np.clip(self.MAX_IMPROVEMENT_RATE * qf * dwell_time * tif_footprint, 0.0, 0.9)
        self.roughness = np.where(
            mask,
            self.roughness * (1.0 - decay),
            self.roughness,
        )
        self.roughness = np.clip(self.roughness, self.MIN_ROUGHNESS_NM, self.MAX_ROUGHNESS_NM)

    def get_mean_roughness(self, mask: np.ndarray) -> float:
        """Mean Ra over the valid aperture (nm)."""
        return float(np.mean(self.roughness[mask]))

    def get_roughness_map(self) -> np.ndarray:
        return self.roughness.copy()
