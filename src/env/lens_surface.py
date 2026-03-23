"""2D lens surface model for polishing simulation.

Represents a convex lens as a 2D height map on a square grid.
Provides target surface generation, initial error generation,
and material removal operations.
"""

import numpy as np


class LensSurface:
    """Manages the 2D surface height map of a convex lens."""

    def __init__(
        self,
        grid_size: int = 128,
        lens_diameter_mm: float = 50.0,
        curvature_radius_mm: float = 200.0,
    ):
        self.grid_size = grid_size
        self.lens_diameter_mm = lens_diameter_mm
        self.curvature_radius_mm = curvature_radius_mm

        self.pixel_size_mm = lens_diameter_mm / grid_size
        half = lens_diameter_mm / 2.0

        lin = np.linspace(-half, half, grid_size)
        self.xx, self.yy = np.meshgrid(lin, lin)

        self.r2 = self.xx**2 + self.yy**2
        self.mask = self.r2 <= (half**2)

        self.target = self._make_target_surface()
        self.current = None
        self.roughness = None

    def _make_target_surface(self) -> np.ndarray:
        """Ideal convex spherical surface: h = R - sqrt(R^2 - r^2).
        Values in microns.
        """
        R = self.curvature_radius_mm
        r2_clipped = np.clip(self.r2, 0, R**2 - 1e-6)
        h = R - np.sqrt(R**2 - r2_clipped)
        h *= 1000.0  # mm -> micron
        h[~self.mask] = 0.0
        return h

    def reset(self, error_scale: float = 1.0, seed: int | None = None) -> np.ndarray:
        """Reset surface with a new random initial error.

        Returns the initial error map (microns).
        """
        rng = np.random.RandomState(seed)
        error = self._generate_initial_error(rng, error_scale)
        self.current = self.target + error
        return self.get_error_map()

    def _generate_initial_error(self, rng: np.random.RandomState, scale: float) -> np.ndarray:
        """Generate a non-uniform initial form error (in microns).

        Simulates realistic grinding residuals: a thin positive baseline with
        strong localized bumps, asymmetric lobes, and mid-frequency ripple.
        Some regions are near-zero while others have 3-5x the average error.
        This forces RL to learn spatially-adaptive dwell times.
        """
        from scipy.ndimage import gaussian_filter

        D = self.lens_diameter_mm
        half = D / 2
        x_norm = self.xx / half
        y_norm = self.yy / half
        r_norm = np.sqrt(x_norm**2 + y_norm**2)

        baseline = 0.5 * scale

        n_bumps = rng.randint(3, 7)
        bumps = np.zeros_like(self.xx)
        for _ in range(n_bumps):
            cx = rng.uniform(-0.7, 0.7)
            cy = rng.uniform(-0.7, 0.7)
            amp = rng.uniform(1.5, 4.0) * scale
            sigma = rng.uniform(0.15, 0.4)
            bumps += amp * np.exp(-((x_norm - cx)**2 + (y_norm - cy)**2) / (2 * sigma**2))

        k1 = rng.uniform(1.0, 3.0)
        k2 = rng.uniform(1.5, 4.0)
        phi1 = rng.uniform(0, 2 * np.pi)
        phi2 = rng.uniform(0, 2 * np.pi)
        A_asym = rng.uniform(0.5, 1.5) * scale
        asymmetry = A_asym * np.cos(k1 * np.pi * x_norm + phi1) * np.sin(k2 * np.pi * y_norm + phi2)

        noise_raw = rng.randn(self.grid_size, self.grid_size)
        noise_smooth = gaussian_filter(noise_raw, sigma=max(3, self.grid_size // 15))
        noise_smooth *= 0.3 * scale / (noise_smooth.std() + 1e-8)

        error = baseline + bumps + asymmetry + noise_smooth
        error = np.maximum(error, 0.05 * scale)
        error[~self.mask] = 0.0
        return error

    def get_error_map(self) -> np.ndarray:
        """Current surface error = current - target (microns)."""
        err = self.current - self.target
        err[~self.mask] = 0.0
        return err

    def get_rms_error(self) -> float:
        """RMS of the error map over the valid aperture (microns)."""
        err = self.get_error_map()
        valid = err[self.mask]
        return float(np.sqrt(np.mean(valid**2)))

    def get_pv_error(self) -> float:
        """Peak-to-valley of the error map (microns)."""
        err = self.get_error_map()
        valid = err[self.mask]
        return float(valid.max() - valid.min())

    def apply_removal(self, removal_map: np.ndarray):
        """Subtract the removal map (microns) from the current surface."""
        self.current -= removal_map * self.mask

    def pos_mm_to_pixel(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        """Convert mm position to grid pixel indices."""
        half = self.lens_diameter_mm / 2.0
        col = int((x_mm + half) / self.pixel_size_mm)
        row = int((y_mm + half) / self.pixel_size_mm)
        col = np.clip(col, 0, self.grid_size - 1)
        row = np.clip(row, 0, self.grid_size - 1)
        return int(row), int(col)

    def pixel_to_pos_mm(self, row: int, col: int) -> tuple[float, float]:
        """Convert pixel indices to mm position."""
        half = self.lens_diameter_mm / 2.0
        x_mm = col * self.pixel_size_mm - half
        y_mm = row * self.pixel_size_mm - half
        return x_mm, y_mm
