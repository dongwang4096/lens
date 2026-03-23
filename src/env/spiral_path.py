"""Variable-pitch Archimedean spiral for multi-pass lens polishing.

Center-sparse: pitch is larger near center, smaller near edge, so that
the tool's coverage per unit area is approximately uniform across the lens.
Always clockwise rotation. Alternates radial direction between passes.
"""

import numpy as np


class SpiralPath:
    """Variable-pitch clockwise spiral with center-sparse geometry.

    Uses r(t) = r_min + (r_max - r_min) * t^alpha where alpha < 1
    makes center sparser (larger radial steps) and edge denser.
    alpha=0.5 gives roughly uniform area coverage for a fixed-width tool.
    """

    def __init__(
        self,
        lens_radius_mm: float = 25.0,
        pitch_mm: float = 7.0,
        points_per_revolution: int = 36,
        min_radius_mm: float = 5.0,
        center_sparse_power: float = 0.5,
    ):
        self.lens_radius_mm = lens_radius_mm
        self.pitch_mm = pitch_mm
        self.points_per_revolution = points_per_revolution
        self.min_radius_mm = min_radius_mm
        self.center_sparse_power = center_sparse_power

    def _generate_one_pass(self, outward: bool, theta_start: float = 0.0):
        """Generate one spiral pass, always clockwise.

        Radius varies as r = r_min + (r_max - r_min) * t^alpha
        where t is uniformly spaced in [0, 1] and theta advances steadily.
        """
        r_min = self.min_radius_mm
        r_max = self.lens_radius_mm
        alpha = self.center_sparse_power

        n_revolutions = (r_max - r_min) / self.pitch_mm
        d_theta = 2.0 * np.pi / self.points_per_revolution
        n_points = max(int(n_revolutions * self.points_per_revolution), 10)

        thetas = theta_start + np.arange(n_points) * d_theta

        t = np.linspace(0, 1, n_points)
        if outward:
            radii = r_min + (r_max - r_min) * (t ** alpha)
        else:
            radii = r_min + (r_max - r_min) * ((1.0 - t) ** alpha)

        x = radii * np.cos(thetas)
        y = radii * np.sin(thetas)

        theta_end = float(thetas[-1] + d_theta) if n_points > 0 else theta_start
        return np.stack([x, y], axis=1).astype(np.float32), theta_end

    def get_single_pass_length(self) -> int:
        wp, _ = self._generate_one_pass(outward=True)
        return len(wp)

    def build_multi_pass(self, num_passes: int) -> np.ndarray:
        """Build multi-pass path: alternating outward/inward, always clockwise.

        Pass 0: center -> edge (outward, center-sparse)
        Pass 1: edge -> center (inward, still clockwise, center-sparse)
        Pass 2: center -> edge ...
        """
        segments = []
        theta = 0.0
        for i in range(num_passes):
            outward = (i % 2 == 0)
            wp, theta = self._generate_one_pass(outward, theta_start=theta)
            segments.append(wp)
        return np.concatenate(segments, axis=0)

    def get_radial_position(self, waypoint: np.ndarray) -> float:
        r = float(np.sqrt(waypoint[0]**2 + waypoint[1]**2))
        return min(r / self.lens_radius_mm, 1.0)
