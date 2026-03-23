"""Gymnasium environment for multi-pass spiral lens polishing.

The tool follows alternating outward/inward Archimedean spirals across
multiple passes. Each pass uses a preset pressure (decreasing from coarse
to fine). The RL agent only controls dwell time at each waypoint.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .lens_surface import LensSurface
from .roughness_model import RoughnessModel
from .spiral_path import SpiralPath

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tif_model import TIFModel


class LensPolishingEnv(gym.Env):
    """Multi-pass spiral polishing environment.

    One episode = num_passes spiral passes (alternating in/out).
    Each pass uses a different preset pressure (coarse -> fine).
    The agent decides dwell_time at every waypoint.

    Observation (flat vector):
        - local_patch (patch_size^2): error around tool
        - radial_position (1)
        - pass_ratio (1): current_pass / num_passes
        - step_in_pass_ratio (1): progress within current pass
        - current_pressure_norm (1): current pass pressure normalized
        - local_error_stats (3): mean, max, std
        Total: patch_size^2 + 7

    Action: (1,) continuous [-1, 1] -> dwell_time
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 128,
        lens_diameter_mm: float = 50.0,
        curvature_radius_mm: float = 200.0,
        initial_roughness_nm: float = 200.0,
        target_rms_um: float = 0.1,
        reward_form: float = 5.0,
        reward_roughness: float = 1.0,
        reward_bonus: float = 10.0,
        num_passes: int = 10,
        pass_pressures: list = None,
        fixed_speed: float = 800.0,
        concentration: float = 5.0,
        dwell_range: tuple = (0.1, 5.0),
        pitch_mm: float = 7.0,
        patch_size: int = 16,
        tif_checkpoint: str | None = None,
        error_scale: float = 1.0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.lens_diameter_mm = lens_diameter_mm
        self.target_rms_um = target_rms_um
        self.concentration = concentration
        self.error_scale = error_scale
        self.patch_size = patch_size
        self.num_passes = num_passes

        self.w_form = reward_form
        self.w_rough = reward_roughness
        self.w_bonus = reward_bonus

        if pass_pressures is None:
            pass_pressures = np.linspace(40, 8, num_passes).tolist()
        self.pass_pressures = pass_pressures

        self.fixed_speed = fixed_speed
        self.dt_range = dwell_range

        self.surface = LensSurface(grid_size, lens_diameter_mm, curvature_radius_mm)
        self.roughness_model = RoughnessModel(grid_size, initial_roughness_nm)
        self.tif_model = TIFModel(tif_checkpoint)

        self.spiral = SpiralPath(
            lens_radius_mm=lens_diameter_mm / 2.0,
            pitch_mm=pitch_mm,
            min_radius_mm=3.0,
            center_sparse_power=0.7,
        )
        self.single_pass_len = self.spiral.get_single_pass_length()
        self.all_waypoints = self.spiral.build_multi_pass(num_passes)
        self.max_steps = len(self.all_waypoints)

        self.lookahead = 50
        obs_dim = (self.lookahead + 1) + 7
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.current_step = 0
        self.initial_rms = 0.0
        self.prev_rms = 0.0
        self.prev_roughness = 0.0
        self.dwell_history = []

    def _current_pass(self) -> int:
        return min(self.current_step // self.single_pass_len, self.num_passes - 1)

    def _current_pressure(self) -> float:
        return self.pass_pressures[self._current_pass()]

    def _step_in_pass(self) -> int:
        return self.current_step % self.single_pass_len

    def _sample_error_at(self, x_mm: float, y_mm: float) -> float:
        """Sample the surface error at a single (x, y) position."""
        row, col = self.surface.pos_mm_to_pixel(x_mm, y_mm)
        err = self.surface.get_error_map()
        return float(err[row, col]) / (self.initial_rms + 1e-8)

    def _get_lookahead_profile(self) -> np.ndarray:
        """Sample error at current + next N waypoints along the spiral.

        Returns (lookahead+1,) array: [error_here, error_wp+1, ..., error_wp+N]
        This lets the agent see what's coming and plan dwell time accordingly.
        """
        profile = np.zeros(self.lookahead + 1, dtype=np.float32)
        for i in range(self.lookahead + 1):
            idx = min(self.current_step + i, self.max_steps - 1)
            wp = self.all_waypoints[idx]
            profile[i] = self._sample_error_at(wp[0], wp[1])
        return np.clip(profile, -5.0, 5.0)

    def _get_obs(self) -> np.ndarray:
        profile = self._get_lookahead_profile()

        idx = min(self.current_step, self.max_steps - 1)
        wp = self.all_waypoints[idx]

        radial = self.spiral.get_radial_position(wp)
        pass_ratio = self._current_pass() / max(self.num_passes - 1, 1)
        step_in_pass_ratio = self._step_in_pass() / max(self.single_pass_len - 1, 1)
        pressure_norm = self._current_pressure() / 50.0

        ahead = profile[1:] if len(profile) > 1 else profile
        lm = float(np.mean(ahead))
        lx = float(np.max(ahead))
        ls = float(np.std(ahead))

        scalars = np.array(
            [radial, pass_ratio, step_in_pass_ratio, pressure_norm, lm, lx, ls],
            dtype=np.float32,
        )
        return np.concatenate([profile, scalars])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.surface.reset(error_scale=self.error_scale, seed=42)
        self.roughness_model.reset(seed=43)

        self.current_step = 0
        self.dwell_history = []

        self.initial_rms = self.surface.get_rms_error()
        self.prev_rms = self.initial_rms
        self.prev_roughness = self.roughness_model.get_mean_roughness(self.surface.mask)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        a = float(np.clip(action[0], -1.0, 1.0))
        dwell = self.dt_range[0] + (a + 1.0) / 2.0 * (self.dt_range[1] - self.dt_range[0])
        pressure = self._current_pressure()

        wp = self.all_waypoints[self.current_step]
        tool_x, tool_y = float(wp[0]), float(wp[1])
        cur_pass = self._current_pass()

        tif_2d = self.tif_model.generate_tif_2d(
            pressure, self.fixed_speed, self.concentration,
            (self.surface.xx, self.surface.yy),
            (tool_x, tool_y),
        )

        removal = tif_2d * dwell
        self.surface.apply_removal(removal)

        tif_max = tif_2d.max()
        footprint = tif_2d / (tif_max + 1e-12)
        self.roughness_model.update(
            footprint, pressure, self.fixed_speed,
            self.concentration, dwell, self.surface.mask,
        )

        self.dwell_history.append(dwell)
        self.current_step += 1

        cur_rms = self.surface.get_rms_error()
        cur_rough = self.roughness_model.get_mean_roughness(self.surface.mask)

        rms_improve = (self.prev_rms - cur_rms) / (self.initial_rms + 1e-8)
        rough_improve = (self.prev_roughness - cur_rough) / (self.roughness_model.initial_ra_nm + 1e-8)

        err = self.surface.get_error_map()
        neg_err = np.minimum(err, 0.0)
        overpolish = float(np.mean(neg_err[self.surface.mask] ** 2)) / (self.initial_rms**2 + 1e-8)

        reward = (
            self.w_form * rms_improve
            + self.w_rough * rough_improve
            - 10.0 * overpolish
        )
        reward = np.clip(reward, -5.0, 5.0)

        terminated = cur_rms < self.target_rms_um
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward += self.w_bonus

        if truncated or terminated:
            final_ratio = cur_rms / (self.initial_rms + 1e-8)
            reward += 5.0 * max(0, 1.0 - final_ratio)

        self.prev_rms = cur_rms
        self.prev_roughness = cur_rough

        info = {
            "rms_error": cur_rms,
            "pv_error": self.surface.get_pv_error(),
            "mean_roughness": cur_rough,
            "tool_x": tool_x,
            "tool_y": tool_y,
            "dwell_time": dwell,
            "pressure": pressure,
            "speed": self.fixed_speed,
            "step": self.current_step,
            "current_pass": cur_pass,
            "radial_progress": self.spiral.get_radial_position(wp),
        }

        return self._get_obs(), float(reward), terminated, truncated, info
