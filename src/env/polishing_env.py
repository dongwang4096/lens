"""Gymnasium environment for convex lens polishing with RL.

The agent controls the tool position, pressure, speed, and dwell time
to polish a convex lens surface toward a target form while minimizing
roughness and total processing time.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .lens_surface import LensSurface
from .roughness_model import RoughnessModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tif_model import TIFModel


class LensPolishingEnv(gym.Env):
    """2D convex lens polishing environment.

    Observation:
        Dict with:
        - "error_map": (1, H, W) normalized error map
        - "roughness_map": (1, H, W) normalized roughness map
        - "tool_pos": (2,) normalized tool position
        - "time_ratio": (1,) fraction of time budget used

    Action: (5,) continuous
        [dx, dy, pressure, speed, dwell_time] all in [-1, 1],
        mapped to physical ranges.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 128,
        lens_diameter_mm: float = 50.0,
        curvature_radius_mm: float = 200.0,
        initial_roughness_nm: float = 200.0,
        max_steps: int = 300,
        time_budget_s: float = 600.0,
        target_rms_um: float = 0.01,
        reward_form: float = 10.0,
        reward_roughness: float = 3.0,
        reward_time: float = 0.1,
        reward_bonus: float = 50.0,
        pressure_range: tuple = (5.0, 50.0),
        speed_range: tuple = (200.0, 1300.0),
        dwell_range: tuple = (0.5, 10.0),
        concentration: float = 5.0,
        tif_checkpoint: str | None = None,
        error_scale: float = 1.0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.lens_diameter_mm = lens_diameter_mm
        self.max_steps = max_steps
        self.time_budget_s = time_budget_s
        self.target_rms_um = target_rms_um
        self.concentration = concentration
        self.error_scale = error_scale

        self.w_form = reward_form
        self.w_rough = reward_roughness
        self.w_time = reward_time
        self.w_bonus = reward_bonus

        self.p_range = pressure_range
        self.v_range = speed_range
        self.dt_range = dwell_range

        self.max_move_mm = lens_diameter_mm * 0.15

        self.surface = LensSurface(grid_size, lens_diameter_mm, curvature_radius_mm)
        self.roughness_model = RoughnessModel(grid_size, initial_roughness_nm)
        self.tif_model = TIFModel(tif_checkpoint)

        self.observation_space = spaces.Dict({
            "maps": spaces.Box(
                low=-5.0, high=5.0,
                shape=(2, grid_size, grid_size),
                dtype=np.float32,
            ),
            "scalar": spaces.Box(
                low=-1.0, high=1.0,
                shape=(3,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self.tool_x = 0.0
        self.tool_y = 0.0
        self.elapsed_time = 0.0
        self.step_count = 0
        self.prev_rms = 0.0
        self.prev_roughness = 0.0
        self.initial_rms = 0.0

    def _decode_action(self, action: np.ndarray):
        """Map [-1,1]^5 to physical parameter space."""
        a = np.clip(action, -1.0, 1.0)
        dx = a[0] * self.max_move_mm
        dy = a[1] * self.max_move_mm
        pressure = self._lerp(a[2], *self.p_range)
        speed = self._lerp(a[3], *self.v_range)
        dwell = self._lerp(a[4], *self.dt_range)
        return dx, dy, pressure, speed, dwell

    @staticmethod
    def _lerp(t: float, lo: float, hi: float) -> float:
        return lo + (t + 1.0) / 2.0 * (hi - lo)

    def _get_obs(self) -> dict:
        err = self.surface.get_error_map().astype(np.float32)
        err_norm = np.clip(err / (self.initial_rms + 1e-8), -5.0, 5.0)

        rough = self.roughness_model.get_roughness_map().astype(np.float32)
        rough_norm = np.clip(rough / (self.roughness_model.initial_ra_nm + 1e-8), 0.0, 2.0)

        maps = np.stack([err_norm, rough_norm], axis=0)

        half = self.lens_diameter_mm / 2.0
        scalar = np.array([
            self.tool_x / half,
            self.tool_y / half,
            self.elapsed_time / self.time_budget_s,
        ], dtype=np.float32)

        return {"maps": maps, "scalar": scalar}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng_seed = self.np_random.integers(0, 2**31) if seed is None else seed

        self.surface.reset(error_scale=self.error_scale, seed=rng_seed)
        self.roughness_model.reset(seed=rng_seed + 1)

        self.tool_x = 0.0
        self.tool_y = 0.0
        self.elapsed_time = 0.0
        self.step_count = 0

        self.initial_rms = self.surface.get_rms_error()
        self.prev_rms = self.initial_rms
        self.prev_roughness = self.roughness_model.get_mean_roughness(self.surface.mask)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        dx, dy, pressure, speed, dwell = self._decode_action(action)

        half = self.lens_diameter_mm / 2.0
        self.tool_x = np.clip(self.tool_x + dx, -half, half)
        self.tool_y = np.clip(self.tool_y + dy, -half, half)

        tif_2d = self.tif_model.generate_tif_2d(
            pressure, speed, self.concentration,
            (self.surface.xx, self.surface.yy),
            (self.tool_x, self.tool_y),
        )

        removal = tif_2d * dwell
        self.surface.apply_removal(removal)

        tif_max = tif_2d.max()
        footprint = tif_2d / (tif_max + 1e-12)
        self.roughness_model.update(
            footprint, pressure, speed, self.concentration, dwell, self.surface.mask
        )

        self.elapsed_time += dwell
        self.step_count += 1

        cur_rms = self.surface.get_rms_error()
        cur_rough = self.roughness_model.get_mean_roughness(self.surface.mask)

        rms_improve_frac = (self.prev_rms - cur_rms) / (self.initial_rms + 1e-8)
        rough_improve_frac = (self.prev_roughness - cur_rough) / (self.roughness_model.initial_ra_nm + 1e-8)

        err = self.surface.get_error_map()
        neg_err = np.minimum(err, 0.0)
        overpolish_penalty = float(np.mean(neg_err[self.surface.mask] ** 2)) / (self.initial_rms**2 + 1e-8)

        reward = (
            self.w_form * rms_improve_frac
            + self.w_rough * rough_improve_frac
            - self.w_time * (dwell / self.time_budget_s)
            - 2.0 * overpolish_penalty
        )
        reward = np.clip(reward, -5.0, 5.0)

        terminated = False
        if cur_rms < self.target_rms_um:
            reward += self.w_bonus
            terminated = True

        truncated = (
            self.step_count >= self.max_steps
            or self.elapsed_time >= self.time_budget_s
        )

        self.prev_rms = cur_rms
        self.prev_roughness = cur_rough

        info = {
            "rms_error": cur_rms,
            "pv_error": self.surface.get_pv_error(),
            "mean_roughness": cur_rough,
            "elapsed_time": self.elapsed_time,
            "tool_x": self.tool_x,
            "tool_y": self.tool_y,
            "pressure": pressure,
            "speed": speed,
            "dwell_time": dwell,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
