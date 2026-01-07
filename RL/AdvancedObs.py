import math
from typing import List, Dict, Any, Tuple

import numpy as np

from rlgym.api import ObsBuilder, AgentID
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM
from rlgym.rocket_league.obs_builders.default_obs import DefaultObs
# ===== AdvancedObs: DefaultObs + low-res grid + touch buffer + extra PO vars =====
from collections import deque

class AdvancedObs(DefaultObs):
    """
    Extends DefaultObs by appending:
      - low-res 2D occupancy grid (4x6) with 3 channels: [ball, allies, enemies]
      - touch_buffer: last-K touches encoded as [+1 (blue), -1 (orange), 0]
      - extra partially observable vars per-agent:
          car.boost_active_time (float), car.supersonic_time (float),
          car.wheels_with_contact (4 bools -> ints)
      - aerial rotation data for self + ball:
          self.angular_velocity[3], self.euler_angles[3], ball.angular_velocity[3]
    The adapter supplies 'touch_buffer' via shared_info.
    """
    def __init__(self, grid_bins=(4, 6), x_max=4096.0, y_max=5120.0, touch_k=8, **kwargs):
        super().__init__(**kwargs)
        self.grid_bins = grid_bins
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.touch_k = int(touch_k)

    def _grid_index(self, pos_xy):
        # map world (x,y) in [-x_max..x_max]x[-y_max..y_max] -> bins (gx, gy)
        x, y = float(pos_xy[0]), float(pos_xy[1])
        gx = int(np.clip((x + self.x_max) / (2 * self.x_max) * self.grid_bins[0], 0, self.grid_bins[0] - 1))
        gy = int(np.clip((y + self.y_max) / (2 * self.y_max) * self.grid_bins[1], 0, self.grid_bins[1] - 1))
        return gx, gy

    def _build_grid(self, agent: AgentID, state: GameState) -> np.ndarray:
        car = state.cars[agent]
        inverted = (car.team_num == ORANGE_TEAM)
        ball = state.inverted_ball if inverted else state.ball

        C, W, H = 3, self.grid_bins[0], self.grid_bins[1]
        grid = np.zeros((C, W, H), dtype=np.float32)

        # ball
        bx, by = self._grid_index((ball.position[0], ball.position[1]))
        grid[0, bx, by] = 1.0

        # cars (team-relative)
        for aid, c in state.cars.items():
            phys = c.inverted_physics if inverted else c.physics
            px, py = self._grid_index((phys.position[0], phys.position[1]))
            chan = 1 if c.team_num == car.team_num else 2
            grid[chan, px, py] = 1.0

        return grid.reshape(C * W * H)

    def _extra_po(self, car) -> np.ndarray:
        # some backends may not expose these; default gracefully
        bat = float(getattr(car, "boost_active_time", 0.0))
        sst = float(getattr(car, "supersonic_time", 0.0))
        w = getattr(car, "wheels_with_contact", (False, False, False, False))
        wheels = np.array(
            [int(bool(x)) for x in (w if isinstance(w, (list, tuple)) else (False, False, False, False))],
            dtype=np.float32
        )
        return np.concatenate([np.array([bat, sst], dtype=np.float32), wheels])

    def _rot_features(self, agent: AgentID, state: GameState) -> np.ndarray:
        """
        Aerial rotation features in team-relative frame:
          - self.angular_velocity[3]
          - self.euler_angles[3]  (pitch, yaw, roll)
          - ball.angular_velocity[3]
        """
        car = state.cars[agent]
        inverted = (car.team_num == ORANGE_TEAM)

        # self physics in team frame
        phys = car.inverted_physics if inverted else car.physics
        ang_self = np.asarray(phys.angular_velocity, dtype=np.float32)  # [3]
        euler = np.asarray(phys.euler_angles, dtype=np.float32)         # [3] (pitch, yaw, roll)

        # ball physics in same frame
        ball = state.inverted_ball if inverted else state.ball
        ang_ball = np.asarray(ball.angular_velocity, dtype=np.float32)  # [3]

        return np.concatenate([ang_self, euler, ang_ball], dtype=np.float32)  # [9]

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        base = super()._build_obs(agent, state, shared_info)  # DefaultObs vector
        car = state.cars[agent]
        grid = self._build_grid(agent, state)

        # touch buffer comes from adapter (already a fixed-length list of floats)
        tb = shared_info.get("touch_buffer", [])
        if len(tb) < getattr(self, "touch_k", 8):
            tb = list(tb) + [0.0] * (self.touch_k - len(tb))
        tb = np.asarray(tb[: self.touch_k], dtype=np.float32)

        po_extra = self._extra_po(car)
        rot_extra = self._rot_features(agent, state)

        return np.concatenate([base, grid, tb, po_extra, rot_extra], dtype=np.float32)

    def get_obs_space(self, agent: AgentID) -> tuple[str, int]:
        kind, base_n = super().get_obs_space(agent)
        C, W, H = 3, self.grid_bins[0], self.grid_bins[1]
        # po_extra: 6 (bat, sst, 4 wheels)
        # rot_extra: 9 (ang_self[3], euler[3], ang_ball[3])
        extra = (C * W * H) + self.touch_k + 15
        return kind, base_n + extra
