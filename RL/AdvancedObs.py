import math
from typing import List, Dict, Any, Tuple

import numpy as np

from rlgym.api import ObsBuilder, AgentID
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM
from rlgym.rocket_league.obs_builders.default_obs import DefaultObs
# ===== AdvancedObs: DefaultObs + low-res grid + touch buffer + extra PO vars =====

class AdvancedObs(DefaultObs):
    """
    Extends DefaultObs by appending:
      - low-res occupancy grid with altitude (W x H x Z) and channels:
          [ball, allies, enemies] x Z
      - touch_buffer: last-K touches encoded as [+1 (blue), -1 (orange), 0]
      - extra partially observable vars per-agent:
          car.boost_active_time (float), car.supersonic_time (float),
          car.wheels_with_contact (4 bools -> ints)

      - profile/role one-hot for self (from shared_info)
      - teammate profile one-hots (distance-sorted, from shared_info)  [teammates only]
      - streamlined team-shape features (team-relative frame):
          * distances to teammates (sorted)
          * distance between teammates (pairwise among teammates, for 3v3 this is one value)
          * team centroids (ally/opponent) and centroid distance
          * triangle area for each team (3v3)
          * summed velocities: teammates-only (excluding self) and opponents (all opponents)

    The adapter supplies:
      - 'touch_buffer'
      - 'profile_by_agent' and 'role_by_agent'
    """
    def __init__(
        self,
        grid_bins: tuple[int, int] = (4, 6),
        z_bins: int = 2,
        x_max: float = 4096.0,
        y_max: float = 5120.0,
        z_max: float = 2044.0,
        air_threshold: float = 300.0,
        touch_k: int = 8,
        profile_names=None,
        role_names=("striker", "positioning", "defender"),
        max_teammates: int = 2,  # for 3v3
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grid_bins = grid_bins
        self.z_bins = int(z_bins)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.z_max = float(z_max)
        self.air_threshold = float(air_threshold)
        self.touch_k = int(touch_k)

        # profile + role names for one-hots
        self.profile_names = list(profile_names or [])
        self.profile_name_to_idx = {n: i for i, n in enumerate(self.profile_names)}
        self.role_names = list(role_names)
        self.role_to_idx = {r: i for i, r in enumerate(self.role_names)}

        self.max_teammates = int(max_teammates)

    # ---------------------------- helpers ----------------------------

    def _grid_index_xy(self, pos_xy):
        x, y = float(pos_xy[0]), float(pos_xy[1])
        gx = int(np.clip((x + self.x_max) / (2 * self.x_max) * self.grid_bins[0], 0, self.grid_bins[0] - 1))
        gy = int(np.clip((y + self.y_max) / (2 * self.y_max) * self.grid_bins[1], 0, self.grid_bins[1] - 1))
        return gx, gy

    def _grid_index_z(self, z: float):
        # If z_bins==2, we treat z as {ground, air} using air_threshold.
        if self.z_bins <= 1:
            return 0
        if self.z_bins == 2:
            return 1 if float(z) > self.air_threshold else 0
        # More than 2 bins: uniform binning up to z_max.
        zc = float(np.clip(z, 0.0, self.z_max))
        return int(np.clip(zc / self.z_max * self.z_bins, 0, self.z_bins - 1))

    def _triangle_area_xy(self, pts_xy: np.ndarray) -> float:
        """
        Shoelace area for 3 points in XY. If not 3 points, returns 0.
        pts_xy: shape (3,2)
        """
        if pts_xy.shape != (3, 2):
            return 0.0
        x1, y1 = pts_xy[0]
        x2, y2 = pts_xy[1]
        x3, y3 = pts_xy[2]
        return 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

    def _safe_vec3(self, v) -> np.ndarray:
        try:
            a = np.asarray(v, dtype=np.float32).reshape(-1)
        except Exception:
            return np.zeros((3,), dtype=np.float32)
        if a.shape[0] >= 3:
            return a[:3].astype(np.float32)
        out = np.zeros((3,), dtype=np.float32)
        out[:a.shape[0]] = a
        return out

    # ---------------------------- grid (with altitude) ----------------------------

    def _build_grid(self, agent: AgentID, state: GameState) -> np.ndarray:
        """
        Returns flattened (channels * z_bins * W * H) occupancy.
        Channels: 0=ball, 1=allies, 2=enemies
        Z: 0=ground, 1=air (for z_bins==2) or more bins if configured.
        """
        car = state.cars[agent]
        inverted = (car.team_num == ORANGE_TEAM)
        ball = state.inverted_ball if inverted else state.ball

        C, Z, W, H = 3, self.z_bins, self.grid_bins[0], self.grid_bins[1]
        grid = np.zeros((C, Z, W, H), dtype=np.float32)

        # ball
        bx, by = self._grid_index_xy((ball.position[0], ball.position[1]))
        bz = self._grid_index_z(ball.position[2] if len(ball.position) > 2 else 0.0)
        grid[0, bz, bx, by] = 1.0

        # cars (team-relative)
        for aid, c in state.cars.items():
            phys = c.inverted_physics if inverted else c.physics
            px, py = self._grid_index_xy((phys.position[0], phys.position[1]))
            pz = self._grid_index_z(phys.position[2] if len(phys.position) > 2 else 0.0)
            chan = 1 if c.team_num == car.team_num else 2
            grid[chan, pz, px, py] = 1.0

        return grid.reshape(C * Z * W * H)

    # ---------------------------- extra partially observable vars ----------------------------

    def _extra_po(self, car) -> np.ndarray:
        bat = float(getattr(car, "boost_active_time", 0.0))
        sst = float(getattr(car, "supersonic_time", 0.0))
        w = getattr(car, "wheels_with_contact", (False, False, False, False))
        wheels = np.array(
            [int(bool(x)) for x in (w if isinstance(w, (list, tuple)) else (False, False, False, False))],
            dtype=np.float32
        )
        return np.concatenate([np.array([bat, sst], dtype=np.float32), wheels])

    # ---------------------------- profile features ----------------------------

    def _self_profile_features(self, agent: AgentID, shared_info: Dict[str, Any]) -> np.ndarray:
        role_by_agent = shared_info.get("role_by_agent", {})
        prof_by_agent = shared_info.get("profile_by_agent", {})

        role = str(role_by_agent.get(agent, "")).lower()
        prof = str(prof_by_agent.get(agent, ""))

        role_oh = np.zeros((len(self.role_names),), dtype=np.float32)
        if role in self.role_to_idx:
            role_oh[self.role_to_idx[role]] = 1.0

        prof_oh = np.zeros((len(self.profile_names),), dtype=np.float32)
        if prof in self.profile_name_to_idx:
            prof_oh[self.profile_name_to_idx[prof]] = 1.0

        return np.concatenate([role_oh, prof_oh]).astype(np.float32)

    def _teammate_profile_features(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        """
        Teammate-only deployed profile one-hots, ordered by distance to self (closest first).
        Shape: max_teammates * len(profile_names)
        If profile_names is empty, returns empty vector.
        """
        if len(self.profile_names) == 0 or self.max_teammates <= 0:
            return np.zeros((0,), dtype=np.float32)

        car = state.cars[agent]
        inverted = (car.team_num == ORANGE_TEAM)
        self_phys = car.inverted_physics if inverted else car.physics
        self_xy = np.asarray(self_phys.position[:2], dtype=np.float32)

        prof_by_agent = shared_info.get("profile_by_agent", {})

        teammates = []
        for aid, c in state.cars.items():
            if aid == agent:
                continue
            if c.team_num != car.team_num:
                continue
            phys = c.inverted_physics if inverted else c.physics
            xy = np.asarray(phys.position[:2], dtype=np.float32)
            d = float(np.linalg.norm(xy - self_xy))
            teammates.append((d, aid))

        teammates.sort(key=lambda x: x[0])
        teammates = teammates[:self.max_teammates]

        out = np.zeros((self.max_teammates, len(self.profile_names)), dtype=np.float32)
        for i, (_, aid) in enumerate(teammates):
            prof = str(prof_by_agent.get(aid, ""))
            j = self.profile_name_to_idx.get(prof, None)
            if j is not None:
                out[i, j] = 1.0
        return out.reshape(-1)

    # ---------------------------- team-shape features ----------------------------

    def _team_shape_features(self, agent: AgentID, state: GameState) -> np.ndarray:
        """
        Streamlined positional patterns in team-relative frame (for 3v3 this is compact):
          - dist_to_teammates_sorted: [d1, d2]
          - teammate_pair_dist: [dist(teammate1, teammate2)]  (0 if not available)
          - ally_centroid_xy, opp_centroid_xy, centroid_dist: [ax, ay, ox, oy, d]
          - ally_triangle_area, opp_triangle_area: [A_ally, A_opp]
          - sum_teammate_vel (excluding self), sum_opponent_vel: [tvx,tvy,tvz, ovx,ovy,ovz]
        """
        car = state.cars[agent]
        inverted = (car.team_num == ORANGE_TEAM)

        self_phys = car.inverted_physics if inverted else car.physics
        self_xy = np.asarray(self_phys.position[:2], dtype=np.float32)

        ally_xy = []
        ally_vel = []
        opp_xy = []
        opp_vel = []

        teammate_xy = []  # allies excluding self for pair distance
        teammate_vel = [] # allies excluding self for sum

        for aid, c in state.cars.items():
            phys = c.inverted_physics if inverted else c.physics
            xy = np.asarray(phys.position[:2], dtype=np.float32)
            vel = self._safe_vec3(getattr(phys, "linear_velocity", (0, 0, 0)))

            if c.team_num == car.team_num:
                ally_xy.append(xy)
                ally_vel.append(vel)
                if aid != agent:
                    teammate_xy.append(xy)
                    teammate_vel.append(vel)
            else:
                opp_xy.append(xy)
                opp_vel.append(vel)

        # distances to teammates sorted
        d_sorted = []
        for xy in teammate_xy:
            d_sorted.append(float(np.linalg.norm(xy - self_xy)))
        d_sorted.sort()
        # pad/truncate to max_teammates
        while len(d_sorted) < self.max_teammates:
            d_sorted.append(0.0)
        d_sorted = d_sorted[:self.max_teammates]

        # teammate distance from each other (for 3v3, only one)
        if len(teammate_xy) >= 2:
            pair_d = float(np.linalg.norm(teammate_xy[0] - teammate_xy[1]))
        else:
            pair_d = 0.0

        # centroids (include self in ally centroid)
        ally_cent = np.mean(np.stack(ally_xy), axis=0) if len(ally_xy) else np.zeros((2,), dtype=np.float32)
        opp_cent = np.mean(np.stack(opp_xy), axis=0) if len(opp_xy) else np.zeros((2,), dtype=np.float32)
        cent_dist = float(np.linalg.norm(ally_cent - opp_cent))

        # triangle areas (3v3)
        ally_area = 0.0
        if len(ally_xy) == 3:
            ally_area = float(self._triangle_area_xy(np.stack(ally_xy)))
        opp_area = 0.0
        if len(opp_xy) == 3:
            opp_area = float(self._triangle_area_xy(np.stack(opp_xy)))

        # summed velocities
        sum_mate_vel = np.sum(np.stack(teammate_vel), axis=0) if len(teammate_vel) else np.zeros((3,), dtype=np.float32)
        sum_opp_vel = np.sum(np.stack(opp_vel), axis=0) if len(opp_vel) else np.zeros((3,), dtype=np.float32)

        return np.asarray(
            list(d_sorted) +
            [pair_d] +
            [float(ally_cent[0]), float(ally_cent[1]), float(opp_cent[0]), float(opp_cent[1]), cent_dist] +
            [ally_area, opp_area] +
            [float(sum_mate_vel[0]), float(sum_mate_vel[1]), float(sum_mate_vel[2]),
             float(sum_opp_vel[0]), float(sum_opp_vel[1]), float(sum_opp_vel[2])],
            dtype=np.float32
        )

    # ---------------------------- main obs ----------------------------

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        base = super()._build_obs(agent, state, shared_info)

        grid = self._build_grid(agent, state)

        tb = shared_info.get("touch_buffer", [])
        if len(tb) < self.touch_k:
            tb = list(tb) + [0.0] * (self.touch_k - len(tb))
        tb = np.asarray(tb[: self.touch_k], dtype=np.float32)

        car = state.cars[agent]
        po_extra = self._extra_po(car)
        self_prof = self._self_profile_features(agent, shared_info)
        teammate_prof = self._teammate_profile_features(agent, state, shared_info)

        team_shape = self._team_shape_features(agent, state)

        return np.concatenate(
            [base, grid, tb, po_extra, self_prof, teammate_prof, team_shape],
            dtype=np.float32
        )

    def get_obs_space(self, agent: AgentID) -> tuple[str, int]:
        kind, base_n = super().get_obs_space(agent)
        C, Z, W, H = 3, self.z_bins, self.grid_bins[0], self.grid_bins[1]

        po_extra = 6  # bat,sst + 4 wheels
        self_prof = len(self.role_names) + len(self.profile_names)
        teammate_prof = self.max_teammates * len(self.profile_names)

        # team_shape dims:
        #   dists_to_teammates: max_teammates
        #   pair_d: 1
        #   centroids+dist: 5
        #   areas: 2
        #   summed vels: 6
        team_shape = self.max_teammates + 1 + 5 + 2 + 6

        extra = (C * Z * W * H) + self.touch_k + po_extra + self_prof + teammate_prof + team_shape
        return kind, base_n + extra
