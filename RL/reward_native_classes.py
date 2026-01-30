# reward_native_classes_normalized.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.common_values import BOOST_LOCATIONS, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM


# ----------------------- normalization constants -----------------------
# These are conservative arena/physics caps in Unreal units.
# They are used ONLY to bound/normalize dense shaping signals so that
# composite weights become interpretable and goal events can dominate.
ARENA_HALF_WIDTH_X = 4096.0
ARENA_HALF_LENGTH_Y = 5120.0
ARENA_MAX_HEIGHT_Z  = 2044.0

# Approx maximum 3D distance between two points in playable space
MAX_ARENA_DIST = float(np.linalg.norm(np.array([
    2.0 * ARENA_HALF_WIDTH_X,
    2.0 * ARENA_HALF_LENGTH_Y,
    ARENA_MAX_HEIGHT_Z
], dtype=np.float32)))

MAX_CAR_SPEED  = 2300.0   # uu/s (supersonic cap)
MAX_BALL_SPEED = 6000.0   # uu/s (hard-ish cap; actual may exceed slightly in edge cases)
MAX_BOOST      = 100.0    # 0..100
MAX_BOOST_PAD_COOLDOWN = 10.0  # seconds (large pad respawn ~10s; small pads are smaller)


# ----------------------- helpers -----------------------
def _safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if n > 1e-9 else 1e-9

def _safe_touches(car) -> int:
    val = getattr(car, "ball_touches", 0)
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _norm_dist(d: float, dmax: float = MAX_ARENA_DIST) -> float:
    # returns in [0,1]
    return _clip01(float(d) / float(dmax))

def _norm_speed(s: float, smax: float) -> float:
    # returns in [0,1]
    return _clip01(float(s) / float(smax))

def _rel(state: GameState, agent: AgentID):
    """
    Return useful, team-relative pieces for this agent:
      car        - Car object
      phys       - PhysicsObject in team-relative frame
      ball       - PhysicsObject in team-relative frame
      opp_goal   - np.ndarray (3,)
      own_goal   - np.ndarray (3,)
      is_orange  - bool

    For ORANGE we use inverted physics and inverted ball/pad timers so +Y is attack.
    """
    car = state.cars[agent]
    is_orange = (car.team_num == ORANGE_TEAM)
    phys = car.inverted_physics if is_orange else car.physics
    ball = state.inverted_ball   if is_orange else state.ball
    opp_goal = np.asarray(BLUE_GOAL_BACK if is_orange else ORANGE_GOAL_BACK, dtype=np.float32)
    own_goal = np.asarray(ORANGE_GOAL_BACK if is_orange else BLUE_GOAL_BACK, dtype=np.float32)
    return car, phys, ball, opp_goal, own_goal, is_orange


# ---------------- elementary rewards (pure GameState) ---------------
class DistToBall(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: closer to ball is better, normalized to [-1, 0]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, b, *_ = _rel(state, a)
            d = np.linalg.norm(np.asarray(p.position) - np.asarray(b.position))
            out[a] = self.weight * (-_norm_dist(d))
        return out


class CarSpeed(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: speed magnitude, normalized to [0, 1]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, *_ = _rel(state, a)
            s = np.linalg.norm(np.asarray(p.linear_velocity))
            out[a] = self.weight * _norm_speed(s, MAX_CAR_SPEED)
        return out


class BoostRemaining(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: remaining boost, normalized to [0, 1]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        return {a: self.weight * _norm_speed(float(state.cars[a].boost_amount), MAX_BOOST) for a in agents}


class BallHit(RewardFunction[AgentID, GameState, float]):
    """Touch-window 'ball progress' reward (horizontal, goal-directed).

    Motivation: exact touch timing/attribution can be noisy (simultaneous pinches, etc.).
    Instead of paying only on the touch tick, we start a short decay window after a
    detected touch and dole out a small reward each tick for up to `window_ticks`.

    Trigger:
      - Primary: per-agent `car.ball_touches` counter increases.
      - Secondary (optional): if shared_info contains `last_touch` with matching {"aid","tick"},
        we also treat that as a touch event. (Note: env-level last_touch may be ambiguous
        on simultaneous touches, so we do NOT rely on it exclusively.)

    Reward signal per active tick:
      - Compute horizontal ball velocity v_xy (ignore Z).
      - Compute horizontal direction from ball -> opponent goal dir_goal_xy (ignore Z).
      - alignment = dot( normalize(v_xy), normalize(dir_goal_xy) )  in [-1, 1]
      - speed_norm = clip(||v_xy|| / MAX_BALL_SPEED, 0, 1)
      - signal = alignment * speed_norm  in [-1, 1]

    We then divide by `window_ticks` so that the *total* contribution per touch window
    is O(1) and does not become a dense 'always-on' reward at 120 Hz.

    Output per tick while active:
        reward = weight * (signal / window_ticks)

    Notes:
      - Because alignment is with opponent-goal direction, moving the ball toward your
        own goal yields negative reward naturally.
      - Vertical ball motion does not directly affect the direction term.
    """
    def __init__(self, weight: float = 1.0, window_ticks: int = 120):
        super().__init__()
        self.weight = float(weight)
        self.window_ticks = int(window_ticks)
        self._prev_touches: Dict[AgentID, int] = {}
        self._ticks_left: Dict[AgentID, int] = {}

    def reset(self, agents, initial_state, shared):
        self._prev_touches = {a: _safe_touches(initial_state.cars[a]) for a in agents}
        self._ticks_left = {a: 0 for a in agents}

    def _touch_event(self, a: AgentID, state: GameState, shared: Dict[str, Any]) -> bool:
        # Optional shared_info hook
        last = shared.get("last_touch") if isinstance(shared, dict) else None
        if isinstance(last, dict):
            try:
                if last.get("aid") == a and int(last.get("tick", -999999)) == int(state.tick_count):
                    return True
            except Exception:
                pass

        # Robust fallback: per-agent touch counter
        car = state.cars[a]
        touches = _safe_touches(car)
        prev = int(self._prev_touches.get(a, 0))
        self._prev_touches[a] = touches
        return touches > prev

    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out: Dict[AgentID, float] = {}
        inv_window = 1.0 / max(1, self.window_ticks)

        for a in agents:
            # Reset/extend window on touch
            if self._touch_event(a, state, shared):
                self._ticks_left[a] = self.window_ticks

            if int(self._ticks_left.get(a, 0)) <= 0:
                out[a] = 0.0
                continue

            # Active tick: compute goal-directed horizontal velocity alignment
            _, _, b, opp_goal, *_ = _rel(state, a)

            ball_vel = np.asarray(b.linear_velocity, dtype=np.float32)
            v_xy = ball_vel[:2]
            v_xy_n = float(np.linalg.norm(v_xy))
            if v_xy_n < 1e-6:
                signal = 0.0
            else:
                ball_pos = np.asarray(b.position, dtype=np.float32)
                goal_pos = np.asarray(opp_goal, dtype=np.float32)
                dir_xy = (goal_pos - ball_pos)[:2]
                dir_n = float(np.linalg.norm(dir_xy))
                if dir_n < 1e-6:
                    signal = 0.0
                else:
                    v_hat = v_xy / v_xy_n
                    d_hat = dir_xy / dir_n
                    align = float(np.clip(np.dot(v_hat, d_hat), -1.0, 1.0))
                    speed_n = float(_norm_speed(v_xy_n, MAX_BALL_SPEED))
                    signal = align * speed_n  # [-1,1]

            out[a] = self.weight * (signal * inv_window)

            # Decrement window
            self._ticks_left[a] = int(self._ticks_left[a]) - 1

        return out


class DefHit(RewardFunction[AgentID, GameState, float]):
    """Defensive touch-window 'ball safety' reward (horizontal, away-from-own-goal).

    Same windowing mechanics as BallHit, but the direction signal is:
      + when ball horizontal velocity points away from our OWN goal
      - when it points toward our OWN goal

    This is intended for defender composites to encourage clears and to discourage
    touches that drive the ball toward your own net. Vertical component is ignored.
    """
    def __init__(self, weight: float = 1.0, window_ticks: int = 120):
        super().__init__()
        self.weight = float(weight)
        self.window_ticks = int(window_ticks)
        self._prev_touches: Dict[AgentID, int] = {}
        self._ticks_left: Dict[AgentID, int] = {}

    def reset(self, agents, initial_state, shared):
        self._prev_touches = {a: _safe_touches(initial_state.cars[a]) for a in agents}
        self._ticks_left = {a: 0 for a in agents}

    def _touch_event(self, a: AgentID, state: GameState, shared: Dict[str, Any]) -> bool:
        # Optional shared_info hook
        last = shared.get("last_touch") if isinstance(shared, dict) else None
        if isinstance(last, dict):
            try:
                if last.get("aid") == a and int(last.get("tick", -999999)) == int(state.tick_count):
                    return True
            except Exception:
                pass

        # Robust fallback: per-agent touch counter
        car = state.cars[a]
        touches = _safe_touches(car)
        prev = int(self._prev_touches.get(a, 0))
        self._prev_touches[a] = touches
        return touches > prev

    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out: Dict[AgentID, float] = {}
        inv_window = 1.0 / max(1, self.window_ticks)

        for a in agents:
            # Reset/extend window on touch
            if self._touch_event(a, state, shared):
                self._ticks_left[a] = self.window_ticks

            if int(self._ticks_left.get(a, 0)) <= 0:
                out[a] = 0.0
                continue

            # Active tick: compute away-from-own-goal horizontal velocity alignment
            _, _, b, _, own_goal, _ = _rel(state, a)

            ball_vel = np.asarray(b.linear_velocity, dtype=np.float32)
            v_xy = ball_vel[:2]
            v_xy_n = float(np.linalg.norm(v_xy))
            if v_xy_n < 1e-6:
                signal = 0.0
            else:
                ball_pos = np.asarray(b.position, dtype=np.float32)
                own_pos = np.asarray(own_goal, dtype=np.float32)
                away_xy = (ball_pos - own_pos)[:2]  # direction away from own goal
                away_n = float(np.linalg.norm(away_xy))
                if away_n < 1e-6:
                    signal = 0.0
                else:
                    v_hat = v_xy / v_xy_n
                    d_hat = away_xy / away_n
                    align = float(np.clip(np.dot(v_hat, d_hat), -1.0, 1.0))
                    speed_n = float(_norm_speed(v_xy_n, MAX_BALL_SPEED))
                    signal = align * speed_n  # [-1,1]

            out[a] = self.weight * (signal * inv_window)

            # Decrement window
            self._ticks_left[a] = int(self._ticks_left[a]) - 1

        return out


class BallDistToGoal(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: closer ball is to opponent goal, normalized to [-1, 0]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, _, b, opp_goal, *_ = _rel(state, a)
            d = np.linalg.norm(np.asarray(b.position) - opp_goal)
            out[a] = self.weight * (-_norm_dist(d))
        return out


class FaceBall(RewardFunction[AgentID, GameState, float]):
    """Cosine alignment with ball direction, bounded in [-1,1]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, b, *_ = _rel(state, a)
            to_ball = np.asarray(b.position) - np.asarray(p.position)
            cos = float(np.dot(p.forward, to_ball) / (_safe_norm(p.forward) * _safe_norm(to_ball)))
            out[a] = self.weight * cos
        return out


class ShotAlignment(RewardFunction[AgentID, GameState, float]):
    """Cosine alignment of ball velocity toward opponent goal, bounded [-1,1]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, _, b, opp_goal, *_ = _rel(state, a)
            bpos = np.asarray(b.position, np.float32)
            # Only reward if ball is in our attacking half (opponent half)
            if bpos[1] <= 0.0:
                out[a] = 0.0
                continue

            bvel = np.asarray(b.linear_velocity, np.float32)
            to_goal = opp_goal - bpos
            cos = float(np.dot(bvel, to_goal) / (_safe_norm(bvel) * _safe_norm(to_goal)))
            out[a] = self.weight * cos
        return out


class SupersonicBonus(RewardFunction[AgentID, GameState, float]):
    """Binary event, (0,1)."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        return {a: self.weight * (1.0 if state.cars[a].is_supersonic else 0.0) for a in agents}


class BehindBallDefensive(RewardFunction[AgentID, GameState, float]):
    """Binary event, (-1,1)."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, b, _, own_goal, _ = _rel(state, a)
            me = np.asarray(p.position); ball = np.asarray(b.position)
            gm = np.linalg.norm(me - own_goal); gb = np.linalg.norm(ball - own_goal)
            out[a] = self.weight * (1.0 if gm < gb else -1.0)
        return out


class CenterlineProximity(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: closer to centerline (y=0) is better, normalized to [-1,0]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, *_ = _rel(state, a)
            y = float(np.asarray(p.position)[1])
            out[a] = self.weight * (-_clip01(abs(y) / ARENA_HALF_LENGTH_Y))
        return out


class NearestBoostInverseDistance(RewardFunction[AgentID, GameState, float]):
    """Distance to nearest boost, bounded in (0,1]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        boost_positions = np.asarray(BOOST_LOCATIONS, dtype=np.float32)
        for a in agents:
            _, p, _, _, _, is_orange = _rel(state, a)
            pos = np.asarray(p.position, dtype=np.float32)
            # keep pad coords in the same inverted frame; flip Y for orange
            bpos = boost_positions.copy()
            if is_orange:
                bpos[:, 1] *= -1.0
            d = np.linalg.norm(bpos - pos[None, :], axis=1)
            nearest = float(np.min(d))
            out[a] = self.weight * (1.0 / (1.0 + nearest))
        return out


class NearestBoostAvailability(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: negative cooldown to nearest pad, normalized to [-1,0]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        boost_positions = np.asarray(BOOST_LOCATIONS, dtype=np.float32)
        for a in agents:
            _, p, _, _, _, is_orange = _rel(state, a)
            pos = np.asarray(p.position, dtype=np.float32)
            bpos = boost_positions.copy()
            timers = state.inverted_boost_pad_timers if is_orange else state.boost_pad_timers
            if is_orange:
                bpos[:, 1] *= -1.0
            d = np.linalg.norm(bpos - pos[None, :], axis=1)
            idx = int(np.argmin(d))
            cooldown = float(timers[idx])  # 0 = available, positive = not
            out[a] = self.weight * (-_norm_speed(cooldown, MAX_BOOST_PAD_COOLDOWN))
        return out


class MeanDistToTeammates(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: mean distance to teammates, normalized to [0,1]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            car, p, *_ = _rel(state, a)
            my_pos = np.asarray(p.position, dtype=np.float32)
            dists = []
            for aid, c in state.cars.items():
                if aid == a or c.team_num != car.team_num:
                    continue
                _, tp, *_ = _rel(state, aid)
                dists.append(np.linalg.norm(my_pos - np.asarray(tp.position, dtype=np.float32)))
            out[a] = self.weight * (_norm_dist(float(np.mean(dists))) if dists else 0.0)
        return out


class MeanDistToOpponents(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: mean distance to opponents, normalized to [0,1]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            car, p, *_ = _rel(state, a)
            my_pos = np.asarray(p.position, dtype=np.float32)
            dists = []
            for aid, c in state.cars.items():
                if aid == a or c.team_num == car.team_num:
                    continue
                _, op, *_ = _rel(state, aid)
                dists.append(np.linalg.norm(my_pos - np.asarray(op.position, dtype=np.float32)))
            out[a] = self.weight * (_norm_dist(float(np.mean(dists))) if dists else 0.0)
        return out


class FaceGoal(RewardFunction[AgentID, GameState, float]):
    """Cosine of car forward vs vector to OPPONENT goal (team-relative), bounded [-1,1]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, _, opp_goal, *_ = _rel(state, a)
            to_goal = opp_goal - np.asarray(p.position, np.float32)
            cos = float(np.dot(p.forward, to_goal) / (_safe_norm(p.forward) * _safe_norm(to_goal)))
            out[a] = self.weight * cos
        return out


class HomeGoalProximity(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: closer to OWN goal, normalized to [-1,0]."""
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, _, _, own_goal, _ = _rel(state, a)
            d = np.linalg.norm(np.asarray(p.position, np.float32) - own_goal)
            out[a] = self.weight * (-_norm_dist(d))
        return out


class BehindOtherPlayers(RewardFunction[AgentID, GameState, float]):
    """
    Fraction of other cars that are 'ahead' of me relative to my OWN goal.
    Output in [0,1]; already bounded.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, _, _, own_goal, _ = _rel(state, a)
            me_d = np.linalg.norm(np.asarray(p.position, np.float32) - own_goal)
            others = []
            for aid in state.cars.keys():
                if aid == a:
                    continue
                _, p2, _, _, own_goal2, _ = _rel(state, aid)
                others.append(np.linalg.norm(np.asarray(p2.position, np.float32) - own_goal2))
            frac = 0.0 if not others else float(sum(d < me_d for d in others)) / float(len(others))
            out[a] = self.weight * frac
        return out


class BlockAlignment(RewardFunction[AgentID, GameState, float]):
    """
    Defensive analogue of ShotAlignment.

    +1 when ball velocity is aligned directly away from our OWN goal,
    -1 when ball is moving straight toward our own goal.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, _, b, _, own_goal, _ = _rel(state, a)
            bpos = np.asarray(b.position, np.float32)
            # Only reward if ball is in our defensive half (our half)
            if bpos[1] >= 0.0:
                out[a] = 0.0
                continue
            bvel = np.asarray(b.linear_velocity, np.float32)
            to_own = own_goal - bpos
            away_from_own = -to_own
            cos = float(np.dot(bvel, away_from_own) / (_safe_norm(bvel) * _safe_norm(away_from_own)))
            out[a] = self.weight * cos
        return out


# ---------------- composites with mutable dicts ----------------
class _CompositeBase(RewardFunction[AgentID, GameState, float]):
    DEFAULT_SIGMA = 0.10
    def __init__(self, weights: Dict[str, float] | None = None):
        super().__init__()
        self._weights: Dict[str, float] = self.default_weights() if weights is None else dict(weights)
        self._combo = CombinedReward(*self.parts(self._weights))

    # override in subclasses
    def default_weights(self) -> Dict[str, float]: return {}
    def parts(self, w: Dict[str, float]) -> List[Tuple[RewardFunction, float]]: raise NotImplementedError

    def set_weights(self, new_w: Dict[str, float]) -> None:
        self._weights.update(new_w)
        self._combo = CombinedReward(*self.parts(self._weights))
    def get_weights(self) -> Dict[str, float]: return dict(self._weights)
    def mutate(self, sigma: float = DEFAULT_SIGMA, clip: Tuple[float, float] | None = None) -> None:
        lo, hi = (-np.inf, np.inf) if clip is None else clip
        m = {k: float(np.clip(v + np.random.normal(0.0, sigma), lo, hi)) for k, v in self._weights.items()}
        self.set_weights(m)

    def reset(self, agents, initial_state, shared): self._combo.reset(agents, initial_state, shared)
    def get_rewards(self, agents, state, is_term, is_trunc, shared): return self._combo.get_rewards(agents, state, is_term, is_trunc, shared)


class StrikerCompositeReward(_CompositeBase):
    def default_weights(self) -> Dict[str, float]:
        return {
            'dist_to_ball': 1.0e-5,
            'car_speed': 1.0e-5,
            'boost_remaining': 1.0e-5,
            'ball_hit': 1.5e-2,
            'ball_dist_to_goal': 1.0e-5,
            'face_ball': 5.0e-5,
            'shot_alignment': 5.0e-5,
            'supersonic_bonus': 1.0e-5,
            'goal': 10.0,
            'touch': 2.0e-2,
        }
    def parts(self, w: Dict[str, float]):
        return [
            (GoalReward(), w['goal']),
            (TouchReward(), w['touch']),
            (DistToBall(), w['dist_to_ball']),
            (CarSpeed(), w['car_speed']),
            (BoostRemaining(), w['boost_remaining']),
            (BallHit(), w['ball_hit']),
            (BallDistToGoal(), w['ball_dist_to_goal']),
            (FaceBall(), w['face_ball']),
            (ShotAlignment(), w['shot_alignment']),
            (SupersonicBonus(), w['supersonic_bonus']),
        ]


class DefenderCompositeReward(_CompositeBase):
    def default_weights(self) -> Dict[str, float]:
        return {
            'boost_remaining': 1.0e-5,
            'dist_to_ball': 1.0e-5,
            'centerline_proximity': 2.0e-5,
            'home_goal_proximity': 5.0e-5,
            'face_ball': 5.0e-5,
            'block_alignment': 5.0e-5,
            'behind_ball_defensive': 5.0e-5,
            'behind_other_players': 5.0e-5,
            'def_hit': 3.0e-2,
            'touch': 1.0e-2,
            'goal': 10.0,
        }
    def parts(self, w: Dict[str, float]):
        return [
            (GoalReward(), w['goal']),
            (TouchReward(), w['touch']),
            (DistToBall(), w['dist_to_ball']),
            (BoostRemaining(), w['boost_remaining']),
            (DefHit(), w['def_hit']),
            (BehindBallDefensive(), w['behind_ball_defensive']),
            (CenterlineProximity(), w['centerline_proximity']),
            (FaceBall(), w['face_ball']),
            (BehindOtherPlayers(), w['behind_other_players']),
            (HomeGoalProximity(), w['home_goal_proximity']),
            (BlockAlignment(), w['block_alignment']),
        ]


class PositioningCompositeReward(_CompositeBase):
    def default_weights(self) -> Dict[str, float]:
        return {
            'mean_dist_to_teammates': 1.0e-5,
            'mean_dist_to_opponents': 1.0e-5,
            'centerline_proximity': 1.0e-5,
            'face_ball': 5.0e-5,
            'face_goal': 5.0e-5,
            'car_speed': 1.0e-5,
            'boost_remaining': 1.0e-5,
            'def_hit': 1.0e-2,
            'shot_alignment': 5.0e-5,
            'block_alignment': 5.0e-5,
            'goal': 10.0,
        }
    def parts(self, w: Dict[str, float]):
        return [
            (GoalReward(), w['goal']),
            (MeanDistToTeammates(), w['mean_dist_to_teammates']),
            (MeanDistToOpponents(), w['mean_dist_to_opponents']),
            (CenterlineProximity(), w['centerline_proximity']),
            (FaceBall(), w['face_ball']),
            (FaceGoal(), w['face_goal']),
            (CarSpeed(), w['car_speed']),
            (BoostRemaining(), w['boost_remaining']),
            (DefHit(), w['def_hit']),
            (ShotAlignment(), w['shot_alignment']),
            (BlockAlignment(), w['block_alignment']),
        ]


# Tiny factory if you want names -> classes quickly
COMPOSITE_FACTORIES = {
    'striker': StrikerCompositeReward,
    'defender': DefenderCompositeReward,
    'positioning': PositioningCompositeReward,
}
def make_composite(name: str, weights: Dict[str, float] | None = None) -> _CompositeBase:
    name = name.lower()
    if name not in COMPOSITE_FACTORIES:
        raise KeyError(f"Unknown composite '{name}'. Options: {sorted(COMPOSITE_FACTORIES.keys())}")
    return COMPOSITE_FACTORIES[name](weights=weights)
