# reward_native_classes_normalized.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

from rlgym.rocket_league.reward_functions import GoalReward, TouchReward
from rlgym.rocket_league.common_values import BOOST_LOCATIONS, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED
from combined_reward import CombinedReward, LoggedCombinedReward


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
# BALL_MAX_SPEED imported from common_values (~6000 uu/s hard cap)
# BallHit/DefHit normalize against 3/4 of that so realistic hard shots saturate the signal
BALL_HIT_SPEED_NORM = BALL_MAX_SPEED * 0.75
MAX_BOOST      = 100.0    # 0..100
MAX_BOOST_PAD_COOLDOWN = 10.0  # seconds (large pad respawn ~10s; small pads are smaller)

# Standard Rocket League goal is ~1786 uu wide; half-width used for post targets
GOAL_HALF_WIDTH = 893.0


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
    # In the inverted frame both teams attack +Y (ORANGE_GOAL_BACK) and defend -Y (BLUE_GOAL_BACK).
    # These constants are the same for blue and orange — do NOT condition on is_orange here.
    opp_goal = np.asarray(ORANGE_GOAL_BACK, dtype=np.float32)
    own_goal = np.asarray(BLUE_GOAL_BACK,   dtype=np.float32)
    return car, phys, ball, opp_goal, own_goal, is_orange


# ---------------- elementary rewards (pure GameState) ---------------
_DIST_TO_BALL_DEAD_ZONE = 900.0   # uu — no penalty within this radius
_DIST_TO_BALL_CONTACT_TICKS = 75  # ~5s at action_repeat=8, 120Hz → suppress penalty after touch

class DistToBall(RewardFunction[AgentID, GameState, float]):
    """Dense shaping: penalty ramps from 0 at 900uu to -1 at max arena dist.

    Within 900uu: reward = 0 (dead zone, no pressure to flee the ball).
    Beyond 900uu: reward = -clip((d - dead_zone) / (MAX_ARENA_DIST - dead_zone), 0, 1).
    Contact bonus: after any touch, suppress the penalty for ~5 in-game seconds
    (75 action steps), giving agents time to follow up without being penalized.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)
        self._contact_ticks: Dict[AgentID, int] = {}
        self._prev_touches: Dict[AgentID, int] = {}

    def reset(self, agents, initial_state, shared):
        self._contact_ticks = {a: 0 for a in agents}
        self._prev_touches  = {a: _safe_touches(initial_state.cars[a]) for a in agents}

    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            # Detect touch to start/extend contact bonus window
            touches = _safe_touches(state.cars[a])
            if touches > int(self._prev_touches.get(a, 0)):
                self._contact_ticks[a] = _DIST_TO_BALL_CONTACT_TICKS
            self._prev_touches[a] = touches

            if int(self._contact_ticks.get(a, 0)) > 0:
                self._contact_ticks[a] -= 1
                out[a] = 0.0  # suppress penalty during contact window
                continue

            _, p, b, *_ = _rel(state, a)
            d = float(np.linalg.norm(np.asarray(p.position) - np.asarray(b.position)))
            if d <= _DIST_TO_BALL_DEAD_ZONE:
                out[a] = 0.0
            else:
                penalty = _clip01((d - _DIST_TO_BALL_DEAD_ZONE) / (MAX_ARENA_DIST - _DIST_TO_BALL_DEAD_ZONE))
                out[a] = self.weight * (-penalty)
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
    def __init__(self, weight: float = 1.0, window_ticks: int = 60):
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

            # Active tick: compute best alignment across goal center and near posts
            _, _, b, opp_goal, *_ = _rel(state, a)

            ball_vel = np.asarray(b.linear_velocity, dtype=np.float32)
            v_xy = ball_vel[:2]
            v_xy_n = float(np.linalg.norm(v_xy))
            if v_xy_n < 1e-6:
                signal = 0.0
            else:
                ball_pos_xy = np.asarray(b.position, dtype=np.float32)[:2]
                goal_center = np.asarray(opp_goal, dtype=np.float32)[:2]
                # Left and right goal posts in team-relative frame (X is lateral, same for both teams)
                left_post  = goal_center.copy(); left_post[0]  -= GOAL_HALF_WIDTH
                right_post = goal_center.copy(); right_post[0] += GOAL_HALF_WIDTH

                v_hat = v_xy / v_xy_n
                speed_n = float(_norm_speed(v_xy_n, BALL_HIT_SPEED_NORM))

                def _target_align(tgt: np.ndarray) -> float:
                    d = tgt - ball_pos_xy
                    dn = float(np.linalg.norm(d))
                    if dn < 1e-6:
                        return 0.0
                    return float(np.clip(np.dot(v_hat, d / dn), -1.0, 1.0))

                # Best on-frame target wins; shots past the posts get lower alignment naturally
                align = max(_target_align(goal_center), _target_align(left_post), _target_align(right_post))
                signal = max(0.0, align * speed_n)  # [0, 1] — positive touches only

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
    def __init__(self, weight: float = 1.0, window_ticks: int = 60):
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
                away_xy = (ball_pos - own_pos)[:2]  # direction away from own goal center
                away_n = float(np.linalg.norm(away_xy))
                if away_n < 1e-6:
                    signal = 0.0
                else:
                    v_hat = v_xy / v_xy_n
                    d_hat = away_xy / away_n
                    align = float(np.clip(np.dot(v_hat, d_hat), -1.0, 1.0))
                    speed_n = float(_norm_speed(v_xy_n, BALL_HIT_SPEED_NORM))
                    signal = max(0.0, align * speed_n)  # [0, 1] — positive clears only

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


class BallDistFromGoal(RewardFunction[AgentID, GameState, float]):
    """Defender cost: 0 when ball is in the opponent half; approaches -1 as ball nears own net.

    Only active in own half (team-relative ball_py <= 0).  Uses distance from own goal so
    the signal is smooth at the midfield boundary (dist ≈ ARENA_HALF_LENGTH_Y → signal ≈ 0)
    and saturates at -1 when the ball is at the own goal mouth (dist ≈ 0).
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, _, b, _, own_goal, _ = _rel(state, a)
            if float(b.position[1]) > 0.0:  # ball in opponent half — no cost
                out[a] = 0.0
                continue
            dist = float(np.linalg.norm(np.asarray(b.position) - own_goal))
            # 0 at midfield (dist ≈ 5120), -1 at own goal (dist ≈ 0)
            out[a] = self.weight * (-_clip01(1.0 - dist / ARENA_HALF_LENGTH_Y))
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
    """
    +1 when the car is between the ball and its own goal along Y (team-relative frame).
    -1 otherwise.

    In the inverted frame both teams defend -Y, so car is "behind ball"
    (closer to own goal) when car_y < ball_y.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, b, *_ = _rel(state, a)
            behind = float(p.position[1]) < float(b.position[1])
            out[a] = self.weight * (1.0 if behind else -1.0)
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
    """
    Zone-based teammate spacing reward.

    Signal is negative when teammates are too close (crowding), zero at the
    inner radius r_eff, and positive up to +1 at the outer radius R_eff.

    Both radii scale linearly with each agent's distance to the nearest goal:
      - Near a goal  (dist ≈ 0)      → r_eff = r_near, R_eff = R_near  (tighter)
      - At midfield  (dist ≈ 5120)   → r_eff = r_far,  R_eff = R_far   (wider)

    This allows natural bunching near the net (transitions, blocks, shots)
    while still rewarding spread-out play in the open field.

    Output in [-1, 1]:
      -1  →  mean teammate dist = 0 (fully stacked)
       0  →  mean teammate dist = r_eff
      +1  →  mean teammate dist ≥ R_eff
    """
    def __init__(
        self,
        weight: float = 1.0,
        r_far:  float = 900.0,
        R_far:  float = 5000.0,
        r_near: float = 300.0,
        R_near: float = 3500.0,
    ):
        super().__init__()
        self.weight = float(weight)
        self.r_far  = float(r_far)
        self.R_far  = float(R_far)
        self.r_near = float(r_near)
        self.R_near = float(R_near)

    def reset(self, agents, initial_state, shared): pass

    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        blue_goal_w   = np.asarray(BLUE_GOAL_BACK,   dtype=np.float32)
        orange_goal_w = np.asarray(ORANGE_GOAL_BACK, dtype=np.float32)

        for a in agents:
            car, p, *_ = _rel(state, a)
            my_pos = np.asarray(p.position, dtype=np.float32)

            # World-frame position for goal-proximity scaling
            my_pos_w = np.asarray(state.cars[a].physics.position, dtype=np.float32)
            dist_to_goal = float(min(
                np.linalg.norm(my_pos_w - blue_goal_w),
                np.linalg.norm(my_pos_w - orange_goal_w),
            ))
            # 0 = at a goal, 1 = at half-field distance or beyond
            goal_frac = _clip01(dist_to_goal / ARENA_HALF_LENGTH_Y)
            r_eff = self.r_near + goal_frac * (self.r_far  - self.r_near)
            R_eff = self.R_near + goal_frac * (self.R_far  - self.R_near)

            dists = []
            for aid, c in state.cars.items():
                if aid == a or c.team_num != car.team_num:
                    continue
                _, tp, *_ = _rel(state, aid)
                dists.append(float(np.linalg.norm(my_pos - np.asarray(tp.position, dtype=np.float32))))

            if not dists:
                out[a] = 0.0
                continue

            mean_d = float(np.mean(dists))
            if mean_d < r_eff:
                # Crowding: -1 at d=0, 0 at d=r_eff
                signal = (mean_d / r_eff) - 1.0
            else:
                # Reward: 0 at r_eff, +1 at R_eff, flat beyond
                signal = min((mean_d - r_eff) / max(R_eff - r_eff, 1.0), 1.0)

            out[a] = self.weight * float(signal)
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
    Fraction of other cars (all 5) that are farther from THIS agent's own goal
    than the agent itself, measured in world-frame coordinates.

    Output in [0, 1]:  1.0 = agent is the last car back (most defensive).
    Formula: players_farther_from_homegoal / (total_players - 1)
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            car_a = state.cars[a]
            is_orange = (car_a.team_num == ORANGE_TEAM)
            # All distances in world frame so every car is on the same scale
            own_goal = np.asarray(
                ORANGE_GOAL_BACK if is_orange else BLUE_GOAL_BACK, dtype=np.float32
            )
            my_pos = np.asarray(car_a.physics.position, np.float32)
            me_d = float(np.linalg.norm(my_pos - own_goal))

            n_farther = 0
            n_others = 0
            for aid in state.cars.keys():
                if aid == a:
                    continue
                n_others += 1
                other_pos = np.asarray(state.cars[aid].physics.position, np.float32)
                if float(np.linalg.norm(other_pos - own_goal)) > me_d:
                    n_farther += 1

            frac = 0.0 if n_others == 0 else float(n_farther) / float(n_others)
            out[a] = self.weight * frac
        return out


class BlockAlignment(RewardFunction[AgentID, GameState, float]):
    """
    Defensive analogue of ShotAlignment. Uses team-relative frame via _rel().

    +1 when ball velocity is aligned directly away from our own goal,
    -1 when ball is moving straight toward our own goal.
    Only active when ball is in our defensive half (team-relative ball_y < 0).
    In the inverted frame both teams defend -Y (own_goal = BLUE_GOAL_BACK).
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, _, b, _, own_goal, _ = _rel(state, a)
            bpos = np.asarray(b.position, np.float32)
            if float(bpos[1]) >= 0.0:
                out[a] = 0.0
                continue
            bvel = np.asarray(b.linear_velocity, np.float32)
            away_from_own = bpos - own_goal
            cos = float(np.dot(bvel, away_from_own) / (_safe_norm(bvel) * _safe_norm(away_from_own)))
            out[a] = self.weight * cos
        return out


# ---------------- composites with mutable dicts ----------------
class _CompositeBase(RewardFunction[AgentID, GameState, float]):
    DEFAULT_SIGMA = 0.10
    def __init__(self, weights: Dict[str, float] | None = None):
        super().__init__()
        self._weights: Dict[str, float] = self.default_weights() if weights is None else dict(weights)
        self._combo = LoggedCombinedReward(*self.parts(self._weights))

    # override in subclasses
    def default_weights(self) -> Dict[str, float]: return {}
    def parts(self, w: Dict[str, float]) -> List[Tuple[RewardFunction, float]]: raise NotImplementedError

    def set_weights(self, new_w: Dict[str, float]) -> None:
        self._weights.update(new_w)
        self._combo = LoggedCombinedReward(*self.parts(self._weights))
    def get_weights(self) -> Dict[str, float]: return dict(self._weights)
    def mutate(self, sigma: float = DEFAULT_SIGMA, clip: Tuple[float, float] | None = None) -> None:
        lo, hi = (-np.inf, np.inf) if clip is None else clip
        m = {k: float(np.clip(v + np.random.normal(0.0, sigma), lo, hi)) for k, v in self._weights.items()}
        self.set_weights(m)

    def reset(self, agents, initial_state, shared): self._combo.reset(agents, initial_state, shared)
    def get_rewards(self, agents, state, is_term, is_trunc, shared): return self._combo.get_rewards(agents, state, is_term, is_trunc, shared)
    def get_last_breakdown(self, aid: AgentID) -> Dict[str, float]:
        lb = getattr(self._combo, "last_breakdown", None)
        if not isinstance(lb, dict):
            return {}
        d = lb.get(aid, {})
        return dict(d) if isinstance(d, dict) else {}


class StrikerCompositeReward(_CompositeBase):
    def default_weights(self) -> Dict[str, float]:
        return {
            'dist_to_ball': 8.0e-5,
            'car_speed': 5.0e-6,
            'ball_hit': 1.0e-1,
            'ball_dist_to_goal': 3.0e-5,
            'face_ball': 2.0e-5,
            'shot_alignment': 7.5e-5,
            'goal': 20.0,
            'touch': 1.0e-1,
        }
    def parts(self, w: Dict[str, float]):
        return [
            ("goal", GoalReward(), w["goal"]),
            ("touch", TouchReward(), w["touch"]),
            ("dist_to_ball", DistToBall(), w["dist_to_ball"]),
            ("car_speed", CarSpeed(), w["car_speed"]),
            ("ball_hit", BallHit(), w["ball_hit"]),
            ("ball_dist_to_goal", BallDistToGoal(), w["ball_dist_to_goal"]),
            ("face_ball", FaceBall(), w["face_ball"]),
            ("shot_alignment", ShotAlignment(), w["shot_alignment"]),
        ]


class DefenderCompositeReward(_CompositeBase):
    def default_weights(self) -> Dict[str, float]:
        return {
            'dist_to_ball': 4.0e-5,
            'face_ball': 2.0e-5,
            'block_alignment': 5.0e-5,
            'behind_ball_defensive': 2.0e-6,
            'def_hit': 1.0e-1,
            'touch': 1.0e-1,
            'ball_dist_from_goal': 5.0e-5,
            'goal': 20.0,
        }
    def parts(self, w: Dict[str, float]):
        return [
            ("goal", GoalReward(), w["goal"]),
            ("touch", TouchReward(), w["touch"]),
            ("dist_to_ball", DistToBall(), w["dist_to_ball"]),
            ("def_hit", DefHit(), w["def_hit"]),
            ("behind_ball_defensive", BehindBallDefensive(), w["behind_ball_defensive"]),
            ("face_ball", FaceBall(), w["face_ball"]),
            ("block_alignment", BlockAlignment(), w["block_alignment"]),
            ("ball_dist_from_goal", BallDistFromGoal(), w["ball_dist_from_goal"]),
        ]


class PositioningCompositeReward(_CompositeBase):
    def default_weights(self) -> Dict[str, float]:
        return {
            'dist_to_ball': 4.0e-5,
            'ball_dist_to_goal': 3.0e-5,
            'mean_dist_to_teammates': 5.0e-6,
            'behind_ball_defensive': 2.0e-6,
            'face_ball': 2.0e-5,
            'car_speed': 5.0e-6,
            'def_hit': 1.0e-1,
            'ball_hit': 1.0e-1,
            'shot_alignment': 5.0e-5,
            'block_alignment': 5.0e-5,
            'touch': 1.0e-1,
            'goal': 20.0,
        }
    def parts(self, w: Dict[str, float]):
        return [
            ("goal", GoalReward(), w["goal"]),
            ("dist_to_ball", DistToBall(), w["dist_to_ball"]),
            ("ball_dist_to_goal", BallDistToGoal(), w["ball_dist_to_goal"]),
            ("mean_dist_to_teammates", MeanDistToTeammates(), w["mean_dist_to_teammates"]),
            ("behind_ball_defensive", BehindBallDefensive(), w["behind_ball_defensive"]),
            ("face_ball", FaceBall(), w["face_ball"]),
            ("car_speed", CarSpeed(), w["car_speed"]),
            ("def_hit", DefHit(), w["def_hit"]),
            ("ball_hit", BallHit(), w["ball_hit"]),
            ("shot_alignment", ShotAlignment(), w["shot_alignment"]),
            ("block_alignment", BlockAlignment(), w["block_alignment"]),
            ("touch", TouchReward(), w["touch"]),
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
