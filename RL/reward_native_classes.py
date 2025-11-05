# reward_native_classes.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward

from rlgym.rocket_league.common_values import BOOST_LOCATIONS, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM


# ----------------------- helpers -----------------------
def _safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if n > 1e-9 else 1e-9

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
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, b, *_ = _rel(state, a)
            val = -np.linalg.norm(np.asarray(p.position) - np.asarray(b.position))
            out[a] = self.weight * float(val)
        return out


class CarSpeed(RewardFunction[AgentID, GameState, float]):
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, *_ = _rel(state, a)
            out[a] = self.weight * float(np.linalg.norm(np.asarray(p.linear_velocity)))
        return out


class BoostRemaining(RewardFunction[AgentID, GameState, float]):
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        return {a: self.weight * float(state.cars[a].boost_amount) for a in agents}


class BallHit(RewardFunction[AgentID, GameState, float]):
    """
    Simple shaping: if agent touched ball this tick, reward proportional to
    ball speed and alignment with car forward->ball direction.
    """
    def __init__(self, weight: float = 1.0, dir_bonus: float = 1.0):
        super().__init__(); self.weight = float(weight); self.dir_bonus = float(dir_bonus)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            car, p, b, *_ = _rel(state, a)
            touched = getattr(car, "ball_touches", 0) > 0
            if not touched:
                out[a] = 0.0
                continue
            ball_vel = np.asarray(b.linear_velocity, dtype=np.float32)
            speed = _safe_norm(ball_vel)
            to_ball = np.asarray(b.position) - np.asarray(p.position)
            align = float(np.dot(p.forward, to_ball / _safe_norm(to_ball)))
            val = speed * (1.0 + self.dir_bonus * align)
            out[a] = self.weight * val
        return out


class BallDistToGoal(RewardFunction[AgentID, GameState, float]):
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, _, b, opp_goal, *_ = _rel(state, a)
            val = -np.linalg.norm(np.asarray(b.position) - opp_goal)
            out[a] = self.weight * float(val)
        return out


class FaceBall(RewardFunction[AgentID, GameState, float]):
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
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, _, b, opp_goal, *_ = _rel(state, a)
            bvel = np.asarray(b.linear_velocity, np.float32)
            to_goal = opp_goal - np.asarray(b.position, np.float32)
            cos = float(np.dot(bvel, to_goal) / (_safe_norm(bvel) * _safe_norm(to_goal)))
            out[a] = self.weight * cos
        return out


class SupersonicBonus(RewardFunction[AgentID, GameState, float]):
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        return {a: self.weight * (1.0 if state.cars[a].is_supersonic else 0.0) for a in agents}


class BehindBallDefensive(RewardFunction[AgentID, GameState, float]):
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
    def __init__(self, weight: float = 1.0):
        super().__init__(); self.weight = float(weight)
    def reset(self, agents, initial_state, shared): pass
    def get_rewards(self, agents, state, is_term, is_trunc, shared):
        out = {}
        for a in agents:
            _, p, *_ = _rel(state, a)
            y = float(np.asarray(p.position)[1])
            out[a] = self.weight * (-abs(y))
        return out


class NearestBoostInverseDistance(RewardFunction[AgentID, GameState, float]):
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
            cooldown = float(timers[idx])  # 0 = available, big positive = not
            out[a] = self.weight * (-cooldown)
        return out


class MeanDistToTeammates(RewardFunction[AgentID, GameState, float]):
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
                _, tp, *_ = _rel(state, aid)  # team-relative for me as well
                dists.append(np.linalg.norm(my_pos - np.asarray(tp.position, dtype=np.float32)))
            out[a] = self.weight * (float(np.mean(dists)) if dists else 0.0)
        return out


class MeanDistToOpponents(RewardFunction[AgentID, GameState, float]):
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
            out[a] = self.weight * (float(np.mean(dists)) if dists else 0.0)
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
            'dist_to_ball': 0.0,
            'car_speed': 0.10,
            'boost_remaining': 0.05,
            'ball_hit': 0.50,
            'ball_dist_to_goal': 0.10,
            'face_ball': 0.25,
            'shot_alignment': 0.75,
            'supersonic_bonus': 0.20,
            'goal': 10.0,
            'touch': 0.50,
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
            'dist_to_ball': 0.10,
            'boost_remaining': 0.10,
            'ball_hit': 0.30,
            'behind_ball_defensive': 1.00,
            'centerline_proximity': 0.40,
            'nearest_boost_inverse_distance': 0.20,
            'nearest_boost_availability': 0.10,
            'face_ball': 0.30,
            'goal': 8.0,
            'touch': 0.30,
        }
    def parts(self, w: Dict[str, float]):
        return [
            (GoalReward(), w['goal']),
            (TouchReward(), w['touch']),
            (DistToBall(), w['dist_to_ball']),
            (BoostRemaining(), w['boost_remaining']),
            (BallHit(), w['ball_hit']),
            (BehindBallDefensive(), w['behind_ball_defensive']),
            (CenterlineProximity(), w['centerline_proximity']),
            (NearestBoostInverseDistance(), w['nearest_boost_inverse_distance']),
            (NearestBoostAvailability(), w['nearest_boost_availability']),
            (FaceBall(), w['face_ball']),
        ]


class PositioningCompositeReward(_CompositeBase):
    def default_weights(self) -> Dict[str, float]:
        return {
            'mean_dist_to_teammates': 0.30,
            'mean_dist_to_opponents': 0.20,
            'centerline_proximity': 0.50,
            'face_ball': 0.40,
        }
    def parts(self, w: Dict[str, float]):
        return [
            (MeanDistToTeammates(), w['mean_dist_to_teammates']),
            (MeanDistToOpponents(), w['mean_dist_to_opponents']),
            (CenterlineProximity(), w['centerline_proximity']),
            (FaceBall(), w['face_ball']),
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
