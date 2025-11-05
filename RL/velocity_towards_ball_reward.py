# velocity_towards_ball_reward.py
from typing import List, Dict, Any
import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

def _safe_norm(x: np.ndarray) -> float:
    n = float(np.linalg.norm(x))
    return n if n > 1e-9 else 1e-9

class VelocityTowardsBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward = projection of the car's velocity along the direction to the ball.
    Uses the same team-relative inversion as DefaultObs:
      - for ORANGE, use car.inverted_physics and state.inverted_ball
      - for BLUE, use car.physics and state.ball
    """
    def __init__(self, scale: float = 1.0, normalize: bool = True):
        super().__init__()
        self.scale = float(scale)
        self.normalize = bool(normalize)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            is_orange = (car.team_num == 1)

            # team-relative physics/ball, matching DefaultObs’ inversion
            phys = car.inverted_physics if is_orange else car.physics
            ball = state.inverted_ball   if is_orange else state.ball

            v = np.asarray(phys.linear_velocity, dtype=np.float32)
            to_ball = np.asarray(ball.position, dtype=np.float32) - np.asarray(phys.position, dtype=np.float32)

            unit = to_ball / _safe_norm(to_ball)
            proj = float(np.dot(v, unit))  # positive if moving toward ball along the line-of-sight

            if self.normalize:
                proj /= _safe_norm(v)  # becomes cos(theta), ∈ [-1, 1]

            rewards[agent] = self.scale * proj
        return rewards
