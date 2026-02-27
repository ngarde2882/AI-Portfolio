from typing import List, Dict, Any, Union, Tuple

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState


class CombinedReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that does a weighted sum of multiple reward functions.
    """

    def __init__(self, *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]]):
        """
        :param rewards_and_weights: A list of reward functions and their corresponding weights.
        """
        reward_fns = []
        weights = []

        for value in rewards_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.
            reward_fns.append(r)
            weights.append(w)

        self.reward_fns = tuple(reward_fns)
        self.weights = tuple(weights)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for reward_fn in self.reward_fns:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        # TODO optimize this double for loop with a numpy matrix?
        combined_rewards = {agent: 0. for agent in agents}
        for reward_fn, weight in zip(self.reward_fns, self.weights):
            rewards = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent, reward in rewards.items():
                combined_rewards[agent] += reward * weight

        return combined_rewards

class LoggedCombinedReward(RewardFunction[AgentID, GameState, float]):
    """
    Like CombinedReward, but stores a per-agent per-component breakdown for the most recent call.
    - Does NOT re-call reward functions.
    - Breakdown values are AFTER weighting (reward * weight).
    """

    def __init__(self, *items: Union[RewardFunction, Tuple[RewardFunction, float], Tuple[str, RewardFunction, float]]):
        reward_fns = []
        weights = []
        names = []

        auto_i = 0
        for value in items:
            if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], str):
                name, r, w = value
            elif isinstance(value, tuple) and len(value) == 2:
                r, w = value
                name = type(r).__name__
            else:
                r, w = value, 1.0
                name = type(r).__name__

            reward_fns.append(r)
            weights.append(float(w))
            names.append(str(name))
            auto_i += 1

        self.reward_fns = tuple(reward_fns)
        self.weights = tuple(weights)
        self.names = tuple(names)

        # last_breakdown[aid][name] = weighted contribution for that component
        self.last_breakdown: Dict[AgentID, Dict[str, float]] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_breakdown = {a: {} for a in agents}
        for reward_fn in self.reward_fns:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        combined = {agent: 0.0 for agent in agents}
        breakdown = {agent: {} for agent in agents}

        for reward_fn, weight, name in zip(self.reward_fns, self.weights, self.names):
            rewards = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent, reward in rewards.items():
                wr = float(reward) * float(weight)
                combined[agent] += wr
                breakdown[agent][name] = breakdown[agent].get(name, 0.0) + wr

        self.last_breakdown = breakdown
        return combined
