"""
Hotswap HRL scaffolding:
- AgentProfile: wraps a composite RewardFunction with a role/name + mutable weights; supports serialize/deserialize.
- TeamProfilePool: manages a team's library of profiles (striker/defender/positioning or custom roles).
- HotswapManager: plugs into your EngineEnvAdapter to assign per-agent profiles and swap them during play.

Assumptions:
- You have reward_native_classes.{StrikerCompositeReward, DefenderCompositeReward, PositioningCompositeReward}.
- Your EngineEnvAdapter accepts reward_function=None; we'll compute rewards per-agent here.
- Actions are still parsed via your action_parser (90-way LookupTableAction + RepeatAction) upstream.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Callable, List, Optional, Any
import json
import numpy as np

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

from reward_native_classes import (
    StrikerCompositeReward,
    DefenderCompositeReward,
    PositioningCompositeReward,
    _CompositeBase,
)

# ----------------------------------------------------------------------------
# Agent profile abstraction
# ----------------------------------------------------------------------------
@dataclass
class AgentProfile:
    name: str
    role: str  # e.g. "striker" | "defender" | "positioning" | custom
    weights: Dict[str, float]

    def build_composite(self) -> _CompositeBase:
        role_lc = self.role.lower()
        if role_lc == "striker":
            return StrikerCompositeReward(weights=self.weights)
        if role_lc == "defender":
            return DefenderCompositeReward(weights=self.weights)
        if role_lc == "positioning":
            return PositioningCompositeReward(weights=self.weights)
        raise KeyError(f"Unknown role '{self.role}' for profile '{self.name}'")

    # --- persistence ---
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "AgentProfile":
        data = json.loads(s)
        return AgentProfile(**data)


# ----------------------------------------------------------------------------
# Team-wide pool of profiles
# ----------------------------------------------------------------------------
class TeamProfilePool:
    def __init__(self):
        self._by_name: Dict[str, AgentProfile] = {}
        self._by_role: Dict[str, List[str]] = {}

    def add(self, profile: AgentProfile) -> None:
        self._by_name[profile.name] = profile
        self._by_role.setdefault(profile.role.lower(), []).append(profile.name)

    def get(self, name: str) -> AgentProfile:
        return self._by_name[name]

    def names_for_role(self, role: str) -> List[str]:
        return list(self._by_role.get(role.lower(), []))

    def sample_for_role(self, role: str, rng=np.random) -> Optional[AgentProfile]:
        names = self.names_for_role(role)
        if not names:
            return None
        return self._by_name[rng.choice(names)]

    def dump_json(self) -> str:
        return json.dumps([asdict(p) for p in self._by_name.values()], indent=2)

    @staticmethod
    def load_json(s: str) -> "TeamProfilePool":
        pool = TeamProfilePool()
        for data in json.loads(s):
            pool.add(AgentProfile(**data))
        return pool


# ----------------------------------------------------------------------------
# Hotswap manager: maps each AgentID -> current composite, and can swap on the fly
# ----------------------------------------------------------------------------
class HotswapManager:
    def __init__(self, profile_pool: TeamProfilePool, policy: Callable[[AgentID, GameState], str]):
        """
        policy: function that returns the *profile name* to use for a given agent & state.
        You can implement AC-based or heuristic policies; we provide a default heuristic below.
        """
        self.pool = profile_pool
        self.policy = policy
        self.current: Dict[AgentID, RewardFunction] = {}
        self.current_name: Dict[AgentID, str] = {}

    def ensure_initialized(self, agents: List[AgentID], state: GameState) -> None:
        for aid in agents:
            if aid not in self.current:
                name = self.policy(aid, state)
                prof = self.pool.get(name)
                comp = prof.build_composite()
                comp.reset([aid], state, {})
                self.current[aid] = comp
                self.current_name[aid] = name

    def maybe_hotswap(self, agents: List[AgentID], state: GameState) -> None:
        for aid in agents:
            desired = self.policy(aid, state)
            if self.current_name.get(aid) != desired:
                prof = self.pool.get(desired)
                comp = prof.build_composite()
                comp.reset([aid], state, {})
                self.current[aid] = comp
                self.current_name[aid] = desired

    def get_rewards(self, agents: List[AgentID], state: GameState) -> Dict[AgentID, float]:
        # Call after maybe_hotswap to compute rewards with the active composite per agent
        rewards: Dict[AgentID, float] = {}
        for aid in agents:
            comp = self.current[aid]
            rmap = comp.get_rewards([aid], state, {aid: False}, {aid: False}, {})
            rewards[aid] = float(rmap[aid])
        return rewards


# ----------------------------------------------------------------------------
# Heuristic policy (starter): role by game context
# ----------------------------------------------------------------------------
# This can be replaced by an Actor-Critic that outputs a distribution over profile names.

def default_policy_factory(pool: TeamProfilePool) -> Callable[[AgentID, GameState], str]:
    """Pick a role by simple cues: if ball is on our half or we're behind the ball -> defender; else striker.
    If neither role is available in pool, fall back to any available profile.
    """
    from rlgym.rocket_league.common_values import ORANGE_TEAM

    def choose(aid: AgentID, state: GameState) -> str:
        car = state.cars[aid]
        is_orange = (car.team_num == ORANGE_TEAM)
        # team-relative Y: use inverted ball for orange so +Y is attack
        ball_y = float(state.inverted_ball.position[1] if is_orange else state.ball.position[1])
        # heuristic thresholds (field center at y=0 in team-relative frame)
        prefer_def = (ball_y < 0.0)
        role = "defender" if prefer_def and pool.names_for_role("defender") else "striker"
        names = pool.names_for_role(role)
        if not names:
            # fallback to any profile
            all_names = list(pool._by_name.keys())
            return all_names[0]
        # simple deterministic pick: first for reproducibility (replace with learned AC later)
        return names[0]

    return choose


# ----------------------------------------------------------------------------
# Glue: step-time reward with hotswapping
# ----------------------------------------------------------------------------
class HotswapRewardAdapter:
    """
    Adapter that you can pass to EngineEnvAdapter as `reward_function`, which in turn delegates to
    the HotswapManager each step. It initializes the current profiles on reset and allows on-the-fly swaps.
    """
    def __init__(self, hotswap: HotswapManager):
        self.hotswap = hotswap

    # Match RewardFunction API used by EngineEnvAdapter
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.hotswap.ensure_initialized(agents, initial_state)

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        self.hotswap.maybe_hotswap(agents, state)
        return self.hotswap.get_rewards(agents, state)
