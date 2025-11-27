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
from typing import Dict, Callable, List, Optional, Tuple, Any
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
    def __init__(self, profile_pool: TeamProfilePool, policy: Optional[Callable[[AgentID, GameState], str]] = None):
        self.pool = profile_pool
        self.policy = policy
        self.current: Dict[AgentID, RewardFunction] = {}
        self.current_name: Dict[AgentID, str] = {}

    # NEW: let AC set a profile directly
    def set_profile(self, aid: AgentID, state: GameState, profile_name: str) -> None:
        prof = self.pool.get(profile_name)
        comp = prof.build_composite()
        comp.reset([aid], state, {})
        self.current[aid] = comp
        self.current_name[aid] = profile_name

    def ensure_initialized(self, agents: List[AgentID], state: GameState) -> None:
        for aid in agents:
            if aid not in self.current:
                # if no policy, just pick the first profile available in the pool
                name = self.policy(aid, state) if self.policy else next(iter(self.pool._by_name.keys()))
                self.set_profile(aid, state, name)

    def maybe_hotswap(self, agents: List[AgentID], state: GameState) -> None:
        if not self.policy:
            return  # AC will drive swaps externally
        for aid in agents:
            desired = self.policy(aid, state)
            if self.current_name.get(aid) != desired:
                self.set_profile(aid, state, desired)
    
    def get_rewards(self, agents, state):
        """
        Compute per-agent rewards using the currently assigned composite for each agent.
        AC should have already set profiles via set_profile(...). If not, we lazy-init.
        """
        # make sure every agent has a composite
        self.ensure_initialized(agents, state)

        # rlgym RewardFunction API expects dicts for term/trunc even if all False
        is_term   = {aid: False for aid in agents}
        is_trunc  = {aid: False for aid in agents}
        shared    = {}

        rewards = {}
        for aid in agents:
            comp = self.current[aid]                  # composite RewardFunction for this agent
            rmap = comp.get_rewards([aid], state, is_term, is_trunc, shared)
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
# AC policy: HRL
# ----------------------------------------------------------------------------
"""
AC profile policy to choose low-level agent assignments (hotswap roles/profiles).

Features:
- Per-agent Actor-Critic that picks a profile name from the team's pool.
- Switch penalty that decays to 0 over 25 seconds of game time since last switch.
- Optional event features (ball speed/dir change, field half, spacing) appended to DefaultObs.
- One forward pass per step for all agents; integrates with HotswapManager.

Assumptions:
- You already have TeamProfilePool with profile names (actions) and HotswapManager.
- Your Engine/State exposes seconds via `state.seconds_elapsed` (fallback: internal step counter * dt=0.0167).
- You have rlgym's DefaultObs available to build AC observations.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.common_values import ORANGE_TEAM, TICKS_PER_SECOND

from reward_native_classes import (
    StrikerCompositeReward,
    DefenderCompositeReward,
    PositioningCompositeReward,
    _CompositeBase,
)

# ------------------------------ utils ------------------------------

def _safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if n > 1e-9 else 1e-9


def _get_time_seconds(state: GameState) -> float:
    # absolute game time in seconds
    # t = getattr(state, "tick_count", None)
    return float(state.tick_count) / TICKS_PER_SECOND # if t is not None else 0.0



# ----------------------- Event feature extractor -----------------------
class ACEventFeatures:
    """Tracks simple event deltas to encourage timely reassignments."""
    def __init__(self):
        self._prev_ball_vel: np.ndarray | None = None
        self._prev_ball_pos: np.ndarray | None = None

    def reset(self):
        self._prev_ball_vel = None
        self._prev_ball_pos = None

    def build(self, state: GameState, team_is_orange: bool) -> np.ndarray:
        # team-relative ball (so +Y is attack for both teams)
        ball = state.inverted_ball if team_is_orange else state.ball
        pos = np.asarray(ball.position, np.float32)
        vel = np.asarray(ball.linear_velocity, np.float32)

        # speed & direction changes
        dv = 0.0
        ddir = 0.0
        if self._prev_ball_vel is not None:
            dv = float(np.linalg.norm(vel) - np.linalg.norm(self._prev_ball_vel))
            u = vel / _safe_norm(vel)
            u_prev = self._prev_ball_vel / _safe_norm(self._prev_ball_vel)
            ddir = float(np.clip(np.dot(u, u_prev), -1.0, 1.0))  # cosine similarity of direction
        self._prev_ball_vel = vel

        # coarse field context: team half (ball_y sign) and speed
        half = float(np.sign(pos[1]))  # -1 back half, +1 front half in team frame
        speed = float(np.linalg.norm(vel))

        # very simple spacing proxy: use car positions to get variance in x,y (team-relative)
        xs, ys = [], []
        for aid, car in state.cars.items():
            is_orange = (car.team_num == ORANGE_TEAM)
            phys = car.inverted_physics if is_orange else car.physics
            p = np.asarray(phys.position, np.float32)
            xs.append(p[0]); ys.append(p[1])
        if xs:
            x_var = float(np.var(xs)); y_var = float(np.var(ys))
        else:
            x_var = y_var = 0.0

        self._prev_ball_pos = pos
        return np.array([dv, ddir, half, speed, x_var, y_var], dtype=np.float32)


# -------------------------- AC network --------------------------
class ACNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.body(x)
        return self.pi(z), self.v(z)


@dataclass
class ACConfig:
    switch_penalty_base: float = 0.5     # starting penalty applied right after a switch
    switch_decay_seconds: float = 25.0   # linearly decay to 0 over this many seconds
    lr: float = 3e-4


class ACProfilePolicy:
    """Actor-Critic over profile names with decaying switch penalty (lazy init).

    Now uses a single team-level observation vector for all agents, composed of:
      - Ball state: [ball_xyz, ball_vel_xyz, ball_speed]
      - Team shape (relative to ball): [mean_rel_xyz, var_x, var_y, closest_idx_norm]
      - Opponent threat: [nearest_dist, opp_speeds[3], opp_aligns[3]]
      - Event features (ACEventFeatures): [dv, ddir, half, speed, x_var, y_var]
    """
    def __init__(self, ac_obs_builder: DefaultObs, profile_names: List[str],
                 cfg: ACConfig = ACConfig(), device: str = "cpu"):
        self.ac_obs_builder = ac_obs_builder  # kept for compatibility/logging, not used in obs
        self.profile_names  = list(profile_names)
        self.name_to_idx    = {n: i for i, n in enumerate(self.profile_names)}
        self.cfg            = cfg
        self.device         = device

        # event features appended to team-level obs (6 dims)
        self.event_feats = ACEventFeatures()

        # lazy init members
        self.obs_dim: int | None = None
        self.net: ACNet | None   = None
        self.opt: optim.Optimizer | None = None

        # switch bookkeeping
        self.current_choice: Dict[Any, str] = {}
        self.last_switch_time: Dict[Any, float] = {}
        self._step_counter = 0

    # ---------- helpers ----------

    def _ensure_net(self, obs_dim: int):
        """Lazy-init AC network once we know obs_dim."""
        if self.net is not None:
            return
        self.obs_dim = obs_dim
        self.net = ACNet(obs_dim, len(self.profile_names)).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    def _build_team_obs(self, state: GameState, team_agents: List[Any]) -> np.ndarray:
        """
        Build a single team-level observation vector.

        Ball + team shape + opponent threat + event features, all in a frame where
        our team attacks +Y (via 'frame_is_orange' based on first team agent).
        """
        assert len(team_agents) > 0, "ACProfilePolicy._build_team_obs: no team agents."

        # Choose which team we are controlling (team of first agent)
        first_car = state.cars[team_agents[0]]
        team_id = first_car.team_num
        frame_is_orange = (team_id == ORANGE_TEAM)

        # --- Ball state in team frame ---
        ball = state.inverted_ball if frame_is_orange else state.ball
        bpos = np.asarray(ball.position, dtype=np.float32)
        bvel = np.asarray(ball.linear_velocity, dtype=np.float32)
        bspeed = _safe_norm(bvel)  # scalar speed

        ball_feats = np.concatenate([bpos, bvel, np.array([bspeed], dtype=np.float32)], axis=0)
        # shape: [7]

        # --- Team shape: only our team's cars, in ball-relative coords ---
        team_positions = []
        for aid in team_agents:
            car = state.cars[aid]
            if car.team_num != team_id:
                continue  # skip opponents
            phys = car.inverted_physics if frame_is_orange else car.physics
            p = np.asarray(phys.position, dtype=np.float32)
            team_positions.append(p)

        if not team_positions:
            # fall back to origin-relative dummy
            team_positions = [np.zeros(3, dtype=np.float32)]

        team_positions = np.stack(team_positions, axis=0)  # [N_team, 3]
        rel_team = team_positions - bpos[None, :]          # ball-relative
        mean_rel = rel_team.mean(axis=0)                   # [3]
        var_rel = rel_team.var(axis=0)                     # [3]
        dists = np.linalg.norm(rel_team, axis=1)           # [N_team]
        closest_idx = int(np.argmin(dists))
        # normalize index to [0,1]; 0 if only one teammate
        closest_norm = float(closest_idx / max(1, len(team_positions) - 1)) if len(team_positions) > 1 else 0.0

        mean_x, mean_y, mean_z = mean_rel.tolist()
        var_x, var_y, _ = var_rel.tolist()

        team_shape = np.array(
            [mean_x, mean_y, mean_z, var_x, var_y, closest_norm],
            dtype=np.float32,
        )
        # shape: [6]

        # --- Opponent threat: nearest opponent, their speeds and alignment to ball ---
        opp_dists: List[float] = []
        opp_speeds: List[float] = []
        opp_align: List[float] = []

        for car in state.cars.values():
            if car.team_num == team_id:
                continue  # skip our team; we only want opponents

            phys = car.inverted_physics if frame_is_orange else car.physics
            pos = np.asarray(phys.position, dtype=np.float32)
            vel = np.asarray(phys.linear_velocity, dtype=np.float32)

            rel = pos - bpos
            dist = float(np.linalg.norm(rel))
            speed = _safe_norm(vel)

            # "Aligned to ball": cosine between opponent velocity and direction-to-ball
            if dist > 1e-3 and speed > 1e-3:
                to_ball = -rel  # vector from opponent to ball
                cos = float(np.dot(vel, to_ball) / (_safe_norm(vel) * _safe_norm(to_ball)))
            else:
                cos = 0.0

            opp_dists.append(dist)
            opp_speeds.append(speed)
            opp_align.append(cos)

        if opp_dists:
            nearest = float(min(opp_dists))
            order = np.argsort(np.asarray(opp_dists))
            max_opps = 3  # track up to 3 opponents
            speeds_sorted = [opp_speeds[i] for i in order[:max_opps]]
            align_sorted = [opp_align[i] for i in order[:max_opps]]

            # zero-pad to length 3
            while len(speeds_sorted) < max_opps:
                speeds_sorted.append(0.0)
                align_sorted.append(0.0)
        else:
            nearest = 0.0
            speeds_sorted = [0.0, 0.0, 0.0]
            align_sorted = [0.0, 0.0, 0.0]

        opponent_feats = np.array(
            [nearest, *speeds_sorted, *align_sorted],
            dtype=np.float32,
        )
        # shape: [1 + 3 + 3] = [7]

        # --- Event features (ball deltas, half, speed, spacing) ---
        # Use same frame as above (frame_is_orange) so everything is consistent.
        ev = self.event_feats.build(state, frame_is_orange)
        # shape: [6]

        # Final team-level observation
        team_obs = np.concatenate([ball_feats, team_shape, opponent_feats, ev], axis=0)
        return team_obs.astype(np.float32)

    def _switch_penalty(self, agent, state: GameState, new_choice: str) -> float:
        prev_choice = self.current_choice.get(agent)
        if prev_choice is None or prev_choice == new_choice:
            return 0.0
        t0 = self.last_switch_time.get(agent, -1e9)
        dt = max(0.0, _get_time_seconds(state) - t0)
        if dt >= self.cfg.switch_decay_seconds:
            return 0.0
        frac = 1.0 - (dt / self.cfg.switch_decay_seconds)
        return self.cfg.switch_penalty_base * frac

    def reset(self):
        self.event_feats.reset()
        self.current_choice.clear()
        self.last_switch_time.clear()
        self._step_counter = 0
        # keep net/opt alive across episodes

    @torch.no_grad()
    def act(self, state: GameState, team_agents: List[Any], team_is_orange: bool) -> Dict[Any, str]:
        """
        Decide profile for each agent given the current GameState.
        `team_is_orange` is ignored for building obs; we infer the controlled team from the first agent.
        """
        if not team_agents:
            return {}

        # Build shared team-level obs and ensure AC net exists
        team_obs = self._build_team_obs(state, team_agents)
        self._ensure_net(team_obs.size)

        # Same team_obs for each agent in this call
        x_np = np.repeat(team_obs[None, :], len(team_agents), axis=0)
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)

        logits, _ = self.net(x)

        # apply per-agent switch penalty in logit space
        logits_np = logits.cpu().numpy()
        for i, aid in enumerate(team_agents):
            for name, idx in self.name_to_idx.items():
                logits_np[i, idx] -= self._switch_penalty(aid, state, name)

        probs = torch.softmax(
            torch.as_tensor(logits_np, dtype=torch.float32, device=self.device),
            dim=-1
        )
        # Deterministic for now; you can change to sampling for exploration
        actions = torch.argmax(probs, dim=-1)

        out: Dict[Any, str] = {}
        for i, aid in enumerate(team_agents):
            choice = self.profile_names[int(actions[i].item())]
            if self.current_choice.get(aid) != choice:
                self.last_switch_time[aid] = _get_time_seconds(state)
                self.current_choice[aid] = choice
            out[aid] = choice

        self._step_counter += 1
        return out

    # training hook unchanged; you can feed returns/adv built from win-condition rewards
    def step_update(self, batch_obs: np.ndarray, acts_idx: np.ndarray, returns: np.ndarray, adv: np.ndarray):
        x = torch.as_tensor(batch_obs, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(acts_idx, dtype=torch.int64, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        logits, v = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        # simple A2C-style update
        pi_loss = -(logp * adv_t).mean()
        v_loss = 0.5 * (ret - v.squeeze(-1)).pow(2).mean()
        ent = dist.entropy().mean()
        loss = pi_loss + v_loss * 0.5 - ent * 0.01
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


# -------------------------- Integration helper --------------------------
class HotswapACAdapter:
    """Adapter that lets a team use ACProfilePolicy as the hotswap policy.

    Call `decide_and_update(manager, state, team_agents, team_is_orange)` each step.
    """
    def __init__(self, policy: ACProfilePolicy):
        self.policy = policy

    def decide_and_update(self, manager, state: GameState, team_agents: List[Any], team_is_orange: bool):
        choices = self.policy.act(state, team_agents, team_is_orange)
        for aid, name in choices.items():
            if manager.current_name.get(aid) != name:
                manager.set_profile(aid, state, name)   # <- use setter
        return choices



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
        return self.hotswap.get_rewards(agents, state)
