
import os, csv
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.common_values import ORANGE_TEAM, TICKS_PER_SECOND
from rlgym.rocket_league.done_conditions.timeout_condition import TimeoutCondition
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

from hotswap_hrl import (
    AgentProfile,
    TeamProfilePool,
    HotswapManager,
    ACProfilePolicy,
    ACConfig,
    HotswapACAdapter,
)

# Your local modules
from reward_native_classes import StrikerCompositeReward, DefenderCompositeReward, PositioningCompositeReward
from rlgym.rocket_league.sim import RocketSimEngine
from AdvancedObs import AdvancedObs


# =============================================================================
# PPO (low-level) — one policy per *player profile name*
# =============================================================================

class PPONet(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.v = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.pi(obs), self.v(obs)


@dataclass
class PPOHyper:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    batch_size: int = 2048
    epochs: int = 4


class PPOBuffer:
    def __init__(self, obs_dim: int, size: int):
        self.obs  = np.zeros((size, obs_dim), np.float32)
        self.acts = np.zeros((size,), np.int64)
        self.logp = np.zeros((size,), np.float32)
        self.rew  = np.zeros((size,), np.float32)
        self.val  = np.zeros((size,), np.float32)
        self.done = np.zeros((size,), np.float32)
        self.ptr = 0
        self.max_size = size

    def add(self, obs, act, logp, rew, val, done) -> bool:
        if self.ptr >= self.max_size:
            return False
        i = self.ptr
        self.obs[i]  = obs
        self.acts[i] = act
        self.logp[i] = logp
        self.rew[i]  = rew
        self.val[i]  = val
        self.done[i] = float(done)
        self.ptr += 1
        return True

    def full(self) -> bool:
        return self.ptr >= self.max_size

    def reset(self):
        self.ptr = 0

    def compute_gae(self, gamma: float, lam: float):
        adv = np.zeros_like(self.rew)
        ret = np.zeros_like(self.rew)

        gae = 0.0
        next_value = 0.0
        for t in reversed(range(self.ptr)):
            mask = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * next_value * mask - self.val[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
            ret[t] = adv[t] + self.val[t]
            next_value = self.val[t]

        # normalize adv
        a = adv[:self.ptr]
        adv[:self.ptr] = (a - a.mean()) / (a.std() + 1e-8)
        return adv[:self.ptr], ret[:self.ptr]


class PPOAgent:
    def __init__(self, obs_size: int, n_actions: int, hyper: PPOHyper = PPOHyper(), device: str = "cpu"):
        self.net = PPONet(obs_size, n_actions).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=hyper.lr)
        self.h = hyper
        self.device = device
        self.buffer = PPOBuffer(obs_size, hyper.batch_size)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, v = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(v.item())

    def store(self, obs, act, logp, rew, val, done):
        if not self.buffer.add(obs, act, logp, rew, val, done):
            self.update()
            ok = self.buffer.add(obs, act, logp, rew, val, done)
            if not ok:
                raise RuntimeError("PPOBuffer still full after update/reset.")

    def update(self):
        buf = self.buffer
        adv, ret = buf.compute_gae(self.h.gamma, self.h.lam)

        obs = torch.as_tensor(buf.obs[:buf.ptr], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(buf.acts[:buf.ptr], dtype=torch.int64, device=self.device)
        logp_old = torch.as_tensor(buf.logp[:buf.ptr], dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        for _ in range(self.h.epochs):
            logits, values = self.net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts)

            ratio = (logp - logp_old).exp()
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1.0 - self.h.clip_eps, 1.0 + self.h.clip_eps) * adv_t
            pi_loss = -torch.min(unclipped, clipped).mean()

            v_loss = 0.5 * (ret_t - values.squeeze(-1)).pow(2).mean()
            ent = dist.entropy().mean()

            loss = pi_loss + self.h.vf_coef * v_loss - self.h.ent_coef * ent
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        buf.reset()


# =============================================================================
# Helper funcs (touches)
# =============================================================================

def _safe_touches(car) -> int:
    val = getattr(car, "ball_touches", 0)
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


# =============================================================================
# Two-team reward adapter: AC swaps determine which *player profile* is deployed per car
# =============================================================================

class TwoTeamAssignedHotswapRewardAdapter(RewardFunction):
    """
    - Owns two HotswapManagers (blue/orange), each with a pool restricted by TEAM_SPECS.
    - AC calls set_profile(aid, state, pname) to deploy a player to a specific car.
    - get_rewards() returns per-car rewards computed by that car's currently-deployed player's composite.
    """
    def __init__(self, global_profiles: Dict[str, AgentProfile],
                 team_specs: Dict[str, List[str]],
                 blue_team_name: str,
                 orange_team_name: str):
        self.global_profiles = global_profiles
        self.team_specs = team_specs
        self.blue_team_name = blue_team_name
        self.orange_team_name = orange_team_name

        self.pool_by_team: Dict[int, TeamProfilePool] = {0: TeamProfilePool(), 1: TeamProfilePool()}
        self.mgr_by_team: Dict[int, HotswapManager] = {0: None, 1: None}

        # merged: AgentID -> profile name
        self.current_name: Dict[AgentID, str] = {}

        # initial assignments are pushed by env.reset()
        self._assigned_player: Dict[AgentID, str] = {}

        self._build_pools()

    def _build_pools(self):
        assert self.blue_team_name in self.team_specs, f"Missing TEAM_SPECS for '{self.blue_team_name}'"
        assert self.orange_team_name in self.team_specs, f"Missing TEAM_SPECS for '{self.orange_team_name}'"

        def add_names(team_id: int, names: List[str]):
            pool = TeamProfilePool()
            for n in dict.fromkeys(names):  # preserve order, unique
                if n not in self.global_profiles:
                    raise KeyError(f"TEAM_SPECS references unknown profile '{n}'")
                pool.add(self.global_profiles[n])
            self.pool_by_team[team_id] = pool
            self.mgr_by_team[team_id] = HotswapManager(pool, policy=None)

        add_names(0, list(self.team_specs[self.blue_team_name]))
        add_names(1, list(self.team_specs[self.orange_team_name]))

    def set_assignments(self, assigned_player: Dict[AgentID, str]) -> None:
        self._assigned_player = dict(assigned_player)

    def set_profile(self, aid: AgentID, state: GameState, profile_name: str) -> None:
        team = int(state.cars[aid].team_num)
        self.mgr_by_team[team].set_profile(aid, state, profile_name)
        self.current_name[aid] = profile_name

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.current_name.clear()
        for team_id in (0, 1):
            self.mgr_by_team[team_id].current.clear()
            self.mgr_by_team[team_id].current_name.clear()

        for aid in agents:
            pname = self._assigned_player.get(aid)
            if pname is None:
                team = int(initial_state.cars[aid].team_num)
                pname = next(iter(self.pool_by_team[team]._by_name.keys()))
            self.set_profile(aid, initial_state, pname)

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        out: Dict[AgentID, float] = {}
        for aid in agents:
            team = int(state.cars[aid].team_num)
            comp = self.mgr_by_team[team].current[aid]
            rmap = comp.get_rewards([aid], state, is_terminated, is_truncated, shared_info)
            out[aid] = float(rmap[aid])
        return out


# =============================================================================
# Engine Env Adapter (two teams)
# =============================================================================

def initialize_engine_with_state(engine, initial_state=None, blue_size=1, orange_size=1):
    gs = initial_state if initial_state is not None else engine.create_base_state()
    mutators = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size),
        KickoffMutator(),
    )
    shared = {}
    mutators.apply(gs, shared)
    engine.set_state(gs, shared)
    return engine, gs


class EngineEnvAdapter:
    def __init__(
        self,
        engine,
        action_parser,
        reward_function: TwoTeamAssignedHotswapRewardAdapter,
        ll_obs_builder,
        blue_size=3,
        orange_size=3,
        blue_team_name: str = "",
        orange_team_name: str = "",
        team_specs: Optional[Dict[str, List[str]]] = None,
        global_profiles: Optional[Dict[str, AgentProfile]] = None,
        ac_adapter_blue: Optional[HotswapACAdapter] = None,
        ac_adapter_orange: Optional[HotswapACAdapter] = None,
    ):
        self.engine = engine
        self.action_parser = action_parser
        self.reward_function = reward_function
        self.ll_obs_builder = ll_obs_builder

        self._blue_size = blue_size
        self._orange_size = orange_size

        self.team_specs = team_specs or {}
        self.global_profiles = global_profiles or {}
        self.blue_team_name = blue_team_name
        self.orange_team_name = orange_team_name

        self.ac_blue = ac_adapter_blue
        self.ac_orange = ac_adapter_orange

        self.player_by_agent: Dict[AgentID, str] = {}
        self.role_by_agent: Dict[AgentID, str] = {}

        self._last_touches: Dict[AgentID, int] = {}
        self._touch_buffer = deque(maxlen=getattr(ll_obs_builder, "touch_k", 8))
        self._last_touch = {"aid": None, "team": None, "tick": -1}
        self._team_touch_streak = {0: 0, 1: 0}

    def _shared_info(self):
        return {
            "touch_buffer": list(self._touch_buffer),
            "last_touch": dict(self._last_touch),
            "team_touch_streak": dict(self._team_touch_streak),
            "profile_by_agent": dict(self.player_by_agent),
            "role_by_agent": dict(self.role_by_agent),
        }

    def _build_ll_obs(self, state: GameState):
        obs_map = {}
        shared = self._shared_info()
        for aid in state.cars.keys():
            obs_map[aid] = self.ll_obs_builder._build_obs(aid, state, shared)
        return obs_map

    def _assign_players_for_match(self, state: GameState):
        blue_list = list(self.team_specs[self.blue_team_name])
        orange_list = list(self.team_specs[self.orange_team_name])

        blue_aids = sorted([aid for aid, car in state.cars.items() if int(car.team_num) == 0])
        orange_aids = sorted([aid for aid, car in state.cars.items() if int(car.team_num) == 1])

        self.player_by_agent.clear()
        self.role_by_agent.clear()

        for i, aid in enumerate(blue_aids):
            pname = blue_list[i % len(blue_list)]
            self.player_by_agent[aid] = pname
            self.role_by_agent[aid] = self.global_profiles[pname].role

        for i, aid in enumerate(orange_aids):
            pname = orange_list[i % len(orange_list)]
            self.player_by_agent[aid] = pname
            self.role_by_agent[aid] = self.global_profiles[pname].role

    def _update_touch_tracking(self, state: GameState):
        for aid, car in state.cars.items():
            prev = self._last_touches.get(aid, 0)
            cur = _safe_touches(car)
            if cur > prev:
                prev_team = self._last_touch.get("team", None)
                t_team = int(car.team_num)
                if prev_team is None or prev_team == t_team:
                    self._team_touch_streak[t_team] = self._team_touch_streak.get(t_team, 0) + 1
                else:
                    self._team_touch_streak[t_team] = 1
                    self._team_touch_streak[1 - t_team] = 0

                sign = +1.0 if t_team != ORANGE_TEAM else -1.0
                self._touch_buffer.append(sign)
                self._last_touch = {"aid": aid, "team": t_team, "tick": int(state.tick_count)}
            self._last_touches[aid] = cur

    def _refresh_deployments_from_reward_adapter(self):
        self.player_by_agent = dict(self.reward_function.current_name)
        self.role_by_agent = {aid: self.global_profiles[p].role for aid, p in self.player_by_agent.items()}

    def reset(self, initial_state=None):
        _, state = initialize_engine_with_state(
            self.engine,
            initial_state=initial_state,
            blue_size=self._blue_size,
            orange_size=self._orange_size,
        )
        self.state = state

        self._touch_buffer.clear()
        self._team_touch_streak = {0: 0, 1: 0}
        self._last_touch = {"aid": None, "team": None, "tick": -1}
        self._last_touches = {aid: _safe_touches(c) for aid, c in state.cars.items()}

        self._assign_players_for_match(state)
        self.reward_function.set_assignments(self.player_by_agent)
        agent_ids = list(state.cars.keys())
        self.reward_function.reset(agent_ids, state, self._shared_info())
        self._refresh_deployments_from_reward_adapter()

        ll_obs = self._build_ll_obs(state)
        first_obs = ll_obs[agent_ids[0]]
        info = {
            "ll_obs": ll_obs,
            "profile_by_agent": dict(self.player_by_agent),
            "role_by_agent": dict(self.role_by_agent),
        }
        return first_obs, info

    def step(self, actions_dict: Dict[AgentID, np.ndarray], shared_info: Optional[Dict[str, Any]] = None):
        prev_state = self.engine.state
        controls_map = self.action_parser.parse_actions(actions_dict, prev_state, shared_info or {})
        state = self.engine.step(controls_map, shared_info or {})
        self.state = state

        agent_ids = list(state.cars.keys())

        if self.ac_blue is not None:
            blue_aids = [aid for aid in agent_ids if int(state.cars[aid].team_num) == 0]
            if blue_aids:
                self.ac_blue.decide_and_update(self.reward_function, state, blue_aids)

        if self.ac_orange is not None:
            orange_aids = [aid for aid in agent_ids if int(state.cars[aid].team_num) == 1]
            if orange_aids:
                self.ac_orange.decide_and_update(self.reward_function, state, orange_aids)

        self._refresh_deployments_from_reward_adapter()
        self._update_touch_tracking(state)

        is_term = {aid: False for aid in agent_ids}
        is_trunc = {aid: False for aid in agent_ids}
        rmap = self.reward_function.get_rewards(agent_ids, state, is_term, is_trunc, self._shared_info())

        ll_obs = self._build_ll_obs(state)
        done = bool(getattr(state, "goal_scored", False))

        first_obs = ll_obs[agent_ids[0]]
        info = {
            "ll_obs": ll_obs,
            "rewards": rmap,
            "profile_by_agent": dict(self.player_by_agent),
            "role_by_agent": dict(self.role_by_agent),
            "touch_buffer": list(self._touch_buffer),
            "last_touch": dict(self._last_touch),
            "touch_streaks": dict(self._team_touch_streak),
        }
        reward_scalar = float(sum(rmap.values()))
        return first_obs, reward_scalar, done, info


# =============================================================================
# MatchRunner (two teams): uses deployed profiles to pick actions + store per-profile rollouts
# =============================================================================

class MatchRunner:
    def __init__(self, env: EngineEnvAdapter, ppo_players: Dict[str, PPOAgent], kickoffs: Optional[KickoffMutator] = None):
        self.env = env
        self.ppo_players = ppo_players
        self.kickoffs = kickoffs or KickoffMutator()
        self.timer = TimeoutCondition(timeout_seconds=300.0)
        self.scores = {"BLUE": 0, "ORANGE": 0}

        self._prev_obs: Dict[AgentID, np.ndarray] = {}
        self._prev_logp: Dict[AgentID, float] = {}
        self._prev_val: Dict[AgentID, float] = {}
        self._prev_profile: Dict[AgentID, str] = {}

    def _score_and_reset_kickoff(self, state: GameState):
        ball_y = float(state.ball.position[1])
        if ball_y > 0:
            self.scores["BLUE"] += 1
        else:
            self.scores["ORANGE"] += 1
        gs = state
        MutatorSequence(self.kickoffs).apply(gs, shared_info={})
        self.env.engine.set_state(gs, shared_info={})
        if hasattr(self.env, "_touch_buffer"):
            self.env._touch_buffer.clear()

    def run(self, log_path: str = "out/two_team_match_log.csv"):
        _, info = self.env.reset()
        agent_ids = list(self.env.state.cars.keys())
        self.timer.reset(agent_ids, self.env.state, shared_info={})

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = None
            fieldnames = None

            while True:
                actions: Dict[AgentID, np.ndarray] = {}
                self._prev_obs.clear()
                self._prev_logp.clear()
                self._prev_val.clear()
                self._prev_profile = dict(info["profile_by_agent"])

                for aid in agent_ids:
                    pname = info["profile_by_agent"][aid]
                    a, logp, v = self.ppo_players[pname].act(info["ll_obs"][aid])
                    actions[aid] = np.array([a], dtype=np.int64)
                    self._prev_obs[aid] = info["ll_obs"][aid]
                    self._prev_logp[aid] = logp
                    self._prev_val[aid] = v

                _, _, done, info = self.env.step(actions)

                rmap = info["rewards"]
                for aid in agent_ids:
                    pname_prev = self._prev_profile[aid]
                    self.ppo_players[pname_prev].store(
                        self._prev_obs[aid],
                        int(actions[aid][0]),
                        float(self._prev_logp[aid]),
                        float(rmap[aid]),
                        float(self._prev_val[aid]),
                        bool(done),
                    )

                for agent in self.ppo_players.values():
                    if agent.buffer.full():
                        agent.update()

                # lightweight logging (tick + deployed profiles)
                row = {"tick": int(self.env.state.tick_count), "time_s": float(self.env.state.tick_count) / float(TICKS_PER_SECOND)}
                for aid in agent_ids:
                    row[f"{aid}_team"] = int(self.env.state.cars[aid].team_num)
                    row[f"{aid}_profile"] = info["profile_by_agent"].get(aid, "")
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)

                dones = self.timer.is_done(agent_ids, self.env.state, shared_info={})
                if any(dones.values()):
                    break

                if getattr(self.env.state, "goal_scored", False):
                    self._score_and_reset_kickoff(self.env.state)

        return {"scores": dict(self.scores), "log_path": log_path}


# =============================================================================
# Example wiring
# =============================================================================

def build_globals():
    S_base = StrikerCompositeReward().get_weights()
    P_base = PositioningCompositeReward().get_weights()
    D_base = DefenderCompositeReward().get_weights()

    def tweak(base: dict, **kwargs):
        out = dict(base)
        out.update(kwargs)
        return out

    GLOBAL_PROFILES = {
        "s0": AgentProfile("s0", "striker", dict(S_base)),
        "s1": AgentProfile("s1", "striker", dict(S_base)),
        "s2": AgentProfile("s2", "striker", tweak(S_base, dist_to_ball=0.0, car_speed=0.0, boost_remaining=1.0e-5, supersonic_bonus=0.0, ball_dist_to_goal=2.0e-5, face_ball=1.0e-5, shot_alignment=6.0e-5, ball_hit=3.0e-2, touch=2.5e-2)),
        "s3": AgentProfile("s3", "striker", tweak(S_base, dist_to_ball=1.5e-5, car_speed=2.0e-5, boost_remaining=0.0, supersonic_bonus=1.5e-5, ball_dist_to_goal=2.0e-5, face_ball=2.0e-5, shot_alignment=7.0e-5, ball_hit=2.0e-2, touch=1.5e-2, goal=12.0)),
        "s4": AgentProfile("s4", "striker", tweak(S_base, dist_to_ball=1.0e-5, car_speed=0.0, boost_remaining=1.0e-5, supersonic_bonus=0.0, ball_dist_to_goal=1.5e-5, face_ball=2.0e-5, shot_alignment=6.0e-5, ball_hit=3.0e-2, touch=1.0e-2)),
        "s5": AgentProfile("s5", "striker", tweak(S_base, dist_to_ball=2.0e-5, car_speed=1.0e-5, boost_remaining=0.0, supersonic_bonus=0.0, ball_dist_to_goal=2.0e-5, face_ball=2.0e-5, shot_alignment=7.0e-5, ball_hit=4.0e-2, touch=2.0e-2, goal=12.0)),
        "p1": AgentProfile("p1", "positioning", dict(P_base)),
        "p2": AgentProfile("p2", "positioning", tweak(P_base, car_speed=0.0, boost_remaining=2.0e-5, mean_dist_to_teammates=2.0e-5, mean_dist_to_opponents=0.0, centerline_proximity=1.0e-5, face_ball=7.5e-5, face_goal=2.5e-5, shot_alignment=5.0e-5, block_alignment=7.5e-5, def_hit=2.0e-2)),
        "p3": AgentProfile("p3", "positioning", tweak(P_base, car_speed=0.5e-5, boost_remaining=2.0e-5, mean_dist_to_teammates=1.0e-5, mean_dist_to_opponents=1.0e-5, centerline_proximity=0.0, face_ball=5.0e-5, face_goal=7.5e-5, shot_alignment=7.5e-5, block_alignment=5.0e-5, def_hit=2.0e-2)),
        "d1": AgentProfile("d1", "defender", dict(D_base)),
        "d2": AgentProfile("d2", "defender", tweak(D_base, boost_remaining=0.5e-5, dist_to_ball=0.0, centerline_proximity=0.5e-5, home_goal_proximity=2.0e-5, face_ball=5.0e-5, block_alignment=7.5e-5, behind_other_players=7.5e-5, def_hit=5.0e-2, touch=0.0)),
        "d3": AgentProfile("d3", "defender", tweak(D_base, boost_remaining=1.0e-5, dist_to_ball=0.0, centerline_proximity=1.0e-5, home_goal_proximity=2.0e-5, face_ball=5.0e-5, block_alignment=7.5e-5, behind_other_players=3.0e-5, def_hit=5.0e-2, touch=2.0e-2, goal=15.0)),
        "d4": AgentProfile("d4", "defender", tweak(D_base, boost_remaining=1.0e-5, dist_to_ball=1.0e-5, centerline_proximity=5.0e-5, home_goal_proximity=5.0e-5, face_ball=5.0e-5, block_alignment=3.0e-5, behind_other_players=2.0e-5, def_hit=7.5e-2, touch=3.0e-2, goal=15.0)),
    }

    TEAM_SPECS = {
        "team1_striker_balance": ["s1","s2","s3","p1","p2","d1","d2"],
        "team2_striker_heavy":   ["s1","s2","s3","s4","s5","p1","p2"],
        "team3_balance":         ["s4","s5","p1","p2","p3","d3","d4"],
        "team4_pos_striker":     ["s1","s2","s3","p1","p2","p3","d1"],
        "team5_balance2":        ["s4","s5","p1","p2","p3","d1","d2"],
        "team6_def_striker":     ["s1","s2","s3","d1","d2","d3","d4"],
        "team7_def_balance":     ["s4","s5","p3","d2","d3","d4"],
        "team8_baseline":        ["s0"],
    }
    return GLOBAL_PROFILES, TEAM_SPECS


def main_smoke_test():
    engine = RocketSimEngine(rlbot_delay=True)
    action_parser = RepeatAction(LookupTableAction(), repeats=8)
    ll_obs_builder = AdvancedObs()

    GLOBAL_PROFILES, TEAM_SPECS = build_globals()
    blue_team_name = "team1_striker_balance"
    orange_team_name = "team6_def_striker"

    reward_adapter = TwoTeamAssignedHotswapRewardAdapter(
        global_profiles=GLOBAL_PROFILES,
        team_specs=TEAM_SPECS,
        blue_team_name=blue_team_name,
        orange_team_name=orange_team_name,
    )

    blue_profiles = list(dict.fromkeys(TEAM_SPECS[blue_team_name]))
    orange_profiles = list(dict.fromkeys(TEAM_SPECS[orange_team_name]))

    blue_policy = ACProfilePolicy(blue_profiles, cfg=ACConfig())
    orange_policy = ACProfilePolicy(orange_profiles, cfg=ACConfig())

    ac_blue = HotswapACAdapter(blue_policy)
    ac_orange = HotswapACAdapter(orange_policy)

    env = EngineEnvAdapter(
        engine=engine,
        action_parser=action_parser,
        reward_function=reward_adapter,
        ll_obs_builder=ll_obs_builder,
        blue_size=3,
        orange_size=3,
        blue_team_name=blue_team_name,
        orange_team_name=orange_team_name,
        team_specs=TEAM_SPECS,
        global_profiles=GLOBAL_PROFILES,
        ac_adapter_blue=ac_blue,
        ac_adapter_orange=ac_orange,
    )

    _, info = env.reset()
    obs_dim = next(iter(info["ll_obs"].values())).shape[0]
    N_ACTIONS = 90
    ppo_players = {name: PPOAgent(obs_dim, N_ACTIONS) for name in GLOBAL_PROFILES.keys()}

    runner = MatchRunner(env, ppo_players, kickoffs=KickoffMutator())
    result = runner.run(log_path="out/two_team_match_log.csv")
    print("finished:", result)


if __name__ == "__main__":
    main_smoke_test()
