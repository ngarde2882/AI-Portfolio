
import os, csv
import traceback
from collections import deque
from dataclasses import dataclass

# =============================================================================
# Debug instrumentation
# =============================================================================

@dataclass
class DebugConfig:
    enabled: bool = False
    log_path: str = "out/debug_env.log"
    # Print warnings when agents appear "stuck" (low speed + repeated same action)
    warn_stuck_agents: bool = True
    stuck_speed_uu_per_s: float = 25.0          # below this, treat as not moving
    stuck_action_window: int = 64               # number of decisions to consider "stuck"
    stuck_same_action_frac: float = 0.95        # if >= this fraction are same action -> stuck

    # Validate observation vectors (NaNs/inf/near-constant)
    validate_obs: bool = True
    obs_nan_inf_warn: bool = True
    obs_low_variance_warn: bool = True
    obs_low_variance_eps: float = 1e-8

    # Reward sanity checks
    reward_nan_inf_warn: bool = True

    # Print cadence
    print_every_ticks: int = 480                # ~4 seconds at 120 Hz

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.common_values import ORANGE_TEAM, TICKS_PER_SECOND, BLUE_GOAL_BACK, ORANGE_GOAL_BACK
from rlgym.rocket_league.done_conditions.timeout_condition import TimeoutCondition
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
from rlgym.rocket_league.sim import RocketSimEngine

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
from AdvancedObs import AdvancedObs


# =============================================================================
# High-level agent ball-half reward helper
# =============================================================================

_HL_BALL_HALF_BASE = 1e-4   # per-tick reward for ball being in correct half
_HL_BALL_HALF_NEAR = 1e-2   # elevated reward when ball is within GOAL_RANGE of a net
_HL_GOAL_RANGE     = 2000.0 # UU radius around each goal center that triggers the near-goal scale
_HL_TOUCH_REWARD   = 0.02   # one-shot HL reward when any player on the team touches the ball

_ORANGE_GOAL_POS = np.asarray(ORANGE_GOAL_BACK, dtype=np.float32)
_BLUE_GOAL_POS   = np.asarray(BLUE_GOAL_BACK,   dtype=np.float32)

def _hl_ball_half_reward(state: "GameState", team: int) -> float:
    """Per-tick HL reward for ball position relative to field halves.

    team=0 (blue) attacks +Y (orange goal); team=1 (orange) attacks -Y (blue goal).
    Returns +_HL_BALL_HALF_BASE  if ball is on the attacking half,
            -_HL_BALL_HALF_BASE  if ball is on the defending half,
    scaled to ±_HL_BALL_HALF_NEAR when ball is within _HL_GOAL_RANGE of the relevant net.
    """
    ball_pos = np.asarray(state.ball.position, dtype=np.float32)
    ball_y   = float(ball_pos[1])
    attacking_half = (team == 0 and ball_y > 0) or (team == 1 and ball_y < 0)
    if attacking_half:
        atk_goal = _ORANGE_GOAL_POS if team == 0 else _BLUE_GOAL_POS
        if float(np.linalg.norm(ball_pos - atk_goal)) < _HL_GOAL_RANGE:
            return _HL_BALL_HALF_NEAR
        return _HL_BALL_HALF_BASE
    else:
        def_goal = _BLUE_GOAL_POS if team == 0 else _ORANGE_GOAL_POS
        if float(np.linalg.norm(ball_pos - def_goal)) < _HL_GOAL_RANGE:
            return -_HL_BALL_HALF_NEAR
        return -_HL_BALL_HALF_BASE


# =============================================================================
# PPO (low-level) — one policy per *player profile name*
# =============================================================================

class PPONet(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, trunk_hidden: int = 512, head_hidden: int = 256):
        super().__init__()
        # Shared trunk: obs → 512 → 512 → 512 (3 layers for richer feature extraction)
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )
        # Separate heads: 512 → 256 → output
        self.pi = nn.Sequential(
            nn.Linear(trunk_hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, n_actions),
        )
        self.v = nn.Sequential(
            nn.Linear(trunk_hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, obs):
        z = self.trunk(obs)
        return self.pi(z), self.v(z)


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
        # Guard against overflow: caller should flush/update before adding more.
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
        # If the buffer is full, run an update before storing the next transition.
        if not self.buffer.add(obs, act, logp, rew, val, done):
            self.update()
            # After update(), buffer is reset, so this should succeed.
            ok = self.buffer.add(obs, act, logp, rew, val, done)
            if not ok:
                raise RuntimeError("PPOBuffer is still full after update/reset; check batch_size configuration.")

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

        self.last_breakdown_by_aid: Dict[AgentID, Dict[str, float]] = {}

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
        self.last_breakdown_by_aid.clear()
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

            bd = {}
            try:
                bd = comp.get_last_breakdown(aid)
            except Exception:
                bd = {}
            self.last_breakdown_by_aid[aid] = bd
        return out


# =============================================================================
# Engine Env Adapter (two teams)
# =============================================================================

def initialize_engine_with_state(engine, initial_state=None, blue_size=1, orange_size=1):
    gs = initial_state if initial_state is not None else engine.create_base_state()
    shared = {}
    if len(gs.cars) == 0:
        # Fresh state — let mutators add cars and set kickoff positions.
        mutators = MutatorSequence(
            FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size),
            KickoffMutator(),
        )
        mutators.apply(gs, shared)
    # else: state already has cars (e.g. cluster replay) — load exact positions as-is.
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
        debug: Optional[DebugConfig] = None,
        cost_of_living_per_tick: float = 0.0,
        reward_scale: float = 1.0,
        goal_anneal_cap: float = 400.0,
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

        self.debug = debug or DebugConfig(enabled=False)
        self.cost_of_living_per_tick = float(cost_of_living_per_tick)
        self.reward_scale = float(reward_scale)
        # Goal component of reward is capped at ±goal_anneal_cap after scaling so that
        # sparse terminal events don't overwhelm dense shaping at high annealing scales.
        # At scale=1.0 (end of annealing) goal*1=10 < cap, so the cap is transparent.
        self.goal_anneal_cap = float(goal_anneal_cap)

        # debug logging to file (avoid interfering with tqdm)
        self._dbg_fh = None
        if self.debug.enabled and getattr(self.debug, "log_path", None):
            os.makedirs(os.path.dirname(self.debug.log_path) or ".", exist_ok=True)
            self._dbg_fh = open(self.debug.log_path, "a", buffering=1)
        self._dbg_last_print_bucket = None
        self._dbg_action_hist = { }

        # action history buffer for observations (always-on)
        self._action_hist = {}
        self._action_hist_k = int(getattr(ll_obs_builder, 'action_hist_k', 8))

        self.player_by_agent: Dict[AgentID, str] = {}

        self._last_touches: Dict[AgentID, int] = {}
        self._touch_buffer = deque(maxlen=getattr(ll_obs_builder, "touch_k", 8))
        self._last_touch = {"aid": None, "team": None, "tick": -1}
        self._team_touch_streak = {0: 0, 1: 0}

    def _shared_info(self):
        return {
            "touch_buffer": list(self._touch_buffer),
            "action_hist_by_agent": {aid: list(hist) for aid, hist in self._action_hist.items()},
            "last_touch": dict(self._last_touch),
            "team_touch_streak": dict(self._team_touch_streak),
            "profile_by_agent": dict(self.player_by_agent),
            "profile_names": list(self.global_profiles.keys()),
        }

    def __del__(self):
        try:
            if getattr(self, "_dbg_fh", None) is not None:
                self._dbg_fh.close()
        except Exception:
            pass


    # ---------------------------- debug helpers ----------------------------

    def _dbg_log(self, msg: str) -> None:
        if not (getattr(self, "debug", None) and self.debug.enabled):
            return
        try:
            line = str(msg).rstrip()
            if self._dbg_fh is not None:
                self._dbg_fh.write(line + "\n")
        except Exception:
            pass

    def _dbg_note_actions(self, actions_idx: Dict[AgentID, int]) -> None:
        if not getattr(self, "debug", DebugConfig()).enabled:
            return
        for aid, a in actions_idx.items():
            h = self._dbg_action_hist.setdefault(aid, deque(maxlen=max(8, int(self.debug.stuck_action_window))))
            h.append(int(a))

    def _dbg_check_obs(self, aid: AgentID, obs: np.ndarray) -> None:
        dbg = getattr(self, "debug", None)
        if not (dbg and dbg.enabled and dbg.validate_obs):
            return
        if obs is None:
            return
        arr = np.asarray(obs, dtype=np.float32)
        if dbg.obs_nan_inf_warn and (not np.isfinite(arr).all()):
            n_bad = int(np.size(arr) - np.isfinite(arr).sum())
            self._dbg_log(f"[DBG][obs] aid={aid} nonfinite={n_bad}/{arr.size}")
        if dbg.obs_low_variance_warn:
            v = float(np.var(arr))
            if v <= float(dbg.obs_low_variance_eps):
                mn = float(np.min(arr)); mx = float(np.max(arr))
                self._dbg_log(f"[DBG][obs] aid={aid} low-variance var={v:.3e} min={mn:.3e} max={mx:.3e}")

    def _dbg_warn_stuck(self, state: GameState, actions_idx: Dict[AgentID, int]) -> None:
        dbg = getattr(self, "debug", None)
        if not (dbg and dbg.enabled and dbg.warn_stuck_agents):
            return

        # throttle printing by tick buckets
        tick = int(getattr(state, "tick_count", 0))
        bucket = tick // max(1, int(dbg.print_every_ticks))
        if self._dbg_last_print_bucket is not None and bucket == self._dbg_last_print_bucket:
            return
        self._dbg_last_print_bucket = bucket

        for aid, car in state.cars.items():
            phys = car.physics
            speed = float(np.linalg.norm(np.asarray(phys.linear_velocity, dtype=np.float32)))
            if speed > float(dbg.stuck_speed_uu_per_s):
                continue

            h = self._dbg_action_hist.get(aid)
            if not h or len(h) < max(8, int(dbg.stuck_action_window) // 2):
                continue

            # fraction of most-common action in window
            vals, counts = np.unique(np.asarray(h, dtype=np.int64), return_counts=True)
            frac = float(np.max(counts)) / float(len(h))
            if frac >= float(dbg.stuck_same_action_frac):
                a_mode = int(vals[int(np.argmax(counts))])
                pname = self.player_by_agent.get(aid, "?")
                self._dbg_log(f"[DBG][stuck] tick={tick} aid={aid} profile={pname} speed={speed:.1f} action_mode={a_mode} frac={frac:.2f} window={len(h)}")

    def _build_ll_obs(self, state: GameState):
        obs_map = {}
        shared = self._shared_info()
        for aid in state.cars.keys():
            obs = self.ll_obs_builder._build_obs(aid, state, shared)
            obs_map[aid] = obs
            self._dbg_check_obs(aid, obs)
        return obs_map

    def _assign_players_for_match(self, state: GameState):
        blue_list = list(self.team_specs[self.blue_team_name])
        orange_list = list(self.team_specs[self.orange_team_name])

        blue_aids = sorted([aid for aid, car in state.cars.items() if int(car.team_num) == 0])
        orange_aids = sorted([aid for aid, car in state.cars.items() if int(car.team_num) == 1])

        self.player_by_agent.clear()

        for i, aid in enumerate(blue_aids):
            pname = blue_list[i % len(blue_list)]
            self.player_by_agent[aid] = pname

        for i, aid in enumerate(orange_aids):
            pname = orange_list[i % len(orange_list)]
            self.player_by_agent[aid] = pname

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


        # reset per-car action history
        self._action_hist = {aid: deque([0.0]*self._action_hist_k, maxlen=self._action_hist_k) for aid in state.cars.keys()}

        self._assign_players_for_match(state)
        self.reward_function.set_assignments(self.player_by_agent)
        agent_ids = list(state.cars.keys())
        self.reward_function.reset(agent_ids, state, self._shared_info())
        self._refresh_deployments_from_reward_adapter()

        # Let AC managers choose initial deployments at tick=0 (before first obs)
        agent_ids = list(state.cars.keys())
        try:
            if self.ac_blue is not None:
                blue_aids = [aid for aid in agent_ids if int(state.cars[aid].team_num) == 0]
                if blue_aids:
                    self.ac_blue.decide_and_update(self.reward_function, state, blue_aids)
            if self.ac_orange is not None:
                orange_aids = [aid for aid in agent_ids if int(state.cars[aid].team_num) == 1]
                if orange_aids:
                    self.ac_orange.decide_and_update(self.reward_function, state, orange_aids)
        except Exception:
            self._dbg_log("[ERR][ac_init] exception during tick=0 deployment decision")
            self._dbg_log(traceback.format_exc().rstrip())
            raise

        self._refresh_deployments_from_reward_adapter()

        ll_obs = self._build_ll_obs(state)
        first_obs = ll_obs[agent_ids[0]]
        info = {
            "ll_obs": ll_obs,
            "profile_by_agent": dict(self.player_by_agent),
            "profile_names": list(self.global_profiles.keys()),
        }
        return first_obs, info

    def step(self, actions_dict: Dict[AgentID, np.ndarray], shared_info: Optional[Dict[str, Any]] = None):
        prev_state = self.engine.state
        controls_map = self.action_parser.parse_actions(actions_dict, prev_state, shared_info or {})
        state = self.engine.step(controls_map, shared_info or {})
        self.state = state

        agent_ids = list(state.cars.keys())

        # Debug: record chosen low-level actions (if provided)
        if shared_info and isinstance(shared_info, dict) and 'actions_idx' in shared_info:
            try:
                self._dbg_note_actions(shared_info['actions_idx'])
            except Exception:
                pass

            try:
                for aid, aidx in shared_info['actions_idx'].items():
                    hist = self._action_hist.setdefault(aid, deque(maxlen=self._action_hist_k))
                    # normalize discrete action index to [0,1] for obs compactness
                    hist.append(float(int(aidx)) / 89.0)
            except Exception:
                pass
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

        # Debug: warn on apparent stuck policies
        try:
            self._dbg_warn_stuck(state, (shared_info or {}).get('actions_idx', {}))
        except Exception:
            pass

        is_term = {aid: False for aid in agent_ids}
        is_trunc = {aid: False for aid in agent_ids}
        rmap = self.reward_function.get_rewards(agent_ids, state, is_term, is_trunc, self._shared_info())
        # Pull per-agent breakdown from reward function (weighted, pre-COL, pre-scale)
        breakdown_by_aid = {}
        try:
            breakdown_by_aid = {aid: dict(self.reward_function.last_breakdown_by_aid.get(aid, {})) for aid in agent_ids}
        except Exception:
            breakdown_by_aid = {aid: {} for aid in agent_ids}

        # Flat per-tick cost of living to discourage idle policies (set via cost_of_living_per_tick)
        if self.cost_of_living_per_tick != 0.0:
            col = float(self.cost_of_living_per_tick)
            for _aid in agent_ids:
                rmap[_aid] = float(rmap.get(_aid, 0.0)) - col
                breakdown_by_aid[_aid]["cost_of_living"] = breakdown_by_aid[_aid].get("cost_of_living", 0.0) - col

        # Reward annealing scale (driven by gauntlet; 1.0 means no scaling).
        # Dense shaping components are scaled freely; the 'goal' component is
        # additionally capped at ±goal_anneal_cap so terminal events don't
        # dominate the value function before agents can reliably reach the ball.
        sc = float(self.reward_scale)
        cap = float(self.goal_anneal_cap)
        if sc != 1.0 or True:  # always recompute total from breakdown to stay consistent
            for _aid in agent_ids:
                bd = breakdown_by_aid.get(_aid, {})
                new_total = 0.0
                for k in list(bd.keys()):
                    raw = float(bd[k])
                    scaled = raw * sc
                    if k == "goal":
                        # Clamp goal contribution so it can't exceed ±cap during annealing.
                        # When scale=1 and goal=10, 10 < 400, so the cap is invisible.
                        scaled = float(np.clip(scaled, -cap, cap))
                    bd[k] = scaled
                    new_total += scaled
                breakdown_by_aid[_aid] = bd
                rmap[_aid] = new_total

        ll_obs = self._build_ll_obs(state)
        done = bool(state.goal_scored)

        first_obs = ll_obs[agent_ids[0]]
        info = {
            "ll_obs": ll_obs,
            "rewards": rmap,
            "profile_by_agent": dict(self.player_by_agent),
            "profile_names": list(self.global_profiles.keys()),
            "touch_buffer": list(self._touch_buffer),
            "last_touch": dict(self._last_touch),
            "touch_streaks": dict(self._team_touch_streak),
            "reward_breakdown": breakdown_by_aid,
            "reward_scale": float(self.reward_scale),
        }
        reward_scalar = float(sum(rmap.values()))
        return first_obs, reward_scalar, done, info


# =============================================================================
# Logging
# =============================================================================

class LowLevelLogger:
    """Append-only low-level match summaries across training."""
    def __init__(self, path: str, profile_names: List[str]):
        self.path = path
        self.profile_names = list(profile_names)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

        self._f = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self._fieldnames())
        if is_new:
            self._writer.writeheader()

    def _fieldnames(self) -> List[str]:
        cols = [
            "match_id",
            "blue_team",
            "orange_team",
            "blue_score",
            "orange_score",
            "winner",
            "goal_diff",
            "blue_switches",
            "orange_switches",
        ]
        # One column per profile: tuple string (blue_frac, orange_frac)
        cols += [f"contrib_{name}" for name in self.profile_names]
        return cols

    def log_match(
        self,
        match_id: int,
        blue_team: str,
        orange_team: str,
        blue_score: int,
        orange_score: int,
        blue_switches: int,
        orange_switches: int,
        contrib_blue: Dict[str, float],
        contrib_orange: Dict[str, float],
    ) -> None:
        if blue_score > orange_score:
            winner = "BLUE"
        elif orange_score > blue_score:
            winner = "ORANGE"
        else:
            winner = "TIE"

        row: Dict[str, Any] = {
            "match_id": int(match_id),
            "blue_team": str(blue_team),
            "orange_team": str(orange_team),
            "blue_score": int(blue_score),
            "orange_score": int(orange_score),
            "winner": winner,
            "goal_diff": int(blue_score - orange_score),
            "blue_switches": int(blue_switches),
            "orange_switches": int(orange_switches),
        }

        for name in self.profile_names:
            b = float(contrib_blue.get(name, 0.0))
            o = float(contrib_orange.get(name, 0.0))
            row[f"contrib_{name}"] = f"({b:.6f},{o:.6f})"

        self._writer.writerow(row)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# Lookup table for decoding discrete action -> control vector (8 floats/ints)
# Lookup table for decoding discrete action -> control vector (8 floats/ints)
try:
    _LOOKUP_TABLE = LookupTableAction.make_lookup_table()
except Exception:
    # fallback: instantiate and read the internal table if exposed
    _tmp = LookupTableAction()
    _LOOKUP_TABLE = getattr(_tmp, "_lookup_table", None)
    if _LOOKUP_TABLE is None:
        raise RuntimeError("Could not obtain lookup table for action decoding.")


class HighLevelMatchLogger:
    """Per-match high-detail logs for visualization/debug."""

    def __init__(self, out_dir: str, match_id: int, blue_team: str, orange_team: str, sample_every_ticks: int = 16):
        self.match_id = int(match_id)
        self.blue_team = str(blue_team)
        self.orange_team = str(orange_team)
        self.sample_every_ticks = int(sample_every_ticks)

        os.makedirs(out_dir, exist_ok=True)
        fname = f"{self.match_id}){self.blue_team}__v__{self.orange_team}.csv"
        self.path = os.path.join(out_dir, fname)

        self._f = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self._fieldnames())
        self._writer.writeheader()

        # per-agent action aggregation since last sample
        self._agg: Dict[AgentID, Dict[str, Any]] = {}
        self._prev_jump: Dict[AgentID, int] = {}
        self._bucket = None  # tick bucket of last written sample

    def _fieldnames(self) -> List[str]:
        cols = [
            "match_id",
            "tick",
            "time_s",
            "blue_score",
            "orange_score",
            "ball_px","ball_py","ball_pz",
            "ball_vx","ball_vy","ball_vz",
            "ball_avx","ball_avy","ball_avz",
        ]
        # Slot-based columns for stable 3v3 layout
        for side in ("blue", "orange"):
            for i in range(3):
                pfx = f"{side}-{i}"
                cols += [
                    f"{pfx}_aid",
                    f"{pfx}_profile",
                    f"{pfx}_px", f"{pfx}_py", f"{pfx}_pz",
                    f"{pfx}_vx", f"{pfx}_vy", f"{pfx}_vz",
                    f"{pfx}_fx", f"{pfx}_fy", f"{pfx}_fz",
                    f"{pfx}_ux", f"{pfx}_uy", f"{pfx}_uz",
                    f"{pfx}_avx", f"{pfx}_avy", f"{pfx}_avz",
                    f"{pfx}_boost",
                    f"{pfx}_jump_presses",
                    f"{pfx}_double_jump",
                    f"{pfx}_handbrake",
                    f"{pfx}_action_idx_last",
                    f"{pfx}_switches_window",
                    f"{pfx}_demoed",
                    f"{pfx}_demo_timer",
                ]
        return cols

    def _get_slots(self, state: GameState) -> Dict[str, List[AgentID]]:
        blue = sorted([aid for aid, car in state.cars.items() if int(car.team_num) == 0], key=lambda x: str(x))
        orange = sorted([aid for aid, car in state.cars.items() if int(car.team_num) == 1], key=lambda x: str(x))
        # pad to length 3 with None
        while len(blue) < 3: blue.append(None)
        while len(orange) < 3: orange.append(None)
        return {"blue": blue[:3], "orange": orange[:3]}

    def note_switches(self, switches: Dict[AgentID, int]) -> None:
        """Increment per-agent switch counters for the current sampling window."""
        for aid, k in switches.items():
            if aid is None:
                continue
            a = self._agg.setdefault(aid, {})
            a["switches_window"] = int(a.get("switches_window", 0)) + int(k)

    def observe_actions(self, actions_idx: Dict[AgentID, int], state: GameState) -> None:
        """Aggregate action events for the current sampling window."""
        for aid, idx in actions_idx.items():
            if aid is None:
                continue
            idx_i = int(idx)
            ctrl = _LOOKUP_TABLE[idx_i]  # [thr, steer/yaw, pitch, yaw, roll, jump, boost, handbrake]
            jump = int(ctrl[5] > 0.5)
            boost = int(ctrl[6] > 0.5)
            handbrake = int(ctrl[7] > 0.5)

            car = state.cars.get(aid, None)
            on_ground = True
            demoed = 0
            demo_timer = 0.0
            if car is not None:
                w = car.wheels_with_contact
                if not (isinstance(w, (list, tuple)) and len(w) == 4):
                    raise TypeError(f"car.wheels_with_contact malformed: {w!r}")
                on_ground = bool(car.on_ground)
                demoed = int(car.is_demoed)
                demo_timer = float(car.demo_respawn_timer)


            a = self._agg.setdefault(aid, {})
            a["boost_any"] = int(a.get("boost_any", 0) or boost)
            a["handbrake_any"] = int(a.get("handbrake_any", 0) or handbrake)
            a["action_idx_last"] = idx_i
            a["demoed"] = int(a.get("demoed", 0) or demoed)
            a["demo_timer"] = max(float(a.get("demo_timer", 0.0)), demo_timer)

            prev_j = int(self._prev_jump.get(aid, 0))
            if jump == 1 and prev_j == 0:
                # rising edge -> a "jump press"
                a["jump_presses"] = int(a.get("jump_presses", 0)) + 1
                # crude double-jump detection: 2nd press while not on ground
                if int(a["jump_presses"]) >= 2 and (not on_ground):
                    a["double_jump"] = 1
            self._prev_jump[aid] = jump

    def maybe_sample(self, state: GameState, profile_by_agent: Dict[AgentID, str], scores: Dict[str, int], force: bool = False) -> None:
        tick = int(state.tick_count)
        bucket = tick // max(1, self.sample_every_ticks)
        if (not force) and (self._bucket is not None) and (bucket == self._bucket):
            return

        self._bucket = bucket

        row: Dict[str, Any] = {
            "match_id": self.match_id,
            "tick": tick,
            "time_s": float(tick) / float(TICKS_PER_SECOND),
            "blue_score": int(scores.get("BLUE", 0)),
            "orange_score": int(scores.get("ORANGE", 0)),
        }

        ball = state.ball
        row.update({
            "ball_px": float(ball.position[0]), "ball_py": float(ball.position[1]), "ball_pz": float(ball.position[2]),
            "ball_vx": float(ball.linear_velocity[0]), "ball_vy": float(ball.linear_velocity[1]), "ball_vz": float(ball.linear_velocity[2]),
            "ball_avx": float(ball.angular_velocity[0]), "ball_avy": float(ball.angular_velocity[1]), "ball_avz": float(ball.angular_velocity[2]),
        })

        slots = self._get_slots(state)
        for side, aids in slots.items():
            for i, aid in enumerate(aids):
                pfx = f"{side}-{i}"
                if aid is None or aid not in state.cars:
                    # fill blanks
                    row[f"{pfx}_aid"] = ""
                    row[f"{pfx}_profile"] = ""
                    for k in ("px","py","pz","vx","vy","vz","fx","fy","fz","ux","uy","uz","avx","avy","avz"):
                        row[f"{pfx}_{k}"] = ""
                    row[f"{pfx}_boost"] = 0
                    row[f"{pfx}_jump_presses"] = 0
                    row[f"{pfx}_double_jump"] = 0
                    row[f"{pfx}_handbrake"] = 0
                    row[f"{pfx}_action_idx_last"] = -1
                    row[f"{pfx}_switches_window"] = 0
                    row[f"{pfx}_demoed"] = 0
                    row[f"{pfx}_demo_timer"] = 0.0
                    continue

                car = state.cars[aid]
                phys = car.physics
                row[f"{pfx}_aid"] = str(aid)
                row[f"{pfx}_profile"] = str(profile_by_agent.get(aid, ""))

                row[f"{pfx}_px"] = float(phys.position[0]); row[f"{pfx}_py"] = float(phys.position[1]); row[f"{pfx}_pz"] = float(phys.position[2])
                lv = phys.linear_velocity
                row[f"{pfx}_vx"] = float(lv[0]); row[f"{pfx}_vy"] = float(lv[1]); row[f"{pfx}_vz"] = float(lv[2])

                fwd = phys.forward
                up  = phys.up
                row[f"{pfx}_fx"] = float(fwd[0]); row[f"{pfx}_fy"] = float(fwd[1]); row[f"{pfx}_fz"] = float(fwd[2])
                row[f"{pfx}_ux"] = float(up[0]);  row[f"{pfx}_uy"] = float(up[1]);  row[f"{pfx}_uz"] = float(up[2])

                av = phys.angular_velocity
                row[f"{pfx}_avx"] = float(av[0]); row[f"{pfx}_avy"] = float(av[1]); row[f"{pfx}_avz"] = float(av[2])

                a = self._agg.get(aid, {})
                row[f"{pfx}_boost"] = int(a.get("boost_any", 0))
                row[f"{pfx}_jump_presses"] = int(a.get("jump_presses", 0))
                row[f"{pfx}_double_jump"] = int(a.get("double_jump", 0))
                row[f"{pfx}_handbrake"] = int(a.get("handbrake_any", 0))
                row[f"{pfx}_action_idx_last"] = int(a.get("action_idx_last", -1))
                row[f"{pfx}_switches_window"] = int(a.get("switches_window", 0))
                row[f"{pfx}_demoed"] = int(a.get("demoed", 0))
                row[f"{pfx}_demo_timer"] = float(a.get("demo_timer", 0.0))

        self._writer.writerow(row)
        self._f.flush()

        # reset window aggregates (keep prev_jump for edge detection across windows)
        for aid in list(self._agg.keys()):
            last_idx = int(self._agg[aid].get("action_idx_last", -1))
            demoed = int(self._agg[aid].get("demoed", 0))
            demo_timer = float(self._agg[aid].get("demo_timer", 0.0))
            # preserve last action + demo info for viz continuity, but clear event-flags
            self._agg[aid] = {
                "switches_window": 0,
                "action_idx_last": last_idx,
                "demoed": demoed,
                "demo_timer": demo_timer,
            }


    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

class RewardContributionLogger:
    """
    One row per match:
      match_id, (blue_sum, orange_sum) for each reward component AFTER annealing multiplier.
    Also includes postgame contribution outcome reward (fixed +/-10; not annealed).
    """
    def __init__(self, path: str, reward_keys: List[str]):
        self.path = path
        self.reward_keys = list(reward_keys)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

        self._f = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self._fieldnames())
        if is_new:
            self._writer.writeheader()

    def _fieldnames(self) -> List[str]:
        cols = [
            "match_id",
            "blue_team",
            "orange_team",
            "reward_scale",
            "blue_score",
            "orange_score",
            "goal_diff",
            "postgame_contrib_goal_diff",
            "ac_tick_reward_blue",   # total per-tick ball-half HL reward accumulated this match
            "ac_tick_reward_orange",
        ]
        cols += [f"r_{k}" for k in self.reward_keys]
        return cols

    def log_match(
        self,
        *,
        match_id: int,
        blue_team: str,
        orange_team: str,
        reward_scale: float,
        blue_score: int,
        orange_score: int,
        ac_tick_reward_blue: float,
        ac_tick_reward_orange: float,
        team_sums_blue: Dict[str, float],
        team_sums_orange: Dict[str, float],
    ) -> None:
        goal_diff = int(blue_score - orange_score)

        row: Dict[str, Any] = {
            "match_id": int(match_id),
            "blue_team": str(blue_team),
            "orange_team": str(orange_team),
            "reward_scale": float(reward_scale),
            "blue_score": int(blue_score),
            "orange_score": int(orange_score),
            "goal_diff": goal_diff,
            # team-total postgame contribution outcome reward (fixed +/-10)
            "postgame_contrib_goal_diff": f"({(10.0 if goal_diff>0 else (-10.0 if goal_diff<0 else 0.0)):.6f},{(-10.0 if goal_diff>0 else (10.0 if goal_diff<0 else 0.0)):.6f})",
            "ac_tick_reward_blue": float(ac_tick_reward_blue),
            "ac_tick_reward_orange": float(ac_tick_reward_orange),
        }

        for k in self.reward_keys:
            b = float(team_sums_blue.get(k, 0.0))
            o = float(team_sums_orange.get(k, 0.0))
            row[f"r_{k}"] = f"({b:.6f},{o:.6f})"

        self._writer.writerow(row)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

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

    def run(
        self,
        match_id: int,
        low_logger: Optional[LowLevelLogger] = None,
        high_dir: Optional[str] = None,
        sample_every_ticks: int = 16,
        high_logger: Optional[HighLevelMatchLogger] = None,
        reward_logger: Optional[RewardContributionLogger] = None,
    ):
        # Reset match-level tracking
        self.scores = {"BLUE": 0, "ORANGE": 0}
        # Reward tracking
        team_reward_sums = {0: {}, 1: {}}
        # Accumulated per-tick HL ball-half rewards for logging (not scaled by reward_scale)
        ac_tick_total_blue = 0.0
        ac_tick_total_orange = 0.0

        _, info = self.env.reset()
        agent_ids = list(self.env.state.cars.keys())
        self.timer.reset(agent_ids, self.env.state, shared_info={})

        if high_logger is None and high_dir is not None:
            high_logger = HighLevelMatchLogger(
                out_dir=high_dir,
                match_id=match_id,
                blue_team=self.env.blue_team_name,
                orange_team=self.env.orange_team_name,
                sample_every_ticks=sample_every_ticks,
            )


        # per-team switch counts
        switches_team = {0: 0, 1: 0}

        # per-team deployment tick totals per profile
        dep_ticks = {0: {}, 1: {}}
        # per-agent deployment ticks (for postgame contribution reward)
        dep_ticks_agent: Dict[AgentID, float] = {aid: 0.0 for aid in agent_ids}

        # initial sample at tick 0
        if high_logger is not None: high_logger.maybe_sample(self.env.state, info.get("profile_by_agent", {}), self.scores, force=True)

        while True:
            state_before = self.env.state
            tick_before = int(state_before.tick_count)

            actions: Dict[AgentID, np.ndarray] = {}
            actions_idx: Dict[AgentID, int] = {}

            self._prev_obs.clear()
            self._prev_logp.clear()
            self._prev_val.clear()
            prev_profiles = dict(info["profile_by_agent"])
            self._prev_profile = dict(prev_profiles)

            # choose action per agent using the currently deployed profile's policy
            for aid in agent_ids:
                pname = prev_profiles[aid]
                a, logp, v = self.ppo_players[pname].act(info["ll_obs"][aid])
                actions[aid] = np.array([a], dtype=np.int64)
                actions_idx[aid] = int(a)
                self._prev_obs[aid] = info["ll_obs"][aid]
                self._prev_logp[aid] = logp
                self._prev_val[aid] = v

            # step env
            _, _, done, info = self.env.step(actions, shared_info={'actions_idx': actions_idx})

            state_after = self.env.state
            tick_after = int(state_after.tick_count)
            dticks = max(1, tick_after - tick_before)

            # Reward tracking
            rb = info.get("reward_breakdown", {})
            for aid in agent_ids:
                team = int(state_after.cars[aid].team_num)
                bd = rb.get(aid, {})
                if not isinstance(bd, dict):
                    continue
                for k, v in bd.items():
                    team_reward_sums[team][k] = float(team_reward_sums[team].get(k, 0.0)) + float(v)

            # account deployment time for the interval [before, after)
            for aid in agent_ids:
                team = int(state_before.cars[aid].team_num)
                pname = prev_profiles.get(aid, "")
                dep_ticks[team][pname] = float(dep_ticks[team].get(pname, 0.0)) + float(dticks)
                dep_ticks_agent[aid] = float(dep_ticks_agent.get(aid, 0.0)) + float(dticks)

            # detect switches (between prev_profiles and current)
            cur_profiles = dict(info["profile_by_agent"])
            per_agent_switches: Dict[AgentID, int] = {}
            for aid in agent_ids:
                if cur_profiles.get(aid) != prev_profiles.get(aid):
                    team = int(state_after.cars[aid].team_num)
                    switches_team[team] += 1
                    per_agent_switches[aid] = per_agent_switches.get(aid, 0) + 1

            # Per-tick HL reward: ball-half position + touch bonus
            _r_tick_blue   = _hl_ball_half_reward(state_after, team=0)
            _r_tick_orange = _hl_ball_half_reward(state_after, team=1)
            _last_touch = info.get("last_touch", {})
            if isinstance(_last_touch, dict) and _last_touch.get("tick") == int(state_after.tick_count):
                _touch_team = _last_touch.get("team")
                if _touch_team == 0:
                    _r_tick_blue  += _HL_TOUCH_REWARD
                elif _touch_team == 1:
                    _r_tick_orange += _HL_TOUCH_REWARD
            ac_tick_total_blue   += _r_tick_blue
            ac_tick_total_orange += _r_tick_orange
            if hasattr(self.env, "ac_blue") and self.env.ac_blue is not None:
                self.env.ac_blue.add_tick_reward(_r_tick_blue)
            if hasattr(self.env, "ac_orange") and self.env.ac_orange is not None:
                self.env.ac_orange.add_tick_reward(_r_tick_orange)

            # termination checks (need to know if this is the last transition before storing)
            dones = self.timer.is_done(agent_ids, state_after, shared_info={})
            done_flag = bool(any(dones.values()))

            # store PPO transitions using reward from this step (optionally with postgame bonus on terminal step)
            rmap = dict(info["rewards"])

            if done_flag:
                # Postgame contribution reward:
                # - Fixed +/-10 team total based on win/loss (independent of goal_diff magnitude).
                # - Distributed across *agents* proportional to their deployment time in this match.
                gd = int(self.scores["BLUE"] - self.scores["ORANGE"])
                if gd != 0:
                    outcome = 1.0 if gd > 0 else -1.0  # +1 => BLUE win, -1 => ORANGE win
                    postgame_total = 10.0
                    blue_total = outcome * postgame_total
                    orange_total = -blue_total

                    team_ticks = {0: 0.0, 1: 0.0}
                    for _aid in agent_ids:
                        t = int(state_after.cars[_aid].team_num)
                        team_ticks[t] += float(dep_ticks_agent.get(_aid, 0.0))

                    for _aid in agent_ids:
                        t = int(state_after.cars[_aid].team_num)
                        denom = float(team_ticks.get(t, 0.0))
                        if denom <= 0.0:
                            continue
                        frac = float(dep_ticks_agent.get(_aid, 0.0)) / denom
                        bonus = (blue_total if t == 0 else orange_total) * frac
                        rmap[_aid] = float(rmap.get(_aid, 0.0)) + float(bonus)

            for aid in agent_ids:
                pname_prev = prev_profiles[aid]
                self.ppo_players[pname_prev].store(
                    self._prev_obs[aid],
                    int(actions[aid][0]),
                    float(self._prev_logp[aid]),
                    float(rmap.get(aid, 0.0)),
                    float(self._prev_val[aid]),
                    bool(done_flag),
                )

            # update any buffers that filled
            for agent in self.ppo_players.values():
                if agent.buffer.full():
                    agent.update()

            # high-level action aggregation + sampling
            if high_logger is not None:
                high_logger.observe_actions(actions_idx, state_after)
                if per_agent_switches:
                    high_logger.note_switches(per_agent_switches)
                high_logger.maybe_sample(state_after, cur_profiles, self.scores, force=False)
            if done_flag:
                break

            if state_after.goal_scored:
                # Per-goal HL training signal: train on segment since last goal then reset.
                # Blue scores when ball_y > 0 (same convention as _score_and_reset_kickoff).
                _blue_scored = float(state_after.ball.position[1]) > 0
                if hasattr(self.env, "ac_blue") and self.env.ac_blue is not None:
                    self.env.ac_blue.finalize_goal(+1.0 if _blue_scored else -1.0)
                if hasattr(self.env, "ac_orange") and self.env.ac_orange is not None:
                    self.env.ac_orange.finalize_goal(+1.0 if not _blue_scored else -1.0)
                # score and kickoff reset inside engine
                self._score_and_reset_kickoff(state_after)
                # after kickoff reset, force a sample so viz sees the reset point
                if high_logger is not None:
                    high_logger.maybe_sample(self.env.engine.state, cur_profiles, self.scores, force=True)


        
        # ---------------- High-level AC match close ----------------
        # Per-goal ±1 signals are issued at each goal via finalize_goal().
        # finalize_match(0.0) here flushes any remaining decisions since the last goal
        # using only accumulated tick rewards — no additional win/loss signal.
        if hasattr(self.env, "ac_blue") and self.env.ac_blue is not None:
            self.env.ac_blue.finalize_match(0.0)
        if hasattr(self.env, "ac_orange") and self.env.ac_orange is not None:
            self.env.ac_orange.finalize_match(0.0)

        # finalize high log
        if high_logger is not None:
            high_logger.maybe_sample(self.env.state, info.get("profile_by_agent", {}), self.scores, force=True)
            high_logger.close()

        # build low-level match summary (contrib fractions per team)
        def norm_contrib(team_id: int) -> Dict[str, float]:
            d = dep_ticks.get(team_id, {})
            denom = float(sum(d.values()))
            if denom <= 0.0:
                return {}
            return {k: float(v) / denom for k, v in d.items()}

        contrib_blue = norm_contrib(0)
        contrib_orange = norm_contrib(1)

        if low_logger is not None:
            low_logger.log_match(
                match_id=match_id,
                blue_team=self.env.blue_team_name,
                orange_team=self.env.orange_team_name,
                blue_score=int(self.scores["BLUE"]),
                orange_score=int(self.scores["ORANGE"]),
                blue_switches=int(switches_team[0]),
                orange_switches=int(switches_team[1]),
                contrib_blue=contrib_blue,
                contrib_orange=contrib_orange,
            )
        
        if reward_logger is not None:
            reward_logger.log_match(
                match_id=match_id,
                blue_team=self.env.blue_team_name,
                orange_team=self.env.orange_team_name,
                reward_scale=float(getattr(self.env, "reward_scale", 1.0)),
                blue_score=int(self.scores["BLUE"]),
                orange_score=int(self.scores["ORANGE"]),
                ac_tick_reward_blue=float(ac_tick_total_blue),
                ac_tick_reward_orange=float(ac_tick_total_orange),
                team_sums_blue=dict(team_reward_sums[0]),
                team_sums_orange=dict(team_reward_sums[1]),
            )

        return {
            "match_id": match_id,
            "scores": dict(self.scores),
            "high_log_path": (high_logger.path if high_logger is not None else None),
        }

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
        "s2": AgentProfile("s2", "striker", tweak(S_base, dist_to_ball=6.0e-5, car_speed=5.0e-6, ball_dist_to_goal=5.0e-5, face_ball=1.0e-5, shot_alignment=6.0e-5, ball_hit=1.5e-1, touch=1.25e-1)),
        "s3": AgentProfile("s3", "striker", tweak(S_base, dist_to_ball=6.0e-5, car_speed=5.0e-6, ball_dist_to_goal=3.0e-5, face_ball=1.0e-5, shot_alignment=7.0e-5, ball_hit=1.0e-1, touch=7.5e-2, goal=24.0)),
        "s4": AgentProfile("s4", "striker", tweak(S_base, dist_to_ball=7.0e-5, car_speed=5.0e-6, ball_dist_to_goal=4.0e-5, face_ball=2.0e-5, shot_alignment=6.0e-5, ball_hit=1.0e-1, touch=7.5e-2)),
        "s5": AgentProfile("s5", "striker", tweak(S_base, dist_to_ball=8.0e-5, car_speed=5.0e-6, ball_dist_to_goal=4.0e-5, face_ball=1.0e-5, shot_alignment=7.0e-5, ball_hit=1.0e-1, touch=1.0e-1, goal=24.0)),
        "p1": AgentProfile("p1", "positioning", dict(P_base)),
        "p2": AgentProfile("p2", "positioning", tweak(P_base, dist_to_ball=5.0e-5, ball_dist_to_goal=4.0e-5, car_speed=5.0e-6, mean_dist_to_teammates=1.0e-5, behind_ball_defensive=4.0e-6, face_ball=2.0e-5, shot_alignment=5.0e-5, block_alignment=7.5e-5, def_hit=2.0e-1, ball_hit=2.0e-1, touch=1.25e-1)),
        "p3": AgentProfile("p3", "positioning", tweak(P_base, dist_to_ball=3.0e-5, ball_dist_to_goal=2.0e-5, car_speed=5.0e-6, mean_dist_to_teammates=5.0e-6, behind_ball_defensive=2.0e-6, face_ball=3.0e-5, shot_alignment=7.5e-5, block_alignment=5.0e-5, def_hit=1.0e-1, ball_hit=1.0e-1, touch=1.0e-1)),
        "d1": AgentProfile("d1", "defender", dict(D_base)),
        "d2": AgentProfile("d2", "defender", tweak(D_base, dist_to_ball=3.0e-5, face_ball=3.0e-5, block_alignment=7.5e-5, behind_ball_defensive=1.0e-6, def_hit=2.0e-1, touch=1.5e-1, ball_dist_from_goal=4.0e-5)),
        "d3": AgentProfile("d3", "defender", tweak(D_base, dist_to_ball=4.0e-5, face_ball=1.0e-5, block_alignment=7.5e-5, behind_ball_defensive=1.0e-6, def_hit=2.0e-1, touch=1.5e-1, ball_dist_from_goal=3.0e-5, goal=30.0)),
        "d4": AgentProfile("d4", "defender", tweak(D_base, dist_to_ball=3.0e-5, face_ball=2.0e-5, block_alignment=3.0e-5, behind_ball_defensive=2.0e-6, def_hit=2.0e-1, touch=1.5e-1, ball_dist_from_goal=5.0e-5, goal=30.0)),
    }

    TEAM_SPECS = {
        "team1_striker_balance": ["s1","s2","s3","p1","p2","d1","d2"],
        "team2_striker_heavy":   ["s1","s2","s3","s4","s5","p1","p2"],
        "team3_balance":         ["s4","s5","p1","p2","p3","d3","d4"],
        "team4_pos_striker":     ["s1","s2","s3","p1","p2","p3","d1"],
        "team5_balance2":        ["s4","s5","p1","p2","p3","d1","d2"],
        "team6_def_striker":     ["s1","s2","s3","d1","d2","d3","d4"],
        "team7_def_balance":     ["s3","s4","s5","p3","d2","d3","d4"],
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
    low_logger = LowLevelLogger("out/low_level_log.csv", profile_names=list(GLOBAL_PROFILES.keys()))
    result = runner.run(match_id=1, low_logger=low_logger, high_dir="out/high_logs", sample_every_ticks=16)
    low_logger.close()
    print("finished:", result)


if __name__ == "__main__":
    main_smoke_test()
