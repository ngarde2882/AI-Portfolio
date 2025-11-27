import numpy as np
import csv, os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Optional

from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.done_conditions.goal_condition import GoalCondition
from rlgym.rocket_league.done_conditions.timeout_condition import TimeoutCondition
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

from rlgym.rocket_league.common_values import ORANGE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, BOOST_LOCATIONS

from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction

N_ACTIONS = 90
base_parser   = LookupTableAction()
action_parser = RepeatAction(base_parser, repeats=8)  # adjust repeats for your tick_skip

# ===============================
# HRL TEAM AGENT
# ===============================
class TeamActorCritic(nn.Module):
    def __init__(self, obs_size, n_categories, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_categories),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        probs = self.actor(obs)
        value = self.critic(obs)
        return probs, value


# ===============================
# PPO for low-level agents
# ===============================
import math
from dataclasses import dataclass

class PPONet(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden=256):
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

    def forward(self, obs):  # obs: (B, obs_size)
        logits = self.pi(obs)
        value = self.v(obs)
        return logits, value


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
    def __init__(self, obs_dim, size, n_actions):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.acts = np.zeros((size,), np.int64)
        self.logp = np.zeros((size,), np.float32)
        self.rew = np.zeros((size,), np.float32)
        self.val = np.zeros((size,), np.float32)
        self.done = np.zeros((size,), np.float32)
        self.ptr = 0
        self.max_size = size

    def add(self, obs, act, logp, rew, val, done):
        i = self.ptr
        self.obs[i] = obs
        self.acts[i] = act
        self.logp[i] = logp
        self.rew[i] = rew
        self.val[i] = val
        self.done[i] = float(done)
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def full(self):
        return self.ptr >= self.max_size

    def compute_gae(self, gamma, lam):
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
        # normalize advantages
        a_std = adv[:self.ptr].std() + 1e-8
        a_mean = adv[:self.ptr].mean()
        adv[:self.ptr] = (adv[:self.ptr] - a_mean) / a_std
        return adv[:self.ptr], ret[:self.ptr]

class PPOAgent:
    """
    One PPO policy per-player (you can share weights if you want).
    Uses discrete 90-way actions.
    """
    def __init__(self, obs_size, n_actions, hyper=PPOHyper(), device="cpu"):
        self.net = PPONet(obs_size, n_actions).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=hyper.lr)
        self.h = hyper
        self.device = device
        self.buffer = PPOBuffer(obs_size, hyper.batch_size, n_actions)
        self.n_actions = n_actions

    @torch.no_grad()
    def act(self, obs_np):
        # obs_np: (obs_size,)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, v = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(v.item())

    def store(self, *transition):
        self.buffer.add(*transition)

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
            # policy loss (clipped)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1.0 - self.h.clip_eps, 1.0 + self.h.clip_eps) * adv_t
            pi_loss = -torch.min(unclipped, clipped).mean()
            # value loss
            v_loss = 0.5 * (ret_t - values.squeeze(-1)).pow(2).mean()
            # entropy bonus
            ent = dist.entropy().mean()
            loss = pi_loss + self.h.vf_coef * v_loss - self.h.ent_coef * ent

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        buf.reset()


# ===============================
# INFO/STATE EXTRACTION HELPERS
# ===============================

def _safe_norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return n if n > 1e-9 else 1e-9

def build_info_from_state(state, agent_id):
    car = state.cars[agent_id]
    is_orange = (car.team_num == ORANGE_TEAM)

    # --- choose inverted vs normal, like DefaultObs ---
    if is_orange:
        physics = car.inverted_physics       # PhysicsObject.inverted()
        ball    = state.inverted_ball
        pads    = state.inverted_boost_pad_timers
        boost_positions = np.asarray(BOOST_LOCATIONS, dtype=np.float32).copy()
        boost_positions[:, 1] *= -1   # keep pad coords in the same inverted frame
        opp_goal = np.asarray(BLUE_GOAL_BACK,   dtype=np.float32)  # attacking toward +Y
        own_goal = np.asarray(ORANGE_GOAL_BACK, dtype=np.float32)
    else:
        physics = car.physics
        ball    = state.ball
        pads    = state.boost_pad_timers
        boost_positions = np.asarray(BOOST_LOCATIONS, dtype=np.float32)
        opp_goal = np.asarray(ORANGE_GOAL_BACK, dtype=np.float32)
        own_goal = np.asarray(BLUE_GOAL_BACK,   dtype=np.float32)

    me_pos = np.asarray(physics.position, dtype=np.float32)
    me_vel = np.asarray(physics.linear_velocity, dtype=np.float32)
    car_forward = np.asarray(physics.forward, dtype=np.float32)

    ball_pos = np.asarray(ball.position, dtype=np.float32)
    ball_vel = np.asarray(ball.linear_velocity, dtype=np.float32)

    to_ball = ball_pos - me_pos
    to_ball_unit = to_ball / _safe_norm(to_ball)

    # teammates/opponents in the SAME frame
    teammate_positions, opponent_positions = [], []
    for aid, other in state.cars.items():
        if aid == agent_id:
            continue
        other_phys = other.inverted_physics if is_orange else other.physics
        pos = np.asarray(other_phys.position, dtype=np.float32)
        (teammate_positions if other.team_num == car.team_num else opponent_positions).append(pos)

    # nearest boost (in this frame)
    dists = np.linalg.norm(boost_positions - me_pos[None, :], axis=1)
    nearest_idx = int(np.argmin(dists))

    info = {
        "car_position": me_pos,
        "car_velocity": me_vel,
        "car_forward": car_forward,
        "ball_position": ball_pos,
        "ball_velocity": ball_vel,
        "ball_speed": float(_safe_norm(ball_vel)),
        "to_ball_vec": to_ball_unit,
        "goal_position": opp_goal,
        "own_goal_position": own_goal,
        "boost_amount": float(car.boost_amount),
        "on_ground": int(car.on_ground) if hasattr(car, "on_ground") else 0,
        "is_boosting": int(car.is_boosting) if hasattr(car, "is_boosting") else 0,
        "is_supersonic": int(car.is_supersonic) if hasattr(car, "is_supersonic") else 0,
        "teammate_positions": teammate_positions,
        "opponent_positions": opponent_positions,
        "ball_touched": bool(getattr(car, "ball_touches", 0) > 0),
        "nearest_boost_dist": float(dists[nearest_idx]),
        "nearest_boost_cooldown": float(pads[nearest_idx]),
        "team_sign": 1.0,  # in this unified frame, everyone attacks +Y
    }
    return info

def _safe_touches(car) -> int:
    """Return a sane integer touch count even if backend leaves it None/str."""
    val = getattr(car, "ball_touches", 0)
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0

# ===============================
# REWARD FEATURE FUNCTIONS (obs, info) -> float
# ===============================

def dist_to_ball(obs, info):
    ball_pos = info.get('ball_position', np.zeros(3))
    car_pos = info.get('car_position', np.zeros(3))
    return -float(np.linalg.norm(car_pos - ball_pos))


def car_speed(obs, info):
    vel = info.get('car_velocity', np.zeros(3))
    return float(np.linalg.norm(vel))


def boost_remaining(obs, info):
    return float(info.get('boost_amount', 0.0))


def ball_hit(obs, info):
    if info.get('ball_touched', False):
        return float(info.get('ball_speed', 0.0)) * (
            1.0 + float(np.dot(info.get('car_forward', np.zeros(3)), info.get('to_ball_vec', np.zeros(3)))))
    return 0.0


def ball_dist_to_goal(obs, info):
    ball_pos = info.get('ball_position', np.zeros(3))
    goal_pos = info.get('goal_position', np.zeros(3))
    return -float(np.linalg.norm(ball_pos - goal_pos))


def mean_dist_to_teammates(obs, info):
    teammates = info.get('teammate_positions', [])
    car_pos = info.get('car_position', np.zeros(3))
    if not teammates:
        return 0.0
    d = [np.linalg.norm(car_pos - t) for t in teammates]
    return float(np.mean(d))


def mean_dist_to_opponents(obs, info):
    opponents = info.get('opponent_positions', [])
    car_pos = info.get('car_position', np.zeros(3))
    if not opponents:
        return 0.0
    d = [np.linalg.norm(car_pos - o) for o in opponents]
    return float(np.mean(d))


def centerline_proximity(obs, info):
    # Reward being central on Y (defensive rotation helper)
    car_pos = info.get('car_position', np.zeros(3))
    return -float(abs(car_pos[1]))


def face_ball(obs, info):
    fwd = info.get('car_forward', np.zeros(3))
    to_ball = info.get('to_ball_vec', np.zeros(3))
    return float(np.dot(fwd / _safe_norm(fwd), to_ball / _safe_norm(to_ball)))


def behind_ball_defensive(obs, info):
    # Positive if agent stays between own goal and ball (defensive positioning)
    own_goal = info.get('own_goal_position', np.zeros(3))
    ball = info.get('ball_position', np.zeros(3))
    me = info.get('car_position', np.zeros(3))
    # Compare distances goal->me vs goal->ball along field y-axis importance
    gm = np.linalg.norm(me - own_goal)
    gb = np.linalg.norm(ball - own_goal)
    return 1.0 if gm < gb else -1.0


def shot_alignment(obs, info):
    # Encourage aligning ball velocity toward opponent goal
    goal = info.get('goal_position', np.zeros(3))
    ball = info.get('ball_position', np.zeros(3))
    bvel = info.get('ball_velocity', np.zeros(3))
    to_goal = goal - ball
    return float(np.dot(bvel, to_goal) / (_safe_norm(bvel) * _safe_norm(to_goal)))


def nearest_boost_inverse_distance(obs, info):
    d = float(info.get('nearest_boost_dist', 0.0))
    return 1.0 / (1.0 + d)


def nearest_boost_availability(obs, info):
    # Positive if the closest pad is up (cooldown ~ 0)
    cd = float(info.get('nearest_boost_cooldown', 0.0))
    return -cd


def supersonic_bonus(obs, info):
    return 1.0 if info.get('is_supersonic', False) else 0.0

def face_goal(obs, info):
    fwd = info.get('car_forward', np.zeros(3))
    goal = info.get('goal_position', np.zeros(3))
    me   = info.get('car_position', np.zeros(3))
    to_goal = goal - me
    return float(np.dot(fwd / _safe_norm(fwd), to_goal / _safe_norm(to_goal)))


def home_goal_proximity(obs, info):
    own = info.get('own_goal_position', np.zeros(3))
    me  = info.get('car_position', np.zeros(3))
    return -float(np.linalg.norm(me - own))


def behind_other_players(obs, info):
    """Count how many other cars are 'ahead' of me (closer to own goal -> I'm behind them)."""
    own  = info.get('own_goal_position', np.zeros(3))
    me   = info.get('car_position', np.zeros(3))
    me_d = np.linalg.norm(me - own)
    others = info.get('teammate_positions', []) + info.get('opponent_positions', [])
    if not others:
        return 0.0
    cnt = sum(np.linalg.norm(np.asarray(p, np.float32) - own) < me_d for p in others)
    # scale to [0,1] by num others
    return float(cnt / max(1, len(others)))



# ===============================
# HIERARCHICAL CONTROLLER  (obs_size, n_actions, hyper=PPOHyper(), device="cpu")
# ===============================
class HierarchicalRLAgent:
    def __init__(self, obs_size, n_players, reward_categories, n_actions):
        self.n_categories = len(reward_categories)
        self.team_controller = TeamActorCritic(obs_size, self.n_categories)
        self.behaviors = [
            [PPOAgent(obs_size, n_actions, reward_features=wset) for wset in category]
            for category in reward_categories
        ]
        self.n_players = n_players
        self.optimizer = optim.Adam(self.team_controller.parameters(), lr=3e-4)

    def assign_category(self, team_obs):
        probs, _ = self.team_controller(team_obs)
        category_indices = torch.multinomial(probs, num_samples=self.n_players, replacement=True)
        return category_indices.tolist()

    def act(self, player_obs_list, team_obs):
        assigned = self.assign_category(team_obs)
        actions = []
        for i, obs in enumerate(player_obs_list):
            category_idx = assigned[i]
            behavior = random.choice(self.behaviors[category_idx])
            # NOTE: placeholder — assumes obs already mapped to a discrete index
            state_idx = int(obs) if np.isscalar(obs) else 0
            actions.append(behavior.select_action(state_idx))
        return actions


# ===============================
# RLGym Environment Integration (with InfoInjector)
# ===============================


def build_observation_from_info(info):
    """
    Team-relative compact observation vector.
    """
    me_pos   = info["car_position"]
    me_vel   = info["car_velocity"]
    me_fwd   = info["car_forward"]
    ball_pos = info["ball_position"]
    ball_vel = info["ball_velocity"]
    to_ball  = info["to_ball_vec"]

    boost    = np.array([info["boost_amount"]], dtype=np.float32)
    center_y = np.array([-abs(info["car_position"][1])], dtype=np.float32)
    pad_d    = np.array([info["nearest_boost_dist"]], dtype=np.float32)
    pad_cd   = np.array([info["nearest_boost_cooldown"]], dtype=np.float32)

    vec = np.concatenate([
        me_pos, me_vel, me_fwd,
        ball_pos, ball_vel, to_ball,
        boost, center_y, pad_d, pad_cd
    ]).astype(np.float32)

    return vec


# --- Engine boot & adapter ----------------------------------------------------

# old initialize_engine_with_state(...) -> replace with:

def initialize_engine_with_state(engine, initial_state=None, blue_size=1, orange_size=1):
    """
    Create a fresh GameState and apply our mutators:
      1) FixedTeamSizeMutator(blue_size, orange_size) - insert cars (expects empty state)
      2) KickoffMutator() - standard ball+car kickoff positions/angles/boost
    """
    gs = initial_state if initial_state is not None else engine.create_base_state()

    # Build and apply the sequence (order matters)
    mutators = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size),
        KickoffMutator(),
    )
    shared = {}
    mutators.apply(gs, shared)  # add cars, then set kickoff

    engine.set_state(gs, shared)
    return engine, gs


# ---- Next-touch bonus scaling (tunable) ----
NEXT_TOUCH_BASE = 0.1     # base bonus
NEXT_TOUCH_STEP = 0.05    # per-streak-step increment
NEXT_TOUCH_CAP  = 0.25    # clamp to avoid runaway

def _next_touch_bonus(streak: int) -> float:
    # linear ramp: base + step*(streak-1), clamped
    bonus = NEXT_TOUCH_BASE + NEXT_TOUCH_STEP * max(0, streak - 1)
    return float(min(bonus, NEXT_TOUCH_CAP))

class EngineEnvAdapter:
    def __init__(self,
                 engine,
                 action_parser,
                 reward_function,               # HotswapRewardAdapter (per-player rewards)
                 ll_obs_builder,                # use AdvancedObs()
                 ac_obs_builder,                # use AdvancedObs() (can be pruned later)
                 action_mode="discrete",
                 agent_selector=None,
                 ac_adapter=None,
                 blue_size=1, orange_size=1):              # <-- NEW: AC hook (HotswapACAdapter)
        self.engine = engine
        self.action_parser = action_parser
        self.action_mode = action_mode
        self.reward_function = reward_function
        self.ll_obs_builder = ll_obs_builder
        self.ac_obs_builder = ac_obs_builder
        self.agent_selector = agent_selector
        self.ac_adapter = ac_adapter
        self._blue_size = blue_size
        self._orange_size = orange_size

        # touch tracking
        self._last_touches = {}   # AgentID -> int (car.ball_touches)
        self._touch_buffer = deque(maxlen=getattr(ll_obs_builder, "touch_k", 8))
        self._last_touch = {"aid": None, "team": None, "tick": -1}
        self._await_next_team_touch = {}  # aid -> bool
        self._team_touch_streak = {0: 0, 1: 0}  # {BLUE: count, ORANGE: count}


    def _shared_info(self):
        # share the same extras with both obs builders
        return {"touch_buffer": list(self._touch_buffer)}

    def _build_ll_obs(self, state):
        obs_map = {}
        shared = self._shared_info()
        for aid in state.cars.keys():
            obs_map[aid] = self.ll_obs_builder._build_obs(aid, state, shared)  # use protected to pass shared
        return obs_map

    def _build_ac_obs(self, state):
        # You can build a real team-level tensor later; for now use the first agent's AdvancedObs
        aid0 = next(iter(state.cars.keys()))
        return self.ac_obs_builder._build_obs(aid0, state, self._shared_info())
    

    def _update_touch_buffer(self, prev_state, state):
        for aid, car in state.cars.items():
            prev = self._last_touches.get(aid, 0)
            cur = _safe_touches(car)
            if cur > prev:
                # team sign: blue +1, orange -1
                sign = +1.0 if car.team_num != ORANGE_TEAM else -1.0
                self._touch_buffer.append(sign)
                # mark who touched now
                self._last_touch = {"aid": aid, "team": car.team_num, "tick": int(state.tick_count)}
                # update team streaks
                t_team = car.team_num  # 0 blue, 1 orange
                prev_team = self._last_touch.get("team", None)
                if prev_team is None or prev_team == t_team:
                    # same-team consecutive touch -> grow this team’s streak
                    self._team_touch_streak[t_team] = self._team_touch_streak.get(t_team, 0) + 1
                else:
                    # opponent touched -> reset our streak to 1 and their streak to 0
                    self._team_touch_streak[t_team] = 1
                    self._team_touch_streak[1 - t_team] = 0
                # the toucher is now awaiting the *next* team touch bonus
                self._await_next_team_touch[aid] = True
            self._last_touches[aid] = cur

    def _compute_role_bonuses(self, state, manager, await_map):
        """
        Returns {aid: bonus} based on current profile role:
        - striker/defender/positioning: 'next team touch' bonus after this agent's last hit
        - defender: + behind_other_players + home_goal_proximity
        - positioning: + face_goal, + car_speed (already exists), + extra on 'teammate->agent' touch chain
        """
        bonuses = {aid: 0.0 for aid in state.cars.keys()}
        if manager is None or not hasattr(manager, "current_name"):
            return bonuses

        # detect if a new touch happened this step and by which team/aid
        # (env already updated _last_touch before calling this)
        # we'll use info dicts to evaluate features
        for aid in state.cars.keys():
            info = build_info_from_state(state, aid)

            role_name = manager.current_name.get(aid, "").lower()
            # --- defender spatial helpers
            if role_name == "defender":
                bonuses[aid] += behind_other_players(None, info)
                bonuses[aid] += 0.1 * home_goal_proximity(None, info)

            # --- positioning helpers
            if role_name == "positioning":
                bonuses[aid] += 0.1 * face_goal(None, info)
                bonuses[aid] += 0.02 * car_speed(None, info)  # gentle speed term

        return bonuses

    def reset(self, initial_state=None):
        _, state = initialize_engine_with_state(self.engine, initial_state=initial_state, blue_size=self._blue_size, orange_size=self._orange_size)
        self.state = state
        self._last_touches = {aid: _safe_touches(c) for aid, c in state.cars.items()}
        self._touch_buffer.clear()
        self._team_touch_streak = {0: 0, 1: 0}
        self._await_next_team_touch = {aid: False for aid in state.cars.keys()}
        self._last_touch = {"aid": None, "team": None, "tick": -1}


        agent_ids = list(state.cars.keys())
        self.reward_function.reset(agent_ids, state, {})
        ll_obs = self._build_ll_obs(state)
        ac_obs = self._build_ac_obs(state)
        first_obs = ll_obs[agent_ids[0]]
        info = {"ll_obs": ll_obs, "ac_obs": ac_obs}
        return first_obs, info

    def step(self, actions_dict, shared_info=None):
        prev_state = self.engine.state
        controls_map = self.action_parser.parse_actions(actions_dict, prev_state, shared_info or {})
        state = self.engine.step(controls_map, shared_info or {})
        self.state = state

        # AC-driven profile selection (hotswap) BEFORE reward calc
        if self.ac_adapter is not None and hasattr(self.reward_function, "hotswap"):
            team_agents = list(state.cars.keys())
            team_is_orange = any(state.cars[aid].team_num == ORANGE_TEAM for aid in team_agents)
            self.ac_adapter.decide_and_update(self.reward_function.hotswap, state, team_agents, team_is_orange)

        # update touch buffer off new state
        self._update_touch_buffer(prev_state, state)

        # role-aware bonuses (spatial/angle/speed)
        mgr = getattr(self.reward_function, "hotswap", None)
        bonuses = self._compute_role_bonuses(state, mgr, self._await_next_team_touch)

        # team next-touch bonuses:
        lt = getattr(self, "_last_touch", None) or {}
        if lt["aid"] is not None:
            t_aid = lt["aid"]
            t_team = lt["team"]
            # 1) if any agent was awaiting a next *team* touch and this touch's team matches -> bonus
            for aid in list(self._await_next_team_touch.keys()):
                if self._await_next_team_touch.get(aid, False):
                    if state.cars[aid].team_num == t_team:
                        # add a team next-touch bonus (works for striker/defender/positioning), modest weight
                        bonuses[aid] = bonuses.get(aid, 0.0) + 0.5
                        self._await_next_team_touch[aid] = False

        agent_ids = list(state.cars.keys())
        # rewards via hotswap composites (per-agent)
        rmap = self.reward_function.get_rewards(agent_ids, state, {aid: False for aid in agent_ids},
                                                {aid: False for aid in agent_ids}, shared_info or {})

        # --- next-touch streak bonus (to the current toucher only) ---
        toucher = lt.get("aid", None)
        team    = lt.get("team", None)
        if toucher is not None and team in (0, 1):
            streak = int(self._team_touch_streak.get(team, 0))
            rmap[toucher] = float(rmap.get(toucher, 0.0)) + _next_touch_bonus(streak)

        # add role-specific bonuses
        for aid in agent_ids:
            rmap[aid] = float(rmap[aid]) + float(bonuses.get(aid, 0.0))

        ll_obs = self._build_ll_obs(state)
        ac_obs = self._build_ac_obs(state)
        done = bool(getattr(state, "goal_scored", False))

        first_obs = ll_obs[agent_ids[0]]
        info = {"ll_obs": ll_obs, "ac_obs": ac_obs, "rewards": rmap, "touch_buffer": list(self._touch_buffer)}
        reward_scalar = float(sum(rmap.values()))
        return first_obs, reward_scalar, done, info


class MatchRunner:
    """
    Runs a continuous 5-minute match:
      - Ends only on TimeoutCondition(300s) (not on goals)
      - On goal: applies KickoffMutator and continues
      - Logs per-tick state & current profile assignment for review
    """
    def __init__(self, env, ppo_agents, kickoffs: Optional[KickoffMutator] = None):
        self.env = env
        self.ppo_agents = ppo_agents
        self.kickoffs = kickoffs or KickoffMutator()
        self.timer = TimeoutCondition(timeout_seconds=300.0)  # 5 minutes  :contentReference[oaicite:8]{index=8}
        self.scores = {"BLUE": 0, "ORANGE": 0}
        self._initialized = False

    def _score_and_reset_kickoff(self, state):
        # naive team guess from goal_scored flag + ball y; adapt if your engine exposes scorer
        ball_y = state.ball.position[1]
        if ball_y > 0:   # ball ended in ORANGE half -> BLUE scored
            self.scores["BLUE"] += 1
        else:
            self.scores["ORANGE"] += 1
        # re-apply a kickoff without touching the timer
        gs = state
        seq = MutatorSequence(self.kickoffs)
        seq.apply(gs, shared_info={})
        self.env.engine.set_state(gs, shared_info={})
        # also clear touch buffer in the env if present
        if hasattr(self.env, "_touch_buffer"):
            self.env._touch_buffer.clear()

    def _log_row(self, state, profiles_by_agent):
        row = {
            "tick": int(state.tick_count),
            "time_s": float(state.tick_count) / 120.0,  # TICKS_PER_SECOND
            "ball_x": float(state.ball.position[0]),
            "ball_y": float(state.ball.position[1]),
            "ball_z": float(state.ball.position[2]),
            "ball_vx": float(state.ball.linear_velocity[0]),
            "ball_vy": float(state.ball.linear_velocity[1]),
            "ball_vz": float(state.ball.linear_velocity[2]),
            "score_blue": self.scores["BLUE"],
            "score_orange": self.scores["ORANGE"],
        }
        # append each car
        lt = getattr(self.env, "_last_touch", {"aid": None, "team": None, "tick": -1})
        row["last_touch_aid"]  = "" if lt.get("aid") is None else str(lt["aid"])
        row["last_touch_team"] = -1 if lt.get("team") is None else int(lt["team"])
        row["last_touch_tick"] = int(lt.get("tick", -1))
        # cars...
        for aid, car in state.cars.items():
            p = car.physics
            row.update({
                f"{aid}_team": int(car.team_num),
                f"{aid}_x": float(p.position[0]),   f"{aid}_y": float(p.position[1]),   f"{aid}_z": float(p.position[2]),
                f"{aid}_vx": float(p.linear_velocity[0]), f"{aid}_vy": float(p.linear_velocity[1]), f"{aid}_vz": float(p.linear_velocity[2]),
                f"{aid}_boost": float(getattr(car, "boost_amount", 0.0)),
                f"{aid}_profile": profiles_by_agent.get(aid, ""),
            })
        row["touch_streak_blue"]   = int(getattr(self.env, "_team_touch_streak", {}).get(0, 0))
        row["touch_streak_orange"] = int(getattr(self.env, "_team_touch_streak", {}).get(1, 0))
        row["last_touch_bonus"]    = float(_next_touch_bonus(row["touch_streak_blue"] if row["last_touch_team"] == 0
            else row["touch_streak_orange"]) if row["last_touch_team"] in (0,1) else 0.0)
        return row

    def run(self, log_path="match_log.csv"):
        # reset env + timer
        first_obs, info = self.env.reset()
        agent_ids = list(self.env.state.cars.keys())
        self.timer.reset(agent_ids, self.env.state, shared_info={})
        self._initialized = True

        # open CSV
        fieldnames = None
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        f = open(log_path, "w", newline="")
        writer = None

        try:
            while True:
                # PPO picks low-level actions per agent
                actions = {}
                for aid in agent_ids:
                    a, logp, v = self.ppo_agents[aid].act(info["ll_obs"][aid])
                    actions[aid] = np.array([a], dtype=np.int64)

                # step sim (AC hotswap happens inside env.step)
                obs, reward_sum, done, info = self.env.step(actions)

                # read current profile names (if available) for logging
                profiles = {}
                mgr = getattr(self.env.reward_function, "hotswap", None)
                if mgr is not None and hasattr(mgr, "current_name"):
                    profiles = dict(mgr.current_name)

                # log frame
                row = self._log_row(self.env.state, profiles)
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)

                # timeout condition to end match
                dones = self.timer.is_done(agent_ids, self.env.state, shared_info={})  # 5-min gate  :contentReference[oaicite:9]{index=9}
                if any(dones.values()):
                    break

                # goal handling: restart kickoff, continue same match/time window
                if getattr(self.env.state, "goal_scored", False):
                    self._score_and_reset_kickoff(self.env.state)

        finally:
            f.close()

        return {"scores": dict(self.scores), "log_path": log_path}


# ===============================
# Reward feature sets
# ===============================
striker_rewards = {
    'dist_to_ball': dist_to_ball,
    'car_speed': car_speed,
    'boost_remaining': boost_remaining,
    'ball_hit': ball_hit,
    'ball_dist_to_goal': ball_dist_to_goal,
    'face_ball': face_ball,
    'shot_alignment': shot_alignment,
    'supersonic_bonus': supersonic_bonus,
    'next_team_touch': lambda obs, info: 0.0,
}

defender_rewards = {
    'dist_to_ball': dist_to_ball,
    'boost_remaining': boost_remaining,
    'ball_hit': ball_hit,
    'behind_ball_defensive': behind_ball_defensive,
    'centerline_proximity': centerline_proximity,
    'nearest_boost_inverse_distance': nearest_boost_inverse_distance,
    'nearest_boost_availability': nearest_boost_availability,
    'face_ball': face_ball,
    'next_team_touch': lambda obs, info: 0.0,
    'behind_other_players': behind_other_players,
    'home_goal_proximity': home_goal_proximity,
}

positioning_rewards = {
    'dist_to_ball': dist_to_ball,
    'boost_remaining': boost_remaining,
    'mean_dist_to_teammates': mean_dist_to_teammates,
    'mean_dist_to_opponents': mean_dist_to_opponents,
    'centerline_proximity': centerline_proximity,
    'face_ball': face_ball,
    'face_goal': face_goal,
    'car_speed': car_speed,
    'next_team_touch': lambda obs, info: 0.0,
}

from hotswap_hrl import AgentProfile, TeamProfilePool, HotswapManager, HotswapRewardAdapter, default_policy_factory, ACProfilePolicy, ACConfig, HotswapACAdapter
from reward_native_classes import StrikerCompositeReward, DefenderCompositeReward, PositioningCompositeReward
from rlgym.rocket_league.sim import RocketSimEngine
from AdvancedObs import AdvancedObs
# engine + action parser
engine = RocketSimEngine(rlbot_delay=True)
N_ACTIONS = 90
action_parser = RepeatAction(LookupTableAction(), repeats=8)

# profile pools / hotswap
# blue_pool = TeamProfilePool()
# blue_pool.add(AgentProfile("S_base", "striker",     StrikerCompositeReward().get_weights()))
# blue_pool.add(AgentProfile("S_agro", "striker",     {**StrikerCompositeReward().get_weights(), "shot_alignment": 1.2, "goal": 12.0}))
# blue_pool.add(AgentProfile("D_base", "defender",    DefenderCompositeReward().get_weights()))
# blue_pool.add(AgentProfile("P_base", "positioning", PositioningCompositeReward().get_weights()))
# blue_policy  = default_policy_factory(blue_pool)   # replace with AC policy later
# hotswap_reward = HotswapRewardAdapter(HotswapManager(blue_pool, policy=blue_policy))

# separate obs builders
ll_obs_builder = AdvancedObs()
ac_obs_builder = AdvancedObs()

# Build profile name list for AC
blue_pool = TeamProfilePool()
blue_pool.add(AgentProfile("S_base", "striker",     StrikerCompositeReward().get_weights()))
blue_pool.add(AgentProfile("S_agro", "striker",     {**StrikerCompositeReward().get_weights(), "shot_alignment": 1.2, "goal": 12.0}))
blue_pool.add(AgentProfile("D_base", "defender",    DefenderCompositeReward().get_weights()))
blue_pool.add(AgentProfile("P_base", "positioning", PositioningCompositeReward().get_weights()))
# -- manager WITHOUT a policy (AC will drive swaps) --
hotswap_mgr = HotswapManager(blue_pool, policy=None)
hotswap_reward = HotswapRewardAdapter(hotswap_mgr)

# -- AC profile policy over all profile names --
profile_names = list({*blue_pool.names_for_role("striker"),
                      *blue_pool.names_for_role("defender"),
                      *blue_pool.names_for_role("positioning")})
ac_policy  = ACProfilePolicy(ac_obs_builder=ac_obs_builder, profile_names=profile_names,
                             cfg=ACConfig(switch_penalty_base=0.5, switch_decay_seconds=25.0))
ac_adapter = HotswapACAdapter(ac_policy)

# -- env that runs AC swaps *inside* step(), then computes rewards from current composites --
env = EngineEnvAdapter(
    engine=engine,
    action_parser=action_parser,
    reward_function=hotswap_reward,
    ll_obs_builder=ll_obs_builder,
    ac_obs_builder=ac_obs_builder,
    action_mode="discrete",
    ac_adapter=ac_adapter,
    blue_size=2, orange_size=2,
)


# reset -> build PPO agents with correct input size
first_obs, info = env.reset()
agent_ids = list(env.state.cars.keys())
obs_size_ll = first_obs.shape[0]
ppo_agents = {aid: PPOAgent(obs_size=obs_size_ll, n_actions=N_ACTIONS) for aid in agent_ids}

# --- one step smoke test ---
# AC would pick profiles here (via policy internally), PPO picks actions per player using ll_obs
# actions = {}
# for aid in agent_ids:
#     a, logp, val = ppo_agents[aid].act(info["ll_obs"][aid])
#     actions[aid] = np.array([a], dtype=np.int64)

# obs1, reward_sum, done, info = env.step(actions)
# print("step ok; reward_sum=", reward_sum, "done=", done)


# Run one full match and save CSV
runner = MatchRunner(env, ppo_agents, kickoffs=KickoffMutator())
result = runner.run(log_path="out/match_log.csv")
print("Match finished:", result)