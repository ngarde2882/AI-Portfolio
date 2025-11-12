import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.done_conditions.goal_condition import GoalCondition
from rlgym.rocket_league.done_conditions.no_touch_condition import NoTouchTimeoutCondition
from rlgym.rocket_league.action_parsers import DiscreteAction
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


def teammate_hit_ball(obs, info):
    # If you later track last_toucher, replace this heuristic
    return 0.0


def opponent_bumped(obs, info):
    # Placeholder; requires explicit event logging outside GameState
    return 0.0


def opponent_demolished(obs, info):
    # Placeholder; requires explicit event logging outside GameState
    return 0.0


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

# ---- NEW, useful micro-features ----

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


# ===============================
# HIERARCHICAL CONTROLLER  (obs_size, n_actions, hyper=PPOHyper(), device="cpu") TODO
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

def initialize_engine_with_state(engine, initial_state=None):
    gs = initial_state if initial_state is not None else engine.create_base_state()
    engine.set_state(gs, shared_info={})
    return engine, gs


class EngineEnvAdapter:
    """
    Discrete low-level control with hotswap rewards.
    Exposes BOTH:
      - ll_obs[agent_id]: low-level obs for PPO
      - ac_obs: team-level obs for top AC (built via its own DefaultObs)
    """
    def __init__(self,
                 engine,
                 action_parser,
                 reward_function,               # HotswapRewardAdapter (per-player rewards)
                 ll_obs_builder: DefaultObs,    # low-level obs builder
                 ac_obs_builder: DefaultObs,    # AC obs builder (separate instance)
                 action_mode="discrete",
                 agent_selector=None):
        self.engine = engine
        self.action_parser = action_parser
        self.action_mode = action_mode
        self.reward_function = reward_function
        self.ll_obs_builder = ll_obs_builder
        self.ac_obs_builder = ac_obs_builder
        self.agent_selector = agent_selector

    def _build_ll_obs(self, state):
        # build per-agent observations
        obs_map = {}
        for aid in state.cars.keys():
            obs_map[aid] = self.ll_obs_builder.build_obs(aid, state, shared_info={})
        return obs_map

    def _build_ac_obs(self, state):
        # simplest path: build obs for each friendly agent and concatenate (or just use the first)
        # Here we use the first agent's obs as AC observation placeholder; replace with your team encoding.
        aid0 = next(iter(state.cars.keys()))
        return self.ac_obs_builder.build_obs(aid0, state, shared_info={})

    def reset(self, initial_state=None):
        _, state = initialize_engine_with_state(self.engine, initial_state=initial_state)
        # reset reward function with current agents
        agent_ids = list(state.cars.keys())
        self.reward_function.reset(agent_ids, state, {})
        ll_obs = self._build_ll_obs(state)
        ac_obs = self._build_ac_obs(state)
        # Gym-like reset returns something — we’ll return the first agent’s LL obs,
        # but stash both in info for training code.
        first_obs = ll_obs[agent_ids[0]]
        info = {"ll_obs": ll_obs, "ac_obs": ac_obs}
        return first_obs, info

    def step(self, actions_dict, shared_info=None):
        # actions_dict: {AgentID: np.array([discrete_id])}
        prev_state = self.engine.state
        controls_map = self.action_parser.parse_actions(actions_dict, prev_state, shared_info or {})
        state = self.engine.step(controls_map, shared_info or {})

        agent_ids = list(state.cars.keys())
        # rewards via hotswap composites (per-agent)
        rmap = self.reward_function.get_rewards(agent_ids, state, {aid: False for aid in agent_ids},
                                                {aid: False for aid in agent_ids}, shared_info or {})
        # observations
        ll_obs = self._build_ll_obs(state)
        ac_obs = self._build_ac_obs(state)
        # termination heuristic
        done = bool(getattr(state, "goal_scored", False))
        # Return first agent’s obs in obs slot; everything else in info
        first_obs = ll_obs[agent_ids[0]]
        info = {"ll_obs": ll_obs, "ac_obs": ac_obs, "rewards": rmap}
        # reward: sum or first — for compatibility, return sum; training can use info['rewards'] per agent
        reward_scalar = float(sum(rmap.values()))
        return first_obs, reward_scalar, done, info


# ===============================
# Example reward feature sets
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
}

defender_rewards = {
    'dist_to_ball': dist_to_ball,
    'boost_remaining': boost_remaining,
    'ball_hit': ball_hit,
    'behind_ball_defensive': behind_ball_defensive,
    # behind_other_players (teammates/opponents distances)
    'centerline_proximity': centerline_proximity,
    # home_goal_proximity
    'nearest_boost_inverse_distance': nearest_boost_inverse_distance,
    'nearest_boost_availability': nearest_boost_availability,
    'face_ball': face_ball,
}

positioning_rewards = {
    'mean_dist_to_teammates': mean_dist_to_teammates,
    'mean_dist_to_opponents': mean_dist_to_opponents,
    'centerline_proximity': centerline_proximity,
    'face_ball': face_ball,
    # face_goal
}

from hotswap_hrl import AgentProfile, TeamProfilePool, HotswapManager, HotswapRewardAdapter, default_policy_factory
from reward_native_classes import StrikerCompositeReward, DefenderCompositeReward, PositioningCompositeReward
from rlgym.rocket_league.sim import RocketSimEngine
# engine + action parser
engine = RocketSimEngine(rlbot_delay=True)
N_ACTIONS = 90
action_parser = RepeatAction(LookupTableAction(), repeats=8)

# profile pools / hotswap
blue_pool = TeamProfilePool()
blue_pool.add(AgentProfile("S_base", "striker",     StrikerCompositeReward().get_weights()))
blue_pool.add(AgentProfile("S_agro", "striker",     {**StrikerCompositeReward().get_weights(), "shot_alignment": 1.2, "goal": 12.0}))
blue_pool.add(AgentProfile("D_base", "defender",    DefenderCompositeReward().get_weights()))
blue_pool.add(AgentProfile("P_base", "positioning", PositioningCompositeReward().get_weights()))
blue_policy  = default_policy_factory(blue_pool)   # replace with AC policy later
hotswap_reward = HotswapRewardAdapter(HotswapManager(blue_pool, policy=blue_policy))

# separate obs builders
ll_obs_builder = DefaultObs()
ac_obs_builder = DefaultObs()

# env
env = EngineEnvAdapter(
    engine=engine,
    action_parser=action_parser,
    reward_function=hotswap_reward,
    ll_obs_builder=ll_obs_builder,
    ac_obs_builder=ac_obs_builder,
    action_mode="discrete",
)

# reset -> build PPO agents with correct input size
first_obs, info = env.reset()
agent_ids = list(env.state.cars.keys())
obs_size_ll = first_obs.shape[0]
ppo_agents = {aid: PPOAgent(obs_size=obs_size_ll, n_actions=N_ACTIONS) for aid in agent_ids}

# --- one step smoke test ---
# AC would pick profiles here (via policy internally), PPO picks actions per player using ll_obs
actions_dict = {}
for aid in agent_ids:
    a, logp, val = ppo_agents[aid].act(info["ll_obs"][aid])
    actions_dict[aid] = np.array([a], dtype=np.int64)

obs1, reward_scalar, done, info = env.step(actions_dict)
print("step ok; reward_sum=", reward_scalar, "done=", done)
