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
# LOW-LEVEL BEHAVIOR (tabular placeholder)
# ===============================
class QLearningBehavior:
    def __init__(self, n_actions, obs_size, reward_features, lr=0.01, gamma=0.99, epsilon=0.1):
        # NOTE: this tabular Q assumes a discrete observation index; adapt to function approximation for real usage
        self.q_table = np.zeros((obs_size, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        # reward_features: dict[name -> callable(obs, info) -> float]
        self.reward_features = reward_features

    def compute_feature_rewards(self, obs, info):
        results = {}
        for name, fn in self.reward_features.items():
            try:
                results[name] = float(fn(obs, info))
            except Exception:
                results[name] = 0.0
        return results

    def total_reward(self, feature_rewards, weights=None):
        if weights is None:
            weights = {k: 1.0 for k in feature_rewards.keys()}
        return sum(feature_rewards[k] * weights.get(k, 1.0) for k in feature_rewards)

    def select_action(self, state_idx):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, state_idx, action, reward, next_state_idx, done):
        max_next = np.max(self.q_table[next_state_idx]) if not done else 0.0
        td_target = reward + self.gamma * max_next
        self.q_table[state_idx, action] += self.lr * (td_target - self.q_table[state_idx, action])


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
# HIERARCHICAL CONTROLLER
# ===============================
class HierarchicalRLAgent:
    def __init__(self, obs_size, n_players, reward_categories, n_actions):
        self.n_categories = len(reward_categories)
        self.team_controller = TeamActorCritic(obs_size, self.n_categories)
        self.behaviors = [
            [QLearningBehavior(n_actions, obs_size, reward_features=wset) for wset in category]
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
    """
    No mutators for now.
    1) base_state = engine.create_base_state() if initial_state is None
    2) engine.set_state(state, shared)
    """
    gs = initial_state if initial_state is not None else engine.create_base_state()
    engine.set_state(gs, shared_info={})
    return engine, gs


class EngineEnvAdapter:
    def __init__(self, engine, action_parser=None, action_mode="discrete",
                 agent_selector=None, obs_fn=None, reward_fn=None, reward_function=None):  # <-- add reward_function
        self.engine = engine
        self.action_parser = action_parser
        self.action_mode = action_mode
        self.agent_selector = agent_selector
        self.obs_fn = obs_fn or build_observation_from_info
        self.reward_fn = reward_fn
        self.reward_function = reward_function  # <-- store it

    def reset(self, initial_state=None):
        _, state = initialize_engine_with_state(self.engine, initial_state=initial_state)
        agent_id = (self.agent_selector(state) if self.agent_selector else next(iter(state.cars.keys())))
        # if using a RewardFunction (hotswap), reset it now
        if self.reward_function is not None:
            self.reward_function.reset([agent_id], state, {})
        info = build_info_from_state(state, agent_id)
        obs = self.obs_fn(info)
        return obs

    def step(self, actions_dict, shared_info=None):
        prev_state = self.engine.state
        if self.action_mode == "discrete":
            assert self.action_parser is not None, "action_parser required for discrete action mode"
            controls_map = self.action_parser.parse_actions(actions_dict, prev_state, shared_info or {})
        else:
            controls_map = actions_dict

        state = self.engine.step(controls_map, shared_info or {})
        agent_id = (self.agent_selector(state) if self.agent_selector else next(iter(state.cars.keys())))
        info = build_info_from_state(state, agent_id)
        obs = self.obs_fn(info)

        if self.reward_function is not None:
            rmap = self.reward_function.get_rewards([agent_id], state, {agent_id: False}, {agent_id: False}, shared_info or {})
            reward = float(rmap[agent_id])
        elif self.reward_fn is not None:
            reward = float(self.reward_fn(obs, info))
        else:
            reward = 0.0

        done = bool(getattr(state, "goal_scored", False))
        return obs, reward, done, info


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
engine = RocketSimEngine(rlbot_delay=True)
# --- Build a profile pool with a few genomes (weights are mutable and saved verbatim) ---
blue_pool = TeamProfilePool()
blue_pool.add(AgentProfile(name="S_base", role="striker",     weights=StrikerCompositeReward().get_weights()))
blue_pool.add(AgentProfile(name="S_agro", role="striker",     weights={**StrikerCompositeReward().get_weights(), "shot_alignment": 1.2, "goal": 12.0}))
blue_pool.add(AgentProfile(name="D_base", role="defender",    weights=DefenderCompositeReward().get_weights()))
blue_pool.add(AgentProfile(name="P_base", role="positioning", weights=PositioningCompositeReward().get_weights()))

# simple policy now; swap in your AC policy later
blue_policy  = default_policy_factory(blue_pool)
blue_manager = HotswapManager(blue_pool, policy=blue_policy)
hotswap_reward = HotswapRewardAdapter(blue_manager)

env = EngineEnvAdapter(
    engine,
    action_parser=action_parser,
    action_mode="discrete",
    obs_fn=build_observation_from_info,
    reward_fn=None,
    reward_function=hotswap_reward,   # <<— use hotswap here
)
