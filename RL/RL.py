import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import rlgym
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, VelocityTowardsBallReward, EventReward
from rlgym.rocket_league.done_conditions.goal_condition import GoalCondition
from rlgym.rocket_league.done_conditions.no_touch_condition import NoTouchTimeoutCondition
from rlgym.rocket_league.action_parsers import DiscreteAction
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence


# --- HRL TEAM AGENT ---
# High-level Actor-Critic that assigns roles/behaviors to players based on team game state.
# Low-level behaviors are Q-learners for individual player control, evolved via reward mutation.


class TeamActorCritic(nn.Module):
    def __init__(self, obs_size, n_behaviors, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_behaviors),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        probs = self.actor(obs)
        value = self.critic(obs)
        return probs, value


class QLearningBehavior:
    def __init__(self, n_actions, obs_size, lr=0.01, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((obs_size, n_actions))  # Simplified discrete obs assumption
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        max_next = np.max(self.q_table[next_state]) if not done else 0
        td_target = reward + self.gamma * max_next
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])


class HierarchicalRLAgent:
    def __init__(self, obs_size, n_players, n_behaviors, n_actions):
        self.team_controller = TeamActorCritic(obs_size, n_behaviors)
        self.behaviors = [QLearningBehavior(n_actions, obs_size) for _ in range(n_behaviors)]
        self.n_players = n_players
        self.optimizer = optim.Adam(self.team_controller.parameters(), lr=3e-4)

    def assign_behaviors(self, team_obs):
        probs, _ = self.team_controller(team_obs)
        behavior_indices = torch.multinomial(probs, num_samples=self.n_players, replacement=True)
        return behavior_indices.tolist()

    def act(self, player_obs_list, team_obs):
        assigned_behaviors = self.assign_behaviors(team_obs)
        actions = []
        for i, obs in enumerate(player_obs_list):
            behavior_idx = assigned_behaviors[i]
            action = self.behaviors[behavior_idx].select_action(obs)
            actions.append(action)
        return actions

    def update_team_controller(self, team_obs, reward, next_obs, done):
        probs, value = self.team_controller(team_obs)
        with torch.no_grad():
            _, next_value = self.team_controller(next_obs)
            target = reward + (1 - done) * 0.99 * next_value
        advantage = target - value

        # Simple advantage actor-critic loss
        policy_loss = -torch.mean(torch.log(probs + 1e-8) * advantage.detach())
        value_loss = torch.mean(advantage.pow(2))
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- Evolutionary Mutation for Behaviors ---
def mutate_behavior_params(behavior, mutation_rate=0.1):
    behavior.lr *= np.clip(1 + np.random.uniform(-mutation_rate, mutation_rate), 0.001, 1.0)
    behavior.gamma = np.clip(behavior.gamma + np.random.uniform(-0.05, 0.05), 0.8, 0.999)
    behavior.epsilon = np.clip(behavior.epsilon + np.random.uniform(-0.05, 0.05), 0.01, 1.0)
    return behavior


def evolve_behaviors(behaviors, scores, elite_fraction=0.2):
    n_elite = max(1, int(len(behaviors) * elite_fraction))
    elite_idxs = np.argsort(scores)[-n_elite:]
    elites = [behaviors[i] for i in elite_idxs]

    new_behaviors = []
    for _ in range(len(behaviors)):
        parent = random.choice(elites)
        child = mutate_behavior_params(parent)
        new_behaviors.append(child)
    return new_behaviors


# --- RLGym Environment Integration ---
def make_rlgym_env():
    reward_fn = CombinedReward(
        (VelocityTowardsBallReward(), 1.0),
        (EventReward(goal=10, concede=-10), 1.0)
    )

    term_cond = [GoalCondition(), NoTouchTimeoutCondition(300)]

    # Use MutatorSequence to initialize teams and kickoff positions
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )

    env = rlgym.make(
        obs_builder=DefaultObs(),
        reward_fn=reward_fn,
        action_parser=DiscreteAction(),
        terminal_conditions=term_cond,
        state_mutator=state_mutator,
        tick_skip=8
    )
    return env


def train_hrl_agent(n_episodes=1000):
    env = make_rlgym_env()
    obs = env.reset()

    obs_size = len(obs[0])  # Assuming list of player obs
    n_players = len(obs)
    n_actions = env.action_space.n

    agent = HierarchicalRLAgent(obs_size=obs_size, n_players=n_players, n_behaviors=4, n_actions=n_actions)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            team_obs = torch.tensor(np.concatenate(obs), dtype=torch.float32).unsqueeze(0)
            actions = agent.act(obs, team_obs)
            next_obs, rewards, done, info = env.step(actions)

            total_reward = sum(rewards)
            next_team_obs = torch.tensor(np.concatenate(next_obs), dtype=torch.float32).unsqueeze(0)

            agent.update_team_controller(team_obs, total_reward, next_team_obs, done)

            for i, r in enumerate(rewards):
                behavior_idx = i % len(agent.behaviors)
                agent.behaviors[behavior_idx].update(i, actions[i], r, i, done)

            obs = next_obs
            ep_reward += total_reward

        print(f"Episode {ep+1}/{n_episodes} | Total Reward: {ep_reward:.2f}")


if __name__ == "__main__":
    train_hrl_agent(n_episodes=10)