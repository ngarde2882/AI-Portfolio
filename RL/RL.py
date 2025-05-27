import rlgym
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition
from rlgym.utils.action_parsers.continuous_action import ContinuousAction


import numpy as np
from rlgym.utils.gamestates import GameState
from rlgym.utils.obs_builders import ObsBuilder

class AdvancedStrikeObs(ObsBuilder):
    def __init__(self):
        self.GOAL_Y = 5120  # Target goal line
        self.BALL_RADIUS = 92.75

    def build_obs(self, player, state: GameState, previous_action):
        obs = []

        # Ball position and velocity
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        obs.extend(ball_pos)
        obs.extend(ball_vel)

        # --- Compute Optimal Strike Surface ---
        goal_target = np.array([0, self.GOAL_Y, 320])  # Center of opponent goal
        impact_direction = goal_target - ball_pos
        impact_direction /= np.linalg.norm(impact_direction)  # Normalize

        # Compute the best impact point
        impact_point = ball_pos + (self.BALL_RADIUS * impact_direction)
        obs.extend(impact_point)  # Add impact point to observation

        # --- Player Data ---
        car_pos = player.car_data.position
        car_vel = player.car_data.linear_velocity
        obs.extend(car_pos)
        obs.extend(car_vel)

        # --- Car Orientation (Facing Direction) ---
        car_facing = player.car_data.forward()  # Get the forward vector
        obs.extend(car_facing)

        # --- Opponent & Teammate Tracking ---
        for other in state.players:
            if other.car_id != player.car_id:  # Ignore self
                other_pos = other.car_data.position
                other_facing = other.car_data.forward()  # Facing direction of the opponent/teammate
                obs.extend(other_pos)
                obs.extend(other_facing)

        return np.array(obs, dtype=np.float32)


import rlgym
from stable_baselines3 import PPO
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition
from rlgym.utils.action_parsers.continuous_action import ContinuousAction

# Create RLGym environment with the new observation builder
env = rlgym.make(
    obs_builder=AdvancedStrikeObs(),
    reward_fn=VelocityBallToGoalReward(),
    terminal_conditions=[TimeoutCondition(225), NoTouchTimeoutCondition(30)],
    action_parser=ContinuousAction()
)

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)  # Train for 100k steps
model.save("rocket_league_advanced_agent")