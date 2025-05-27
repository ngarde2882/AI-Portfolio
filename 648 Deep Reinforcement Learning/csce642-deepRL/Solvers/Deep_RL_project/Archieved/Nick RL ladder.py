import numpy as np
import asyncio
from stable_baselines3 import A2C,DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium.spaces import Box

from poke_env.data import GenData
from poke_env.environment import SideCondition
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer, SimpleHeuristicsPlayer, Player
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

import klefki

import time
import random
from tqdm import tqdm

random.seed(389)

class TrainingMonitorCallback(BaseCallback):
    def __init__(self, env_player, verbose=0):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.env_player = env_player

    def _on_step(self) -> bool:
        """This method is required but not used for reset logic."""
        return True  # Allow training to continue.

    def on_training_end(self) -> None:
        """Called when the training phase ends, ensuring cleanup."""
        print("Training phase completed. Resetting environment.")
        self.env_player.reset_env()  # Clean up battles safely

MAX_TURNS = 255
# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class Agent(Gen9EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # hp
        remaining_health = np.ones(6)
        opponent_remaining_health = np.ones(6)
        # stats
        stats = np.zeros(7)
        opponent_stats = np.zeros(7)
        # status
        status = np.zeros(6)
        opponent_status = [mon.status for mon in battle.opponent_team.values()]
        for i, mon in enumerate(battle.team.values()):
            remaining_health[i] = mon.current_hp_fraction
            if mon.status is not None:
                status[i] = mon.status.value
            if mon.active:
                for j,b in enumerate(mon.boosts.values()):
                    stats[j] = b
        for i, mon in enumerate(battle.opponent_team.values()):
            opponent_remaining_health[i] = mon.current_hp_fraction
            if mon.status is not None:
                opponent_status[i] = mon.status.value
            if mon.active:
                for j,b in enumerate(mon.boosts.values()):
                    opponent_stats[j] = b

        # tera
        tera = bool(battle.can_tera)
        opponent_tera = bool(battle._opponent_can_terrastallize)

        # hazards
        hazarddex = { # relevant side conditions to track (to reduce this dimension by half and not have variable sized input)
            SideCondition.UNKNOWN:0,
            SideCondition.AURORA_VEIL:1,
            SideCondition.LIGHT_SCREEN:2,
            SideCondition.MIST:3,
            SideCondition.REFLECT:4,
            SideCondition.SAFEGUARD:5,
            SideCondition.SPIKES:6,
            SideCondition.STEALTH_ROCK:7,
            SideCondition.STICKY_WEB:8,
            SideCondition.TAILWIND:9,
            SideCondition.TOXIC_SPIKES:10,
        }
        hazards = np.zeros(11)
        for cond,val in battle.side_conditions.items():
            if cond in hazarddex:
                hazards[hazarddex[cond]] = val
        opponent_hazards = np.zeros(11)
        for cond,val in battle.opponent_side_conditions.items():
            if cond in hazarddex:
                opponent_hazards[hazarddex[cond]] = val

        # terrain
        fields = np.zeros(13)
        for f,val in battle.fields.items():
            fields[f.value-1] = val

        # weather
        weather = np.zeros(9)
        for w,val in battle.weather.items():
            weather[w.value-1] = val

        # turn
        turn = battle.turn

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power, # 4* -1,3
                moves_dmg_multiplier, # 4* 0,4
                [remaining_mon_team, remaining_mon_opponent], # 1*1* 0,1 0,1
                remaining_health, # 6* 0,1
                opponent_remaining_health, # 6* 0,1
                [tera, opponent_tera], # 1*1* 0,1 0,1
                stats, # 7* -6,6
                opponent_stats, # 7* -6,6
                hazards, # 11* 0,MAX_TURNS
                opponent_hazards, # 11* 0,MAX_TURNS
                fields, # 13* 0,MAX_TURNS
                weather, # 9* 0,MAX_TURNS
                [turn], # 1* 0,MAX_TURNS
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )+25/current_state.turn
    
    def describe_embedding(self):
        low = np.concatenate([[-1]*4, [0]*4, [0]*2, [0]*6, [0]*6, [0]*2, [-6]*7, [-6]*7, [0]*11, [0]*11, [0]*13, [0]*9, [0]])
        high = np.concatenate([[3]*4, [4]*4, [1]*2, [1]*6, [1]*6, [1]*2, [6]*7, [6]*7, [MAX_TURNS]*11, [MAX_TURNS]*11, [MAX_TURNS]*13, [MAX_TURNS]*9, [MAX_TURNS]])
        return Box(
            np.array(low, dtype=np.float64),
            np.array(high, dtype=np.float64),
            dtype=np.float64,
        )

class OnlineAgent(Player):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
    
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # hp
        remaining_health = np.ones(6)
        opponent_remaining_health = np.ones(6)
        # stats
        stats = np.zeros(7)
        opponent_stats = np.zeros(7)
        # status
        status = np.zeros(6)
        opponent_status = [mon.status for mon in battle.opponent_team.values()]
        for i, mon in enumerate(battle.team.values()):
            remaining_health[i] = mon.current_hp_fraction
            if mon.status is not None:
                status[i] = mon.status.value
            if mon.active:
                for j,b in enumerate(mon.boosts.values()):
                    stats[j] = b
        for i, mon in enumerate(battle.opponent_team.values()):
            opponent_remaining_health[i] = mon.current_hp_fraction
            if mon.status is not None:
                opponent_status[i] = mon.status.value
            if mon.active:
                for j,b in enumerate(mon.boosts.values()):
                    opponent_stats[j] = b

        # tera
        tera = bool(battle.can_tera)
        opponent_tera = bool(battle._opponent_can_terrastallize)

        # hazards
        hazarddex = { # relevant side conditions to track (to reduce this dimension by half and not have variable sized input)
            SideCondition.UNKNOWN:0,
            SideCondition.AURORA_VEIL:1,
            SideCondition.LIGHT_SCREEN:2,
            SideCondition.MIST:3,
            SideCondition.REFLECT:4,
            SideCondition.SAFEGUARD:5,
            SideCondition.SPIKES:6,
            SideCondition.STEALTH_ROCK:7,
            SideCondition.STICKY_WEB:8,
            SideCondition.TAILWIND:9,
            SideCondition.TOXIC_SPIKES:10,
        }
        hazards = np.zeros(11)
        for cond,val in battle.side_conditions.items():
            if cond in hazarddex:
                hazards[hazarddex[cond]] = val
        opponent_hazards = np.zeros(11)
        for cond,val in battle.opponent_side_conditions.items():
            if cond in hazarddex:
                opponent_hazards[hazarddex[cond]] = val

        # terrain
        fields = np.zeros(13)
        for f,val in battle.fields.items():
            fields[f.value-1] = val

        # weather
        weather = np.zeros(9)
        for w,val in battle.weather.items():
            weather[w.value-1] = val

        # turn
        turn = battle.turn

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power, # 4* -1,3
                moves_dmg_multiplier, # 4* 0,4
                [remaining_mon_team, remaining_mon_opponent], # 1*1* 0,1 0,1
                remaining_health, # 6* 0,1
                opponent_remaining_health, # 6* 0,1
                [tera, opponent_tera], # 1*1* 0,1 0,1
                stats, # 7* -6,6
                opponent_stats, # 7* -6,6
                hazards, # 11* 0,MAX_TURNS
                opponent_hazards, # 11* 0,MAX_TURNS
                fields, # 13* 0,MAX_TURNS
                weather, # 9* 0,MAX_TURNS
                [turn], # 1* 0,MAX_TURNS
            ]
        )
    
    def describe_embedding(self):
        low = np.concatenate([[-1]*4, [0]*4, [0]*2, [0]*6, [0]*6, [0]*2, [-6]*7, [-6]*7, [0]*11, [0]*11, [0]*13, [0]*9, [0]])
        high = np.concatenate([[3]*4, [4]*4, [1]*2, [1]*6, [1]*6, [1]*2, [6]*7, [6]*7, [MAX_TURNS]*11, [MAX_TURNS]*11, [MAX_TURNS]*13, [MAX_TURNS]*9, [MAX_TURNS]])
        return Box(
            np.array(low, dtype=np.float64),
            np.array(high, dtype=np.float64),
            dtype=np.float64,
        )
    
    def action_to_move(self, action, battle):
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 8], mega=True
            )
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 12], dynamax=True
            )
        elif (
            battle.can_tera
            and 0 <= action - 16 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 16], terastallize=True
            )
        elif 0 <= action - 20 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 20])
        else:
            return self.choose_random_move(battle)

    def choose_move(self, battle):
        obs = self.embed_battle(battle)
        action, _ = self.model.predict(obs, deterministic=True)
        time.sleep(random.uniform(10, 30))
        return self.action_to_move(action)

class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


np.random.seed(0)
from typing import Dict
model_store: Dict[str, BaseAlgorithm] = {}

GEN_9_DATA = GenData.from_gen(9)
NB_RANDOM_TRAINING_STEPS = 10_000
NB_ADV_TRAINING_STEPS = 5000
TEST_EPISODES = 100
LADDER_EPISODES = 500

# for i, opponent in enumerate(opponents):
#     opponent.__name__ = names[i]

# Training functions
# def a2c_training():
#     # Initialize the environment player with the first opponent
#     env_player = Agent(opponent=opponents[0])
#     check_env(env_player)  # Ensure the environment complies with OpenAI Gym standards

#     # Initialize the A2C model
#     model = A2C("MlpPolicy", env_player, verbose=2)

#     # Train against each opponent sequentially
#     for i, opponent in enumerate(opponents):
#         print(f"Training against opponent {i + 1}: {names[i]}")
        
#         # Update opponent in the environment
#         env_player._opponent = opponent

#         # Attach the callback for automatic environment resetting
#         callback = TrainingMonitorCallback(env_player)

#         if i>0: NB_TRAINING_STEPS = NB_ADV_TRAINING_STEPS
#         else: NB_TRAINING_STEPS = NB_RANDOM_TRAINING_STEPS

#         # Train the model with the current opponent
#         model.set_env(env_player)
#         try:
#             model.learn(total_timesteps=NB_TRAINING_STEPS, callback=callback)
#         except Exception as e:
#             print(f"Error occurred during training: {e}")

#     # Store the trained model
#     # model_store['a2c'] = model
#     model.save('A2C_model')

def a2c_training():
    """
    Train the agent using A2C sequentially against each opponent.
    """
    print("Starting A2C training...")
    opponents = [
        RandomPlayer(battle_format='gen9randombattle'),
        MaxDamagePlayer(battle_format='gen9randombattle'),
        SimpleHeuristicsPlayer(battle_format='gen9randomebattle'),
    ]
    names = ['RandomPlayer', 'MaxDamagePlayer', 'SimpleHeuristicsPlayer']

    # Initialize the environment player with the first opponent
    env_player = Agent(opponent=opponents[0])
    check_env(env_player)  # Ensure the environment complies with OpenAI Gym standards

    # Initialize the A2C model
    model = A2C("MlpPolicy", env_player, verbose=1)

    # Train against each opponent sequentially
    for i, opponent in enumerate(opponents):
        print(f"Training against opponent {i + 1}: {names[i]}")

        # Update opponent in the environment
        # env_player._opponent = opponent
        env_player.reset_env(opponent)

        # Determine the number of training steps
        training_steps = NB_ADV_TRAINING_STEPS if i > 0 else NB_RANDOM_TRAINING_STEPS

        # Attach the callback for automatic environment resetting
        callback = TrainingMonitorCallback(env_player)

        # Train the model with the current opponent
        model.set_env(env_player)
        model.learn(total_timesteps=training_steps, callback=callback)

        print(f"Completed training against {names[i]}.")

    # # Quitters training
    # callback = TrainingMonitorCallback(env_player)
    # model.learn(total_timesteps=NB_RANDOM_TRAINING_STEPS, callback=callback)

    # Save the trained model
    model.save("A2C_model")
    print("A2C training completed and model saved.")



def dqn_training():
    # Initialize the environment player with the first opponent
    env_player = Agent(opponent=opponents[0])
    check_env(env_player)  # Ensure the environment complies with OpenAI Gym standards

    # Initialize the A2C model
    model = DQN("MlpPolicy", env_player, verbose=1)

    # Train against each opponent sequentially
    for i, opponent in enumerate(opponents):
        print(f"Training against opponent {i + 1}: {type(opponent).__name__}")
        
        # Update opponent in the environment
        env_player._opponent = opponent

        # Attach the callback for automatic environment resetting
        callback = TrainingMonitorCallback(env_player)

        if i>0: NB_TRAINING_STEPS = NB_ADV_TRAINING_STEPS
        else: NB_TRAINING_STEPS = NB_RANDOM_TRAINING_STEPS

        # Train the model with the current opponent
        model.set_env(env_player)
        model.learn(total_timesteps=NB_TRAINING_STEPS, callback=callback)
        env_player.reset_battles()

    # Store the trained model
    model_store['dqn'] = model
# evaluation runner
def evaluate_policy(model):
    print('Model Evaluation...')
    opponents = [
        RandomPlayer(battle_format='gen9randombattle'),
        MaxDamagePlayer(battle_format='gen9randombattle'),
        SimpleHeuristicsPlayer(battle_format='gen9randomebattle'),
    ]
    names = ['RandomPlayer', 'MaxDamagePlayer', 'SimpleHeuristicsPlayer']
    results = {}
    for i, opponent in enumerate(opponents):
        env_player = Agent(opponent=opponent)
        # env_player._opponent = opponent
        print(f'\t{names[i]}')
        model.set_env(env_player)
        print('Environment Set')

        finished_episodes = 0
        obs, _ = env_player.reset()
        print('Battle init')
        # obs, _, done, _, _ = env_player.step(0)
        while finished_episodes < TEST_EPISODES:
            action, _ = model.predict(obs, deterministic=True)
            print(f'pred {action}')
            obs, _, done, _, _ = env_player.step(action)
            print(f'step {action},{done}')

            if done:
                finished_episodes += 1
                print(f'Battle {finished_episodes} complete')
                obs, _ = env_player.reset()
                print(f'Environment Reset')
        results[names[i]] = env_player.n_won_battles
        print(f"Won {env_player.n_won_battles}/{TEST_EPISODES} battles against {names[i]}")
        env_player.reset_env()
    model.save('A2C_model')
    # return results
# evaluation functions
def a2c_evaluation():
    # Reset battle statistics
    model = A2C.load('A2C_model')
    results = evaluate_policy(model)
    if results is not None:
        print('A2C Evaluation Results:')
        for opponent, victories in results.items():
            print(f'Agent vs {opponent}: {victories}/{TEST_EPISODES} wins')

def dqn_evaluation():
    # Reset battle statistics
    model = model_store['dqn']
    results = evaluate_policy(model)
    if results is not None:
        print('DQN Evaluation Results:')
        for opponent, victories in results.items():
            print(f'{opponent}: {victories}/{TEST_EPISODES} wins')
    model.save('DQN_model')

# play online
async def a2cladder():
    print('Entering the ladder...')
    # We create a random player
    model = A2C.load('A2C_model')
    player = OnlineAgent(
        account_configuration=AccountConfiguration(username=klefki.a2cuser, password=klefki.password),
        server_configuration=ShowdownServerConfiguration,
        model=model,
        start_timer_on_battle_start=True,
    )
    print('It\'s a bad day to be a human pokemon trainer')
    await player.ladder(LADDER_EPISODES)
    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
         print(battle.rating, battle.opponent_rating)

async def dqnladder():
    print('Entering the ladder...')
    # We create a random player
    model = A2C.load('DQN_model')
    player = OnlineAgent(
        account_configuration=AccountConfiguration(username=klefki.a2cuser, password=klefki.password),
        server_configuration=ShowdownServerConfiguration,
        model=model,
        start_timer_on_battle_start=True,
    )
    print('It\'s a bad day to be a human pokemon trainer')
    await player.ladder(LADDER_EPISODES)
    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
         print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    a2c_training()
    a2c_evaluation()

    # asyncio.get_event_loop().run_until_complete(a2cladder())
