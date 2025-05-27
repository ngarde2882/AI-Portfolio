import numpy as np
import asyncio
from stable_baselines3 import A2C,DQN
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.environment import SideCondition
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer,SimpleHeuristicsPlayer

from poke_env import AccountConfiguration, ShowdownServerConfiguration

import klefki

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
    
    def choose_move(self, battle):
        obs = self.embed_battle(battle)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.action_to_move(action)
    
    def complete_current_episode(self):
        # Check if all ongoing battles are complete
        return all(battle.finished for battle in self.battles.values())


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

model_store = {}

GEN_9_DATA = GenData.from_gen(9)
NB_TRAINING_STEPS = 30_000
TEST_EPISODES = 100
LADDER_EPISODES = 2
# Training functions
# def a2c_training():
#     opponent = RandomPlayer(battle_format='gen9randombattle')
#     second_opponent = MaxDamagePlayer(battle_format='gen9randombattle')
#     third_opponent = SimpleHeuristicsPlayer(battle_format='gen9randombattle')
#     env_player = Agent(opponent=opponent)
#     check_env(env_player)
#     model = A2C("MlpPolicy", env_player, verbose=1)
#     model.learn(total_timesteps=NB_TRAINING_STEPS)
#     # model.env._opponent=second_opponent
#     # model.env.reset_env(restart=True, opponent=second_opponent)
#     env_player._opponent = second_opponent
#     model.set_env(env_player)
#     model.learn(total_timesteps=NB_TRAINING_STEPS)
#     env_player._opponent = third_opponent
#     model.set_env(env_player)
#     model.learn(total_timesteps=NB_TRAINING_STEPS)

#     model_store['a2c'] = model

def a2c_training():
    # Initialize opponents
    opponents = [
        RandomPlayer(battle_format='gen9randombattle'),
        MaxDamagePlayer(battle_format='gen9randombattle'),
        SimpleHeuristicsPlayer(battle_format='gen9randombattle')
    ]

    # Initialize the environment player with the first opponent
    env_player = Agent(opponent=opponents[0])
    check_env(env_player)  # Ensure the environment complies with OpenAI Gym standards

    # Create the model
    model = DQN("MlpPolicy", env_player, verbose=1)
    model.learn(total_timesteps=NB_TRAINING_STEPS)  # Train model

    # Train against each opponent sequentially
    for i, opponent in enumerate(opponents):
        pass
        #print(f"Training against opponent {i + 1}: {type(opponent).__name__}")
        #env_player._opponent = opponent  # Change opponent
        
        # Wait for any ongoing episodes to finish
        #while not env_player.complete_current_episode():
        #pass  # Loop until the environment signals completion

        #env_player.reset_battles()  # Reset battles cleanly after completion
        #model.set_env(env_player)  # Set the updated environment
        #model.learn(total_timesteps=NB_TRAINING_STEPS)  # Train model

    # Store the trained model
    model_store['a2c'] = model

def dqn_training():
    opponent = RandomPlayer()
    second_opponent = MaxDamagePlayer()
    third_opponent = SimpleHeuristicsPlayer()
    env_player = Agent(opponent=opponent)

    model = DQN("MlpPolicy", env_player, verbose=1)
    model.learn(total_timesteps=NB_TRAINING_STEPS)
    model.env._opponent=second_opponent
    model.learn(total_timesteps=NB_TRAINING_STEPS)
    model.env._opponent=third_opponent
    model.learn(total_timesteps=NB_TRAINING_STEPS)

    model_store['dqn'] = model

# evaluation runner
def evaluate_policy(model):
    opponents = [
        RandomPlayer(battle_format='gen9randombattle'),
        MaxDamagePlayer(battle_format='gen9randombattle'),
        SimpleHeuristicsPlayer(battle_format='gen9randombattle'),
    ]
    results = {}
    for opponent in opponents:
        env_player = Agent(opponent=opponent)
        env_player.reset_battles()
        model.set_env(env_player)

        finished_episodes = 0
        obs, _ = env_player.reset()

        while finished_episodes < TEST_EPISODES:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env_player.step(action)

            if done:
                finished_episodes += 1
                if finished_episodes < TEST_EPISODES:
                    break
                obs, _ = env_player.reset()
        results[type(opponent).__name__] = env_player.n_won_battles
        print("Won", env_player.n_won_battles, "battles against", env_player._opponent)
# evaluation functions
def a2c_evaluation():
    # Reset battle statistics
    model = model_store['a2c']
    results = evaluate_policy(model)
    if results is not None:
        print('A2C Evaluation Results:')
        for opponent, victories in results.items():
            print(f'{opponent}: {victories}/{TEST_EPISODES} wins')

def dqn_evaluation():
    # Reset battle statistics
    model = model_store['dqn']
    model.env.reset_battles()
    evaluate_policy(model)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (model.env.n_won_battles, TEST_EPISODES)
    )

# play online
async def a2cladder():
    # We create a random player
    model = model_store["a2c"]
    player = Agent(
        account_configuration=AccountConfiguration(username=klefki.dqnuser, password=klefki.password),
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=True,
    )

    model.env = player
    await model.env.ladder(LADDER_EPISODES)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)

async def dqnladder():
    # We create a random player
    model = model_store["dqn"]
    player = Agent(
        player_configuration=AccountConfiguration(username=klefki.dqnuser, password=klefki.password),
        server_configuration=ShowdownServerConfiguration,
    )

    model.env = player
    await model.env.ladder(LADDER_EPISODES)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    a2c_training()
    a2c_evaluation()

    #asyncio.get_event_loop().run_until_complete(a2cladder())
    
    