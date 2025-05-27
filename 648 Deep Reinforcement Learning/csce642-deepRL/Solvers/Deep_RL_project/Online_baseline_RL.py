import numpy as np
import pandas as pd
import asyncio
from stable_baselines3 import A2C,DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.spaces import Box

from poke_env.data import GenData
from poke_env.environment import SideCondition
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer, SimpleHeuristicsPlayer, Player
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
GEN_9_DATA = GenData.from_gen(9)
class Agentbaseline(Player):
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

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:

        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
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
        return self.action_to_move(action,battle)
    
async def main():
    # We create a random player
    player = Agentbaseline(model=DQN.load("dqn_model_30k"),
        account_configuration=AccountConfiguration("RL_baseline", "Wagner777"),
        server_configuration=ShowdownServerConfiguration,
    )
    # Sending challenges to 'your_username'
    #await player.send_challenges("Joe_wag_RL", n_challenges=1)

    # Accepting one challenge from any user
    #await player.accept_challenges(None, 1)

    # Accepting three challenges from 'your_username'
    await player.accept_challenges('Joe_Wag_RL', 5)

    # Playing 5 games on the ladder
    #await player.ladder(5)

    # Print the rating of the player and its opponent after each battle
    # for battle in player.battles.values():
    #     print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
