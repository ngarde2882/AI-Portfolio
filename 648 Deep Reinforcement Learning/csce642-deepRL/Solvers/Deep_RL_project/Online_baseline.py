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

async def main():
    player = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("RL_baseline", "Wagner777"),
        server_configuration=ShowdownServerConfiguration,
        )
    # Sending challenges to 'your_username'
    #await player.send_challenges("Joe_wag_RL", n_challenges=1)

    # Accepting one challenge from any user
    #await player.accept_challenges(None, 1)

    # Accepting three challenges from 'your_username'
    await player.accept_challenges('Joe_Wag_RL', 1)

    # Playing 5 games on the ladder
    #await player.ladder(5)

    # Print the rating of the player and its opponent after each battle
    # for battle in player.battles.values():
    #     print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
