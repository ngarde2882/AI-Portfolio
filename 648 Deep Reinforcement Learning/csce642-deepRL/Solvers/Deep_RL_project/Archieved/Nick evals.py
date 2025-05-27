GEN_9_DATA = GenData.from_gen(9)
NB_RANDOM_TRAINING_STEPS = 2_000
NB_ADV_TRAINING_STEPS = 1000
TEST_EPISODES = 10
LADDER_EPISODES = 500
opponents = [
        RandomPlayer(battle_format='gen9randombattle'),
        MaxDamagePlayer(battle_format='gen9randombattle'),
        SimpleHeuristicsPlayer(battle_format='gen9randomebattle'),
    ]
names = ['RandomPlayer', 'MaxDamagePlayer', 'SimpleHeuristicsPlayer']
# evaluation runner
def evaluate_policy(model):
    print('Model Evaluation...')
    results = {}
    for i, opponent in enumerate(opponents):
        env_player = Agent(opponent=opponent)
        # env_player._opponent = opponent
        print(f'\t{names[i]}')
        model.set_env(env_player)
        print('Environment Set')

        finished_episodes = 0
        obs, _ = env_player.reset()
        print('Battle innit')
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
    # return results
def a2c_evaluation():
    # Reset battle statistics
    model = model_store['a2c']
    results = evaluate_policy(model)
    if results is not None:
        print('A2C Evaluation Results:')
        for opponent, victories in results.items():
            print(f'Agent vs {opponent}: {victories}/{TEST_EPISODES} wins')
    model.save('A2C_model')

def dqn_evaluation():
    # Reset battle statistics
    model = model_store['dqn']
    results = evaluate_policy(model)
    if results is not None:
        print('DQN Evaluation Results:')
        for opponent, victories in results.items():
            print(f'{opponent}: {victories}/{TEST_EPISODES} wins')
    model.save('DQN_model')