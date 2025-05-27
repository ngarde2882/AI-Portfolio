# CSCE 648: Deep Reinforcement Learning

This course focused on the theory and practice of reinforcement learning. I implemented a wide range of algorithms and developed a Pokémon Showdown battle agent that progressed through increasingly capable opponents.

## Implemented Algorithms

- Q-Learning
- SARSA
- REINFORCE
- Monte Carlo
- Value & Policy Iteration
- Deep Q-Network (DQN)
- Advantage Actor-Critic (A2C)
- Deep Deterministic Policy Gradient (DDPG)

## Pokémon RL Agent

- Trained via curriculum learning:
  - Stage 1: vs. random move selector
  - Stage 2: vs. heuristic opponent
  - Stage 3: self-play
- Final stage: played online — successful against casual players but couldn't outperform human strategic play

> Agent architecture includes both A2C and DQN implementations.
