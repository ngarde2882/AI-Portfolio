# Rocket League Reinforcement Learning Agent

This project is a hierarchical reinforcement learning agent designed to play Rocket League using the RLGym framework. The long-term goal is to train a capable agent through curriculum learning, replay-based observation, and competitive self-play.

## Objectives

- Simulate Rocket League training using RLGym and Carball
- Train an agent using PPO and curriculum-based reward shaping
- Develop modular environments and reward functions for skill-specific training (e.g., shooting, passing, blocking)

## Current Progress

- Environment setup with RLGym
- HRL agents composed (and saved during training)
- Detailed and tiered reward composition
- Match logging
- Stage 1 training: gauntlets
    We run every team matchup for a set number of matches to gain full-length episode training
    This step generates low-level logs of basic match info
- Stage 2 validation: tournaments
    We create 2 brackets of our teams that play a series of matches until the winner of both brackets play
    This step generates high-level logs for (planned) visualization and real break-downs of matches
- Stage 3 training: state-similarity
    We collect high-level logs to find patterns in player formations and game situations for intense replaying in one-goal/one-minute matches

- This training pipeline was promising and resulted in agents properly discovering rules of the game, but often one or two teama performing well would diminish the training of the others
- At this point I deemed that the hierarchical approach was slowing training due to overlapping but not shared observation states between hierarchy levels, and difficulty attributing rewards to distinct actions for the high-level agent