# Rocket League Reinforcement Learning Agent

This project is an early-stage reinforcement learning agent designed to play Rocket League using the RLGym framework. The long-term goal is to train a capable teammate agent through curriculum learning, replay-based observation, and competitive self-play.

## Objectives

- Simulate Rocket League training using RLGym and Carball
- Train an agent using PPO and curriculum-based reward shaping
- Develop modular environments and reward functions for skill-specific training (e.g., shooting, passing)

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
- Stage 3 training (planned): state-similarity
    We collect high-level logs to find patterns in player formations and game situations for intense replaying in one-goal matches
