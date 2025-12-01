# Deep Reinforcement Learning Project
Super Mario Bros RL Agent (Double DQN)

An autonomous agent trained to beat Super Mario Bros (World 1-1) using Deep Reinforcement Learning.

This project implements a Double Deep Q-Network (DDQN) with a custom Convolutional Neural Network (CNN). It overcomes the "sparse reward" problem common in side-scrollers through distinct Reward Shaping techniques, specifically a stagnation penalty that forces the agent to overcome local optima.
 
(The agent navigating World 1-1 after 8,000 episodes of training)

Key Features
Double Deep Q-Network (DDQN): Decouples action selection from target evaluation to minimize Q-value overestimation, leading to more stable training.
Perceptual Preprocessing: Converts raw NES pixels ($240 \times 256 \times 3$) into a stack of 4 grayscale frames ($4 \times 84 \times 84$) to allow the agent to perceive velocity and direction.

Custom Reward Shaping:
Velocity Reward: Positive feedback for moving right.
Stagnation Penalty: A rigid penalty ($-1.0$) applied if the agent stops progressing for 50 frames, preventing "camping" behaviors.
Death Penalty: Massive penalty ($-50$) for dying.
Efficient Memory: Utilizes TorchRL's LazyMemmapStorage to manage a Replay Buffer of 60,000 frames without crashing system RAM
