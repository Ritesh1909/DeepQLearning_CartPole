# Deep Q-Network for OpenAI's Cartpole Environment

This repository contains the implementation of a Deep Q-Network (DQN) designed to solve the Cartpole balancing problem, a classic reinforcement learning task from OpenAI's Gym. The project implements advanced techniques like experience replay and a target network, along with extensive hyperparameter tuning and ablation studies to optimize performance.

## Project Overview

- **Objective**: To implement and refine a DQN model capable of balancing a pole on a cart in a simulation.
- **Technologies Used**: Python, PyTorch, OpenAI Gym
- **Key Features**:
  - Experience replay to enhance training stability.
  - A separate target network to improve convergence.
  - Exploration strategies including ε-greedy and Boltzmann (softmax) policies.
  - Comprehensive hyperparameter tuning and ablation study to find optimal settings.

## Environment Setup

The project is based on the CartPole-v1 environment provided by OpenAI Gym, which involves a pole attached to a cart moving along a frictionless track. The goal is to prevent the pole from falling over by moving the cart left or right.

### State Space

The state space consists of four continuous values:
- Cart Position (range: -4.8 to 4.8)
- Cart Velocity (range: -inf to inf)
- Pole Angle (range: -24° to 24°)
- Pole Angular Velocity (range: -inf to inf)

### Actions

The agent can take two discrete actions:
- Push the cart to the left (0)
- Push the cart to the right (1)

## Deep Q-Learning Implementation

### Neural Network Architecture

The DQN model uses a fully connected neural network with the following specifications:
- Input layer corresponding to the state space dimension (4 neurons).
- Two hidden layers with 128 neurons each.
- Output layer with 2 neurons (one for each possible action).

### Training

The training process involves:
- Collecting state, action, reward, and next state tuples in an experience replay buffer.
- Sampling from this buffer to train the network, using the Bellman equation to update Q-values.
- Periodically updating the weights of the target network to match those of the policy network.

## Results and Observations

Through rigorous experimentation with various configurations, the optimal hyperparameters were identified. Notably, the project explores the effects of different learning rates, discount factors, and network update strategies on the performance of the DQN agent.

### Ablation Study

The ablation study assesses the impact of experience replay and the target network on the DQN's performance. It was observed that using both techniques together yielded the best results, demonstrating quicker learning and greater stability compared to configurations that omitted one or both of these components.

## Conclusion

The project successfully demonstrates the application of a Deep Q-Network to a classic control task, with detailed analysis and optimization of the learning process. Future work could explore more complex environments or integrate other reinforcement learning advancements to further enhance the agent's performance.

## How to Run

In order to run all the experiments performed on DQN:
```python
python Experiment.py
```
In order to run DQN with Experience Replay and Target Network:
```python
 python dqn.py --experience_replay --target_network
```
In order to run DQN with Experience Replay only:
```python
 python dqn.py --experience_replay
```
In order to run DQN with Target Network only:
```python
 python dqn.py --target_network
```
In order to run DQN without Experience Replay and Target Network:
```python
 python dqn.py
```
