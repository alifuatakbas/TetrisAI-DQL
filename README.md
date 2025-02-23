# TetrisAI-DQL

A Deep Q-Learning implementation for Tetris game environment.

## Overview

This project implements a Deep Q-Learning algorithm to train an AI agent to play Tetris. The agent learns through experience, utilizing prioritized experience replay and a custom reward system.

## Technical Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pygame
- Keras
- Matplotlib (For training result graphics)

## Implementation Details

### State Representation
- Maximum height of the stack
- Number of holes in the structure
- Surface roughness measurement
- Number of completed lines

### Neural Network Architecture
- Input layer: 4 neurons (state features)
- Hidden layers: 2 layers of 64 neurons each
- Output layer: 40 neurons (possible actions)
- Activation function: ReLU
- Output activation: Linear

### Learning Parameters
- Learning rate: 0.001
- Discount factor (γ): 0.99
- Epsilon decay: 0.9995
- Minimum epsilon: 0.01
- Batch size: 64
- Memory capacity: 20,000 experiences

- ## Performance Metrics

The agent's performance is evaluated based on:
- Number of lines cleared
- Average game duration
- Maximum height achieved
- Learning convergence rate

## Future Development

Planned improvements include:
- Implementation of alternative DQL variants
- Enhanced visualization capabilities
- Refined reward system
- Performance optimization
