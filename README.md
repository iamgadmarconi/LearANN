# LearANN - Learning Artificial Neural Networks

A Python project for implementing and understanding neural networks from scratch, focusing on Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks.

## Project Overview

I created LearANN as a project to understand neural network architectures from the ground up. Instead of relying on high-level deep learning frameworks, I built everything from first principles to gain deeper insights into how these models actually work under the hood.

## Features

- Custom implementation of neural network layers:
  - Dense (fully connected) layers
  - LSTM cells
- Various activation functions (ReLU, Sigmoid, Tanh, Linear)
- Different optimization algorithms:
  - Gradient Descent
  - Adagrad
  - Adam
- Loss functions:
  - Mean Squared Error (MSE)
  - Cross Entropy
- CUDA acceleration support for improved performance
- Test cases demonstrating capabilities:
  - Simple regression tasks
  - Sine wave prediction

## Project Structure

- `src/python/`: Python implementation
  - `networks/`: Neural network architectures
    - `rnn/`: Recurrent Neural Network implementation
    - `cnn/`: Convolutional Neural Network implementation (work in progress)
  - `utils/`: Utility functions and components
    - `layers/`: Layer implementations (Dense, LSTM)
    - `optimizers/`: Optimization algorithms
    - `cuda/`: CUDA acceleration utilities

## Performance Optimization

The project utilizes Numba JIT compilation to accelerate computations and includes optional CUDA support for GPU acceleration, demonstrating an understanding of performance considerations in machine learning implementations.

## Educational Purpose

I built this project as a hands-on learning experience to:

- Understand forward and backward propagation algorithms at a fundamental level
- Implement various gradient-based optimization techniques
- Learn about memory management challenges in neural networks
- Explore acceleration techniques for numerical computation

While I didn't design this for production use, implementing these concepts from scratch gave me insights into neural network mechanics that often remain hidden behind abstractions in higher-level frameworks.

## License

This project is licensed under the terms included in the LICENSE file.