# Reflex and Multi-Agent Search Strategies for Game AI Rolit

This repository contains a Python implementation of various artificial intelligence (AI) strategies for game agents, focusing on reflex agents and multi-agent search techniques. The project demonstrates the application of these strategies in a game setting, where agents must make decisions based on the game's state to achieve their objectives. Below is a detailed overview of the components and functionalities of this project.

## Overview

The project is structured into several key components, each responsible for different aspects of AI strategy implementation in games. These components include the ReflexAgent, which acts based on a state evaluation function, and MultiAgentSearchAgent, which provides a foundation for adversarial search strategies like Minimax, AlphaBeta pruning, and Expectimax.

### ReflexAgent

- **Purpose**: Chooses actions at each decision point by examining alternatives through a state evaluation function.
- **Key Methods**:
  - `getAction`: Selects the best action based on the evaluation function.
  - `evaluationFunction`: Evaluates the game state to guide action selection.
- **Implementation Notes**: This agent's behavior is guided by the evaluation of the current and successor game states, aiming to maximize the agent's score.

### MultiAgentSearchAgent

- **Purpose**: Serves as a base class for agents employing adversarial search strategies.
- **Strategies Implemented**:
  - **MinimaxAgent**: Utilizes the Minimax algorithm to choose actions considering the possible moves of all agents.
  - **AlphaBetaAgent**: Implements Alpha-Beta pruning to optimize the Minimax search process by eliminating unnecessary branches.
  - **ExpectimaxAgent**: Applies the Expectimax algorithm, accounting for the probabilistic outcomes of other agents' actions.
- **Key Methods**:
  - `getAction`: Determines the best action using the respective search strategy.

### Utility Functions

- **`scoreEvaluationFunction`**: A default evaluation function that returns the game state's score.
- **`betterEvaluationFunction`**: An advanced evaluation function that considers multiple aspects of the game state, such as coin parity, mobility, corners captured, and stability, to provide a more nuanced evaluation.

## Dependencies

- `Agents`: Module containing the abstract base classes for game agents.
- `util`: Provides utility functions and data structures for implementing search algorithms.
- `random`: Used for selecting among equally good actions randomly.

## Usage

To use these agents in a game, instantiate the desired agent class and pass the game state to its `getAction` method. The agent will then determine the best action to take based on its strategy.

## Conclusion

This project showcases the implementation of reflex and multi-agent search strategies in a game environment. Through these agents, we can explore foundational AI
