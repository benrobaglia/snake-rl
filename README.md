# **Snake-RL**

A reinforcement learning project where an agent learns to play the classic Snake game using Proximal Policy Optimization (PPO) or Deep Q-Networks (DQN).

![Snake Game Screenshot](images/snake_game_screenshot.png) <!-- Optional: Add a screenshot of your game -->

## **Table of Contents**

- [Introduction](#introduction)
- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [How to Run](#how-to-run)
- [TODO](#todo)

---

## **Introduction**

The `snake-rl` project implements a Snake game environment and trains an AI agent to play the game using Deep Reinforcement Learning. The primary goal is to explore how an agent can learn to navigate and make decisions in a dynamic environment with delayed rewards.

---

## **Installation**

To set up the environment for this project, follow these steps:

1. **Ensure you have Python 3.11 installed**:
    - You can download Python 3.11 from [python.org](https://www.python.org/downloads/).

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/snake-rl.git
   cd snake-rl

3. Install packages
    ```bash
    pip install -r requirements.txt


## **File Descriptions**

Here is a brief description of the key files in the project:

- **`env.py`**: Defines the `SnakeEnv` class, which is the custom OpenAI Gym-compatible environment for the Snake game. This includes the game's rules, state representation, and reward structure.
  
- **`human_play.py`**: A script to manually play the Snake game. Useful for testing the environment and understanding the dynamics of the game before training the agent.
  
- **`tensorboard_callback.py`**: Contains the `PrintEveryNTimesteps` callback to log custom metrics. This was intended to be used with TensorBoard but may not be fully functional due to compatibility issues.
  
- **`train.py`**: The main script to train the reinforcement learning agent. It sets up the environment, defines the neural network architecture, and configures the PPO algorithm from Stable Baselines3 for training.
  
- **`test.py`**: A script to test the trained agent by loading a saved model and running it in the environment to see the agent's performance.

---

## **How to Run**

1. **Train the Agent**: To train the agent, simply run the `train.py` script:

    ```bash
    python train.py
    ```

    This will start training the agent in the Snake environment using PPO and log metrics for monitoring.

2. **Play as a Human**: If you'd like to play the Snake game manually, you can use:

    ```bash
    python human_play.py
    ```

3. **Test the Trained Model**: To test a previously trained model, use:

    ```bash
    python test.py
    ```

---

## **TODO**

- **TensorBoard Callback**: tensorboard callback is not printing anything. Need to fix

---

