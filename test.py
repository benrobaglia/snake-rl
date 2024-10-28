from stable_baselines3 import PPO
from env import SnakeEnv
import time


def main():
    # Create a single environment for evaluation
    env = SnakeEnv(grid_size=10)

    # Load the trained model
    model = PPO.load("ppo_snake_cnn_vec", env=env)

    # Evaluate the agent
    episodes = 5  # Number of episodes to run
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.05)  # Slow down the rendering for visibility
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
