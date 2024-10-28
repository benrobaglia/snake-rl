from stable_baselines3 import PPO
from env import SnakeEnv
from tensorboard_callback import PrintEveryNTimesteps
from stable_baselines3.common.callbacks import EveryNTimesteps


def main():
    env = SnakeEnv(grid_size=10)

    # Define the policy kwargs
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    # Set PPO hyperparameters
    ppo_hyperparams = dict(
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="./ppo_snake_tensorboard/",
        **ppo_hyperparams
    )

    print_callback = EveryNTimesteps(n_steps=10000, callback=PrintEveryNTimesteps(n_steps=10000))

    # Train the agent
    total_timesteps = 10_000_000  # Adjust as needed
    model.learn(total_timesteps=total_timesteps, callback=print_callback)

    # Save the trained model
    model.save("ppo_snake_cnn_vec")

    # Close the vectorized environment
    env.close()

if __name__ == "__main__":
    main()

