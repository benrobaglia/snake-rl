from stable_baselines3.common.callbacks import BaseCallback

class PrintEveryNTimesteps(BaseCallback):
    def __init__(self, n_steps=10000, verbose=0):
        super(PrintEveryNTimesteps, self).__init__(verbose)
        self.n_steps = n_steps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.n_steps == 0:
            # Print timestep information
            print(f"Timestep {self.num_timesteps}:")

            # Access relevant metrics from self.locals
            ep_len_mean = self.locals.get("rollout/ep_len_mean", "N/A")
            ep_rew_mean = self.locals.get("rollout/ep_rew_mean", "N/A")
            entropy_loss = self.locals.get("train/entropy_loss", "N/A")
            policy_loss = self.locals.get("train/policy_gradient_loss", "N/A")
            value_loss = self.locals.get("train/value_loss", "N/A")

            # Print the metrics
            print(f"  Mean episode length: {ep_len_mean}")
            print(f"  Mean episode reward: {ep_rew_mean}")
            print(f"  Entropy loss: {entropy_loss}")
            print(f"  Policy gradient loss: {policy_loss}")
            print(f"  Value loss: {value_loss}")
            
        return True
