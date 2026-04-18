import os
import time
import argparse
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from mec_env_threats_und import MECEnvThreatsUnd  
import warnings
import torch

class ProgressAndSaveCallback(BaseCallback):
    def __init__(self, total_timesteps, save_path, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.save_path = save_path
        self.start_time = None
        self.best_mean_reward = -np.inf

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        elapsed = time.time() - self.start_time
        progress = self.num_timesteps / self.total_timesteps
        eta = elapsed / progress - elapsed if progress > 0 else 0
        print(f"\r[Progress] {progress*100:.2f}% | Steps: {self.num_timesteps}/{self.total_timesteps} | "
              f"Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min", end="")
        if self.n_calls % 1000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    print(f"\n💾 Best model saved with mean reward {mean_reward:.2f}")
        return True

def main(args):

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    env = MECEnvThreatsUnd(
        num_users=args.num_users,
        num_servers=args.num_servers,
        num_services=args.num_services,
        latency_threshold=args.latency_threshold
    )
    env = Monitor(env, args.log_dir)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=args.log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Starting training for {args.timesteps} timesteps...")
    callback = ProgressAndSaveCallback(total_timesteps=args.timesteps, save_path=args.output_model)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    print(f"\n Training complete! Best model saved at {args.output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30000, help="Total training timesteps")
    parser.add_argument("--num_users", type=int, default=200, help="Number of users")
    parser.add_argument("--num_servers", type=int, default=50, help="Number of servers")
    parser.add_argument("--num_services", type=int, default=2, help="Number of services")
    parser.add_argument("--latency_threshold", type=float, default=50, help="Latency threshold")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)  
    parser.add_argument("--exploration_fraction", type=float, default=0.1)
    parser.add_argument("--exploration_final_eps", type=float, default=0.02)
    parser.add_argument("--log_dir", type=str, default="logs/threats_und")
    parser.add_argument("--output_model", type=str, default="saved_models/AAnew_edge_agent_threats_und.zip")
    args = parser.parse_args()
    main(args)
