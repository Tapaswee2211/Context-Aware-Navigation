# train_a200.py
# ~/clearpath_ws/src/context_aware_nav/context_aware_nav/train_a200.py
#
# Run from inside the package folder:
#   cd ~/clearpath_ws/src/context_aware_nav/context_aware_nav/
#   python3 train_a200.py

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback)
from stable_baselines3.common.monitor import Monitor

# Import your env (same folder)
from pic4rl_camera_env import A200NavEnv


# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--timesteps",  type=int,   default=200_000)
parser.add_argument("--save_path",  type=str,   default="./models/a200_lidar")
parser.add_argument("--log_dir",    type=str,   default="./rl_logs")
parser.add_argument("--resume",     type=str,   default=None,
                    help="Path to a .zip model to resume from")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.log_dir,   exist_ok=True)


# ── Custom callback: live training graphs ─────────────────────────────────────
class TrainingGraphCallback(BaseCallback):
    """Saves reward and episode-length graphs every N steps."""

    def __init__(self, log_dir, save_every=5000, verbose=0):
        super().__init__(verbose)
        self.log_dir    = log_dir
        self.save_every = save_every
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_dists   = []   # final dist_to_goal per episode
        self._ep_reward = 0.0
        self._ep_len    = 0

    def _on_step(self) -> bool:
        reward   = self.locals["rewards"][0]
        done     = self.locals["dones"][0]
        info     = self.locals["infos"][0]

        self._ep_reward += reward
        self._ep_len    += 1

        if done:
            self.ep_rewards.append(self._ep_reward)
            self.ep_lengths.append(self._ep_len)
            dist = info.get("dist_to_goal", float('nan'))
            self.ep_dists.append(dist)
            self._ep_reward = 0.0
            self._ep_len    = 0

        if self.num_timesteps % self.save_every == 0 and len(self.ep_rewards) > 1:
            self._save_graphs()

        return True

    def _save_graphs(self):
        # ── 1. Episode reward ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        ep_idx = np.arange(len(self.ep_rewards))
        ax.plot(ep_idx, self.ep_rewards, color='steelblue',
                linewidth=0.8, alpha=0.6, label='Episode reward')
        # Rolling mean (window 20)
        if len(self.ep_rewards) > 20:
            rm = np.convolve(self.ep_rewards,
                             np.ones(20)/20, mode='valid')
            ax.plot(np.arange(19, len(self.ep_rewards)), rm,
                    color='navy', linewidth=1.8, label='Rolling mean (20)')
        ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title(f"Training reward — {self.num_timesteps:,} steps")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "rl_01_reward.png"), dpi=120)
        plt.close(fig)

        # ── 2. Episode length ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(ep_idx, self.ep_lengths, color='#e67e22',
                linewidth=0.8, alpha=0.6)
        if len(self.ep_lengths) > 20:
            rm = np.convolve(self.ep_lengths, np.ones(20)/20, mode='valid')
            ax.plot(np.arange(19, len(self.ep_lengths)), rm,
                    color='#d35400', linewidth=1.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Episode length")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "rl_02_ep_length.png"), dpi=120)
        plt.close(fig)

        # ── 3. Distance to goal at episode end ────────────────────────────
        valid = [(i, d) for i, d in enumerate(self.ep_dists)
                 if not np.isnan(d)]
        if valid:
            xi, yi = zip(*valid)
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.scatter(xi, yi, s=8, color='#27ae60', alpha=0.5)
            if len(yi) > 20:
                rm = np.convolve(yi, np.ones(20)/20, mode='valid')
                ax.plot(np.arange(19, len(yi)), rm,
                        color='#1e8449', linewidth=1.8)
            ax.axhline(0.4, color='red', linestyle='--',
                       linewidth=1, label='Goal tolerance (0.4m)')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Final dist to goal (m)")
            ax.set_title("Distance to goal at episode end")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir,
                        "rl_03_goal_dist.png"), dpi=120)
            plt.close(fig)

        print(f"[TrainingGraph] Saved graphs at step {self.num_timesteps:,} "
              f"| episodes: {len(self.ep_rewards)} "
              f"| last reward: {self.ep_rewards[-1]:.1f}")


# ── Build environment ─────────────────────────────────────────────────────────
print("Creating environment...")
env = A200NavEnv(use_camera=False, goal_position=(5.0, 0.0))
env = Monitor(env, os.path.join(args.log_dir, "monitor"))

# ── Build or load model ───────────────────────────────────────────────────────
if args.resume:
    print(f"Resuming from {args.resume}")
    model = SAC.load(args.resume, env=env)
else:
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
    )

# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq=10_000,
    save_path=args.save_path,
    name_prefix="a200_lidar")

graph_cb = TrainingGraphCallback(
    log_dir=args.log_dir,
    save_every=2_000)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training for {args.timesteps:,} steps...")
model.learn(
    total_timesteps=args.timesteps,
    callback=[checkpoint_cb, graph_cb],
    progress_bar=True)

# ── Save final model ──────────────────────────────────────────────────────────
final_path = os.path.join(args.save_path, "a200_lidar_final")
model.save(final_path)
print(f"Model saved → {final_path}.zip")

env.close()
