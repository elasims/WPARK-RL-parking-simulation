"""
train.py — Train a PPO agent on the MallCarParkEnv
====================================================
Usage:
    python train.py                        # synthetic data
    python train.py --data mall_carpark_rl_dataset.xlsx

Output:
    models/ppo_carpark_final.zip           trained model
    models/ppo_carpark_best/               best checkpoint
    logs/ppo_carpark/                      TensorBoard logs
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from environment import MallCarParkEnv
from baselines import RandomAgent, GreedyAgent


# ---------------------------------------------------------------------------
# Custom callback: prints episode stats to console
# ---------------------------------------------------------------------------

class EpisodeLogCallback(BaseCallback):
    def __init__(self, log_freq: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq   = log_freq
        self.ep_count   = 0
        self.ep_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_count  += 1
                r = info["episode"]["r"]
                self.ep_rewards.append(r)
                if self.ep_count % self.log_freq == 0:
                    mean_r = np.mean(self.ep_rewards[-self.log_freq:])
                    print(f"  Episode {self.ep_count:>5}  "
                          f"mean reward (last {self.log_freq}): {mean_r:+.1f}")
        return True


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark(agent, env_fn, n_episodes: int = 5, label: str = "") -> dict:
    rewards, nav_times, prox_scores = [], [], []
    for _ in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.predict(obs, env)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        s = env.summary()
        rewards.append(s["episode_reward"])
        if s["avg_nav_time_min"]:
            nav_times.append(s["avg_nav_time_min"])
        if s["avg_proximity"]:
            prox_scores.append(s["avg_proximity"])
    result = {
        "agent":           label,
        "mean_reward":     round(float(np.mean(rewards)), 2),
        "mean_nav_time":   round(float(np.mean(nav_times)), 2)  if nav_times  else None,
        "mean_proximity":  round(float(np.mean(prox_scores)), 2) if prox_scores else None,
    }
    print(f"  [{label}]  reward={result['mean_reward']:+.1f}  "
          f"nav={result['mean_nav_time']} min  prox={result['mean_proximity']}/5")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str,  default=None,
                        help="Path to mall_carpark_rl_dataset.xlsx")
    parser.add_argument("--timesteps",  type=int,  default=200_000,
                        help="Total PPO training timesteps")
    parser.add_argument("--n_envs",     type=int,  default=4,
                        help="Number of parallel environments")
    parser.add_argument("--n_steps",    type=int,  default=500,
                        help="Steps per episode (cars per day)")
    parser.add_argument("--eval_freq",  type=int,  default=10_000)
    parser.add_argument("--no_baseline", action="store_true")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset (optional)
    # ------------------------------------------------------------------
    df = None
    if args.data and os.path.exists(args.data):
        print(f"\nLoading dataset: {args.data}")
        df = pd.read_excel(args.data, sheet_name="Transaction Log", header=1)
        print(f"  {len(df):,} rows loaded.")
    else:
        print("\nNo dataset provided — using synthetic car generator.")

    env_fn = lambda: MallCarParkEnv(df=df, n_steps=args.n_steps)

    # ------------------------------------------------------------------
    # Baseline benchmarks (before training)
    # ------------------------------------------------------------------
    if not args.no_baseline:
        print("\n── Pre-training baselines ──────────────────────────────")
        benchmark(RandomAgent(), env_fn, n_episodes=5, label="Random")
        benchmark(GreedyAgent(), env_fn, n_episodes=5, label="Greedy")

    # ------------------------------------------------------------------
    # Build vectorised + normalised training envs
    # ------------------------------------------------------------------
    print(f"\n── Training PPO  ({args.timesteps:,} steps, {args.n_envs} envs) ──")

    def make_env():
        env = MallCarParkEnv(df=df, n_steps=args.n_steps)
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = VecNormalize(
        make_vec_env(make_env, n_envs=1),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )

    # ------------------------------------------------------------------
    # PPO model
    # ------------------------------------------------------------------
    model = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 256,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,    # exploration bonus
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        tensorboard_log = "logs/",
        policy_kwargs = dict(net_arch=[256, 256]),   # 2-layer MLP
        verbose       = 0,
    )

    callbacks = [
        EpisodeLogCallback(log_freq=20),
        EvalCallback(
            eval_env,
            best_model_save_path = "models/ppo_carpark_best/",
            log_path             = "logs/eval/",
            eval_freq            = args.eval_freq,
            n_eval_episodes      = 5,
            deterministic        = True,
            render               = False,
            verbose              = 0,
        ),
        CheckpointCallback(
            save_freq  = 50_000,
            save_path  = "models/checkpoints/",
            name_prefix = "ppo_carpark",
            verbose     = 0,
        ),
    ]

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                tb_log_name="ppo_carpark")
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s")

    # Save final model + normalisation stats
    model.save("models/ppo_carpark_final")
    vec_env.save("models/vec_normalize.pkl")
    print("Saved → models/ppo_carpark_final.zip")

    # ------------------------------------------------------------------
    # Post-training evaluation
    # ------------------------------------------------------------------
    print("\n── Post-training evaluation ────────────────────────────────")

    # Wrap PPO for the benchmark helper
    class PPOWrapper:
        def __init__(self, m):
            self.m = m
        def predict(self, obs, env):
            # obs here is the raw env obs; for fair comparison we use it directly
            action, _ = self.m.predict(obs, deterministic=True)
            return int(action)

    if not args.no_baseline:
        benchmark(RandomAgent(), env_fn, n_episodes=10, label="Random")
        benchmark(GreedyAgent(), env_fn, n_episodes=10, label="Greedy")
    benchmark(PPOWrapper(model), env_fn, n_episodes=10, label="PPO (trained)")

    print("\nDone. To visualise training run:")
    print("  tensorboard --logdir logs/")


if __name__ == "__main__":
    main()
