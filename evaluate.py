"""
evaluate.py — Head-to-head comparison of Random, Greedy, and PPO agents
=========================================================================
Usage:
    python evaluate.py                                  # synthetic data
    python evaluate.py --data mall_carpark_rl_dataset.xlsx
    python evaluate.py --episodes 50 --steps 500
"""

import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from environment import MallCarParkEnv, TOTAL_SPACES
from baselines import RandomAgent, GreedyAgent


def evaluate_agent(agent_fn, env_fn, n_episodes: int, label: str) -> dict:
    """
    Runs n_episodes and collects granular metrics.
    agent_fn() → agent with .predict(obs, env) → int
    """
    metrics = defaultdict(list)
    for ep in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done   = False
        while not done:
            action = agent_fn().predict(obs, env) if callable(agent_fn) else agent_fn.predict(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        s = env.summary()
        metrics["reward"].append(s["episode_reward"])
        metrics["assigned"].append(s["assigned"])
        metrics["rejected"].append(s["rejected"])
        metrics["invalid"].append(s["invalid_actions"])
        metrics["medical_served"].append(s["medical_served"])
        if s["avg_nav_time_min"] is not None:
            metrics["nav_time"].append(s["avg_nav_time_min"])
        if s["avg_proximity"] is not None:
            metrics["proximity"].append(s["avg_proximity"])

    def _fmt(vals):
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

    result = {
        "agent":          label,
        "episodes":       n_episodes,
        "mean_reward":    round(float(np.mean(metrics["reward"])), 2),
        "std_reward":     round(float(np.std(metrics["reward"])),  2),
        "mean_assigned":  round(float(np.mean(metrics["assigned"])), 1),
        "mean_rejected":  round(float(np.mean(metrics["rejected"])), 1),
        "mean_invalid":   round(float(np.mean(metrics["invalid"])),  1),
        "mean_medical":   round(float(np.mean(metrics["medical_served"])), 1),
        "mean_nav_time":  round(float(np.mean(metrics["nav_time"])),   2) if metrics["nav_time"]  else None,
        "mean_proximity": round(float(np.mean(metrics["proximity"])),  2) if metrics["proximity"] else None,
    }
    return result


def print_table(results: list[dict]):
    cols = [
        ("Agent",         "agent",          "<18"),
        ("Reward",        "mean_reward",    ">9"),
        ("Assigned",      "mean_assigned",  ">9"),
        ("Rejected",      "mean_rejected",  ">9"),
        ("Medical OK",    "mean_medical",   ">10"),
        ("Nav time (m)",  "mean_nav_time",  ">12"),
        ("Prox (1-5)",    "mean_proximity", ">10"),
    ]
    header = "  ".join(f"{h:{fmt}}" for h, _, fmt in cols)
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        row = "  ".join(
            f"{str(r.get(k, 'N/A')):{fmt}}" for _, k, fmt in cols
        )
        print(row)
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps",    type=int, default=500)
    parser.add_argument("--ppo",      action="store_true",
                        help="Include PPO if models/ directory exists")
    args = parser.parse_args()

    df = None
    if args.data and os.path.exists(args.data):
        df = pd.read_excel(args.data, sheet_name="Transaction Log", header=1)
        print(f"Dataset loaded: {len(df):,} rows")

    env_fn = lambda: MallCarParkEnv(df=df, n_steps=args.steps)

    print(f"\nEvaluating over {args.episodes} episodes × {args.steps} steps …\n")

    results = []
    results.append(evaluate_agent(RandomAgent, env_fn, args.episodes, "Random"))
    results.append(evaluate_agent(GreedyAgent, env_fn, args.episodes, "Greedy"))

    if args.ppo and os.path.exists("models/ppo_carpark_final.zip"):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        vec_env = DummyVecEnv([lambda: MallCarParkEnv(df=df, n_steps=args.steps)])
        if os.path.exists("models/vec_normalize.pkl"):
            vec_env = VecNormalize.load("models/vec_normalize.pkl", vec_env)
            vec_env.training = False
        model = PPO.load("models/ppo_carpark_final", env=vec_env)

        class PPOAgent:
            def predict(self, obs, env):
                norm_obs = vec_env.normalize_obs(obs[np.newaxis, :])
                action, _ = model.predict(norm_obs, deterministic=True)
                return int(action[0])

        ppo_agent = PPOAgent()
        results.append(evaluate_agent(
            lambda: ppo_agent, env_fn, args.episodes, "PPO (trained)"
        ))

    print_table(results)

    # Save to CSV
    out = pd.DataFrame(results)
    out.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved → evaluation_results.csv")


if __name__ == "__main__":
    main()
